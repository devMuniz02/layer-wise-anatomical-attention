import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig, GPT2Tokenizer, PreTrainedModel

from .configuration_lana import LanaConfig
from .gpt2_modified import create_decoder
from .layerwise_anatomical_attention import build_layerwise_attention_bias, build_legacy_gaussian_attention_bias
from .modeling_outputs import LanaModelOutput
from .segmenters import AnatomicalSegmenter

logger = logging.getLogger(__name__)
PAD_TOKEN = "<|pad|>"


def _resolve_repo_root(config: LanaConfig) -> Path | None:
    for candidate in [getattr(config, "local_repo_path", ""), getattr(config, "_name_or_path", "")]:
        if not candidate:
            continue
        path = Path(str(candidate))
        if path.exists():
            return path
    return None


def _resolve_source(reference: str, repo_root: Path | None) -> str:
    if not reference:
        return reference
    path = Path(reference)
    if path.is_absolute() and path.exists():
        return str(path)
    if repo_root is not None:
        repo_path = repo_root / reference
        if repo_path.exists():
            return str(repo_path)
    if path.exists():
        return str(path)
    return reference


def _resolve_tokenizer_source(config: LanaConfig, repo_root: Path | None) -> str:
    for reference in [
        getattr(config, "bundled_tokenizer_name", ""),
        "",
    ]:
        if reference:
            resolved = _resolve_source(reference, repo_root)
            if resolved and Path(resolved).exists():
                return resolved
    if repo_root is not None and (repo_root / "tokenizer_config.json").exists():
        return str(repo_root)
    return _resolve_source(config.text_model_name, repo_root)


def _is_local_source(reference: str, repo_root: Path | None) -> bool:
    resolved = _resolve_source(reference, repo_root)
    return bool(resolved) and Path(resolved).exists()


def build_visual_projection(config: LanaConfig) -> nn.Module:
    if config.visual_projection_type == "linear":
        return nn.Linear(config.visual_feature_dim, config.text_hidden_size)
    if config.visual_projection_type == "mlp4":
        return nn.Sequential(
            nn.Linear(config.visual_feature_dim, config.text_hidden_size),
            nn.GELU(),
            nn.Linear(config.text_hidden_size, config.text_hidden_size),
            nn.GELU(),
            nn.Linear(config.text_hidden_size, config.text_hidden_size),
            nn.GELU(),
            nn.Linear(config.text_hidden_size, config.text_hidden_size),
        )
    raise ValueError(f"Unsupported visual projection type: {config.visual_projection_type}")


class LanaForConditionalGeneration(PreTrainedModel):
    config_class = LanaConfig
    base_model_prefix = "lana"
    supports_gradient_checkpointing = True

    def __init__(self, config: LanaConfig):
        super().__init__(config)
        repo_root = _resolve_repo_root(config)
        vision_model_name = _resolve_source(getattr(config, "bundled_vision_model_name", "") or config.vision_model_name, repo_root)
        text_model_name = _resolve_source(getattr(config, "bundled_text_model_name", "") or config.text_model_name, repo_root)
        segmentation_model_name = _resolve_source(
            getattr(config, "bundled_segmentation_model_name", "") or config.segmentation_model_name,
            repo_root,
        )
        tokenizer_source = _resolve_tokenizer_source(config, repo_root)
        lung_checkpoint = _resolve_source(config.lung_segmenter_checkpoint, repo_root)
        heart_checkpoint = _resolve_source(config.heart_segmenter_checkpoint, repo_root)
        segmenter_weights_in_model_state = bool(getattr(config, "segmenter_weights_in_model_state", False))

        vision_config = AutoConfig.from_pretrained(vision_model_name, trust_remote_code=True)
        if getattr(vision_config, "hidden_size", None) is not None:
            config.visual_feature_dim = vision_config.hidden_size

        vision_load_pretrained = not _is_local_source(vision_model_name, repo_root)
        if vision_load_pretrained:
            self.vision_encoder = AutoModel.from_pretrained(vision_model_name, trust_remote_code=True)
        else:
            self.vision_encoder = AutoModel.from_config(vision_config, trust_remote_code=True)
        decoder_kwargs = {
            "ignore_mismatched_sizes": True,
            "use_cache": config.use_cache,
        }
        if config.decoder_load_in_4bit:
            compute_dtype = getattr(torch, config.decoder_compute_dtype, torch.float16)
            decoder_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            decoder_kwargs["device_map"] = {"": 0}
        self.text_decoder = create_decoder(
            text_model_name=text_model_name,
            attention_implementation=config.segmentation_attention_implementation,
            max_position_embeddings=config.max_position_embeddings,
            load_pretrained=not _is_local_source(text_model_name, repo_root),
            vocab_size=getattr(config, "vocab_size", None),
            **decoder_kwargs,
        )
        if _is_local_source(tokenizer_source, repo_root):
            self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_source)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True, use_fast=False)
        if self.tokenizer.pad_token_id is None:
            target_vocab_size = getattr(config, "vocab_size", None)
            if target_vocab_size and target_vocab_size > len(self.tokenizer):
                self.tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
            else:
                fallback_pad = self.tokenizer.eos_token or self.tokenizer.bos_token or PAD_TOKEN
                self.tokenizer.pad_token = fallback_pad
        if self.text_decoder.get_input_embeddings().weight.shape[0] != len(self.tokenizer):
            self.text_decoder.resize_token_embeddings(len(self.tokenizer))
        self.text_decoder.config.pad_token_id = self.tokenizer.pad_token_id
        if hasattr(self.text_decoder, "generation_config") and self.text_decoder.generation_config is not None:
            self.text_decoder.generation_config.pad_token_id = self.tokenizer.pad_token_id
            self.text_decoder.generation_config.eos_token_id = None

        config.vocab_size = self.text_decoder.config.vocab_size
        config.text_hidden_size = self.text_decoder.config.hidden_size
        config.num_attention_layers = self.text_decoder.config.n_layer

        self.visual_projection = build_visual_projection(config)
        self.segmenter = None
        if config.use_segmentation_mask:
            assume_segmenter_weights_from_model_state = segmenter_weights_in_model_state and not (
                Path(lung_checkpoint).exists() or Path(heart_checkpoint).exists()
            )
            self.segmenter = AnatomicalSegmenter(
                model_name=segmentation_model_name,
                freeze=config.freeze_segmenter,
                lung_checkpoint=lung_checkpoint,
                heart_checkpoint=heart_checkpoint,
                load_pretrained=not _is_local_source(segmentation_model_name, repo_root),
                assume_weights_from_model_state=assume_segmenter_weights_from_model_state,
            )
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        kwargs.setdefault("low_cpu_mem_usage", False)
        config = kwargs.get("config")
        if config is not None and getattr(config, "local_repo_path", ""):
            return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        repo_path = str(pretrained_model_name_or_path)
        if not Path(repo_path).exists():
            repo_path = snapshot_download(repo_path)

        if config is None:
            config = LanaConfig.from_pretrained(repo_path, trust_remote_code=True)
        config.local_repo_path = repo_path
        kwargs["config"] = config
        return super().from_pretrained(repo_path, *model_args, **kwargs)

    def move_non_quantized_modules(self, device: torch.device) -> None:
        self.vision_encoder.to(device)
        self.visual_projection.to(device)
        if self.segmenter is not None:
            self.segmenter.to(device)
        if not getattr(self.config, "decoder_load_in_4bit", False):
            self.text_decoder.to(device)

    def _encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if any(param.requires_grad for param in self.vision_encoder.parameters()):
            outputs = self.vision_encoder(pixel_values=pixel_values)
        else:
            with torch.no_grad():
                outputs = self.vision_encoder(pixel_values=pixel_values)
        hidden = outputs.last_hidden_state
        tokens_to_skip = max(0, int(getattr(self.config, "vision_feature_prefix_tokens_to_skip", 1)))
        if hidden.shape[1] > tokens_to_skip:
            hidden = hidden[:, tokens_to_skip:, :]
        return self.visual_projection(hidden)

    def _build_layerwise_bias(
        self,
        anatomical_masks: Optional[torch.Tensor],
        total_sequence_length: int,
        vision_prefix_length: int,
    ) -> Optional[torch.Tensor]:
        if anatomical_masks is None:
            return None
        if getattr(self.config, "attention_bias_mode", "layerwise") == "gaussian_legacy":
            vision_key_tokens = max(1, min(int(vision_prefix_length), int(total_sequence_length)))
            legacy_bias = build_legacy_gaussian_attention_bias(
                masks=anatomical_masks,
                num_layers=self.config.num_attention_layers,
                target_query_tokens=total_sequence_length,
                target_key_tokens=vision_key_tokens,
                base_kernel_size=self.config.layer_mask_base_kernel_size,
                kernel_growth=self.config.layer_mask_kernel_growth,
                strength=self.config.anatomical_attention_bias,
            )
            if vision_key_tokens == total_sequence_length:
                return legacy_bias
            full_bias = legacy_bias.new_zeros(
                legacy_bias.shape[0],
                legacy_bias.shape[1],
                total_sequence_length,
                total_sequence_length,
            )
            full_bias[:, :, :, :vision_key_tokens] = legacy_bias
            return full_bias
        return build_layerwise_attention_bias(
            masks=anatomical_masks,
            num_layers=self.config.num_attention_layers,
            target_tokens=total_sequence_length,
            base_kernel_size=self.config.layer_mask_base_kernel_size,
            kernel_growth=self.config.layer_mask_kernel_growth,
            strength=self.config.anatomical_attention_bias,
        )

    def _resolve_attention_bias(
        self,
        pixel_values: torch.Tensor,
        anatomical_masks: Optional[torch.Tensor],
        total_sequence_length: int,
        vision_prefix_length: int,
    ):
        if anatomical_masks is not None:
            return self._build_layerwise_bias(
                anatomical_masks,
                total_sequence_length=total_sequence_length,
                vision_prefix_length=vision_prefix_length,
            )
        if self.segmenter is None:
            return None
        if getattr(self.config, "attention_bias_mode", "layerwise") == "gaussian_legacy":
            combined_mask = self.segmenter.predict_mask(pixel_values)
            if combined_mask is None:
                logger.warning("Segmentation attention is enabled but no segmenter checkpoints were loaded; continuing without anatomical attention.")
                return None
            return self._build_layerwise_bias(
                combined_mask,
                total_sequence_length=total_sequence_length,
                vision_prefix_length=vision_prefix_length,
            )
        layerwise_bias = self.segmenter(
            pixel_values,
            num_layers=self.config.num_attention_layers,
            target_tokens=total_sequence_length,
            strength=self.config.anatomical_attention_bias,
        )
        if layerwise_bias is None:
            logger.warning("Segmentation attention is enabled but no segmenter checkpoints were loaded; continuing without anatomical attention.")
        return layerwise_bias

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        anatomical_masks: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> LanaModelOutput:
        vision_features = self._encode_images(pixel_values)
        batch_size, prefix_length, _ = vision_features.shape

        if input_ids is None:
            bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
            input_ids = torch.full((batch_size, 1), bos, device=vision_features.device, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
        elif attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        text_embeds = self.text_decoder.transformer.wte(input_ids)
        inputs_embeds = torch.cat([vision_features, text_embeds], dim=1)
        merged_attention_mask = torch.cat(
            [
                torch.ones((batch_size, prefix_length), device=attention_mask.device, dtype=attention_mask.dtype),
                attention_mask,
            ],
            dim=1,
        )

        merged_labels = None
        if labels is not None:
            ignore_prefix = torch.full((batch_size, prefix_length), -100, device=labels.device, dtype=labels.dtype)
            merged_labels = torch.cat([ignore_prefix, labels], dim=1)

        layerwise_bias = self._resolve_attention_bias(
            pixel_values=pixel_values,
            anatomical_masks=anatomical_masks,
            total_sequence_length=inputs_embeds.shape[1],
            vision_prefix_length=prefix_length,
        )
        decoder_outputs = self.text_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=merged_attention_mask,
            labels=merged_labels,
            segmentation_mask=layerwise_bias,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        return LanaModelOutput(
            loss=decoder_outputs.loss,
            logits=decoder_outputs.logits,
            attentions=decoder_outputs.attentions,
            layerwise_attentions=layerwise_bias,
            hidden_states=decoder_outputs.hidden_states,
            vision_features=vision_features,
        )

    @torch.inference_mode()
    def generate(
        self,
        pixel_values: torch.Tensor,
        anatomical_masks: Optional[torch.Tensor] = None,
        max_new_tokens: int = 150,
        **kwargs,
    ):
        vision_features = self._encode_images(pixel_values)
        batch_size = pixel_values.shape[0]
        if getattr(self.config, "generation_use_bos_token", True):
            bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
            start_tokens = torch.full((batch_size, 1), bos, device=pixel_values.device, dtype=torch.long)
            text_embeds = self.text_decoder.transformer.wte(start_tokens)
            inputs_embeds = torch.cat([vision_features, text_embeds], dim=1)
            attention_mask = torch.ones(inputs_embeds.shape[:2], device=pixel_values.device, dtype=torch.long)
        else:
            inputs_embeds = vision_features
            attention_mask = None

        layerwise_bias = self._resolve_attention_bias(
            pixel_values=pixel_values,
            anatomical_masks=anatomical_masks,
            total_sequence_length=inputs_embeds.shape[1] + max_new_tokens,
            vision_prefix_length=vision_features.shape[1],
        )
        generation_kwargs = dict(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,
            num_beams=1,
            segmentation_mask=layerwise_bias,
            use_cache=True,
        )
        if attention_mask is not None:
            generation_kwargs["attention_mask"] = attention_mask

        repetition_penalty = float(getattr(self.config, "generation_repetition_penalty", 1.0))
        if repetition_penalty > 1.0:
            generation_kwargs["repetition_penalty"] = repetition_penalty

        eos_token_id = self.tokenizer.eos_token_id
        if getattr(self.config, "generation_stop_on_eos", False):
            generation_kwargs["eos_token_id"] = eos_token_id
        else:
            generation_kwargs["eos_token_id"] = None
            generation_kwargs["forced_eos_token_id"] = None
            if eos_token_id is not None:
                generation_kwargs["suppress_tokens"] = [int(eos_token_id)]

        generation_kwargs.update(kwargs)
        return self.text_decoder.generate(**generation_kwargs)
