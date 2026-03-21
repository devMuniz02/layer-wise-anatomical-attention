import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel

from .attention import build_layerwise_attention_bias
from .configuration_lana import LanaConfig
from .gpt2_modified import create_decoder
from .modeling_outputs import LanaModelOutput
from .segmenters import AnatomicalSegmenter

logger = logging.getLogger(__name__)


class LanaForConditionalGeneration(PreTrainedModel):
    config_class = LanaConfig
    base_model_prefix = "lana"
    supports_gradient_checkpointing = True

    def __init__(self, config: LanaConfig):
        super().__init__(config)
        vision_config = AutoConfig.from_pretrained(config.vision_model_name, trust_remote_code=True)
        if getattr(vision_config, "hidden_size", None) is not None:
            config.visual_feature_dim = vision_config.hidden_size

        self.vision_encoder = AutoModel.from_pretrained(config.vision_model_name, trust_remote_code=True)
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
            text_model_name=config.text_model_name,
            attention_implementation=config.segmentation_attention_implementation,
            max_position_embeddings=config.max_position_embeddings,
            **decoder_kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        config.vocab_size = self.text_decoder.config.vocab_size
        config.text_hidden_size = self.text_decoder.config.hidden_size
        config.num_attention_layers = self.text_decoder.config.n_layer

        self.visual_projection = nn.Linear(config.visual_feature_dim, config.text_hidden_size)
        self.segmenter = None
        if config.use_segmentation_mask:
            self.segmenter = AnatomicalSegmenter(
                model_name=config.segmentation_model_name,
                freeze=config.freeze_segmenter,
                lung_checkpoint=config.lung_segmenter_checkpoint,
                heart_checkpoint=config.heart_segmenter_checkpoint,
            )
        self.post_init()

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
        if hidden.shape[1] > 1:
            hidden = hidden[:, 1:, :]
        return self.visual_projection(hidden)

    def _build_layerwise_bias(self, anatomical_masks: Optional[torch.Tensor], total_sequence_length: int) -> Optional[torch.Tensor]:
        if anatomical_masks is None:
            return None
        return build_layerwise_attention_bias(
            masks=anatomical_masks,
            num_layers=self.config.num_attention_layers,
            target_tokens=total_sequence_length,
            base_kernel_size=self.config.layer_mask_base_kernel_size,
            kernel_growth=self.config.layer_mask_kernel_growth,
            strength=self.config.anatomical_attention_bias,
        )

    def _resolve_attention_bias(self, pixel_values: torch.Tensor, anatomical_masks: Optional[torch.Tensor], total_sequence_length: int):
        if anatomical_masks is not None:
            return self._build_layerwise_bias(anatomical_masks, total_sequence_length=total_sequence_length)
        if self.segmenter is None:
            return None
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
        max_new_tokens: int = 128,
        **kwargs,
    ):
        vision_features = self._encode_images(pixel_values)
        batch_size = pixel_values.shape[0]
        bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        start_tokens = torch.full((batch_size, 1), bos, device=pixel_values.device, dtype=torch.long)
        text_embeds = self.text_decoder.transformer.wte(start_tokens)
        inputs_embeds = torch.cat([vision_features, text_embeds], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=pixel_values.device, dtype=torch.long)

        layerwise_bias = self._resolve_attention_bias(
            pixel_values=pixel_values,
            anatomical_masks=anatomical_masks,
            total_sequence_length=inputs_embeds.shape[1] + max_new_tokens,
        )
        return self.text_decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            segmentation_mask=layerwise_bias,
            use_cache=True,
            **kwargs,
        )
