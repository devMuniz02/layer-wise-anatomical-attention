from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import PretrainedConfig


class LanaConfig(PretrainedConfig):
    model_type = "lana_radgen"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        loaded = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if isinstance(loaded, tuple):
            config, unused_kwargs = loaded
        else:
            config, unused_kwargs = loaded, None
        repo_path = str(pretrained_model_name_or_path)
        if not Path(repo_path).exists():
            try:
                repo_path = snapshot_download(repo_path)
            except Exception:
                repo_path = str(pretrained_model_name_or_path)
        config.local_repo_path = repo_path
        if unused_kwargs is not None:
            return config, unused_kwargs
        return config

    def __init__(
        self,
        vision_model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        text_model_name: str = "gpt2",
        image_size: int = 512,
        mask_size: int = 32,
        num_attention_layers: int = 12,
        max_position_embeddings: int = 2048,
        visual_feature_dim: int = 384,
        text_hidden_size: int = 768,
        visual_projection_type: str = "mlp4",
        vocab_size: int = 50257,
        layer_mask_base_kernel_size: int = 3,
        layer_mask_kernel_growth: int = 2,
        anatomical_attention_bias: float = 2.0,
        attention_bias_mode: str = "layerwise",
        vision_feature_prefix_tokens_to_skip: int = 1,
        use_segmentation_mask: bool = True,
        segmentation_model_name: str = "facebook/dinov3-convnext-small-pretrain-lvd1689m",
        segmentation_attention_implementation: str = "sdpa",
        freeze_segmenter: bool = True,
        generation_use_bos_token: bool = True,
        generation_stop_on_eos: bool = False,
        generation_repetition_penalty: float = 1.0,
        lung_segmenter_checkpoint: str = "",
        heart_segmenter_checkpoint: str = "",
        bundled_vision_model_name: str = "",
        bundled_segmentation_model_name: str = "",
        bundled_text_model_name: str = "",
        bundled_tokenizer_name: str = "",
        segmenter_weights_in_model_state: bool = False,
        local_repo_path: str = "",
        use_cache: bool = True,
        decoder_load_in_4bit: bool = False,
        decoder_compute_dtype: str = "float16",
        **kwargs,
    ):
        self.vision_model_name = vision_model_name
        self.text_model_name = text_model_name
        self.image_size = image_size
        self.mask_size = mask_size
        self.num_attention_layers = num_attention_layers
        self.max_position_embeddings = max_position_embeddings
        self.visual_feature_dim = visual_feature_dim
        self.text_hidden_size = text_hidden_size
        self.visual_projection_type = visual_projection_type
        self.vocab_size = vocab_size
        self.layer_mask_base_kernel_size = layer_mask_base_kernel_size
        self.layer_mask_kernel_growth = layer_mask_kernel_growth
        self.anatomical_attention_bias = anatomical_attention_bias
        self.attention_bias_mode = attention_bias_mode
        self.vision_feature_prefix_tokens_to_skip = vision_feature_prefix_tokens_to_skip
        self.use_segmentation_mask = use_segmentation_mask
        self.segmentation_model_name = segmentation_model_name
        self.segmentation_attention_implementation = segmentation_attention_implementation
        self.freeze_segmenter = freeze_segmenter
        self.generation_use_bos_token = generation_use_bos_token
        self.generation_stop_on_eos = generation_stop_on_eos
        self.generation_repetition_penalty = generation_repetition_penalty
        self.lung_segmenter_checkpoint = lung_segmenter_checkpoint
        self.heart_segmenter_checkpoint = heart_segmenter_checkpoint
        self.bundled_vision_model_name = bundled_vision_model_name
        self.bundled_segmentation_model_name = bundled_segmentation_model_name
        self.bundled_text_model_name = bundled_text_model_name
        self.bundled_tokenizer_name = bundled_tokenizer_name
        self.segmenter_weights_in_model_state = segmenter_weights_in_model_state
        self.local_repo_path = local_repo_path
        self.use_cache = use_cache
        self.decoder_load_in_4bit = decoder_load_in_4bit
        self.decoder_compute_dtype = decoder_compute_dtype
        super().__init__(**kwargs)
