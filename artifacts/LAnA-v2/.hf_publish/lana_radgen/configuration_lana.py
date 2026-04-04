from transformers import PretrainedConfig


class LanaConfig(PretrainedConfig):
    model_type = "lana_radgen"

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
        vocab_size: int = 50257,
        layer_mask_base_kernel_size: int = 3,
        layer_mask_kernel_growth: int = 2,
        anatomical_attention_bias: float = 2.0,
        use_segmentation_mask: bool = True,
        segmentation_model_name: str = "facebook/dinov3-convnext-small-pretrain-lvd1689m",
        segmentation_attention_implementation: str = "sdpa",
        freeze_segmenter: bool = True,
        lung_segmenter_checkpoint: str = "",
        heart_segmenter_checkpoint: str = "",
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
        self.vocab_size = vocab_size
        self.layer_mask_base_kernel_size = layer_mask_base_kernel_size
        self.layer_mask_kernel_growth = layer_mask_kernel_growth
        self.anatomical_attention_bias = anatomical_attention_bias
        self.use_segmentation_mask = use_segmentation_mask
        self.segmentation_model_name = segmentation_model_name
        self.segmentation_attention_implementation = segmentation_attention_implementation
        self.freeze_segmenter = freeze_segmenter
        self.lung_segmenter_checkpoint = lung_segmenter_checkpoint
        self.heart_segmenter_checkpoint = heart_segmenter_checkpoint
        self.use_cache = use_cache
        self.decoder_load_in_4bit = decoder_load_in_4bit
        self.decoder_compute_dtype = decoder_compute_dtype
        super().__init__(**kwargs)
