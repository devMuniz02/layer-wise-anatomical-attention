from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import PretrainedConfig


class LanaArxivConfig(PretrainedConfig):
    model_type = "lana_arxiv"

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
        segmentation_model_name: str = "facebook/dinov3-convnext-small-pretrain-lvd1689m",
        image_size: int = 512,
        source_space_repo_id: str = "manu02/LAnA-Arxiv",
        source_space_revision: str = "arxiv paper checkpoint",
        source_weight_name: str = "arxiv_paper_model.pth",
        generation_repetition_penalty: float = 1.2,
        generation_stop_on_eos: bool = True,
        vision_feature_prefix_tokens_to_skip: int = 5,
        local_repo_path: str = "",
        **kwargs,
    ):
        self.vision_model_name = vision_model_name
        self.text_model_name = text_model_name
        self.segmentation_model_name = segmentation_model_name
        self.image_size = image_size
        self.source_space_repo_id = source_space_repo_id
        self.source_space_revision = source_space_revision
        self.source_weight_name = source_weight_name
        self.generation_repetition_penalty = generation_repetition_penalty
        self.generation_stop_on_eos = generation_stop_on_eos
        self.vision_feature_prefix_tokens_to_skip = vision_feature_prefix_tokens_to_skip
        self.local_repo_path = local_repo_path
        super().__init__(**kwargs)
