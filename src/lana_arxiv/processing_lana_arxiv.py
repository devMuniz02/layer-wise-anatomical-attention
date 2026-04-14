from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import GPT2Tokenizer
from transformers.processing_utils import ProcessorMixin
import torch

try:
    from .image_processing_lana_arxiv import LanaArxivImageProcessor
except ImportError:  # pragma: no cover - supports flattened HF remote-code layout
    from image_processing_lana_arxiv import LanaArxivImageProcessor


class LanaArxivProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "LanaArxivImageProcessor"
    tokenizer_class = "GPT2Tokenizer"

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        super().__init__(image_processor, tokenizer, **kwargs)

    def __call__(self, images=None, text=None, **kwargs):
        if images is None and text is None:
            raise ValueError("LanaArxivProcessor expected `images`, `text`, or both.")

        encoded = {}
        if images is not None:
            encoded.update(self.image_processor(images=images, **kwargs))
        if text is not None:
            encoded.update(self.tokenizer(text, **kwargs))
        return encoded

    def _normalize_ids(self, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.ndim == 0:
                return [int(value.item())]
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return [self._normalize_ids(item) for item in value]
        return value

    def batch_decode(self, *args, **kwargs):
        normalized_args = list(args)
        if normalized_args:
            normalized_args[0] = self._normalize_ids(normalized_args[0])
        return self.tokenizer.batch_decode(*normalized_args, **kwargs)

    def decode(self, *args, **kwargs):
        normalized_args = list(args)
        if normalized_args:
            normalized_args[0] = self._normalize_ids(normalized_args[0])
        return self.tokenizer.decode(*normalized_args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        kwargs = dict(kwargs)
        kwargs.pop("trust_remote_code", None)
        image_processor = LanaArxivImageProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        source = Path(str(pretrained_model_name_or_path))
        if not source.exists():
            source = Path(snapshot_download(str(pretrained_model_name_or_path), local_files_only=bool(kwargs.get("local_files_only", False))))
        tokenizer_source = source
        if not ((tokenizer_source / "tokenizer.json").exists() or (tokenizer_source / "vocab.json").exists()):
            tokenizer_source = source / "bundled_backbones" / "text_decoder"
        if not ((tokenizer_source / "tokenizer.json").exists() or (tokenizer_source / "vocab.json").exists()):
            raise FileNotFoundError(f"Expected GPT-2 tokenizer files under {tokenizer_source}")
        tokenizer = GPT2Tokenizer.from_pretrained(
            str(tokenizer_source),
            local_files_only=bool(kwargs.get("local_files_only", False)),
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return cls(image_processor=image_processor, tokenizer=tokenizer)
