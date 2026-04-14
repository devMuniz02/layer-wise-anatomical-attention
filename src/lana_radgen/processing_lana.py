from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer, GPT2Tokenizer
from transformers.processing_utils import ProcessorMixin

from .image_processing_lana import LanaImageProcessor


class LanaProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "LanaImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        super().__init__(image_processor, tokenizer, **kwargs)

    def __call__(self, images=None, text=None, **kwargs):
        if images is None and text is None:
            raise ValueError("LanaProcessor expected `images`, `text`, or both.")

        encoded = {}
        if images is not None:
            encoded.update(self.image_processor(images=images, **kwargs))
        if text is not None:
            encoded.update(self.tokenizer(text, **kwargs))
        return encoded

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        kwargs = dict(kwargs)
        kwargs.pop("trust_remote_code", None)
        image_processor = LanaImageProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        source = Path(str(pretrained_model_name_or_path))
        if source.exists():
            tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=True,
                use_fast=False,
                **kwargs,
            )
        return cls(image_processor=image_processor, tokenizer=tokenizer)
