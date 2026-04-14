from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import MethodType

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_model
from transformers import GPT2Tokenizer, PreTrainedModel

try:
    from .configuration_lana_arxiv import LanaArxivConfig
except ImportError:  # pragma: no cover - supports flattened HF remote-code layout
    from configuration_lana_arxiv import LanaArxivConfig


class LanaArxivForConditionalGeneration(PreTrainedModel):
    config_class = LanaArxivConfig
    base_model_prefix = "legacy_model"
    main_input_name = "pixel_values"

    def __init__(self, config: LanaArxivConfig):
        super().__init__(config)
        self.legacy_model = None
        self.tokenizer = None
        self._runtime_device = torch.device("cpu")

    def _resolve_repo_path(self, pretrained_model_name_or_path) -> Path:
        repo_path = Path(str(pretrained_model_name_or_path))
        if repo_path.exists():
            return repo_path.resolve()
        return Path(snapshot_download(str(pretrained_model_name_or_path))).resolve()

    def _load_legacy_model(self, repo_path: Path) -> None:
        candidate_paths = [
            repo_path / "utils" / "complete_model.py",
            repo_path / "utils" / "models" / "complete_model.py",
        ]
        complete_model_path = next((path for path in candidate_paths if path.exists()), candidate_paths[0])
        if not complete_model_path.exists():
            raise FileNotFoundError(f"Expected legacy model loader at {complete_model_path}")

        weights_path = repo_path / self.config.source_weight_name
        if not weights_path.exists():
            raise FileNotFoundError(f"Expected legacy report weights at {weights_path}")

        added_to_syspath = False
        repo_path_str = str(repo_path)
        if repo_path_str not in sys.path:
            sys.path.insert(0, repo_path_str)
            added_to_syspath = True
        try:
            os.environ["LANA_ARXIV_REPO_ROOT"] = str(repo_path)
            for module_name in [name for name in list(sys.modules) if name == "utils" or name.startswith("utils.")]:
                del sys.modules[module_name]

            legacy_spec = importlib.util.spec_from_file_location("lana_arxiv_legacy_complete_model", complete_model_path)
            legacy_module = importlib.util.module_from_spec(legacy_spec)
            legacy_spec.loader.exec_module(legacy_module)

            legacy_model = legacy_module.create_complete_model(device="cpu", attention_implementation="eager")
            decoder = getattr(legacy_model, "decoder", None)
            transformer = getattr(decoder, "transformer", None)
            if decoder is not None and not hasattr(decoder, "model_parallel"):
                decoder.model_parallel = False
            if transformer is not None and not hasattr(transformer, "get_head_mask"):
                def _legacy_get_head_mask(this, head_mask, num_hidden_layers, is_attention_chunked: bool = False):
                    if head_mask is None:
                        return [None] * num_hidden_layers
                    return head_mask

                transformer.get_head_mask = MethodType(_legacy_get_head_mask, transformer)
            if transformer is not None and not hasattr(transformer, "model_parallel"):
                transformer.model_parallel = False

            if weights_path.suffix.lower() == ".safetensors":
                load_model(legacy_model, str(weights_path))
            else:
                checkpoint = torch.load(weights_path, map_location="cpu")
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    checkpoint = checkpoint["model_state_dict"]
                if not isinstance(checkpoint, dict):
                    raise TypeError(f"Unsupported checkpoint payload in {weights_path}")
                model_state = legacy_model.state_dict()
                compatible_checkpoint = {}
                for key, value in checkpoint.items():
                    if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
                        compatible_checkpoint[key] = value
                legacy_model.load_state_dict(compatible_checkpoint, strict=False)
            tokenizer = GPT2Tokenizer.from_pretrained(str(repo_path))
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            legacy_model.tokenizer = tokenizer
            legacy_model.pad_token_id = tokenizer.pad_token_id

            self.legacy_model = legacy_model
            self.tokenizer = tokenizer
        finally:
            os.environ.pop("LANA_ARXIV_REPO_ROOT", None)
            if added_to_syspath:
                sys.path.remove(repo_path_str)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, config=None, **kwargs):
        kwargs = dict(kwargs)
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("state_dict", None)
        kwargs.pop("ignore_mismatched_sizes", None)
        kwargs.pop("low_cpu_mem_usage", None)
        if config is None:
            config = LanaArxivConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = cls(config)
        repo_path = model._resolve_repo_path(getattr(config, "local_repo_path", "") or pretrained_model_name_or_path)
        model._load_legacy_model(repo_path)
        return model

    def move_non_quantized_modules(self, device: torch.device) -> None:
        if self.legacy_model is None:
            return
        self._runtime_device = torch.device(device)
        self.legacy_model.to(self._runtime_device)
        if hasattr(self.legacy_model, "device"):
            self.legacy_model.device = self._runtime_device

    def eval(self):
        super().eval()
        if self.legacy_model is not None:
            self.legacy_model.eval()
        return self

    def forward(self, pixel_values: torch.Tensor, **kwargs):
        if self.legacy_model is None:
            raise RuntimeError("Legacy model is not loaded. Use from_pretrained() to initialize the wrapper.")
        pixel_values = pixel_values.to(self._runtime_device)
        return self.legacy_model(pixel_values=pixel_values, **kwargs)

    @torch.inference_mode()
    def generate(self, pixel_values: torch.Tensor, max_new_tokens: int = 100, **kwargs):
        if self.legacy_model is None:
            raise RuntimeError("Legacy model is not loaded. Use from_pretrained() to initialize the wrapper.")
        pixel_values = pixel_values.to(self._runtime_device)
        generated_ids, _, _ = self.legacy_model.generate(
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            output_attentions=bool(kwargs.pop("output_attentions", False)),
        )
        return generated_ids
