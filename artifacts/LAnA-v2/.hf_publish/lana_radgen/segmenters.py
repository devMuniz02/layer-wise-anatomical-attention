import logging
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel

from .attention.layerwise_anatomical_attention import build_layerwise_attention_bias

LOGGER = logging.getLogger(__name__)


def _freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


class _DinoUNetLung(nn.Module):
    def __init__(self, model_name: str, freeze: bool = True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.channel_adapter = nn.Conv2d(768, 512, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )
        if freeze:
            _freeze_module(self)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_feats = self.encoder(x, output_hidden_states=True, return_dict=True)
        feats = next(h for h in reversed(enc_feats.hidden_states) if isinstance(h, torch.Tensor) and h.ndim == 4)
        feats = self.channel_adapter(feats)
        pred = self.decoder(feats)
        return (torch.sigmoid(pred) > 0.5).float()


class _DinoUNetHeart(nn.Module):
    def __init__(self, model_name: str, freeze: bool = True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.adapter = nn.Conv2d(768, 512, 1)
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 1),
        )
        if freeze:
            _freeze_module(self)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(x, output_hidden_states=True, return_dict=True)
        feat = next(h for h in reversed(enc.hidden_states) if isinstance(h, torch.Tensor) and h.ndim == 4)
        feat = self.adapter(feat)
        logits = self.decoder(feat)
        pred = torch.argmax(logits, dim=1)
        return (pred == 2).unsqueeze(1).float()


class AnatomicalSegmenter(nn.Module):
    def __init__(
        self,
        model_name: str,
        freeze: bool = True,
        lung_checkpoint: str = "",
        heart_checkpoint: str = "",
    ):
        super().__init__()
        self.lung_model = _DinoUNetLung(model_name=model_name, freeze=freeze)
        self.heart_model = _DinoUNetHeart(model_name=model_name, freeze=freeze)
        self.loaded_lung_checkpoint = self._load_submodule(self.lung_model, lung_checkpoint, "lung")
        self.loaded_heart_checkpoint = self._load_submodule(self.heart_model, heart_checkpoint, "heart")

    @staticmethod
    def _load_submodule(module: nn.Module, checkpoint_path: str, label: str) -> bool:
        if not checkpoint_path:
            return False
        path = Path(checkpoint_path)
        if not path.exists():
            LOGGER.warning("Requested %s segmenter checkpoint does not exist: %s", label, path)
            return False
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        module.load_state_dict(state, strict=False)
        LOGGER.info("Loaded %s segmenter checkpoint from %s", label, path)
        return True

    @property
    def has_any_checkpoint(self) -> bool:
        return self.loaded_lung_checkpoint or self.loaded_heart_checkpoint

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor, num_layers: int, target_tokens: int, strength: float) -> torch.Tensor | None:
        if not self.has_any_checkpoint:
            return None

        masks = []
        if self.loaded_heart_checkpoint:
            masks.append(self.heart_model(pixel_values))
        if self.loaded_lung_checkpoint:
            masks.append(self.lung_model(pixel_values))
        if not masks:
            return None

        combined_mask = torch.clamp(sum(masks), 0.0, 1.0)
        return build_layerwise_attention_bias(
            masks=combined_mask,
            num_layers=num_layers,
            target_tokens=target_tokens,
            strength=strength,
        )
