import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)


@dataclass
class ReportSample:
    image_path: str
    text: str
    mask_path: Optional[str] = None


class ResizeCachedReportDataset(Dataset):
    def __init__(
        self,
        manifest,
        tokenizer,
        image_mean=None,
        image_std=None,
        image_size: Optional[int] = None,
        max_text_length: Optional[int] = None,
        resize_loaded_images: bool = True,
        prepend_bos_token: bool = True,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.image_mean = torch.tensor(image_mean or [0.485, 0.456, 0.406]).view(3, 1, 1)
        self.image_std = torch.tensor(image_std or [0.229, 0.224, 0.225]).view(3, 1, 1)
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.resize_loaded_images = resize_loaded_images
        bos_token_id = getattr(tokenizer, "bos_token_id", None)
        self.bos_token_id = bos_token_id if prepend_bos_token else None
        self._tokenized_reports = [self._tokenize_report(str(text)) for text in self.manifest["report_text"].tolist()]

    def __len__(self) -> int:
        return len(self.manifest)

    def _load_png_tensor(self, path: str) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        if self.resize_loaded_images and self.image_size is not None:
            image = image.resize((self.image_size, self.image_size), resample=Image.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - self.image_mean) / self.image_std

    def _load_mask(self, path: Optional[str], fallback_shape) -> torch.Tensor:
        if not path:
            return torch.ones(1, fallback_shape[-2], fallback_shape[-1], dtype=torch.float32)
        image = Image.open(path).convert("L")
        if self.resize_loaded_images and self.image_size is not None:
            image = image.resize((self.image_size, self.image_size), resample=Image.NEAREST)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)

    def _tokenize_report(self, report_text: str) -> Dict[str, torch.Tensor]:
        normalized = report_text.rstrip() + "\n"
        token_limit = self.max_text_length
        reserve = int(self.bos_token_id is not None)
        if token_limit is not None:
            token_limit = max(1, token_limit - reserve)
        tokenized = self.tokenizer(
            normalized,
            truncation=token_limit is not None,
            max_length=token_limit,
            padding=False,
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        prefix = []
        if self.bos_token_id is not None:
            prefix.append(self.bos_token_id)

        if prefix:
            prefix_tensor = torch.tensor(prefix, dtype=input_ids.dtype)
            input_ids = torch.cat([prefix_tensor, input_ids], dim=0)
            attention_mask = torch.cat(
                [torch.ones(len(prefix), dtype=attention_mask.dtype), attention_mask],
                dim=0,
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        dataset_size = len(self.manifest)
        for offset in range(dataset_size):
            current_idx = (idx + offset) % dataset_size
            row = self.manifest.iloc[current_idx]
            try:
                pixel_values = self._load_png_tensor(row["processed_image_path"])
                mask = self._load_mask(row.get("processed_mask_path"), pixel_values.shape)
            except (FileNotFoundError, OSError, UnidentifiedImageError, ValueError) as exc:
                LOGGER.warning("Skipping unreadable sample at %s: %s", row.get("processed_image_path", "<unknown>"), exc)
                continue
            tokenized = self._tokenized_reports[current_idx]
            return {
                "pixel_values": pixel_values,
                "anatomical_masks": mask,
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "report_id": row.get("report_id", current_idx),
            }
        raise RuntimeError("No readable image samples remain in the dataset.")
