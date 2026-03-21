from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class ReportSample:
    image_path: str
    text: str
    mask_path: Optional[str] = None


class ResizeCachedReportDataset(Dataset):
    def __init__(self, manifest, tokenizer, image_mean=None, image_std=None, image_size: Optional[int] = None):
        self.manifest = manifest.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.image_mean = torch.tensor(image_mean or [0.485, 0.456, 0.406]).view(3, 1, 1)
        self.image_std = torch.tensor(image_std or [0.229, 0.224, 0.225]).view(3, 1, 1)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.manifest)

    def _load_png_tensor(self, path: str) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        if self.image_size is not None:
            image = image.resize((self.image_size, self.image_size), resample=Image.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - self.image_mean) / self.image_std

    def _load_mask(self, path: Optional[str], fallback_shape) -> torch.Tensor:
        if not path:
            return torch.ones(1, fallback_shape[-2], fallback_shape[-1], dtype=torch.float32)
        image = Image.open(path).convert("L")
        if self.image_size is not None:
            image = image.resize((self.image_size, self.image_size), resample=Image.NEAREST)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.manifest.iloc[idx]
        pixel_values = self._load_png_tensor(row["processed_image_path"])
        mask = self._load_mask(row.get("processed_mask_path"), pixel_values.shape)
        tokenized = self.tokenizer(
            row["report_text"],
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "anatomical_masks": mask,
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "report_id": row.get("report_id", idx),
        }
