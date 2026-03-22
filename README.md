---
license: mit
library_name: transformers
pipeline_tag: image-to-text
tags:
  - medical-ai
  - radiology
  - chest-xray
  - report-generation
  - segmentation
  - anatomical-attention
metrics:
  - BLEU
  - METEOR
  - ROUGE
  - CIDEr
---

# LAnA

**Layer-Wise Anatomical Attention model**

[![ArXiv](https://img.shields.io/badge/ArXiv-2512.16841-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2512.16841)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devmuniz-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/devmuniz)
[![GitHub Profile](https://img.shields.io/badge/GitHub-devMuniz02-181717?logo=github&logoColor=white)](https://github.com/devMuniz02)
[![Portfolio](https://img.shields.io/badge/Portfolio-devmuniz02.github.io-0F172A?logo=googlechrome&logoColor=white)](https://devmuniz02.github.io/)
[![GitHub Repo](https://img.shields.io/badge/Repository-layer--wise--anatomical--attention-181717?logo=github&logoColor=white)](https://github.com/devMuniz02/layer-wise-anatomical-attention)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-manu02-FFD21E?logoColor=black)](https://huggingface.co/manu02)

![Layer-Wise Anatomical Attention](assets/AnatomicalAttention.gif)

## Status

- Project status: `Training in progress`
- Release status: `Research preview checkpoint`
- Current checkpoint status: `Not final`
- Training completion toward planned run: `20.24%` (`0.607` / `3` epochs)
- Current published metrics are intermediate and will change as training continues.

## Overview

LAnA is a medical report-generation project for chest X-ray images. The completed project is intended to generate radiology reports with a vision-language model guided by layer-wise anatomical attention built from predicted anatomical masks.

The architecture combines a DINOv3 vision encoder, lung and heart segmentation heads, and a GPT-2 decoder modified so each transformer layer receives a different anatomical attention bias derived from the segmentation mask.

## Intended Use

- Input: a chest X-ray image resized to `512x512` and normalized with ImageNet mean/std.
- Output: a generated radiology report.
- Best fit: research use, report-generation experiments, and anatomical-attention ablations.

## Data

- Full project datasets: CheXpert and MIMIC-CXR.
- Intended project scope: train on curated chest X-ray/report data from both datasets and evaluate on MIMIC-CXR test studies.
- Current released checkpoint datasets: `CheXpert, MIMIC-CXR` for training and `CheXpert, MIMIC-CXR` for validation.
- Current published evaluation: MIMIC-CXR test split, `frontal-only (PA/AP)` studies.

## Evaluation

- Text-generation metrics used in this project include BLEU, METEOR, ROUGE, and CIDEr.
- Medical report metrics implemented in the repository include RadGraph F1 and CheXpert F1.

## Training Snapshot

- Run: `full_3_epoch_mask_run`
- This section describes the current public checkpoint, not the final completed project.
- Method: `lora_adamw`
- Vision encoder: `facebook/dinov3-vits16-pretrain-lvd1689m`
- Text decoder: `gpt2`
- Segmentation encoder: `facebook/dinov3-convnext-small-pretrain-lvd1689m`
- Image size: `512`
- Local batch size: `1`
- Effective global batch size: `8`
- Scheduler: `cosine`
- Warmup steps: `5114`
- Weight decay: `0.01`
- Steps completed: `20702`
- Planned total steps: `102276`
- Images seen: `165639`
- Total training time: `5.1668` hours
- Hardware: `NVIDIA GeForce RTX 5070`
- Final train loss: `2.8459`
- Validation loss: `1.6727`

## MIMIC Test Results

Frontal-only evaluation using `PA/AP` studies only.

| Metric | Value |
| --- | --- |
| Number of studies | TBD |
| RadGraph F1 | TBD |
| CheXpert F1 micro | TBD |
| CheXpert F1 macro | TBD |

## Inference

### Option 1: Local `lana_radgen` package

Warning: this path only works if the repository code is available in your runtime environment.
In practice, run it from the project root or install the package so `lana_radgen` is importable.

```python
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from lana_radgen import LanaForConditionalGeneration

repo_id = "manu02/LAnA"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LanaForConditionalGeneration.from_pretrained(repo_id).to(device)
model.eval()

lung_ckpt = hf_hub_download(repo_id=repo_id, filename="segmenters/lung_segmenter_dinounet_finetuned.pth")
heart_ckpt = hf_hub_download(repo_id=repo_id, filename="segmenters/heart_segmenter_dinounet_best.pth")
print(lung_ckpt, heart_ckpt)

image_path = Path("example.png")
image = Image.open(image_path).convert("RGB")

# If the input image is not already 512x512, resize it before inference.
image = image.resize((512, 512), resample=Image.BICUBIC)
array = np.asarray(image, dtype=np.float32) / 255.0
pixel_values = torch.from_numpy(array).permute(2, 0, 1)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
pixel_values = ((pixel_values - mean) / std).unsqueeze(0).to(device)

with torch.no_grad():
    generated = model.generate(pixel_values=pixel_values, max_new_tokens=128)

report = model.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
print(report)
```

### Option 2: Hugging Face `AutoModel` with remote code

Use this if you do not want to import `lana_radgen` locally.
Because LAnA has custom architecture code, this path requires `trust_remote_code=True`.

```python
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

repo_id = "manu02/LAnA"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModel.from_pretrained(repo_id, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
model.eval()

lung_ckpt = hf_hub_download(repo_id=repo_id, filename="segmenters/lung_segmenter_dinounet_finetuned.pth")
heart_ckpt = hf_hub_download(repo_id=repo_id, filename="segmenters/heart_segmenter_dinounet_best.pth")
print(lung_ckpt, heart_ckpt)

image_path = Path("example.png")
image = Image.open(image_path).convert("RGB")
image = image.resize((512, 512), resample=Image.BICUBIC)
array = np.asarray(image, dtype=np.float32) / 255.0
pixel_values = torch.from_numpy(array).permute(2, 0, 1)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
pixel_values = ((pixel_values - mean) / std).unsqueeze(0).to(device)

with torch.no_grad():
    generated = model.generate(pixel_values=pixel_values, max_new_tokens=128)

report = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
print(report)
```

## Notes

- `segmenters/` contains the lung and heart segmentation checkpoints used to build anatomical attention masks.
- `evaluations/mimic_test_metrics.json` contains the latest saved MIMIC test metrics.

<!-- EVAL_RESULTS_START -->
## Latest Evaluation

- Dataset: `MIMIC-CXR test`
- View filter: `frontal-only (PA/AP)`
- Number of examples: `3041`
- CheXpert F1 micro: `0.1375`
- CheXpert F1 macro: `0.0923`
- RadGraph F1: `0.0847`
- RadGraph entity F1: `0.1524`
- RadGraph relation F1: `0.1304`
- RadGraph available: `True`
- RadGraph error: `None`

- Evaluation file: `evaluations/mimic_test_metrics.json`
- Predictions file: `evaluations/mimic_test_predictions.csv`
<!-- EVAL_RESULTS_END -->

<!-- MIMIC_TEST_RESULTS_START -->
## MIMIC Test Results

Frontal-only evaluation using `PA/AP` studies only. Number of evaluated studies: `3041`.

| Metric | Value |
| --- | --- |
| RadGraph F1 | `0.0847` |
| CheXpert F1 micro | `0.1375` |
| CheXpert F1 macro | `0.0923` |
<!-- MIMIC_TEST_RESULTS_END -->
