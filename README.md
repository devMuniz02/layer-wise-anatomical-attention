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

## Overview

LAnA is a medical report-generation project for chest X-ray images. The completed project is intended to generate radiology reports with a vision-language model guided by layer-wise anatomical attention built from predicted anatomical masks.

The architecture combines a DINOv3 vision encoder, lung and heart segmentation heads, and a GPT-2 decoder modified so each transformer layer receives a different anatomical attention bias derived from the segmentation mask.

## How to Run

Standard `AutoModel.from_pretrained(..., trust_remote_code=True)` loading is currently blocked for this repo because the custom model constructor performs nested pretrained submodel loads.
Use the verified manual load path below instead: download the HF repo snapshot, import the downloaded package, and load the exported `model.safetensors` directly.
You must set an `HF_TOKEN` environment variable with permission to access the DINOv3 model repositories used by this project, otherwise the required vision backbones cannot be downloaded.

```python
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

repo_dir = Path(snapshot_download('manu02/LAnA-v4'))
sys.path.insert(0, str(repo_dir))

from lana_radgen import LanaConfig, LanaForConditionalGeneration

config = LanaConfig.from_pretrained(repo_dir)
config.lung_segmenter_checkpoint = str(repo_dir / "segmenters" / "lung_segmenter_dinounet_finetuned.pth")
config.heart_segmenter_checkpoint = str(repo_dir / "segmenters" / "heart_segmenter_dinounet_best.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LanaForConditionalGeneration(config)
state_dict = load_file(str(repo_dir / "model.safetensors"))
missing, unexpected = model.load_state_dict(state_dict, strict=True)
assert not missing and not unexpected

model.tokenizer = AutoTokenizer.from_pretrained(repo_dir, trust_remote_code=True)
model.move_non_quantized_modules(device)
model.eval()

image_path = Path("example.png")
image = Image.open(image_path).convert("RGB")
image = image.resize((512, 512), resample=Image.BICUBIC)
array = np.asarray(image, dtype=np.float32) / 255.0
pixel_values = torch.from_numpy(array).permute(2, 0, 1)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
pixel_values = ((pixel_values - mean) / std).unsqueeze(0).to(device)

with torch.no_grad():
    generated = model.generate(pixel_values=pixel_values, max_new_tokens=150)

report = model.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
print(report)
```

## Intended Use

- Input: a chest X-ray image resized to `512x512` and normalized with ImageNet mean/std.
- Output: a generated radiology report.
- Best fit: research use, report-generation experiments, and anatomical-attention ablations.

## MIMIC Test Results

Frontal-only evaluation using `PA/AP` studies only.

These comparison tables are refreshed across the full LAnA collection whenever any collection model is evaluated.

### Cross-Model Comparison: All Frontal Test Studies

| Metric | LAnA-MIMIC-CHEXPERT | LAnA-MIMIC | LAnA | LAnA-v2 | LAnA-v3 | LAnA-v4 |
| --- | --- | --- | --- | --- | --- | --- |
| Number of studies | `3041` | `3041` | `3041` | `3041` | `3041` | `3041` |
| ROUGE-L | `0.1513` | `0.1653` | `0.1686` | `0.1670` | `0.1745` | `0.1572` |
| BLEU-1 | `0.1707` | `0.1916` | `0.2091` | `0.2174` | `0.2346` | `0.2036` |
| BLEU-4 | `0.0357` | `0.0386` | `0.0417` | `0.0417` | `0.0484` | `0.0357` |
| METEOR | `0.2079` | `0.2202` | `0.2298` | `0.2063` | `0.2129` | `0.1704` |
| RadGraph F1 | `0.0918` | `0.0921` | `0.1024` | `0.1057` | `0.0939` | `0.0818` |
| RadGraph entity F1 | `0.1399` | `0.1459` | `0.1587` | `0.1569` | `0.1441` | `0.1278` |
| RadGraph relation F1 | `0.1246` | `0.1322` | `0.1443` | `0.1474` | `0.1280` | `0.1198` |
| CheXpert F1 14-micro | `0.1829` | `0.1565` | `0.2116` | `0.1401` | `0.3116` | `0.1111` |
| CheXpert F1 5-micro | `0.2183` | `0.1530` | `0.2512` | `0.2506` | `0.2486` | `0.0140` |
| CheXpert F1 14-macro | `0.1095` | `0.0713` | `0.1095` | `0.0401` | `0.1363` | `0.0439` |
| CheXpert F1 5-macro | `0.1634` | `0.1007` | `0.1644` | `0.1004` | `0.1686` | `0.0105` |

### Cross-Model Comparison: Findings-Only Frontal Test Studies

| Metric | LAnA-MIMIC-CHEXPERT | LAnA-MIMIC | LAnA | LAnA-v2 | LAnA-v3 | LAnA-v4 |
| --- | --- | --- | --- | --- | --- | --- |
| Number of studies | `2210` | `2210` | `2210` | `2210` | `2210` | `2210` |
| ROUGE-L | `0.1576` | `0.1720` | `0.1771` | `0.1771` | `0.1848` | `0.1668` |
| BLEU-1 | `0.1754` | `0.2003` | `0.2177` | `0.2263` | `0.2480` | `0.2206` |
| BLEU-4 | `0.0405` | `0.0449` | `0.0484` | `0.0487` | `0.0573` | `0.0423` |
| METEOR | `0.2207` | `0.2347` | `0.2466` | `0.2240` | `0.2310` | `0.1864` |
| RadGraph F1 | `0.1010` | `0.1000` | `0.1119` | `0.1181` | `0.1046` | `0.0942` |
| RadGraph entity F1 | `0.1517` | `0.1577` | `0.1713` | `0.1739` | `0.1584` | `0.1444` |
| RadGraph relation F1 | `0.1347` | `0.1413` | `0.1549` | `0.1628` | `0.1405` | `0.1351` |
| CheXpert F1 14-micro | `0.1651` | `0.1442` | `0.1907` | `0.1365` | `0.2921` | `0.1101` |
| CheXpert F1 5-micro | `0.2152` | `0.1716` | `0.2415` | `0.2455` | `0.2394` | `0.0158` |
| CheXpert F1 14-macro | `0.1047` | `0.0700` | `0.1039` | `0.0381` | `0.1326` | `0.0435` |
| CheXpert F1 5-macro | `0.1611` | `0.1112` | `0.1578` | `0.0952` | `0.1636` | `0.0124` |

## Data

- Full project datasets: CheXpert and MIMIC-CXR.
- Intended project scope: train on curated chest X-ray/report data from both datasets and evaluate on MIMIC-CXR test studies.
- Current released checkpoint datasets: `MIMIC-CXR (findings-only)` for training and `MIMIC-CXR (findings-only)` for validation.
- Current published evaluation: MIMIC-CXR test split, `frontal-only (PA/AP)` studies.

## Evaluation

- Medical report metrics implemented in the repository include RadGraph F1 and CheXpert F1 (`14-micro`, `5-micro`, `14-macro`, `5-macro`).

## Training Snapshot

- Run: `LAnA-v4`
- This section describes the current public checkpoint, not the final completed project.
- Method: `full_adamw`
- Vision encoder: `facebook/dinov3-vits16-pretrain-lvd1689m`
- Text decoder: `gpt2`
- Visual projection: `linear`
- Segmentation encoder: `facebook/dinov3-convnext-small-pretrain-lvd1689m`
- Image size: `512`
- Local batch size: `1`
- Effective global batch size: `128`
- Scheduler: `cosine`
- Warmup steps: `165`
- Weight decay: `0.01`
- Steps completed: `1457`
- Planned total steps: `3297`
- Images seen: `186757`
- Total training time: `3.5000` hours
- Hardware: `NVIDIA GeForce RTX 5070`
- Final train loss: `1.1516`
- Validation loss: `1.7666`

## Status

- Project status: `Training in progress`
- Release status: `Research preview checkpoint`
- Current checkpoint status: `Not final`
- Training completion toward planned run: `44.29%` (`1` / `3` epochs)
- Current published metrics are intermediate and will change as training continues.

## Notes

- Set `HF_TOKEN` with permission to access the DINOv3 repositories required by this model before downloading or running inference.
- `segmenters/` contains the lung and heart segmentation checkpoints used to build anatomical attention masks.
- `evaluations/mimic_test_metrics.json` contains the latest saved MIMIC test metrics.
