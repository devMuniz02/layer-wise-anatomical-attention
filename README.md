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
- Training completion toward planned run: `48.46%` (`1.454` / `3` epochs)
- Current published metrics are intermediate and will change as training continues.

## Overview

LAnA is a medical report-generation project for chest X-ray images. The completed project is intended to generate radiology reports with a vision-language model guided by layer-wise anatomical attention built from predicted anatomical masks.

The architecture combines a DINOv3 vision encoder, lung and heart segmentation heads, and a GPT-2 decoder modified so each transformer layer receives a different anatomical attention bias derived from the segmentation mask.

## How to Run

For local inference instructions, go to the [Inference](#inference) section.

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
- Medical report metrics implemented in the repository include RadGraph F1 and CheXpert F1 (`14-micro`, `5-micro`, `14-macro`, `5-macro`).

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
- Steps completed: `49562`
- Planned total steps: `102276`
- Images seen: `396535`
- Total training time: `11.6893` hours
- Hardware: `NVIDIA GeForce RTX 5070`
- Final train loss: `1.3791`
- Validation loss: `1.4219`

## MIMIC Test Results

Frontal-only evaluation using `PA/AP` studies only.

### Current Checkpoint Results

| Metric | Value |
| --- | --- |
| Number of studies | `3041` |
| RadGraph F1 | `0.0966` |
| RadGraph entity F1 | `0.1455` |
| RadGraph relation F1 | `0.1320` |
| CheXpert F1 14-micro | `0.2313` |
| CheXpert F1 5-micro | `0.2747` |
| CheXpert F1 14-macro | `0.1357` |
| CheXpert F1 5-macro | `0.2098` |

### Final Completed Training Results

The final table will be populated when the planned training run is completed. Until then, final-report metrics remain `TBD`.

| Metric | Value |
| --- | --- |
| Number of studies | TBD |
| RadGraph F1 | TBD |
| RadGraph entity F1 | TBD |
| RadGraph relation F1 | TBD |
| CheXpert F1 14-micro | TBD |
| CheXpert F1 5-micro | TBD |
| CheXpert F1 14-macro | TBD |
| CheXpert F1 5-macro | TBD |

## Inference

Standard `AutoModel.from_pretrained(..., trust_remote_code=True)` loading is currently blocked for this repo because the custom model constructor performs nested pretrained submodel loads.
Use the verified manual load path below instead: download the HF repo snapshot, import the downloaded package, and load the exported `model.safetensors` directly.

```python
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

repo_dir = Path(snapshot_download("manu02/LAnA"))
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
    generated = model.generate(pixel_values=pixel_values, max_new_tokens=128)

report = model.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
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
- CheXpert F1 14-micro: `0.2313`
- CheXpert F1 5-micro: `0.2747`
- CheXpert F1 14-macro: `0.1357`
- CheXpert F1 5-macro: `0.2098`
- RadGraph F1: `0.0966`
- RadGraph entity F1: `0.1455`
- RadGraph relation F1: `0.1320`
- RadGraph available: `True`
- RadGraph error: `None`

- Evaluation file: `evaluations/mimic_test_metrics.json`
- Predictions file: `evaluations/mimic_test_predictions.csv`
<!-- EVAL_RESULTS_END -->
