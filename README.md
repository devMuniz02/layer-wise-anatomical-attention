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

> Best current model in this collection: [`manu02/LAnA-v5`](https://huggingface.co/manu02/LAnA-v5)

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

## Intended Use

- Input: a chest X-ray image resized to `512x512` and normalized with ImageNet mean/std.
- Output: a generated radiology report.
- Best fit: research use, report-generation experiments, and anatomical-attention ablations.

## How to Run

New users should prefer the standard Hugging Face flow below.
The legacy snapshot/manual implementation lives on the `snapshot-legacy` branch for backward compatibility.

### Implementation 1: Standard Hugging Face loading

```python
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

repo_id = "manu02/LAnA-v5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
model.move_non_quantized_modules(device)
model.eval()

image = Image.open("example.png").convert("RGB")
inputs = processor(images=image, return_tensors="pt")
inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

with torch.inference_mode():
    generated = model.generate(**inputs, max_new_tokens=150)

report = processor.batch_decode(generated, skip_special_tokens=True)[0]
print(report)
```

Batched inference uses the same path:

```python
batch = processor(images=[image_a, image_b], return_tensors="pt")
batch = {name: tensor.to(device) for name, tensor in batch.items()}
generated = model.generate(**batch, max_new_tokens=150)
reports = processor.batch_decode(generated, skip_special_tokens=True)
```

`HF_TOKEN` is optional for this public standard-loading path. If you do not set one, the model still loads,
but Hugging Face may show lower-rate-limit warnings.

### Legacy snapshot branch

Use the snapshot/manual branch only if you specifically need the older import-based workflow:

- Branch: [`snapshot-legacy`](https://huggingface.co/manu02/LAnA-v5/tree/snapshot-legacy)
- Download example: `snapshot_download("manu02/LAnA-v5", revision="snapshot-legacy")`

## Licensing and Redistribution Notice

This checkpoint bundles or derives from Meta DINOv3 model materials. Redistribution of those components must follow
the DINOv3 license terms included in this repository. The project code remains available under the repository's own
license, but the full packaged checkpoint should not be treated as MIT-only.

## Research and Safety Disclaimer

This model is intended for research and educational use only. It is not a medical device, has not been validated
for clinical deployment, and should not be used as a substitute for professional radiology review.

## MIMIC Test Results

Frontal-only evaluation using `PA/AP` studies only.

These comparison tables are refreshed across the full LAnA collection whenever any collection model is evaluated.

### Cross-Model Comparison: All Frontal Test Studies

| Metric | LAnA-MIMIC-CHEXPERT | LAnA-MIMIC | LAnA | LAnA-v2 | LAnA-v3 | LAnA-v5 (Model still training) |
| --- | --- | --- | --- | --- | --- | --- |
| Run status | `Completed` | `Completed` | `Completed` | `Completed` | `Completed` | `Model still training` |
| Number of studies | `3041` | `3041` | `3041` | `3041` | `3041` | `3041` |
| ROUGE-L | `0.1513` | `0.1653` | `0.1686` | `0.1670` | `0.1745` | `0.1688` |
| BLEU-1 | `0.1707` | `0.1916` | `0.2091` | `0.2174` | `0.2346` | `0.2711` |
| BLEU-4 | `0.0357` | `0.0386` | `0.0417` | `0.0417` | `0.0484` | `0.0490` |
| METEOR | `0.2079` | `0.2202` | `0.2298` | `0.2063` | `0.2129` | `0.2568` |
| RadGraph F1 | `0.0918` | `0.0921` | `0.1024` | `0.1057` | `0.0939` | `0.0825` |
| RadGraph entity F1 | `0.1399` | `0.1459` | `0.1587` | `0.1569` | `0.1441` | `0.1440` |
| RadGraph relation F1 | `0.1246` | `0.1322` | `0.1443` | `0.1474` | `0.1280` | `0.1273` |
| CheXpert F1 14-micro | `0.1829` | `0.1565` | `0.2116` | `0.1401` | `0.3116` | `0.3497` |
| CheXpert F1 5-micro | `0.2183` | `0.1530` | `0.2512` | `0.2506` | `0.2486` | `0.3766` |
| CheXpert F1 14-macro | `0.1095` | `0.0713` | `0.1095` | `0.0401` | `0.1363` | `0.1755` |
| CheXpert F1 5-macro | `0.1634` | `0.1007` | `0.1644` | `0.1004` | `0.1686` | `0.2698` |

### Cross-Model Comparison: Findings-Only Frontal Test Studies

| Metric | LAnA-MIMIC-CHEXPERT | LAnA-MIMIC | LAnA | LAnA-v2 | LAnA-v3 | LAnA-v5 (Model still training) |
| --- | --- | --- | --- | --- | --- | --- |
| Run status | `Completed` | `Completed` | `Completed` | `Completed` | `Completed` | `Model still training` |
| Number of studies | `2210` | `2210` | `2210` | `2210` | `2210` | `2210` |
| ROUGE-L | `0.1576` | `0.1720` | `0.1771` | `0.1771` | `0.1848` | `0.1769` |
| BLEU-1 | `0.1754` | `0.2003` | `0.2177` | `0.2263` | `0.2480` | `0.2761` |
| BLEU-4 | `0.0405` | `0.0449` | `0.0484` | `0.0487` | `0.0573` | `0.0562` |
| METEOR | `0.2207` | `0.2347` | `0.2466` | `0.2240` | `0.2310` | `0.2722` |
| RadGraph F1 | `0.1010` | `0.1000` | `0.1119` | `0.1181` | `0.1046` | `0.0910` |
| RadGraph entity F1 | `0.1517` | `0.1577` | `0.1713` | `0.1739` | `0.1584` | `0.1532` |
| RadGraph relation F1 | `0.1347` | `0.1413` | `0.1549` | `0.1628` | `0.1405` | `0.1354` |
| CheXpert F1 14-micro | `0.1651` | `0.1442` | `0.1907` | `0.1365` | `0.2921` | `0.3112` |
| CheXpert F1 5-micro | `0.2152` | `0.1716` | `0.2415` | `0.2455` | `0.2394` | `0.3375` |
| CheXpert F1 14-macro | `0.1047` | `0.0700` | `0.1039` | `0.0381` | `0.1326` | `0.1583` |
| CheXpert F1 5-macro | `0.1611` | `0.1112` | `0.1578` | `0.0952` | `0.1636` | `0.2378` |

## Data

- Full project datasets: CheXpert and MIMIC-CXR.
- Intended project scope: train on curated chest X-ray/report data from both datasets and evaluate on MIMIC-CXR test studies.
- Current released checkpoint datasets: `MIMIC-CXR (findings-only)` for training and `MIMIC-CXR (findings-only)` for validation.
- Current published evaluation: MIMIC-CXR test split, `frontal-only (PA/AP)` studies.

## Evaluation

- Medical report metrics implemented in the repository include RadGraph F1 and CheXpert F1 (`14-micro`, `5-micro`, `14-macro`, `5-macro`).

## Experiment Model Descriptions

- `LAnA-MIMIC-CHEXPERT`: This variant was trained on a combined dataset of `CheXpert` and `MIMIC-CXR` using LoRA fine-tuning with the `AdamW` optimizer.
- `LAnA-MIMIC`: This model was trained on the `MIMIC-CXR (findings-only)` dataset using LoRA fine-tuning with the `AdamW` optimizer.
- `LAnA`: This model was trained on the `MIMIC-CXR (findings-only)` dataset using full-model optimization with `AdamW` instead of LoRA.
- `LAnA-v2`: This version keeps the same training setup as `LAnA`, but increases the effective global batch size from `16` to `128`.
- `LAnA-v3`: This version keeps the same training setup as `LAnA`, including the effective global batch size of `16`, but changes how EOS is handled so training and generation follow the same behavior. The model no longer uses the EOS token during training, and generation remained greedy without stopping when an EOS token was produced. In the previous setup, decoding was also greedy, stopped at EOS, and used a maximum of `128` new tokens.
- `LAnA-v4`: This version keeps the same decoding behavior as `LAnA-v3`, but increases the effective global batch size from `16` to `128`.
- `LAnA-v5`: This version uses the training recipe from the original `LAnA` paper, while switching to the legacy [`CXR-Findings-AI`](https://huggingface.co/spaces/manu02/CXR-Findings-AI) generation behavior.

## Training Snapshot

- Run: `LAnA-v5`
- This section describes the current public checkpoint, not the final completed project.
- Method: `full_adamw`
- Vision encoder: `facebook/dinov3-vits16-pretrain-lvd1689m`
- Text decoder: `gpt2`
- Visual projection: `linear`
- Segmentation encoder: `facebook/dinov3-convnext-small-pretrain-lvd1689m`
- Image size: `512`
- Local batch size: `1`
- Effective global batch size: `16`
- Scheduler: `cosine`
- Warmup steps: `1318`
- Weight decay: `0.01`
- Steps completed: `21569`
- Planned total steps: `26358`
- Images seen: `345172`
- Total training time: `7.1667` hours
- Hardware: `NVIDIA GeForce RTX 5070`
- Final train loss: `1.5532`
- Validation loss: `1.5243`

## Status

- Project status: `Training in progress`
- Release status: `Research preview checkpoint`
- Current checkpoint status: `Not final`
- Training completion toward planned run: `81.85%` (`2` / `3` epochs)
- Current published metrics are intermediate and will change as training continues.

## Notes

- Set `HF_TOKEN` with permission to access the DINOv3 repositories required by this model before downloading or running inference.
- `segmenters/` contains the lung and heart segmentation checkpoints used to build anatomical attention masks.
- `evaluations/mimic_test_metrics.json` contains the latest saved MIMIC test metrics.
