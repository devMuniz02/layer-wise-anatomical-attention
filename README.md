[![ArXiv](https://img.shields.io/badge/ArXiv-2512.16841-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2512.16841)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devmuniz-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/devmuniz)
[![GitHub Profile](https://img.shields.io/badge/GitHub-devMuniz02-181717?logo=github&logoColor=white)](https://github.com/devMuniz02)
[![Portfolio](https://img.shields.io/badge/Portfolio-devmuniz02.github.io-0F172A?logo=googlechrome&logoColor=white)](https://devmuniz02.github.io/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-manu02-FFD21E?logoColor=black)](https://huggingface.co/manu02)

# LAnA

--- license: mit library_name: transformers pipeline_tag: image-to-text tags: - medical-ai - radiology - chest-xray - report-generation - segmentation - anatomical-attention metrics: - BLEU - METEOR - ROUGE - CIDEr ---

- Project status: `Training in progress` - Release status: `Research preview checkpoint` - Current checkpoint status: `Not final` - Training completion toward planned run: `56.90%` (`1.707` / `3` epochs) - Current published metrics are intermediate and will change as training continues.

## Overview

Code for arXiv paper: https://arxiv.org/abs/2512.16841

## Repository Structure

| Path | Description |
| --- | --- |
| `assets/` | Images, figures, or other supporting media used by the project. |
| `data_utils/` | Top-level project directory containing repository-specific resources. |
| `scripts/` | Top-level project directory containing repository-specific resources. |
| `src/` | Primary source code for the application or library. |
| `tests/` | Top-level project directory containing repository-specific resources. |
| `.gitignore` | Top-level file included in the repository. |
| `benchmark_mask_sweep.json` | Top-level file included in the repository. |
| `benchmark_stage1_mask.json` | Top-level file included in the repository. |
| `HOWTORUN.txt` | Top-level file included in the repository. |
| `LICENSE` | Repository license information. |

## Getting Started

1. Clone the repository.

   ```bash
   git clone https://github.com/devMuniz02/layer-wise-anatomical-attention.git
   cd layer-wise-anatomical-attention
   ```

2. Prepare the local environment.

Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Run or inspect the project entry point.

Use the project-specific scripts or notebooks in the repository root to run the workflow.

## MIMIC Test Results

Frontal-only evaluation using `PA/AP` studies only.

### Current Checkpoint Results

| Metric | Value |
| --- | --- |
| Number of studies | `3041` |
| RadGraph F1 | `0.0741` |
| RadGraph entity F1 | `0.1189` |
| RadGraph relation F1 | `0.1071` |
| CheXpert F1 14-micro | `0.1827` |
| CheXpert F1 5-micro | `0.1578` |
| CheXpert F1 14-macro | `0.0996` |
| CheXpert F1 5-macro | `0.1271` |

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

## Status

- Project status: `Training in progress`
- Release status: `Research preview checkpoint`
- Current checkpoint status: `Not final`
- Training completion toward planned run: `56.90%` (`1.707` / `3` epochs)
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
- Steps completed: `58191`
- Planned total steps: `102276`
- Images seen: `465570`
- Total training time: `13.6893` hours
- Hardware: `NVIDIA GeForce RTX 5070`
- Final train loss: `1.4760`
- Validation loss: `1.4079`
