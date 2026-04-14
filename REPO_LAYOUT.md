# LAnA Repo Layout

This repo is organized around one canonical source layer plus three artifact families for published model payloads.

## Canonical Ownership

- `src/lana_radgen/`: canonical implementation for the standard LAnA family.
- `src/lana_arxiv/`: canonical implementation for the ArXiv-specific runtime.
- `scripts/`: canonical operational layer for training, evaluation, packaging, republishing, benchmarking, and cleanup.
- `tests/`: canonical verification layer for shared source behavior.

Code under `artifacts/*/.hf_publish/**` is treated as generated export payload unless that family still requires vendored files for Hugging Face compatibility.

## Version Families

### Pre-v3 Family

- `artifacts/LAnA`
- `artifacts/LAnA-paper`
- `artifacts/LAnA-MIMIC-TERM`
- `artifacts/full_3_epoch_mask_run`
- `artifacts/LAnA-v2`

Shared behavior:

- Standard LAnA runtime based on `src/lana_radgen`
- Differences are primarily dataset recipe, training run, and release payload

Canonical source:

- `src/lana_radgen/`
- `scripts/`

Generated/exported payloads:

- `.hf_publish/`
- `.hf_model_card_publish/`
- `.hf_eval_publish/`

Safe to rebuild:

- publish directories
- model cards generated from shared scripts
- evaluation publish outputs

### v3-v5 Family

- `artifacts/LAnA-v3`
- `artifacts/LAnA-v4`
- `artifacts/LAnA-v5`

Shared behavior:

- Later standard-runtime export approach
- Version-specific packaged payloads may vendor generated compatibility files

Canonical source:

- `src/lana_radgen/`
- `scripts/`

Generated/exported payloads:

- `.hf_publish/`
- `.hf_model_card_publish/`
- bundled backbone metadata in exported folders

Notes:

- `LAnA-v3` and `LAnA-v4` previously contained stale vendored `lana_radgen` package copies under `.hf_publish/`; those should not be treated as canonical source.

Safe to rebuild:

- publish directories
- model cards
- bundled backbone metadata used only for export

### ArXiv Family

- `artifacts/LAnA-Arxiv`

Shared behavior:

- ArXiv-paper runtime isolated from the standard LAnA family

Canonical source:

- `src/lana_arxiv/`
- `scripts/package_lana_arxiv.py`
- `scripts/evaluate_lana_arxiv.py`

Generated/exported payloads:

- `.hf_publish/`
- `.hf_model_card_publish/`
- validation residue such as `.hf_modules_cache_check/`

Safe to rebuild:

- publish directories
- model cards
- module cache validation outputs

## Tracking Rules

Keep tracked:

- human-authored source under `src/`, `scripts/`, and `tests/`
- root documentation
- intentional release metadata and minimal model payloads that the repo chooses to store

Regenerate instead of track broadly:

- `.hf_model_card_publish/`
- `.hf_eval_publish/`
- duplicated vendored code inside publish folders when equivalent canonical source exists

Remove or ignore locally generated content:

- `.cache/`
- `.plan_tmp/`
- `bert-base-uncased/`
- checkpoint-local tokenizer duplicates under `artifacts/*/checkpoints/**/tokenizer/`
- `.hf_modules_cache_check/`

## GitHub Push Guidance

- Treat `src/` and `scripts/` as the review surface for behavior changes.
- Treat artifact publish folders as release payloads, not the primary place to edit logic.
- Prefer regenerating publish outputs from shared scripts instead of hand-editing duplicate vendored files in multiple artifact folders.
- Do not inspect or modify `deletions/` or dataset paths as part of this repo cleanup flow.
