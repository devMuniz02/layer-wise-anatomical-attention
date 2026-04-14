from __future__ import annotations

import re
from textwrap import dedent


DINO_V3_NOTICE = dedent(
    """
    ## Licensing and Redistribution Notice

    This checkpoint bundles or derives from Meta DINOv3 model materials. Redistribution of those components must follow
    the DINOv3 license terms included in this repository. The project code remains available under the repository's own
    license, but the full packaged checkpoint should not be treated as MIT-only.
    """
).strip()


RESEARCH_USE_NOTICE = dedent(
    """
    ## Research and Safety Disclaimer

    This model is intended for research and educational use only. It is not a medical device, has not been validated
    for clinical deployment, and should not be used as a substitute for professional radiology review.
    """
).strip()


DEFAULT_BEST_MODEL_REPO_ID = "manu02/LAnA-v3"


def build_best_model_notice(repo_id: str = DEFAULT_BEST_MODEL_REPO_ID) -> str:
    return dedent(
        f"""
        > Best current model in this collection: [`{repo_id}`](https://huggingface.co/{repo_id})
        """
    ).strip()


BEST_CURRENT_MODEL_NOTICE = build_best_model_notice()


def upsert_best_model_notice(current: str, repo_id: str = DEFAULT_BEST_MODEL_REPO_ID) -> str:
    notice = build_best_model_notice(repo_id)
    pattern = re.compile(r"> Best current model in this collection: \[`[^`]+`\]\(https://huggingface\.co/[^)]+\)")
    if pattern.search(current):
        return pattern.sub(notice, current, count=1)

    marker = "**Layer-Wise Anatomical Attention model**"
    if marker in current:
        return current.replace(marker, f"{marker}\n\n{notice}", 1)

    first_heading_end = current.find("\n", current.find("# "))
    if first_heading_end != -1:
        return current[: first_heading_end + 1] + "\n" + notice + "\n" + current[first_heading_end + 1 :]
    return current.rstrip() + "\n\n" + notice + "\n"


def build_dual_usage_section(repo_id: str, snapshot_revision: str | None = None) -> str:
    snapshot_download_call = (
        f'snapshot_download("{repo_id}", revision="{snapshot_revision}")' if snapshot_revision else f'snapshot_download("{repo_id}")'
    )
    return dedent(
        f"""
        ## How to Run

        New users should prefer the standard Hugging Face flow below. The legacy snapshot-download path remains supported
        for backward compatibility with earlier notebooks and scripts.

        ### Implementation 1: Standard Hugging Face loading

        ```python
        import torch
        from PIL import Image
        from transformers import AutoModel, AutoProcessor

        repo_id = "{repo_id}"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
        model.move_non_quantized_modules(device)
        model.eval()

        image = Image.open("example.png").convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {{name: tensor.to(device) for name, tensor in inputs.items()}}

        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=150)

        report = processor.batch_decode(generated, skip_special_tokens=True)[0]
        print(report)
        ```

        Batched inference uses the same path:

        ```python
        batch = processor(images=[image_a, image_b], return_tensors="pt")
        batch = {{name: tensor.to(device) for name, tensor in batch.items()}}
        generated = model.generate(**batch, max_new_tokens=150)
        reports = processor.batch_decode(generated, skip_special_tokens=True)
        ```

        `HF_TOKEN` is optional for this public standard-loading path. If you do not set one, the model still loads,
        but Hugging Face may show lower-rate-limit warnings.

        ### Implementation 2: Snapshot download / legacy manual loading

        ```python
        from pathlib import Path
        import sys

        import torch
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        repo_dir = Path({snapshot_download_call})
        sys.path.insert(0, str(repo_dir))

        from lana_radgen import LanaConfig, LanaForConditionalGeneration

        config = LanaConfig.from_pretrained(repo_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = LanaForConditionalGeneration(config)
        state_dict = load_file(str(repo_dir / "model.safetensors"))
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        assert not missing and not unexpected

        model.tokenizer = AutoTokenizer.from_pretrained(repo_dir, trust_remote_code=True)
        model.move_non_quantized_modules(device)
        model.eval()
        ```

        For the legacy snapshot/manual path, set `HF_TOKEN` with access to the gated DINOv3 encoder repos used by this
        project if your local code or environment still resolves those upstream references:

        - Vision encoder: [`facebook/dinov3-vits16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
        - Segmentation encoder: [`facebook/dinov3-convnext-small-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-convnext-small-pretrain-lvd1689m)
        """
    ).strip()


def build_main_branch_usage_section(repo_id: str, snapshot_revision: str = "snapshot-legacy") -> str:
    return dedent(
        f"""
        ## How to Run

        New users should prefer the standard Hugging Face flow below.
        The legacy snapshot/manual implementation lives on the `{snapshot_revision}` branch for backward compatibility.

        ### Implementation 1: Standard Hugging Face loading

        ```python
        import torch
        from PIL import Image
        from transformers import AutoModel, AutoProcessor

        repo_id = "{repo_id}"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
        model.move_non_quantized_modules(device)
        model.eval()

        image = Image.open("example.png").convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {{name: tensor.to(device) for name, tensor in inputs.items()}}

        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=150)

        report = processor.batch_decode(generated, skip_special_tokens=True)[0]
        print(report)
        ```

        Batched inference uses the same path:

        ```python
        batch = processor(images=[image_a, image_b], return_tensors="pt")
        batch = {{name: tensor.to(device) for name, tensor in batch.items()}}
        generated = model.generate(**batch, max_new_tokens=150)
        reports = processor.batch_decode(generated, skip_special_tokens=True)
        ```

        `HF_TOKEN` is optional for this public standard-loading path. If you do not set one, the model still loads,
        but Hugging Face may show lower-rate-limit warnings.

        ### Legacy snapshot branch

        Use the snapshot/manual branch only if you specifically need the older import-based workflow:

        - Branch: [`{snapshot_revision}`](https://huggingface.co/{repo_id}/tree/{snapshot_revision})
        - Download example: `snapshot_download("{repo_id}", revision="{snapshot_revision}")`
        """
    ).strip()


def build_snapshot_branch_usage_section(repo_id: str, snapshot_revision: str = "snapshot-legacy") -> str:
    usage = build_dual_usage_section(repo_id, snapshot_revision=snapshot_revision)
    return usage.replace(
        "## How to Run",
        dedent(
            f"""
            ## How to Run

            This branch preserves the legacy snapshot/manual layout for older notebooks and experiments.
            If you only need standard inference, prefer the [`main`](https://huggingface.co/{repo_id}/tree/main) branch instead.
            """
        ).strip(),
        1,
    )
