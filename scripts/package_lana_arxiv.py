from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from textwrap import dedent

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, GPT2Tokenizer, dynamic_module_utils

from lana_arxiv import LanaArxivConfig, LanaArxivImageProcessor, LanaArxivProcessor
from lana_radgen.logging_utils import configure_logging
from lana_radgen.model_card import DINO_V3_NOTICE, RESEARCH_USE_NOTICE, upsert_best_model_notice


PAPER_METRICS_SOURCE = (
    Path("deletions")
    / "ReportGeneration"
    / "experiments"
    / "lstm_vs_gpt"
    / "results"
    / "cloud_best_model_7_MIMIC.json"
)
LEGACY_MODEL_SOURCE = Path("deletions") / "models" / "model_best7.pth"
LEGACY_UTILS_SOURCE = Path(".tmp") / "model_best7_eval_space" / "utils"
PACKAGED_MODEL_FILENAME = "arxiv_paper_model.pth"
BUNDLED_VISION_MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"
BUNDLED_SEGMENTATION_MODEL_NAME = "facebook/dinov3-convnext-small-pretrain-lvd1689m"
BUNDLED_TEXT_MODEL_NAME = "gpt2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package the paper-era LAnA-Arxiv legacy report generator as a local upload-ready artifact.")
    parser.add_argument("--output-dir", default="artifacts/LAnA-Arxiv")
    parser.add_argument("--repo-id", default="manu02/LAnA-Arxiv")
    parser.add_argument("--model-path", default=str(LEGACY_MODEL_SOURCE))
    parser.add_argument("--legacy-utils-dir", default=str(LEGACY_UTILS_SOURCE))
    parser.add_argument("--paper-metrics-json", default=str(PAPER_METRICS_SOURCE))
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-smoke-test", action="store_true")
    parser.add_argument("--skip-collection-refresh", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _assert_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _resolve_snapshot_dir(model_name: str) -> Path:
    sanitized = model_name.replace("/", "--")
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{sanitized}"
    snapshots_root = cache_root / "snapshots"
    if not snapshots_root.exists():
        raise FileNotFoundError(f"No local Hugging Face snapshot found for {model_name}")
    snapshots = [path for path in snapshots_root.iterdir() if path.is_dir()]
    if not snapshots:
        raise FileNotFoundError(f"No snapshot directory found for {model_name}")
    return max(snapshots, key=lambda path: path.stat().st_mtime)


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))


def _copy_selected_backbone_files(src: Path, dst: Path, filenames: list[str]) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        src_file = src / name
        if src_file.exists():
            shutil.copy2(src_file, dst / name)


def _bundle_required_backbones(output_dir: Path) -> None:
    bundled_root = output_dir / "bundled_backbones"
    bundled_root.mkdir(parents=True, exist_ok=True)
    backbone_sources = {
        "vision_encoder": {
            "model_name": BUNDLED_VISION_MODEL_NAME,
            "filenames": ["config.json", "preprocessor_config.json"],
        },
        "segmenter_encoder": {
            "model_name": BUNDLED_SEGMENTATION_MODEL_NAME,
            "filenames": ["config.json", "preprocessor_config.json"],
        },
        "text_decoder": {
            "model_name": BUNDLED_TEXT_MODEL_NAME,
            "filenames": ["config.json", "generation_config.json"],
        },
    }
    for dirname, bundle_spec in backbone_sources.items():
        snapshot_dir = _resolve_snapshot_dir(bundle_spec["model_name"])
        _copy_selected_backbone_files(snapshot_dir, bundled_root / dirname, bundle_spec["filenames"])


def _patch_packaged_legacy_runtime(output_dir: Path) -> None:
    utils_dir = output_dir / "utils"
    complete_model_path = utils_dir / "complete_model.py"
    modified_gpt_path = utils_dir / "modifiedGPT2.py"

    complete_model_text = complete_model_path.read_text(encoding="utf-8")
    complete_model_text = complete_model_text.replace(
        "from transformers import AutoModel, GPT2Tokenizer\n",
        "from transformers import AutoConfig, AutoModel, GPT2Tokenizer\n",
        1,
    )
    if "from pathlib import Path" not in complete_model_text:
        complete_model_text = complete_model_text.replace(
            "import os\n",
            "import os\nfrom pathlib import Path\n",
            1,
        )
    if "_REPO_ROOT = Path(__file__).resolve().parents[1]" not in complete_model_text:
        complete_model_text = complete_model_text.replace(
            "from utils.layer_mask import gaussian_layer_stack_pipeline\n",
            "from utils.layer_mask import gaussian_layer_stack_pipeline\n\n"
            "_REPO_ROOT = Path(os.environ.get(\"LANA_ARXIV_REPO_ROOT\", Path(__file__).resolve().parents[1]))\n"
            "_BUNDLED_VISION_MODEL = (_REPO_ROOT / \"bundled_backbones\" / \"vision_encoder\").resolve().as_posix()\n"
            "_BUNDLED_SEGMENTATION_MODEL = (_REPO_ROOT / \"bundled_backbones\" / \"segmenter_encoder\").resolve().as_posix()\n"
            "_BUNDLED_TEXT_MODEL = (_REPO_ROOT / \"bundled_backbones\" / \"text_decoder\").resolve().as_posix()\n",
            1,
        )
    complete_model_text = complete_model_text.replace(
        'model_id="facebook/dinov3-vits16-pretrain-lvd1689m"',
        'model_id=_BUNDLED_VISION_MODEL',
    )
    complete_model_text = complete_model_text.replace(
        'model_name="facebook/dinov3-convnext-small-pretrain-lvd1689m"',
        'model_name=_BUNDLED_SEGMENTATION_MODEL',
    )
    complete_model_text = complete_model_text.replace(
        'GPT2Tokenizer.from_pretrained("gpt2")',
        'GPT2Tokenizer.from_pretrained(_REPO_ROOT.as_posix(), local_files_only=True)',
    )
    complete_model_text = complete_model_text.replace(
        "self.model = AutoModel.from_pretrained(model_id)",
        "self.model = AutoModel.from_config(AutoConfig.from_pretrained(model_id, local_files_only=True))",
    )
    complete_model_text = complete_model_text.replace(
        "self.encoder = AutoModel.from_pretrained(model_name)",
        "self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_name, local_files_only=True))",
    )
    complete_model_text = complete_model_text.replace(
        "self.tokenizer = GPT2Tokenizer.from_pretrained(_BUNDLED_TEXT_MODEL)",
        "self.tokenizer = GPT2Tokenizer.from_pretrained(_REPO_ROOT.as_posix(), local_files_only=True)",
    )
    complete_model_path.write_text(complete_model_text, encoding="utf-8")

    modified_gpt_text = modified_gpt_path.read_text(encoding="utf-8")
    if "from pathlib import Path" not in modified_gpt_text:
        modified_gpt_text = modified_gpt_text.replace(
            "from torch import nn\n",
            "from torch import nn\nimport os\nfrom pathlib import Path\n",
            1,
        )
    elif "import os" not in modified_gpt_text:
        modified_gpt_text = modified_gpt_text.replace(
            "from torch import nn\n",
            "from torch import nn\nimport os\n",
            1,
        )
    if "_REPO_ROOT = Path(__file__).resolve().parents[1]" not in modified_gpt_text:
        modified_gpt_text = modified_gpt_text.replace(
            "logger = logging.get_logger(__name__)\n",
            "logger = logging.get_logger(__name__)\n"
            "_REPO_ROOT = Path(os.environ.get(\"LANA_ARXIV_REPO_ROOT\", Path(__file__).resolve().parents[1]))\n"
            "_BUNDLED_TEXT_MODEL = (_REPO_ROOT / \"bundled_backbones\" / \"text_decoder\").resolve().as_posix()\n",
            1,
        )
    modified_gpt_text = modified_gpt_text.replace(
        "        max_positions = 2048\n",
        "        max_positions = getattr(config, \"n_positions\", getattr(config, \"n_ctx\", 1024))\n",
        1,
    )
    modified_gpt_text = modified_gpt_text.replace(
        'GPT2Config.from_pretrained("gpt2")',
        'GPT2Config.from_pretrained(_BUNDLED_TEXT_MODEL)',
    )
    if "def expand_gpt2_attention_bias(" not in modified_gpt_text:
        modified_gpt_text = modified_gpt_text.replace(
            "def create_decoder(attention = \"sdpa\"):\n",
            dedent(
                """
                def expand_gpt2_attention_bias(model: torch.nn.Module, new_max_positions: int):
                    for block in getattr(model.transformer, "h", []):
                        attn = getattr(block, "attn", None)
                        if attn is None:
                            continue
                        bias = torch.tril(torch.ones((new_max_positions, new_max_positions), dtype=torch.bool)).view(
                            1, 1, new_max_positions, new_max_positions
                        )
                        attn.register_buffer("bias", bias, persistent=False)
                    return model

                def create_decoder(attention = "sdpa"):
                """
            ).lstrip(),
            1,
        )
    legacy_create_decoder = [
        'def create_decoder(attention = "sdpa"):',
        '    config = GPT2Config.from_pretrained(_BUNDLED_TEXT_MODEL, local_files_only=True)',
        '    config._attn_implementation = attention',
        '    new_max_positions = 2048',
        '    decoder = GPT2LMHeadModelModified(config)',
        '    decoder.config._attn_implementation = attention',
        '    decoder = expand_gpt2_positional_embeddings(decoder, new_max_positions=new_max_positions, mode="linear")',
        '    decoder = expand_gpt2_attention_bias(decoder, new_max_positions=new_max_positions)',
        '    return decoder',
    ]
    modified_gpt_lines = modified_gpt_text.splitlines()
    start_idx = next((idx for idx, line in enumerate(modified_gpt_lines) if line.startswith("def create_decoder(")), None)
    if start_idx is None:
        raise RuntimeError("Failed to patch create_decoder() in packaged modifiedGPT2.py")
    end_idx = start_idx + 1
    while end_idx < len(modified_gpt_lines) and modified_gpt_lines[end_idx].startswith("    "):
        end_idx += 1
    modified_gpt_lines = modified_gpt_lines[:start_idx] + legacy_create_decoder + modified_gpt_lines[end_idx:]
    modified_gpt_path.write_text("\n".join(modified_gpt_lines) + "\n", encoding="utf-8")


def _copy_legacy_runtime_payload(output_dir: Path, model_path: Path, legacy_utils_dir: Path, packaged_model_filename: str) -> str:
    _assert_exists(model_path, "legacy model file")
    complete_model_source = legacy_utils_dir / "complete_model.py"
    modified_gpt_source = legacy_utils_dir / "modifiedGPT2.py"
    layer_mask_source = legacy_utils_dir / "layer_mask.py"
    processing_source = legacy_utils_dir / "processing.py"

    _assert_exists(complete_model_source, "legacy complete_model.py")
    _assert_exists(modified_gpt_source, "legacy modifiedGPT2.py")
    _assert_exists(layer_mask_source, "legacy layer_mask.py")

    shutil.copy2(model_path, output_dir / packaged_model_filename)

    utils_dir = output_dir / "utils"
    utils_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(layer_mask_source, utils_dir / "layer_mask.py")
    if processing_source.exists():
        shutil.copy2(processing_source, utils_dir / "processing.py")
    shutil.copy2(complete_model_source, utils_dir / "complete_model.py")
    shutil.copy2(modified_gpt_source, utils_dir / "modifiedGPT2.py")
    (utils_dir / "__init__.py").write_text("", encoding="utf-8")
    _bundle_required_backbones(output_dir)
    _patch_packaged_legacy_runtime(output_dir)

    source_package = _repo_root() / "src" / "lana_arxiv"
    for source_file in [
        "__init__.py",
        "configuration_lana_arxiv.py",
        "image_processing_lana_arxiv.py",
        "processing_lana_arxiv.py",
        "modeling_lana_arxiv.py",
    ]:
        shutil.copy2(source_package / source_file, output_dir / source_file)

    assets_source = _repo_root() / "assets" / "AnatomicalAttention.gif"
    if assets_source.exists():
        assets_dir = output_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(assets_source, assets_dir / assets_source.name)

    license_source = _repo_root() / "licenses" / "DINOv3-LICENSE.txt"
    if license_source.exists():
        shutil.copy2(license_source, output_dir / "DINOv3-LICENSE.txt")
    return packaged_model_filename


def _save_config_and_processor(output_dir: Path, args: argparse.Namespace, model_filename: str) -> None:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(output_dir)
    gpt2_snapshot = _resolve_snapshot_dir(BUNDLED_TEXT_MODEL_NAME)
    for extra_name in ["tokenizer.json", "vocab.json", "merges.txt", "special_tokens_map.json"]:
        extra_path = gpt2_snapshot / extra_name
        if extra_path.exists():
            shutil.copy2(extra_path, output_dir / extra_name)

    image_processor = LanaArxivImageProcessor(size={"height": args.image_size, "width": args.image_size})
    image_processor.save_pretrained(output_dir)
    processor = LanaArxivProcessor(image_processor=image_processor, tokenizer=tokenizer)
    processor.save_pretrained(output_dir)

    processor_config_path = output_dir / "processor_config.json"
    processor_config = json.loads(processor_config_path.read_text(encoding="utf-8")) if processor_config_path.exists() else {}
    processor_config["processor_class"] = "LanaArxivProcessor"
    processor_config["image_processor_type"] = "LanaArxivImageProcessor"
    processor_config["tokenizer_class"] = "GPT2Tokenizer"
    processor_config["auto_map"] = {"AutoProcessor": "processing_lana_arxiv.LanaArxivProcessor"}
    processor_config_path.write_text(json.dumps(processor_config, indent=2), encoding="utf-8")

    preprocessor_config_path = output_dir / "preprocessor_config.json"
    preprocessor_config = json.loads(preprocessor_config_path.read_text(encoding="utf-8")) if preprocessor_config_path.exists() else {}
    preprocessor_config["processor_class"] = "LanaArxivProcessor"
    preprocessor_config["image_processor_type"] = "LanaArxivImageProcessor"
    preprocessor_config["tokenizer_class"] = "GPT2Tokenizer"
    preprocessor_config["auto_map"] = {"AutoProcessor": "processing_lana_arxiv.LanaArxivProcessor"}
    preprocessor_config_path.write_text(json.dumps(preprocessor_config, indent=2), encoding="utf-8")

    config = LanaArxivConfig(
        vision_model_name=BUNDLED_VISION_MODEL_NAME,
        text_model_name=BUNDLED_TEXT_MODEL_NAME,
        segmentation_model_name=BUNDLED_SEGMENTATION_MODEL_NAME,
        image_size=args.image_size,
        source_space_repo_id=args.repo_id,
        source_space_revision="arxiv paper checkpoint",
        source_weight_name=model_filename,
        generation_repetition_penalty=1.2,
        generation_stop_on_eos=True,
        vision_feature_prefix_tokens_to_skip=5,
        bundled_vision_model_name="bundled_backbones/vision_encoder",
        bundled_segmentation_model_name="bundled_backbones/segmenter_encoder",
        bundled_text_model_name="bundled_backbones/text_decoder",
        architectures=["LanaArxivForConditionalGeneration"],
    )
    config.auto_map = {
        "AutoConfig": "configuration_lana_arxiv.LanaArxivConfig",
        "AutoModel": "modeling_lana_arxiv.LanaArxivForConditionalGeneration",
        "AutoProcessor": "processing_lana_arxiv.LanaArxivProcessor",
    }
    config.save_pretrained(output_dir)


def _empty_metric_section() -> dict:
    return {
        "split": "test",
        "subset": "",
        "dataset": "mimic-cxr",
        "view_filter": "",
        "num_examples": None,
        "bleu_1": None,
        "bleu_4": None,
        "meteor": None,
        "rouge_l": None,
        "chexpert_f1_14_micro": None,
        "chexpert_f1_5_micro": None,
        "chexpert_f1_14_macro": None,
        "chexpert_f1_5_macro": None,
        "chexpert_f1_micro": None,
        "chexpert_f1_macro": None,
        "chexpert_per_label_f1": {},
        "radgraph_f1": None,
        "radgraph_f1_entity": None,
        "radgraph_f1_relation": None,
        "radgraph_available": None,
        "radgraph_error": None,
    }


def _paper_metric_sections(paper_metrics_json: Path) -> tuple[dict, dict]:
    _assert_exists(paper_metrics_json, "paper metrics JSON")
    paper_metrics = json.loads(paper_metrics_json.read_text(encoding="utf-8"))

    all_test = _empty_metric_section()
    all_test.update(
        {
            "subset": "all frontal studies",
            "view_filter": "not reported in cloud_best_model_7_MIMIC.json",
        }
    )

    findings_only = _empty_metric_section()
    findings_only.update(
        {
            "subset": "findings-only frontal studies",
            "view_filter": "values copied from cloud_best_model_7_MIMIC.json",
            "chexpert_f1_14_micro": paper_metrics.get("chexbert_f1_micro"),
            "chexpert_f1_5_micro": paper_metrics.get("chexbert_f1_micro_5"),
            "chexpert_f1_14_macro": paper_metrics.get("chexbert_f1_macro"),
            "chexpert_f1_5_macro": paper_metrics.get("chexbert_f1_macro_5"),
            "chexpert_f1_micro": paper_metrics.get("chexbert_f1_micro"),
            "chexpert_f1_macro": paper_metrics.get("chexbert_f1_macro"),
            "radgraph_f1": paper_metrics.get("radgraph_f1_RG_E"),
            "radgraph_f1_entity": paper_metrics.get("radgraph_f1_RG_E"),
            "radgraph_f1_relation": paper_metrics.get("radgraph_f1_RG_ER"),
            "radgraph_available": True,
            "radgraph_error": None,
        }
    )
    return all_test, findings_only


def _write_evaluations(output_dir: Path, paper_metrics_json: Path) -> dict:
    evaluations_dir = output_dir / "evaluations"
    evaluations_dir.mkdir(parents=True, exist_ok=True)
    all_test, findings_only = _paper_metric_sections(paper_metrics_json)
    metrics_bundle = {
        "evaluation_suite": "mimic_test_dual",
        "all_test": all_test,
        "findings_only_test": findings_only,
    }
    (evaluations_dir / "mimic_test_metrics.json").write_text(json.dumps(metrics_bundle, indent=2), encoding="utf-8")
    (evaluations_dir / "mimic_test_findings_only_metrics.json").write_text(json.dumps(findings_only, indent=2), encoding="utf-8")
    return metrics_bundle


def _build_readme(repo_id: str, image_size: int, max_new_tokens: int, model_filename: str) -> str:
    overview = dedent(
        f"""
        ---
        license: mit
        library_name: transformers
        pipeline_tag: image-to-text
        tags:
          - medical-ai
          - radiology
          - chest-xray
          - report-generation
          - arxiv
          - legacy-model
        ---

        # LAnA-Arxiv

        **Legacy report-generation model created for the arXiv paper**

        [![ArXiv](https://img.shields.io/badge/ArXiv-2512.16841-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2512.16841)
        [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-manu02-FFD21E?logoColor=black)](https://huggingface.co/manu02)

        ![Layer-Wise Anatomical Attention](assets/AnatomicalAttention.gif)

        ## Overview

        `LAnA-Arxiv` packages the legacy report-generation model created for the arXiv paper.
        This artifact ships the paper checkpoint as `{model_filename}`, keeps the legacy generation path, and excludes the separate classification model.

        The packaged model preserves the original generation behavior:

        - skips the first `5` visual prefix tokens
        - uses the legacy Gaussian anatomical attention path
        - uses greedy decoding
        - stops generation on EOS
        - uses a repetition penalty of `1.2`

        {DINO_V3_NOTICE}

        {RESEARCH_USE_NOTICE}

        ## How to Run

        The artifact can be loaded from either the future repo id or a local folder path with the same code.

        ```python
        import torch
        from PIL import Image
        from transformers import AutoModel, AutoProcessor

        repo_id_or_path = "{repo_id}"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        processor = AutoProcessor.from_pretrained(repo_id_or_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(repo_id_or_path, trust_remote_code=True)
        model.move_non_quantized_modules(device)
        model.eval()

        image = Image.open("example.png").convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {{name: tensor.to(device) for name, tensor in inputs.items()}}

        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens={max_new_tokens})

        report = processor.batch_decode(generated, skip_special_tokens=True)[0]
        print(report)
        ```

        ## Intended Use

        - Input: a chest X-ray image resized to `{image_size}x{image_size}` and normalized with ImageNet mean/std.
        - Output: a generated radiology report.
        - Best fit: research comparison against the model created in the arXiv paper.

        ## MIMIC Test Results

        Canonical `LAnA-Arxiv` values in the comparison tables are copied directly from `cloud_best_model_7_MIMIC.json`.
        Metrics not reported in that paper JSON are intentionally left empty.

        ## Data

        - Packaged weight file: `{model_filename}`
        - Source family: `model_best7`
        - Packaged component: report-generation model only
        - Excluded component: separate classification checkpoint and classifier pipeline

        ## Evaluation

        - The canonical reported values come directly from `deletions/ReportGeneration/experiments/lstm_vs_gpt/results/cloud_best_model_7_MIMIC.json`.
        - This artifact does not substitute reproduced ROUGE, BLEU, or METEOR values into the public tables.

        ## Experiment Model Descriptions

        This section is refreshed locally together with the rest of the LAnA collection tables.

        ## Training Snapshot

        - Run: `LAnA-Arxiv`
        - Model identity: `model_best7` family
        - Shipped weight: `{model_filename}`
        - This artifact packages the model created in the arXiv paper.
        - Vision encoder: `facebook/dinov3-vits16-pretrain-lvd1689m`
        - Text decoder: `gpt2`
        - Segmentation encoder: `facebook/dinov3-convnext-small-pretrain-lvd1689m`
        - Image size: `{image_size}`
        - Example generation max new tokens: `{max_new_tokens}`

        ## Status

        - Project status: `Legacy paper model packaged locally`
        - Release status: `Ready for later upload`
        - Current checkpoint status: `Report generator only`

        ## Notes

        - `classification.pth` is intentionally excluded from this artifact.
        - `{model_filename}` is a renamed packaged copy of the paper-era `model_best7.pth` checkpoint.
        - `{model_filename}` and the vendored legacy generation code are preserved for reproducible loading.
        - Replace `repo_id_or_path` with the local folder path before upload if you want to test the artifact locally.
        """
    ).strip()
    return upsert_best_model_notice(overview)


def _write_readme(output_dir: Path, repo_id: str, image_size: int, max_new_tokens: int, model_filename: str) -> None:
    (output_dir / "README.md").write_text(
        _build_readme(repo_id=repo_id, image_size=image_size, max_new_tokens=max_new_tokens, model_filename=model_filename) + "\n",
        encoding="utf-8",
    )


def _write_run_summary(output_dir: Path, args: argparse.Namespace, model_filename: str, metrics_bundle: dict) -> None:
    all_test = metrics_bundle["all_test"]
    findings_only = metrics_bundle["findings_only_test"]
    summary = {
        "run_name": "LAnA-Arxiv",
        "repo_id": args.repo_id,
        "repo_url": f"https://huggingface.co/{args.repo_id}",
        "completed": True,
        "method": "legacy_paper_model_package",
        "vision_model_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "text_model_name": "gpt2",
        "segmentation_model_name": "facebook/dinov3-convnext-small-pretrain-lvd1689m",
        "image_size": int(args.image_size),
        "visual_projection_type": "linear",
        "attention_bias_mode": "gaussian_legacy",
        "vision_prefix_tokens_to_skip": 5,
        "generation_use_bos_token": False,
        "generation_stop_on_eos": True,
        "generation_repetition_penalty": 1.2,
        "source_space_repo_id": args.repo_id,
        "source_space_revision": "arxiv paper checkpoint",
        "source_checkpoint_path": "deletions/models/model_best7.pth",
        "source_weight_name": model_filename,
        "packaged_from_legacy_space": False,
        "packaged_from_arxiv_model": True,
        "packaged_report_generator_only": True,
        "excluded_assets": ["classification.pth"],
        "paper_metrics_source": str(Path(args.paper_metrics_json).as_posix()),
        "latest_evaluation": all_test,
        "latest_evaluations": {
            "all_test": all_test,
            "findings_only_test": findings_only,
        },
        "latest_evaluation_settings": {
            "model_source": "paper_json_reference",
            "image_size": int(args.image_size),
            "max_new_tokens": int(args.max_new_tokens),
            "batch_size": None,
            "output_tag": "paper_json_reference",
        },
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _smoke_test_local_loading(output_dir: Path, device: str) -> None:
    dynamic_cache_dir = output_dir / ".hf_modules_cache"
    dynamic_cache_dir.mkdir(parents=True, exist_ok=True)
    dynamic_module_utils.HF_MODULES_CACHE = str(dynamic_cache_dir)
    try:
        processor = AutoProcessor.from_pretrained(str(output_dir), trust_remote_code=True)
        model = AutoModel.from_pretrained(str(output_dir), trust_remote_code=True)
        runtime_device = torch.device("cpu" if device.startswith("cuda") else device)
        model.move_non_quantized_modules(runtime_device)
        model.eval()

        image = Image.new("RGB", (512, 512), color="black")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {name: tensor.to(runtime_device) for name, tensor in inputs.items()}
        with torch.inference_mode():
            _ = model.generate(**inputs, max_new_tokens=4)
    finally:
        shutil.rmtree(dynamic_cache_dir, ignore_errors=True)


def _refresh_collection_model_cards(output_dir: Path, repo_id: str, metrics_bundle: dict) -> None:
    repo_root = _repo_root()
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import evaluate as evaluate_script  # type: ignore

    evaluate_script._update_collection_model_cards(output_dir.resolve(), repo_id, metrics_bundle)


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level)

    repo_root = _repo_root()
    output_dir = (repo_root / args.output_dir).resolve()
    model_path = (repo_root / args.model_path).resolve()
    legacy_utils_dir = (repo_root / args.legacy_utils_dir).resolve()
    paper_metrics_json = (repo_root / args.paper_metrics_json).resolve()

    _reset_output_dir(output_dir)
    packaged_model_filename = _copy_legacy_runtime_payload(output_dir, model_path, legacy_utils_dir, PACKAGED_MODEL_FILENAME)
    _save_config_and_processor(output_dir, args, packaged_model_filename)
    metrics_bundle = _write_evaluations(output_dir, paper_metrics_json)
    _write_readme(output_dir, repo_id=args.repo_id, image_size=args.image_size, max_new_tokens=args.max_new_tokens, model_filename=packaged_model_filename)
    _write_run_summary(output_dir, args, packaged_model_filename, metrics_bundle)

    if not args.skip_smoke_test:
        _smoke_test_local_loading(output_dir, args.device)

    if not args.skip_collection_refresh:
        _refresh_collection_model_cards(output_dir, args.repo_id, metrics_bundle)


if __name__ == "__main__":
    main()
