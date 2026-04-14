import argparse
import gc
import importlib.util
import json
import logging
import os
import re
import sys
import zipfile
from pathlib import Path
from types import MethodType
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors.torch import load_file, load_model

from lana_radgen import LanaConfig, LanaForConditionalGeneration
from lana_radgen.hub import push_directory_to_hub, push_model_card_update_to_hub
from lana_radgen.logging_utils import configure_logging
from lana_radgen.metrics import chexpert_label_f1_from_reference_labels, corpus_bleu_1, corpus_bleu_4, meteor_score, radgraph_f1, rouge_l
from lana_radgen.model_card import upsert_best_model_notice

LOGGER = logging.getLogger("evaluate")

CHEXPERT_LABEL_COLUMNS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Atelectasis",
    "Consolidation",
    "Pneumonia",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]

README_EVAL_START = "<!-- EVAL_RESULTS_START -->"


def release_cached_memory() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
README_EVAL_END = "<!-- EVAL_RESULTS_END -->"
MIMIC_RESULTS_START = "<!-- MIMIC_TEST_RESULTS_START -->"
MIMIC_RESULTS_END = "<!-- MIMIC_TEST_RESULTS_END -->"
ALL_TEST_KEY = "all_test"
FINDINGS_ONLY_TEST_KEY = "findings_only_test"
LANA_COLLECTION_MODELS = [
    {
        "artifact_dir": "full_3_epoch_mask_run",
        "repo_id": "manu02/LAnA-MIMIC-CHEXPERT",
        "model_name": "LAnA-MIMIC-CHEXPERT",
    },
    {
        "artifact_dir": "LAnA-MIMIC-TERM",
        "repo_id": "manu02/LAnA-MIMIC",
        "model_name": "LAnA-MIMIC",
    },
    {
        "artifact_dir": "LAnA-paper",
        "repo_id": "manu02/LAnA",
        "model_name": "LAnA",
    },
    {
        "artifact_dir": "LAnA-Arxiv",
        "repo_id": "manu02/LAnA-Arxiv",
        "model_name": "LAnA-Arxiv",
    },
    {
        "artifact_dir": "LAnA-v2",
        "repo_id": "manu02/LAnA-v2",
        "model_name": "LAnA-v2",
    },
    {
        "artifact_dir": "LAnA-v3",
        "repo_id": "manu02/LAnA-v3",
        "model_name": "LAnA-v3",
    },
    {
        "artifact_dir": "LAnA-v4",
        "repo_id": "manu02/LAnA-v4",
        "model_name": "LAnA-v4",
    },
    {
        "artifact_dir": "LAnA-v5",
        "repo_id": "manu02/LAnA-v5",
        "model_name": "LAnA-v5",
    },
]
EXPERIMENT_DESCRIPTION_HEADING = "## Experiment Model Descriptions"
EXPERIMENT_DESCRIPTION_ORDER = [
    "LAnA-MIMIC-CHEXPERT",
    "LAnA-MIMIC",
    "LAnA",
    "LAnA-Arxiv",
    "LAnA-v2",
    "LAnA-v3",
    "LAnA-v4",
    "LAnA-v5",
]
EXPERIMENT_DESCRIPTION_TEXT = {
    "LAnA-MIMIC-CHEXPERT": "This variant was trained on a combined dataset of `CheXpert` and `MIMIC-CXR` using LoRA fine-tuning with the `AdamW` optimizer.",
    "LAnA-MIMIC": "This model was trained on the `MIMIC-CXR (findings-only)` dataset using LoRA fine-tuning with the `AdamW` optimizer.",
    "LAnA": "This model was trained on the `MIMIC-CXR (findings-only)` dataset using full-model optimization with `AdamW` instead of LoRA.",
    "LAnA-Arxiv": "This model is the report-generation model created in the arXiv paper, packaged locally with its original legacy generation code and without the separate classification model.",
    "LAnA-v2": "This version keeps the same training setup as `LAnA`, but increases the effective global batch size from `16` to `128`.",
    "LAnA-v3": "This version keeps the same training setup as `LAnA`, including the effective global batch size of `16`, but changes how EOS is handled so training and generation follow the same behavior. The model no longer uses the EOS token during training, and generation remained greedy without stopping when an EOS token was produced. In the previous setup, decoding was also greedy, stopped at EOS, and used a maximum of `128` new tokens.",
    "LAnA-v4": "This version keeps the same decoding behavior as `LAnA-v3`, but increases the effective global batch size from `16` to `128`.",
    "LAnA-v5": "This version uses the training recipe from the original `LAnA` paper, while switching to the legacy [`CXR-Findings-AI`](https://huggingface.co/spaces/manu02/CXR-Findings-AI) generation behavior.",
}


class GenerationModelAdapter:
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, **kwargs) -> torch.Tensor:
        output = self.model.generate(**kwargs)
        if isinstance(output, tuple):
            output = output[0]
        elif hasattr(output, "sequences"):
            output = output.sequences
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Expected tensor output from generate(), got {type(output)!r}")
        if output.ndim == 1:
            output = output.unsqueeze(0)
        return output

    def eval(self):
        self.model.eval()
        return self


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the latest exported LANA model on the MIMIC test split.")
    parser.add_argument("--run-dir", default="", help="Training artifact directory. Defaults to the most recently modified artifacts/* run.")
    parser.add_argument("--repo-id", default="manu02/lana-radgen-benchmark")
    parser.add_argument(
        "--model-source",
        choices=["auto", "run_export", "space_legacy", "deletions_complete_model"],
        default="auto",
        help="Select which model loader to use. 'auto' preserves the current behavior.",
    )
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of test examples.")
    parser.add_argument("--mimic-root", default="Datasets/MIMIC")
    parser.add_argument("--predictions-csv", default="", help="Optional existing predictions CSV to recompute metrics without running generation.")
    parser.add_argument("--output-tag", default="", help="Optional suffix used for evaluation output filenames, e.g. '120_tokens'.")
    parser.add_argument("--external-space-repo-id", default="", help="Optional Hugging Face Space repo id to evaluate via a temporary snapshot loader.")
    parser.add_argument("--external-space-revision", default="main")
    parser.add_argument("--external-space-local-dir", default="", help="Optional local directory where the Space snapshot should be downloaded.")
    parser.add_argument(
        "--deletions-model-path",
        default="",
        help="Checkpoint path used when --model-source deletions_complete_model is selected.",
    )
    parser.add_argument(
        "--deletions-loader-path",
        default="deletions/utils/models/complete_model.py",
        help="Legacy deletions complete_model.py loader path.",
    )
    parser.add_argument(
        "--generate-chunk",
        action="store_true",
        help="Generate one chunk of predictions and save it immediately without computing final metrics.",
    )
    parser.add_argument(
        "--generate-all-chunks",
        action="store_true",
        help="Generate all chunks for the selected manifest range, skipping already-complete chunk files.",
    )
    parser.add_argument(
        "--merge-chunks",
        action="store_true",
        help="Merge saved chunk CSVs and compute final metrics without running generation again.",
    )
    parser.add_argument("--chunk-dir", default="", help="Directory used to save or merge chunk CSV files.")
    parser.add_argument("--chunk-size", type=int, default=100, help="Number of studies per chunk when using chunked generation.")
    parser.add_argument("--start-idx", type=int, default=0, help="Start index within the filtered MIMIC manifest.")
    parser.add_argument("--end-idx", type=int, default=-1, help="Exclusive end index within the filtered MIMIC manifest. -1 means full range.")
    parser.add_argument("--skip-collection-model-card-update", action="store_true", help="Skip rewriting local collection model cards after evaluation.")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def _latest_run_dir() -> Path:
    candidates = []
    artifacts_root = Path("artifacts")
    if not artifacts_root.exists():
        raise FileNotFoundError("No artifacts directory found.")
    for summary_path in artifacts_root.glob("*/run_summary.json"):
        candidates.append(summary_path.parent)
    if not candidates:
        raise FileNotFoundError("No run_summary.json found under artifacts/.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _parse_findings_section(report_text: str) -> str:
    normalized = report_text.replace("\r\n", "\n")
    match = re.search(r"FINDINGS:\s*(.*?)(?:\n\s*[A-Z ]+:\s|$)", normalized, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()
    impression_match = re.search(r"IMPRESSION:\s*(.*?)(?:\n\s*[A-Z ]+:\s|$)", normalized, flags=re.IGNORECASE | re.DOTALL)
    if impression_match:
        return re.sub(r"\s+", " ", impression_match.group(1)).strip()
    return re.sub(r"\s+", " ", normalized).strip()


def _load_report_texts(report_zip_path: Path) -> dict[tuple[int, int], str]:
    reports = {}
    with zipfile.ZipFile(report_zip_path) as archive:
        for name in archive.namelist():
            if not name.endswith(".txt"):
                continue
            match = re.search(r"/p(\d+)/s(\d+)\.txt$", name)
            if not match:
                continue
            subject_id = int(match.group(1))
            study_id = int(match.group(2))
            text = archive.read(name).decode("utf-8", errors="ignore")
            reports[(subject_id, study_id)] = _parse_findings_section(text)
    return reports


def _resolve_processed_image_path(subject_id: int, study_id: int, dicom_id: str, image_root: Path) -> Path:
    return image_root / f"p{subject_id}" / f"s{study_id}" / f"{dicom_id}.png"


def _build_mimic_test_manifest(mimic_root: Path, limit: int = 0, findings_only: bool = False) -> pd.DataFrame:
    split_df = pd.read_csv(mimic_root / "mimic-cxr-2.0.0-split.csv.gz", compression="gzip")
    records_df = pd.read_csv(mimic_root / "cxr-record-list.csv.gz", compression="gzip")
    metadata_df = pd.read_csv(mimic_root / "mimic-cxr-2.0.0-metadata.csv")
    chexpert_df = pd.read_csv(mimic_root / "mimic-cxr-2.0.0-chexpert.csv.gz", compression="gzip")
    reports = _load_report_texts(mimic_root / "mimic-cxr-reports.zip")

    df = split_df[split_df["split"] == "test"].copy()
    if findings_only:
        findings_df = pd.read_csv(mimic_root / "mimic-cxr-2.0.0-metadata-findings-only.csv")
        df = df.merge(findings_df[["subject_id", "study_id", "dicom_id"]], on=["subject_id", "study_id", "dicom_id"], how="inner")
    df = df.merge(records_df, on=["subject_id", "study_id", "dicom_id"], how="left")
    df = df.merge(metadata_df[["subject_id", "study_id", "dicom_id", "ViewPosition"]], on=["subject_id", "study_id", "dicom_id"], how="left")
    df = df.merge(chexpert_df[["subject_id", "study_id"] + CHEXPERT_LABEL_COLUMNS], on=["subject_id", "study_id"], how="left")

    df["ViewPosition"] = df["ViewPosition"].astype(str).str.upper()
    df = df[df["ViewPosition"].isin({"PA", "AP"})].copy()
    df = df.sort_values(by=["subject_id", "study_id", "dicom_id"]).drop_duplicates(subset=["subject_id", "study_id"], keep="first")

    image_root = mimic_root / "images" / "datos"
    df["processed_image_path"] = df.apply(
        lambda row: str(_resolve_processed_image_path(int(row["subject_id"]), int(row["study_id"]), str(row["dicom_id"]), image_root).resolve()),
        axis=1,
    )
    df["report_text"] = df.apply(lambda row: reports.get((int(row["subject_id"]), int(row["study_id"])), ""), axis=1)
    for label in CHEXPERT_LABEL_COLUMNS:
        df[label] = df[label].fillna(0.0).map(lambda value: 1 if float(value) == 1.0 else 0)

    df = df[df["report_text"].map(bool)]
    df = df[df["processed_image_path"].map(lambda value: Path(value).exists())]
    if limit > 0:
        df = df.head(limit)
    if df.empty:
        raise RuntimeError("No usable MIMIC test examples were found after joins and file checks.")
    return df.reset_index(drop=True)


def _build_findings_only_lookup(mimic_root: Path) -> set[tuple[int, int]]:
    manifest = _build_mimic_test_manifest(mimic_root=mimic_root, findings_only=True)
    return {
        (int(row.subject_id), int(row.study_id))
        for row in manifest[["subject_id", "study_id"]].drop_duplicates().itertuples(index=False)
    }


def _load_chexpert_reference_labels(mimic_root: Path) -> pd.DataFrame:
    chexpert_df = pd.read_csv(mimic_root / "mimic-cxr-2.0.0-chexpert.csv.gz", compression="gzip")
    for label in CHEXPERT_LABEL_COLUMNS:
        chexpert_df[label] = chexpert_df[label].fillna(0.0).map(lambda value: 1 if float(value) == 1.0 else 0)
    return chexpert_df[["subject_id", "study_id"] + CHEXPERT_LABEL_COLUMNS].drop_duplicates(subset=["subject_id", "study_id"], keep="first")


def _load_image_tensor(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), resample=Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


def _load_model_from_export(model_dir: Path, device: torch.device) -> LanaForConditionalGeneration:
    config = LanaConfig.from_pretrained(str(model_dir))
    model = LanaForConditionalGeneration(config)
    state_dict = load_file(str(model_dir / "model.safetensors"))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        LOGGER.warning("Export model loaded with non-strict state dict. Missing=%s Unexpected=%s", missing, unexpected)
    model.move_non_quantized_modules(device)
    model.eval()
    return model


def _load_dotenv_variables(dotenv_path: Path) -> dict[str, str]:
    if not dotenv_path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _ensure_hf_auth_environment(repo_root: Path) -> None:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        return
    env_values = _load_dotenv_variables(repo_root / ".env")
    token = env_values.get("HF_TOKEN") or env_values.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)


def _download_external_space_snapshot(repo_root: Path, repo_id: str, revision: str, local_dir: str) -> Path:
    if local_dir:
        local_path = Path(local_dir).resolve()
        if (local_path / "utils" / "complete_model.py").exists() and (local_path / "complete_model.safetensor").exists():
            return local_path
    _ensure_hf_auth_environment(repo_root)
    download_dir = Path(local_dir) if local_dir else repo_root / ".tmp" / f"space_snapshot_{repo_id.replace('/', '__')}"
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="space",
        revision=revision,
        local_dir=str(download_dir),
    )
    return Path(snapshot_path).resolve()


def _load_external_space_model(repo_root: Path, repo_id: str, revision: str, local_dir: str, device: torch.device) -> GenerationModelAdapter:
    snapshot_dir = _download_external_space_snapshot(repo_root, repo_id=repo_id, revision=revision, local_dir=local_dir)
    complete_model_path = snapshot_dir / "utils" / "complete_model.py"
    if not complete_model_path.exists():
        raise FileNotFoundError(f"Expected legacy Space loader at {complete_model_path}")
    if not (snapshot_dir / "complete_model.safetensor").exists():
        raise FileNotFoundError(f"Expected legacy Space weights at {snapshot_dir / 'complete_model.safetensor'}")

    added_to_syspath = False
    snapshot_dir_str = str(snapshot_dir)
    if snapshot_dir_str not in sys.path:
        sys.path.insert(0, snapshot_dir_str)
        added_to_syspath = True
    try:
        for module_name in [name for name in list(sys.modules) if name == "utils" or name.startswith("utils.")]:
            del sys.modules[module_name]
        legacy_spec = importlib.util.spec_from_file_location("legacy_space_complete_model", complete_model_path)
        legacy_module = importlib.util.module_from_spec(legacy_spec)
        legacy_spec.loader.exec_module(legacy_module)
        model = legacy_module.create_complete_model(device=str(device), attention_implementation="eager")
        decoder = getattr(model, "decoder", None)
        transformer = getattr(getattr(model, "decoder", None), "transformer", None)
        if decoder is not None and not hasattr(decoder, "model_parallel"):
            decoder.model_parallel = False
        if transformer is not None and not hasattr(transformer, "get_head_mask"):
            def _legacy_get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked: bool = False):
                if head_mask is None:
                    return [None] * num_hidden_layers
                return head_mask

            transformer.get_head_mask = MethodType(_legacy_get_head_mask, transformer)
        if transformer is not None and not hasattr(transformer, "model_parallel"):
            transformer.model_parallel = False
        load_model(model, str(snapshot_dir / "complete_model.safetensor"))
    finally:
        if added_to_syspath:
            sys.path.remove(snapshot_dir_str)

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        tokenizer = getattr(getattr(model, "decoder", None), "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Legacy Space model did not expose a tokenizer after loading.")
    model.eval()
    return GenerationModelAdapter(model, tokenizer)


def _load_model(
    run_dir: Path,
    device: torch.device,
    *,
    model_source: str = "auto",
    external_space_repo_id: str = "",
    external_space_revision: str = "main",
    external_space_local_dir: str = "",
    deletions_model_path: str = "",
    deletions_loader_path: str = "deletions/utils/models/complete_model.py",
) -> GenerationModelAdapter | LanaForConditionalGeneration:
    resolved_source = _resolve_model_source(model_source, external_space_repo_id=external_space_repo_id)
    if resolved_source == "space_legacy":
        repo_root = Path(__file__).resolve().parents[1]
        return _load_external_space_model(
            repo_root=repo_root,
            repo_id=external_space_repo_id,
            revision=external_space_revision,
            local_dir=external_space_local_dir,
            device=device,
        )
    if resolved_source == "deletions_complete_model":
        repo_root = Path(__file__).resolve().parents[1]
        return _load_deletions_complete_model(
            repo_root=repo_root,
            device=device,
            checkpoint_path=deletions_model_path,
            loader_path=deletions_loader_path,
        )

    summary_path = run_dir / "run_summary.json"
    checkpoints_root = run_dir / "checkpoints"
    latest_checkpoint_path = checkpoints_root / "latest_checkpoint.json"
    if summary_path.exists() and latest_checkpoint_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        latest_payload = json.loads(latest_checkpoint_path.read_text(encoding="utf-8"))
        checkpoint_dir = Path(latest_payload["path"])
        training_state_path = checkpoint_dir / "training_state.pt"
        if training_state_path.exists():
            train_spec = importlib.util.spec_from_file_location("train_runtime", Path(__file__).with_name("train.py"))
            train_mod = importlib.util.module_from_spec(train_spec)
            train_spec.loader.exec_module(train_mod)

            args = SimpleNamespace(
                vision_model_name=summary["vision_model_name"],
                text_model_name=summary["text_model_name"],
                image_size=int(summary["image_size"]),
                visual_projection_type=summary.get("visual_projection_type", "mlp4"),
                attention_bias_mode=summary.get("attention_bias_mode", "layerwise"),
                vision_prefix_tokens_to_skip=int(summary.get("vision_prefix_tokens_to_skip", 1)),
                segmentation_model_name=summary["segmentation_model_name"],
                lung_segmenter_checkpoint=summary["lung_segmenter_checkpoint"],
                heart_segmenter_checkpoint=summary["heart_segmenter_checkpoint"],
                disable_segmentation_mask=False,
                generation_use_bos_token=bool(summary.get("generation_use_bos_token", True)),
                generation_stop_on_eos=bool(summary.get("generation_stop_on_eos", False)),
                generation_repetition_penalty=float(summary.get("generation_repetition_penalty", 1.0)),
                batch_size=int(summary.get("batch_size", 1)),
                global_batch_size=int(summary.get("global_batch_size", 0)),
                num_workers=0,
                learning_rate=1e-4,
                seed=int(summary.get("seed", 42)),
            )
            model = train_mod.build_model(summary["method"], args, device)
            payload = torch.load(training_state_path, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["model_state"], strict=True)
            model.eval()
            return model

    return _load_model_from_export(run_dir / "model", device)


def _resolve_model_source(model_source: str, *, external_space_repo_id: str) -> str:
    if model_source != "auto":
        return model_source
    if external_space_repo_id:
        return "space_legacy"
    return "run_export"


def _load_deletions_complete_model(
    repo_root: Path,
    device: torch.device,
    *,
    checkpoint_path: str,
    loader_path: str,
) -> GenerationModelAdapter:
    loader_file = (repo_root / loader_path).resolve()
    if not loader_file.exists():
        raise FileNotFoundError(f"Expected legacy deletions loader at {loader_file}")
    checkpoint_file = Path(checkpoint_path).resolve() if checkpoint_path else (repo_root / "deletions" / "models" / "model_best7.pth").resolve()
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Expected legacy deletions checkpoint at {checkpoint_file}")

    deletions_root = loader_file.parents[2]
    added_to_syspath = False
    deletions_root_str = str(deletions_root)
    if deletions_root_str not in sys.path:
        sys.path.insert(0, deletions_root_str)
        added_to_syspath = True
    try:
        for module_name in [name for name in list(sys.modules) if name == "utils" or name.startswith("utils.")]:
            del sys.modules[module_name]
        legacy_spec = importlib.util.spec_from_file_location("legacy_deletions_complete_model", loader_file)
        legacy_module = importlib.util.module_from_spec(legacy_spec)
        legacy_spec.loader.exec_module(legacy_module)
        model = legacy_module.CustomModel(device=str(device), attention_implementation="eager")
        decoder = getattr(model, "decoder", None)
        transformer = getattr(getattr(model, "decoder", None), "transformer", None)
        if decoder is not None and not hasattr(decoder, "model_parallel"):
            decoder.model_parallel = False
        if transformer is not None and not hasattr(transformer, "get_head_mask"):
            def _legacy_get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked: bool = False):
                if head_mask is None:
                    return [None] * num_hidden_layers
                return head_mask

            transformer.get_head_mask = MethodType(_legacy_get_head_mask, transformer)
        if transformer is not None and not hasattr(transformer, "model_parallel"):
            transformer.model_parallel = False

        payload = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            payload = payload.get("model_state_dict", payload.get("state_dict", payload))
        if not isinstance(payload, dict):
            raise TypeError(f"Unsupported deletions checkpoint payload type: {type(payload)!r}")
        cleaned_state = {(key[7:] if key.startswith("module.") else key): value for key, value in payload.items()}
        model.load_state_dict(cleaned_state, strict=False)
    finally:
        if added_to_syspath:
            sys.path.remove(deletions_root_str)

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        tokenizer = getattr(getattr(model, "decoder", None), "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Legacy deletions model did not expose a tokenizer after loading.")
    model.eval()
    return GenerationModelAdapter(model, tokenizer)


def _decode_prediction(model: LanaForConditionalGeneration, token_ids: torch.Tensor) -> str:
    text = model.tokenizer.decode(token_ids, skip_special_tokens=True)
    return re.sub(r"\s+", " ", text).strip()


def _compute_metrics(records_df: pd.DataFrame, split_name: str, subset_name: str, view_filter: str) -> dict:
    predictions = records_df["prediction"].fillna("").astype(str).tolist()
    references = records_df["reference"].fillna("").astype(str).tolist()
    reference_labels = []
    for _, row in records_df.iterrows():
        reference_labels.append({label: int(row[label]) for label in CHEXPERT_LABEL_COLUMNS})

    chexpert_metrics = chexpert_label_f1_from_reference_labels(predictions, reference_labels)
    radgraph_metrics = radgraph_f1(predictions, references)
    return {
        "split": split_name,
        "subset": subset_name,
        "dataset": "mimic-cxr",
        "view_filter": view_filter,
        "num_examples": len(records_df),
        "bleu_1": corpus_bleu_1(predictions, references),
        "bleu_4": corpus_bleu_4(predictions, references),
        "meteor": meteor_score(predictions, references),
        "rouge_l": rouge_l(predictions, references),
        "chexpert_f1_14_micro": chexpert_metrics["chexpert_f1_14_micro"],
        "chexpert_f1_5_micro": chexpert_metrics["chexpert_f1_5_micro"],
        "chexpert_f1_14_macro": chexpert_metrics["chexpert_f1_14_macro"],
        "chexpert_f1_5_macro": chexpert_metrics["chexpert_f1_5_macro"],
        "chexpert_f1_micro": chexpert_metrics["chexpert_f1_micro"],
        "chexpert_f1_macro": chexpert_metrics["chexpert_f1_macro"],
        "chexpert_per_label_f1": chexpert_metrics["chexpert_per_label_f1"],
        "radgraph_f1": radgraph_metrics["radgraph_f1"],
        "radgraph_f1_entity": radgraph_metrics["radgraph_f1_entity"],
        "radgraph_f1_relation": radgraph_metrics["radgraph_f1_relation"],
        "radgraph_available": radgraph_metrics["radgraph_available"],
        "radgraph_error": radgraph_metrics["radgraph_error"],
    }


def _evaluate_model(
    model: LanaForConditionalGeneration,
    manifest: pd.DataFrame,
    device: torch.device,
    image_size: int,
    max_new_tokens: int,
    batch_size: int,
) -> tuple[list[dict], dict]:
    records = []

    with torch.inference_mode():
        for start_idx in range(0, len(manifest), batch_size):
            batch_df = manifest.iloc[start_idx : start_idx + batch_size]
            batch_tensors = [
                _load_image_tensor(Path(row["processed_image_path"]), image_size=image_size) for _, row in batch_df.iterrows()
            ]
            pixel_values = torch.stack(batch_tensors, dim=0).to(device)
            generated_ids = model.generate(pixel_values=pixel_values, max_new_tokens=max_new_tokens)

            for sample_idx, (_, row) in enumerate(batch_df.iterrows()):
                prediction_text = _decode_prediction(model, generated_ids[sample_idx].detach().cpu())
                reference_text = str(row["report_text"]).strip()
                label_map = {label: int(row[label]) for label in CHEXPERT_LABEL_COLUMNS}

                records.append(
                    {
                        "subject_id": int(row["subject_id"]),
                        "study_id": int(row["study_id"]),
                        "dicom_id": str(row["dicom_id"]),
                        "image_path": row["processed_image_path"],
                        "processed_image_path": row["processed_image_path"],
                        "prediction": prediction_text,
                        "reference": reference_text,
                        **label_map,
                    }
                )

            completed = min(start_idx + len(batch_df), len(manifest))
            if completed % 100 == 0 or completed == len(manifest):
                LOGGER.info("Generated %s / %s test reports", completed, len(manifest))

    metrics = _compute_metrics(
        pd.DataFrame(records),
        split_name="test",
        subset_name="all frontal studies",
        view_filter="frontal-only (PA/AP)",
    )
    return records, metrics


def _slice_manifest(manifest: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
    start = max(0, int(start_idx))
    stop = len(manifest) if int(end_idx) < 0 else min(len(manifest), int(end_idx))
    if stop < start:
        raise ValueError(f"Invalid manifest slice: start_idx={start} end_idx={stop}")
    return manifest.iloc[start:stop].reset_index(drop=True)


def _resolve_effective_batch_size(args: argparse.Namespace, resolved_model_source: str) -> int:
    if resolved_model_source == "deletions_complete_model" and "--batch-size" not in sys.argv:
        return 1
    return max(1, int(args.batch_size))


def _resolve_chunk_dir(run_dir: Path, chunk_dir: str, output_tag: str) -> Path:
    if chunk_dir.strip():
        return Path(chunk_dir).resolve()
    suffix = f"_{output_tag.strip()}" if output_tag.strip() else ""
    return (run_dir / "evaluations" / f"chunks{suffix}").resolve()


def _chunk_csv_path(chunk_dir: Path, start_idx: int, end_idx: int) -> Path:
    return chunk_dir / f"chunk_{start_idx:04d}_{end_idx - 1:04d}.csv"


def _prediction_columns() -> list[str]:
    return [
        "subject_id",
        "study_id",
        "dicom_id",
        "image_path",
        "processed_image_path",
        "prediction",
        "reference",
        *CHEXPERT_LABEL_COLUMNS,
    ]


def _is_valid_chunk_csv(chunk_path: Path, expected_rows: int) -> bool:
    if not chunk_path.exists() or chunk_path.stat().st_size == 0:
        return False
    try:
        chunk_df = pd.read_csv(chunk_path)
    except Exception:
        return False
    required_columns = set(_prediction_columns())
    return required_columns.issubset(chunk_df.columns) and len(chunk_df) == expected_rows


def _append_records_to_csv(chunk_path: Path, records: list[dict]) -> None:
    if not records:
        return
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_df = pd.DataFrame(records)
    write_header = not chunk_path.exists() or chunk_path.stat().st_size == 0
    chunk_df.to_csv(chunk_path, mode="a", header=write_header, index=False)


def _generate_chunk_predictions(
    model: LanaForConditionalGeneration | GenerationModelAdapter,
    manifest: pd.DataFrame,
    device: torch.device,
    image_size: int,
    max_new_tokens: int,
    batch_size: int,
    chunk_path: Path,
) -> Path:
    if chunk_path.exists():
        chunk_path.unlink()

    with torch.inference_mode():
        for start_idx in range(0, len(manifest), batch_size):
            batch_df = manifest.iloc[start_idx : start_idx + batch_size]
            batch_tensors = [
                _load_image_tensor(Path(row["processed_image_path"]), image_size=image_size) for _, row in batch_df.iterrows()
            ]
            pixel_values = torch.stack(batch_tensors, dim=0).to(device)
            generated_ids = model.generate(pixel_values=pixel_values, max_new_tokens=max_new_tokens)

            records = []
            for sample_idx, (_, row) in enumerate(batch_df.iterrows()):
                prediction_text = _decode_prediction(model, generated_ids[sample_idx].detach().cpu())
                reference_text = str(row["report_text"]).strip()
                label_map = {label: int(row[label]) for label in CHEXPERT_LABEL_COLUMNS}
                records.append(
                    {
                        "subject_id": int(row["subject_id"]),
                        "study_id": int(row["study_id"]),
                        "dicom_id": str(row["dicom_id"]),
                        "image_path": row["processed_image_path"],
                        "processed_image_path": row["processed_image_path"],
                        "prediction": prediction_text,
                        "reference": reference_text,
                        **label_map,
                    }
                )
            _append_records_to_csv(chunk_path, records)
            completed = min(start_idx + len(batch_df), len(manifest))
            if completed % 100 == 0 or completed == len(manifest):
                LOGGER.info("Generated %s / %s records for %s", completed, len(manifest), chunk_path.name)

            del pixel_values, generated_ids
            release_cached_memory()
    return chunk_path


def _load_chunk_records(chunk_dir: Path) -> pd.DataFrame:
    chunk_files = sorted(chunk_dir.glob("chunk_*.csv"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk CSV files found in {chunk_dir}")
    frames = []
    for chunk_path in chunk_files:
        chunk_df = pd.read_csv(chunk_path)
        missing_columns = [column for column in _prediction_columns() if column not in chunk_df.columns]
        if missing_columns:
            raise RuntimeError(f"Chunk file {chunk_path} is missing columns: {missing_columns}")
        frames.append(chunk_df)
    merged_df = pd.concat(frames, ignore_index=True)
    merged_df = merged_df.sort_values(by=["subject_id", "study_id", "dicom_id"]).drop_duplicates(
        subset=["subject_id", "study_id"],
        keep="first",
    )
    return merged_df.reset_index(drop=True)


def _evaluate_saved_predictions(predictions_path: Path, mimic_root: Path) -> tuple[list[dict], dict]:
    records_df = pd.read_csv(predictions_path)
    missing_label_columns = [label for label in CHEXPERT_LABEL_COLUMNS if label not in records_df.columns]
    if missing_label_columns:
        chexpert_df = _load_chexpert_reference_labels(mimic_root)
        records_df = records_df.merge(chexpert_df, on=["subject_id", "study_id"], how="left")
    if records_df[CHEXPERT_LABEL_COLUMNS].isnull().any().any():
        missing = records_df[records_df[CHEXPERT_LABEL_COLUMNS].isnull().any(axis=1)][["subject_id", "study_id"]].drop_duplicates()
        raise RuntimeError(f"Missing CheXpert labels for {len(missing)} prediction rows.")

    metrics = _compute_metrics(
        records_df,
        split_name="test",
        subset_name="all frontal studies",
        view_filter="frontal-only (PA/AP)",
    )
    records = records_df.to_dict(orient="records")
    return records, metrics


def _filter_findings_only_records(records_df: pd.DataFrame, mimic_root: Path) -> pd.DataFrame:
    findings_lookup = _build_findings_only_lookup(mimic_root)
    filtered = records_df[
        records_df.apply(lambda row: (int(row["subject_id"]), int(row["study_id"])) in findings_lookup, axis=1)
    ].copy()
    if filtered.empty:
        raise RuntimeError("No findings-only rows matched the saved MIMIC test predictions.")
    return filtered


def _format_metric(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _repo_id_from_summary(summary: dict) -> str:
    repo_id = str(summary.get("repo_id", "") or "").strip()
    if repo_id:
        return repo_id
    repo_url = str(summary.get("repo_url", "") or "").strip().rstrip("/")
    marker = "huggingface.co/"
    if marker in repo_url:
        return repo_url.split(marker, 1)[1]
    return ""


def _collection_entries(repo_root: Path, trigger_run_dir: Path | None = None, trigger_repo_id: str | None = None) -> list[dict]:
    artifacts_root = repo_root / "artifacts"
    seeded_entries = []
    for item in LANA_COLLECTION_MODELS:
        entry = dict(item)
        entry["run_dir"] = artifacts_root / item["artifact_dir"]
        seeded_entries.append(entry)

    repo_id_by_run_dir: dict[Path, str] = {}
    model_name_by_run_dir: dict[Path, str] = {}
    for entry in seeded_entries:
        resolved = entry["run_dir"].resolve()
        repo_id_by_run_dir[resolved] = entry["repo_id"]
        model_name_by_run_dir[resolved] = entry["model_name"]

    entries = []
    seen: set[Path] = set()
    for summary_path in sorted(artifacts_root.glob("*/run_summary.json")):
        run_dir = summary_path.parent.resolve()
        if "benchmark" in run_dir.name.lower():
            continue
        if run_dir in seen:
            continue
        seen.add(run_dir)
        summary = _load_run_summary(run_dir)
        entries.append(
            {
                "artifact_dir": run_dir.name,
                "repo_id": repo_id_by_run_dir.get(run_dir, _repo_id_from_summary(summary)),
                "model_name": model_name_by_run_dir.get(run_dir, str(summary.get("run_name", run_dir.name))),
                "run_dir": run_dir,
                "summary": summary,
            }
        )

    if trigger_run_dir is not None and trigger_repo_id:
        trigger_resolved = trigger_run_dir.resolve()
        if not any(entry["run_dir"].resolve() == trigger_resolved for entry in entries):
            summary = _load_run_summary(trigger_resolved)
            entries.append(
                {
                    "artifact_dir": trigger_resolved.name,
                    "repo_id": trigger_repo_id,
                    "model_name": str(summary.get("run_name", trigger_resolved.name)),
                    "run_dir": trigger_resolved,
                    "summary": summary,
                }
            )
    ordered_entries = []
    for model_name in EXPERIMENT_DESCRIPTION_ORDER:
        for entry in entries:
            if entry["model_name"] == model_name:
                ordered_entries.append(entry)
    for entry in entries:
        if entry not in ordered_entries:
            ordered_entries.append(entry)
    return ordered_entries


def _load_saved_metrics_bundle(run_dir: Path) -> dict | None:
    metrics_path = run_dir / "evaluations" / "mimic_test_metrics.json"
    if metrics_path.exists():
        try:
            raw = metrics_path.read_text(encoding="utf-8").strip()
            if raw:
                return json.loads(raw)
            LOGGER.warning("Saved metrics bundle at %s is empty; falling back to run_summary.json.", metrics_path)
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("Saved metrics bundle at %s is invalid; falling back to run_summary.json: %s", metrics_path, exc)

    summary = _load_run_summary(run_dir)
    latest_evaluations = summary.get("latest_evaluations")
    if isinstance(latest_evaluations, dict) and latest_evaluations:
        all_test_metrics = latest_evaluations.get(ALL_TEST_KEY) or summary.get("latest_evaluation")
        findings_metrics = latest_evaluations.get(FINDINGS_ONLY_TEST_KEY)
        if all_test_metrics and findings_metrics:
            return _build_metrics_bundle(all_test_metrics, findings_metrics)

    all_test_metrics = summary.get("latest_evaluation")
    findings_metrics_path = run_dir / "evaluations" / "mimic_test_findings_only_metrics.json"
    if all_test_metrics and findings_metrics_path.exists():
        try:
            findings_raw = findings_metrics_path.read_text(encoding="utf-8").strip()
            if findings_raw:
                findings_metrics = json.loads(findings_raw)
                return _build_metrics_bundle(all_test_metrics, findings_metrics)
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("Findings-only metrics at %s are invalid: %s", findings_metrics_path, exc)

    return None


def _display_model_name(row: dict) -> str:
    if row.get("completed", True):
        return row["model_name"]
    return f"{row['model_name']} (Model still training)"


def _metric_row(metric_name: str, metrics_key: str, rows: list[dict], section_key: str) -> str:
    values = []
    for row in rows:
        metrics = row.get("metrics", {})
        section = metrics.get(section_key, {})
        values.append(f"`{_format_metric(section.get(metrics_key))}`")
    return f"| {metric_name} | " + " | ".join(values) + " |"


def _comparison_table_lines(title: str, rows: list[dict], section_key: str) -> list[str]:
    header = "| Metric | " + " | ".join(_display_model_name(row) for row in rows) + " |"
    separator = "| --- | " + " | ".join(["---"] * len(rows)) + " |"
    return [
        f"### {title}",
        "",
        header,
        separator,
        "| Run status | " + " | ".join(f"`{'Completed' if row.get('completed', True) else 'Model still training'}`" for row in rows) + " |",
        _metric_row("Number of studies", "num_examples", rows, section_key),
        _metric_row("ROUGE-L", "rouge_l", rows, section_key),
        _metric_row("BLEU-1", "bleu_1", rows, section_key),
        _metric_row("BLEU-4", "bleu_4", rows, section_key),
        _metric_row("METEOR", "meteor", rows, section_key),
        _metric_row("RadGraph F1", "radgraph_f1", rows, section_key),
        _metric_row("RadGraph entity F1", "radgraph_f1_entity", rows, section_key),
        _metric_row("RadGraph relation F1", "radgraph_f1_relation", rows, section_key),
        _metric_row("CheXpert F1 14-micro", "chexpert_f1_14_micro", rows, section_key),
        _metric_row("CheXpert F1 5-micro", "chexpert_f1_5_micro", rows, section_key),
        _metric_row("CheXpert F1 14-macro", "chexpert_f1_14_macro", rows, section_key),
        _metric_row("CheXpert F1 5-macro", "chexpert_f1_5_macro", rows, section_key),
        "",
    ]


def _build_collection_results_section(rows: list[dict]) -> str:
    return [
        "## MIMIC Test Results",
        "",
        "Frontal-only evaluation using `PA/AP` studies only.",
        "",
        "These comparison tables are refreshed across the full LAnA collection whenever any collection model is evaluated.",
        "",
        *_comparison_table_lines("Cross-Model Comparison: All Frontal Test Studies", rows, ALL_TEST_KEY),
        *_comparison_table_lines("Cross-Model Comparison: Findings-Only Frontal Test Studies", rows, FINDINGS_ONLY_TEST_KEY),
    ]


def _replace_mimic_results_section(current: str, mimic_results_section: str) -> str:
    updated = current
    top_level_mimic_pattern = re.compile(r"## MIMIC Test Results\s+.*?(?=\n## |\Z)", flags=re.DOTALL)
    if top_level_mimic_pattern.search(updated):
        updated = top_level_mimic_pattern.sub(mimic_results_section, updated, count=1)
    else:
        mimic_pattern = re.compile(re.escape(MIMIC_RESULTS_START) + r".*?" + re.escape(MIMIC_RESULTS_END), flags=re.DOTALL)
        if mimic_pattern.search(updated):
            updated = mimic_pattern.sub(mimic_results_section, updated)
        else:
            updated = updated.rstrip() + "\n\n" + mimic_results_section + "\n"

    marker_block_pattern = re.compile(r"\n*" + re.escape(MIMIC_RESULTS_START) + r".*?" + re.escape(MIMIC_RESULTS_END) + r"\n*", flags=re.DOTALL)
    updated = marker_block_pattern.sub("\n\n", updated)
    eval_block_pattern = re.compile(r"\n*" + re.escape(README_EVAL_START) + r".*?" + re.escape(README_EVAL_END) + r"\n*", flags=re.DOTALL)
    updated = eval_block_pattern.sub("\n\n", updated)
    updated = re.sub(r"\n{3,}", "\n\n", updated).rstrip() + "\n"
    return updated


def _build_experiment_descriptions_section(entries: list[dict]) -> str:
    entry_by_name = {entry["model_name"]: entry for entry in entries}
    ordered_names = list(EXPERIMENT_DESCRIPTION_ORDER)
    for entry in entries:
        if entry["model_name"] not in entry_by_name:
            continue
        if entry["model_name"] not in ordered_names:
            ordered_names.append(entry["model_name"])

    lines = [EXPERIMENT_DESCRIPTION_HEADING, ""]
    for model_name in ordered_names:
        text = EXPERIMENT_DESCRIPTION_TEXT.get(model_name, "")
        lines.append(f"- `{model_name}`: {text}".rstrip())
    lines.append("")
    return "\n".join(lines)


def _replace_or_insert_experiment_descriptions(current: str, experiment_section: str) -> str:
    pattern = re.compile(r"## Experiment Model Descriptions\s+.*?(?=\n## |\Z)", flags=re.DOTALL)
    if pattern.search(current):
        updated = pattern.sub(experiment_section, current, count=1)
    else:
        training_snapshot_pattern = re.compile(r"\n## Training Snapshot")
        if training_snapshot_pattern.search(current):
            updated = training_snapshot_pattern.sub("\n\n" + experiment_section + "\n\n## Training Snapshot", current, count=1)
        else:
            updated = current.rstrip() + "\n\n" + experiment_section + "\n"
    return re.sub(r"\n{3,}", "\n\n", updated)


def _best_model_repo_id_from_collection_rows(collection_entries: list[dict], collection_rows: list[dict]) -> str | None:
    if not collection_rows:
        return None
    repo_id_by_name = {entry["model_name"]: entry.get("repo_id", "") for entry in collection_entries}

    def sort_key(row: dict) -> tuple[float, float]:
        metrics = row.get("metrics", {}).get(ALL_TEST_KEY, row.get("metrics", {}))
        primary = metrics.get("chexpert_f1_14_micro")
        secondary = metrics.get("rouge_l")
        return (
            float(primary) if primary is not None else float("-inf"),
            float(secondary) if secondary is not None else float("-inf"),
        )

    best_row = max(collection_rows, key=sort_key)
    repo_id = repo_id_by_name.get(best_row["model_name"], "")
    return repo_id or None


def _load_run_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _update_model_card(run_dir: Path, repo_id: str, metrics_bundle: dict) -> None:
    readme_path = run_dir / "README.md"
    if readme_path.exists():
        current = readme_path.read_text(encoding="utf-8")
    else:
        current = f"# {run_dir.name}\n"

    summary = _load_run_summary(run_dir)
    run_completed = bool(summary.get("completed"))
    if run_completed:
        status_replacements = [
            (
                r"- Project status: `.*?`",
                "- Project status: `Training completed`",
            ),
            (
                r"- Release status: `.*?`",
                "- Release status: `Completed training run`",
            ),
            (
                r"- Current checkpoint status: `.*?`",
                "- Current checkpoint status: `Final completed run`",
            ),
            (
                r"- This section describes the current public checkpoint, not the final completed project\.",
                "- This section describes the completed public training run.",
            ),
        ]
        for pattern_text, replacement in status_replacements:
            current = re.sub(pattern_text, replacement, current)
    else:
        current = re.sub(r"- Project status: `.*?`", "- Project status: `Training in progress`", current)
    repo_root = Path(__file__).resolve().parents[1]
    collection_entries = _collection_entries(repo_root, run_dir, repo_id)
    experiment_section = _build_experiment_descriptions_section(collection_entries)
    updated = _replace_or_insert_experiment_descriptions(current, experiment_section)

    collection_rows = []
    for entry in collection_entries:
        saved_metrics = metrics_bundle if entry["run_dir"].resolve() == run_dir.resolve() else _load_saved_metrics_bundle(entry["run_dir"])
        if saved_metrics is None:
            continue
        entry_summary = entry.get("summary") or _load_run_summary(entry["run_dir"])
        collection_rows.append(
            {
                "model_name": entry["model_name"],
                "metrics": saved_metrics,
                "completed": bool(entry_summary.get("completed")),
            }
        )
    if not collection_rows:
        collection_rows = [{"model_name": run_dir.name, "metrics": metrics_bundle, "completed": bool(summary.get("completed"))}]
    best_repo_id = _best_model_repo_id_from_collection_rows(collection_entries, collection_rows)
    if best_repo_id:
        updated = upsert_best_model_notice(updated, best_repo_id)
    mimic_results_section = "\n".join(_build_collection_results_section(collection_rows))
    updated = _replace_mimic_results_section(updated, mimic_results_section)
    readme_path.write_text(updated, encoding="utf-8")


def _update_collection_model_cards(trigger_run_dir: Path, trigger_repo_id: str, metrics_bundle: dict) -> list[dict]:
    repo_root = Path(__file__).resolve().parents[1]
    updated_entries = []
    for entry in _collection_entries(repo_root, trigger_run_dir, trigger_repo_id):
        run_dir = entry["run_dir"]
        if not run_dir.exists():
            continue
        bundle = metrics_bundle if run_dir.resolve() == trigger_run_dir.resolve() else _load_saved_metrics_bundle(run_dir)
        if bundle is None:
            continue
        _update_model_card(run_dir, entry["repo_id"], bundle)
        updated_entries.append(entry)
    return updated_entries


def _merge_existing_metrics(metrics_path: Path, metrics: dict) -> dict:
    if not metrics_path.exists():
        return metrics
    try:
        raw = metrics_path.read_text(encoding="utf-8").strip()
        if not raw:
            return metrics
        existing = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Ignoring invalid existing metrics file at %s: %s", metrics_path, exc)
        return metrics
    if not metrics.get("radgraph_available") and existing.get("radgraph_f1") is not None:
        metrics["radgraph_f1"] = existing.get("radgraph_f1")
        metrics["radgraph_f1_entity"] = existing.get("radgraph_f1_entity")
        metrics["radgraph_f1_relation"] = existing.get("radgraph_f1_relation")
        metrics["radgraph_available"] = existing.get("radgraph_available", False)
        metrics["radgraph_error"] = existing.get("radgraph_error")
    return metrics


def _build_metrics_bundle(all_test_metrics: dict, findings_metrics: dict) -> dict:
    return {
        **all_test_metrics,
        "evaluation_suite": "mimic_test_dual",
        ALL_TEST_KEY: all_test_metrics,
        FINDINGS_ONLY_TEST_KEY: findings_metrics,
    }


def _generation_settings_payload(
    *,
    model_source: str,
    image_size: int,
    max_new_tokens: int,
    batch_size: int,
    output_tag: str,
) -> dict:
    return {
        "model_source": model_source,
        "image_size": int(image_size),
        "max_new_tokens": int(max_new_tokens),
        "batch_size": int(batch_size),
        "output_tag": output_tag,
    }


def _write_final_evaluation_outputs(
    *,
    run_dir: Path,
    repo_id: str,
    mimic_root: Path,
    records_df: pd.DataFrame,
    all_test_metrics: dict,
    output_tag: str,
    model_source: str,
    image_size: int,
    max_new_tokens: int,
    batch_size: int,
    skip_collection_model_card_update: bool,
    push_to_hub: bool,
) -> None:
    findings_records_df = _filter_findings_only_records(records_df, mimic_root)
    findings_metrics = _compute_metrics(
        findings_records_df,
        split_name="test",
        subset_name="findings-only frontal studies",
        view_filter="frontal-only (PA/AP), structured Findings section only",
    )

    evaluations_dir = run_dir / "evaluations"
    evaluations_dir.mkdir(parents=True, exist_ok=True)
    predictions_path, findings_predictions_path, metrics_path, findings_metrics_path = _evaluation_output_paths(
        evaluations_dir,
        output_tag,
    )
    all_test_metrics = _merge_existing_metrics(metrics_path, all_test_metrics)
    findings_metrics = _merge_existing_metrics(findings_metrics_path, findings_metrics)
    metrics_bundle = _build_metrics_bundle(all_test_metrics, findings_metrics)
    metrics_bundle.update(
        _generation_settings_payload(
            model_source=model_source,
            image_size=image_size,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            output_tag=output_tag,
        )
    )
    records_df.to_csv(predictions_path, index=False)
    findings_records_df.to_csv(findings_predictions_path, index=False)
    metrics_path.write_text(json.dumps(metrics_bundle, indent=2), encoding="utf-8")

    findings_metrics_payload = dict(findings_metrics)
    findings_metrics_payload.update(
        _generation_settings_payload(
            model_source=model_source,
            image_size=image_size,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            output_tag=output_tag,
        )
    )
    findings_metrics_path.write_text(json.dumps(findings_metrics_payload, indent=2), encoding="utf-8")

    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {}
    summary["latest_evaluation"] = all_test_metrics
    summary["latest_evaluations"] = {
        ALL_TEST_KEY: all_test_metrics,
        FINDINGS_ONLY_TEST_KEY: findings_metrics,
    }
    summary["latest_evaluation_settings"] = _generation_settings_payload(
        model_source=model_source,
        image_size=image_size,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        output_tag=output_tag,
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if skip_collection_model_card_update or model_source in {"space_legacy", "deletions_complete_model"}:
        updated_collection_entries = []
    else:
        updated_collection_entries = _update_collection_model_cards(run_dir, repo_id, metrics_bundle)

    LOGGER.info("Saved evaluation metrics to %s", metrics_path)
    LOGGER.info("Saved predictions to %s", predictions_path)
    LOGGER.info("Saved findings-only predictions to %s", findings_predictions_path)

    if push_to_hub:
        for entry in updated_collection_entries:
            if entry["run_dir"].exists():
                if entry["run_dir"].resolve() == run_dir.resolve() and entry["repo_id"] == repo_id:
                    continue
                if not entry["repo_id"]:
                    LOGGER.info("Skipping model-card push for %s because no repo_id is known.", entry["run_dir"])
                    continue
                try:
                    repo_url = push_model_card_update_to_hub(
                        str(entry["run_dir"]),
                        entry["repo_id"],
                        commit_message="Refresh LAnA collection comparison tables",
                    )
                    LOGGER.info("Uploaded updated model card to %s", repo_url)
                except Exception as exc:
                    LOGGER.warning("Skipping failed collection model-card push for %s: %s", entry["repo_id"], exc)
        repo_url = push_directory_to_hub(str(run_dir), repo_id, commit_message="Upload MIMIC test evaluation results")
        LOGGER.info("Uploaded current run artifacts to %s", repo_url)


def _evaluation_output_paths(evaluations_dir: Path, output_tag: str) -> tuple[Path, Path, Path, Path]:
    suffix = f"_{output_tag.strip()}" if output_tag.strip() else ""
    predictions_path = evaluations_dir / f"mimic_test_predictions{suffix}.csv"
    findings_predictions_path = evaluations_dir / f"mimic_test_findings_only_predictions{suffix}.csv"
    metrics_path = evaluations_dir / f"mimic_test_metrics{suffix}.json"
    findings_metrics_path = evaluations_dir / f"mimic_test_findings_only_metrics{suffix}.json"
    return predictions_path, findings_predictions_path, metrics_path, findings_metrics_path


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    release_cached_memory()

    try:
        run_dir = Path(args.run_dir) if args.run_dir else _latest_run_dir()
        mimic_root = Path(args.mimic_root)
        device = torch.device(args.device)
        resolved_model_source = _resolve_model_source(args.model_source, external_space_repo_id=args.external_space_repo_id)
        effective_batch_size = _resolve_effective_batch_size(args, resolved_model_source)
        output_tag = args.output_tag.strip()

        if args.generate_chunk and args.merge_chunks:
            raise ValueError("--generate-chunk and --merge-chunks cannot be used together.")
        if args.generate_all_chunks and args.merge_chunks:
            raise ValueError("--generate-all-chunks and --merge-chunks cannot be used together.")
        if args.generate_chunk and args.generate_all_chunks:
            raise ValueError("--generate-chunk and --generate-all-chunks cannot be used together.")

        if args.predictions_csv:
            predictions_path = Path(args.predictions_csv)
            LOGGER.info("Recomputing evaluation metrics from saved predictions at %s", predictions_path)
            records, all_test_metrics = _evaluate_saved_predictions(predictions_path=predictions_path, mimic_root=mimic_root)
            records_df = pd.DataFrame(records)
            _write_final_evaluation_outputs(
                run_dir=run_dir,
                repo_id=args.repo_id,
                mimic_root=mimic_root,
                records_df=records_df,
                all_test_metrics=all_test_metrics,
                output_tag=output_tag,
                model_source=resolved_model_source,
                image_size=args.image_size,
                max_new_tokens=args.max_new_tokens,
                batch_size=effective_batch_size,
                skip_collection_model_card_update=args.skip_collection_model_card_update,
                push_to_hub=args.push_to_hub,
            )
            return

        manifest = _build_mimic_test_manifest(mimic_root=mimic_root, limit=args.limit)
        manifest = _slice_manifest(manifest, args.start_idx, args.end_idx)

        if args.generate_chunk or args.generate_all_chunks:
            chunk_dir = _resolve_chunk_dir(run_dir, args.chunk_dir, output_tag)
            chunk_dir.mkdir(parents=True, exist_ok=True)
            ranges = []
            if args.generate_chunk:
                chunk_start = 0
                chunk_end = len(manifest)
                ranges = [(chunk_start, chunk_end)]
            else:
                chunk_size = max(1, int(args.chunk_size))
                for chunk_start in range(0, len(manifest), chunk_size):
                    chunk_end = min(chunk_start + chunk_size, len(manifest))
                    ranges.append((chunk_start, chunk_end))

            pending_ranges = []
            for chunk_start, chunk_end in ranges:
                expected_rows = chunk_end - chunk_start
                chunk_path = _chunk_csv_path(chunk_dir, chunk_start + args.start_idx, chunk_end + args.start_idx)
                if _is_valid_chunk_csv(chunk_path, expected_rows):
                    LOGGER.info("Skipping existing completed chunk %s", chunk_path.name)
                    continue
                pending_ranges.append((chunk_start, chunk_end, chunk_path))

            if not pending_ranges:
                LOGGER.info("All requested chunks already exist in %s", chunk_dir)
                return

            LOGGER.info("Evaluating run_dir=%s on %s MIMIC test studies", run_dir, len(manifest))
            model = _load_model(
                run_dir=run_dir,
                device=device,
                model_source=resolved_model_source,
                external_space_repo_id=args.external_space_repo_id,
                external_space_revision=args.external_space_revision,
                external_space_local_dir=args.external_space_local_dir,
                deletions_model_path=args.deletions_model_path,
                deletions_loader_path=args.deletions_loader_path,
            )

            for chunk_start, chunk_end, chunk_path in pending_ranges:
                chunk_manifest = manifest.iloc[chunk_start:chunk_end].reset_index(drop=True)
                LOGGER.info(
                    "Generating chunk %s (%s rows, model_source=%s, image_size=%s, max_new_tokens=%s, batch_size=%s)",
                    chunk_path.name,
                    len(chunk_manifest),
                    resolved_model_source,
                    args.image_size,
                    args.max_new_tokens,
                    effective_batch_size,
                )
                _generate_chunk_predictions(
                    model=model,
                    manifest=chunk_manifest,
                    device=device,
                    image_size=args.image_size,
                    max_new_tokens=args.max_new_tokens,
                    batch_size=effective_batch_size,
                    chunk_path=chunk_path,
                )
            model = None
            release_cached_memory()
            settings_path = chunk_dir / "chunk_generation_config.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "model_source": resolved_model_source,
                        "image_size": int(args.image_size),
                        "max_new_tokens": int(args.max_new_tokens),
                        "batch_size": int(effective_batch_size),
                        "start_idx": int(args.start_idx),
                        "end_idx": int(args.end_idx),
                        "limit": int(args.limit),
                        "output_tag": output_tag,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            LOGGER.info("Saved chunk predictions under %s", chunk_dir)
            return

        if args.merge_chunks:
            chunk_dir = _resolve_chunk_dir(run_dir, args.chunk_dir, output_tag)
            records_df = _load_chunk_records(chunk_dir)
            expected_examples = len(manifest)
            if len(records_df) != expected_examples:
                raise RuntimeError(
                    f"Merged chunk rows ({len(records_df)}) did not match expected manifest size ({expected_examples})."
                )
            all_test_metrics = _compute_metrics(
                records_df,
                split_name="test",
                subset_name="all frontal studies",
                view_filter="frontal-only (PA/AP)",
            )
        else:
            LOGGER.info("Evaluating run_dir=%s on %s MIMIC test studies", run_dir, len(manifest))
            model = _load_model(
                run_dir=run_dir,
                device=device,
                model_source=resolved_model_source,
                external_space_repo_id=args.external_space_repo_id,
                external_space_revision=args.external_space_revision,
                external_space_local_dir=args.external_space_local_dir,
                deletions_model_path=args.deletions_model_path,
                deletions_loader_path=args.deletions_loader_path,
            )
            records, all_test_metrics = _evaluate_model(
                model,
                manifest,
                device=device,
                image_size=args.image_size,
                max_new_tokens=args.max_new_tokens,
                batch_size=effective_batch_size,
            )
            records_df = pd.DataFrame(records)
            model = None
            release_cached_memory()

        _write_final_evaluation_outputs(
            run_dir=run_dir,
            repo_id=args.repo_id,
            mimic_root=mimic_root,
            records_df=records_df,
            all_test_metrics=all_test_metrics,
            output_tag=output_tag,
            model_source=resolved_model_source,
            image_size=args.image_size,
            max_new_tokens=args.max_new_tokens,
            batch_size=effective_batch_size,
            skip_collection_model_card_update=args.skip_collection_model_card_update,
            push_to_hub=args.push_to_hub,
        )
    finally:
        release_cached_memory()


if __name__ == "__main__":
    main()
