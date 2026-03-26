import argparse
import importlib.util
import json
import logging
import re
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from PIL import Image
from safetensors.torch import load_file

from lana_radgen import LanaConfig, LanaForConditionalGeneration
from lana_radgen.hub import push_directory_to_hub
from lana_radgen.logging_utils import configure_logging
from lana_radgen.metrics import chexpert_label_f1_from_reference_labels, radgraph_f1

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
README_EVAL_END = "<!-- EVAL_RESULTS_END -->"
MIMIC_RESULTS_START = "<!-- MIMIC_TEST_RESULTS_START -->"
MIMIC_RESULTS_END = "<!-- MIMIC_TEST_RESULTS_END -->"
ALL_TEST_KEY = "all_test"
FINDINGS_ONLY_TEST_KEY = "findings_only_test"


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the latest exported LANA model on the MIMIC test split.")
    parser.add_argument("--run-dir", default="", help="Training artifact directory. Defaults to the most recently modified artifacts/* run.")
    parser.add_argument("--repo-id", default="manu02/lana-radgen-benchmark")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of test examples.")
    parser.add_argument("--mimic-root", default="Datasets/MIMIC")
    parser.add_argument("--predictions-csv", default="", help="Optional existing predictions CSV to recompute metrics without running generation.")
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


def _load_model(run_dir: Path, device: torch.device) -> LanaForConditionalGeneration:
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
                segmentation_model_name=summary["segmentation_model_name"],
                lung_segmenter_checkpoint=summary["lung_segmenter_checkpoint"],
                heart_segmenter_checkpoint=summary["heart_segmenter_checkpoint"],
                disable_segmentation_mask=False,
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
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)

def _results_table_lines(title: str, metrics: dict) -> list[str]:
    return [
        f"### {title}",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Number of studies | `{metrics['num_examples']}` |",
        f"| RadGraph F1 | `{_format_metric(metrics['radgraph_f1'])}` |",
        f"| RadGraph entity F1 | `{_format_metric(metrics['radgraph_f1_entity'])}` |",
        f"| RadGraph relation F1 | `{_format_metric(metrics['radgraph_f1_relation'])}` |",
        f"| CheXpert F1 14-micro | `{_format_metric(metrics['chexpert_f1_14_micro'])}` |",
        f"| CheXpert F1 5-micro | `{_format_metric(metrics['chexpert_f1_5_micro'])}` |",
        f"| CheXpert F1 14-macro | `{_format_metric(metrics['chexpert_f1_14_macro'])}` |",
        f"| CheXpert F1 5-macro | `{_format_metric(metrics['chexpert_f1_5_macro'])}` |",
        "",
    ]


def _load_run_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _update_model_card(run_dir: Path, metrics_bundle: dict) -> None:
    readme_path = run_dir / "README.md"
    if readme_path.exists():
        current = readme_path.read_text(encoding="utf-8")
    else:
        current = f"# {run_dir.name}\n"

    summary = _load_run_summary(run_dir)
    run_completed = bool(summary.get("completed"))
    all_test_metrics = metrics_bundle[ALL_TEST_KEY]
    findings_metrics = metrics_bundle[FINDINGS_ONLY_TEST_KEY]

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
                r"- Current published metrics are .*",
                "- Current published metrics correspond to the completed training run.",
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

    if run_completed:
        mimic_results_section = "\n".join(
            [
                "## MIMIC Test Results",
                "",
                "Frontal-only evaluation using `PA/AP` studies only.",
                "",
                "### Final Completed Training Results",
                "",
                "These final-report metrics correspond to the completed training run.",
                "",
                *_results_table_lines("All Frontal Test Studies", all_test_metrics),
                *_results_table_lines("Findings-Only Frontal Test Studies", findings_metrics),
            ]
        )
    else:
        mimic_results_section = "\n".join(
            [
                "## MIMIC Test Results",
                "",
                "Frontal-only evaluation using `PA/AP` studies only.",
                "",
                "### Current Checkpoint Results",
                "",
                *_results_table_lines("All Frontal Test Studies", all_test_metrics),
                *_results_table_lines("Findings-Only Frontal Test Studies", findings_metrics),
                "### Final Completed Training Results",
                "",
                "The final table will be populated when the planned training run is completed. Until then, final-report metrics remain `TBD`.",
                "",
                "| Metric | Value |",
                "| --- | --- |",
                "| Number of studies | TBD |",
                "| RadGraph F1 | TBD |",
                "| RadGraph entity F1 | TBD |",
                "| RadGraph relation F1 | TBD |",
                "| CheXpert F1 14-micro | TBD |",
                "| CheXpert F1 5-micro | TBD |",
                "| CheXpert F1 14-macro | TBD |",
                "| CheXpert F1 5-macro | TBD |",
                "",
            ]
        )

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

    readme_path.write_text(updated, encoding="utf-8")


def _merge_existing_metrics(metrics_path: Path, metrics: dict) -> dict:
    if not metrics_path.exists():
        return metrics
    existing = json.loads(metrics_path.read_text(encoding="utf-8"))
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


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level)

    run_dir = Path(args.run_dir) if args.run_dir else _latest_run_dir()
    mimic_root = Path(args.mimic_root)
    device = torch.device(args.device)
    if args.predictions_csv:
        predictions_path = Path(args.predictions_csv)
        LOGGER.info("Recomputing evaluation metrics from saved predictions at %s", predictions_path)
        records, all_test_metrics = _evaluate_saved_predictions(predictions_path=predictions_path, mimic_root=mimic_root)
    else:
        manifest = _build_mimic_test_manifest(mimic_root=mimic_root, limit=args.limit)
        LOGGER.info("Evaluating run_dir=%s on %s MIMIC test studies", run_dir, len(manifest))

        model = _load_model(run_dir=run_dir, device=device)
        records, all_test_metrics = _evaluate_model(
            model,
            manifest,
            device=device,
            image_size=args.image_size,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )

    records_df = pd.DataFrame(records)
    findings_records_df = _filter_findings_only_records(records_df, mimic_root)
    findings_metrics = _compute_metrics(
        findings_records_df,
        split_name="test",
        subset_name="findings-only frontal studies",
        view_filter="frontal-only (PA/AP), structured Findings section only",
    )

    evaluations_dir = run_dir / "evaluations"
    evaluations_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = evaluations_dir / "mimic_test_predictions.csv"
    findings_predictions_path = evaluations_dir / "mimic_test_findings_only_predictions.csv"
    metrics_path = evaluations_dir / "mimic_test_metrics.json"
    all_test_metrics = _merge_existing_metrics(metrics_path, all_test_metrics)
    findings_metrics = _merge_existing_metrics(
        evaluations_dir / "mimic_test_findings_only_metrics.json",
        findings_metrics,
    )
    metrics_bundle = _build_metrics_bundle(all_test_metrics, findings_metrics)
    records_df.to_csv(predictions_path, index=False)
    findings_records_df.to_csv(findings_predictions_path, index=False)
    metrics_path.write_text(json.dumps(metrics_bundle, indent=2), encoding="utf-8")
    (evaluations_dir / "mimic_test_findings_only_metrics.json").write_text(json.dumps(findings_metrics, indent=2), encoding="utf-8")

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
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _update_model_card(run_dir, metrics_bundle)
    LOGGER.info("Saved evaluation metrics to %s", metrics_path)
    LOGGER.info("Saved predictions to %s", predictions_path)
    LOGGER.info("Saved findings-only predictions to %s", findings_predictions_path)

    if args.push_to_hub:
        repo_url = push_directory_to_hub(str(run_dir), args.repo_id, commit_message="Upload MIMIC test evaluation results")
        LOGGER.info("Uploaded updated model card and evaluation artifacts to %s", repo_url)


if __name__ == "__main__":
    main()
