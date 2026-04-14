import argparse
import gc
import importlib.util
import json
import logging
import sys
import zipfile
from pathlib import Path
from types import MethodType
from typing import Any

import pandas as pd
import torch
from safetensors.torch import load_model as load_safetensors_model


LOGGER = logging.getLogger("reproduce_cloud_best_model7")

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

REQUIRED_CHUNK_COLUMNS = [
    "subject_id",
    "study_id",
    "dicom_id",
    "processed_image_path",
    "reference_text",
    "prediction_text",
]


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _ensure_transformers_compat() -> None:
    try:
        from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
    except Exception:
        return

    def _shim_encode_plus(self, *args, **kwargs):
        if args:
            first = args[0]
            if isinstance(first, (list, tuple)) and all(isinstance(token, str) for token in first):
                pair = None
                if len(args) > 1 and isinstance(args[1], (list, tuple)) and all(isinstance(token, str) for token in args[1]):
                    pair = list(args[1])
                token_ids_0 = self.convert_tokens_to_ids(list(first))
                token_ids_1 = self.convert_tokens_to_ids(pair) if pair is not None else None
                input_ids = self.build_inputs_with_special_tokens(token_ids_0, token_ids_1)
                token_type_ids = [0] * len(input_ids)
                attention_mask = [1] * len(input_ids)
                return {
                    "input_ids": input_ids,
                    "token_type_ids": token_type_ids,
                    "attention_mask": attention_mask,
                }
        if hasattr(self, "_encode_plus"):
            return self._encode_plus(*args, **kwargs)
        return self(*args, **kwargs)

    def _shim_build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if hasattr(self, "prepare_for_model"):
            prepared = self.prepare_for_model(
                token_ids_0,
                pair_ids=token_ids_1,
                add_special_tokens=True,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            return prepared["input_ids"]
        if token_ids_1 is None:
            return list(token_ids_0)
        return list(token_ids_0) + list(token_ids_1)

    for cls in (PreTrainedTokenizer, PreTrainedTokenizerFast):
        if cls is None:
            continue
        if not hasattr(cls, "encode_plus"):
            cls.encode_plus = _shim_encode_plus
        if not hasattr(cls, "build_inputs_with_special_tokens"):
            cls.build_inputs_with_special_tokens = _shim_build_inputs_with_special_tokens


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


def _parse_findings_section(report_text: str) -> str:
    normalized = report_text.replace("\r\n", "\n")
    import re

    match = re.search(r"FINDINGS:\s*(.*?)(?:\n\s*[A-Z ]+:\s|$)", normalized, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()
    impression_match = re.search(r"IMPRESSION:\s*(.*?)(?:\n\s*[A-Z ]+:\s|$)", normalized, flags=re.IGNORECASE | re.DOTALL)
    if impression_match:
        return re.sub(r"\s+", " ", impression_match.group(1)).strip()
    return re.sub(r"\s+", " ", normalized).strip()


def _load_report_texts(report_zip_path: Path) -> dict[tuple[int, int], str]:
    reports: dict[tuple[int, int], str] = {}
    import re

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
    df["reference_text"] = df.apply(lambda row: reports.get((int(row["subject_id"]), int(row["study_id"])), ""), axis=1)
    for label in CHEXPERT_LABEL_COLUMNS:
        df[label] = df[label].fillna(0.0).map(lambda value: 1 if float(value) == 1.0 else 0)

    df = df[df["reference_text"].map(bool)]
    df = df[df["processed_image_path"].map(lambda value: Path(value).exists())]
    if limit > 0:
        df = df.head(limit)
    if df.empty:
        raise RuntimeError("No usable MIMIC test examples were found after joins and file checks.")
    return df.reset_index(drop=True)


def _build_findings_only_lookup(mimic_root: Path) -> set[tuple[int, int]]:
    findings_df = pd.read_csv(mimic_root / "mimic-cxr-2.0.0-metadata-findings-only.csv")
    return {
        (int(row.subject_id), int(row.study_id))
        for row in findings_df[["subject_id", "study_id"]].drop_duplicates().itertuples(index=False)
    }


def _chunk_csv_path(chunk_dir: Path, start_idx: int, end_idx: int, output_tag: str) -> Path:
    suffix = f"_{output_tag}" if output_tag else ""
    return chunk_dir / f"chunk_{start_idx:04d}_{end_idx - 1:04d}{suffix}.csv"


def _is_valid_chunk_csv(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    return all(column in df.columns for column in REQUIRED_CHUNK_COLUMNS) and not df.empty


def _append_records_to_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records(records)
    write_header = not path.exists() or path.stat().st_size == 0
    frame.to_csv(path, mode="a", index=False, header=write_header)


def _load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module


def _load_deletions_modules(loader_path: Path) -> tuple[Any, Any, Any]:
    loader_path = loader_path.resolve()
    metrics_root = Path(__file__).resolve().parents[1] / "deletions"

    if loader_path.parent.name == "models":
        deletions_root = loader_path.parents[2]
        if str(deletions_root) not in sys.path:
            sys.path.insert(0, str(deletions_root))
        from utils.models.complete_model import create_complete_model  # type: ignore
        from utils.processing import image_transform  # type: ignore
    else:
        package_root = loader_path.parents[1]
        package_root_str = str(package_root)
        if package_root_str not in sys.path:
            sys.path.insert(0, package_root_str)
        for module_name in [name for name in list(sys.modules) if name == "utils" or name.startswith("utils.")]:
            del sys.modules[module_name]
        create_complete_model = _load_module_from_path("lana_arxiv_legacy_complete_model", loader_path).create_complete_model
        processing_module = _load_module_from_path("lana_arxiv_legacy_processing", loader_path.parent / "processing.py")
        image_transform = processing_module.image_transform

    text_metrics_module = _load_module_from_path(
        "legacy_deletions_text_metrics",
        metrics_root / "utils" / "text_metrics.py",
    )
    evaluate_all_metrics = text_metrics_module.evaluate_all_metrics
    save_metrics_to_json = text_metrics_module.save_metrics_to_json

    return create_complete_model, image_transform, (evaluate_all_metrics, save_metrics_to_json)


def _load_legacy_model(args: argparse.Namespace) -> Any:
    create_complete_model, _, _ = _load_deletions_modules(Path(args.deletions_loader_path))
    model = create_complete_model(device=args.device)
    decoder = getattr(model, "decoder", None)
    transformer = getattr(decoder, "transformer", None)
    if decoder is not None and not hasattr(decoder, "model_parallel"):
        decoder.model_parallel = False
    if transformer is not None and not hasattr(transformer, "get_head_mask"):
        def _legacy_get_head_mask(this, head_mask, num_hidden_layers, is_attention_chunked: bool = False):
            if head_mask is None:
                return [None] * num_hidden_layers
            return head_mask

        transformer.get_head_mask = MethodType(_legacy_get_head_mask, transformer)
    if transformer is not None and not hasattr(transformer, "model_parallel"):
        transformer.model_parallel = False
    checkpoint_path = Path(args.model_path)
    if checkpoint_path.suffix.lower() in {".safetensor", ".safetensors"}:
        load_safetensors_model(model, str(checkpoint_path))
    else:
        checkpoint = torch.load(args.model_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            LOGGER.warning("Legacy model loaded non-strictly. Missing=%s Unexpected=%s", missing, unexpected)
    model.eval()
    return model


def _generate_chunk(
    model: Any,
    manifest: pd.DataFrame,
    transform: Any,
    start_idx: int,
    end_idx: int,
    chunk_csv_path: Path,
    max_new_tokens: int,
) -> Path:
    if _is_valid_chunk_csv(chunk_csv_path):
        LOGGER.info("Skipping existing valid chunk: %s", chunk_csv_path)
        return chunk_csv_path

    if chunk_csv_path.exists():
        chunk_csv_path.unlink()

    records: list[dict[str, Any]] = []
    chunk_df = manifest.iloc[start_idx:end_idx].reset_index(drop=True)
    total = len(chunk_df)
    for local_idx, (_, row) in enumerate(chunk_df.iterrows(), start=1):
        image_path = Path(str(row["processed_image_path"]))
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        pixel_values = transform(image).unsqueeze(0).to(args.device)
        with torch.inference_mode():
            _, generated_texts, _ = model.generate(pixel_values=pixel_values, max_new_tokens=max_new_tokens, output_attentions=False)
        prediction_text = generated_texts[0] if generated_texts else ""
        record = {
            "subject_id": int(row["subject_id"]),
            "study_id": int(row["study_id"]),
            "dicom_id": str(row["dicom_id"]),
            "processed_image_path": str(image_path),
            "reference_text": str(row["reference_text"]),
            "prediction_text": str(prediction_text),
        }
        for label in CHEXPERT_LABEL_COLUMNS:
            record[label] = int(row[label])
        records.append(record)
        _append_records_to_csv(chunk_csv_path, [record])
        records.clear()
        if local_idx % 10 == 0 or local_idx == total:
            LOGGER.info("Generated %s / %s for chunk %s", local_idx, total, chunk_csv_path.name)
        release_cached_memory()
    return chunk_csv_path


def _merge_chunk_predictions(chunk_dir: Path) -> pd.DataFrame:
    chunk_paths = sorted(path for path in chunk_dir.glob("chunk_*.csv") if _is_valid_chunk_csv(path))
    if not chunk_paths:
        raise FileNotFoundError(f"No valid chunk CSVs found in {chunk_dir}")
    merged = pd.concat((pd.read_csv(path) for path in chunk_paths), ignore_index=True)
    merged = merged.drop_duplicates(subset=["subject_id", "study_id"], keep="first")
    merged = merged.sort_values(by=["subject_id", "study_id"]).reset_index(drop=True)
    return merged


def _compare_against_target(metrics: dict[str, Any], target_path: Path) -> dict[str, dict[str, float]]:
    target = json.loads(target_path.read_text(encoding="utf-8"))
    deltas: dict[str, dict[str, float]] = {}
    for key, value in metrics.items():
        if key in target and isinstance(value, (float, int)) and isinstance(target[key], (float, int)):
            deltas[key] = {
                "current": float(value),
                "target": float(target[key]),
                "delta": float(value) - float(target[key]),
            }
    return deltas


def _rich_legacy_metrics(evaluate_all_metrics, predictions: list[str], references: list[str]) -> dict[str, Any]:
    rich_metrics = evaluate_all_metrics(predictions, references, evaluation_mode="legacy_full")
    chexagent_metrics = evaluate_all_metrics(predictions, references, evaluation_mode="CheXagent")
    merged = dict(rich_metrics)
    merged.update(chexagent_metrics)
    return merged


def _parse_token_sweep(raw: str) -> list[int]:
    values: list[int] = []
    for piece in raw.split(","):
        item = piece.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("All token counts in --max-new-tokens-list must be positive.")
        values.append(value)
    deduped: list[int] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def _run_single_pass(args: argparse.Namespace, *, max_new_tokens: int, output_tag: str) -> dict[str, Any]:
    mimic_root = Path(args.mimic_root)
    run_dir = Path(args.run_dir)
    chunk_dir = (Path(args.chunk_dir) if args.chunk_dir else (run_dir / "evaluations" / f"chunks_{output_tag}"))
    chunk_dir.mkdir(parents=True, exist_ok=True)

    manifest = _build_mimic_test_manifest(mimic_root=mimic_root, limit=args.limit, findings_only=args.findings_only_only)
    end_idx = len(manifest) if args.end_idx < 0 else min(args.end_idx, len(manifest))
    if args.start_idx < 0 or args.start_idx >= end_idx:
        raise ValueError("Invalid start/end index range.")

    _, image_transform, metrics_modules = _load_deletions_modules(Path(args.deletions_loader_path))
    evaluate_all_metrics, save_metrics_to_json = metrics_modules
    transform = image_transform(img_size=args.image_size)

    model = _load_legacy_model(args)
    current = args.start_idx
    while current < end_idx:
        next_end = min(current + args.chunk_size, end_idx)
        chunk_path = _chunk_csv_path(chunk_dir, current, next_end, output_tag)
        _generate_chunk(model, manifest, transform, current, next_end, chunk_path, max_new_tokens)
        current = next_end
    del model
    release_cached_memory()

    merged = _merge_chunk_predictions(chunk_dir)
    evaluations_dir = run_dir / "evaluations"
    evaluations_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{output_tag}" if output_tag else ""
    predictions_path = evaluations_dir / f"cloud_best_model7_predictions{suffix}.csv"
    merged.to_csv(predictions_path, index=False)

    findings_lookup = _build_findings_only_lookup(mimic_root)
    findings_only = merged[
        merged.apply(lambda row: (int(row["subject_id"]), int(row["study_id"])) in findings_lookup, axis=1)
    ].reset_index(drop=True)
    findings_path = evaluations_dir / f"cloud_best_model7_findings_only_predictions{suffix}.csv"
    findings_only.to_csv(findings_path, index=False)

    all_metrics = _rich_legacy_metrics(
        evaluate_all_metrics,
        merged["prediction_text"].fillna("").tolist(),
        merged["reference_text"].fillna("").tolist(),
    )
    findings_metrics = _rich_legacy_metrics(
        evaluate_all_metrics,
        findings_only["prediction_text"].fillna("").tolist(),
        findings_only["reference_text"].fillna("").tolist(),
    )
    metrics_path = evaluations_dir / f"cloud_best_model7_metrics{suffix}.json"
    findings_metrics_path = evaluations_dir / f"cloud_best_model7_findings_only_metrics{suffix}.json"
    save_metrics_to_json(all_metrics, str(metrics_path))
    save_metrics_to_json(findings_metrics, str(findings_metrics_path))

    comparison = {
        "all_test": _compare_against_target(all_metrics, Path(args.target_json)),
        "findings_only_test": _compare_against_target(findings_metrics, Path(args.target_json)),
    }
    comparison_path = evaluations_dir / f"cloud_best_model7_comparison{suffix}.json"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    LOGGER.info("Saved predictions to %s", predictions_path)
    LOGGER.info("Saved all-test metrics to %s", metrics_path)
    LOGGER.info("Saved findings-only metrics to %s", findings_metrics_path)
    LOGGER.info("Saved target comparison to %s", comparison_path)
    return {
        "output_tag": output_tag,
        "max_new_tokens": max_new_tokens,
        "findings_only_only": bool(args.findings_only_only),
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "findings_metrics_path": str(findings_metrics_path),
        "comparison_path": str(comparison_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce cloud_best_model_7 metrics with the legacy deletions pipeline.")
    parser.add_argument("--model-path", default="deletions/models/model_best7.pth")
    parser.add_argument("--deletions-loader-path", default="deletions/utils/models/complete_model.py")
    parser.add_argument("--mimic-root", default="Datasets/MIMIC")
    parser.add_argument("--run-dir", default="artifacts/repro_cloud_best_model7")
    parser.add_argument("--chunk-dir", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument(
        "--max-new-tokens-list",
        default="",
        help="Optional comma-separated token sweep, e.g. '100,128,150'. Runs the exact legacy pipeline separately for each value.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=-1)
    parser.add_argument("--output-tag", default="")
    parser.add_argument(
        "--findings-only-only",
        action="store_true",
        help="Generate and score only the findings-only subset to reduce runtime.",
    )
    parser.add_argument("--generate-chunk", action="store_true")
    parser.add_argument("--generate-all-chunks", action="store_true")
    parser.add_argument("--merge-score", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument(
        "--target-json",
        default="deletions/ReportGeneration/experiments/lstm_vs_gpt/results/cloud_best_model_7_MIMIC.json",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    global args
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)
    _ensure_transformers_compat()
    token_sweep = _parse_token_sweep(args.max_new_tokens_list)

    if token_sweep and any([args.generate_chunk, args.generate_all_chunks, args.merge_score]):
        raise ValueError("--max-new-tokens-list cannot be combined with explicit chunk-only/merge-only modes.")

    if token_sweep:
        sweep_results = []
        for token_count in token_sweep:
            output_tag = f"{args.output_tag}_{token_count}".strip("_") if args.output_tag else f"{token_count}_tokens"
            LOGGER.info("Running exact legacy sweep for max_new_tokens=%s", token_count)
            sweep_results.append(_run_single_pass(args, max_new_tokens=token_count, output_tag=output_tag))
        sweep_summary_path = Path(args.run_dir) / "evaluations" / "cloud_best_model7_token_sweep.json"
        sweep_summary_path.parent.mkdir(parents=True, exist_ok=True)
        sweep_summary_path.write_text(json.dumps({"runs": sweep_results}, indent=2), encoding="utf-8")
        LOGGER.info("Saved sweep summary to %s", sweep_summary_path)
        return

    if not any([args.generate_chunk, args.generate_all_chunks, args.merge_score, args.run_all]):
        args.run_all = True
        LOGGER.info("No explicit mode selected; defaulting to --run-all.")

    mimic_root = Path(args.mimic_root)
    run_dir = Path(args.run_dir)
    chunk_dir = Path(args.chunk_dir) if args.chunk_dir else (run_dir / "evaluations" / "chunks")
    chunk_dir.mkdir(parents=True, exist_ok=True)

    manifest = _build_mimic_test_manifest(mimic_root=mimic_root, limit=args.limit, findings_only=args.findings_only_only)
    end_idx = len(manifest) if args.end_idx < 0 else min(args.end_idx, len(manifest))
    if args.start_idx < 0 or args.start_idx >= end_idx:
        raise ValueError("Invalid start/end index range.")

    _, image_transform, metrics_modules = _load_deletions_modules(Path(args.deletions_loader_path))
    evaluate_all_metrics, save_metrics_to_json = metrics_modules
    transform = image_transform(img_size=args.image_size)

    should_generate = args.generate_chunk or args.generate_all_chunks or args.run_all
    if should_generate:
        model = _load_legacy_model(args)
        if args.generate_chunk:
            chunk_path = _chunk_csv_path(chunk_dir, args.start_idx, end_idx, args.output_tag)
            _generate_chunk(model, manifest, transform, args.start_idx, end_idx, chunk_path, args.max_new_tokens)
        else:
            current = args.start_idx
            while current < end_idx:
                next_end = min(current + args.chunk_size, end_idx)
                chunk_path = _chunk_csv_path(chunk_dir, current, next_end, args.output_tag)
                _generate_chunk(model, manifest, transform, current, next_end, chunk_path, args.max_new_tokens)
                current = next_end
        del model
        release_cached_memory()

    if args.merge_score or args.run_all:
        merged = _merge_chunk_predictions(chunk_dir)
        evaluations_dir = run_dir / "evaluations"
        evaluations_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{args.output_tag}" if args.output_tag else ""
        predictions_path = evaluations_dir / f"cloud_best_model7_predictions{suffix}.csv"
        merged.to_csv(predictions_path, index=False)

        findings_lookup = _build_findings_only_lookup(mimic_root)
        findings_only = merged[
            merged.apply(lambda row: (int(row["subject_id"]), int(row["study_id"])) in findings_lookup, axis=1)
        ].reset_index(drop=True)
        findings_path = evaluations_dir / f"cloud_best_model7_findings_only_predictions{suffix}.csv"
        findings_only.to_csv(findings_path, index=False)

        all_metrics = _rich_legacy_metrics(
            evaluate_all_metrics,
            merged["prediction_text"].fillna("").tolist(),
            merged["reference_text"].fillna("").tolist(),
        )
        findings_metrics = _rich_legacy_metrics(
            evaluate_all_metrics,
            findings_only["prediction_text"].fillna("").tolist(),
            findings_only["reference_text"].fillna("").tolist(),
        )
        metrics_path = evaluations_dir / f"cloud_best_model7_metrics{suffix}.json"
        findings_metrics_path = evaluations_dir / f"cloud_best_model7_findings_only_metrics{suffix}.json"
        save_metrics_to_json(all_metrics, str(metrics_path))
        save_metrics_to_json(findings_metrics, str(findings_metrics_path))

        comparison = {
            "all_test": _compare_against_target(all_metrics, Path(args.target_json)),
            "findings_only_test": _compare_against_target(findings_metrics, Path(args.target_json)),
        }
        comparison_path = evaluations_dir / f"cloud_best_model7_comparison{suffix}.json"
        comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

        LOGGER.info("Saved predictions to %s", predictions_path)
        LOGGER.info("Saved all-test metrics to %s", metrics_path)
        LOGGER.info("Saved findings-only metrics to %s", findings_metrics_path)
        LOGGER.info("Saved target comparison to %s", comparison_path)


if __name__ == "__main__":
    main()
