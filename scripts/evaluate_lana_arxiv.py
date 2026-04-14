from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import AutoConfig, GPT2Tokenizer

import evaluate as eval_mod


LOGGER = logging.getLogger("evaluate_lana_arxiv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the local LAnA-Arxiv package with a resumable chunked workflow."
    )
    parser.add_argument("--run-dir", default="artifacts/LAnA-Arxiv")
    parser.add_argument("--repo-id", default="manu02/LAnA-Arxiv")
    parser.add_argument("--mimic-root", default="Datasets/MIMIC")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument(
        "--max-new-tokens-list",
        default="",
        help="Optional comma-separated token sweep, e.g. '100,128,150'. Runs separate evaluations and saves tagged outputs for each.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--predictions-csv", default="")
    parser.add_argument("--chunk-dir", default="")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=-1)
    parser.add_argument("--generate-all-chunks", action="store_true")
    parser.add_argument("--merge-chunks", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument(
        "--overwrite-findings-only",
        action="store_true",
        help="Recompute and overwrite findings-only metrics instead of preserving existing paper values.",
    )
    parser.add_argument(
        "--refresh-readme",
        action="store_true",
        help="Refresh only the local LAnA-Arxiv README after writing metrics.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def _parse_token_sweep(raw: str) -> list[int]:
    values: list[int] = []
    for piece in raw.split(","):
        item = piece.strip()
        if not item:
            continue
        token_count = int(item)
        if token_count <= 0:
            raise ValueError("All token counts in --max-new-tokens-list must be positive.")
        values.append(token_count)
    deduped: list[int] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def _load_local_model(run_dir: Path, device: torch.device):
    config_path = run_dir / "configuration_lana_arxiv.py"
    image_processing_path = run_dir / "image_processing_lana_arxiv.py"
    processing_path = run_dir / "processing_lana_arxiv.py"
    modeling_path = run_dir / "modeling_lana_arxiv.py"

    for module_name in [
        "configuration_lana_arxiv",
        "image_processing_lana_arxiv",
        "processing_lana_arxiv",
        "modeling_lana_arxiv",
    ]:
        if module_name in sys.modules:
            del sys.modules[module_name]

    def _load_module(module_name: str, module_path: Path):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        sys.modules[module_name] = module
        return module

    _load_module("configuration_lana_arxiv", config_path)
    image_processing_mod = _load_module("image_processing_lana_arxiv", image_processing_path)
    processing_mod = _load_module("processing_lana_arxiv", processing_path)
    modeling_mod = _load_module("modeling_lana_arxiv", modeling_path)

    config = AutoConfig.from_pretrained(str(run_dir), trust_remote_code=True)
    image_processor = image_processing_mod.LanaArxivImageProcessor.from_pretrained(str(run_dir))
    tokenizer = GPT2Tokenizer.from_pretrained(str(run_dir))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    processor = processing_mod.LanaArxivProcessor(image_processor=image_processor, tokenizer=tokenizer)
    model = modeling_mod.LanaArxivForConditionalGeneration.from_pretrained(str(run_dir), config=config)
    if hasattr(model, "move_non_quantized_modules"):
        model.move_non_quantized_modules(device)
    else:
        model.to(device)
    model.eval()
    return processor, model


def _decode_prediction(processor, generated_ids: torch.Tensor) -> str:
    return processor.decode(generated_ids, skip_special_tokens=True).strip()


def _chunk_csv_path(chunk_dir: Path, start_idx: int, end_idx: int) -> Path:
    return chunk_dir / f"chunk_{start_idx:04d}_{end_idx - 1:04d}.csv"


def _resolve_chunk_dir(run_dir: Path, chunk_dir: str, output_tag: str) -> Path:
    if chunk_dir.strip():
        return Path(chunk_dir).resolve()
    suffix = f"_{output_tag.strip()}" if output_tag.strip() else ""
    return (run_dir / "evaluations" / f"chunks_lana_arxiv{suffix}").resolve()


def _generate_chunk_predictions(
    *,
    model,
    processor,
    manifest: pd.DataFrame,
    device: torch.device,
    max_new_tokens: int,
    batch_size: int,
    chunk_path: Path,
) -> None:
    if chunk_path.exists():
        chunk_path.unlink()

    with torch.inference_mode():
        for start_idx in range(0, len(manifest), batch_size):
            batch_df = manifest.iloc[start_idx : start_idx + batch_size]
            images = [Image.open(Path(row["processed_image_path"])).convert("RGB") for _, row in batch_df.iterrows()]
            encoded = processor(images=images, return_tensors="pt")
            pixel_values = encoded["pixel_values"].to(device)
            generated_ids = model.generate(pixel_values=pixel_values, max_new_tokens=max_new_tokens)

            records = []
            for sample_idx, (_, row) in enumerate(batch_df.iterrows()):
                prediction_text = _decode_prediction(processor, generated_ids[sample_idx].detach().cpu())
                records.append(
                    {
                        "subject_id": int(row["subject_id"]),
                        "study_id": int(row["study_id"]),
                        "dicom_id": str(row["dicom_id"]),
                        "image_path": str(row["processed_image_path"]),
                        "processed_image_path": str(row["processed_image_path"]),
                        "prediction": prediction_text,
                        "reference": str(row["report_text"]).strip(),
                        **{label: int(row[label]) for label in eval_mod.CHEXPERT_LABEL_COLUMNS},
                    }
                )
            pd.DataFrame.from_records(records).to_csv(
                chunk_path,
                mode="a",
                index=False,
                header=not chunk_path.exists() or chunk_path.stat().st_size == 0,
            )
            LOGGER.info("Generated %s / %s rows for %s", min(start_idx + batch_size, len(manifest)), len(manifest), chunk_path.name)
            eval_mod.release_cached_memory()


def _load_chunk_records(chunk_dir: Path) -> pd.DataFrame:
    chunk_paths = sorted(chunk_dir.glob("chunk_*.csv"))
    valid_paths = [path for path in chunk_paths if eval_mod._is_valid_chunk_csv(path, len(pd.read_csv(path)) if path.exists() else 0)]
    if not valid_paths:
        raise FileNotFoundError(f"No valid chunk CSVs found in {chunk_dir}")
    merged = pd.concat((pd.read_csv(path) for path in valid_paths), ignore_index=True)
    merged = merged.drop_duplicates(subset=["subject_id", "study_id"], keep="first")
    return merged.sort_values(by=["subject_id", "study_id"]).reset_index(drop=True)


def _metrics_nonempty(section: dict | None) -> bool:
    if not section:
        return False
    metric_keys = [
        "bleu_1",
        "bleu_4",
        "meteor",
        "rouge_l",
        "chexpert_f1_14_micro",
        "chexpert_f1_5_micro",
        "chexpert_f1_14_macro",
        "chexpert_f1_5_macro",
        "radgraph_f1",
    ]
    return any(section.get(key) is not None for key in metric_keys)


def _write_outputs(
    *,
    run_dir: Path,
    repo_id: str,
    mimic_root: Path,
    records_df: pd.DataFrame,
    output_tag: str,
    image_size: int,
    max_new_tokens: int,
    batch_size: int,
    overwrite_findings_only: bool,
    refresh_readme: bool,
) -> None:
    evaluations_dir = run_dir / "evaluations"
    evaluations_dir.mkdir(parents=True, exist_ok=True)
    predictions_path, findings_predictions_path, metrics_path, findings_metrics_path = eval_mod._evaluation_output_paths(
        evaluations_dir,
        output_tag,
    )

    all_test_metrics = eval_mod._compute_metrics(
        records_df,
        split_name="test",
        subset_name="all frontal studies",
        view_filter="frontal-only (PA/AP)",
    )
    findings_records_df = eval_mod._filter_findings_only_records(records_df, mimic_root)

    existing_bundle = {}
    if metrics_path.exists():
        try:
            existing_bundle = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            existing_bundle = {}

    existing_findings = existing_bundle.get(eval_mod.FINDINGS_ONLY_TEST_KEY)
    if overwrite_findings_only or not _metrics_nonempty(existing_findings):
        findings_metrics = eval_mod._compute_metrics(
            findings_records_df,
            split_name="test",
            subset_name="findings-only frontal studies",
            view_filter="frontal-only (PA/AP), structured Findings section only",
        )
    else:
        findings_metrics = existing_findings

    metrics_bundle = eval_mod._build_metrics_bundle(all_test_metrics, findings_metrics)
    metrics_bundle.update(
        eval_mod._generation_settings_payload(
            model_source="lana_arxiv_local_hf",
            image_size=image_size,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            output_tag=output_tag,
        )
    )

    records_df.to_csv(predictions_path, index=False)
    findings_records_df.to_csv(findings_predictions_path, index=False)
    metrics_path.write_text(json.dumps(metrics_bundle, indent=2), encoding="utf-8")

    findings_payload = dict(findings_metrics)
    findings_payload.update(
        eval_mod._generation_settings_payload(
            model_source="lana_arxiv_local_hf",
            image_size=image_size,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            output_tag=output_tag,
        )
    )
    findings_metrics_path.write_text(json.dumps(findings_payload, indent=2), encoding="utf-8")

    summary_path = run_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary["latest_evaluation"] = all_test_metrics
    summary["latest_evaluations"] = {
        eval_mod.ALL_TEST_KEY: all_test_metrics,
        eval_mod.FINDINGS_ONLY_TEST_KEY: findings_metrics,
    }
    summary["latest_evaluation_settings"] = eval_mod._generation_settings_payload(
        model_source="lana_arxiv_local_hf",
        image_size=image_size,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        output_tag=output_tag,
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if refresh_readme:
        eval_mod._update_model_card(run_dir, repo_id, metrics_bundle)

    LOGGER.info("Saved all-test metrics to %s", metrics_path)
    LOGGER.info("Saved findings-only metrics to %s", findings_metrics_path)
    LOGGER.info("Saved merged predictions to %s", predictions_path)


def _run_single_evaluation(
    *,
    run_dir: Path,
    repo_id: str,
    mimic_root: Path,
    device: torch.device,
    limit: int,
    start_idx: int,
    end_idx: int,
    chunk_dir_arg: str,
    chunk_size: int,
    batch_size: int,
    image_size: int,
    max_new_tokens: int,
    output_tag: str,
    overwrite_findings_only: bool,
    refresh_readme: bool,
    predictions_csv: str,
) -> dict:
    if predictions_csv:
        predictions_path = Path(predictions_csv).resolve()
        records, _ = eval_mod._evaluate_saved_predictions(predictions_path=predictions_path, mimic_root=mimic_root)
        records_df = pd.DataFrame(records)
        _write_outputs(
            run_dir=run_dir,
            repo_id=repo_id,
            mimic_root=mimic_root,
            records_df=records_df,
            output_tag=output_tag,
            image_size=image_size,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            overwrite_findings_only=overwrite_findings_only,
            refresh_readme=refresh_readme,
        )
    else:
        manifest = eval_mod._build_mimic_test_manifest(mimic_root=mimic_root, limit=limit)
        manifest = eval_mod._slice_manifest(manifest, start_idx, end_idx)
        chunk_dir = _resolve_chunk_dir(run_dir, chunk_dir_arg, output_tag)
        chunk_dir.mkdir(parents=True, exist_ok=True)

        processor, model = _load_local_model(run_dir, device)
        for chunk_start in range(0, len(manifest), max(1, int(chunk_size))):
            chunk_end = min(chunk_start + max(1, int(chunk_size)), len(manifest))
            chunk_path = _chunk_csv_path(chunk_dir, chunk_start + start_idx, chunk_end + start_idx)
            expected_rows = chunk_end - chunk_start
            if eval_mod._is_valid_chunk_csv(chunk_path, expected_rows):
                LOGGER.info("Skipping existing completed chunk %s", chunk_path.name)
                continue
            chunk_manifest = manifest.iloc[chunk_start:chunk_end].reset_index(drop=True)
            _generate_chunk_predictions(
                model=model,
                processor=processor,
                manifest=chunk_manifest,
                device=device,
                max_new_tokens=max_new_tokens,
                batch_size=max(1, int(batch_size)),
                chunk_path=chunk_path,
            )
        model = None
        processor = None
        eval_mod.release_cached_memory()

        records_df = _load_chunk_records(chunk_dir)
        if len(records_df) != len(manifest):
            raise RuntimeError(
                f"Merged chunk rows ({len(records_df)}) did not match expected manifest size ({len(manifest)})."
            )
        _write_outputs(
            run_dir=run_dir,
            repo_id=repo_id,
            mimic_root=mimic_root,
            records_df=records_df,
            output_tag=output_tag,
            image_size=image_size,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            overwrite_findings_only=overwrite_findings_only,
            refresh_readme=refresh_readme,
        )

    metrics_path = run_dir / "evaluations" / (
        f"mimic_test_metrics_{output_tag}.json" if output_tag else "mimic_test_metrics.json"
    )
    findings_metrics_path = run_dir / "evaluations" / (
        f"mimic_test_findings_only_metrics_{output_tag}.json"
        if output_tag
        else "mimic_test_findings_only_metrics.json"
    )
    return {
        "output_tag": output_tag,
        "max_new_tokens": max_new_tokens,
        "metrics_path": str(metrics_path),
        "findings_metrics_path": str(findings_metrics_path),
    }


def main() -> None:
    args = build_parser().parse_args()
    eval_mod.configure_logging(args.log_level)
    run_dir = Path(args.run_dir).resolve()
    mimic_root = Path(args.mimic_root).resolve()
    device = torch.device(args.device)
    output_tag = args.output_tag.strip()
    token_sweep = _parse_token_sweep(args.max_new_tokens_list)

    if token_sweep and any([args.generate_all_chunks, args.merge_chunks, args.predictions_csv]):
        raise ValueError("--max-new-tokens-list cannot be combined with chunk-only or predictions-only modes.")

    if not token_sweep and not any([args.generate_all_chunks, args.merge_chunks, args.run_all, args.predictions_csv]):
        args.run_all = True
        LOGGER.info("No explicit mode selected; defaulting to --run-all.")

    if token_sweep:
        summary_path = run_dir / "run_summary.json"
        original_summary_text = summary_path.read_text(encoding="utf-8") if summary_path.exists() else None
        sweep_results = []
        try:
            for token_count in token_sweep:
                token_tag = f"{token_count}_tokens"
                LOGGER.info("Running LAnA-Arxiv token sweep for max_new_tokens=%s", token_count)
                sweep_results.append(
                    _run_single_evaluation(
                        run_dir=run_dir,
                        repo_id=args.repo_id,
                        mimic_root=mimic_root,
                        device=device,
                        limit=args.limit,
                        start_idx=args.start_idx,
                        end_idx=args.end_idx,
                        chunk_dir_arg="",
                        chunk_size=args.chunk_size,
                        batch_size=args.batch_size,
                        image_size=args.image_size,
                        max_new_tokens=token_count,
                        output_tag=token_tag,
                        overwrite_findings_only=True,
                        refresh_readme=False,
                        predictions_csv="",
                    )
                )
        finally:
            if original_summary_text is not None:
                summary_path.write_text(original_summary_text, encoding="utf-8")

        summary_path = run_dir / "evaluations" / "lana_arxiv_token_sweep.json"
        summary_path.write_text(json.dumps({"runs": sweep_results}, indent=2), encoding="utf-8")
        LOGGER.info("Saved token sweep summary to %s", summary_path)
        return

    if args.predictions_csv:
        _run_single_evaluation(
            run_dir=run_dir,
            repo_id=args.repo_id,
            mimic_root=mimic_root,
            device=device,
            limit=args.limit,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            chunk_dir_arg=args.chunk_dir,
            chunk_size=args.chunk_size,
            batch_size=args.batch_size,
            image_size=args.image_size,
            max_new_tokens=args.max_new_tokens,
            output_tag=output_tag,
            overwrite_findings_only=args.overwrite_findings_only,
            refresh_readme=args.refresh_readme,
            predictions_csv=args.predictions_csv,
        )
        return

    manifest = eval_mod._build_mimic_test_manifest(mimic_root=mimic_root, limit=args.limit)
    manifest = eval_mod._slice_manifest(manifest, args.start_idx, args.end_idx)
    chunk_dir = _resolve_chunk_dir(run_dir, args.chunk_dir, output_tag)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    if args.generate_all_chunks or args.run_all:
        processor, model = _load_local_model(run_dir, device)
        for chunk_start in range(0, len(manifest), max(1, int(args.chunk_size))):
            chunk_end = min(chunk_start + max(1, int(args.chunk_size)), len(manifest))
            chunk_path = _chunk_csv_path(chunk_dir, chunk_start + args.start_idx, chunk_end + args.start_idx)
            expected_rows = chunk_end - chunk_start
            if eval_mod._is_valid_chunk_csv(chunk_path, expected_rows):
                LOGGER.info("Skipping existing completed chunk %s", chunk_path.name)
                continue
            chunk_manifest = manifest.iloc[chunk_start:chunk_end].reset_index(drop=True)
            _generate_chunk_predictions(
                model=model,
                processor=processor,
                manifest=chunk_manifest,
                device=device,
                max_new_tokens=args.max_new_tokens,
                batch_size=max(1, int(args.batch_size)),
                chunk_path=chunk_path,
            )
        model = None
        processor = None
        eval_mod.release_cached_memory()

    if args.merge_chunks or args.run_all:
        records_df = _load_chunk_records(chunk_dir)
        if len(records_df) != len(manifest):
            raise RuntimeError(
                f"Merged chunk rows ({len(records_df)}) did not match expected manifest size ({len(manifest)})."
            )
        _write_outputs(
            run_dir=run_dir,
            repo_id=args.repo_id,
            mimic_root=mimic_root,
            records_df=records_df,
            output_tag=output_tag,
            image_size=args.image_size,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            overwrite_findings_only=args.overwrite_findings_only,
            refresh_readme=args.refresh_readme,
        )


if __name__ == "__main__":
    main()
