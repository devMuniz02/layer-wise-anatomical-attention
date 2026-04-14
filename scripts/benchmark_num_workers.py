import argparse
import gc
import importlib.util
import json
import logging
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

LOGGER = logging.getLogger("benchmark_num_workers")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark DataLoader num_workers settings for the training pipeline.")
    parser.add_argument("--dataset", default="combined", choices=["chexpert", "mimic", "combined"])
    parser.add_argument("--metadata-path", default="Datasets/CheXpert/df_chexpert_plus_240401_findings.csv")
    parser.add_argument("--image-root", default="Datasets/CheXpert/images")
    parser.add_argument("--mimic-root", default="Datasets/MIMIC")
    parser.add_argument("--mimic-findings-only", action="store_true")
    parser.add_argument("--text-model-name", default="gpt2")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-batches", type=int, default=100)
    parser.add_argument("--worker-values", default="0,2,4,8")
    parser.add_argument("--skip-cached-resize", action="store_true")
    parser.add_argument("--cache-size-audit-path", default=".cache/image_size_audit.json")
    parser.add_argument("--output", default=".cache/num_workers_benchmark.json")
    parser.add_argument("--log-level", default="INFO")
    return parser


def load_train_module():
    train_path = Path(__file__).with_name("train.py")
    spec = importlib.util.spec_from_file_location("train_runtime", train_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_worker_values(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("worker-values must contain at least one integer.")
    return values


def build_train_manifest(train_mod, args) -> tuple[object, list[str]]:
    train_parts = []
    dataset_names = []
    if args.dataset in {"chexpert", "combined"}:
        train_parts.append(train_mod.build_chexpert_manifest(args.metadata_path, args.image_root, split="train"))
        dataset_names.append("CheXpert")
    if args.dataset in {"mimic", "combined"}:
        train_parts.append(
            train_mod.build_mimic_manifest(args.mimic_root, split="train", findings_only=args.mimic_findings_only)
        )
        dataset_names.append("MIMIC-CXR (findings-only)" if args.mimic_findings_only else "MIMIC-CXR")
    manifest = train_mod.combine_manifests(train_parts)
    return manifest, dataset_names


def benchmark_loader(train_mod, manifest, tokenizer, args, num_workers: int, skip_cached_resize: bool) -> dict:
    create_start = time.perf_counter()
    _, loader, _ = train_mod.make_dataloader(
        manifest,
        tokenizer,
        args.batch_size,
        args.image_size,
        num_workers,
        True,
        args.seed,
        skip_cached_resize=skip_cached_resize,
    )
    create_elapsed = time.perf_counter() - create_start

    iterator = iter(loader)
    fetch_times = []
    images_seen = 0
    batches_seen = 0
    batch_sizes = []

    while batches_seen < args.num_batches:
        batch_start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        fetch_elapsed = time.perf_counter() - batch_start
        fetch_times.append(fetch_elapsed)
        current_batch_size = int(batch["pixel_values"].shape[0])
        batch_sizes.append(current_batch_size)
        images_seen += current_batch_size
        batches_seen += 1

    del iterator
    del loader
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    if not fetch_times:
        raise RuntimeError(f"No batches were produced for num_workers={num_workers}.")

    first_batch_sec = fetch_times[0]
    steady_fetch_times = fetch_times[1:] or fetch_times
    steady_total_sec = sum(steady_fetch_times)
    steady_images = sum(batch_sizes[1:]) if len(batch_sizes) > 1 else images_seen

    return {
        "num_workers": num_workers,
        "loader_create_sec": create_elapsed,
        "batches_measured": batches_seen,
        "images_seen": images_seen,
        "first_batch_sec": first_batch_sec,
        "mean_batch_fetch_sec": sum(fetch_times) / len(fetch_times),
        "median_batch_fetch_sec": statistics.median(fetch_times),
        "steady_mean_batch_fetch_sec": sum(steady_fetch_times) / len(steady_fetch_times),
        "steady_images_per_sec": (steady_images / steady_total_sec) if steady_total_sec > 0 else 0.0,
    }


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    train_mod = load_train_module()
    train_mod.set_global_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    manifest, dataset_names = build_train_manifest(train_mod, args)
    skip_cached_resize = train_mod.should_skip_cached_resize(args)
    worker_values = parse_worker_values(args.worker_values)

    LOGGER.info(
        "Benchmarking num_workers=%s on %s train rows from %s | skip_cached_resize=%s",
        worker_values,
        len(manifest),
        ", ".join(dataset_names),
        skip_cached_resize,
    )

    results = []
    for num_workers in worker_values:
        LOGGER.info("Running benchmark for num_workers=%s", num_workers)
        try:
            result = benchmark_loader(train_mod, manifest, tokenizer, args, num_workers, skip_cached_resize)
            result["status"] = "ok"
            LOGGER.info(
                "num_workers=%s first_batch=%.4fs steady_mean=%.4fs steady_images_per_sec=%.2f",
                num_workers,
                result["first_batch_sec"],
                result["steady_mean_batch_fetch_sec"],
                result["steady_images_per_sec"],
            )
        except Exception as exc:
            result = {
                "num_workers": num_workers,
                "status": "failed",
                "error": str(exc),
            }
            LOGGER.warning("Benchmark failed for num_workers=%s: %s", num_workers, exc)
        results.append(result)

    successful = [item for item in results if item.get("status") == "ok"]
    best = max(successful, key=lambda item: item["steady_images_per_sec"]) if successful else None
    payload = {
        "dataset": args.dataset,
        "dataset_names": dataset_names,
        "train_rows": len(manifest),
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "num_batches": args.num_batches,
        "worker_values": worker_values,
        "skip_cached_resize": skip_cached_resize,
        "best_num_workers": best["num_workers"] if best is not None else None,
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if best is not None:
        LOGGER.info("Best num_workers=%s", best["num_workers"])
    else:
        LOGGER.warning("No successful worker benchmark results were recorded.")
    LOGGER.info("Wrote benchmark summary to %s", output_path.resolve())


if __name__ == "__main__":
    main()
