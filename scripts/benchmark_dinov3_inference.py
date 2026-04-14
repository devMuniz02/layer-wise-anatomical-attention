import argparse
import json
import logging
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel

from lana_radgen.logging_utils import configure_logging

LOGGER = logging.getLogger("benchmark_dinov3_inference")

LVD_DINOV3_MODELS = [
    "facebook/dinov3-vits16-pretrain-lvd1689m",
    "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    "facebook/dinov3-convnext-small-pretrain-lvd1689m",
    "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "facebook/dinov3-convnext-base-pretrain-lvd1689m",
    "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "facebook/dinov3-convnext-large-pretrain-lvd1689m",
    "facebook/dinov3-vith16plus-pretrain-lvd1689m",
]


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark DINOv3 image encoder inference latency and throughput.")
    parser.add_argument("--models", nargs="*", default=LVD_DINOV3_MODELS, help="Model ids to benchmark. Defaults to all LVD DINOv3 encoders.")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--benchmark-iters", type=int, default=20)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default="artifacts/dinov3_inference_benchmark/results.json")
    parser.add_argument("--log-level", default="INFO")
    return parser


def resolve_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def format_table(rows: list[dict]) -> str:
    headers = [
        "model",
        "status",
        "params_m",
        "hidden_size",
        "seq_len",
        "latency_ms",
        "throughput_img_s",
        "peak_mem_mb",
    ]
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row.get(header, ""))))

    def render_line(values: dict) -> str:
        return " | ".join(str(values.get(header, "")).ljust(widths[header]) for header in headers)

    separator = "-+-".join("-" * widths[header] for header in headers)
    lines = [render_line({header: header for header in headers}), separator]
    for row in rows:
        lines.append(render_line(row))
    return "\n".join(lines)


def benchmark_model(
    model_name: str,
    image_size: int,
    batch_size: int,
    warmup_iters: int,
    benchmark_iters: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    result = {
        "model": model_name,
        "status": "failed",
        "params_m": "",
        "hidden_size": "",
        "seq_len": "",
        "latency_ms": "",
        "throughput_img_s": "",
        "peak_mem_mb": "",
    }
    model = None
    try:
        LOGGER.info("[%s] Loading config", model_name)
        config = AutoConfig.from_pretrained(model_name)
        hidden_size = getattr(config, "hidden_size", "")
        LOGGER.info("[%s] Config loaded: hidden_size=%s image_size=%s", model_name, hidden_size, image_size)
        LOGGER.info("[%s] Loading model weights with AutoModel.from_pretrained", model_name)
        model = AutoModel.from_pretrained(model_name)
        LOGGER.info("[%s] Model weights loaded", model_name)
        model.eval()
        model.to(device)
        LOGGER.info("[%s] Model moved to device=%s", model_name, device)

        params_m = count_parameters(model) / 1_000_000.0
        LOGGER.info("[%s] Parameter count: %.1fM", model_name, params_m)
        input_tensor = torch.randn((batch_size, 3, image_size, image_size), device=device, dtype=torch.float32)
        LOGGER.info("[%s] Direct pixel_values prepared: shape=%s dtype=%s", model_name, tuple(input_tensor.shape), input_tensor.dtype)

        use_autocast = device.type == "cuda" and dtype != torch.float32
        amp_dtype = dtype if use_autocast else None
        LOGGER.info("[%s] Autocast enabled=%s dtype=%s", model_name, use_autocast, amp_dtype)

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            LOGGER.info("[%s] CUDA peak memory stats reset", model_name)

        with torch.inference_mode():
            for warmup_idx in range(warmup_iters):
                LOGGER.info("[%s] Warmup %s/%s", model_name, warmup_idx + 1, warmup_iters)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast):
                    outputs = model(pixel_values=input_tensor)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)

            latencies = []
            output_seq_len = ""
            LOGGER.info("[%s] Starting timed benchmark iterations", model_name)
            for bench_idx in range(benchmark_iters):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast):
                    outputs = model(pixel_values=input_tensor)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start
                latencies.append(elapsed)
                if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                    output_seq_len = int(outputs.last_hidden_state.shape[1])
                LOGGER.info(
                    "[%s] Timed iter %s/%s: %.4f s%s",
                    model_name,
                    bench_idx + 1,
                    benchmark_iters,
                    elapsed,
                    f" seq_len={output_seq_len}" if output_seq_len else "",
                )

        mean_latency_ms = statistics.mean(latencies) * 1000.0
        throughput = batch_size / statistics.mean(latencies)
        peak_mem_mb = ""
        if device.type == "cuda":
            peak_mem_mb = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)
        LOGGER.info(
            "[%s] Completed: mean_latency_ms=%.2f throughput_img_s=%.2f peak_mem_mb=%s",
            model_name,
            mean_latency_ms,
            throughput,
            peak_mem_mb,
        )

        result.update(
            {
                "status": "ok",
                "params_m": f"{params_m:.1f}",
                "hidden_size": str(hidden_size),
                "seq_len": str(output_seq_len),
                "latency_ms": f"{mean_latency_ms:.2f}",
                "throughput_img_s": f"{throughput:.2f}",
                "peak_mem_mb": str(peak_mem_mb),
                "mean_latency_seconds": statistics.mean(latencies),
                "median_latency_seconds": statistics.median(latencies),
                "stdev_latency_seconds": statistics.pstdev(latencies) if len(latencies) > 1 else 0.0,
                "min_latency_seconds": min(latencies),
                "max_latency_seconds": max(latencies),
                "batch_size": batch_size,
                "image_size": image_size,
                "dtype": str(dtype).replace("torch.", ""),
                "device": str(device),
            }
        )
    except torch.cuda.OutOfMemoryError as exc:
        LOGGER.exception("[%s] CUDA OOM during benchmark", model_name)
        result["status"] = "oom"
        result["error"] = str(exc)
    except Exception as exc:
        LOGGER.exception("[%s] Benchmark failed", model_name)
        result["status"] = "error"
        result["error"] = str(exc)
    finally:
        LOGGER.info("[%s] Cleaning up model and CUDA cache", model_name)
        if model is not None:
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return result


def write_payload(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable.")
    dtype = resolve_dtype(args.dtype)
    if device.type == "cpu" and dtype != torch.float32:
        LOGGER.warning("CPU benchmark requested with %s; falling back to float32.", args.dtype)
        dtype = torch.float32

    output_path = Path(args.output_json)
    payload = {
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "warmup_iters": args.warmup_iters,
        "benchmark_iters": args.benchmark_iters,
        "models": args.models,
        "results": [],
    }
    write_payload(output_path, payload)
    LOGGER.info("Initialized benchmark output at %s", output_path)

    results = []
    for model_name in args.models:
        LOGGER.info("Benchmarking %s", model_name)
        result = (
            benchmark_model(
                model_name=model_name,
                image_size=args.image_size,
                batch_size=args.batch_size,
                warmup_iters=args.warmup_iters,
                benchmark_iters=args.benchmark_iters,
                device=device,
                dtype=dtype,
            )
        )
        results.append(result)
        payload["results"] = results
        write_payload(output_path, payload)
        LOGGER.info("Saved partial benchmark results after %s", model_name)

    table_rows = []
    for item in results:
        table_rows.append(
            {
                "model": item["model"],
                "status": item["status"],
                "params_m": item.get("params_m", ""),
                "hidden_size": item.get("hidden_size", ""),
                "seq_len": item.get("seq_len", ""),
                "latency_ms": item.get("latency_ms", ""),
                "throughput_img_s": item.get("throughput_img_s", ""),
                "peak_mem_mb": item.get("peak_mem_mb", ""),
            }
        )
    print(format_table(table_rows))
    LOGGER.info("Saved benchmark results to %s", output_path)


if __name__ == "__main__":
    main()
