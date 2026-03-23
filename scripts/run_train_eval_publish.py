import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

from lana_radgen.logging_utils import configure_logging

LOGGER = logging.getLogger("run_train_eval_publish")

DEFAULT_TRAIN_CANDIDATES = [
    {"method": "lora_adamw", "batch_size": 1, "global_batch_size": 8},
    {"method": "lora_adamw", "batch_size": 2, "global_batch_size": 8},
    {"method": "lora_adamw", "batch_size": 2, "global_batch_size": 4},
    {"method": "full_adam8bit", "batch_size": 1, "global_batch_size": 8},
    {"method": "full_adamw", "batch_size": 1, "global_batch_size": 8},
    {"method": "qlora_paged_adamw8bit", "batch_size": 1, "global_batch_size": 8},
]

DEFAULT_EVAL_BATCH_CANDIDATES = [1, 2, 4, 8]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Auto-tune train/eval config once, then run timed training, evaluate, push once to HF after evaluation, sync the local README, and optionally commit/push GitHub."
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--run-name", default="full_3_epoch_mask_run")
    parser.add_argument("--dataset", default="combined", choices=["chexpert", "mimic", "combined"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--duration", required=True, help="Examples: 30m, 40m, 2h, 01:30:00")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--eval-generation-batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="artifacts/full_3_epoch_mask_run")
    parser.add_argument("--method", default="lora_adamw")
    parser.add_argument("--repo-id", default="manu02/LAnA")
    parser.add_argument("--git-remote", default="origin")
    parser.add_argument("--git-branch", default="main")
    parser.add_argument("--git-commit-message", default="")
    parser.add_argument("--skip-github-push", action="store_true")
    parser.add_argument("--skip-git-commit", action="store_true")
    parser.add_argument("--rebenchmark", action="store_true")
    parser.add_argument("--benchmark-train-steps", type=int, default=16)
    parser.add_argument("--benchmark-eval-limit", type=int, default=64)
    parser.add_argument("--log-level", default="INFO")
    return parser


def run_command(command: list[str], cwd: Path) -> None:
    LOGGER.info("Running command: %s", " ".join(command))
    subprocess.run(command, cwd=str(cwd), check=True)


def run_command_capture(command: list[str], cwd: Path) -> subprocess.CompletedProcess:
    LOGGER.info("Running command: %s", " ".join(command))
    return subprocess.run(command, cwd=str(cwd), check=True, capture_output=True, text=True)


def sync_root_readme(repo_root: Path, output_dir: Path) -> None:
    source = output_dir / "README.md"
    target = repo_root / "README.md"
    if not source.exists():
        raise FileNotFoundError(f"Expected run README not found: {source}")
    shutil.copyfile(source, target)
    LOGGER.info("Synced root README from %s", source)


def commit_and_push(repo_root: Path, args) -> None:
    commit_message = args.git_commit_message.strip() or f"Update training/evaluation after {args.duration} run"
    run_command(["git", "add", "README.md"], repo_root)
    status = subprocess.run(
        ["git", "status", "--short", "--", "README.md"],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    if not status.stdout.strip():
        LOGGER.info("No README changes to commit.")
        return
    if not args.skip_git_commit:
        run_command(["git", "commit", "-m", commit_message], repo_root)
    if not args.skip_github_push:
        run_command(["git", "push", args.git_remote, args.git_branch], repo_root)


def autotune_path(output_dir: Path) -> Path:
    return output_dir / "pipeline_autotune.json"


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def benchmark_training(repo_root: Path, args, output_dir: Path) -> dict:
    benchmark_root = output_dir / "_autotune" / "train"
    benchmark_root.mkdir(parents=True, exist_ok=True)
    results = []
    for idx, candidate in enumerate(DEFAULT_TRAIN_CANDIDATES):
        candidate_dir = benchmark_root / f"candidate_{idx}_{candidate['method']}_b{candidate['batch_size']}_g{candidate['global_batch_size']}"
        command = [
            args.python_executable,
            "scripts/train.py",
            "--run-name",
            f"autotune_train_{idx}",
            "--dataset",
            args.dataset,
            "--epochs",
            "1",
            "--batch-size",
            str(candidate["batch_size"]),
            "--global-batch-size",
            str(candidate["global_batch_size"]),
            "--eval-batch-size",
            str(args.eval_batch_size),
            "--image-size",
            str(args.image_size),
            "--device",
            args.device,
            "--output-dir",
            str(candidate_dir),
            "--method",
            candidate["method"],
            "--max-train-steps",
            str(args.benchmark_train_steps),
            "--save-every-n-steps",
            "1000",
            "--log-level",
            args.log_level,
            "--disable-wandb",
        ]
        started = time.perf_counter()
        try:
            run_command(command, repo_root)
            elapsed = time.perf_counter() - started
            summary = load_json(candidate_dir / "run_summary.json")
            result = {
                **candidate,
                "candidate_dir": str(candidate_dir),
                "status": "ok",
                "elapsed_seconds": elapsed,
                "images_per_second": float(summary.get("images_per_second", 0.0)),
                "steps": int(summary.get("steps", 0)),
                "train_loss_last": float(summary.get("train_loss_last", 0.0)),
            }
        except Exception as exc:
            result = {**candidate, "status": "failed", "error": str(exc)}
        results.append(result)

    successful = [item for item in results if item["status"] == "ok"]
    if not successful:
        raise RuntimeError("All training autotune candidates failed.")
    best = max(successful, key=lambda item: item["images_per_second"])
    return {"results": results, "best": best}


def benchmark_evaluation(repo_root: Path, args, eval_run_dir: Path) -> dict:
    results = []
    for batch_size in DEFAULT_EVAL_BATCH_CANDIDATES:
        command = [
            args.python_executable,
            "scripts/evaluate.py",
            "--run-dir",
            str(eval_run_dir),
            "--batch-size",
            str(batch_size),
            "--device",
            args.device,
            "--repo-id",
            args.repo_id,
            "--limit",
            str(args.benchmark_eval_limit),
            "--log-level",
            args.log_level,
        ]
        started = time.perf_counter()
        try:
            run_command(command, repo_root)
            elapsed = time.perf_counter() - started
            result = {
                "batch_size": batch_size,
                "status": "ok",
                "elapsed_seconds": elapsed,
                "examples_per_second": args.benchmark_eval_limit / elapsed if elapsed > 0 else 0.0,
            }
        except Exception as exc:
            result = {"batch_size": batch_size, "status": "failed", "error": str(exc)}
        results.append(result)

    successful = [item for item in results if item["status"] == "ok"]
    if not successful:
        raise RuntimeError("All evaluation autotune candidates failed.")
    best = max(successful, key=lambda item: item["examples_per_second"])
    return {"results": results, "best": best}


def resolve_autotuned_config(repo_root: Path, args, output_dir: Path) -> dict:
    config_path = autotune_path(output_dir)
    if config_path.exists() and not args.rebenchmark:
        payload = load_json(config_path)
        LOGGER.info("Loaded saved autotune config from %s", config_path)
        return payload

    LOGGER.info("Autotune config not found or rebenchmark requested. Running train/eval benchmarks.")
    train_benchmark = benchmark_training(repo_root, args, output_dir)
    best_train = train_benchmark["best"]
    best_train_dir = Path(best_train["candidate_dir"])
    original_method = args.method
    original_batch_size = args.batch_size
    original_global_batch_size = args.global_batch_size

    args.method = best_train["method"]
    args.batch_size = int(best_train["batch_size"])
    args.global_batch_size = int(best_train["global_batch_size"])
    eval_benchmark = benchmark_evaluation(repo_root, args, best_train_dir)
    best_eval = eval_benchmark["best"]

    args.method = original_method
    args.batch_size = original_batch_size
    args.global_batch_size = original_global_batch_size

    payload = {
        "train": best_train,
        "eval": best_eval,
        "benchmarks": {
            "train": train_benchmark["results"],
            "eval": eval_benchmark["results"],
        },
    }
    save_json(config_path, payload)
    LOGGER.info("Saved autotune config to %s", config_path)
    return payload


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level)

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    autotuned = resolve_autotuned_config(repo_root, args, output_dir)
    selected_method = str(autotuned["train"]["method"])
    selected_batch_size = int(autotuned["train"]["batch_size"])
    selected_global_batch_size = int(autotuned["train"]["global_batch_size"])
    selected_eval_batch_size = int(autotuned["eval"]["batch_size"])

    LOGGER.info(
        "Using autotuned config: method=%s train_batch=%s global_batch=%s eval_batch=%s",
        selected_method,
        selected_batch_size,
        selected_global_batch_size,
        selected_eval_batch_size,
    )

    train_cmd = [
        args.python_executable,
        "scripts/train.py",
        "--run-name",
        args.run_name,
        "--dataset",
        args.dataset,
        "--epochs",
        str(args.epochs),
        "--duration",
        args.duration,
        "--batch-size",
        str(selected_batch_size),
        "--global-batch-size",
        str(selected_global_batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--image-size",
        str(args.image_size),
        "--device",
        args.device,
        "--output-dir",
        str(output_dir),
        "--method",
        selected_method,
        "--repo-id",
        args.repo_id,
        "--log-level",
        args.log_level,
        "--disable-wandb",
    ]
    eval_cmd = [
        args.python_executable,
        "scripts/evaluate.py",
        "--run-dir",
        str(output_dir),
        "--batch-size",
        str(selected_eval_batch_size),
        "--device",
        args.device,
        "--repo-id",
        args.repo_id,
        "--push-to-hub",
        "--log-level",
        args.log_level,
    ]

    run_command(train_cmd, repo_root)
    run_command(eval_cmd, repo_root)
    sync_root_readme(repo_root, output_dir)
    commit_and_push(repo_root, args)


if __name__ == "__main__":
    main()
