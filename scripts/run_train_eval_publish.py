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


def resolve_model_variant_args(args) -> dict[str, object]:
    if args.model_variant != "lana_v5":
        return {
            "attention_bias_mode": args.attention_bias_mode,
            "vision_prefix_tokens_to_skip": int(args.vision_prefix_tokens_to_skip),
            "generation_use_bos_token": bool(args.generation_use_bos_token),
            "generation_stop_on_eos": bool(args.generation_stop_on_eos),
            "generation_repetition_penalty": float(args.generation_repetition_penalty),
        }
    return {
        "attention_bias_mode": "gaussian_legacy",
        "vision_prefix_tokens_to_skip": 5,
        "generation_use_bos_token": False,
        "generation_stop_on_eos": True,
        "generation_repetition_penalty": 1.2,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Auto-tune train/eval config once, then run timed training, evaluate, push once to HF after evaluation, sync the local README, and optionally commit/push GitHub."
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--model-variant", default="default", choices=["default", "lana_v5"])
    parser.add_argument("--run-name", default="full_3_epoch_mask_run")
    parser.add_argument("--vision-model-name", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--text-model-name", default="gpt2")
    parser.add_argument("--dataset", default="combined", choices=["chexpert", "mimic", "combined"])
    parser.add_argument("--metadata-path", default="Datasets/CheXpert/df_chexpert_plus_240401_findings.csv")
    parser.add_argument("--image-root", default="Datasets/CheXpert/images")
    parser.add_argument("--mimic-root", default="Datasets/MIMIC")
    parser.add_argument("--mimic-findings-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--duration", required=True, help="Examples: 30m, 40m, 2h, 01:30:00")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--eval-generation-batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--visual-projection-type", default="mlp4", choices=["mlp4", "linear"])
    parser.add_argument("--attention-bias-mode", default="layerwise", choices=["layerwise", "gaussian_legacy"])
    parser.add_argument("--vision-prefix-tokens-to-skip", type=int, default=1)
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--skip-cached-resize", action="store_true")
    parser.add_argument("--cache-size-audit-path", default=".cache/image_size_audit.json")
    parser.add_argument("--segmentation-model-name", default="facebook/dinov3-convnext-small-pretrain-lvd1689m")
    parser.add_argument("--lung-segmenter-checkpoint", default="models/lung_segmenter_dinounet_finetuned.pth")
    parser.add_argument("--heart-segmenter-checkpoint", default="models/heart_segmenter_dinounet_best.pth")
    parser.add_argument("--generation-use-bos-token", dest="generation_use_bos_token", action="store_true")
    parser.add_argument("--no-generation-use-bos-token", dest="generation_use_bos_token", action="store_false")
    parser.set_defaults(generation_use_bos_token=True)
    parser.add_argument("--generation-stop-on-eos", action="store_true")
    parser.add_argument("--generation-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--enable-wandb", dest="enable_wandb", action="store_true")
    parser.add_argument("--disable-wandb", dest="enable_wandb", action="store_false")
    parser.set_defaults(enable_wandb=True)
    parser.add_argument("--wandb-project", default="lana-radgen")
    parser.add_argument("--output-dir", default="artifacts/full_3_epoch_mask_run")
    parser.add_argument("--method", default="lora_adamw")
    parser.add_argument("--repo-id", default="manu02/LAnA")
    parser.add_argument("--root-readme-name", default="README.md")
    parser.add_argument("--git-remote", default="origin")
    parser.add_argument("--git-branch", default="main")
    parser.add_argument("--git-commit-message", default="")
    parser.add_argument("--skip-github-push", action="store_true")
    parser.add_argument("--skip-git-commit", action="store_true")
    parser.add_argument("--skip-autotune", action="store_true")
    parser.add_argument("--rebenchmark", action="store_true")
    parser.add_argument("--benchmark-steps", type=int, default=1)
    parser.add_argument("--time-limit-minutes", type=int, default=0)
    parser.add_argument("--max-train-steps", type=int, default=1000000)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--save-every-n-steps", type=int, default=1000)
    parser.add_argument("--log-every-n-steps", type=int, default=100)
    parser.add_argument("--keep-last-n-checkpoints", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmark-train-steps", type=int, default=16)
    parser.add_argument("--benchmark-eval-limit", type=int, default=64)
    parser.add_argument("--log-level", default="INFO")
    return parser


def build_train_command(args, *, run_name: str, output_dir: Path, method: str, batch_size: int, global_batch_size: int, eval_batch_size: int, max_train_steps: int, push_to_hub: bool = False) -> list[str]:
    variant_args = resolve_model_variant_args(args)
    return [
        args.python_executable,
        "scripts/train.py",
        "--model-variant",
        args.model_variant,
        "--vision-model-name",
        args.vision_model_name,
        "--text-model-name",
        args.text_model_name,
        "--run-name",
        run_name,
        "--dataset",
        args.dataset,
        "--metadata-path",
        args.metadata_path,
        "--image-root",
        args.image_root,
        "--mimic-root",
        args.mimic_root,
        *(["--mimic-findings-only"] if args.mimic_findings_only else []),
        "--batch-size",
        str(batch_size),
        "--global-batch-size",
        str(global_batch_size),
        "--eval-batch-size",
        str(eval_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--image-size",
        str(args.image_size),
        "--visual-projection-type",
        args.visual_projection_type,
        "--attention-bias-mode",
        str(variant_args["attention_bias_mode"]),
        "--vision-prefix-tokens-to-skip",
        str(variant_args["vision_prefix_tokens_to_skip"]),
        "--precision",
        args.precision,
        *(["--skip-cached-resize"] if args.skip_cached_resize else []),
        "--cache-size-audit-path",
        args.cache_size_audit_path,
        "--segmentation-model-name",
        args.segmentation_model_name,
        "--lung-segmenter-checkpoint",
        args.lung_segmenter_checkpoint,
        "--heart-segmenter-checkpoint",
        args.heart_segmenter_checkpoint,
        *(["--generation-use-bos-token"] if variant_args["generation_use_bos_token"] else ["--no-generation-use-bos-token"]),
        *(["--generation-stop-on-eos"] if variant_args["generation_stop_on_eos"] else []),
        "--generation-repetition-penalty",
        str(variant_args["generation_repetition_penalty"]),
        "--device",
        args.device,
        "--benchmark-steps",
        str(args.benchmark_steps),
        "--time-limit-minutes",
        str(args.time_limit_minutes),
        "--duration",
        args.duration,
        "--max-train-steps",
        str(max_train_steps),
        "--epochs",
        str(args.epochs),
        "--method",
        method,
        "--output-dir",
        str(output_dir),
        "--repo-id",
        args.repo_id,
        *(["--push-to-hub"] if push_to_hub else []),
        "--save-every-n-steps",
        str(args.save_every_n_steps),
        "--log-every-n-steps",
        str(args.log_every_n_steps),
        "--keep-last-n-checkpoints",
        str(args.keep_last_n_checkpoints),
        "--seed",
        str(args.seed),
        "--log-level",
        args.log_level,
        *(["--wandb-project", args.wandb_project] if args.enable_wandb else ["--disable-wandb"]),
    ]


def build_eval_command(args, *, output_dir: Path, batch_size: int, limit: int = 0, push_to_hub: bool = False) -> list[str]:
    command = [
        args.python_executable,
        "scripts/evaluate.py",
        "--run-dir",
        str(output_dir),
        "--repo-id",
        args.repo_id,
        "--device",
        args.device,
        "--batch-size",
        str(batch_size),
        "--image-size",
        str(args.image_size),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--mimic-root",
        args.mimic_root,
        "--log-level",
        args.log_level,
    ]
    if limit > 0:
        command.extend(["--limit", str(limit)])
    if push_to_hub:
        command.append("--push-to-hub")
    return command


def run_command(command: list[str], cwd: Path) -> None:
    LOGGER.info("Running command: %s", " ".join(command))
    subprocess.run(command, cwd=str(cwd), check=True)


def run_command_capture(command: list[str], cwd: Path) -> subprocess.CompletedProcess:
    LOGGER.info("Running command: %s", " ".join(command))
    return subprocess.run(command, cwd=str(cwd), check=True, capture_output=True, text=True)


def sync_root_readme(repo_root: Path, output_dir: Path, target_name: str) -> None:
    source = output_dir / "README.md"
    target = repo_root / target_name
    if not source.exists():
        raise FileNotFoundError(f"Expected run README not found: {source}")
    shutil.copyfile(source, target)
    LOGGER.info("Synced root README from %s", source)


def commit_and_push(repo_root: Path, args) -> None:
    commit_message = args.git_commit_message.strip() or f"Update training/evaluation after {args.duration} run"
    run_command(["git", "add", args.root_readme_name], repo_root)
    status = subprocess.run(
        ["git", "status", "--short", "--", args.root_readme_name],
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


def autotune_root(output_dir: Path) -> Path:
    return output_dir / "_autotune"


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def strip_candidate_dir(payload: dict) -> dict:
    sanitized = dict(payload)
    sanitized.pop("candidate_dir", None)
    return sanitized


def cleanup_autotune_artifacts(output_dir: Path) -> None:
    root = autotune_root(output_dir)
    if not root.exists():
        return
    shutil.rmtree(root)
    LOGGER.info("Deleted autotune artifacts from %s after saving metrics.", root)


def benchmark_training(repo_root: Path, args, output_dir: Path) -> dict:
    benchmark_root = autotune_root(output_dir) / "train"
    benchmark_root.mkdir(parents=True, exist_ok=True)
    results = []
    for idx, candidate in enumerate(DEFAULT_TRAIN_CANDIDATES):
        candidate_dir = benchmark_root / f"candidate_{idx}_{candidate['method']}_b{candidate['batch_size']}_g{candidate['global_batch_size']}"
        command = build_train_command(
            args,
            run_name=f"autotune_train_{idx}",
            output_dir=candidate_dir,
            method=candidate["method"],
            batch_size=int(candidate["batch_size"]),
            global_batch_size=int(candidate["global_batch_size"]),
            eval_batch_size=int(args.eval_batch_size),
            max_train_steps=int(args.benchmark_train_steps),
            push_to_hub=False,
        )
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
        command = build_eval_command(args, output_dir=eval_run_dir, batch_size=batch_size, limit=int(args.benchmark_eval_limit), push_to_hub=False)
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
    if args.skip_autotune:
        return {
            "train": {
                "method": args.method,
                "batch_size": int(args.batch_size),
                "global_batch_size": int(args.global_batch_size),
            },
            "eval": {
                "batch_size": int(args.eval_batch_size),
            },
            "benchmarks": {
                "train": [],
                "eval": [],
            },
        }
    config_path = autotune_path(output_dir)
    if config_path.exists() and not args.rebenchmark:
        payload = load_json(config_path)
        LOGGER.info("Loaded saved autotune config from %s", config_path)
        return payload

    LOGGER.info("Autotune config not found or rebenchmark requested. Running train/eval benchmarks.")
    original_method = args.method
    original_batch_size = args.batch_size
    original_global_batch_size = args.global_batch_size
    try:
        train_benchmark = benchmark_training(repo_root, args, output_dir)
        best_train = train_benchmark["best"]
        best_train_dir = Path(best_train["candidate_dir"])

        args.method = best_train["method"]
        args.batch_size = int(best_train["batch_size"])
        args.global_batch_size = int(best_train["global_batch_size"])
        eval_benchmark = benchmark_evaluation(repo_root, args, best_train_dir)
        best_eval = eval_benchmark["best"]
    finally:
        args.method = original_method
        args.batch_size = original_batch_size
        args.global_batch_size = original_global_batch_size

    payload = {
        "train": strip_candidate_dir(best_train),
        "eval": best_eval,
        "benchmarks": {
            "train": [strip_candidate_dir(item) for item in train_benchmark["results"]],
            "eval": eval_benchmark["results"],
        },
    }
    save_json(config_path, payload)
    LOGGER.info("Saved autotune config to %s", config_path)
    cleanup_autotune_artifacts(output_dir)
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

    train_cmd = build_train_command(
        args,
        run_name=args.run_name,
        output_dir=output_dir,
        method=selected_method,
        batch_size=selected_batch_size,
        global_batch_size=selected_global_batch_size,
        eval_batch_size=int(args.eval_batch_size),
        max_train_steps=int(args.max_train_steps),
        push_to_hub=bool(args.push_to_hub),
    )
    eval_cmd = build_eval_command(args, output_dir=output_dir, batch_size=selected_eval_batch_size, push_to_hub=bool(args.push_to_hub))

    run_command(train_cmd, repo_root)
    run_command(eval_cmd, repo_root)
    sync_root_readme(repo_root, output_dir, args.root_readme_name)
    commit_and_push(repo_root, args)


if __name__ == "__main__":
    main()
