import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from lana_radgen.logging_utils import configure_logging

LOGGER = logging.getLogger("run_train_eval_publish")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a timed training block, evaluate, push once to HF after evaluation, sync the local README, and optionally commit/push GitHub."
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
    parser.add_argument("--log-level", default="INFO")
    return parser


def run_command(command: list[str], cwd: Path) -> None:
    LOGGER.info("Running command: %s", " ".join(command))
    subprocess.run(command, cwd=str(cwd), check=True)


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


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level)

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = (repo_root / args.output_dir).resolve()

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
        str(args.batch_size),
        "--global-batch-size",
        str(args.global_batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--image-size",
        str(args.image_size),
        "--device",
        args.device,
        "--output-dir",
        str(output_dir),
        "--method",
        args.method,
        "--repo-id",
        args.repo_id,
        "--log-level",
        args.log_level,
    ]
    eval_cmd = [
        args.python_executable,
        "scripts/evaluate.py",
        "--run-dir",
        str(output_dir),
        "--batch-size",
        str(args.eval_generation_batch_size),
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
