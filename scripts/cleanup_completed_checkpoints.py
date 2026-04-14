import argparse
import json
import logging
import shutil
from pathlib import Path

from lana_radgen.logging_utils import configure_logging

LOGGER = logging.getLogger("cleanup_completed_checkpoints")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Delete checkpoint folders for completed artifact runs while keeping final exported model files."
    )
    parser.add_argument(
        "--artifacts-root",
        default="artifacts",
        help="Root directory that contains per-run artifact folders.",
    )
    parser.add_argument(
        "--run-dir",
        default="",
        help="Optional specific run directory to clean instead of scanning all artifacts/* runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be deleted without removing any files.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def latest_checkpoint_file(run_dir: Path) -> Path:
    return run_dir / "checkpoints" / "latest_checkpoint.json"


def load_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def iter_candidate_runs(artifacts_root: Path, run_dir_arg: str) -> list[Path]:
    if run_dir_arg:
        run_dir = Path(run_dir_arg)
        return [run_dir]
    if not artifacts_root.exists():
        raise FileNotFoundError(f"Artifacts root not found: {artifacts_root}")
    return sorted(path for path in artifacts_root.iterdir() if path.is_dir())


def directory_size_bytes(path: Path) -> int:
    return sum(
        file_path.stat().st_size
        for file_path in path.rglob("*")
        if file_path.is_file()
    )


def cleanup_run_checkpoints(run_dir: Path, dry_run: bool) -> dict:
    summary = load_summary(run_dir)
    checkpoints_dir = run_dir / "checkpoints"
    latest_file = latest_checkpoint_file(run_dir)

    result = {
        "run_dir": str(run_dir.resolve()),
        "completed": bool(summary.get("completed", False)),
        "checkpoints_exist": checkpoints_dir.exists(),
        "size_bytes": directory_size_bytes(checkpoints_dir) if checkpoints_dir.exists() else 0,
        "deleted": False,
        "skipped_reason": "",
    }

    if not summary:
        result["skipped_reason"] = "missing run_summary.json"
        return result
    if not result["completed"]:
        result["skipped_reason"] = "run not marked completed"
        return result
    if not checkpoints_dir.exists():
        result["skipped_reason"] = "no checkpoints directory"
        return result

    if dry_run:
        result["skipped_reason"] = "dry run"
        return result

    if latest_file.exists():
        latest_file.unlink(missing_ok=True)
    shutil.rmtree(checkpoints_dir, ignore_errors=True)
    result["deleted"] = True
    result["checkpoints_exist"] = False
    return result


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level)

    artifacts_root = Path(args.artifacts_root)
    run_dirs = iter_candidate_runs(artifacts_root=artifacts_root, run_dir_arg=args.run_dir)
    if not run_dirs:
        LOGGER.info("No run directories found.")
        return

    reclaimed_bytes = 0
    deleted_runs = 0
    for run_dir in run_dirs:
        result = cleanup_run_checkpoints(run_dir=run_dir, dry_run=args.dry_run)
        size_mb = result["size_bytes"] / (1024 * 1024)
        if result["deleted"]:
            reclaimed_bytes += result["size_bytes"]
            deleted_runs += 1
            LOGGER.info("Deleted checkpoints for %s (%.1f MB)", result["run_dir"], size_mb)
            continue
        if result["completed"] and result["checkpoints_exist"] and args.dry_run:
            LOGGER.info("Would delete checkpoints for %s (%.1f MB)", result["run_dir"], size_mb)
            continue
        LOGGER.info("Skipping %s: %s", result["run_dir"], result["skipped_reason"])

    reclaimed_mb = reclaimed_bytes / (1024 * 1024)
    if args.dry_run:
        LOGGER.info("Dry run complete.")
    else:
        LOGGER.info("Cleanup complete. Deleted checkpoints for %s run(s), reclaimed %.1f MB.", deleted_runs, reclaimed_mb)


if __name__ == "__main__":
    main()
