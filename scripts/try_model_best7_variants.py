from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path


LOGGER = logging.getLogger("try_model_best7_variants")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a matrix of legacy model_best7 generation/evaluation variants and save resumable status."
    )
    parser.add_argument("--python-executable", default=str((_repo_root() / "venv310" / "Scripts" / "python.exe").resolve()))
    parser.add_argument("--model-path", default="deletions/models/model_best7.pth")
    parser.add_argument("--loader-path", default="deletions/utils/models/complete_model.py")
    parser.add_argument("--mimic-root", default="Datasets/MIMIC")
    parser.add_argument("--run-root", default="artifacts/model_best7_variants")
    parser.add_argument("--variants", default="", help="Optional comma-separated subset of preset names to run.")
    parser.add_argument("--force", action="store_true", help="Re-run variants even if they were previously marked completed.")
    parser.add_argument("--log-level", default="INFO")
    return parser


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {"variants": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"variants": {}}


def _save_state(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _variant_matrix(args: argparse.Namespace) -> list[dict]:
    python = str(Path(args.python_executable).resolve())
    model_path = str((_repo_root() / args.model_path).resolve())
    loader_path = str((_repo_root() / args.loader_path).resolve())
    mimic_root = str((_repo_root() / args.mimic_root).resolve())
    run_root = Path(args.run_root)

    return [
        {
            "name": "paper_notebook_like_150",
            "description": "Notebook-style legacy generation/eval with current reliable dataloading and legacy findings extraction.",
            "cmd": [
                python,
                "scripts/run_deletions_cloud_eval.py",
                "--run-dir",
                str(run_root / "paper_notebook_like_150"),
                "--model-path",
                model_path,
                "--max-new-tokens",
                "150",
                "--log-level",
                args.log_level,
            ],
        },
        {
            "name": "exact_legacy_all_test_150",
            "description": "Chunked exact-legacy generation/eval on all frontal test studies at 150 tokens.",
            "cmd": [
                python,
                "scripts/reproduce_cloud_best_model7.py",
                "--model-path",
                model_path,
                "--deletions-loader-path",
                loader_path,
                "--mimic-root",
                mimic_root,
                "--run-dir",
                str(run_root / "exact_legacy_all_test_150"),
                "--chunk-size",
                "50",
                "--image-size",
                "512",
                "--max-new-tokens",
                "150",
                "--log-level",
                args.log_level,
            ],
        },
        {
            "name": "exact_legacy_findings_sweep_100_128_150",
            "description": "Chunked exact-legacy findings-only sweep across 100, 128, and 150 tokens.",
            "cmd": [
                python,
                "scripts/reproduce_cloud_best_model7.py",
                "--model-path",
                model_path,
                "--deletions-loader-path",
                loader_path,
                "--mimic-root",
                mimic_root,
                "--run-dir",
                str(run_root / "exact_legacy_findings_sweep_100_128_150"),
                "--chunk-size",
                "50",
                "--image-size",
                "512",
                "--findings-only-only",
                "--max-new-tokens-list",
                "100,128,150",
                "--log-level",
                args.log_level,
            ],
        },
    ]


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level)

    repo_root = _repo_root()
    run_root = (repo_root / args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    state_path = run_root / "variant_state.json"
    state = _load_state(state_path)
    state.setdefault("variants", {})

    requested = {name.strip() for name in args.variants.split(",") if name.strip()}
    variants = _variant_matrix(args)
    if requested:
        variants = [variant for variant in variants if variant["name"] in requested]
        missing = requested - {variant["name"] for variant in variants}
        if missing:
            raise ValueError(f"Unknown variants requested: {sorted(missing)}")

    for variant in variants:
        name = variant["name"]
        current = state["variants"].get(name, {})
        if current.get("status") == "completed" and not args.force:
            LOGGER.info("Skipping completed variant %s", name)
            continue

        LOGGER.info("Running variant %s", name)
        LOGGER.info("Description: %s", variant["description"])
        try:
            subprocess.run(variant["cmd"], cwd=str(repo_root), check=True)
            state["variants"][name] = {
                "status": "completed",
                "description": variant["description"],
                "command": variant["cmd"],
            }
        except subprocess.CalledProcessError as exc:
            LOGGER.exception("Variant %s failed", name)
            state["variants"][name] = {
                "status": "failed",
                "description": variant["description"],
                "command": variant["cmd"],
                "returncode": exc.returncode,
            }
        finally:
            _save_state(state_path, state)


if __name__ == "__main__":
    main()
