import argparse
import json
import re
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

from lana_radgen.hub import push_split_inference_and_snapshot_layout
from lana_radgen.model_card import (
    DINO_V3_NOTICE,
    RESEARCH_USE_NOTICE,
    build_main_branch_usage_section,
    build_snapshot_branch_usage_section,
    upsert_best_model_notice,
)


COLLECTION_MODELS = [
    {
        "artifact_dir": "full_3_epoch_mask_run",
        "repo_id": "manu02/LAnA-MIMIC-CHEXPERT",
        "local_runtime": False,
    },
    {
        "artifact_dir": "LAnA-MIMIC-TERM",
        "repo_id": "manu02/LAnA-MIMIC",
        "local_runtime": False,
    },
    {
        "artifact_dir": "LAnA-paper",
        "repo_id": "manu02/LAnA",
        "local_runtime": False,
    },
    {
        "artifact_dir": "LAnA-Arxiv",
        "repo_id": "manu02/LAnA-Arxiv",
        "local_runtime": True,
    },
    {
        "artifact_dir": "LAnA-v2",
        "repo_id": "manu02/LAnA-v2",
        "local_runtime": False,
    },
    {
        "artifact_dir": "LAnA-v3",
        "repo_id": "manu02/LAnA-v3",
        "local_runtime": True,
    },
    {
        "artifact_dir": "LAnA-v4",
        "repo_id": "manu02/LAnA-v4",
        "local_runtime": True,
    },
    {
        "artifact_dir": "LAnA-v5",
        "repo_id": "manu02/LAnA-v5",
        "local_runtime": True,
    },
]


def _best_collection_repo_id(repo_root: Path) -> str | None:
    best_repo_id = None
    best_score = float("-inf")
    best_rouge = float("-inf")
    for entry in COLLECTION_MODELS:
        summary_path = repo_root / "artifacts" / entry["artifact_dir"] / "run_summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        metrics = summary.get("latest_evaluations", {}).get("all_test") or summary.get("latest_evaluation") or {}
        score = metrics.get("chexpert_f1_14_micro")
        rouge = metrics.get("rouge_l")
        score_value = float(score) if score is not None else float("-inf")
        rouge_value = float(rouge) if rouge is not None else float("-inf")
        if (score_value, rouge_value) > (best_score, best_rouge):
            best_score, best_rouge = score_value, rouge_value
            best_repo_id = entry["repo_id"]
    return best_repo_id


def _replace_section(text: str, heading: str, replacement: str, next_heading_pattern: str = r"\n## ") -> str:
    pattern = rf"{re.escape(heading)}\n.*?(?={next_heading_pattern}|\Z)"
    if re.search(pattern, text, flags=re.DOTALL):
        return re.sub(pattern, replacement.rstrip() + "\n\n", text, flags=re.DOTALL)
    return text.rstrip() + "\n\n" + replacement.rstrip() + "\n"


def _ensure_main_branch_model_card(readme_path: Path, repo_id: str, best_repo_id: str | None = None) -> None:
    current = readme_path.read_text(encoding="utf-8") if readme_path.exists() else f"# {repo_id}\n"
    updated = _replace_section(current, "## How to Run", build_main_branch_usage_section(repo_id))
    updated = upsert_best_model_notice(updated, best_repo_id or repo_id)
    if DINO_V3_NOTICE not in updated and "## Intended Use" in updated:
        updated = updated.replace("## Intended Use", f"{DINO_V3_NOTICE}\n\n{RESEARCH_USE_NOTICE}\n\n## Intended Use", 1)
    elif DINO_V3_NOTICE not in updated:
        updated = updated.rstrip() + f"\n\n{DINO_V3_NOTICE}\n\n{RESEARCH_USE_NOTICE}\n"
    readme_path.write_text(updated, encoding="utf-8")


def _ensure_snapshot_legacy_model_card(readme_path: Path, repo_id: str, best_repo_id: str | None = None) -> None:
    current = readme_path.read_text(encoding="utf-8") if readme_path.exists() else f"# {repo_id}\n"
    updated = _replace_section(current, "## How to Run", build_snapshot_branch_usage_section(repo_id))
    updated = upsert_best_model_notice(updated, best_repo_id or repo_id)
    if DINO_V3_NOTICE not in updated and "## Intended Use" in updated:
        updated = updated.replace("## Intended Use", f"{DINO_V3_NOTICE}\n\n{RESEARCH_USE_NOTICE}\n\n## Intended Use", 1)
    elif DINO_V3_NOTICE not in updated:
        updated = updated.rstrip() + f"\n\n{DINO_V3_NOTICE}\n\n{RESEARCH_USE_NOTICE}\n"
    readme_path.write_text(updated, encoding="utf-8")


def _ensure_license_notice(target_dir: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root / "licenses" / "DINOv3-LICENSE.txt"
    if source.exists():
        shutil.copyfile(source, target_dir / "DINOv3-LICENSE.txt")


def _stage_remote_snapshot(repo_id: str, staging_root: Path) -> Path:
    snapshot_dir = Path(snapshot_download(repo_id))
    target_dir = staging_root / repo_id.replace("/", "__")
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(snapshot_dir, target_dir)
    return target_dir
def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the full LAnA collection with dual-path model cards and republish local-compatible artifacts.")
    parser.add_argument("--staging-root", default=".tmp/hf_collection")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--commit-message", default="Republish dual-compatible LAnA checkpoint")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    staging_root = (repo_root / args.staging_root).resolve()
    staging_root.mkdir(parents=True, exist_ok=True)
    best_repo_id = _best_collection_repo_id(repo_root)

    for entry in COLLECTION_MODELS:
        repo_id = entry["repo_id"]
        if entry["local_runtime"]:
            local_dir = repo_root / "artifacts" / entry["artifact_dir"]
            _ensure_main_branch_model_card(local_dir / "README.md", repo_id, best_repo_id=best_repo_id)
            _ensure_snapshot_legacy_model_card(local_dir / "README.snapshot-legacy.md", repo_id, best_repo_id=best_repo_id)
            _ensure_license_notice(local_dir)
            if args.push:
                push_split_inference_and_snapshot_layout(str(local_dir), repo_id, args.commit_message)
            continue

        staged_dir = _stage_remote_snapshot(repo_id, staging_root)
        _ensure_main_branch_model_card(staged_dir / "README.md", repo_id, best_repo_id=best_repo_id)
        _ensure_snapshot_legacy_model_card(staged_dir / "README.snapshot-legacy.md", repo_id, best_repo_id=best_repo_id)
        _ensure_license_notice(staged_dir)
        if args.push:
            push_split_inference_and_snapshot_layout(str(staged_dir), repo_id, args.commit_message)


if __name__ == "__main__":
    main()
