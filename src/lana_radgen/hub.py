import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi


PACKAGE_FILES = [
    "__init__.py",
    "configuration_lana.py",
    "gpt2_modified.py",
    "modeling_lana.py",
    "modeling_outputs.py",
    "segmenters.py",
]

PACKAGE_DIRS = [
    "attention",
]


def _write_remote_code_wrappers(package_dir: Path) -> None:
    (package_dir / "configuration_lana.py").write_text(
        "from lana_radgen.configuration_lana import LanaConfig\n\n__all__ = [\"LanaConfig\"]\n",
        encoding="utf-8",
    )
    (package_dir / "modeling_lana.py").write_text(
        "from lana_radgen.modeling_lana import LanaForConditionalGeneration\n\n__all__ = [\"LanaForConditionalGeneration\"]\n",
        encoding="utf-8",
    )


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))


def _patch_config_for_hf(config_path: Path) -> None:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["auto_map"] = {
        "AutoConfig": "configuration_lana.LanaConfig",
        "AutoModel": "modeling_lana.LanaForConditionalGeneration",
    }
    payload["architectures"] = ["LanaForConditionalGeneration"]
    payload["lung_segmenter_checkpoint"] = "segmenters/lung_segmenter_dinounet_finetuned.pth"
    payload["heart_segmenter_checkpoint"] = "segmenters/heart_segmenter_dinounet_best.pth"
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_hf_package(local_dir: str) -> Path:
    source_dir = Path(local_dir)
    package_dir = source_dir / ".hf_publish"
    if package_dir.exists():
        shutil.rmtree(package_dir, ignore_errors=True)
    package_dir.mkdir(parents=True, exist_ok=True)

    readme_path = source_dir / "README.md"
    if readme_path.exists():
        shutil.copy2(readme_path, package_dir / "README.md")

    for optional_file in ["run_summary.json", "benchmark_results.json", "pipeline_autotune.json"]:
        src = source_dir / optional_file
        if src.exists():
            shutil.copy2(src, package_dir / optional_file)

    for optional_dir in ["assets", "evaluations", "segmenters"]:
        src = source_dir / optional_dir
        if src.exists():
            _copy_tree(src, package_dir / optional_dir)

    model_dir = source_dir / "model"
    tokenizer_dir = source_dir / "tokenizer"
    if model_dir.exists():
        shutil.copy2(model_dir / "config.json", package_dir / "config.json")
        shutil.copy2(model_dir / "model.safetensors", package_dir / "model.safetensors")
        _patch_config_for_hf(package_dir / "config.json")
    if tokenizer_dir.exists():
        for tokenizer_file in tokenizer_dir.iterdir():
            if tokenizer_file.is_file():
                shutil.copy2(tokenizer_file, package_dir / tokenizer_file.name)

    local_package_dir = package_dir / "lana_radgen"
    local_package_dir.mkdir(parents=True, exist_ok=True)
    source_package_root = Path(__file__).resolve().parent
    for filename in PACKAGE_FILES:
        shutil.copy2(source_package_root / filename, local_package_dir / filename)
    for dirname in PACKAGE_DIRS:
        _copy_tree(source_package_root / dirname, local_package_dir / dirname)
    shutil.copy2(source_package_root / "__init__.py", local_package_dir / "__init__.py")
    _write_remote_code_wrappers(package_dir)
    return package_dir


def push_directory_to_hub(local_dir: str, repo_id: str, commit_message: str = "Upload model") -> str:
    api = HfApi()
    folder = _build_hf_package(local_dir)
    api.create_repo(repo_id=repo_id, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(folder),
        commit_message=commit_message,
        ignore_patterns=["checkpoints", "checkpoints/**", "_autotune", "_autotune/**", ".hf_publish", ".hf_publish/**"],
        delete_patterns=[
            "checkpoints",
            "checkpoints/**",
            "_autotune",
            "_autotune/**",
            "model",
            "model/**",
            "tokenizer",
            "tokenizer/**",
        ],
    )
    return f"https://huggingface.co/{repo_id}"
