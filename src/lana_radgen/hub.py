import json
import logging
import os
import shutil
import time
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from safetensors import safe_open
from transformers import AutoTokenizer, GPT2Tokenizer

from .image_processing_lana import LanaImageProcessor
from .processing_lana import LanaProcessor

PACKAGE_FILES = [
    "__init__.py",
    "configuration_lana.py",
    "gpt2_modified.py",
    "image_processing_lana.py",
    "layerwise_anatomical_attention.py",
    "modeling_lana.py",
    "modeling_outputs.py",
    "processing_lana.py",
    "segmenters.py",
]

PACKAGE_DIRS = [
    "attention",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)


def _upload_folder_with_retry(api: HfApi, max_attempts: int = 3, retry_delay_seconds: float = 5.0, **kwargs):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return api.upload_folder(**kwargs)
        except HfHubHTTPError as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code is None or status_code < 500 or attempt >= max_attempts:
                raise
            last_error = exc
            LOGGER.warning(
                "Transient Hugging Face upload failure (status=%s) on attempt %s/%s for repo %s; retrying in %.1fs.",
                status_code,
                attempt,
                max_attempts,
                kwargs.get("repo_id", "<unknown>"),
                retry_delay_seconds,
            )
            time.sleep(retry_delay_seconds)
    if last_error is not None:
        raise last_error


def _write_root_remote_code_files(package_dir: Path, source_package_root: Path, include_legacy_dirs: bool = True) -> None:
    for filename in PACKAGE_FILES:
        (package_dir / filename).write_text((source_package_root / filename).read_text(encoding="utf-8"), encoding="utf-8")
    if include_legacy_dirs:
        for dirname in PACKAGE_DIRS:
            _copy_tree(source_package_root / dirname, package_dir / dirname)


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))


def _link_or_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _mirror_tree_with_links(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    for path in src.rglob("*"):
        if path.name == "__pycache__" or path.suffix in {".pyc", ".pyo"}:
            continue
        relative_path = path.relative_to(src)
        target = dst / relative_path
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            _link_or_copy_file(path, target)


def _infer_visual_projection_type(model_weights_path: Path) -> str | None:
    if not model_weights_path.exists():
        return None
    with safe_open(str(model_weights_path), framework="pt") as tensor_file:
        keys = set(tensor_file.keys())
    if "visual_projection.weight" in keys:
        return "linear"
    if "visual_projection.0.weight" in keys:
        return "mlp4"
    return None


def _patch_config_for_hf(config_path: Path, model_weights_path: Path | None = None) -> None:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["auto_map"] = {
        "AutoConfig": "configuration_lana.LanaConfig",
        "AutoModel": "modeling_lana.LanaForConditionalGeneration",
        "AutoProcessor": "processing_lana.LanaProcessor",
    }
    payload["architectures"] = ["LanaForConditionalGeneration"]
    payload["lung_segmenter_checkpoint"] = "segmenters/lung_segmenter_dinounet_finetuned.pth"
    payload["heart_segmenter_checkpoint"] = "segmenters/heart_segmenter_dinounet_best.pth"
    payload["bundled_vision_model_name"] = "bundled_backbones/vision_encoder"
    payload["bundled_segmentation_model_name"] = "bundled_backbones/segmenter_encoder"
    payload["bundled_text_model_name"] = "bundled_backbones/text_decoder"
    payload["bundled_tokenizer_name"] = "."
    payload["segmenter_weights_in_model_state"] = True
    projection_type = _infer_visual_projection_type(model_weights_path) if model_weights_path is not None else None
    if projection_type:
        payload["visual_projection_type"] = projection_type
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_exported_model_paths(source_dir: Path) -> tuple[Path, Path]:
    model_dir = source_dir / "model"
    if (model_dir / "config.json").exists() and (model_dir / "model.safetensors").exists():
        return model_dir / "config.json", model_dir / "model.safetensors"
    if (source_dir / "config.json").exists() and (source_dir / "model.safetensors").exists():
        return source_dir / "config.json", source_dir / "model.safetensors"
    raise FileNotFoundError(f"Expected exported model config and weights under {model_dir} or {source_dir}")


def _resolve_tokenizer_source_dir(source_dir: Path) -> Path | None:
    tokenizer_dir = source_dir / "tokenizer"
    if tokenizer_dir.exists():
        return tokenizer_dir
    tokenizer_files = ["tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt"]
    if any((source_dir / name).exists() for name in tokenizer_files):
        return source_dir
    return None


def _resolve_snapshot_dir(model_name: str) -> Path:
    sanitized = model_name.replace("/", "--")
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{sanitized}"
    snapshots_root = cache_root / "snapshots"
    if not snapshots_root.exists():
        raise FileNotFoundError(f"No local Hugging Face snapshot found for {model_name}")
    snapshots = [path for path in snapshots_root.iterdir() if path.is_dir()]
    if not snapshots:
        raise FileNotFoundError(f"No snapshot directory found for {model_name}")
    return max(snapshots, key=lambda path: path.stat().st_mtime)


def _bundle_required_backbones(package_dir: Path, config_payload: dict, *, include_weights: bool = True) -> None:
    bundled_root = package_dir / "bundled_backbones"
    bundled_root.mkdir(parents=True, exist_ok=True)
    backbone_sources = {
        "vision_encoder": config_payload["vision_model_name"],
        "segmenter_encoder": config_payload["segmentation_model_name"],
        "text_decoder": config_payload["text_model_name"],
    }
    for dirname, model_name in backbone_sources.items():
        snapshot_dir = _resolve_snapshot_dir(model_name)
        target_dir = bundled_root / dirname
        target_dir.mkdir(parents=True, exist_ok=True)
        _link_or_copy_file(snapshot_dir / "config.json", target_dir / "config.json")
        if include_weights:
            if (snapshot_dir / "model.safetensors").exists():
                _link_or_copy_file(snapshot_dir / "model.safetensors", target_dir / "model.safetensors")
            for optional_name in ["generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]:
                src = snapshot_dir / optional_name
                if src.exists():
                    _link_or_copy_file(src, target_dir / optional_name)


def _build_hf_package(
    local_dir: str,
    *,
    include_local_package: bool = True,
    include_legacy_dirs: bool = True,
    include_backbone_weights: bool = True,
    include_segmenter_checkpoints: bool = True,
    include_assets: bool = True,
    include_optional_artifacts: bool = True,
    readme_override_path: str | None = None,
) -> Path:
    source_dir = Path(local_dir)
    package_dir = source_dir / ".hf_publish"
    config_payload = None
    if package_dir.exists():
        shutil.rmtree(package_dir, ignore_errors=True)
    package_dir.mkdir(parents=True, exist_ok=True)

    readme_path = Path(readme_override_path) if readme_override_path else source_dir / "README.md"
    if readme_path.exists():
        _link_or_copy_file(readme_path, package_dir / "README.md")
    license_notice = REPO_ROOT / "licenses" / "DINOv3-LICENSE.txt"
    if license_notice.exists():
        _link_or_copy_file(license_notice, package_dir / "DINOv3-LICENSE.txt")

    if include_assets:
        assets_src = source_dir / "assets"
        if assets_src.exists():
            _mirror_tree_with_links(assets_src, package_dir / "assets")

    if include_optional_artifacts:
        for optional_file in ["run_summary.json", "benchmark_results.json", "pipeline_autotune.json"]:
            src = source_dir / optional_file
            if src.exists():
                _link_or_copy_file(src, package_dir / optional_file)

        evaluations_src = source_dir / "evaluations"
        if evaluations_src.exists():
            _mirror_tree_with_links(evaluations_src, package_dir / "evaluations")

    if include_segmenter_checkpoints:
        segmenters_src = source_dir / "segmenters"
        if segmenters_src.exists():
            _mirror_tree_with_links(segmenters_src, package_dir / "segmenters")

    config_src, model_weights_src = _resolve_exported_model_paths(source_dir)
    _link_or_copy_file(config_src, package_dir / "config.json")
    _link_or_copy_file(model_weights_src, package_dir / "model.safetensors")
    _patch_config_for_hf(package_dir / "config.json", model_weights_src)
    config_payload = json.loads((package_dir / "config.json").read_text(encoding="utf-8"))
    _bundle_required_backbones(package_dir, config_payload, include_weights=include_backbone_weights)

    tokenizer_dir = _resolve_tokenizer_source_dir(source_dir)
    tokenizer_source = _resolve_snapshot_dir(config_payload["text_model_name"])
    if tokenizer_dir is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True, use_fast=False)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token_id is None:
        if config_payload.get("vocab_size", 0) > len(tokenizer):
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        else:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
    tokenizer.save_pretrained(package_dir, legacy_format=True)
    for required_name in ["vocab.json", "merges.txt", "special_tokens_map.json"]:
        src = tokenizer_source / required_name
        if src.exists() and not (package_dir / required_name).exists():
            _link_or_copy_file(src, package_dir / required_name)
    image_processor = LanaImageProcessor(size={"height": config_payload["image_size"], "width": config_payload["image_size"]})
    processor = LanaProcessor(image_processor=image_processor, tokenizer=tokenizer)
    image_processor.save_pretrained(package_dir)
    processor.save_pretrained(package_dir)
    tokenizer.save_pretrained(package_dir, legacy_format=True)
    processor_config_path = package_dir / "processor_config.json"
    processor_payload = {
        "image_processor": image_processor.to_dict(),
        "processor_class": "LanaProcessor",
        "auto_map": {"AutoProcessor": "processing_lana.LanaProcessor"},
    }
    processor_config_path.write_text(json.dumps(processor_payload, indent=2), encoding="utf-8")
    preprocessor_config_path = package_dir / "preprocessor_config.json"
    if preprocessor_config_path.exists():
        preprocessor_payload = json.loads(preprocessor_config_path.read_text(encoding="utf-8"))
        preprocessor_payload["auto_map"] = {"AutoProcessor": "processing_lana.LanaProcessor"}
        preprocessor_payload["processor_class"] = "LanaProcessor"
        preprocessor_config_path.write_text(json.dumps(preprocessor_payload, indent=2), encoding="utf-8")

    source_package_root = Path(__file__).resolve().parent
    if include_local_package:
        local_package_dir = package_dir / "lana_radgen"
        local_package_dir.mkdir(parents=True, exist_ok=True)
        for filename in PACKAGE_FILES:
            shutil.copy2(source_package_root / filename, local_package_dir / filename)
        for dirname in PACKAGE_DIRS:
            _copy_tree(source_package_root / dirname, local_package_dir / dirname)
        shutil.copy2(source_package_root / "__init__.py", local_package_dir / "__init__.py")
    _write_root_remote_code_files(package_dir, source_package_root, include_legacy_dirs=include_legacy_dirs)
    return package_dir


def _build_hf_model_card_update_package(local_dir: str) -> Path:
    source_dir = Path(local_dir)
    package_dir = source_dir / ".hf_model_card_publish"
    if package_dir.exists():
        shutil.rmtree(package_dir, ignore_errors=True)
    package_dir.mkdir(parents=True, exist_ok=True)

    readme_path = source_dir / "README.md"
    if readme_path.exists():
        _link_or_copy_file(readme_path, package_dir / "README.md")
    license_notice = REPO_ROOT / "licenses" / "DINOv3-LICENSE.txt"
    if license_notice.exists():
        _link_or_copy_file(license_notice, package_dir / "DINOv3-LICENSE.txt")

    return package_dir


def push_directory_to_hub(
    local_dir: str,
    repo_id: str,
    commit_message: str = "Upload model",
    *,
    revision: str | None = None,
    include_local_package: bool = True,
    include_legacy_dirs: bool = True,
    include_backbone_weights: bool = True,
    include_segmenter_checkpoints: bool = True,
    include_assets: bool = True,
    include_optional_artifacts: bool = True,
    readme_override_path: str | None = None,
) -> str:
    api = HfApi()
    folder = _build_hf_package(
        local_dir,
        include_local_package=include_local_package,
        include_legacy_dirs=include_legacy_dirs,
        include_backbone_weights=include_backbone_weights,
        include_segmenter_checkpoints=include_segmenter_checkpoints,
        include_assets=include_assets,
        include_optional_artifacts=include_optional_artifacts,
        readme_override_path=readme_override_path,
    )
    api.create_repo(repo_id=repo_id, exist_ok=True)
    if revision and revision != "main":
        try:
            api.create_branch(repo_id=repo_id, branch=revision, exist_ok=True)
        except Exception:
            pass
    delete_patterns = [
        "checkpoints",
        "checkpoints/**",
        "_autotune",
        "_autotune/**",
        "model",
        "model/**",
        "tokenizer",
        "tokenizer/**",
    ]
    if not include_local_package:
        delete_patterns.extend(["lana_radgen", "lana_radgen/**"])
    if not include_legacy_dirs:
        delete_patterns.extend(["attention", "attention/**"])
    if not include_backbone_weights:
        delete_patterns.extend(
            [
                "bundled_backbones/vision_encoder/model.safetensors",
                "bundled_backbones/segmenter_encoder/model.safetensors",
                "bundled_backbones/text_decoder/model.safetensors",
                "bundled_backbones/text_decoder/generation_config.json",
                "bundled_backbones/text_decoder/tokenizer.json",
                "bundled_backbones/text_decoder/tokenizer_config.json",
                "bundled_backbones/text_decoder/vocab.json",
                "bundled_backbones/text_decoder/merges.txt",
            ]
        )
    if not include_segmenter_checkpoints:
        delete_patterns.extend(["segmenters", "segmenters/**"])
    if not include_assets:
        delete_patterns.extend(["assets", "assets/**"])
    if not include_optional_artifacts:
        delete_patterns.extend(
            [
                "evaluations",
                "evaluations/**",
                "benchmark_results.json",
                "model_card.py",
                "run_summary.json",
                "pipeline_autotune.json",
            ]
        )
    _upload_folder_with_retry(
        api,
        repo_id=repo_id,
        folder_path=str(folder),
        commit_message=commit_message,
        revision=revision,
        ignore_patterns=["checkpoints", "checkpoints/**", "_autotune", "_autotune/**", ".hf_publish", ".hf_publish/**"],
        delete_patterns=delete_patterns,
    )
    return f"https://huggingface.co/{repo_id}"


def push_split_inference_and_snapshot_layout(local_dir: str, repo_id: str, commit_message: str = "Upload model") -> str:
    push_directory_to_hub(
        local_dir,
        repo_id,
        commit_message="Publish snapshot-compatible legacy branch for manual loading",
        revision="snapshot-legacy",
        include_local_package=True,
        include_legacy_dirs=True,
        include_backbone_weights=True,
        include_segmenter_checkpoints=True,
        include_assets=True,
        include_optional_artifacts=True,
        readme_override_path=str(Path(local_dir) / "README.snapshot-legacy.md"),
    )
    return push_directory_to_hub(
        local_dir,
        repo_id,
        commit_message=commit_message,
        revision="main",
        include_local_package=False,
        include_legacy_dirs=False,
        include_backbone_weights=False,
        include_segmenter_checkpoints=False,
        include_assets=True,
        include_optional_artifacts=False,
    )


def push_model_card_update_to_hub(local_dir: str, repo_id: str, commit_message: str = "Update model card") -> str:
    api = HfApi()
    folder = _build_hf_model_card_update_package(local_dir)
    api.create_repo(repo_id=repo_id, exist_ok=True)
    _upload_folder_with_retry(
        api,
        repo_id=repo_id,
        folder_path=str(folder),
        commit_message=commit_message,
    )
    return f"https://huggingface.co/{repo_id}"
