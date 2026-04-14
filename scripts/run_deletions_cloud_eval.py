from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import logging
import sys
import zipfile
from pathlib import Path
from types import MethodType
import types

import torch
from safetensors.torch import load_model as load_safetensors_model
from tqdm import tqdm
from PIL import Image


LOGGER = logging.getLogger("run_deletions_cloud_eval")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _ensure_transformers_compat() -> None:
    try:
        from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
    except Exception:
        return

    def _shim_encode_plus(self, *args, **kwargs):
        if args:
            first = args[0]
            if isinstance(first, (list, tuple)) and all(isinstance(token, str) for token in first):
                pair = None
                if len(args) > 1 and isinstance(args[1], (list, tuple)) and all(isinstance(token, str) for token in args[1]):
                    pair = list(args[1])
                token_ids_0 = self.convert_tokens_to_ids(list(first))
                token_ids_1 = self.convert_tokens_to_ids(pair) if pair is not None else None
                input_ids = self.build_inputs_with_special_tokens(token_ids_0, token_ids_1)
                token_type_ids = [0] * len(input_ids)
                attention_mask = [1] * len(input_ids)
                return {
                    "input_ids": input_ids,
                    "token_type_ids": token_type_ids,
                    "attention_mask": attention_mask,
                }
        if hasattr(self, "_encode_plus"):
            return self._encode_plus(*args, **kwargs)
        return self(*args, **kwargs)

    def _shim_build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if hasattr(self, "prepare_for_model"):
            prepared = self.prepare_for_model(
                token_ids_0,
                pair_ids=token_ids_1,
                add_special_tokens=True,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            return prepared["input_ids"]
        if token_ids_1 is None:
            return list(token_ids_0)
        return list(token_ids_0) + list(token_ids_1)

    for cls in (PreTrainedTokenizer, PreTrainedTokenizerFast):
        if cls is None:
            continue
        if not hasattr(cls, "encode_plus"):
            cls.encode_plus = _shim_encode_plus
        if not hasattr(cls, "build_inputs_with_special_tokens"):
            cls.build_inputs_with_special_tokens = _shim_build_inputs_with_special_tokens


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module


def _prepare_deletions_imports() -> None:
    deletions_root = (_repo_root() / "deletions").resolve()
    deletions_root_str = str(deletions_root)
    if deletions_root_str not in sys.path:
        sys.path.insert(0, deletions_root_str)
    for module_name in [name for name in list(sys.modules) if name == "utils" or name.startswith("utils.")]:
        del sys.modules[module_name]
    if "gcsfs" not in sys.modules:
        fake_gcsfs = types.ModuleType("gcsfs")

        class _DummyGCSFileSystem:
            def __init__(self, *args, **kwargs):
                pass

            def exists(self, *args, **kwargs):
                return False

        fake_gcsfs.GCSFileSystem = _DummyGCSFileSystem
        sys.modules["gcsfs"] = fake_gcsfs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the original deletions cloud evaluation flow and save outputs into a repro folder.")
    parser.add_argument("--run-dir", default="artifacts/repro_deletions_cloud_eval")
    parser.add_argument("--model-path", default="deletions/models/model_best7.pth")
    parser.add_argument("--models-dir", default="")
    parser.add_argument("--all-models", action="store_true", help="Evaluate all report-generation model files inside --models-dir.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--limit-batches", type=int, default=0)
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip models whose metrics JSON already exists.")
    parser.add_argument("--force", action="store_true", help="Re-evaluate even if outputs already exist.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-level", default="INFO")
    return parser


def _load_notebook_style_modules():
    _prepare_deletions_imports()
    from utils.train_comparison import build_tokenizer_from_labels  # type: ignore
    from utils.processing import image_transform  # type: ignore
    from utils.data.mimic_dataset import extract_findings  # type: ignore
    from utils.text_metrics import evaluate_all_metrics, save_metrics_to_json  # type: ignore
    from utils.models.complete_model import create_complete_model  # type: ignore

    return {
        "build_tokenizer_from_labels": build_tokenizer_from_labels,
        "image_transform": image_transform,
        "extract_findings": extract_findings,
        "evaluate_all_metrics": evaluate_all_metrics,
        "save_metrics_to_json": save_metrics_to_json,
        "create_complete_model": create_complete_model,
    }


def _resolve_default_paths() -> tuple[dict, dict]:
    repo_root = _repo_root()
    chexpert_dir = repo_root / "Datasets" / "CheXpert"
    mimic_dir = repo_root / "Datasets" / "MIMIC"
    chexpert_paths = {
        "chexpert_data_path": str((repo_root / "Datasets" / "CHEXPERT516").resolve()),
        "chexpert_data_csv": str((chexpert_dir / "df_chexpert_plus_240401_findings.csv").resolve()),
    }
    mimic_paths = {
        "mimic_data_path": str(mimic_dir.resolve()),
        "mimic_splits_csv": str((mimic_dir / "mimic-cxr-2.0.0-split.csv.gz").resolve()),
        "mimic_metadata_csv": str((mimic_dir / "mimic-cxr-2.0.0-metadata-findings-only.csv").resolve()),
        "mimic_reports_path": str((mimic_dir / "cxr-record-list.csv.gz").resolve()),
        "mimic_images_dir": str((mimic_dir / "images" / "datos").resolve()),
    }
    return chexpert_paths, mimic_paths


def _apply_legacy_transformer_shims(model: object) -> None:
    decoder = getattr(model, "decoder", None)
    transformer = getattr(decoder, "transformer", None)
    if decoder is not None and not hasattr(decoder, "model_parallel"):
        decoder.model_parallel = False
    if transformer is not None and not hasattr(transformer, "get_head_mask"):
        def _legacy_get_head_mask(this, head_mask, num_hidden_layers, is_attention_chunked: bool = False):
            if head_mask is None:
                return [None] * num_hidden_layers
            return head_mask

        transformer.get_head_mask = MethodType(_legacy_get_head_mask, transformer)
    if transformer is not None and not hasattr(transformer, "model_parallel"):
        transformer.model_parallel = False


def _load_weights(model: object, model_path: Path) -> None:
    if model_path.suffix.lower() in {".safetensor", ".safetensors"}:
        load_safetensors_model(model, str(model_path))
        return
    ckpt = torch.load(model_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        LOGGER.warning("Model loaded non-strictly. Missing=%s Unexpected=%s", missing, unexpected)


def _load_raw_report_texts(report_zip_path: Path) -> dict[tuple[int, int], str]:
    reports: dict[tuple[int, int], str] = {}
    import re

    with zipfile.ZipFile(report_zip_path) as archive:
        for name in archive.namelist():
            if not name.endswith(".txt"):
                continue
            match = re.search(r"/p(\d+)/s(\d+)\.txt$", name)
            if not match:
                continue
            subject_id = int(match.group(1))
            study_id = int(match.group(2))
            reports[(subject_id, study_id)] = archive.read(name).decode("utf-8", errors="ignore")
    return reports


def _load_current_manifest(mimic_root: Path, limit: int = 0):
    evaluate_path = _repo_root() / "scripts" / "evaluate.py"
    module = _load_module_from_path("_current_eval_manifest", evaluate_path)
    manifest = module._build_mimic_test_manifest(mimic_root=mimic_root, limit=limit, findings_only=True)
    raw_reports = _load_raw_report_texts(mimic_root / "mimic-cxr-reports.zip")
    manifest["report_text"] = manifest.apply(
        lambda row: raw_reports.get((int(row["subject_id"]), int(row["study_id"])), ""),
        axis=1,
    )
    manifest = manifest[manifest["report_text"].map(bool)].reset_index(drop=True)
    return manifest


def _release_memory() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def _model_paths(args: argparse.Namespace) -> list[Path]:
    if args.all_models:
        models_dir = Path(args.models_dir or "deletions/models").resolve()
        candidates = sorted(
            path for path in models_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".pth", ".safetensor", ".safetensors"}
        )
        report_models = [path for path in candidates if "classification" not in path.name.lower()]
        if not report_models:
            raise FileNotFoundError(f"No report-generation model files found in {models_dir}")
        return report_models
    return [Path(args.model_path).resolve()]


def _safe_tag(model_path: Path, explicit_tag: str) -> str:
    return explicit_tag.strip() if explicit_tag.strip() else model_path.stem


def _generated_texts_path(run_dir: Path, tag: str, token_count: int) -> Path:
    suffix = f"_{tag}" if tag else ""
    return run_dir / f"bestmodelcloud_generated_texts{suffix}_{token_count}.json"


def _metrics_path(run_dir: Path, tag: str, token_count: int) -> Path:
    suffix = f"_{tag}" if tag else ""
    return run_dir / f"cloud_best_model7_MIMIC{suffix}_{token_count}.json"


def _output_paths(run_dir: Path, tag: str, token_count: int) -> tuple[Path, Path]:
    return (
        _generated_texts_path(run_dir, tag, token_count),
        _metrics_path(run_dir, tag, token_count),
    )


def _load_sweep_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {"models": {}}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {"models": {}}


def _save_sweep_state(state_path: Path, payload: dict) -> None:
    state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _truncate_text_with_tokenizer(tokenizer, text: str, max_tokens: int) -> str:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    truncated_ids = token_ids[:max_tokens]
    try:
        decoded = tokenizer.decode(truncated_ids, skip_special_tokens=True)
    except TypeError:
        decoded = tokenizer.decode(truncated_ids)
    return str(decoded).strip()


def _write_generated_variant(
    *,
    run_dir: Path,
    tag: str,
    token_count: int,
    generated_text: list[str],
    target_text: list[str],
    model_path: Path,
) -> Path:
    generated_texts_path = _generated_texts_path(run_dir, tag, token_count)
    generated_payload = {
        "generated": generated_text,
        "target": target_text,
        "model_path": str(model_path.resolve()),
        "max_new_tokens": int(token_count),
    }
    generated_texts_path.write_text(json.dumps(generated_payload, indent=2), encoding="utf-8")
    return generated_texts_path


def _generate_single_model_texts(
    *,
    args: argparse.Namespace,
    model_path: Path,
    run_dir: Path,
    manifest,
    transform,
    extract_findings,
    tok,
    create_complete_model,
) -> dict[int, Path]:
    tag = _safe_tag(model_path, args.output_tag)
    generation_token_count = max(150, int(args.max_new_tokens))
    generated_texts_path_150 = _generated_texts_path(run_dir, tag, generation_token_count)
    generated_texts_path_128 = _generated_texts_path(run_dir, tag, 128)
    generated_texts_path_100 = _generated_texts_path(run_dir, tag, 100)
    if args.skip_existing and not args.force and generated_texts_path_150.exists() and generated_texts_path_128.exists() and generated_texts_path_100.exists():
        LOGGER.info("Skipping generation for %s because all generated-text variants already exist", model_path.name)
        return {
            generation_token_count: generated_texts_path_150,
            128: generated_texts_path_128,
            100: generated_texts_path_100,
        }

    pad_id = tok.pad_token_id
    eos_id = tok.eos_token_id
    device = torch.device(args.device)
    repo_root = _repo_root()
    lung_path = str((repo_root / "models" / "dino_unet_decoder_finetuned.pth").resolve())
    heart_path = str((repo_root / "models" / "dino_unet_organos_best.pth").resolve())
    model = create_complete_model(
        device=device,
        SEGMENTER_MODEL_PATH_LUNG=lung_path,
        SEGMENTER_MODEL_PATH_HEART=heart_path,
        freeze_encoder=False,
        mask_implementation="default",
    )
    _apply_legacy_transformer_shims(model)
    _load_weights(model, model_path.resolve())
    model.eval()

    generated_text: list[str] = []
    target_text: list[str] = []

    try:
        with torch.inference_mode():
            total_batches = (len(manifest) + max(1, args.batch_size) - 1) // max(1, args.batch_size)
            for iteration, start_idx in enumerate(tqdm(range(0, len(manifest), max(1, args.batch_size)), total=total_batches, desc=model_path.stem), start=1):
                batch_df = manifest.iloc[start_idx : start_idx + max(1, args.batch_size)]
                image_tensors = []
                batch_targets: list[str] = []
                for _, row in batch_df.iterrows():
                    image = Image.open(Path(row["processed_image_path"])).convert("RGB")
                    image_tensors.append(transform(image))
                    batch_targets.append(extract_findings(str(row["report_text"])))

                pixel_values = torch.stack(image_tensors, dim=0).to(model.device, non_blocking=True)
                patches = model.encoder(pixel_values)
                projected_patches = model.linear_projection(patches)
                segmented_layers = model.segmenter(pixel_values, model.num_layers)

                gen_ids = model.decoder.generate(
                    inputs_embeds=projected_patches,
                    max_new_tokens=generation_token_count,
                    do_sample=False,
                    top_k=50,
                    top_p=0.95,
                    temperature=1.0,
                    repetition_penalty=1.2,
                    num_beams=1,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                    use_cache=True,
                    segmentation_mask=segmented_layers,
                    prefix_allowed_length=0,
                    plot_attention_mask=False,
                    plot_attention_mask_layer=[],
                    plot_attention_map=False,
                    plot_attention_map_layer=[],
                    plot_attention_map_generation=0,
                )
                texts = model.tokenizer.batch_decode(gen_ids.detach().cpu(), skip_special_tokens=True)
                generated_text.extend(texts)
                target_text.extend(batch_targets)

                if args.limit_batches > 0 and iteration >= args.limit_batches:
                    break

        variant_paths: dict[int, Path] = {}
        variant_paths[generation_token_count] = _write_generated_variant(
            run_dir=run_dir,
            tag=tag,
            token_count=generation_token_count,
            generated_text=generated_text,
            target_text=target_text,
            model_path=model_path,
        )
        truncated_128 = [_truncate_text_with_tokenizer(tok, text, 128) for text in generated_text]
        truncated_100 = [_truncate_text_with_tokenizer(tok, text, 100) for text in generated_text]
        variant_paths[128] = _write_generated_variant(
            run_dir=run_dir,
            tag=tag,
            token_count=128,
            generated_text=truncated_128,
            target_text=target_text,
            model_path=model_path,
        )
        variant_paths[100] = _write_generated_variant(
            run_dir=run_dir,
            tag=tag,
            token_count=100,
            generated_text=truncated_100,
            target_text=target_text,
            model_path=model_path,
        )
        for token_count, path in variant_paths.items():
            LOGGER.info("Saved generated texts (%s tokens) to %s", token_count, path)
        return variant_paths
    finally:
        del model
        _release_memory()


def _evaluate_generated_json(
    *,
    run_dir: Path,
    model_path: Path,
    tag: str,
    batch_size: int,
    image_size: int,
    token_count: int,
    generated_json_path: Path,
    evaluate_all_metrics,
    save_metrics_to_json,
) -> Path:
    payload = json.loads(generated_json_path.read_text(encoding="utf-8"))
    generated_text = [str(text) for text in payload.get("generated", [])]
    target_text = [str(text) for text in payload.get("target", [])]
    eval_results = evaluate_all_metrics(
        generated=generated_text,
        original=target_text,
        evaluation_mode="CheXagent",
    )
    eval_results["training_time_seconds"] = 0
    eval_results["max_new_tokens"] = int(token_count)
    eval_results["batch_size"] = int(batch_size)
    eval_results["image_size"] = int(image_size)
    eval_results["model_path"] = str(model_path.resolve())
    metrics_path = _metrics_path(run_dir, tag, token_count)
    save_metrics_to_json(eval_results, str(metrics_path))
    LOGGER.info("Saved metrics (%s tokens) to %s", token_count, metrics_path)
    return metrics_path


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    _ensure_transformers_compat()

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.force:
        args.skip_existing = False

    modules = _load_notebook_style_modules()
    build_tokenizer_from_labels = modules["build_tokenizer_from_labels"]
    extract_findings = modules["extract_findings"]
    image_transform = modules["image_transform"]
    evaluate_all_metrics = modules["evaluate_all_metrics"]
    save_metrics_to_json = modules["save_metrics_to_json"]
    create_complete_model = modules["create_complete_model"]

    tok = build_tokenizer_from_labels(gpt2=True)

    _, mimic_paths = _resolve_default_paths()
    manifest = _load_current_manifest(Path(mimic_paths["mimic_data_path"]), limit=0)
    transform = image_transform(img_size=args.image_size)
    state_path = run_dir / "evaluation_state.json"
    state = _load_sweep_state(state_path)
    state.setdefault("models", {})
    token_variants = [max(150, int(args.max_new_tokens)), 128, 100]
    generation_registry: dict[str, dict[str, str]] = {}

    for model_path in _model_paths(args):
        tag = _safe_tag(model_path, args.output_tag)
        model_state = state["models"].get(tag, {})
        try:
            generated_paths = _generate_single_model_texts(
                args=args,
                model_path=model_path,
                run_dir=run_dir,
                manifest=manifest,
                transform=transform,
                extract_findings=extract_findings,
                tok=tok,
                create_complete_model=create_complete_model,
            )
            generation_registry[tag] = {str(token_count): str(path) for token_count, path in generated_paths.items()}
            state["models"][tag] = {
                "model_path": str(model_path.resolve()),
                "generation_status": "completed",
                "generated_texts": generation_registry[tag],
                "evaluation_status": model_state.get("evaluation_status", "pending"),
                "evaluated_metrics": model_state.get("evaluated_metrics", {}),
            }
        except Exception as exc:
            LOGGER.exception("Failed while generating %s", model_path.name)
            state["models"][tag] = {
                "model_path": str(model_path.resolve()),
                "generation_status": "failed",
                "evaluation_status": model_state.get("evaluation_status", "pending"),
                "generated_texts": model_state.get("generated_texts", {}),
                "evaluated_metrics": model_state.get("evaluated_metrics", {}),
                "error": repr(exc),
            }
        finally:
            _save_sweep_state(state_path, state)
            _release_memory()

    LOGGER.info("Generation phase complete. Starting evaluation phase.")
    for model_path in _model_paths(args):
        tag = _safe_tag(model_path, args.output_tag)
        model_state = state["models"].get(tag, {})
        if model_state.get("generation_status") != "completed":
            LOGGER.info("Skipping evaluation for %s because generation did not complete", model_path.name)
            continue
        generated_texts = model_state.get("generated_texts", {})
        evaluated_metrics = dict(model_state.get("evaluated_metrics", {}))
        for token_count in token_variants:
            token_key = str(token_count)
            generated_json = Path(generated_texts.get(token_key, ""))
            metrics_path = _metrics_path(run_dir, tag, token_count)
            if args.skip_existing and not args.force and metrics_path.exists():
                evaluated_metrics[token_key] = str(metrics_path)
                LOGGER.info("Skipping evaluation for %s at %s tokens because metrics already exist", model_path.name, token_count)
                continue
            if not generated_json.exists():
                LOGGER.warning("Missing generated texts for %s at %s tokens; skipping evaluation", model_path.name, token_count)
                continue
            try:
                saved_metrics_path = _evaluate_generated_json(
                    run_dir=run_dir,
                    model_path=model_path,
                    tag=tag,
                    batch_size=args.batch_size,
                    image_size=args.image_size,
                    token_count=token_count,
                    generated_json_path=generated_json,
                    evaluate_all_metrics=evaluate_all_metrics,
                    save_metrics_to_json=save_metrics_to_json,
                )
                evaluated_metrics[token_key] = str(saved_metrics_path)
                state["models"][tag] = {
                    **state["models"].get(tag, {}),
                    "evaluation_status": "completed",
                    "evaluated_metrics": evaluated_metrics,
                }
            except Exception as exc:
                LOGGER.exception("Failed while evaluating %s at %s tokens", model_path.name, token_count)
                state["models"][tag] = {
                    **state["models"].get(tag, {}),
                    "evaluation_status": "failed",
                    "evaluated_metrics": evaluated_metrics,
                    "evaluation_error": repr(exc),
                }
            finally:
                _save_sweep_state(state_path, state)
                _release_memory()


if __name__ == "__main__":
    main()
