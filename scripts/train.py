import argparse
import copy
import gc
import json
import logging
import math
import os
import random
import re
import shutil
import time
import zipfile
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Sampler
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from lana_radgen import LanaConfig, LanaForConditionalGeneration
from lana_radgen.data import ResizeCachedReportDataset
from lana_radgen.hub import push_directory_to_hub, push_split_inference_and_snapshot_layout
from lana_radgen.logging_utils import configure_logging
from lana_radgen.model_card import DINO_V3_NOTICE, RESEARCH_USE_NOTICE, build_best_model_notice, build_dual_usage_section, build_main_branch_usage_section

LOGGER = logging.getLogger("train")
SPLIT_INFERENCE_REPOS = {
    "manu02/LAnA-MIMIC-CHEXPERT",
    "manu02/LAnA-MIMIC",
    "manu02/LAnA",
    "manu02/LAnA-v2",
    "manu02/LAnA-v3",
    "manu02/LAnA-v4",
    "manu02/LAnA-v5",
}

BENCHMARK_METHODS = ["qlora_paged_adamw8bit", "lora_adamw", "full_adam", "full_adam8bit"]
METHOD_CHOICES = BENCHMARK_METHODS + ["full_adamw", "lora_adamw8bit"]
PAD_TOKEN = "<|pad|>"


class StatefulShuffleSampler(Sampler[int]):
    def __init__(self, dataset_size: int, seed: int = 42):
        self.dataset_size = dataset_size
        self.seed = seed
        self.epoch = 0
        self.position = 0
        self.order = self._build_order(0)

    def _build_order(self, epoch: int) -> list[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + epoch)
        return torch.randperm(self.dataset_size, generator=generator).tolist()

    def set_epoch(self, epoch: int, position: int = 0) -> None:
        self.epoch = epoch
        self.position = position
        self.order = self._build_order(epoch)

    def state_dict(self) -> dict:
        return {
            "dataset_size": self.dataset_size,
            "seed": self.seed,
            "epoch": self.epoch,
            "position": self.position,
            "order": self.order,
        }

    def load_state_dict(self, state: dict) -> None:
        self.dataset_size = int(state["dataset_size"])
        self.seed = int(state["seed"])
        self.epoch = int(state["epoch"])
        self.position = int(state["position"])
        self.order = [int(idx) for idx in state["order"]]

    def __iter__(self):
        while self.position < len(self.order):
            idx = self.order[self.position]
            self.position += 1
            yield idx

    def __len__(self) -> int:
        return max(0, len(self.order) - self.position)


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark and train LANA radiology report generation model.")
    parser.add_argument("--model-variant", default="default", choices=["default", "lana_v5"])
    parser.add_argument("--vision-model-name", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--text-model-name", default="gpt2")
    parser.add_argument("--wandb-project", default="lana-radgen")
    parser.add_argument("--run-name", default="benchmark-train")
    parser.add_argument("--dataset", default="combined", choices=["chexpert", "mimic", "combined"])
    parser.add_argument("--metadata-path", default="Datasets/CheXpert/df_chexpert_plus_240401_findings.csv")
    parser.add_argument("--image-root", default="Datasets/CheXpert/images")
    parser.add_argument("--mimic-root", default="Datasets/MIMIC")
    parser.add_argument("--mimic-findings-only", action="store_true", help="Restrict MIMIC rows to studies with a structured FINDINGS section only.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--global-batch-size", type=int, default=0, help="Effective batch size via gradient accumulation. 0 uses local batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--visual-projection-type", default="mlp4", choices=["mlp4", "linear"])
    parser.add_argument("--attention-bias-mode", default="layerwise", choices=["layerwise", "gaussian_legacy"])
    parser.add_argument("--vision-prefix-tokens-to-skip", type=int, default=1)
    parser.add_argument("--precision", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--skip-cached-resize", action="store_true", help="Skip runtime resize for cached PNGs that already match --image-size.")
    parser.add_argument("--cache-size-audit-path", default=".cache/image_size_audit.json")
    parser.add_argument("--torch-compile", action="store_true", help="Enable torch.compile for supported training methods.")
    parser.add_argument("--torch-compile-mode", default="default", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--segmentation-model-name", default="facebook/dinov3-convnext-small-pretrain-lvd1689m")
    parser.add_argument("--lung-segmenter-checkpoint", default="models/lung_segmenter_dinounet_finetuned.pth")
    parser.add_argument("--heart-segmenter-checkpoint", default="models/heart_segmenter_dinounet_best.pth")
    parser.add_argument("--disable-segmentation-mask", action="store_true")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--generation-use-bos-token", dest="generation_use_bos_token", action="store_true")
    parser.add_argument("--no-generation-use-bos-token", dest="generation_use_bos_token", action="store_false")
    parser.set_defaults(generation_use_bos_token=True)
    parser.add_argument("--generation-stop-on-eos", action="store_true")
    parser.add_argument("--generation-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--benchmark-steps", type=int, default=1)
    parser.add_argument("--time-limit-minutes", type=int, default=0)
    parser.add_argument("--duration", default="", help="Wall-clock training duration. Examples: 1800, 30m, 2h, 01:30:00")
    parser.add_argument("--max-train-steps", type=int, default=1000000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--method", default="auto", choices=["auto"] + METHOD_CHOICES)
    parser.add_argument("--output-dir", default="artifacts/benchmark_train")
    parser.add_argument("--repo-id", default="manu02/LAnA")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument("--export-only", action="store_true", help="Skip training and export the latest saved checkpoint into the normal artifact layout.")
    parser.add_argument("--save-every-n-steps", type=int, default=1000)
    parser.add_argument("--log-every-n-steps", type=int, default=100)
    parser.add_argument("--keep-last-n-checkpoints", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser


def apply_model_variant_presets(args) -> None:
    if args.model_variant != "lana_v5":
        return
    args.attention_bias_mode = "gaussian_legacy"
    args.vision_prefix_tokens_to_skip = 5
    args.generation_use_bos_token = False
    args.generation_stop_on_eos = True
    args.generation_repetition_penalty = 1.2


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_runtime(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def resolve_training_precision(device: torch.device, requested: str) -> dict[str, object]:
    requested = requested.lower()
    if device.type != "cuda":
        return {
            "requested": requested,
            "resolved": "fp32",
            "amp_dtype": None,
            "use_grad_scaler": False,
            "decoder_compute_dtype": "float16",
        }

    bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    if requested == "auto":
        resolved = "bf16" if bf16_supported else "fp16"
    elif requested == "bf16" and not bf16_supported:
        LOGGER.warning("bf16 was requested but is not supported on this CUDA device; falling back to fp16.")
        resolved = "fp16"
    else:
        resolved = requested

    if resolved == "fp32":
        amp_dtype = None
    elif resolved == "fp16":
        amp_dtype = torch.float16
    elif resolved == "bf16":
        amp_dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported precision mode: {requested}")

    return {
        "requested": requested,
        "resolved": resolved,
        "amp_dtype": amp_dtype,
        "use_grad_scaler": resolved == "fp16",
        "decoder_compute_dtype": "bfloat16" if resolved == "bf16" else "float16",
    }


def autocast_context(device: torch.device, amp_dtype):
    if amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def parse_duration_to_seconds(duration: str) -> int:
    raw = duration.strip().lower()
    if not raw:
        return 0
    if ":" in raw:
        parts = [int(part) for part in raw.split(":")]
        if len(parts) == 2:
            minutes, seconds = parts
            return minutes * 60 + seconds
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return hours * 3600 + minutes * 60 + seconds
        raise ValueError(f"Unsupported duration format: {duration}")
    match = re.fullmatch(r"(\d+)([smh]?)", raw)
    if not match:
        raise ValueError(f"Unsupported duration format: {duration}")
    value = int(match.group(1))
    unit = match.group(2) or "s"
    if unit == "s":
        return value
    if unit == "m":
        return value * 60
    if unit == "h":
        return value * 3600
    raise ValueError(f"Unsupported duration unit: {duration}")


def should_skip_cached_resize(args) -> bool:
    if args.skip_cached_resize:
        return True

    audit_path = Path(args.cache_size_audit_path)
    if not audit_path.exists():
        return False

    try:
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Could not read cache size audit file %s: %s", audit_path, exc)
        return False

    expected_image_size = int(args.image_size)
    datasets_payload = payload.get("datasets", {})
    required: list[tuple[str, str, str]] = []
    if args.dataset in {"chexpert", "combined"}:
        required.append(("chexpert", "image_root", str(Path(args.image_root).resolve())))
    if args.dataset in {"mimic", "combined"}:
        required.append(("mimic", "mimic_root", str(Path(args.mimic_root).resolve())))

    if not required:
        return False

    for dataset_name, path_key, expected_root in required:
        entry = datasets_payload.get(dataset_name)
        if not entry:
            return False
        if not entry.get("verified", False):
            return False
        if int(entry.get("expected_size", -1)) != expected_image_size:
            return False
        if str(entry.get(path_key, "")) != expected_root:
            return False

    LOGGER.info("Using cache size audit %s to skip runtime resize.", audit_path)
    return True


def unwrap_model(model):
    return getattr(model, "_orig_mod", model)


def maybe_compile_model(model, args, method: str, device: torch.device):
    if not getattr(args, "torch_compile", False):
        return model
    if device.type != "cuda":
        LOGGER.info("Skipping torch.compile because the selected device is not CUDA.")
        return model
    if not hasattr(torch, "compile"):
        LOGGER.warning("torch.compile is unavailable in this PyTorch build.")
        return model
    if method.startswith("qlora"):
        LOGGER.warning("Skipping torch.compile for method=%s because quantized decoder paths are less reliable.", method)
        return model
    try:
        compiled = torch.compile(model, mode=args.torch_compile_mode)
        LOGGER.info("Enabled torch.compile with mode=%s for method=%s", args.torch_compile_mode, method)
        return compiled
    except Exception as exc:
        LOGGER.warning("torch.compile failed for method=%s: %s", method, exc)
        return model


def load_env_file(env_path: str = ".env") -> None:
    path = Path(env_path)
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def build_chexpert_manifest(metadata_path: str, image_root: str, split: str) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    metadata = metadata[metadata["split"].astype(str) == split].copy()
    findings = metadata["section_findings"].fillna("").astype(str).str.strip()
    image_paths = metadata["path_to_image"].astype(str).str.replace(".jpg", ".png", regex=False)
    processed_paths = image_paths.map(lambda relative: str((Path(image_root) / relative).resolve()))
    manifest = pd.DataFrame(
        {
            "report_id": metadata.index.astype(str),
            "processed_image_path": processed_paths,
            "report_text": findings,
            "dataset_source": "chexpert",
        }
    )
    manifest = manifest[manifest["report_text"] != ""]
    manifest = manifest[manifest["processed_image_path"].map(lambda p: Path(p).exists())]
    manifest = manifest.reset_index(drop=True)
    if manifest.empty:
        raise RuntimeError(f"No valid CheXpert rows found for split={split}.")
    return manifest


def _extract_report_section(report_text: str, allow_impression_fallback: bool = True) -> str:
    normalized = report_text.replace("\r\n", "\n")
    match = re.search(r"FINDINGS:\s*(.*?)(?:\n\s*[A-Z ]+:\s|$)", normalized, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()
    if allow_impression_fallback:
        impression_match = re.search(r"IMPRESSION:\s*(.*?)(?:\n\s*[A-Z ]+:\s|$)", normalized, flags=re.IGNORECASE | re.DOTALL)
        if impression_match:
            return re.sub(r"\s+", " ", impression_match.group(1)).strip()
        return re.sub(r"\s+", " ", normalized).strip()
    return ""


def _load_report_texts(report_zip_path: Path, allow_impression_fallback: bool = True) -> dict[tuple[int, int], str]:
    reports = {}
    with zipfile.ZipFile(report_zip_path) as archive:
        for name in archive.namelist():
            if not name.endswith(".txt"):
                continue
            match = re.search(r"/p(\d+)/s(\d+)\.txt$", name)
            if not match:
                continue
            subject_id = int(match.group(1))
            study_id = int(match.group(2))
            text = archive.read(name).decode("utf-8", errors="ignore")
            reports[(subject_id, study_id)] = _extract_report_section(text, allow_impression_fallback=allow_impression_fallback)
    return reports


def _resolve_mimic_processed_image_path(subject_id: int, study_id: int, dicom_id: str, image_root: Path) -> Path:
    return image_root / f"p{subject_id}" / f"s{study_id}" / f"{dicom_id}.png"


def build_mimic_manifest(mimic_root: str, split: str, findings_only: bool = False) -> pd.DataFrame:
    root = Path(mimic_root)
    split_df = pd.read_csv(root / "mimic-cxr-2.0.0-split.csv.gz", compression="gzip")
    records_df = pd.read_csv(root / "cxr-record-list.csv.gz", compression="gzip")
    metadata_df = pd.read_csv(root / "mimic-cxr-2.0.0-metadata.csv")
    reports = _load_report_texts(root / "mimic-cxr-reports.zip", allow_impression_fallback=not findings_only)

    split_name = "validate" if split == "valid" else split
    df = split_df[split_df["split"] == split_name].copy()
    if findings_only:
        findings_df = pd.read_csv(root / "mimic-cxr-2.0.0-metadata-findings-only.csv")
        df = df.merge(findings_df[["subject_id", "study_id", "dicom_id"]], on=["subject_id", "study_id", "dicom_id"], how="inner")
    df = df.merge(records_df, on=["subject_id", "study_id", "dicom_id"], how="left")
    df = df.merge(metadata_df[["subject_id", "study_id", "dicom_id", "ViewPosition"]], on=["subject_id", "study_id", "dicom_id"], how="left")

    df["ViewPosition"] = df["ViewPosition"].astype(str).str.upper()
    df = df[df["ViewPosition"].isin({"PA", "AP"})].copy()
    df = df.sort_values(by=["subject_id", "study_id", "dicom_id"]).drop_duplicates(subset=["subject_id", "study_id"], keep="first")

    image_root = root / "images" / "datos"
    df["processed_image_path"] = df.apply(
        lambda row: str(
            _resolve_mimic_processed_image_path(
                int(row["subject_id"]),
                int(row["study_id"]),
                str(row["dicom_id"]),
                image_root,
            ).resolve()
        ),
        axis=1,
    )
    df["report_text"] = df.apply(lambda row: reports.get((int(row["subject_id"]), int(row["study_id"])), ""), axis=1)
    manifest = pd.DataFrame(
        {
            "report_id": df["subject_id"].astype(str) + "_" + df["study_id"].astype(str),
            "processed_image_path": df["processed_image_path"],
            "report_text": df["report_text"].astype(str).str.strip(),
            "dataset_source": "mimic",
        }
    )
    manifest = manifest[manifest["report_text"] != ""]
    manifest = manifest[manifest["processed_image_path"].map(lambda p: Path(p).exists())]
    manifest = manifest.reset_index(drop=True)
    if manifest.empty:
        raise RuntimeError(f"No valid MIMIC rows found for split={split}.")
    return manifest


def combine_manifests(manifests: list[pd.DataFrame]) -> pd.DataFrame:
    filtered = [manifest.copy() for manifest in manifests if manifest is not None and not manifest.empty]
    if not filtered:
        raise RuntimeError("No manifests available to combine.")
    combined = pd.concat(filtered, ignore_index=True)
    combined = combined.reset_index(drop=True)
    if combined.empty:
        raise RuntimeError("Combined manifest is empty.")
    return combined


def collate_batch(batch, pad_token_id: int):
    pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)
    anatomical_masks = torch.stack([item["anatomical_masks"] for item in batch], dim=0)
    input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=pad_token_id)
    attention_mask = pad_sequence([item["attention_mask"] for item in batch], batch_first=True, padding_value=0)
    labels = input_ids.clone()
    labels[labels == pad_token_id] = -100
    return {
        "pixel_values": pixel_values,
        "anatomical_masks": anatomical_masks,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def make_dataloader(
    manifest,
    tokenizer,
    batch_size: int,
    image_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    skip_cached_resize: bool = False,
    prepend_bos_token: bool = True,
):
    max_text_length = getattr(tokenizer, "model_max_length", None)
    if isinstance(max_text_length, int) and max_text_length > 100000:
        max_text_length = 1024
    dataset = ResizeCachedReportDataset(
        manifest=manifest,
        tokenizer=tokenizer,
        image_size=image_size,
        max_text_length=max_text_length,
        resize_loaded_images=not skip_cached_resize,
        prepend_bos_token=prepend_bos_token,
    )
    sampler = StatefulShuffleSampler(len(dataset), seed=seed) if shuffle else None
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "sampler": sampler,
        "num_workers": num_workers,
        "collate_fn": lambda batch: collate_batch(batch, tokenizer.pad_token_id),
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(**loader_kwargs)
    return dataset, loader, sampler


def freeze_module(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = True


def build_model(method: str, args, device: torch.device):
    config = LanaConfig(
        vision_model_name=args.vision_model_name,
        text_model_name=args.text_model_name,
        image_size=args.image_size,
        visual_projection_type=args.visual_projection_type,
        attention_bias_mode=args.attention_bias_mode,
        vision_feature_prefix_tokens_to_skip=args.vision_prefix_tokens_to_skip,
        segmentation_model_name=args.segmentation_model_name,
        generation_use_bos_token=args.generation_use_bos_token,
        generation_stop_on_eos=args.generation_stop_on_eos,
        generation_repetition_penalty=args.generation_repetition_penalty,
        lung_segmenter_checkpoint=args.lung_segmenter_checkpoint,
        heart_segmenter_checkpoint=args.heart_segmenter_checkpoint,
        use_segmentation_mask=not args.disable_segmentation_mask,
        decoder_load_in_4bit=(method == "qlora_paged_adamw8bit"),
        decoder_compute_dtype=getattr(args, "decoder_compute_dtype", "float16"),
    )
    model = LanaForConditionalGeneration(config)
    if model.tokenizer.pad_token_id is None:
        model.tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
    model.text_decoder.config.pad_token_id = model.tokenizer.pad_token_id
    if hasattr(model.text_decoder, "generation_config") and model.text_decoder.generation_config is not None:
        model.text_decoder.generation_config.pad_token_id = model.tokenizer.pad_token_id
        model.text_decoder.generation_config.eos_token_id = None
    freeze_module(model.vision_encoder)
    if model.segmenter is not None and args.disable_segmentation_mask:
        freeze_module(model.segmenter)
    if method.startswith("qlora"):
        model.text_decoder = prepare_model_for_kbit_training(model.text_decoder)
    if method.startswith("lora") or method.startswith("qlora"):
        freeze_module(model.text_decoder)
        unfreeze_module(model.visual_projection)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["c_attn", "c_proj"],
            fan_in_fan_out=True,
        )
        model.text_decoder = get_peft_model(model.text_decoder, lora_config)
    else:
        unfreeze_module(model.text_decoder)
        unfreeze_module(model.visual_projection)
    model.move_non_quantized_modules(device)
    model.train()
    return maybe_compile_model(model, args, method, device)


def build_optimizer(method: str, model: torch.nn.Module, learning_rate: float, weight_decay: float):
    params = [param for param in model.parameters() if param.requires_grad]
    if method == "full_adam":
        return torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    if method == "full_adam8bit":
        import bitsandbytes as bnb

        return bnb.optim.Adam8bit(params, lr=learning_rate, weight_decay=weight_decay)
    if method == "lora_adamw8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(params, lr=learning_rate, weight_decay=weight_decay)
    if method == "qlora_paged_adamw8bit":
        import bitsandbytes as bnb

        return bnb.optim.PagedAdamW8bit(params, lr=learning_rate, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)


def compute_scheduler_steps(dataset_size: int, effective_global_batch_size: int, epochs: int, max_train_steps: int) -> tuple[int, int, int]:
    steps_per_epoch = max(1, math.ceil(dataset_size / max(1, effective_global_batch_size)))
    total_training_steps = steps_per_epoch * max(1, epochs)
    if max_train_steps > 0:
        total_training_steps = min(total_training_steps, max_train_steps)
    warmup_steps = max(1, math.ceil(total_training_steps * 0.05))
    return steps_per_epoch, total_training_steps, warmup_steps


def build_scheduler(optimizer, total_training_steps: int, warmup_steps: int):
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(1, total_training_steps),
    )


def compute_accumulation_steps(local_batch_size: int, global_batch_size: int) -> tuple[int, int]:
    if global_batch_size <= 0:
        return 1, local_batch_size
    accumulation_steps = max(1, math.ceil(global_batch_size / local_batch_size))
    effective_global_batch_size = local_batch_size * accumulation_steps
    return accumulation_steps, effective_global_batch_size


def move_batch_to_device(batch, device: torch.device):
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=device.type == "cuda")
        else:
            moved[key] = value
    return moved


def train_step(
    model,
    optimizer,
    scheduler,
    scaler,
    batch,
    device: torch.device,
    accumulation_steps: int,
    step_optimizer: bool,
    amp_dtype,
    measure_time: bool = False,
):
    batch = move_batch_to_device(batch, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with autocast_context(device, amp_dtype):
        outputs = model(**batch)
        scaled_loss = outputs.loss / accumulation_steps
    if scaler is not None and scaler.is_enabled():
        scaler.scale(scaled_loss).backward()
    else:
        scaled_loss.backward()
    if step_optimizer:
        if scaler is not None and scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    if measure_time and device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start if measure_time else 0.0
    return float(outputs.loss.detach().cpu()), elapsed, int(batch["pixel_values"].shape[0])


def evaluate_loss(model, eval_loader, device: torch.device, amp_dtype, max_batches: int = 5):
    model.eval()
    losses = []
    with torch.no_grad():
        for idx, batch in enumerate(eval_loader):
            batch = move_batch_to_device(batch, device)
            with autocast_context(device, amp_dtype):
                outputs = model(**batch)
            losses.append(float(outputs.loss.detach().cpu()))
            if idx + 1 >= max_batches:
                break
    model.train()
    return sum(losses) / len(losses) if losses else math.nan


def release_cached_memory() -> None:
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


def cleanup_model(model):
    if model is not None:
        del model
    release_cached_memory()


def benchmark_methods(args, train_loader, device: torch.device):
    results = []
    benchmark_batch = next(iter(train_loader))
    accumulation_steps, effective_global_batch_size = compute_accumulation_steps(args.batch_size, args.global_batch_size)
    for method in BENCHMARK_METHODS:
        LOGGER.info("Benchmarking method=%s", method)
        model = None
        optimizer = None
        scheduler = None
        scaler = None
        try:
            model = build_model(method, args, device)
            optimizer = build_optimizer(method, model, args.learning_rate, args.weight_decay)
            scaler = torch.amp.GradScaler(device.type, enabled=bool(getattr(args, "use_grad_scaler", False)))
            _, total_training_steps, warmup_steps = compute_scheduler_steps(
                len(train_loader.dataset), effective_global_batch_size, args.epochs, args.max_train_steps
            )
            scheduler = build_scheduler(optimizer, total_training_steps, warmup_steps)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            times = []
            images = 0
            for _ in range(args.benchmark_steps):
                loss, elapsed, batch_images = train_step(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    benchmark_batch,
                    device,
                    accumulation_steps=accumulation_steps,
                    step_optimizer=True,
                    amp_dtype=args.amp_dtype,
                    measure_time=True,
                )
                losses.append(loss)
                times.append(elapsed)
                images += batch_images
            result = {
                "method": method,
                "status": "ok",
                "avg_step_time_sec": sum(times) / len(times),
                "images_per_sec": images / sum(times),
                "loss": sum(losses) / len(losses),
                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "local_batch_size": args.batch_size,
                "effective_global_batch_size": effective_global_batch_size,
                "gradient_accumulation_steps": accumulation_steps,
            }
            results.append(result)
            LOGGER.info("Benchmark result %s", result)
        except Exception as exc:
            result = {"method": method, "status": "failed", "error": str(exc)}
            results.append(result)
            LOGGER.exception("Benchmark failed for method=%s", method)
        finally:
            model = None
            optimizer = None
            scheduler = None
            scaler = None
            release_cached_memory()
    successful = [item for item in results if item["status"] == "ok"]
    if not successful:
        raise RuntimeError("All benchmark methods failed.")
    return results, min(successful, key=lambda item: item["avg_step_time_sec"])


def get_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_existing_benchmark_results() -> list[dict]:
    benchmark_path = Path("benchmark_mask_sweep.json")
    if not benchmark_path.exists():
        return []
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return payload.get("results", [])


def checkpoint_root(output_dir: Path) -> Path:
    root = output_dir / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    return root


def latest_checkpoint_file(output_dir: Path) -> Path:
    return checkpoint_root(output_dir) / "latest_checkpoint.json"


def prune_old_checkpoints(root: Path, keep_last_n: int) -> None:
    if keep_last_n <= 0:
        return
    checkpoints = sorted([path for path in root.glob("step_*") if path.is_dir()], key=lambda path: path.name)
    while len(checkpoints) > keep_last_n:
        stale = checkpoints.pop(0)
        shutil.rmtree(stale, ignore_errors=True)


def delete_all_checkpoints(output_dir: Path) -> None:
    root = output_dir / "checkpoints"
    latest_file = root / "latest_checkpoint.json"
    if latest_file.exists():
        latest_file.unlink(missing_ok=True)
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    LOGGER.info("Deleted checkpoint artifacts under %s", root)


def link_or_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def mirror_tree_with_links(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    for path in src.rglob("*"):
        if path.name == "__pycache__" or path.suffix in {".pyc", ".pyo"}:
            continue
        target = dst / path.relative_to(src)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            link_or_copy_file(path, target)


def save_or_link_checkpoint_tokenizer(root: Path, ckpt_dir: Path, tokenizer) -> None:
    checkpoint_tokenizer_dir = ckpt_dir / "tokenizer"
    existing_checkpoints = sorted(
        [path for path in root.glob("step_*") if path.is_dir() and path != ckpt_dir],
        key=lambda path: path.name,
        reverse=True,
    )
    for existing_checkpoint in existing_checkpoints:
        source_dir = existing_checkpoint / "tokenizer"
        if source_dir.exists():
            mirror_tree_with_links(source_dir, checkpoint_tokenizer_dir)
            return
    tokenizer.save_pretrained(checkpoint_tokenizer_dir)


def save_export(output_dir: Path, model, tokenizer, summary: dict, args, benchmark_results: list[dict]) -> None:
    base_model = unwrap_model(model)
    export_state_dict = base_model.state_dict()
    if hasattr(base_model.text_decoder, "merge_and_unload"):
        LOGGER.info("Merging LoRA decoder into a standalone decoder for export.")
        merged_decoder = copy.deepcopy(base_model.text_decoder).cpu().merge_and_unload()
        export_state_dict = OrderedDict(
            (name, tensor.detach().cpu()) for name, tensor in base_model.state_dict().items() if not name.startswith("text_decoder.")
        )
        for name, tensor in merged_decoder.state_dict().items():
            export_state_dict[f"text_decoder.{name}"] = tensor.detach().cpu()
    if "text_decoder.lm_head.weight" in export_state_dict:
        export_state_dict["text_decoder.lm_head.weight"] = export_state_dict["text_decoder.lm_head.weight"].clone()
    base_model.save_pretrained(output_dir / "model", state_dict=export_state_dict)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    segmenter_dir = output_dir / "segmenters"
    segmenter_dir.mkdir(parents=True, exist_ok=True)
    for source, target_name in [
        (args.lung_segmenter_checkpoint, "lung_segmenter_dinounet_finetuned.pth"),
        (args.heart_segmenter_checkpoint, "heart_segmenter_dinounet_best.pth"),
    ]:
        source_path = Path(source)
        if source_path.exists():
            link_or_copy_file(source_path, segmenter_dir / target_name)
    gif_source = Path("assets") / "AnatomicalAttention.gif"
    if gif_source.exists():
        asset_dir = output_dir / "assets"
        asset_dir.mkdir(parents=True, exist_ok=True)
        link_or_copy_file(gif_source, asset_dir / "AnatomicalAttention.gif")
    save_json(output_dir / "benchmark_results.json", {"results": benchmark_results})
    save_json(output_dir / "run_summary.json", summary)


def save_checkpoint(
    output_dir: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    tokenizer,
    sampler: StatefulShuffleSampler,
    summary: dict,
    args,
    benchmark_results: list[dict],
    latest_loss: float,
    reason: str,
) -> Path:
    root = checkpoint_root(output_dir)
    ckpt_dir = root / f"step_{summary['steps']:07d}"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": unwrap_model(model).state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
            "sampler_state": sampler.state_dict(),
            "summary": summary,
            "rng_state": get_rng_state(),
            "args": vars(args),
            "benchmark_results": benchmark_results,
            "latest_loss": latest_loss,
            "reason": reason,
        },
        ckpt_dir / "training_state.pt",
    )
    save_or_link_checkpoint_tokenizer(root, ckpt_dir, tokenizer)
    save_json(
        latest_checkpoint_file(output_dir),
        {"path": str(ckpt_dir.resolve()), "step": summary["steps"], "reason": reason},
    )
    prune_old_checkpoints(root, args.keep_last_n_checkpoints)
    LOGGER.info("Saved checkpoint to %s", ckpt_dir)
    return ckpt_dir


def resolve_resume_path(output_dir: Path, requested: str) -> Path | None:
    if requested:
        return Path(requested)
    latest_file = latest_checkpoint_file(output_dir)
    if latest_file.exists():
        payload = json.loads(latest_file.read_text(encoding="utf-8"))
        return Path(payload["path"])
    return None


def restore_checkpoint(output_dir: Path, args, device: torch.device, sampler: StatefulShuffleSampler, scaler):
    resume_path = resolve_resume_path(output_dir, args.resume_from_checkpoint)
    if resume_path is None:
        return None
    try:
        payload = torch.load(resume_path / "training_state.pt", map_location="cpu", weights_only=False)
        method = payload["summary"]["method"]
        checkpoint_summary = payload["summary"]
        requested_accumulation_steps, requested_effective_global_batch_size = compute_accumulation_steps(
            args.batch_size,
            args.global_batch_size,
        )
        previous_batch_size = int(checkpoint_summary.get("batch_size", args.batch_size))
        previous_effective_global_batch_size = int(
            checkpoint_summary.get("global_batch_size", requested_effective_global_batch_size)
        )
        previous_accumulation_steps = int(
            checkpoint_summary.get("gradient_accumulation_steps", requested_accumulation_steps)
        )
        if (
            previous_batch_size != int(args.batch_size)
            or previous_effective_global_batch_size != int(requested_effective_global_batch_size)
            or previous_accumulation_steps != int(requested_accumulation_steps)
        ):
            raise RuntimeError(
                "Checkpoint optimization config does not match the requested resume config. "
                f"checkpoint batch/global/accum={previous_batch_size}/{previous_effective_global_batch_size}/{previous_accumulation_steps}, "
                f"requested={args.batch_size}/{requested_effective_global_batch_size}/{requested_accumulation_steps}. "
                "Start a fresh run or resume with the same batch settings."
            )
        model = build_model(method, args, device)
        missing, unexpected = unwrap_model(model).load_state_dict(payload["model_state"], strict=False)
        if missing or unexpected:
            LOGGER.warning("Checkpoint loaded with non-strict state dict. Missing=%s Unexpected=%s", missing, unexpected)
        optimizer = build_optimizer(method, model, args.learning_rate, args.weight_decay)
        optimizer.load_state_dict(payload["optimizer_state"])
        effective_global_batch_size = int(payload["summary"].get("global_batch_size", args.global_batch_size or args.batch_size))
        _, total_training_steps, warmup_steps = compute_scheduler_steps(
            sampler.dataset_size,
            effective_global_batch_size,
            args.epochs,
            args.max_train_steps,
        )
        scheduler = build_scheduler(optimizer, total_training_steps, warmup_steps)
        scheduler_state = payload.get("scheduler_state")
        if scheduler_state is None:
            raise RuntimeError("Checkpoint does not contain scheduler state for the current training configuration.")
        scheduler.load_state_dict(scheduler_state)
        scaler_state = payload.get("scaler_state")
        if scaler is not None and scaler.is_enabled() and scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        sampler.load_state_dict(payload["sampler_state"])
        set_rng_state(payload["rng_state"])
        LOGGER.info("Resumed from checkpoint %s", resume_path)
        return {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "summary": payload["summary"],
            "benchmark_results": payload.get("benchmark_results", []),
            "latest_loss": float(payload.get("latest_loss", math.nan)),
        }
    except Exception as exc:
        LOGGER.warning("Could not resume checkpoint %s. Starting a fresh run instead. Reason: %s", resume_path, exc)
        return None


def export_checkpoint_only(args, device: torch.device) -> dict:
    output_dir = Path(args.output_dir)
    resume_path = resolve_resume_path(output_dir, args.resume_from_checkpoint)
    if resume_path is None:
        raise FileNotFoundError("No checkpoint available to export. Expected latest_checkpoint.json or --resume-from-checkpoint.")

    payload = torch.load(resume_path / "training_state.pt", map_location="cpu", weights_only=False)
    summary = dict(payload["summary"])
    benchmark_results = payload.get("benchmark_results", [])
    method = summary["method"]
    model = build_model(method, args, device)
    missing, unexpected = unwrap_model(model).load_state_dict(payload["model_state"], strict=False)
    if missing or unexpected:
        LOGGER.warning("Checkpoint export loaded with non-strict state dict. Missing=%s Unexpected=%s", missing, unexpected)

    summary["repo_id"] = args.repo_id
    save_export(output_dir, model, model.tokenizer, summary, args, benchmark_results)
    save_json(output_dir / "run_summary.json", summary)
    write_model_card(output_dir, summary, benchmark_results, args.repo_id)
    cleanup_model(model)
    LOGGER.info("Exported checkpoint %s into %s", resume_path, output_dir)
    return summary


def timed_training(args, method: str, train_loader, train_sampler, eval_loader, tokenizer, device: torch.device, benchmark_results: list[dict]):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accumulation_steps, effective_global_batch_size = compute_accumulation_steps(args.batch_size, args.global_batch_size)
    steps_per_epoch, total_training_steps, warmup_steps = compute_scheduler_steps(
        train_sampler.dataset_size,
        effective_global_batch_size,
        args.epochs,
        args.max_train_steps,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=bool(getattr(args, "use_grad_scaler", False)))
    restored = restore_checkpoint(output_dir, args, device, train_sampler, scaler)
    if restored is None:
        model = build_model(method, args, device)
        optimizer = build_optimizer(method, model, args.learning_rate, args.weight_decay)
        scheduler = build_scheduler(optimizer, total_training_steps, warmup_steps)
        optimizer.zero_grad(set_to_none=True)
        summary = {
            "method": method,
            "run_name": args.run_name,
            "steps": 0,
            "epochs_completed": 0,
            "epoch_index": 0,
            "target_epochs": args.epochs,
            "progress_epochs": 0.0,
            "training_completion_percent": 0.0,
            "elapsed_seconds": 0.0,
            "images_seen": 0,
            "train_loss_last": math.nan,
            "train_loss_mean": math.nan,
            "val_loss": math.nan,
            "images_per_second": 0.0,
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "vision_model_name": args.vision_model_name,
            "text_model_name": args.text_model_name,
            "repo_id": args.repo_id,
            "model_variant": args.model_variant,
            "visual_projection_type": args.visual_projection_type,
            "attention_bias_mode": args.attention_bias_mode,
            "vision_prefix_tokens_to_skip": args.vision_prefix_tokens_to_skip,
            "segmentation_model_name": args.segmentation_model_name,
            "generation_use_bos_token": args.generation_use_bos_token,
            "generation_stop_on_eos": args.generation_stop_on_eos,
            "generation_repetition_penalty": args.generation_repetition_penalty,
            "lung_segmenter_checkpoint": args.lung_segmenter_checkpoint,
            "heart_segmenter_checkpoint": args.heart_segmenter_checkpoint,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "global_batch_size": effective_global_batch_size,
            "gradient_accumulation_steps": accumulation_steps,
            "steps_per_epoch": steps_per_epoch,
            "planned_total_steps": total_training_steps,
            "scheduler": "cosine",
            "warmup_steps": warmup_steps,
            "warmup_ratio": 0.05,
            "weight_decay": args.weight_decay,
            "precision": getattr(args, "resolved_precision", args.precision),
            "torch_compile": bool(getattr(args, "torch_compile", False)),
            "torch_compile_mode": getattr(args, "torch_compile_mode", "default"),
            "hardware": torch.cuda.get_device_name(device.index or 0) if device.type == "cuda" else str(device),
            "seed": args.seed,
            "resume_supported": True,
            "checkpoint_every_n_steps": args.save_every_n_steps,
            "cumulative_loss_sum": 0.0,
            "cumulative_loss_count": 0,
            "completed": False,
        }
        train_sampler.set_epoch(0, 0)
        latest_loss = math.nan
        benchmark_results_to_save = benchmark_results
    else:
        model = restored["model"]
        optimizer = restored["optimizer"]
        scheduler = restored["scheduler"]
        summary = restored["summary"]
        summary["run_name"] = args.run_name
        summary["repo_id"] = args.repo_id
        summary["target_epochs"] = args.epochs
        summary["image_size"] = args.image_size
        summary["batch_size"] = args.batch_size
        summary["global_batch_size"] = effective_global_batch_size
        summary["gradient_accumulation_steps"] = accumulation_steps
        summary["steps_per_epoch"] = steps_per_epoch
        summary["planned_total_steps"] = total_training_steps
        summary["scheduler"] = "cosine"
        summary["warmup_steps"] = warmup_steps
        summary["warmup_ratio"] = 0.05
        summary["weight_decay"] = args.weight_decay
        summary["precision"] = getattr(args, "resolved_precision", args.precision)
        summary["torch_compile"] = bool(getattr(args, "torch_compile", False))
        summary["torch_compile_mode"] = getattr(args, "torch_compile_mode", "default")
        latest_loss = restored["latest_loss"]
        benchmark_results_to_save = restored["benchmark_results"] or benchmark_results
        optimizer.zero_grad(set_to_none=True)

    start_time = time.perf_counter()
    prior_elapsed = float(summary.get("elapsed_seconds", 0.0))
    running_losses = []
    duration_seconds = parse_duration_to_seconds(args.duration) if args.duration else 0
    if duration_seconds <= 0 and args.time_limit_minutes > 0:
        duration_seconds = args.time_limit_minutes * 60
    time_limit_sec = duration_seconds if duration_seconds > 0 else None
    summary["target_duration_seconds"] = time_limit_sec or 0
    summary["target_duration_mode"] = "per_invocation"

    def update_training_progress() -> None:
        dataset_size = max(int(train_sampler.dataset_size), 1)
        epoch_fraction = train_sampler.position / dataset_size
        progress_epochs = float(summary.get("epoch_index", 0)) + epoch_fraction
        target_epochs = max(int(summary.get("target_epochs", args.epochs)), 1)
        summary["progress_epochs"] = progress_epochs
        summary["training_completion_percent"] = min(100.0, max(0.0, (progress_epochs / target_epochs) * 100.0))

    micro_step_in_update = 0
    pending_step_elapsed = 0.0
    try:
        while summary["epoch_index"] < args.epochs and summary["steps"] < args.max_train_steps:
            if train_sampler.position >= train_sampler.dataset_size:
                summary["epochs_completed"] += 1
                summary["epoch_index"] += 1
                if summary["epoch_index"] >= args.epochs:
                    break
                train_sampler.set_epoch(summary["epoch_index"], 0)

            for batch in train_loader:
                micro_step_in_update += 1
                should_step_optimizer = micro_step_in_update >= accumulation_steps
                loss, elapsed, batch_images = train_step(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    batch,
                    device,
                    accumulation_steps=accumulation_steps,
                    step_optimizer=should_step_optimizer,
                    amp_dtype=args.amp_dtype,
                    measure_time=False,
                )
                summary["images_seen"] += batch_images
                latest_loss = loss
                running_losses.append(loss)
                summary["cumulative_loss_sum"] += loss
                summary["cumulative_loss_count"] += 1
                update_training_progress()
                pending_step_elapsed += elapsed
                current_run_elapsed = time.perf_counter() - start_time
                elapsed_total = prior_elapsed + current_run_elapsed
                if should_step_optimizer:
                    summary["steps"] += 1
                    micro_step_in_update = 0
                    if args.log_every_n_steps > 0 and summary["steps"] % args.log_every_n_steps == 0:
                        LOGGER.info(
                            "method=%s epoch=%s step=%s sampler_pos=%s loss=%.4f step_time=%.3fs",
                            method,
                            summary["epoch_index"] + 1,
                            summary["steps"],
                            train_sampler.position,
                            loss,
                            pending_step_elapsed,
                        )
                    if not args.disable_wandb:
                        current_lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else float("nan")
                        wandb.log(
                            {
                                "train/loss": loss,
                                "train/step": summary["steps"],
                                "train/step_time_sec": pending_step_elapsed,
                                "train/epoch": summary["epoch_index"] + (train_sampler.position / train_sampler.dataset_size),
                                "train/learning_rate": current_lr,
                            }
                        )
                    pending_step_elapsed = 0.0
                if should_step_optimizer and args.save_every_n_steps > 0 and summary["steps"] % args.save_every_n_steps == 0:
                    summary["elapsed_seconds"] = elapsed_total
                    summary["train_loss_last"] = latest_loss
                    if summary["cumulative_loss_count"] > 0:
                        summary["train_loss_mean"] = summary["cumulative_loss_sum"] / summary["cumulative_loss_count"]
                    summary["images_per_second"] = summary["images_seen"] / elapsed_total if elapsed_total > 0 else 0.0
                    save_checkpoint(
                        output_dir,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        tokenizer,
                        train_sampler,
                        summary,
                        args,
                        benchmark_results_to_save,
                        latest_loss,
                        "periodic",
                    )
                if time_limit_sec is not None and current_run_elapsed >= time_limit_sec:
                    LOGGER.info(
                        "Stopping run because requested duration for this invocation was reached at %.2f seconds (cumulative %.2f seconds)",
                        current_run_elapsed,
                        elapsed_total,
                    )
                    break
                if should_step_optimizer and summary["steps"] >= args.max_train_steps:
                    break

            if train_sampler.position >= train_sampler.dataset_size:
                summary["epochs_completed"] += 1
                summary["epoch_index"] += 1
                if summary["epoch_index"] < args.epochs:
                    train_sampler.set_epoch(summary["epoch_index"], 0)

            current_run_elapsed = time.perf_counter() - start_time
            elapsed_total = prior_elapsed + current_run_elapsed
            if time_limit_sec is not None and current_run_elapsed >= time_limit_sec:
                break
            if summary["steps"] >= args.max_train_steps:
                break
    except KeyboardInterrupt:
        update_training_progress()
        summary["elapsed_seconds"] = prior_elapsed + (time.perf_counter() - start_time)
        summary["train_loss_last"] = latest_loss
        if summary["cumulative_loss_count"] > 0:
            summary["train_loss_mean"] = summary["cumulative_loss_sum"] / summary["cumulative_loss_count"]
        summary["images_per_second"] = summary["images_seen"] / summary["elapsed_seconds"] if summary["elapsed_seconds"] > 0 else 0.0
        save_checkpoint(
            output_dir,
            model,
            optimizer,
            scheduler,
            scaler,
            tokenizer,
            train_sampler,
            summary,
            args,
            benchmark_results_to_save,
            latest_loss,
            "interrupt",
        )
        save_export(output_dir, model, tokenizer, summary, args, benchmark_results_to_save)
        raise

    update_training_progress()
    summary["elapsed_seconds"] = prior_elapsed + (time.perf_counter() - start_time)
    summary["train_loss_last"] = latest_loss
    if summary["cumulative_loss_count"] > 0:
        summary["train_loss_mean"] = summary["cumulative_loss_sum"] / summary["cumulative_loss_count"]
    summary["images_per_second"] = summary["images_seen"] / summary["elapsed_seconds"] if summary["elapsed_seconds"] > 0 else 0.0
    summary["completed"] = summary["epoch_index"] >= args.epochs
    summary["val_loss"] = evaluate_loss(model, eval_loader, device, args.amp_dtype)
    save_export(output_dir, model, tokenizer, summary, args, benchmark_results_to_save)
    save_checkpoint(
        output_dir,
        model,
        optimizer,
        scheduler,
        scaler,
        tokenizer,
        train_sampler,
        summary,
        args,
        benchmark_results_to_save,
        latest_loss,
        "final",
    )
    if summary["completed"]:
        delete_all_checkpoints(output_dir)
    return model, summary, benchmark_results_to_save


def write_model_card(output_dir: Path, summary: dict, benchmark_results: list[dict], repo_id: str):
    total_hours = summary["elapsed_seconds"] / 3600 if summary["elapsed_seconds"] else 0.0
    target_epochs = max(int(summary.get("target_epochs", 0) or 0), 1)
    progress_epochs = float(summary.get("progress_epochs", summary.get("epoch_index", 0)))
    completion_percent = float(summary.get("training_completion_percent", (progress_epochs / target_epochs) * 100.0))
    repo_id = str(repo_id or summary.get("repo_id") or "manu02/LAnA")
    display_epochs = progress_epochs
    if bool(summary.get("completed")):
        epochs_completed = int(summary.get("epochs_completed", target_epochs) or target_epochs)
        display_epochs = float(min(max(epochs_completed, 0), target_epochs))
    benchmark_lines = []
    for item in benchmark_results:
        if item["status"] == "ok":
            benchmark_lines.append(
                f"| `{item['method']}` | {item['local_batch_size']} | {item['effective_global_batch_size']} | "
                f"{item['gradient_accumulation_steps']} | {item['optimizer_step_time_sec']:.4f} | {item['images_per_sec']:.4f} | ok |"
            )
        else:
            benchmark_lines.append(
                f"| `{item['method']}` | {item.get('local_batch_size', '-')} | {item.get('global_batch_size_requested', '-')} | "
                f"- | - | - | failed: {item['error']} |"
            )

    def best_collection_repo_id() -> str:
        repo_root = Path(__file__).resolve().parents[1]
        candidates = [
            ("manu02/LAnA-MIMIC-CHEXPERT", repo_root / "artifacts" / "full_3_epoch_mask_run" / "run_summary.json"),
            ("manu02/LAnA-MIMIC", repo_root / "artifacts" / "LAnA-MIMIC-TERM" / "run_summary.json"),
            ("manu02/LAnA", repo_root / "artifacts" / "LAnA-paper" / "run_summary.json"),
            ("manu02/LAnA-v2", repo_root / "artifacts" / "LAnA-v2" / "run_summary.json"),
            ("manu02/LAnA-v3", repo_root / "artifacts" / "LAnA-v3" / "run_summary.json"),
            ("manu02/LAnA-v4", repo_root / "artifacts" / "LAnA-v4" / "run_summary.json"),
            ("manu02/LAnA-v5", repo_root / "artifacts" / "LAnA-v5" / "run_summary.json"),
        ]
        best_repo = repo_id
        best_pair = (float("-inf"), float("-inf"))
        for candidate_repo_id, summary_path in candidates:
            if not summary_path.exists():
                continue
            try:
                candidate_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            metrics = candidate_summary.get("latest_evaluations", {}).get("all_test") or candidate_summary.get("latest_evaluation") or {}
            score = metrics.get("chexpert_f1_14_micro")
            rouge = metrics.get("rouge_l")
            pair = (
                float(score) if score is not None else float("-inf"),
                float(rouge) if rouge is not None else float("-inf"),
            )
            if pair > best_pair:
                best_pair = pair
                best_repo = candidate_repo_id
        return best_repo

    lines = [
        "---",
        "license: mit",
        "library_name: transformers",
        "pipeline_tag: image-to-text",
        "tags:",
        "  - medical-ai",
        "  - radiology",
        "  - chest-xray",
        "  - report-generation",
        "  - segmentation",
        "  - anatomical-attention",
        "metrics:",
        "  - BLEU",
        "  - METEOR",
        "  - ROUGE",
        "  - CIDEr",
        "---",
        "",
        "# LAnA",
        "",
        "**Layer-Wise Anatomical Attention model**",
        "",
        build_best_model_notice(best_collection_repo_id()),
        "",
        "[![ArXiv](https://img.shields.io/badge/ArXiv-2512.16841-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2512.16841)",
        "[![LinkedIn](https://img.shields.io/badge/LinkedIn-devmuniz-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/devmuniz)",
        "[![GitHub Profile](https://img.shields.io/badge/GitHub-devMuniz02-181717?logo=github&logoColor=white)](https://github.com/devMuniz02)",
        "[![Portfolio](https://img.shields.io/badge/Portfolio-devmuniz02.github.io-0F172A?logo=googlechrome&logoColor=white)](https://devmuniz02.github.io/)",
        "[![GitHub Repo](https://img.shields.io/badge/Repository-layer--wise--anatomical--attention-181717?logo=github&logoColor=white)](https://github.com/devMuniz02/layer-wise-anatomical-attention)",
        "[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-manu02-FFD21E?logoColor=black)](https://huggingface.co/manu02)",
        "",
        "![Layer-Wise Anatomical Attention](assets/AnatomicalAttention.gif)",
        "",
        "## Overview",
        "",
        "LAnA is a medical report-generation project for chest X-ray images. The completed project is intended to generate radiology reports with a vision-language model guided by layer-wise anatomical attention built from predicted anatomical masks.",
        "",
        "The architecture combines a DINOv3 vision encoder, lung and heart segmentation heads, and a GPT-2 decoder modified so each transformer layer receives a different anatomical attention bias derived from the segmentation mask.",
        "",
        "## Intended Use",
        "",
        "- Input: a chest X-ray image resized to `512x512` and normalized with ImageNet mean/std.",
        "- Output: a generated radiology report.",
        "- Best fit: research use, report-generation experiments, and anatomical-attention ablations.",
        "",
        *(build_main_branch_usage_section(repo_id).splitlines() if repo_id in SPLIT_INFERENCE_REPOS else build_dual_usage_section(repo_id).splitlines()),
        "",
        *DINO_V3_NOTICE.splitlines(),
        "",
        *RESEARCH_USE_NOTICE.splitlines(),
        "",
        "## MIMIC Test Results",
        "",
        "Frontal-only evaluation using `PA/AP` studies only.",
        "",
        "### Current Checkpoint Results",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        "| Number of studies | TBD |",
        "| RadGraph F1 | TBD |",
        "| RadGraph entity F1 | TBD |",
        "| RadGraph relation F1 | TBD |",
        "| CheXpert F1 14-micro | TBD |",
        "| CheXpert F1 5-micro | TBD |",
        "| CheXpert F1 14-macro | TBD |",
        "| CheXpert F1 5-macro | TBD |",
        "",
        "### Final Completed Training Results",
        "",
        "The final table will be populated when the planned training run is completed. Until then, final-report metrics remain `TBD`.",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        "| Number of studies | TBD |",
        "| RadGraph F1 | TBD |",
        "| RadGraph entity F1 | TBD |",
        "| RadGraph relation F1 | TBD |",
        "| CheXpert F1 14-micro | TBD |",
        "| CheXpert F1 5-micro | TBD |",
        "| CheXpert F1 14-macro | TBD |",
        "| CheXpert F1 5-macro | TBD |",
        "",
        "",
        "## Data",
        "",
        "- Full project datasets: CheXpert and MIMIC-CXR.",
        "- Intended project scope: train on curated chest X-ray/report data from both datasets and evaluate on MIMIC-CXR test studies.",
        f"- Current released checkpoint datasets: `{summary.get('train_datasets', 'unknown')}` for training and `{summary.get('validation_datasets', 'unknown')}` for validation.",
        "- Current published evaluation: MIMIC-CXR test split, `frontal-only (PA/AP)` studies.",
        "",
        "## Evaluation",
        "",
        "- Medical report metrics implemented in the repository include RadGraph F1 and CheXpert F1 (`14-micro`, `5-micro`, `14-macro`, `5-macro`).",
        "",
        "## Training Snapshot",
        "",
        f"- Run: `{summary.get('run_name', 'unknown')}`",
        "- This section describes the current public checkpoint, not the final completed project.",
        f"- Method: `{summary['method']}`",
        f"- Vision encoder: `{summary['vision_model_name']}`",
        f"- Text decoder: `{summary['text_model_name']}`",
        f"- Visual projection: `{summary.get('visual_projection_type', 'mlp4')}`",
        f"- Segmentation encoder: `{summary.get('segmentation_model_name', 'unknown')}`",
        f"- Image size: `{summary['image_size']}`",
        f"- Local batch size: `{summary['batch_size']}`",
        f"- Effective global batch size: `{summary['global_batch_size']}`",
        f"- Scheduler: `{summary.get('scheduler', 'cosine')}`",
        f"- Warmup steps: `{summary.get('warmup_steps', 'unknown')}`",
        f"- Weight decay: `{summary.get('weight_decay', 'unknown')}`",
        f"- Steps completed: `{summary['steps']}`",
        f"- Planned total steps: `{summary.get('planned_total_steps', 'unknown')}`",
        f"- Images seen: `{summary['images_seen']}`",
        f"- Total training time: `{total_hours:.4f}` hours",
        f"- Hardware: `{summary.get('hardware', 'unknown')}`",
        f"- Final train loss: `{summary['train_loss_last']:.4f}`",
        f"- Validation loss: `{summary['val_loss']:.4f}`",
        "",
        "## Status",
        "",
        "- Project status: `Training in progress`",
        "- Release status: `Research preview checkpoint`",
        "- Current checkpoint status: `Not final`",
        f"- Training completion toward planned run: `{completion_percent:.2f}%` (`{display_epochs:.0f}` / `{target_epochs}` epochs)",
        "- Current published metrics are intermediate and will change as training continues.",
        "",
        "## Notes",
        "",
        "- Set `HF_TOKEN` with permission to access the DINOv3 repositories required by this model before downloading or running inference.",
        "- `segmenters/` contains the lung and heart segmentation checkpoints used to build anatomical attention masks.",
        "- `evaluations/mimic_test_metrics.json` contains the latest saved MIMIC test metrics.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    apply_model_variant_presets(args)
    configure_logging(args.log_level)
    load_env_file()
    set_global_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable in the current environment.")
    configure_runtime(device)
    precision = resolve_training_precision(device, args.precision)
    args.resolved_precision = str(precision["resolved"])
    args.amp_dtype = precision["amp_dtype"]
    args.amp_dtype_name = str(args.amp_dtype).replace("torch.", "") if args.amp_dtype is not None else "none"
    args.use_grad_scaler = bool(precision["use_grad_scaler"])
    args.decoder_compute_dtype = str(precision["decoder_compute_dtype"])
    LOGGER.info("Training precision requested=%s resolved=%s", args.precision, args.resolved_precision)
    release_cached_memory()

    use_wandb = not args.disable_wandb and not args.export_only
    try:
        if use_wandb:
            if not os.environ.get("WANDB_API_KEY"):
                raise RuntimeError("W&B logging was requested but WANDB_API_KEY was not found in the environment or .env file.")
            wandb_config = vars(args).copy()
            wandb_config["amp_dtype"] = args.amp_dtype_name
            wandb.init(project=args.wandb_project, name=args.run_name, config=wandb_config)

        tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token or PAD_TOKEN

        if args.export_only:
            summary = export_checkpoint_only(args, device)
            if args.push_to_hub:
                if args.repo_id in SPLIT_INFERENCE_REPOS:
                    repo_url = push_split_inference_and_snapshot_layout(
                        str(output_dir),
                        args.repo_id,
                        commit_message="Upload exported LANA checkpoint artifact",
                    )
                else:
                    repo_url = push_directory_to_hub(str(output_dir), args.repo_id, commit_message="Upload exported LANA checkpoint artifact")
                LOGGER.info("Uploaded exported checkpoint to %s", repo_url)
                summary["repo_url"] = repo_url
                save_json(output_dir / "run_summary.json", summary)
            return

        train_parts = []
        valid_parts = []
        train_dataset_names = []
        valid_dataset_names = []
        if args.dataset in {"chexpert", "combined"}:
            train_parts.append(build_chexpert_manifest(args.metadata_path, args.image_root, split="train"))
            valid_parts.append(build_chexpert_manifest(args.metadata_path, args.image_root, split="valid"))
            train_dataset_names.append("CheXpert")
            valid_dataset_names.append("CheXpert")
        if args.dataset in {"mimic", "combined"}:
            train_parts.append(build_mimic_manifest(args.mimic_root, split="train", findings_only=args.mimic_findings_only))
            valid_parts.append(build_mimic_manifest(args.mimic_root, split="valid", findings_only=args.mimic_findings_only))
            mimic_label = "MIMIC-CXR (findings-only)" if args.mimic_findings_only else "MIMIC-CXR"
            train_dataset_names.append(mimic_label)
            valid_dataset_names.append(mimic_label)

        train_manifest = combine_manifests(train_parts)
        valid_manifest = combine_manifests(valid_parts)
        skip_cached_resize = should_skip_cached_resize(args)
        prepend_bos_token = bool(getattr(args, "generation_use_bos_token", True))
        _, train_loader, train_sampler = make_dataloader(
            train_manifest,
            tokenizer,
            args.batch_size,
            args.image_size,
            args.num_workers,
            True,
            args.seed,
            skip_cached_resize=skip_cached_resize,
            prepend_bos_token=prepend_bos_token,
        )
        _, eval_loader, _ = make_dataloader(
            valid_manifest,
            tokenizer,
            args.eval_batch_size,
            args.image_size,
            args.num_workers,
            False,
            args.seed,
            skip_cached_resize=skip_cached_resize,
            prepend_bos_token=prepend_bos_token,
        )

        LOGGER.info(
            "Train rows: %s | Valid rows: %s | Train datasets: %s | Valid datasets: %s",
            len(train_manifest),
            len(valid_manifest),
            ", ".join(train_dataset_names),
            ", ".join(valid_dataset_names),
        )
        LOGGER.info("Device: %s", device)
        LOGGER.info("Runtime cached resize enabled=%s", not skip_cached_resize)

        if args.method == "auto":
            benchmark_results, fastest = benchmark_methods(args, train_loader, device)
            selected_method = fastest["method"]
            if train_sampler is not None:
                train_sampler.set_epoch(0, 0)
            LOGGER.info("Selected fastest method=%s", selected_method)
        else:
            benchmark_results = load_existing_benchmark_results()
            selected_method = args.method

        model = None
        try:
            model, summary, benchmark_results = timed_training(
                args, selected_method, train_loader, train_sampler, eval_loader, tokenizer, device, benchmark_results
            )
        except KeyboardInterrupt:
            summary_path = output_dir / "run_summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary["train_datasets"] = ", ".join(train_dataset_names)
                summary["validation_datasets"] = ", ".join(valid_dataset_names)
                write_model_card(output_dir, summary, benchmark_results, args.repo_id)
                if args.push_to_hub:
                    if args.repo_id in SPLIT_INFERENCE_REPOS:
                        repo_url = push_split_inference_and_snapshot_layout(
                            str(output_dir),
                            args.repo_id,
                            commit_message="Upload interrupted LANA model checkpoint export",
                        )
                    else:
                        repo_url = push_directory_to_hub(
                            str(output_dir),
                            args.repo_id,
                            commit_message="Upload interrupted LANA model checkpoint export",
                        )
                    LOGGER.info("Uploaded interrupted run artifacts to %s", repo_url)
                    summary["repo_url"] = repo_url
                    save_json(summary_path, summary)
            LOGGER.warning("Training interrupted. Latest checkpoint saved and ready to resume.")
            return

        summary["train_datasets"] = ", ".join(train_dataset_names)
        summary["validation_datasets"] = ", ".join(valid_dataset_names)
        save_json(output_dir / "run_summary.json", summary)
        write_model_card(output_dir, summary, benchmark_results, args.repo_id)
        if args.push_to_hub:
            if args.repo_id in SPLIT_INFERENCE_REPOS:
                repo_url = push_split_inference_and_snapshot_layout(
                    str(output_dir),
                    args.repo_id,
                    commit_message="Upload benchmarked LANA model",
                )
            else:
                repo_url = push_directory_to_hub(str(output_dir), args.repo_id, commit_message="Upload benchmarked LANA model")
            LOGGER.info("Uploaded model to %s", repo_url)
            summary["repo_url"] = repo_url
            save_json(output_dir / "run_summary.json", summary)
    finally:
        if use_wandb:
            wandb.finish()
        release_cached_memory()


if __name__ == "__main__":
    main()
