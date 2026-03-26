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
from lana_radgen.hub import push_directory_to_hub
from lana_radgen.logging_utils import configure_logging

LOGGER = logging.getLogger("train")

BENCHMARK_METHODS = ["qlora_paged_adamw8bit", "lora_adamw", "full_adam", "full_adam8bit"]
METHOD_CHOICES = BENCHMARK_METHODS + ["full_adamw", "lora_adamw8bit"]


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
    parser.add_argument("--vision-model-name", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--text-model-name", default="gpt2")
    parser.add_argument("--wandb-project", default="lana-radgen")
    parser.add_argument("--run-name", default="benchmark-train")
    parser.add_argument("--dataset", default="combined", choices=["chexpert", "mimic", "combined"])
    parser.add_argument("--metadata-path", default="Datasets/CheXpert/df_chexpert_plus_240401_findings.csv")
    parser.add_argument("--image-root", default="Datasets/CheXpert/images")
    parser.add_argument("--mimic-root", default="Datasets/MIMIC")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--global-batch-size", type=int, default=0, help="Effective batch size via gradient accumulation. 0 uses local batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--segmentation-model-name", default="facebook/dinov3-convnext-small-pretrain-lvd1689m")
    parser.add_argument("--lung-segmenter-checkpoint", default="models/lung_segmenter_dinounet_finetuned.pth")
    parser.add_argument("--heart-segmenter-checkpoint", default="models/heart_segmenter_dinounet_best.pth")
    parser.add_argument("--disable-segmentation-mask", action="store_true")
    parser.add_argument("--device", default=default_device())
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
    parser.add_argument("--save-every-n-steps", type=int, default=1000)
    parser.add_argument("--keep-last-n-checkpoints", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _parse_findings_section(report_text: str) -> str:
    normalized = report_text.replace("\r\n", "\n")
    match = re.search(r"FINDINGS:\s*(.*?)(?:\n\s*[A-Z ]+:\s|$)", normalized, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()
    impression_match = re.search(r"IMPRESSION:\s*(.*?)(?:\n\s*[A-Z ]+:\s|$)", normalized, flags=re.IGNORECASE | re.DOTALL)
    if impression_match:
        return re.sub(r"\s+", " ", impression_match.group(1)).strip()
    return re.sub(r"\s+", " ", normalized).strip()


def _load_report_texts(report_zip_path: Path) -> dict[tuple[int, int], str]:
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
            reports[(subject_id, study_id)] = _parse_findings_section(text)
    return reports


def _resolve_mimic_processed_image_path(subject_id: int, study_id: int, dicom_id: str, image_root: Path) -> Path:
    return image_root / f"p{subject_id}" / f"s{study_id}" / f"{dicom_id}.png"


def build_mimic_manifest(mimic_root: str, split: str) -> pd.DataFrame:
    root = Path(mimic_root)
    split_df = pd.read_csv(root / "mimic-cxr-2.0.0-split.csv.gz", compression="gzip")
    records_df = pd.read_csv(root / "cxr-record-list.csv.gz", compression="gzip")
    metadata_df = pd.read_csv(root / "mimic-cxr-2.0.0-metadata.csv")
    reports = _load_report_texts(root / "mimic-cxr-reports.zip")

    split_name = "validate" if split == "valid" else split
    df = split_df[split_df["split"] == split_name].copy()
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


def make_dataloader(manifest, tokenizer, batch_size: int, image_size: int, num_workers: int, shuffle: bool, seed: int):
    max_text_length = getattr(tokenizer, "model_max_length", None)
    if isinstance(max_text_length, int) and max_text_length > 100000:
        max_text_length = 1024
    dataset = ResizeCachedReportDataset(
        manifest=manifest,
        tokenizer=tokenizer,
        image_size=image_size,
        max_text_length=max_text_length,
    )
    sampler = StatefulShuffleSampler(len(dataset), seed=seed) if shuffle else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
    )
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
        segmentation_model_name=args.segmentation_model_name,
        lung_segmenter_checkpoint=args.lung_segmenter_checkpoint,
        heart_segmenter_checkpoint=args.heart_segmenter_checkpoint,
        use_segmentation_mask=not args.disable_segmentation_mask,
        decoder_load_in_4bit=(method == "qlora_paged_adamw8bit"),
    )
    model = LanaForConditionalGeneration(config)
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
    return model


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
    return {key: value.to(device) for key, value in batch.items()}


def train_step(model, optimizer, scheduler, batch, device: torch.device, accumulation_steps: int, step_optimizer: bool):
    batch = move_batch_to_device(batch, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = model(**batch)
    scaled_loss = outputs.loss / accumulation_steps
    scaled_loss.backward()
    if step_optimizer:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return float(outputs.loss.detach().cpu()), elapsed, int(batch["pixel_values"].shape[0])


def evaluate_loss(model, eval_loader, device: torch.device, max_batches: int = 5):
    model.eval()
    losses = []
    with torch.no_grad():
        for idx, batch in enumerate(eval_loader):
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            losses.append(float(outputs.loss.detach().cpu()))
            if idx + 1 >= max_batches:
                break
    model.train()
    return sum(losses) / len(losses) if losses else math.nan


def cleanup_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def benchmark_methods(args, train_loader, device: torch.device):
    results = []
    benchmark_batch = next(iter(train_loader))
    accumulation_steps, effective_global_batch_size = compute_accumulation_steps(args.batch_size, args.global_batch_size)
    for method in BENCHMARK_METHODS:
        LOGGER.info("Benchmarking method=%s", method)
        try:
            model = build_model(method, args, device)
            optimizer = build_optimizer(method, model, args.learning_rate, args.weight_decay)
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
                    model, optimizer, scheduler, benchmark_batch, device, accumulation_steps=accumulation_steps, step_optimizer=True
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
            if "model" in locals():
                cleanup_model(model)
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


def save_export(output_dir: Path, model, tokenizer, summary: dict, args, benchmark_results: list[dict]) -> None:
    export_state_dict = model.state_dict()
    if hasattr(model.text_decoder, "merge_and_unload"):
        LOGGER.info("Merging LoRA decoder into a standalone decoder for export.")
        merged_decoder = copy.deepcopy(model.text_decoder).cpu().merge_and_unload()
        export_state_dict = OrderedDict(
            (name, tensor.detach().cpu()) for name, tensor in model.state_dict().items() if not name.startswith("text_decoder.")
        )
        for name, tensor in merged_decoder.state_dict().items():
            export_state_dict[f"text_decoder.{name}"] = tensor.detach().cpu()
    if "text_decoder.lm_head.weight" in export_state_dict:
        export_state_dict["text_decoder.lm_head.weight"] = export_state_dict["text_decoder.lm_head.weight"].clone()
    model.save_pretrained(output_dir / "model", state_dict=export_state_dict)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    segmenter_dir = output_dir / "segmenters"
    segmenter_dir.mkdir(parents=True, exist_ok=True)
    for source, target_name in [
        (args.lung_segmenter_checkpoint, "lung_segmenter_dinounet_finetuned.pth"),
        (args.heart_segmenter_checkpoint, "heart_segmenter_dinounet_best.pth"),
    ]:
        source_path = Path(source)
        if source_path.exists():
            shutil.copy2(source_path, segmenter_dir / target_name)
    gif_source = Path("assets") / "AnatomicalAttention.gif"
    if gif_source.exists():
        asset_dir = output_dir / "assets"
        asset_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(gif_source, asset_dir / "AnatomicalAttention.gif")
    save_json(output_dir / "benchmark_results.json", {"results": benchmark_results})
    save_json(output_dir / "run_summary.json", summary)


def save_checkpoint(
    output_dir: Path,
    model,
    optimizer,
    scheduler,
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
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
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
    tokenizer.save_pretrained(ckpt_dir / "tokenizer")
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


def restore_checkpoint(output_dir: Path, args, device: torch.device, sampler: StatefulShuffleSampler):
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
        missing, unexpected = model.load_state_dict(payload["model_state"], strict=False)
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
    restored = restore_checkpoint(output_dir, args, device, train_sampler)
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
            "segmentation_model_name": args.segmentation_model_name,
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
                    batch,
                    device,
                    accumulation_steps=accumulation_steps,
                    step_optimizer=should_step_optimizer,
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
                        wandb.log(
                            {
                                "train/loss": loss,
                                "train/step": summary["steps"],
                                "train/step_time_sec": pending_step_elapsed,
                                "train/epoch": summary["epoch_index"] + (train_sampler.position / train_sampler.dataset_size),
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
    summary["val_loss"] = evaluate_loss(model, eval_loader, device)
    save_export(output_dir, model, tokenizer, summary, args, benchmark_results_to_save)
    save_checkpoint(
        output_dir,
        model,
        optimizer,
        scheduler,
        tokenizer,
        train_sampler,
        summary,
        args,
        benchmark_results_to_save,
        latest_loss,
        "final",
    )
    return model, summary, benchmark_results_to_save


def write_model_card(output_dir: Path, summary: dict, benchmark_results: list[dict], repo_id: str):
    total_hours = summary["elapsed_seconds"] / 3600 if summary["elapsed_seconds"] else 0.0
    target_epochs = max(int(summary.get("target_epochs", 0) or 0), 1)
    progress_epochs = float(summary.get("progress_epochs", summary.get("epoch_index", 0)))
    completion_percent = float(summary.get("training_completion_percent", (progress_epochs / target_epochs) * 100.0))
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
        "[![ArXiv](https://img.shields.io/badge/ArXiv-2512.16841-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2512.16841)",
        "[![LinkedIn](https://img.shields.io/badge/LinkedIn-devmuniz-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/devmuniz)",
        "[![GitHub Profile](https://img.shields.io/badge/GitHub-devMuniz02-181717?logo=github&logoColor=white)](https://github.com/devMuniz02)",
        "[![Portfolio](https://img.shields.io/badge/Portfolio-devmuniz02.github.io-0F172A?logo=googlechrome&logoColor=white)](https://devmuniz02.github.io/)",
        "[![GitHub Repo](https://img.shields.io/badge/Repository-layer--wise--anatomical--attention-181717?logo=github&logoColor=white)](https://github.com/devMuniz02/layer-wise-anatomical-attention)",
        "[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-manu02-FFD21E?logoColor=black)](https://huggingface.co/manu02)",
        "",
        "![Layer-Wise Anatomical Attention](assets/AnatomicalAttention.gif)",
        "",
        "## Status",
        "",
        "- Project status: `Training in progress`",
        "- Release status: `Research preview checkpoint`",
        "- Current checkpoint status: `Not final`",
        f"- Training completion toward planned run: `{completion_percent:.2f}%` (`{display_epochs:.0f}` / `{target_epochs}` epochs)",
        "- Current published metrics are intermediate and will change as training continues.",
        "",
        "## Overview",
        "",
        "LAnA is a medical report-generation project for chest X-ray images. The completed project is intended to generate radiology reports with a vision-language model guided by layer-wise anatomical attention built from predicted anatomical masks.",
        "",
        "The architecture combines a DINOv3 vision encoder, lung and heart segmentation heads, and a GPT-2 decoder modified so each transformer layer receives a different anatomical attention bias derived from the segmentation mask.",
        "",
        "## How to Run",
        "",
        "Use the local inference flow below to run the model from the exported checkpoint.",
        "",
        "### Inference",
        "",
        "Standard `AutoModel.from_pretrained(..., trust_remote_code=True)` loading is currently blocked for this repo because the custom model constructor performs nested pretrained submodel loads.",
        "Use the verified manual load path below instead: download the HF repo snapshot, import the downloaded package, and load the exported `model.safetensors` directly.",
        "",
        "```python",
        "from pathlib import Path",
        "import sys",
        "",
        "import numpy as np",
        "import torch",
        "from PIL import Image",
        "from huggingface_hub import snapshot_download",
        "from safetensors.torch import load_file",
        "from transformers import AutoTokenizer",
        "",
        "repo_dir = Path(snapshot_download(\"manu02/LAnA\"))",
        "sys.path.insert(0, str(repo_dir))",
        "",
        "from lana_radgen import LanaConfig, LanaForConditionalGeneration",
        "",
        "config = LanaConfig.from_pretrained(repo_dir)",
        "config.lung_segmenter_checkpoint = str(repo_dir / \"segmenters\" / \"lung_segmenter_dinounet_finetuned.pth\")",
        "config.heart_segmenter_checkpoint = str(repo_dir / \"segmenters\" / \"heart_segmenter_dinounet_best.pth\")",
        "",
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")",
        "",
        "model = LanaForConditionalGeneration(config)",
        "state_dict = load_file(str(repo_dir / \"model.safetensors\"))",
        "missing, unexpected = model.load_state_dict(state_dict, strict=True)",
        "assert not missing and not unexpected",
        "",
        "model.tokenizer = AutoTokenizer.from_pretrained(repo_dir, trust_remote_code=True)",
        "model.move_non_quantized_modules(device)",
        "model.eval()",
        "",
        "image_path = Path(\"example.png\")",
        "image = Image.open(image_path).convert(\"RGB\")",
        "image = image.resize((512, 512), resample=Image.BICUBIC)",
        "array = np.asarray(image, dtype=np.float32) / 255.0",
        "pixel_values = torch.from_numpy(array).permute(2, 0, 1)",
        "mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)",
        "std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)",
        "pixel_values = ((pixel_values - mean) / std).unsqueeze(0).to(device)",
        "",
        "with torch.no_grad():",
        "    generated = model.generate(pixel_values=pixel_values, max_new_tokens=128)",
        "",
        "report = model.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]",
        "print(report)",
        "```",
        "",
        "## Intended Use",
        "",
        "- Input: a chest X-ray image resized to `512x512` and normalized with ImageNet mean/std.",
        "- Output: a generated radiology report.",
        "- Best fit: research use, report-generation experiments, and anatomical-attention ablations.",
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
        "## Notes",
        "",
        "- `segmenters/` contains the lung and heart segmentation checkpoints used to build anatomical attention masks.",
        "- `evaluations/mimic_test_metrics.json` contains the latest saved MIMIC test metrics.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    load_env_file()
    set_global_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable in the current environment.")

    use_wandb = not args.disable_wandb
    if use_wandb:
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError("W&B logging was requested but WANDB_API_KEY was not found in the environment or .env file.")
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        train_parts.append(build_mimic_manifest(args.mimic_root, split="train"))
        valid_parts.append(build_mimic_manifest(args.mimic_root, split="valid"))
        train_dataset_names.append("MIMIC-CXR")
        valid_dataset_names.append("MIMIC-CXR")

    train_manifest = combine_manifests(train_parts)
    valid_manifest = combine_manifests(valid_parts)
    _, train_loader, train_sampler = make_dataloader(train_manifest, tokenizer, args.batch_size, args.image_size, args.num_workers, True, args.seed)
    _, eval_loader, _ = make_dataloader(valid_manifest, tokenizer, args.eval_batch_size, args.image_size, args.num_workers, False, args.seed)

    LOGGER.info(
        "Train rows: %s | Valid rows: %s | Train datasets: %s | Valid datasets: %s",
        len(train_manifest),
        len(valid_manifest),
        ", ".join(train_dataset_names),
        ", ".join(valid_dataset_names),
    )
    LOGGER.info("Device: %s", device)

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
                repo_url = push_directory_to_hub(
                    str(output_dir),
                    args.repo_id,
                    commit_message="Upload interrupted LANA model checkpoint export",
                )
                LOGGER.info("Uploaded interrupted run artifacts to %s", repo_url)
                summary["repo_url"] = repo_url
                save_json(summary_path, summary)
        LOGGER.warning("Training interrupted. Latest checkpoint saved and ready to resume.")
        if use_wandb:
            wandb.finish()
        return

    summary["train_datasets"] = ", ".join(train_dataset_names)
    summary["validation_datasets"] = ", ".join(valid_dataset_names)
    save_json(output_dir / "run_summary.json", summary)
    write_model_card(output_dir, summary, benchmark_results, args.repo_id)
    if args.push_to_hub:
        repo_url = push_directory_to_hub(str(output_dir), args.repo_id, commit_message="Upload benchmarked LANA model")
        LOGGER.info("Uploaded model to %s", repo_url)
        summary["repo_url"] = repo_url
        save_json(output_dir / "run_summary.json", summary)

    if model is not None:
        cleanup_model(model)
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
