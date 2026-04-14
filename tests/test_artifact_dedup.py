import os
from pathlib import Path

from lana_radgen.hub import _build_hf_package
from scripts.train import save_or_link_checkpoint_tokenizer


def _same_file_identity(left: Path, right: Path) -> bool:
    left_stat = left.stat()
    right_stat = right.stat()
    return (left_stat.st_dev, left_stat.st_ino) == (right_stat.st_dev, right_stat.st_ino)


def test_build_hf_package_hardlinks_large_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "model").mkdir(parents=True)
    (run_dir / "tokenizer").mkdir()
    (run_dir / "segmenters").mkdir()
    (run_dir / "assets").mkdir()
    (run_dir / "evaluations").mkdir()

    (run_dir / "README.md").write_text("readme", encoding="utf-8")
    (run_dir / "run_summary.json").write_text("{}", encoding="utf-8")
    (run_dir / "benchmark_results.json").write_text("{}", encoding="utf-8")
    (run_dir / "pipeline_autotune.json").write_text("{}", encoding="utf-8")
    (run_dir / "model" / "config.json").write_text("{}", encoding="utf-8")
    (run_dir / "model" / "model.safetensors").write_bytes(b"weights")
    (run_dir / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (run_dir / "segmenters" / "lung_segmenter_dinounet_finetuned.pth").write_bytes(b"lung")
    (run_dir / "segmenters" / "heart_segmenter_dinounet_best.pth").write_bytes(b"heart")
    (run_dir / "assets" / "AnatomicalAttention.gif").write_bytes(b"gif")
    (run_dir / "evaluations" / "mimic_test_predictions.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    package_dir = _build_hf_package(str(run_dir))

    assert _same_file_identity(run_dir / "model" / "model.safetensors", package_dir / "model.safetensors")
    assert _same_file_identity(
        run_dir / "segmenters" / "lung_segmenter_dinounet_finetuned.pth",
        package_dir / "segmenters" / "lung_segmenter_dinounet_finetuned.pth",
    )
    assert _same_file_identity(run_dir / "assets" / "AnatomicalAttention.gif", package_dir / "assets" / "AnatomicalAttention.gif")


class _DummyTokenizer:
    def save_pretrained(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "tokenizer.json").write_text('{"dummy": true}', encoding="utf-8")
        (output_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")


def test_checkpoint_tokenizer_reuses_existing_snapshot(tmp_path: Path) -> None:
    checkpoints_root = tmp_path / "checkpoints"
    first_ckpt = checkpoints_root / "step_0001000"
    second_ckpt = checkpoints_root / "step_0002000"

    tokenizer = _DummyTokenizer()
    save_or_link_checkpoint_tokenizer(checkpoints_root, first_ckpt, tokenizer)
    save_or_link_checkpoint_tokenizer(checkpoints_root, second_ckpt, tokenizer)

    assert _same_file_identity(
        first_ckpt / "tokenizer" / "tokenizer.json",
        second_ckpt / "tokenizer" / "tokenizer.json",
    )
    assert _same_file_identity(
        first_ckpt / "tokenizer" / "tokenizer_config.json",
        second_ckpt / "tokenizer" / "tokenizer_config.json",
    )
