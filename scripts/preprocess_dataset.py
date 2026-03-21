import argparse
import logging
from pathlib import Path

import pandas as pd

from data_utils.cache_io import write_json
from data_utils.image_ops import resize_rgb_image
from data_utils.mask_ops import resize_mask

LOGGER = logging.getLogger("preprocess_dataset")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resize raw radiology images into a training cache.")
    parser.add_argument("--manifest", required=True, help="CSV with columns image_path, report_text, optional mask_path, report_id.")
    parser.add_argument("--output-dir", required=True, help="Directory where resized PNG cache and manifest will be stored.")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite cached files if they already exist.")
    parser.add_argument("--log-level", default="INFO")
    return parser


def preprocess_dataset(manifest_path: str, output_dir: str, image_size: int, overwrite: bool) -> None:
    manifest = pd.read_csv(manifest_path)
    output_root = Path(output_dir)
    image_root = output_root / "images"
    mask_root = output_root / "masks"
    processed_rows = []

    for index, row in manifest.iterrows():
        report_id = str(row.get("report_id", index))
        image_target = image_root / f"{report_id}.png"
        if overwrite or not image_target.exists():
            resize_rgb_image(str(row["image_path"]), str(image_target), image_size)

        processed_row = {
            "report_id": report_id,
            "processed_image_path": str(image_target.resolve()),
            "report_text": row["report_text"],
            "source_image_path": str(row["image_path"]),
        }

        mask_path = row.get("mask_path")
        if isinstance(mask_path, str) and mask_path:
            mask_target = mask_root / f"{report_id}.png"
            if overwrite or not mask_target.exists():
                resize_mask(mask_path, str(mask_target), image_size)
            processed_row["processed_mask_path"] = str(mask_target.resolve())
            processed_row["source_mask_path"] = mask_path

        processed_rows.append(processed_row)

    processed_manifest_path = output_root / "processed_manifest.csv"
    pd.DataFrame(processed_rows).to_csv(processed_manifest_path, index=False)

    summary = {
        "input_manifest": str(Path(manifest_path).resolve()),
        "output_dir": str(output_root.resolve()),
        "num_examples": len(processed_rows),
        "image_size": image_size,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "cached_artifact_format": "png",
        "mask_alignment": "Masks are resized to the same square resolution as the image cache.",
        "note": "This script only caches resized images and masks. Tensor normalization remains inside the training code.",
    }
    write_json(str(output_root / "preprocessing_summary.json"), summary)
    LOGGER.info("Wrote %s resized samples to %s", len(processed_rows), output_root)


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    preprocess_dataset(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        image_size=args.image_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
