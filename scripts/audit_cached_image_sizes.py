import argparse
import json
import logging
import struct
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger("audit_cached_image_sizes")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit cached PNG sizes for training datasets.")
    parser.add_argument("--dataset", default="combined", choices=["chexpert", "mimic", "combined"])
    parser.add_argument("--metadata-path", default="Datasets/CheXpert/df_chexpert_plus_240401_findings.csv")
    parser.add_argument("--image-root", default="Datasets/CheXpert/images")
    parser.add_argument("--mimic-root", default="Datasets/MIMIC")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--output", default=".cache/image_size_audit.json")
    parser.add_argument("--log-level", default="INFO")
    return parser


def png_size(path: Path) -> tuple[int, int] | None:
    with path.open("rb") as handle:
        if handle.read(8) != PNG_SIGNATURE:
            return None
        chunk_length = struct.unpack(">I", handle.read(4))[0]
        if handle.read(4) != b"IHDR" or chunk_length < 8:
            return None
        return struct.unpack(">II", handle.read(8))


def build_chexpert_paths(metadata_path: Path, image_root: Path) -> list[str]:
    metadata = pd.read_csv(metadata_path)
    metadata = metadata[metadata["split"].astype(str).isin({"train", "valid"})].copy()
    findings = metadata["section_findings"].fillna("").astype(str).str.strip()
    metadata = metadata[findings != ""].copy()
    image_paths = metadata["path_to_image"].astype(str).str.replace(".jpg", ".png", regex=False)
    return [str((image_root / relative).resolve()) for relative in image_paths.tolist()]


def build_mimic_paths(mimic_root: Path) -> list[str]:
    split_df = pd.read_csv(mimic_root / "mimic-cxr-2.0.0-split.csv.gz", compression="gzip")
    records_df = pd.read_csv(mimic_root / "cxr-record-list.csv.gz", compression="gzip")
    metadata_df = pd.read_csv(mimic_root / "mimic-cxr-2.0.0-metadata.csv")

    df = split_df[split_df["split"].isin({"train", "validate"})].copy()
    df = df.merge(records_df, on=["subject_id", "study_id", "dicom_id"], how="left")
    df = df.merge(
        metadata_df[["subject_id", "study_id", "dicom_id", "ViewPosition"]],
        on=["subject_id", "study_id", "dicom_id"],
        how="left",
    )
    df["ViewPosition"] = df["ViewPosition"].astype(str).str.upper()
    df = df[df["ViewPosition"].isin({"PA", "AP"})].copy()
    df = df.sort_values(by=["subject_id", "study_id", "dicom_id"]).drop_duplicates(
        subset=["subject_id", "study_id"],
        keep="first",
    )

    image_root = mimic_root / "images" / "datos"
    return [
        str((image_root / f"p{int(row.subject_id)}" / f"s{int(row.study_id)}" / f"{row.dicom_id}.png").resolve())
        for row in df[["subject_id", "study_id", "dicom_id"]].itertuples(index=False)
    ]


def audit_paths(dataset_name: str, image_paths: list[str], expected_size: int) -> dict:
    counts: dict[str, int] = {}
    checked = 0
    missing = 0
    invalid = 0

    for index, raw_path in enumerate(image_paths, start=1):
        path = Path(raw_path)
        if not path.exists():
            missing += 1
            continue
        size = png_size(path)
        if size is None:
            invalid += 1
            counts["invalid"] = counts.get("invalid", 0) + 1
            continue
        key = f"{size[0]}x{size[1]}"
        counts[key] = counts.get(key, 0) + 1
        checked += 1
        if index % 5000 == 0:
            LOGGER.info("[%s] audited %s/%s images", dataset_name, index, len(image_paths))

    expected_key = f"{expected_size}x{expected_size}"
    verified = checked > 0 and missing == 0 and invalid == 0 and set(counts.keys()) == {expected_key}
    return {
        "dataset": dataset_name,
        "expected_size": expected_size,
        "verified": verified,
        "checked_images": checked,
        "missing_images": missing,
        "invalid_png_headers": invalid,
        "size_counts": counts,
    }


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "image_size": args.image_size,
        "datasets": {},
    }

    if args.dataset in {"chexpert", "combined"}:
        chexpert_root = Path(args.image_root).resolve()
        chexpert_result = audit_paths(
            "chexpert",
            build_chexpert_paths(Path(args.metadata_path), chexpert_root),
            args.image_size,
        )
        chexpert_result["image_root"] = str(chexpert_root)
        chexpert_result["metadata_path"] = str(Path(args.metadata_path).resolve())
        payload["datasets"]["chexpert"] = chexpert_result

    if args.dataset in {"mimic", "combined"}:
        mimic_root = Path(args.mimic_root).resolve()
        mimic_result = audit_paths("mimic", build_mimic_paths(mimic_root), args.image_size)
        mimic_result["mimic_root"] = str(mimic_root)
        payload["datasets"]["mimic"] = mimic_result

    payload["verified_all"] = bool(payload["datasets"]) and all(
        entry.get("verified", False) for entry in payload["datasets"].values()
    )
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("Wrote image size audit to %s", output_path.resolve())


if __name__ == "__main__":
    main()
