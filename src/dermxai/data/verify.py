"""Check that the raw HAM10000 download is complete and consistent, and write a JSON report."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from dermxai import CLASSES
from dermxai.data.layout import METADATA_CSV, raw_image_path, raw_mask_path

logger = logging.getLogger(__name__)

EXPECTED_IMAGES = 10015  # stated on the Dataverse page and in Tschandl et al. (2018)
REQUIRED_COLUMNS = ("lesion_id", "image_id", "dx")


def load_metadata(raw_dir: Path) -> pd.DataFrame:
    """Read HAM10000_metadata.csv."""
    return pd.read_csv(raw_dir / METADATA_CSV)


def verify(raw_dir: Path, report_path: Path) -> bool:
    """Run every check, log a readable report, save it as JSON and return True if all checks pass."""
    checks: dict[str, bool] = {}
    info: dict[str, Any] = {}

    meta_file = raw_dir / METADATA_CSV
    if not meta_file.exists():
        logger.error("Metadata not found: %s", meta_file)
        return False
    df = load_metadata(raw_dir)
    checks["metadata has required columns"] = all(c in df.columns for c in REQUIRED_COLUMNS)
    checks[f"metadata has {EXPECTED_IMAGES} rows"] = len(df) == EXPECTED_IMAGES
    checks["image_id is unique"] = bool(df["image_id"].is_unique)
    checks["all dx codes are known"] = set(df["dx"]) <= set(CLASSES)
    dx_per_lesion = df.groupby("lesion_id")["dx"].nunique()
    checks["every lesion_id has exactly one diagnosis"] = bool((dx_per_lesion == 1).all())
    info["n_images"] = int(len(df))
    info["n_lesions"] = int(df["lesion_id"].nunique())
    info["images_per_lesion"] = {int(k): int(v) for k, v in df.groupby("lesion_id").size().value_counts().sort_index().items()}
    info["images_per_class"] = {k: int(v) for k, v in df["dx"].value_counts().items()}
    info["lesions_per_class"] = {k: int(v) for k, v in df.groupby("dx")["lesion_id"].nunique().items()}

    ids = list(df["image_id"])
    missing_img = [i for i in ids if not raw_image_path(raw_dir, i).exists()]
    missing_mask = [i for i in ids if not raw_mask_path(raw_dir, i).exists()]
    n_jpg = len(list((raw_dir / "images").glob("*.jpg"))) if (raw_dir / "images").exists() else 0
    n_png = len(list((raw_dir / "masks").glob("*.png"))) if (raw_dir / "masks").exists() else 0
    checks["every metadata image has a JPEG"] = not missing_img
    checks["every metadata image has a mask"] = not missing_mask
    checks[f"images folder holds {EXPECTED_IMAGES} JPEGs"] = n_jpg == EXPECTED_IMAGES
    checks[f"masks folder holds {EXPECTED_IMAGES} PNGs"] = n_png == EXPECTED_IMAGES
    info["missing_images"] = missing_img[:20]
    info["missing_masks"] = missing_mask[:20]

    # Geometry and mask content, for every image that has both files.
    sizes: dict[str, int] = {}
    size_mismatch, nonbinary, empty = [], [], []
    mask_frac: list[tuple[str, float]] = []
    dx_of = dict(zip(df["image_id"], df["dx"]))
    skip = set(missing_img) | set(missing_mask)
    for k, image_id in enumerate(i for i in ids if i not in skip):
        with Image.open(raw_image_path(raw_dir, image_id)) as im:
            img_size = im.size  # header only, no full decode
        with Image.open(raw_mask_path(raw_dir, image_id)) as m:
            mask = np.asarray(m.convert("L"))
        sizes[f"{img_size[0]}x{img_size[1]}"] = sizes.get(f"{img_size[0]}x{img_size[1]}", 0) + 1
        if (mask.shape[1], mask.shape[0]) != img_size:
            size_mismatch.append(image_id)
        if not set(np.unique(mask)) <= {0, 255}:
            nonbinary.append(image_id)
        frac = float((mask > 127).mean())
        if frac == 0.0:
            empty.append(image_id)
        mask_frac.append((dx_of[image_id], frac))
        if (k + 1) % 2000 == 0:
            logger.info("  checked %d images ...", k + 1)
    checks["every mask has the same size as its image"] = not size_mismatch
    checks["no empty masks"] = not empty
    info["image_sizes"] = sizes
    info["size_mismatch"] = size_mismatch[:20]
    info["empty_masks"] = empty[:20]
    info["n_nonbinary_masks"] = len(nonbinary)  # reported, not a failure: we binarise at >127 anyway
    fr = pd.DataFrame(mask_frac, columns=["dx", "frac"])
    if len(fr):
        info["mask_area_fraction"] = {
            "all": {"mean": round(float(fr["frac"].mean()), 4), "median": round(float(fr["frac"].median()), 4)},
            **{dx: {"mean": round(float(g["frac"].mean()), 4), "median": round(float(g["frac"].median()), 4)}
               for dx, g in fr.groupby("dx")},
        }

    ok = all(checks.values())
    report = {"ok": ok, "checks": checks, "info": info}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info("==== HAM10000 verification report ====")
    for name, passed in checks.items():
        logger.info("  [%s] %s", "PASS" if passed else "FAIL", name)
    logger.info("  images: %d, lesions: %d", info["n_images"], info["n_lesions"])
    logger.info("  images per class: %s", info["images_per_class"])
    logger.info("  lesions per class: %s", info["lesions_per_class"])
    logger.info("  images per lesion (count -> lesions): %s", info["images_per_lesion"])
    logger.info("  image sizes: %s", sizes)
    logger.info("  non-binary masks (binarised at >127): %d", len(nonbinary))
    if "mask_area_fraction" in info:
        logger.info("  mean mask area fraction (= random-baseline for 'energy in mask'): %s",
                    info["mask_area_fraction"]["all"])
    logger.info("  overall: %s (report: %s)", "OK" if ok else "PROBLEMS FOUND", report_path)
    return ok
