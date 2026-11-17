"""The one place that knows how HAM10000 files are named on disk (raw and cached)."""

from __future__ import annotations

from pathlib import Path

METADATA_CSV = "HAM10000_metadata.csv"


def raw_image_path(raw_dir: Path, image_id: str) -> Path:
    """Original dermoscopy JPEG, e.g. images/ISIC_0024306.jpg."""
    return raw_dir / "images" / f"{image_id}.jpg"


def raw_mask_path(raw_dir: Path, image_id: str) -> Path:
    """Original binary lesion mask, e.g. masks/ISIC_0024306_segmentation.png."""
    return raw_dir / "masks" / f"{image_id}_segmentation.png"


def cached_image_path(cache_dir: Path, image_id: str) -> Path:
    """Resized image in the preprocessing cache (lossless PNG)."""
    return cache_dir / "images" / f"{image_id}.png"


def cached_mask_path(cache_dir: Path, image_id: str) -> Path:
    """Resized mask in the preprocessing cache (PNG, values 0/255)."""
    return cache_dir / "masks" / f"{image_id}.png"
