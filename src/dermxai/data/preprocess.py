"""Resize every image and its mask ONCE (same geometry) and cache them as PNG in data/cache/.

Why: decoding a 600x450 JPEG and resizing it costs more CPU time than a ResNet-18 forward pass
on a small crop, so doing it every epoch would dominate training time on a laptop.

Geometry: the short side becomes `short_side` px and the aspect ratio is kept (600x450 -> 341x256).
Nothing is cropped here, so image borders (vignettes, rulers) are preserved for later analysis.
Images use INTER_AREA (the right filter for down-scaling); masks use INTER_NEAREST so they stay binary.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np

from dermxai.data.layout import cached_image_path, cached_mask_path, raw_image_path, raw_mask_path
from dermxai.utils import check_disk

logger = logging.getLogger(__name__)

EST_BYTES_PER_PAIR = 250_000  # conservative estimate for a 341x256 PNG image + mask; checked against reality in the log


def resized_shape(height: int, width: int, short_side: int) -> tuple[int, int]:
    """(new_height, new_width) so that the shorter side equals short_side, aspect ratio kept."""
    scale = short_side / min(height, width)
    return max(1, round(height * scale)), max(1, round(width * scale))


def resize_pair(image: np.ndarray, mask: np.ndarray, short_side: int) -> tuple[np.ndarray, np.ndarray]:
    """Resize an HxWx3 uint8 image and its HxW mask to the same new size; mask returned as 0/255 uint8."""
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(f"image {image.shape[:2]} and mask {mask.shape[:2]} differ in size")
    new_h, new_w = resized_shape(image.shape[0], image.shape[1], short_side)
    shrinking = new_h < image.shape[0]
    img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR)
    binary = np.where(mask > 127, 255, 0).astype(np.uint8)
    msk = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return img, msk


def read_image_bgr(path: Path) -> np.ndarray:
    """Read a colour image as HxWx3 uint8 BGR (np.fromfile handles any Windows path)."""
    arr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise IOError(f"Could not decode image {path}")
    return arr


def read_gray(path: Path) -> np.ndarray:
    """Read a single-channel image (e.g. a mask) as HxW uint8."""
    arr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise IOError(f"Could not decode image {path}")
    return arr


def write_png_atomic(path: Path, array: np.ndarray) -> None:
    """Write a PNG via a temp file + rename, so Ctrl+C can never leave a half-written file."""
    ok, buf = cv2.imencode(".png", array)
    if not ok:
        raise IOError(f"PNG encoding failed for {path}")
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(buf.tobytes())
    os.replace(tmp, path)


def preprocess(raw_dir: Path, cache_dir: Path, image_ids: list[str], short_side: int,
               min_free_gb: float, limit: int | None = None) -> dict[str, float]:
    """Cache resized image/mask pairs; already-cached pairs are skipped, so re-running resumes."""
    (cache_dir / "images").mkdir(parents=True, exist_ok=True)
    (cache_dir / "masks").mkdir(parents=True, exist_ok=True)
    todo = [i for i in image_ids
            if not (cached_image_path(cache_dir, i).exists() and cached_mask_path(cache_dir, i).exists())]
    logger.info("%d of %d pairs already cached, %d to do", len(image_ids) - len(todo), len(image_ids), len(todo))
    if limit is not None:
        todo = todo[:limit]
        logger.info("--limit %d: processing only the first %d (timing run)", limit, len(todo))
    free = check_disk(cache_dir, min_free_gb, needed_bytes=len(todo) * EST_BYTES_PER_PAIR)
    logger.info("Free disk: %.1f GB; estimated cache growth: %.2f GB", free, len(todo) * EST_BYTES_PER_PAIR / 1024**3)

    t0 = time.perf_counter()
    written_bytes = 0
    done = 0
    try:
        for done, image_id in enumerate(todo, start=1):
            img = read_image_bgr(raw_image_path(raw_dir, image_id))
            msk = read_gray(raw_mask_path(raw_dir, image_id))
            img_s, msk_s = resize_pair(img, msk, short_side)
            write_png_atomic(cached_image_path(cache_dir, image_id), img_s)
            write_png_atomic(cached_mask_path(cache_dir, image_id), msk_s)
            written_bytes += cached_image_path(cache_dir, image_id).stat().st_size
            written_bytes += cached_mask_path(cache_dir, image_id).stat().st_size
            if done == 20 or done % 1000 == 0 or done == len(todo):
                elapsed = time.perf_counter() - t0
                per = elapsed / done
                logger.info("  %d/%d done, %.1f ms per pair, ETA %.1f min, avg %.0f KB per pair",
                            done, len(todo), 1000 * per, per * (len(todo) - done) / 60,
                            written_bytes / done / 1024)
    except KeyboardInterrupt:
        logger.warning("Interrupted after %d pairs. Every finished file is complete; re-run to resume.", done - 1)
        raise
    elapsed = time.perf_counter() - t0
    stats = {"processed": done if todo else 0, "seconds": round(elapsed, 1),
             "ms_per_pair": round(1000 * elapsed / max(1, done), 2),
             "projected_minutes_for_all": round(elapsed / max(1, done) * len(image_ids) / 60, 1),
             "avg_kb_per_pair": round(written_bytes / max(1, done) / 1024, 1)}
    if limit is None:
        manifest = {"short_side": short_side, "n_pairs": len(image_ids), "format": "PNG, image BGR uint8, mask 0/255",
                    "image_interpolation": "INTER_AREA", "mask_interpolation": "INTER_NEAREST (after >127 binarisation)"}
        (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Preprocessing stats: %s", stats)
    return stats
