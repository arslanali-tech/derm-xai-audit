"""Draw the lesion-mask outline on a few cached images to check by eye that images and masks line up.

One image per diagnosis, chosen with a fixed seed from the training split.
Output: results/data/cache_samples.png

Usage:  .venv\\Scripts\\python.exe scripts\\show_cache_samples.py
"""

from __future__ import annotations

import logging

from dermxai.utils import use_project_local_caches

use_project_local_caches()

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from dermxai import CLASSES  # noqa: E402
from dermxai.config import load_config  # noqa: E402
from dermxai.data.layout import cached_image_path, cached_mask_path  # noqa: E402
from dermxai.data.preprocess import read_gray, read_image_bgr  # noqa: E402
from dermxai.utils import resolve, setup_logging  # noqa: E402

logger = logging.getLogger("cache_samples")


def main() -> None:
    """Save a 1 x 7 grid: one cached training image per class with its mask outline in green."""
    setup_logging()
    cfg = load_config(resolve("configs/resnet18.yaml"))
    cache = resolve(cfg.paths.cache_dir)
    split = pd.read_csv(resolve(cfg.paths.splits_csv))
    train = split[split["split"] == "train"]
    picks = [train[train["dx"] == dx].sample(1, random_state=cfg.runtime.seed).iloc[0] for dx in CLASSES]

    fig, axes = plt.subplots(1, len(picks), figsize=(3 * len(picks), 2.6))
    for ax, row in zip(axes, picks):
        img = cv2.cvtColor(read_image_bgr(cached_image_path(cache, row.image_id)), cv2.COLOR_BGR2RGB)
        mask = read_gray(cached_mask_path(cache, row.image_id))
        contours, _ = cv2.findContours((mask > 0).astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
        ax.imshow(img)
        ax.set_title(f"{row.dx}  ({(mask > 0).mean():.0%} lesion)\n{row.image_id}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Cached images (341x256) with lesion-mask outline (green)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.88))  # leave room for the suptitle
    out = resolve(cfg.paths.results_dir) / "data" / "cache_samples.png"
    fig.savefig(out, dpi=110)
    logger.info("Saved %s", out)


if __name__ == "__main__":
    main()
