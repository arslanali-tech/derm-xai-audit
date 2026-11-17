"""Shared helpers: project paths, logging, seeding, and laptop-safety guards (threads, priority, RAM, disk).

torch is imported lazily inside functions so that data-only commands start quickly.
"""

from __future__ import annotations

import logging
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import psutil

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
GB = 1024**3


def project_root() -> Path:
    """Return the repository root (the folder containing pyproject.toml)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not find pyproject.toml above " + str(here))


def resolve(path: str | Path) -> Path:
    """Turn a project-relative path from the config into an absolute path inside the project."""
    p = Path(path)
    return p if p.is_absolute() else project_root() / p


def use_project_local_caches() -> None:
    """Point torch-hub and matplotlib caches into <project>/.cache for THIS process only.

    Without this, pretrained weights and font caches are written to the user profile.
    os.environ changes affect only the current process and its children; nothing is permanent.
    """
    cache = project_root() / ".cache"
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("MPLBACKEND", "Agg")


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    """Log to the console and, optionally, to a file (e.g. results/<run>/log.txt)."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=handlers, force=True)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and (if installed) torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_num_threads(requested: int | None = None) -> int:
    """Number of compute threads: physical cores - 1 (at least 1), or `requested` if smaller."""
    physical = psutil.cpu_count(logical=False) or 1
    limit = max(1, physical - 1)
    return limit if requested is None else max(1, min(requested, limit))


def configure_torch_threads(requested: int | None = None) -> int:
    """Limit torch's intra-op thread pool so one core always stays free for Windows."""
    import torch

    n = safe_num_threads(requested)
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)  # may only be called once per process
    except RuntimeError:
        pass
    return n


def lower_process_priority() -> None:
    """Run this Python process at below-normal priority so the desktop stays responsive."""
    proc = psutil.Process()
    if sys.platform == "win32":
        proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        proc.nice(10)


def resolve_device(name: str) -> "torch.device":  # noqa: F821 (lazy import)
    """Map 'cpu' / 'cuda' / 'auto' to a torch.device; never fails if CUDA is missing."""
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        logging.getLogger(__name__).warning("CUDA requested but not available; using CPU.")
        return torch.device("cpu")
    return torch.device(name)


def ram_budget_bytes(max_fraction: float) -> int:
    """Memory this job may use: the smaller of max_fraction x total RAM and 80% of what is free now."""
    vm = psutil.virtual_memory()
    return int(min(max_fraction * vm.total, 0.8 * vm.available))


def log_ram(logger: logging.Logger, max_fraction: float) -> None:
    """Log total / available RAM and this job's budget."""
    vm = psutil.virtual_memory()
    logger.info(
        "RAM: total %.1f GB, available %.1f GB, budget for this job %.1f GB",
        vm.total / GB, vm.available / GB, ram_budget_bytes(max_fraction) / GB,
    )


def check_disk(path: Path, min_free_gb: float, needed_bytes: int = 0) -> float:
    """Raise if writing `needed_bytes` under `path` would leave less than `min_free_gb` free; return free GB."""
    probe = path
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    after = (free - needed_bytes) / GB
    if after < min_free_gb:
        raise RuntimeError(
            f"Not enough disk: {free / GB:.1f} GB free, this step needs ~{needed_bytes / GB:.1f} GB, "
            f"which would leave {after:.1f} GB (< {min_free_gb} GB safety floor)."
        )
    return free / GB


def prepare_long_job(threads: int | None, low_priority: bool, logger: logging.Logger) -> int:
    """Apply the laptop-safety settings for a compute-heavy job; return the thread count used."""
    n = configure_torch_threads(threads)
    if low_priority:
        lower_process_priority()
    logger.info("Using %d torch threads (physical cores: %s), low priority: %s",
                n, psutil.cpu_count(logical=False), low_priority)
    return n
