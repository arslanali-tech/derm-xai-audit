"""Typed configuration: dataclasses filled from a YAML file.

Every tunable number lives here (with a default) or in configs/*.yaml.
Unknown YAML keys raise an error, so a typo can never be silently ignored.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathsConfig:
    """Where data and results live, relative to the project root."""

    raw_dir: str = "data/raw/HAM10000"
    cache_dir: str = "data/cache"
    splits_csv: str = "data/splits.csv"
    results_dir: str = "results"


@dataclass(frozen=True)
class DataConfig:
    """Dataset, preprocessing and split settings."""

    cache_short_side: int = 256  # images+masks are resized once so their short side is this many px
    image_size: int = 224  # network input size (train crops are image_size x image_size)
    val_frac: float = 0.15  # fraction of LESIONS (not images) in validation
    test_frac: float = 0.15  # fraction of LESIONS (not images) in test
    split_seed: int = 42


@dataclass(frozen=True)
class RuntimeConfig:
    """Laptop-safety and hardware settings."""

    device: str = "cpu"  # "cpu", "cuda" or "auto"; the project is designed for CPU
    num_threads: int | None = None  # None = physical cores - 1 (minimum 1)
    num_workers: int = 0  # DataLoader workers; 0 is safest on Windows
    low_priority: bool = True  # run long jobs at below-normal OS priority
    max_ram_fraction: float = 0.5  # keep this process under this fraction of total RAM
    min_free_disk_gb: float = 20.0  # refuse to write if the drive would drop below this
    seed: int = 42


@dataclass(frozen=True)
class Config:
    """Top-level config; later phases add their own sections."""

    run_name: str = "default"
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _build(cls: type, values: dict[str, Any]) -> Any:
    """Recursively build dataclass `cls` from a dict, rejecting unknown keys."""
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in values.items():
        default = known[name].default_factory if known[name].default_factory is not dataclasses.MISSING else None
        if default is not None and dataclasses.is_dataclass(default) and isinstance(value, dict):
            kwargs[name] = _build(default, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: Path | str | None) -> Config:
    """Load a YAML config file into a Config (defaults if path is None)."""
    if path is None:
        return Config()
    with Path(path).open("r", encoding="utf-8") as fh:
        values = yaml.safe_load(fh) or {}
    return _build(Config, values)
