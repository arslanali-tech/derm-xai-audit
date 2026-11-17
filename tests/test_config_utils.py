"""Config loading rejects typos; safety helpers respect the laptop limits."""

import psutil
import pytest

from dermxai.config import Config, load_config
from dermxai.utils import check_disk, project_root, safe_num_threads


def test_project_config_loads() -> None:
    cfg = load_config(project_root() / "configs" / "resnet18.yaml")
    assert cfg.run_name == "resnet18"
    assert cfg.data.image_size == 224


def test_unknown_key_is_rejected(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("data:\n  image_sise: 224\n", encoding="utf-8")  # typo on purpose
    with pytest.raises(ValueError, match="image_sise"):
        load_config(bad)


def test_defaults_without_file() -> None:
    assert load_config(None) == Config()


def test_threads_leave_one_core_free() -> None:
    physical = psutil.cpu_count(logical=False) or 1
    assert safe_num_threads() == max(1, physical - 1)
    assert safe_num_threads(1000) == max(1, physical - 1)  # never more than the limit
    assert safe_num_threads(1) == 1


def test_disk_check_refuses_huge_write(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="Not enough disk"):
        check_disk(tmp_path, min_free_gb=20, needed_bytes=10**15)  # 1 PB
