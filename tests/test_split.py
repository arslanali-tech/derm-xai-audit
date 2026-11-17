"""The lesion-level split must never put two images of one lesion into different splits."""

import numpy as np
import pandas as pd
import pytest

from dermxai import CLASSES
from dermxai.data.split import assert_no_leakage, make_lesion_split


def toy_metadata(lesions_per_class: int = 40, seed: int = 0) -> pd.DataFrame:
    """Fake HAM10000-style metadata: every lesion has 1-4 images, one diagnosis per lesion."""
    rng = np.random.default_rng(seed)
    rows, img = [], 0
    for dx in CLASSES:
        for k in range(lesions_per_class):
            for _ in range(rng.integers(1, 5)):
                rows.append({"lesion_id": f"HAM_{dx}_{k}", "image_id": f"ISIC_{img:07d}", "dx": dx})
                img += 1
    return pd.DataFrame(rows)


def test_no_lesion_in_two_splits() -> None:
    split = make_lesion_split(toy_metadata(), 0.15, 0.15, seed=42)
    assert (split.groupby("lesion_id")["split"].nunique() == 1).all()


def test_every_image_assigned_once() -> None:
    df = toy_metadata()
    split = make_lesion_split(df, 0.15, 0.15, seed=42)
    assert len(split) == len(df)
    assert set(split["image_id"]) == set(df["image_id"])
    assert set(split["split"]) == {"train", "val", "test"}


def test_stratified_every_class_in_every_split() -> None:
    split = make_lesion_split(toy_metadata(), 0.15, 0.15, seed=42)
    assert (pd.crosstab(split["dx"], split["split"]) > 0).all().all()


def test_lesion_fractions_close_to_requested() -> None:
    split = make_lesion_split(toy_metadata(lesions_per_class=100), 0.15, 0.15, seed=42)
    frac = split.groupby("split")["lesion_id"].nunique() / split["lesion_id"].nunique()
    assert frac["test"] == pytest.approx(0.15, abs=0.02)
    assert frac["val"] == pytest.approx(0.15, abs=0.02)


def test_same_seed_same_split_different_seed_different_split() -> None:
    df = toy_metadata()
    a = make_lesion_split(df, 0.15, 0.15, seed=1)
    b = make_lesion_split(df, 0.15, 0.15, seed=1)
    c = make_lesion_split(df, 0.15, 0.15, seed=2)
    assert a.equals(b)
    assert not a.equals(c)


def test_leakage_detector_catches_a_leak() -> None:
    leaked = pd.DataFrame({"image_id": ["a", "b"], "lesion_id": ["L1", "L1"], "dx": ["nv", "nv"],
                           "split": ["train", "test"]})
    with pytest.raises(AssertionError):
        assert_no_leakage(leaked)


def test_lesion_with_two_diagnoses_is_rejected() -> None:
    df = toy_metadata()
    assert df.loc[0, "dx"] != "vasc"
    # Add a second image of lesion 0 with a different diagnosis -> inconsistent metadata.
    df = pd.concat([df, df.iloc[[0]].assign(image_id="ISIC_extra", dx="vasc")], ignore_index=True)
    with pytest.raises(ValueError):
        make_lesion_split(df, 0.15, 0.15, seed=42)
