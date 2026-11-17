"""Leakage-free train/val/test split: split LESIONS, then give each image its lesion's split.

HAM10000 contains several photos of the same lesion (same `lesion_id`). If two photos of one
lesion land in train and test, the test set partly measures memorisation, not generalisation.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


def make_lesion_split(df: pd.DataFrame, val_frac: float, test_frac: float, seed: int) -> pd.DataFrame:
    """Return image_id, lesion_id, dx, split; lesions are split stratified by diagnosis."""
    if not 0 < val_frac < 1 or not 0 < test_frac < 1 or val_frac + test_frac >= 1:
        raise ValueError("val_frac and test_frac must be in (0, 1) and sum to < 1")
    # One row per lesion. verify-data checks that each lesion has exactly one diagnosis.
    lesions = df.groupby("lesion_id", sort=True)["dx"].first().reset_index()
    if (df.groupby("lesion_id")["dx"].nunique() > 1).any():
        raise ValueError("Some lesion_id has more than one dx; cannot stratify by lesion.")

    # Step 1: carve off the test lesions. Step 2: split the rest into train / val.
    rest, test = train_test_split(lesions, test_size=test_frac, stratify=lesions["dx"], random_state=seed)
    val_share_of_rest = val_frac / (1.0 - test_frac)
    train, val = train_test_split(rest, test_size=val_share_of_rest, stratify=rest["dx"], random_state=seed)

    split_of = {**dict.fromkeys(train["lesion_id"], "train"),
                **dict.fromkeys(val["lesion_id"], "val"),
                **dict.fromkeys(test["lesion_id"], "test")}
    out = df[["image_id", "lesion_id", "dx"]].copy()
    out["split"] = out["lesion_id"].map(split_of)
    assert_no_leakage(out)
    return out.sort_values("image_id").reset_index(drop=True)


def assert_no_leakage(split_df: pd.DataFrame) -> None:
    """Raise if any image lacks a split or any lesion_id appears in more than one split."""
    if split_df["split"].isna().any():
        raise AssertionError("Some images have no split assigned.")
    n_splits_per_lesion = split_df.groupby("lesion_id")["split"].nunique()
    leaked = n_splits_per_lesion[n_splits_per_lesion > 1]
    if len(leaked):
        raise AssertionError(f"{len(leaked)} lesion_ids appear in more than one split, e.g. {list(leaked.index[:5])}")


def summarise_split(split_df: pd.DataFrame) -> pd.DataFrame:
    """Table of image counts per class and split, plus each split's share of all images."""
    table = pd.crosstab(split_df["dx"], split_df["split"]).reindex(columns=list(SPLITS), fill_value=0)
    table.loc["TOTAL"] = table.sum()
    table.loc["share of images"] = (table.loc["TOTAL"] / table.loc["TOTAL"].sum()).round(3)
    return table
