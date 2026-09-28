"""Load a split's features and labels for P3, and pick the model's feature columns."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

# main_category_id is left out (decided 2026-09-28, leakage.md 3.4):
# main_category_conversion_rate carries the same information and, being recomputed at
# every cutoff, has no unseen-category problem on the test set.
NOT_FEATURES = ("user_id", "label", "main_category_id")


def load_split(features_dir: str, split: str, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Features joined to the label, one row per user; every user must have a label.

    columns limits the feature columns read; user_id is always read for the join.
    """
    if columns is not None:
        columns = ["user_id", *columns]
    features = pd.read_parquet(f"{features_dir}/{split}/features", columns=columns)
    labels = pd.read_parquet(f"{features_dir}/{split}/labels")
    df = features.merge(labels, on="user_id", how="left", validate="one_to_one")
    if df["label"].isna().any():
        raise ValueError(f"{split}: {df['label'].isna().sum()} users have no label")
    return df


def model_features(df: pd.DataFrame) -> list[str]:
    """Every column except the id, the label and main_category_id, in table order."""
    return [c for c in df.columns if c not in NOT_FEATURES]
