"""Score the RFM baseline and print the numbers the P3 models must beat.

Run under WSL from the repository root, after scripts/build_features.py:
    PTE_ENV=wsl python scripts/rfm_baseline.py

Fits the M bins on the training buyers, scores the validation set (the test set stays
untouched until the models are frozen) and prints ROC-AUC, AP and lift at 10% / 20%.
On the data of 2026-09-28: ROC-AUC 0.670, AP 0.185, lift@10% 4.1249, lift@20% 2.3889.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from pte.config import load_config, resolve
from pte.model.metrics import lift_at
from pte.model.rfm import COLUMNS, fit_spend_bins, rfm_score


def load_split(features_dir: str, split: str) -> pd.DataFrame:
    """RFM input columns joined to the label; every user must have one."""
    features = pd.read_parquet(f"{features_dir}/{split}/features", columns=["user_id", *COLUMNS])
    labels = pd.read_parquet(f"{features_dir}/{split}/labels")
    df = features.merge(labels, on="user_id", how="left", validate="one_to_one")
    if df["label"].isna().any():
        raise ValueError(f"{split}: {df['label'].isna().sum()} users have no label")
    return df


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", help="configs/<env>.yaml (default $PTE_ENV, then dev)")
    args = parser.parse_args(argv)
    features_dir = str(resolve(load_config(args.env)["paths"]["features_dir"]))

    spend_bins = fit_spend_bins(load_split(features_dir, "train"))
    print("M bins (training buyers):", [round(b, 2) for b in spend_bins[1:-1]])

    valid = load_split(features_dir, "valid")
    score = rfm_score(valid, spend_bins)
    rate = valid["label"].mean()
    ap = average_precision_score(valid["label"], score)
    print(f"valid: {len(valid):,} users, positive rate {rate:.2%}")
    print(f"ROC-AUC  {roc_auc_score(valid['label'], score):.4f}")
    print(f"AP       {ap:.4f}  ({ap / rate:.2f}x the positive rate)")
    for frac in (0.1, 0.2):
        print(f"lift@{frac:.0%}  {lift_at(score, valid['label'], frac):.4f}")


if __name__ == "__main__":
    main()
