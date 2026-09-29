"""Paired bootstrap of the test-set lift@10% gap between the logistic and LightGBM.

Run under WSL from the repository root, after scripts/register_model.py:
    PTE_ENV=wsl python scripts/bootstrap_lift.py

Added after the one test-set evaluation, outside the checklist in leakage.md 5.1 (decided
2026-09-30, Andy): on test the logistic's lift@10% is 3.7673 against LightGBM's 3.7407, and
the top decile is where P4 sends coupons. It selects nothing, as the final model is frozen
either way; it only puts an interval on a number already reported. Fixed before running:

- statistic: lift@10% of the logistic minus lift@10% of LightGBM (version 1, raw p; Platt
  does not change the ranking)
- each round draws as many users as the test set, with replacement, and scores both models
  on the same draw (paired, so the luck of the draw is the same for both)
- ROUNDS rounds, seed SEED, 95% interval from the 2.5th and 97.5th percentiles
- reading: an interval containing 0 means no distinguishable gap in the top decile;
  otherwise the sign of the interval says which model is ahead

Run 2026-09-30 (13 minutes): observed gap +0.0267; 1000 rounds, mean +0.0265, sd 0.0098,
95% interval [+0.0077, +0.0461], gap <= 0 in 0.5% of rounds. So in the test period the
logistic is ahead in the top decile. The final model stays LightGBM (Andy, 2026-09-30):
switching on a test result would make the test set a selection set.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

import mlflow
import numpy as np
import pandas as pd

from pte.config import load_config, resolve
from pte.model.data import load_split, model_features
from pte.model.logreg import build_logreg
from pte.model.metrics import lift_at
from pte.model.registry import EXPERIMENT, MODEL_NAME
from pte.model.tracking import setup_tracking

MODEL_VERSION = "1"
ROUNDS = 1000
SEED = 0
FRAC = 0.1


def gap(table: pd.DataFrame) -> float:
    """lift@10% of the logistic minus that of LightGBM, on the rows of table."""
    lift_logreg = lift_at(table["p_logreg"], table["label"], FRAC)
    return lift_logreg - lift_at(table["p_lgbm"], table["label"], FRAC)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", help="configs/<env>.yaml (default $PTE_ENV, then dev)")
    args = parser.parse_args(argv)
    paths = load_config(args.env)["paths"]
    features_dir = str(resolve(paths["features_dir"]))
    setup_tracking(resolve(paths["mlflow_dir"]), EXPERIMENT)

    train = load_split(features_dir, "train")
    test = load_split(features_dir, "test")
    features = model_features(train)
    logreg = build_logreg().fit(train[features], train["label"])
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")
    # only the three columns each round needs, so a resample copies little
    table = pd.DataFrame(
        {
            "label": test["label"].to_numpy(),
            "p_lgbm": model.predict(test)["p_raw"].to_numpy(),
            "p_logreg": logreg.predict_proba(test[features])[:, 1],
        }
    )
    observed = gap(table)
    print(f"Observed gap on test: {observed:+.4f}")

    rng = np.random.default_rng(SEED)
    n = len(table)
    gaps = np.empty(ROUNDS)
    start = time.perf_counter()
    for i in range(ROUNDS):
        gaps[i] = gap(table.iloc[rng.integers(0, n, size=n)])
        if (i + 1) % 100 == 0:
            print(f"  {i + 1} rounds, {time.perf_counter() - start:.0f} s")
    low, high = np.percentile(gaps, [2.5, 97.5])
    print(f"{ROUNDS} rounds: mean {gaps.mean():+.4f}, sd {gaps.std(ddof=1):.4f}")
    print(f"95% interval [{low:+.4f}, {high:+.4f}]; rounds with gap <= 0: {(gaps <= 0).mean():.1%}")


if __name__ == "__main__":
    main()
