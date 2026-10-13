"""PSI drift checks for daily_score, from Andy's P5 notebook (2026-10-12/13).

Decided in pipeline.md section 5: the reference is Valid (cutoff 2019-11-01); the variables are
the score p and the top five SHAP features; each is cut at the reference's deciles, the cuts
are frozen, and every later day is cut on those same cuts. Recutting a day on its own deciles
puts 10% in every bin, so its PSI is always 0.

PSI = sum over bins of (a - e) * ln(a / e), with e the reference's share in a bin and a the
day's. Both are floored first, because a bin nobody falls into gives ln(0).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

VARIABLES = [
    "p",
    "n_purchases_per_day",
    "avg_session_seconds",
    "days_since_last_event",
    "n_categories",
    "active_days",
]
N_BINS = 10
# Valid: the model never trained on it, and it is the last population seen before launch
REFERENCE_CUTOFF = "2019-11-01"
REFERENCE_FILE = "psi_reference.json"
# 1e-6, not 1e-4 (both decided 2026-10-12): Valid's smallest bin is 4.55% (n_categories = 4),
# and 1e-4 only covers a bin emptying down to 4.16%; 1e-6 covers it down to 2.47%.
FLOOR = 1e-6
# Fixed before the replay (pipeline.md section 5): any variable at or above it fails alert.
ALERT_PSI = 0.25


def load_day(
    scores_path: Path, features_path: Path, variables: list[str] = VARIABLES
) -> pd.DataFrame:
    """One row per user: user_id and the variables, p from the scores and the rest from features.

    score_users scores every row of the feature table, so both sides hold the same users; a
    count that differs means the two files are from different runs, and an inner join would
    drop the difference without a word.
    """
    scores = pd.read_parquet(scores_path, columns=["user_id", "p"])
    features = pd.read_parquet(
        features_path, columns=["user_id", *[v for v in variables if v != "p"]]
    )
    day = scores.merge(features, on="user_id", how="inner", validate="one_to_one")
    if not len(scores) == len(features) == len(day):
        raise ValueError(
            f"{len(scores):,} scores and {len(features):,} feature rows share {len(day):,} users"
        )
    return day[["user_id", *variables]]


def make_cuts(reference: ArrayLike, n_bins: int = N_BINS) -> np.ndarray:
    """The reference's n_bins quantiles, deduplicated; fit once and frozen.

    Ties merge cuts: on Valid, n_purchases_per_day is 0 for 90.1% of users, so all nine of
    its decile cuts are 0 and it keeps two bins.
    """
    quantiles = np.arange(1, n_bins) / n_bins
    return np.unique(np.quantile(reference, quantiles))


def bin_shares(values: ArrayLike, cuts: ArrayLike) -> np.ndarray:
    """Each bin's share of values, len(cuts) + 1 of them; a value equal to a cut goes left.

    bincount with minlength gives an empty bin a 0; value_counts would leave its row out.
    """
    bins = np.searchsorted(cuts, values, side="left")
    count_per_bin = np.bincount(bins, minlength=len(cuts) + 1)
    return count_per_bin / len(values)


def psi(expected: ArrayLike, actual: ArrayLike, floor: float = FLOOR) -> float:
    """sum (a - e) * ln(a / e) over the bins, with both sets of shares floored first."""
    expected = np.maximum(expected, floor)
    actual = np.maximum(actual, floor)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def check_floor(reference: dict, floor: float = FLOOR, threshold: float = ALERT_PSI) -> float:
    """The term of the smallest reference bin emptying; raises if it falls short of threshold.

    A bin of share e that empties adds (e - floor) * ln(e / floor). If that is below the
    threshold, losing the whole bin alone does not fail alert, and nothing reports it: the
    grid stays green. This turns that silent miss into an error when the reference is built.

    Only bins with e > 0 count. Interpolated quantiles can leave a reference bin empty (cuts
    0.4 and 1.2 among integers), and an empty bin cannot empty; taking its 0 as the smallest
    gives inf and passes the check without looking at the real smallest bin.
    """
    smallest = min(e for r in reference.values() for e in r["expected"] if e > 0)
    worst = (smallest - floor) * np.log(smallest / floor)
    if worst < threshold:
        raise ValueError(
            f"floor {floor} too large: the smallest bin ({smallest:.4f}) emptying adds only "
            f"{worst:.3f}, below {threshold}"
        )
    return float(worst)


def build_reference(
    valid: pd.DataFrame, variables: list[str] = VARIABLES, floor: float = FLOOR
) -> dict:
    """{variable: {"cuts": [...], "expected": [...]}}, as lists so it can be saved as JSON."""
    reference = {}
    for a in variables:
        valid_cut = make_cuts(valid[a])
        valid_shares = bin_shares(valid[a], valid_cut)
        reference[a] = {"cuts": valid_cut.tolist(), "expected": valid_shares.tolist()}
    check_floor(reference, floor)
    return reference


def psi_table(day: pd.DataFrame, reference: dict, floor: float = FLOOR) -> pd.DataFrame:
    """One row per variable in reference: variable, psi; day is cut on the frozen cuts."""
    rows_psi = []
    for a, r in reference.items():
        actual_shares = bin_shares(day[a], r["cuts"])
        res_psi = psi(r["expected"], actual_shares, floor)
        rows_psi.append({"variable": a, "psi": res_psi})
    return pd.DataFrame(rows_psi)
