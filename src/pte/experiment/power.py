"""Power analysis for the coupon experiment, from Andy's P4 notebook (2026-10-02).

The population and the split are fixed (experiment_design.md sections 1-2), so the analysis
runs backwards: given n, the minimum detectable effect, for all users and for each decile.
Decided 2026-10-02 (experiment_design.md section 4): two-sided alpha 0.05, power 80%.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import norm


def mde(
    p: ArrayLike, n_t: ArrayLike, n_c: ArrayLike, alpha: float = 0.05, power: float = 0.8
) -> np.ndarray:
    """Minimum detectable difference in conversion rate, absolute (0.01 = 1 pp).

    MDE = (z_{1-alpha/2} + z_{power}) * sqrt(p(1-p) * (1/n_t + 1/n_c)), both arms at the
    baseline rate p. The z values come from norm.ppf: 1.95996 + 0.84162 = 2.80159, not the
    rounded 2.80. Takes arrays, so one call covers every decile.
    """
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    p = np.asarray(p, dtype=float)
    se = np.sqrt(p * (1 - p) * (1 / np.asarray(n_t) + 1 / np.asarray(n_c)))
    return (z_alpha + z_beta) * se


def mde_table(
    valid: pd.DataFrame, test: pd.DataFrame, alpha: float = 0.05, power: float = 0.8
) -> pd.DataFrame:
    """The MDE for each decile and for all users, under a 50/50 split of the test population.

    valid has one row per user with decile and label: the baseline r_d is valid's actual rate
    (decided 2026-10-02, experiment_design.md section 4). test has one row per user with
    decile only; its labels are the control arm's outcome, so only its head counts are used.
    The "all" row takes its rate and counts from the users, not from the decile rows.
    mde is the absolute difference, mde_pp the same in percentage points, mde_rel it over r_d.
    """
    valid_by_decile = (
        valid.groupby("decile").agg(n_valid=("label", "size"), r_d=("label", "mean")).reset_index()
    )
    test_by_decile = test.groupby("decile").agg(n_test=("decile", "size")).reset_index()
    table = valid_by_decile.merge(test_by_decile, on="decile", how="inner", validate="one_to_one")
    total = pd.DataFrame(
        [
            {
                "decile": "all",
                "n_valid": len(valid),
                "r_d": valid["label"].mean(),
                "n_test": len(test),
            }
        ]
    )
    table = pd.concat([table, total], ignore_index=True)
    table["n_arm"] = table["n_test"] / 2
    table["mde"] = mde(table["r_d"], table["n_arm"], table["n_arm"], alpha, power)
    table["mde_pp"] = table["mde"] * 100
    table["mde_rel"] = table["mde"] / table["r_d"]
    return table
