"""The per-decile analysis, Holm and Cochran's Q, from Andy's P4 notebook (2026-10-09).

Decided 2026-10-03 and 2026-10-09 (experiment_design.md section 3), before any decile's arms
were compared: each decile is estimated as the main analysis estimates decile 1, raw, with no
CUPED. Conversion takes z_test for p and the unpooled SE for its CI; GMV per user takes
mean_diff_ci and Welch's t. Estimates and CIs stay uncorrected, so step 10 can check that each
CI covers 95% of the time; which deciles are significant is read from Holm's p, one family of
10 per metric. Decile 1's Holm p does not touch the primary result.

Holm steps down through the sorted p values, each against a looser threshold:

    p_(1) <= ... <= p_(m),  adjusted p_(i) = max over j <= i of min(1, (m - j + 1) p_(j))

The running max keeps the stop rule: once one p fails, every larger one fails with it.

Cochran's Q asks whether the 10 effects are all equal, which "decile 5 significant, decile 1
not" does not answer:

    w = 1 / se^2,  diff_bar = sum(w diff) / sum(w),  Q = sum(w (diff - diff_bar)^2) ~ chi2(m - 1)

A significant Q says the effects differ, not which decile is largest (step 9).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import chi2, ttest_ind

from pte.experiment.estimate import mean_diff_ci, prop_diff_ci
from pte.experiment.ztest import z_test

DECILES = range(1, 11)


def conversion_row(users: pd.DataFrame, decile: int, truth: float) -> dict[str, float]:
    """One decile's conversion: the arms' rates, diff, unpooled CI, z_test, and the truth.

    users has decile, arm and y_obs (see estimate.observed); nothing is added to it. truth is
    the decile's true effect as a rate, not pp.
    """
    part = users.loc[users["decile"] == decile]
    y_t = part.loc[part["arm"] == "T", "y_obs"]
    y_c = part.loc[part["arm"] == "C", "y_obs"]
    est = prop_diff_ci(y_t, y_c)
    test = z_test(y_t, y_c)
    return {
        "decile": decile,
        "n_t": len(y_t),
        "n_c": len(y_c),
        "r_t": est["r_t"],
        "r_c": est["r_c"],
        "diff": est["diff"],
        "se": est["se"],
        "ci_low": est["ci_low"],
        "ci_high": est["ci_high"],
        "z": test["z"],
        "p": test["p"],
        "truth": truth,
        "err_in_se": (est["diff"] - truth) / est["se"],
        "covers": est["ci_low"] <= truth <= est["ci_high"],
    }


def gmv_row(users: pd.DataFrame, decile: int, truth: float) -> dict[str, float]:
    """One decile's GMV per user: the arms' means, diff, Welch CI and t, and the truth."""
    part = users.loc[users["decile"] == decile]
    x_t = part.loc[part["arm"] == "T", "gmv_obs"]
    x_c = part.loc[part["arm"] == "C", "gmv_obs"]
    est = mean_diff_ci(x_t, x_c)
    test = ttest_ind(x_t, x_c, equal_var=False)
    return {
        "decile": decile,
        "n_t": len(x_t),
        "n_c": len(x_c),
        "mean_t": est["mean_t"],
        "mean_c": est["mean_c"],
        "diff": est["diff"],
        "se": est["se"],
        "ci_low": est["ci_low"],
        "ci_high": est["ci_high"],
        "t": float(test.statistic),
        "p": float(test.pvalue),
        "truth": truth,
        "err_in_se": (est["diff"] - truth) / est["se"],
        "covers": est["ci_low"] <= truth <= est["ci_high"],
    }


def holm_adjust(p: ArrayLike) -> np.ndarray:
    """Holm's adjusted p values, in the order given.

    The p values go into a table with their position, are sorted, adjusted and sorted back by
    position, so putting them back does not rest on two arrays lining up.
    """
    p = np.asarray(p, dtype=float)
    m = len(p)
    t = pd.DataFrame({"pos": range(m), "p": p}).sort_values("p")
    t["rank"] = np.arange(1, m + 1)
    t["p_adj"] = (t["p"] * (m - t["rank"] + 1)).clip(upper=1).cummax()
    return t.sort_values("pos")["p_adj"].to_numpy()


def cochran_q(diff: ArrayLike, se: ArrayLike) -> dict[str, float]:
    """Cochran's Q for equal effects across segments, its degrees of freedom and p."""
    diff = np.asarray(diff, dtype=float)
    se = np.asarray(se, dtype=float)
    w = 1 / se**2
    diff_bar = (w * diff).sum() / w.sum()
    q = (w * (diff - diff_bar) ** 2).sum()
    df = len(diff) - 1
    return {"Q": float(q), "df": df, "p": float(chi2.sf(q, df))}


def heterogeneity_analysis(
    users: pd.DataFrame, truth: Mapping[tuple[str, int], float]
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Per metric, one row per decile with Holm's p; and one row of Cochran's Q per metric.

    truth maps (metric, decile) to the true effect, conversion as a rate. Holm runs within each
    metric's 10 deciles, never over the 20 together.
    """
    rows = {"conversion": conversion_row, "gmv": gmv_row}
    tables = {}
    q_rows = []
    for metric, row in rows.items():
        table = pd.DataFrame([row(users, d, truth[(metric, d)]) for d in DECILES])
        table["p_holm"] = holm_adjust(table["p"])
        tables[metric] = table
        q_rows.append({"metric": metric, **cochran_q(table["diff"], table["se"])})
    return tables, pd.DataFrame(q_rows)
