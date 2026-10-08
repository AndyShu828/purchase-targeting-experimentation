"""The main analysis's estimates and CIs, from Andy's P4 notebook (2026-10-08).

Decided 2026-10-08 (experiment_design.md section 3), before any y1 was compared with y0:
treatment reads y1 and gmv1, control y0 and gmv0, on the official split with no user dropped.
Significance on conversion comes from z_test (pooled SE, under the null the arms share one
rate); its 95% CI uses the unpooled SE, as a CI must cover effects other than 0:

    se = sqrt(r_t (1 - r_t) / n_t + r_c (1 - r_c) / n_c), CI = diff +/- 1.96 se

r (1 - r) is the variance of one 0/1 outcome, so it holds for conversion only. GMV per user
takes each arm's sample variance (ddof 1) and Welch's t, as the injected buyers make the
treated arm's variance differ:

    se = sqrt(s_t^2 / n_t + s_c^2 / n_c), CI = diff +/- 1.96 se
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import ttest_ind

from pte.experiment.ztest import z_test

Z = 1.96
SEGMENTS = ("decile_1", "all")
METRICS = {"conversion": "y_obs", "gmv": "gmv_obs"}


def observed(users: pd.DataFrame) -> pd.DataFrame:
    """users with y_obs and gmv_obs: y1 and gmv1 in arm "T", y0 and gmv0 in arm "C".

    The one column an experiment sees per user. y0 and y1 are both kept: they hold for any split,
    while y_obs belongs to this one.
    """
    out = users.copy()
    treated = out["arm"] == "T"
    out["y_obs"] = np.where(treated, out["y1"], out["y0"])
    out["gmv_obs"] = np.where(treated, out["gmv1"], out["gmv0"])
    return out


def prop_diff_ci(y_t: ArrayLike, y_c: ArrayLike) -> dict[str, float]:
    """diff, unpooled SE and 95% CI for two arms' 0/1 outcomes, with each arm's rate."""
    y_t = np.asarray(y_t)
    y_c = np.asarray(y_c)
    n_t, n_c = len(y_t), len(y_c)
    r_t, r_c = y_t.sum() / n_t, y_c.sum() / n_c
    diff = r_t - r_c
    se = np.sqrt(r_t * (1 - r_t) / n_t + r_c * (1 - r_c) / n_c)
    return {
        "diff": float(diff),
        "se": float(se),
        "ci_low": float(diff - Z * se),
        "ci_high": float(diff + Z * se),
        "r_t": float(r_t),
        "r_c": float(r_c),
    }


def mean_diff_ci(x_t: ArrayLike, x_c: ArrayLike) -> dict[str, float]:
    """diff of means, Welch SE and 95% CI, with each arm's mean.

    The sample variance is taken with ddof 1 whatever the input: numpy's var divides by n by
    default and pandas' by n - 1, so the same call on an array and a Series would differ.
    """
    x_t = np.asarray(x_t, dtype=float)
    x_c = np.asarray(x_c, dtype=float)
    n_t, n_c = len(x_t), len(x_c)
    mean_t, mean_c = x_t.mean(), x_c.mean()
    diff = mean_t - mean_c
    se = np.sqrt(x_t.var(ddof=1) / n_t + x_c.var(ddof=1) / n_c)
    return {
        "diff": float(diff),
        "se": float(se),
        "ci_low": float(diff - Z * se),
        "ci_high": float(diff + Z * se),
        "mean_t": float(mean_t),
        "mean_c": float(mean_c),
    }


def main_analysis(users: pd.DataFrame, truth: Mapping[tuple[str, str], float]) -> pd.DataFrame:
    """One row per metric and segment: the estimate, its CI and test, and the truth.

    users has one row per user with arm, decile, y_obs and gmv_obs (see observed). truth maps
    (metric, segment) to the true effect, in the same units as diff (a rate, not pp). The
    conversion rows' stat is z_test's z, the GMV rows' Welch's t. rel_diff is diff over the
    control arm's level; err_in_se is (diff - truth) / se and covers whether the CI holds the
    truth, which only a simulation can report.
    """
    rows = []
    for segment in SEGMENTS:
        part = users if segment == "all" else users[users["decile"] == 1]
        treated = part["arm"] == "T"
        for metric, column in METRICS.items():
            x_t, x_c = part.loc[treated, column], part.loc[~treated, column]
            if metric == "conversion":
                est = prop_diff_ci(x_t, x_c)
                test = z_test(x_t, x_c)
                level, stat, p = est["r_c"], test["z"], test["p"]
            else:
                est = mean_diff_ci(x_t, x_c)
                test = ttest_ind(x_t, x_c, equal_var=False)
                level, stat, p = est["mean_c"], float(test.statistic), float(test.pvalue)
            true = truth[(metric, segment)]
            rows.append(
                {
                    "metric": metric,
                    "segment": segment,
                    "n_t": len(x_t),
                    "n_c": len(x_c),
                    "diff": est["diff"],
                    "rel_diff": est["diff"] / level,
                    "se": est["se"],
                    "ci_low": est["ci_low"],
                    "ci_high": est["ci_high"],
                    "stat": stat,
                    "p": p,
                    "truth": true,
                    "err_in_se": (est["diff"] - true) / est["se"],
                    "covers": est["ci_low"] <= true <= est["ci_high"],
                }
            )
    return pd.DataFrame(rows)
