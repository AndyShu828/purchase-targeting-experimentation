"""Choosing the CUPED covariate, from Andy's P4 notebook (2026-10-04).

Decided 2026-10-03 (experiment_design.md section 6): the covariate for conversion is the
candidate with the largest rho squared with the label on valid, so the test outcomes play no
part in the choice. CUPED leaves Var(Y) x (1 - rho^2), so the MDE shrinks by sqrt(1 - rho^2).

The per-decile analysis compares users within a decile, so the deciles have already absorbed
the differences between them; what CUPED can still use there is the correlation within each
decile, which correlations_by_decile measures. Where a candidate is constant within a decile
(no one in decile 10 bought before the cutoff) its SD is 0 and the correlation is NaN.

The CUPED estimate itself, from Andy's P4 notebook (2026-10-09), as pre-registered that day
before any of it was computed on test (experiment_design.md section 6):

    theta = Cov(X, Y) / Var(X),  Y_cv = Y - theta (X - mean(X)),  both arms pooled

Y_cv averages to the mean of Y over the pooled users, but not within an arm: the diff moves by
-theta x (mean X_T - mean X_C), the part of the arms' chance imbalance X can explain. Y_cv is
no longer 0/1, so every row, conversion too, takes mean_diff_ci and Welch's t.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
from scipy.stats import ttest_ind

from pte.experiment.estimate import SEGMENTS, mean_diff_ci

# metric -> (covariate X, outcome Y); decided 2026-10-03 and 2026-10-04 (section 6)
COVARIATES = {"conversion": ("p", "y_obs"), "gmv": ("total_spend_per_day", "gmv_obs")}

# The prediction: rho squared on valid, written down before CUPED touched test (section 6;
# conversion 2026-10-04, GMV 2026-10-09)
PRED_RHO2 = {
    ("conversion", "decile_1"): 0.112,
    ("conversion", "all"): 0.1099,
    ("gmv", "decile_1"): 0.1942,
    ("gmv", "all"): 0.1675,
}


def covariate_correlations(
    df: pd.DataFrame, candidates: Sequence[str], outcome: str = "label"
) -> pd.DataFrame:
    """One row per candidate: rho with the outcome over all users, and rho squared."""
    rows = []
    for c in candidates:
        rho = df[c].corr(df[outcome])
        rows.append({"candidate": c, "rho": rho, "rho2": rho**2})
    return pd.DataFrame(rows)


def correlations_by_decile(
    df: pd.DataFrame, candidates: Sequence[str], outcome: str = "label"
) -> pd.DataFrame:
    """One row per decile, one column per candidate: rho with the outcome within the decile."""
    cols = {}
    for c in candidates:
        # c=c binds this round's name into the lambda; apply runs at once, so it is the same
        # either way, but a lambda that reads a loop variable later would see only the last one
        cols[c] = df.groupby("decile").apply(
            lambda g, c=c: g[c].corr(g[outcome]), include_groups=False
        )
    return pd.DataFrame(cols)


def cuped_row(part: pd.DataFrame, x: str, y: str, true_value: float) -> dict[str, float]:
    """CUPED on one segment: theta and Y_cv over both arms, then the arms compared on Y_cv.

    part has arm, x and y; nothing is added to it. var_reduction is 1 - Var(Y_cv) / Var(Y);
    with theta fitted on the same users it equals rho^2 exactly. se_ratio is the SE over the
    SE that mean_diff_ci gives on the raw y, the same formula on both sides.
    """
    if part[x].isna().any():
        # pre-registered: no fill rule, stop instead of choosing one after seeing the data
        raise ValueError(f"covariate {x} has missing values")
    theta = part[x].cov(part[y]) / part[x].var()
    y_cv = part[y] - theta * (part[x] - part[x].mean())
    treated = part["arm"] == "T"
    est = mean_diff_ci(y_cv[treated], y_cv[~treated])
    raw = mean_diff_ci(part.loc[treated, y], part.loc[~treated, y])
    p = ttest_ind(y_cv[treated], y_cv[~treated], equal_var=False).pvalue
    return {
        "theta": float(theta),
        "var_reduction": float(1 - y_cv.var() / part[y].var()),
        "diff_raw": raw["diff"],
        "diff": est["diff"],
        "se_raw": raw["se"],
        "se": est["se"],
        "se_ratio": est["se"] / raw["se"],
        "ci_low": est["ci_low"],
        "ci_high": est["ci_high"],
        "p": float(p),
        "truth": true_value,
        "err_in_se": (est["diff"] - true_value) / est["se"],
        "covers": est["ci_low"] <= true_value <= est["ci_high"],
    }


def cuped_analysis(users: pd.DataFrame, truth: Mapping[tuple[str, str], float]) -> pd.DataFrame:
    """One row per metric and segment, in the notebook's order, with the predicted rho^2.

    users is main_analysis' input plus the covariates. theta is fitted within each segment:
    decile 1 on its own users, all users on everyone.
    """
    rows = []
    for metric, (x, y) in COVARIATES.items():
        for segment in SEGMENTS:
            part = users if segment == "all" else users[users["decile"] == 1]
            row = cuped_row(part, x, y, truth[(metric, segment)])
            row["metric"] = metric
            row["segment"] = segment
            row["pred_rho2"] = PRED_RHO2[(metric, segment)]
            rows.append(row)
    return pd.DataFrame(rows)
