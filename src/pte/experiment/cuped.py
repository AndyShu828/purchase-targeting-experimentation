"""Choosing the CUPED covariate, from Andy's P4 notebook (2026-10-04).

Decided 2026-10-03 (experiment_design.md section 6): the covariate for conversion is the
candidate with the largest rho squared with the label on valid, so the test outcomes play no
part in the choice. CUPED leaves Var(Y) x (1 - rho^2), so the MDE shrinks by sqrt(1 - rho^2).

The per-decile analysis compares users within a decile, so the deciles have already absorbed
the differences between them; what CUPED can still use there is the correlation within each
decile, which correlations_by_decile measures. Where a candidate is constant within a decile
(no one in decile 10 bought before the cutoff) its SD is 0 and the correlation is NaN.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


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
