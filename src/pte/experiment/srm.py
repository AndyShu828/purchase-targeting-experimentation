"""The sample ratio mismatch (SRM) check, from Andy's P4 notebook (2026-10-05).

Decided 2026-10-04 (experiment_design.md section 2): a chi-square test of the two arms' head
counts against the designed 50/50, df 1, alarms at p < 0.001 over all users; the same test
within each decile is a diagnostic only and raises no alarm. Only head counts are read, never
an outcome.

chi_sq = sum over the arms of (O - E)^2 / E, with E = N x the arm's designed share. With two
arms the counts' deviations are +d and -d, so chi_sq = d^2 / (N x share x (1 - share)) = z^2,
and the chi-square p (right tail, df 1) is the two-sided z test's p.
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import chi2

SRM_ALPHA = 0.001
TREATED_SHARE = 0.5


def srm_test(n_t: int, n_c: int, treated_share: float = TREATED_SHARE) -> dict[str, float]:
    """Counts, expected counts, chi-square and p for one pair of arm head counts."""
    n = n_t + n_c
    e_t = n * treated_share
    e_c = n * (1 - treated_share)
    chi_sq = (n_t - e_t) ** 2 / e_t + (n_c - e_c) ** 2 / e_c
    return {
        "n_t": n_t,
        "n_c": n_c,
        "e_t": e_t,
        "e_c": e_c,
        "share_t": n_t / n,
        "chi_sq": chi_sq,
        "p": float(chi2.sf(chi_sq, df=1)),
    }


def srm_table(split: pd.DataFrame, treated_share: float = TREATED_SHARE) -> pd.DataFrame:
    """One row per decile, then an "all" row; only the all row's p is checked for the alarm.

    split has one row per user with arm ("T" or "C") and decile. Counts are taken by the
    arm's label, never by position: value_counts orders by count, so its first row is
    whichever arm is larger. alarm is p < SRM_ALPHA on the all row and None on the deciles,
    which are a diagnostic.
    """

    def row(decile, arms: pd.Series) -> dict:
        counts = arms.value_counts()
        return {"decile": decile, **srm_test(counts.get("T", 0), counts.get("C", 0), treated_share)}

    rows = [row(decile, g["arm"]) for decile, g in split.groupby("decile")]
    rows.append(row("all", split["arm"]))
    table = pd.DataFrame(rows)
    table["alarm"] = None
    table.loc[table["decile"] == "all", "alarm"] = table["p"].iloc[-1] < SRM_ALPHA
    return table
