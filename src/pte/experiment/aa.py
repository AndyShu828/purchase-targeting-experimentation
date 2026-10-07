"""The A/A simulation, from Andy's P4 notebook (2026-10-06).

Decided 2026-10-06 (experiment_design.md section 2), before any A/A split was computed: 1000
salts "aa-0" .. "aa-999", each a new split of the test set; both arms read y0, so the true
effect is zero and every significant result is a false positive. Each split is tested twice
with the z test, over all users and within decile 1 (the primary test's decile).

Each metric passes on two criteria: its count of p < 0.05 in 37-64, the 2.5% and 97.5%
quantiles of Binomial(1000, 0.05); and a KS test of its 1000 p values against Uniform(0, 1)
at p >= 0.05. Under the null p = 2 (1 - Phi(|z|)) is uniform, as P(p <= a) = a for any a.
diff_sd, the spread of the 1000 diffs, is a diagnostic against the formula's SE, not a
criterion.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from scipy.stats import kstest

from pte.experiment.assignment import SALT, assign
from pte.experiment.ztest import z_test

# Fixed before any A/A split was computed; none is the official salt
AA_SALTS = [f"aa-{i}" for i in range(1000)]
assert SALT not in AA_SALTS
ALPHA = 0.05
# binom.ppf(0.025 / 0.975, 1000, 0.05); a correct test falls outside with probability 4.19%
COUNT_RANGE = (37, 64)
KS_ALPHA = 0.05


def aa_run(users: pd.DataFrame, salt: str) -> list[dict]:
    """One split of users (user_id, decile, y0): a row for all users and one for decile 1.

    The split is joined back on user_id, not attached by position. Both arms read y0.
    """
    split = users.merge(
        assign(users["user_id"], salt), on="user_id", how="inner", validate="one_to_one"
    )
    assert len(split) == len(users)
    rows = []
    for metric, part in [("all", split), ("decile_1", split.loc[split["decile"] == 1])]:
        res = z_test(
            part.loc[part["arm"] == "T", "y0"],
            part.loc[part["arm"] == "C", "y0"],
        )
        res["salt"] = salt
        res["metric"] = metric
        rows.append(res)
    return rows


def aa_simulation(users: pd.DataFrame, salts: Sequence[str] = AA_SALTS) -> pd.DataFrame:
    """Two rows per salt: diff, z, p, salt, metric."""
    rows = []
    for salt in salts:
        rows.extend(aa_run(users, salt))
    return pd.DataFrame(rows)


def aa_summary(results: pd.DataFrame, count_range: tuple[int, int] = COUNT_RANGE) -> pd.DataFrame:
    """One row per metric: the count of p < 0.05, the KS p, both verdicts and diff_sd.

    Each metric is judged on its own rows only; the two metrics' counts are never pooled.
    """
    rows = []
    for metric in ["all", "decile_1"]:
        part = results.loc[results["metric"] == metric]
        cnt = int((part["p"] < ALPHA).sum())
        ks_p = float(kstest(part["p"], "uniform").pvalue)
        rows.append(
            {
                "metric": metric,
                "n": len(part),
                "cnt": cnt,
                "pass_count": count_range[0] <= cnt <= count_range[1],
                "ks_p": ks_p,
                "pass_ks": ks_p >= KS_ALPHA,
                "diff_sd": float(part["diff"].std()),
            }
        )
    return pd.DataFrame(rows)
