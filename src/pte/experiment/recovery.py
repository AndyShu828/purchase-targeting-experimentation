"""The recovery check, does the analysis get the injected effects back, from Andy's P4 notebook.

Decided 2026-10-11 (experiment_design.md section 8), before any of its splits was computed:
outcomes stays as injected, so every truth is fixed, and only the split changes. Each salt is
a new split; each split gives 22 rows, the 2 "all" rows of main_analysis (its 2 decile-1 rows
repeat heterogeneity_analysis's) and the 20 of heterogeneity_analysis. Over the 1000 salts,
each row is judged on its own:

    coverage: n_covered, the times the CI held the truth, in 936-963
              (the 2.5% and 97.5% quantiles of Binomial(1000, 0.95))
    bias:     z_bias = mean(err_in_se) / (sd(err_in_se) / sqrt(n_runs)),  |z_bias| <= 1.96

Up to 3 of the 22 rows out on either criterion is normal; 4 or more call for a cause first.
The first 1000 salts had 4 rows out on bias; a confirmation run on 1000 new salts, set down
before it ran, brought none of the 4 back (section 8).

Each salt's rows depend on nothing but the outcomes and the salt, so the salts can run in
parallel processes; recovery_simulation returns the same rows, in the same order, either way.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from pte.experiment.assignment import assign
from pte.experiment.estimate import main_analysis, observed
from pte.experiment.heterogeneity import DECILES, heterogeneity_analysis

# Fixed before any recovery split was computed; none is the official or an A/A salt
RECOVERY_SALTS = [f"recovery-{i}" for i in range(1000)]
# Added after the first run, and fixed before the confirmation run computed (section 8)
CONFIRM_SALTS = [f"recovery-{i}" for i in range(1000, 2000)]
# binom.ppf(0.025 / 0.975, 1000, 0.95)
COVERED_RANGE = (936, 963)
Z_LIMIT = 1.96
ROW_COLS = ["salt", "metric", "segment", "diff", "se", "truth", "err_in_se", "covers"]
METRICS = ("conversion", "gmv")


def truths(outcomes: pd.DataFrame) -> tuple[dict, dict]:
    """The true effects from y0, y1, gmv0, gmv1: one dict for main_analysis, one for the deciles.

    The first is keyed (metric, "all" or "decile_1"), the second (metric, decile). Both hold
    for any split, as each user's y0 and y1 do.
    """
    by_decile = outcomes.groupby("decile")
    conv = (by_decile["y1"].mean() - by_decile["y0"].mean()).to_dict()
    gmv = (by_decile["gmv1"].mean() - by_decile["gmv0"].mean()).to_dict()
    truth_main = {
        ("conversion", "all"): (outcomes["y1"] - outcomes["y0"]).mean(),
        ("gmv", "all"): (outcomes["gmv1"] - outcomes["gmv0"]).mean(),
        ("conversion", "decile_1"): conv[1],
        ("gmv", "decile_1"): gmv[1],
    }
    truth_decile = {}
    for d in DECILES:
        truth_decile[("conversion", d)] = conv[d]
        truth_decile[("gmv", d)] = gmv[d]
    return truth_main, truth_decile


def split_and_observe(users: pd.DataFrame, salt: str) -> pd.DataFrame:
    """users split under salt, with arm, y_obs and gmv_obs; users itself is left as it was.

    The arm is joined back on user_id, not attached by position.
    """
    split = assign(users["user_id"], salt)[["user_id", "arm"]]
    out = users.merge(split, on="user_id", how="left", validate="one_to_one")
    assert len(out) == len(users)
    return observed(out)


def recovery_run(
    users: pd.DataFrame,
    salt: str,
    truth_main: Mapping[tuple[str, str], float],
    truth_decile: Mapping[tuple[str, int], float],
) -> pd.DataFrame:
    """One split's 22 rows: "all", then decile_1 to decile_10, for conversion and for GMV."""
    obs = split_and_observe(users, salt)
    res = main_analysis(obs, truth_main)
    parts = [res.loc[res["segment"] == "all"]]
    tables, _ = heterogeneity_analysis(obs, truth_decile)
    for metric in METRICS:
        table = tables[metric]
        table["metric"] = metric
        table["segment"] = "decile_" + table["decile"].astype(str)
        parts.append(table)
    result = pd.concat(parts)
    result["salt"] = salt
    return result[ROW_COLS].reset_index(drop=True)


# Each worker process keeps one copy of the outcomes, sent once when it starts rather than
# once per salt (2.3M rows). Workers are spawned, not forked: forking a process that already
# runs threads (pandas, pyarrow) can deadlock the child
_worker_args: tuple | None = None


def _init_worker(users: pd.DataFrame, truth_main: dict, truth_decile: dict) -> None:
    global _worker_args
    _worker_args = (users, truth_main, truth_decile)


def _run_in_worker(salt: str) -> pd.DataFrame:
    users, truth_main, truth_decile = _worker_args
    return recovery_run(users, salt, truth_main, truth_decile)


def recovery_simulation(
    users: pd.DataFrame, salts: Sequence[str], workers: int = 1
) -> pd.DataFrame:
    """22 rows per salt, in the order of salts, run in one process or across workers.

    workers > 1 spreads the salts over that many processes; map keeps the order of salts, and
    each salt's rows depend on nothing else, so the result is the same as with one.
    """
    truth_main, truth_decile = truths(users)
    if workers == 1:
        rows = [recovery_run(users, s, truth_main, truth_decile) for s in salts]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(users, truth_main, truth_decile),
        ) as pool:
            rows = list(pool.map(_run_in_worker, salts))
    return pd.concat(rows, ignore_index=True)


def default_workers() -> int:
    """All but two of the machine's cores, at least one."""
    return max(1, (os.cpu_count() or 1) - 2)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """One row per metric and segment: bias, the SD of err_in_se, coverage, and both verdicts.

    Rows keep the order they first appear in, then conversion's rows come before GMV's, so
    decile_10 follows decile_9 rather than decile_1.
    """
    s = results.groupby(["metric", "segment"], as_index=False, sort=False).agg(
        truth=("truth", "first"),
        n_runs=("salt", "count"),
        mean_diff=("diff", "mean"),
        mean_err_in_se=("err_in_se", "mean"),
        sd_err_in_se=("err_in_se", "std"),
        n_covered=("covers", "sum"),
    )
    s["bias"] = s["mean_diff"] - s["truth"]
    s["z_bias"] = s["mean_err_in_se"] / (s["sd_err_in_se"] / np.sqrt(s["n_runs"]))
    s["pass_coverage"] = s["n_covered"].between(COVERED_RANGE[0], COVERED_RANGE[1])
    s["pass_bias"] = s["z_bias"].abs() <= Z_LIMIT
    return s.sort_values("metric", kind="stable").reset_index(drop=True)
