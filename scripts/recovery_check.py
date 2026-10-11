"""The recovery check on 1000 re-splits of the test set, from Andy's P4 notebook.

Run under WSL from the repository root, after scripts/inject_effects.py:
    PTE_ENV=wsl python scripts/recovery_check.py --batch first
    PTE_ENV=wsl python scripts/recovery_check.py --batch confirm

Pre-registered 2026-10-11 (experiment_design.md section 8) before any of its splits was
computed: outcomes stays as injected and only the split changes, 22 rows per split (the "all"
rows of main_analysis and the 20 of heterogeneity_analysis), each row judged on coverage
(936-963 of 1000) and bias (|z_bias| <= 1.96). "first" runs RECOVERY_SALTS and writes
<features_dir>/test/recovery; "confirm" runs CONFIRM_SALTS, set down after the first run put 4
rows out on bias and before it ran, and writes <features_dir>/test/recovery_confirm. Files are
written only when the whole batch runs. The salts run in parallel processes (--workers).

Run on 2026-10-11 on 12 workers: 681 s for "first", 610 s for "confirm" (one process took
40-46 minutes, so about 4 times faster, not 12). Both files equal Andy's notebook's, row for
row in the same order (DataFrame.equals), and a third run of Claude's own, one process.

| run     | coverage rows out (936-963)    | bias rows out (abs z_bias > 1.96)                    |
| :------ | :----------------------------- | :--------------------------------------------------- |
| first   | 0                              | 4: conversion deciles 2, 3; GMV all, decile 1        |
| confirm | 2: GMV deciles 5, 7 (970, 969) | 0                                                    |

The 4 first-run rows z_bias +2.05, -2.27, -2.23, -2.59 came back at -0.97, +0.88, -1.24,
-1.50: none past 1.96 with the same sign, so none came back; the 2 confirm rows out covered
too often, the conservative side. Over both runs SD of err_in_se 0.925-1.042, coverage
940-970. Read in experiment_design.md section 8.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

import pandas as pd

from pte.config import load_config, resolve
from pte.experiment.recovery import (
    CONFIRM_SALTS,
    RECOVERY_SALTS,
    default_workers,
    recovery_simulation,
    summarize,
)

BATCHES = {
    "first": (RECOVERY_SALTS, "recovery"),
    "confirm": (CONFIRM_SALTS, "recovery_confirm"),
}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", help="configs/<env>.yaml (default $PTE_ENV, then dev)")
    parser.add_argument("--batch", choices=BATCHES, default="first")
    parser.add_argument("--n-salts", type=int, help="first n salts of the batch only")
    parser.add_argument("--workers", type=int, default=default_workers())
    args = parser.parse_args(argv)
    features_dir = str(resolve(load_config(args.env)["paths"]["features_dir"]))

    salts, name = BATCHES[args.batch]
    if args.n_salts is not None:
        salts = salts[: args.n_salts]
    users = pd.read_parquet(f"{features_dir}/test/outcomes")
    start = time.time()
    results = recovery_simulation(users, salts, workers=args.workers)
    print(f"{len(salts)} salts on {args.workers} workers in {time.time() - start:.0f} s")
    if len(salts) == len(BATCHES[args.batch][0]):
        results.to_parquet(f"{features_dir}/test/{name}", index=False)

    summary = summarize(results)
    cols = ["metric", "segment", "n_covered", "mean_err_in_se", "sd_err_in_se", "z_bias"]
    print(summary[cols].round(3).to_string(index=False))
    for verdict in ["pass_coverage", "pass_bias"]:
        out = summary.loc[~summary[verdict], ["metric", "segment"]]
        print(f"{verdict}: {len(out)} of 22 rows out", out.values.tolist())


if __name__ == "__main__":
    main()
