"""The per-decile analysis on the official split, from Andy's P4 notebook, with Holm and Q.

Run under WSL from the repository root, after scripts/inject_effects.py:
    PTE_ENV=wsl python scripts/heterogeneity_analysis.py

Pre-registered 2026-10-09 (experiment_design.md section 3) before any decile's arms were
compared: raw estimates in every decile (conversion z_test with the unpooled CI, GMV per user
Welch's t), CIs uncorrected, Holm within each metric's 10 deciles, Cochran's Q per metric.

Run on 2026-10-09. It agrees with Andy's notebook to every printed digit, once z_test took
norm.sf (decile 5's p had printed as 0), and with a separate run (chi-square on each decile's
2 x 2 table, Holm by a loop, Q by np.average).

| decile | conv diff pp |    conv p | conv Holm | gmv diff |    gmv p | gmv Holm |
| -----: | -----------: | --------: | --------: | -------: | -------: | -------: |
|      1 |       +0.047 |     0.763 |     0.763 |    +2.82 |    0.489 |    1.000 |
|      2 |       +0.320 |   0.00518 |    0.0155 |    -0.95 |    0.412 |    1.000 |
|      3 |       +0.347 |  2.94e-04 |   0.00118 |    +1.83 |   0.0630 |    0.315 |
|      4 |       +0.439 |  7.98e-08 |  4.79e-07 |    +2.26 |  0.00174 |   0.0139 |
|      5 |       +0.639 |  8.35e-19 |  8.35e-18 |    +2.65 | 5.00e-04 |  0.00450 |
|      6 |       +0.512 |  6.40e-16 |  5.76e-15 |    +2.38 | 1.46e-04 |  0.00146 |
|      7 |       +0.433 |  4.68e-13 |  3.74e-12 |    +0.96 |    0.144 |    0.575 |
|      8 |       +0.340 |  4.34e-10 |  3.04e-09 |    +1.59 |  0.00260 |   0.0182 |
|      9 |       +0.217 |  5.92e-06 |  2.96e-05 |    +1.11 |  0.00450 |   0.0270 |
|     10 |       +0.078 |    0.0427 |    0.0854 |    -0.09 |    0.810 |    1.000 |

Cochran's Q: conversion 81.15 (df 9, p 9.5e-14), GMV 25.95 (df 9, p 0.0021).
All 20 CIs cover their truths. Holm leaves conversion significant in deciles 2-9 (decile 10
drops out: a miss, as its truth is 0.1 pp) and GMV in deciles 4, 5, 6, 8 and 9.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import pandas as pd

from pte.config import load_config, resolve
from pte.experiment.assignment import SALT, assign
from pte.experiment.estimate import observed
from pte.experiment.heterogeneity import DECILES, heterogeneity_analysis
from pte.experiment.inject import injected_effects


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", help="configs/<env>.yaml (default $PTE_ENV, then dev)")
    args = parser.parse_args(argv)
    features_dir = str(resolve(load_config(args.env)["paths"]["features_dir"]))

    outcomes = pd.read_parquet(f"{features_dir}/test/outcomes")
    effects = injected_effects(outcomes).set_index("decile")
    truth = {}
    for d in DECILES:
        truth[("conversion", d)] = effects.loc[d, "conv_pp"] / 100
        truth[("gmv", d)] = effects.loc[d, "gmv_effect"]

    split = assign(outcomes["user_id"])
    assert (split["user_id"].to_numpy() == outcomes["user_id"].to_numpy()).all()
    users = observed(outcomes.assign(arm=split["arm"].to_numpy()))

    tables, q = heterogeneity_analysis(users, truth)
    print(f'salt "{SALT}"; treatment reads y1 and gmv1, control y0 and gmv0')
    with pd.option_context("display.width", 250, "display.max_columns", None):
        for metric, table in tables.items():
            print(f"\n{metric}")
            print(table.to_string(index=False))
        print()
        print(q.to_string(index=False))


if __name__ == "__main__":
    main()
