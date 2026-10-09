"""The coupon experiment's main analysis on the official split, from Andy's P4 notebook.

Run under WSL from the repository root, after scripts/inject_effects.py:
    PTE_ENV=wsl python scripts/main_analysis.py

Splits the test set under the official salt; treatment reads y1 and gmv1, control y0 and gmv0
(experiment_design.md section 3). The primary test is decile 1's conversion difference by
z_test; all users' conversion and both segments' GMV per user are secondary. The true effects
come from the same outcomes table (injected_effects), the values section 5 records.

Run on 2026-10-08. It agrees with Andy's notebook to the digit except err_in_se in the sixth
digit, as the notebook took the truths rounded to six significant digits; and with a separate
run that hashed the ids itself and tested conversion with chi-square on the 2 x 2 table.
All users' conversion p was the exception: z_test returned 0 then (1 - Phi(|z|) rounds to 0),
and 9.51e-36 is the chi-square's; since z_test takes norm.sf (2026-10-09) it returns 9.51e-36.

| metric     | segment  |     diff | rel_diff |    ci_low |   ci_high |   stat |        p | err_se |
| :--------- | :------- | -------: | -------: | --------: | --------: | -----: | -------: | -----: |
| conversion | decile_1 | 0.000467 |   0.0028 | -0.002574 |  0.003508 |  0.301 |    0.763 | -0.345 |
| gmv        | decile_1 | 2.824516 |   0.0192 | -5.173901 | 10.822933 |  0.692 |    0.489 |  0.478 |
| conversion | all      | 0.003444 |   0.0778 |  0.002903 |  0.003985 | 12.481 | 9.51e-36 |  0.522 |
| gmv        | all      | 1.523330 |   0.0576 |  0.608740 |  2.437919 |  3.265 |   0.0011 |  0.274 |

Truths: conversion 0.001002 in decile 1 and 0.003300 over all users, GMV 0.875912 and 1.395263.

Every CI covers its truth. The primary test is null, as designed (power 8.5%): decile 1 shows
no detectable effect, and its CI excludes effects above 0.351 pp.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import pandas as pd

from pte.config import load_config, resolve
from pte.experiment.assignment import SALT, assign
from pte.experiment.estimate import main_analysis, observed
from pte.experiment.inject import injected_effects


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", help="configs/<env>.yaml (default $PTE_ENV, then dev)")
    args = parser.parse_args(argv)
    features_dir = str(resolve(load_config(args.env)["paths"]["features_dir"]))

    outcomes = pd.read_parquet(f"{features_dir}/test/outcomes")
    effects = injected_effects(outcomes).set_index("decile")
    truth = {}
    for segment, decile in [("decile_1", 1), ("all", "all")]:
        truth[("conversion", segment)] = effects.loc[decile, "conv_pp"] / 100
        truth[("gmv", segment)] = effects.loc[decile, "gmv_effect"]

    split = assign(outcomes["user_id"])
    assert (split["user_id"].to_numpy() == outcomes["user_id"].to_numpy()).all()
    users = observed(outcomes.assign(arm=split["arm"].to_numpy()))

    table = main_analysis(users, truth)
    print(f'salt "{SALT}"; treatment reads y1 and gmv1, control y0 and gmv0')
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.to_string(index=False))
    primary = table[(table["metric"] == "conversion") & (table["segment"] == "decile_1")].iloc[0]
    print(
        f"primary: diff {primary['diff'] * 100:+.4f} pp, p {primary['p']:.4f}, "
        f"95% CI [{primary['ci_low'] * 100:.4f}, {primary['ci_high'] * 100:.4f}] pp"
    )


if __name__ == "__main__":
    main()
