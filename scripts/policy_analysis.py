"""The step 9 strategy comparison on the official split, from Andy's P4 notebook.

Run under WSL from the repository root, after scripts/inject_effects.py:
    PTE_ENV=wsl python scripts/policy_analysis.py

Pre-registered 2026-10-10 (experiment_design.md section 7) before any strategy was compared:
the coupon costs k x the GMV of everyone who uses it, k = 0, 5%, 10%, 15%; coupons go out a
decile at a time, 10% to 100%; by_score, by_est_lift, random and the true-value ceilings,
every one judged on the true values. Profit is profit / m, in GMV units with no currency.

Run on 2026-10-10. It agrees with Andy's notebook to every printed digit, and its 10% rows with
a separate run summing y0, y1, gmv0 and gmv1 by decile without any of these functions.

Coupons to 10%: the decile sent, extra orders, profit / m at k = 0 / 5% / 10% / 15%

| strategy       | decile | orders | k = 0    | k = 5%     | k = 10%    | k = 15%    |
| -------------- | -----: | -----: | -------: | ---------: | ---------: | ---------: |
| by_score       |      1 |    231 | +201,957 | -1,516,714 | -3,235,386 | -4,954,057 |
| by_est_lift    |      5 |  1,383 | +513,934 |   +358,876 |   +203,818 |    +48,761 |
| random         |      - |  760.8 | +321,703 |       -541 |   -322,785 |   -645,029 |
| by_true_profit |  5 (7) |  1,383 | +513,934 |   +358,876 |   +203,818 |    +87,491 |

and ROI at k = 5% / 10% / 15%: by_score -0.88 / -0.94 / -0.96, by_est_lift +2.31 / +0.66 /
+0.10, random -0.00 / -0.50 / -0.67, by_true_profit +2.31 / +0.66 / +0.25 (decile 7 at 15%,
922 orders).

The most profitable share, by_est_lift against the ceiling: 100% / 100% at k = 0, 60% / 80%
(1,583,637 against 1,723,419) at 5%, 40% / 60% (686,281 / 821,369) at 10%, 20% / 40%
(115,120 / 206,128) at 15%. Once coupons cost anything, by_score loses money at every share.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import pandas as pd

from pte.config import load_config, resolve
from pte.experiment.assignment import SALT, assign
from pte.experiment.estimate import observed
from pte.experiment.heterogeneity import DECILES, heterogeneity_analysis
from pte.experiment.policy import best_share, build_decile_table, decile_truth, policy_curves


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", help="configs/<env>.yaml (default $PTE_ENV, then dev)")
    args = parser.parse_args(argv)
    features_dir = str(resolve(load_config(args.env)["paths"]["features_dir"]))

    outcomes = pd.read_parquet(f"{features_dir}/test/outcomes")
    truth = decile_truth(outcomes)
    by_decile = truth.set_index("decile")
    truth_by_metric = {}
    for d in DECILES:
        truth_by_metric[("conversion", d)] = by_decile.loc[d, "conv_lift_true"]
        truth_by_metric[("gmv", d)] = by_decile.loc[d, "gmv_lift_true"]

    split = assign(outcomes["user_id"])
    assert (split["user_id"].to_numpy() == outcomes["user_id"].to_numpy()).all()
    users = observed(outcomes.assign(arm=split["arm"].to_numpy()))
    tables, _ = heterogeneity_analysis(users, truth_by_metric)

    curves = policy_curves(build_decile_table(tables, truth))
    cols = ["strategy", "k", "share", "decile", "extra_orders_true", "profit_true", "roi_true"]
    print(f'salt "{SALT}"; estimates rank, true values judge; profit is profit / m')
    with pd.option_context(
        "display.width", 250, "display.max_rows", None, "display.float_format", "{:,.4f}".format
    ):
        print("\ncoupons to 10%")
        print(curves.loc[curves["share"] == 0.1, cols].to_string(index=False))
        print("\nthe most profitable share per strategy and k")
        print(best_share(curves)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
