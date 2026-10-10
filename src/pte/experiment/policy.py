"""Who gets the coupon: the step 9 strategy comparison, from Andy's P4 notebook (2026-10-10).

Decided 2026-10-10 (experiment_design.md section 7), before any strategy was compared: the
coupon (0% installment fee) costs only when used, so every coupon holder who buys costs the
waived fee on all they spend, and the incremental GMV earns the margin m. Per user in a decile:

    profit / m = gmv_lift - k * gmv_with_coupon,  k = f / m
    gmv_with_coupon = gmv_no_coupon + gmv_lift

so only k = f / m matters, reported at 0, 5%, 10% and 15%. Coupons go out a decile at a time,
10% to 100%, in an order set by each strategy:

    by_score        the model's score, decile 1 first
    by_est_lift     the conversion lift this experiment estimated (section 7's "a")
    random          no ranking: the expected value, share x the total, nothing drawn
    by_true_lift    the true conversion lift, the ceiling for extra orders
    by_true_profit  the true profit per user at each k, the ceiling for profit

Estimates only rank; every strategy is judged on the true values (section 7's "ii"). The
narrowed scope (7654d6b) leaves out ranking by the estimated profit and the split-half check.

Totals per decile, not per user, are what add up across deciles: a curve's row j is the first
j deciles summed, and profit and ROI are taken from those sums,

    coupon_cost = k * coupon_gmv,  profit = extra_gmv - coupon_cost,  roi = profit / coupon_cost

with ROI undefined (NaN) at k = 0, where nothing is spent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

K_VALUES = (0, 0.05, 0.10, 0.15)
# The six totals per decile that curve adds up; profit and ROI are taken from their sums
CUMSUM_COLS = [
    "extra_orders_est",
    "extra_orders_true",
    "extra_gmv_est",
    "extra_gmv_true",
    "coupon_gmv_est",
    "coupon_gmv_true",
]


def decile_truth(outcomes: pd.DataFrame) -> pd.DataFrame:
    """One row per decile: its size and the true lifts and GMV levels, from y0, y1, gmv0, gmv1.

    Only a simulation has both outcomes for every user; these judge the strategies, never rank
    the estimated ones.
    """
    g = outcomes.groupby("decile")
    truth = pd.DataFrame(
        {
            "n": g.size(),
            "conv_lift_true": g["y1"].mean() - g["y0"].mean(),
            "gmv_lift_true": g["gmv1"].mean() - g["gmv0"].mean(),
            "gmv_no_coupon_true": g["gmv0"].mean(),
            "gmv_with_coupon_true": g["gmv1"].mean(),
        }
    )
    return truth.reset_index()


def build_decile_table(tables: Mapping[str, pd.DataFrame], truth: pd.DataFrame) -> pd.DataFrame:
    """One row per decile: step 8's estimates, the truth, and the six totals curve adds up.

    tables is heterogeneity_analysis's per-metric tables; truth is decile_truth's. Every table is
    joined on decile, never lined up by position.
    """
    conv = tables["conversion"][["decile", "diff"]].rename(columns={"diff": "conv_lift_est"})
    gmv = tables["gmv"][["decile", "mean_t", "mean_c"]].rename(
        columns={"mean_t": "gmv_treated", "mean_c": "gmv_control"}
    )
    t = conv.merge(gmv, on="decile").merge(truth, on="decile")

    t["extra_orders_est"] = t["n"] * t["conv_lift_est"]
    t["extra_orders_true"] = t["n"] * t["conv_lift_true"]
    t["extra_gmv_est"] = t["n"] * (t["gmv_treated"] - t["gmv_control"])
    t["extra_gmv_true"] = t["n"] * t["gmv_lift_true"]
    t["coupon_gmv_est"] = t["n"] * t["gmv_treated"]
    t["coupon_gmv_true"] = t["n"] * t["gmv_with_coupon_true"]
    return t


def curve(decile_table: pd.DataFrame, key: str) -> pd.DataFrame:
    """Coupons a decile at a time, largest key first (ties: the smaller decile first).

    Row j is the first j deciles: share j / (number of deciles) and the totals summed over them.
    """
    t_sort = decile_table.sort_values([key, "decile"], ascending=[False, True])
    t_sort["share"] = np.arange(1, len(t_sort) + 1) / len(t_sort)
    t_sort[CUMSUM_COLS] = t_sort[CUMSUM_COLS].cumsum()
    return t_sort[["decile", "share"] + CUMSUM_COLS].reset_index(drop=True)


def add_profit(res: pd.DataFrame, k: float, name: str) -> pd.DataFrame:
    """The coupon cost, profit and ROI on the true values, labelled with k and the strategy."""
    res["k"] = k
    res["strategy"] = name
    res["coupon_cost_true"] = k * res["coupon_gmv_true"]
    res["profit_true"] = res["extra_gmv_true"] - res["coupon_cost_true"]
    if k == 0:
        res["roi_true"] = np.nan  # nothing spent, no return on it
    else:
        res["roi_true"] = res["profit_true"] / res["coupon_cost_true"]
    return res


def evaluate(decile_table: pd.DataFrame, key: str, k: float, name: str) -> pd.DataFrame:
    """One strategy's curve at one k: ranked by key, judged on the true values."""
    return add_profit(curve(decile_table, key), k, name)


def random_curve(decile_table: pd.DataFrame, k: float) -> pd.DataFrame:
    """Coupons to a random share of everyone: on average share x each total, a straight line.

    Not a ranking of whole deciles, so not curve: a random share takes that share of every
    decile, and averaging over every order the deciles could come in gives the same line.
    """
    res = pd.DataFrame({"share": np.arange(1, len(decile_table) + 1) / len(decile_table)})
    for col in ["extra_orders_true", "extra_gmv_true", "coupon_gmv_true"]:
        res[col] = res["share"] * decile_table[col].sum()
    return add_profit(res, k, "random")


def policy_curves(decile_table: pd.DataFrame, k_values: Sequence[float] = K_VALUES) -> pd.DataFrame:
    """Every strategy's curve at every k, one row per strategy x k x share.

    by_true_profit ranks by the true profit per user at each k, so its order can change with k;
    the decile table passed in is left as it was.
    """
    t = decile_table.copy()
    t["neg_decile"] = -t["decile"]  # curve ranks largest first; decile 1 has the largest -decile
    results = []
    for k in k_values:
        t["true_profit_per_user"] = t["gmv_lift_true"] - k * t["gmv_with_coupon_true"]
        results.append(evaluate(t, "neg_decile", k, "by_score"))
        results.append(evaluate(t, "conv_lift_est", k, "by_est_lift"))
        results.append(evaluate(t, "conv_lift_true", k, "by_true_lift"))
        results.append(evaluate(t, "true_profit_per_user", k, "by_true_profit"))
        results.append(random_curve(t, k))
    return pd.concat(results, ignore_index=True)


def best_share(curves: pd.DataFrame) -> pd.DataFrame:
    """Per strategy and k, the share with the largest true profit."""
    best = curves.sort_values("profit_true", ascending=False).drop_duplicates(["strategy", "k"])
    return best.sort_values(["strategy", "k"]).reset_index(drop=True)
