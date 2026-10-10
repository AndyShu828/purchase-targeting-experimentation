import numpy as np
import pandas as pd
import pytest

from pte.experiment.policy import (
    CUMSUM_COLS,
    best_share,
    build_decile_table,
    curve,
    decile_truth,
    evaluate,
    policy_curves,
    random_curve,
)

# Three deciles of toy users: decile 3's rows first, so lining tables up by position would show
OUTCOMES = pd.DataFrame(
    {
        "decile": [3, 3, 1, 1, 1, 1, 2, 2],
        "y0": [0, 0, 1, 0, 0, 0, 0, 1],
        "y1": [1, 0, 1, 1, 0, 0, 1, 1],
        "gmv0": [0.0, 0, 100, 0, 0, 0, 0, 20],
        "gmv1": [10.0, 0, 100, 40, 0, 0, 30, 20],
    }
)


def test_decile_truth_takes_both_outcomes_of_every_user():
    truth = decile_truth(OUTCOMES).set_index("decile")
    assert truth["n"].to_dict() == {1: 4, 2: 2, 3: 2}
    assert truth.loc[1, "conv_lift_true"] == pytest.approx(0.25)  # 2 of 4 buy, then 1 of 4
    assert truth.loc[1, "gmv_no_coupon_true"] == pytest.approx(25.0)
    assert truth.loc[1, "gmv_with_coupon_true"] == pytest.approx(35.0)
    assert truth.loc[1, "gmv_lift_true"] == pytest.approx(10.0)
    lift_plus_base = truth["gmv_no_coupon_true"] + truth["gmv_lift_true"]
    assert truth["gmv_with_coupon_true"].to_numpy() == pytest.approx(lift_plus_base.to_numpy())


def toy_tables():
    # step 8's tables, rows in an order of their own: decile 2, 3, 1 and 3, 1, 2
    conversion = pd.DataFrame({"decile": [2, 3, 1], "diff": [0.6, 0.2, 0.1], "se": [9.0] * 3})
    gmv = pd.DataFrame(
        {
            "decile": [3, 1, 2],
            "mean_t": [6.0, 38.0, 31.0],
            "mean_c": [4.0, 30.0, 15.0],
            "diff": [7.0] * 3,
        }
    )
    return {"conversion": conversion, "gmv": gmv}


def test_build_decile_table_joins_on_decile_and_adds_the_six_totals():
    t = build_decile_table(toy_tables(), decile_truth(OUTCOMES)).set_index("decile")
    # each estimate stays with its decile although every table came in a different row order
    assert t["conv_lift_est"].to_dict() == {1: 0.1, 2: 0.6, 3: 0.2}
    assert t["gmv_treated"].to_dict() == {1: 38.0, 2: 31.0, 3: 6.0}
    assert t["gmv_control"].to_dict() == {1: 30.0, 2: 15.0, 3: 4.0}
    assert "diff" not in t.columns  # step 8's two diffs are renamed, never both carried along
    row = t.loc[1]  # n 4: the estimates and truths times the decile's size
    assert row["extra_orders_est"] == pytest.approx(0.4)
    assert row["extra_orders_true"] == pytest.approx(1.0)
    assert row["extra_gmv_est"] == pytest.approx(32.0)
    assert row["extra_gmv_true"] == pytest.approx(40.0)
    assert row["coupon_gmv_est"] == pytest.approx(152.0)
    assert row["coupon_gmv_true"] == pytest.approx(140.0)


DECILE_TABLE = pd.DataFrame(
    {
        "decile": [1, 2, 3],
        "key": [0.1, 0.5, 0.5],  # deciles 2 and 3 tie
        **{col: [1.0, 10.0, 100.0] for col in CUMSUM_COLS},
    }
)


def test_curve_ranks_largest_first_breaks_ties_by_decile_and_sums():
    res = curve(DECILE_TABLE, "key")
    assert res["decile"].tolist() == [2, 3, 1]
    assert res["share"].to_numpy() == pytest.approx([1 / 3, 2 / 3, 1.0])
    for col in CUMSUM_COLS:
        assert res[col].tolist() == pytest.approx([10.0, 110.0, 111.0])
    assert list(res.index) == [0, 1, 2]
    assert DECILE_TABLE[CUMSUM_COLS[0]].tolist() == [1.0, 10.0, 100.0]  # input left as it was


def test_evaluate_takes_profit_and_roi_from_the_summed_totals():
    t = DECILE_TABLE.assign(extra_gmv_true=[3.0, 4.0, 5.0], coupon_gmv_true=[20.0, 30.0, 40.0])
    res = evaluate(t, "key", 0.05, "toy")
    # deciles 2, 3, 1: extra GMV 4, 9, 12; coupon GMV 30, 70, 90; cost 1.5, 3.5, 4.5
    assert res["coupon_cost_true"].tolist() == pytest.approx([1.5, 3.5, 4.5])
    assert res["profit_true"].tolist() == pytest.approx([2.5, 5.5, 7.5])
    assert res["roi_true"].tolist() == pytest.approx([2.5 / 1.5, 5.5 / 3.5, 7.5 / 4.5])
    assert (res["k"] == 0.05).all() and (res["strategy"] == "toy").all()


def test_evaluate_leaves_roi_undefined_when_coupons_cost_nothing():
    res = evaluate(DECILE_TABLE, "key", 0, "toy")
    assert res["roi_true"].isna().all()
    assert res["profit_true"].tolist() == pytest.approx(res["extra_gmv_true"].tolist())


def test_random_curve_takes_each_share_of_every_total():
    res = random_curve(DECILE_TABLE, 0.1)
    assert res["share"].to_numpy() == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert res["extra_orders_true"].tolist() == pytest.approx([37.0, 74.0, 111.0])
    assert res["coupon_cost_true"].tolist() == pytest.approx([3.7, 7.4, 11.1])
    assert res["profit_true"].tolist() == pytest.approx([33.3, 66.6, 99.9])
    assert (res["strategy"] == "random").all()


def toy_decile_table():
    return build_decile_table(toy_tables(), decile_truth(OUTCOMES))


def test_policy_curves_has_every_strategy_k_and_share():
    t = toy_decile_table()
    curves = policy_curves(t, k_values=(0, 0.5))
    assert len(curves) == 5 * 2 * 3
    assert set(curves["strategy"]) == {
        "by_score",
        "by_est_lift",
        "random",
        "by_true_lift",
        "by_true_profit",
    }
    assert curves.index.is_unique
    assert "neg_decile" not in t.columns and "true_profit_per_user" not in t.columns
    # everyone gets a coupon at share 1, so every strategy ends at the same totals
    end = curves.loc[curves["share"] == 1.0]
    for col in ["extra_orders_true", "extra_gmv_true", "coupon_gmv_true", "profit_true"]:
        for k in (0, 0.5):
            assert end.loc[end["k"] == k, col].to_numpy() == pytest.approx(
                [end.loc[end["k"] == k, col].iloc[0]] * 5
            )


def first_decile(curves, strategy, k):
    return curves.loc[(curves["strategy"] == strategy) & (curves["k"] == k), "decile"].iloc[0]


def test_policy_curves_orders_each_strategy_by_its_own_key():
    curves = policy_curves(toy_decile_table(), k_values=(0, 0.8))
    assert first_decile(curves, "by_score", 0) == 1
    assert first_decile(curves, "by_est_lift", 0) == 2  # conv_lift_est 0.6
    assert first_decile(curves, "by_true_lift", 0) == 2  # conv_lift_true 0.5
    # true profit per user = lift - k x with-coupon GMV. At k = 0 it is the lift (decile 2's 15
    # leads); at k = 0.8 decile 1 has 10 - 28, decile 2 has 15 - 20 and decile 3 has 5 - 4
    assert first_decile(curves, "by_true_profit", 0) == 2
    assert first_decile(curves, "by_true_profit", 0.8) == 3


def test_best_share_picks_the_most_profitable_row_per_strategy_and_k():
    curves = pd.DataFrame(
        {
            "strategy": ["a", "a", "a", "a", "b"],
            "k": [0.1, 0.1, 0.1, 0.2, 0.1],
            "share": [0.1, 0.2, 0.3, 0.1, 0.1],
            "profit_true": [5.0, 9.0, 7.0, -1.0, 3.0],
        }
    )
    best = best_share(curves)
    assert best[["strategy", "k", "share"]].values.tolist() == [
        ["a", 0.1, 0.2],
        ["a", 0.2, 0.1],
        ["b", 0.1, 0.1],
    ]
    assert np.isclose(best["profit_true"], [9.0, -1.0, 3.0]).all()
