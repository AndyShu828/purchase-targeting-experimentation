import numpy as np
import pandas as pd
import pytest
from scipy.stats import ttest_ind

from pte.experiment.estimate import mean_diff_ci, prop_diff_ci
from pte.experiment.heterogeneity import (
    cochran_q,
    conversion_row,
    gmv_row,
    heterogeneity_analysis,
    holm_adjust,
)
from pte.experiment.ztest import z_test


def test_holm_adjust_matches_the_hand_worked_toys():
    # Andy's notebook self-checks: the running max lifts the 0.04 to 0.06, and 0.6 rises to 1.0
    assert holm_adjust([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm_adjust([0.5, 0.6]) == pytest.approx([1.0, 1.0])
    # 0.5 x 2 is 1.0 on the dot, so the cap only shows when the product passes 1: 0.6 x 2
    assert holm_adjust([0.6, 0.7]) == pytest.approx([1.0, 1.0])


def test_holm_adjust_is_not_bonferroni():
    # the session's worked example: Holm keeps all four, Bonferroni (x 4) only two
    adjusted = holm_adjust([0.01, 0.015, 0.04, 0.005])
    assert adjusted == pytest.approx([0.03, 0.03, 0.04, 0.02])
    assert (adjusted < 0.05).all()


def test_holm_adjust_keeps_the_order_given_and_never_lowers_p():
    p = np.array([0.2, 0.001, 0.04, 0.03, 0.5, 0.01])
    adjusted = holm_adjust(p)
    assert (adjusted >= p).all()
    # larger p, larger or equal adjusted p, wherever they sit
    order = np.argsort(p)
    assert (np.diff(adjusted[order]) >= 0).all()
    assert adjusted[np.argmin(p)] == pytest.approx(6 * 0.001)


def test_cochran_q_matches_the_hand_worked_toys():
    res = cochran_q([0.1, 0.6], [0.1, 0.1])
    assert res["Q"] == pytest.approx(12.5)
    assert res["df"] == 1
    assert res["p"] == pytest.approx(0.000407, abs=1e-6)
    # unequal SEs: the far decile weighs 25, not 100, and diff_bar is 55 / 225
    res = cochran_q([0.1, 0.6, 0.3], [0.1, 0.2, 0.1])
    assert res["Q"] == pytest.approx(50 / 9)
    assert res["df"] == 2
    assert res["p"] == pytest.approx(0.062177, abs=1e-6)


def test_cochran_q_is_zero_for_equal_effects():
    assert cochran_q([0.3, 0.3, 0.3], [0.1, 0.2, 0.3])["Q"] == pytest.approx(0.0)


# Two deciles of toy users; decile 2's rows come first so a filter on the wrong table would show
USERS = pd.DataFrame(
    {
        "decile": [2, 2, 2, 2, 1, 1, 1, 1, 1, 1],
        "arm": ["T", "C", "T", "C", "T", "T", "T", "C", "C", "C"],
        "y_obs": [1, 0, 0, 0, 1, 1, 0, 1, 0, 0],
        "gmv_obs": [50.0, 0, 0, 0, 120, 80, 0, 30, 0, 0],
    }
)


def test_conversion_row_compares_one_deciles_arms():
    row = conversion_row(USERS, 1, truth=0.1)
    est = prop_diff_ci([1, 1, 0], [1, 0, 0])
    test = z_test([1, 1, 0], [1, 0, 0])
    assert (row["decile"], row["n_t"], row["n_c"]) == (1, 3, 3)
    assert (row["r_t"], row["r_c"]) == pytest.approx((2 / 3, 1 / 3))
    assert row["diff"] == pytest.approx(1 / 3)
    for key in ("se", "ci_low", "ci_high"):
        assert row[key] == pytest.approx(est[key])
    assert (row["z"], row["p"]) == pytest.approx((test["z"], test["p"]))
    assert row["err_in_se"] == pytest.approx((1 / 3 - 0.1) / est["se"])
    assert row["covers"] == (est["ci_low"] <= 0.1 <= est["ci_high"])
    assert list(USERS.columns) == ["decile", "arm", "y_obs", "gmv_obs"]


def test_gmv_row_takes_mean_diff_ci_and_welchs_t():
    row = gmv_row(USERS, 1, truth=20.0)
    est = mean_diff_ci([120, 80, 0], [30, 0, 0])
    welch = ttest_ind([120, 80, 0], [30, 0, 0], equal_var=False)
    assert (row["n_t"], row["n_c"]) == (3, 3)
    assert (row["mean_t"], row["mean_c"]) == pytest.approx((200 / 3, 10))
    for key in ("diff", "se", "ci_low", "ci_high"):
        assert row[key] == pytest.approx(est[key])
    assert (row["t"], row["p"]) == pytest.approx((welch.statistic, welch.pvalue))
    assert row["err_in_se"] == pytest.approx((est["diff"] - 20.0) / est["se"])
    assert row["covers"] is True
    # a truth beyond either end of the CI is not covered
    assert gmv_row(USERS, 1, truth=est["ci_high"] + 1)["covers"] is False
    assert gmv_row(USERS, 1, truth=est["ci_low"] - 1)["covers"] is False


def test_heterogeneity_analysis_covers_ten_deciles_with_holm_per_metric():
    rng = np.random.default_rng(0)
    n = 4_000
    users = pd.DataFrame(
        {
            "decile": np.repeat(np.arange(1, 11), n // 10),
            "arm": np.tile(["T", "C"], n // 2),
            "y_obs": rng.integers(0, 2, n),
            "gmv_obs": rng.exponential(30, n),
        }
    )
    truth = {(m, d): 0.0 for m in ("conversion", "gmv") for d in range(1, 11)}
    tables, q = heterogeneity_analysis(users, truth)
    for metric in ("conversion", "gmv"):
        table = tables[metric]
        assert table["decile"].tolist() == list(range(1, 11))
        assert (table["n_t"] + table["n_c"]).sum() == n
        # each metric is its own family of 10
        assert table["p_holm"].to_numpy() == pytest.approx(holm_adjust(table["p"]))
        row = q.set_index("metric").loc[metric]
        expected = cochran_q(table["diff"], table["se"])
        assert (row["Q"], row["df"], row["p"]) == pytest.approx((expected["Q"], 9, expected["p"]))
    assert "r_t" in tables["conversion"] and "mean_t" in tables["gmv"]
