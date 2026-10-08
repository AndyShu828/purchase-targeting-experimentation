import numpy as np
import pandas as pd
import pytest
from scipy.stats import ttest_ind

from pte.experiment.estimate import main_analysis, mean_diff_ci, observed, prop_diff_ci
from pte.experiment.ztest import z_test


def arm(x: int, n: int) -> np.ndarray:
    """n users of whom the first x bought."""
    return np.repeat([1, 0], [x, n - x])


TOY_T = [0, 0, 0, 0, 0, 0, 120, 80, 0, 300]
TOY_C = [0, 0, 0, 0, 0, 0, 0, 50, 0, 100]


def test_prop_diff_ci_matches_the_worked_exercise():
    # Andy's notebook self-check: 60 of 1000 against 50 of 1000
    res = prop_diff_ci(arm(60, 1000), arm(50, 1000))
    assert res["diff"] == pytest.approx(0.01)
    assert res["se"] == pytest.approx(0.0101931, abs=1e-7)
    assert (res["ci_low"], res["ci_high"]) == pytest.approx((-0.0099785, 0.0299785), abs=1e-7)
    assert (res["r_t"], res["r_c"]) == pytest.approx((0.06, 0.05))


def test_prop_diff_ci_uses_each_arms_own_n_and_rate():
    # 30 of 200 against 20 of 300, by hand: a pooled rate or n/2 per arm would differ
    res = prop_diff_ci(arm(30, 200), arm(20, 300))
    assert res["diff"] == pytest.approx(0.0833333, abs=1e-7)
    assert res["se"] == pytest.approx(0.0290673, abs=1e-7)
    assert (res["ci_low"], res["ci_high"]) == pytest.approx((0.0263614, 0.1403052), abs=1e-7)
    # the pooled SE of the test is a different number on these arms
    assert res["diff"] / res["se"] != pytest.approx(z_test(arm(30, 200), arm(20, 300))["z"])


def test_mean_diff_ci_matches_the_worked_exercise_and_welch():
    # Andy's notebook self-check, the same for a Series and an array (ddof 1 either way)
    for t, c in [(pd.Series(TOY_T), pd.Series(TOY_C)), (np.array(TOY_T), np.array(TOY_C))]:
        res = mean_diff_ci(t, c)
        assert res["diff"] == pytest.approx(35.0)
        assert res["se"] == pytest.approx(32.668367, abs=1e-6)
        assert (res["ci_low"], res["ci_high"]) == pytest.approx((-29.03, 99.03), abs=1e-6)
        assert (res["mean_t"], res["mean_c"]) == pytest.approx((50.0, 15.0))
        welch = ttest_ind(t, c, equal_var=False)
        assert res["diff"] / res["se"] == pytest.approx(welch.statistic)
    assert welch.statistic == pytest.approx(1.071373, abs=1e-6)
    assert welch.pvalue == pytest.approx(0.306704, abs=1e-6)


def test_observed_reads_y1_in_treatment_and_y0_in_control():
    users = pd.DataFrame(
        {
            "arm": ["T", "C", "T", "C"],
            "y0": [0, 0, 1, 1],
            "y1": [1, 1, 1, 1],
            "gmv0": [0.0, 0.0, 30.0, 40.0],
            "gmv1": [50.0, 60.0, 30.0, 40.0],
        }
    )
    out = observed(users)
    assert out["y_obs"].tolist() == [1, 0, 1, 1]
    assert out["gmv_obs"].tolist() == [50.0, 0.0, 30.0, 40.0]
    assert "y_obs" not in users.columns


@pytest.fixture
def users() -> pd.DataFrame:
    # decile 1 buys far more often and spends more, so a row taken from the wrong segment,
    # metric or arm changes the result; outcomes differ between arms within each segment
    rng = np.random.default_rng(0)
    n = 2000
    decile = np.where(np.arange(n) < 600, 1, 2)
    arm_ = np.where(rng.random(n) < 0.5, "T", "C")
    rate = np.where(decile == 1, 0.4, 0.05) + np.where(arm_ == "T", 0.05, 0.0)
    y_obs = (rng.random(n) < rate).astype(int)
    gmv_obs = y_obs * rng.exponential(np.where(decile == 1, 200.0, 50.0))
    return pd.DataFrame({"decile": decile, "arm": arm_, "y_obs": y_obs, "gmv_obs": gmv_obs})


TRUTH = {
    ("conversion", "decile_1"): 0.05,
    ("conversion", "all"): 0.05,
    ("gmv", "decile_1"): 10.0,
    ("gmv", "all"): 1000.0,
}


def test_main_analysis_rows_match_each_segment_metric_and_test(users):
    table = main_analysis(users, TRUTH)
    assert list(zip(table["metric"], table["segment"], strict=True)) == [
        ("conversion", "decile_1"),
        ("gmv", "decile_1"),
        ("conversion", "all"),
        ("gmv", "all"),
    ]
    for segment, part in [("decile_1", users[users["decile"] == 1]), ("all", users)]:
        t, c = part[part["arm"] == "T"], part[part["arm"] == "C"]
        conv = table[(table["metric"] == "conversion") & (table["segment"] == segment)].iloc[0]
        est = prop_diff_ci(t["y_obs"], c["y_obs"])
        test = z_test(t["y_obs"], c["y_obs"])
        assert (conv["n_t"], conv["n_c"]) == (len(t), len(c))
        assert conv["diff"] == pytest.approx(est["diff"])
        assert conv["se"] == pytest.approx(est["se"])
        assert conv["stat"] == pytest.approx(test["z"])
        assert conv["p"] == pytest.approx(test["p"])
        assert conv["rel_diff"] == pytest.approx(est["diff"] / c["y_obs"].mean())
        gmv = table[(table["metric"] == "gmv") & (table["segment"] == segment)].iloc[0]
        est = mean_diff_ci(t["gmv_obs"], c["gmv_obs"])
        welch = ttest_ind(t["gmv_obs"], c["gmv_obs"], equal_var=False)
        assert gmv["diff"] == pytest.approx(est["diff"])
        assert gmv["se"] == pytest.approx(est["se"])
        assert gmv["stat"] == pytest.approx(welch.statistic)
        assert gmv["p"] == pytest.approx(welch.pvalue)
        assert gmv["rel_diff"] == pytest.approx(est["diff"] / c["gmv_obs"].mean())


def test_main_analysis_compares_each_row_with_its_own_truth(users):
    table = main_analysis(users, TRUTH)
    assert table["truth"].tolist() == [
        TRUTH[(m, s)] for m, s in zip(table["metric"], table["segment"], strict=True)
    ]
    assert table["err_in_se"].tolist() == pytest.approx(
        ((table["diff"] - table["truth"]) / table["se"]).tolist()
    )
    expected = (table["ci_low"] <= table["truth"]) & (table["truth"] <= table["ci_high"])
    assert table["covers"].tolist() == expected.tolist()
    # GMV over all users: a truth of 1000 lies far outside the CI
    assert table["covers"].tolist()[-1] is False
