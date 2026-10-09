import numpy as np
import pandas as pd
import pytest
from scipy.stats import ttest_ind

from pte.experiment.cuped import (
    PRED_RHO2,
    correlations_by_decile,
    covariate_correlations,
    cuped_analysis,
    cuped_row,
)
from pte.experiment.estimate import mean_diff_ci

# The toy table from the 2026-10-04 session, plus a column z that is constant in decile 2.
TOY = pd.DataFrame(
    {
        "decile": [1, 1, 1, 2, 2, 2],
        "x": [3, 0, 2, 1, 0, 0],
        "p": [0.30, 0.20, 0.25, 0.05, 0.02, 0.04],
        "z": [1, 0, 1, 0, 0, 0],
        "label": [1, 0, 1, 0, 0, 1],
    }
)


def test_covariate_correlations_match_corr_and_square_it():
    table = covariate_correlations(TOY, ["x", "p"])
    assert table["candidate"].tolist() == ["x", "p"]
    expected = [TOY["x"].corr(TOY["label"]), TOY["p"].corr(TOY["label"])]
    assert table["rho"].tolist() == pytest.approx(expected)
    assert table["rho2"].tolist() == pytest.approx([r**2 for r in expected])


def test_correlations_by_decile_match_the_hand_worked_toy():
    table = correlations_by_decile(TOY, ["x", "p", "z"])
    assert table.index.tolist() == [1, 2]
    assert table["x"].tolist() == pytest.approx([0.9449, -0.5], abs=1e-4)
    assert table["p"].tolist() == pytest.approx([0.8660, 0.1890], abs=1e-4)
    # within decile 1 z equals the label; in decile 2 it is constant, so its SD is 0
    assert table.loc[1, "z"] == pytest.approx(1.0)
    assert pd.isna(table.loc[2, "z"])


# The toy table from the 2026-10-09 session: treatment drew the users with the higher X
ARMS = pd.DataFrame({"arm": ["T", "T", "C", "C"], "x": [4, 2, 0, 2], "y": [10.0, 6, 4, 6]})


def test_cuped_row_matches_the_hand_worked_toy():
    # mean X 2, Cov(X, Y) = 12 / 3 = 4, Var(X) = 8 / 3, so theta = 1.5 and Y_cv = 7, 6, 7, 6
    row = cuped_row(ARMS, "x", "y", true_value=1.0)
    assert row["theta"] == pytest.approx(1.5)
    # raw diff 8 - 5 = 3, minus theta x (mean X_T - mean X_C) = 1.5 x (3 - 1)
    assert row["diff_raw"] == pytest.approx(3.0)
    assert row["diff"] == pytest.approx(3.0 - 1.5 * 2)
    # Var(Y) 19 / 3, Var(Y_cv) 1 / 3; with theta fitted on these users it is rho^2 exactly
    assert row["var_reduction"] == pytest.approx(18 / 19)
    assert row["var_reduction"] == pytest.approx(ARMS["x"].corr(ARMS["y"]) ** 2)
    # each arm's Y_cv is (7, 6), variance 0.5; raw arms (10, 6) and (4, 6), variances 8 and 2
    assert row["se"] == pytest.approx(np.sqrt(0.5 / 2 + 0.5 / 2))
    assert row["se_raw"] == pytest.approx(np.sqrt(8 / 2 + 2 / 2))
    assert row["se_ratio"] == pytest.approx(row["se"] / row["se_raw"])
    assert (row["ci_low"], row["ci_high"]) == pytest.approx((-1.96 * row["se"], 1.96 * row["se"]))
    assert row["err_in_se"] == pytest.approx((0.0 - 1.0) / row["se"])
    assert row["covers"] is True
    assert "y_cv" not in ARMS.columns


def test_cuped_row_takes_welchs_t_on_y_cv():
    # unequal arms with unequal spreads, so Student's t and the raw y give other p values
    rng = np.random.default_rng(1)
    n_t, n_c = 300, 100
    x = rng.normal(size=n_t + n_c)
    y = 2 * x + rng.normal(scale=np.r_[np.full(n_t, 0.5), np.full(n_c, 3.0)])
    part = pd.DataFrame({"arm": ["T"] * n_t + ["C"] * n_c, "x": x, "y": y})
    row = cuped_row(part, "x", "y", true_value=0.0)
    theta = part["x"].cov(part["y"]) / part["x"].var()
    y_cv = part["y"] - theta * (part["x"] - part["x"].mean())
    treated = part["arm"] == "T"
    welch = ttest_ind(y_cv[treated], y_cv[~treated], equal_var=False)
    assert row["p"] == pytest.approx(welch.pvalue)
    assert row["p"] != pytest.approx(ttest_ind(y_cv[treated], y_cv[~treated]).pvalue)
    assert row["se"] == pytest.approx(mean_diff_ci(y_cv[treated], y_cv[~treated])["se"])


def test_cuped_row_stops_on_a_missing_covariate():
    part = ARMS.astype({"x": float})
    part.loc[0, "x"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        cuped_row(part, "x", "y", true_value=0.0)


@pytest.fixture
def users() -> pd.DataFrame:
    # decile 1 differs from decile 2, and each metric's covariate tracks its own outcome only,
    # so a row fitted on the wrong segment or columns gets a different theta
    rng = np.random.default_rng(0)
    n = 2000
    decile = np.where(np.arange(n) < 600, 1, 2)
    arm_ = np.where(rng.random(n) < 0.5, "T", "C")
    p = np.where(decile == 1, 0.3, 0.05) * rng.uniform(0.5, 1.5, n)
    y_obs = (rng.random(n) < p + np.where(arm_ == "T", 0.05, 0.0)).astype(int)
    spend = rng.exponential(np.where(decile == 1, 20.0, 2.0))
    gmv_obs = y_obs * (5 * spend + rng.exponential(50.0, n))
    return pd.DataFrame(
        {
            "decile": decile,
            "arm": arm_,
            "p": p,
            "total_spend_per_day": spend,
            "y_obs": y_obs,
            "gmv_obs": gmv_obs,
        }
    )


TRUTH = {
    ("conversion", "decile_1"): 0.05,
    ("conversion", "all"): 0.04,
    ("gmv", "decile_1"): 10.0,
    ("gmv", "all"): 1000.0,
}


def test_cuped_analysis_fits_each_row_on_its_own_segment_and_columns(users):
    table = cuped_analysis(users, TRUTH)
    keys = list(zip(table["metric"], table["segment"], strict=True))
    assert keys == [
        ("conversion", "decile_1"),
        ("conversion", "all"),
        ("gmv", "decile_1"),
        ("gmv", "all"),
    ]
    parts = {"decile_1": users[users["decile"] == 1], "all": users}
    columns = {"conversion": ("p", "y_obs"), "gmv": ("total_spend_per_day", "gmv_obs")}
    for (metric, segment), (_, row) in zip(keys, table.iterrows(), strict=True):
        x, y = columns[metric]
        expected = cuped_row(parts[segment], x, y, TRUTH[(metric, segment)])
        for k, v in expected.items():
            assert row[k] == pytest.approx(v), (metric, segment, k)
        assert row["pred_rho2"] == PRED_RHO2[(metric, segment)]
    # GMV over all users: a truth of 1000 lies far outside the CI
    assert table["covers"].tolist() == [True, True, True, False]
    # the four thetas all differ, so no row reused another's segment or columns
    assert table["theta"].round(6).nunique() == 4
    assert "y_cv" not in users.columns
