import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom

from pte.experiment.aa import AA_SALTS
from pte.experiment.assignment import SALT, assign
from pte.experiment.estimate import main_analysis, observed
from pte.experiment.heterogeneity import heterogeneity_analysis
from pte.experiment.recovery import (
    CONFIRM_SALTS,
    COVERED_RANGE,
    RECOVERY_SALTS,
    ROW_COLS,
    recovery_run,
    recovery_simulation,
    split_and_observe,
    summarize,
    truths,
)

SEGMENTS = ["all"] + [f"decile_{d}" for d in range(1, 11)]


@pytest.fixture
def users() -> pd.DataFrame:
    # 2000 users, 200 per decile; y1 flips some of y0's zeros, more in the middle deciles,
    # and each flipped user gets an amount, so every decile has a nonzero truth on both metrics
    rng = np.random.default_rng(0)
    decile = np.repeat(np.arange(1, 11), 200)
    y0 = (rng.random(2000) < 0.3).astype(int)
    flip = (y0 == 0) & (rng.random(2000) < np.where((decile > 3) & (decile < 8), 0.3, 0.1))
    y1 = np.where(flip, 1, y0)
    gmv0 = np.where(y0 == 1, rng.exponential(50, 2000), 0.0)
    gmv1 = np.where(flip, rng.exponential(50, 2000), gmv0)
    return pd.DataFrame(
        {
            "user_id": np.arange(5000, 7000),
            "decile": decile,
            "y0": y0,
            "gmv0": gmv0,
            "y1": y1,
            "gmv1": gmv1,
        }
    )


def test_salts_are_fixed_and_new():
    assert RECOVERY_SALTS[0] == "recovery-0" and RECOVERY_SALTS[-1] == "recovery-999"
    assert CONFIRM_SALTS[0] == "recovery-1000" and CONFIRM_SALTS[-1] == "recovery-1999"
    everything = RECOVERY_SALTS + CONFIRM_SALTS
    assert len(set(everything)) == 2000
    assert SALT not in everything
    assert not set(everything) & set(AA_SALTS)


def test_covered_range_is_the_binomial_95_percent_range():
    assert COVERED_RANGE == (binom.ppf(0.025, 1000, 0.95), binom.ppf(0.975, 1000, 0.95))


def test_truths_are_mean_y1_minus_mean_y0(users):
    truth_main, truth_decile = truths(users)
    assert truth_main[("conversion", "all")] == pytest.approx((users["y1"] - users["y0"]).mean())
    assert truth_main[("gmv", "all")] == pytest.approx((users["gmv1"] - users["gmv0"]).mean())
    d3 = users[users["decile"] == 3]
    assert truth_decile[("conversion", 3)] == pytest.approx((d3["y1"] - d3["y0"]).mean())
    assert truth_decile[("gmv", 3)] == pytest.approx((d3["gmv1"] - d3["gmv0"]).mean())
    assert truth_main[("gmv", "decile_1")] == truth_decile[("gmv", 1)]
    assert len(truth_decile) == 20


def test_split_and_observe_joins_the_arm_on_user_id(users):
    # shuffled and re-indexed, so position and index no longer line up with user_id
    shuffled = users.sample(frac=1, random_state=1).reset_index(drop=True)
    obs = split_and_observe(shuffled, "recovery-5").set_index("user_id")
    arms = assign(users["user_id"], "recovery-5").set_index("user_id")["arm"]
    assert (obs["arm"] == arms.reindex(obs.index)).all()
    treated = obs["arm"] == "T"
    assert (obs.loc[treated, "y_obs"] == obs.loc[treated, "y1"]).all()
    assert (obs.loc[~treated, "gmv_obs"] == obs.loc[~treated, "gmv0"]).all()


def test_split_and_observe_leaves_users_alone(users):
    before = users.copy()
    split_and_observe(users, "recovery-5")
    pd.testing.assert_frame_equal(users, before)


def test_recovery_run_has_22_rows_from_the_two_analyses(users):
    truth_main, truth_decile = truths(users)
    rows = recovery_run(users, "recovery-7", truth_main, truth_decile)
    assert list(rows.columns) == ROW_COLS
    assert len(rows) == 22
    assert (rows["salt"] == "recovery-7").all()
    assert set(zip(rows["metric"], rows["segment"], strict=True)) == {
        (m, s) for m in ["conversion", "gmv"] for s in SEGMENTS
    }
    # the same numbers as calling the two analyses on the split by hand
    obs = observed(users.assign(arm=assign(users["user_id"], "recovery-7")["arm"].to_numpy()))
    main = main_analysis(obs, truth_main).set_index(["metric", "segment"])
    tables, _ = heterogeneity_analysis(obs, truth_decile)
    got = rows.set_index(["metric", "segment"])
    for metric in ["conversion", "gmv"]:
        assert got.loc[(metric, "all"), "diff"] == main.loc[(metric, "all"), "diff"]
        assert got.loc[(metric, "decile_1"), "diff"] == main.loc[(metric, "decile_1"), "diff"]
        for col in ["diff", "se", "truth", "err_in_se", "covers"]:
            expected = tables[metric].set_index("decile")[col]
            for d in range(1, 11):
                assert got.loc[(metric, f"decile_{d}"), col] == expected[d]


def test_recovery_run_changes_with_the_salt(users):
    truth_main, truth_decile = truths(users)
    a = recovery_run(users, "recovery-1", truth_main, truth_decile)
    b = recovery_run(users, "recovery-2", truth_main, truth_decile)
    assert (a["truth"] == b["truth"]).all()
    assert not np.allclose(a["diff"], b["diff"])


def test_parallel_gives_the_same_rows_as_one_process(users):
    salts = RECOVERY_SALTS[:6]
    one = recovery_simulation(users, salts, workers=1)
    two = recovery_simulation(users, salts, workers=2)
    pd.testing.assert_frame_equal(one, two)
    assert len(one) == 6 * 22
    assert one["salt"].tolist() == [s for s in salts for _ in range(22)]


def results_of(rows: list[tuple]) -> pd.DataFrame:
    """Long results from (salt, metric, segment, diff, truth, err_in_se, covers) tuples."""
    out = pd.DataFrame(
        rows, columns=["salt", "metric", "segment", "diff", "truth", "err_in_se", "covers"]
    )
    out["se"] = 1.0
    return out[ROW_COLS]


def test_summarize_matches_a_hand_worked_toy():
    # GMV all comes first, as recovery_run puts the "all" rows first; summarize still lists
    # conversion before GMV, and decile_10 after decile_2
    results = results_of(
        [
            ("s1", "gmv", "all", 3.0, 2.0, 1.0, True),
            ("s1", "conversion", "decile_2", 0.5, 0.4, 2.0, False),
            ("s1", "conversion", "decile_10", 0.2, 0.1, 0.5, True),
            ("s2", "gmv", "all", 1.0, 2.0, -1.0, True),
            ("s2", "conversion", "decile_2", 0.7, 0.4, 4.0, True),
            ("s2", "conversion", "decile_10", 0.0, 0.1, -0.5, False),
        ]
    )
    s = summarize(results)
    assert s[["metric", "segment"]].values.tolist() == [
        ["conversion", "decile_2"],
        ["conversion", "decile_10"],
        ["gmv", "all"],
    ]
    d2 = s.iloc[0]
    assert d2["n_runs"] == 2
    assert d2["mean_diff"] == pytest.approx(0.6)
    assert d2["bias"] == pytest.approx(0.2)
    assert d2["mean_err_in_se"] == pytest.approx(3.0)
    # sd of (2, 4) with n - 1 = 1: sqrt(2); z = 3 / (sqrt(2) / sqrt(2)) = 3
    assert d2["sd_err_in_se"] == pytest.approx(np.sqrt(2))
    assert d2["z_bias"] == pytest.approx(3.0)
    assert not d2["pass_bias"]
    # covers counted, not rows: 1 of 2
    assert d2["n_covered"] == 1
    gmv_all = s.iloc[2]
    assert gmv_all["n_covered"] == 2
    assert gmv_all["bias"] == pytest.approx(0.0)
    assert gmv_all["pass_bias"]


@pytest.mark.parametrize(
    ("n_covered", "passes"), [(935, False), (936, True), (950, True), (963, True), (964, False)]
)
def test_summarize_coverage_range_includes_both_ends(n_covered, passes):
    covers = [True] * n_covered + [False] * (1000 - n_covered)
    results = results_of(
        [(f"s{i}", "gmv", "all", 0.0, 0.0, (-1) ** i, c) for i, c in enumerate(covers)]
    )
    s = summarize(results)
    assert s.loc[0, "n_covered"] == n_covered
    assert s.loc[0, "pass_coverage"] == passes


def test_summarize_bias_limit_is_1_96_either_side():
    # 1000 runs at err m +- 1: sd = sqrt(1000 / 999), so z_bias = m / (sd / sqrt(1000)); m is
    # set from the z wanted, just inside and just outside 1.96 on both sides
    sd = np.sqrt(1000 / 999)
    for z_wanted, passes in [(1.95, True), (-1.95, True), (1.97, False), (-1.97, False)]:
        m = z_wanted * sd / np.sqrt(1000)
        errs = [m + (-1) ** i for i in range(1000)]
        res = results_of([(f"s{i}", "gmv", "all", 0.0, 0.0, e, True) for i, e in enumerate(errs)])
        row = summarize(res).loc[0]
        assert row["z_bias"] == pytest.approx(z_wanted, rel=1e-9)
        assert row["pass_bias"] == passes
