import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom

from pte.experiment.aa import AA_SALTS, COUNT_RANGE, aa_run, aa_simulation, aa_summary
from pte.experiment.assignment import SALT, assign
from pte.experiment.ztest import z_test


@pytest.fixture
def users() -> pd.DataFrame:
    # 600 users, deciles 1-3; decile 1 buys far more often, so a row taken from the wrong
    # deciles changes the decile_1 result
    rng = np.random.default_rng(0)
    decile = np.repeat([1, 2, 3], 200)
    y0 = (rng.random(600) < np.where(decile == 1, 0.5, 0.05)).astype(int)
    return pd.DataFrame({"user_id": np.arange(1000, 1600), "decile": decile, "y0": y0})


def test_salts_are_fixed_and_exclude_the_official_one():
    assert len(AA_SALTS) == len(set(AA_SALTS)) == 1000
    assert AA_SALTS[0] == "aa-0" and AA_SALTS[-1] == "aa-999"
    assert SALT not in AA_SALTS


def test_count_range_is_the_binomial_95_percent_range():
    assert COUNT_RANGE == (binom.ppf(0.025, 1000, 0.05), binom.ppf(0.975, 1000, 0.05))


def test_aa_run_tests_all_users_and_decile_1_on_y0(users):
    rows = aa_run(users, "aa-7")
    assert [(r["salt"], r["metric"]) for r in rows] == [("aa-7", "all"), ("aa-7", "decile_1")]
    arms = assign(users["user_id"], "aa-7")["arm"].to_numpy()
    y0, d1 = users["y0"].to_numpy(), users["decile"].to_numpy() == 1
    expected_all = z_test(y0[arms == "T"], y0[arms == "C"])
    expected_d1 = z_test(y0[d1 & (arms == "T")], y0[d1 & (arms == "C")])
    for key in ["diff", "z", "p"]:
        assert rows[0][key] == pytest.approx(expected_all[key])
        assert rows[1][key] == pytest.approx(expected_d1[key])


def test_aa_run_does_not_depend_on_row_order(users):
    shuffled = users.sample(frac=1, random_state=1).reset_index(drop=True)
    for got, expected in zip(aa_run(shuffled, "aa-3"), aa_run(users, "aa-3"), strict=True):
        assert got["metric"] == expected["metric"]
        assert [got[k] for k in ["diff", "z", "p"]] == pytest.approx(
            [expected[k] for k in ["diff", "z", "p"]]
        )


def test_aa_simulation_has_two_rows_per_salt(users):
    results = aa_simulation(users, AA_SALTS[:4])
    assert len(results) == 8
    assert results["salt"].tolist() == [s for s in AA_SALTS[:4] for _ in range(2)]
    assert results["metric"].tolist() == ["all", "decile_1"] * 4


def results_of(p_all, p_d1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "diff": np.arange(len(p_all) + len(p_d1), dtype=float),
            "p": np.concatenate([p_all, p_d1]),
            "metric": ["all"] * len(p_all) + ["decile_1"] * len(p_d1),
        }
    )


def test_aa_summary_judges_each_metric_on_its_own_rows():
    # all: evenly spread p, 50 below 0.05; decile_1: every p at 0.01, so 1000 false positives.
    # Pooled, the two would give 1050 of 2000
    even = (np.arange(1000) + 0.5) / 1000
    summary = aa_summary(results_of(even, np.full(1000, 0.01)))
    assert summary["metric"].tolist() == ["all", "decile_1"]
    assert summary["n"].tolist() == [1000, 1000]
    assert summary["cnt"].tolist() == [50, 1000]
    assert summary["pass_count"].tolist() == [True, False]
    assert summary["pass_ks"].tolist() == [True, False]
    assert summary["ks_p"].iloc[0] > 0.99
    assert summary["diff_sd"].tolist() == pytest.approx(
        [np.arange(1000).std(ddof=1), np.arange(1000, 2000).std(ddof=1)]
    )


@pytest.mark.parametrize("cnt, passes", [(36, False), (37, True), (64, True), (65, False)])
def test_aa_summary_count_range_is_inclusive(cnt, passes):
    p = np.concatenate([np.full(cnt, 0.01), np.full(1000 - cnt, 0.5)])
    summary = aa_summary(results_of(p, p))
    assert summary["cnt"].tolist() == [cnt, cnt]
    assert summary["pass_count"].tolist() == [passes, passes]
