import pandas as pd
import pytest
from scipy.stats import chisquare, norm

from pte.experiment.srm import srm_table, srm_test


def test_srm_test_matches_the_worked_exercises():
    # the 2026-10-05 exercises: d = 20 and d = 100 against E = N/2
    small = srm_test(520, 480)
    assert (small["e_t"], small["e_c"]) == (500, 500)
    assert small["chi_sq"] == pytest.approx(1.6)
    assert small["p"] == pytest.approx(0.20590, abs=1e-5)
    assert srm_test(5100, 4900)["chi_sq"] == pytest.approx(4.0)
    assert srm_test(5100, 4900)["p"] == pytest.approx(0.04550, abs=1e-5)


def test_srm_test_reproduces_the_official_split():
    # Andy's notebook, salt "p4-coupon-2026-10" on the 2,305,679 test users
    result = srm_test(1_154_643, 1_151_036)
    assert result["e_t"] == result["e_c"] == 1_152_839.5
    assert result["share_t"] == pytest.approx(0.500782, abs=1e-6)
    assert result["chi_sq"] == pytest.approx(5.642784, abs=1e-6)
    assert result["p"] == pytest.approx(0.017527352145, rel=1e-9)


def test_srm_test_agrees_with_scipy_and_with_the_z_test():
    # scipy's chisquare on the same counts, at 50/50 and at a 90/10 design
    for n_t, n_c, share in [(1_154_643, 1_151_036, 0.5), (9_130, 870, 0.9)]:
        n = n_t + n_c
        ref = chisquare([n_t, n_c], [n * share, n * (1 - share)])
        result = srm_test(n_t, n_c, share)
        assert result["chi_sq"] == pytest.approx(ref.statistic)
        assert result["p"] == pytest.approx(ref.pvalue)
        z = (n_t - n * share) / (n * share * (1 - share)) ** 0.5
        assert result["chi_sq"] == pytest.approx(z**2)
        assert result["p"] == pytest.approx(2 * norm.sf(abs(z)))


def test_srm_table_counts_by_label_and_alarms_only_on_all():
    # decile 1 has more T, decile 2 more C, so a count taken by position would swap the arms
    # in one of them; decile 3 has no C at all
    split = pd.DataFrame(
        {
            "decile": [1, 1, 1, 2, 2, 2, 2, 3],
            "arm": ["T", "T", "C", "C", "C", "C", "T", "T"],
        }
    )
    table = srm_table(split)
    assert table["decile"].tolist() == [1, 2, 3, "all"]
    assert table["n_t"].tolist() == [2, 1, 1, 4]
    assert table["n_c"].tolist() == [1, 3, 0, 4]
    assert table["chi_sq"].tolist() == pytest.approx(
        [srm_test(2, 1)["chi_sq"], srm_test(1, 3)["chi_sq"], srm_test(1, 0)["chi_sq"], 0.0]
    )
    assert table["alarm"].tolist() == [None, None, None, False]


def test_srm_table_alarms_below_0001():
    # 52,000 against 48,000: chi_sq 160, far past 10.83 (p = 0.001); the decile with the same
    # counts carries no alarm, being a diagnostic
    split = pd.DataFrame({"decile": 1, "arm": ["T"] * 52_000 + ["C"] * 48_000})
    table = srm_table(split)
    assert table["chi_sq"].tolist() == pytest.approx([160.0, 160.0])
    assert table["alarm"].tolist() == [None, True]
