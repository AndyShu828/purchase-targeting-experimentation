import numpy as np
import pytest
from scipy.stats import chi2_contingency

from pte.experiment.ztest import z_test


def arm(x: int, n: int) -> np.ndarray:
    """n users of whom the first x bought."""
    return np.repeat([1, 0], [x, n - x])


def test_z_test_matches_the_worked_exercise():
    # Andy's notebook self-check: 60 of 1000 against 50 of 1000
    res = z_test(arm(60, 1000), arm(50, 1000))
    assert res["diff"] == pytest.approx(0.01)
    assert res["z"] == pytest.approx(0.980816, abs=1e-6)
    assert res["p"] == pytest.approx(0.326683, abs=1e-6)


def test_z_test_reproduces_aa_0():
    # the A/A split under "aa-0", from its arms' counts: all users, then decile 1
    res = z_test(arm(51_042, 1_152_673), arm(51_248, 1_153_006))
    assert res["diff"] == pytest.approx(-0.00016587, abs=1e-8)
    assert res["z"] == pytest.approx(-0.611626, abs=1e-6)
    assert res["p"] == pytest.approx(0.540785, abs=1e-6)
    res = z_test(arm(18_975, 114_904), arm(19_288, 115_664))
    assert res["diff"] == pytest.approx(-0.00162103, abs=1e-8)
    assert res["z"] == pytest.approx(-1.046101, abs=1e-6)
    assert res["p"] == pytest.approx(0.295515, abs=1e-6)


def test_z_test_agrees_with_the_chi_square_on_the_2x2_table():
    # unequal arms, so taking half the total as each arm's n would fail; z keeps diff's sign
    for x_t, n_t, x_c, n_c in [(30, 100, 20, 300), (5, 400, 40, 600), (120, 1000, 120, 1000)]:
        res = z_test(arm(x_t, n_t), arm(x_c, n_c))
        table = [[x_t, n_t - x_t], [x_c, n_c - x_c]]
        chi_sq, p, _, _ = chi2_contingency(table, correction=False)
        assert res["diff"] == pytest.approx(x_t / n_t - x_c / n_c)
        assert res["z"] ** 2 == pytest.approx(chi_sq)
        assert res["p"] == pytest.approx(p)
        assert np.sign(res["z"]) == np.sign(res["diff"])


def test_z_test_keeps_small_p_values():
    # 1 - Phi(|z|) rounds to 0 once Phi(|z|) is within 1e-16 of 1 (decile 5 of the experiment, z 8.86)
    for x_t, n_t, x_c, n_c in [(3_941, 115_354, 3_200, 115_213), (600, 2_000, 200, 2_000)]:
        res = z_test(arm(x_t, n_t), arm(x_c, n_c))
        table = [[x_t, n_t - x_t], [x_c, n_c - x_c]]
        _, p, _, _ = chi2_contingency(table, correction=False)
        assert p < 1e-16
        assert res["p"] == pytest.approx(p, rel=1e-9, abs=0)  # approx allows 1e-12 by default
