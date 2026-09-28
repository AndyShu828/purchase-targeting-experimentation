import numpy as np
import pytest
from sklearn.metrics import brier_score_loss

from pte.model.calibration import (
    calibrate,
    cross_fit,
    fit_isotonic,
    fit_platt,
    logit,
    random_halves,
)

# The toy table from the 2026-09-29 session: every p twice, once in each half.
TOY_P = [0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.7, 0.7]
TOY_Y = [0, 0, 1, 0, 0, 1, 1, 1]
TOY_HALF = [0, 1, 0, 1, 0, 1, 0, 1]


def test_isotonic_pools_adjacent_violators():
    # half 0 has y = 0, 1, 0, 1 in p order; the middle 1, 0 pools to 0.5, 0.5
    model = fit_isotonic([0.1, 0.3, 0.5, 0.7], [0, 1, 0, 1])
    assert calibrate(model, [0.1, 0.3, 0.5, 0.7]).tolist() == [0, 0.5, 0.5, 1]


def test_cross_fit_matches_the_hand_worked_toy():
    q = cross_fit(TOY_P, TOY_Y, TOY_HALF, fit_isotonic)
    assert q.tolist() == [0, 0, 0, 0.5, 1, 0.5, 1, 1]
    # in-sample the same method looks better than doing nothing, out of sample it is worse
    in_sample = calibrate(fit_isotonic(TOY_P, TOY_Y), TOY_P)
    assert brier_score_loss(TOY_Y, TOY_P) == pytest.approx(0.16)
    assert brier_score_loss(TOY_Y, in_sample) == pytest.approx(0.125)
    assert brier_score_loss(TOY_Y, q) == pytest.approx(0.3125)


def test_cross_fit_never_scores_a_user_with_their_own_label():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.99, size=400)
    y = (rng.random(400) < p).astype(int)
    half = random_halves(400, seed=1)
    flipped = np.where(half == 0, 1 - y, y)
    for fit in (fit_platt, fit_isotonic):
        before, after = cross_fit(p, y, half, fit), cross_fit(p, flipped, half, fit)
        # half 0's labels only reach the calibrator that scores half 1
        assert np.array_equal(before[half == 0], after[half == 0])
        assert not np.array_equal(before[half == 1], after[half == 1])


def test_cross_fit_rejects_a_half_that_is_not_zero_and_one():
    with pytest.raises(ValueError, match="both 0 and 1"):
        cross_fit(TOY_P, TOY_Y, [0] * 8, fit_isotonic)
    with pytest.raises(ValueError, match="both 0 and 1"):
        cross_fit(TOY_P, TOY_Y, [0, 1, 2, 1, 0, 1, 0, 1], fit_isotonic)


def synthetic(n, w, c, seed):
    """Raw p whose true probability is logit^-1(w * logit(p) + c)."""
    rng = np.random.default_rng(seed)
    p = 1 / (1 + np.exp(-rng.normal(-2.5, 1.0, size=n)))
    truth = 1 / (1 + np.exp(-(w * logit(p) + c)))
    return p, (rng.random(n) < truth).astype(int)


def test_platt_recovers_slope_and_shift_on_the_logit_scale():
    p, y = synthetic(200_000, w=0.8, c=-0.3, seed=0)
    model = fit_platt(p, y)
    assert model.coef_[0][0] == pytest.approx(0.8, abs=0.03)
    assert model.intercept_[0] == pytest.approx(-0.3, abs=0.05)


def test_platt_leaves_calibrated_p_nearly_unchanged():
    p, y = synthetic(200_000, w=1.0, c=0.0, seed=1)
    q = calibrate(fit_platt(p, y), p)
    assert np.abs(q - p).max() < 0.01


def test_platt_mean_equals_the_actual_rate():
    # the intercept's estimating equation sum(y - q) = 0, which needs the solver run through
    p, y = synthetic(200_000, w=1.0, c=-0.1, seed=2)
    q = calibrate(fit_platt(p, y), p)
    assert q.mean() == pytest.approx(y.mean(), abs=1e-7)


def test_platt_keeps_the_ranking_isotonic_ties_it():
    p, y = synthetic(20_000, w=1.0, c=-0.1, seed=3)
    order = np.argsort(p)
    platt = calibrate(fit_platt(p, y), p)[order]
    assert (np.diff(platt) > 0).all()
    isotonic = calibrate(fit_isotonic(p, y), p)[order]
    assert (np.diff(isotonic) >= 0).all()
    assert len(np.unique(isotonic)) < len(np.unique(p))


def test_logit_stays_finite_at_zero_and_one():
    assert np.isfinite(logit([0.0, 0.5, 1.0])).all()
    assert logit([0.5])[0] == 0


def test_random_halves_is_reproducible_and_binary():
    half = random_halves(1000)
    assert np.array_equal(half, random_halves(1000, seed=42))
    assert set(half.tolist()) == {0, 1}
    # the notebook's rng = np.random.default_rng(42); rng.integers(0, 2, size=n)
    assert np.array_equal(half, np.random.default_rng(42).integers(0, 2, size=1000))
