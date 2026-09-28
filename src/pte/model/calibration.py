"""Probability calibration: Platt scaling, and the cross-fit that chose it, from Andy's P3 notebook
(2026-09-29).

Decided 2026-09-29 (leakage.md 3.4): the LightGBM probabilities are recalibrated with Platt
scaling, a one-feature logistic regression on logit(p), fit on the validation set. The model
never saw the validation labels, so that is where its overestimate shows (mean prediction
5.49% against 5.00% actual; on the training set the two match). Platt beat isotonic regression
under cross-fitting: 0.042317 against 0.042321 Brier, uncalibrated 0.042358.

Any calibrator that can leave p unchanged scores at least as well on the data it was fit on as
before, so an in-sample before/after comparison is no evidence. cross_fit scores each user with
a calibrator fit on the other half only.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

Calibrator = LogisticRegression | IsotonicRegression

# keeps logit finite if a model ever outputs exactly 0 or 1
EPS = 1e-12


def logit(p: ArrayLike) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_platt(p: ArrayLike, y: ArrayLike) -> LogisticRegression:
    """q = 1 / (1 + exp(-(w * logit(p) + c))); w = 1, c = 0 leaves p unchanged.

    Unpenalized (C=inf) and solved to a tight tolerance, so that the intercept's estimating
    equation sum(y - q) = 0 holds and the mean calibrated probability equals the actual rate.
    With sklearn's defaults the solver stopped after 7 iterations at c = -0.0959 and a mean of
    5.0044% against 5.0003% actual; solved through, c = -0.0970.
    """
    model = LogisticRegression(C=np.inf, tol=1e-10, max_iter=1000)
    return model.fit(logit(p)[:, None], np.asarray(y))


def fit_isotonic(p: ArrayLike, y: ArrayLike) -> IsotonicRegression:
    """The step function kept for comparison; clip holds new p beyond the fitted range flat."""
    return IsotonicRegression(out_of_bounds="clip").fit(np.asarray(p, dtype=float), np.asarray(y))


def calibrate(model: Calibrator, p: ArrayLike) -> np.ndarray:
    """Map raw probabilities to calibrated ones. Needs no labels."""
    if isinstance(model, IsotonicRegression):
        return model.predict(np.asarray(p, dtype=float))
    return model.predict_proba(logit(p)[:, None])[:, 1]


def random_halves(n: int, seed: int = 42) -> np.ndarray:
    """0 or 1 per user at random. The validation set has one row per user."""
    return np.random.default_rng(seed).integers(0, 2, size=n)


def cross_fit(
    p: ArrayLike, y: ArrayLike, half: ArrayLike, fit: Callable[[ArrayLike, ArrayLike], Calibrator]
) -> np.ndarray:
    """Calibrated p for every user, each from a calibrator fit on the other half only.

    The result keeps the input order, so it lines up with y by position; stacking the two
    halves instead would reorder the rows.
    """
    p, y, half = np.asarray(p, dtype=float), np.asarray(y), np.asarray(half)
    if sorted(np.unique(half).tolist()) != [0, 1]:
        raise ValueError("half must hold both 0 and 1 and nothing else")
    q = np.empty(len(p))
    for h in (0, 1):
        fitted = fit(p[half != h], y[half != h])
        q[half == h] = calibrate(fitted, p[half == h])
    return q
