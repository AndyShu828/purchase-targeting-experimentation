"""The two-proportion z test, from Andy's P4 notebook (2026-10-06).

Decided 2026-10-06 (experiment_design.md section 2): two-sided, alpha 0.05, with the standard
error taken at the pooled rate, as under the null the two arms share one conversion rate.

    diff = r_t - r_c, r_pool = (x_t + x_c) / (n_t + n_c)
    z = diff / sqrt(r_pool (1 - r_pool) (1/n_t + 1/n_c)), p = 2 (1 - Phi(|z|))

The A/A simulation checks it on the test set's y0 before the primary test uses it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm


def z_test(y_t: ArrayLike, y_c: ArrayLike) -> dict[str, float]:
    """diff, z and two-sided p for two arms' 0/1 outcomes.

    n is each arm's length and x its sum, so the arms' own head counts are used, not half of
    the total. z carries the sign of diff (treatment minus control); p reads only |z|.
    """
    y_t = np.asarray(y_t)
    y_c = np.asarray(y_c)
    n_t, n_c = len(y_t), len(y_c)
    x_t, x_c = y_t.sum(), y_c.sum()
    diff = x_t / n_t - x_c / n_c
    r_pool = (x_t + x_c) / (n_t + n_c)
    se = np.sqrt(r_pool * (1 - r_pool) * (1 / n_t + 1 / n_c))
    z = diff / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return {"diff": float(diff), "z": float(z), "p": float(p)}
