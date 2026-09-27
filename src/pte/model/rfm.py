"""RFM baseline: a rule score that sees no label, the yardstick for the P3 models.

From Andy's P3 notebook (2026-09-27). The spec is decisions 1-4 of that session:
- users who never bought score 0, all tied at the bottom
- R: days_since_last_purchase 1-2 / 3-4 / 5-7 / 8-10 / 11+ -> 5..1
- F: purchase_days 1 / 2 / 3 / 4-5 / 6+ -> 1..5 (visits, not basket size)
- M: total_spend cut at the 20/40/60/80% quantiles of the training buyers, frozen, so
  validation and test are scored on the training edges (leakage.md 3.3)
- total = R + F + M, equal weights, 3..15 for buyers
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# pd.cut bins are right-closed: (0, 2] -> 5, (2, 4] -> 4, ...
R_BINS = [0, 2, 4, 7, 10, np.inf]
R_LABELS = [5, 4, 3, 2, 1]
F_BINS = [0, 1, 2, 3, 5, np.inf]
F_LABELS = [1, 2, 3, 4, 5]
M_QUANTILES = [0.2, 0.4, 0.6, 0.8]
M_LABELS = [1, 2, 3, 4, 5]

COLUMNS = ["n_purchases", "purchase_days", "days_since_last_purchase", "total_spend"]


def fit_spend_bins(train: pd.DataFrame) -> list[float]:
    """M bins from the training buyers' total_spend, open at both ends."""
    buyers = train[train["n_purchases"] > 0]
    return [-np.inf, *buyers["total_spend"].quantile(M_QUANTILES), np.inf]


def rfm_score(features: pd.DataFrame, spend_bins: list[float]) -> pd.Series:
    """R + F + M for buyers, 0 for everyone else; spend_bins comes from fit_spend_bins."""
    r = pd.cut(features["days_since_last_purchase"], bins=R_BINS, labels=R_LABELS)
    f = pd.cut(features["purchase_days"], bins=F_BINS, labels=F_LABELS)
    m = pd.cut(features["total_spend"], bins=spend_bins, labels=M_LABELS)
    total = r.astype(float) + f.astype(float) + m.astype(float)

    buyer = features["n_purchases"] > 0
    # NaN for a buyer means a value outside the bins or a misaligned column. The notebook
    # once left 9,527 buyers' M score NaN (another frame's column, aligned by index) and
    # only sklearn's NaN check caught it; the where() below would hide it as a 0.
    if total[buyer].isna().any():
        raise ValueError(f"{total[buyer].isna().sum()} buyers fall outside the RFM bins")
    return total.where(buyer, 0.0).rename("rfm_score")
