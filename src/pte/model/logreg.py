"""Baseline 2: logistic regression, from Andy's P3 notebook (2026-09-28).

Decided 2026-09-28 (leakage.md 3.4): NULLs are filled with the training median plus a
missing-indicator column, then standardized, all inside one Pipeline so that fit sees
only the training set and the median, mean and scale are frozen for scoring.
Features go in raw: log1p on the long-tailed counts improved Brier but cost lift@10%
(4.1454 -> 4.0456 on four features), because one coefficient per feature has to serve
buyers and non-buyers at once.
"""

from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def build_logreg() -> Pipeline:
    """Unfitted pipeline; add_indicator adds one 0/1 column per feature with a training NULL."""
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(),
    )
