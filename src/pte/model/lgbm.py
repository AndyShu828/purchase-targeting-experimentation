"""LightGBM with early stopping on the validation set, from Andy's P3 notebook (2026-09-28).

Decided 2026-09-28 (leakage.md 3.4): the number of trees is picked on the validation
set, so its scores are slightly optimistic and the test set is kept for the report.
NULLs go in as they are (LightGBM learns a direction for them at each split) and
nothing is scaled or logged: splits only compare values, so monotone transforms change
nothing.
"""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd
from numpy.typing import ArrayLike

MAX_TREES = 2000  # an upper bound only; early stopping ends far sooner (106 on 23 features)
EARLY_STOPPING_ROUNDS = 50


def train_lgbm(
    x_train: pd.DataFrame,
    y_train: ArrayLike,
    x_valid: pd.DataFrame,
    y_valid: ArrayLike,
    **params,
) -> lgb.LGBMClassifier:
    """Fit until the validation log loss has not improved for EARLY_STOPPING_ROUNDS trees.

    The returned model keeps only the best_iteration_ trees. verbose=0 keeps LightGBM's
    warnings and drops its info lines; it is set explicitly because verbose=-1 once in a
    process silences every later model that does not set it (LightGBM 4.7), and that is
    how a categorical_feature that had silently turned into NaN went unnoticed.
    """
    params = {"n_estimators": MAX_TREES, "verbose": 0, **params}
    model = lgb.LGBMClassifier(**params)
    model.fit(
        x_train,
        y_train,
        eval_X=x_valid,
        eval_y=y_valid,
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return model
