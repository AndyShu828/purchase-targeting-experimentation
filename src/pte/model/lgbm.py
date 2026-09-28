"""LightGBM with early stopping on the validation set, from Andy's P3 notebook (2026-09-28).

Decided 2026-09-28 (leakage.md 3.4): the number of trees is picked on the validation
set, so its scores are slightly optimistic and the test set is kept for the report.
NULLs go in as they are (LightGBM learns a direction for them at each split) and
nothing is scaled or logged. Splits only compare values, but monotone transforms are not
quite free: with log1p on n_events, n_purchases and total_spend, num_leaves 31 with
min_child_samples 500 stopped at 118 trees instead of 205 (validation log loss 0.169338
against 0.169319), while 31 / 20 and 63 / 500 came out identical. Not investigated; the
histogram bin edges are the likely cause.

Decided 2026-09-29 (Andy): TUNED_PARAMS, picked by validation log loss from a 12-run grid
(num_leaves 15 / 31 / 63 / 127 x min_child_samples 20 / 100 / 500 at learning rate 0.1,
logged to MLflow), then rerun at learning rate 0.05. min_child_samples 500 won at every
num_leaves. The whole grid spans 0.169309-0.169400; against the untuned model (31 / 20 /
0.1, 106 trees, 0.169400) the tuned one (145 trees, 0.169269) is 0.000131 lower per user,
z = -8.8 paired over the validation users. Real, but 0.08%.
"""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd
from numpy.typing import ArrayLike

MAX_TREES = 2000  # an upper bound only; early stopping ends far sooner (145 on 20 features)
EARLY_STOPPING_ROUNDS = 50
TUNED_PARAMS = {"num_leaves": 63, "min_child_samples": 500, "learning_rate": 0.05}


def train_lgbm(
    x_train: pd.DataFrame,
    y_train: ArrayLike,
    x_valid: pd.DataFrame,
    y_valid: ArrayLike,
    **params,
) -> lgb.LGBMClassifier:
    """Fit until the validation log loss has not improved for EARLY_STOPPING_ROUNDS trees.

    Uses TUNED_PARAMS unless params override them. The returned model keeps only the
    best_iteration_ trees. verbose=0 keeps LightGBM's warnings and drops its info lines;
    it is set explicitly because verbose=-1 once in a process silences every later model
    that does not set it (LightGBM 4.7), and that is how a categorical_feature that had
    silently turned into NaN went unnoticed.
    """
    params = {"n_estimators": MAX_TREES, "verbose": 0, **TUNED_PARAMS, **params}
    model = lgb.LGBMClassifier(**params)
    model.fit(
        x_train,
        y_train,
        eval_X=x_valid,
        eval_y=y_valid,
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return model
