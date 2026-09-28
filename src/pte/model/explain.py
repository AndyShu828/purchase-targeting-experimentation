"""SHAP values for the LightGBM model, from LightGBM's own TreeSHAP (predict(pred_contrib=True)).

A user's SHAP values split the model's raw score (the logit) into one contribution per
feature: base + sum of contributions = logit, exactly (checked to 1e-14 on valid). No shap
package is needed. The exact computation is slow, 7 minutes for all 1.79M validation users
on 145 trees against 23 seconds for 100,000, so a random sample stands in for the set.

Three importances answer different questions and disagree here (2026-09-29, valid):
gain is the loss reduction from a feature's splits, split is how often it is split on, and
mean |SHAP| is how far it moves a typical user's logit. A feature split once near the root
over everyone gets a large gain from few splits; one used deep in many small branches gets
many splits and a small effect on each user.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

SAMPLE_SIZE = 100_000


def shap_values(
    model: lgb.LGBMClassifier, x: pd.DataFrame, n: int | None = SAMPLE_SIZE, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """(feature values, SHAP values, base) for a random sample of n rows of x (all if None).

    Both frames keep x's index and columns, one SHAP column per feature, so a user's values
    and contributions line up by label. base is the logit before any feature is known.
    """
    if n is not None and n < len(x):
        x = x.sample(n, random_state=seed)
    contrib = model.predict(x, pred_contrib=True)
    values = pd.DataFrame(contrib[:, :-1], index=x.index, columns=x.columns)
    return x, values, float(contrib[0, -1])


def importance(model: lgb.LGBMClassifier, values: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| per feature with its share, next to gain and split shares, largest first."""
    booster = model.booster_
    gain = booster.feature_importance("gain")
    split = booster.feature_importance("split")
    table = pd.DataFrame(
        {
            "mean_abs_shap": values.abs().mean(),
            "gain_share": pd.Series(gain / gain.sum(), index=booster.feature_name()),
            "split_share": pd.Series(split / split.sum(), index=booster.feature_name()),
        }
    )
    table.insert(1, "shap_share", table["mean_abs_shap"] / table["mean_abs_shap"].sum())
    return table.sort_values("mean_abs_shap", ascending=False)


def dependence(
    feature: pd.Series, shap: pd.Series, bins: int = 10, max_levels: int = 15
) -> pd.DataFrame:
    """Mean SHAP by value of a feature, low to high, NULLs in a row of their own.

    Reads the direction of a feature's effect: whether higher values push the logit up or
    down, and where it turns. A feature with at most max_levels distinct values gets a row
    per value (active_days, 1-14); otherwise 0 gets its own row and the rest is cut into
    quantile bins. Without that, a mostly-zero feature such as n_purchases_per_day (90% of
    users at 0) collapses into a single bin; ties can still merge bins elsewhere.
    """
    df = pd.DataFrame({"value": feature, "shap": shap})
    null = df["value"].isna()
    if df["value"].nunique() <= max_levels:
        df["bin"] = df["value"].astype(str)
    else:
        zero = df["value"] == 0
        df.loc[zero, "bin"] = "0"
        rest = ~null & ~zero
        df.loc[rest, "bin"] = pd.qcut(df.loc[rest, "value"], bins, duplicates="drop").astype(str)
    df.loc[null, "bin"] = "NULL"
    table = df.groupby("bin", sort=False).agg(
        n=("shap", "size"),
        low=("value", "min"),
        high=("value", "max"),
        mean_shap=("shap", "mean"),
    )
    order = table["low"].fillna(np.inf).sort_values(kind="stable").index
    return table.loc[order]
