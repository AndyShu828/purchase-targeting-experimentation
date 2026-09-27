"""Evaluation metrics sklearn does not provide: lift at the top frac of a ranking.

lift_at follows Andy's function from the P3 notebook (2026-09-28) step for step, with
input checks added.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def lift_at(score: ArrayLike, label: ArrayLike, frac: float) -> float:
    """Positive rate among the top k = int(n * frac) users by score, over the overall rate.

    Ties: decided 2026-09-27 (decision 5), a tied group that straddles the cutoff counts
    its expected positives. With n_A users and P_A positives ranked above the boundary
    group T (n_T users, P_T positives), m = k - n_A of T are taken, and they hold
    m * P_T / n_T positives on average over every order of T (the hypergeometric mean).
    So P_k = P_A + m * P_T / n_T and lift = (P_k / k) / (P / n).
    Taking T's rows in table order instead is not neutral: on the RFM validation set it
    gave lift@20% = 2.4056 against the expected 2.3889.

    score and label are taken by position, not index, so two columns from different
    frames cannot misalign into NaN.
    """
    df = pd.DataFrame({"score": np.asarray(score), "label": np.asarray(label)})
    if df["score"].isna().any():
        raise ValueError("score has NaN; groupby would silently drop those users")
    k = int(len(df) * frac)
    if not 1 <= k <= len(df):
        raise ValueError(f"frac={frac} selects k={k} of {len(df)} users")

    groups = (
        df.groupby("score")
        .agg(n=("label", "size"), pos=("label", "sum"))
        .sort_index(ascending=False)
    )
    groups["n_before"] = groups["n"].cumsum() - groups["n"]
    groups["pos_before"] = groups["pos"].cumsum() - groups["pos"]

    # n_before < k <= n_before + n holds for exactly one group, even when k falls on
    # the end of a group; .item() raises if that ever breaks.
    boundary = groups[(groups["n_before"] < k) & (groups["n_before"] + groups["n"] >= k)]
    m = k - boundary["n_before"].item()
    pos_k = boundary["pos_before"].item() + m * boundary["pos"].item() / boundary["n"].item()
    return float((pos_k / k) / (groups["pos"].sum() / len(df)))
