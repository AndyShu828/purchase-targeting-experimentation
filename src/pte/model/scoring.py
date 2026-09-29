"""Batch scoring: every user in a cutoff's population, with a calibrated probability and a decile.

Decile 1 is the tenth of users with the highest scores, decile 10 the lowest, so decile 1 is
the group lift@10% measures. Deciles are cut within the users scored at that cutoff, by rank
and not by a fixed probability threshold: P4 carries its operating point to a new cutoff as a
top-k% share (leakage.md 3.4), and a threshold would move with the calibration drift.

Ties: pd.qcut cuts on the score values, so users with the same score always share a decile,
at the cost of deciles not being exactly equal. On valid (2026-09-29) the largest tie is
4,098 users, but at most 235 sit on a decile boundary, and the deciles hold 179,227-179,400
users against 179,312 exact. Splitting ties by row order would give exact sizes and put two
identical users in different deciles.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

N_DECILES = 10


def assign_deciles(p: ArrayLike) -> np.ndarray:
    """1 for the highest-scoring tenth, N_DECILES for the lowest; equal scores, equal decile.

    Raises ValueError when ties leave a decile empty. Large ties can do that without two
    edges coinciding: in 20 scores with 0.12 three times, the edges at 50% and 60% are
    0.095 and 0.12 and the one at 70% is 0.129, so the bin (0.12, 0.129] holds nobody
    and the bin below it holds 4. qcut raises on its own only when edges coincide.
    """
    # qcut numbers its bins from the lowest scores up, 0 to N_DECILES - 1
    deciles = N_DECILES - pd.qcut(np.asarray(p, dtype=float), N_DECILES, labels=False)
    empty = sorted(set(range(1, N_DECILES + 1)) - set(deciles.tolist()))
    if empty:
        raise ValueError(f"ties leave decile(s) {empty} empty; too few distinct scores")
    return deciles


def score_users(model, features: pd.DataFrame, cutoff: date, version: str) -> pd.DataFrame:
    """One row per row of features: user_id, cutoff, model_version, p_raw, p, decile.

    model is the registered pyfunc (registry.CalibratedLGBM): it picks its own columns from
    features, and p is the calibrated probability. Rows keep the order of features.
    """
    scores = model.predict(features)
    return pd.DataFrame(
        {
            "user_id": features["user_id"].to_numpy(),
            "cutoff": cutoff,
            "model_version": version,
            "p_raw": scores["p_raw"].to_numpy(),
            "p": scores["p"].to_numpy(),
            "decile": assign_deciles(scores["p"]),
        }
    )


def check_cutoff(cutoff: date, first_day: date, last_day: date, window_days: int) -> None:
    """The feature window [cutoff - window_days, cutoff) must lie inside the data.

    The latest cutoff is the day after the data ends: scoring as of that morning, with
    nothing yet known about the days being predicted. An earlier start would silently
    shorten the window and push every count down.
    """
    earliest, latest = first_day + timedelta(days=window_days), last_day + timedelta(days=1)
    if not earliest <= cutoff <= latest:
        raise ValueError(
            f"cutoff {cutoff} is outside {earliest} .. {latest}: the {window_days}-day "
            f"window needs data on {first_day} .. {last_day}"
        )


def decile_summary(scores: pd.DataFrame) -> pd.DataFrame:
    """Users, mean and lowest calibrated p per decile, decile 1 first."""
    return scores.groupby("decile").agg(
        users=("p", "size"), mean_p=("p", "mean"), min_p=("p", "min")
    )
