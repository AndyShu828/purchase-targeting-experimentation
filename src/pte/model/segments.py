"""Segment evaluation, from Andy's P3 notebook (2026-09-29).

Decided 2026-09-29: "new vs returning" is bought vs not bought in the feature window, since
the population (anyone active in the 14 days before the cutoff) holds no truly new users and
the training window starts on the first day of data, so "seen before" cannot be computed.
"High vs low activity" is active_days 1 vs 2 or more, cut by value so nothing needs freezing.

Segments are chosen from the features only: at scoring time the label window has not
happened yet.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import rankdata

from pte.model.metrics import evaluate

SEGMENTS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "all": lambda df: pd.Series(True, index=df.index),
    "bought": lambda df: df["n_purchases"] > 0,
    "not bought": lambda df: df["n_purchases"] == 0,
    "active 1 day": lambda df: df["active_days"] == 1,
    "active 2+ days": lambda df: df["active_days"] >= 2,
}


def evaluate_segments(df: pd.DataFrame, score: ArrayLike, probability: bool = True) -> pd.DataFrame:
    """One row per segment: size, positive rate and evaluate()'s metrics on that segment.

    score lines up with df's rows by position. For a probability, mean_score next to
    positive_rate shows whether the segment is calibrated. AUC does not depend on the
    positive rate, but AP starts from it (a random ranking scores about the rate) and
    lift@k% is capped at 1 / rate, so compare those across segments with that in mind.
    """
    score = np.asarray(score, dtype=float)
    label = df["label"].to_numpy()
    rows = {}
    for name, segment in SEGMENTS.items():
        mask = segment(df).to_numpy()
        row = {"n": int(mask.sum()), "positive_rate": float(label[mask].mean())}
        if probability:
            row["mean_score"] = float(score[mask].mean())
        rows[name] = row | evaluate(label[mask], score[mask], probability)
    return pd.DataFrame.from_dict(rows, orient="index")


def pair_auc(score: ArrayLike, label: ArrayLike, group: ArrayLike) -> pd.DataFrame:
    """ROC-AUC split by which group the positive and the negative of each pair come from.

    AUC is the share of (positive, negative) pairs the score puts in the right order, ties
    counting half. Grouping both sides by a 0/1 group gives four kinds of pair; the overall
    AUC is their average weighted by pair counts, which is how both halves of a split can
    score below the whole: the pairs across the halves are missing from each.
    """
    score, label = np.asarray(score, dtype=float), np.asarray(label)
    group = np.asarray(group).astype(bool)
    total = (label == 1).sum() * (label == 0).sum()
    rows = []
    for pos_group in (True, False):
        for neg_group in (True, False):
            pos = score[(label == 1) & (group == pos_group)]
            neg = score[(label == 0) & (group == neg_group)]
            pairs = len(pos) * len(neg)
            auc = np.nan
            if pairs:
                # rank-sum form of the pair count: wins over all negatives, ties as half
                ranks = rankdata(np.concatenate([pos, neg]))
                auc = (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / pairs
            rows.append(
                {
                    "positive_in_group": pos_group,
                    "negative_in_group": neg_group,
                    "auc": auc,
                    "pair_share": pairs / total,
                }
            )
    return pd.DataFrame(rows)
