"""Assemble the feature groups for one cutoff, and the labels beside them.

Features and labels are built and written separately (plan v2: label generation is its
own function, kept apart from the features): P3 joins them for training, while P3's batch
scoring and P4's experiment scoring read the features alone, as there are no labels yet.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

from pyspark.sql import DataFrame

from pte.features.activity import build_activity_features
from pte.features.intent import build_intent_features
from pte.features.labels import build_labels, build_population
from pte.features.purchase import build_purchase_features

# Each takes (events, population, cutoff, window_days) and returns one row per population
# user. New groups are appended here.
FEATURE_GROUPS: list[Callable[[DataFrame, DataFrame, date, int], DataFrame]] = [
    build_activity_features,
    build_purchase_features,
    build_intent_features,
]

# Decided 2026-09-24 (docs/incident_2019_11_15.md section 4.3): the cutoff is the first
# label day; the test labels stop at 11-30, the last day of data, so they span 13 days.
CUTOFFS = {
    "train": (date(2019, 10, 15), 14),
    "valid": (date(2019, 11, 1), 14),
    "test": (date(2019, 11, 18), 13),
}

# The window every group looks back over. Equal to the population's 14 days for now.
WINDOW_DAYS = 14


def build_feature_table(
    events: DataFrame, cutoff: date, window_days: int = WINDOW_DAYS
) -> DataFrame:
    """One row per population user, every feature group side by side. No label column."""
    population = build_population(events, cutoff)
    table = population
    for build_group in FEATURE_GROUPS:
        # Every group already holds exactly the population, so this join adds columns
        # only; left keeps that true even if a group ever drops a user.
        table = table.join(
            build_group(events, population, cutoff, window_days), on="user_id", how="left"
        )
    return table


def build_label_table(events: DataFrame, cutoff: date, label_days: int) -> DataFrame:
    """One row per population user: user_id and label, from events on and after cutoff."""
    return build_labels(events, build_population(events, cutoff), cutoff, label_days)
