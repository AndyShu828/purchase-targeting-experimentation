"""Trend feature group: is a user's activity rising or falling going into the cutoff.

Translated from Andy's DuckDB query (2026-09-25). Plan v2 defines it as events in the
last 7 days divided by events in the 7 days before, so the window is fixed at
[cutoff - 14, cutoff) on event_date and split at cutoff - 7, whatever window_days is.
"""

from __future__ import annotations

from datetime import date, timedelta

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

HALF_DAYS = 7


def build_trend_features(
    events: DataFrame, population: DataFrame, cutoff: date, window_days: int
) -> DataFrame:
    """One row per population user: recent_to_previous_event_ratio, all event types.

    Decided 2026-09-25 (option a, the plan's own ratio): NULL when the previous 7 days
    have no event, 0 when the last 7 have none. The share recent / (recent + previous)
    was rejected: it equals this ratio with NULL filled by 1, and on the training cutoff
    those users buy at 4.97% against 13.79% for their neighbours at 0.8-1.0.

    window_days is accepted for the common signature (build.FEATURE_GROUPS) and unused.
    Both counts come from one aggregation, so an empty half counts 0; nullif turns a zero
    denominator into NULL, where Spark's ANSI mode would raise DIVIDE_BY_ZERO.
    """
    split = cutoff - timedelta(days=HALF_DAYS)
    start = cutoff - timedelta(days=2 * HALF_DAYS)
    is_recent = F.col("event_date") >= F.lit(split)
    counts = (
        events.where((F.col("event_date") >= F.lit(start)) & (F.col("event_date") < F.lit(cutoff)))
        .groupBy("user_id")
        .agg(
            F.count(F.when(is_recent, 1)).alias("n_recent_events"),
            F.count(F.when(~is_recent, 1)).alias("n_previous_events"),
        )
    )
    return population.join(counts, on="user_id", how="left").select(
        "user_id",
        (F.col("n_recent_events") / F.nullif(F.col("n_previous_events"), F.lit(0))).alias(
            "recent_to_previous_event_ratio"
        ),
    )
