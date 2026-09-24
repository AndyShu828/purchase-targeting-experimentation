"""Activity feature group: how much, and how recently, a user was active before cutoff.

Translated from Andy's DuckDB query (2026-09-25); same cutoff convention as labels.py:
the window is [cutoff - window_days, cutoff) on event_date.
"""

from __future__ import annotations

from datetime import date, timedelta

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# Counts are 0 for a population user with no event in the window.
COUNT_COLUMNS = ["n_events", "n_views", "n_carts", "active_days"]


def build_activity_features(
    events: DataFrame, population: DataFrame, cutoff: date, window_days: int
) -> DataFrame:
    """One row per population user: four counts and days_since_last_event.

    The population looks back 14 days, the window may be shorter, so some users have no
    event in it. Their counts are 0 and days_since_last_event stays NULL (decided
    2026-09-24: LightGBM handles missing values; the logistic baseline imputes in P3,
    with statistics fitted on the training set only).
    """
    start = cutoff - timedelta(days=window_days)
    active = (
        events.where((F.col("event_date") >= F.lit(start)) & (F.col("event_date") < F.lit(cutoff)))
        .groupBy("user_id")
        .agg(
            F.count("*").alias("n_events"),
            F.count(F.when(F.col("event_type") == "view", 1)).alias("n_views"),
            F.count(F.when(F.col("event_type") == "cart", 1)).alias("n_carts"),
            F.countDistinct("event_date").alias("active_days"),
            # Spark's datediff is (end, start); DuckDB's date_diff is ('day', start, end).
            F.datediff(F.lit(cutoff), F.max("event_date")).alias("days_since_last_event"),
        )
    )
    return population.join(active, on="user_id", how="left").select(
        "user_id",
        *[F.coalesce(F.col(c), F.lit(0)).alias(c) for c in COUNT_COLUMNS],
        "days_since_last_event",
    )
