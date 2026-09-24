"""Preference feature group: what, and at which price level, a user looks at.

Translated from Andy's DuckDB queries (2026-09-25); same window as the other groups:
[cutoff - window_days, cutoff) on event_date. main_category_conversion_rate is to come.
"""

from __future__ import annotations

from datetime import date, timedelta

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

# Counts are 0 for a population user with no event in the window.
COUNT_COLUMNS = ["n_categories", "n_brands"]


def build_preference_features(
    events: DataFrame, population: DataFrame, cutoff: date, window_days: int
) -> DataFrame:
    """One row per population user: n_categories, n_brands, price_percentile, main_category_id.

    price_percentile averages, over the user's events with price > 0, each event's
    PERCENT_RANK by price within its category_id. Zero prices are a collection fault
    (P1) and are filtered out before ranking, and the ranking sees only the window, so
    nothing after the cutoff moves it. NULL when the user has no positive-price event.

    main_category_id is the category with the most events; ties go to the one touched
    most recently, then to the smaller id, so the pick is deterministic (decided
    2026-09-25). NULL only when the user has no event in the window.
    """
    start = cutoff - timedelta(days=window_days)
    windowed = events.where(
        (F.col("event_date") >= F.lit(start)) & (F.col("event_date") < F.lit(cutoff))
    )

    counts = windowed.groupBy("user_id").agg(
        F.countDistinct("category_id").alias("n_categories"),
        # countDistinct skips NULL, so events without a brand add nothing.
        F.countDistinct("brand").alias("n_brands"),
    )

    by_category = Window.partitionBy("category_id").orderBy("price")
    price_percentiles = (
        windowed.where(F.col("price") > 0)
        .withColumn("percent_rank", F.percent_rank().over(by_category))
        .groupBy("user_id")
        .agg(F.avg("percent_rank").alias("price_percentile"))
    )

    # Spark sorts NULLs first in ascending order and DuckDB last; category_id has no
    # NULLs in the data, but nulls_last keeps the two engines' tie-break identical.
    by_user = Window.partitionBy("user_id").orderBy(
        F.desc("n_events"), F.desc("last_event_time"), F.asc_nulls_last("category_id")
    )
    main_categories = (
        windowed.groupBy("user_id", "category_id")
        .agg(F.count("*").alias("n_events"), F.max("event_time").alias("last_event_time"))
        .withColumn("rn", F.row_number().over(by_user))
        .where(F.col("rn") == 1)
        .select("user_id", F.col("category_id").alias("main_category_id"))
    )

    return (
        population.join(counts, on="user_id", how="left")
        .join(price_percentiles, on="user_id", how="left")
        .join(main_categories, on="user_id", how="left")
        .select(
            "user_id",
            *[F.coalesce(F.col(c), F.lit(0)).alias(c) for c in COUNT_COLUMNS],
            "price_percentile",
            "main_category_id",
        )
    )
