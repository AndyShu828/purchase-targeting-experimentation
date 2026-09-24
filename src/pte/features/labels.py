"""Sample population and purchase labels, kept apart from the feature code.

Cutoff convention (docs/incident_2019_11_15.md section 4.3): the cutoff is the FIRST day
of the label window. Features see event_date < cutoff; labels see event_date >= cutoff.

Windows are cut on event_date, not event_time: event_date is the UTC date of event_time
(0 mismatches in all 109,950,743 rows), a whole-day boundary needs no timestamp literal,
and event_date is the partition column, so Spark reads only the partitions it needs.
"""

from __future__ import annotations

from datetime import date, timedelta

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# Decided 2026-09-24: the sample is every user with any event in the 14 days before cutoff.
POPULATION_DAYS = 14


def build_population(
    events: DataFrame, cutoff: date, lookback_days: int = POPULATION_DAYS
) -> DataFrame:
    """Users with any event in [cutoff - lookback_days, cutoff): one row per user_id.

    Features and labels are both built for this population, so it is defined once, here.
    Users first seen on or after cutoff are not in it (cold start, out of scope).
    """
    start = cutoff - timedelta(days=lookback_days)
    return (
        events.where((F.col("event_date") >= F.lit(start)) & (F.col("event_date") < F.lit(cutoff)))
        .select("user_id")
        .distinct()
    )


def build_labels(
    events: DataFrame, population: DataFrame, cutoff: date, label_days: int
) -> DataFrame:
    """label = 1 if the user purchased in [cutoff, cutoff + label_days), else 0.

    Returns exactly one row per population user. Negatives come from the population
    (left join, then 0), never from "users who did not show up among the buyers":
    an inner join would silently drop every negative and every row would be 1.

    The data ends on 2019-11-30. A window that runs past it is not an error here; it just
    counts fewer days. So the 11-18 test cutoff must be called with label_days=13.
    """
    end = cutoff + timedelta(days=label_days)
    buyers = (
        events.where(
            (F.col("event_type") == "purchase")
            & (F.col("event_date") >= F.lit(cutoff))
            & (F.col("event_date") < F.lit(end))
        )
        .select("user_id")
        .distinct()
        .withColumn("label", F.lit(1))
    )
    return population.join(buyers, on="user_id", how="left").select(
        "user_id", F.coalesce(F.col("label"), F.lit(0)).alias("label")
    )
