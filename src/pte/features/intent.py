"""Intent feature group: view-to-cart and cart abandonment over (session, category) pairs.

Translated from Andy's DuckDB query (2026-09-25; view_to_cart_rate 2026-09-26). Same unit
as P1's cart_abandonment_by_category.sql: a pair is abandoned if it holds a cart and no
purchase. Carting A and buying A' of the same category counts as bought (decided
2026-09-25).
"""

from __future__ import annotations

from datetime import date, timedelta

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from pte.features.sessions import cut_sessions


def build_intent_features(
    events: DataFrame, population: DataFrame, cutoff: date, window_days: int
) -> DataFrame:
    """One row per population user: cart_abandon_rate and view_to_cart_rate.

    cart_abandon_rate = abandoned pairs / cart pairs. Pairs in sessions overlapping the
    purchase outage are dropped: they record carts but cannot record purchases. A user
    with no remaining cart pair gets NULL (decided 2026-09-25): 0 would claim every cart
    was bought, 1 that none was.

    view_to_cart_rate = pairs with a view and a cart / pairs with a view, order ignored.
    Outage pairs are kept (decided 2026-09-26): only features that use purchases drop
    them. A user with no view pair gets NULL.
    """
    start = cutoff - timedelta(days=window_days)
    windowed = events.where(
        (F.col("event_date") >= F.lit(start)) & (F.col("event_date") < F.lit(cutoff))
    )
    pairs = (
        cut_sessions(windowed)
        .groupBy("user_id", "user_session", "session_index", "category_id")
        .agg(
            F.max((F.col("event_type") == "view").cast("int")).alias("has_view"),
            F.max((F.col("event_type") == "cart").cast("int")).alias("has_cart"),
            F.max((F.col("event_type") == "purchase").cast("int")).alias("has_purchase"),
            F.max(F.col("is_outage_session").cast("int")).alias("is_outage_session"),
        )
    )
    counted = (F.col("is_outage_session") == 0) & (F.col("has_cart") == 1)
    viewed = F.col("has_view") == 1
    user_pair_counts = pairs.groupBy("user_id").agg(
        F.count(F.when(counted, 1)).alias("cart_pairs"),
        F.count(F.when(counted & (F.col("has_purchase") == 0), 1)).alias("abandoned_pairs"),
        F.count(F.when(viewed, 1)).alias("view_pairs"),
        F.count(F.when(viewed & (F.col("has_cart") == 1), 1)).alias("view_cart_pairs"),
    )
    return population.join(user_pair_counts, on="user_id", how="left").select(
        "user_id",
        (F.col("abandoned_pairs") / F.nullif(F.col("cart_pairs"), F.lit(0))).alias(
            "cart_abandon_rate"
        ),
        (F.col("view_cart_pairs") / F.nullif(F.col("view_pairs"), F.lit(0))).alias(
            "view_to_cart_rate"
        ),
    )
