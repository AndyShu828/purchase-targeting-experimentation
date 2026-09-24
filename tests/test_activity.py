from datetime import date

CUTOFF = date(2019, 11, 18)  # 7-day window: 11-11 .. 11-17; population: 11-04 .. 11-17


def test_activity_features(spark, make_events):
    from pyspark.testing import assertDataFrameEqual

    from pte.features.activity import build_activity_features
    from pte.features.labels import build_population

    events = make_events(
        [
            # 1: active on three days of the window; one event each just outside both ends
            {"user_id": 1, "event_type": "view", "event_time": "2019-11-10 23:59:59"},
            {"user_id": 1, "event_type": "view", "event_time": "2019-11-11 00:00:00"},
            {"user_id": 1, "event_type": "view", "event_time": "2019-11-15 12:00:00"},
            {"user_id": 1, "event_type": "cart", "event_time": "2019-11-17 08:00:00"},
            {"user_id": 1, "event_type": "purchase", "event_time": "2019-11-17 09:00:00"},
            {"user_id": 1, "event_type": "view", "event_time": "2019-11-18 00:00:00"},
            # 2: in the population, nothing in the window -> 0 counts, NULL recency
            {"user_id": 2, "event_type": "view", "event_time": "2019-11-06 10:00:00"},
            # 3: only on the window's first day -> recency 7
            {"user_id": 3, "event_type": "cart", "event_time": "2019-11-11 00:00:00"},
            # 4: first seen after cutoff -> not in the population, no row
            {"user_id": 4, "event_type": "view", "event_time": "2019-11-20 10:00:00"},
        ]
    )
    features = build_activity_features(
        events, build_population(events, CUTOFF), CUTOFF, window_days=7
    )

    expected = spark.createDataFrame(
        [(1, 4, 2, 1, 3, 1), (2, 0, 0, 0, 0, None), (3, 1, 0, 1, 1, 7)],
        "user_id bigint, n_events bigint, n_views bigint, n_carts bigint, "
        "active_days bigint, days_since_last_event int",
    )
    assertDataFrameEqual(features, expected)
