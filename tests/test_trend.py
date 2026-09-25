from datetime import date

CUTOFF = date(2019, 11, 18)  # previous 11-04 .. 11-10, recent 11-11 .. 11-17


def test_trend_features(spark, make_events):
    from pyspark.testing import assertDataFrameEqual

    from pte.features.labels import build_population
    from pte.features.trend import build_trend_features

    def event(user_id, event_time, event_type="view"):
        return {"user_id": user_id, "event_type": event_type, "event_time": event_time}

    events = make_events(
        [
            # 1: the split day's first second is recent, the second before it previous
            #    -> 2 recent / 1 previous = 2.0
            event(1, "2019-11-10 23:59:59"),
            event(1, "2019-11-11 00:00:00"),
            event(1, "2019-11-15 12:00:00"),
            # 2: only the previous half -> 0 recent / 2 previous = 0.0, not NULL
            event(2, "2019-11-05 10:00:00"),
            event(2, "2019-11-06 10:00:00"),
            # 3: only the recent half -> no previous event, NULL rather than inf
            event(3, "2019-11-16 10:00:00"),
            # 4: one event per half on the window's first and last days -> 1.0; the cutoff
            #    day and the day before the window must not count
            event(4, "2019-11-03 23:59:59"),
            event(4, "2019-11-04 00:00:00"),
            event(4, "2019-11-17 23:59:59"),
            event(4, "2019-11-18 00:00:00", "purchase"),
            # 5: every event type counts -> 3 recent / 2 previous = 1.5
            event(5, "2019-11-07 10:00:00"),
            event(5, "2019-11-08 10:00:00", "cart"),
            event(5, "2019-11-12 10:00:00"),
            event(5, "2019-11-12 10:05:00", "cart"),
            event(5, "2019-11-12 10:10:00", "purchase"),
            # 6: first seen after cutoff -> not in the population, no row
            event(6, "2019-11-20 10:00:00"),
        ]
    )
    features = build_trend_features(
        events, build_population(events, CUTOFF), CUTOFF, window_days=14
    )

    expected = spark.createDataFrame(
        [(1, 2.0), (2, 0.0), (3, None), (4, 1.0), (5, 1.5)],
        "user_id bigint, recent_to_previous_event_ratio double",
    )
    assertDataFrameEqual(features, expected)


def test_trend_window_ignores_window_days(spark, make_events):
    """The plan fixes the halves at 7 days; a shorter feature window must not move them."""
    from pte.features.labels import build_population
    from pte.features.trend import build_trend_features

    # 11-12 is recent under the 7-day split (11-11) but previous under any later one, and
    # 11-05 drops out of any window shorter than 14 days: either slip changes the 1.0.
    events = make_events(
        [
            {"user_id": 1, "event_type": "view", "event_time": "2019-11-05 10:00:00"},
            {"user_id": 1, "event_type": "view", "event_time": "2019-11-12 10:00:00"},
        ]
    )
    population = build_population(events, CUTOFF)
    for window_days in (7, 14):
        rows = build_trend_features(events, population, CUTOFF, window_days).collect()
        assert [(r.user_id, r.recent_to_previous_event_ratio) for r in rows] == [(1, 1.0)]


def test_population_user_with_no_event_in_the_trend_window_keeps_a_null_row(spark, make_events):
    """The population is 14 days today, like the trend window; a longer one must keep rows."""
    from pte.features.labels import build_population
    from pte.features.trend import build_trend_features

    events = make_events(
        [
            # 11-01 is inside a 21-day population but before the trend window
            {"user_id": 1, "event_type": "view", "event_time": "2019-11-01 10:00:00"},
            {"user_id": 2, "event_type": "view", "event_time": "2019-11-12 10:00:00"},
        ]
    )
    population = build_population(events, CUTOFF, lookback_days=21)
    rows = build_trend_features(events, population, CUTOFF, window_days=14).collect()
    assert sorted((r.user_id, r.recent_to_previous_event_ratio) for r in rows) == [
        (1, None),
        (2, None),
    ]
