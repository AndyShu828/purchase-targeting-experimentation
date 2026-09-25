from datetime import date

CUTOFF = date(2019, 11, 18)  # window 11-04 .. 11-17, outage 11-14 18:09:37 .. 11-16 08:32:34


def _event(user, time, cookie):
    return {"user_id": user, "user_session": cookie, "event_type": "view", "event_time": time}


def test_session_features(spark, make_events):
    from pyspark.testing import assertDataFrameEqual

    from pte.features.labels import build_population
    from pte.features.session import build_session_features

    events = make_events(
        [
            # 1: gap 1800 s stays in the session, 1801 s starts one; a second cookie adds a
            #    third -> 3 sessions of 2, 1 and 2 events, (1800 + 0 + 60) / 3 = 620 s
            _event(1, "2019-11-05 10:00:00", "c1"),
            _event(1, "2019-11-05 10:30:00", "c1"),
            _event(1, "2019-11-05 11:00:01", "c1"),
            _event(1, "2019-11-06 09:00:00", "d1"),
            _event(1, "2019-11-06 09:01:00", "d1"),
            # 2: on user 1's cookie in between; its own single-event session -> 0 s
            _event(2, "2019-11-05 10:45:00", "c1"),
            # 3, 4: NULL cookies are kept and never shared between users
            _event(3, "2019-11-07 10:00:00", None),
            _event(3, "2019-11-07 10:10:00", None),
            _event(4, "2019-11-07 10:05:00", None),
            # 5: sessions crossing the window start and the cutoff keep only their inside
            #    event each -> 2 single-event sessions
            _event(5, "2019-11-03 23:55:00", "c5"),
            _event(5, "2019-11-04 00:05:00", "c5"),
            _event(5, "2019-11-17 23:50:00", "c5"),
            _event(5, "2019-11-18 00:05:00", "c5"),
            # 6: an outage session is kept (decided 2026-09-26) -> 300 s
            _event(6, "2019-11-15 10:00:00", "c6"),
            _event(6, "2019-11-15 10:05:00", "c6"),
            # 7: two events in the same second are one session, not two
            _event(7, "2019-11-08 12:00:00", "c7"),
            _event(7, "2019-11-08 12:00:00", "c7"),
            _event(7, "2019-11-08 13:00:00", "c7"),
            # 8: first seen after cutoff -> not in the population, no row
            _event(8, "2019-11-20 10:00:00", "c8"),
        ]
    )
    features = build_session_features(
        events, build_population(events, CUTOFF), CUTOFF, window_days=14
    )

    expected = spark.createDataFrame(
        [
            (1, 620.0, 5 / 3, 3),
            (2, 0.0, 1.0, 1),
            (3, 600.0, 2.0, 1),
            (4, 0.0, 1.0, 1),
            (5, 0.0, 1.0, 2),
            (6, 300.0, 2.0, 1),
            (7, 0.0, 1.5, 2),
        ],
        "user_id bigint, avg_session_seconds double, events_per_session double, n_sessions bigint",
    )
    assertDataFrameEqual(features, expected)


def test_population_user_with_no_event_in_the_window_keeps_a_row(spark, make_events):
    """0 sessions and NULL averages: a count is 0 when nothing happened, a mean undefined."""
    from pte.features.labels import build_population
    from pte.features.session import build_session_features

    events = make_events(
        [
            # 11-01 is inside a 21-day population but before the 14-day window
            _event(1, "2019-11-01 10:00:00", "c1"),
            _event(2, "2019-11-12 10:00:00", "c2"),
        ]
    )
    population = build_population(events, CUTOFF, lookback_days=21)
    rows = build_session_features(events, population, CUTOFF, window_days=14).collect()
    assert sorted(
        (r.user_id, r.avg_session_seconds, r.events_per_session, r.n_sessions) for r in rows
    ) == [(1, None, None, 0), (2, 0.0, 1.0, 1)]
