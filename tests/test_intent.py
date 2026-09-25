from datetime import date, datetime

CUTOFF = date(2019, 11, 18)  # window 11-04 .. 11-17, outage 11-14 18:09:37 .. 11-16 08:32:34


def _event(user, time, event_type="view", category=10, cookie=None):
    return {
        "user_id": user,
        "user_session": cookie or f"c{user}",
        "event_type": event_type,
        "event_time": time,
        "category_id": category,
    }


def test_cut_sessions_on_the_30_minute_gap_per_user(spark, make_events):
    from pte.features.sessions import cut_sessions

    events = make_events(
        [
            _event(1, "2019-11-05 10:00:00"),
            _event(1, "2019-11-05 10:30:00"),  # gap 1800 s: same session
            _event(1, "2019-11-05 11:00:01"),  # gap 1801 s: new session
            # user 2 shares user 1's cookie; its events must not bridge user 1's gap
            _event(2, "2019-11-05 10:45:00", cookie="c1"),
        ]
    )
    rows = {
        (r.user_id, r.event_time): r.session_index
        for r in cut_sessions(events).select("user_id", "event_time", "session_index").collect()
    }
    assert rows == {
        (1, datetime(2019, 11, 5, 10, 0, 0)): 0,
        (1, datetime(2019, 11, 5, 10, 30, 0)): 0,
        (1, datetime(2019, 11, 5, 11, 0, 1)): 1,
        (2, datetime(2019, 11, 5, 10, 45, 0)): 0,
    }


def test_outage_flag_needs_real_overlap(spark, make_events):
    from pte.features.sessions import cut_sessions

    events = make_events(
        [
            _event(1, "2019-11-14 18:00:00"),
            _event(1, "2019-11-14 18:09:37"),  # ends exactly at the outage start: no overlap
            _event(2, "2019-11-16 08:32:34"),  # starts exactly at the outage end: no overlap
            _event(3, "2019-11-14 18:09:38"),  # one second inside
            _event(4, "2019-11-16 08:20:00"),
            _event(4, "2019-11-16 08:40:00"),  # spans the outage end
        ]
    )
    flags = {
        r.user_id: r.is_outage_session
        for r in cut_sessions(events).select("user_id", "is_outage_session").distinct().collect()
    }
    assert flags == {1: False, 2: False, 3: True, 4: True}


def test_cart_abandon_rate(spark, make_events):
    from pyspark.testing import assertDataFrameEqual

    from pte.features.intent import build_intent_features
    from pte.features.labels import build_population

    events = make_events(
        [
            # 1: three cart pairs, two abandoned (the conversation's worked example) -> 2/3
            _event(1, "2019-11-05 10:00:00", "cart", category=10),
            _event(1, "2019-11-05 10:01:00", "cart", category=10),
            _event(1, "2019-11-05 10:05:00", "purchase", category=10),
            _event(1, "2019-11-05 10:06:00", "cart", category=20),  # same session, other category
            _event(1, "2019-11-05 10:07:00", "view", category=30),  # no cart: not counted
            _event(1, "2019-11-08 09:00:00", "cart", category=10),  # new session, same category
            _event(1, "2019-11-15 09:00:00", "cart", category=40),  # outage session: dropped
            # 2: views only -> NULL
            _event(2, "2019-11-06 10:00:00"),
            # 3: only carted during the outage -> no reliable cart pair -> NULL
            _event(3, "2019-11-15 12:00:00", "cart"),
            # 4: carts A, buys A' of the same category -> bought -> 0
            _event(4, "2019-11-07 10:00:00", "cart", category=50),
            _event(4, "2019-11-07 10:03:00", "purchase", category=50),
            # 5: carts before cutoff, buys after it in the same visit -> abandoned as of cutoff
            _event(5, "2019-11-17 23:50:00", "cart", category=60),
            _event(5, "2019-11-18 00:05:00", "purchase", category=60),
        ]
    )
    features = build_intent_features(
        events, build_population(events, CUTOFF), CUTOFF, window_days=14
    ).select("user_id", "cart_abandon_rate")

    expected = spark.createDataFrame(
        [(1, 2 / 3), (2, None), (3, None), (4, 0.0), (5, 1.0)],
        "user_id bigint, cart_abandon_rate double",
    )
    assertDataFrameEqual(features, expected)


def test_population_user_with_no_event_in_a_shorter_window_keeps_a_null_row(spark, make_events):
    from pyspark.testing import assertDataFrameEqual

    from pte.features.intent import build_intent_features
    from pte.features.labels import build_population

    # population 11-04 .. 11-17; the 7-day window 11-11 .. 11-17 holds nothing of user 1
    events = make_events([_event(1, "2019-11-05 10:00:00", "cart")])
    features = build_intent_features(
        events, build_population(events, CUTOFF), CUTOFF, window_days=7
    )

    expected = spark.createDataFrame(
        [(1, None, None)], "user_id bigint, cart_abandon_rate double, view_to_cart_rate double"
    )
    assertDataFrameEqual(features, expected)


def test_view_to_cart_rate(spark, make_events):
    from pyspark.testing import assertDataFrameEqual

    from pte.features.intent import build_intent_features
    from pte.features.labels import build_population

    events = make_events(
        [
            # 1: in one session, category 10 viewed twice and carted, 20 viewed only -> 1/2
            _event(1, "2019-11-05 10:00:00", "view", category=10),
            _event(1, "2019-11-05 10:01:00", "view", category=10),
            _event(1, "2019-11-05 10:02:00", "cart", category=10),
            _event(1, "2019-11-05 10:03:00", "view", category=20),
            # 2: a cart without a view in its pair is not a conversion -> 0/1
            _event(2, "2019-11-06 10:00:00", "cart", category=30),
            _event(2, "2019-11-06 10:01:00", "view", category=40),
            # 3: only an outage session, which is kept here (decided 2026-09-26) -> 1/1
            _event(3, "2019-11-15 12:00:00", "view", category=50),
            _event(3, "2019-11-15 12:01:00", "cart", category=50),
            # 4: carts but no view -> no view pair -> NULL
            _event(4, "2019-11-07 10:00:00", "cart", category=10),
            # 5: viewed, then carted the same category in a later session -> two pairs -> 0/1
            _event(5, "2019-11-08 10:00:00", "view", category=10),
            _event(5, "2019-11-08 10:31:00", "cart", category=10),
            # 6: viewed one category and carted another in one session -> 0/1
            _event(6, "2019-11-09 10:00:00", "view", category=60),
            _event(6, "2019-11-09 10:01:00", "cart", category=70),
            # 7: viewed before cutoff, carted after it in the same visit -> 0/1 as of cutoff
            _event(7, "2019-11-17 23:50:00", "view", category=10),
            _event(7, "2019-11-18 00:05:00", "cart", category=10),
        ]
    )
    features = build_intent_features(
        events, build_population(events, CUTOFF), CUTOFF, window_days=14
    ).select("user_id", "view_to_cart_rate")

    expected = spark.createDataFrame(
        [(1, 0.5), (2, 0.0), (3, 1.0), (4, None), (5, 0.0), (6, 0.0), (7, 0.0)],
        "user_id bigint, view_to_cart_rate double",
    )
    assertDataFrameEqual(features, expected)
