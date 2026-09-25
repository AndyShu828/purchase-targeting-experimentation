from datetime import date

CUTOFF = date(2019, 11, 18)  # window 11-04 .. 11-17


def _event(user, time, category, price=10.0, brand=None, event_type="view"):
    return {
        "user_id": user,
        "event_type": event_type,
        "event_time": time,
        "category_id": category,
        "price": price,
        "brand": brand,
    }


def test_preference_features(spark, make_events):
    from pyspark.testing import assertDataFrameEqual

    from pte.features.labels import build_population
    from pte.features.preference import build_preference_features

    # Positive prices inside the window, by category:
    #   10: 100 (u1), 200 (u5), 300 (u1)  -> ranks 0.0, 0.5, 1.0
    #   20: 80 (u1), 80 (u1), 90 (u5)     -> ranks 0.0, 0.0, 1.0 (ties share a rank)
    #   30: 10 (u1)                        -> 0.0 (alone in its category)
    # The zero price and the events outside the window would each shift category 10.
    events = make_events(
        [
            # 1: most events tie between 10 and 20; 20 was touched last -> main 20.
            #    Ranks 0, 1, 0, 0, 0 -> 0.2. Brands a, a, b and two NULLs -> 2.
            _event(1, "2019-11-05 10:00:00", 10, 100.0, "a"),
            _event(1, "2019-11-06 10:00:00", 10, 300.0, "a"),
            _event(1, "2019-11-10 10:00:00", 20, 80.0, "b"),
            _event(1, "2019-11-12 10:00:00", 20, 80.0),
            _event(1, "2019-11-08 10:00:00", 30, 10.0),
            # on the cutoff day: label side, must not make 10 the main category or rank
            _event(1, "2019-11-18 00:00:00", 10, 50.0, "z"),
            # before the window: must not add a category or a brand
            _event(1, "2019-11-03 23:59:59", 99, 10.0, "y"),
            # 2: one event each in 50 and 40 at the same instant -> the smaller id, 40
            _event(2, "2019-11-09 12:00:00", 50, 5.0),
            _event(2, "2019-11-09 12:00:00", 40, 5.0),
            # 3: zero price only -> no rank, NULL; no brand -> 0
            _event(3, "2019-11-10 10:00:00", 60, 0.0),
            # 5: the zero price is dropped before ranking -> (0.5 + 1.0) / 2 = 0.75
            _event(5, "2019-11-07 09:00:00", 10, 200.0),
            _event(5, "2019-11-07 09:05:00", 10, 0.0),
            _event(5, "2019-11-08 10:00:00", 20, 90.0),
            _event(5, "2019-11-03 10:00:00", 10, 400.0),  # before the window
        ]
    )
    features = build_preference_features(
        events, build_population(events, CUTOFF), CUTOFF, window_days=14
    ).drop("main_category_conversion_rate")

    expected = spark.createDataFrame(
        [
            (1, 3, 2, 0.2, 20),
            (2, 2, 0, 0.0, 40),
            (3, 1, 0, None, 60),
            (5, 2, 0, 0.75, 10),
        ],
        "user_id bigint, n_categories bigint, n_brands bigint, "
        "price_percentile double, main_category_id bigint",
    )
    assertDataFrameEqual(features, expected)


def test_population_user_with_no_event_in_a_shorter_window_keeps_a_row(spark, make_events):
    from pyspark.testing import assertDataFrameEqual

    from pte.features.labels import build_population
    from pte.features.preference import build_preference_features

    # population 11-04 .. 11-17, 7-day window 11-11 .. 11-17: user 1 is in the first only
    events = make_events([_event(1, "2019-11-06 10:00:00", 70, 15.0, "a")])
    features = build_preference_features(
        events, build_population(events, CUTOFF), CUTOFF, window_days=7
    )

    expected = spark.createDataFrame([(1, 0, 0, None, None, None)], features.schema)
    assertDataFrameEqual(features, expected)


def test_main_category_conversion_rate(spark, make_events):
    from pyspark.testing import assertDataFrameEqual

    from pte.features.labels import build_population
    from pte.features.preference import build_preference_features

    # Category 10 has five (session, category) pairs that count, one with a purchase: 1/5.
    events = make_events(
        [
            # 1: bought twice in one visit -> one pair with a purchase, not two
            _event(1, "2019-11-05 10:00:00", 10),
            _event(1, "2019-11-05 10:05:00", 10, event_type="purchase"),
            _event(1, "2019-11-05 10:06:00", 10, event_type="purchase"),
            # 1: back the next day, no purchase -> a second pair
            _event(1, "2019-11-06 10:00:00", 10),
            # 2: 31 minutes apart on one cookie -> two sessions, two pairs
            _event(2, "2019-11-07 10:00:00", 10),
            _event(2, "2019-11-07 10:31:00", 10),
            # 3: during the outage -> the session is dropped, its pair does not count
            _event(3, "2019-11-15 12:00:00", 10),
            _event(3, "2019-11-15 12:01:00", 10, event_type="cart"),
            # 4: buys after the cutoff in the same visit -> a pair without a purchase
            _event(4, "2019-11-17 23:50:00", 10),
            _event(4, "2019-11-18 00:05:00", 10, event_type="purchase"),
            # 5: its own category, bought -> 1.0
            _event(5, "2019-11-10 10:00:00", 20),
            _event(5, "2019-11-10 10:02:00", 20, event_type="purchase"),
            # 6: its main category only occurs in an outage session -> no rate, NULL
            _event(6, "2019-11-15 09:00:00", 30),
        ]
    )
    features = build_preference_features(
        events, build_population(events, CUTOFF), CUTOFF, window_days=14
    ).select("user_id", "main_category_id", "main_category_conversion_rate")

    expected = spark.createDataFrame(
        [(1, 10, 0.2), (2, 10, 0.2), (3, 10, 0.2), (4, 10, 0.2), (5, 20, 1.0), (6, 30, None)],
        "user_id bigint, main_category_id bigint, main_category_conversion_rate double",
    )
    assertDataFrameEqual(features, expected)
