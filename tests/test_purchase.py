from datetime import date

import pytest

from pte.features.outage import effective_days

OUTAGE_DAYS = 38.3825 / 24


@pytest.mark.parametrize(
    ("cutoff", "window_days", "expected"),
    [
        (date(2019, 10, 15), 14, 14.0),  # training: October, no overlap
        (date(2019, 11, 1), 14, 14.0),  # validation: ends 10-31, no overlap
        (date(2019, 11, 18), 14, 14 - OUTAGE_DAYS),  # test: whole outage inside
        (date(2019, 11, 18), 7, 7 - OUTAGE_DAYS),  # 11-11 .. 11-17 still covers all of it
        (date(2019, 11, 15), 1, 1 - (24 - (18 + 9 / 60 + 37 / 3600)) / 24),  # 11-14 only
        (date(2019, 11, 14), 7, 7.0),  # ends at 11-14 00:00, before the outage starts
    ],
)
def test_effective_days(cutoff, window_days, expected):
    assert effective_days(cutoff, window_days) == pytest.approx(expected)


def test_purchase_features(spark, make_events):
    from pyspark.testing import assertDataFrameEqual

    from pte.features.labels import build_population
    from pte.features.purchase import build_purchase_features

    cutoff = date(2019, 10, 15)  # window 10-01 .. 10-14, no outage: rates are count / 14
    events = make_events(
        [
            # 1: two purchases on one day, one on another -> 3 purchases on 2 days
            {
                "user_id": 1,
                "event_type": "purchase",
                "event_time": "2019-10-05 09:00:00",
                "price": 10.0,
            },
            {
                "user_id": 1,
                "event_type": "purchase",
                "event_time": "2019-10-05 10:00:00",
                "price": 20.0,
            },
            {
                "user_id": 1,
                "event_type": "view",
                "event_time": "2019-10-08 10:00:00",
                "price": 99.0,
            },
            {
                "user_id": 1,
                "event_type": "purchase",
                "event_time": "2019-10-10 23:59:59",
                "price": 30.0,
            },
            # 2: views only -> zeros, NULL avg_price and recency
            {"user_id": 2, "event_type": "view", "event_time": "2019-10-03 10:00:00", "price": 5.0},
            # 3: buys on the cutoff day -> label side, not a feature
            {"user_id": 3, "event_type": "view", "event_time": "2019-10-02 10:00:00", "price": 7.0},
            {
                "user_id": 3,
                "event_type": "purchase",
                "event_time": "2019-10-15 00:00:00",
                "price": 7.0,
            },
            # 4: buys at the first instant of the window -> recency 14
            {
                "user_id": 4,
                "event_type": "purchase",
                "event_time": "2019-10-01 00:00:00",
                "price": 4.0,
            },
        ]
    )
    features = build_purchase_features(
        events, build_population(events, cutoff), cutoff, window_days=14
    )

    expected = spark.createDataFrame(
        [
            (1, 3, 2, 60.0, 3 / 14, 2 / 14, 60 / 14, 20.0, 5),
            (2, 0, 0, 0.0, 0.0, 0.0, 0.0, None, None),
            (3, 0, 0, 0.0, 0.0, 0.0, 0.0, None, None),
            (4, 1, 1, 4.0, 1 / 14, 1 / 14, 4 / 14, 4.0, 14),
        ],
        "user_id bigint, n_purchases bigint, purchase_days bigint, total_spend double, "
        "n_purchases_per_day double, purchase_days_per_day double, "
        "total_spend_per_day double, avg_price double, days_since_last_purchase int",
    )
    assertDataFrameEqual(features, expected)


def test_population_user_with_no_event_in_a_shorter_window_keeps_a_row(spark, make_events):
    from pyspark.testing import assertDataFrameEqual

    from pte.features.labels import build_population
    from pte.features.purchase import build_purchase_features

    cutoff = date(2019, 10, 15)  # population 10-01 .. 10-14, 7-day window 10-08 .. 10-14
    events = make_events(
        [{"user_id": 1, "event_type": "purchase", "event_time": "2019-10-02 10:00:00"}]
    )
    features = build_purchase_features(
        events, build_population(events, cutoff), cutoff, window_days=7
    )

    expected = spark.createDataFrame([(1, 0, 0, 0.0, 0.0, 0.0, 0.0, None, None)], features.schema)
    assertDataFrameEqual(features, expected)


def test_purchase_rates_use_effective_days_across_the_outage(spark, make_events):
    from pyspark.testing import assertDataFrameEqual

    from pte.features.labels import build_population
    from pte.features.purchase import build_purchase_features

    cutoff = date(2019, 11, 18)
    events = make_events(
        [
            {
                "user_id": 1,
                "event_type": "purchase",
                "event_time": "2019-11-10 12:00:00",
                "price": 12.4,
            },
        ]
    )
    features = build_purchase_features(
        events, build_population(events, cutoff), cutoff, window_days=14
    ).select("n_purchases_per_day")

    expected = spark.createDataFrame([(1 / (14 - OUTAGE_DAYS),)], "n_purchases_per_day double")
    assertDataFrameEqual(features, expected)
