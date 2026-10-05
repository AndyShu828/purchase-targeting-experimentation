from datetime import date

import pytest

CUTOFF = date(2019, 11, 18)


@pytest.fixture
def events(make_events):
    """Six users around the 2019-11-18 cutoff, one edge case each."""
    return make_events(
        [
            # 1: active before cutoff, buys on the cutoff day at 00:00:00 -> label 1
            {"user_id": 1, "event_type": "view", "event_time": "2019-11-10 12:00:00"},
            {"user_id": 1, "event_type": "purchase", "event_time": "2019-11-18 00:00:00"},
            # 2: buys one second before cutoff, nothing after -> feature side, label 0
            {"user_id": 2, "event_type": "purchase", "event_time": "2019-11-17 23:59:59"},
            # 3: only seen inside the label window -> not in the population at all
            {"user_id": 3, "event_type": "purchase", "event_time": "2019-11-20 09:00:00"},
            # 6: first seen on the cutoff day itself -> label side only, not in the population
            {"user_id": 6, "event_type": "view", "event_time": "2019-11-18 00:00:00"},
            # 4: last event 15 days before cutoff -> outside the 14-day lookback
            {"user_id": 4, "event_type": "view", "event_time": "2019-11-03 23:59:59"},
            # 5: first lookback day, then three purchases in the window -> one row, label 1
            {"user_id": 5, "event_type": "cart", "event_time": "2019-11-04 00:00:00"},
            {"user_id": 5, "event_type": "purchase", "event_time": "2019-11-19 10:00:00"},
            {"user_id": 5, "event_type": "purchase", "event_time": "2019-11-19 11:00:00"},
            {"user_id": 5, "event_type": "purchase", "event_time": "2019-11-30 23:59:59"},
        ]
    )


def test_population_is_users_active_in_the_14_days_before_cutoff(spark, events):
    from pyspark.testing import assertDataFrameEqual

    from pte.features.labels import build_population

    expected = spark.createDataFrame([(1,), (2,), (5,)], "user_id bigint")
    assertDataFrameEqual(build_population(events, CUTOFF), expected)


def test_labels_one_row_per_population_user(spark, events):
    from pyspark.testing import assertDataFrameEqual

    from pte.features.labels import build_labels, build_population

    labels = build_labels(events, build_population(events, CUTOFF), CUTOFF, label_days=13)

    expected = spark.createDataFrame([(1, 1), (2, 0), (5, 1)], "user_id bigint, label int")
    assertDataFrameEqual(labels, expected)


def test_label_window_end_is_exclusive(spark, make_events):
    from pyspark.testing import assertDataFrameEqual

    from pte.features.labels import build_labels, build_population

    events = make_events(
        [
            {"user_id": 1, "event_type": "view", "event_time": "2019-10-10 00:00:00"},
            {"user_id": 1, "event_type": "purchase", "event_time": "2019-10-29 00:00:00"},
            {"user_id": 2, "event_type": "view", "event_time": "2019-10-10 00:00:00"},
            {"user_id": 2, "event_type": "purchase", "event_time": "2019-10-28 23:59:59"},
        ]
    )
    cutoff = date(2019, 10, 15)  # 14-day window: 10-15 .. 10-28, so 10-29 is outside
    labels = build_labels(events, build_population(events, cutoff), cutoff, label_days=14)

    expected = spark.createDataFrame([(1, 0), (2, 1)], "user_id bigint, label int")
    assertDataFrameEqual(labels, expected)


def test_gmv_sums_the_window_once_per_purchase_and_fills_zero(spark, make_events):
    from pyspark.testing import assertDataFrameEqual

    from pte.features.labels import build_gmv, build_labels, build_population

    seen = {"event_type": "view", "event_time": "2019-11-10 12:00:00"}
    events = make_events(
        [
            # 1: an exact duplicate (counted once) and another item at the same price
            # (counted): 9.99 + 9.99, not 3 x 9.99 and not one distinct price
            {**seen, "user_id": 1},
            *[
                {
                    "user_id": 1,
                    "event_type": "purchase",
                    "product_id": 10,
                    "price": 9.99,
                    "user_session": "s1",
                    "event_time": "2019-11-20 10:00:00",
                }
            ]
            * 2,
            {
                "user_id": 1,
                "event_type": "purchase",
                "product_id": 11,
                "price": 9.99,
                "user_session": "s1",
                "event_time": "2019-11-20 10:05:00",
            },
            # 2: in the population, no purchase in the window -> 0.0, not NULL; a view with a
            # price, a purchase the second before cutoff and one on the first day after the
            # window are all left out
            {**seen, "user_id": 2},
            {
                "user_id": 2,
                "event_type": "view",
                "price": 80.0,
                "event_time": "2019-11-25 12:00:00",
            },
            {
                "user_id": 2,
                "event_type": "purchase",
                "price": 5.0,
                "event_time": "2019-11-17 23:59:59",
            },
            {
                "user_id": 2,
                "event_type": "purchase",
                "price": 7.0,
                "event_time": "2019-12-01 00:00:00",
            },
            # 3: buys on the cutoff day at 00:00:00 and on the last day of the window
            {**seen, "user_id": 3},
            {
                "user_id": 3,
                "event_type": "purchase",
                "price": 50.0,
                "event_time": "2019-11-18 00:00:00",
            },
            {
                "user_id": 3,
                "event_type": "purchase",
                "price": 0.1,
                "event_time": "2019-11-30 23:59:59",
            },
            # 4: buys in the window but is not in the population -> no row
            {
                "user_id": 4,
                "event_type": "purchase",
                "price": 30.0,
                "event_time": "2019-11-20 10:00:00",
            },
        ]
    )
    population = build_population(events, CUTOFF)
    gmv = build_gmv(events, population, CUTOFF, label_days=13)

    expected = spark.createDataFrame(
        [(1, 19.98), (2, 0.0), (3, 50.1)], "user_id bigint, gmv double"
    )
    assertDataFrameEqual(gmv, expected)
    # gmv > 0 exactly where label = 1
    labels = build_labels(events, population, CUTOFF, label_days=13)
    joined = labels.join(gmv, "user_id").collect()
    assert all((r["gmv"] > 0) == (r["label"] == 1) for r in joined)
