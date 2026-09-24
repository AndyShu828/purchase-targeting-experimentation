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
