from datetime import date, datetime

import pytest

from pte.config import load_config, resolve


def test_make_events_fills_nulls_and_event_date(spark, make_events):
    from pyspark.testing import assertDataFrameEqual

    from pte.spark import EVENTS_SCHEMA

    df = make_events(
        [{"user_id": 1, "event_type": "purchase", "event_time": "2019-11-17 23:59:59"}]
    )

    expected = spark.createDataFrame(
        [
            (
                "purchase",
                None,
                None,
                None,
                None,
                None,
                1,
                None,
                datetime(2019, 11, 17, 23, 59, 59),
                date(2019, 11, 17),
            )
        ],
        EVENTS_SCHEMA,
    )
    assertDataFrameEqual(df, expected)


def test_make_events_rejects_unknown_columns(make_events):
    with pytest.raises(KeyError, match="not an events column"):
        make_events([{"user_id": 1, "session_id": "x"}])


def test_events_schema_matches_parquet(spark):
    """Hand-written test rows are only valid if EVENTS_SCHEMA is what Spark reads."""
    from pyspark.sql.types import StructType

    from pte.spark import EVENTS_SCHEMA, read_events

    if not resolve(load_config()["paths"]["sample_dir"]).exists():
        pytest.skip("no local sample data (CI)")
    actual = read_events(spark).schema
    assert actual.simpleString() == StructType.fromDDL(EVENTS_SCHEMA).simpleString()
