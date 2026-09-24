"""Spark session and event loading shared by feature code, scripts and tests.

Spark runs under WSL / Linux only: on Windows its Hadoop layer misses native libraries.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping

from pyspark.sql import DataFrame, SparkSession

from pte.config import load_config, resolve

# Schema of the date-partitioned Parquet dataset as Spark reads it.
# event_time is timestamp_ntz: the raw data is UTC with the zone dropped at ingest,
# so it is compared against naive datetimes, never shifted by the session time zone.
EVENTS_SCHEMA = (
    "event_type string, product_id bigint, category_id bigint, category_code string, "
    "brand string, price double, user_id bigint, user_session string, "
    "event_time timestamp_ntz, event_date date"
)


def build_session(
    app_name: str = "pte",
    master: str = "local[*]",
    conf: Mapping[str, str] | None = None,
) -> SparkSession:
    """Local session whose Python workers use the active virtualenv interpreter."""
    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .config("spark.sql.session.timeZone", "UTC")
        # In local mode the driver JVM runs every task. Its 1g default ran out of heap on
        # the session windows over 14 days of events (2026-09-25); WSL here has 15 GB.
        # Only takes effect if this call starts the JVM, i.e. before any other session.
        .config("spark.driver.memory", "8g")
    )
    for key, value in (conf or {}).items():
        builder = builder.config(key, value)
    return builder.getOrCreate()


def read_events(spark: SparkSession, *, full: bool = False, env: str | None = None) -> DataFrame:
    """Read the event dataset named in configs/<env>.yaml: the dev sample, or all events."""
    cfg = load_config(env)
    path = resolve(cfg["paths"]["events_dir" if full else "sample_dir"])
    # Spark needs a file:// URI here; a bare Windows drive path is read as a bad scheme.
    return spark.read.parquet(path.as_uri())
