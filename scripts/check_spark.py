"""Check that the date-partitioned Parquet dataset reads correctly in PySpark.

Run from the repository root:  python scripts/check_spark.py [--full]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from pyspark.sql import SparkSession

from pte.config import load_config, resolve


def build_session() -> SparkSession:
    """Local session whose Python workers use the active virtualenv interpreter."""
    return (
        SparkSession.builder.appName("pte-check-spark")
        .master("local[*]")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--full", action="store_true", help="read all events, not the dev sample")
    parser.add_argument("--env", default=None, help="config name under configs/ (default: dev)")
    args = parser.parse_args(argv)

    cfg = load_config(args.env)
    path = resolve(cfg["paths"]["events_dir" if args.full else "sample_dir"])
    # Spark needs a file:// URI here; a bare Windows drive path is read as a bad scheme.
    source = path.as_uri()

    spark = build_session()
    spark.sparkContext.setLogLevel("WARN")
    try:
        df = spark.read.parquet(source)
        print(f"source={source}")
        df.printSchema()
        by_date = df.groupBy("event_date").count().orderBy("event_date")
        by_date.show(100, truncate=False)
        print(f"total_rows={df.count():,}")

        dtype = dict(df.dtypes)["event_date"]
        print(f"event_date dtype={dtype}")
        if dtype != "date":
            raise SystemExit(f"FAIL: expected event_date to be date, got {dtype}")
        print("OK")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
