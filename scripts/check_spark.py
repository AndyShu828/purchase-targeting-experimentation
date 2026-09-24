"""Check that the date-partitioned Parquet dataset reads correctly in PySpark.

Run from the repository root:  python scripts/check_spark.py [--full]
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from pte.spark import build_session, read_events


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--full", action="store_true", help="read all events, not the dev sample")
    parser.add_argument("--env", default=None, help="config name under configs/ (default: dev)")
    args = parser.parse_args(argv)

    spark = build_session("pte-check-spark")
    spark.sparkContext.setLogLevel("WARN")
    try:
        df = read_events(spark, full=args.full, env=args.env)
        print(f"source={df.inputFiles()[0].rsplit('/event_date=', 1)[0]}")
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
