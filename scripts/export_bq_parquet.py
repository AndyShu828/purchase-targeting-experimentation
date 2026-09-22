"""Write a BigQuery-bound copy of one month of events, with price as DECIMAL.

BigQuery derives a Parquet load's schema from the file, so a column's type has
to be settled here rather than in the DDL: `price` is DOUBLE locally and has to
arrive as NUMERIC, which means writing DECIMAL(10,2) into the export. The local
tree under data/parquet stays DOUBLE and untouched -- the two sides differ by
this one column on purpose, and docs/data_quality.md records why.

Usage:
    python scripts/export_bq_parquet.py --month 2019-10
    python scripts/export_bq_parquet.py --month 2019-10 --out data/bq_export --overwrite
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

import duckdb

from pte.config import load_config, resolve
from pte.ingest.convert import parquet_source

# 2,574.07 is the largest price in the data and no row carries more than two
# decimals, so 10 digits with 2 after the point is far wider than needed and
# still exact. DECIMAL maps to BigQuery NUMERIC on load.
PRICE_TYPE = "DECIMAL(10, 2)"

# day_index is the partition key on BigQuery: days since DAY_ZERO, 0..60 for
# the two months. A DATE partition is impossible in a sandbox project, where
# partition expiry maxes out at 60 days and is measured from the partition's
# own date -- 2019 partitions expire the moment they land. An integer range
# partition carries no such clock. The anchor is fixed here rather than derived
# from the month being exported, so October and November share one numbering.
DAY_ZERO = "2019-10-01"


def month_bounds(month: str) -> tuple[str, str]:
    """Return [first day, first day of next month) for a YYYY-MM string."""
    try:
        year, mon = (int(part) for part in month.split("-"))
    except ValueError:
        raise SystemExit(f"--month wants YYYY-MM, got {month!r}") from None
    next_year, next_mon = (year + 1, 1) if mon == 12 else (year, mon + 1)
    return f"{year:04d}-{mon:02d}-01", f"{next_year:04d}-{next_mon:02d}-01"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--month", default="2019-10", help="month to export (default: 2019-10)")
    parser.add_argument(
        "--out", default="data/bq_export", help="output root (default: data/bq_export)"
    )
    parser.add_argument("--overwrite", action="store_true", help="replace an existing export")
    parser.add_argument("--env", default=None, help="config name under configs/ (default: dev)")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    start_day, end_day = month_bounds(args.month)

    cfg = load_config(args.env)
    source_dir = resolve(cfg["paths"]["events_dir"])
    if not source_dir.is_dir():
        raise SystemExit(f"no parquet at {source_dir} -- run `python -m pte.ingest.convert` first")
    out_dir = resolve(args.out) / f"events_{args.month.replace('-', '')}"
    if out_dir.exists() and not args.overwrite:
        raise SystemExit(f"{out_dir} exists -- pass --overwrite to replace it")
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    temp_dir = cfg.get("duckdb", {}).get("temp_directory")
    if temp_dir:
        resolve(temp_dir).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = '{resolve(temp_dir).as_posix()}'")
    con.execute("SET preserve_insertion_order = false")

    # One flat file per day, NOT a PARTITION_BY tree. DuckDB's partitioning
    # writes the key into the directory name and drops it from the file, and it
    # reads that back by inferring hive partitions from the path. BigQuery does
    # no such inference on a local load: the column would simply be absent and
    # PARTITION BY event_date would have nothing to partition on. `bq load`
    # takes one local file per call anyway, so a flat day-named file is also
    # the shape the load loop wants.
    days = [
        day
        for (day,) in con.execute(
            f"""
            SELECT DISTINCT event_date
            FROM {parquet_source(source_dir)}
            WHERE event_date >= DATE '{start_day}' AND event_date < DATE '{end_day}'
            ORDER BY event_date
            """
        ).fetchall()
    ]
    if not days:
        raise SystemExit(f"no rows in {args.month} under {source_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    for day in days:
        con.execute(f"""
            COPY (
                -- event_date arrives from the hive path and is written into
                -- the file here, because this COPY does not partition. Adding
                -- it again as a literal would not raise: DuckDB renames the
                -- second one event_date_1 and BigQuery would receive both.
                SELECT * EXCLUDE (price),
                       CAST(price AS {PRICE_TYPE}) AS price,
                       CAST(
                           DATE_DIFF('day', DATE '{DAY_ZERO}', event_date) AS BIGINT
                       ) AS day_index
                FROM {parquet_source(source_dir)}
                WHERE event_date = DATE '{day}'
            ) TO '{(out_dir / f"events_{day}.parquet").as_posix()}'
            (FORMAT parquet, COMPRESSION zstd)
        """)
    elapsed = time.perf_counter() - start

    files = sorted(out_dir.glob("events_*.parquet"))
    (rows,) = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{(out_dir / '*.parquet').as_posix()}')"
    ).fetchone()
    written = sum(f.stat().st_size for f in files)
    print(f"{out_dir} <- {source_dir}")
    print(f"{rows:,} rows, {len(files)} files, {written:,} B in {elapsed:.1f}s")
    print(f"largest file {max(f.stat().st_size for f in files):,} B")


if __name__ == "__main__":
    try:
        main()
    except duckdb.Error as exc:
        raise SystemExit(f"{type(exc).__name__}: {exc}") from exc
