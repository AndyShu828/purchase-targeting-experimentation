"""Convert REES46 monthly CSVs to date-partitioned Parquet and extract the dev sample.

Usage:
    python -m pte.ingest.convert convert [--overwrite]
    python -m pte.ingest.convert sample [--overwrite]
    python -m pte.ingest.convert verify
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import duckdb

from pte.config import load_config, resolve

# event_time carries a " UTC" suffix, so it is read as text and parsed explicitly.
# 19-digit category_id must stay BIGINT: a float would silently lose precision.
CSV_TYPES_SQL = (
    "{'event_time': 'VARCHAR', 'product_id': 'BIGINT', 'category_id': 'BIGINT', "
    "'user_id': 'BIGINT', 'price': 'DOUBLE'}"
)


def _sql_list(paths: Sequence[Path]) -> str:
    return "[" + ", ".join(f"'{p.as_posix()}'" for p in paths) + "]"


def _write_options(overwrite: bool) -> str:
    options = "FORMAT parquet, PARTITION_BY (event_date), COMPRESSION zstd"
    return options + (", OVERWRITE" if overwrite else "")


def parquet_source(directory: Path) -> str:
    """SQL table expression reading a hive-partitioned Parquet directory."""
    pattern = (directory / "**" / "*.parquet").as_posix()
    return f"read_parquet('{pattern}', hive_partitioning = true)"


def convert(
    csv_paths: Sequence[Path],
    out_dir: Path,
    overwrite: bool = False,
    con: duckdb.DuckDBPyConnection | None = None,
) -> None:
    """Write events partitioned by UTC event_date."""
    con = con or duckdb.connect()
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"""
        COPY (
            SELECT * EXCLUDE (event_time, ts),
                   ts AS event_time,
                   CAST(ts AS DATE) AS event_date
            FROM (
                SELECT *, CAST(replace(event_time, ' UTC', '') AS TIMESTAMP) AS ts
                FROM read_csv({_sql_list(csv_paths)}, header = true, types = {CSV_TYPES_SQL})
            )
        ) TO '{out_dir.as_posix()}' ({_write_options(overwrite)})
    """)


def extract_sample(
    events_dir: Path,
    out_dir: Path,
    start: str,
    end: str,
    overwrite: bool = False,
    con: duckdb.DuckDBPyConnection | None = None,
) -> None:
    """Copy the partitions between start and end (inclusive) into out_dir."""
    con = con or duckdb.connect()
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"""
        COPY (
            SELECT * FROM {parquet_source(events_dir)}
            WHERE event_date BETWEEN DATE '{start}' AND DATE '{end}'
        ) TO '{out_dir.as_posix()}' ({_write_options(overwrite)})
    """)


def count_rows(
    csv_paths: Sequence[Path],
    events_dir: Path,
    con: duckdb.DuckDBPyConnection | None = None,
) -> tuple[int, int]:
    """Return (CSV data rows, Parquet rows) for reconciliation."""
    con = con or duckdb.connect()
    csv_sql = (
        f"SELECT count(*) FROM read_csv({_sql_list(csv_paths)}, header = true, all_varchar = true)"
    )
    csv_rows = con.sql(csv_sql).fetchone()[0]
    parquet_rows = con.sql(f"SELECT count(*) FROM {parquet_source(events_dir)}").fetchone()[0]
    return csv_rows, parquet_rows


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("step", choices=["convert", "sample", "verify"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--env", default=None, help="config name under configs/ (default: dev)")
    args = parser.parse_args(argv)

    cfg = load_config(args.env)
    paths, ingest = cfg["paths"], cfg["ingest"]
    raw_dir = resolve(paths["raw_dir"])
    events_dir = resolve(paths["events_dir"])
    csv_paths = [raw_dir / f"{month}.csv" for month in ingest["months"]]

    con = duckdb.connect()
    temp_dir = cfg.get("duckdb", {}).get("temp_directory")
    if temp_dir:
        resolve(temp_dir).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = '{resolve(temp_dir).as_posix()}'")

    if args.step == "convert":
        convert(csv_paths, events_dir, args.overwrite, con)
    elif args.step == "sample":
        extract_sample(
            events_dir,
            resolve(paths["sample_dir"]),
            ingest["sample_start"],
            ingest["sample_end"],
            args.overwrite,
            con,
        )
    else:
        csv_rows, parquet_rows = count_rows(csv_paths, events_dir, con)
        status = "OK" if csv_rows == parquet_rows else "MISMATCH"
        print(f"csv={csv_rows:,} parquet={parquet_rows:,} {status}")
        if status != "OK":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
