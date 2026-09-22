"""Run a .sql file against the local DuckDB dev sample and print rows plus timing.

The file is executed as-is, so it stays valid BigQuery SQL: the source table is
exposed as a view named `events`, which matches `FROM events` under a BigQuery
default dataset. Dialect gaps are not patched over -- SAFE_DIVIDE and friends
will fail here, which is the point of testing locally first.

The one exception is BIGQUERY_SHIMS: time differences and week truncation have
no spelling both engines accept (docs/sql_dialects.md), so the BigQuery form is
defined as a DuckDB macro. A macro cannot read a bare date part like SECOND --
DuckDB only accepts it because the parameter is never used -- so each shim
hard-codes the one part it supports. tests/test_sql_dialect.py fails CI if a
file uses any other part.

Usage:
    python scripts/run_sql.py sql/analysis/data_quality.sql
    python scripts/run_sql.py sql/analysis/funnel.sql --full --limit 20
    python scripts/run_sql.py -e "SELECT event_type, count(*) AS n FROM events GROUP BY 1"
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb

from pte.config import load_config, resolve
from pte.ingest.convert import parquet_source

# BigQuery function -> (the only date part it supports here, DuckDB macro).
# Never name a shim after a DuckDB built-in: CREATE MACRO date_diff silently
# replaces the built-in for the whole connection (checked on 1.5.5).
BIGQUERY_SHIMS = {
    "TIMESTAMP_DIFF": (
        "SECOND",
        "CREATE MACRO timestamp_diff(end_ts, start_ts, part) AS "
        "date_diff('second', start_ts, end_ts)",
    ),
    "TIMESTAMP_TRUNC": (
        "ISOWEEK",
        "CREATE MACRO timestamp_trunc(ts, part) AS date_trunc('week', ts)",
    ),
}


def format_table(columns: Sequence[str], rows: Sequence[tuple[Any, ...]]) -> str:
    """Render rows as a fixed-width text table."""
    cells = [list(columns)] + [["NULL" if v is None else str(v) for v in row] for row in rows]
    widths = [max(len(c[i]) for c in cells) for i in range(len(columns))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(cells[0], widths, strict=True)).rstrip()]
    lines.append("  ".join("-" * w for w in widths))
    lines += [
        "  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)).rstrip()
        for row in cells[1:]
    ]
    return "\n".join(lines)


def read_sql(args: argparse.Namespace) -> tuple[str, str]:
    """Return (label, statement) from either --execute or the .sql file argument."""
    if args.execute is not None:
        return "-e", args.execute
    try:
        return str(args.sql_file), args.sql_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"cannot read {args.sql_file}: {exc.strerror}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("sql_file", nargs="?", type=Path, help="path to a .sql file")
    source.add_argument("-e", "--execute", metavar="SQL", help="run this statement instead")
    parser.add_argument(
        "--full", action="store_true", help="read all 61 partitions instead of the dev sample"
    )
    parser.add_argument("--limit", type=int, default=50, help="rows to print (default: 50)")
    parser.add_argument("--env", default=None, help="config name under configs/ (default: dev)")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    label, text = read_sql(args)

    cfg = load_config(args.env)
    paths = cfg["paths"]
    source_dir = resolve(paths["events_dir"] if args.full else paths["sample_dir"])
    if not source_dir.is_dir():
        raise SystemExit(f"no parquet at {source_dir} -- run `python -m pte.ingest.convert` first")

    con = duckdb.connect()
    temp_dir = cfg.get("duckdb", {}).get("temp_directory")
    if temp_dir:
        resolve(temp_dir).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = '{resolve(temp_dir).as_posix()}'")
    for _part, macro in BIGQUERY_SHIMS.values():
        con.execute(macro)
    con.execute(f"CREATE VIEW events AS SELECT * FROM {parquet_source(source_dir)}")

    print(f"{label}  events -> {source_dir}")
    start = time.perf_counter()
    result = con.sql(text)
    rows = result.fetchall() if result is not None else []
    elapsed = time.perf_counter() - start

    if result is not None:
        print(format_table(result.columns, rows[: args.limit]))
        if len(rows) > args.limit:
            print(f"... {len(rows) - args.limit} more rows")
    print(f"\n{len(rows)} rows in {elapsed:.3f}s")


if __name__ == "__main__":
    try:
        main()
    except duckdb.Error as exc:
        sys.exit(f"{type(exc).__name__}: {exc}")
