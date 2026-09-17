"""Run a .sql file against the local DuckDB dev sample and print rows plus timing.

The file is executed as-is, so it stays valid BigQuery SQL: the source table is
exposed as a view named `events`, which matches `FROM events` under a BigQuery
default dataset. Dialect gaps are not patched over -- SAFE_DIVIDE and friends
will fail here, which is the point of testing locally first.

Usage:
    python scripts/run_sql.py sql/analysis/data_quality.sql
    python scripts/run_sql.py sql/analysis/funnel.sql --full --limit 20
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb

from pte.config import load_config, resolve
from pte.ingest.convert import parquet_source


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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sql_file", type=Path)
    parser.add_argument(
        "--full", action="store_true", help="read all 61 partitions instead of the dev sample"
    )
    parser.add_argument("--limit", type=int, default=50, help="rows to print (default: 50)")
    parser.add_argument("--env", default=None, help="config name under configs/ (default: dev)")
    args = parser.parse_args(argv)

    cfg = load_config(args.env)
    paths = cfg["paths"]
    source_dir = resolve(paths["events_dir"] if args.full else paths["sample_dir"])
    text = args.sql_file.read_text(encoding="utf-8")

    con = duckdb.connect()
    temp_dir = cfg.get("duckdb", {}).get("temp_directory")
    if temp_dir:
        resolve(temp_dir).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = '{resolve(temp_dir).as_posix()}'")
    con.execute(f"CREATE VIEW events AS SELECT * FROM {parquet_source(source_dir)}")

    print(f"{args.sql_file}  events -> {source_dir}")
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
    main()
