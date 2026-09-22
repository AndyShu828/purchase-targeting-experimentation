"""Does carrying a wide, high-cardinality column through a window pipeline cost?

Conclusion 12 tested this once with `category_code` and measured nothing: 82.5 s
against 84.3 s over four rounds each, intervals overlapping, one round reversed.
The suspicion was that 129 distinct values dictionary-encode to almost nothing,
so "keeps the sort narrow" was never really under test. This retries with
columns that cannot dictionary away: `product_id` (166k distinct integers) and
`user_session` (23M distinct 36-char strings, 42% of the Parquet by size per
docs/storage.md).

THE MEASUREMENT WAS WRONG THREE TIMES BEFORE IT WAS RIGHT, and the history is
the point:

  1. Each variant's rounds run together. Clean monotone trend by cardinality --
     and the narrow variant alone drifted 18.3 s to 41.4 s across its own four
     rounds. Variant order was time order, so the trend and the drift were the
     same column of numbers.
  2. Interleaved. Still ordered inside each round, and the per-position penalty
     (~4 s) was the size of the effect, so the last variant paid it every time.
  3. Interleaved and reversed on alternate rounds. The trend collapsed: means
     within 1.5 s across a cardinality range of 0 to 23M, while each variant
     walked from 12 s to 66 s.
  4. One fresh process per measurement. Repeating a single query in one process
     goes 11.9 -> 33.5 s over six runs with nothing changed; three separate
     processes give 11.5, 11.2, 11.8. The drift is state accumulating in the
     DuckDB connection, not the machine -- checked against thermal throttling
     (clock ratio stayed at 100-109% of nominal), spill (temp directory empty
     while running) and memory (19 GB free, CPU only 20-29% busy).

So the final design: a fresh process per measurement, and the order still
alternates, because a fresh process only removes the drift we know about.

The pipeline is the sessionisation shape used across sql/analysis: LAG over
(user_id, event_time), a 30-minute cut, a running SUM to number sessions, then
one aggregate. Only the carried columns differ between variants, and every
variant ends in the same COUNT so none can be optimised away differently.

Usage:
    python scripts/bench_sort_width.py                            # 4 rounds, full
    python scripts/bench_sort_width.py --rounds 6 --only narrow   # drift control
    python scripts/bench_sort_width.py --rounds 2 --sample
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import duckdb
from run_sql import BIGQUERY_SHIMS

from pte.config import load_config, resolve
from pte.ingest.convert import parquet_source

# Each variant carries a different set of extra columns through both window
# layers. The aggregate at the end is identical, so any difference in time is
# the cost of dragging those columns through the sort.
VARIANTS = {
    "narrow (nothing extra)": "",
    "carry category_code (129 values)": ", category_code",
    "carry product_id (166k values)": ", product_id",
    "carry user_session (23M values)": ", user_session",
}

TEMPLATE = """
WITH previous_event_times AS (
    SELECT
        user_id,
        event_time{extra},
        LAG(event_time, 1) OVER (
            PARTITION BY user_id ORDER BY event_time
        ) AS previous_event_time
    FROM events
),

session_cuts AS (
    SELECT
        user_id,
        event_time{extra},
        SUM(
            CASE
                WHEN TIMESTAMP_DIFF(event_time, previous_event_time, SECOND) > 1800 THEN 1
                ELSE 0
            END
        ) OVER (PARTITION BY user_id ORDER BY event_time) AS session_index
    FROM previous_event_times
)

SELECT COUNT(*) AS rows_seen, COUNT(DISTINCT session_index) AS session_indexes
FROM session_cuts
"""


def time_one(label: str, source_dir: Path, cfg: dict) -> float:
    """Time one variant once in this process and return the seconds."""
    con = duckdb.connect()
    temp_dir = cfg.get("duckdb", {}).get("temp_directory")
    if temp_dir:
        resolve(temp_dir).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = '{resolve(temp_dir).as_posix()}'")
    # The same shims run_sql.py installs, imported rather than copied, so the
    # SQL here is the BigQuery form used everywhere else in the repo.
    for _part, macro in BIGQUERY_SHIMS.values():
        con.execute(macro)
    con.execute(f"CREATE VIEW events AS SELECT * FROM {parquet_source(source_dir)}")

    sql = TEMPLATE.format(extra=VARIANTS[label])
    start = time.perf_counter()
    con.sql(sql).fetchall()
    elapsed = time.perf_counter() - start
    con.close()
    return elapsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rounds", type=int, default=4, help="timed rounds per variant")
    parser.add_argument("--sample", action="store_true", help="dev sample instead of full data")
    parser.add_argument("--env", default=None, help="config name under configs/")
    parser.add_argument(
        "--only", default=None, help="one variant by name prefix, as a drift control"
    )
    # Internal: time one variant once and print the seconds. This is what the
    # parent spawns, and it is the whole reason each measurement starts clean.
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    cfg = load_config(args.env)
    paths = cfg["paths"]
    source_dir = resolve(paths["sample_dir"] if args.sample else paths["events_dir"])
    if not source_dir.is_dir():
        raise SystemExit(f"no parquet at {source_dir}")

    if args.worker is not None:
        print(f"{time_one(args.worker, source_dir, cfg):.6f}")
        return

    variants = list(VARIANTS)
    if args.only:
        variants = [label for label in VARIANTS if label.startswith(args.only)]
        if not variants:
            raise SystemExit(f"--only {args.only!r} matches none of {list(VARIANTS)}")

    print(f"events -> {source_dir}, {args.rounds} rounds each\n")
    header = f"{'variant':34}" + "".join(f"{f'run{i + 1}':>9}" for i in range(args.rounds))
    print(header + f"{'min':>9}{'mean':>9}")
    print("-" * len(header + "         " * 2))

    times: dict[str, list[float]] = {label: [] for label in variants}
    for round_index in range(args.rounds):
        order = list(reversed(variants)) if round_index % 2 else variants
        for label in order:
            command = [sys.executable, __file__, "--worker", label]
            if args.sample:
                command.append("--sample")
            if args.env:
                command += ["--env", args.env]
            finished = subprocess.run(command, capture_output=True, text=True, check=True)
            times[label].append(float(finished.stdout.strip()))

    for label, runs in times.items():
        cells = "".join(f"{t:9.1f}" for t in runs)
        print(f"{label:34}{cells}{min(runs):9.1f}{statistics.mean(runs):9.1f}")

    print("\nOne fresh process per measurement, order alternating -- see the module docstring.")


if __name__ == "__main__":
    try:
        main()
    except duckdb.Error as exc:
        raise SystemExit(f"{type(exc).__name__}: {exc}") from exc
