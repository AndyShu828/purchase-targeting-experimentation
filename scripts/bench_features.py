"""Time each feature group in DuckDB and in Spark, and check the two agree row by row.

Plan v2 (P2): compute the same features in DuckDB, compare the time, confirm the results
match. The DuckDB side is Andy's SQL in sql/features/; the Spark side is
src/pte/features/. Run under WSL from the repository root, where both engines read the
same Parquet files:
    PTE_ENV=wsl python scripts/bench_features.py [--split test] [--runs 3]

Choices made here, for review in docs/spark_vs_duckdb.md:
- each run computes one group end to end and writes it to Parquet, so both engines do
  the same work, including the population and the write; Spark's JVM start is excluded
- runs repeat back to back, so every run after the first reads from a warm page cache;
  all runs are printed and the median is reported
- both engines use every core: DuckDB by default, Spark via local[*]
- a result matches if every user is on both sides and every column is equal, NULL to
  NULL, floats to within 1e-9
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import tempfile
import time
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

import duckdb

from pte.config import load_config, resolve
from pte.features.activity import build_activity_features
from pte.features.build import CUTOFFS, WINDOW_DAYS
from pte.features.intent import build_intent_features
from pte.features.labels import build_population
from pte.features.outage import effective_days
from pte.features.preference import build_preference_features
from pte.features.purchase import build_purchase_features
from pte.features.trend import build_trend_features
from pte.spark import build_session, read_events

SQL_DIR = Path(__file__).resolve().parents[1] / "sql" / "features"
GROUPS = {
    "activity": build_activity_features,
    "purchase": build_purchase_features,
    "intent": build_intent_features,
    "preference": build_preference_features,
    "trend": build_trend_features,
}


def time_duckdb(con: duckdb.DuckDBPyConnection, sql: str, out: Path) -> float:
    start = time.perf_counter()
    con.execute(f"COPY ({sql}) TO '{out}' (FORMAT parquet)")
    return time.perf_counter() - start


def time_spark(df, out: Path) -> float:
    start = time.perf_counter()
    df.write.mode("overwrite").parquet(out.as_uri())
    return time.perf_counter() - start


def mismatches(con: duckdb.DuckDBPyConnection, duck: Path, spark: Path) -> int:
    """Users missing on one side, or with any column that differs."""
    columns = [c for c in con.sql(f"SELECT * FROM '{duck}' LIMIT 0").columns if c != "user_id"]
    differs = " OR ".join(
        f"(d.{c} IS NULL) <> (s.{c} IS NULL) OR abs(d.{c} - s.{c}) > 1e-9" for c in columns
    )
    return con.sql(
        f"""SELECT count(*) FROM '{duck}' AS d
            FULL JOIN read_parquet('{spark}/*.parquet') AS s ON d.user_id = s.user_id
            WHERE d.user_id IS NULL OR s.user_id IS NULL OR {differs}"""
    ).fetchone()[0]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", choices=sorted(CUTOFFS), default="test")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--env", default=None, help="config name under configs/")
    args = parser.parse_args(argv)

    cfg = load_config(args.env)
    events_dir = resolve(cfg["paths"]["events_dir"])
    cutoff, _ = CUTOFFS[args.split]
    temp_root = resolve(cfg["duckdb"]["temp_directory"])
    temp_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="pte-bench-", dir=temp_root))

    con = duckdb.connect()
    con.execute(
        "CREATE VIEW events AS SELECT * FROM "
        f"read_parquet('{events_dir}/**/*.parquet', hive_partitioning = true)"
    )
    con.execute(f"SET VARIABLE cutoff = DATE '{cutoff}'")
    con.execute(f"SET VARIABLE window_start = DATE '{cutoff - timedelta(days=WINDOW_DAYS)}'")
    con.execute(f"SET VARIABLE effective_days = {effective_days(cutoff, WINDOW_DAYS)!r}")

    spark = build_session("pte-bench-features")
    spark.sparkContext.setLogLevel("WARN")
    try:
        events = read_events(spark, full=True, env=args.env)
        population = build_population(events, cutoff)
        print(f"split={args.split} cutoff={cutoff} window_days={WINDOW_DAYS} runs={args.runs}")
        print("| group | DuckDB median (s) | Spark median (s) | Spark / DuckDB | rows differing |")
        print("| --- | --: | --: | --: | --: |")
        for name, build in GROUPS.items():
            sql = (SQL_DIR / f"{name}.sql").read_text(encoding="utf-8")
            duck_out, spark_out = work / f"{name}_duckdb.parquet", work / f"{name}_spark"
            duck_runs = [time_duckdb(con, sql, duck_out) for _ in range(args.runs)]
            spark_runs = [
                time_spark(build(events, population, cutoff, WINDOW_DAYS), spark_out)
                for _ in range(args.runs)
            ]
            duck_s, spark_s = statistics.median(duck_runs), statistics.median(spark_runs)
            print(
                f"| {name} | {duck_s:.1f} | {spark_s:.1f} | {spark_s / duck_s:.1f}x "
                f"| {mismatches(con, duck_out, spark_out):,} |"
            )
            print(
                f"<!-- {name}: duckdb runs {[round(r, 1) for r in duck_runs]}, "
                f"spark runs {[round(r, 1) for r in spark_runs]} -->"
            )
    finally:
        spark.stop()
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
