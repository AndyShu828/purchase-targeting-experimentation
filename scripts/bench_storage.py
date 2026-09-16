"""Benchmark: the same daily-purchase query on CSV vs Parquet (October 2019).

Run from the repository root:  python scripts/bench_storage.py
"""

import time

import duckdb

from pte.config import load_config, resolve
from pte.ingest.convert import parquet_source

RUNS = 3

cfg = load_config()
csv_path = (resolve(cfg["paths"]["raw_dir"]) / "2019-Oct.csv").as_posix()
events_dir = resolve(cfg["paths"]["events_dir"])

QUERIES = {
    "csv": f"""
        SELECT CAST(CAST(replace(event_time, ' UTC', '') AS TIMESTAMP) AS DATE) AS d, count(*)
        FROM read_csv('{csv_path}', header = true, types = {{'event_time': 'VARCHAR'}})
        WHERE event_type = 'purchase'
        GROUP BY 1
    """,
    "parquet": f"""
        SELECT event_date AS d, count(*)
        FROM {parquet_source(events_dir)}
        WHERE event_type = 'purchase' AND event_date < DATE '2019-11-01'
        GROUP BY 1
    """,
}

con = duckdb.connect()
results = {}
for name, sql in QUERIES.items():
    times = []
    for _ in range(RUNS):
        start = time.perf_counter()
        rows = con.sql(sql).fetchall()
        times.append(time.perf_counter() - start)
    results[name] = sorted(rows)
    print(f"{name:8s} best={min(times):.3f}s  runs={[round(t, 3) for t in times]}")

if results["csv"] != results["parquet"]:
    raise SystemExit("MISMATCH: CSV and Parquet give different results")
print(f"results match ({len(results['csv'])} days)")