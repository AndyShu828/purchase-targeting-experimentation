"""Time Spark feature groups under a few shuffle and join settings (P2's Spark optimisation).

Plan v2 (P2): record one Spark optimisation, shuffle partitions or a broadcast join, with
the time before and after. Run under WSL from the repository root:
    PTE_ENV=wsl python scripts/bench_spark_tuning.py [--split test] [--runs 3]
        [--groups preference session activity] [--variants baseline aqe_off_200 ...]

Choices made here, for review in docs/spark_vs_duckdb.md:
- every variant is a set of runtime SQL settings, switched inside one session, so all of
  them share one JVM start and one warm page cache
- runs go round-robin, one of each variant per round, after one discarded warm-up round,
  so drift over the half hour (cache, JIT, the machine) lands on every variant alike
- each run writes the group to Parquet, as bench_features.py does; the median is reported
- per run it also reads the Spark UI's REST API: shuffle bytes written, the tasks of the
  stages that read a shuffle (AQE coalescing shows up here) and the join strategies of
  the final plan, since a broadcast variant only means something if the join changed
- each variant's output must match the baseline's row for row, or the run is flagged
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import tempfile
import time
import urllib.request
from collections.abc import Sequence
from pathlib import Path

from pte.config import load_config, resolve
from pte.features.activity import build_activity_features
from pte.features.build import CUTOFFS, WINDOW_DAYS
from pte.features.intent import build_intent_features
from pte.features.labels import build_population
from pte.features.preference import build_preference_features
from pte.features.purchase import build_purchase_features
from pte.features.session import build_session_features
from pte.features.trend import build_trend_features
from pte.spark import build_session, read_events

GROUPS = {
    "activity": build_activity_features,
    "purchase": build_purchase_features,
    "intent": build_intent_features,
    "preference": build_preference_features,
    "trend": build_trend_features,
    "session": build_session_features,
}

# 2 x 2 for shuffle partitions (AQE on / off x 200 / 16 = the cores here), plus a broadcast
# variant: population LEFT JOIN features can only broadcast its right side, the per-user
# aggregate (~2M rows), which the 10 MB default keeps on a sort-merge join.
BROADCAST_BYTES = str(512 * 1024 * 1024)
VARIANTS = {
    "baseline": {"spark.sql.adaptive.enabled": "true", "spark.sql.shuffle.partitions": "200"},
    "aqe_on_16": {"spark.sql.adaptive.enabled": "true", "spark.sql.shuffle.partitions": "16"},
    "aqe_off_200": {"spark.sql.adaptive.enabled": "false", "spark.sql.shuffle.partitions": "200"},
    "aqe_off_16": {"spark.sql.adaptive.enabled": "false", "spark.sql.shuffle.partitions": "16"},
    "broadcast": {
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.shuffle.partitions": "200",
        "spark.sql.autoBroadcastJoinThreshold": BROADCAST_BYTES,
        "spark.sql.adaptive.autoBroadcastJoinThreshold": BROADCAST_BYTES,
    },
}
DEFAULTS = {"spark.sql.autoBroadcastJoinThreshold": "10485760"}


def rest(spark, path: str):
    url = (
        f"{spark.sparkContext.uiWebUrl}/api/v1/applications/"
        f"{spark.sparkContext.applicationId}{path}"
    )
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def apply(spark, settings: dict[str, str]) -> None:
    spark.conf.unset("spark.sql.adaptive.autoBroadcastJoinThreshold")
    for key, value in {**DEFAULTS, **settings}.items():
        spark.conf.set(key, value)


def run_once(spark, df, out: Path) -> dict:
    stages_before = {s["stageId"] for s in rest(spark, "/stages")}
    start = time.perf_counter()
    df.write.mode("overwrite").parquet(out.as_uri())
    seconds = time.perf_counter() - start
    plan = rest(spark, "/sql?details=true&planDescription=true")[-1]["planDescription"]
    stages = [s for s in rest(spark, "/stages") if s["stageId"] not in stages_before]
    return {
        "seconds": seconds,
        "shuffle_gb": sum(s["shuffleWriteBytes"] for s in stages) / 1e9,
        "read_tasks": sorted(s["numTasks"] for s in stages if s["shuffleReadBytes"] > 0),
        # The description lists each node twice (tree, then details), and under AQE the
        # tree holds a final and an initial plan: count the final tree only.
        "joins": re.findall(
            r"(BroadcastHashJoin|SortMergeJoin|ShuffledHashJoin)",
            plan.split("\n\n")[0].split("== Initial Plan ==")[0],
        ),
    }


def differs(spark, left: Path, right: Path) -> int:
    """Users missing on one side, or with a column that differs (floats within 1e-9).

    A tolerance, not exceptAll: a new partition count changes the order of a sum, and with
    it the last bits of every average.
    """
    from pyspark.sql import functions as F

    a, b = spark.read.parquet(left.as_uri()), spark.read.parquet(right.as_uri())
    columns = [c for c in a.columns if c != "user_id"]
    joined = a.alias("a").join(b.alias("b"), on="user_id", how="full_outer")
    differ = F.col("a.user_id").isNull() | F.col("b.user_id").isNull()
    for c in columns:
        x, y = F.col(f"a.{c}"), F.col(f"b.{c}")
        differ = differ | (x.isNull() != y.isNull()) | (F.abs(x - y) > 1e-9)
    return joined.where(differ).count()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", choices=sorted(CUTOFFS), default="test")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--groups", nargs="+", choices=sorted(GROUPS), default=["preference"])
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--env", default=None, help="config name under configs/")
    args = parser.parse_args(argv)

    cfg = load_config(args.env)
    cutoff, _ = CUTOFFS[args.split]
    temp_root = resolve(cfg["duckdb"]["temp_directory"])
    temp_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="pte-tuning-", dir=temp_root))

    spark = build_session("pte-bench-spark-tuning")
    spark.sparkContext.setLogLevel("WARN")
    try:
        events = read_events(spark, full=True, env=args.env)
        population = build_population(events, cutoff)
        print(f"split={args.split} cutoff={cutoff} runs={args.runs} + 1 warm-up round")
        for name in args.groups:
            build = GROUPS[name]
            results: dict[str, list[dict]] = {v: [] for v in args.variants}
            for round_index in range(args.runs + 1):
                for variant in args.variants:
                    apply(spark, VARIANTS[variant])
                    df = build(events, population, cutoff, WINDOW_DAYS)
                    result = run_once(spark, df, work / f"{name}_{variant}")
                    if round_index > 0:
                        results[variant].append(result)
            print(f"\n### {name}")
            print(
                "| variant | median (s) | runs (s) | shuffle write (GB) | "
                "shuffle-read tasks | joins in final plan | rows differing |"
            )
            print("| --- | --: | --- | --: | --- | --- | --: |")
            for variant in args.variants:
                runs = results[variant]
                apply(spark, VARIANTS["baseline"])
                diff = differs(spark, work / f"{name}_baseline", work / f"{name}_{variant}")
                joins = sorted(set(runs[-1]["joins"]))
                counts = {j: runs[-1]["joins"].count(j) for j in joins}
                print(
                    f"| {variant} | {statistics.median(r['seconds'] for r in runs):.1f} "
                    f"| {[round(r['seconds'], 1) for r in runs]} "
                    f"| {statistics.median(r['shuffle_gb'] for r in runs):.2f} "
                    f"| {runs[-1]['read_tasks']} | {counts} | {diff:,} |"
                )
    finally:
        spark.stop()
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
