"""Write the test set's GMV in the label window, the coupon experiment's untreated spend.

Run under WSL from the repository root, after scripts/build_features.py:
    PTE_ENV=wsl python scripts/build_gmv.py

Writes <features_dir>/test/gmv as Parquet: user_id and gmv, one row per test user, from
build_gmv over 11-18 to 11-30 (experiment_design.md section 5). Checks that gmv > 0 exactly
where the label is 1, as both read the same purchases in the same window.

Run on 2026-10-05: 2,305,679 users, each with one label and one gmv; 102,290 buyers (gmv > 0),
and gmv > 0 disagrees with label = 1 for none of them. Total 61,231,737.69 (price has no
currency unit, data_quality.md), 26.5569 per user. The window holds 212,347 purchase rows by
test users, 212,346 after dropping exact duplicates. A DuckDB query of the same steps (filter,
DISTINCT over the nine columns, sum per user) gives the same 102,290 buyers and total, and
gmv differs from this table for no user; dropping duplicates before the filter gives the same.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import timedelta

from pyspark.sql import functions as F

from pte.config import load_config, resolve
from pte.features.build import CUTOFFS
from pte.features.labels import DUPLICATE_KEY, build_gmv, build_population
from pte.spark import build_session, read_events


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", help="configs/<env>.yaml (default $PTE_ENV, then dev)")
    args = parser.parse_args(argv)

    out_dir = resolve(load_config(args.env)["paths"]["features_dir"]) / "test"
    cutoff, label_days = CUTOFFS["test"]
    spark = build_session("pte-build-gmv")
    spark.sparkContext.setLogLevel("WARN")
    try:
        events = read_events(spark, full=True, env=args.env)
        population = build_population(events, cutoff)
        build_gmv(events, population, cutoff, label_days).write.mode("overwrite").parquet(
            (out_dir / "gmv").as_uri()
        )

        gmv = spark.read.parquet((out_dir / "gmv").as_uri())
        labels = spark.read.parquet((out_dir / "labels").as_uri())
        joined = labels.join(gmv, on="user_id", how="full")
        n_users, n_labels, n_gmv, buyers, total, mismatched = joined.agg(
            F.count("*"),
            F.count("label"),
            F.count("gmv"),
            F.sum(F.when(F.col("gmv") > 0, 1).otherwise(0)),
            F.sum("gmv"),
            F.sum(F.when((F.col("gmv") > 0) != (F.col("label") == 1), 1).otherwise(0)),
        ).first()

        end = cutoff + timedelta(days=label_days)
        window = events.where(
            (F.col("event_type") == "purchase")
            & (F.col("event_date") >= F.lit(cutoff))
            & (F.col("event_date") < F.lit(end))
        ).join(population, on="user_id", how="inner")
        rows = window.count()
        distinct_rows = window.dropDuplicates(list(DUPLICATE_KEY)).count()
    finally:
        spark.stop()

    print(f"test cutoff={cutoff} label_days={label_days}")
    print(f"users={n_users:,} with_label={n_labels:,} with_gmv={n_gmv:,}")
    print(f"buyers (gmv > 0)={buyers:,} total_gmv={total:,.2f} gmv_per_user={total / n_users:.4f}")
    print(f"gmv > 0 disagreeing with label = 1: {mismatched}")
    print(f"purchase rows in the window={rows:,} after dropping exact duplicates={distinct_rows:,}")


if __name__ == "__main__":
    main()
