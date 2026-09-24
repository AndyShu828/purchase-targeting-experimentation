"""Write the feature and label tables for the three cutoffs, and print their summary.

Run under WSL from the repository root:
    PTE_ENV=wsl python scripts/build_features.py [--split test]

Writes <features_dir>/<split>/features and <features_dir>/<split>/labels as Parquet.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

from pyspark.sql import functions as F

from pte.config import load_config, resolve
from pte.features.build import CUTOFFS, WINDOW_DAYS, build_feature_table, build_label_table
from pte.spark import build_session, read_events


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", choices=sorted(CUTOFFS), action="append")
    parser.add_argument("--env", default=None, help="config name under configs/")
    args = parser.parse_args(argv)

    out_dir = resolve(load_config(args.env)["paths"]["features_dir"])
    spark = build_session("pte-build-features")
    spark.sparkContext.setLogLevel("WARN")
    try:
        events = read_events(spark, full=True, env=args.env)
        for split in args.split or ["train", "valid", "test"]:
            cutoff, label_days = CUTOFFS[split]
            start = time.perf_counter()
            build_feature_table(events, cutoff).write.mode("overwrite").parquet(
                (out_dir / split / "features").as_uri()
            )
            build_label_table(events, cutoff, label_days).write.mode("overwrite").parquet(
                (out_dir / split / "labels").as_uri()
            )
            seconds = time.perf_counter() - start

            features = spark.read.parquet((out_dir / split / "features").as_uri())
            labels = spark.read.parquet((out_dir / split / "labels").as_uri())
            n_users, positives = labels.agg(F.count("*"), F.sum("label")).first()
            print(
                f"{split} cutoff={cutoff} window_days={WINDOW_DAYS} label_days={label_days} "
                f"users={n_users:,} positives={positives:,} "
                f"positive_rate={100 * positives / n_users:.2f}% seconds={seconds:.0f}"
            )
            nulls = features.select(
                [F.count(F.when(F.col(c).isNull(), 1)).alias(c) for c in features.columns]
            ).first()
            means = features.select([F.avg(c).alias(c) for c in features.columns]).first()
            for column in features.columns[1:]:
                print(f"  {column:32s} null={nulls[column]:>10,}  mean={means[column]:.6g}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
