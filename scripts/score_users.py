"""Score every user in a cutoff's population with the registered model: probability and decile.

Run under WSL from the repository root:
    PTE_ENV=wsl python scripts/score_users.py --cutoff 2019-12-01

Builds the feature table for the cutoff from all events with Spark (features see only
event_date < cutoff, labels are not built), scores it with the champion model (or
--version), and writes <scores_dir>/<cutoff>/scores.parquet next to the features/ it
scored. 2019-12-01 is the latest cutoff, the day after the data ends: the run a live
system would make that morning. --features scores a table already built instead, such as
<features_dir>/valid/features for cutoff 2019-11-01, and skips Spark.

Checked 2026-09-29 on version 1: cutoff 2019-11-01 rebuilt the valid feature table cell
for cell (56 s) and scored its 1,793,124 users exactly as the champion scores valid (max
gap 0.0, mean p 5.0003%). Cutoff 2019-12-01: 2,281,094 users in 79 s, mean p 5.4996%,
decile 1 from p = 0.1224 up. A cutoff whose window leaves the data, such as 2019-12-02,
is refused.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from pte.config import load_config, resolve
from pte.model.registry import CHAMPION, EXPERIMENT, MODEL_NAME
from pte.model.scoring import check_cutoff, decile_summary, score_users
from pte.model.tracking import setup_tracking


def build_features(cutoff: date, out: Path, env: str | None) -> None:
    """Write the cutoff's feature table to out with Spark, after checking the data covers it."""
    from pyspark.sql import functions as F

    from pte.features.build import WINDOW_DAYS, build_feature_table
    from pte.spark import build_session, read_events

    # 8 threads, not local[*]'s 16: in the daily_score replay (2026-10-13) cutoffs 11-24, 11-25,
    # 11-26 and 11-29 ran out of Spark memory twice each (UNABLE_TO_ACQUIRE_MEMORY), since the
    # 8g driver's execution pool is split across the running tasks; at 8 threads each task
    # gets twice as much, and 11-24 scored in 110 s.
    spark = build_session("pte-score-users", master="local[8]")
    spark.sparkContext.setLogLevel("WARN")
    try:
        events = read_events(spark, full=True, env=env)
        first_day, last_day = events.agg(F.min("event_date"), F.max("event_date")).first()
        check_cutoff(cutoff, first_day, last_day, WINDOW_DAYS)
        build_feature_table(events, cutoff).write.mode("overwrite").parquet(out.as_uri())
    finally:
        spark.stop()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cutoff", required=True, type=date.fromisoformat, help="YYYY-MM-DD")
    parser.add_argument("--version", help=f"model version (default: @{CHAMPION})")
    parser.add_argument("--features", type=Path, help="a feature table already built")
    parser.add_argument("--env", help="configs/<env>.yaml (default $PTE_ENV, then dev)")
    args = parser.parse_args(argv)
    paths = load_config(args.env)["paths"]
    out_dir = resolve(paths["scores_dir"]) / args.cutoff.isoformat()

    setup_tracking(resolve(paths["mlflow_dir"]), EXPERIMENT)
    # resolved to a number once, so the file names the version that scored it even if the
    # alias moves later
    version = args.version or str(
        MlflowClient().get_model_version_by_alias(MODEL_NAME, CHAMPION).version
    )
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{version}")

    features_path = args.features
    if features_path is None:
        features_path = out_dir / "features"
        start = time.perf_counter()
        build_features(args.cutoff, features_path, args.env)
        print(f"Built features for {args.cutoff} in {time.perf_counter() - start:.0f} s")
    features = pd.read_parquet(features_path)

    scores = score_users(model, features, args.cutoff, version)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(out_dir / "scores.parquet", index=False)
    print(
        f"Scored {len(scores):,} users as of {args.cutoff} with {MODEL_NAME} version {version}: "
        f"mean p {scores['p'].mean():.4%} -> {out_dir / 'scores.parquet'}"
    )
    print(decile_summary(scores).round(5).to_string())


if __name__ == "__main__":
    main()
