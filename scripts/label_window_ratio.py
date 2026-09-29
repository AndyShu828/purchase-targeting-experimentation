"""How much lower a 13-day label rate is than a 14-day one, measured on October cutoffs.

Run under WSL from the repository root:
    PTE_ENV=wsl python scripts/label_window_ratio.py

The model predicts purchase within 14 days; the test labels (cutoff 11-18) count 13, as the
data ends on 11-30 (docs/incident_2019_11_15.md). Even a perfect model's mean prediction
would sit above the test rate. r = 13-day rate / 14-day rate over the same users measures
that gap, so the test mean prediction can be read as mean p * r against the actual rate.

r is fixed here, before the test labels are read (leakage.md section 5), on the train cutoff
10-15; 10-22 and 10-29 show how much it moves. Every label window here ends by 11-12, before
the outage and the test period. Also prints the rate for every window length 1-14 days.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date, timedelta

from pyspark.sql import functions as F

from pte.features.build import CUTOFFS
from pte.features.labels import build_labels, build_population
from pte.spark import build_session, read_events

LABEL_DAYS = CUTOFFS["train"][1]  # 14
TEST_LABEL_DAYS = CUTOFFS["test"][1]  # 13
CUTOFF_DATES = [CUTOFFS["train"][0], date(2019, 10, 22), date(2019, 10, 29)]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", help="configs/<env>.yaml (default $PTE_ENV, then dev)")
    args = parser.parse_args(argv)

    spark = build_session("pte-label-window-ratio")
    spark.sparkContext.setLogLevel("WARN")
    try:
        events = read_events(spark, full=True, env=args.env)
        for cutoff in CUTOFF_DATES:
            population = build_population(events, cutoff).cache()
            n = population.count()
            rates = {}
            for days in (TEST_LABEL_DAYS, LABEL_DAYS):
                labels = build_labels(events, population, cutoff, days)
                rates[days] = labels.agg(F.avg("label")).first()[0]
            r = rates[TEST_LABEL_DAYS] / rates[LABEL_DAYS]
            print(
                f"cutoff {cutoff}: {n:,} users, 13-day rate {rates[TEST_LABEL_DAYS]:.4%}, "
                f"14-day rate {rates[LABEL_DAYS]:.4%}, r = {r:.4f} (13/14 = {13 / 14:.4f})"
            )

            # the day each user first purchases in the 14-day window gives every length at once
            first = (
                events.where(
                    (F.col("event_type") == "purchase")
                    & (F.col("event_date") >= F.lit(cutoff))
                    & (F.col("event_date") < F.lit(cutoff + timedelta(days=LABEL_DAYS)))
                )
                .join(population, "user_id")
                .groupBy("user_id")
                .agg(F.datediff(F.min("event_date"), F.lit(cutoff)).alias("day"))
                .groupBy("day")
                .count()
                .orderBy("day")
                .toPandas()
            )
            first["rate"] = first["count"].cumsum() / n
            first["days"] = first["day"] + 1
            print(first[["days", "count", "rate"]].to_string(index=False))
            population.unpersist()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
