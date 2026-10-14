"""One cutoff's PSI against the frozen reference, one row per variable.

Run under WSL from the repository root, after score_users.py has scored the cutoff:
    PTE_ENV=wsl python scripts/daily_psi.py --cutoff 2019-12-01

Reads <scores_dir>/<cutoff>/scores.parquet, the features/ it scored and
<monitoring_dir>/psi_reference.json; cuts the day on the reference's frozen cuts and writes
<monitoring_dir>/psi/<cutoff>.parquet (cutoff, variable, psi, n_users), overwriting it, so a
rerun of the same cutoff gives the same file. It does not judge the numbers: alerting on
them is the alert task's job. n_users is there because PSI compares shares, so a day with
half the users and the same shares reads 0.

Checked 2026-10-13 on 2019-12-01: 2,281,094 users; p 0.0102, n_purchases_per_day 0.0041,
avg_session_seconds 0.0007, days_since_last_event 0.0203, n_categories 0.0026, active_days
0.0052, as in Andy's notebook.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import date

from pte.config import load_config, resolve
from pte.monitoring.psi import REFERENCE_FILE, load_day, psi_table


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cutoff", required=True, type=date.fromisoformat, help="YYYY-MM-DD")
    parser.add_argument("--env", help="configs/<env>.yaml (default $PTE_ENV, then dev)")
    args = parser.parse_args(argv)
    paths = load_config(args.env)["paths"]
    day_dir = resolve(paths["scores_dir"]) / args.cutoff.isoformat()
    monitoring_dir = resolve(paths["monitoring_dir"])

    with open(monitoring_dir / REFERENCE_FILE) as f:
        reference = json.load(f)["variables"]
    day = load_day(day_dir / "scores.parquet", day_dir / "features", list(reference))

    table = psi_table(day, reference)
    table.insert(0, "cutoff", args.cutoff)
    table["n_users"] = len(day)
    out = monitoring_dir / "psi" / f"{args.cutoff.isoformat()}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    print(f"PSI for {args.cutoff}, {len(day):,} users -> {out}")
    print(table[["variable", "psi"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
