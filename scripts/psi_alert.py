"""Fail the alert task when any variable's PSI reaches ALERT_PSI; log the ones from 0.1 up."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date

import pandas as pd

from pte.config import load_config, resolve
from pte.monitoring.psi import ALERT_PSI

LOG_PSI = 0.1


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", required=True, type=date.fromisoformat, help="YYYY-MM-DD")
    parser.add_argument("--env", help="configs/<env>.yaml (default $PTE_ENV, then dev)")
    args = parser.parse_args(argv)
    monitoring_dir = resolve(load_config(args.env)["paths"]["monitoring_dir"])
    table = pd.read_parquet(monitoring_dir / "psi" / f"{args.cutoff.isoformat()}.parquet")

    for row in table.itertuples():
        if LOG_PSI <= row.psi < ALERT_PSI:
            print(row.variable, row.psi)
        if row.psi >= ALERT_PSI:
            raise ValueError(f"{row.variable} PSI {row.psi:.4f} >= {ALERT_PSI}")


if __name__ == "__main__":
    main()
