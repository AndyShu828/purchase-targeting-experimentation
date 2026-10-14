"""Freeze the PSI reference: Valid's cuts and shares for p and the top five SHAP features.

Run under WSL from the repository root:
    PTE_ENV=wsl python scripts/build_psi_reference.py

Reads Valid's scores (<scores_dir>/2019-11-01/scores.parquet) and features
(<features_dir>/valid/features), cuts each variable at its deciles, and writes
<monitoring_dir>/psi_reference.json. daily_psi.py reads only that file afterwards, never Valid.
Fails before writing if the floor cannot flag the smallest bin emptying (check_floor).

Checked 2026-10-13: the file matches the one Andy's notebook saved, cell for cell; 1,793,124
users; bins 10 / 2 / 8 / 10 / 5 / 4; the smallest bin is n_categories at 0.0455, whose
emptying adds 0.488.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from pte.config import load_config, resolve
from pte.monitoring.psi import REFERENCE_CUTOFF, REFERENCE_FILE, build_reference, load_day


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--env", help="configs/<env>.yaml (default $PTE_ENV, then dev)")
    args = parser.parse_args(argv)
    paths = load_config(args.env)["paths"]

    valid = load_day(
        resolve(paths["scores_dir"]) / REFERENCE_CUTOFF / "scores.parquet",
        resolve(paths["features_dir"]) / "valid" / "features",
    )
    reference = build_reference(valid)

    out = resolve(paths["monitoring_dir"]) / REFERENCE_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"reference_cutoff": REFERENCE_CUTOFF, "variables": reference}, f, indent=2)
    print(f"Reference from {len(valid):,} users as of {REFERENCE_CUTOFF} -> {out}")
    for v, r in reference.items():
        print(f"  {v}: {len(r['expected'])} bins, smallest {min(r['expected']):.4f}")


if __name__ == "__main__":
    main()
