"""The A/A simulation on the test set's y0, from Andy's P4 notebook.

Run under WSL from the repository root, after scripts/inject_effects.py:
    PTE_ENV=wsl python scripts/aa_simulation.py

Splits the test set under each of the 1000 A/A salts and runs the z test over all users and
within decile 1, both arms reading y0 (experiment_design.md section 2). Reads only user_id,
decile and y0 from <features_dir>/test/outcomes; never y1 or the official salt. Writes the
2000 rows to <features_dir>/test/aa and prints the verdict per metric. About 40 minutes.

Run on 2026-10-06 (the summary agrees with Andy's notebook to the digit; all 2000 rows agree,
to 1e-13, with a separate run that hashed the ids itself and tested with scipy's chi-square
on the 2 x 2 table):

| metric   |    n | cnt | pass_count |     ks_p | pass_ks |  diff_sd |
| :------- | ---: | --: | :--------- | -------: | :------ | -------: |
| all      | 1000 |  50 | True       | 0.769500 | True    | 0.000269 |
| decile_1 | 1000 |  39 | True       | 0.611160 | True    | 0.001494 |

The formula's SE on the official arms' sizes is 0.000271 over all users and 0.00155 in
decile 1, so diff_sd is 0.99 and 0.96 of it.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

import numpy as np
import pandas as pd

from pte.config import load_config, resolve
from pte.experiment.aa import AA_SALTS, aa_simulation, aa_summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", help="configs/<env>.yaml (default $PTE_ENV, then dev)")
    parser.add_argument("--n-salts", type=int, default=len(AA_SALTS), help="first n salts only")
    args = parser.parse_args(argv)
    features_dir = str(resolve(load_config(args.env)["paths"]["features_dir"]))

    users = pd.read_parquet(f"{features_dir}/test/outcomes", columns=["user_id", "decile", "y0"])
    start = time.time()
    results = aa_simulation(users, AA_SALTS[: args.n_salts])
    print(f"{args.n_salts} salts in {time.time() - start:.0f} s")
    if args.n_salts == len(AA_SALTS):
        results.to_parquet(f"{features_dir}/test/aa", index=False)

    print(aa_summary(results).round(6).to_string(index=False))
    for metric, part in results.groupby("metric"):
        counts = np.histogram(part["p"], bins=20, range=(0, 1))[0]
        print(f"{metric}: p in 20 bins of 0.05 (50 each expected): {counts.tolist()}")


if __name__ == "__main__":
    main()
