"""The SRM check on the coupon experiment's official split, from Andy's P4 notebook.

Run under WSL from the repository root, after scripts/build_features.py:
    PTE_ENV=wsl python scripts/srm_check.py

Splits the test set (cutoff 11-18) under the official salt (assignment.SALT) and tests the two
arms' head counts against 50/50 (experiment_design.md section 2): the alarm is p < 0.001 over
all users; each decile's test is a diagnostic only. Deciles come from the frozen model, MLflow
version MODEL_VERSION, as in power_analysis.py. Only features are read, never a label.

Run on 2026-10-05 (the all row agrees with Andy's notebook to the digit):

| decile |     n_t |     n_c | share_t | chi_sq |      p |
| -----: | ------: | ------: | ------: | -----: | -----: |
|      1 | 115,735 | 114,833 |  0.5020 |  3.529 | 0.0603 |
|      2 | 115,262 | 115,306 |  0.4999 |  0.008 | 0.9270 |
|      3 | 115,860 | 114,708 |  0.5025 |  5.756 | 0.0164 |
|      4 | 115,359 | 115,209 |  0.5003 |  0.098 | 0.7547 |
|      5 | 115,354 | 115,213 |  0.5003 |  0.086 | 0.7690 |
|      6 | 115,887 | 114,681 |  0.5026 |  6.308 | 0.0120 |
|      7 | 115,245 | 115,323 |  0.4998 |  0.026 | 0.8710 |
|      8 | 115,165 | 115,250 |  0.4998 |  0.031 | 0.8594 |
|      9 | 115,215 | 115,101 |  0.5002 |  0.056 | 0.8122 |
|     10 | 115,561 | 115,412 |  0.5003 |  0.096 | 0.7565 |
|    all | 1,154,643 | 1,151,036 | 0.5008 | 5.643 | 0.0175 |

No alarm: p = 0.0175 over all users is not below 0.001. Deciles 3 and 6 are below 0.05; with
Holm over the ten the smallest adjusted p is 0.12, so no decile stands out.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import mlflow
import pandas as pd

from pte.config import load_config, resolve
from pte.experiment.assignment import SALT, assign
from pte.experiment.srm import SRM_ALPHA, srm_table
from pte.model.registry import EXPERIMENT, MODEL_NAME
from pte.model.scoring import assign_deciles
from pte.model.tracking import setup_tracking

MODEL_VERSION = "1"  # frozen at v0.3-p3 (leakage.md section 5)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", help="configs/<env>.yaml (default $PTE_ENV, then dev)")
    args = parser.parse_args(argv)
    paths = load_config(args.env)["paths"]
    features_dir = str(resolve(paths["features_dir"]))
    setup_tracking(resolve(paths["mlflow_dir"]), EXPERIMENT)
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")

    test = pd.read_parquet(f"{features_dir}/test/features")
    test["decile"] = assign_deciles(model.predict(test)["p"])
    split = assign(test["user_id"]).merge(
        test[["user_id", "decile"]], on="user_id", how="inner", validate="one_to_one"
    )
    assert len(split) == len(test)

    table = srm_table(split)
    print(f'salt "{SALT}"; {MODEL_NAME} version {MODEL_VERSION}; alarm at p < {SRM_ALPHA}')
    print(table.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
