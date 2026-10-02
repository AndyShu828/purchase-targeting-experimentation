"""The coupon experiment's minimum detectable effect for each decile, from Andy's P4 notebook.

Run under WSL from the repository root, after scripts/build_features.py:
    PTE_ENV=wsl python scripts/power_analysis.py

The experiment population is the test set (cutoff 11-18), split 50/50 (experiment_design.md
sections 1-2). The baseline rate of each decile is valid's actual rate (cutoff 11-01); the
test set's labels are the control arm's outcome, so only its features are read here. Both
sets are scored with the frozen model, MLflow version MODEL_VERSION, and cut into deciles on
their own scores. Two-sided alpha 0.05, power 80% (section 4).

Run on 2026-10-03 (Andy's notebook gave the same table):

| decile | n_valid |    r_d |  n_test |     n_arm | mde_pp | mde_rel |
| -----: | ------: | -----: | ------: | --------: | -----: | ------: |
|      1 | 179,313 | 0.2146 | 230,568 |   115,284 |  0.479 |    2.2% |
|      2 | 179,312 | 0.0734 | 230,568 |   115,284 |  0.304 |    4.1% |
|      3 | 179,312 | 0.0493 | 230,568 |   115,284 |  0.253 |    5.1% |
|      4 | 179,313 | 0.0379 | 230,568 |   115,284 |  0.223 |    5.9% |
|      5 | 179,309 | 0.0318 | 230,567 | 115,283.5 |  0.205 |    6.4% |
|      6 | 179,315 | 0.0262 | 230,568 |   115,284 |  0.186 |    7.1% |
|      7 | 179,282 | 0.0220 | 230,568 |   115,284 |  0.171 |    7.8% |
|      8 | 179,341 | 0.0181 | 230,415 | 115,207.5 |  0.156 |    8.6% |
|      9 | 179,227 | 0.0157 | 230,316 |   115,158 |  0.145 |    9.3% |
|     10 | 179,400 | 0.0110 | 230,973 | 115,486.5 |  0.122 |   11.0% |
|    all | 1,793,124 | 0.0500 | 2,305,679 | 1,152,839.5 | 0.080 | 1.6% |

The primary test (decile 1) needs an injected effect of 0.48 pp for 80% power.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import mlflow
import pandas as pd

from pte.config import load_config, resolve
from pte.experiment.power import mde_table
from pte.model.data import load_split
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

    valid = load_split(features_dir, "valid")
    # features only: the test labels are the control arm's outcome
    test = pd.read_parquet(f"{features_dir}/test/features")
    for df in (valid, test):
        df["p"] = model.predict(df)["p"]
        df["decile"] = assign_deciles(df["p"])

    table = mde_table(valid, test)
    print(f"{MODEL_NAME} version {MODEL_VERSION}; two-sided alpha 0.05, power 80%, 50/50 split")
    print(table.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
