"""CUPED on the official split, from Andy's P4 notebook, next to the raw estimates.

Run under WSL from the repository root, after scripts/inject_effects.py:
    PTE_ENV=wsl python scripts/cuped_analysis.py

Pre-registered 2026-10-09 (experiment_design.md section 6) before any of it was computed:
conversion's covariate is the model score p, GMV's total_spend_per_day; theta is fitted on the
test set over both arms, within decile 1 and over all users; every row takes mean_diff_ci and
Welch's t. The primary result stays the raw z test of the main analysis; CUPED is secondary.
p is scored here by the same registered model at the same cutoff, so its deciles must equal
the ones the effects were injected by.

Run on 2026-10-09. It agrees with Andy's notebook to every printed digit, and diff_raw with the
main analysis; p on test has no missing value.

| row         | var_red |   pred | diff_raw |     diff |       se | se_ratio |        p | err_se |
| :---------- | ------: | -----: | -------: | -------: | -------: | -------: | -------: | -----: |
| conv dec. 1 |  0.0783 | 0.1120 | 0.000467 | 0.000787 | 0.001489 |    0.960 |    0.597 | -0.144 |
| conv all    |  0.0699 | 0.1099 | 0.003444 | 0.003397 | 0.000266 |    0.964 | 2.53e-37 |  0.365 |
| gmv dec. 1  |  0.1159 | 0.1942 | 2.824516 | 5.794612 | 3.837041 |    0.940 |    0.131 |  1.282 |
| gmv all     |  0.0999 | 0.1675 | 1.523330 | 1.757121 | 0.442708 |    0.949 | 7.22e-05 |  0.817 |

theta 0.7797, 0.5816, 2.8753 and 2.7995 in the same order; pred is pred_rho2.
Every CI covers its truth. The variance falls by 60-70% of the predicted rho^2 in every row:
rho on test is lower than on valid even against the untreated y0 and gmv0. Decile 1 stays null.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import mlflow
import pandas as pd

from pte.config import load_config, resolve
from pte.experiment.assignment import SALT, assign
from pte.experiment.cuped import cuped_analysis
from pte.experiment.estimate import observed
from pte.experiment.inject import injected_effects
from pte.model.registry import EXPERIMENT, MODEL_NAME
from pte.model.scoring import assign_deciles
from pte.model.tracking import setup_tracking

MODEL_VERSION = "1"  # frozen at v0.3-p3 (leakage.md section 5)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", help="configs/<env>.yaml (default $PTE_ENV, then dev)")
    args = parser.parse_args(argv)
    paths = load_config(args.env)["paths"]
    test_dir = resolve(paths["features_dir"]) / "test"
    setup_tracking(resolve(paths["mlflow_dir"]), EXPERIMENT)
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")

    features = pd.read_parquet(test_dir / "features")
    covariates = pd.DataFrame(
        {
            "user_id": features["user_id"],
            "p": model.predict(features)["p"].to_numpy(),
            "total_spend_per_day": features["total_spend_per_day"],
        }
    )
    covariates["decile_check"] = assign_deciles(covariates["p"])

    outcomes = pd.read_parquet(test_dir / "outcomes")
    effects = injected_effects(outcomes).set_index("decile")
    truth = {}
    for segment, decile in [("decile_1", 1), ("all", "all")]:
        truth[("conversion", segment)] = effects.loc[decile, "conv_pp"] / 100
        truth[("gmv", segment)] = effects.loc[decile, "gmv_effect"]

    split = assign(outcomes["user_id"])
    assert (split["user_id"].to_numpy() == outcomes["user_id"].to_numpy()).all()
    users = observed(outcomes.assign(arm=split["arm"].to_numpy())).merge(
        covariates, on="user_id", how="left", validate="one_to_one"
    )
    assert len(users) == len(outcomes)
    assert (users["decile"] == users["decile_check"]).all()

    table = cuped_analysis(users, truth)
    print(f'salt "{SALT}"; {MODEL_NAME} version {MODEL_VERSION}; {len(users):,} users')
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
