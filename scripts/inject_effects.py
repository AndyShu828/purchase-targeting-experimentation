"""Inject the simulated coupon effect into the test set and print the true effects.

Run under WSL from the repository root, after scripts/build_features.py and build_gmv.py:
    PTE_ENV=wsl python scripts/inject_effects.py

Writes <features_dir>/test/outcomes as Parquet, one row per test user: user_id, decile, y0, y1,
gmv0, gmv1 (experiment_design.md section 5). Deciles come from the frozen model, MLflow
version MODEL_VERSION, as in power_analysis.py. The arm is not read: Y(1) is built for every
user, and every later split is compared with the same truth.

Run on 2026-10-05 (conv_pp and gmv_effect agree with Andy's notebook to the digit, and with
a separate run that shuffled the rows first). 7,608 users flipped; gmv in price units, which
have no currency:

| decile |   n_d | tau_pp | conv_pp | gmv0 per user | gmv_effect |
| -----: | ----: | -----: | ------: | ------------: | ---------: |
|      1 | 230,568 | 0.1 | 0.100187 | 148.21 | 0.875912 |
|      2 | 230,568 | 0.2 | 0.199941 |  35.82 | 0.917059 |
|      3 | 230,568 | 0.4 | 0.399882 |  23.11 | 1.784483 |
|      4 | 230,568 | 0.5 | 0.500069 |  14.64 | 2.061336 |
|      5 | 230,567 | 0.6 | 0.599826 |  11.22 | 2.228999 |
|      6 | 230,568 | 0.5 | 0.500069 |   8.75 | 1.883209 |
|      7 | 230,568 | 0.4 | 0.399882 |   8.16 | 1.886322 |
|      8 | 230,415 | 0.3 | 0.299894 |   6.75 | 1.208786 |
|      9 | 230,316 | 0.2 | 0.200160 |   5.15 | 0.764714 |
|     10 | 230,973 | 0.1 | 0.100012 |   3.76 | 0.342846 |
|    all | 2,305,679 |  | 0.329968 |  26.56 | 1.395263 |

The GMV effect does not follow tau: it is about tau x the decile's spend per buyer, and
decile 1's buyers spend about 893 each against 402 in decile 5.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import mlflow
import pandas as pd

from pte.config import load_config, resolve
from pte.experiment.inject import SEED, inject, injected_effects
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
    users = (
        pd.DataFrame(
            {
                "user_id": features["user_id"],
                "decile": assign_deciles(model.predict(features)["p"]),
            }
        )
        .merge(
            pd.read_parquet(test_dir / "labels").rename(columns={"label": "y0"}),
            on="user_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            pd.read_parquet(test_dir / "gmv").rename(columns={"gmv": "gmv0"}),
            on="user_id",
            how="inner",
            validate="one_to_one",
        )
        .sort_values("user_id", ignore_index=True)
    )
    assert len(users) == len(features)

    outcomes = inject(users)
    outcomes.to_parquet(test_dir / "outcomes", index=False)

    table = injected_effects(outcomes)
    print(f"{MODEL_NAME} version {MODEL_VERSION}; seed {SEED}; {len(outcomes):,} users")
    print(table.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
