"""Register the final P3 model in MLflow: the tuned LightGBM and its Platt calibration as one model.

The final model (decided 2026-09-29, Andy) is LightGBM on the 20 model_features, TUNED_PARAMS,
trained on train with early stopping on valid, then Platt-calibrated on valid. Batch scoring
and P4 need the calibrated probability, so the two are registered together as one pyfunc
model: loading it gives both the raw and the calibrated probability, and nobody can score with
one fit's trees and another fit's Platt.

A registered version is fixed; the alias CHAMPION points at the version to score with, so a
retrained model moves the alias instead of every caller's version number.

Loading needs the pte package installed: the pyfunc stores CalibratedLGBM by reference to this
module, not a copy of its code.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from mlflow.entities.model_registry import ModelVersion
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from numpy.typing import ArrayLike
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from pte.model.calibration import logit
from pte.model.metrics import evaluate
from pte.model.tracking import git_commit, metric_name

EXPERIMENT = "p3-final-model"
MODEL_NAME = "pte-purchase-propensity"
CHAMPION = "champion"
# rows kept as the input example, enough to show the columns and a few NULLs
EXAMPLE_ROWS = 5


class CalibratedLGBM(mlflow.pyfunc.PythonModel):
    """LightGBM's probability p, then Platt's q = 1 / (1 + exp(-(w * logit(p) + c))).

    predict returns one row per input row, in the same order and with the same index, with
    columns p_raw and p (calibrated). Columns other than the model's features are ignored, so
    a whole features table (user_id included) can go in as it is.
    """

    def load_context(self, context) -> None:
        self.booster = lgb.Booster(model_file=context.artifacts["booster"])
        platt = json.loads(Path(context.artifacts["platt"]).read_text())
        self.w, self.c = platt["w"], platt["c"]

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        # Through the pyfunc, MLflow's signature check has already cut the input down to the
        # features in signature order; by name keeps direct calls safe as well, since the
        # booster itself scores whatever columns it is given by position.
        p = self.booster.predict(model_input[self.booster.feature_name()])
        q = 1 / (1 + np.exp(-(self.w * logit(p) + self.c)))
        return pd.DataFrame({"p_raw": p, "p": q}, index=model_input.index)


def platt_params(platt: LogisticRegression) -> dict[str, float]:
    return {"w": float(platt.coef_[0][0]), "c": float(platt.intercept_[0])}


def register_final_model(
    model: lgb.LGBMClassifier,
    platt: LogisticRegression,
    x_valid: pd.DataFrame,
    y_valid: ArrayLike,
    tags: dict[str, str] | None = None,
) -> ModelVersion:
    """Log LightGBM + Platt as one run, register it as a new version of MODEL_NAME and point
    CHAMPION at it; returns the new version, whose run_id leads back to the run.

    x_valid is the validation set the early stopping and Platt were fit on. Its metrics are
    logged for the record and are optimistic for that reason; the test set is scored once,
    elsewhere. Ranking metrics are the same before and after Platt (increasing in p for w > 0),
    so they are logged once; log loss, Brier and the mean prediction are logged for both.
    """
    y_valid = np.asarray(y_valid)
    with tempfile.TemporaryDirectory() as tmp:
        booster_path = Path(tmp) / "booster.txt"
        platt_path = Path(tmp) / "platt.json"
        # LightGBM already trims the booster to the trees early stopping kept; named anyway
        model.booster_.save_model(booster_path, num_iteration=model.best_iteration_)
        platt_path.write_text(json.dumps(platt_params(platt)))
        artifacts = {"booster": str(booster_path), "platt": str(platt_path)}

        # score valid through the saved files, so the logged metrics are the saved model's;
        # a separate instance, since the one logged is pickled and must not carry a booster
        scorer = CalibratedLGBM()
        scorer.load_context(mlflow.pyfunc.PythonModelContext(artifacts, None))
        scores = scorer.predict(None, x_valid)
        p, q = scores["p_raw"].to_numpy(), scores["p"].to_numpy()

        metrics = {metric_name(k): v for k, v in evaluate(y_valid, q).items() if k != "brier"}
        for suffix, probs in [("raw", p), ("platt", q)]:
            metrics |= {
                f"log_loss_{suffix}": log_loss(y_valid, probs),
                f"brier_{suffix}": brier_score_loss(y_valid, probs),
                f"mean_p_{suffix}": probs.mean(),
            }
        metrics["actual_rate"] = y_valid.mean()

        params = model.get_params()
        example = x_valid.head(EXAMPLE_ROWS)
        with mlflow.start_run(run_name="final: LightGBM + Platt"):
            mlflow.log_params(
                {k: params[k] for k in ["num_leaves", "min_child_samples", "learning_rate"]}
                | {"n_trees": model.best_iteration_}
                | {f"platt_{k}": v for k, v in platt_params(platt).items()}
            )
            mlflow.log_metrics(metrics)
            mlflow.set_tags(
                {"git_commit": git_commit(), "n_features": model.n_features_in_} | (tags or {})
            )
            info = mlflow.pyfunc.log_model(
                name="model",
                python_model=CalibratedLGBM(),
                artifacts=artifacts,
                signature=infer_signature(example, scorer.predict(None, example)),
                input_example=example,
                # pinned to what trained it; listing them skips MLflow's slow inference
                pip_requirements=[
                    f"lightgbm=={lgb.__version__}",
                    f"mlflow=={mlflow.__version__}",
                    f"numpy=={np.__version__}",
                    f"pandas=={pd.__version__}",
                ],
                registered_model_name=MODEL_NAME,
            )

    client = MlflowClient()
    version = client.get_model_version(MODEL_NAME, str(info.registered_model_version))
    client.set_registered_model_alias(MODEL_NAME, CHAMPION, version.version)
    return version


def load_champion() -> mlflow.pyfunc.PyFuncModel:
    """The version CHAMPION points at, from the tracking URI set by setup_tracking."""
    return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{CHAMPION}")
