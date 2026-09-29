"""MLflow tracking for the P3 models.

Decided 2026-09-29: every LightGBM tuning run is logged, with its parameters, the number of
trees early stopping kept and its validation metrics, and the run with the lowest validation
log loss wins. The two baselines are logged too, in their own experiment (BASELINE_EXPERIMENT;
added 2026-09-30, as the plan asks for every run), since RFM has no log loss to rank by. Runs
go to a SQLite file under paths.mlflow_dir (outside the repository, like the data). Browse
them with
    mlflow ui --backend-store-uri sqlite:///<mlflow_dir>/mlflow.db
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import lightgbm as lgb
import mlflow
from numpy.typing import ArrayLike
from sklearn.metrics import log_loss

from pte.config import PROJECT_ROOT
from pte.model.metrics import evaluate

EXPERIMENT = "p3-lgbm-tuning"
BASELINE_EXPERIMENT = "p3-baselines"
# the LightGBM parameters the tuning grid touches, plus the ones that set its scale
LOGGED_PARAMS = ["num_leaves", "min_child_samples", "learning_rate", "n_estimators"]


def setup_tracking(mlflow_dir: str | Path, experiment: str = EXPERIMENT) -> None:
    """Point MLflow at <mlflow_dir>/mlflow.db and select the experiment, creating both."""
    mlflow_dir = Path(mlflow_dir).resolve()
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_dir.as_posix()}/mlflow.db")
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(mlflow_dir / "artifacts").as_uri())
    mlflow.set_experiment(experiment)


def git_commit() -> str:
    """The commit the code ran from, marked dirty when the working tree has changes."""

    def git(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True)
        return result.stdout.strip()

    return git("rev-parse", "--short", "HEAD") + ("-dirty" if git("status", "--porcelain") else "")


def metric_name(name: str) -> str:
    """MLflow metric names allow no @ or %: lift@10% is logged as lift_at_10pct."""
    return name.replace("@", "_at_").replace("%", "pct")


def log_lgbm_run(
    model: lgb.LGBMClassifier,
    y_valid: ArrayLike,
    p_valid: ArrayLike,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """Log one fitted LightGBM and its validation predictions as a run; returns the run id.

    Metrics are the validation log loss (the tuning criterion), best_iteration and evaluate()'s
    ROC-AUC, AP, lift@10%, lift@20% and Brier. tags carry what the model cannot report about
    itself, such as how the features were transformed: the 2026-09-29 grid ran on log1p
    features, and that changed some runs.
    """
    params = model.get_params()
    metrics = {"log_loss": log_loss(y_valid, p_valid), "best_iteration": model.best_iteration_}
    metrics |= {metric_name(name): value for name, value in evaluate(y_valid, p_valid).items()}
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({name: params[name] for name in LOGGED_PARAMS})
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {"git_commit": git_commit(), "n_features": model.n_features_in_} | (tags or {})
        )
    return run.info.run_id


def log_baseline_run(
    run_name: str,
    y_valid: ArrayLike,
    score_valid: ArrayLike,
    probability: bool = True,
    params: dict[str, object] | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """Log one baseline's validation metrics as a run in the current experiment; returns its id.

    Metrics are evaluate()'s, plus log loss for a probability. RFM's 0-15 score is no
    probability, so it gets neither log loss nor Brier. params describe the baseline, since
    unlike LightGBM there is no fitted model to read them from.
    """
    metrics = {metric_name(k): v for k, v in evaluate(y_valid, score_valid, probability).items()}
    if probability:
        metrics["log_loss"] = log_loss(y_valid, score_valid)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params or {})
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"git_commit": git_commit()} | (tags or {}))
    return run.info.run_id
