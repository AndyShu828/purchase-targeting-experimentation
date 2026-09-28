import subprocess

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss

from pte.model.lgbm import TUNED_PARAMS, train_lgbm
from pte.model.tracking import EXPERIMENT, git_commit, log_lgbm_run, metric_name, setup_tracking


@pytest.fixture
def tracking(tmp_path):
    setup_tracking(tmp_path / "mlflow")
    yield tmp_path / "mlflow"
    mlflow.set_tracking_uri(None)


def small_lgbm(**params):
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"signal": rng.normal(size=1500), "noise": rng.normal(size=1500)})
    y = (rng.random(1500) < 1 / (1 + np.exp(-x["signal"]))).astype(int)
    model = train_lgbm(x[:1000], y[:1000], x[1000:], y[1000:], **params)
    return model, y[1000:], model.predict_proba(x[1000:])[:, 1]


def test_setup_creates_the_store_and_experiment(tracking):
    assert (tracking / "mlflow.db").exists()
    assert mlflow.get_experiment_by_name(EXPERIMENT) is not None
    # running it again reuses both
    setup_tracking(tracking)
    assert len(mlflow.search_experiments(filter_string=f"name = '{EXPERIMENT}'")) == 1


def test_a_run_holds_the_grid_params_and_validation_metrics(tracking):
    model, y_valid, p_valid = small_lgbm(num_leaves=7, min_child_samples=40)
    run_id = log_lgbm_run(model, y_valid, p_valid, run_name="leaves 7", tags={"features": "raw"})
    run = mlflow.get_run(run_id)
    assert run.data.tags["features"] == "raw"
    assert run.info.run_name == "leaves 7"
    assert run.data.params["num_leaves"] == "7"
    assert run.data.params["min_child_samples"] == "40"
    assert run.data.params["learning_rate"] == str(TUNED_PARAMS["learning_rate"])
    metrics = run.data.metrics
    assert metrics["log_loss"] == pytest.approx(log_loss(y_valid, p_valid))
    assert metrics["best_iteration"] == model.best_iteration_
    assert set(metrics) == {
        "log_loss",
        "best_iteration",
        "roc_auc",
        "ap",
        "lift_at_10pct",
        "lift_at_20pct",
        "brier",
    }
    assert run.data.tags["n_features"] == "2"
    assert run.data.tags["git_commit"] == git_commit()


def test_runs_can_be_ranked_by_log_loss(tracking):
    for leaves in (3, 15):
        model, y_valid, p_valid = small_lgbm(num_leaves=leaves)
        log_lgbm_run(model, y_valid, p_valid)
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT], order_by=["metrics.log_loss ASC"])
    assert len(runs) == 2
    assert runs["metrics.log_loss"].is_monotonic_increasing


@pytest.mark.parametrize("status, expected", [("", "abc1234"), (" M src/x.py", "abc1234-dirty")])
def test_git_commit_marks_uncommitted_changes(monkeypatch, status, expected):
    outputs = {"rev-parse": "abc1234\n", "status": status}

    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout=outputs[args[1]])

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert git_commit() == expected


def test_metric_names_drop_characters_mlflow_rejects():
    assert metric_name("lift@10%") == "lift_at_10pct"
    assert metric_name("roc_auc") == "roc_auc"
