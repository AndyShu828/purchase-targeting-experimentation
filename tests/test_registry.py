import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from pte.model.calibration import calibrate, fit_platt
from pte.model.lgbm import train_lgbm
from pte.model.registry import (
    CHAMPION,
    EXPERIMENT,
    MODEL_NAME,
    load_champion,
    register_final_model,
)
from pte.model.tracking import setup_tracking


@pytest.fixture
def tracking(tmp_path):
    setup_tracking(tmp_path / "mlflow", EXPERIMENT)
    yield
    mlflow.set_tracking_uri(None)


def fitted(seed=0, n=3000):
    """A small LightGBM + Platt fit the way the real one is: train, then valid for both."""
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(
        {
            "signal": rng.normal(size=n),
            "count": rng.integers(0, 5, size=n),
            "sparse": np.where(rng.random(n) < 0.3, rng.normal(size=n), np.nan),
        }
    )
    y = (rng.random(n) < 1 / (1 + np.exp(-(x["signal"] - 2)))).astype(int)
    train, valid = slice(0, n // 2), slice(n // 2, n)
    model = train_lgbm(x[train], y[train], x[valid], y[valid], min_child_samples=20)
    platt = fit_platt(model.predict_proba(x[valid])[:, 1], y[valid])
    return model, platt, x[valid], y[valid]


def test_the_loaded_model_scores_like_the_fit_it_came_from(tracking):
    model, platt, x_valid, y_valid = fitted()
    register_final_model(model, platt, x_valid, y_valid)
    scores = load_champion().predict(x_valid)
    p = model.predict_proba(x_valid)[:, 1]
    np.testing.assert_allclose(scores["p_raw"], p, rtol=0, atol=1e-12)
    np.testing.assert_allclose(scores["p"], calibrate(platt, p), rtol=0, atol=1e-12)


def test_features_are_taken_by_name_and_other_columns_ignored(tracking):
    model, platt, x_valid, y_valid = fitted()
    register_final_model(model, platt, x_valid, y_valid)
    loaded = load_champion()
    expected = loaded.predict(x_valid)
    # a features table as it is read: an id first, the columns in another order, a new index
    table = x_valid[["sparse", "count", "signal"]].assign(user_id=np.arange(len(x_valid)))
    table.index = table.index + 1000
    scores = loaded.predict(table[["user_id", "sparse", "count", "signal"]])
    assert list(scores.index) == list(table.index)
    np.testing.assert_array_equal(scores["p"], expected["p"])
    with pytest.raises(Exception, match="signal"):
        loaded.predict(x_valid.drop(columns="signal"))
    # called directly, past MLflow's signature check, the wrapper still picks by name
    direct = loaded.unwrap_python_model().predict(None, table)
    np.testing.assert_array_equal(direct["p"], expected["p"])


def test_each_registration_is_a_new_version_and_moves_the_champion(tracking):
    first, second = fitted(seed=0), fitted(seed=1)
    v1 = register_final_model(*first)
    v2 = register_final_model(*second)
    assert (str(v1.version), str(v2.version)) == ("1", "2")
    client = MlflowClient()
    assert str(client.get_model_version_by_alias(MODEL_NAME, CHAMPION).version) == "2"
    # the old version stays loadable by number
    old = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/1").predict(first[2])
    p = first[0].predict_proba(first[2])[:, 1]
    np.testing.assert_allclose(old["p"], calibrate(first[1], p), rtol=0, atol=1e-12)


def test_the_run_records_the_trees_platt_and_both_calibrations(tracking):
    model, platt, x_valid, y_valid = fitted()
    version = register_final_model(model, platt, x_valid, y_valid, tags={"features": "test"})
    run = mlflow.get_run(version.run_id)
    params, metrics = run.data.params, run.data.metrics
    assert params["n_trees"] == str(model.best_iteration_)
    assert float(params["platt_w"]) == platt.coef_[0][0]
    assert float(params["platt_c"]) == platt.intercept_[0]
    assert run.data.tags["features"] == "test"
    # unpenalized Platt puts the mean calibrated probability on the actual rate
    assert metrics["mean_p_platt"] == pytest.approx(metrics["actual_rate"], abs=1e-6)
    assert metrics["brier_platt"] <= metrics["brier_raw"]
    assert {"roc_auc", "ap", "lift_at_10pct", "lift_at_20pct"} <= set(metrics)
