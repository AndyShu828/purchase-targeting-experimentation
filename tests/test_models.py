import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from pte.model.data import load_split, model_features
from pte.model.lgbm import EARLY_STOPPING_ROUNDS, MAX_TREES, train_lgbm
from pte.model.logreg import build_logreg


def test_model_features_drops_id_label_and_category_id():
    df = pd.DataFrame(columns=["user_id", "n_events", "main_category_id", "total_spend", "label"])
    assert model_features(df) == ["n_events", "total_spend"]


def write_split(root, split, features, labels):
    for kind, df in [("features", features), ("labels", labels)]:
        (root / split / kind).mkdir(parents=True)
        df.to_parquet(root / split / kind / "part-0.parquet")


def test_load_split_joins_labels(tmp_path):
    write_split(
        tmp_path,
        "train",
        pd.DataFrame({"user_id": [1, 2], "n_events": [3, 4], "total_spend": [0.0, 9.5]}),
        pd.DataFrame({"user_id": [2, 1], "label": [1, 0]}),
    )
    df = load_split(str(tmp_path), "train", columns=["n_events"])
    assert list(df.columns) == ["user_id", "n_events", "label"]
    assert df.set_index("user_id")["label"].to_dict() == {1: 0, 2: 1}


def test_load_split_rejects_users_without_label(tmp_path):
    write_split(
        tmp_path,
        "valid",
        pd.DataFrame({"user_id": [1, 2, 3], "n_events": [3, 4, 5]}),
        pd.DataFrame({"user_id": [1, 2], "label": [1, 0]}),
    )
    with pytest.raises(ValueError, match="1 users have no label"):
        load_split(str(tmp_path), "valid")


def test_logreg_freezes_the_training_median_and_flags_nulls():
    train = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, np.nan, np.nan, 10.0], "z": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]}
    )
    model = build_logreg().fit(train, [1, 1, 0, 0, 0, 1])
    imputer = model[0]
    # median of the non-NULL training x (1, 2, 3, 10) is 2.5; z has no NULL
    assert list(imputer.statistics_) == [2.5, 0.5]
    # a scoring set with its own, different median is still filled with 2.5
    valid = pd.DataFrame({"x": [np.nan, 100.0, 200.0], "z": [0.0, 0.0, 1.0]})
    filled = imputer.transform(valid)
    assert filled[:, 0].tolist() == [2.5, 100.0, 200.0]
    # one indicator column, for x only: z had no NULL in training
    assert filled[:, 2].tolist() == [1.0, 0.0, 0.0]
    assert filled.shape == (3, 3)
    # then standardized on the training set: mean 0, standard deviation 1 there
    scaled = model[:-1].transform(train)
    assert scaled.mean(axis=0) == pytest.approx(0, abs=1e-12)
    assert scaled.std(axis=0) == pytest.approx(1)


def noisy_data(n, seed):
    """One informative feature plus noise, so extra trees only fit noise."""
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({"signal": rng.normal(size=n), "noise": rng.normal(size=n)})
    y = (rng.random(n) < 1 / (1 + np.exp(-x["signal"]))).astype(int)
    return x, y


def test_lgbm_stops_early_and_keeps_only_the_best_trees():
    x_train, y_train = noisy_data(2000, 0)
    x_valid, y_valid = noisy_data(2000, 1)
    model = train_lgbm(x_train, y_train, x_valid, y_valid)
    best = model.best_iteration_
    assert 1 <= best < MAX_TREES - EARLY_STOPPING_ROUNDS
    # the EARLY_STOPPING_ROUNDS trees grown after the best one are dropped, so
    # predict_proba cannot use them by accident
    assert model.booster_.current_iteration() == best


def test_lgbm_warnings_survive_an_earlier_silent_model(capfd):
    """verbose=-1 anywhere in the process must not silence train_lgbm's warnings."""
    x, y = noisy_data(500, 2)
    lgb.LGBMClassifier(n_estimators=2, verbose=-1).fit(x, y)
    capfd.readouterr()
    # two aliases of one parameter: LightGBM warns which one it ignores
    model = train_lgbm(x, y, x, y, n_estimators=2, min_data_in_leaf=5)
    assert model.n_estimators == 2  # the caller's params override the defaults
    out = capfd.readouterr().out
    assert "[LightGBM] [Warning] min_data_in_leaf is set=5" in out
    assert "[LightGBM] [Info]" not in out
