import numpy as np
import pandas as pd
import pytest

from pte.model.explain import dependence, importance, shap_values
from pte.model.lgbm import train_lgbm


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    n = 3000
    x = pd.DataFrame(
        {
            # columns not in importance order, so an unsorted table would show
            "noise": rng.normal(size=n),
            "weak": rng.normal(size=n),
            "strong": rng.normal(size=n),
        }
    )
    x.loc[rng.random(n) < 0.1, "weak"] = np.nan
    logit = -2 + 1.5 * x["strong"] + 0.4 * x["weak"].fillna(0)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    model = train_lgbm(x[:2000], y[:2000], x[2000:], y[2000:], min_child_samples=20)
    return model, x


def test_shap_values_add_up_to_the_raw_score(fitted):
    model, x = fitted
    sample, values, base = shap_values(model, x, n=500, seed=1)
    assert len(sample) == len(values) == 500
    assert list(values.columns) == list(x.columns)
    assert values.index.equals(sample.index)
    raw = model.predict(sample, raw_score=True)
    assert np.allclose(base + values.sum(axis=1), raw, atol=1e-10)


def test_shap_values_sample_only_when_asked(fitted):
    model, x = fitted
    sample, values, _ = shap_values(model, x, n=None)
    assert sample.index.equals(x.index)
    # asking for at least every row keeps them all, in order, instead of shuffling
    assert shap_values(model, x, n=len(x))[0].index.equals(x.index)
    again, _, _ = shap_values(model, x, n=500, seed=1)
    assert again.index.equals(shap_values(model, x, n=500, seed=1)[0].index)


def test_importance_ranks_by_mean_abs_shap_and_shares_sum_to_one(fitted):
    model, x = fitted
    _, values, _ = shap_values(model, x, n=None)
    table = importance(model, values)
    assert table.index[0] == "strong"
    assert table["mean_abs_shap"].is_monotonic_decreasing
    for share in ("shap_share", "gain_share", "split_share"):
        assert table[share].sum() == pytest.approx(1)
    assert table.loc["weak", "mean_abs_shap"] == pytest.approx(values["weak"].abs().mean())


def test_dependence_by_hand():
    # few levels: a row per value, in value order, NULLs last
    feature = pd.Series([3, 1, 1, 2, np.nan, 3])
    shap = pd.Series([0.6, -0.2, -0.4, 0.1, 0.5, 0.2])
    table = dependence(feature, shap)
    assert table.index.tolist() == ["1.0", "2.0", "3.0", "NULL"]
    assert table["n"].tolist() == [2, 1, 2, 1]
    assert table["mean_shap"].tolist() == pytest.approx([-0.3, 0.1, 0.4, 0.5])


def test_dependence_keeps_zero_apart_from_the_bins():
    rng = np.random.default_rng(2)
    values = np.where(rng.random(1000) < 0.9, 0.0, rng.uniform(0.1, 8, 1000))
    feature = pd.Series(values)
    table = dependence(feature, pd.Series(values * 0.1), bins=4, max_levels=5)
    assert table.index[0] == "0"
    assert table.loc["0", "n"] == (values == 0).sum()
    assert len(table) == 5  # 0 plus four quantile bins of the non-zero values
    assert table["mean_shap"].is_monotonic_increasing
