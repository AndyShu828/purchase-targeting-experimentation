from datetime import date

import numpy as np
import pandas as pd
import pytest

from pte.model.scoring import assign_deciles, check_cutoff, decile_summary, score_users


def test_decile_1_is_the_highest_tenth():
    p = np.random.default_rng(0).permutation(np.arange(100) / 100)
    deciles = assign_deciles(p)
    assert np.bincount(deciles).tolist() == [0] + [10] * 10
    assert set(deciles[p >= 0.9]) == {1}
    assert set(deciles[p < 0.1]) == {10}


def test_equal_scores_share_a_decile():
    # 0.5 appears 15 times across what would be the boundary between two exact deciles
    p = np.concatenate([np.arange(40) / 100, np.full(15, 0.5), 0.6 + np.arange(45) / 100])
    deciles = assign_deciles(p)
    assert len(set(deciles[p == 0.5])) == 1
    assert np.bincount(deciles).sum() == 100


def test_ties_too_large_to_cut_raise():
    with pytest.raises(ValueError):
        assign_deciles(np.r_[np.zeros(50), np.arange(50)])


class FakeModel:
    """Stands in for the pyfunc: scores by the column signal, returns a new index."""

    def predict(self, features):
        p = features["signal"].to_numpy()
        return pd.DataFrame({"p_raw": p, "p": p / 2})


def test_scores_line_up_with_the_users_they_came_from():
    features = pd.DataFrame(
        {"user_id": [30, 10, 20] * 10, "signal": np.linspace(0.9, 0.1, 30)},
        index=range(100, 130),
    )
    scores = score_users(FakeModel(), features, date(2019, 12, 1), "3")
    assert list(scores.columns) == ["user_id", "cutoff", "model_version", "p_raw", "p", "decile"]
    assert scores["user_id"].tolist() == features["user_id"].tolist()
    assert scores["p_raw"].tolist() == features["signal"].tolist()
    assert (scores["p"] == scores["p_raw"] / 2).all()
    assert scores["decile"].tolist() == [d for d in range(1, 11) for _ in range(3)]
    assert (scores["cutoff"] == date(2019, 12, 1)).all()
    assert (scores["model_version"] == "3").all()

    # shuffled, as real scores come in user order, not decile order
    summary = decile_summary(scores.sample(frac=1, random_state=0))
    assert summary.index.tolist() == list(range(1, 11))
    assert summary["users"].tolist() == [3] * 10
    assert summary["mean_p"].is_monotonic_decreasing


@pytest.mark.parametrize(
    "cutoff, ok",
    [
        (date(2019, 10, 14), False),  # window starts 09-30, a day before the data
        (date(2019, 10, 15), True),  # the train cutoff: window is exactly 10-01 .. 10-14
        (date(2019, 12, 1), True),  # the morning after the last day
        (date(2019, 12, 2), False),  # 12-01 would be missing from the window
    ],
)
def test_the_window_must_lie_inside_the_data(cutoff, ok):
    args = (cutoff, date(2019, 10, 1), date(2019, 11, 30), 14)
    if ok:
        check_cutoff(*args)
    else:
        with pytest.raises(ValueError, match="outside"):
            check_cutoff(*args)
