import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from pte.model.metrics import evaluate
from pte.model.segments import SEGMENTS, evaluate_segments, pair_auc


def toy():
    """20 users: bought or not crossed with 1 or 2+ active days, labels mixed in each."""
    rng = np.random.default_rng(0)
    n = 20
    df = pd.DataFrame(
        {
            "n_purchases": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 3, 1, 1, 2, 1, 1, 4],
            "active_days": [1, 1, 1, 1, 1, 2, 3, 2, 5, 2, 1, 1, 1, 1, 1, 2, 2, 4, 3, 2],
            "label": [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0],
        }
    )
    return df, rng.random(n)


def test_segments_split_the_population_both_ways():
    df, _ = toy()
    masks = {name: segment(df).to_numpy() for name, segment in SEGMENTS.items()}
    assert masks["all"].all()
    assert (masks["bought"] ^ masks["not bought"]).all()
    assert (masks["active 1 day"] ^ masks["active 2+ days"]).all()
    assert masks["bought"].sum() == 10
    assert masks["active 1 day"].sum() == 10


def test_each_row_is_evaluate_on_that_segment():
    df, score = toy()
    table = evaluate_segments(df, score)
    assert list(table.index) == list(SEGMENTS)
    bought = df["n_purchases"].to_numpy() > 0
    expected = evaluate(df["label"][bought], score[bought])
    for metric, value in expected.items():
        assert table.loc["bought", metric] == pytest.approx(value)
    assert table.loc["bought", "n"] == 10
    assert table.loc["bought", "positive_rate"] == pytest.approx(0.5)
    assert table.loc["bought", "mean_score"] == pytest.approx(score[bought].mean())
    assert table.loc["all", "roc_auc"] == pytest.approx(roc_auc_score(df["label"], score))


def test_a_rule_score_gets_no_brier_or_mean():
    df, score = toy()
    table = evaluate_segments(df, np.round(score * 3), probability=False)
    assert "brier" not in table.columns
    assert "mean_score" not in table.columns


def test_evaluate_segments_reads_score_by_position_not_index():
    df, score = toy()
    shuffled = df.set_index(pd.Index(np.arange(20)[::-1]))
    assert evaluate_segments(shuffled, score).equals(evaluate_segments(df, score))


def test_pair_auc_by_hand():
    # group 1: positive 0.9, negative 0.5; group 0: positive 0.6, negatives 0.2 and 0.6 (a tie)
    score = [0.9, 0.5, 0.6, 0.2, 0.6]
    label = [1, 0, 1, 0, 0]
    group = [1, 1, 0, 0, 0]
    table = pair_auc(score, label, group).set_index(["positive_in_group", "negative_in_group"])
    assert table.loc[(True, True), "auc"] == 1  # 0.9 > 0.5
    assert table.loc[(True, False), "auc"] == 1  # 0.9 > 0.2, 0.6
    assert table.loc[(False, True), "auc"] == 1  # 0.6 > 0.5
    assert table.loc[(False, False), "auc"] == 0.75  # 0.6 > 0.2, 0.6 ties 0.6
    # 2 positives x 3 negatives = 6 pairs: 1, 2, 1 and 2 of them
    assert table["pair_share"].tolist() == pytest.approx([1 / 6, 2 / 6, 1 / 6, 2 / 6])


def test_pair_auc_averages_back_to_the_overall_auc():
    rng = np.random.default_rng(1)
    group = rng.random(3000) < 0.2
    label = (rng.random(3000) < np.where(group, 0.3, 0.05)).astype(int)
    # a score that mostly reads the group, like LightGBM leaning on past purchases
    score = np.round(group * 1.0 + rng.normal(0, 0.8, 3000) + label * 0.5, 2)
    table = pair_auc(score, label, group)
    assert table["pair_share"].sum() == pytest.approx(1)
    assert (table["auc"] * table["pair_share"]).sum() == pytest.approx(roc_auc_score(label, score))
    within = table[table["positive_in_group"] == table["negative_in_group"]]
    assert within.iloc[0]["auc"] == pytest.approx(roc_auc_score(label[group], score[group]))
    assert within.iloc[1]["auc"] == pytest.approx(roc_auc_score(label[~group], score[~group]))


def test_pair_auc_is_nan_when_a_cell_has_no_pairs():
    table = pair_auc([0.9, 0.1, 0.4], [1, 0, 0], [1, 1, 1])
    assert table.loc[0, "auc"] == 1
    assert table.loc[1:, "auc"].isna().all()
    assert table["pair_share"].tolist() == [1, 0, 0, 0]
