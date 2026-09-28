from itertools import permutations

import pandas as pd
import pytest

from pte.model.metrics import evaluate, lift_at

# The notebook's toy table: score 4 is a tie of 4 users holding 1 positive.
SCORE = [5, 5, 4, 4, 4, 4, 2, 1]
LABEL = [1, 1, 1, 0, 0, 0, 1, 0]


@pytest.mark.parametrize(
    ("frac", "expected"),
    [
        (0.25, 2.0),  # k=2: both score-5 users, both positive -> (2/2) / (4/8)
        (0.5, 1.25),  # k=4: 2 + 2 * 1/4 = 2.5 positives -> (2.5/4) / (4/8)
        (0.75, 1.0),  # k=6 ends exactly on the score-4 group: all of it, no fraction
        (0.875, 8 / 7),  # k=7: the score-2 user, a single-user "tie"
        (1.0, 1.0),  # everyone: the overall rate by definition
    ],
)
def test_lift_at_toy(frac, expected):
    assert lift_at(SCORE, LABEL, frac) == pytest.approx(expected)


def brute_force_lift(score, label, frac):
    """Mean plain top-k lift over every row order: ties broken each possible way."""
    k = int(len(score) * frac)
    rate = sum(label) / len(label)
    total, count = 0.0, 0
    for order in permutations(range(len(score))):
        # sorted() is stable, so tied users keep the permuted order among themselves
        top = sorted(order, key=lambda i: -score[i])[:k]
        total += sum(label[i] for i in top) / k / rate
        count += 1
    return total / count


@pytest.mark.parametrize("frac", [0.25, 0.375, 0.5, 0.625, 0.75])
def test_lift_at_is_mean_over_tie_orders(frac):
    """The m * P_T / n_T rule equals the average over all 8! orders, not an approximation."""
    score = [3, 3, 3, 2, 2, 2, 2, 1]
    label = [1, 0, 0, 1, 1, 0, 0, 1]
    assert lift_at(score, label, frac) == pytest.approx(brute_force_lift(score, label, frac))


def test_lift_at_without_ties_is_plain_top_k():
    score = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    label = [1, 0, 1, 1, 0, 0, 0, 1, 0, 0]
    # top 3 hold 2 positives: (2/3) / (4/10)
    assert lift_at(score, label, 0.3) == pytest.approx((2 / 3) / 0.4)


def test_lift_at_ignores_row_order():
    """Taking tied rows in table order would give 2.0 here, and 0.0 reversed."""
    score = [1, 1, 1, 1]
    label = [1, 1, 0, 0]
    assert lift_at(score, label, 0.5) == lift_at(score[::-1], label[::-1], 0.5) == 1.0


def test_lift_at_uses_position_not_index():
    """Two Series with different indexes must pair by position, not align into NaN."""
    score = pd.Series(SCORE, index=range(100, 108))
    label = pd.Series(LABEL)
    assert lift_at(score, label, 0.5) == pytest.approx(1.25)


@pytest.mark.parametrize("frac", [0.0, 0.1, 1.5])
def test_lift_at_rejects_empty_or_oversized_top(frac):
    # 0.1 of 8 users is int(0.8) = 0 users
    with pytest.raises(ValueError, match="selects k="):
        lift_at(SCORE, LABEL, frac)


def test_lift_at_rejects_nan_score():
    with pytest.raises(ValueError, match="NaN"):
        lift_at([1.0, float("nan"), 0.5], [1, 0, 0], 0.5)


def test_evaluate_matches_sklearn_and_lift_at():
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    prob = [0.9, 0.8, 0.4, 0.4, 0.4, 0.4, 0.3, 0.1, 0.05, 0.01]
    # the second user is a negative, so lift@10% (3.33) and lift@20% (1.67) differ
    label = [1, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    assert evaluate(label, prob) == {
        "roc_auc": pytest.approx(roc_auc_score(label, prob)),
        "ap": pytest.approx(average_precision_score(label, prob)),
        "lift@10%": pytest.approx(lift_at(prob, label, 0.1)),
        "lift@20%": pytest.approx(lift_at(prob, label, 0.2)),
        "brier": pytest.approx(brier_score_loss(label, prob)),
    }


def test_evaluate_gives_a_rule_score_no_brier():
    # RFM's 0-15 is a ranking, not a probability; brier_score_loss would reject it anyway
    label = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    result = evaluate(label, [15, 0, 3, 0, 0, 0, 0, 0, 0, 0], probability=False)
    assert "brier" not in result
    assert result["roc_auc"] == pytest.approx(1.0)
