import numpy as np
import pandas as pd
import pytest

from pte.model.rfm import fit_spend_bins, rfm_score

# Wide enough that every spend in these tests lands in M = 1 unless a test sets bins.
LOW_M_BINS = [-np.inf, 1000, 2000, 3000, 4000, np.inf]


def buyers(days_since=1, purchase_days=1, spend=1.0, n=1):
    """n buyers; any argument may be a list, one value per buyer."""
    return pd.DataFrame(
        {
            "n_purchases": 1,
            "purchase_days": purchase_days,
            "days_since_last_purchase": days_since,
            "total_spend": spend,
        },
        index=range(n),
    )


def test_r_edges():
    days = [1, 2, 3, 4, 5, 7, 8, 10, 11, 14]
    # F = 1 and M = 1 throughout, so the total is R + 2
    score = rfm_score(buyers(days_since=days, n=len(days)), LOW_M_BINS)
    assert list(score - 2) == [5, 5, 4, 4, 3, 3, 2, 2, 1, 1]


def test_f_edges():
    days = [1, 2, 3, 4, 5, 6, 14]
    # R = 5 and M = 1 throughout, so the total is F + 6
    score = rfm_score(buyers(purchase_days=days, n=len(days)), LOW_M_BINS)
    assert list(score - 6) == [1, 2, 3, 4, 4, 5, 5]


def test_spend_bins_come_from_buyers_only():
    train = pd.concat(
        [
            buyers(spend=[float(s) for s in range(1, 11)], n=10),
            # many non-buyers at 0 would drag every quantile down if they counted
            pd.DataFrame({"n_purchases": [0] * 50, "total_spend": [0.0] * 50}),
        ]
    )
    # linear interpolation over 1..10: 1 + q * 9
    assert fit_spend_bins(train) == pytest.approx([-np.inf, 2.8, 4.6, 6.4, 8.2, np.inf])


def test_m_uses_the_frozen_bins():
    """Scoring another set reuses the training bins, whatever that set's own spread."""
    bins = [-np.inf, 2.8, 4.6, 6.4, 8.2, np.inf]
    spend = [2.8, 2.81, 5.0, 8.2, 1e6]
    score = rfm_score(buyers(spend=spend, n=len(spend)), bins)
    # R = 5, F = 1; bins are right-closed, so 2.8 is still M = 1
    assert list(score - 6) == [1, 2, 3, 4, 5]


def test_non_buyers_score_zero():
    features = pd.DataFrame(
        {
            "n_purchases": [0, 0, 1],
            "purchase_days": [0, 0, 6],
            "days_since_last_purchase": [None, None, 1],
            "total_spend": [0.0, 0.0, 5000.0],
        }
    )
    assert list(rfm_score(features, LOW_M_BINS)) == [0, 0, 15]


@pytest.mark.parametrize(
    ("column", "value"),
    [("days_since_last_purchase", None), ("days_since_last_purchase", 0), ("purchase_days", 0)],
)
def test_buyer_outside_the_bins_raises(column, value):
    """A buyer's NaN must not be folded into the non-buyers' 0."""
    features = buyers(n=2)
    features[column] = features[column].astype(float)
    features.loc[1, column] = value
    with pytest.raises(ValueError, match="1 buyers fall outside"):
        rfm_score(features, LOW_M_BINS)
