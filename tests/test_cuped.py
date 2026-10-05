import pandas as pd
import pytest

from pte.experiment.cuped import correlations_by_decile, covariate_correlations

# The toy table from the 2026-10-04 session, plus a column z that is constant in decile 2.
TOY = pd.DataFrame(
    {
        "decile": [1, 1, 1, 2, 2, 2],
        "x": [3, 0, 2, 1, 0, 0],
        "p": [0.30, 0.20, 0.25, 0.05, 0.02, 0.04],
        "z": [1, 0, 1, 0, 0, 0],
        "label": [1, 0, 1, 0, 0, 1],
    }
)


def test_covariate_correlations_match_corr_and_square_it():
    table = covariate_correlations(TOY, ["x", "p"])
    assert table["candidate"].tolist() == ["x", "p"]
    expected = [TOY["x"].corr(TOY["label"]), TOY["p"].corr(TOY["label"])]
    assert table["rho"].tolist() == pytest.approx(expected)
    assert table["rho2"].tolist() == pytest.approx([r**2 for r in expected])


def test_correlations_by_decile_match_the_hand_worked_toy():
    table = correlations_by_decile(TOY, ["x", "p", "z"])
    assert table.index.tolist() == [1, 2]
    assert table["x"].tolist() == pytest.approx([0.9449, -0.5], abs=1e-4)
    assert table["p"].tolist() == pytest.approx([0.8660, 0.1890], abs=1e-4)
    # within decile 1 z equals the label; in decile 2 it is constant, so its SD is 0
    assert table.loc[1, "z"] == pytest.approx(1.0)
    assert pd.isna(table.loc[2, "z"])
