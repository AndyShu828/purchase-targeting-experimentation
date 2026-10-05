import pandas as pd
import pytest

from pte.experiment.power import mde, mde_table


def test_mde_matches_the_hand_worked_exercises():
    # the 2026-10-01 exercises, worked by hand with 2.80; norm.ppf gives 2.80159
    assert mde(0.05, 10_000, 10_000) == pytest.approx(0.00863, abs=1e-5)
    assert mde(0.15, 115_000, 115_000) == pytest.approx(0.00417, abs=1e-5)


def test_mde_takes_one_array_for_every_decile():
    assert mde([0.05, 0.15], 115_000, 115_000).tolist() == pytest.approx(
        [float(mde(0.05, 115_000, 115_000)), float(mde(0.15, 115_000, 115_000))]
    )


def test_mde_table_counts_rates_and_the_all_row():
    # deciles of unequal size, so the all row's rate (3/5) differs from the mean of the
    # decile rates (7/12)
    valid = pd.DataFrame({"decile": [1, 1, 1, 2, 2], "label": [1, 1, 0, 1, 0]})
    test = pd.DataFrame({"decile": [1, 1, 2, 2, 2, 2]})
    table = mde_table(valid, test)
    assert table["decile"].tolist() == [1, 2, "all"]
    assert table["n_valid"].tolist() == [3, 2, 5]
    assert table["r_d"].tolist() == pytest.approx([2 / 3, 1 / 2, 3 / 5])
    assert table["n_test"].tolist() == [2, 4, 6]
    assert table["n_arm"].tolist() == [1, 2, 3]
    assert table["mde"].tolist() == pytest.approx(
        [float(mde(2 / 3, 1, 1)), float(mde(1 / 2, 2, 2)), float(mde(3 / 5, 3, 3))]
    )
    assert table["mde_pp"].tolist() == pytest.approx((table["mde"] * 100).tolist())
    assert table["mde_rel"].tolist() == pytest.approx((table["mde"] / table["r_d"]).tolist())
