import json

import numpy as np
import pandas as pd
import pytest

from pte.monitoring.psi import (
    bin_shares,
    build_reference,
    check_floor,
    load_day,
    make_cuts,
    psi,
    psi_table,
)

# the notebook's toy columns
TOY_REF = np.array([0, 0, 0, 0, 0, 0, 1, 1, 2, 5])
TOY_TODAY = np.array([0, 0, 0, 1, 1, 1, 2, 3, 4, 6])


def test_make_cuts_drops_repeated_cuts():
    # quartile cuts of TOY_REF are 0, 0, 1
    assert make_cuts(TOY_REF, n_bins=4).tolist() == [0, 1]
    assert len(make_cuts(np.arange(100), n_bins=10)) == 9


def test_bin_shares_puts_a_value_on_a_cut_in_the_left_bin():
    cuts = np.array([0, 1])
    assert bin_shares(TOY_REF, cuts) == pytest.approx([0.6, 0.2, 0.2])
    assert bin_shares(TOY_TODAY, cuts) == pytest.approx([0.3, 0.3, 0.4])


def test_bin_shares_gives_an_empty_bin_a_zero():
    assert bin_shares(np.array([5, 6]), np.array([0, 1])).tolist() == [0, 0, 1]


def test_psi_matches_the_worked_example():
    # four bins at 0.25 against 0.40 / 0.30 / 0.20 / 0.10 (pipeline.md section 5)
    assert psi([0.25] * 4, [0.40, 0.30, 0.20, 0.10]) == pytest.approx(0.2282, abs=1e-4)
    assert psi([0.6, 0.2, 0.2], [0.6, 0.2, 0.2]) == 0


def test_psi_floors_an_empty_bin():
    # without the floor ln(0) makes it inf; floored, 1e-4 -> 4.6043 and 1e-6 -> 6.9077
    assert psi([0.5, 0.5], [1.0, 0.0], floor=1e-4) == pytest.approx(4.6043, abs=1e-4)
    assert psi([0.5, 0.5], [1.0, 0.0]) == pytest.approx(6.9077, abs=1e-4)


def test_check_floor_raises_when_an_emptied_bin_stays_under_the_threshold():
    # measured 2026-10-12: e = 0.04 at floor 1e-4 adds 0.239, a silent miss
    with pytest.raises(ValueError, match="0.239"):
        check_floor({"x": {"expected": [0.04, 0.96]}}, floor=1e-4)
    # Valid's smallest bin, 0.0455 at floor 1e-6, adds 0.488
    assert check_floor({"x": {"expected": [0.0455, 0.9545]}}) == pytest.approx(0.488, abs=1e-3)


def test_check_floor_skips_an_empty_reference_bin():
    # min() over all bins would take the 0, get inf and pass without checking the 0.04
    with pytest.raises(ValueError, match="0.0400"):
        check_floor({"x": {"expected": [0.0, 0.04, 0.96]}}, floor=1e-4)


def test_build_reference_and_psi_table_on_toy_frames():
    valid = pd.DataFrame({"x": TOY_REF, "y": np.arange(10)})
    reference = build_reference(valid, variables=["x", "y"])
    assert list(reference) == ["x", "y"]
    # deciles of TOY_REF interpolate between its integers: 0, 0.4, 1, 1.2, 2.3 after dedup,
    # so two of its six bins hold nobody
    assert reference["x"]["cuts"] == pytest.approx([0, 0.4, 1, 1.2, 2.3])
    assert reference["x"]["expected"] == pytest.approx([0.6, 0, 0.2, 0, 0.1, 0.1])
    assert len(reference["y"]["expected"]) == 10
    json.dumps(reference)  # lists, not numpy arrays

    table = psi_table(valid, reference)
    assert table.columns.tolist() == ["variable", "psi"]
    assert table["variable"].tolist() == ["x", "y"]
    assert table["psi"].tolist() == [0, 0]  # the reference against itself


def test_psi_table_cuts_the_day_on_the_frozen_cuts():
    # TOY_TODAY recut on its own deciles would have equal-ish bins; on TOY_REF's it moves right
    reference = {"x": {"cuts": [0.0, 1.0], "expected": [0.6, 0.2, 0.2]}}
    table = psi_table(pd.DataFrame({"x": TOY_TODAY}), reference)
    assert table["psi"].iloc[0] == pytest.approx(psi([0.6, 0.2, 0.2], [0.3, 0.3, 0.4]))


def test_load_day_joins_scores_and_features_on_user_id(tmp_path):
    pd.DataFrame({"user_id": [2, 1], "p": [0.2, 0.1], "decile": [1, 2]}).to_parquet(
        tmp_path / "scores.parquet"
    )
    pd.DataFrame({"user_id": [1, 2], "x": [10, 20]}).to_parquet(tmp_path / "features.parquet")
    day = load_day(tmp_path / "scores.parquet", tmp_path / "features.parquet", ["p", "x"])
    assert day.columns.tolist() == ["user_id", "p", "x"]
    assert day.set_index("user_id").loc[2].tolist() == [0.2, 20]


def test_load_day_refuses_files_with_different_users(tmp_path):
    # an inner join would quietly keep users 1 and 2 and drop 3
    pd.DataFrame({"user_id": [1, 2, 3], "p": [0.1, 0.2, 0.3]}).to_parquet(tmp_path / "s.parquet")
    pd.DataFrame({"user_id": [1, 2], "x": [10, 20]}).to_parquet(tmp_path / "f.parquet")
    with pytest.raises(ValueError, match="3 scores and 2 feature rows share 2 users"):
        load_day(tmp_path / "s.parquet", tmp_path / "f.parquet", ["p", "x"])
