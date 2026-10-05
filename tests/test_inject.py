import numpy as np
import pandas as pd
import pytest

from pte.experiment.inject import (
    TAU_PP,
    draw_injection,
    flip_counts,
    inject,
    injected_effects,
)

TOY_TAU = {1: 20.0, 2: 10.0}  # pp; large, so a toy decile of 50 flips 10 and 5


@pytest.fixture
def users():
    """Two deciles of 50 users; ids out of order; buyers' gmv0 distinct, so draws are traceable."""
    rng = np.random.default_rng(0)
    ids = rng.permutation(np.arange(1000, 1100))
    decile = np.repeat([1, 2], 50)
    y0 = np.tile([1] * 10 + [0] * 40, 2)  # 10 buyers in each decile
    gmv0 = np.where(y0 == 1, np.arange(100) + 1.5, 0.0)
    return pd.DataFrame({"user_id": ids, "decile": decile, "y0": y0, "gmv0": gmv0})


def test_flip_counts_on_the_test_set_sizes():
    # the decile sizes of the test set; k_d as in Andy's notebook, 7,608 in all
    n = [230568, 230568, 230568, 230568, 230567, 230568, 230568, 230415, 230316, 230973]
    users = pd.DataFrame({"decile": np.repeat(np.arange(1, 11), n)})
    table = flip_counts(users)
    assert table["n_d"].tolist() == n
    assert table["k_d"].tolist() == [231, 461, 922, 1153, 1383, 1153, 922, 691, 461, 231]
    assert table["k_d"].sum() == 7608


def test_flip_counts_needs_a_tau_for_every_decile():
    with pytest.raises(ValueError, match="decile"):
        flip_counts(pd.DataFrame({"decile": [1, 11]}))


def test_draws_follow_the_preregistered_order(users):
    # by hand: one generator; decile 1 then 2; each draws users, then their GMV; pools sorted
    rng = np.random.default_rng(7)
    expected = []
    for d, k in [(1, 10), (2, 5)]:
        g = users[users["decile"] == d]
        pool = np.sort(g.loc[g["y0"] == 0, "user_id"].to_numpy())
        buyers = g[g["y0"] == 1].sort_values("user_id")["gmv0"].to_numpy()
        flipped = rng.choice(pool, size=k, replace=False)
        expected.append(pd.DataFrame({"user_id": flipped, "gmv_draw": rng.choice(buyers, k)}))
    pd.testing.assert_frame_equal(
        draw_injection(users, TOY_TAU, seed=7), pd.concat(expected, ignore_index=True)
    )


def test_inject_flips_exactly_k_non_buyers_and_draws_from_the_same_decile(users):
    out = inject(users, TOY_TAU, seed=7)
    assert out["user_id"].tolist() == users["user_id"].tolist()  # input order kept
    flipped = out["y1"] != out["y0"]
    assert out.loc[flipped].groupby("decile").size().tolist() == [10, 5]
    assert (out.loc[flipped, "y0"] == 0).all()
    assert (out.loc[flipped, "y1"] == 1).all()
    # everyone else, buyers included, keeps Y(0)
    assert (out.loc[~flipped, "gmv1"] == out.loc[~flipped, "gmv0"]).all()
    for d in (1, 2):
        buyers = set(users.loc[(users["decile"] == d) & (users["y0"] == 1), "gmv0"])
        assert set(out.loc[flipped & (out["decile"] == d), "gmv1"]) <= buyers


def test_gmv_is_drawn_with_replacement():
    # one buyer in the decile and 3 to flip: only possible with replacement
    users = pd.DataFrame(
        {"user_id": [1, 2, 3, 4], "decile": 1, "y0": [1, 0, 0, 0], "gmv0": [50.0, 0, 0, 0]}
    )
    out = inject(users, {1: 75.0}, seed=1)
    assert out["gmv1"].tolist() == [50.0, 50.0, 50.0, 50.0]


def test_row_order_does_not_change_who_is_flipped(users):
    a = inject(users, TOY_TAU, seed=7).set_index("user_id").sort_index()
    shuffled = users.sample(frac=1, random_state=3)
    b = inject(shuffled, TOY_TAU, seed=7).set_index("user_id").sort_index()
    pd.testing.assert_frame_equal(a, b)


def test_the_seed_fixes_the_draws(users):
    a = draw_injection(users, TOY_TAU, seed=7)
    pd.testing.assert_frame_equal(a, draw_injection(users, TOY_TAU, seed=7))
    assert set(a["user_id"]) != set(draw_injection(users, TOY_TAU, seed=8)["user_id"])


def test_injected_effects_are_the_flips_and_the_draws(users):
    out = inject(users, TOY_TAU, seed=7)
    drawn = draw_injection(users, TOY_TAU, seed=7).merge(users[["user_id", "decile"]])
    table = injected_effects(out, TOY_TAU)
    assert table["decile"].tolist() == [1, 2, "all"]
    assert table["n_d"].tolist() == [50, 50, 100]
    assert table["conv_pp"].tolist() == pytest.approx([20.0, 10.0, 15.0])
    sums = drawn.groupby("decile")["gmv_draw"].sum()
    assert table["gmv_effect"].tolist() == pytest.approx(
        [sums[1] / 50, sums[2] / 50, sums.sum() / 100]
    )


def test_tau_is_the_preregistered_table():
    assert TAU_PP == {
        1: 0.1,
        2: 0.2,
        3: 0.4,
        4: 0.5,
        5: 0.6,
        6: 0.5,
        7: 0.4,
        8: 0.3,
        9: 0.2,
        10: 0.1,
    }
