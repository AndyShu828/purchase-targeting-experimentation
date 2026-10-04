import pandas as pd
import pytest

from pte.experiment.assignment import assign, bucket


def test_bucket_matches_the_worked_values():
    # worked on 2026-10-04 in Andy's notebook and in a separate process
    assert bucket(101, "p4-coupon") == 63
    assert bucket(102, "p4-coupon") == 10
    assert [bucket(u, "toy-salt") for u in [101, 102, 103, 104]] == [19, 8, 2, 99]


def test_assign_puts_buckets_below_50_in_treatment():
    table = assign(pd.Series([101, 102, 103, 104]), salt="toy-salt")
    assert table["user_id"].tolist() == [101, 102, 103, 104]
    assert table["bucket"].tolist() == [19, 8, 2, 99]
    assert table["arm"].tolist() == ["T", "T", "T", "C"]
    # the boundary: users 35 and 145 sit in buckets 49 and 50 under "toy-salt"
    edge = assign(pd.Series([35, 145]), salt="toy-salt")
    assert edge["bucket"].tolist() == [49, 50]
    assert edge["arm"].tolist() == ["T", "C"]


def test_a_users_arm_does_not_depend_on_order_or_on_other_users():
    ids = pd.Series(range(1000, 1200))
    full = assign(ids, salt="toy-salt").set_index("user_id")["arm"]
    shuffled = assign(ids.sample(frac=1, random_state=0), salt="toy-salt")
    assert (shuffled.set_index("user_id")["arm"] == full.loc[shuffled["user_id"]]).all()
    alone = assign(pd.Series([1100]), salt="toy-salt")["arm"].item()
    assert alone == full.loc[1100]


def test_a_new_salt_is_a_new_independent_split():
    ids = pd.Series(range(100_000))
    a = assign(ids, salt="toy-salt")["arm"]
    b = assign(ids, salt="toy-salt-2")["arm"]
    # 50/50 each; independent splits disagree for half the users. SE of each share is about
    # 0.0016 at 100,000 users, so 0.01 is over 6 SE.
    assert (a == "T").mean() == pytest.approx(0.5, abs=0.01)
    assert (a != b).mean() == pytest.approx(0.5, abs=0.01)
