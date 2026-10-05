"""Salted-hash assignment to the coupon experiment's arms, from Andy's P4 notebook (2026-10-04).

Decided 2026-10-04 (experiment_design.md section 2): bucket = int(sha256("<salt>:<user_id>"),
16) % 100, and buckets 0-49 are treatment. A user's arm depends only on their own user_id and
the salt, not on row order or on the other users; the same salt always gives the same split,
and a new salt gives a new, independent one (the A/A simulation re-splits by changing it).

Python's built-in hash() is not used: string hashing is seeded per process, so the same
string gave buckets 64, 85 and 48 in three runs. sha256 gives the same bucket in any process
(Andy's notebook and a separate run both put user 101 in bucket 63 under "p4-coupon").
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

# Written down before its split was computed; never changed afterwards, as a new salt is a new
# split. "p4-coupon" was looked at in a demonstration (50.112% treated) and is not used.
SALT = "p4-coupon-2026-10"
N_BUCKETS = 100
TREATED_BUCKETS = 50  # buckets 0-49: a 50/50 split


def bucket(user_id: int, salt: str) -> int:
    """0-99, from the user's id and the salt only."""
    text = f"{salt}:{user_id}"
    h = hashlib.sha256(text.encode()).hexdigest()
    return int(h, 16) % N_BUCKETS


def assign(user_ids: pd.Series, salt: str = SALT) -> pd.DataFrame:
    """One row per user, in input order: user_id, bucket, and arm "T" or "C"."""
    buckets = user_ids.apply(lambda u: bucket(u, salt))
    return pd.DataFrame(
        {
            "user_id": user_ids.to_numpy(),
            "bucket": buckets.to_numpy(),
            "arm": np.where(buckets < TREATED_BUCKETS, "T", "C"),
        }
    )
