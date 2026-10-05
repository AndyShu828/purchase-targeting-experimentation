"""Injecting the simulated coupon effect into the test set, from Andy's P4 notebook (2026-10-05).

Decided 2026-10-03 and 2026-10-05 (experiment_design.md section 5). Y(0) is the real outcome:
y0 the label, gmv0 the spend in the label window. Y(1) is built for every user before any
split: in each decile exactly k_d = round(tau_d / 100 x n_d) users with y0 = 0, drawn without
replacement, get y1 = 1, and each is given a gmv1 drawn with replacement from the gmv0 of that
decile's buyers (y0 = 1). Everyone else keeps y1 = y0 and gmv1 = gmv0, so the effect on GMV
comes only from the new buyers. The arm is not read here.

One generator, built once from SEED; deciles in order, each drawing its users and then their
GMV. rng.choice draws by position, so both pools are sorted by user_id first, and the result
does not depend on the order of the input rows.

These are simulated effects, set by assumption, not findings about real coupons.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

# The injected conversion effect per decile, in percentage points (decided 2026-10-03)
TAU_PP = {1: 0.1, 2: 0.2, 3: 0.4, 4: 0.5, 5: 0.6, 6: 0.5, 7: 0.4, 8: 0.3, 9: 0.2, 10: 0.1}
# Written down before the injection was computed (decided 2026-10-05); never changed afterwards
SEED = 20261005


def flip_counts(users: pd.DataFrame, tau_pp: Mapping[int, float] = TAU_PP) -> pd.DataFrame:
    """One row per decile: n_d, tau_pp and k_d = round(tau_pp / 100 x n_d).

    Series.round rounds half to even, like Python's round; no decile of the test set lands
    on a half.
    """
    counts = users.groupby("decile").size()
    table = pd.DataFrame({"decile": counts.index, "n_d": counts.to_numpy()})
    missing = sorted(set(table["decile"]) - set(tau_pp))
    if missing:
        raise ValueError(f"no tau for decile(s) {missing}")
    table["tau_pp"] = table["decile"].map(tau_pp)
    table["k_d"] = (table["n_d"] * (table["tau_pp"] / 100)).round().astype(int)
    return table


def draw_injection(
    users: pd.DataFrame, tau_pp: Mapping[int, float] = TAU_PP, seed: int = SEED
) -> pd.DataFrame:
    """The flipped users and their drawn GMV: one row each, user_id and gmv_draw.

    users has one row per user with user_id, decile, y0 and gmv0.
    """
    counts = flip_counts(users, tau_pp)
    rng = np.random.default_rng(seed)
    parts = []
    for d, k in zip(counts["decile"], counts["k_d"], strict=True):
        decile = users[users["decile"] == d]
        pool = decile.loc[decile["y0"] == 0, "user_id"].sort_values().to_numpy()
        buyer_gmv = decile[decile["y0"] == 1].sort_values("user_id")["gmv0"].to_numpy()
        flipped = rng.choice(pool, size=k, replace=False)
        gmv_draw = rng.choice(buyer_gmv, size=k, replace=True)
        parts.append(pd.DataFrame({"user_id": flipped, "gmv_draw": gmv_draw}))
    return pd.concat(parts, ignore_index=True)


def inject(
    users: pd.DataFrame, tau_pp: Mapping[int, float] = TAU_PP, seed: int = SEED
) -> pd.DataFrame:
    """users with y1 and gmv1 added, rows in input order.

    The draws are joined back by user_id, never by position: a flipped user has a gmv_draw,
    everyone else NaN after the left join.
    """
    drawn = draw_injection(users, tau_pp, seed)
    out = users.merge(drawn, on="user_id", how="left", validate="one_to_one")
    flipped = out["gmv_draw"].notna()
    out["y1"] = np.where(flipped, 1, out["y0"])
    out["gmv1"] = np.where(flipped, out["gmv_draw"], out["gmv0"])
    return out.drop(columns="gmv_draw")


def injected_effects(outcomes: pd.DataFrame, tau_pp: Mapping[int, float] = TAU_PP) -> pd.DataFrame:
    """The true effects, one row per decile and an "all" row: the truth estimates are judged by.

    conv_pp = (mean y1 - mean y0) x 100, which is k_d / n_d and differs from tau_pp by at most
    half a user; gmv_effect = mean gmv1 - mean gmv0 per user, set by the amounts drawn.
    """

    def effects(g: pd.DataFrame) -> dict[str, float]:
        return {
            "n_d": len(g),
            "y0": g["y0"].mean(),
            "y1": g["y1"].mean(),
            "gmv0": g["gmv0"].mean(),
            "gmv1": g["gmv1"].mean(),
        }

    rows = [{"decile": d, **effects(g)} for d, g in outcomes.groupby("decile")]
    rows.append({"decile": "all", **effects(outcomes)})
    table = pd.DataFrame(rows)
    table["tau_pp"] = table["decile"].map(tau_pp)
    table["conv_pp"] = (table["y1"] - table["y0"]) * 100
    table["gmv_effect"] = table["gmv1"] - table["gmv0"]
    return table
