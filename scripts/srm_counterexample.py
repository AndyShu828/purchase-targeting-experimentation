"""The SRM counterexample on the official split, from Andy's P4 notebook.

Run under WSL from the repository root, after scripts/inject_effects.py:
    PTE_ENV=wsl python scripts/srm_counterexample.py

Splits the test set under the official salt, drops the users counterexample_drop picks
(treated, active on one day, bucket < 5 under "srm-drop"; experiment_design.md section 2) and
reports the SRM test and the z test on y0, over all users and within decile 1, before and after
the drop. Reads user_id, decile and y0 from <features_dir>/test/outcomes and active_days from
<features_dir>/test/features; never y1. y0's true effect is 0, so the after - before change in
diff is the bias the drop caused.

Run on 2026-10-08 (agrees with Andy's notebook to the digit, and with a separate run that hashed
the ids itself and tested with scipy's chi-square on the 2 x 2 table):

| rows   |      n_t |      n_c |  srm_p   | all_diff | all_p  | d1_diff   | d1_p   |
| :----- | -------: | -------: | -------: | -------: | -----: | --------: | -----: |
| before | 1,154,643 | 1,151,036 | 0.0175  | 0.000175 | 0.5183 | -0.000509 | 0.7425 |
| after  | 1,120,634 | 1,151,036 | 1.76e-90 | 0.000787 | 0.0041 | -0.000087 | 0.9553 |

34,009 users dropped, 884 of them in decile 1. The alarm goes off; over all users the diff
moves by +0.0612 pp, 2.3 standard errors, to p = 0.004 on a true effect of 0, while decile 1
moves by +0.0422 pp, 0.27 standard errors.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import pandas as pd

from pte.config import load_config, resolve
from pte.experiment.assignment import SALT, assign
from pte.experiment.srm import DROP_BUCKETS, DROP_SALT, counterexample_drop, srm_test
from pte.experiment.ztest import z_test


def summarize(users: pd.DataFrame) -> dict:
    """Arm counts, the SRM p, and the z test on y0 over all users and within decile 1."""
    row = {}
    srm = srm_test(int((users["arm"] == "T").sum()), int((users["arm"] == "C").sum()))
    row.update(n_t=srm["n_t"], n_c=srm["n_c"], srm_chi_sq=srm["chi_sq"], srm_p=srm["p"])
    for name, part in [("all", users), ("d1", users[users["decile"] == 1])]:
        z = z_test(part.loc[part["arm"] == "T", "y0"], part.loc[part["arm"] == "C", "y0"])
        row.update({f"{name}_{k}": v for k, v in z.items()})
    return row


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", help="configs/<env>.yaml (default $PTE_ENV, then dev)")
    args = parser.parse_args(argv)
    features_dir = str(resolve(load_config(args.env)["paths"]["features_dir"]))

    users = pd.read_parquet(f"{features_dir}/test/outcomes", columns=["user_id", "decile", "y0"])
    activity = pd.read_parquet(f"{features_dir}/test/features", columns=["user_id", "active_days"])
    users = users.merge(activity, on="user_id", how="left", validate="one_to_one")
    assert users["active_days"].notna().all()
    split = assign(users["user_id"])
    assert (split["user_id"].to_numpy() == users["user_id"].to_numpy()).all()
    users["arm"] = split["arm"].to_numpy()

    drop = counterexample_drop(users)
    kept = users[~drop]
    table = pd.DataFrame([summarize(users), summarize(kept)], index=["before", "after"])
    print(f'salt "{SALT}"; drop: arm T, active_days 1, bucket < {DROP_BUCKETS} under "{DROP_SALT}"')
    print(f"dropped {drop.sum():,}, {(drop & (users['decile'] == 1)).sum():,} in decile 1")
    print(table.T.to_string())
    bias = (table.loc["after"] - table.loc["before"]) * 100
    print(f"bias in diff: all {bias['all_diff']:+.4f} pp, decile 1 {bias['d1_diff']:+.4f} pp")


if __name__ == "__main__":
    main()
