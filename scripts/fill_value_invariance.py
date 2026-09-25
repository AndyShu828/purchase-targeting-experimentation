"""Why a fill value must be frozen: the numbers behind docs/leakage.md section 3.x.

Run under WSL from the repository root, after scripts/build_features.py:
    PTE_ENV=wsl python scripts/fill_value_invariance.py

Fits logit(p) = b0 + b1*X + b2*I (+ b3*active_days) on the training set, where X is
days_since_last_purchase with NULL filled by a constant c and I = 1 for NULL. The fit is
an unregularized logistic regression solved exactly by Newton's method on grouped cells
(one per distinct X, active_days), so it needs no numpy. It prints:
- the coefficients for c = 6 (the training median) and c = 2 (the test median): b0 and
  b1 do not depend on c, and b2 moves by exactly 4 * b1
- the training-fit model scored on the test set with c frozen at 6 and with c = 2: mean
  prediction against the actual rate, for the NULL group and overall, and AUC

A toy model to check the derivation, not the P3 model.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence

import duckdb

from pte.config import load_config, resolve

NULL_X = -1

# One row per (X, active_days) cell: users and positives.
CELLS_SQL = """
SELECT coalesce(f.days_since_last_purchase, {null_x}) AS x, f.active_days AS a,
    count(*) AS n, sum(l.label) AS y
FROM read_parquet('{split_dir}/features/*.parquet') AS f
JOIN read_parquet('{split_dir}/labels/*.parquet') AS l USING (user_id)
GROUP BY ALL
ORDER BY ALL
"""

Cell = tuple[int, int, int, int]


def load_cells(con: duckdb.DuckDBPyConnection, split_dir: str) -> list[Cell]:
    return con.execute(CELLS_SQL.format(null_x=NULL_X, split_dir=split_dir)).fetchall()


def row(x: int, a: int, fill: float, with_active_days: bool) -> list[float]:
    missing = x == NULL_X
    values = [1.0, float(fill if missing else x), 1.0 if missing else 0.0]
    return values + [float(a)] if with_active_days else values


def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting; the systems here are at most 4 x 4."""
    k = len(vector)
    m = [r[:] + [v] for r, v in zip(matrix, vector, strict=True)]
    for c in range(k):
        pivot = max(range(c, k), key=lambda r: abs(m[r][c]))
        m[c], m[pivot] = m[pivot], m[c]
        for r in range(k):
            if r != c:
                factor = m[r][c] / m[c][c]
                m[r] = [m[r][j] - factor * m[c][j] for j in range(k + 1)]
    return [m[i][k] / m[i][i] for i in range(k)]


def fit(cells: list[Cell], fill: float, with_active_days: bool) -> list[float]:
    """Maximum likelihood by Newton's method: gradient sum (y - n p) f, Hessian n p (1-p) f f'."""
    k = 4 if with_active_days else 3
    b = [0.0] * k
    for _ in range(50):
        gradient = [0.0] * k
        hessian = [[0.0] * k for _ in range(k)]
        for x, a, n, y in cells:
            f = row(x, a, fill, with_active_days)
            p = sigmoid(sum(bi * fi for bi, fi in zip(b, f, strict=True)))
            for i in range(k):
                gradient[i] += (y - n * p) * f[i]
                for j in range(k):
                    hessian[i][j] += n * p * (1 - p) * f[i] * f[j]
        step = solve(hessian, gradient)
        b = [bi + si for bi, si in zip(b, step, strict=True)]
        if max(abs(s) for s in step) < 1e-12:
            return b
    raise RuntimeError("Newton's method did not converge")


def evaluate(cells: list[Cell], b: list[float], fill: float, with_active_days: bool) -> str:
    scored = [
        (sum(bi * fi for bi, fi in zip(b, row(x, a, fill, with_active_days), strict=True)), x, n, y)
        for x, a, n, y in cells
    ]

    def rates(rows: list[tuple[float, int, int, int]]) -> tuple[float, float]:
        users = sum(n for _, _, n, _ in rows)
        predicted = sum(n * sigmoid(s) for s, _, n, _ in rows) / users
        return predicted, sum(y for _, _, _, y in rows) / users

    null_pred, null_actual = rates([r for r in scored if r[1] == NULL_X])
    all_pred, all_actual = rates(scored)

    # AUC over cells: each positive scores the negatives below it, plus half the tied ones.
    scored.sort()
    positives = sum(y for _, _, _, y in scored)
    negatives = sum(n - y for _, _, n, y in scored)
    wins, below, i = 0.0, 0, 0
    while i < len(scored):
        j, pos, neg = i, 0, 0
        while j < len(scored) and abs(scored[j][0] - scored[i][0]) < 1e-12:
            pos += scored[j][3]
            neg += scored[j][2] - scored[j][3]
            j += 1
        wins += pos * (below + neg / 2)
        below += neg
        i = j
    auc = wins / (positives * negatives)

    return (
        f"NULL group predicted {null_pred:.4%} actual {null_actual:.4%} | "
        f"all predicted {all_pred:.4%} actual {all_actual:.4%} | AUC {auc:.4f}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--env", default=None, help="config name under configs/")
    args = parser.parse_args(argv)

    features_dir = resolve(load_config(args.env)["paths"]["features_dir"])
    con = duckdb.connect()
    train = load_cells(con, (features_dir / "train").as_posix())
    test = load_cells(con, (features_dir / "test").as_posix())

    for with_active_days in (False, True):
        name = "M2 (X, I, active_days)" if with_active_days else "M1 (X, I)"
        print(f"=== {name}")
        b6 = fit(train, 6, with_active_days)
        b2 = fit(train, 2, with_active_days)
        print("fill 6:", [round(v, 6) for v in b6])
        print("fill 2:", [round(v, 6) for v in b2])
        print(f"b2(fill 2) - b2(fill 6) = {b2[2] - b6[2]:.6f}, 4 * b1 = {4 * b6[1]:.6f}")
        print("train, fill 6:       ", evaluate(train, b6, 6, with_active_days))
        print("test, frozen fill 6: ", evaluate(test, b6, 6, with_active_days))
        print("test, fill 2:        ", evaluate(test, b6, 2, with_active_days))


if __name__ == "__main__":
    main()
