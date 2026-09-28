"""Train the P3 baselines and LightGBM, and print their validation metrics side by side.

Run under WSL from the repository root, after scripts/build_features.py:
    PTE_ENV=wsl python scripts/compare_models.py

Everything is fit on train and scored on valid; the test set stays untouched until the
models are frozen. LightGBM also uses valid for early stopping, so its valid scores are
slightly optimistic. On the data of 2026-09-28 (Andy's notebook reproduced exactly):

| model                  | ROC-AUC |     AP | lift@10% | lift@20% |   Brier |
| ---------------------- | ------: | -----: | -------: | -------: | ------: |
| RFM                    |  0.6698 | 0.1854 |   4.1249 |   2.3889 |         |
| logistic, 4 features   |  0.7438 | 0.2070 |   4.1454 |   2.7323 | 0.04381 |
| logistic, 23 features  |  0.7444 | 0.2283 |   4.1083 |   2.8069 | 0.04314 |
| LightGBM, 4 features   |  0.7446 | 0.2273 |   4.1207 |   2.7314 | 0.04306 |
| LightGBM, 23 features  |  0.7654 | 0.2524 |   4.2820 |   2.8798 | 0.04236 |

Then the 23-feature LightGBM is calibrated on valid (2026-09-29, leakage.md 3.4). Each half
of valid is scored by a calibrator fit on the other half: Brier 0.042358 uncalibrated,
0.042321 isotonic, 0.042317 Platt. The final Platt, fit on all of valid, has w = 1.0059 and
c = -0.0970 and brings the mean prediction from 5.4939% to the actual 5.0003%.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from sklearn.metrics import brier_score_loss

from pte.config import load_config, resolve
from pte.model.calibration import calibrate, cross_fit, fit_isotonic, fit_platt, random_halves
from pte.model.data import load_split, model_features
from pte.model.lgbm import train_lgbm
from pte.model.logreg import build_logreg
from pte.model.metrics import evaluate
from pte.model.rfm import fit_spend_bins, rfm_score

# The four columns RFM itself reads, as the models' small feature set.
RFM_FEATURES = ["n_events", "n_purchases", "days_since_last_purchase", "total_spend"]
METRICS = ["roc_auc", "ap", "lift@10%", "lift@20%", "brier"]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", help="configs/<env>.yaml (default $PTE_ENV, then dev)")
    args = parser.parse_args(argv)
    features_dir = str(resolve(load_config(args.env)["paths"]["features_dir"]))

    train = load_split(features_dir, "train")
    valid = load_split(features_dir, "valid")
    features = model_features(train)
    y_train, y_valid = train["label"], valid["label"]

    rows = {"RFM": evaluate(y_valid, rfm_score(valid, fit_spend_bins(train)), probability=False)}
    for name, cols in [("4 features", RFM_FEATURES), (f"{len(features)} features", features)]:
        logreg = build_logreg().fit(train[cols], y_train)
        rows[f"logistic, {name}"] = evaluate(y_valid, logreg.predict_proba(valid[cols])[:, 1])
        lgbm = train_lgbm(train[cols], y_train, valid[cols], y_valid)
        print(f"LightGBM, {name}: stopped at {lgbm.best_iteration_} trees")
        p_valid = lgbm.predict_proba(valid[cols])[:, 1]
        rows[f"LightGBM, {name}"] = evaluate(y_valid, p_valid)

    print(f"\n| model | {' | '.join(METRICS)} |")
    print(f"| --- |{' ---: |' * len(METRICS)}")
    for name, result in rows.items():
        cells = [f"{result[m]:.5f}" if m == "brier" else f"{result[m]:.4f}" for m in result]
        cells += [""] * (len(METRICS) - len(cells))
        print(f"| {name} | {' | '.join(cells)} |")

    # p_valid is left from the last loop pass: LightGBM on all the features
    half = random_halves(len(valid))
    print(f"\nCalibrating LightGBM, {len(features)} features (Brier, cross-fit on valid halves)")
    print(f"| none | {brier_score_loss(y_valid, p_valid):.6f} |")
    for method, fit in [("isotonic", fit_isotonic), ("Platt", fit_platt)]:
        q_oos = cross_fit(p_valid, y_valid, half, fit)
        print(f"| {method} | {brier_score_loss(y_valid, q_oos):.6f} |")
    platt = fit_platt(p_valid, y_valid)
    q_valid = calibrate(platt, p_valid)
    print(
        f"Final Platt on all of valid: w = {platt.coef_[0][0]:.4f}, c = {platt.intercept_[0]:.4f}; "
        f"mean {p_valid.mean():.4%} -> {q_valid.mean():.4%}, actual {y_valid.mean():.4%}"
    )


if __name__ == "__main__":
    main()
