"""Fit the final P3 model, LightGBM + Platt, and register it in MLflow as the champion.

Run under WSL from the repository root, on a clean checkout of main (the run records the
commit, marked -dirty otherwise), after scripts/build_features.py:
    PTE_ENV=wsl python scripts/register_model.py

LightGBM is trained on train with TUNED_PARAMS and early stopping on valid, Platt is fit on
valid (leakage.md 3.4), and both are registered together as a new version of
pte-purchase-propensity, with the alias champion moved to it. The test set is not read.
Before exiting, the script loads the champion back and checks that it scores valid exactly
as the fit in memory does.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np

from pte.config import load_config, resolve
from pte.model.calibration import calibrate, fit_platt
from pte.model.data import load_split, model_features
from pte.model.lgbm import train_lgbm
from pte.model.registry import CHAMPION, EXPERIMENT, MODEL_NAME, load_champion, register_final_model
from pte.model.tracking import setup_tracking


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", help="configs/<env>.yaml (default $PTE_ENV, then dev)")
    parser.add_argument("--mlflow-dir", help="another MLflow store, for a dry run")
    args = parser.parse_args(argv)
    paths = load_config(args.env)["paths"]
    features_dir = str(resolve(paths["features_dir"]))
    setup_tracking(args.mlflow_dir or resolve(paths["mlflow_dir"]), EXPERIMENT)

    train = load_split(features_dir, "train")
    valid = load_split(features_dir, "valid")
    features = model_features(train)
    x_valid, y_valid = valid[features], valid["label"]
    model = train_lgbm(train[features], train["label"], x_valid, y_valid)
    p_valid = model.predict_proba(x_valid)[:, 1]
    platt = fit_platt(p_valid, y_valid)
    print(
        f"LightGBM, {len(features)} features, {model.best_iteration_} trees; "
        f"Platt w = {platt.coef_[0][0]:.4f}, c = {platt.intercept_[0]:.4f}"
    )

    version = register_final_model(model, platt, x_valid, y_valid, tags={"features": "per_day"})
    print(f"Registered {MODEL_NAME} version {version.version} (run {version.run_id}) @{CHAMPION}")

    scores = load_champion().predict(valid)
    gap_raw = np.abs(scores["p_raw"].to_numpy() - p_valid).max()
    gap = np.abs(scores["p"].to_numpy() - calibrate(platt, p_valid)).max()
    print(f"Reloaded champion on valid: max gap {gap_raw:.1e} raw, {gap:.1e} calibrated")
    print(f"Mean calibrated p {scores['p'].mean():.4%}, actual {y_valid.mean():.4%}")
    if max(gap_raw, gap) > 1e-12:
        raise SystemExit("the registered model does not score like the fit it came from")


if __name__ == "__main__":
    main()
