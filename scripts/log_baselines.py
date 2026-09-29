"""Log the P3 baselines to MLflow: RFM and the logistic regressions, with validation metrics.

Run under WSL from the repository root, on a clean checkout of main (the runs record the
commit), after scripts/build_features.py:
    PTE_ENV=wsl python scripts/log_baselines.py

The plan asks for every run in MLflow; until 2026-09-30 only LightGBM's were. These are the
baseline rows of scripts/compare_models.py, fit on train and scored on valid, logged to the
experiment p3-baselines. Test metrics are not logged: the test set was evaluated once and
its numbers are in scripts/evaluate_test.py and docs/model_report.md.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from compare_models import RFM_FEATURES

from pte.config import load_config, resolve
from pte.model.data import load_split, model_features
from pte.model.logreg import build_logreg
from pte.model.rfm import F_BINS, M_QUANTILES, R_BINS, fit_spend_bins, rfm_score
from pte.model.tracking import BASELINE_EXPERIMENT, log_baseline_run, setup_tracking


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", help="configs/<env>.yaml (default $PTE_ENV, then dev)")
    args = parser.parse_args(argv)
    paths = load_config(args.env)["paths"]
    features_dir = str(resolve(paths["features_dir"]))
    setup_tracking(resolve(paths["mlflow_dir"]), BASELINE_EXPERIMENT)

    train = load_split(features_dir, "train")
    valid = load_split(features_dir, "valid")
    y_train, y_valid = train["label"], valid["label"]

    spend_bins = fit_spend_bins(train)
    rfm_params = {
        "r_bins": R_BINS,
        "f_bins": F_BINS,
        "m_quantiles": M_QUANTILES,
        "m_edges": [round(edge, 2) for edge in spend_bins[1:-1]],
    }
    run_id = log_baseline_run(
        "RFM",
        y_valid,
        rfm_score(valid, spend_bins),
        probability=False,
        params=rfm_params,
        tags={"model": "rfm"},
    )
    print(f"RFM -> run {run_id}")

    for name, cols in [("4 features", RFM_FEATURES), ("20 features", model_features(train))]:
        logreg = build_logreg().fit(train[cols], y_train)
        params = {"imputer": "median + missing indicators", "C": 1.0, "n_features": len(cols)}
        run_id = log_baseline_run(
            f"logistic, {name}",
            y_valid,
            logreg.predict_proba(valid[cols])[:, 1],
            params=params,
            tags={"model": "logistic", "features": "per_day"},
        )
        print(f"logistic, {name} -> run {run_id}")


if __name__ == "__main__":
    main()
