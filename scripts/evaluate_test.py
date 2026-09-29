"""Evaluate the frozen P3 models on the test set, as pre-registered in docs/leakage.md 5.1.

Run under WSL from the repository root, after scripts/build_features.py and
scripts/register_model.py:
    PTE_ENV=wsl python scripts/evaluate_test.py

The test set is evaluated once. Everything reported here was fixed in leakage.md 5.1 before
any test label was read; do not add, drop or re-cut items after seeing the output. The
models are fit on train (RFM's spend bins, the logistic) or taken from the registry
(LightGBM, version MODEL_VERSION, Platt fit on valid); nothing is fit on test.

Prints, for valid and test side by side: the four models' metrics, mean prediction against
the actual rate (the 14-day prediction also scaled by R for the 13-day test labels), the
test set's own deciles, the segments and the pair split. Saves the calibration curves to
docs/figures/calibration_test.png.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from pte.config import PROJECT_ROOT, load_config, resolve
from pte.model.data import load_split, model_features
from pte.model.logreg import build_logreg
from pte.model.metrics import evaluate
from pte.model.registry import EXPERIMENT, MODEL_NAME
from pte.model.rfm import fit_spend_bins, rfm_score
from pte.model.scoring import assign_deciles
from pte.model.segments import evaluate_segments, pair_auc
from pte.model.tracking import setup_tracking

# pinned, not @champion: the pre-registration names this version
MODEL_VERSION = "1"
# 13-day / 14-day label rate on the train cutoff 10-15 (scripts/label_window_ratio.py);
# 10-22 and 10-29 gave 0.9384 and 0.9498
R = 0.9596
CURVE_BINS = 20
FIGURE = PROJECT_ROOT / "docs" / "figures" / "calibration_test.png"


def calibration_bins(p: pd.Series, y: pd.Series, n_bins: int = CURVE_BINS) -> pd.DataFrame:
    """Mean prediction and actual rate in n_bins equal-count bins of p, lowest first."""
    table = pd.DataFrame({"p": p.to_numpy(), "y": y.to_numpy()})
    table["bin"] = pd.qcut(table["p"], n_bins, labels=False, duplicates="drop")
    return table.groupby("bin").agg(mean_p=("p", "mean"), actual=("y", "mean"))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", help="configs/<env>.yaml (default $PTE_ENV, then dev)")
    args = parser.parse_args(argv)
    paths = load_config(args.env)["paths"]
    features_dir = str(resolve(paths["features_dir"]))
    setup_tracking(resolve(paths["mlflow_dir"]), EXPERIMENT)

    train = load_split(features_dir, "train")
    features = model_features(train)
    spend_bins = fit_spend_bins(train)
    logreg = build_logreg().fit(train[features], train["label"])
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")

    splits = {name: load_split(features_dir, name) for name in ["valid", "test"]}
    rows = []
    for name, df in splits.items():
        # scores go into the table first, so every later filter carries them along
        scores = model.predict(df)
        df["p_raw"] = scores["p_raw"].to_numpy()
        df["p"] = scores["p"].to_numpy()
        df["rfm"] = rfm_score(df, spend_bins)
        df["p_logreg"] = logreg.predict_proba(df[features])[:, 1]
        y = df["label"]
        for model_name, column, probability in [
            ("RFM", "rfm", False),
            ("logistic, 20 features", "p_logreg", True),
            ("LightGBM", "p_raw", True),
            ("LightGBM + Platt", "p", True),
        ]:
            rows.append(
                {"model": model_name, "split": name}
                | evaluate(y, df[column], probability=probability)
            )
    overall = (
        pd.DataFrame(rows)
        .pivot(index="model", columns="split")
        # the pre-registered row order, valid before test within each metric
        .reindex(index=list(dict.fromkeys(row["model"] for row in rows)))
        .reindex(columns=["valid", "test"], level="split")
    )
    print(f"Test cutoff 11-18, {len(splits['test']):,} users; {MODEL_NAME} version {MODEL_VERSION}")
    print("\n1-2. Metrics, valid against test")
    print(overall.round(5).to_string())

    print("\n3. Mean prediction against the actual rate")
    for name, df in splits.items():
        print(
            f"{name}: actual {df['label'].mean():.4%}; mean p_raw {df['p_raw'].mean():.4%}, "
            f"mean p {df['p'].mean():.4%}, mean p x R {df['p'].mean() * R:.4%}"
        )

    test = splits["test"]
    test["decile"] = assign_deciles(test["p"])
    deciles = test.groupby("decile").agg(
        users=("p", "size"), mean_p=("p", "mean"), actual=("label", "mean")
    )
    deciles["mean_p_x_r"] = deciles["mean_p"] * R
    print("\nTest deciles (the test set's own, decile 1 highest)")
    print(deciles.round(5).to_string())

    print("\n4. Test segments (mean_score is before calibration)")
    segments = evaluate_segments(test, test["p_raw"])
    segments.insert(3, "mean_platt", evaluate_segments(test, test["p"])["mean_score"])
    segments["rfm_roc_auc"] = evaluate_segments(test, test["rfm"], probability=False)["roc_auc"]
    print(segments.round(4).to_string())
    print("\nTest ROC-AUC by where each pair's positive and negative come from (group = bought)")
    pairs = pair_auc(test["p_raw"], test["label"], test["n_purchases"] > 0)
    print(pairs.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 6))
    low, high = 1.0, 0.0
    for column, label in [("p_raw", "LightGBM"), ("p", "LightGBM + Platt")]:
        curve = calibration_bins(test[column], test["label"])
        ax.plot(curve["mean_p"], curve["actual"], marker="o", label=label)
        low = min(low, curve["mean_p"].min(), curve["actual"].min())
        high = max(high, curve["mean_p"].max(), curve["actual"].max())
    line = [low, high]
    ax.plot(line, line, color="grey", linestyle="--", label="perfect, 14-day labels")
    ax.plot(line, [v * R for v in line], color="grey", linestyle=":", label=f"perfect, x {R}")
    # log axes: most users sit below 5%, where a linear axis crowds every bin together
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"mean prediction ({CURVE_BINS} equal-count bins, log scale)")
    ax.set_ylabel("actual purchase rate, test (13-day labels, log scale)")
    ax.set_title("Calibration on the test set (cutoff 2019-11-18)")
    ax.legend()
    Path(FIGURE).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=120, bbox_inches="tight")
    print(f"\nCalibration curves -> {FIGURE}")


if __name__ == "__main__":
    main()
