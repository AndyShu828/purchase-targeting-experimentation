"""Explain the tuned LightGBM with SHAP on a validation sample: importances and directions.

Run under WSL from the repository root, after scripts/build_features.py:
    PTE_ENV=wsl python scripts/explain_model.py

Prints the three importances side by side (mean |SHAP|, gain, split) and, for the top
features by mean |SHAP|, the mean SHAP value by decile of the feature, which shows which
way each one pushes the prediction. The business reading of these tables is Andy's
(P3 plan, decision class 3) and goes into docs/model_report.md.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from pte.config import load_config, resolve
from pte.model.data import load_split, model_features
from pte.model.explain import SAMPLE_SIZE, dependence, importance, shap_values
from pte.model.lgbm import train_lgbm


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", help="configs/<env>.yaml (default $PTE_ENV, then dev)")
    parser.add_argument("--top", type=int, default=8, help="features to show directions for")
    args = parser.parse_args(argv)
    features_dir = str(resolve(load_config(args.env)["paths"]["features_dir"]))

    train = load_split(features_dir, "train")
    valid = load_split(features_dir, "valid")
    features = model_features(train)
    model = train_lgbm(train[features], train["label"], valid[features], valid["label"])
    x, values, base = shap_values(model, valid[features])
    print(f"LightGBM, {len(features)} features, {model.best_iteration_} trees")
    print(f"SHAP on {SAMPLE_SIZE:,} validation users; base logit {base:.4f}\n")
    print(importance(model, values).round(4).to_string())
    for name in importance(model, values).index[: args.top]:
        print(f"\n{name}")
        print(dependence(x[name], values[name]).round(4).to_string())


if __name__ == "__main__":
    main()
