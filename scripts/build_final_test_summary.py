"""Build a report-ready final-test summary and CV-vs-test comparison table.

Reads the pipeline's canonical results/tables/final_test_results.csv (produced by
`python -m src.run_30_models --mode full --stages final_test`) and
results/tables/kfold_results.csv, and writes two small report tables plus copies
of the test-set parity plots into results/plots/. Does not train or touch the
test set itself -- it only reads existing result artifacts.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
PLOTS_OUT = ROOT / "results" / "plots"
PARITY_SRC = ROOT / "results" / "figures" / "parity_plots"

WINNERS = {
    "density": "lgbm_regressor",
    "pld": "extra_trees",
    "lcd": "extra_trees",
}


def main() -> None:
    final_test = pd.read_csv(TABLES / "final_test_results.csv")
    kfold = pd.read_csv(TABLES / "kfold_results.csv")

    final_test = final_test[final_test.apply(lambda r: WINNERS.get(r["target"]) == r["model"], axis=1)].copy()

    # Test set size is identical across targets (targets are never imputed / never missing).
    final_test["test_n"] = 2603

    summary_cols = [
        "target", "model", "test_n",
        "test_mae", "test_rmse", "test_r2", "test_median_absolute_error", "test_normalized_rmse",
        "train_validation_mae", "train_validation_rmse", "train_validation_r2",
        "training_seconds", "dataset_hash", "split_hash", "model_config_hash",
    ]
    final_test = final_test[summary_cols].sort_values("target").reset_index(drop=True)
    final_test.to_csv(TABLES / "final_test_summary.csv", index=False)

    rows = []
    for target, model in WINNERS.items():
        cv_row = kfold[(kfold["target"] == target) & (kfold["model"] == model)].iloc[0]
        test_row = final_test[(final_test["target"] == target) & (final_test["model"] == model)].iloc[0]
        cv_rmse = float(cv_row["cv_validation_rmse_mean"])
        test_rmse = float(test_row["test_rmse"])
        cv_r2 = float(cv_row["cv_validation_r2_mean"])
        test_r2 = float(test_row["test_r2"])
        rows.append(
            {
                "target": target,
                "model": model,
                "cv_rmse_mean": cv_rmse,
                "cv_rmse_std": float(cv_row["cv_validation_rmse_std"]),
                "test_rmse": test_rmse,
                "rmse_diff_test_minus_cv": test_rmse - cv_rmse,
                "rmse_diff_pct": (test_rmse - cv_rmse) / cv_rmse * 100.0,
                "cv_r2_mean": cv_r2,
                "test_r2": test_r2,
                "r2_diff_test_minus_cv": test_r2 - cv_r2,
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(TABLES / "cv_vs_test_comparison.csv", index=False)

    PLOTS_OUT.mkdir(parents=True, exist_ok=True)
    for target, model in WINNERS.items():
        src = PARITY_SRC / f"{target}_{model}_test.png"
        if src.exists():
            shutil.copy2(src, PLOTS_OUT / f"{target}_{model}_test_parity.png")

    print(final_test.to_string(index=False))
    print()
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
