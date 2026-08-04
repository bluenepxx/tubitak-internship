"""Regression evaluation, diagnostics, ranking, and plotting helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score


def regression_metrics(y_true, y_pred, prefix: str = "", target_std: float | None = None) -> dict[str, float]:
    """Compute regression metrics; classification metrics are intentionally absent."""

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    rmse = math.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
    metrics = {
        f"{prefix}mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        f"{prefix}rmse": float(rmse),
        f"{prefix}r2": float(r2_score(y_true_arr, y_pred_arr)),
        f"{prefix}median_absolute_error": float(median_absolute_error(y_true_arr, y_pred_arr)),
    }
    if target_std and target_std > 0:
        metrics[f"{prefix}normalized_rmse"] = float(rmse / target_std)
    return metrics


def generalization_gaps(train_metrics: dict[str, float], validation_metrics: dict[str, float]) -> dict[str, float]:
    """Compute train-validation gaps required for overfit diagnostics."""

    return {
        "r2_gap": train_metrics.get("train_r2", np.nan) - validation_metrics.get("validation_r2", np.nan),
        "rmse_gap": validation_metrics.get("validation_rmse", np.nan) - train_metrics.get("train_rmse", np.nan),
    }


def aggregate_fold_results(fold_records: Iterable[dict[str, object]]) -> dict[str, object]:
    """Aggregate k-fold records into mean/std fields."""

    records = list(fold_records)
    frame = pd.DataFrame(records)
    base = {key: records[0][key] for key in ("target", "model", "stage") if key in records[0]} if records else {}
    for column in [c for c in frame.columns if c.endswith(("_mae", "_rmse", "_r2", "_median_absolute_error"))]:
        base[f"{column}_mean"] = float(frame[column].mean())
        base[f"{column}_std"] = float(frame[column].std(ddof=1)) if len(frame) > 1 else 0.0
    base["fold_count"] = int(len(frame))
    return base


def rank_results(results: pd.DataFrame) -> pd.DataFrame:
    """Rank models without using test metrics."""

    if results.empty:
        return results
    sort_columns = [
        "cv_validation_rmse_mean",
        "cv_validation_r2_mean",
        "validation_rmse",
        "cv_validation_rmse_std",
    ]
    for column in sort_columns:
        if column not in results.columns:
            results[column] = np.nan
    ranked = results.sort_values(
        by=sort_columns,
        ascending=[True, False, True, True],
        na_position="last",
    ).copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def save_parity_plot(y_true, y_pred, path: str | Path, title: str) -> None:
    """Save true-vs-predicted parity plot."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    lo = float(min(y_true_arr.min(), y_pred_arr.min()))
    hi = float(max(y_true_arr.max(), y_pred_arr.max()))
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true_arr, y_pred_arr, s=14, alpha=0.7)
    plt.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def save_residual_plot(y_true, y_pred, path: str | Path, title: str) -> None:
    """Save predicted-vs-residual plot."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    residuals = y_true_arr - y_pred_arr
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred_arr, residuals, s=14, alpha=0.7)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def save_learning_curve_plot(curve: pd.DataFrame, path: str | Path, title: str) -> None:
    """Save train/validation RMSE learning curve."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(curve["train_fraction"], curve["train_rmse"], marker="o", label="Train RMSE")
    plt.plot(curve["train_fraction"], curve["validation_rmse"], marker="o", label="Validation RMSE")
    plt.xlabel("Training Fraction")
    plt.ylabel("RMSE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def save_convergence_plot(history: pd.DataFrame, path: str | Path, title: str) -> None:
    """Save convergence/loss history for iterative estimators and neural models."""

    if history.empty:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    for column in [c for c in history.columns if c.endswith("loss")]:
        plt.plot(history.index + 1, history[column], marker="o", label=column)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()
