"""SHAP analysis for the three winning models (density/pld/lcd).

Loads the already-fitted final pipelines from results/models/ (trained on
train+validation by src.train.run_final_test) and explains them with
shap.TreeExplainer on a deterministic sample drawn from the validation split
(never the test split, to keep the test set reserved for the single final
evaluation). Produces per-target bar/beeswarm/waterfall plots, a SHAP
feature-group summary table, and a permutation-based feature-group importance
table computed on the same sample for direct comparison.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import FEATURE_COLUMNS, load_dataset_bundle, select_columns, select_target  # noqa: E402

SHAP_DIR = ROOT / "results" / "figures" / "shap"
TABLES = ROOT / "results" / "tables"
SAMPLE_SIZE = 500
BACKGROUND_SIZE = 100
RANDOM_STATE = 42
N_PERM_REPEATS = 10

WINNERS = {
    "density": "lgbm_regressor",
    "pld": "extra_trees",
    "lcd": "extra_trees",
}

# lightgbm trees are shallow (num_leaves=31) so the fast path-dependent TreeSHAP
# algorithm is exact and cheap. The winning ExtraTreesRegressor models grow
# extremely deep, unbalanced trees (min_samples_leaf=2, no max_depth -> depths
# of 100-140 observed) which makes the path-dependent algorithm numerically
# unstable (additivity check fails by ~1e19 in testing). Interventional
# TreeSHAP with an explicit background sample is exact-ish and stable there.
SHAP_METHOD = {
    "lgbm_regressor": "tree_path_dependent",
    "extra_trees": "interventional",
}


def load_config() -> dict:
    experiment_config = yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text(encoding="utf-8"))
    model_config = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    experiment_config["models"] = model_config.get("models", {})
    return experiment_config


def feature_group(name: str) -> str:
    prefix = name.split("__", 1)[0]
    return {
        "node_tfidf": "node_selfies",
        "linker_tfidf": "linker_selfies",
        "point_group": "point_group",
        "topology": "topology",
        "topology_missing": "topology_missing",
    }.get(prefix, prefix)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main() -> None:
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()
    bundle = load_dataset_bundle(config)

    group_rows = []
    perm_rows = []

    for target, model_name in WINNERS.items():
        t0 = time.perf_counter()
        pipeline = joblib.load(ROOT / "results" / "models" / f"{target}_{model_name}.joblib")
        preprocessor = pipeline.named_steps["preprocessor"]
        to_dense = pipeline.named_steps.get("to_dense")
        estimator = pipeline.named_steps["model"]

        pool_size = min(SAMPLE_SIZE + BACKGROUND_SIZE, len(bundle.validation))
        pool = bundle.validation.sample(n=pool_size, random_state=RANDOM_STATE)
        X_pool_raw = select_columns(pool, list(FEATURE_COLUMNS))
        Xt_pool = preprocessor.transform(X_pool_raw)
        if to_dense is not None:
            Xt_pool = to_dense.transform(Xt_pool)
        Xt_pool = np.asarray(Xt_pool)
        feature_names = list(preprocessor.get_feature_names_out())

        method = SHAP_METHOD[model_name]
        if method == "interventional":
            background_n = min(BACKGROUND_SIZE, pool_size - 1)
            Xt_background = Xt_pool[:background_n]
            Xt = Xt_pool[background_n:]
            y_sample = select_target(pool.iloc[background_n:], target).to_numpy(dtype=float)
            explainer = shap.TreeExplainer(
                estimator, data=Xt_background, feature_perturbation="interventional", model_output="raw"
            )
            shap_values = explainer.shap_values(Xt, check_additivity=False)
        else:
            Xt = Xt_pool
            y_sample = select_target(pool, target).to_numpy(dtype=float)
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(Xt)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = np.asarray(base_value).ravel()[0]

        recon_err = float(np.max(np.abs((base_value + shap_values.sum(axis=1)) - estimator.predict(Xt))))
        print(f"[{target}/{model_name}] method={method} max additivity error={recon_err:.3e}", flush=True)

        explanation = shap.Explanation(
            values=shap_values,
            base_values=np.full(shap_values.shape[0], base_value),
            data=Xt,
            feature_names=feature_names,
        )

        print(f"[{target}/{model_name}] SHAP computed on {Xt.shape[0]} rows x {Xt.shape[1]} features "
              f"in {time.perf_counter() - t0:.1f}s", flush=True)

        # bar plot: top 20 mean |SHAP|
        plt.figure()
        shap.plots.bar(explanation, max_display=20, show=False)
        plt.title(f"{target} ({model_name}) - top 20 SHAP features")
        plt.tight_layout()
        plt.savefig(SHAP_DIR / f"{target}_{model_name}_bar.png", dpi=160, bbox_inches="tight")
        plt.close()

        # beeswarm plot
        plt.figure()
        shap.plots.beeswarm(explanation, max_display=20, show=False)
        plt.title(f"{target} ({model_name}) - SHAP beeswarm")
        plt.tight_layout()
        plt.savefig(SHAP_DIR / f"{target}_{model_name}_beeswarm.png", dpi=160, bbox_inches="tight")
        plt.close()

        # waterfall plots for 3 illustrative predictions (low / median / high)
        preds = estimator.predict(Xt)
        order = np.argsort(preds)
        picks = {
            "low": int(order[0]),
            "median": int(order[len(order) // 2]),
            "high": int(order[-1]),
        }
        for label, idx in picks.items():
            plt.figure()
            shap.plots.waterfall(explanation[idx], max_display=15, show=False)
            plt.title(f"{target} ({model_name}) - waterfall ({label} prediction)")
            plt.tight_layout()
            plt.savefig(SHAP_DIR / f"{target}_{model_name}_waterfall_{label}.png", dpi=160, bbox_inches="tight")
            plt.close()

        # SHAP group summary
        groups = pd.Series([feature_group(n) for n in feature_names])
        abs_shap = np.abs(shap_values)
        mean_abs_per_feature = abs_shap.mean(axis=0)
        total_mean_abs = mean_abs_per_feature.sum()
        for group_name in sorted(groups.unique()):
            mask = (groups == group_name).to_numpy()
            group_mean_abs = float(mean_abs_per_feature[mask].sum())
            group_rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "feature_group": group_name,
                    "n_features_in_group": int(mask.sum()),
                    "sum_mean_abs_shap": group_mean_abs,
                    "pct_of_total_abs_shap": group_mean_abs / total_mean_abs * 100.0 if total_mean_abs > 0 else 0.0,
                }
            )

        # permutation-based group importance, as a comparison baseline (same sample)
        rng = np.random.default_rng(RANDOM_STATE)
        baseline_pred = estimator.predict(Xt)
        baseline_rmse = rmse(y_sample, baseline_pred)
        for group_name in sorted(groups.unique()):
            mask = (groups == group_name).to_numpy()
            degradations = []
            for _ in range(N_PERM_REPEATS):
                Xt_perm = Xt.copy()
                perm_idx = rng.permutation(Xt_perm.shape[0])
                Xt_perm[:, mask] = Xt_perm[perm_idx][:, mask]
                perm_pred = estimator.predict(Xt_perm)
                degradations.append(rmse(y_sample, perm_pred) - baseline_rmse)
            perm_rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "feature_group": group_name,
                    "n_features_in_group": int(mask.sum()),
                    "baseline_rmse": baseline_rmse,
                    "mean_rmse_increase": float(np.mean(degradations)),
                    "std_rmse_increase": float(np.std(degradations, ddof=1)),
                }
            )

    group_summary = pd.DataFrame(group_rows)
    group_summary.to_csv(TABLES / "shap_group_summary.csv", index=False)

    perm_summary = pd.DataFrame(perm_rows)
    perm_summary["pct_of_total_rmse_increase"] = perm_summary.groupby(["target", "model"])["mean_rmse_increase"].transform(
        lambda s: s / s.sum() * 100.0 if s.sum() > 0 else 0.0
    )
    perm_summary.to_csv(TABLES / "permutation_group_importance.csv", index=False)

    print()
    print(group_summary.to_string(index=False))
    print()
    print(perm_summary.to_string(index=False))


if __name__ == "__main__":
    main()
