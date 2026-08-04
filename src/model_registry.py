"""Structured registry for the 30 requested regression algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

RANDOM_STATE = 42
MODEL_REGISTRY_VERSION = "2026-08-04-poisson-nystroem-v1"


@dataclass(frozen=True)
class ModelSpec:
    """One model entry in the reproducible benchmark catalog."""

    name: str
    family: str
    factory: Callable[[dict[str, Any] | None], Any]
    parameters: dict[str, Any] = field(default_factory=dict)
    requires_dense: bool = False
    requires_scaling: bool = False
    tree_based: bool = False
    ensemble_based: bool = False
    iterative: bool = False
    attention_based: bool = False
    supports_n_jobs: bool = False
    enabled: bool = True

    def create(self, overrides: dict[str, Any] | None = None) -> Any:
        params = dict(self.parameters)
        if overrides:
            params.update(overrides)
        return self.factory(params)


def _linear_regression(params):
    from sklearn.linear_model import LinearRegression

    return LinearRegression(**params)


def _ridge(params):
    from sklearn.linear_model import Ridge

    return Ridge(**params)


def _lasso(params):
    from sklearn.linear_model import Lasso

    return Lasso(**params)


def _elastic_net(params):
    from sklearn.linear_model import ElasticNet

    return ElasticNet(**params)


def _bayesian_ridge(params):
    from sklearn.linear_model import BayesianRidge

    return BayesianRidge(**params)


def _huber(params):
    from sklearn.linear_model import HuberRegressor

    return HuberRegressor(**params)


def _sgd(params):
    from sklearn.linear_model import SGDRegressor

    return SGDRegressor(**params)


def _passive_aggressive(params):
    from sklearn.linear_model import PassiveAggressiveRegressor

    return PassiveAggressiveRegressor(**params)


def _knn(params):
    from sklearn.neighbors import KNeighborsRegressor

    return KNeighborsRegressor(**params)


def _linear_svr(params):
    from sklearn.svm import LinearSVR

    return LinearSVR(**params)


def _svr(params):
    from sklearn.svm import SVR

    return SVR(**params)


def _nu_svr(params):
    from sklearn.svm import NuSVR

    return NuSVR(**params)


def _poisson(params):
    from sklearn.linear_model import PoissonRegressor

    return PoissonRegressor(**params)


def _nystroem_ridge(params):
    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline

    params = dict(params)
    alpha = params.pop("alpha", 1.0)
    solver = params.pop("solver", "lsqr")
    params.setdefault("kernel", "rbf")
    params.setdefault("n_components", 512)
    params.setdefault("random_state", RANDOM_STATE)
    return Pipeline(
        steps=[
            ("nystroem", Nystroem(**params)),
            ("ridge", Ridge(alpha=alpha, solver=solver)),
        ]
    )


def _pls(params):
    from sklearn.cross_decomposition import PLSRegression

    return PLSRegression(**params)


def _mlp(params):
    from sklearn.neural_network import MLPRegressor

    return MLPRegressor(**params)


def _decision_tree(params):
    from sklearn.tree import DecisionTreeRegressor

    return DecisionTreeRegressor(**params)


def _extra_tree(params):
    from sklearn.tree import ExtraTreeRegressor

    return ExtraTreeRegressor(**params)


def _random_forest(params):
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(**params)


def _extra_trees(params):
    from sklearn.ensemble import ExtraTreesRegressor

    return ExtraTreesRegressor(**params)


def _gradient_boosting(params):
    from sklearn.ensemble import GradientBoostingRegressor

    return GradientBoostingRegressor(**params)


def _hist_gradient_boosting(params):
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(**params)


def _adaboost(params):
    from sklearn.ensemble import AdaBoostRegressor

    return AdaBoostRegressor(**params)


def _bagging(params):
    from sklearn.ensemble import BaggingRegressor
    from sklearn.tree import DecisionTreeRegressor

    params = dict(params)
    if "estimator" not in params:
        params["estimator"] = DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=12)
    return BaggingRegressor(**params)


def _voting(params):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
    from sklearn.linear_model import Ridge

    params = dict(params)
    params.setdefault(
        "estimators",
        [
            ("ridge", Ridge(alpha=1.0)),
            ("rf", RandomForestRegressor(n_estimators=60, random_state=RANDOM_STATE, n_jobs=1)),
            ("gbr", GradientBoostingRegressor(random_state=RANDOM_STATE)),
        ],
    )
    return VotingRegressor(**params)


def _stacking(params):
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, StackingRegressor
    from sklearn.linear_model import Ridge

    params = dict(params)
    params.setdefault(
        "estimators",
        [
            ("ridge", Ridge(alpha=1.0)),
            ("extra", ExtraTreesRegressor(n_estimators=40, random_state=RANDOM_STATE, n_jobs=1)),
            ("gbr", GradientBoostingRegressor(random_state=RANDOM_STATE)),
        ],
    )
    params.setdefault("final_estimator", Ridge(alpha=1.0))
    return StackingRegressor(**params)


def _xgb(params):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("xgboost is required for XGBRegressor. Install it with `pip install xgboost`.") from exc

    return XGBRegressor(**params)


def _lgbm(params):
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise ImportError("lightgbm is required for LGBMRegressor. Install it with `pip install lightgbm`.") from exc

    return LGBMRegressor(**params)


def _catboost(params):
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise ImportError("catboost is required for CatBoostRegressor. Install it with `pip install catboost`.") from exc

    return CatBoostRegressor(**params)


def _attention(params):
    try:
        from .attention_models import SelfiesTransformerRegressor
    except ImportError as exc:
        raise ImportError("torch is required for the SELFIES Transformer. Install it with `pip install torch`.") from exc

    return SelfiesTransformerRegressor(**params)


def build_model_registry(enabled_overrides: dict[str, bool] | None = None) -> dict[str, ModelSpec]:
    """Return exactly the 30 requested algorithms, excluding the dummy baseline."""

    specs = [
        ModelSpec("linear_regression", "linear", _linear_regression),
        ModelSpec("ridge", "linear", _ridge, {"alpha": 1.0, "solver": "lsqr"}, requires_scaling=True),
        ModelSpec("lasso", "linear", _lasso, {"alpha": 0.001, "max_iter": 3000, "random_state": RANDOM_STATE}, requires_scaling=True, iterative=True),
        ModelSpec("elastic_net", "linear", _elastic_net, {"alpha": 0.001, "l1_ratio": 0.5, "max_iter": 3000, "random_state": RANDOM_STATE}, requires_scaling=True, iterative=True),
        ModelSpec("bayesian_ridge", "linear", _bayesian_ridge, requires_dense=True, requires_scaling=True),
        ModelSpec("huber_regressor", "linear", _huber, {"max_iter": 300, "epsilon": 1.35}, requires_dense=True, requires_scaling=True, iterative=True),
        ModelSpec("sgd_regressor", "linear", _sgd, {"loss": "squared_error", "penalty": "elasticnet", "max_iter": 2000, "tol": 1e-3, "random_state": RANDOM_STATE}, requires_scaling=True, iterative=True),
        ModelSpec("passive_aggressive_regressor", "linear", _passive_aggressive, {"max_iter": 1000, "tol": 1e-3, "random_state": RANDOM_STATE}, requires_scaling=True, iterative=True),
        ModelSpec("kneighbors_regressor", "neighbors", _knn, {"n_neighbors": 7, "weights": "distance"}, requires_scaling=True),
        ModelSpec("linear_svr", "svm", _linear_svr, {"C": 1.0, "max_iter": 5000, "random_state": RANDOM_STATE}, requires_scaling=True, iterative=True),
        ModelSpec("poisson_regressor", "linear", _poisson, {"alpha": 1.0, "max_iter": 1000, "tol": 1e-6}, requires_scaling=True, iterative=True),
        ModelSpec("svr_rbf", "svm", _svr, {"kernel": "rbf", "C": 10.0, "epsilon": 0.1, "gamma": "scale", "tol": 1e-3, "cache_size": 2048, "shrinking": True, "max_iter": 20000}, requires_scaling=True, iterative=True),
        ModelSpec("nu_svr", "svm", _nu_svr, {"C": 10.0, "nu": 0.5, "kernel": "rbf", "gamma": "scale", "tol": 1e-3, "cache_size": 2048, "shrinking": True, "max_iter": 20000}, requires_scaling=True, iterative=True),
        ModelSpec("nystroem_ridge", "kernel_approximation", _nystroem_ridge, {"kernel": "rbf", "gamma": None, "n_components": 512, "alpha": 1.0, "solver": "lsqr", "random_state": RANDOM_STATE}, requires_dense=True, requires_scaling=True),
        ModelSpec("pls_regression", "linear", _pls, {"n_components": 8}, requires_dense=True, requires_scaling=True),
        ModelSpec("mlp_regressor", "neural_network", _mlp, {"hidden_layer_sizes": (128, 64), "activation": "relu", "early_stopping": True, "max_iter": 250, "random_state": RANDOM_STATE}, requires_dense=True, requires_scaling=True, iterative=True),
        ModelSpec("decision_tree", "tree", _decision_tree, {"max_depth": 24, "min_samples_leaf": 2, "random_state": RANDOM_STATE}, requires_dense=True, tree_based=True),
        ModelSpec("extra_tree", "tree", _extra_tree, {"max_depth": 24, "min_samples_leaf": 2, "random_state": RANDOM_STATE}, requires_dense=True, tree_based=True),
        ModelSpec("random_forest", "ensemble_tree", _random_forest, {"n_estimators": 200, "min_samples_leaf": 2, "random_state": RANDOM_STATE, "n_jobs": -1}, requires_dense=True, tree_based=True, ensemble_based=True, supports_n_jobs=True),
        ModelSpec("extra_trees", "ensemble_tree", _extra_trees, {"n_estimators": 200, "min_samples_leaf": 2, "random_state": RANDOM_STATE, "n_jobs": -1}, requires_dense=True, tree_based=True, ensemble_based=True, supports_n_jobs=True),
        ModelSpec("gradient_boosting", "ensemble_tree", _gradient_boosting, {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 3, "random_state": RANDOM_STATE}, requires_dense=True, tree_based=True, ensemble_based=True, iterative=True),
        ModelSpec("hist_gradient_boosting", "ensemble_tree", _hist_gradient_boosting, {"max_iter": 150, "learning_rate": 0.05, "random_state": RANDOM_STATE}, requires_dense=True, tree_based=True, ensemble_based=True, iterative=True),
        ModelSpec("adaboost", "ensemble_tree", _adaboost, {"n_estimators": 150, "learning_rate": 0.05, "random_state": RANDOM_STATE}, requires_dense=True, tree_based=True, ensemble_based=True),
        ModelSpec("bagging", "ensemble_tree", _bagging, {"n_estimators": 80, "random_state": RANDOM_STATE, "n_jobs": -1}, requires_dense=True, tree_based=True, ensemble_based=True, supports_n_jobs=True),
        ModelSpec("voting_regressor", "ensemble", _voting, {"n_jobs": -1}, requires_dense=True, requires_scaling=True, ensemble_based=True, supports_n_jobs=True),
        ModelSpec("stacking_regressor", "ensemble", _stacking, {"cv": 3, "n_jobs": -1}, requires_dense=True, requires_scaling=True, ensemble_based=True, supports_n_jobs=True),
        ModelSpec("xgb_regressor", "gradient_boosting", _xgb, {"n_estimators": 250, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.9, "objective": "reg:squarederror", "random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": 0}, requires_dense=True, tree_based=True, ensemble_based=True, iterative=True, supports_n_jobs=True),
        ModelSpec("lgbm_regressor", "gradient_boosting", _lgbm, {"n_estimators": 250, "learning_rate": 0.05, "num_leaves": 31, "random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1}, requires_dense=True, tree_based=True, ensemble_based=True, iterative=True, supports_n_jobs=True),
        ModelSpec("catboost_regressor", "gradient_boosting", _catboost, {"iterations": 250, "learning_rate": 0.05, "depth": 6, "loss_function": "RMSE", "random_seed": RANDOM_STATE, "verbose": False, "allow_writing_files": False}, requires_dense=True, tree_based=True, ensemble_based=True, iterative=True),
        ModelSpec("selfies_transformer", "attention", _attention, {}, attention_based=True, iterative=True),
    ]
    registry = {spec.name: spec for spec in specs}
    if len(registry) != 30:
        raise AssertionError(f"Expected exactly 30 benchmark algorithms, found {len(registry)}.")
    if enabled_overrides:
        registry = {
            name: ModelSpec(
                **{
                    **spec.__dict__,
                    "enabled": bool(enabled_overrides.get(name, spec.enabled)),
                }
            )
            for name, spec in registry.items()
        }
    return registry


def dummy_spec() -> ModelSpec:
    """Return the external baseline that is not counted among the 30 algorithms."""

    def _dummy(params):
        from sklearn.dummy import DummyRegressor

        return DummyRegressor(**params)

    return ModelSpec("dummy_mean", "baseline", _dummy, {"strategy": "mean"}, enabled=True)


def catalog_table(registry: dict[str, ModelSpec] | None = None) -> pd.DataFrame:
    """Create the requested model catalog table."""

    specs = list((registry or build_model_registry()).values())
    rows = [
        {
            "name": spec.name,
            "family": spec.family,
            "tree_based": spec.tree_based,
            "ensemble_based": spec.ensemble_based,
            "attention_based": spec.attention_based,
            "requires_dense": spec.requires_dense,
            "requires_scaling": spec.requires_scaling,
            "enabled": spec.enabled,
        }
        for spec in specs
    ]
    return pd.DataFrame(rows)


def validate_model_coverage(registry: dict[str, ModelSpec]) -> None:
    """Fail if the requested model families are not represented."""

    specs = [spec for spec in registry.values() if spec.enabled]
    if len(registry) != 30:
        raise ValueError(f"The registry must contain exactly 30 requested algorithms, found {len(registry)}.")
    if sum(spec.tree_based for spec in specs) < 5:
        raise ValueError("The enabled registry must include at least five tree-based algorithms.")
    if sum(spec.ensemble_based for spec in specs) < 2:
        raise ValueError("The enabled registry must include at least two ensemble algorithms.")
    if not any(spec.family == "svm" for spec in specs):
        raise ValueError("The enabled registry must include support vector machines.")
    if not any(spec.attention_based for spec in specs):
        raise ValueError("The enabled registry must include an attention architecture.")
