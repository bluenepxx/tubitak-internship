"""Benchmark training orchestration used by the small CLI entry point."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupKFold, KFold

from .data_loader import (
    FEATURE_COLUMNS,
    DatasetBundle,
    dataset_content_hash,
    select_columns,
    select_target,
    split_content_hash,
)
from .evaluation import (
    generalization_gaps,
    rank_results,
    regression_metrics,
    save_convergence_plot,
    save_learning_curve_plot,
    save_parity_plot,
    save_residual_plot,
)
from .model_registry import MODEL_REGISTRY_VERSION, ModelSpec, dummy_spec
from .preprocessing import build_classical_pipeline

DEFAULT_STAGES = ("holdout", "kfold", "groupkfold", "learning_curve", "final_test")
RUN_IDENTITY_COLUMNS = ("stage", "target", "model", "dataset_hash", "model_config_hash")
SUCCESS_COLUMNS = ("training_seconds", "validation_rmse", "cv_validation_rmse_mean", "test_rmse", "train_fraction")
MERGEABLE_TABLES = {
    "holdout_results.csv",
    "kfold_results.csv",
    "groupkfold_results.csv",
    "learning_curve_summary.csv",
    "final_test_results.csv",
    "failed_runs.csv",
}


def ensure_result_dirs(config: dict) -> dict[str, Path]:
    """Create all benchmark output folders and return resolved paths."""

    paths_cfg = config.get("results", {})
    root = Path(paths_cfg.get("root", "results"))
    paths = {
        "root": root,
        "tables": Path(paths_cfg.get("tables", root / "tables")),
        "figures": Path(paths_cfg.get("figures", root / "figures")),
        "learning_curves": Path(paths_cfg.get("learning_curves", root / "figures" / "learning_curves")),
        "parity_plots": Path(paths_cfg.get("parity_plots", root / "figures" / "parity_plots")),
        "residual_plots": Path(paths_cfg.get("residual_plots", root / "figures" / "residual_plots")),
        "convergence": Path(paths_cfg.get("convergence", root / "figures" / "convergence")),
        "histories": Path(paths_cfg.get("histories", root / "histories")),
        "models": Path(paths_cfg.get("models", root / "models")),
        "logs": Path(paths_cfg.get("logs", root / "logs")),
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def configure_logging(paths: dict[str, Path]) -> logging.Logger:
    """Configure file and terminal logging."""

    logger = logging.getLogger("qmof_benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(paths["logs"] / "benchmark.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


class FileLock:
    """Small cross-process lock based on atomic lock-file creation."""

    def __init__(self, path: Path, timeout_seconds: float = 120.0) -> None:
        self.path = path
        self.timeout_seconds = timeout_seconds
        self._fd: int | None = None

    def __enter__(self):
        start = time.perf_counter()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                self._fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(self._fd, str(os.getpid()).encode("ascii"))
                return self
            except FileExistsError:
                if time.perf_counter() - start > self.timeout_seconds:
                    raise TimeoutError(f"Timed out waiting for result table lock: {self.path}")
                time.sleep(0.2)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, suffix=".tmp", encoding="utf-8", newline="") as handle:
        temporary_path = Path(handle.name)
        frame.to_csv(handle, index=False)
    temporary_path.replace(path)


def _row_success_score(row: pd.Series) -> tuple[int, float, float]:
    has_error = any(column in row.index and pd.notna(row[column]) and str(row[column]) for column in ("exception_type", "message"))
    success_rank = 0 if has_error else 1
    metric = np.inf
    for column in ("validation_rmse", "cv_validation_rmse_mean", "test_rmse", "train_rmse", "train_fraction"):
        if column in row.index and pd.notna(row[column]):
            value = float(row[column])
            metric = -value if column == "train_fraction" else value
            break
    elapsed = float(row.get("training_seconds", np.inf)) if pd.notna(row.get("training_seconds", np.nan)) else np.inf
    return success_rank, -metric, -elapsed


def deduplicate_results(frame: pd.DataFrame, identity_columns: Sequence[str] = RUN_IDENTITY_COLUMNS) -> pd.DataFrame:
    """Preserve one best successful row for each exact run identity."""

    if frame.empty or any(column not in frame.columns for column in identity_columns):
        return frame
    frame = frame.copy()
    frame["_score"] = frame.apply(_row_success_score, axis=1)
    frame = frame.sort_values("_score", ascending=False)
    frame = frame.drop_duplicates(subset=list(identity_columns), keep="first")
    return frame.drop(columns=["_score"]).sort_index(axis=0).reset_index(drop=True)


def normalize_result_schema(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return a frame with all requested columns, preserving extra incoming columns."""

    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan
    ordered = list(dict.fromkeys([*columns, *out.columns]))
    return out.loc[:, ordered]


def append_csv(path: Path, record: dict[str, Any], identity_columns: Sequence[str] = RUN_IDENTITY_COLUMNS) -> None:
    """Atomically upsert one result for checkpoint/resume safety."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(path.with_suffix(path.suffix + ".lock")):
        existing = safe_read_csv(path)
        frame = pd.concat([existing, pd.DataFrame([record])], ignore_index=True, sort=False)
        frame = deduplicate_results(frame, identity_columns=identity_columns)
        _atomic_write_csv(frame, path)


def merge_result_table(local_path: Path, incoming_frame: pd.DataFrame, bundle: DatasetBundle) -> pd.DataFrame:
    """Merge one external result table after identity and dataset/split validation."""

    missing = [column for column in RUN_IDENTITY_COLUMNS if column not in incoming_frame.columns]
    if missing:
        raise ValueError(f"Incoming table {local_path.name} is missing identity columns: {missing}")
    if "split_hash" not in incoming_frame.columns:
        raise ValueError(f"Incoming table {local_path.name} is missing required split_hash column.")
    dataset_hashes = set(incoming_frame["dataset_hash"].dropna().astype(str))
    split_hashes = set(incoming_frame["split_hash"].dropna().astype(str))
    if dataset_hashes != {bundle.dataset_hash}:
        raise ValueError(
            f"Incoming table {local_path.name} has incompatible dataset_hash values: {sorted(dataset_hashes)} "
            f"expected {bundle.dataset_hash}"
        )
    if split_hashes != {bundle.split_hash}:
        raise ValueError(
            f"Incoming table {local_path.name} has incompatible split_hash values: {sorted(split_hashes)} "
            f"expected {bundle.split_hash}"
        )

    with FileLock(local_path.with_suffix(local_path.suffix + ".lock")):
        local = safe_read_csv(local_path)
        all_columns = list(dict.fromkeys([*local.columns, *incoming_frame.columns]))
        merged = pd.concat(
            [
                normalize_result_schema(local, all_columns),
                normalize_result_schema(incoming_frame, all_columns),
            ],
            ignore_index=True,
            sort=False,
        )
        merged = deduplicate_results(merged)
        _atomic_write_csv(merged, local_path)
    return merged


def merge_result_tables(source_paths: Sequence[Path], paths: dict[str, Path], bundle: DatasetBundle) -> list[Path]:
    """Merge Colab result tables back into canonical local result tables."""

    merged_paths: list[Path] = []
    candidates: list[Path] = []
    for source in source_paths:
        if not source.exists():
            raise FileNotFoundError(f"Merge source does not exist: {source}")
        if source.is_dir():
            candidates.extend(sorted(path for path in source.glob("*.csv") if path.name in MERGEABLE_TABLES))
            tables_dir = source / "tables"
            if tables_dir.is_dir():
                candidates.extend(sorted(path for path in tables_dir.glob("*.csv") if path.name in MERGEABLE_TABLES))
        elif source.name in MERGEABLE_TABLES:
            candidates.append(source)
        else:
            raise ValueError(f"Unsupported merge source: {source}")

    seen: set[Path] = set()
    for incoming_path in candidates:
        if incoming_path in seen:
            continue
        seen.add(incoming_path)
        incoming = safe_read_csv(incoming_path)
        if incoming.empty:
            continue
        local_path = paths["tables"] / incoming_path.name
        merge_result_table(local_path, incoming, bundle)
        merged_paths.append(local_path)
    return merged_paths


def safe_read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV, returning an empty frame for missing or headerless empty files."""

    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_completed(path: Path, key_columns: Iterable[str]) -> set[tuple[Any, ...]]:
    """Load successful result keys for resume mode."""

    frame = safe_read_csv(path)
    if frame.empty or any(column not in frame.columns for column in key_columns):
        return set()
    return set(map(tuple, frame[list(key_columns)].to_numpy()))


def completed_identities(
    path: Path,
    dataset_hash: str,
    model_hashes: dict[str, str],
    allowed_models: set[str],
) -> set[tuple[Any, ...]]:
    """Return completed identities for the current dataset and per-model hashes."""

    frame = safe_read_csv(path)
    if frame.empty:
        return set()
    required = set(RUN_IDENTITY_COLUMNS)
    if required.issubset(frame.columns):
        exact = frame[
            (frame["dataset_hash"] == dataset_hash)
            & (frame["model"].isin(allowed_models))
            & (frame.apply(lambda row: model_hashes.get(str(row["model"])) == row["model_config_hash"], axis=1))
        ]
        completed = set(map(tuple, exact[list(RUN_IDENTITY_COLUMNS)].to_numpy()))
        legacy_mask = (
            frame["model"].isin(allowed_models)
            & (frame["dataset_hash"].isna() | frame["model_config_hash"].isna())
        )
        for row in frame.loc[legacy_mask, ["stage", "target", "model"]].itertuples(index=False):
            model = str(row.model)
            if model in model_hashes:
                completed.add((row.stage, row.target, model, dataset_hash, model_hashes[model]))
        return completed

    legacy_columns = ["stage", "target", "model"]
    if all(column in frame.columns for column in legacy_columns):
        frame = frame[frame["model"].isin(allowed_models)]
        return {
            (row.stage, row.target, row.model, dataset_hash, model_hashes[str(row.model)])
            for row in frame[legacy_columns].itertuples(index=False)
            if str(row.model) in model_hashes
        }
    return set()


def effective_model_params(config: dict, spec: ModelSpec, mode: str) -> dict[str, Any]:
    """Return effective model parameters without stage-specific n_jobs throttling."""

    models_cfg = config.get("models", {}).get(spec.name, {})
    params = dict(models_cfg.get("parameters", spec.parameters) or {})
    if mode == "smoke":
        params.update(models_cfg.get("smoke_parameters", {}) or {})
    if spec.supports_n_jobs:
        params["n_jobs"] = int(config.get("n_jobs", -1))
    return params


def model_config_hash(spec: ModelSpec, config: dict, mode: str) -> str:
    """Hash only the selected model's effective configuration."""

    payload = {
        "model_registry_version": MODEL_REGISTRY_VERSION,
        "name": spec.name,
        "family": spec.family,
        "parameters": effective_model_params(config, spec, mode),
        "requires_dense": spec.requires_dense,
        "requires_scaling": spec.requires_scaling,
        "preprocessing": config.get("preprocessing", {}),
        "random_seed": config.get("random_seed", 42),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def model_hash_map(specs: Iterable[ModelSpec], config: dict, mode: str) -> dict[str, str]:
    return {spec.name: model_config_hash(spec, config, mode) for spec in specs}


def add_run_identity(record: dict[str, Any], bundle: DatasetBundle, spec: ModelSpec, config: dict, mode: str) -> dict[str, Any]:
    """Attach dataset/split/model identity fields to one result record."""

    enriched = dict(record)
    enriched["dataset_hash"] = bundle.dataset_hash
    enriched["split_hash"] = bundle.split_hash
    enriched["model_config_hash"] = model_config_hash(spec, config, mode)
    enriched["model_registry_version"] = MODEL_REGISTRY_VERSION
    return enriched


def record_failure(
    path: Path,
    stage: str,
    target: str,
    spec: ModelSpec,
    exc: BaseException,
    bundle: DatasetBundle,
    config: dict,
    mode: str,
) -> None:
    """Append an estimator failure and keep the wider benchmark alive."""

    append_csv(
        path,
        add_run_identity(
            {
                "stage": stage,
                "target": target,
                "model": spec.name,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=8),
            },
            bundle,
            spec,
            config,
            mode,
        ),
    )


def _log_file_only(logger: logging.Logger, level: int, message: str) -> None:
    record = logger.makeRecord(logger.name, level, __file__, 0, message, args=(), exc_info=None)
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.handle(record)


def progress_start(stage: str, target: str, model: str, logger: logging.Logger) -> float:
    """Emit a flushed progress start line and return a monotonic timer."""

    start = time.perf_counter()
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"[START] stage={stage} target={target} model={model} start={timestamp}"
    print(message, flush=True)
    _log_file_only(logger, logging.INFO, message)
    return start


def _record_metric(record: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = record.get(name)
        if value is not None and not pd.isna(value):
            return value
    return "NA"


def progress_success(stage: str, target: str, model: str, start: float, record: dict[str, Any], logger: logging.Logger) -> None:
    """Emit a flushed progress success line with core metrics."""

    elapsed = time.perf_counter() - start
    train_rmse = _record_metric(record, "train_rmse", "cv_train_rmse_mean", "train_validation_rmse")
    validation_rmse = _record_metric(record, "validation_rmse", "cv_validation_rmse_mean", "test_rmse")
    validation_r2 = _record_metric(record, "validation_r2", "cv_validation_r2_mean", "test_r2")
    message = (
        f"[SUCCESS] stage={stage} target={target} model={model} elapsed={elapsed:.2f}s "
        f"train_rmse={train_rmse} validation_rmse={validation_rmse} validation_r2={validation_r2}"
    )
    print(message, flush=True)
    _log_file_only(logger, logging.INFO, message)


def progress_failure(stage: str, target: str, model: str, start: float, exc: BaseException, logger: logging.Logger) -> None:
    """Emit a flushed progress failure line."""

    elapsed = time.perf_counter() - start
    message = (
        f"[FAILURE] stage={stage} target={target} model={model} elapsed={elapsed:.2f}s "
        f"error={type(exc).__name__}: {exc}"
    )
    print(message, flush=True)
    _log_file_only(logger, logging.ERROR, message)


def _model_params(config: dict, spec: ModelSpec, mode: str) -> dict[str, Any]:
    return effective_model_params(config, spec, mode)


def _cv_model_params(config: dict, spec: ModelSpec, mode: str) -> dict[str, Any]:
    params = _model_params(config, spec, mode)
    if spec.supports_n_jobs:
        params["n_jobs"] = 1
    return params


def _min_df(config: dict, mode: str) -> int:
    if mode == "smoke":
        return int(config.get("preprocessing", {}).get("smoke_min_df", 1))
    return int(config.get("preprocessing", {}).get("tfidf_min_df", 2))


def _max_dense_mb(config: dict) -> float:
    return float(config.get("preprocessing", {}).get("max_dense_mb", 1024.0))


def _validate_poisson_target(spec: ModelSpec, target: str, y: pd.Series, split_name: str) -> None:
    if spec.name != "poisson_regressor":
        return
    negative_mask = y < 0
    if negative_mask.any():
        examples = y.loc[negative_mask].head(5).tolist()
        raise ValueError(
            f"PoissonRegressor requires non-negative target values. "
            f"target={target} split={split_name} negative_count={int(negative_mask.sum())} examples={examples}"
        )


def _estimator_n_iter(estimator: Any) -> Any:
    n_iter = getattr(estimator, "n_iter_", np.nan)
    try:
        has_value = not bool(pd.isna(n_iter).all())
    except AttributeError:
        has_value = not bool(pd.isna(n_iter))
    if has_value:
        return n_iter
    named_steps = getattr(estimator, "named_steps", None)
    if named_steps:
        last_step = list(named_steps.values())[-1]
        return getattr(last_step, "n_iter_", np.nan)
    return np.nan


def _convergence_reached(caught: list[warnings.WarningMessage]) -> bool:
    return not any(issubclass(warning.category, ConvergenceWarning) for warning in caught)


def _fit_predict_classical(
    spec: ModelSpec,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    target: str,
    params: dict[str, Any],
    config: dict,
    validation_split_name: str = "validation",
):
    X_train = select_columns(train_df, FEATURE_COLUMNS)
    y_train = select_target(train_df, target)
    X_validation = select_columns(validation_df, FEATURE_COLUMNS)
    y_validation = select_target(validation_df, target)
    _validate_poisson_target(spec, target, y_train, "train")
    _validate_poisson_target(spec, target, y_validation, validation_split_name)

    estimator = spec.factory(params)
    pipeline = build_classical_pipeline(
        estimator,
        requires_dense=spec.requires_dense,
        requires_scaling=spec.requires_scaling,
        min_df=_min_df(config, config.get("mode", "full")),
        max_dense_mb=_max_dense_mb(config),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(X_train, y_train)
    train_pred = pipeline.predict(X_train)
    validation_pred = pipeline.predict(X_validation)
    convergence = "; ".join(str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning))
    n_iter = _estimator_n_iter(pipeline.named_steps["model"])
    return pipeline, train_pred, validation_pred, convergence, n_iter, _convergence_reached(caught)


def _fit_predict_attention(
    spec: ModelSpec,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    target: str,
    params: dict[str, Any],
):
    model = spec.factory(params)
    y_train = select_target(train_df, target)
    y_validation = select_target(validation_df, target)
    model.fit(train_df, y_train, validation_data=(validation_df, y_validation))
    return model, model.predict(train_df), model.predict(validation_df), "", getattr(model, "n_iter_", np.nan), True


def train_one_holdout(
    spec: ModelSpec,
    target: str,
    bundle: DatasetBundle,
    config: dict,
    mode: str,
) -> tuple[Any, dict[str, Any], np.ndarray]:
    """Train one model-target pair and return fitted model, metrics, validation predictions."""

    params = _model_params(config, spec, mode)
    start = time.perf_counter()
    if spec.attention_based:
        fitted, train_pred, validation_pred, convergence, n_iter, convergence_reached = _fit_predict_attention(
            spec, bundle.train, bundle.validation, target, params
        )
    else:
        fitted, train_pred, validation_pred, convergence, n_iter, convergence_reached = _fit_predict_classical(
            spec, bundle.train, bundle.validation, target, params, {**config, "mode": mode}
        )
    elapsed = time.perf_counter() - start
    y_train = select_target(bundle.train, target)
    y_validation = select_target(bundle.validation, target)
    target_std = float(y_train.std(ddof=1))
    train_metrics = regression_metrics(y_train, train_pred, prefix="train_", target_std=target_std)
    validation_metrics = regression_metrics(
        y_validation, validation_pred, prefix="validation_", target_std=target_std
    )
    record = {
        "stage": "holdout",
        "target": target,
        "model": spec.name,
        "training_seconds": float(elapsed),
        "convergence_warnings": convergence,
        "convergence_reached": convergence_reached,
        "n_iter": n_iter,
        **train_metrics,
        **validation_metrics,
        **generalization_gaps(train_metrics, validation_metrics),
    }
    return fitted, add_run_identity(record, bundle, spec, config, mode), np.asarray(validation_pred)


def run_holdout(
    specs: list[ModelSpec],
    targets: list[str],
    bundle: DatasetBundle,
    config: dict,
    mode: str,
    paths: dict[str, Path],
    logger: logging.Logger,
    resume: bool,
    include_dummy_baseline: bool = True,
) -> dict[tuple[str, str], Any]:
    """Run all holdout fits, including the dummy baseline."""

    output = paths["tables"] / "holdout_results.csv"
    failed = paths["tables"] / "failed_runs.csv"
    all_specs = [dummy_spec(), *specs] if include_dummy_baseline else list(specs)
    model_hashes = model_hash_map(all_specs, config, mode)
    completed = completed_identities(output, bundle.dataset_hash, model_hashes, {spec.name for spec in all_specs}) if resume else set()
    fitted_models: dict[tuple[str, str], Any] = {}
    for target in targets:
        for spec in all_specs:
            key = ("holdout", target, spec.name, bundle.dataset_hash, model_hashes[spec.name])
            if key in completed:
                logger.info("Skipping completed holdout %s/%s", target, spec.name)
                continue
            stage_start = progress_start("holdout", target, spec.name, logger)
            try:
                fitted, record, validation_pred = train_one_holdout(spec, target, bundle, config, mode)
                append_csv(output, record)
                fitted_models[(target, spec.name)] = fitted
                if spec.iterative and hasattr(fitted, "history_"):
                    history_path = paths["histories"] / f"{target}_{spec.name}_history.csv"
                    fitted.save_history(history_path)
                    save_convergence_plot(
                        pd.read_csv(history_path),
                        paths["convergence"] / f"{target}_{spec.name}_convergence.png",
                        f"{target} {spec.name}",
                    )
                save_parity_plot(
                    select_target(bundle.validation, target),
                    validation_pred,
                    paths["parity_plots"] / f"{target}_{spec.name}_validation.png",
                    f"{target} {spec.name} validation",
                )
                save_residual_plot(
                    select_target(bundle.validation, target),
                    validation_pred,
                    paths["residual_plots"] / f"{target}_{spec.name}_validation.png",
                    f"{target} {spec.name} validation",
                )
                progress_success("holdout", target, spec.name, stage_start, record, logger)
            except Exception as exc:
                progress_failure("holdout", target, spec.name, stage_start, exc, logger)
                record_failure(failed, "holdout", target, spec, exc, bundle, config, mode)
    return fitted_models


def run_cross_validation(
    specs: list[ModelSpec],
    targets: list[str],
    bundle: DatasetBundle,
    config: dict,
    mode: str,
    paths: dict[str, Path],
    logger: logging.Logger,
    resume: bool,
) -> None:
    """Run train-only KFold CV for all enabled models and targets."""

    output = paths["tables"] / "kfold_results.csv"
    failed = paths["tables"] / "failed_runs.csv"
    model_hashes = model_hash_map(specs, config, mode)
    completed = completed_identities(output, bundle.dataset_hash, model_hashes, {spec.name for spec in specs}) if resume else set()
    folds = int(config.get("cv_folds", 5))
    if mode == "smoke":
        folds = int(config.get("smoke", {}).get("cv_folds", 2))
    folds = min(folds, len(bundle.train))
    splitter = KFold(n_splits=max(2, folds), shuffle=True, random_state=int(config.get("random_seed", 42)))
    for target in targets:
        for spec in specs:
            key = ("kfold", target, spec.name, bundle.dataset_hash, model_hashes[spec.name])
            if key in completed:
                logger.info("Skipping completed kfold %s/%s", target, spec.name)
                continue
            stage_start = progress_start("kfold", target, spec.name, logger)
            try:
                fold_rows = []
                for fold, (train_idx, validation_idx) in enumerate(splitter.split(bundle.train), start=1):
                    fold_train = bundle.train.iloc[train_idx].copy()
                    fold_validation = bundle.train.iloc[validation_idx].copy()
                    params = _cv_model_params(config, spec, mode)
                    if spec.attention_based:
                        fitted, train_pred, validation_pred, convergence, n_iter, convergence_reached = _fit_predict_attention(
                            spec, fold_train, fold_validation, target, params
                        )
                    else:
                        fitted, train_pred, validation_pred, convergence, n_iter, convergence_reached = _fit_predict_classical(
                            spec, fold_train, fold_validation, target, params, {**config, "mode": mode}
                        )
                    y_fold_train = select_target(fold_train, target)
                    y_fold_validation = select_target(fold_validation, target)
                    target_std = float(y_fold_train.std(ddof=1))
                    train_metrics = regression_metrics(y_fold_train, train_pred, prefix="cv_train_", target_std=target_std)
                    validation_metrics = regression_metrics(
                        y_fold_validation, validation_pred, prefix="cv_validation_", target_std=target_std
                    )
                    fold_rows.append(
                        {
                            "stage": "kfold",
                            "target": target,
                            "model": spec.name,
                            "fold": fold,
                            "convergence_warnings": convergence,
                            "convergence_reached": convergence_reached,
                            "n_iter": n_iter,
                            **train_metrics,
                            **validation_metrics,
                        }
                    )
                summary = _summarize_cv_rows(fold_rows, "kfold", target, spec.name)
                summary = add_run_identity(summary, bundle, spec, config, mode)
                append_csv(output, summary)
                progress_success("kfold", target, spec.name, stage_start, summary, logger)
            except Exception as exc:
                progress_failure("kfold", target, spec.name, stage_start, exc, logger)
                record_failure(failed, "kfold", target, spec, exc, bundle, config, mode)


def _summarize_cv_rows(rows: list[dict[str, Any]], stage: str, target: str, model: str) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    summary: dict[str, Any] = {"stage": stage, "target": target, "model": model, "fold_count": int(len(frame))}
    for column in [c for c in frame.columns if c.startswith(("cv_train_", "cv_validation_"))]:
        summary[f"{column}_mean"] = float(frame[column].mean())
        summary[f"{column}_std"] = float(frame[column].std(ddof=1)) if len(frame) > 1 else 0.0
    summary["convergence_warnings"] = " | ".join(sorted(set(str(x) for x in frame["convergence_warnings"] if str(x))))
    summary["convergence_reached"] = bool(frame["convergence_reached"].all()) if "convergence_reached" in frame.columns else True
    summary["n_iter"] = " | ".join(str(x) for x in frame["n_iter"].tolist())
    return summary


def _filter_current_results(
    frame: pd.DataFrame,
    target: str,
    bundle: DatasetBundle,
    model_hashes: dict[str, str],
    valid_models: set[str],
) -> pd.DataFrame:
    if frame.empty or "target" not in frame.columns or "model" not in frame.columns:
        return pd.DataFrame()
    out = frame[(frame["target"] == target) & (frame["model"].isin(valid_models))].copy()
    if {"dataset_hash", "model_config_hash"}.issubset(out.columns):
        exact = out[
            (out["dataset_hash"] == bundle.dataset_hash)
            & (out.apply(lambda row: model_hashes.get(str(row["model"])) == row["model_config_hash"], axis=1))
        ]
        legacy = out[out["dataset_hash"].isna() | out["model_config_hash"].isna()]
        out = pd.concat([exact, legacy], ignore_index=True, sort=False)
    return out


def _top_models_for_target(
    paths: dict[str, Path],
    target: str,
    k: int,
    bundle: DatasetBundle,
    model_hashes: dict[str, str],
    valid_models: set[str],
) -> list[str]:
    holdout_path = paths["tables"] / "holdout_results.csv"
    kfold_path = paths["tables"] / "kfold_results.csv"
    holdout = safe_read_csv(holdout_path)
    cv = safe_read_csv(kfold_path)
    if cv.empty or "target" not in cv.columns or holdout.empty or "target" not in holdout.columns:
        return []
    target_cv = _filter_current_results(cv, target, bundle, model_hashes, valid_models)
    target_holdout = _filter_current_results(holdout, target, bundle, model_hashes, valid_models)
    merged = target_cv.merge(
        target_holdout[["model", "validation_rmse"]],
        on="model",
        how="left",
    )
    ranked = rank_results(merged)
    return ranked["model"].head(k).tolist()


def run_groupkfold(
    registry: dict[str, ModelSpec],
    targets: list[str],
    bundle: DatasetBundle,
    config: dict,
    mode: str,
    paths: dict[str, Path],
    logger: logging.Logger,
    resume: bool,
) -> None:
    """Run GroupKFold on top models selected without test data."""

    output = paths["tables"] / "groupkfold_results.csv"
    failed = paths["tables"] / "failed_runs.csv"
    model_hashes = model_hash_map(registry.values(), config, mode)
    completed = completed_identities(output, bundle.dataset_hash, model_hashes, set(registry)) if resume else set()
    top_k = int(config.get("groupkfold_top_k", 5))
    if mode == "smoke":
        top_k = int(config.get("smoke", {}).get("groupkfold_top_k", 2))
    groups = bundle.train["input_signature"]
    n_groups = groups.nunique()
    if n_groups < 2:
        logger.info("Skipping GroupKFold: fewer than two unique input signatures.")
        return
    n_splits = min(int(config.get("cv_folds", 5)), n_groups)
    for target in targets:
        for model_name in _top_models_for_target(paths, target, top_k, bundle, model_hashes, set(registry)):
            if model_name not in registry:
                continue
            spec = registry[model_name]
            key = ("groupkfold", target, spec.name, bundle.dataset_hash, model_hashes[spec.name])
            if key in completed:
                continue
            stage_start = progress_start("groupkfold", target, spec.name, logger)
            try:
                splitter = GroupKFold(n_splits=max(2, n_splits))
                fold_rows = []
                for fold, (train_idx, validation_idx) in enumerate(splitter.split(bundle.train, groups=groups), start=1):
                    fold_train = bundle.train.iloc[train_idx].copy()
                    fold_validation = bundle.train.iloc[validation_idx].copy()
                    if set(fold_train["input_signature"]).intersection(set(fold_validation["input_signature"])):
                        raise ValueError("Exact input signature leakage detected between GroupKFold train and validation.")
                    params = _cv_model_params(config, spec, mode)
                    if spec.attention_based:
                        _, train_pred, validation_pred, convergence, n_iter, convergence_reached = _fit_predict_attention(
                            spec, fold_train, fold_validation, target, params
                        )
                    else:
                        _, train_pred, validation_pred, convergence, n_iter, convergence_reached = _fit_predict_classical(
                            spec, fold_train, fold_validation, target, params, {**config, "mode": mode}
                        )
                    y_fold_train = select_target(fold_train, target)
                    y_fold_validation = select_target(fold_validation, target)
                    target_std = float(y_fold_train.std(ddof=1))
                    fold_rows.append(
                        {
                            "stage": "groupkfold",
                            "target": target,
                            "model": spec.name,
                            "fold": fold,
                            "convergence_warnings": convergence,
                            "convergence_reached": convergence_reached,
                            "n_iter": n_iter,
                            **regression_metrics(y_fold_train, train_pred, prefix="cv_train_", target_std=target_std),
                            **regression_metrics(
                                y_fold_validation, validation_pred, prefix="cv_validation_", target_std=target_std
                            ),
                        }
                    )
                summary = _summarize_cv_rows(fold_rows, "groupkfold", target, spec.name)
                summary = add_run_identity(summary, bundle, spec, config, mode)
                append_csv(output, summary)
                progress_success("groupkfold", target, spec.name, stage_start, summary, logger)
            except Exception as exc:
                progress_failure("groupkfold", target, spec.name, stage_start, exc, logger)
                record_failure(failed, "groupkfold", target, spec, exc, bundle, config, mode)


def run_learning_curves(
    registry: dict[str, ModelSpec],
    targets: list[str],
    bundle: DatasetBundle,
    config: dict,
    mode: str,
    paths: dict[str, Path],
    logger: logging.Logger,
    resume: bool,
) -> None:
    """Train top models on increasing training fractions."""

    output = paths["tables"] / "learning_curve_summary.csv"
    failed = paths["tables"] / "failed_runs.csv"
    model_hashes = model_hash_map(registry.values(), config, mode)
    completed = completed_identities(output, bundle.dataset_hash, model_hashes, set(registry)) if resume else set()
    top_k = int(config.get("learning_curve_top_k", 3))
    if mode == "smoke":
        top_k = int(config.get("smoke", {}).get("learning_curve_top_k", 1))
    fractions = config.get("learning_curve_fractions", [0.10, 0.25, 0.50, 0.75, 1.0])
    rng = np.random.default_rng(int(config.get("random_seed", 42)))
    for target in targets:
        for model_name in _top_models_for_target(paths, target, top_k, bundle, model_hashes, set(registry)):
            if model_name not in registry:
                continue
            spec = registry[model_name]
            key = ("learning_curve", target, spec.name, bundle.dataset_hash, model_hashes[spec.name])
            if key in completed:
                continue
            stage_start = progress_start("learning_curve", target, spec.name, logger)
            try:
                rows = []
                indices = np.arange(len(bundle.train))
                for fraction in fractions:
                    sample_size = max(2, int(round(len(indices) * float(fraction))))
                    sample_idx = rng.choice(indices, size=min(sample_size, len(indices)), replace=False)
                    train_part = bundle.train.iloc[sample_idx].copy()
                    fitted, record, _ = train_one_holdout(spec, target, DatasetBundle(
                        df=bundle.df,
                        train=train_part,
                        validation=bundle.validation,
                        test=bundle.test,
                        split_assignments=bundle.split_assignments,
                        dataset_summary=bundle.dataset_summary,
                        split_summary=bundle.split_summary,
                        dataset_hash=bundle.dataset_hash,
                        split_hash=bundle.split_hash,
                    ), config, mode)
                    rows.append({"train_fraction": float(fraction), **record})
                curve = pd.DataFrame(rows)
                curve["stage"] = "learning_curve"
                summary_record = add_run_identity(rows[-1], bundle, spec, config, mode)
                summary_record["stage"] = "learning_curve"
                append_csv(output, summary_record)
                save_learning_curve_plot(
                    curve,
                    paths["learning_curves"] / f"{target}_{spec.name}.png",
                    f"{target} {spec.name}",
                )
                progress_success("learning_curve", target, spec.name, stage_start, summary_record, logger)
            except Exception as exc:
                progress_failure("learning_curve", target, spec.name, stage_start, exc, logger)
                record_failure(failed, "learning_curve", target, spec, exc, bundle, config, mode)


def write_leaderboards(
    targets: list[str],
    paths: dict[str, Path],
    registry: dict[str, ModelSpec],
    bundle: DatasetBundle,
    config: dict,
    mode: str,
) -> dict[str, str]:
    """Write one leaderboard per target from CV and validation results."""

    holdout_path = paths["tables"] / "holdout_results.csv"
    kfold_path = paths["tables"] / "kfold_results.csv"
    holdout = safe_read_csv(holdout_path)
    kfold = safe_read_csv(kfold_path)
    winners: dict[str, str] = {}
    for target in targets:
        if kfold.empty or "target" not in kfold.columns or holdout.empty or "target" not in holdout.columns:
            pd.DataFrame().to_csv(paths["tables"] / f"leaderboard_{target}.csv", index=False)
            continue
        valid_models = set(registry)
        hashes = model_hash_map(registry.values(), config, mode)
        target_kfold = _filter_current_results(kfold, target, bundle, hashes, valid_models)
        target_holdout = _filter_current_results(holdout, target, bundle, hashes, valid_models)
        merged = target_kfold.merge(
            target_holdout[["model", "validation_rmse", "validation_r2", "train_r2", "train_rmse"]],
            on="model",
            how="left",
        )
        ranked = rank_results(merged)
        ranked.to_csv(paths["tables"] / f"leaderboard_{target}.csv", index=False)
        if not ranked.empty:
            winners[target] = str(ranked.iloc[0]["model"])
    return winners


def run_final_test(
    registry: dict[str, ModelSpec],
    winners: dict[str, str],
    bundle: DatasetBundle,
    config: dict,
    mode: str,
    paths: dict[str, Path],
    logger: logging.Logger,
    resume: bool,
) -> None:
    """Evaluate the final winner for each target on the test set exactly once."""

    output = paths["tables"] / "final_test_results.csv"
    failed = paths["tables"] / "failed_runs.csv"
    model_hashes = model_hash_map(registry.values(), config, mode)
    completed = completed_identities(output, bundle.dataset_hash, model_hashes, set(registry)) if resume else set()
    for target, model_name in winners.items():
        spec = registry[model_name]
        key = ("final_test", target, spec.name, bundle.dataset_hash, model_hashes[spec.name])
        if key in completed:
            continue
        stage_start = progress_start("final_test", target, spec.name, logger)
        try:
            train_plus_validation = pd.concat([bundle.train, bundle.validation], axis=0).reset_index(drop=True)
            params = _model_params(config, spec, mode)
            start = time.perf_counter()
            if spec.attention_based:
                model = spec.factory(params)
                y_train_validation = select_target(train_plus_validation, target)
                y_validation = select_target(bundle.validation, target)
                model.fit(train_plus_validation, y_train_validation, validation_data=(bundle.validation, y_validation))
                test_pred = model.predict(bundle.test)
                train_pred = model.predict(train_plus_validation)
                convergence = ""
                n_iter = getattr(model, "n_iter_", np.nan)
                convergence_reached = True
            else:
                model, train_pred, test_pred, convergence, n_iter, convergence_reached = _fit_predict_classical(
                    spec,
                    train_plus_validation,
                    bundle.test,
                    target,
                    params,
                    {**config, "mode": mode},
                    validation_split_name="test",
                )
            elapsed = time.perf_counter() - start
            y_train_validation = select_target(train_plus_validation, target)
            y_test = select_target(bundle.test, target)
            target_std = float(y_train_validation.std(ddof=1))
            record = {
                "stage": "final_test",
                "target": target,
                "model": spec.name,
                "training_seconds": float(elapsed),
                "convergence_warnings": convergence,
                "convergence_reached": convergence_reached,
                "n_iter": n_iter,
                **regression_metrics(y_train_validation, train_pred, prefix="train_validation_", target_std=target_std),
                **regression_metrics(y_test, test_pred, prefix="test_", target_std=target_std),
            }
            record = add_run_identity(record, bundle, spec, config, mode)
            append_csv(output, record)
            if spec.attention_based and hasattr(model, "save_checkpoint"):
                model.save_checkpoint(paths["models"] / f"{target}_{spec.name}.pt")
            else:
                joblib.dump(model, paths["models"] / f"{target}_{spec.name}.joblib")
            save_parity_plot(y_test, test_pred, paths["parity_plots"] / f"{target}_{spec.name}_test.png", f"{target} {spec.name} test")
            save_residual_plot(y_test, test_pred, paths["residual_plots"] / f"{target}_{spec.name}_test.png", f"{target} {spec.name} test")
            progress_success("final_test", target, spec.name, stage_start, record, logger)
        except Exception as exc:
            progress_failure("final_test", target, spec.name, stage_start, exc, logger)
            record_failure(failed, "final_test", target, spec, exc, bundle, config, mode)


def make_smoke_bundle(bundle: DatasetBundle, config: dict) -> DatasetBundle:
    """Use a small deterministic subset while preserving train/validation/test labels."""

    smoke_cfg = config.get("smoke", {})
    per_split = int(smoke_cfg.get("rows_per_split", 30))
    parts = [
        split.sample(n=min(per_split, len(split)), random_state=int(config.get("random_seed", 42)))
        for split in (bundle.train, bundle.validation, bundle.test)
    ]
    df = pd.concat(parts, axis=0).reset_index(drop=True)
    return DatasetBundle(
        df=df,
        train=df[df["split"] == "train"].copy(),
        validation=df[df["split"] == "validation"].copy(),
        test=df[df["split"] == "test"].copy(),
        split_assignments=bundle.split_assignments,
        dataset_summary=bundle.dataset_summary,
        split_summary=bundle.split_summary,
        dataset_hash=dataset_content_hash(df),
        split_hash=split_content_hash(df[["qmof_id", "split"]]),
    )


def write_manifest(bundle: DatasetBundle, specs: list[ModelSpec], config: dict, paths: dict[str, Path], mode: str) -> None:
    """Write a machine-readable run manifest."""

    model_configuration = {
        spec.name: {
            "family": spec.family,
            "parameters": effective_model_params(config, spec, mode),
            "enabled": spec.enabled,
            "requires_dense": spec.requires_dense,
            "requires_scaling": spec.requires_scaling,
        }
        for spec in specs
    }
    model_configuration_hash = hashlib.sha256(
        json.dumps(model_configuration, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    per_model_hashes = {spec.name: model_config_hash(spec, config, mode) for spec in specs}

    packages = {}
    for name in ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "xgboost", "lightgbm", "catboost", "torch"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_commit = None
    device_info = {"platform": platform.platform(), "processor": platform.processor()}
    try:
        import torch

        device_info["torch_cuda_available"] = torch.cuda.is_available()
        device_info["torch_device"] = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device_info["torch_cuda_available"] = None
    manifest = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "python_version": sys.version,
        "package_versions": packages,
        "random_seed": int(config.get("random_seed", 42)),
        "dataset_row_count": int(len(bundle.df)),
        "dataset_hash": bundle.dataset_hash,
        "split_hash": bundle.split_hash,
        "split_sizes": bundle.split_summary.to_dict(orient="records"),
        "enabled_algorithms": [spec.name for spec in specs if spec.enabled],
        "model_registry_version": MODEL_REGISTRY_VERSION,
        "model_configuration_hash": model_configuration_hash,
        "per_model_config_hashes": per_model_hashes,
        "git_commit": git_commit,
        "device_information": device_info,
    }
    (paths["root"] / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def ensure_required_tables_exist(paths: dict[str, Path], targets: list[str]) -> None:
    """Create empty required CSV outputs that a partial/successful run did not touch."""

    required = [
        "model_catalog.csv",
        "dataset_summary.csv",
        "split_summary.csv",
        "holdout_results.csv",
        "kfold_results.csv",
        "groupkfold_results.csv",
        "learning_curve_summary.csv",
        "final_test_results.csv",
        "failed_runs.csv",
    ]
    for name in required:
        path = paths["tables"] / name
        if not path.exists():
            pd.DataFrame().to_csv(path, index=False)
    for target in targets:
        path = paths["tables"] / f"leaderboard_{target}.csv"
        if not path.exists():
            pd.DataFrame().to_csv(path, index=False)


def run_benchmark(
    registry: dict[str, ModelSpec],
    targets: list[str],
    bundle: DatasetBundle,
    config: dict,
    mode: str,
    resume: bool,
    stages: Sequence[str] = DEFAULT_STAGES,
    include_dummy_baseline: bool = True,
) -> None:
    """Run the configured benchmark stages."""

    selected_stages = tuple(stages)
    unknown_stages = sorted(set(selected_stages) - set(DEFAULT_STAGES))
    if unknown_stages:
        raise ValueError(f"Unknown benchmark stages: {unknown_stages}")

    paths = ensure_result_dirs(config)
    logger = configure_logging(paths)
    specs = [spec for spec in registry.values() if spec.enabled]
    if mode == "smoke":
        bundle = make_smoke_bundle(bundle, config)
    logger.info("Dataset rows: %s | train=%s validation=%s test=%s", len(bundle.df), len(bundle.train), len(bundle.validation), len(bundle.test))
    bundle.dataset_summary.to_csv(paths["tables"] / "dataset_summary.csv", index=False)
    bundle.split_summary.to_csv(paths["tables"] / "split_summary.csv", index=False)
    write_manifest(bundle, specs, config, paths, mode)
    if "holdout" in selected_stages:
        run_holdout(specs, targets, bundle, config, mode, paths, logger, resume, include_dummy_baseline=include_dummy_baseline)
    if "kfold" in selected_stages:
        run_cross_validation(specs, targets, bundle, config, mode, paths, logger, resume)
    if "groupkfold" in selected_stages:
        run_groupkfold(registry, targets, bundle, config, mode, paths, logger, resume)
    if "learning_curve" in selected_stages:
        run_learning_curves(registry, targets, bundle, config, mode, paths, logger, resume)
    winners = write_leaderboards(targets, paths, registry, bundle, config, mode)
    if "final_test" in selected_stages:
        run_final_test(registry, winners, bundle, config, mode, paths, logger, resume)
    ensure_required_tables_exist(paths, targets)
