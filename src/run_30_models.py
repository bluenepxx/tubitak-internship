"""Small CLI entry point for the 30-model QMOF regression benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .data_loader import TARGET_COLUMNS, load_dataset_bundle, select_columns
from .model_registry import build_model_registry, catalog_table, validate_model_coverage
from .train import DEFAULT_STAGES, ensure_result_dirs, merge_result_tables, run_benchmark


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML config file."""

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def parse_args() -> argparse.Namespace:
    """Parse benchmark CLI arguments."""

    parser = argparse.ArgumentParser(description="Run the QMOF 30-model regression benchmark.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--resume", action="store_true", help="Skip successful stage/target/model records already in CSVs.")
    parser.add_argument("--targets", nargs="+", default=list(TARGET_COLUMNS), choices=list(TARGET_COLUMNS))
    parser.add_argument("--include-models", nargs="+", default=None, metavar="MODEL")
    parser.add_argument("--exclude-models", nargs="+", default=None, metavar="MODEL")
    parser.add_argument("--stages", nargs="+", default=list(DEFAULT_STAGES), choices=list(DEFAULT_STAGES))
    parser.add_argument("--merge-results", nargs="+", default=None, metavar="PATH", help="Merge external result CSVs or result directories, then exit.")
    parser.add_argument("--experiment-config", default="configs/experiment.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    return parser.parse_args()


def smoke_test_feature_selection(frame: pd.DataFrame) -> None:
    """Verify the exact requested multi-column feature selection succeeds."""

    requested = [
        "node_token_text",
        "linker_token_text",
        "point_group",
        "topology",
        "topology_missing",
    ]
    selected = select_columns(frame, requested)
    if list(selected.columns) != requested:
        raise AssertionError(f"Feature selection returned unexpected columns: {list(selected.columns)}")


def smoke_test_estimator_construction(registry: dict, model_config: dict, mode: str, paths: dict[str, Path]) -> pd.DataFrame:
    """Attempt estimator construction for all registered algorithms and save the result."""

    rows = []
    for name, spec in registry.items():
        cfg = model_config.get("models", {}).get(name, {})
        params = dict(cfg.get("parameters", spec.parameters) or {})
        if mode == "smoke":
            params.update(cfg.get("smoke_parameters", {}) or {})
        try:
            spec.factory(params)
            rows.append({"model": name, "construction_ok": True, "exception_type": "", "message": ""})
        except Exception as exc:
            rows.append(
                {
                    "model": name,
                    "construction_ok": False,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(paths["tables"] / "estimator_construction_check.csv", index=False)
    return result


def apply_model_filters(registry: dict, include_models: list[str] | None, exclude_models: list[str] | None) -> dict:
    """Apply CLI model include/exclude filters without mutating the canonical registry."""

    known = set(registry)
    include = set(include_models or [])
    exclude = set(exclude_models or [])
    unknown = sorted((include | exclude) - known)
    if unknown:
        raise ValueError(f"Unknown model names in filters: {unknown}")
    if include:
        selected_names = include - exclude
    else:
        selected_names = {name for name, spec in registry.items() if spec.enabled} - exclude
    return {name: spec for name, spec in registry.items() if name in selected_names and spec.enabled}


def main() -> None:
    """Load configs, validate model catalog, and run the benchmark."""

    args = parse_args()
    experiment_config = load_yaml(Path(args.experiment_config))
    model_config = load_yaml(Path(args.models_config))
    experiment_config["models"] = model_config.get("models", {})

    enabled = {name: bool(cfg.get("enabled", True)) for name, cfg in experiment_config["models"].items()}
    registry = build_model_registry(enabled)
    validate_model_coverage(registry)

    paths = ensure_result_dirs(experiment_config)
    catalog = catalog_table(registry)
    catalog.to_csv(paths["tables"] / "model_catalog.csv", index=False)
    enabled_count = int(catalog["enabled"].sum())
    if len(registry) != 30 or enabled_count != 30:
        raise ValueError(f"Expected exactly 30 enabled requested algorithms; registry={len(registry)} enabled={enabled_count}.")

    selected_registry = apply_model_filters(registry, args.include_models, args.exclude_models)
    excluded_names = sorted(set(registry) - set(selected_registry))
    if not selected_registry:
        raise ValueError("No models selected after applying include/exclude filters.")

    bundle = load_dataset_bundle(experiment_config)
    if args.merge_results:
        merged = merge_result_tables([Path(path) for path in args.merge_results], paths, bundle)
        print(f"Merged result tables: {', '.join(str(path) for path in merged) if merged else 'none'}", flush=True)
        return

    smoke_test_feature_selection(bundle.train)
    construction = smoke_test_estimator_construction(selected_registry, model_config, args.mode, paths)
    constructed = int(construction["construction_ok"].sum())
    print(f"Registered algorithms: {len(registry)} (DummyRegressor is external baseline)")
    print(f"Selected models: {', '.join(selected_registry)}", flush=True)
    print(f"Excluded models: {', '.join(excluded_names) if excluded_names else 'none'}", flush=True)
    print(f"Selected stages: {', '.join(args.stages)}", flush=True)
    print(f"Estimator construction check: {constructed}/{len(selected_registry)} succeeded")
    print(f"Targets: {', '.join(args.targets)}")
    print(f"Mode: {args.mode}")
    run_benchmark(
        selected_registry,
        list(args.targets),
        bundle,
        experiment_config,
        mode=args.mode,
        resume=args.resume,
        stages=tuple(args.stages),
        include_dummy_baseline=not (args.include_models or args.exclude_models),
    )
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
