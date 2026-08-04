"""Data loading, schema validation, split management, and dataset summaries."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

from .imputation import handle_input_missingness

TARGET_COLUMNS = ("density", "pld", "lcd")
FEATURE_COLUMNS = ("node_token_text", "linker_token_text", "point_group", "topology", "topology_missing")
REQUIRED_COLUMNS = (
    "qmof_id",
    "node_tokens",
    "linker_tokens",
    "node_token_text",
    "linker_token_text",
    "point_group",
    "topology",
    "topology_missing",
    *TARGET_COLUMNS,
)
SPLIT_LABELS = ("train", "validation", "test")


@dataclass(frozen=True)
class DatasetBundle:
    """Loaded data and leakage-safe split views."""

    df: pd.DataFrame
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    split_assignments: pd.DataFrame
    dataset_summary: pd.DataFrame
    split_summary: pd.DataFrame
    dataset_hash: str
    split_hash: str


def select_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Select multiple columns safely, converting tuple constants at the pandas boundary."""

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(
            f"Missing required columns: {missing}. "
            f"Available columns: {list(frame.columns)}"
        )
    return frame.loc[:, list(columns)].copy()


def select_target(frame: pd.DataFrame, target: str) -> pd.Series:
    """Select one target column with the same clear missing-column error."""

    return select_columns(frame, [target]).iloc[:, 0]


def project_root() -> Path:
    """Resolve the repository root from this module location."""

    return Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path, root: Path | None = None) -> Path:
    """Resolve a config path relative to the project root."""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else (root or project_root()) / candidate


def load_jsonl_dataset(path: str | Path) -> pd.DataFrame:
    """Load the prepared JSONL forward-model dataset."""

    dataset_path = resolve_project_path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {dataset_path}")
    return pd.read_json(dataset_path, lines=True)


def validate_schema(df: pd.DataFrame, required_columns: tuple[str, ...] = REQUIRED_COLUMNS) -> None:
    """Validate required columns, duplicate ids, and target availability."""

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Prepared dataset is missing required columns: {missing}")
    if df["qmof_id"].duplicated().any():
        examples = df.loc[df["qmof_id"].duplicated(keep=False), "qmof_id"].astype(str).head(10).tolist()
        raise ValueError(f"Duplicate qmof_id values found in prepared dataset: {examples}")
    for target in TARGET_COLUMNS:
        if df[target].isna().any():
            examples = df.loc[df[target].isna(), "qmof_id"].astype(str).head(10).tolist()
            raise ValueError(f"Target {target} contains missing values; targets are never imputed. Examples: {examples}")


def create_input_signature(df: pd.DataFrame) -> pd.Series:
    """Hash the exact model inputs used by GroupKFold leakage checks."""

    required = ["node_token_text", "linker_token_text", "point_group", "topology"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Cannot build exact input signatures; missing columns: {missing}")

    def signature(row: pd.Series) -> str:
        text = "\u241f".join(str(row[column]) for column in required)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    return select_columns(df, required).apply(signature, axis=1)


def dataframe_hash(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    """Create a deterministic content hash for a DataFrame."""

    data = select_columns(frame, columns) if columns is not None else frame.copy()
    data = data.sort_index(axis=1)
    if "qmof_id" in data.columns:
        data = data.sort_values("qmof_id").reset_index(drop=True)
    elif {"qmof_id", "split"}.issubset(data.columns):
        data = data.sort_values(["qmof_id", "split"]).reset_index(drop=True)
    else:
        data = data.reset_index(drop=True)
    payload = data.to_json(orient="split", date_format="iso", force_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dataset_content_hash(df: pd.DataFrame) -> str:
    """Hash the model-relevant dataset content independently of split labels."""

    columns = [
        "qmof_id",
        "node_token_text",
        "linker_token_text",
        "point_group",
        "topology",
        "topology_missing",
        *TARGET_COLUMNS,
    ]
    return dataframe_hash(df, columns)


def split_content_hash(split_assignments: pd.DataFrame) -> str:
    """Hash qmof_id to split assignments."""

    return dataframe_hash(split_assignments, ["qmof_id", "split"])


def load_or_create_splits(
    df: pd.DataFrame,
    split_path: str | Path,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Reuse qmof_id-based split assignments or create deterministic 70/15/15 splits."""

    path = resolve_project_path(split_path)
    if path.exists():
        splits = pd.read_csv(path)
        missing = {"qmof_id", "split"} - set(splits.columns)
        if missing:
            raise ValueError(f"Split file {path} is missing columns: {sorted(missing)}")
        if splits["qmof_id"].duplicated().any():
            raise ValueError(f"Split file {path} contains duplicate qmof_id values.")
        merged = select_columns(df, ["qmof_id"]).merge(
            select_columns(splits, ["qmof_id", "split"]),
            on="qmof_id",
            how="left",
            validate="one_to_one",
        )
        if merged["split"].isna().any():
            examples = merged.loc[merged["split"].isna(), "qmof_id"].astype(str).head(10).tolist()
            raise ValueError(f"Split file does not cover all dataset qmof_id values. Examples: {examples}")
        invalid = sorted(set(merged["split"]) - set(SPLIT_LABELS))
        if invalid:
            raise ValueError(f"Split file contains invalid split labels: {invalid}")
        if logger:
            logger.info("Loaded existing split assignments from %s", path)
        return merged

    if abs((train_ratio + validation_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("Train/validation/test ratios must sum to 1.0")

    ids = df["qmof_id"].to_numpy()
    train_ids, temporary_ids = train_test_split(ids, test_size=1.0 - train_ratio, random_state=random_state, shuffle=True)
    validation_fraction = validation_ratio / (validation_ratio + test_ratio)
    validation_ids, test_ids = train_test_split(
        temporary_ids,
        test_size=1.0 - validation_fraction,
        random_state=random_state,
        shuffle=True,
    )
    assignment = pd.DataFrame({"qmof_id": ids, "split": "unassigned"})
    assignment.loc[assignment["qmof_id"].isin(train_ids), "split"] = "train"
    assignment.loc[assignment["qmof_id"].isin(validation_ids), "split"] = "validation"
    assignment.loc[assignment["qmof_id"].isin(test_ids), "split"] = "test"
    path.parent.mkdir(parents=True, exist_ok=True)
    assignment.to_csv(path, index=False)
    if logger:
        logger.info("Created deterministic split assignments at %s", path)
    return assignment


def summarize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact machine-readable dataset summary table."""

    records: list[dict[str, object]] = [
        {"metric": "row_count", "value": int(len(df))},
        {"metric": "unique_qmof_id", "value": int(df["qmof_id"].nunique())},
        {"metric": "duplicate_qmof_id", "value": int(df["qmof_id"].duplicated().sum())},
        {"metric": "unique_input_signatures", "value": int(df["input_signature"].nunique())},
    ]
    for target in TARGET_COLUMNS:
        records.extend(
            [
                {"metric": f"{target}_missing", "value": int(df[target].isna().sum())},
                {"metric": f"{target}_mean", "value": float(df[target].mean())},
                {"metric": f"{target}_std", "value": float(df[target].std(ddof=1))},
            ]
        )
    return pd.DataFrame(records)


def summarize_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize split sizes and duplicate-signature pressure by split."""

    records: list[dict[str, object]] = []
    for split in SPLIT_LABELS:
        part = df[df["split"] == split]
        records.append(
            {
                "split": split,
                "rows": int(len(part)),
                "unique_qmof_id": int(part["qmof_id"].nunique()),
                "unique_input_signatures": int(part["input_signature"].nunique()),
            }
        )
    return pd.DataFrame(records)


def load_dataset_bundle(config: dict, logger: logging.Logger | None = None) -> DatasetBundle:
    """Load, validate, sanitize, split, and summarize the prepared dataset."""

    data_cfg = config.get("data", {})
    split_cfg = config.get("splits", {})
    df = load_jsonl_dataset(data_cfg.get("dataset_path", "data/processed/forward_model_selfies.jsonl"))
    validate_schema(df)
    df, report = handle_input_missingness(df, logger=logger)
    if logger:
        logger.info("Input imputation report: %s", report.as_dict())
    df["input_signature"] = create_input_signature(df)

    splits = load_or_create_splits(
        df,
        data_cfg.get("split_path", "data/processed/split_assignments.csv"),
        train_ratio=float(split_cfg.get("train_ratio", 0.70)),
        validation_ratio=float(split_cfg.get("validation_ratio", 0.15)),
        test_ratio=float(split_cfg.get("test_ratio", 0.15)),
        random_state=int(config.get("random_seed", 42)),
        logger=logger,
    )
    df = df.merge(splits, on="qmof_id", how="left", validate="one_to_one")
    if df["split"].isna().any():
        raise ValueError("Internal split merge failed; some rows are unassigned.")

    return DatasetBundle(
        df=df,
        train=df[df["split"] == "train"].copy(),
        validation=df[df["split"] == "validation"].copy(),
        test=df[df["split"] == "test"].copy(),
        split_assignments=splits,
        dataset_summary=summarize_dataset(df),
        split_summary=summarize_splits(df),
        dataset_hash=dataset_content_hash(df),
        split_hash=split_content_hash(splits),
    )
