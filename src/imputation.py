"""Input-only missing-data handling for the QMOF regression benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterable

import pandas as pd

MISSING_TOPOLOGY = "__MISSING_TOPOLOGY__"
MISSING_POINT_GROUP = "__MISSING_POINT_GROUP__"
TOKEN_COLUMNS = ("node_tokens", "linker_tokens")
TOKEN_TEXT_COLUMNS = ("node_token_text", "linker_token_text")


@dataclass(frozen=True)
class ImputationReport:
    """Counts of input values changed during input sanitation."""

    topology_filled: int
    point_group_filled: int
    topology_missing_created: bool
    topology_missing_positive: int

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "topology_filled": self.topology_filled,
            "point_group_filled": self.point_group_filled,
            "topology_missing_created": self.topology_missing_created,
            "topology_missing_positive": self.topology_missing_positive,
        }


def _is_missing_text(series: pd.Series) -> pd.Series:
    text = series.astype("string")
    return series.isna() | text.str.strip().str.lower().isin({"", "nan", "none", "null", "-1"})


def validate_token_columns(df: pd.DataFrame, columns: Iterable[str] = TOKEN_COLUMNS) -> None:
    """Fail clearly if node/linker token lists are missing or empty."""

    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required token columns: {missing_columns}")

    bad_rows: dict[str, list[str]] = {}
    for column in columns:
        bad_mask = df[column].isna() | ~df[column].map(lambda value: isinstance(value, list) and len(value) > 0)
        if bad_mask.any():
            bad_rows[column] = df.loc[bad_mask, "qmof_id"].astype(str).head(10).tolist()

    if bad_rows:
        raise ValueError(
            "Invalid or empty SELFIES token lists found. The benchmark does not invent chemistry. "
            f"Examples by column: {bad_rows}"
        )


def ensure_token_text(df: pd.DataFrame) -> pd.DataFrame:
    """Create token text columns from validated token lists when needed."""

    validate_token_columns(df)
    out = df.copy()
    for token_column, text_column in zip(TOKEN_COLUMNS, TOKEN_TEXT_COLUMNS):
        if text_column not in out.columns or out[text_column].isna().any():
            out[text_column] = out[token_column].map(lambda tokens: " ".join(tokens))
        empty_mask = out[text_column].astype("string").str.strip().eq("") | out[text_column].isna()
        if empty_mask.any():
            ids = out.loc[empty_mask, "qmof_id"].astype(str).head(10).tolist()
            raise ValueError(f"Empty {text_column} values after token-text validation. Example qmof_id values: {ids}")
    return out


def handle_input_missingness(df: pd.DataFrame, logger: logging.Logger | None = None) -> tuple[pd.DataFrame, ImputationReport]:
    """Fill only input categorical values and validate chemistry token fields.

    Target columns are intentionally untouched.
    """

    out = ensure_token_text(df)

    topology_missing_before = _is_missing_text(out["topology"]) if "topology" in out.columns else pd.Series(True, index=out.index)
    point_group_missing_before = (
        _is_missing_text(out["point_group"]) if "point_group" in out.columns else pd.Series(True, index=out.index)
    )

    out["topology"] = out.get("topology", pd.Series(index=out.index, dtype="object")).astype("object")
    out.loc[topology_missing_before, "topology"] = MISSING_TOPOLOGY

    out["point_group"] = out.get("point_group", pd.Series(index=out.index, dtype="object")).astype("object")
    out.loc[point_group_missing_before, "point_group"] = MISSING_POINT_GROUP

    created_indicator = "topology_missing" not in out.columns
    if created_indicator:
        out["topology_missing"] = topology_missing_before.astype(int)
    else:
        indicator = pd.to_numeric(out["topology_missing"], errors="coerce")
        out["topology_missing"] = indicator.fillna(topology_missing_before.astype(int)).astype(int).clip(0, 1)
        out.loc[topology_missing_before, "topology_missing"] = 1

    report = ImputationReport(
        topology_filled=int(topology_missing_before.sum()),
        point_group_filled=int(point_group_missing_before.sum()),
        topology_missing_created=created_indicator,
        topology_missing_positive=int(out["topology_missing"].sum()),
    )

    if logger:
        logger.info("Input missingness handled: %s", report.as_dict())

    return out, report
