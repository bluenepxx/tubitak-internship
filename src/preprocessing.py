"""Leakage-safe preprocessing builders for classical regressors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURE_COLUMNS = ["node_token_text", "linker_token_text", "point_group", "topology", "topology_missing"]


@dataclass(frozen=True)
class DenseGuard:
    """Upper bound for guarded sparse-to-dense conversion."""

    max_dense_mb: float = 1024.0


class SparseToDenseTransformer(BaseEstimator, TransformerMixin):
    """Convert sparse matrices to dense only after estimating memory use."""

    def __init__(self, max_dense_mb: float = 1024.0) -> None:
        self.max_dense_mb = max_dense_mb

    def fit(self, X, y=None):  # noqa: N803
        return self

    def transform(self, X):  # noqa: N803
        if not sparse.issparse(X):
            return X
        estimated_mb = (X.shape[0] * X.shape[1] * np.dtype(np.float64).itemsize) / (1024**2)
        if estimated_mb > self.max_dense_mb:
            raise MemoryError(
                f"Dense conversion would require about {estimated_mb:.1f} MB, "
                f"above configured limit {self.max_dense_mb:.1f} MB."
            )
        return X.toarray()


def make_tfidf_vectorizer(min_df: int = 2) -> TfidfVectorizer:
    """Notebook-compatible SELFIES TF-IDF vectorizer."""

    return TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        # unigrams + bigrams over SELFIES tokens, min_df=2 to drop one-off tokens.
        # Chosen as a practical default, no separate grid search over these two.
        ngram_range=(1, 2),
        min_df=min_df,
    )


def make_one_hot_encoder() -> OneHotEncoder:
    """Create a sparse OneHotEncoder compatible with modern sklearn."""

    return OneHotEncoder(handle_unknown="ignore", sparse_output=True)


def build_feature_preprocessor(min_df: int = 2) -> ColumnTransformer:
    """Build separate node/linker TF-IDF and categorical preprocessing."""

    return ColumnTransformer(
        transformers=[
            ("node_tfidf", make_tfidf_vectorizer(min_df=min_df), "node_token_text"),
            ("linker_tfidf", make_tfidf_vectorizer(min_df=min_df), "linker_token_text"),
            ("point_group", make_one_hot_encoder(), ["point_group"]),
            ("topology", make_one_hot_encoder(), ["topology"]),
            ("topology_missing", "passthrough", ["topology_missing"]),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def build_classical_pipeline(
    estimator,
    requires_dense: bool,
    requires_scaling: bool,
    min_df: int = 2,
    max_dense_mb: float = 1024.0,
) -> Pipeline:
    """Assemble a leakage-safe sklearn Pipeline for one estimator."""

    steps = [("preprocessor", build_feature_preprocessor(min_df=min_df))]
    if requires_dense:
        steps.append(("to_dense", SparseToDenseTransformer(max_dense_mb=max_dense_mb)))
    if requires_scaling:
        steps.append(("scaler", StandardScaler(with_mean=requires_dense)))
    steps.append(("model", estimator))
    return Pipeline(steps=steps)
