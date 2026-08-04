"""Reusable SELFIES token utilities for classical and neural benchmarks."""

from __future__ import annotations

from collections import Counter
import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
MOL_SEP_TOKEN = "<MOL_SEP>"
SPECIAL_TOKENS = (PAD_TOKEN, UNK_TOKEN, MOL_SEP_TOKEN)


def validate_token_list(value: object, column: str = "tokens") -> list[str]:
    """Return a clean token list or raise a chemistry-preserving validation error."""

    if not isinstance(value, list) or not value:
        raise ValueError(f"{column} must be a non-empty list of SELFIES tokens.")
    tokens = [str(token).strip() for token in value]
    if any(not token or token.lower() in {"nan", "none", "null"} for token in tokens):
        raise ValueError(f"{column} contains empty or null-like SELFIES tokens.")
    return tokens


def token_lists_to_text(values: Iterable[Sequence[str]]) -> list[str]:
    """Convert token lists into whitespace-separated text for TF-IDF."""

    return [" ".join(validate_token_list(list(tokens))) for tokens in values]


def build_vocabulary(token_lists: Iterable[Sequence[str]], min_freq: int = 1) -> dict[str, int]:
    """Build a train-only vocabulary with stable special-token indices."""

    counter: Counter[str] = Counter()
    for tokens in token_lists:
        counter.update(validate_token_list(list(tokens)))

    vocab = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    for token, count in sorted(counter.items()):
        if count >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def numericalize(tokens: Sequence[str], vocab: dict[str, int]) -> list[int]:
    """Map tokens to integer ids, using ``<UNK>`` for unseen validation/test tokens."""

    unk = vocab[UNK_TOKEN]
    return [vocab.get(token, unk) for token in validate_token_list(list(tokens))]


def pad_sequences(sequences: Sequence[Sequence[int]], max_length: int, pad_value: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Pad/truncate integer sequences and return both ids and boolean attention masks."""

    ids = np.full((len(sequences), max_length), pad_value, dtype=np.int64)
    masks = np.zeros((len(sequences), max_length), dtype=bool)
    for row, sequence in enumerate(sequences):
        clipped = list(sequence)[:max_length]
        if clipped:
            ids[row, : len(clipped)] = clipped
            masks[row, : len(clipped)] = True
    return ids, masks


def length_statistics(token_lists: Iterable[Sequence[str]], percentile: float = 99.0, cap: int = 256) -> dict[str, float | int]:
    """Compute token-length statistics from training data only."""

    lengths = np.array([len(validate_token_list(list(tokens))) for tokens in token_lists], dtype=float)
    if len(lengths) == 0:
        raise ValueError("Cannot compute token length statistics for an empty training split.")
    percentile_length = int(math.ceil(float(np.percentile(lengths, percentile))))
    max_length = max(1, min(percentile_length, cap))
    return {
        "count": int(len(lengths)),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "p99": float(np.percentile(lengths, percentile)),
        "max_seen": int(lengths.max()),
        "selected_max_length": int(max_length),
    }
