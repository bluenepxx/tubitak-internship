"""PyTorch SELFIES Transformer regressor with sklearn-like fit/predict methods."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd

from .selfies_features import PAD_TOKEN, build_vocabulary, length_statistics, numericalize, pad_sequences

UNKNOWN_CATEGORY = "__UNK_CATEGORY__"


@dataclass
class AttentionHistory:
    """Training history returned by the Transformer regressor."""

    records: list[dict[str, float | int]]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)


class SelfiesTransformerRegressor:
    """Sequence model using separate node/linker inputs and categorical embeddings."""

    def __init__(
        self,
        random_state: int = 42,
        embedding_dim: int = 64,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        feedforward_dim: int = 128,
        dropout: float = 0.1,
        hidden_dim: int = 128,
        epochs: int = 30,
        batch_size: int = 64,
        patience: int = 6,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        node_max_length_cap: int = 64,
        linker_max_length_cap: int = 256,
        validation_fraction: float = 0.15,
        device: str = "auto",
        verbose: bool = False,
    ) -> None:
        self.random_state = random_state
        self.embedding_dim = embedding_dim
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.feedforward_dim = feedforward_dim
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.node_max_length_cap = node_max_length_cap
        self.linker_max_length_cap = linker_max_length_cap
        self.validation_fraction = validation_fraction
        self.device = device
        self.verbose = verbose

    def _require_torch(self):
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError("torch is required for SelfiesTransformerRegressor. Install it with `pip install torch`.") from exc
        return torch, nn

    def _set_seed(self, torch) -> None:
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass

    def _device(self, torch):
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _prepare_vocabularies(self, X: pd.DataFrame) -> None:
        self.node_vocab_ = build_vocabulary(X["node_tokens"])
        self.linker_vocab_ = build_vocabulary(X["linker_tokens"])
        self.node_length_stats_ = length_statistics(X["node_tokens"], cap=self.node_max_length_cap)
        self.linker_length_stats_ = length_statistics(X["linker_tokens"], cap=self.linker_max_length_cap)
        self.node_max_length_ = int(self.node_length_stats_["selected_max_length"])
        self.linker_max_length_ = int(self.linker_length_stats_["selected_max_length"])
        self.point_group_vocab_ = self._category_vocab(X["point_group"])
        self.topology_vocab_ = self._category_vocab(X["topology"])

    @staticmethod
    def _category_vocab(values: pd.Series) -> dict[str, int]:
        categories = [UNKNOWN_CATEGORY, *sorted(set(values.astype(str).fillna(UNKNOWN_CATEGORY)))]
        return {category: index for index, category in enumerate(dict.fromkeys(categories))}

    @staticmethod
    def _encode_categories(vocab: dict[str, int], values: pd.Series) -> np.ndarray:
        unknown = vocab[UNKNOWN_CATEGORY]
        return values.astype(str).map(lambda value: vocab.get(value, unknown)).to_numpy(dtype=np.int64)

    def _arrays(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        node_ids, node_mask = pad_sequences(
            [numericalize(tokens, self.node_vocab_) for tokens in X["node_tokens"]],
            self.node_max_length_,
            pad_value=self.node_vocab_[PAD_TOKEN],
        )
        linker_ids, linker_mask = pad_sequences(
            [numericalize(tokens, self.linker_vocab_) for tokens in X["linker_tokens"]],
            self.linker_max_length_,
            pad_value=self.linker_vocab_[PAD_TOKEN],
        )
        return {
            "node_ids": node_ids,
            "node_mask": node_mask,
            "linker_ids": linker_ids,
            "linker_mask": linker_mask,
            "point_group": self._encode_categories(self.point_group_vocab_, X["point_group"]),
            "topology": self._encode_categories(self.topology_vocab_, X["topology"]),
            "topology_missing": pd.to_numeric(X["topology_missing"], errors="coerce").fillna(0).to_numpy(dtype=np.float32),
        }

    def fit(self, X: pd.DataFrame, y, validation_data: tuple[pd.DataFrame, Any] | None = None):
        torch, nn = self._require_torch()
        self._set_seed(torch)
        self.device_ = self._device(torch)
        self._prepare_vocabularies(X)
        arrays = self._arrays(X)
        y_arr = np.asarray(y, dtype=np.float32)
        if validation_data is None:
            rng = np.random.default_rng(self.random_state)
            indices = rng.permutation(len(X))
            val_size = max(1, int(round(len(indices) * self.validation_fraction))) if len(indices) > 3 else 1
            val_idx = indices[:val_size]
            train_idx = indices[val_size:]
            if len(train_idx) == 0:
                train_idx = val_idx
        else:
            train_idx = np.arange(len(X))
            X_val, y_val = validation_data
            validation_arrays = self._arrays(X_val)
            validation_y = np.asarray(y_val, dtype=np.float32)

        if validation_data is None:
            validation_arrays = {key: value[val_idx] for key, value in arrays.items()}
            validation_y = y_arr[val_idx]
        train_arrays = {key: value[train_idx] for key, value in arrays.items()}
        train_y = y_arr[train_idx]

        self.model_ = _TransformerModule(
            torch=torch,
            nn=nn,
            node_vocab_size=len(self.node_vocab_),
            linker_vocab_size=len(self.linker_vocab_),
            point_group_count=len(self.point_group_vocab_),
            topology_count=len(self.topology_vocab_),
            embedding_dim=self.embedding_dim,
            heads=self.transformer_heads,
            layers=self.transformer_layers,
            feedforward_dim=self.feedforward_dim,
            dropout=self.dropout,
            hidden_dim=self.hidden_dim,
        ).to(self.device_)

        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        self.history_ = AttentionHistory(records=[])
        best_state = None
        best_val = float("inf")
        stale_epochs = 0

        for epoch in range(1, self.epochs + 1):
            train_loss = self._run_epoch(torch, train_arrays, train_y, optimizer, loss_fn, training=True)
            val_loss = self._run_epoch(torch, validation_arrays, validation_y, optimizer, loss_fn, training=False)
            self.history_.records.append({"epoch": epoch, "train_loss": float(train_loss), "validation_loss": float(val_loss)})
            if self.verbose:
                print(f"SELFIES Transformer epoch {epoch}: train={train_loss:.5f} validation={val_loss:.5f}")
            if val_loss < best_val:
                best_val = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in self.model_.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.n_iter_ = len(self.history_.records)
        self.best_validation_loss_ = float(best_val)
        self.device_report_ = str(self.device_)
        return self

    def _tensor_batch(self, torch, arrays: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, Any]:
        return {
            "node_ids": torch.as_tensor(arrays["node_ids"][indices], dtype=torch.long, device=self.device_),
            "node_mask": torch.as_tensor(arrays["node_mask"][indices], dtype=torch.bool, device=self.device_),
            "linker_ids": torch.as_tensor(arrays["linker_ids"][indices], dtype=torch.long, device=self.device_),
            "linker_mask": torch.as_tensor(arrays["linker_mask"][indices], dtype=torch.bool, device=self.device_),
            "point_group": torch.as_tensor(arrays["point_group"][indices], dtype=torch.long, device=self.device_),
            "topology": torch.as_tensor(arrays["topology"][indices], dtype=torch.long, device=self.device_),
            "topology_missing": torch.as_tensor(arrays["topology_missing"][indices], dtype=torch.float32, device=self.device_).unsqueeze(1),
        }

    def _run_epoch(self, torch, arrays, y, optimizer, loss_fn, training: bool) -> float:
        self.model_.train(training)
        indices = np.arange(len(y))
        if training:
            np.random.shuffle(indices)
        losses: list[float] = []
        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            batch = self._tensor_batch(torch, arrays, batch_idx)
            target = torch.as_tensor(y[batch_idx], dtype=torch.float32, device=self.device_).unsqueeze(1)
            with torch.set_grad_enabled(training):
                prediction = self.model_(**batch)
                loss = loss_fn(prediction, target)
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        return float(np.mean(losses))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        torch, _ = self._require_torch()
        arrays = self._arrays(X)
        self.model_.eval()
        indices = np.arange(len(X))
        preds: list[np.ndarray] = []
        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            batch = self._tensor_batch(torch, arrays, batch_idx)
            with torch.no_grad():
                pred = self.model_(**batch).detach().cpu().numpy().reshape(-1)
            preds.append(pred)
        return np.concatenate(preds) if preds else np.array([], dtype=float)

    def save_history(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.history_.to_frame().to_csv(output, index=False)

    def save_checkpoint(self, path: str | Path) -> None:
        """Persist model weights and preprocessing vocabularies without pickling the dynamic module."""

        torch, _ = self._require_torch()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model_.state_dict(),
                "node_vocab": self.node_vocab_,
                "linker_vocab": self.linker_vocab_,
                "point_group_vocab": self.point_group_vocab_,
                "topology_vocab": self.topology_vocab_,
                "node_max_length": self.node_max_length_,
                "linker_max_length": self.linker_max_length_,
                "params": self.get_params(),
                "history": self.history_.records,
            },
            output,
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return sklearn-style constructor parameters."""

        return {
            "random_state": self.random_state,
            "embedding_dim": self.embedding_dim,
            "transformer_heads": self.transformer_heads,
            "transformer_layers": self.transformer_layers,
            "feedforward_dim": self.feedforward_dim,
            "dropout": self.dropout,
            "hidden_dim": self.hidden_dim,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "node_max_length_cap": self.node_max_length_cap,
            "linker_max_length_cap": self.linker_max_length_cap,
            "validation_fraction": self.validation_fraction,
            "device": self.device,
            "verbose": self.verbose,
        }


class _TransformerModule:
    """Small wrapper to construct the actual torch.nn.Module after torch import."""

    def __new__(
        cls,
        torch,
        nn,
        node_vocab_size: int,
        linker_vocab_size: int,
        point_group_count: int,
        topology_count: int,
        embedding_dim: int,
        heads: int,
        layers: int,
        feedforward_dim: int,
        dropout: float,
        hidden_dim: int,
    ):
        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.node_embedding = nn.Embedding(node_vocab_size, embedding_dim, padding_idx=0)
                self.linker_embedding = nn.Embedding(linker_vocab_size, embedding_dim, padding_idx=0)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=heads,
                    dim_feedforward=feedforward_dim,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                )
                self.node_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
                linker_layer = nn.TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=heads,
                    dim_feedforward=feedforward_dim,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                )
                self.linker_encoder = nn.TransformerEncoder(linker_layer, num_layers=layers)
                self.point_group_embedding = nn.Embedding(max(point_group_count, 1), embedding_dim // 2)
                self.topology_embedding = nn.Embedding(max(topology_count, 1), embedding_dim // 2)
                combined = embedding_dim * 2 + embedding_dim + 1
                self.head = nn.Sequential(
                    nn.Linear(combined, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 2, 1),
                )

            @staticmethod
            def masked_mean(encoded, mask):
                weights = mask.unsqueeze(-1).float()
                summed = (encoded * weights).sum(dim=1)
                denom = weights.sum(dim=1).clamp_min(1.0)
                return summed / denom

            def forward(self, node_ids, node_mask, linker_ids, linker_mask, point_group, topology, topology_missing):
                node_emb = self.node_embedding(node_ids)
                linker_emb = self.linker_embedding(linker_ids)
                node_encoded = self.node_encoder(node_emb, src_key_padding_mask=~node_mask)
                linker_encoded = self.linker_encoder(linker_emb, src_key_padding_mask=~linker_mask)
                node_repr = self.masked_mean(node_encoded, node_mask)
                linker_repr = self.masked_mean(linker_encoded, linker_mask)
                categorical_repr = torch.cat(
                    [self.point_group_embedding(point_group), self.topology_embedding(topology)],
                    dim=1,
                )
                features = torch.cat([node_repr, linker_repr, categorical_repr, topology_missing], dim=1)
                return self.head(features)

        return Module()
