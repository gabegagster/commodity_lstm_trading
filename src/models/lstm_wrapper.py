"""Scikit-learn compatible estimator wrapper for PyTorch LSTM models."""

import copy
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.lstm_architecture import CommodityLSTM, get_default_device, set_seed


def _create_sliding_sequences(
    X: np.ndarray,
    y: Optional[np.ndarray],
    lookback: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Build 3D sliding sequence inputs from 2D time series array.

    For index t >= lookback - 1, the sequence is X[t - lookback + 1 : t + 1].
    The corresponding target is y[t].

    Args:
        X: 2D feature matrix of shape (T, num_features).
        y: Optional 2D target matrix of shape (T, num_outputs).
        lookback: Number of historical time steps in each sequence.

    Returns:
        X_seq: 3D array of shape (T - lookback + 1, lookback, num_features).
        y_seq: 2D array of shape (T - lookback + 1, num_outputs), or None.
    """
    n_samples, n_features = X.shape
    if n_samples < lookback:
        raise ValueError(
            f"Input samples ({n_samples}) must be at least lookback window ({lookback})."
        )

    n_sequences = n_samples - lookback + 1
    # Vectorized / efficient strided sequence extraction
    X_seq = np.empty((n_sequences, lookback, n_features), dtype=np.float32)
    for i in range(n_sequences):
        X_seq[i] = X[i : i + lookback]

    if y is not None:
        y_seq = np.asarray(y[lookback - 1 :], dtype=np.float32)
        if y_seq.ndim == 1:
            y_seq = y_seq[:, np.newaxis]
    else:
        y_seq = None

    return X_seq, y_seq


class PyTorchLSTMRegressor(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible wrapper for CommodityLSTM.

    Integrates into skfin's Backtester and sklearn's Pipeline (implements
    fit, predict, and transform). Buffers the trailing window of training data
    so that walk-forward test splits (where test_size < lookback) are automatically
    evaluated with full historical context and zero look-ahead bias.

    Args:
        hidden_size: Hidden units in LSTM (default: 16).
        num_layers: Number of LSTM layers (default: 1).
        lookback: Historical sequence length in time steps (default: 60).
        dropout: Dropout rate (default: 0.2).
        lr: Learning rate for Adam optimizer (default: 1e-3).
        weight_decay: L2 regularization penalty (default: 1e-4).
        batch_size: Mini-batch size for training (default: 32).
        epochs: Maximum training epochs per walk-forward fold (default: 40).
        loss_type: 'huber' for fat-tailed robustness or 'mse' (default: 'huber').
        huber_delta: Delta threshold for Huber loss (default: 0.01).
        val_split: Fraction of chronological training data for validation early stopping (default: 0.15).
        patience: Epochs to wait for validation improvement before early stopping (default: 7).
        random_state: Random seed for initialization (default: 42).
        device: Device to run PyTorch model ('cpu', 'cuda', 'mps', or None for auto).
        verbose: Print training progress if True.
    """

    def __init__(
        self,
        hidden_size: int = 16,
        num_layers: int = 1,
        lookback: int = 60,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        epochs: int = 40,
        loss_type: Literal["huber", "mse"] = "huber",
        huber_delta: float = 0.01,
        val_split: float = 0.15,
        patience: int = 7,
        random_state: int = 42,
        device: Optional[str] = None,
        verbose: bool = False,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lookback = lookback
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        self.val_split = val_split
        self.patience = patience
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def _resolve_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return get_default_device()

    def _init_criterion(self) -> nn.Module:
        if self.loss_type == "huber":
            return nn.HuberLoss(delta=self.huber_delta)
        elif self.loss_type == "mse":
            return nn.MSELoss()
        raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.DataFrame, pd.Series]):
        """Fit LSTM on training features and targets.

        Converts 2D tabular features into sliding sequences. Allocates the final
        val_split portion chronologically (without shuffling) for early stopping.
        Buffers trailing lookback rows in self.last_train_X_ for subsequent test predictions.
        """
        set_seed(self.random_state)
        dev = self._resolve_device()

        X_arr = X.values if isinstance(X, (pd.DataFrame, pd.Series)) else np.asarray(X)
        y_arr = y.values if isinstance(y, (pd.DataFrame, pd.Series)) else np.asarray(y)

        if X_arr.ndim == 1:
            X_arr = X_arr[:, np.newaxis]
        if y_arr.ndim == 1:
            y_arr = y_arr[:, np.newaxis]

        # Ensure C-contiguous float32 arrays for PyTorch C++ LSTM stability
        X_arr = np.ascontiguousarray(X_arr, dtype=np.float32)
        y_arr = np.ascontiguousarray(y_arr, dtype=np.float32)

        self.n_features_in_ = X_arr.shape[1]
        self.n_outputs_ = y_arr.shape[1]

        # Buffer trailing rows for walk-forward test predictions
        self.last_train_X_ = X_arr[-self.lookback :].copy()

        # Build sliding sequences
        X_seq, y_seq = _create_sliding_sequences(X_arr, y_arr, self.lookback)
        n_total = len(X_seq)

        # Chronological validation split (no shuffling to preserve time ordering)
        if self.val_split > 0 and n_total > 10:
            n_val = max(1, int(n_total * self.val_split))
            n_train = n_total - n_val
            X_train_seq, y_train_seq = X_seq[:n_train], y_seq[:n_train]
            X_val_seq, y_val_seq = X_seq[n_train:], y_seq[n_train:]
        else:
            X_train_seq, y_train_seq = X_seq, y_seq
            X_val_seq, y_val_seq = None, None

        train_dataset = TensorDataset(
            torch.from_numpy(X_train_seq).float(),
            torch.from_numpy(y_train_seq).float(),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, len(train_dataset)),
            shuffle=True,  # Shuffle mini-batches within train slice to break autocorrelation
        )

        # Initialize PyTorch network
        self.model_ = CommodityLSTM(
            input_size=self.n_features_in_,
            hidden_size=self.hidden_size,
            output_size=self.n_outputs_,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(dev)

        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        criterion = self._init_criterion()

        best_val_loss = float("inf")
        best_weights = copy.deepcopy(self.model_.state_dict())
        patience_counter = 0

        self.train_loss_history_ = []
        self.val_loss_history_ = []

        self.model_.train()
        for epoch in range(self.epochs):
            batch_losses = []
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(dev)
                batch_y = batch_y.to(dev)

                optimizer.zero_grad()
                pred = self.model_(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()

                # Gradient clipping to stabilize recurrent updates
                nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                optimizer.step()
                batch_losses.append(loss.item())

            train_epoch_loss = float(np.mean(batch_losses))
            self.train_loss_history_.append(train_epoch_loss)

            # Validation step
            if X_val_seq is not None:
                self.model_.eval()
                with torch.no_grad():
                    vx = torch.from_numpy(X_val_seq).float().to(dev)
                    vy = torch.from_numpy(y_val_seq).float().to(dev)
                    val_pred = self.model_(vx)
                    val_loss = criterion(val_pred, vy).item()
                self.val_loss_history_.append(val_loss)
                scheduler.step(val_loss)
                self.model_.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = copy.deepcopy(self.model_.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                best_weights = copy.deepcopy(self.model_.state_dict())

        # Load best weights
        self.model_.load_state_dict(best_weights)
        self.model_.eval()
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict expected returns for test samples.

        In walk-forward cross-validation (e.g. skfin Backtester), test splits
        contain out-of-sample data whose length is often shorter than lookback.
        The wrapper prepends buffered training history (self.last_train_X_)
        so that every test point has a complete lookback window.

        Args:
            X: Test feature matrix.

        Returns:
            2D numpy array of predicted returns, shape (len(X), n_outputs_).
        """
        dev = self._resolve_device()
        X_arr = X.values if isinstance(X, (pd.DataFrame, pd.Series)) else np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, np.newaxis]

        X_arr = np.ascontiguousarray(X_arr, dtype=np.float32)
        n_test = len(X_arr)

        if hasattr(self, "last_train_X_") and self.last_train_X_ is not None:
            # Prepend buffered training history
            X_full = np.ascontiguousarray(
                np.concatenate([self.last_train_X_, X_arr], axis=0), dtype=np.float32
            )
            X_seq, _ = _create_sliding_sequences(X_full, y=None, lookback=self.lookback)
            # The last n_test sequences correspond to the test steps
            X_seq = X_seq[-n_test:]
        else:
            if n_test < self.lookback:
                raise ValueError(
                    f"Test length ({n_test}) < lookback ({self.lookback}) and no trailing history buffered."
                )
            X_seq, _ = _create_sliding_sequences(X_arr, y=None, lookback=self.lookback)

        self.model_.eval()
        with torch.no_grad():
            x_tensor = torch.from_numpy(X_seq).float().to(dev)
            preds = self.model_(x_tensor).cpu().numpy()

        return preds

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Alias predict to transform to enable native sklearn Pipeline integration."""
        return self.predict(X)
