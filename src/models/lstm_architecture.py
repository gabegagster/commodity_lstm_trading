"""PyTorch LSTM architecture for multi-asset commodity sequence forecasting."""

import os
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Single-thread execution prevents OpenMP / Accelerate thread contention on macOS
torch.set_num_threads(1)


def set_seed(seed: int = 42) -> None:
    """Set random seeds across libraries to ensure reproducible results."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Single thread per process prevents OpenMP / Accelerate thread contention on macOS
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_default_device() -> torch.device:
    """Select the best available device for small financial LSTM forecasting.

    Defaults to CPU because for small sequence networks (e.g. 16-32 hidden units)
    CPU execution has zero kernel launch latency and is significantly faster and more stable
    in sequential walk-forward loops than MPS.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class CommodityLSTM(nn.Module):
    """Small regularized LSTM network for commodity return forecasting.

    Designed specifically for low signal-to-noise financial time series to
    mitigate overfitting via compact capacity and dropout regularization.

    Args:
        input_size: Number of features per time step (e.g. 5 for 5 ETFs).
        hidden_size: Number of hidden units in LSTM (default: 16).
        output_size: Number of target predictions (e.g. 5 for multi-asset returns).
        num_layers: Number of stacked LSTM layers (default: 1).
        dropout: Dropout probability applied to representations (default: 0.2).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 16,
        output_size: int = 5,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_rate = dropout

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier initialization for linear layers and orthogonal for LSTM gates."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name:
                param.data.fill_(0.0)
                # Forget gate bias trick (set forget gate bias to 1.0)
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.0)

        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass of LSTM.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size).
            state: Optional initial (h_0, c_0) state tuple.

        Returns:
            Predicted returns of shape (batch_size, output_size).
        """
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x, state)

        # Extract last time step representation: shape (batch_size, hidden_size)
        last_hidden = lstm_out[:, -1, :]

        # Regularize representation and project to asset predictions
        out = self.drop(last_hidden)
        predictions = self.fc(out)
        return predictions
