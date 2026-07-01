from __future__ import annotations

import math

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, horizon: int, hidden: int = 48, layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.head(hidden[-1])


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        positions = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        horizon: int,
        dim: int = 48,
        heads: int = 4,
        layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(n_features, dim)
        self.pos = PositionalEncoding(dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(self.pos(self.proj(x)))
        return self.head(z[:, -1])


class CNNTransformerForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        horizon: int,
        dim: int = 48,
        heads: int = 4,
        layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv1d(n_features, dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.pos = PositionalEncoding(dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.local(x.transpose(1, 2)).transpose(1, 2)
        z = self.encoder(self.pos(z))
        pooled = z.mean(dim=1)
        gated = pooled * self.gate(z[:, -1])
        return self.head(gated)
