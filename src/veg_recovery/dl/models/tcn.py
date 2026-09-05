"""Малый bidirectional residual TCN для интерполяции, не forecasting."""

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TCNConfig:
    input_features: int
    n_crops: int
    hidden_size: int = 64
    layers: int = 3
    kernel_size: int = 5
    dropout: float = 0.1
    crop_embedding_dim: int = 8
    residual_bound: float = 0.3

    def __post_init__(self):
        if (
            self.input_features < 1
            or self.n_crops < 1
            or self.hidden_size < 4
            or not 1 <= self.layers <= 5
            or self.kernel_size < 3
            or self.kernel_size % 2 == 0
            or not 0 <= self.dropout < 1
            or self.crop_embedding_dim < 1
            or not 0 < self.residual_bound <= 2
        ):
            raise ValueError("Invalid TCN configuration")


class _Block(nn.Module):
    def __init__(self, hidden, kernel, dilation, dropout):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.conv = nn.Conv1d(hidden, hidden, kernel, dilation=dilation, padding=pad)
        self.norm = nn.LayerNorm(hidden)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, valid):
        residual = self.conv(x).transpose(1, 2)
        residual = self.dropout(self.activation(self.norm(residual))).transpose(1, 2)
        return (x + residual) * valid


class ResidualTCN(nn.Module):
    def __init__(self, config: TCNConfig):
        super().__init__()
        self.config = config
        self.crop_embedding = nn.Embedding(
            config.n_crops, config.crop_embedding_dim, padding_idx=0
        )
        self.projection = nn.Conv1d(
            config.input_features + config.crop_embedding_dim, config.hidden_size, 1
        )
        self.blocks = nn.ModuleList(
            [
                _Block(config.hidden_size, config.kernel_size, 2**i, config.dropout)
                for i in range(config.layers)
            ]
        )
        self.head = nn.Conv1d(config.hidden_size, 1, 1)
        # Начало обучения точно соответствует безопасной linear base, а не случайной поправке.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features, crop_ids, base, valid_calendar_mask):
        valid = valid_calendar_mask[:, None, :].to(features.dtype)
        embedded = self.crop_embedding(crop_ids)
        x = torch.cat([features, embedded], dim=-1).transpose(1, 2) * valid
        x = self.projection(x) * valid
        for block in self.blocks:
            x = block(x, valid)
        correction = self.config.residual_bound * torch.tanh(self.head(x).squeeze(1))
        return (base + correction) * valid.squeeze(1)

    def configuration(self):
        return asdict(self.config)


def masked_loss(prediction, labels, loss_mask, *, kind="huber"):
    if prediction.shape != labels.shape or labels.shape != loss_mask.shape:
        raise ValueError("Loss shape mismatch")
    if loss_mask.dtype is not torch.bool or not loss_mask.any():
        raise ValueError("Loss requires a nonempty boolean supervision mask")
    # Индексация ДО арифметики: NaN * 0 всё равно NaN и портит градиент.
    y, p = labels[loss_mask], prediction[loss_mask]
    if not torch.isfinite(y).all() or not torch.isfinite(p).all():
        raise ValueError("Nonfinite supervised values")
    if kind == "mse":
        return nn.functional.mse_loss(p, y)
    if kind == "huber":
        return nn.functional.huber_loss(p, y, delta=0.1)
    raise ValueError("Loss must be mse or huber")
