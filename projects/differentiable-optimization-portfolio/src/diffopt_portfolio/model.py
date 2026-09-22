from __future__ import annotations

import torch
from torch import nn

from .layer import MeanVarianceLayer


class EndToEndPortfolio(nn.Module):
    def __init__(self, n_features: int, n_assets: int, layer: MeanVarianceLayer):
        super().__init__()
        self.forecaster = nn.Linear(n_features, n_assets)
        self.optimization = layer

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        mu = self.forecaster(features)
        return self.optimization(mu)


def realized_mean_variance_loss(
    weights: torch.Tensor,
    realized_returns: torch.Tensor,
    risk_penalty: float = 0.0,
) -> torch.Tensor:
    pnl = (weights * realized_returns).sum(dim=-1)
    risk = weights.square().sum(dim=-1)
    return (-pnl + risk_penalty * risk).mean()
