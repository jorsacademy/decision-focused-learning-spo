"""Prediction models used in decision-focused learning experiments."""

from __future__ import annotations

from torch import nn


class LinearCostPredictor(nn.Module):
    """Simple linear cost model, intentionally misspecified for nonlinear synthetic data."""

    def __init__(self, n_features: int, n_items: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, n_items)

    def forward(self, features):
        return self.linear(features)
