"""Synthetic contextual cost data for predict-then-optimize experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class DatasetSplit:
    features: Tensor
    costs: Tensor


def generate_contextual_costs(
    n_samples: int,
    n_features: int,
    n_items: int,
    *,
    seed: int = 0,
    noise_std: float = 0.25,
    nonlinear_strength: float = 0.8,
) -> DatasetSplit:
    """Generate features and mildly misspecified nonlinear item costs."""
    if n_samples <= 0 or n_features <= 0 or n_items <= 1:
        raise ValueError("n_samples/n_features must be positive and n_items must exceed one")
    if noise_std < 0 or nonlinear_strength < 0:
        raise ValueError("noise_std and nonlinear_strength must be nonnegative")

    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, n_features, generator=g)
    weights = torch.randn(n_features, n_items, generator=g)
    linear = x @ weights
    nonlinear = nonlinear_strength * torch.sin(linear)
    noise = noise_std * torch.randn(n_samples, n_items, generator=g)
    item_offsets = torch.linspace(-0.5, 0.5, n_items)
    costs = linear + nonlinear + noise + item_offsets
    return DatasetSplit(features=x, costs=costs)
