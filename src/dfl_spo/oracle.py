"""Exact optimization oracle for fixed-cardinality cost minimization."""

from __future__ import annotations

import torch
from torch import Tensor


def select_k(costs: Tensor, k: int) -> Tensor:
    """Return binary decisions selecting the k smallest costs per instance."""
    if costs.ndim != 2:
        raise ValueError("costs must have shape [batch, items]")
    n_items = costs.shape[1]
    if not 1 <= k <= n_items:
        raise ValueError("k must satisfy 1 <= k <= number of items")
    indices = torch.topk(costs, k=k, dim=1, largest=False).indices
    decision = torch.zeros_like(costs)
    decision.scatter_(1, indices, 1.0)
    return decision


def objective_value(costs: Tensor, decisions: Tensor) -> Tensor:
    """Evaluate linear objective values c^T w for batched decisions."""
    if costs.shape != decisions.shape or costs.ndim != 2:
        raise ValueError("costs and decisions must have identical [batch, items] shape")
    return (costs * decisions).sum(dim=1)
