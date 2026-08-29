"""Prediction and decision-focused losses."""

from __future__ import annotations

import torch
from torch import Tensor

from dfl_spo.oracle import objective_value, select_k


def mse_loss(predicted_costs: Tensor, true_costs: Tensor) -> Tensor:
    """Standard prediction loss."""
    return torch.mean((predicted_costs - true_costs) ** 2)


def spo_plus_loss(predicted_costs: Tensor, true_costs: Tensor, k: int) -> Tensor:
    """Return the mean SPO+ surrogate for fixed-cardinality minimization.

    For minimization, let w_true = w*(c) and
    w_spo = w*(2*c_hat - c). The sample surrogate is

        (c - 2*c_hat)^T w_spo + 2*c_hat^T w_true - c^T w_true.

    The oracle decisions are treated as fixed subgradient selections.
    """
    if predicted_costs.shape != true_costs.shape:
        raise ValueError("predicted_costs and true_costs must have identical shape")

    with torch.no_grad():
        w_true = select_k(true_costs, k)
        w_spo = select_k(2.0 * predicted_costs.detach() - true_costs, k)

    surrogate = objective_value(true_costs - 2.0 * predicted_costs, w_spo)
    surrogate = surrogate + 2.0 * objective_value(predicted_costs, w_true)
    surrogate = surrogate - objective_value(true_costs, w_true)
    return surrogate.mean()
