"""Evaluation metrics for predict-then-optimize experiments."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from dfl_spo.oracle import objective_value, select_k


def decision_regret(predicted_costs: Tensor, true_costs: Tensor, k: int) -> Tensor:
    """Return per-instance regret under true costs."""
    chosen = select_k(predicted_costs, k)
    optimal = select_k(true_costs, k)
    return objective_value(true_costs, chosen) - objective_value(true_costs, optimal)


def normalized_regret(predicted_costs: Tensor, true_costs: Tensor, k: int) -> Tensor:
    """Normalize regret by the absolute clairvoyant objective with numerical protection."""
    optimal = select_k(true_costs, k)
    optimum = objective_value(true_costs, optimal)
    regret = decision_regret(predicted_costs, true_costs, k)
    return regret / optimum.abs().clamp_min(1e-8)


@torch.inference_mode()
def evaluate_model(model: nn.Module, features: Tensor, true_costs: Tensor, k: int) -> dict[str, float]:
    """Evaluate prediction quality and downstream decision quality."""
    model.eval()
    predicted = model(features)
    regret = decision_regret(predicted, true_costs, k)
    nregret = normalized_regret(predicted, true_costs, k)
    return {
        "mse": float(torch.mean((predicted - true_costs) ** 2)),
        "mean_regret": float(regret.mean()),
        "median_regret": float(regret.median()),
        "mean_normalized_regret": float(nregret.mean()),
        "zero_regret_rate": float((regret.abs() <= 1e-8).float().mean()),
    }
