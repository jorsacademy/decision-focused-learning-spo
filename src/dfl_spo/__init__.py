"""Decision-focused learning with Smart Predict-then-Optimize."""

from dfl_spo.data import DatasetSplit, generate_contextual_costs
from dfl_spo.evaluation import decision_regret, evaluate_model, normalized_regret
from dfl_spo.losses import mse_loss, spo_plus_loss
from dfl_spo.models import LinearCostPredictor
from dfl_spo.oracle import objective_value, select_k
from dfl_spo.shortest_path import (
    GridGraph,
    evaluate_shortest_path_model,
    generate_shortest_path_data,
    path_objective,
    shortest_path_decision,
    shortest_path_spo_plus_loss,
    train_shortest_path_model,
)
from dfl_spo.training import train_model

__all__ = [
    "DatasetSplit",
    "GridGraph",
    "LinearCostPredictor",
    "decision_regret",
    "evaluate_model",
    "evaluate_shortest_path_model",
    "generate_contextual_costs",
    "generate_shortest_path_data",
    "mse_loss",
    "normalized_regret",
    "objective_value",
    "path_objective",
    "select_k",
    "shortest_path_decision",
    "shortest_path_spo_plus_loss",
    "spo_plus_loss",
    "train_model",
    "train_shortest_path_model",
]
