"""Decision-focused learning with Smart Predict-then-Optimize."""

from dfl_spo.data import DatasetSplit, generate_contextual_costs
from dfl_spo.evaluation import decision_regret, evaluate_model, normalized_regret
from dfl_spo.losses import mse_loss, spo_plus_loss
from dfl_spo.models import LinearCostPredictor
from dfl_spo.oracle import objective_value, select_k
from dfl_spo.training import train_model

__all__ = [
    "DatasetSplit",
    "LinearCostPredictor",
    "decision_regret",
    "evaluate_model",
    "generate_contextual_costs",
    "mse_loss",
    "normalized_regret",
    "objective_value",
    "select_k",
    "spo_plus_loss",
    "train_model",
]
