import torch

from dfl_spo.data import generate_contextual_costs
from dfl_spo.evaluation import evaluate_model
from dfl_spo.models import LinearCostPredictor
from dfl_spo.training import train_model


def test_dataset_generation_is_reproducible() -> None:
    a = generate_contextual_costs(16, 4, 6, seed=7)
    b = generate_contextual_costs(16, 4, 6, seed=7)
    assert torch.allclose(a.features, b.features)
    assert torch.allclose(a.costs, b.costs)


def test_mse_and_spo_training_smoke() -> None:
    data = generate_contextual_costs(64, 4, 6, seed=1)
    for method in ("mse", "spo+"):
        torch.manual_seed(3)
        model = LinearCostPredictor(4, 6)
        history = train_model(model, data.features, data.costs, method=method, k=2, epochs=3)
        metrics = evaluate_model(model, data.features, data.costs, 2)
        assert len(history) == 3
        assert all(torch.isfinite(torch.tensor(v)) for v in history)
        assert metrics["mean_regret"] >= -1e-7
