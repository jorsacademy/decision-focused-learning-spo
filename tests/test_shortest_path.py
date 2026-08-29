import torch

from dfl_spo.models import LinearCostPredictor
from dfl_spo.shortest_path import (
    GridGraph,
    evaluate_shortest_path_model,
    generate_shortest_path_data,
    path_objective,
    shortest_path_decision,
    shortest_path_spo_plus_loss,
    train_shortest_path_model,
)


def test_grid_oracle_finds_known_two_by_two_path() -> None:
    graph = GridGraph(2)
    costs = torch.tensor([[1.0, 5.0, 1.0, 1.0]])
    decision = shortest_path_decision(costs, graph)
    assert decision.shape == costs.shape
    assert torch.allclose(path_objective(costs, decision), torch.tensor([2.0]))
    assert int(decision.sum()) == 2


def test_shortest_path_spo_plus_has_gradient() -> None:
    graph = GridGraph(3)
    _, true_costs = generate_shortest_path_data(4, 5, graph, seed=2)
    predicted = true_costs.clone().requires_grad_(True)
    loss = shortest_path_spo_plus_loss(predicted, true_costs, graph)
    loss.backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()


def test_shortest_path_training_and_evaluation_run() -> None:
    graph = GridGraph(3)
    features, costs = generate_shortest_path_data(24, 4, graph, seed=3)
    model = LinearCostPredictor(4, graph.n_edges)
    history = train_shortest_path_model(
        model, features, costs, graph, method="spo+", epochs=3, learning_rate=1e-2
    )
    metrics = evaluate_shortest_path_model(model, features, costs, graph)
    assert len(history) == 3
    assert metrics["mean_regret"] >= -1e-6
    assert 0.0 <= metrics["zero_regret_rate"] <= 1.0
