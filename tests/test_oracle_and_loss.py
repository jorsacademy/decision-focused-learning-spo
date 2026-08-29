import torch

from dfl_spo.evaluation import decision_regret
from dfl_spo.losses import spo_plus_loss
from dfl_spo.oracle import objective_value, select_k


def test_select_k_returns_exact_cardinality_and_minimum_items() -> None:
    costs = torch.tensor([[3.0, 1.0, 4.0, 2.0]])
    decision = select_k(costs, 2)
    assert decision.tolist() == [[0.0, 1.0, 0.0, 1.0]]
    assert torch.allclose(objective_value(costs, decision), torch.tensor([3.0]))


def test_regret_is_zero_for_perfect_prediction() -> None:
    costs = torch.tensor([[2.0, -1.0, 4.0, 0.0]])
    regret = decision_regret(costs, costs, 2)
    assert torch.allclose(regret, torch.zeros(1))


def test_spo_plus_is_zero_for_perfect_prediction() -> None:
    costs = torch.tensor([[2.0, -1.0, 4.0, 0.0]])
    predicted = costs.clone().requires_grad_(True)
    loss = spo_plus_loss(predicted, costs, 2)
    assert torch.allclose(loss, torch.tensor(0.0))
    loss.backward()
    assert torch.allclose(predicted.grad, torch.zeros_like(predicted))


def test_spo_plus_produces_finite_gradient() -> None:
    true_costs = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    predicted = torch.tensor([[3.0, 2.0, 1.0, 0.0]], requires_grad=True)
    loss = spo_plus_loss(predicted, true_costs, 2)
    loss.backward()
    assert torch.isfinite(loss)
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
