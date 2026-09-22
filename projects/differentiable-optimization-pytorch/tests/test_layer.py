"""Tests for the PyTorch differentiable QP layer."""

import torch

from differentiable_qp import (
    PredictThenOptimize,
    UnconstrainedQPLayer,
    positive_definite_from_factor,
)


def test_solution_satisfies_first_order_condition() -> None:
    layer = UnconstrainedQPLayer()

    Q = torch.tensor(
        [
            [[2.0, 0.2], [0.2, 1.5]],
            [[1.8, -0.1], [-0.1, 2.2]],
        ],
        dtype=torch.float64,
    )
    p = torch.tensor(
        [[-1.0, 0.5], [0.4, -0.8]],
        dtype=torch.float64,
    )

    x = layer(Q, p)
    residual = (Q @ x.unsqueeze(-1)).squeeze(-1) + p

    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-10)


def test_gradient_with_respect_to_p_matches_analytic_result() -> None:
    layer = UnconstrainedQPLayer()

    Q = torch.tensor(
        [[2.0, 0.3], [0.3, 1.4]],
        dtype=torch.float64,
    )
    p = torch.tensor([-0.7, 0.2], dtype=torch.float64, requires_grad=True)

    x = layer(Q, p)
    x.sum().backward()

    expected = -torch.linalg.solve(Q.T, torch.ones(2, dtype=torch.float64))
    assert torch.allclose(p.grad, expected, atol=1e-10)


def test_positive_definite_construction() -> None:
    torch.manual_seed(0)
    factor = torch.randn(4, 3, 3, dtype=torch.float64)
    Q = positive_definite_from_factor(factor, jitter=1e-3)

    eigenvalues = torch.linalg.eigvalsh(Q)
    assert torch.all(eigenvalues > 0)


def test_model_backpropagates_through_qp_layer() -> None:
    Q = torch.tensor(
        [[2.0, 0.2], [0.2, 1.5]],
        dtype=torch.float32,
    )
    model = PredictThenOptimize(input_dim=3, hidden_dim=8, qp_dim=2, Q=Q)

    features = torch.randn(5, 3)
    target = torch.randn(5, 2)

    loss = torch.nn.functional.mse_loss(model(features), target)
    loss.backward()

    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)
