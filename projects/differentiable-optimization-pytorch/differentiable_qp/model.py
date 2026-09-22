"""Small neural model used in the differentiable-optimization tutorial."""

from __future__ import annotations

from torch import Tensor, nn

from .layer import UnconstrainedQPLayer


class PredictThenOptimize(nn.Module):
    """Predict a QP parameter and optimize it inside the forward pass.

    The neural network predicts the linear cost vector ``p``. A fixed
    positive-definite matrix ``Q`` defines the quadratic term. The model output
    is the exact minimizer of the resulting unconstrained quadratic program.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        qp_dim: int,
        Q: Tensor,
    ) -> None:
        super().__init__()

        if Q.shape != (qp_dim, qp_dim):
            raise ValueError(f"Q must have shape ({qp_dim}, {qp_dim}).")

        self.predict_p = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, qp_dim),
        )
        self.qp = UnconstrainedQPLayer()
        self.register_buffer("Q", Q.detach().clone())

    def forward(self, features: Tensor) -> Tensor:
        """Predict ``p`` and return the corresponding optimal QP solution."""
        p = self.predict_p(features)
        Q_batch = self.Q.expand(features.shape[0], -1, -1)
        return self.qp(Q_batch, p)
