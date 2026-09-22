from __future__ import annotations

import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer


class MeanVarianceLayer(torch.nn.Module):
    """Long-only fully-invested mean-variance portfolio layer.

    Forward pass solves:
        min_w  risk_aversion * w' Sigma w - mu' w
        s.t.   1'w = 1, w >= 0
    Gradients are obtained by differentiating through the convex program.
    """

    def __init__(self, covariance: np.ndarray, risk_aversion: float = 1.0):
        super().__init__()
        covariance = np.asarray(covariance, dtype=float)
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ValueError("covariance must be square")
        if risk_aversion <= 0:
            raise ValueError("risk_aversion must be positive")
        n = covariance.shape[0]
        covariance = covariance + 1e-6 * np.eye(n)
        w = cp.Variable(n)
        mu = cp.Parameter(n)
        objective = cp.Minimize(float(risk_aversion) * cp.quad_form(w, covariance) - mu @ w)
        constraints = [cp.sum(w) == 1, w >= 0]
        problem = cp.Problem(objective, constraints)
        if not problem.is_dpp():
            raise ValueError("optimization layer must satisfy DPP")
        self.layer = CvxpyLayer(problem, parameters=[mu], variables=[w])

    def forward(self, expected_returns: torch.Tensor) -> torch.Tensor:
        (weights,) = self.layer(expected_returns)
        return weights
