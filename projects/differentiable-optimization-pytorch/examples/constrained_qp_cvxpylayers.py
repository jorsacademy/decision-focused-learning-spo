"""Constrained differentiable QP example using CVXPYLayers.

Install the optional dependencies first:

    pip install cvxpy cvxpylayers

Unlike the manual projected-gradient approximation in the original draft, this
example delegates the constrained convex problem to CVXPY/CVXPYLayers.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer


def build_layer() -> CvxpyLayer:
    """Build a DPP-compliant constrained quadratic-program layer."""
    n = 2

    x = cp.Variable(n)
    p = cp.Parameter(n)
    h = cp.Parameter(3)

    Q = np.array([[2.0, 0.5], [0.5, 1.5]])
    G = np.array(
        [
            [-1.0, 0.0],
            [0.0, -1.0],
            [1.0, 1.0],
        ]
    )

    objective = cp.Minimize(0.5 * cp.quad_form(x, Q) + p @ x)
    constraints = [G @ x <= h]
    problem = cp.Problem(objective, constraints)

    if not problem.is_dpp():
        raise RuntimeError("The CVXPY problem must satisfy DPP for CvxpyLayer.")

    return CvxpyLayer(problem, parameters=[p, h], variables=[x])


def main() -> None:
    """Solve a small batch and differentiate a scalar loss through the solutions."""
    layer = build_layer()

    p = torch.tensor(
        [[-1.0, -1.0], [-2.0, -1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    h = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.2]],
        dtype=torch.float64,
        requires_grad=True,
    )

    solution, = layer(p, h)
    loss = solution.square().sum()
    loss.backward()

    print("Solutions:")
    print(solution)
    print("\nGradient of loss with respect to p:")
    print(p.grad)
    print("\nGradient of loss with respect to h:")
    print(h.grad)


if __name__ == "__main__":
    main()
