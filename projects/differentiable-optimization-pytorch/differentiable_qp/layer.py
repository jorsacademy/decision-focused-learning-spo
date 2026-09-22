"""Differentiable quadratic-program layers implemented with PyTorch."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class UnconstrainedQPLayer(nn.Module):
    r"""Solve a batch of strictly convex unconstrained quadratic programs.

    The layer solves

        minimize_x  0.5 * x^T Q x + p^T x

    where ``Q`` is symmetric positive definite. The unique minimizer satisfies

        Q x* + p = 0,

    so the implementation uses ``torch.linalg.solve`` rather than explicitly
    forming ``Q^{-1}``.

    PyTorch differentiates through ``torch.linalg.solve``, which makes the
    solution differentiable with respect to both ``Q`` and ``p`` whenever the
    solve is well-defined.
    """

    def forward(self, Q: Tensor, p: Tensor) -> Tensor:
        """Return the optimal solution ``x*`` for each problem in the batch.

        Args:
            Q: Tensor with shape ``(..., n, n)``.
            p: Tensor with shape ``(..., n)``.

        Returns:
            Tensor with shape ``(..., n)``.

        Raises:
            ValueError: If the tensor shapes are incompatible.
        """
        if Q.ndim < 2 or p.ndim < 1:
            raise ValueError("Q must have shape (..., n, n) and p (..., n).")
        if Q.shape[-1] != Q.shape[-2]:
            raise ValueError("Q must be square in its last two dimensions.")
        if Q.shape[-1] != p.shape[-1]:
            raise ValueError("The last dimension of p must match Q.")
        if Q.shape[:-2] != p.shape[:-1]:
            raise ValueError("Q and p must have matching batch dimensions.")

        return torch.linalg.solve(Q, -p.unsqueeze(-1)).squeeze(-1)


def positive_definite_from_factor(factor: Tensor, jitter: float = 1e-3) -> Tensor:
    """Construct a symmetric positive-definite matrix from an unconstrained factor.

    For a matrix ``L``, the construction is

        Q = L L^T + jitter * I.

    Args:
        factor: Tensor with shape ``(..., n, n)``.
        jitter: Strictly positive diagonal regularization.

    Returns:
        Positive-definite tensor with the same shape as ``factor``.
    """
    if factor.ndim < 2 or factor.shape[-1] != factor.shape[-2]:
        raise ValueError("factor must have shape (..., n, n).")
    if jitter <= 0:
        raise ValueError("jitter must be strictly positive.")

    n = factor.shape[-1]
    identity = torch.eye(n, dtype=factor.dtype, device=factor.device)
    return factor @ factor.transpose(-1, -2) + jitter * identity
