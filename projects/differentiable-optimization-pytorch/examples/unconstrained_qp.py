"""Train a neural network through an exact differentiable QP solve.

This example intentionally uses a fixed positive-definite Q matrix. The neural
network predicts the linear objective coefficient p, and the QP layer maps p to
the optimal decision x*. Synthetic targets are generated from a known mapping,
so a falling validation loss is meaningful rather than an artifact of fitting
independent random labels.
"""

from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn

from differentiable_qp import PredictThenOptimize, UnconstrainedQPLayer

SEED = 7


def set_seed(seed: int = SEED) -> None:
    """Make this small tutorial reasonably reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_dataset(
    n_samples: int = 600,
    input_dim: int = 6,
    qp_dim: int = 3,
) -> tuple[Tensor, Tensor, Tensor]:
    """Create features and QP-optimal targets from a known ground-truth map."""
    features = torch.randn(n_samples, input_dim)

    Q = torch.tensor(
        [
            [2.5, 0.4, 0.2],
            [0.4, 1.8, 0.3],
            [0.2, 0.3, 1.4],
        ],
        dtype=torch.float32,
    )
    if qp_dim != 3:
        raise ValueError("This tutorial dataset is defined for qp_dim=3.")

    true_weight = torch.tensor(
        [
            [0.8, -0.3, 0.2, 0.5, -0.4, 0.1],
            [-0.2, 0.6, -0.5, 0.1, 0.3, 0.4],
            [0.3, 0.2, 0.7, -0.4, 0.1, -0.6],
        ],
        dtype=torch.float32,
    )
    true_bias = torch.tensor([0.2, -0.1, 0.3], dtype=torch.float32)

    p_true = features @ true_weight.T + true_bias
    p_true = p_true + 0.1 * torch.sin(p_true)

    qp = UnconstrainedQPLayer()
    Q_batch = Q.expand(n_samples, -1, -1)
    targets = qp(Q_batch, p_true).detach()
    return features, targets, Q


def train() -> None:
    """Train the predict-then-optimize model and plot train/validation loss."""
    set_seed()

    features, targets, Q = make_dataset()
    split = int(0.8 * len(features))
    X_train, X_val = features[:split], features[split:]
    y_train, y_val = targets[:split], targets[split:]

    model = PredictThenOptimize(
        input_dim=X_train.shape[1],
        hidden_dim=32,
        qp_dim=y_train.shape[1],
        Q=Q,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    train_history: list[float] = []
    val_history: list[float] = []

    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()

        prediction = model(X_train)
        train_loss = criterion(prediction, y_train)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val)

        train_history.append(train_loss.item())
        val_history.append(val_loss.item())

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"Epoch {epoch:3d}/200 | "
                f"train MSE: {train_loss.item():.6f} | "
                f"validation MSE: {val_loss.item():.6f}"
            )

    plt.figure(figsize=(8, 5))
    plt.plot(train_history, label="Train")
    plt.plot(val_history, label="Validation")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared error")
    plt.title("Learning Through a Differentiable QP Layer")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train()
