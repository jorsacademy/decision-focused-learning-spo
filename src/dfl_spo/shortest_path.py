"""Contextual shortest-path problem and SPO+ utilities on directed grid DAGs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class GridGraph:
    """Directed square grid with right/down edges from source to sink."""

    size: int

    def __post_init__(self) -> None:
        if self.size < 2:
            raise ValueError("grid size must be at least 2")

    @property
    def n_nodes(self) -> int:
        return self.size * self.size

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        n = self.size
        edges: list[tuple[int, int]] = []
        for row in range(n):
            for col in range(n):
                node = row * n + col
                if col + 1 < n:
                    edges.append((node, node + 1))
                if row + 1 < n:
                    edges.append((node, node + n))
        return tuple(edges)

    @property
    def n_edges(self) -> int:
        return len(self.edges)


def shortest_path_decision(costs: Tensor, graph: GridGraph) -> Tensor:
    """Return exact source-to-sink path incidence vectors for batched edge costs.

    The graph is acyclic, so dynamic programming is exact even when reflected
    SPO+ costs are negative.
    """
    if costs.ndim != 2 or costs.shape[1] != graph.n_edges:
        raise ValueError("costs must have shape [batch, graph.n_edges]")

    edges = graph.edges
    incoming: list[list[tuple[int, int]]] = [[] for _ in range(graph.n_nodes)]
    for edge_idx, (u, v) in enumerate(edges):
        incoming[v].append((u, edge_idx))

    decisions = torch.zeros_like(costs)
    for batch_idx in range(costs.shape[0]):
        distance = [float("inf")] * graph.n_nodes
        parent_edge = [-1] * graph.n_nodes
        distance[0] = 0.0

        for node in range(1, graph.n_nodes):
            best_cost = float("inf")
            best_edge = -1
            for predecessor, edge_idx in incoming[node]:
                candidate = distance[predecessor] + float(costs[batch_idx, edge_idx])
                if candidate < best_cost:
                    best_cost = candidate
                    best_edge = edge_idx
            distance[node] = best_cost
            parent_edge[node] = best_edge

        node = graph.n_nodes - 1
        while node != 0:
            edge_idx = parent_edge[node]
            if edge_idx < 0:
                raise RuntimeError("sink is unreachable")
            decisions[batch_idx, edge_idx] = 1.0
            node = edges[edge_idx][0]

    return decisions


def path_objective(costs: Tensor, decisions: Tensor) -> Tensor:
    """Evaluate batched path decisions under batched edge costs."""
    if costs.shape != decisions.shape:
        raise ValueError("costs and decisions must have identical shape")
    return (costs * decisions).sum(dim=1)


def generate_shortest_path_data(
    n_samples: int,
    n_features: int,
    graph: GridGraph,
    *,
    seed: int = 0,
    noise_std: float = 0.1,
) -> tuple[Tensor, Tensor]:
    """Generate contextual features and nonlinear positive edge costs."""
    if n_samples <= 0 or n_features <= 0:
        raise ValueError("n_samples and n_features must be positive")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(n_samples, n_features, generator=generator)
    linear = torch.randn(n_features, graph.n_edges, generator=generator) / n_features**0.5
    nonlinear = torch.randn(n_features, graph.n_edges, generator=generator) / n_features**0.5
    raw = features @ linear + 0.35 * (features.square() @ nonlinear)
    noise = noise_std * torch.randn(n_samples, graph.n_edges, generator=generator)
    costs = torch.nn.functional.softplus(raw + noise) + 0.05
    return features, costs


def shortest_path_spo_plus_loss(
    predicted_costs: Tensor,
    true_costs: Tensor,
    graph: GridGraph,
) -> Tensor:
    """Compute the SPO+ convex surrogate using the exact shortest-path oracle."""
    if predicted_costs.shape != true_costs.shape:
        raise ValueError("predicted_costs and true_costs must have identical shape")

    with torch.no_grad():
        w_true = shortest_path_decision(true_costs, graph)
        w_spo = shortest_path_decision(2.0 * predicted_costs.detach() - true_costs, graph)

    surrogate = path_objective(true_costs - 2.0 * predicted_costs, w_spo)
    surrogate += 2.0 * path_objective(predicted_costs, w_true)
    surrogate -= path_objective(true_costs, w_true)
    return surrogate.mean()


def train_shortest_path_model(
    model: nn.Module,
    features: Tensor,
    costs: Tensor,
    graph: GridGraph,
    *,
    method: str,
    epochs: int = 100,
    learning_rate: float = 1e-2,
) -> list[float]:
    """Train a cost predictor with MSE or shortest-path SPO+."""
    if method not in {"mse", "spo+"}:
        raise ValueError("method must be 'mse' or 'spo+'").
    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[float] = []
    for _ in range(epochs):
        predicted = model(features)
        if method == "mse":
            loss = torch.mean((predicted - costs) ** 2)
        else:
            loss = shortest_path_spo_plus_loss(predicted, costs, graph)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history


@torch.inference_mode()
def evaluate_shortest_path_model(
    model: nn.Module,
    features: Tensor,
    true_costs: Tensor,
    graph: GridGraph,
) -> dict[str, float]:
    """Report prediction error and downstream shortest-path decision quality."""
    model.eval()
    predicted = model(features)
    predicted_decision = shortest_path_decision(predicted, graph)
    optimal_decision = shortest_path_decision(true_costs, graph)
    achieved = path_objective(true_costs, predicted_decision)
    optimum = path_objective(true_costs, optimal_decision)
    regret = achieved - optimum
    normalized = regret / optimum.abs().clamp_min(1e-8)
    return {
        "mse": float(torch.mean((predicted - true_costs) ** 2)),
        "mean_path_cost": float(achieved.mean()),
        "mean_optimal_cost": float(optimum.mean()),
        "mean_regret": float(regret.mean()),
        "mean_normalized_regret": float(normalized.mean()),
        "zero_regret_rate": float((regret.abs() <= 1e-7).float().mean()),
    }
