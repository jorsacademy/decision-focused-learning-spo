from __future__ import annotations

import argparse

import torch

from dfl_spo.models import LinearCostPredictor
from dfl_spo.shortest_path import (
    GridGraph,
    evaluate_shortest_path_model,
    generate_shortest_path_data,
    train_shortest_path_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare MSE and SPO+ on contextual shortest path.")
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--test-samples", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    graph = GridGraph(args.grid_size)
    x_train, c_train = generate_shortest_path_data(
        args.train_samples, args.features, graph, seed=args.seed
    )
    x_test, c_test = generate_shortest_path_data(
        args.test_samples, args.features, graph, seed=args.seed + 10_000
    )

    models = {
        "mse": LinearCostPredictor(args.features, graph.n_edges),
        "spo+": LinearCostPredictor(args.features, graph.n_edges),
    }
    models["spo+"].load_state_dict(models["mse"].state_dict())

    for method, model in models.items():
        train_shortest_path_model(
            model,
            x_train,
            c_train,
            graph,
            method=method,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )
        metrics = evaluate_shortest_path_model(model, x_test, c_test, graph)
        print(f"[{method}]")
        for key, value in metrics.items():
            print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
