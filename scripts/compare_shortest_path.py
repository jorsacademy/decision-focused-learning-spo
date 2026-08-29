from __future__ import annotations

import argparse
import math
import statistics

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
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    return parser.parse_args()


def summarize(values: list[float]) -> tuple[float, float, float]:
    if not values:
        raise ValueError("values must be non-empty")
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, 0.0, 0.0
    std = statistics.stdev(values)
    ci95 = 1.96 * std / math.sqrt(len(values))
    return mean, std, ci95


def run_seed(args: argparse.Namespace, seed: int) -> dict[str, dict[str, float]]:
    torch.manual_seed(seed)
    graph = GridGraph(args.grid_size)
    x_train, c_train = generate_shortest_path_data(
        args.train_samples, args.features, graph, seed=seed
    )
    x_test, c_test = generate_shortest_path_data(
        args.test_samples, args.features, graph, seed=seed + 10_000
    )

    models = {
        "mse": LinearCostPredictor(args.features, graph.n_edges),
        "spo+": LinearCostPredictor(args.features, graph.n_edges),
    }
    models["spo+"].load_state_dict(models["mse"].state_dict())

    results: dict[str, dict[str, float]] = {}
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
        results[method] = evaluate_shortest_path_model(model, x_test, c_test, graph)
    return results


def main() -> None:
    args = parse_args()
    if not args.seeds:
        raise ValueError("at least one seed is required")

    per_seed = {seed: run_seed(args, seed) for seed in args.seeds}
    metrics = next(iter(per_seed.values()))["mse"].keys()

    print("method,metric,mean,std,ci95,n_seeds")
    for method in ("mse", "spo+"):
        for metric in metrics:
            values = [per_seed[seed][method][metric] for seed in args.seeds]
            mean, std, ci95 = summarize(values)
            print(f"{method},{metric},{mean:.6f},{std:.6f},{ci95:.6f},{len(values)}")

    regret_deltas = [
        per_seed[seed]["mse"]["mean_normalized_regret"]
        - per_seed[seed]["spo+"]["mean_normalized_regret"]
        for seed in args.seeds
    ]
    mean, std, ci95 = summarize(regret_deltas)
    print(
        "spo+_improvement,mean_normalized_regret_delta,"
        f"{mean:.6f},{std:.6f},{ci95:.6f},{len(regret_deltas)}"
    )


if __name__ == "__main__":
    main()
