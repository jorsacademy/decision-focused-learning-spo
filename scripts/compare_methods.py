from __future__ import annotations

import argparse
import copy
import statistics

import torch

from dfl_spo.data import generate_contextual_costs
from dfl_spo.evaluation import evaluate_model
from dfl_spo.models import LinearCostPredictor
from dfl_spo.training import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare MSE and SPO+ training.")
    parser.add_argument("--train-samples", type=int, default=1024)
    parser.add_argument("--test-samples", type=int, default=512)
    parser.add_argument("--features", type=int, default=6)
    parser.add_argument("--items", type=int, default=10)
    parser.add_argument("--select", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    return parser.parse_args()


def run_seed(args: argparse.Namespace, seed: int) -> dict[str, dict[str, float]]:
    torch.manual_seed(seed)
    dataset = generate_contextual_costs(
        args.train_samples + args.test_samples,
        args.features,
        args.items,
        seed=seed,
    )
    train_x = dataset.features[: args.train_samples]
    train_c = dataset.costs[: args.train_samples]
    test_x = dataset.features[args.train_samples :]
    test_c = dataset.costs[args.train_samples :]

    base = LinearCostPredictor(args.features, args.items)
    mse_model = copy.deepcopy(base)
    spo_model = copy.deepcopy(base)

    train_model(
        mse_model,
        train_x,
        train_c,
        method="mse",
        k=args.select,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    train_model(
        spo_model,
        train_x,
        train_c,
        method="spo+",
        k=args.select,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    return {
        "mse": evaluate_model(mse_model, test_x, test_c, args.select),
        "spo+": evaluate_model(spo_model, test_x, test_c, args.select),
    }


def main() -> None:
    args = parse_args()
    if not 1 <= args.select <= args.items:
        raise ValueError("--select must satisfy 1 <= select <= items")

    results = [run_seed(args, seed) for seed in args.seeds]
    metrics = ["mse", "mean_regret", "mean_normalized_regret", "zero_regret_rate"]
    print("method,metric,mean,std")
    for method in ("mse", "spo+"):
        for metric in metrics:
            values = [result[method][metric] for result in results]
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            print(f"{method},{metric},{statistics.mean(values):.6f},{std:.6f}")


if __name__ == "__main__":
    main()
