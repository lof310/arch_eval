"""Basic usage examples for arch_eval."""

import torch
import torch.nn as nn
from arch_eval import Trainer, Benchmark, TrainingConfig, BenchmarkConfig
from arch_eval.core.callbacks import EarlyStopping, ModelCheckpoint, TensorBoardLogger
from arch_eval.distributed import init_distributed, cleanup_distributed
from arch_eval.hpo import HyperparameterOptimizer


class SimpleMLP(nn.Module):
    def __init__(self, input_size=10, hidden=20, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_single():
    config = TrainingConfig(
        realtime=True,
        dataset="synthetic classification",
        dataset_params={"n_samples": 10000, "n_features": 10, "n_classes": 2},
        training_args={"num_epochs": 10, "batch_size": 16, "learning_rate": 0.001},
        save_plot=["loss", "accuracy"],
        task="classification",
        mixed_precision=True,
        checkpoint_dir="./checkpoints",
        save_best_only=True,
        callbacks=[EarlyStopping(monitor="val_loss", patience=2), TensorBoardLogger()],
    )
    model = SimpleMLP(input_size=10, num_classes=2)
    trainer = Trainer(model, config)
    history = trainer.train()
    print("Training completed!")
    return history


def benchmark_models():
    models = [
        {"name": "Small MLP", "model": SimpleMLP(input_size=10, hidden=10, num_classes=2)},
        {"name": "Large MLP", "model": SimpleMLP(input_size=10, hidden=50, num_classes=2)},
    ]
    config = BenchmarkConfig(
        realtime=False,
        dataset="synthetic classification",
        dataset_params={"n_samples": 10000, "n_features": 10, "n_classes": 2},
        training_args={"num_epochs": 3, "batch_size": 16},
        compare_metrics=["accuracy", "loss"],
        parallel=True,
    )
    bench = Benchmark(models, config)
    results = bench.run()
    print("\nBenchmark Results:\n", results)
    return results


def hyperparameter_search():
    def model_fn():
        return SimpleMLP(input_size=10, hidden=20, num_classes=2)

    base_config = TrainingConfig(
        dataset="synthetic classification",
        dataset_params={"n_samples": 500, "n_features": 10, "n_classes": 2},
        training_args={"num_epochs": 3, "batch_size": 32},
        task="classification",
        realtime=False
    )
    param_grid = {
        "learning_rate": [0.001, 0.01, 0.1],
        "hidden": [10, 20, 50],
    }
    opt = HyperparameterOptimizer(model_fn, base_config, param_grid, search_type="grid", metric="val_accuracy", mode="max")
    results = opt.run()
    return results


if __name__ == "__main__":
    print("Example 1: Single model training")
    train_single()
    print("\n" + "=" * 50 + "\n")
    print("Example 2: Benchmarking")
    benchmark_models()
    print("\n" + "=" * 50 + "\n")
    print("Example 3: Hyperparameter search")
    hyperparameter_search()
