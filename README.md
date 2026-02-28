# arch_eval

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub Repo](https://img.shields.io/badge/GitHub-lof310%2Farch__eval-blue)](https://github.com/lof310/arch_eval)

**arch_eval** is a high-level library for efficient architecture evaluation of machine learning models. It provides a unified interface for training, benchmarking, and hyperparameter optimization with features like distributed training, mixed precision, and real-time visualization.

## Features

- **Unified Training Interface** – Train single models with comprehensive configuration options.
- **Multi-Model Benchmarking** – Compare multiple architectures sequentially or in parallel (thread/process-based).
- **Distributed Training** – Built-in support for DataParallel, DistributedDataParallel (DDP), and FSDP.
- **Advanced Mixed Precision** – AMP with float16, bfloat16, and experimental FP8 support.
- **Gradient Checkpointing** – Reduce memory footprint for large models.
- **Rich Visualization** – Real-time training windows, video recording of metrics, and publication‑ready plots.
- **Comprehensive Logging** – Integration with Weights & Biases and TensorBoard.
- **Hyperparameter Optimization** – Grid search and random search out of the box.
- **Extensible Plugin System** – Custom hooks and callbacks for maximum flexibility.
- **Robust Data Handling** – Supports PyTorch Datasets, synthetic data, torchvision datasets, Hugging Face datasets, and streaming.
- **Production-Ready** – Configurable timeouts, retry logic, checkpointing with corruption protection, and deterministic execution.

## Installation

Install directly from the GitHub repository:

```bash
# Clone the repository
git clone https://github.com/lof310/arch_eval.git
cd arch_eval

# Install in development mode (recommended)
pip install -e .

# Or install normally
pip install .
```

## Quick Start

### 1. Train a Single Model

```python
import torch.nn as nn
from arch_eval import Trainer, TrainingConfig

# Define a simple model
class SimpleMLP(nn.Module):
    def __init__(self, input_size=10, hidden=20, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc1(x))

# Configure training
config = TrainingConfig(
    dataset="synthetic classification",
    dataset_params={"n_samples": 1000, "n_features": 10, "n_classes": 2},
    training_args={"num_epochs": 5, "batch_size": 32},
    task="classification",
    realtime=True,
    save_plot=["loss", "accuracy"]
)

model = SimpleMLP()
trainer = Trainer(model, config)
history = trainer.train()
```

### 2. Benchmark Multiple Models

```python
from arch_eval import Benchmark, BenchmarkConfig

models = [
    {"name": "Small MLP", "model": SimpleMLP(hidden=10)},
    {"name": "Large MLP", "model": SimpleMLP(hidden=50)}
]

config = BenchmarkConfig(
    dataset="synthetic classification",
    dataset_params={"n_samples": 500, "n_features": 10, "n_classes": 2},
    compare_metrics=["accuracy", "loss"],
    parallel=True
)

benchmark = Benchmark(models, config)
results = benchmark.run()
print(results)
```

### 3. Hyperparameter Search

```python
from arch_eval import HyperparameterOptimizer

def model_fn():
    return SimpleMLP()

base_config = TrainingConfig(
    dataset="synthetic classification",
    dataset_params={"n_samples": 500, "n_features": 10, "n_classes": 2},
    training_args={"num_epochs": 3},
    task="classification",
    realtime=False  # disable live plots during search
)

param_grid = {
    "learning_rate": [0.001, 0.01, 0.1],
    "hidden": [10, 20, 50]
}

optimizer = HyperparameterOptimizer(
    model_fn, base_config, param_grid,
    search_type="grid", metric="val_accuracy", mode="max"
)
results = optimizer.run()
```

## Documentation

Documentation is under development. For now, please refer to the example scripts in the `examples/` directory and the in-code docstrings.

## Contributing

Contributions are welcome!

## License

Distributed under the Apache License 2.0. See `LICENSE` for more information.

## Citation

If you use arch_eval in your research, please cite:

```bibtex
@software{arch_eval2026,
  author = {Leinier Orama},
  title = {arch_eval: High-level Library for Architecture Evaluation of ML Models},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/lof310/arch_eval}
}
```
