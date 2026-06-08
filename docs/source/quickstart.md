# Quick Start

## Train a Single Model

```python
import torch.nn as nn
from arch_eval import Trainer, TrainingConfig

# Define a global configuration
# Dataset
n_samples, n_features, n_classes = 5000, 128, 64

# Model
input_size, hidden = n_features, n_features*2

# Training
batch_size, num_epochs = 16, 4

# Define a simple model
class MLP(nn.Module):
    def __init__(self, input_size=128, hidden=256, num_classes=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Configure training
config = TrainingConfig(
    dataset="synthetic classification",
    dataset_params={"n_samples": n_samples, "n_features": n_features, "n_classes": n_classes},
    training_args={"num_epochs": num_epochs, "batch_size": batch_size},
    task="classification",
    realtime="auto",  # Options: "auto", "gui", "terminal", "none"
    save_plot=["loss", "accuracy"]
)

model = MLP(input_size, hidden, n_classes)
trainer = Trainer(model, config)
history = trainer.train()
```

## Benchmark Multiple Models

```python
from arch_eval import Benchmark, BenchmarkConfig

models = [
    {"name": "Small MLP", "model": MLP(hidden=256)},
    {"name": "Large MLP", "model": MLP(hidden=512)}
]

config = BenchmarkConfig(
    dataset="synthetic classification",
    dataset_params={"n_samples": 10000, "n_features": 128, "n_classes": 64},
    compare_metrics=["accuracy", "loss"],
    parallel=True
)

benchmark = Benchmark(models, config)
results = benchmark.run()
print(results)
```

## Hyperparameter Search

```python
from arch_eval import HyperparameterOptimizer

def model_fn():
    return MLP()

base_config = TrainingConfig(
    dataset="synthetic classification",
    dataset_params={"n_samples": 1000, "n_features": 128, "n_classes": 64},
    training_args={"num_epochs": 3},
    task="classification",
    realtime="none"  # disable live plots during search
)

param_grid = {
    "learning_rate": [0.001, 0.01, 0.1],
    "hidden": [64, 128, 256]
}

optimizer = HyperparameterOptimizer(
    model_fn, base_config, param_grid,
    search_type="grid", metric="val_accuracy", mode="max"
)
results = optimizer.run()
```

## Using Callbacks

```python
from arch_eval import Trainer, TrainingConfig, EarlyStopping, ModelCheckpoint, TensorBoardLogger

config = TrainingConfig(
    dataset="synthetic classification",
    dataset_params={"n_samples": 1000, "n_features": 20, "n_classes": 5},
    training_args={"num_epochs": 50, "batch_size": 32},
    task="classification",
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=5),
        ModelCheckpoint(filepath="checkpoints/model.pt", monitor="val_accuracy", save_best_only=True),
        TensorBoardLogger(log_dir="./logs")
    ],
    checkpoint_dir="./checkpoints"
)

model = nn.Linear(20, 5)
trainer = Trainer(model, config)
history = trainer.train()
```
