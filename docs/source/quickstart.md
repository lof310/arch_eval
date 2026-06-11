# Quick Start

Welcome to **arch_eval**! This guide will get you up and running quickly with practical examples. By the end of this guide, you'll understand how to train models, compare architectures, and optimize hyperparameters.

## Prerequisites

Before starting, ensure you have:
- Python 3.9 or higher installed
- PyTorch installed (with CUDA if using GPU)
- arch_eval installed (`pip install -e .` or `pip install arch_eval`)

## Train a Single Model

Let's start with the most common use case: training a single neural network model.

### Complete Example

```python
import torch.nn as nn
from arch_eval import Trainer, TrainingConfig

# Define a global configuration
# Dataset parameters
n_samples, n_features, n_classes = 5000, 128, 64

# Model parameters
input_size, hidden = n_features, n_features*2

# Training parameters
batch_size, num_epochs = 16, 4

# Define a simple model
class MLP(nn.Module):
    """Multi-Layer Perceptron for classification."""
    
    def __init__(self, input_size=128, hidden=256, num_classes=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.GELU(),  # Activation function
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Configure training
config = TrainingConfig(
    dataset="synthetic classification",  # Use built-in synthetic data generator
    dataset_params={
        "n_samples": n_samples, 
        "n_features": n_features, 
        "n_classes": n_classes
    },
    training_args={
        "num_epochs": num_epochs, 
        "batch_size": batch_size,
        "learning_rate": 0.001  # Default is 1e-3
    },
    task="classification",
    realtime="auto",  # Options: "auto", "gui", "terminal", "none"
                      # "auto" tries GUI first, falls back to terminal
    save_plot=["loss", "accuracy"],  # Save plots after training
    seed=42  # For reproducibility
)

# Create model and trainer
model = MLP(input_size, hidden, n_classes)
trainer = Trainer(model, config)

# Train the model
history = trainer.train()

# Access results
print(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
print(f"Final training loss: {history['train_loss'][-1]:.4f}")
```

### Understanding the Components

**TrainingConfig**: This dataclass holds all configuration parameters:
- `dataset`: Can be a string identifier, dataset object, or tuple of (data, targets)
- `dataset_params`: Parameters passed to the dataset factory
- `training_args`: Dictionary with batch_size, learning_rate, num_epochs, etc.
- `task`: Type of task ("classification", "regression", "next-token-prediction")
- `realtime`: Visualization mode during training
- `save_plot`: List of metrics to plot and save after training

**Trainer**: The main class that handles the training loop:
- Automatically creates data loaders
- Handles device placement (CPU/GPU)
- Manages optimization and learning rate scheduling
- Computes and logs metrics
- Supports callbacks for customization

### What You Get

After training completes, you'll have:
1. **History dictionary**: Contains lists of metric values per epoch
   - `train_loss`, `val_loss`: Loss values
   - `accuracy`, `val_accuracy`: Accuracy (for classification)
   - Other metrics depending on task type
2. **Saved plots**: PNG files in your working directory
3. **Model checkpoints**: If configured (see Callbacks section)
4. **Logs**: Console output and optional WandB/TensorBoard logs

## Benchmark Multiple Models

One of the key features of arch_eval is comparing multiple architectures easily.

### Basic Benchmarking

```python
from arch_eval import Benchmark, BenchmarkConfig

# Define models to compare
models = [
    {"name": "Small MLP", "model": MLP(hidden=256)},
    {"name": "Large MLP", "model": MLP(hidden=512)},
    {"name": "Deep MLP", "model": MLP(hidden=256, num_layers=3)},
]

# Configure benchmark
config = BenchmarkConfig(
    dataset="synthetic classification",
    dataset_params={"n_samples": 10000, "n_features": 128, "n_classes": 64},
    training_args={"num_epochs": 10, "batch_size": 32, "learning_rate": 0.001},
    compare_metrics=["accuracy", "loss"],  # Metrics to extract from history
    parallel=True,  # Run models concurrently (uses threads)
    device="cpu"  # Force CPU for this example
)

# Run benchmark
benchmark = Benchmark(models, config)
results = benchmark.run()

# View results
print(results)
print("\nBest model by accuracy:")
print(results.loc[results["accuracy"].idxmax()])
```

### Benchmark Output

The `run()` method returns a pandas DataFrame with one row per model:

```
         name  accuracy      loss
0   Small MLP    0.8523    0.4231
1   Large MLP    0.8891    0.3156
2    Deep MLP    0.8756    0.3542
```

### Parallel vs Sequential

- `parallel=False`: Models run sequentially (safer for GPU memory)
- `parallel=True`: Models run in parallel using threads
- `use_processes=True`: Use process-based parallelism (requires cloudpickle)

**Note**: For GPU training, parallel execution may cause memory issues. Consider running sequentially or limiting parallel models.

## Hyperparameter Search

Find optimal hyperparameters automatically with grid or random search.

### Grid Search Example

Grid search tries all combinations of specified hyperparameters:

```python
from arch_eval import HyperparameterOptimizer, TrainingConfig

def model_fn():
    """Factory function that returns a fresh model instance."""
    return MLP(hidden=128)

# Base configuration
base_config = TrainingConfig(
    dataset="synthetic classification",
    dataset_params={"n_samples": 1000, "n_features": 128, "n_classes": 64},
    training_args={"num_epochs": 3},  # Fewer epochs for search
    task="classification",
    realtime="none"  # Disable live plots during search (faster)
)

# Define parameter grid
param_grid = {
    "learning_rate": [0.001, 0.01, 0.1],
    "hidden": [64, 128, 256],
    "batch_size": [16, 32]
}

# Create optimizer
optimizer = HyperparameterOptimizer(
    model_fn, 
    base_config, 
    param_grid,
    search_type="grid",  # Try all combinations
    metric="val_accuracy",  # Metric to optimize
    mode="max"  # Maximize the metric
)

# Run search
results = optimizer.run()

# View all results
print(results)

# Find best configuration
best_idx = results["val_accuracy"].idxmax()
print(f"\nBest configuration:")
print(results.loc[best_idx])
```

### Random Search Example

For large parameter spaces, random search is more efficient:

```python
param_grid = {
    "learning_rate": [0.0001, 0.001, 0.01, 0.1],
    "hidden": [32, 64, 128, 256, 512],
    "batch_size": [16, 32, 64, 128]
}

optimizer = HyperparameterOptimizer(
    model_fn, 
    base_config, 
    param_grid,
    search_type="random",  # Random sampling
    n_trials=10,  # Number of random combinations to try
    metric="val_loss",
    mode="min"  # Minimize validation loss
)

results = optimizer.run()
```

### Understanding Results

The results DataFrame contains:
- All tested hyperparameter combinations
- The target metric value for each trial
- Additional metrics from training history

You can analyze results to understand hyperparameter sensitivity:

```python
# Group by learning rate and see average performance
print(results.groupby("learning_rate")["val_accuracy"].mean())

# Plot results (requires matplotlib)
import matplotlib.pyplot as plt
plt.scatter(results["learning_rate"], results["val_accuracy"])
plt.xscale("log")
plt.xlabel("Learning Rate")
plt.ylabel("Validation Accuracy")
plt.show()
```

## Using Callbacks

Callbacks allow you to customize training behavior without modifying the core library.

### Built-in Callbacks

```python
from arch_eval import (
    Trainer, TrainingConfig, 
    EarlyStopping, ModelCheckpoint, 
    TensorBoardLogger, LRSchedulerLogger,
    SlopeEarlyStopping
)

# Configure with multiple callbacks
config = TrainingConfig(
    dataset="synthetic classification",
    dataset_params={"n_samples": 1000, "n_features": 20, "n_classes": 5},
    training_args={"num_epochs": 50, "batch_size": 32},
    task="classification",
    callbacks=[
        # Stop early if validation loss doesn't improve
        EarlyStopping(monitor="val_loss", patience=5, mode="min"),
        
        # Save best model checkpoint
        ModelCheckpoint(
            filepath="checkpoints/model.pt", 
            monitor="val_accuracy", 
            save_best_only=True,
            mode="max"
        ),
        
        # Log to TensorBoard
        TensorBoardLogger(log_dir="./logs"),
        
        # Log learning rate after each epoch
        LRSchedulerLogger()
    ],
    checkpoint_dir="./checkpoints"
)

model = nn.Linear(20, 5)
trainer = Trainer(model, config)
history = trainer.train()
```

### Available Callbacks

| Callback | Description | Key Parameters |
|----------|-------------|----------------|
| `EarlyStopping` | Stops training when metric stops improving | `monitor`, `patience`, `mode` |
| `ModelCheckpoint` | Saves model checkpoints | `filepath`, `monitor`, `save_best_only` |
| `LRSchedulerLogger` | Logs learning rates | None |
| `TensorBoardLogger` | Logs metrics to TensorBoard | `log_dir` |
| `SlopeEarlyStopping` | Stops when metric slope becomes flat | `monitor`, `window_size`, `threshold` |
| `GradientModifierCallback` | Custom gradient modification | `modifier_fn` |
| `TextGeneratorCallback` | Generates text samples for LMs | `prompt`, `generation_kwargs` |

### Creating Custom Callbacks

You can create custom callbacks by subclassing `Callback`:

```python
from arch_eval import Callback

class CustomCallback(Callback):
    def on_epoch_start(self, trainer, epoch):
        print(f"Starting epoch {epoch}")
    
    def on_epoch_end(self, trainer, epoch, metrics):
        print(f"Epoch {epoch} completed: {metrics}")
    
    def on_train_end(self, trainer):
        print("Training finished!")

# Use custom callback
config = TrainingConfig(
    ...,
    callbacks=[CustomCallback()]
)
```

## Next Steps

Now that you've learned the basics, explore:

1. **User Guide** - Detailed explanations of all features
2. **Examples** - Complete working examples for various use cases
3. **API Reference** - Full API documentation
4. **Advanced Features** - Distributed training, mixed precision, profiling, etc.
