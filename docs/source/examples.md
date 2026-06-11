# Usage Examples

This page provides several complete, working examples demonstrating the main features of **arch_eval**. Each example is self-contained and can be run as-is (with minor modifications for your specific use case).

## Table of Contents

- [Basic Training with MNIST](#basic-training-with-mnist)
- [Benchmarking Two MLP Variants](#benchmarking-two-mlp-variants)
- [Hyperparameter Search with Random Search](#hyperparameter-search-with-random-search)
- [Using Callbacks – Early Stopping and Checkpointing](#using-callbacks--early-stopping-and-checkpointing)
- [Custom Dataset from NumPy Arrays](#custom-dataset-from-numpy-arrays)
- [Distributed Training with DDP](#distributed-training-with-ddp)
- [Profiling and Video Recording](#profiling-and-video-recording)
- [Using a HuggingFace Dataset](#using-a-huggingface-dataset)
- [Custom Callback – Logging to File](#custom-callback--logging-to-file)
- [Using the Plugin System](#using-the-plugin-system)

---

## Basic Training with MNIST

Train a simple CNN on the MNIST digit classification dataset using torchvision data. This example demonstrates:
- Loading real-world image data
- Using transforms for normalization
- Training with GPU acceleration (if available)
- Saving plots and logging

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from arch_eval import Trainer, TrainingConfig
from torchvision import transforms

# ---------- Model Definition ----------
class SimpleCNN(nn.Module):
    """Simple Convolutional Neural Network for MNIST classification."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ---------- Data Transforms ----------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ---------- Configuration ----------
config = TrainingConfig(
    dataset="mnist",
    dataset_params={"root": "./data", "split": "train", "download": True},
    transform=transform,
    training_args={"batch_size": 64, "learning_rate": 0.001, "num_epochs": 5},
    task="classification",
    device="cuda" if torch.cuda.is_available() else "cpu",
    realtime=True,
    save_plot=["loss", "accuracy"],
    log_to_wandb=False,
    seed=42,
)

# ---------- Train ----------
model = SimpleCNN()
trainer = Trainer(model, config)
history = trainer.train()

print(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
```

### Key Points

1. **Dataset Loading**: The library automatically handles downloading and loading MNIST from torchvision
2. **Transforms**: Normalization improves training stability
3. **Device Selection**: Automatically uses GPU if available
4. **Visualization**: Real-time plots help monitor training progress

---

## Benchmarking Two MLP Variants

Compare a small and a large Multi-Layer Perceptron (MLP) on synthetic classification data.

```python
import torch.nn as nn
from arch_eval import Benchmark, BenchmarkConfig

class MLP(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 64)
        )

    def forward(self, x):
        return self.net(x)

models = [
    {"name": "Small MLP", "model": MLP(hidden=128)},
    {"name": "Large MLP", "model": MLP(hidden=512)},
]

config = BenchmarkConfig(
    dataset="synthetic classification",
    dataset_params={"n_samples": 5000, "n_features": 128, "n_classes": 64},
    training_args={"batch_size": 32, "learning_rate": 0.001, "num_epochs": 10},
    compare_metrics=["accuracy", "loss"],
    parallel=True,
    device="cpu",
)

benchmark = Benchmark(models, config)
results = benchmark.run()

print(results)
print(f"\nBest model: {results.loc[results['accuracy'].idxmax()]['name']}")
```

### Tips for Benchmarking

1. **Parallel Execution**: Use `parallel=True` for faster benchmarking
2. **Consistent Data**: All models see the same data splits for fair comparison
3. **Multiple Metrics**: Compare on various metrics (accuracy, loss, training time)

---

## Hyperparameter Search with Random Search

Optimize learning rate and hidden size for a regression model.

```python
import numpy as np
import torch.nn as nn
from arch_eval import HyperparameterOptimizer, TrainingConfig

class Regressor(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x)

def model_fn():
    return Regressor()

base_config = TrainingConfig(
    dataset="synthetic regression",
    dataset_params={"n_samples": 2000, "n_features": 20, "noise": 0.1},
    training_args={"num_epochs": 5, "batch_size": 32},
    task="regression",
    realtime=False,
)

param_grid = {
    "learning_rate": [0.0001, 0.001, 0.01, 0.1],
    "hidden": [32, 64, 128],
}

optimizer = HyperparameterOptimizer(
    model_fn, base_config, param_grid,
    search_type="random", n_trials=6,
    metric="val_mse", mode="min"
)

results = optimizer.run()
print("Best trial:")
print(results.loc[results["val_mse"].idxmin()])
```

### Grid Search vs Random Search

| Aspect | Grid Search | Random Search |
|--------|-------------|---------------|
| Coverage | Exhaustive | Sampling |
| Efficiency | Good for small spaces | Better for large spaces |
| Configuration | No n_trials needed | Specify n_trials |

---

## Using Callbacks – Early Stopping and Checkpointing

Train a model with early stopping and model checkpointing.

```python
from arch_eval import (
    Trainer, TrainingConfig,
    EarlyStopping, ModelCheckpoint, LRSchedulerLogger
)

config = TrainingConfig(
    dataset="synthetic classification",
    dataset_params={"n_samples": 1000, "n_features": 20, "n_classes": 5},
    training_args={"num_epochs": 50, "batch_size": 32, "learning_rate": 0.01},
    task="classification",
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=5, mode="min"),
        ModelCheckpoint(
            filepath="./checkpoints/model-{epoch:02d}.pt",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max"
        ),
        LRSchedulerLogger()
    ],
    checkpoint_dir="./checkpoints",
)

model = nn.Linear(20, 5)
trainer = Trainer(model, config)
history = trainer.train()
```

### Callback Parameters Explained

**EarlyStopping:**
- `monitor`: Which metric to track
- `patience`: How many epochs to wait before stopping
- `mode`: "min" for losses, "max" for accuracies

**ModelCheckpoint:**
- `filepath`: Where to save (can include `{epoch}` placeholder)
- `monitor`: Metric to track for saving decisions
- `save_best_only`: Only save when metric improves

---

## Custom Dataset from NumPy Arrays

Use your own data stored as NumPy arrays.

```python
import numpy as np
import torch
from arch_eval import Trainer, TrainingConfig

# Generate random data
np.random.seed(42)
X = np.random.randn(1000, 50).astype(np.float32)
y = (X.sum(axis=1) > 0).astype(np.int64)

config = TrainingConfig(
    dataset=(X, y),
    training_args={"batch_size": 64, "learning_rate": 0.001, "num_epochs": 5},
    task="classification",
)

model = torch.nn.Linear(50, 2)
trainer = Trainer(model, config)
trainer.train()
```

### Data Format Options

You can pass data in various formats:

```python
# Tuple of numpy arrays
dataset = (X_numpy, y_numpy)

# Tuple of torch tensors
dataset = (X_tensor, y_tensor)

# PyTorch Dataset instance
from torch.utils.data import TensorDataset
dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

# String identifier (built-in datasets)
dataset = "mnist"
dataset = "cifar10"
```

---

## Distributed Training with DDP

Launch script using `torchrun` for distributed training.

### Training Script (train_ddp.py)

```python
# train_ddp.py
import os
import torch
import torch.nn as nn
from arch_eval import Trainer, TrainingConfig, DistributedBackend

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc(x)

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])

config = TrainingConfig(
    dataset="synthetic classification",
    dataset_params={"n_samples": 10000, "n_features": 128, "n_classes": 10},
    training_args={"batch_size": 64, "num_epochs": 10},
    distributed_backend=DistributedBackend.DISTRIBUTED,
    distributed_world_size=world_size,
    distributed_rank=rank,
    dataset_shard={"num_shards": world_size, "shard_id": rank},
    device=f"cuda:{local_rank}",
)

model = Model()
trainer = Trainer(model, config)
trainer.train()
```

### Running the Script

```bash
# Train on 2 GPUs
torchrun --nproc_per_node=2 train_ddp.py
```

### Important Notes

1. **Batch Size**: Effective batch size is `batch_size × num_gpus`
2. **Learning Rate**: Consider scaling LR with batch size
3. **Data Sharding**: Each GPU processes different samples

---

## Profiling and Video Recording

Enable profiling and record training videos.

```python
config = TrainingConfig(
    dataset="synthetic classification",
    dataset_params={"n_samples": 1000, "n_features": 20, "n_classes": 5},
    training_args={"num_epochs": 5},
    profiler={
        "enabled": True,
        "activities": ["cpu", "cuda"],
        "schedule": {"wait": 1, "warmup": 1, "active": 2},
        "trace_path": "./profiler_trace"
    },
    save_video=["loss"],
    realtime=False,
)

model = nn.Linear(20, 5)
trainer = Trainer(model, config)
trainer.train()
```

### Understanding Profiler Output

The profiler generates trace files viewable in Chrome at `chrome://tracing`.

### Video Recording Requirements

- Requires `ffmpeg` installed
- Videos saved as MP4 format
- Shows metric evolution over time

---

## Using a HuggingFace Dataset

Load the IMDB dataset and train a text classifier.

```python
from datasets import load_dataset
from arch_eval import Trainer, TrainingConfig
import torch
import torch.nn as nn

dataset = load_dataset("imdb")

class TextClassifier(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids):
        emb = self.embedding(input_ids).mean(dim=1)
        return self.fc(emb)

# Simple tokenization for demo
def simple_tokenize(text, max_length=128):
    words = text.lower().split()
    tokens = [hash(word) % 10000 for word in words[:max_length]]
    return tokens + [0] * (max_length - len(tokens))

# Prepare subset
input_ids = torch.tensor([simple_tokenize(item['text']) for item in dataset['train'][:1000]])
labels = torch.tensor([item['label'] for item in dataset['train'][:1000]])

config = TrainingConfig(
    dataset=(input_ids, labels),
    training_args={"batch_size": 16, "num_epochs": 3},
    task="classification",
)

model = TextClassifier()
trainer = Trainer(model, config)
trainer.train()
```

### Note on Production Usage

For real applications:
1. Use proper tokenization (e.g., from transformers library)
2. Consider pre-trained models
3. Use streaming for large datasets

---

## Custom Callback – Logging to File

Create a callback that writes metrics to CSV.

```python
import csv
from arch_eval import Callback

class CSVLogger(Callback):
    def __init__(self, filename="log.csv"):
        self.filename = filename
        self.file = None
        self.writer = None
    
    def on_train_start(self, trainer):
        self.file = open(self.filename, "w", newline="")
    
    def on_epoch_end(self, trainer, epoch, metrics):
        if self.writer is None:
            self.writer = csv.DictWriter(
                self.file, 
                fieldnames=["epoch"] + list(metrics.keys())
            )
            self.writer.writeheader()
        row = {"epoch": epoch, **metrics}
        self.writer.writerow(row)
        self.file.flush()
    
    def on_train_end(self, trainer):
        if self.file:
            self.file.close()

config = TrainingConfig(
    dataset="synthetic classification",
    dataset_params={"n_samples": 500, "n_features": 20, "n_classes": 5},
    training_args={"num_epochs": 10, "batch_size": 32},
    task="classification",
    callbacks=[CSVLogger("training_log.csv")]
)
```

### Callback Lifecycle Methods

Available methods to override:
- `on_train_start(trainer)` - Before training begins
- `on_epoch_start(trainer, epoch)` - At start of each epoch
- `on_batch_end(trainer, batch_idx, loss)` - After each batch
- `on_epoch_end(trainer, epoch, metrics)` - After each epoch
- `on_train_end(trainer)` - After training completes

---

## Using the Plugin System

Plugins extend arch_eval functionality globally.

### Step 1: Create Plugin File

```python
# my_plugin.py
from arch_eval.plugins import hook

@hook("on_epoch_start")
def epoch_start(trainer, epoch):
    print(f"Starting epoch {epoch}!")

@hook("on_train_end")
def training_end(trainer):
    print("Training completed!")
```

### Step 2: Discover and Use Plugins

```python
from arch_eval import Trainer, TrainingConfig, discover_plugins

discover_plugins(["./"])  # Scan current directory

config = TrainingConfig(
    dataset="synthetic classification",
    dataset_params={"n_samples": 500, "n_features": 20, "n_classes": 5},
    training_args={"num_epochs": 5, "batch_size": 32},
    task="classification",
)

model = nn.Linear(20, 5)
trainer = Trainer(model, config)
trainer.train()
```

### Available Hook Points

- `on_train_start` - Before training loop
- `on_train_end` - After training loop
- `on_epoch_start/end` - At epoch boundaries
- `on_batch_start/end` - At batch boundaries
- `on_log` - When metrics are logged
- `on_exception` - When an error occurs

---

## Summary

These examples cover the main features of arch_eval:

1. **Basic Training**: Quick setup for single model training
2. **Benchmarking**: Compare multiple architectures
3. **Hyperparameter Search**: Optimize parameters
4. **Callbacks**: Customize training behavior
5. **Custom Data**: Use your own datasets
6. **Distributed Training**: Scale to multiple GPUs
7. **Profiling**: Analyze performance
8. **External Datasets**: Integrate with Hugging Face
9. **Custom Extensions**: Create callbacks and plugins

For more details, refer to the [User Guide](guide.md) and [API Reference](api.md).
