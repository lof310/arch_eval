# Usage Examples

This page provides several complete examples demonstrating the main features of **arch_eval**.

## Basic Training with MNIST

Train a simple CNN on MNIST using torchvision data.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from arch_eval import Trainer, TrainingConfig
from torchvision import transforms

# ---------- Model ----------
class SimpleCNN(nn.Module):
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

# ---------- Configuration ----------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

config = TrainingConfig(
    dataset="mnist",
    dataset_params={"root": "./data", "split": "train", "download": True},
    transform=transform,
    training_args={
        "batch_size": 64,
        "learning_rate": 0.001,
        "num_epochs": 5,
    },
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

## Benchmarking Two MLP Variants

Compare a small and a large MLP on synthetic data.

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
    dataset_params={
        "n_samples": 5000,
        "n_features": 128,
        "n_classes": 64,
        "n_informative": 64,
    },
    training_args={
        "batch_size": 32,
        "learning_rate": 0.001,
        "num_epochs": 10,
    },
    compare_metrics=["accuracy", "loss"],
    parallel=True,               # run two models concurrently
    use_processes=False,         # use threads (safe for CPU; for GPU, keep sequential)
    device="cpu",                # force CPU for this example
)

benchmark = Benchmark(models, config)
results = benchmark.run()
print(results)
```

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
    dataset_params={
        "n_samples": 2000,
        "n_features": 20,
        "noise": 0.1,
    },
    training_args={
        "num_epochs": 5,
        "batch_size": 32,
    },
    task="regression",
    realtime=False,   # disable plots during search
)

param_grid = {
    "learning_rate": [0.0001, 0.001, 0.01, 0.1],
    "hidden": [32, 64, 128],
}

optimizer = HyperparameterOptimizer(
    model_fn,
    base_config,
    param_grid,
    search_type="random",
    n_trials=6,              # try 6 random combinations
    metric="val_mse",
    mode="min",
)

results = optimizer.run()
print("Best trial:")
print(results.loc[results["val_mse"].idxmin()])
```

## Using Callbacks – Early Stopping and Checkpointing

Train a model with early stopping and model checkpointing.

```python
from arch_eval import Trainer, TrainingConfig
from arch_eval import EarlyStopping, ModelCheckpoint

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
        )
    ],
    checkpoint_dir="./checkpoints",
)

model = nn.Linear(20, 5)
trainer = Trainer(model, config)
history = trainer.train()
```

## Custom Dataset from NumPy Arrays

Use your own data stored as NumPy arrays.

```python
import numpy as np
import torch
from arch_eval import Trainer, TrainingConfig

# Generate random data
X = np.random.randn(1000, 50).astype(np.float32)
y = (X.sum(axis=1) > 0).astype(np.int64)   # binary labels

config = TrainingConfig(
    dataset=(X, y),          # tuple (data, targets)
    training_args={"batch_size": 64, "learning_rate": 0.001, "num_epochs": 5},
    task="classification",
)

model = torch.nn.Linear(50, 2)   # 2 classes
trainer = Trainer(model, config)
trainer.train()
```

## Distributed Training with DDP

Launch script using `torchrun`. Assume the script is `train_ddp.py`.

```python
# train_ddp.py
import torch
import torch.nn as nn
import torch.distributed as dist
from arch_eval import Trainer, TrainingConfig, DistributedBackend

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc(x)

# Get rank and world size from environment (set by torchrun)
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

config = TrainingConfig(
    dataset="synthetic classification",
    dataset_params={"n_samples": 10000, "n_features": 128, "n_classes": 10},
    training_args={"batch_size": 64, "num_epochs": 10, "learning_rate": 0.01},
    distributed_backend=DistributedBackend.DISTRIBUTED,
    distributed_world_size=world_size,
    distributed_rank=rank,
    # Optional: shard dataset so each GPU sees different samples
    dataset_shard={"num_shards": world_size, "shard_id": rank},
    device="cuda",
)

model = Model()
trainer = Trainer(model, config)
trainer.train()
```

Run with:

```bash
torchrun --nproc_per_node=2 train_ddp.py
```

## Profiling and Video Recording

Enable the profiler and record a video of the loss curve.

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
    save_video=["loss"],          # record loss over time
    realtime=False,               # disable live window (optional)
)

model = nn.Linear(20, 5)
trainer = Trainer(model, config)
trainer.train()
# After training, a video file `loss.mp4` will be created (if ffmpeg is installed).
```

## Using a HuggingFace Dataset

Load the IMDB dataset from Hugging Face and train a simple text classifier.

```python
from datasets import load_dataset
from arch_eval import Trainer, TrainingConfig
import torch.nn as nn

# Load dataset
dataset = load_dataset("imdb")

class TextClassifier(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids):
        # input_ids: (batch, seq_len)
        emb = self.embedding(input_ids).mean(dim=1)  # average pooling
        return self.fc(emb)

config = TrainingConfig(
    dataset=dataset["train"],          # pass the dataset object
    dataset_streaming=False,           # set to True for IterableDataset
    training_args={"batch_size": 16, "num_epochs": 1},
    task="classification",
)

model = TextClassifier()
trainer = Trainer(model, config)
trainer.train()
```
_(Note: This is a simplified example; real text classification requires proper tokenization and possibly a collate function.)_


## Custom Callback – Logging to File

Create a callback that writes metrics to a CSV file.

```python
import csv
from arch_eval import Callback

class CSVLogger(Callback):
    def __init__(self, filename="log.csv"):
        self.filename = filename
        self.file = open(filename, "w", newline="")
        self.writer = None

    def on_log(self, trainer, metrics, step):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=["step"] + list(metrics.keys()))
            self.writer.writeheader()
        row = {"step": step, **metrics}
        self.writer.writerow(row)
        self.file.flush()

    def on_train_end(self, trainer):
        self.file.close()

# Use it
config = TrainingConfig(
    ...,
    callbacks=[CSVLogger("training_log.csv")]
)
```

## Using the Plugin System

Create a simple plugin that prints a message at the start of each epoch.

**File: `my_plugin.py`** (place it in your Python path)

```python
from arch_eval.plugins import hook

@hook("on_epoch_start")
def epoch_start(trainer, epoch):
    print(f"Starting epoch {epoch}!")
```

Now run any training script; the plugin will be discovered automatically and the hook will execute.

---

_**These examples cover most of the library’s functionality. For further details, refer to the [Guide](guide.md) and [API Reference](api.md).**_
