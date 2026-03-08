# User Guide

This guide walks you through the main features of **arch_eval** and shows how to use them effectively.

```{contents} Contents
:depth: 2
:local:
```

## Installation

```bash
# Clone the repository
git clone --depth=1 https://github.com/lof310/arch_eval.git
cd arch_eval

# Install in Development Model (Recommended)
pip install -e .

# Install Normally
pip install .

# Or install from PyPI
pip install arch_eval
```

**Dependencies**:
- Python ≥ 3.8
- PyTorch ≥ 1.9
- pandas, numpy, scikit‑learn, psutil, matplotlib, seaborn
- Optional: wandb, transformer_engine (for FP8), ffmpeg (for video)

## Core Concepts

arch_eval is built around a few central objects:

- **`TrainingConfig`** – holds all parameters for a single training run.
- **`Trainer`** – trains a single model and returns a history.
- **`BenchmarkConfig`** – holds parameters for comparing multiple models.
- **`Benchmark`** – runs several models (sequentially or in parallel) and returns a comparison table.
- **`HyperparameterOptimizer`** – performs grid or random search over a hyperparameter space.

All configuration is done via dataclasses, making it easy to serialise and share.

## Basic Training

Here’s the simplest possible training script:

```python
import torch.nn as nn
from arch_eval import Trainer, TrainingConfig

# 1. Define your model
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(20, 10)

    def forward(self, x):
        return self.fc(x)

# 2. Create configuration
config = TrainingConfig(
    dataset="synthetic classification",   # built-in synthetic data
    dataset_params={
        "n_samples": 1000,
        "n_features": 20,
        "n_classes": 10,
    },
    training_args={
        "batch_size": 32,
        "learning_rate": 0.001,
        "num_epochs": 5,
    },
    task="classification",
    realtime=True,                       # show live plot window
)

# 3. Instantiate trainer and run
model = SimpleMLP()
trainer = Trainer(model, config)
history = trainer.train()

print(history["val_accuracy"][-1])       # final validation accuracy
```

### Explanation

- `dataset="synthetic classification"` tells the library to generate a synthetic classification dataset.
- `dataset_params` are passed to `sklearn.datasets.make_classification`.
- `training_args` holds the usual training hyperparameters.
- `realtime=True` opens a matplotlib window that updates every `viz_interval` steps (default 10). It shows metric curves and system resource usage.

After training, `history` is a dictionary mapping metric names (like `"train_loss"`, `"val_accuracy"`) to lists of values per epoch.

## Configuration Deep Dive

All configuration classes inherit from `BaseConfig`, which provides common fields. Below are the most important ones.

### Data Specification

You can specify data in many ways:

| Method | Example |
|--------|---------|
| **Synthetic** | `dataset="synthetic classification"` with `dataset_params` |
| **Torchvision** | `dataset="cifar10"`, `dataset_params={"split": "train"}` |
| **Hugging Face** | `dataset = load_dataset("cifar10")` and pass the dataset object |
| **Custom Dataset** | Pass a `torch.utils.data.Dataset` instance |
| **Tensor/Dict** | Pass a tuple `(data, targets)` or a dict with `"data"` and `"targets"` |
| **Streaming** | Set `dataset_streaming=True` for Hugging Face `IterableDataset` |

For distributed training, you can also shard datasets using `dataset_shard = {"num_shards": 4, "shard_id": rank}`.

### Device and Precision

- `device` – auto‑selects `"cuda"` if available, else `"cpu"`.
- `dtype` – default `torch.float32`.
- `mixed_precision=True` enables AMP. Use `mixed_precision_dtype` to choose `"float16"`, `"bfloat16"`, or `"fp8"` (experimental, requires Transformer Engine).

### Logging and Visualization

- `log_interval` – how often to print to console (steps).
- `viz_interval` – how often to update the realtime window.
- `save_plot` – list of metric names; at the end of training, PNG plots are saved.
- `save_video` – list of metric names; a video of the metric evolution is created (requires ffmpeg).
- `log_to_wandb` – enable Weights & Biases logging. Also set `wandb_project` and optionally `wandb_run_name`.

### Callbacks

Callbacks are passed via the `callbacks` list. Built‑in callbacks:

```python
from arch_eval import EarlyStopping, ModelCheckpoint, TensorBoardLogger

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5),
    ModelCheckpoint(filepath="checkpoints/epoch-{epoch}.pt", monitor="val_accuracy", save_best_only=True),
    TensorBoardLogger(log_dir="./logs")
]
```

You can also write your own by subclassing `Callback` and overriding any of its methods.

## Benchmarking Multiple Models

To compare several architectures, use `Benchmark`:

```python
from arch_eval import Benchmark, BenchmarkConfig

models = [
    {"name": "MLP Small", "model": MLP(hidden=128)},
    {"name": "MLP Large", "model": MLP(hidden=256)},
]

bench_config = BenchmarkConfig(
    dataset="synthetic classification",
    dataset_params={"n_samples": 5000, "n_features": 64, "n_classes": 20},
    training_args={"num_epochs": 10, "batch_size": 64},
    compare_metrics=["accuracy", "loss"],
    parallel=True,          # run models concurrently
    use_processes=False,    # use threads (safe for CPU; for GPU, keep sequential or threads)
)

benchmark = Benchmark(models, bench_config)
results = benchmark.run()   # returns pandas DataFrame
print(results)
```

- `parallel=True` runs models in parallel using threads (or processes if `use_processes=True`). For GPU training, parallelism may cause memory issues – use with caution or keep sequential.
- `compare_metrics` lists the metrics you want to extract from each model’s history. They must appear in the history (e.g., `"accuracy"`, `"val_loss"`).
- The resulting DataFrame contains a row per model with the final value of each requested metric.

## Hyperparameter Optimization

The `HyperparameterOptimizer` class provides grid and random search:

```python
from arch_eval import HyperparameterOptimizer, TrainingConfig

def model_fn():
    return MLP()   # must return a fresh model each time

base_config = TrainingConfig(
    dataset="synthetic classification",
    dataset_params={"n_samples": 1000, "n_features": 64, "n_classes": 10},
    training_args={"num_epochs": 5},
    task="classification",
    realtime=False,   # disable live plots during search
)

param_grid = {
    "learning_rate": [0.001, 0.01, 0.1],
    "batch_size": [32, 64],
}

optimizer = HyperparameterOptimizer(
    model_fn,
    base_config,
    param_grid,
    search_type="grid",           # or "random"
    metric="val_accuracy",
    mode="max",
)

results = optimizer.run()
print(results)
```

- `param_grid` keys can be either top‑level attributes of `TrainingConfig` (like `batch_size`) or keys inside `training_args` (like `learning_rate`). The optimizer updates the config accordingly.
- For random search, set `search_type="random"` and optionally `n_trials`.
- The returned DataFrame includes all tried hyperparameters and the target metric.

## Distributed Training

arch_eval supports three distributed backends:

- `DATAPARALLEL` – `torch.nn.DataParallel` (simple, but slower due to GIL)
- `DISTRIBUTED` – `torch.nn.parallel.DistributedDataParallel` (recommended for multi‑GPU)
- `FSDP` – Fully Sharded Data Parallel (PyTorch ≥ 1.12)

To use DDP:

```python
from arch_eval import TrainingConfig, DistributedBackend

config = TrainingConfig(
    ...,
    distributed_backend=DistributedBackend.DISTRIBUTED,
    distributed_world_size=2,        # number of processes
    distributed_rank=0,               # set per process
    distributed_master_addr="127.0.0.1",
    distributed_master_port="29500",
)
```

You must launch your script with `torch.distributed.launch` or `torchrun`. For example:

```bash
torchrun --nproc_per_node=2 train.py
```

In the script, each process will have a different rank; the trainer automatically handles the wrapping and data sharding if you set `dataset_shard` accordingly.

## Using Plugins

Plugins are external modules that can register global hooks. They are discovered automatically if their module name starts with `arch_eval_plugin_` or ends with `_plugin`.

To create a plugin:

1. Create a Python file (e.g., `my_plugin.py`).
2. Define functions decorated with `@hook("hook_name")`.
3. Place it somewhere in your `PYTHONPATH`.

Example plugin:

```python
from arch_eval.plugins import hook

@hook("on_epoch_end")
def log_extra_info(trainer, epoch, metrics):
    print(f"Epoch {epoch} done, loss={metrics.get('val_loss', -1):.4f}")
```

You can also register local hooks directly on a `Trainer` instance via its `plugin_manager`:

```python
def my_local_hook(trainer, batch_idx, loss):
    print(f"Batch {batch_idx} loss: {loss}")

trainer.plugin_manager.register_local_hook("on_batch_end", my_local_hook)
```

## Advanced Features

### Gradient Checkpointing

Enable to reduce memory for large models:

```python
config.gradient_checkpointing = True
config.gradient_checkpointing_modules = ["layer1", "layer2"]   # optional, if you want to checkpoint only specific modules
```

### Mixed Precision with FP8

Requires NVIDIA Transformer Engine:

```python
config.mixed_precision = True
config.mixed_precision_dtype = "fp8"
```

### Profiling

Enable PyTorch profiler to trace execution:

```python
config.profiler = {
    "enabled": True,
    "activities": ["cpu", "cuda"],
    "schedule": {"wait": 1, "warmup": 1, "active": 3},
    "trace_path": "./traces"
}
```

### Custom Loss Functions

You can provide your own loss function:

```python
def my_loss(output, target):
    return torch.nn.functional.mse_loss(output, target)

config.loss_function = my_loss
```

### Model Output Transformation

If your model returns a tuple or a dict, you can transform it to the expected format (tensor of logits) using `model_output_transform`:

```python
def transform(output):
    return output["logits"] if isinstance(output, dict) else output[0]

config.model_output_transform = transform
```

## Logging and Monitoring

- Console logging is configured via `setup_logging(level="INFO")`.
- WandB integration: set `log_to_wandb=True` and `wandb_project`.
- TensorBoard: use the `TensorBoardLogger` callback.
- Real‑time window: `realtime=True` (requires an interactive backend, e.g., TkAgg).
- Video recording: `save_video=["loss", "accuracy"]` – frames are saved and assembled with ffmpeg at the end.

## Best Practices

1. **Use `seed` for reproducibility** – set `seed=42` and optionally `deterministic=True`.
2. **Start with synthetic data** to quickly test your pipeline.
3. **Monitor GPU memory** with `memory_summary()` or the real‑time window.
4. **For hyperparameter search**, disable realtime plots (`realtime=False`) to avoid GUI overhead.
5. **When benchmarking on GPU**, prefer sequential execution or threads; processes may not work well with CUDA.
6. **Save checkpoints regularly** with `ModelCheckpoint` to recover from interruptions.
7. **Use `profiler` to identify bottlenecks** in your data loading or model forward/backward.

## Next Steps

- See the [Examples](examples.md) page for complete, runnable scripts.
- Browse the [API Reference](api.md) for detailed signatures and parameters.
