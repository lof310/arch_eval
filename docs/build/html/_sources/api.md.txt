# API Reference

This page documents all public classes, functions, and configuration objects provided by **arch_eval**.

## Core Classes

### `Trainer`

```{eval-rst}
.. autoclass:: arch_eval.Trainer
   :members:
   :undoc-members:
   :show-inheritance:
```
**Constructor**

```python
Trainer(model: nn.Module, config: TrainingConfig)
```

- **model**: PyTorch `nn.Module` to train.
- **config**: [`TrainingConfig`](#TrainingConfig) object with all training parameters.

**Methods**

- `train() -> Dict[str, List[float]]` – Runs the training loop and returns a history dictionary mapping metric names to lists of values (one per epoch).
- `load_checkpoint(path: str, load_optimizer: bool = True, load_scheduler: bool = True, weights_only: bool = False)` – Loads a saved checkpoint.

---

### `Benchmark`

```{eval-rst}
.. autoclass:: arch_eval.Benchmark
   :members:
   :undoc-members:
   :show-inheritance:
```

**Constructor**

```python
Benchmark(models: List[Dict[str, Any]], config: BenchmarkConfig)
```

- **models**: List of dictionaries, each with keys `"name"` (optional) and `"model"` (a `nn.Module`).
- **config**: [`BenchmarkConfig`](#BenchmarkConfig) object.

**Methods**

- `run() -> pd.DataFrame` – Executes the benchmark and returns a pandas DataFrame with results.

---

### `HyperparameterOptimizer`

```{eval-rst}
.. autoclass:: arch_eval.HyperparameterOptimizer
   :members:
   :undoc-members:
   :show-inheritance:
```

**Constructor**

```python
HyperparameterOptimizer(
    model_fn: Callable,
    base_config: TrainingConfig,
    param_grid: Dict[str, List[Any]],
    search_type: str = "grid",
    n_trials: Optional[int] = None,
    metric: str = "val_loss",
    mode: str = "min"
)
```

- **model_fn**: A zero‑argument callable that returns a new model instance.
- **base_config**: Base [`TrainingConfig`](#TrainingConfig) that will be copied and updated per trial.
- **param_grid**: Dictionary mapping parameter names to lists of values to try.
- **search_type**: `"grid"` or `"random"`.
- **n_trials**: Number of random trials (ignored for grid search).
- **metric**: Metric to optimize (must appear in the training history).
- **mode**: `"min"` or `"max"`.

**Methods**

- `run() -> pd.DataFrame` – Runs the search and returns a DataFrame with all trials and the target metric.

---

### `PluginManager`

```{eval-rst}
.. autoclass:: arch_eval.PluginManager
   :members:
   :undoc-members:
   :show-inheritance:
```

**Methods**

- `discover_plugins(plugin_paths: Optional[List[str]] = None)` – Scans for plugins (modules whose name starts with `arch_eval_plugin_` or ends with `_plugin`).
- `register_local_hook(hook_name: str, func: Callable)` – Registers a hook function for the current trainer instance.
- `execute_hook(hook_name: str, *args, **kwargs) -> List[Any]` – Executes all global and local hooks for the given hook.
- `get_plugins() -> Dict[str, Any]` – Returns information about discovered plugins.

---

## Configuration Classes

### `TrainingConfig`

```{eval-rst}
.. autoclass:: arch_eval.TrainingConfig
   :members:
   :undoc-members:
   :show-inheritance:
```

Inherits from [`BaseConfig`](#BaseConfig). All fields are configurable via the constructor.

**Key fields (in addition to `BaseConfig` fields):**

- `optimizers: List[Dict[str, Any]]` – List of optimizer specifications (e.g., `{"type": "adam", "lr": 0.001}`). Defaults to a single Adam optimizer.
- `schedulers: List[Dict[str, Any]]` – List of scheduler specs, each with a `"type"` (e.g., `"step"`, `"cosine"`, `"reduce_on_plateau"`) and optional `"optimizer"` index.
- `training_args: Dict[str, Any]` – Holds `batch_size`, `learning_rate`, `num_epochs`, etc.
- `task: Union[str, TaskType]` – One of `"classification"`, `"regression"`, `"next-token-prediction"` or a custom object with a `loss_function`.
- `mixed_precision: bool` – Enable automatic mixed precision (AMP).
- `mixed_precision_dtype: MixedPrecisionDtype` – `"float16"`, `"bfloat16"`, or `"fp8"`.
- `distributed_backend: DistributedBackend` – `"none"`, `"dp"` (DataParallel), `"ddp"` (DistributedDataParallel), or `"fsdp"`.
- `callbacks: List[Callback]` – List of callback instances.
- `checkpoint_dir: Optional[str]` – Directory to save checkpoints.
- `early_stopping_patience: Optional[int]` – Patience for early stopping.
- `gradient_clip: Optional[float]` – Max norm for gradient clipping.
- `profiler: Optional[Dict]` – Profiler configuration (e.g., `{"enabled": True, "activities": ["cpu", "cuda"]}`).

---

### `BenchmarkConfig`

```{eval-rst}
.. autoclass:: arch_eval.BenchmarkConfig
   :members:
   :undoc-members:
   :show-inheritance:
```

Inherits from [`BaseConfig`](#BaseConfig). Adds:

- `training_args: Dict[str, Any]` – Training parameters (batch size, learning rate, epochs) used for all models.
- `task: Union[str, TaskType]` – Task type.
- `parallel: bool` – Whether to run models in parallel.
- `compare_metrics: List[str]` – Metrics to extract from each model’s history.
- `max_workers: Optional[int]` – Maximum parallel workers.
- `use_processes: bool` – Use process‑based (rather than thread‑based) parallelism.
- `timeout_seconds: int` – Timeout per model.
- `retry_failed: bool` – Whether to retry failed models.

---

### `BaseConfig`

```{eval-rst}
.. autoclass:: arch_eval.core.config.BaseConfig
   :members:
   :undoc-members:
   :show-inheritance:
```

**Common fields:**

- **Data**:
  - `dataset: Any` – A dataset object, string identifier, or dictionary.
  - `dataset_params: Dict` – Parameters for synthetic/torchvision datasets.
  - `transform`, `target_transform`, `collate_fn` – Optional callables.
  - `dataset_streaming: bool` – Use streaming (IterableDataset) for Hugging Face datasets.
- **Computation**:
  - `dtype: torch.dtype` – Default `torch.float32`.
  - `device: Optional[str]` – Auto‑selected if `None`.
- **Logging & Viz**:
  - `viz_interval`, `log_interval`, `eval_interval` – Step intervals.
  - `realtime: bool` – Enable live plotting window.
  - `save_video: List[str]` – List of metric names to record as video.
  - `save_plot: List[str]` – List of metric names to save as final plots.
  - `log_to_wandb`, `wandb_project`, etc.
- **Reproducibility**:
  - `seed: Optional[int]`
  - `deterministic: bool`

---

### Enums

- `TaskType`: `REGRESSION`, `CLASSIFICATION`, `NEXT_TOKEN_PREDICTION`
- `DistributedBackend`: `NONE`, `DATAPARALLEL`, `DISTRIBUTED`, `FSDP`
- `MixedPrecisionDtype`: `FLOAT16`, `BFLOAT16`, `FP8`

---

## Callbacks

All callbacks inherit from `Callback` and can be registered via the `callbacks` list in `TrainingConfig`.

### `Callback`

```{eval-rst}
.. autoclass:: arch_eval.Callback
   :members:
   :undoc-members:
   :show-inheritance:
```

All methods are no‑ops by default; override the ones you need.

### Built‑in Callbacks

- **`EarlyStopping`** – Stops training when a monitored metric stops improving.
- **`ModelCheckpoint`** – Saves model checkpoints.
- **`LRSchedulerLogger`** – Logs learning rates after each epoch.
- **`TensorBoardLogger`** – Logs metrics to TensorBoard.

---

## Data Handling

### `DatasetHandler`

```{eval-rst}
.. autoclass:: arch_eval.data.DatasetHandler
   :members:
   :undoc-members:
   :show-inheritance:
```

Internal class used by `Trainer` to prepare data loaders. Usually you do not need to instantiate it directly.

### Synthetic Dataset Classes

#### `SyntheticDataset`

```{eval-rst}
.. autoclass:: arch_eval.data.SyntheticDataset
   :members:
   :undoc-members:
   :show-inheritance:
```

A simple `torch.utils.data.Dataset` wrapping synthetic data tensors for traditional ML tasks (classification, regression, etc.).

#### `TextDataset`

```{eval-rst}
.. autoclass:: arch_eval.data.TextDataset
   :members:
   :undoc-members:
   :show-inheritance:
```

Dataset for language modeling with token sequences. Returns `(input_ids, labels)` pairs suitable for transformer training.

**Example:**
```python
from arch_eval.data import TextDataset
import torch

input_ids = torch.randint(0, 1000, (1000, 128))  # 1000 samples, seq length 128
labels = input_ids.clone()  # For causal LM, labels are same as input
dataset = TextDataset(input_ids, labels)
```

#### `ImageDataset`

```{eval-rst}
.. autoclass:: arch_eval.data.ImageDataset
   :members:
   :undoc-members:
   :show-inheritance:
```

Dataset for vision tasks with images. Returns `(images, labels)` pairs with optional transforms.

**Example:**
```python
from arch_eval.data import ImageDataset
import torch

images = torch.rand(1000, 3, 32, 32)  # 1000 RGB images, 32x32
labels = torch.randint(0, 10, (1000,))  # 10 classes
dataset = ImageDataset(images, labels)
```

#### `VisionLanguageDataset`

```{eval-rst}
.. autoclass:: arch_eval.data.VisionLanguageDataset
   :members:
   :undoc-members:
   :show-inheritance:
```

Dataset for vision-language multi-modal tasks. Returns dictionaries with `pixel_values`, `input_ids`, `attention_mask`, and `labels`.

**Example:**
```python
from arch_eval.data import VisionLanguageDataset
import torch

images = torch.rand(500, 3, 32, 32)  # 500 images
input_ids = torch.randint(0, 1000, (500, 64))  # Token sequences
attention_mask = torch.ones(500, 64)
labels = torch.randint(0, 10, (500,))
dataset = VisionLanguageDataset(images, input_ids, attention_mask, labels)
```

### Synthetic Dataset Generation Functions

#### `create_synthetic_dataset`

```python
create_synthetic_dataset(dataset_type: str, params: Dict[str, Any]) -> SyntheticDataset
```

Creates a synthetic dataset for traditional ML tasks. Supported types:
- `"classification"`, `"regression"`, `"blobs"`, `"circles"`, `"moons"`
- `"friedman1"`, `"friedman2"`, `"friedman3"`, `"sparse_uncorrelated"`, `"multilabel"`

**Example:**
```python
from arch_eval.data import create_synthetic_dataset

# Binary classification
dataset = create_synthetic_dataset("classification", {
    "n_samples": 1000,
    "n_features": 20,
    "n_classes": 2,
    "n_informative": 10
})
```

#### `create_synthetic_text_dataset`

```python
create_synthetic_text_dataset(params: Dict[str, Any]) -> TextDataset
```

Creates synthetic text data for language modeling. Generates token sequences with configurable structure.

**Parameters:**
- `vocab_size`: Size of vocabulary (default: 1000)
- `seq_length`: Sequence length (default: 128)
- `n_samples`: Number of samples (default: 1000)
- `entropy`: Randomness level 0-1, lower = more pattern (default: 0.8)
- `random_state`: Random seed (default: 42)

**Example:**
```python
from arch_eval.data import create_synthetic_text_dataset

# Generate synthetic text for transformer pretraining
dataset = create_synthetic_text_dataset({
    "vocab_size": 32000,
    "seq_length": 512,
    "n_samples": 10000,
    "entropy": 0.7  # Some structure, not purely random
})

config = TrainingConfig(
    dataset="synthetic text",
    dataset_params={
        "vocab_size": 32000,
        "seq_length": 512,
        "n_samples": 10000
    },
    task="next-token-prediction",
    training_args={"batch_size": 32, "num_epochs": 10}
)
```

#### `create_synthetic_image_dataset`

```python
create_synthetic_image_dataset(params: Dict[str, Any]) -> ImageDataset
```

Creates synthetic image data for vision tasks. Supports random noise, gradients, and geometric shapes.

**Parameters:**
- `img_size`: Image size as int or tuple (H, W) (default: 32)
- `channels`: Number of channels (default: 3)
- `n_samples`: Number of samples (default: 1000)
- `n_classes`: Number of classes (default: 10)
- `pattern`: Type of pattern - `'random'`, `'gradient'`, `'shapes'` (default: `'random'`)
- `random_state`: Random seed (default: 42)

**Example:**
```python
from arch_eval.data import create_synthetic_image_dataset

# Generate synthetic images with geometric shapes
dataset = create_synthetic_image_dataset({
    "img_size": 64,
    "channels": 3,
    "n_samples": 5000,
    "n_classes": 10,
    "pattern": "shapes"  # Squares, circles, triangles, lines
})

config = TrainingConfig(
    dataset="synthetic image",
    dataset_params={
        "img_size": 64,
        "channels": 3,
        "pattern": "shapes"
    },
    task="classification",
    training_args={"batch_size": 64, "num_epochs": 20}
)
```

#### `create_synthetic_vision_language_dataset`

```python
create_synthetic_vision_language_dataset(params: Dict[str, Any]) -> VisionLanguageDataset
```

Creates synthetic vision-language paired data for multi-modal tasks.

**Parameters:**
- `img_size`: Image size (default: 32)
- `channels`: Number of image channels (default: 3)
- `vocab_size`: Vocabulary size for text (default: 1000)
- `seq_length`: Text sequence length (default: 64)
- `n_samples`: Number of samples (default: 500)
- `correlation`: How much text correlates with image class 0-1 (default: 0.7)
- `random_state`: Random seed (default: 42)

**Example:**
```python
from arch_eval.data import create_synthetic_vision_language_dataset

# Generate synthetic image-caption pairs
dataset = create_synthetic_vision_language_dataset({
    "img_size": 32,
    "channels": 3,
    "vocab_size": 1000,
    "seq_length": 64,
    "n_samples": 2000,
    "correlation": 0.8  # High correlation between image class and text
})

config = TrainingConfig(
    dataset="synthetic vision_language",
    dataset_params={
        "img_size": 32,
        "vocab_size": 1000,
        "seq_length": 64,
        "correlation": 0.8
    },
    task="next-token-prediction",  # Or custom multi-modal task
    training_args={"batch_size": 32, "num_epochs": 15}
)
```

---

## Distributed Utilities

### `init_distributed`

```python
init_distributed(backend="nccl", world_size=1, rank=0, master_addr="127.0.0.1", master_port="29500")
```

Initializes the distributed process group.

### `cleanup_distributed`

```python
cleanup_distributed()
```

Destroys the process group.

### `get_wrapped_model`

```python
get_wrapped_model(model, config)
```

Returns the model wrapped with the appropriate distributed wrapper (`DataParallel`, `DistributedDataParallel`, or `FSDP`).

---

## Logging

### `setup_logging`

```python
setup_logging(level="INFO", log_file=None, fmt=None)
```

Configures root logger with console output and optional file rotation. Returns the root logger.

### `LoggerAdapter`

```{eval-rst}
.. autoclass:: arch_eval.logging.LoggerAdapter
   :members:
   :undoc-members:
   :show-inheritance:
```

Provides consistent logging with a `"arch_eval."` prefix.

---

## Metrics

### `MetricCalculator`

```{eval-rst}
.. autoclass:: arch_eval.metrics.MetricCalculator
   :members:
   :undoc-members:
   :show-inheritance:
```

Computes task‑specific metrics (accuracy, precision, recall, F1, AUC, R², MSE, perplexity, etc.) and accumulates data for confusion matrices.

**Supported Model Output Formats:**

The `MetricCalculator` automatically handles various output formats from different model architectures:

- **Tensor**: Standard PyTorch tensor output `(batch_size, num_classes)` or `(batch_size, seq_len, vocab_size)`
- **Tuple**: Models returning `(logits, loss)`, `(loss, logits)`, or similar tuple patterns
- **Dict**: Transformer models returning `{"logits": ..., "loss": ...}` or similar dictionaries
- **Hugging Face Style**: Objects with `.logits` and `.loss` attributes (e.g., `CausalLMOutput`, `SequenceClassifierOutput`)

Example with Hugging Face style output:
```python
from transformers.modeling_outputs import CausalLMOutput

class MyTransformer(nn.Module):
    def forward(self, input_ids, labels=None):
        logits = self.model(input_ids)
        loss = compute_loss(logits, labels) if labels is not None else None
        return CausalLMOutput(loss=loss, logits=logits)

# Trainer and MetricCalculator will automatically extract loss and logits
trainer = Trainer(model, config)
history = trainer.train()
```

---

## Plugins

### `hook` decorator

```python
from arch_eval.plugins import hook

@hook("before_training")
def my_hook(trainer, config):
    ...
```

Marks a function as a plugin hook. The function will be called at the appropriate time if the plugin is loaded.

---

## Profiling

### `profiler_context`

```python
from arch_eval.profiler import profiler_context

with profiler_context(config):
    # training code
```

Context manager that conditionally enables PyTorch’s `torch.profiler` based on the `profiler` field in the config.

---

## Utilities

### Device utilities

```python
get_optimal_device() -> str
get_device_info() -> Dict[str, Any]
memory_summary() -> str
```

### `auto_device` decorator

```python
from arch_eval.utils import auto_device

@auto_device
def my_function(tensor):
    ...
```

Automatically moves input tensors to the device of the first argument (or inferred from a tensor) and optionally returns CPU tensors.

---

## Visualization

### `RealtimeWindow`

```{eval-rst}
.. autoclass:: arch_eval.viz.RealtimeWindow
   :members:
   :undoc-members:
   :show-inheritance:
```

Displays live metric plots and system resource usage.

### `VideoRecorder`

```{eval-rst}
.. autoclass:: arch_eval.viz.VideoRecorder
   :members:
   :undoc-members:
   :show-inheritance:
```

Records frames of metric plots and assembles them into a video using `ffmpeg`.

### `PlotSaver`

```{eval-rst}
.. autoclass:: arch_eval.viz.PlotSaver
   :members:
   :undoc-members:
   :show-inheritance:
```

Saves final metric plots (PNG) to disk.

---

## Exceptions

All custom exceptions inherit from `ArchEvalError`:

- `DatasetFormatError`
- `ConfigurationError`
- `ModelError`
- `PluginError`
- `VisualizationError`
- `StopTraining` – can be raised by plugins to gracefully stop training.
- `DistributedError`
```
