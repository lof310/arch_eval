# Arch Eval Library Documentation

```{toctree}
:maxdepth: 2
:caption: Getting Started

quickstart
```

```{toctree}
:maxdepth: 2
:caption: User Guide

guide
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api
```

```{toctree}
:maxdepth: 2
:caption: Examples

examples
```

```{toctree}
:maxdepth: 1
:caption: Project Info

contributing
```

## Overview

**arch_eval** is a high-level library for efficient and fast architecture evaluation and comparison of machine learning models. It provides a unified interface for training, benchmarking, and hyperparameter optimization with features like distributed training, mixed precision, and real-time visualization.

### Key Features

- **Unified Training Interface**: Train single models with easy-to-use configuration options
- **Multi-Model Benchmarking**: Compare multiple architectures sequentially or in parallel (thread/process-based)
- **Distributed Training**: Built-in support for DataParallel, DistributedDataParallel (DDP), and FSDP
- **Advanced Mixed Precision**: AMP with float16, bfloat16, and experimental FP8 support
- **Gradient Checkpointing**: Reduce memory footprint for large models
- **Rich Visualization**: Real-time training windows, video recording of metrics, and publication-ready plots
- **Logging**: Integration with Weights & Biases and TensorBoard
- **Hyperparameter Optimization**: Grid search and random search out of the box
- **Extensible Plugin System**: Custom hooks and callbacks for maximum flexibility
- **Robust Data Handling**: Supports PyTorch Datasets, synthetic data, torchvision datasets, Hugging Face datasets, and streaming
- **Transformer Support**: Seamless compatibility with Hugging Face Transformers and custom transformer architectures
- **Production-Ready**: Configurable timeouts, retry logic, checkpointing, and deterministic execution

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 1.12
- transformers ≥ 4.30.0

### Optional Dependencies

- `wandb` - Weights & Biases logging
- `tensorboard` - TensorBoard logging
- `transformer_engine` - FP8 mixed precision support (NVIDIA GPUs)
- `ffmpeg` - Video recording of metrics
- `cloudpickle` - Enhanced serialization for parallel execution

## Installation

Install from the GitHub repository:

```bash
# Clone the repository
git clone --depth=1 https://github.com/lof310/arch_eval.git
cd arch_eval

# Install in development mode (recommended)
pip install -e .

# Install normally
pip install .
```

Or install directly with pip (when published):

```bash
pip install arch_eval
```

## Quick Example

```python
import torch.nn as nn
from arch_eval import Trainer, TrainingConfig

# Define a simple model
class MLP(nn.Module):
    def __init__(self, input_size=128, hidden=256, num_classes=10):
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
    dataset_params={"n_samples": 5000, "n_features": 128, "n_classes": 10},
    training_args={"num_epochs": 10, "batch_size": 32},
    task="classification",
    realtime="auto",
    save_plot=["loss", "accuracy"]
)

model = MLP()
trainer = Trainer(model, config)
history = trainer.train()
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
