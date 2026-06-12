# Arch Eval Library Documentation

Welcome to the **arch_eval** documentation! This comprehensive guide will help you understand and use the library effectively for evaluating and comparing machine learning model architectures.

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

Whether you're a researcher comparing different neural network architectures, a practitioner looking to optimize model performance, or a student learning about machine learning model evaluation, **arch_eval** streamlines the entire workflow from data loading to results analysis.

### Key Features

- **Unified Training Interface**: Train single models with easy-to-use configuration options. No need to write boilerplate training loops - just define your model and configuration.
- **Multi-Model Benchmarking**: Compare multiple architectures sequentially or in parallel (thread/process-based). Get comparison tables automatically generated.
- **Distributed Training**: Built-in support for DataParallel, DistributedDataParallel (DDP), and FSDP (Fully Sharded Data Parallel) for scaling to multiple GPUs.
- **Advanced Mixed Precision**: AMP with float16, bfloat16, and experimental FP8 support (requires NVIDIA Transformer Engine) for faster training on supported hardware.
- **Gradient Checkpointing**: Reduce memory footprint for large models by trading compute for memory, enabling training of larger architectures.
- **Rich Visualization**: Real-time training windows, video recording of metrics, and publication-ready plots. Monitor your training progress visually.
- **Logging**: Integration with Weights & Biases and TensorBoard for experiment tracking and reproducibility.
- **Hyperparameter Optimization**: Grid search and random search out of the box. Find optimal hyperparameters efficiently.
- **Extensible Plugin System**: Custom hooks and callbacks for maximum flexibility. Extend the library without modifying core code.
- **Robust Data Handling**: Supports PyTorch Datasets, synthetic data, torchvision datasets, Hugging Face datasets, and streaming for large datasets.
- **Transformer Support**: Seamless compatibility with Hugging Face Transformers and custom transformer architectures. Handles various output formats automatically.
- **Production-Ready**: Configurable timeouts, retry logic, checkpointing, and deterministic execution for reliable deployments.

### When to Use arch_eval

**arch_eval** is ideal for:

- **Architecture Search**: Quickly compare different model designs (e.g., varying layer sizes, depths, or activation functions)
- **Hyperparameter Tuning**: Systematically explore learning rates, batch sizes, and other training parameters
- **Educational Purposes**: Focus on model design without worrying about training loop implementation
- **Research Prototyping**: Rapidly test new ideas and get comparable results across experiments
- **Benchmarking**: Create standardized evaluation pipelines for fair model comparisons

### Requirements

- **Python** ≥ 3.9
- **PyTorch** ≥ 1.12
- **transformers** ≥ 4.30.0
- **Additional dependencies**: pandas, numpy, scikit-learn, psutil, matplotlib, seaborn

### Optional Dependencies

- `wandb` - Weights & Biases logging for experiment tracking
- `tensorboard` - TensorBoard logging for local visualization
- `transformer_engine` - FP8 mixed precision support (NVIDIA GPUs only)
- `ffmpeg` - Video recording of metrics evolution during training
- `cloudpickle` - Enhanced serialization for parallel execution and complex objects
- `rich` - Enhanced progress bars and terminal output

## Installation

### From GitHub Repository (Recommended)

Install directly from the source repository for the latest features:

```bash
# Clone the repository
git clone --depth=1 https://github.com/lof310/arch_eval.git
cd arch_eval

# Install in development mode (recommended for contributors)
pip install -e .

# Or install normally
pip install .
```

### From PyPI (When Published)

```bash
pip install arch_eval
```

### With Optional Dependencies

```bash
# Install with all optional dependencies
pip install arch_eval[wandb,tensorboard,video]

# Or install specific optional dependencies
pip install wandb tensorboard
pip install cloudpickle  # For better parallel serialization
```

### Verifying Installation

After installation, verify that arch_eval is properly installed:

```python
import arch_eval
print(arch_eval.__version__)

# Test basic functionality
from arch_eval import Trainer, TrainingConfig
print("Installation successful!")
```

### Troubleshooting Installation

**Issue: CUDA not available**
- Ensure you have the correct PyTorch CUDA version installed for your GPU
- Visit [pytorch.org](https://pytorch.org) for the correct installation command

**Issue: Missing dependencies**
- Install required dependencies: `pip install -r requirements.txt`
- For development: `pip install -r requirements-dev.txt`

**Issue: FP8 support not working**
- FP8 requires NVIDIA Transformer Engine and compatible GPU (H100, A100, etc.)
- Install with: `pip install transformer_engine`

## Quick Example

Here's a complete example showing how to train a simple model:

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
    realtime="auto",  # Shows live training plots
    save_plot=["loss", "accuracy"]  # Saves plots after training
)

model = MLP()
trainer = Trainer(model, config)
history = trainer.train()

# Access training history
print(f"Final accuracy: {history['val_accuracy'][-1]:.4f}")
```

## Next Steps

- **Quick Start**: Jump into using arch_eval with our [Quick Start Guide](quickstart.md)
- **User Guide**: Learn about all features in detail in the [User Guide](guide.md)
- **Examples**: Explore complete working examples in the [Examples](examples.md) section
- **API Reference**: Look up specific classes and functions in the [API Reference](api.md)

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
