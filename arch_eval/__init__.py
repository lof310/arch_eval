"""arch_eval: High-level library for easy architecture evaluation of ML models."""

__version__ = "0.1.0"

from arch_eval.core.benchmark import Benchmark
from arch_eval.core.callbacks import (Callback, EarlyStopping,
                                      LRSchedulerLogger, ModelCheckpoint,
                                      TensorBoardLogger)
from arch_eval.core.config import BenchmarkConfig, TrainingConfig
from arch_eval.core.trainer import Trainer
from arch_eval.distributed import cleanup_distributed, init_distributed
from arch_eval.hpo import HyperparameterOptimizer
from arch_eval.logging.logger_config import setup_logging
from arch_eval.plugins.manager import PluginManager

_plugin_manager = PluginManager()
_plugin_manager.discover_plugins()

__all__ = [
    "Trainer",
    "Benchmark",
    "TrainingConfig",
    "BenchmarkConfig",
    "PluginManager",
    "setup_logging",
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "LRSchedulerLogger",
    "TensorBoardLogger",
    "init_distributed",
    "cleanup_distributed",
    "HyperparameterOptimizer",
]
