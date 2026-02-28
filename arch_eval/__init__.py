"""arch_eval: High-level library for easy architecture evaluation of ML models."""

__version__ = "0.5.0"

from arch_eval.core.trainer import Trainer
from arch_eval.core.benchmark import Benchmark
from arch_eval.core.config import TrainingConfig, BenchmarkConfig
from arch_eval.plugins.manager import PluginManager
from arch_eval.logging.logger_config import setup_logging
from arch_eval.core.callbacks import Callback
from arch_eval.distributed import init_distributed, cleanup_distributed
from arch_eval.hpo import HyperparameterOptimizer
#from arch_eval.interpret import permutation_importance, attention_weights

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
    "init_distributed",
    "cleanup_distributed",
    "HyperparameterOptimizer",
    "permutation_importance",
    "attention_weights",
]
