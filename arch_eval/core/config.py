"""Configuration dataclasses for Trainer and Benchmark."""

import os
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import torch

from arch_eval.core.exceptions import ConfigurationError


class TaskType(str, Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    NEXT_TOKEN_PREDICTION = "next-token-prediction"


class DistributedBackend(str, Enum):
    NONE = "none"
    DATAPARALLEL = "dp"
    DISTRIBUTED = "ddp"
    FSDP = "fsdp"


class MixedPrecisionDtype(str, Enum):
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FP8 = "fp8"  # experimental


def _serialize_callable(obj: Any) -> Any:
    """Convert a callable to a serializable representation."""
    if obj is None:
        return None
    if not callable(obj):
        return obj
    if hasattr(obj, "__name__") and hasattr(obj, "__module__") and obj.__module__ != "__main__":
        return ("__function__", obj.__module__, obj.__name__)
    warnings.warn(f"Callable {obj} may not be picklable.")
    return str(obj)


def _deserialize_callable(rep: Any) -> Any:
    """Restore a callable from its serialized representation."""
    if rep is None or not isinstance(rep, tuple):
        return rep
    if len(rep) == 3 and rep[0] == "__function__":
        module_name, func_name = rep[1], rep[2]
        try:
            module = __import__(module_name, fromlist=[func_name])
            return getattr(module, func_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Could not restore function {module_name}.{func_name}: {e}")
    return rep


def _serialize_dtype(dtype: torch.dtype) -> str:
    """Convert torch.dtype to string."""
    return str(dtype).split(".")[-1]


def _deserialize_dtype(dtype_str: str) -> torch.dtype:
    """Convert string back to torch.dtype."""
    return getattr(torch, dtype_str)


@dataclass
class BaseConfig:
    """Base configuration with common fields."""

    # Data
    dataset: Any = None
    dataset_params: Dict[str, Any] = field(default_factory=dict)
    transform: Optional[Callable] = None
    target_transform: Optional[Callable] = None
    collate_fn: Optional[Callable] = None
    dataset_streaming: bool = False
    dataset_shard: Optional[Dict[str, Any]] = None  # ej. {"num_shards": 4, "shard_id": 0}

    # Computation
    dtype: torch.dtype = torch.float32
    device: Optional[str] = None  # None = auto

    # Logging intervals
    viz_interval: int = 10
    log_interval: int = 10
    eval_interval: int = 100

    # Visualization
    realtime: bool = True
    save_video: List[str] = field(default_factory=list)
    save_plot: List[str] = field(default_factory=list)
    viz_metrics: Optional[List[str]] = None

    # External logging
    log_to_wandb: bool = False
    log_to_tensorboard: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # Reproducibility
    seed: Optional[int] = None
    deterministic: bool = False

    # DataLoader params
    dataloader_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "num_workers": 0,
            "pin_memory": False,
            "prefetch_factor": 2,
            "persistent_workers": False,
        }
    )

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device.startswith("cuda"):
            self.dataloader_params["pin_memory"] = True
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            import random

            import numpy as np

            np.random.seed(self.seed)
            random.seed(self.seed)
        if self.deterministic:
            torch.use_deterministic_algorithms(True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if self.viz_metrics is None:
            self.viz_metrics = self.save_video.copy()

    def __getstate__(self):
        state = self.__dict__.copy()
        for f in ["transform", "target_transform", "collate_fn"]:
            if f in state:
                state[f] = _serialize_callable(state[f])
        if "dtype" in state:
            state["dtype"] = _serialize_dtype(state["dtype"])
        return state

    def __setstate__(self, state):
        for f in ["transform", "target_transform", "collate_fn"]:
            if f in state:
                state[f] = _deserialize_callable(state[f])
        if "dtype" in state and isinstance(state["dtype"], str):
            state["dtype"] = _deserialize_dtype(state["dtype"])
        self.__dict__.update(state)


@dataclass
class TrainingConfig(BaseConfig):
    """Configuration for Trainer."""

    # Training
    optimizers: List[Dict[str, Any]] = field(default_factory=list)
    schedulers: List[Dict[str, Any]] = field(default_factory=list)
    training_args: Dict[str, Any] = field(
        default_factory=lambda: {
            "batch_size": 16,
            "learning_rate": 1e-3,
            "num_epochs": 10,
            "weight_decay": 1e-2,
            "momentum": 0.9,
        }
    )

    # Task
    task: Union[str, Any] = TaskType.CLASSIFICATION

    # Regularization
    gradient_clip: Optional[float] = None
    gradient_accumulation_steps: int = 1

    # Early stopping
    early_stopping_patience: Optional[int] = None
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: str = "min"

    # Mixed precision
    mixed_precision: bool = False
    mixed_precision_dtype: MixedPrecisionDtype = MixedPrecisionDtype.FLOAT16
    grad_scaler: bool = True

    # Checkpointing
    checkpoint_dir: Optional[str] = None
    save_best_only: bool = True
    save_frequency: int = 1
    checkpoint_metric: str = "val_loss"

    # Model output transformation
    model_output_transform: Optional[Callable] = None

    # Scheduler stepping
    scheduler_interval: str = "epoch"  # or "step"

    # Callbacks
    callbacks: List[Any] = field(default_factory=list)

    # Evaluation
    eval_on_test: bool = False

    # Logging
    log_all_metrics: bool = False

    # Model validation
    input_shape: Optional[tuple] = None

    # Custom loss function
    loss_function: Optional[Callable] = None

    # Distributed training
    distributed_backend: DistributedBackend = DistributedBackend.NONE
    distributed_world_size: int = 1
    distributed_rank: int = 0
    distributed_master_addr: str = "127.0.0.1"
    distributed_master_port: str = "29500"

    # Gradient checkpointing
    gradient_checkpointing: bool = False
    gradient_checkpointing_modules: Optional[List[str]] = None

    # Timeout and recovery
    timeout_seconds: int = 3600
    retry_failed: bool = False

    # Profiling
    profiler: Optional[Dict[str, Any]] = None  # ej. {"enabled": True, "activities": ["cpu", "cuda"], "schedule": {...}}

    # Memory
    gc_collect_interval: int = 50

    # Confusion matrix
    log_confusion_matrix: bool = False
    confusion_matrix_labels: Optional[List[str]] = None

    def __post_init__(self):
        super().__post_init__()
        if self.log_to_wandb and self.wandb_project is None:
            raise ConfigurationError("wandb_project must be specified when log_to_wandb=True")
        if isinstance(self.task, str) and self.task not in [t.value for t in TaskType]:
            raise ConfigurationError(f"Unsupported task: {self.task}")
        if self.gradient_clip is not None and self.gradient_clip <= 0:
            raise ConfigurationError("gradient_clip must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ConfigurationError("gradient_accumulation_steps must be >= 1")
        if self.early_stopping_patience is not None and self.early_stopping_mode not in ("min", "max"):
            raise ConfigurationError("early_stopping_mode must be 'min' or 'max'")
        if not all(isinstance(m, str) for m in self.save_video):
            raise ConfigurationError("save_video must be a list of strings")
        if not all(isinstance(m, str) for m in self.save_plot):
            raise ConfigurationError("save_plot must be a list of strings")
        for opt in self.optimizers:
            if "type" not in opt:
                opt["type"] = "adam"
        for sch in self.schedulers:
            if "type" not in sch:
                raise ConfigurationError("Scheduler must have a 'type' field")
        if self.scheduler_interval not in ("epoch", "step"):
            raise ConfigurationError("scheduler_interval must be 'epoch' or 'step'")
        if self.checkpoint_dir and self.save_best_only and not self.checkpoint_metric:
            raise ConfigurationError("checkpoint_metric must be specified when save_best_only=True")
        if self.distributed_backend != DistributedBackend.NONE:
            if self.device == "cpu":
                raise ConfigurationError("Distributed training requires GPU (CUDA).")
            if self.distributed_world_size < 1:
                raise ConfigurationError("distributed_world_size must be >= 1")
        if self.mixed_precision_dtype == MixedPrecisionDtype.FP8:
            try:
                import transformer_engine.pytorch as te  # noqa
            except ImportError:
                raise ConfigurationError("FP8 requires NVIDIA Transformer Engine installed.")

    def __getstate__(self):
        state = super().__getstate__()
        for f in ["model_output_transform", "loss_function"]:
            if f in state:
                state[f] = _serialize_callable(state[f])
        if "callbacks" in state:
            serialized_callbacks = []
            for cb in state["callbacks"]:
                if hasattr(cb, "__getstate__"):
                    serialized_callbacks.append((cb.__class__, cb.__getstate__()))
                else:
                    warnings.warn(f"Callback {cb} may not be picklable.")
                    serialized_callbacks.append((cb.__class__, None))
            state["callbacks"] = serialized_callbacks
        return state

    def __setstate__(self, state):
        for f in ["model_output_transform", "loss_function"]:
            if f in state:
                state[f] = _deserialize_callable(state[f])
        if "callbacks" in state:
            restored_callbacks = []
            for cls, cb_state in state["callbacks"]:
                if cb_state is not None:
                    cb = cls.__new__(cls)
                    cb.__setstate__(cb_state)
                else:
                    cb = cls()
                restored_callbacks.append(cb)
            state["callbacks"] = restored_callbacks
        super().__setstate__(state)


@dataclass
class BenchmarkConfig(BaseConfig):
    """Configuration for Benchmark."""

    training_args: Dict[str, Any] = field(
        default_factory=lambda: {
            "batch_size": 32,
            "learning_rate": 0.001,
            "num_epochs": 10,
        }
    )
    task: Union[str, Any] = TaskType.CLASSIFICATION
    parallel: bool = False
    compare_metrics: List[str] = field(default_factory=lambda: ["accuracy", "loss"])
    max_workers: Optional[int] = None
    use_processes: bool = False
    timeout_seconds: int = 3600
    retry_failed: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.parallel and self.device == "cuda":
            import warnings

            warnings.warn("Parallel execution on GPU may cause memory issues. Use with caution.")
        if self.use_processes and self.device == "cuda":
            warnings.warn("Process-based parallelism with CUDA may not work properly. Consider using sequential.")
