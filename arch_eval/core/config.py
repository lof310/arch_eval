"""Configuration dataclasses for Trainer and Benchmark."""

import os
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import torch

from arch_eval.core.exceptions import ConfigurationError


def _try_import_cloudpickle():
    """Try to import cloudpickle, return None if not available."""
    try:
        import cloudpickle

        return cloudpickle
    except ImportError:
        return None


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
    # Try cloudpickle first for robust serialization of any callable
    cloudpickle = _try_import_cloudpickle()
    if cloudpickle is not None:
        try:
            return ("__cloudpickle__", cloudpickle.dumps(obj))
        except Exception:
            pass  # Fall through to original logic
    # Original fallback logic
    if hasattr(obj, "__name__") and hasattr(obj, "__module__") and obj.__module__ != "__main__":
        return ("__function__", obj.__module__, obj.__name__)
    warnings.warn(f"Callable {obj} may not be picklable.")
    return str(obj)


def _deserialize_callable(rep: Any) -> Any:
    """Restore a callable from its serialized representation."""
    if rep is None or not isinstance(rep, tuple):
        return rep
    # Handle cloudpickle serialization
    if len(rep) == 2 and rep[0] == "__cloudpickle__":
        cloudpickle = _try_import_cloudpickle()
        if cloudpickle is not None:
            try:
                return cloudpickle.loads(rep[1])
            except Exception as e:
                raise ValueError(f"Could not restore cloudpickled callable: {e}")
        else:
            raise ValueError("cloudpickle required to deserialize this callable but not installed")
    # Handle original function reference logic
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
    apply_transforms: bool = True  # Whether to apply transforms to datasets (False for text/VL datasets)

    # Computation
    dtype: torch.dtype = torch.float32
    device: Optional[str] = None  # None = auto

    # Logging intervals
    viz_interval: int = 10
    log_interval: int = 10
    eval_interval: int = 100

    # Visualization
    realtime: Union[str, bool] = "auto"  # "auto", "gui", "terminal", "none" (bool kept for backward compatibility)
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
        # Seed setup removed - moved to Trainer.__init__
        if self.deterministic:
            torch.use_deterministic_algorithms(True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if self.viz_metrics is None:
            self.viz_metrics = self.save_video.copy()
        # Backward compatibility for realtime: True -> "auto", False -> "none"
        if isinstance(self.realtime, bool):
            self.realtime = "auto" if self.realtime else "none"
        if self.realtime not in ("auto", "gui", "terminal", "none"):
            raise ConfigurationError(
                f"Invalid realtime value: {self.realtime}. Must be 'auto', 'gui', 'terminal', or 'none'."
            )

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
    # Loss extraction function (optional, for custom model output formats)
    loss_fn_extractor: Optional[Callable] = None
    # Loss computation mode: "auto" (heuristic), "model" (use model's loss), "criterion" (always apply criterion)
    loss_mode: str = "auto"

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

    # Debug mode for quick pipeline verification
    debug: bool = False

    # torch.compile support
    compile_model: bool = False
    compile_kwargs: Dict[str, Any] = field(default_factory=lambda: {"dynamic": True})

    # Progress bar backend
    progress_bar: str = "auto"  # "auto", "rich", "tqdm", "plain"

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
                # Try cloudpickle first for robust callback serialization
                cloudpickle = _try_import_cloudpickle()
                if cloudpickle is not None:
                    try:
                        serialized_callbacks.append(("__cloudpickle__", cloudpickle.dumps(cb)))
                        continue
                    except Exception:
                        pass  # Fall through to original logic
                # Original fallback logic
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
            for item in state["callbacks"]:
                # Handle cloudpickle serialization
                if isinstance(item, tuple) and len(item) == 2 and item[0] == "__cloudpickle__":
                    cloudpickle = _try_import_cloudpickle()
                    if cloudpickle is not None:
                        try:
                            cb = cloudpickle.loads(item[1])
                            restored_callbacks.append(cb)
                            continue
                        except Exception as e:
                            raise ValueError(f"Could not restore cloudpickled callback: {e}")
                    else:
                        raise ValueError("cloudpickle required to deserialize this callback but not installed")
                # Original fallback logic
                cls, cb_state = item
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
    
    # Training-specific fields (mirrored from TrainingConfig for benchmarking)
    mixed_precision: bool = False
    mixed_precision_dtype: MixedPrecisionDtype = MixedPrecisionDtype.FLOAT16
    gradient_clip: Optional[float] = None
    checkpoint_dir: Optional[str] = None
    optimizers: List[Dict[str, Any]] = field(default_factory=list)
    schedulers: List[Dict[str, Any]] = field(default_factory=list)
    callbacks: List[Any] = field(default_factory=list)
    loss_function: Optional[Callable] = None
    model_output_transform: Optional[Callable] = None
    gradient_accumulation_steps: int = 1
    early_stopping_patience: Optional[int] = None
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: str = "min"
    grad_scaler: bool = True
    save_best_only: bool = True
    save_frequency: int = 1
    checkpoint_metric: str = "val_loss"
    scheduler_interval: str = "epoch"
    eval_on_test: bool = False
    log_all_metrics: bool = False
    input_shape: Optional[tuple] = None
    loss_fn_extractor: Optional[Callable] = None
    loss_mode: str = "auto"
    distributed_backend: DistributedBackend = DistributedBackend.NONE
    distributed_world_size: int = 1
    distributed_rank: int = 0
    distributed_master_addr: str = "127.0.0.1"
    distributed_master_port: str = "29500"
    gradient_checkpointing: bool = False
    gradient_checkpointing_modules: Optional[List[str]] = None
    profiler: Optional[Dict[str, Any]] = None
    gc_collect_interval: int = 50
    log_confusion_matrix: bool = False
    confusion_matrix_labels: Optional[List[str]] = None
    debug: bool = False
    compile_model: bool = False
    compile_kwargs: Dict[str, Any] = field(default_factory=lambda: {"dynamic": True})
    progress_bar: str = "auto"
    
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
