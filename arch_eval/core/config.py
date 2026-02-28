"""Configuration dataclasses for Trainer and Benchmark."""

from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Any, Callable
import torch
from enum import Enum
import os


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
    FP8 = "fp8"          # experimental


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
    dataset_shard: Optional[Dict[str, Any]] = None   # ej. {"num_shards": 4, "shard_id": 0}

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

    # External logging
    log_to_wandb: bool = False
    log_to_tensorboard: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    tensorboard_dir: str = "./logs"

    # Reproducibility
    seed: Optional[int] = None
    deterministic: bool = False

    # DataLoader params
    dataloader_params: Dict[str, Any] = field(default_factory=lambda: {
        "num_workers": 0,
        "pin_memory": False,
        "prefetch_factor": 2,
        "persistent_workers": False,
    })

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device.startswith("cuda"):
            self.dataloader_params["pin_memory"] = True
        else:
            self.dataloader_params["pin_memory"] = False
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            import numpy as np, random
            np.random.seed(self.seed)
            random.seed(self.seed)
        if self.deterministic:
            torch.use_deterministic_algorithms(True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

@dataclass
class TrainingConfig(BaseConfig):
    """Configuration for Trainer."""

    # Training
    optimizers: List[Dict[str, Any]] = field(default_factory=list)
    schedulers: List[Dict[str, Any]] = field(default_factory=list)
    training_args: Dict[str, Any] = field(default_factory=lambda: {
        "batch_size": 16,
        "learning_rate": 1e-3,
        "num_epochs": 10,
        "weight_decay": 1e-2,
        "momentum": 0.9,
    })

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
    profiler: Optional[Dict[str, Any]] = None   # ej. {"enabled": True, "activities": ["cpu", "cuda"], "schedule": {...}}

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
                import transformer_engine.pytorch as te
            except ImportError:
                raise ConfigurationError("FP8 requires NVIDIA Transformer Engine installed.")


@dataclass
class BenchmarkConfig(BaseConfig):
    """Configuration for Benchmark."""

    training_args: Dict[str, Any] = field(default_factory=lambda: {
        "batch_size": 32,
        "learning_rate": 0.001,
        "num_epochs": 10,
    })
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
