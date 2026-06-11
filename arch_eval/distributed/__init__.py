"""Distributed training utilities."""

import os
from typing import Optional

import torch
import torch.distributed as dist

from arch_eval.core.config import DistributedBackend
from arch_eval.core.exceptions import ConfigurationError, DistributedError


def init_distributed(
    config,
):
    """Initialize the distributed process group."""
    from arch_eval.core.config import DistributedBackend

    backend = "nccl"  # default
    if config.distributed_backend == DistributedBackend.DISTRIBUTED:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    elif config.distributed_backend == DistributedBackend.FSDP:
        backend = "nccl"
    elif config.distributed_backend == DistributedBackend.DATAPARALLEL:
        backend = "gloo"

    world_size = config.distributed_world_size
    rank = config.distributed_rank
    master_addr = config.distributed_master_addr
    master_port = config.distributed_master_port

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    # Set LOCAL_RANK to local rank (rank % gpus_per_node), not global rank
    gpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 1
    local_rank = rank % gpus_per_node
    os.environ["LOCAL_RANK"] = str(local_rank)
    if not dist.is_available():
        raise DistributedError("torch.distributed is not available.")
    if not dist.is_initialized():
        # Allow CPU-based distributed training with Gloo backend
        if backend == "gloo" or (not torch.cuda.is_available() and backend != "nccl"):
            pass  # Gloo works on CPU
        elif backend == "nccl" and not torch.cuda.is_available():
            raise DistributedError("NCCL backend requires CUDA")
        dist.init_process_group(backend=backend, world_size=world_size, rank=rank)


def cleanup_distributed():
    """Destroy the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_wrapped_model(model, config):
    """Wrap the model according to the chosen distributed backend."""
    if config.distributed_backend == DistributedBackend.DATAPARALLEL:
        return torch.nn.DataParallel(model)
    elif config.distributed_backend == DistributedBackend.DISTRIBUTED:
        from torch.nn.parallel import DistributedDataParallel

        # Use local_rank for device ID in multi-GPU per process scenarios
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return DistributedDataParallel(model, device_ids=[local_rank])
    elif config.distributed_backend == DistributedBackend.FSDP:
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import ShardingStrategy

            return FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)
        except ImportError:
            raise DistributedError("FSDP requires PyTorch >= 1.12")
    else:
        return model
