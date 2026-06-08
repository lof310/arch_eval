"""Distributed training utilities."""

import os
from typing import Optional

import torch
import torch.distributed as dist

from arch_eval.core.config import DistributedBackend
from arch_eval.core.exceptions import ConfigurationError, DistributedError


def init_distributed(
    backend: str = "nccl",
    world_size: int = 1,
    rank: int = 0,
    master_addr: str = "127.0.0.1",
    master_port: str = "29500",
):
    """Initialize the distributed process group."""
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    if not dist.is_available():
        raise DistributedError("torch.distributed is not available.")
    if not dist.is_initialized():
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

        return DistributedDataParallel(model, device_ids=[config.distributed_rank])
    elif config.distributed_backend == DistributedBackend.FSDP:
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import ShardingStrategy

            return FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)
        except ImportError:
            raise DistributedError("FSDP requires PyTorch >= 1.12")
    else:
        return model
