"""Profiling utilities using torch.profiler."""

import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


@contextmanager
def profiler_context(config):
    """Context manager for PyTorch profiler."""
    if config.profiler and config.profiler.get("enabled", False):
        activities = []
        if "cpu" in config.profiler.get("activities", ["cpu"]):
            activities.append(torch.profiler.ProfilerActivity.CPU)
        if "cuda" in config.profiler.get("activities", []) and torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        # Configurar schedule con valores por defecto
        schedule_config = config.profiler.get("schedule", {})
        wait = schedule_config.get("wait", 1)
        warmup = schedule_config.get("warmup", 1)
        active = schedule_config.get("active", 3)
        repeat = schedule_config.get("repeat", 1)

        schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
        trace_path = config.profiler.get("trace_path", "./trace.json")

        with torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_path.replace(".json", "")),
        ) as prof:
            yield prof

        logger.info(f"Profiling trace saved to {trace_path}")
    else:
        yield None
