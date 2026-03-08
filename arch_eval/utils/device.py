"""Device utilities with auto_device decorator."""

import functools
from typing import Any, Callable, Dict, Optional, Union

import psutil
import torch

__all__ = ["get_optimal_device", "get_device_info", "memory_summary", "auto_device"]


def get_optimal_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_device_info() -> Dict[str, Any]:
    info = {
        "cpu_count": psutil.cpu_count(),
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_total": psutil.virtual_memory().total,
        "memory_available": psutil.virtual_memory().available,
    }
    if torch.cuda.is_available():
        info.update(
            {
                "cuda_available": True,
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cuda_memory_allocated": torch.cuda.memory_allocated(0),
                "cuda_memory_reserved": torch.cuda.memory_reserved(0),
                "cuda_max_memory_allocated": torch.cuda.max_memory_allocated(0),
            }
        )
    else:
        info["cuda_available"] = False
    return info


def memory_summary() -> str:
    lines = []
    mem = psutil.virtual_memory()
    lines.append(f"CPU Memory: {mem.used / 2**30:.2f}GB / {mem.total / 2**30:.2f}GB ({mem.percent}%)")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 2**30
            reserved = torch.cuda.memory_reserved(i) / 2**30
            total = torch.cuda.get_device_properties(i).total_memory / 2**30
            lines.append(
                f"GPU {i} ({torch.cuda.get_device_name(i)}): alloc={alloc:.2f}GB, reserved={reserved:.2f}GB, total={total:.2f}GB"
            )
    return "\n".join(lines)


def auto_device(func: Optional[Callable] = None, *, return_cpu: bool = False):
    """
    Decorator to automatically move input tensors to the device of the first argument (if it has a .device)
    or to the device specified by the instance's `device` attribute (if applied to a method).
    If return_cpu=True, output tensors are moved back to CPU.
    """

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Determine target device
            device = None
            if args and hasattr(args[0], "device"):
                # First argument is self with device attribute (e.g., Trainer)
                device = getattr(args[0], "device", None)
            if device is None:
                # Try to infer from first tensor argument
                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        device = arg.device
                        break
                if device is None:
                    for v in kwargs.values():
                        if isinstance(v, torch.Tensor):
                            device = v.device
                            break
            if device is None:
                device = torch.device("cpu")

            # Move input tensors to device
            new_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    new_args.append(arg.to(device))
                else:
                    new_args.append(arg)
            new_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    new_kwargs[k] = v.to(device)
                else:
                    new_kwargs[k] = v

            result = f(*new_args, **new_kwargs)

            if return_cpu and isinstance(result, torch.Tensor):
                return result.cpu()
            if return_cpu and isinstance(result, (tuple, list)):
                return type(result)((r.cpu() if isinstance(r, torch.Tensor) else r) for r in result)
            return result

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
