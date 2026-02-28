"""Device utilities for optimal hardware selection and memory management."""

import torch
import psutil
from typing import Dict, Any

def get_optimal_device() -> str:
    if torch.cuda.is_available():
        try:
            if torch.cuda.get_device_properties(0).total_memory > 2**30:
                return "cuda"
        except:
            pass
        return "cuda"
    return "cpu"

def get_device_info() -> Dict[str, Any]:
    info = {
        "cpu_count": psutil.cpu_count(),
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_total": psutil.virtual_memory().total,
        "memory_available": psutil.virtual_memory().available,
    }
    if torch.cuda.is_available():
        info.update({
            "cuda_available": True,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name(0),
            "cuda_memory_allocated": torch.cuda.memory_allocated(0),
            "cuda_memory_reserved": torch.cuda.memory_reserved(0),
            "cuda_max_memory_allocated": torch.cuda.max_memory_allocated(0),
        })
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
            lines.append(f"GPU {i} ({torch.cuda.get_device_name(i)}): alloc={alloc:.2f}GB, reserved={reserved:.2f}GB, total={total:.2f}GB")
    return "\n".join(lines)
