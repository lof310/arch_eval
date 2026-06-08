"""Prefetch data loader for overlapping CPU/GPU work."""

from typing import Any, Dict, List, Tuple, Union

import torch


class PrefetchLoader:
    """Wraps a DataLoader to prefetch the next batch on GPU while computing on current batch.
    This helps overlap data transfer with computation, especially useful for CPU training
    or when data loading is a bottleneck.
    """

    def __init__(self, loader, device: torch.device):
        self.loader = loader
        self.device = device

    def __iter__(self):
        stream = torch.cuda.Stream() if self.device.type == "cuda" else None
        loader_iter = iter(self.loader)
        try:
            next_batch = self._to_device(next(loader_iter), stream)
        except StopIteration:
            return
        current_batch = next_batch
        for batch in loader_iter:
            next_batch = self._to_device(batch, stream)
            yield current_batch
            current_batch = next_batch
        yield current_batch

    def _to_device(self, batch, stream=None):
        """Move batch to device, handling various data structures."""
        if batch is None:
            return None
        if isinstance(batch, torch.Tensor):
            if stream:
                with torch.cuda.stream(stream):
                    return batch.to(self.device, non_blocking=True)
            return batch.to(self.device, non_blocking=True)
        if isinstance(batch, (tuple, list)):
            return type(batch)(self._to_device(b, stream) for b in batch)
        if isinstance(batch, dict):
            return {k: self._to_device(v, stream) for k, v in batch.items()}
        return batch

    def __len__(self):
        return len(self.loader)
