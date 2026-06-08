"""Factory for creating datasets from iterable sources."""

import itertools
import logging
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset, IterableDataset

from arch_eval.data.factories.base import DatasetFactory

logger = logging.getLogger(__name__)


class IterableDatasetWrapper(IterableDataset):
    """Wrapper for iterable datasets."""

    def __init__(self, iterable, max_items=None):
        self.iterable = iterable
        self.max_items = max_items

    def __iter__(self):
        if self.max_items:
            return iter(itertools.islice(self.iterable, self.max_items))
        return iter(self.iterable)


class IterableFactory(DatasetFactory):
    """Factory for creating datasets from Python iterables/generators."""

    def can_handle(self, data: Any, config: Any) -> bool:
        # Check if data is an iterable (but not a string/tensor/list/tuple which are handled elsewhere)
        if hasattr(data, "__iter__") and not isinstance(data, (str, list, tuple)):
            try:
                import numpy as np

                if not isinstance(data, np.ndarray):
                    return True
            except ImportError:
                pass
            return True
        return False

    def create(self, data: Any, config: Any) -> Tuple[Dataset, Optional[Dict[str, Any]]]:
        params = getattr(config, "dataset_params", {}) or {}
        max_items = params.get("max_items", None)
        dataset = IterableDatasetWrapper(data, max_items)
        logger.info(f"Created IterableDataset wrapper")
        return dataset, None
