"""Factory for creating datasets from dict specifications."""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset

from arch_eval.data.factories.base import DatasetFactory

logger = logging.getLogger(__name__)


class DictDataset(Dataset):
    """Dataset created from a dict specification."""

    def __init__(self, data_dict, transform=None):
        self.data_dict = data_dict
        self.transform = transform
        self.keys = list(data_dict.keys())
        self.length = len(data_dict[self.keys[0]])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = {k: self.data_dict[k][idx] for k in self.keys}
        if self.transform:
            item = self.transform(item)
        return item


class DictFactory(DatasetFactory):
    """Factory for creating datasets from dict specifications."""

    def can_handle(self, data: Any, config: Any) -> bool:
        return isinstance(data, dict)

    def create(self, data: Any, config: Any) -> Tuple[Dataset, Optional[Dict[str, Any]]]:
        params = getattr(config, "dataset_params", {}) or {}
        transform = params.get("transform")
        # Convert lists to tensors if needed
        processed_data = {}
        for k, v in data.items():
            if isinstance(v, list):
                if isinstance(v[0], (int, float)):
                    processed_data[k] = torch.tensor(v)
                else:
                    processed_data[k] = v
            else:
                processed_data[k] = v
        dataset = DictDataset(processed_data, transform)
        metadata = {
            "num_classes": 1,
            "input_shape": None,
        }
        logger.debug(f"Created DictDataset with {len(dataset)} samples")
        return dataset, metadata
