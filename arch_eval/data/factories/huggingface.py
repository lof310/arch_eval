"""Factory for creating datasets from HuggingFace."""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset

from arch_eval.data.factories.base import DatasetFactory

logger = logging.getLogger(__name__)

try:
    from datasets import Dataset as HFDataset
    from datasets import IterableDataset as HFIterableDataset

    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    HFDataset = object
    HFIterableDataset = object


class HuggingFaceDatasetWrapper(Dataset):
    """Wrapper for HuggingFace datasets."""

    def __init__(self, hf_dataset, text_column="text", label_column=None):
        self.hf_dataset = hf_dataset
        self.text_column = text_column
        self.label_column = label_column

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        text = item[self.text_column]
        label = item[self.label_column] if self.label_column else text
        return text, label


class HuggingFaceFactory(DatasetFactory):
    """Factory for creating datasets from HuggingFace."""

    def can_handle(self, data: Any, config: Any) -> bool:
        if not HUGGINGFACE_AVAILABLE:
            return False
        # Check if data is a string that looks like a HF dataset name
        if isinstance(data, str) and "/" in data:
            return True
        dataset_type = getattr(config, "dataset_type", "")
        if isinstance(dataset_type, str) and dataset_type.startswith("hf:"):
            return True
        return False

    def create(self, data: Any, config: Any) -> Tuple[Dataset, Optional[Dict[str, Any]]]:
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("HuggingFace datasets not available. Install with: pip install datasets")

        # Get dataset name
        if isinstance(data, str) and "/" in data:
            dataset_name = data
        else:
            dataset_type = getattr(config, "dataset_type", "")
            dataset_name = dataset_type.replace("hf:", "") if dataset_type.startswith("hf:") else "glue/mrpc"

        params = getattr(config, "dataset_params", {}) or {}
        split = params.get("split", "train")
        text_column = params.get("text_column", "text")
        label_column = params.get("label_column", "label")

        logger.info(f"Loading HuggingFace dataset: {dataset_name} (split={split})")
        hf_dataset = HFDataset.from_pretrained(dataset_name, split=split, **params)

        wrapper = HuggingFaceDatasetWrapper(hf_dataset, text_column, label_column)
        metadata = {
            "num_classes": (
                len(hf_dataset.features[label_column]) if label_column and label_column in hf_dataset.features else 1
            ),
            "input_shape": None,  # Variable length text
        }
        return wrapper, metadata
