"""Factory for creating synthetic datasets."""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
from sklearn.datasets import (make_blobs, make_circles, make_classification,
                              make_friedman1, make_friedman2, make_friedman3,
                              make_moons, make_multilabel_classification,
                              make_regression, make_sparse_uncorrelated)
from torch.utils.data import Dataset

from arch_eval.data.factories.base import DatasetFactory

logger = logging.getLogger(__name__)


class SyntheticDataset(Dataset):
    """Wrapper for synthetic datasets."""

    def __init__(self, data: torch.Tensor, targets: torch.Tensor):
        self.data, self.targets = data, targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class SyntheticFactory(DatasetFactory):
    """Factory for creating synthetic datasets from sklearn."""

    SYNTHETIC_TYPES = {
        "classification": make_classification,
        "blobs": make_blobs,
        "moons": make_moons,
        "circles": make_circles,
        "regression": make_regression,
        "friedman1": make_friedman1,
        "friedman2": make_friedman2,
        "friedman3": make_friedman3,
        "multilabel": make_multilabel_classification,
        "sparse_uncorrelated": make_sparse_uncorrelated,
    }

    def can_handle(self, data: Any, config: Any) -> bool:
        if isinstance(data, str) and data in self.SYNTHETIC_TYPES:
            return True
        if hasattr(config, "dataset_type") and config.dataset_type in self.SYNTHETIC_TYPES:
            return True
        return False

    def create(self, data: Any, config: Any) -> Tuple[Dataset, Optional[Dict[str, Any]]]:
        dataset_type = data if isinstance(data, str) else getattr(config, "dataset_type", "classification")
        # Get parameters from config
        params = getattr(config, "dataset_params", {}) or {}
        n_samples = params.get("n_samples", 1000)
        n_features = params.get("n_features", 20)
        if dataset_type == "regression":
            X, y = make_regression(n_samples=n_samples, n_features=n_features, random_state=42, **params)
            y = torch.FloatTensor(y)
        elif dataset_type in ["friedman1", "friedman2", "friedman3"]:
            func = self.SYNTHETIC_TYPES[dataset_type]
            X, y = func(n_samples=n_samples, n_features=n_features, random_state=42)
            y = torch.FloatTensor(y)
        elif dataset_type == "multilabel":
            X, y = make_multilabel_classification(n_samples=n_samples, n_features=n_features, random_state=42, **params)
            y = torch.LongTensor(y)
        elif dataset_type == "sparse_uncorrelated":
            X, y = make_sparse_uncorrelated(n_samples=n_samples, n_features=n_features, random_state=42)
            y = torch.LongTensor(y)
        else:
            # Classification variants
            func = self.SYNTHETIC_TYPES.get(dataset_type, make_classification)
            X, y = func(n_samples=n_samples, n_features=n_features, random_state=42, **params)
            y = torch.LongTensor(y)
        X = torch.FloatTensor(X)
        dataset = SyntheticDataset(X, y)
        metadata = {
            "num_classes": len(torch.unique(y)) if y.dtype in [torch.long, torch.int] else 1,
            "input_shape": X.shape[1:],
            "dataset_type": dataset_type,
        }
        logger.info(f"Created synthetic {dataset_type} dataset with {n_samples} samples")
        return dataset, metadata
