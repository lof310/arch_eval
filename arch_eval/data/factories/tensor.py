"""Factory for creating datasets from torch tensors."""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset, TensorDataset

from arch_eval.data.factories.base import DatasetFactory

logger = logging.getLogger(__name__)


class TensorFactory(DatasetFactory):
    """Factory for creating datasets from torch tensors, numpy arrays, or existing TensorDataset."""

    def can_handle(self, data: Any, config: Any) -> bool:
        if isinstance(data, TensorDataset):
            return True
        if isinstance(data, (tuple, list)) and len(data) >= 2:
            return True
        if isinstance(data, torch.Tensor):
            return True
        try:
            import numpy as np

            if isinstance(data, np.ndarray):
                return True
        except ImportError:
            pass
        return False

    def create(self, data: Any, config: Any) -> Tuple[Dataset, Optional[Dict[str, Any]]]:
        # If already a TensorDataset, return it directly
        if isinstance(data, TensorDataset):
            X, y = data.tensors[0], data.tensors[1] if len(data.tensors) > 1 else data.tensors[0]
            dataset = data
            metadata = {
                "num_classes": len(torch.unique(y)) if y.dtype in [torch.long, torch.int] else 1,
                "input_shape": X.shape[1:] if len(X.shape) > 1 else X.shape,
            }
            logger.debug(f"Using existing TensorDataset with {len(X)} samples")
            return dataset, metadata
        if isinstance(data, (tuple, list)):
            X, y = data[0], data[1]
        elif isinstance(data, torch.Tensor):
            # Single tensor - use it as both X and y (for autoencoders etc.)
            X, y = data, data
        else:
            import numpy as np

            if isinstance(data, np.ndarray):
                if len(data.shape) == 1 or data.shape[1] == 1:
                    X = torch.FloatTensor(data)
                    y = X.clone()
                else:
                    X = torch.FloatTensor(data[:, :-1])
                    y = torch.FloatTensor(data[:, -1])
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")

        # Convert to tensors if needed
        if not isinstance(X, torch.Tensor):
            import numpy as np

            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X)
        if not isinstance(y, torch.Tensor):
            import numpy as np

            if isinstance(y, np.ndarray):
                if y.dtype in [np.int32, np.int64]:
                    y = torch.LongTensor(y)
                else:
                    y = torch.FloatTensor(y)

        dataset = TensorDataset(X, y)
        metadata = {
            "num_classes": len(torch.unique(y)) if y.dtype in [torch.long, torch.int] else 1,
            "input_shape": X.shape[1:] if len(X.shape) > 1 else X.shape,
        }
        logger.debug(f"Created TensorDataset with {len(X)} samples")
        return dataset, metadata
