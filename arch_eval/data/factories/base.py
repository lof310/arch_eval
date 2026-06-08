"""Base factory interface for dataset creation."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from torch.utils.data import DataLoader, Dataset


class DatasetFactory(ABC):
    """Abstract base class for dataset factories.
    Factories encapsulate the logic for creating datasets from different sources.
    Each factory should implement can_handle() to check if it can process the given
    data/config, and create() to return the dataset and optional metadata.
    """

    @abstractmethod
    def can_handle(self, data: Any, config: Any) -> bool:
        """Check if this factory can handle the given data/config.
        Args:
            data: The raw data or dataset specification.
            config: The configuration object.
        Returns:
            True if this factory can create a dataset from the data.
        """
        pass

    @abstractmethod
    def create(self, data: Any, config: Any) -> Tuple[Dataset, Optional[Dict[str, Any]]]:
        """Create a dataset from the given data.
        Args:
            data: The raw data or dataset specification.
            config: The configuration object.
        Returns:
            A tuple of (dataset, metadata_dict). Metadata can include info like
            num_classes, input_shape, etc.
        """
        pass
