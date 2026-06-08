"""Factory for creating datasets from torchvision."""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset

from arch_eval.data.factories.base import DatasetFactory

logger = logging.getLogger(__name__)

try:
    import torchvision.datasets as tv_datasets
    import torchvision.transforms as T

    TORCHVISION_AVAILABLE = True
    TORCHVISION_DATASETS = {"mnist", "fashion_mnist", "cifar10", "cifar100", "svhn", "imagenet"}
except ImportError:
    TORCHVISION_AVAILABLE = False
    T = None
    tv_datasets = None
    TORCHVISION_DATASETS = set()


class TorchvisionFactory(DatasetFactory):
    """Factory for creating datasets from torchvision."""

    def can_handle(self, data: Any, config: Any) -> bool:
        if not TORCHVISION_AVAILABLE:
            return False
        if isinstance(data, str) and data.lower() in TORCHVISION_DATASETS:
            return True
        dataset_type = getattr(config, "dataset_type", "")
        if isinstance(dataset_type, str) and dataset_type.lower() in TORCHVISION_DATASETS:
            return True
        return False

    def create(self, data: Any, config: Any) -> Tuple[Dataset, Optional[Dict[str, Any]]]:
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision not available")

        dataset_name = (data if isinstance(data, str) else getattr(config, "dataset_type", "")).lower()
        # Map friendly names to actual class names
        name_map = {
            "mnist": "MNIST",
            "fashion_mnist": "FashionMNIST",
            "cifar10": "CIFAR10",
            "cifar100": "CIFAR100",
            "svhn": "SVHN",
            "imagenet": "ImageNet",
        }
        actual_name = name_map.get(dataset_name, dataset_name.capitalize())
        params = getattr(config, "dataset_params", {}) or {}
        root = params.get("root", "./data")
        train = params.get("train", True)
        download = params.get("download", True)
        # Get transforms
        transform = params.get("transform")
        if transform is None:
            # Default transforms based on dataset
            if dataset_name in ["cifar10", "cifar100"]:
                normalize = T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                if train:
                    transform = T.Compose(
                        [T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4), T.ToTensor(), normalize]
                    )
                else:
                    transform = T.Compose([T.ToTensor(), normalize])
            elif dataset_name == "mnist":
                transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
            else:
                transform = T.Compose([T.ToTensor()])

        # Create dataset
        dataset_class = getattr(tv_datasets, actual_name, None)
        if dataset_class is None:
            raise ValueError(f"Unknown torchvision dataset: {actual_name}")
        if dataset_name == "imagenet":
            # ImageNet requires special handling
            split = "train" if train else "val"
            dataset = dataset_class(root=root, split=split, transform=transform, **params)
        elif dataset_name == "svhn":
            dataset = dataset_class(
                root=root, split="train" if train else "test", transform=transform, download=download, **params
            )
        else:
            dataset = dataset_class(root=root, train=train, transform=transform, download=download, **params)

        metadata = {
            "num_classes": (
                10
                if dataset_name in ["mnist", "fashion_mnist", "cifar10", "svhn"]
                else 100 if dataset_name == "cifar100" else 1000
            ),
            "input_shape": (
                (1, 28, 28)
                if dataset_name == "mnist"
                else (3, 32, 32) if dataset_name in ["cifar10", "cifar100"] else (3, 224, 224)
            ),
        }
        logger.info(f"Created torchvision {dataset_name} dataset (train={train})")
        return dataset, metadata
