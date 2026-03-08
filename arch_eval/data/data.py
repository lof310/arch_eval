"""Dataset handling and synthetic generation."""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.datasets import (
    make_blobs,
    make_circles,
    make_classification,
    make_friedman1,
    make_friedman2,
    make_friedman3,
    make_moons,
    make_multilabel_classification,
    make_regression,
    make_sparse_uncorrelated,
)
from torch.utils.data import DataLoader, Dataset, IterableDataset, TensorDataset

from arch_eval.core.exceptions import DatasetFormatError

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

try:
    from datasets import Dataset as HFDataset
    from datasets import IterableDataset as HFIterableDataset

    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    HFDataset = object
    HFIterableDataset = object


class SyntheticDataset(Dataset):
    """Wrapper for synthetic datasets."""

    def __init__(self, data: torch.Tensor, targets: torch.Tensor):
        self.data, self.targets = data, targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def create_synthetic_dataset(dataset_type: str, params: Dict[str, Any]) -> SyntheticDataset:
    """Create synthetic dataset of specified type."""
    n_samples = params.get("n_samples", 1000)
    n_features = params.get("n_features", 20)
    noise = params.get("noise", 0.1)
    random_state = params.get("random_state", 42)

    if dataset_type == "classification":
        n_classes = params.get("n_classes", 2)
        n_clusters_per_class = params.get("n_clusters_per_class", 2)
        n_informative = params.get("n_informative", n_features // 2)

        # Ensure n_informative is large enough to separate the classes
        required_informative = int(np.ceil(np.log2(n_classes * n_clusters_per_class)))
        if n_informative < required_informative:
            if "n_informative" not in params:   # user didn't set it, we can adjust
                n_informative = min(required_informative, n_features)
                logger.warning(
                    f"n_informative increased from {n_features//2} to {n_informative} "
                    f"to accommodate {n_classes} classes with {n_clusters_per_class} clusters each."
                )
            else:
                raise ValueError(
                    f"n_informative={n_informative} is too small for {n_classes} classes "
                    f"with {n_clusters_per_class} clusters per class. Minimum required is "
                    f"{required_informative}. Either increase n_informative, reduce n_classes, "
                    f"or increase n_features."
                )
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=n_informative,
            n_clusters_per_class=n_clusters_per_class,
            n_redundant=2,
            random_state=random_state,
            flip_y=noise,
        )
    elif dataset_type == "regression":
        n_targets = params.get("n_targets", 1)
        X, y = make_regression(
            n_samples=n_samples, n_features=n_features, n_targets=n_targets, noise=noise, random_state=random_state
        )
    elif dataset_type == "blobs":
        n_centers = params.get("n_centers", 3)
        cluster_std = params.get("cluster_std", 1.0)
        X, y = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=n_centers,
            cluster_std=cluster_std,
            random_state=random_state,
        )
    elif dataset_type == "circles":
        factor = params.get("factor", 0.5)
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)
    elif dataset_type == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif dataset_type == "friedman1":
        X, y = make_friedman1(n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state)
    elif dataset_type == "friedman2":
        X, y = make_friedman2(n_samples=n_samples, noise=noise, random_state=random_state)
    elif dataset_type == "friedman3":
        X, y = make_friedman3(n_samples=n_samples, noise=noise, random_state=random_state)
    elif dataset_type == "sparse_uncorrelated":
        X, y = make_sparse_uncorrelated(n_samples=n_samples, n_features=n_features, random_state=random_state)
    elif dataset_type == "multilabel":
        n_classes = params.get("n_classes", 5)
        n_labels = params.get("n_labels", 2)
        X, y = make_multilabel_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_labels=n_labels,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown synthetic dataset type: {dataset_type}")

    X = torch.from_numpy(X).float()
    if dataset_type in ("regression", "friedman1", "friedman2", "friedman3"):
        y = torch.from_numpy(y).float()
        if y.ndim == 1:
            y = y.unsqueeze(1)
    else:
        y = torch.from_numpy(y).long()
    return SyntheticDataset(X, y)


class TransformDataset(Dataset):
    """Wrapper to apply transforms to a dataset."""

    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            data, target = item[0], item[1]
            if self.transform:
                data = self.transform(data)
            if self.target_transform:
                target = self.target_transform(target)
            return (data, target) + item[2:] if len(item) > 2 else (data, target)
        else:
            data = item
            return self.transform(data) if self.transform else data


class DatasetHandler:
    """Handles conversion of various dataset formats to PyTorch DataLoaders."""

    def __init__(self, config):
        self.config = config
        self.transform = config.transform
        self.target_transform = config.target_transform
        self.collate_fn = config.collate_fn

    def prepare_loaders(self) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        dataset = self.config.dataset
        params = self.config.dataset_params
        batch_size = self.config.training_args.get("batch_size", 32)
        dl_params = self.config.dataloader_params

        # Streaming dataset
        streaming = self.config.dataset_streaming

        if isinstance(dataset, str):
            if dataset.startswith("synthetic "):
                dataset = create_synthetic_dataset(dataset.replace("synthetic ", ""), params)
            elif TORCHVISION_AVAILABLE and dataset.lower() in TORCHVISION_DATASETS:
                dataset = self._load_torchvision(dataset, params)
            else:
                raise DatasetFormatError(f"Unknown dataset string: {dataset}")

        if HUGGINGFACE_AVAILABLE and isinstance(dataset, (HFDataset, HFIterableDataset)):
            dataset = self._from_huggingface(dataset, streaming)

        # Sharding for distributed training
        shard_cfg = self.config.dataset_shard
        if shard_cfg and hasattr(dataset, "shard"):
            num_shards = shard_cfg.get("num_shards", 1)
            shard_id = shard_cfg.get("shard_id", 0)
            dataset = dataset.shard(num_shards=num_shards, index=shard_id)

        if isinstance(dataset, Dataset):
            if self.transform or self.target_transform:
                dataset = TransformDataset(dataset, self.transform, self.target_transform)
            full = dataset
        elif isinstance(dataset, torch.Tensor):
            targets = params.get("targets", torch.zeros(len(dataset)))
            full = TensorDataset(dataset, targets)
        elif isinstance(dataset, dict):
            return self._from_dict(dataset, batch_size, dl_params)
        elif isinstance(dataset, (list, tuple)) and len(dataset) == 2:
            data, targets = dataset
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            if isinstance(targets, np.ndarray):
                targets = torch.from_numpy(targets)
            full = TensorDataset(data, targets)
        else:
            raise DatasetFormatError(f"Unsupported dataset format: {type(dataset)}")

        return self._create_splits(full, batch_size, dl_params, streaming)

    def _load_torchvision(self, name: str, params: Dict) -> Dataset:
        name = name.lower()
        if name not in TORCHVISION_DATASETS:
            raise DatasetFormatError(f"Unknown torchvision dataset: {name}")
        transform = self.transform or T.ToTensor()
        target_transform = self.target_transform
        split = params.get("split", "train")
        download = params.get("download", True)
        root = params.get("root", "./data")
        cls_map = {
            "mnist": tv_datasets.MNIST,
            "fashion_mnist": tv_datasets.FashionMNIST,
            "cifar10": tv_datasets.CIFAR10,
            "cifar100": tv_datasets.CIFAR100,
            "svhn": tv_datasets.SVHN,
            "imagenet": tv_datasets.ImageNet,
        }
        cls = cls_map[name]
        if name == "svhn":
            return cls(
                root=root, split=split, transform=transform, target_transform=target_transform, download=download
            )
        else:
            is_train = split == "train"
            return cls(
                root=root, train=is_train, transform=transform, target_transform=target_transform, download=download
            )

    def _from_huggingface(self, hf_dataset, streaming):
        """Convert HuggingFace dataset to PyTorch Dataset."""
        if streaming:

            class HFDatasetWrapper(torch.utils.data.IterableDataset):
                def __init__(self, hf_ds, transform, target_transform):
                    self.hf_ds = hf_ds
                    self.transform = transform
                    self.target_transform = target_transform

                def __iter__(self):
                    for item in self.hf_ds:
                        # Infer columns
                        if isinstance(item, dict):
                            data = item.get(
                                "image", item.get("pixel_values", item.get("input_ids", list(item.values())[0]))
                            )
                            target = item.get(
                                "label", item.get("labels", list(item.values())[1] if len(item) > 1 else None)
                            )
                        else:
                            data, target = item[0], item[1] if len(item) > 1 else None

                        if self.transform:
                            data = self.transform(data)
                        if target is not None and self.target_transform:
                            target = self.target_transform(target)

                        yield (data, target) if target is not None else (data, None)

            return HFDatasetWrapper(hf_dataset, self.transform, self.target_transform)
        else:

            class HFDatasetWrapper(torch.utils.data.Dataset):
                def __init__(self, hf_ds, transform, target_transform):
                    self.hf_ds = hf_ds
                    self.transform = transform
                    self.target_transform = target_transform
                    cols = hf_ds.column_names
                    self.data_col = (
                        "image"
                        if "image" in cols
                        else (
                            "pixel_values"
                            if "pixel_values" in cols
                            else ("input_ids" if "input_ids" in cols else cols[0])
                        )
                    )
                    self.target_col = (
                        "label"
                        if "label" in cols
                        else ("labels" if "labels" in cols else (cols[1] if len(cols) > 1 else None))
                    )

                def __len__(self):
                    return len(self.hf_ds)

                def __getitem__(self, idx):
                    item = self.hf_ds[idx]
                    data = item[self.data_col]
                    if self.transform:
                        data = self.transform(data)
                    if self.target_col is None:
                        return data, None
                    target = item[self.target_col]
                    if self.target_transform:
                        target = self.target_transform(target)
                    return data, target

            return HFDatasetWrapper(hf_dataset, self.transform, self.target_transform)

    def _from_dict(self, d: Dict, batch_size: int, dl_params: Dict):
        if "train" in d:
            self.train_dataset = self._to_tensor_dataset(d["train"])
            self.val_dataset = self._to_tensor_dataset(d.get("val"))
            self.test_dataset = self._to_tensor_dataset(d.get("test"))
            return self._create_splits(None, batch_size, dl_params)
        data, targets = d.get("data"), d.get("targets")
        if data is None or targets is None:
            raise DatasetFormatError("Dictionary must contain 'data' and 'targets' keys")
        if "train_mask" in d:
            self.train_dataset = self._apply_mask(data, targets, d["train_mask"])
            self.val_dataset = self._apply_mask(data, targets, d.get("val_mask"))
            self.test_dataset = self._apply_mask(data, targets, d.get("test_mask"))
            return self._create_splits(None, batch_size, dl_params)
        return self._create_splits(self._to_tensor_dataset((data, targets)), batch_size, dl_params)

    def _apply_mask(self, data, targets, mask):
        if mask is None:
            return None
        if isinstance(mask, (list, np.ndarray)):
            mask = torch.tensor(mask, dtype=torch.bool)
        idx = torch.where(mask)[0]
        data_sub = data[idx] if isinstance(data, torch.Tensor) else data[idx.numpy()]
        targets_sub = targets[idx] if isinstance(targets, torch.Tensor) else targets[idx.numpy()]
        return self._to_tensor_dataset((data_sub, targets_sub))

    def _to_tensor_dataset(self, item) -> Dataset:
        if isinstance(item, Dataset):
            return item
        if isinstance(item, (tuple, list)) and len(item) == 2:
            data, targets = item
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data).float()
            if isinstance(targets, np.ndarray):
                targets = torch.from_numpy(targets)
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data)
            if not isinstance(targets, torch.Tensor):
                targets = torch.tensor(targets)
            if len(data) != len(targets):
                raise DatasetFormatError(f"Data and targets length mismatch: {len(data)} vs {len(targets)}")
            ds = TensorDataset(data, targets)
            if self.transform or self.target_transform:
                ds = TransformDataset(ds, self.transform, self.target_transform)
            return ds
        raise DatasetFormatError(f"Cannot convert {type(item)} to TensorDataset")

    def _create_splits(self, dataset: Optional[Dataset], batch_size: int, dl_params: Dict, streaming: bool = False):
        if hasattr(self, "train_dataset"):
            train_loader = self._build_loader(
                self.train_dataset, batch_size, shuffle=True, dl_params=dl_params, streaming=streaming
            )
            val_loader = (
                self._build_loader(
                    self.val_dataset, batch_size, shuffle=False, dl_params=dl_params, streaming=streaming
                )
                if self.val_dataset
                else None
            )
            test_loader = (
                self._build_loader(
                    self.test_dataset, batch_size, shuffle=False, dl_params=dl_params, streaming=streaming
                )
                if self.test_dataset
                else None
            )
            return train_loader, val_loader, test_loader

        if dataset is None:
            raise DatasetFormatError("No dataset provided")
        if streaming:
            return (
                self._build_loader(dataset, batch_size, shuffle=False, dl_params=dl_params, streaming=streaming),
                None,
                None,
            )
        total = len(dataset)
        train_len = int(0.9 * total)
        val_len = int(0.05 * total)
        test_len = total - train_len - val_len
        train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])
        return (
            self._build_loader(train_ds, batch_size, shuffle=True, dl_params=dl_params),
            self._build_loader(val_ds, batch_size, shuffle=False, dl_params=dl_params),
            self._build_loader(test_ds, batch_size, shuffle=False, dl_params=dl_params),
        )

    def _build_loader(self, ds, batch_size, shuffle, dl_params, streaming=False):
        if ds is None:
            return None
        kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle and not streaming,
            "collate_fn": self.collate_fn,
            **dl_params,
        }
        if kwargs.get("num_workers", 0) == 0:
            kwargs.pop("prefetch_factor", None)
            kwargs.pop("persistent_workers", None)
        if streaming:
            kwargs.pop("shuffle", None)
        return DataLoader(ds, **kwargs)
