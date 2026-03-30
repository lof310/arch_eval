"""Dataset handling and synthetic generation."""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.datasets import (make_blobs, make_circles, make_classification,
                              make_friedman1, make_friedman2, make_friedman3,
                              make_moons, make_multilabel_classification,
                              make_regression, make_sparse_uncorrelated)
from torch.utils.data import (DataLoader, Dataset, IterableDataset,
                              TensorDataset)

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

# Try to import transformers tokenizer for language modeling
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None


class SyntheticDataset(Dataset):
    """Wrapper for synthetic datasets."""

    def __init__(self, data: torch.Tensor, targets: torch.Tensor):
        self.data, self.targets = data, targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class TextDataset(Dataset):
    """Dataset for language modeling with token sequences."""

    def __init__(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        self.input_ids = input_ids
        self.labels = labels if labels is not None else input_ids.clone()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.input_ids[idx], self.labels[idx]
        return self.input_ids[idx], self.input_ids[idx]


class ImageDataset(Dataset):
    """Dataset for vision tasks with images."""

    def __init__(self, images: torch.Tensor, labels: Optional[torch.Tensor] = None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        if self.labels is not None:
            return img, self.labels[idx]
        return img, img  # For reconstruction tasks


class VisionLanguageDataset(Dataset):
    """Dataset for vision-language tasks with image-text pairs."""

    def __init__(self, images: torch.Tensor, input_ids: torch.Tensor, 
                 attention_mask: Optional[torch.Tensor] = None,
                 labels: Optional[torch.Tensor] = None):
        self.images = images
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels if labels is not None else input_ids.clone()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = {
            "pixel_values": self.images[idx],
            "input_ids": self.input_ids[idx],
        }
        if self.attention_mask is not None:
            item["attention_mask"] = self.attention_mask[idx]
        item["labels"] = self.labels[idx]
        return item


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
            if "n_informative" not in params:  # user didn't set it, we can adjust
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


def create_synthetic_text_dataset(params: Dict[str, Any]) -> TextDataset:
    """
    Create synthetic text dataset for language modeling.
    
    Generates random token sequences simulating text data for transformer training.
    
    :param params: Dictionary containing:
        - vocab_size: Size of vocabulary (default: 1000)
        - seq_length: Sequence length (default: 128)
        - n_samples: Number of samples (default: 1000)
        - entropy: Randomness level 0-1, lower = more pattern (default: 0.8)
    :return: TextDataset with input_ids and labels
    """
    vocab_size = params.get("vocab_size", 1000)
    seq_length = params.get("seq_length", 128)
    n_samples = params.get("n_samples", 1000)
    entropy = params.get("entropy", 0.8)
    random_state = params.get("random_state", 42)
    
    torch.manual_seed(random_state)
    
    # Generate token sequences with some structure (not purely random)
    # Lower entropy means more repeated patterns
    if entropy < 0.5:
        # High structure: use n-gram-like patterns
        base_patterns = torch.randint(0, vocab_size // 10, (n_samples, seq_length // 10))
        input_ids = base_patterns.repeat_interleave(10, dim=1)
        input_ids = input_ids[:, :seq_length]
        # Add some noise
        mask = torch.rand_like(input_ids.float()) < entropy
        noise = torch.randint(0, vocab_size, input_ids.shape)
        input_ids = torch.where(mask, noise, input_ids)
    else:
        # Low structure: mostly random with slight bias
        input_ids = torch.randint(0, vocab_size, (n_samples, seq_length))
        # Add some bigram structure
        for i in range(1, seq_length):
            bias = torch.rand(n_samples) < (1 - entropy) * 0.3
            input_ids[bias, i] = input_ids[bias, i-1]
    
    # Labels are shifted by one for causal language modeling
    labels = input_ids.clone()
    
    return TextDataset(input_ids, labels)


def create_synthetic_image_dataset(params: Dict[str, Any]) -> ImageDataset:
    """
    Create synthetic image dataset for vision tasks.
    
    Generates random images or simple geometric patterns.
    
    :param params: Dictionary containing:
        - img_size: Image size as int or tuple (H, W) (default: 32)
        - channels: Number of channels (default: 3)
        - n_samples: Number of samples (default: 1000)
        - n_classes: Number of classes (default: 10)
        - pattern: Type of pattern - 'random', 'gradient', 'shapes' (default: 'random')
    :return: ImageDataset with images and labels
    """
    img_size = params.get("img_size", 32)
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    channels = params.get("channels", 3)
    n_samples = params.get("n_samples", 1000)
    n_classes = params.get("n_classes", 10)
    pattern = params.get("pattern", "random")
    random_state = params.get("random_state", 42)
    
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    images = torch.zeros(n_samples, channels, img_size[0], img_size[1])
    labels = torch.randint(0, n_classes, (n_samples,))
    
    if pattern == "random":
        images = torch.rand(n_samples, channels, img_size[0], img_size[1])
    elif pattern == "gradient":
        for i in range(n_samples):
            # Create gradient based on class
            angle = (labels[i].item() / n_classes) * 2 * 3.14159
            x = torch.linspace(0, 1, img_size[1])
            y = torch.linspace(0, 1, img_size[0])
            xx, yy = torch.meshgrid(x, y)
            grad = torch.sin(xx * 10 + angle) * torch.cos(yy * 10 + angle)
            grad = (grad - grad.min()) / (grad.max() - grad.min())
            for c in range(channels):
                images[i, c] = grad + torch.rand_like(grad) * 0.1
    elif pattern == "shapes":
        for i in range(n_samples):
            cls = labels[i].item()
            img = torch.ones(channels, img_size[0], img_size[1]) * 0.5
            center_x = img_size[1] // 2
            center_y = img_size[0] // 2
            size = img_size[0] // 4
            
            # Different shapes based on class modulo
            shape_type = cls % 4
            color = torch.rand(channels, 1, 1)
            
            if shape_type == 0:  # Square
                img[:, center_y-size:center_y+size, center_x-size:center_x+size] = color
            elif shape_type == 1:  # Circle
                y_grid, x_grid = torch.meshgrid(torch.arange(img_size[0]), torch.arange(img_size[1]), indexing='ij')
                mask = ((x_grid - center_x)**2 + (y_grid - center_y)**2) < size**2
                img[:, mask] = color.squeeze()
            elif shape_type == 2:  # Triangle
                for cy in range(center_y - size, center_y + size):
                    width = int(size * (1 - abs(cy - center_y) / size))
                    img[:, cy, max(0, center_x-width):min(img_size[1], center_x+width)] = color.squeeze()
            else:  # Lines
                for c in range(channels):
                    if c % 2 == 0:
                        img[c, ::2, :] = color[c].item()
                    else:
                        img[c, :, ::2] = color[c].item()
            
            images[i] = img + torch.rand_like(img) * 0.05
    
    # Normalize to [0, 1]
    images = images.clamp(0, 1)
    
    return ImageDataset(images, labels)


def create_synthetic_vision_language_dataset(params: Dict[str, Any]) -> VisionLanguageDataset:
    """
    Create synthetic vision-language dataset for multi-modal tasks.
    
    Generates paired image and text token sequences.
    
    :param params: Dictionary containing:
        - img_size: Image size (default: 32)
        - channels: Number of image channels (default: 3)
        - vocab_size: Vocabulary size for text (default: 1000)
        - seq_length: Text sequence length (default: 64)
        - n_samples: Number of samples (default: 500)
        - correlation: How much text correlates with image class 0-1 (default: 0.7)
    :return: VisionLanguageDataset with images, input_ids, attention_mask, and labels
    """
    img_size = params.get("img_size", 32)
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    channels = params.get("channels", 3)
    vocab_size = params.get("vocab_size", 1000)
    seq_length = params.get("seq_length", 64)
    n_samples = params.get("n_samples", 500)
    correlation = params.get("correlation", 0.7)
    random_state = params.get("random_state", 42)
    
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    # Generate images with classes
    n_classes = 10
    images = torch.rand(n_samples, channels, img_size[0], img_size[1])
    labels = torch.randint(0, n_classes, (n_samples,))
    
    # Generate text tokens correlated with image class
    input_ids = torch.zeros(n_samples, seq_length, dtype=torch.long)
    attention_mask = torch.ones(n_samples, seq_length, dtype=torch.long)
    
    for i in range(n_samples):
        cls = labels[i].item()
        
        # Start with class-specific tokens (correlation)
        n_correlated = int(seq_length * correlation)
        if n_correlated > 0:
            # Use class as base for first tokens
            base_token = cls * (vocab_size // n_classes)
            input_ids[i, :n_correlated] = base_token + torch.randint(0, vocab_size // n_classes, (n_correlated,))
        
        # Fill rest with random tokens
        if n_correlated < seq_length:
            input_ids[i, n_correlated:] = torch.randint(0, vocab_size, (seq_length - n_correlated,))
    
    return VisionLanguageDataset(images, input_ids, attention_mask, labels)


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
                dataset_type = dataset.replace("synthetic ", "")
                # Handle new synthetic dataset types for vision and language
                if dataset_type == "text":
                    dataset = create_synthetic_text_dataset(params)
                elif dataset_type == "image":
                    dataset = create_synthetic_image_dataset(params)
                elif dataset_type == "vision_language" or dataset_type == "vl":
                    dataset = create_synthetic_vision_language_dataset(params)
                else:
                    dataset = create_synthetic_dataset(dataset_type, params)
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
