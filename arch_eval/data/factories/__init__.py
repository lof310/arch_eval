"""Dataset factories for flexible dataset creation."""

from arch_eval.data.factories.base import DatasetFactory
from arch_eval.data.factories.dict import DictFactory
from arch_eval.data.factories.huggingface import HuggingFaceFactory
from arch_eval.data.factories.iterable import IterableFactory
from arch_eval.data.factories.synthetic import SyntheticFactory
from arch_eval.data.factories.tensor import TensorFactory
from arch_eval.data.factories.torchvision import TorchvisionFactory

__all__ = [
    "DatasetFactory",
    "SyntheticFactory",
    "TensorFactory",
    "TorchvisionFactory",
    "HuggingFaceFactory",
    "IterableFactory",
    "DictFactory",
]
