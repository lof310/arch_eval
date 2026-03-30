"""Comprehensive test for transformer compatibility including vision and language."""
import torch
import torch.nn as nn
from arch_eval import TrainingConfig, Trainer
from arch_eval.core.config import TaskType

print("Testing full transformer compatibility...")

# Test 1: Language Model with dict output (Hugging Face style)
class DictOutputTransformer(nn.Module):
    def __init__(self, vocab_size=100, hidden=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.linear = nn.Linear(hidden, vocab_size)
    
    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids)
        logits = self.linear(x.mean(dim=1))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return {"logits": logits, "loss": loss}

print("\n1. Testing Dict Output Transformer (Hugging Face style)...")
config = TrainingConfig(
    dataset="synthetic text",
    dataset_params={"vocab_size": 100, "seq_length": 32, "n_samples": 100},
    task="next-token-prediction",
    training_args={"batch_size": 16, "num_epochs": 2},
    device="cpu"
)
model = DictOutputTransformer()
trainer = Trainer(model, config)
history = trainer.train()
assert "train_loss" in history, "Dict output model failed"
print("✓ Dict output transformer passed")

# Test 2: Language Model with tuple output (logits, loss)
class TupleOutputTransformer(nn.Module):
    def __init__(self, vocab_size=100, hidden=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.linear = nn.Linear(hidden, vocab_size)
    
    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids)
        logits = self.linear(x.mean(dim=1))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return (logits, loss)

print("\n2. Testing Tuple Output Transformer...")
config = TrainingConfig(
    dataset="synthetic text",
    dataset_params={"vocab_size": 100, "seq_length": 32, "n_samples": 100},
    task="next-token-prediction",
    training_args={"batch_size": 16, "num_epochs": 2},
    device="cpu"
)
model = TupleOutputTransformer()
trainer = Trainer(model, config)
history = trainer.train()
assert "train_loss" in history, "Tuple output model failed"
print("✓ Tuple output transformer passed")

# Test 3: Vision Model with shapes pattern
print("\n3. Testing Vision Model with Synthetic Shapes...")
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)
    
    def forward(self, x, labels=None):
        x = torch.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        logits = self.fc(x)
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
            return {"logits": logits, "loss": loss}
        return {"logits": logits}

config = TrainingConfig(
    dataset="synthetic image",
    dataset_params={"img_size": 32, "channels": 3, "n_samples": 100, "n_classes": 10, "pattern": "shapes"},
    task="classification",
    training_args={"batch_size": 16, "num_epochs": 2},
    device="cpu"
)
model = SimpleCNN()
trainer = Trainer(model, config)
history = trainer.train()
assert "train_loss" in history, "Vision model failed"
print("✓ Vision model with shapes passed")

# Test 4: Vision Model with gradient pattern
print("\n4. Testing Vision Model with Gradient Pattern...")
config = TrainingConfig(
    dataset="synthetic image",
    dataset_params={"img_size": 32, "channels": 3, "n_samples": 100, "n_classes": 10, "pattern": "gradient"},
    task="classification",
    training_args={"batch_size": 16, "num_epochs": 2},
    device="cpu"
)
model = SimpleCNN()
trainer = Trainer(model, config)
history = trainer.train()
assert "train_loss" in history, "Vision gradient model failed"
print("✓ Vision model with gradient passed")

# Test 5: Standard tensor output (baseline)
print("\n5. Testing Standard Tensor Output (Baseline)...")
class StandardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(20, 2)
    
    def forward(self, x, labels=None):
        logits = self.fc(x)
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
            return loss
        return logits

from arch_eval.data import create_synthetic_dataset
dataset = create_synthetic_dataset("classification", {"n_samples": 100, "n_features": 20, "n_classes": 2})

config = TrainingConfig(
    dataset=dataset,
    task="classification",
    training_args={"batch_size": 16, "num_epochs": 2},
    device="cpu"
)
model = StandardModel()
trainer = Trainer(model, config)
history = trainer.train()
assert "train_loss" in history, "Standard model failed"
print("✓ Standard tensor output passed")

print("\n✅ All transformer compatibility tests passed!")
