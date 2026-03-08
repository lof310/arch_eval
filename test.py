"""Comprehensive tests for arch_eval library."""

import torch
import torch.nn as nn
import pytest
from arch_eval import Trainer, TrainingConfig, Benchmark, BenchmarkConfig, HyperparameterOptimizer
from arch_eval.core.exceptions import ModelError, ConfigurationError
from arch_eval.data.data import create_synthetic_dataset
import tempfile
import os


# Simple model for testing
class SimpleMLP(nn.Module):
    def __init__(self, input_size=20, hidden=50, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Test 1: Basic Trainer initialization and training
# ============================================================
def test_trainer_basic():
    """Test basic trainer initialization and training loop."""
    config = TrainingConfig(
        dataset="synthetic classification",
        dataset_params={
            "n_samples": 100,
            "n_features": 20,
            "n_classes": 5,
            "n_informative": 10
        },
        training_args={
            "num_epochs": 2,
            "batch_size": 16,
            "learning_rate": 0.001
        },
        task="classification",
        realtime=False,  # Disable realtime window for tests
        log_interval=1,
        eval_interval=1
    )
    
    model = SimpleMLP(input_size=20, num_classes=5)
    trainer = Trainer(model, config)
    history = trainer.train()
    
    assert isinstance(history, dict)
    assert len(history) > 0
    assert "train_loss" in history or "loss" in history
    print("✓ Trainer basic test passed")


# ============================================================
# Test 2: Trainer with checkpointing
# ============================================================
def test_trainer_checkpoint():
    """Test trainer with checkpoint saving and loading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainingConfig(
            dataset="synthetic classification",
            dataset_params={
                "n_samples": 100,
                "n_features": 20,
                "n_classes": 5
            },
            training_args={
                "num_epochs": 2,
                "batch_size": 16
            },
            task="classification",
            checkpoint_dir=tmpdir,
            save_best_only=True,
            checkpoint_metric="val_loss",
            realtime=False
        )
        
        model = SimpleMLP(input_size=20, num_classes=5)
        trainer = Trainer(model, config)
        trainer.train()
        
        # Check if checkpoint was saved
        checkpoint_files = os.listdir(tmpdir)
        assert len(checkpoint_files) > 0
        assert any("best_model.pt" in f for f in checkpoint_files)
        
        # Test loading checkpoint
        trainer.load_checkpoint(os.path.join(tmpdir, "best_model.pt"))
        print("✓ Checkpoint test passed")


# ============================================================
# Test 3: Benchmark multiple models
# ============================================================
def test_benchmark():
    """Test benchmarking multiple models."""
    models = [
        {"name": "Small MLP", "model": SimpleMLP(hidden=32)},
        {"name": "Large MLP", "model": SimpleMLP(hidden=64)}
    ]
    
    config = BenchmarkConfig(
        dataset="synthetic classification",
        dataset_params={
            "n_samples": 100,
            "n_features": 20,
            "n_classes": 5
        },
        training_args={
            "num_epochs": 1,
            "batch_size": 16
        },
        compare_metrics=["accuracy", "loss"],
        parallel=False,  # Use sequential for testing
        realtime=False
    )
    
    benchmark = Benchmark(models, config)
    results = benchmark.run()
    
    assert len(results) == 2
    assert "model_name" in results.columns
    assert "accuracy" in results.columns or "val_accuracy" in results.columns
    print("✓ Benchmark test passed")


# ============================================================
# Test 4: Hyperparameter optimization
# ============================================================
def test_hpo():
    """Test hyperparameter optimization."""
    def model_fn():
        return SimpleMLP(input_size=20, num_classes=5)
    
    base_config = TrainingConfig(
        dataset="synthetic classification",
        dataset_params={
            "n_samples": 100,
            "n_features": 20,
            "n_classes": 5
        },
        training_args={
            "num_epochs": 1,
            "batch_size": 16
        },
        task="classification",
        realtime=False
    )
    
    param_grid = {
        "learning_rate": [0.001, 0.01],
        "hidden": [32, 64]
    }
    
    optimizer = HyperparameterOptimizer(
        model_fn, base_config, param_grid,
        search_type="grid", metric="val_loss", mode="min"
    )
    
    results = optimizer.run()
    
    assert len(results) == 4  # 2x2 grid
    assert "learning_rate" in results.columns
    assert "hidden" in results.columns
    print("✓ HPO test passed")


# ============================================================
# Test 5: Synthetic dataset creation with automatic n_informative adjustment
# ============================================================
def test_synthetic_dataset_adjustment():
    """Test that synthetic dataset creation automatically adjusts n_informative."""
    # This should work even though default n_informative (n_features//2=10) is too low
    # for 50 classes (requires at least log2(50*2)=~7, but 10 is actually enough here)
    dataset = create_synthetic_dataset(
        "classification",
        {
            "n_samples": 100,
            "n_features": 20,
            "n_classes": 50,
            # No n_informative provided - should auto-adjust
        }
    )
    
    assert len(dataset) == 100
    assert dataset.data.shape[1] == 20
    
    # This should raise an error because user explicitly set n_informative too low
    with pytest.raises(ValueError, match="n_informative=5 is too small"):
        create_synthetic_dataset(
            "classification",
            {
                "n_samples": 100,
                "n_features": 20,
                "n_classes": 50,
                "n_informative": 5  # Explicitly too low
            }
        )
    print("✓ Synthetic dataset adjustment test passed")


# ============================================================
# Test 6: Model validation with wrong input size
# ============================================================
def test_model_validation_failure():
    """Test that model validation fails with wrong input size."""
    config = TrainingConfig(
        dataset="synthetic classification",
        dataset_params={
            "n_samples": 100,
            "n_features": 20,  # Dataset has 20 features
            "n_classes": 5
        },
        training_args={
            "num_epochs": 1,
            "batch_size": 16
        },
        task="classification",
        realtime=False
    )
    
    # Model expects 10 features but dataset has 20
    model = SimpleMLP(input_size=10, num_classes=5)
    
    with pytest.raises(ModelError, match="Model validation failed"):
        Trainer(model, config)
    print("✓ Model validation test passed")


# ============================================================
# Test 7: Configuration validation
# ============================================================
def test_config_validation():
    """Test that configuration validation catches errors."""
    # Missing wandb project when log_to_wandb=True
    with pytest.raises(ConfigurationError, match="wandb_project must be specified"):
        TrainingConfig(
            dataset="synthetic classification",
            dataset_params={"n_samples": 100, "n_features": 20, "n_classes": 5},
            log_to_wandb=True,
            wandb_project=None
        )
    
    # Invalid early stopping mode
    with pytest.raises(ConfigurationError, match="early_stopping_mode must be 'min' or 'max'"):
        TrainingConfig(
            dataset="synthetic classification",
            dataset_params={"n_samples": 100, "n_features": 20, "n_classes": 5},
            early_stopping_patience=5,
            early_stopping_mode="invalid"
        )
    
    # Distributed training on CPU
    with pytest.raises(ConfigurationError, match="Distributed training requires GPU"):
        TrainingConfig(
            dataset="synthetic classification",
            dataset_params={"n_samples": 100, "n_features": 20, "n_classes": 5},
            distributed_backend="ddp",
            device="cpu"
        )
    print("✓ Config validation test passed")


# ============================================================
# Run all tests
# ============================================================
if __name__ == "__main__":
    print("Running arch_eval tests...\n")
    test_trainer_basic()
    test_trainer_checkpoint()
    test_benchmark()
    test_hpo()
    test_synthetic_dataset_adjustment()
    test_model_validation_failure()
    test_config_validation()
    print("\n✅ All tests passed!")
