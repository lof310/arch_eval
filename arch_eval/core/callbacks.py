"""Callback base classes and built-in callbacks."""

import logging
import os
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class Callback:
    """Base class for callbacks. All methods are no-ops by default."""

    def before_training(self, trainer, config):
        pass

    def after_training(self, trainer, final_metrics):
        pass

    def on_train_start(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

    def on_epoch_start(self, trainer, epoch):
        pass

    def on_epoch_end(self, trainer, epoch, metrics):
        pass

    def on_batch_start(self, trainer, batch_idx, data, targets):
        pass

    def on_batch_end(self, trainer, batch_idx, loss):
        pass

    def on_validation_start(self, trainer):
        pass

    def on_validation_end(self, trainer, metrics):
        pass

    def on_backward(self, trainer, loss):
        pass

    def on_optimizer_step(self, trainer):
        pass

    def on_log(self, trainer, metrics, step):
        pass

    def on_checkpoint(self, trainer, checkpoint_path, is_best):
        pass

    def on_exception(self, trainer, exception):
        pass

    def register_hooks(self, plugin_manager):
        """Register all implemented methods as local hooks."""
        hook_map = {
            "before_training": self.before_training,
            "after_training": self.after_training,
            "on_train_start": self.on_train_start,
            "on_train_end": self.on_train_end,
            "on_epoch_start": self.on_epoch_start,
            "on_epoch_end": self.on_epoch_end,
            "on_batch_start": self.on_batch_start,
            "on_batch_end": self.on_batch_end,
            "on_validation_start": self.on_validation_start,
            "on_validation_end": self.on_validation_end,
            "on_backward": self.on_backward,
            "on_optimizer_step": self.on_optimizer_step,
            "on_log": self.on_log,
            "on_checkpoint": self.on_checkpoint,
            "on_exception": self.on_exception,
        }
        for name, method in hook_map.items():
            if method.__func__ is not getattr(Callback, name):
                plugin_manager.register_local_hook(name, method)


class EarlyStopping(Callback):
    """Stop training when a monitored metric has stopped improving."""

    def __init__(self, monitor="val_loss", min_delta=0.001, patience=10, mode="min"):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.best = float("inf") if mode == "min" else -float("inf")
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, trainer, epoch, metrics):
        if self.monitor not in metrics:
            return
        current = metrics[self.monitor]
        if (self.mode == "min" and current < self.best - self.min_delta) or (
            self.mode == "max" and current > self.best + self.min_delta
        ):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                trainer.stop_training = True
                self.stopped_epoch = epoch
                logger.info(f"Early stopping triggered at epoch {epoch}")


class ModelCheckpoint(Callback):
    """Save the model after every epoch if it improves."""

    def __init__(self, filepath, monitor="val_loss", save_best_only=True, mode="min"):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.best = float("inf") if mode == "min" else -float("inf")

    def on_epoch_end(self, trainer, epoch, metrics):
        if self.monitor not in metrics:
            return
        current = metrics[self.monitor]
        filepath = self.filepath.format(epoch=epoch, **metrics)
        if not self.save_best_only:
            self._save(trainer, filepath, is_best=False)
            return
        improved = (self.mode == "min" and current < self.best) or (self.mode == "max" and current > self.best)
        if improved:
            self.best = current
            self._save(trainer, filepath, is_best=True)

    def _save(self, trainer, filepath, is_best):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Use temporary file to avoid corruption
        tmp = filepath + ".tmp"
        torch.save(
            {
                "epoch": trainer.current_epoch,
                "model_state_dict": trainer.model.state_dict(),
                "optimizer_state_dicts": [opt.state_dict() for opt in trainer.optimizers],
                "scheduler_state_dicts": [sch["scheduler"].state_dict() for sch in trainer.schedulers],
                "config": trainer.config,
                "metrics": trainer.history[-1] if trainer.history else {},
            },
            tmp,
        )
        os.replace(tmp, filepath)
        logger.info(f"Checkpoint saved to {filepath}" + (" (best)" if is_best else ""))


class LRSchedulerLogger(Callback):
    """Log learning rates after each epoch/step."""

    def on_epoch_end(self, trainer, epoch, metrics):
        for i, opt in enumerate(trainer.optimizers):
            for j, pg in enumerate(opt.param_groups):
                trainer.plugin_manager.execute_hook("on_log", trainer, {f"lr_opt{i}_group{j}": pg["lr"]}, epoch)


class TensorBoardLogger(Callback):
    """Log metrics to TensorBoard."""

    def __init__(self, log_dir="./logs"):
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir)
        except ImportError:
            logger.warning("TensorBoard not installed. Install with 'pip install tensorboard'")
            self.writer = None

    def on_log(self, trainer, metrics, step):
        if self.writer:
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, step)

    def on_train_end(self, trainer):
        if self.writer:
            self.writer.close()
