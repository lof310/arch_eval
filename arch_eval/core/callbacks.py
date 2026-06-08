"""Callback base classes and built-in callbacks."""

import logging
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
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

    def on_before_optimizer_step(self, trainer, gradients: List[torch.Tensor]):
        """Called before optimizer.step() with list of gradients per param group."""
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
            "on_before_optimizer_step": self.on_before_optimizer_step,
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

    def _sanitize_filepath(self, epoch, metrics):
        """Sanitize filepath by using only epoch and sanitized metric values."""
        # Filter out keys that contain path separators or colons
        safe_metrics = {}
        for k, v in metrics.items():
            if "/" not in k and ":" not in k and "\\" not in k:
                try:
                    # Round numeric values to avoid long decimals
                    if isinstance(v, (int, float)):
                        safe_metrics[k] = round(v, 4)
                    else:
                        safe_metrics[k] = str(v)[:20]  # Truncate strings
                except (TypeError, ValueError):
                    pass
        # Build a safe suffix from metrics
        suffix_parts = []
        for k, v in sorted(safe_metrics.items()):
            suffix_parts.append(f"{k}={v}")
        suffix = "_".join(suffix_parts)[:100]  # Limit length
        # Use epoch as base with optional metric suffix
        base_path = self.filepath.format(epoch=epoch) if "{epoch}" in self.filepath else self.filepath
        if suffix:
            # Insert suffix before extension
            if "." in base_path:
                name, ext = base_path.rsplit(".", 1)
                return f"{name}_{suffix}.{ext}"
            return f"{base_path}_{suffix}"
        return base_path

    def on_epoch_end(self, trainer, epoch, metrics):
        if self.monitor not in metrics:
            return
        current = metrics[self.monitor]
        filepath = self._sanitize_filepath(epoch, metrics)
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


class TextGeneratorCallback(Callback):
    """Generate and log text samples during evaluation for language model sanity checks.
    At each eval_interval, takes a fixed prompt from dataset_params.get("eval_prompt"),
    generates max_new_tokens via the model's generate method (if available), and logs
    the decoded text to wandb or prints it.
    """

    def __init__(self, eval_prompt: str = None, max_new_tokens: int = 50, num_samples: int = 1):
        self.eval_prompt = eval_prompt
        self.max_new_tokens = max_new_tokens
        self.num_samples = num_samples
        self._tokenizer = None

    def _get_tokenizer(self, trainer):
        """Try to get tokenizer from model or config."""
        if self._tokenizer is not None:
            return self._tokenizer
        # Check if model has a tokenizer attribute
        if hasattr(trainer.model, "tokenizer"):
            self._tokenizer = trainer.model.tokenizer
            return self._tokenizer
        # Check if config has tokenizer in dataset_params
        if hasattr(trainer.config, "dataset_params"):
            tok = trainer.config.dataset_params.get("tokenizer")
            if tok is not None:
                self._tokenizer = tok
                return self._tokenizer
        return None

    def on_validation_end(self, trainer, metrics):
        """Generate text samples at the end of validation."""
        # Only generate on specified eval intervals
        if trainer.current_epoch % trainer.config.eval_interval != 0:
            return
        # Get prompt from config if not set
        prompt = self.eval_prompt
        if prompt is None and hasattr(trainer.config, "dataset_params"):
            prompt = trainer.config.dataset_params.get("eval_prompt")
        if prompt is None:
            # Default prompts for demonstration
            default_prompts = [
                "The future of artificial intelligence",
                "In a world where machines can think",
                "Once upon a time",
            ]
            import random

            prompt = random.choice(default_prompts)
        tokenizer = self._get_tokenizer(trainer)
        # Check if model has generate method
        model = trainer.model
        if hasattr(model, "module"):
            model = model.module  # Unwrap DDP/FSDP
        if not hasattr(model, "generate"):
            trainer.logger.info("Model does not have generate method, skipping text generation")
            return
        try:
            import torch

            # Tokenize prompt
            if tokenizer is not None:
                inputs = tokenizer(prompt, return_tensors="pt").to(trainer.device)
                input_ids = inputs.get("input_ids", inputs)
                attention_mask = inputs.get("attention_mask", None)
            else:
                # Fallback: try to use model's encode method or create simple tokenization
                trainer.logger.warning("No tokenizer available, using dummy tokens")
                input_ids = torch.randint(0, 1000, (1, 10)).to(trainer.device)
                attention_mask = None
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    attention_mask=attention_mask,
                    pad_token_id=getattr(tokenizer, "pad_token_id", None) if tokenizer else None,
                )
            # Decode output
            if tokenizer is not None:
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                generated_text = f"[Generated tokens: {outputs.shape}] (no tokenizer for decoding)"
            # Log result
            log_msg = f"\n{'='*50}\nPrompt: {prompt}\nGenerated: {generated_text}\n{'='*50}"
            trainer.logger.info(log_msg)
            if trainer.config.log_to_wandb:
                import wandb

                wandb.log(
                    {
                        "generated_text": wandb.Html(f"<pre>{log_msg}</pre>"),
                        "generation_step": trainer.current_epoch,
                    }
                )
        except Exception as e:
            trainer.logger.warning(f"Text generation failed: {e}")


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


class SlopeEarlyStopping(Callback):
    """Stop training when the metric's slope over a window becomes flat."""

    def __init__(self, monitor: str = "val_loss", window: int = 10, threshold: float = 0.0001, mode: str = "min"):
        self.monitor = monitor
        self.window = window
        self.threshold = threshold
        self.mode = mode
        self.values: List[float] = []

    def on_epoch_end(self, trainer, epoch, metrics):
        if self.monitor not in metrics:
            return
        self.values.append(metrics[self.monitor])
        if len(self.values) < self.window:
            return
        recent = self.values[-self.window :]
        slope = np.polyfit(range(self.window), recent, 1)[0]
        improved = (slope < -self.threshold) if self.mode == "min" else (slope > self.threshold)
        if not improved:
            trainer.stop_training = True
            logger.info(
                f"SlopeEarlyStopping: slope={slope:.6f} {'<' if self.mode == 'min' else '>'} threshold={self.threshold}"
            )


class GradientModifierCallback(Callback):
    """Generic gradient modification hook.
    Stores a user function that is called before optimizer.step() with the list of gradients.
    The function receives (trainer, gradients) where gradients is a flattened list per param group.
    """

    def __init__(self, modify_fn: Callable[[Any, List[torch.Tensor]], None]):
        self.modify_fn = modify_fn

    def on_before_optimizer_step(self, trainer, gradients: List[torch.Tensor]):
        self.modify_fn(trainer, gradients)
