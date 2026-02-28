"""Main Trainer class for single model training."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List, Tuple, Union
import time
import numpy as np
import os
import wandb
import logging
from contextlib import AbstractContextManager
from collections import defaultdict
from arch_eval.core.config import TrainingConfig, DistributedBackend, MixedPrecisionDtype
from arch_eval.core.exceptions import ModelError, ConfigurationError, StopTraining, DistributedError
from arch_eval.data.data import DatasetHandler
from arch_eval.metrics.calculator import MetricCalculator
from arch_eval.viz.viz import RealtimeWindow, VideoRecorder, PlotSaver
from arch_eval.logging.logger_config import LoggerAdapter
from arch_eval.plugins.manager import PluginManager, hook
from arch_eval.utils.device import memory_summary
from arch_eval.distributed import init_distributed, cleanup_distributed, get_wrapped_model
from arch_eval.profiler import profiler_context

logger = logging.getLogger(__name__)

class NullContext(AbstractContextManager):
    """Context manager that does nothing, compatible with autocast parameters."""
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

class Trainer:
    """Main training class for single model."""

    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config
        self.logger = LoggerAdapter("trainer")

        # Distributed setup
        if config.distributed_backend != DistributedBackend.NONE:
            init_distributed(
                backend="nccl",
                world_size=config.distributed_world_size,
                rank=config.distributed_rank,
                master_addr=config.distributed_master_addr,
                master_port=config.distributed_master_port
            )
            self.model = get_wrapped_model(model, config)
        else:
            self.model = model

        self._validate_model()
        self.device = torch.device(config.device)
        self.model = self.model.to(self.device).to(config.dtype)

        self.dataset_handler = DatasetHandler(config)
        self.train_loader, self.val_loader, self.test_loader = self.dataset_handler.prepare_loaders()

        self.metric_calculator = MetricCalculator(
            config.task, config.device, output_transform=config.model_output_transform
        )

        self._setup_optimizers()
        self._setup_schedulers()
        self._setup_loss_function()

        # Mixed precision
        self.use_amp = config.mixed_precision and config.device == "cuda"
        self.amp_dtype = self._get_amp_dtype()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp and config.grad_scaler else None

        # Gradient checkpointing
        if config.gradient_checkpointing:
            self._apply_gradient_checkpointing()

        # Visualization
        self.window = None
        if config.realtime:
            try:
                self.window = RealtimeWindow(config)
                if getattr(self.window, 'disabled', False):
                    self.window = None
            except Exception as e:
                self.logger.warning(f"Failed to initialize realtime window: {e}")
                self.window = None
        self.video_recorder = VideoRecorder(config, config.save_video) if config.save_video else None
        self.plot_saver = None

        # External loggers
        if config.log_to_wandb:
            wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=config.training_args)
        if config.log_to_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(config.tensorboard_dir)
            except ImportError:
                logger.warning("TensorBoard not installed.")
                self.tb_writer = None

        self.plugin_manager = PluginManager()
        for cb in config.callbacks:
            if hasattr(cb, "register_hooks"):
                cb.register_hooks(self.plugin_manager)

        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float("inf") if config.early_stopping_mode == "min" else -float("inf")
        self.patience_counter = 0
        self.history = []
        self.stop_training = False

        if config.checkpoint_dir:
            os.makedirs(config.checkpoint_dir, exist_ok=True)

        self.accumulation_steps = config.gradient_accumulation_steps
        self.current_accum_step = 0

        self.logger.info(f"Trainer initialized on {self.device}\n{memory_summary()}")

    def _get_amp_dtype(self):
        if not self.use_amp:
            return None
        if self.config.mixed_precision_dtype == MixedPrecisionDtype.FLOAT16:
            return torch.float16
        elif self.config.mixed_precision_dtype == MixedPrecisionDtype.BFLOAT16:
            return torch.bfloat16
        elif self.config.mixed_precision_dtype == MixedPrecisionDtype.FP8:
            return torch.float8_e4m3fn if hasattr(torch, 'float8_e4m3fn') else torch.float16
        else:
            return torch.float16

    def _apply_gradient_checkpointing(self):
        """Aplica checkpointing a los módulos especificados."""
        if self.config.gradient_checkpointing_modules:
            for name in self.config.gradient_checkpointing_modules:
                module = dict(self.model.named_modules()).get(name)
                if module:
                    module.apply(lambda m: setattr(m, 'gradient_checkpointing', True))
        else:
            # Intenta aplicar a cualquier módulo que tenga el atributo
            for module in self.model.modules():
                if hasattr(module, 'gradient_checkpointing'):
                    module.gradient_checkpointing = True

    def _validate_model(self):
        shape = self.config.input_shape or (1, 10)
        dummy = torch.randn(1, *shape).to(torch.device(self.config.device))
        try:
            with torch.no_grad():
                self.model(dummy)
        except Exception as e:
            raise ModelError(f"Model validation failed: {e}")

    def _setup_optimizers(self):
        self.optimizers = []
        for opt_cfg in self.config.optimizers:
            opt_cfg = opt_cfg.copy()
            opt_type = opt_cfg.pop("type", "adam").lower()
            if opt_type == "adam":
                opt = torch.optim.Adam(self.model.parameters(), **opt_cfg)
            elif opt_type == "sgd":
                opt = torch.optim.SGD(self.model.parameters(), **opt_cfg)
            elif opt_type == "adamw":
                opt = torch.optim.AdamW(self.model.parameters(), **opt_cfg)
            else:
                raise ConfigurationError(f"Unsupported optimizer: {opt_type}")
            self.optimizers.append(opt)
        if not self.optimizers:
            lr = self.config.training_args.get("learning_rate", 0.001)
            self.optimizers = [torch.optim.Adam(self.model.parameters(), lr=lr)]

    def _setup_schedulers(self):
        self.schedulers = []
        for sch_cfg in self.config.schedulers:
            sch_cfg = sch_cfg.copy()
            sch_type = sch_cfg.pop("type").lower()
            opt_idx = sch_cfg.pop("optimizer", 0)
            if opt_idx >= len(self.optimizers):
                raise ConfigurationError(f"Optimizer index {opt_idx} out of range")
            opt = self.optimizers[opt_idx]

            if sch_type == "step":
                sch = torch.optim.lr_scheduler.StepLR(
                    opt, step_size=sch_cfg.pop("step_size", 30), gamma=sch_cfg.pop("gamma", 0.1), **sch_cfg
                )
            elif sch_type == "cosine":
                T_max = sch_cfg.pop("T_max", self.config.training_args.get("num_epochs", 10))
                sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max, **sch_cfg)
            elif sch_type == "reduce_on_plateau":
                sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, mode=sch_cfg.pop("mode", "min"), factor=sch_cfg.pop("factor", 0.1),
                    patience=sch_cfg.pop("patience", 10), **sch_cfg
                )
            else:
                raise ConfigurationError(f"Unsupported scheduler: {sch_type}")
            self.schedulers.append({
                "scheduler": sch,
                "interval": sch_cfg.get("interval", self.config.scheduler_interval),
                "monitor": sch_cfg.get("monitor", "val_loss") if sch_type == "reduce_on_plateau" else None,
            })

    def _setup_loss_function(self):
        if self.config.loss_function:
            self.criterion = self.config.loss_function
            return
        task = self.config.task
        if isinstance(task, str):
            if task == "classification":
                self.criterion = nn.CrossEntropyLoss()
            elif task == "regression":
                self.criterion = nn.MSELoss()
            elif task == "next-token-prediction":
                self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
            else:
                self.criterion = nn.MSELoss()
        else:
            self.criterion = getattr(task, "loss_function", nn.MSELoss())

    def _compute_loss(self, output, targets):
        if isinstance(output, tuple) and len(output) == 2:
            outputs, loss = output
            return loss
        else:
            return self.criterion(output, targets)

    def train(self) -> Dict[str, List[float]]:
        self.logger.info("Starting training")
        try:
            self.plugin_manager.execute_hook("on_train_start", self)
            self.plugin_manager.execute_hook("before_training", self, self.config)

            num_epochs = self.config.training_args.get("num_epochs", 10)

            with profiler_context(self.config) as prof:
                for epoch in range(num_epochs):
                    if self.stop_training:
                        break
                    self.current_epoch = epoch
                    self.plugin_manager.execute_hook("on_epoch_start", self, epoch)

                    train_metrics = self._train_epoch()
                    val_metrics = self._validate_epoch() if self.val_loader else {}
                    epoch_metrics = {**train_metrics, **val_metrics}
                    self.history.append(epoch_metrics)

                    self._step_schedulers(epoch_metrics, interval="epoch")

                    if epoch % self.config.log_interval == 0:
                        self._log_metrics(epoch_metrics, epoch)

                    plugin_results = self.plugin_manager.execute_hook("on_epoch_end", self, epoch, epoch_metrics)
                    for r in plugin_results:
                        if isinstance(r, dict):
                            epoch_metrics.update(r)

                    if self.config.early_stopping_patience and self._check_early_stopping(epoch_metrics):
                        self.logger.info(f"Early stopping triggered at epoch {epoch}")
                        break

                    self._save_checkpoint(epoch, epoch_metrics)

                    if prof:
                        prof.step()

            if self.config.eval_on_test and self.test_loader:
                test_metrics = self._evaluate(self.test_loader, "test")
                self.logger.info(f"Test metrics: {test_metrics}")
                self.history.append(test_metrics)

            self.plugin_manager.execute_hook("after_training", self, self.history)
            self.plugin_manager.execute_hook("on_train_end", self)

        except StopTraining:
            self.logger.info("Training stopped by plugin")
        except Exception as e:
            self.plugin_manager.execute_hook("on_exception", self, e)
            raise
        finally:
            if self.config.save_plot:
                self.plot_saver = PlotSaver(self.config, self._get_history_dict())
                self.plot_saver.save_plots()
            if self.video_recorder:
                self.video_recorder.save_video("training_video")
            if self.window:
                self.window.close()
            if self.config.log_to_wandb:
                wandb.finish()
            if hasattr(self, "tb_writer") and self.tb_writer:
                self.tb_writer.close()
            if self.config.distributed_backend != DistributedBackend.NONE:
                cleanup_distributed()

        self.logger.info("Training completed")
        return self._get_history_dict()

    def _train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        metric_accum = defaultdict(float)
        count = 0
        autocast = torch.cuda.amp.autocast if self.use_amp else NullContext
        self.current_accum_step = 0

        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            self.plugin_manager.execute_hook("on_batch_start", self, batch_idx, data, targets)

            if self.use_amp and self.amp_dtype is not None:
                with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                    output = self.model(data)
                    loss = self._compute_loss(output, targets)
                    loss = loss / self.accumulation_steps
            else:
                output = self.model(data)
                loss = self._compute_loss(output, targets)
                loss = loss / self.accumulation_steps

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            self.current_accum_step += 1

            if self.current_accum_step % self.accumulation_steps == 0:
                if self.config.gradient_clip:
                    if self.scaler:
                        for opt in self.optimizers:
                            self.scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)

                if self.scaler:
                    for opt in self.optimizers:
                        self.scaler.step(opt)
                    self.scaler.update()
                else:
                    for opt in self.optimizers:
                        opt.step()

                for opt in self.optimizers:
                    opt.zero_grad()

                self.plugin_manager.execute_hook("on_optimizer_step", self)

            # Actualizar métricas sin almacenar todos los outputs
            with torch.no_grad():
                batch_metrics = self.metric_calculator.calculate_batch_metrics(
                    output, targets, loss.item() * self.accumulation_steps, "train"
                )
                for k, v in batch_metrics.items():
                    metric_accum[k] += v
                count += 1

            total_loss += loss.item() * self.accumulation_steps
            self.global_step += 1
            self._step_schedulers({"train_loss": loss.item() * self.accumulation_steps}, interval="step")

            if self.global_step % self.config.viz_interval == 0:
                avg_metrics = {k: v/count for k, v in metric_accum.items()}
                if self.window:
                    self.window.update(avg_metrics)
                if self.video_recorder:
                    self.video_recorder.record_step(self.global_step, avg_metrics)

            self.plugin_manager.execute_hook("on_batch_end", self, batch_idx, loss.item() * self.accumulation_steps)

            del output, loss, data, targets

        metrics = {k: v / count for k, v in metric_accum.items()}
        return metrics

    def _validate_epoch(self) -> Dict[str, float]:
        return self._evaluate(self.val_loader, "val")

    def _evaluate(self, loader: DataLoader, split: str) -> Dict[str, float]:
        if not loader:
            return {}
        self.model.eval()
        total_loss = 0.0
        metric_accum = defaultdict(float)
        count = 0
        autocast = torch.cuda.amp.autocast if self.use_amp else NullContext

        self.plugin_manager.execute_hook("before_eval", self, split)
        self.plugin_manager.execute_hook("on_validation_start", self)

        with torch.no_grad():
            for data, targets in loader:
                data, targets = data.to(self.device), targets.to(self.device)
                if self.use_amp and self.amp_dtype is not None:
                    with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                        output = self.model(data)
                        loss = self._compute_loss(output, targets)
                else:
                    output = self.model(data)
                    loss = self._compute_loss(output, targets)
                total_loss += loss.item()
                batch_metrics = self.metric_calculator.calculate_batch_metrics(output, targets, loss.item(), split)
                for k, v in batch_metrics.items():
                    metric_accum[k] += v
                count += 1
                del output, loss, data, targets

        metrics = {k: v / count for k, v in metric_accum.items()}
        self.plugin_manager.execute_hook("on_validation_end", self, metrics)

        if self.window:
            self.window.update(metrics)
        return metrics

    def _step_schedulers(self, metrics: Dict[str, float], interval: str):
        for sch in self.schedulers:
            if sch["interval"] != interval:
                continue
            sched = sch["scheduler"]
            if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if sch["monitor"] in metrics:
                    sched.step(metrics[sch["monitor"]])
            else:
                sched.step()

    def _log_metrics(self, metrics: Dict[str, float], step: int):
        if self.config.log_all_metrics:
            log_msg = f"Epoch {step}: " + " - ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        else:
            log_msg = f"Epoch {step}: " + " ".join(f"{k}: {v:.4f}" for k, v in metrics.items() if "loss" in k or "accuracy" in k)
        self.logger.info(log_msg)

        if self.config.log_to_wandb:
            wandb.log(metrics, step=step)
        if hasattr(self, "tb_writer") and self.tb_writer:
            for k, v in metrics.items():
                self.tb_writer.add_scalar(k, v, step)

        self.plugin_manager.execute_hook("on_log", self, metrics, step)

    def _check_early_stopping(self, metrics: Dict[str, float]) -> bool:
        if self.config.early_stopping_metric not in metrics:
            return False
        current = metrics[self.config.early_stopping_metric]
        mode = self.config.early_stopping_mode
        improved = (mode == "min" and current < self.best_metric) or (mode == "max" and current > self.best_metric)
        if improved:
            self.best_metric = current
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        return self.patience_counter >= self.config.early_stopping_patience

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        if not self.config.checkpoint_dir:
            return
        save_this, is_best = False, False
        if self.config.save_best_only:
            current = metrics.get(self.config.checkpoint_metric)
            if current is not None:
                mode = getattr(self.config, "early_stopping_mode", "min" if "loss" in self.config.checkpoint_metric else "max")
                improved = (mode == "min" and current < self.best_metric) or (mode == "max" and current > self.best_metric)
                if improved:
                    self.best_metric = current
                    is_best = True
                    save_this = True
        else:
            if epoch % self.config.save_frequency == 0:
                save_this = True
        if not save_this:
            return

        serializable_config = {}
        for k, v in self.config.__dict__.items():
            if k.startswith('_'):
                continue
            try:
                import pickle
                pickle.dumps(v)
                serializable_config[k] = v
            except (TypeError, pickle.PickleError):
                if hasattr(v, '__name__'):
                    serializable_config[k] = v.__name__
                elif hasattr(v, '__class__'):
                    serializable_config[k] = str(v.__class__)
                else:
                    serializable_config[k] = str(v)

        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dicts": [opt.state_dict() for opt in self.optimizers],
            "scheduler_state_dicts": [sch["scheduler"].state_dict() for sch in self.schedulers],
            "config": serializable_config,
            "metrics": metrics,
            "best_metric": self.best_metric,
        }
        if self.scaler:
            ckpt["scaler_state_dict"] = self.scaler.state_dict()

        path = os.path.join(self.config.checkpoint_dir, "best_model.pt" if is_best else f"checkpoint_epoch_{epoch}.pt")
        tmp = path + ".tmp"
        torch.save(ckpt, tmp)
        os.replace(tmp, path)
        self.logger.info(f"Checkpoint saved to {path}")
        self.plugin_manager.execute_hook("on_checkpoint", self, path, is_best)

    def load_checkpoint(self, path: str, load_optimizer: bool = True, load_scheduler: bool = True):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if load_optimizer:
            for opt, state in zip(self.optimizers, ckpt["optimizer_state_dicts"]):
                opt.load_state_dict(state)
        if load_scheduler:
            for sch, state in zip(self.schedulers, ckpt["scheduler_state_dicts"]):
                sch["scheduler"].load_state_dict(state)
        if "scaler_state_dict" in ckpt and self.scaler:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.current_epoch = ckpt.get("epoch", 0)
        self.best_metric = ckpt.get("best_metric", self.best_metric)
        self.logger.info(f"Checkpoint loaded from {path}")

    def _get_history_dict(self) -> Dict[str, List[float]]:
        hist = {}
        for m in self.history:
            for k, v in m.items():
                hist.setdefault(k, []).append(v)
        return hist
