"""Benchmark class for comparing multiple models."""

import copy
import gc
import logging
import multiprocessing as mp
import os
import pickle
import warnings
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)
from typing import Any, Dict, List

import pandas as pd
import torch.nn as nn

from arch_eval.core.config import BenchmarkConfig, TrainingConfig
from arch_eval.core.trainer import Trainer
from arch_eval.logging.logger_config import LoggerAdapter

logger = logging.getLogger(__name__)


def _try_import_cloudpickle():
    """Try to import cloudpickle. Returns the module or None."""
    try:
        import cloudpickle

        return cloudpickle
    except ImportError:
        return None


def _check_model_pickle_integrity(model: nn.Module, model_name: str) -> bool:
    """
    Check if a model can be safely serialized and deserialized.
    Uses cloudpickle for robust serialization. Returns True if the model
    passes the integrity check, False otherwise.
    """
    cloudpickle = _try_import_cloudpickle()
    if cloudpickle is None:
        logger.warning(f"cloudpickle not available, cannot verify integrity of {model_name}")
        return False
    try:
        # Serialize to bytes buffer
        import io

        buffer = io.BytesIO()
        cloudpickle.dump(model, buffer)
        buffer.seek(0)
        # Deserialize back
        restored_model = cloudpickle.load(buffer)
        # Compare state dict keys
        original_keys = set(model.state_dict().keys())
        restored_keys = set(restored_model.state_dict().keys())
        if original_keys != restored_keys:
            logger.error(f"Model {model_name} integrity check failed: state dict keys mismatch")
            logger.error(f"  Original keys: {sorted(original_keys)}")
            logger.error(f"  Restored keys: {sorted(restored_keys)}")
            return False
        return True
    except Exception as e:
        logger.error(f"Model {model_name} integrity check failed: {e}")
        return False


def _train_single_process(args):
    """Helper for process-based parallelism with memory cleanup."""
    model_info, config = args
    name, model = model_info["name"], model_info["model"]
    try:
        trainer_config = TrainingConfig(
            realtime=config.realtime,
            save_video=config.save_video,
            save_plot=config.save_plot,
            dtype=config.dtype,
            device=config.device,
            dataset=copy.deepcopy(config.dataset),
            dataset_params=config.dataset_params.copy(),
            transform=config.transform,
            target_transform=config.target_transform,
            training_args=config.training_args.copy(),
            task=config.task,
            viz_interval=config.viz_interval,
            log_interval=config.log_interval,
            eval_interval=config.eval_interval,
            log_to_wandb=config.log_to_wandb,
            wandb_project=config.wandb_project,
            seed=config.seed,
            dataloader_params=config.dataloader_params.copy(),
        )
        trainer = Trainer(model, trainer_config)
        history = trainer.train()
        final = {}
        for metric in config.compare_metrics:
            if metric in history:
                final[metric] = history[metric][-1] if history.get(metric) else None
            elif f"train_{metric}" in history:
                final[metric] = history[f"train_{metric}"][-1] if history.get(f"train_{metric}") else None
            elif f"val_{metric}" in history:
                final[metric] = history[f"val_{metric}"][-1] if history.get(f"val_{metric}") else None
        return {"model_name": name, **final}
    except Exception as e:
        return {"model_name": name, "error": str(e)}
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class Benchmark:
    """Benchmark multiple models for comparison."""

    def __init__(self, models: List[Dict[str, Any]], config: BenchmarkConfig):
        self.models = models
        self.config = config
        self.logger = LoggerAdapter("benchmark")
        self._validate_models()
        self.results = []
        self.logger.info(f"Benchmark initialized with {len(models)} models")

    def _validate_models(self):
        for i, d in enumerate(self.models):
            if "name" not in d:
                d["name"] = f"Model_{i}"
            if "model" not in d:
                raise ValueError(f"Model {i} missing 'model' key")
            if not isinstance(d["model"], nn.Module):
                raise ValueError(f"Model {i} must be a PyTorch nn.Module")

    def run(self) -> pd.DataFrame:
        self.logger.info("Starting benchmark")
        if self.config.parallel and len(self.models) > 1:
            if self.config.use_processes:
                results = self._run_parallel_process()
            else:
                results = self._run_parallel_thread()
        else:
            results = self._run_sequential()
        df = pd.DataFrame(results)
        self._log_results(df)
        return df

    def _run_sequential(self):
        return [self._train_single(m) for m in self.models]

    def _run_parallel_thread(self):
        results = []
        max_workers = self.config.max_workers or len(self.models)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut = {ex.submit(self._train_single, m): m["name"] for m in self.models}
            for f in as_completed(fut):
                try:
                    results.append(f.result())
                except Exception as e:
                    self.logger.error(f"Model {fut[f]} failed: {e}")
                    results.append({"model_name": fut[f], "error": str(e)})
        return results

    def _run_parallel_process(self):
        ctx = mp.get_context("spawn")
        max_workers = self.config.max_workers or len(self.models)
        # Mandatory integrity check before launching processes
        failed_models = []
        for m in self.models:
            model_name = m["name"]
            model = m["model"]
            if not _check_model_pickle_integrity(model, model_name):
                failed_models.append(model_name)
        if failed_models:
            self.logger.error(f"Model integrity check failed for: {failed_models}")
            self.logger.warning("Falling back to sequential execution due to model serialization issues.")
            return self._run_sequential()
        args_list = [(copy.deepcopy(m), copy.deepcopy(self.config)) for m in self.models]
        results = []
        # Try with ProcessPoolExecutor, fallback to sequential if pickling fails
        try:
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
                futures = [ex.submit(_train_single_process, a) for a in args_list]
                for i, f in enumerate(as_completed(futures)):
                    try:
                        result = f.result(timeout=self.config.timeout_seconds)
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Model {self.models[i]['name']} failed: {e}")
                        error_result = {"model_name": self.models[i]["name"], "error": str(e)}
                        results.append(error_result)
                        if self.config.retry_failed:
                            self.logger.info(f"Retrying {self.models[i]['name']}...")
                            retry_future = ex.submit(_train_single_process, args_list[i])
                            try:
                                result = retry_future.result(timeout=self.config.timeout_seconds)
                                # Replace error result with successful result
                                results[-1] = result
                            except Exception as e2:
                                self.logger.error(f"Retry failed: {e2}")
                                results[-1] = {"model_name": self.models[i]["name"], "error": str(e2)}
        except (pickle.PicklingError, AttributeError) as e:
            self.logger.error(f"Pickling error in process-based parallelism: {e}")
            self.logger.warning("Falling back to sequential execution. Install 'cloudpickle' for better serialization.")
            results = self._run_sequential()
        return results

    def _train_single(self, model_info):
        name, model = model_info["name"], model_info["model"]
        self.logger.info(f"Training model: {name}")
        try:
            trainer_config = TrainingConfig(
                realtime=self.config.realtime,
                save_video=self.config.save_video,
                save_plot=self.config.save_plot,
                dtype=self.config.dtype,
                device=self.config.device,
                dataset=copy.deepcopy(self.config.dataset),
                dataset_params=self.config.dataset_params.copy(),
                transform=self.config.transform,
                target_transform=self.config.target_transform,
                training_args=self.config.training_args.copy(),
                task=self.config.task,
                viz_interval=self.config.viz_interval,
                log_interval=self.config.log_interval,
                eval_interval=self.config.eval_interval,
                log_to_wandb=self.config.log_to_wandb,
                wandb_project=self.config.wandb_project,
                seed=self.config.seed,
                dataloader_params=self.config.dataloader_params.copy(),
            )
            trainer = Trainer(model, trainer_config)
            history = trainer.train()
            final = {}
            for metric in self.config.compare_metrics:
                if metric in history:
                    final[metric] = history[metric][-1] if history.get(metric) else None
                elif f"train_{metric}" in history:
                    final[metric] = history[f"train_{metric}"][-1] if history.get(f"train_{metric}") else None
                elif f"val_{metric}" in history:
                    final[metric] = history[f"val_{metric}"][-1] if history.get(f"val_{metric}") else None
            return {"model_name": name, **final}
        except Exception as e:
            self.logger.error(f"Model {name} failed: {e}")
            return {"model_name": name, "error": str(e)}

    def _log_results(self, df: pd.DataFrame):
        self.logger.info("Benchmark Results:")
        for _, row in df.iterrows():
            parts = [f"{col}: {row[col]:.4f}" for col in df.columns if col != "model_name" and pd.notna(row[col])]
            self.logger.info(f"  {row['model_name']}: " + " - ".join(parts))
