"""Dynamic metric calculation based on task type, with confusion matrix support."""

import logging
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             explained_variance_score, f1_score, max_error,
                             mean_absolute_error, mean_squared_error,
                             median_absolute_error,
                             precision_recall_fscore_support, precision_score,
                             r2_score, recall_score, roc_auc_score,
                             top_k_accuracy_score)

from arch_eval.core.config import TaskType

logger = logging.getLogger(__name__)


class MetricCalculator:
    """Calculates metrics based on task type, with confusion matrix storage."""

    def __init__(self, task: Union[str, Any], device: str, output_transform: Optional[Callable] = None):
        self.task = task
        self.device = device
        self.output_transform = output_transform or (lambda x: x)
        self.history = {}
        # For confusion matrix accumulation
        self._all_preds = []
        self._all_targets = []
        self._conf_matrix = None

    def reset_confusion_matrix(self):
        """Reset accumulated predictions for confusion matrix."""
        self._all_preds.clear()
        self._all_targets.clear()
        self._conf_matrix = None

    def accumulate_confusion_matrix(self, outputs: Any, targets: torch.Tensor, max_samples: int = 10000):
        """Accumulate predictions for later confusion matrix computation (classification only)."""
        if isinstance(self.task, str) and self.task == TaskType.CLASSIFICATION:
            outputs = self.output_transform(outputs)

            # Extract predictions from various output formats
            if isinstance(outputs, torch.Tensor):
                preds = torch.argmax(outputs, dim=-1).cpu().numpy()
            elif isinstance(outputs, (tuple, list)):
                # Find the logits tensor in tuple/list
                for item in outputs:
                    if isinstance(item, torch.Tensor) and item.ndim > 1:
                        preds = torch.argmax(item, dim=-1).cpu().numpy()
                        break
                else:
                    return  # Could not find suitable tensor
            elif isinstance(outputs, dict):
                # Look for common keys
                for key in ["logits", "pred", "output", "predictions"]:
                    if key in outputs and isinstance(outputs[key], torch.Tensor):
                        preds = torch.argmax(outputs[key], dim=-1).cpu().numpy()
                        break
                else:
                    return  # Could not find suitable tensor
            elif hasattr(outputs, "logits") and isinstance(outputs.logits, torch.Tensor):
                # Handle Hugging Face style output objects (CausalLMOutput, SequenceClassifierOutput, etc.)
                preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            else:
                return

            targ = targets.cpu().numpy()
            # Limit memory usage by keeping only recent samples
            if len(self._all_preds) >= max_samples:
                half = max_samples // 2
                self._all_preds = self._all_preds[-half:]
                self._all_targets = self._all_targets[-half:]
            self._all_preds.extend(preds)
            self._all_targets.extend(targ)

    def compute_confusion_matrix(self, labels: Optional[List[str]] = None) -> Optional[np.ndarray]:
        """Compute confusion matrix from accumulated predictions."""
        if not self._all_preds:
            return None
        self._conf_matrix = confusion_matrix(self._all_targets, self._all_preds)
        return self._conf_matrix

    def calculate_batch_metrics(self, outputs: Any, targets: torch.Tensor, loss: float, split: str) -> Dict[str, float]:
        metrics = {f"{split}_loss": loss}
        outputs = self.output_transform(outputs)

        # Extract numpy arrays - handle various output formats
        if isinstance(outputs, torch.Tensor):
            out_np = outputs.detach().cpu().numpy()
        elif isinstance(outputs, (tuple, list)):
            # Try to find logits in tuple/list output
            # Common patterns: (logits,), (logits, loss), (loss, logits), (logits, aux_outputs)
            for item in outputs:
                if isinstance(item, torch.Tensor):
                    # Prefer tensors that look like logits (not scalar losses)
                    if item.numel() > 1 or item.ndim > 0:
                        out_np = item.detach().cpu().numpy()
                        break
            else:
                # Fallback to first element
                out_np = outputs[0].detach().cpu().numpy() if outputs[0] is not None else np.array([])
        elif isinstance(outputs, dict):
            # Look for common keys in transformer models
            for key in ["logits", "pred", "output", "predictions", "hidden_states"]:
                if key in outputs and isinstance(outputs[key], torch.Tensor):
                    out_np = outputs[key].detach().cpu().numpy()
                    break
            else:
                # Fall back to first tensor value
                for v in outputs.values():
                    if isinstance(v, torch.Tensor):
                        out_np = v.detach().cpu().numpy()
                        break
                else:
                    out_np = np.array([])
        elif hasattr(outputs, "logits") and isinstance(outputs.logits, torch.Tensor):
            # Handle Hugging Face style output objects (CausalLMOutput, SequenceClassifierOutput, etc.)
            out_np = outputs.logits.detach().cpu().numpy()
        else:
            out_np = np.array(outputs)

        targ_np = targets.detach().cpu().numpy()

        if isinstance(self.task, str):
            if self.task == TaskType.CLASSIFICATION:
                m = self._classification_metrics(out_np, targ_np)
            elif self.task == TaskType.REGRESSION:
                m = self._regression_metrics(out_np, targ_np)
            elif self.task == TaskType.NEXT_TOKEN_PREDICTION:
                m = self._language_metrics(out_np, targ_np, loss)
            else:
                m = {}
            metrics.update({f"{split}_{k}": v for k, v in m.items()})

        return metrics

    def _classification_metrics(self, out: np.ndarray, targ: np.ndarray) -> Dict[str, float]:
        if out.ndim == 2 and out.shape[1] > 1:
            # Apply softmax to logits for proper probability computation
            exp_out = np.exp(out - np.max(out, axis=1, keepdims=True))
            probs = exp_out / exp_out.sum(axis=1, keepdims=True)
            preds = np.argmax(out, axis=1)
            try:
                if out.shape[1] == 2:
                    auc = roc_auc_score(targ, probs[:, 1])
                else:
                    auc = roc_auc_score(targ, probs, multi_class="ovr")
            except Exception as e:
                logger.debug(f"AUC failed: {e}")
                auc = 0.5
            top5 = None
            # Only compute top-5 accuracy if we have more than 5 classes
            if out.shape[1] > 5:
                try:
                    top5 = top_k_accuracy_score(targ, probs, k=5, labels=list(range(out.shape[1])))
                except Exception:
                    pass
        else:
            preds = (out > 0.5).astype(int)
            try:
                auc = roc_auc_score(targ, out)
            except Exception:
                auc = 0.5
            top5 = None

        precision, recall, f1, _ = precision_recall_fscore_support(targ, preds, average="macro", zero_division=0)
        res = {
            "accuracy": accuracy_score(targ, preds),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc_roc": auc,
        }
        if top5 is not None:
            res["top5_accuracy"] = top5
        return res

    def _regression_metrics(self, out: np.ndarray, targ: np.ndarray) -> Dict[str, float]:
        return {
            "r2": r2_score(targ, out),
            "mse": mean_squared_error(targ, out),
            "mae": mean_absolute_error(targ, out),
            "explained_variance": explained_variance_score(targ, out),
            "max_error": max_error(targ, out),
            "median_absolute_error": median_absolute_error(targ, out),
        }

    def _language_metrics(self, out: np.ndarray, targ: np.ndarray, loss: float) -> Dict[str, float]:
        perplexity = np.exp(loss) if loss < 100 else float("inf")
        preds = np.argmax(out, axis=-1) if out.ndim > 1 else out
        if preds.ndim > 1:
            preds = preds.reshape(-1)
            targ_flat = targ.reshape(-1)
        else:
            targ_flat = targ
        mask = targ_flat != -100
        acc = accuracy_score(targ_flat[mask], preds[mask]) if mask.any() else 0.0
        return {"accuracy": acc, "perplexity": perplexity}

    def get_summary(self) -> Dict[str, float]:
        summ = {}
        for k, vals in self.history.items():
            if vals:
                summ[f"{k}_mean"] = np.mean(vals[-100:])
                summ[f"{k}_std"] = np.std(vals[-100:])
                summ[f"{k}_best"] = np.max(vals) if "loss" not in k else np.min(vals)
        return summ
