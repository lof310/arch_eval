"""Hyperparameter optimization utilities."""

import itertools
import logging
import random
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional

from arch_eval.core.config import TrainingConfig
from arch_eval.core.trainer import Trainer

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """Simple hyperparameter search (grid/random)."""

    def __init__(
        self,
        model_fn: Callable,
        base_config: TrainingConfig,
        param_grid: Dict[str, List[Any]],
        search_type: str = "grid",
        n_trials: Optional[int] = None,
        metric: str = "val_loss",
        mode: str = "min",
    ):
        self.model_fn = model_fn
        self.base_config = base_config
        self.param_grid = param_grid
        self.search_type = search_type
        self.n_trials = n_trials or (
            len(list(itertools.product(*param_grid.values()))) if search_type == "grid" else 10
        )
        self.metric = metric
        self.mode = mode
        self.results = []

    def _safe_deepcopy(self, obj):
        """Safely copy config, falling back to serialization-based copy if deepcopy fails."""
        try:
            return deepcopy(obj)
        except Exception as e:
            logger.warning(f"deepcopy failed ({e}), using serialization-based copy fallback")
            try:
                import pickle

                return pickle.loads(pickle.dumps(obj))
            except Exception as e2:
                raise RuntimeError(f"Failed to copy config: deepcopy and serialization both failed: {e2}")

    def run(self) -> Any:
        """Run hyperparameter search and return results as DataFrame."""
        # Lazy import pandas
        from arch_eval._lazy import lazy_import

        pd = lazy_import("pandas")

        if self.search_type == "grid":
            combinations = list(itertools.product(*self.param_grid.values()))
            keys = list(self.param_grid.keys())
            trials = [dict(zip(keys, combo)) for combo in combinations]
        else:
            trials = []
            for _ in range(self.n_trials):
                trial = {k: random.choice(v) for k, v in self.param_grid.items()}
                trials.append(trial)

        for i, params in enumerate(trials):
            config = self._safe_deepcopy(self.base_config)
            for k, v in params.items():
                if k in config.training_args:
                    config.training_args[k] = v
                else:
                    setattr(config, k, v)

            model = self.model_fn()
            try:
                trainer = Trainer(model, config)
                history = trainer.train()
                final_metric = history.get(self.metric, [None])[-1]
            except Exception as e:
                logger.error(f"Trial {i} failed with params {params}: {e}")
                final_metric = None
            self.results.append({**params, self.metric: final_metric})

        df = pd.DataFrame(self.results)
        # Filter out rows where metric is None (failed trials)
        valid_df = df[df[self.metric].notna()]
        if len(valid_df) == 0:
            logger.warning("All trials failed to produce a valid metric")
            return df
        best_idx = valid_df[self.metric].idxmin() if self.mode == "min" else valid_df[self.metric].idxmax()
        best = valid_df.loc[best_idx]
        print(f"Best {self.metric}: {best[self.metric]} with params: {best.drop(self.metric).to_dict()}")
        return df
