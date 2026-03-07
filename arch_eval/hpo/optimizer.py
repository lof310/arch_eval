"""Hyperparameter optimization utilities."""

import itertools
import random
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from arch_eval.core.config import TrainingConfig
from arch_eval.core.trainer import Trainer


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

    def run(self) -> pd.DataFrame:
        if self.search_type == "grid":
            combinations = list(itertools.product(*self.param_grid.values()))
            keys = list(self.param_grid.keys())
            trials = [dict(zip(keys, combo)) for combo in combinations]
        else:  # random
            trials = []
            for _ in range(self.n_trials):
                trial = {k: random.choice(v) for k, v in self.param_grid.items()}
                trials.append(trial)

        for i, params in enumerate(trials):
            config = deepcopy(self.base_config)
            # Update training_args or top-level attributes
            for k, v in params.items():
                if k in config.training_args:
                    config.training_args[k] = v
                else:
                    setattr(config, k, v)

            model = self.model_fn()
            trainer = Trainer(model, config)
            history = trainer.train()
            final_metric = history.get(self.metric, [None])[-1]
            self.results.append({**params, self.metric: final_metric})

        df = pd.DataFrame(self.results)
        best_idx = df[self.metric].idxmin() if self.mode == "min" else df[self.metric].idxmax()
        best = df.loc[best_idx]
        print(f"Best {self.metric}: {best[self.metric]} with params: {best.drop(self.metric).to_dict()}")
        return df
