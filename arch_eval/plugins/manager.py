"""Plugin management system with improved error handling and per-trainer hooks."""

import importlib
import inspect
import logging
import os
import pkgutil
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from arch_eval.core.exceptions import PluginError, StopTraining

logger = logging.getLogger(__name__)


@dataclass
class HookSpec:
    name: str
    description: str
    args: List[str]
    returns: Optional[str]


class PluginManager:
    """Manages plugin discovery, loading, and hook execution."""

    HOOKS = {
        "before_training": HookSpec("before_training", "Called before training starts", ["trainer", "config"], None),
        "after_batch": HookSpec(
            "after_batch", "Called after each batch", ["trainer", "batch_idx", "outputs", "loss"], "dict"
        ),
        "after_epoch": HookSpec("after_epoch", "Called after each epoch", ["trainer", "epoch", "metrics"], "dict"),
        "before_eval": HookSpec("before_eval", "Called before evaluation", ["trainer", "split"], None),
        "after_training": HookSpec(
            "after_training", "Called after training completes", ["trainer", "final_metrics"], None
        ),
        "on_log": HookSpec("on_log", "Called during logging", ["trainer", "metrics", "step"], None),
        "on_exception": HookSpec("on_exception", "Called when exception occurs", ["trainer", "exception"], None),
        "on_checkpoint": HookSpec(
            "on_checkpoint", "Called when saving a checkpoint", ["trainer", "checkpoint_path", "is_best"], None
        ),
        "on_train_start": HookSpec("on_train_start", "Called at the very beginning of training", ["trainer"], None),
        "on_train_end": HookSpec("on_train_end", "Called at the very end of training", ["trainer"], None),
        "on_epoch_start": HookSpec("on_epoch_start", "Called at the start of each epoch", ["trainer", "epoch"], None),
        "on_epoch_end": HookSpec(
            "on_epoch_end", "Called at the end of each epoch", ["trainer", "epoch", "metrics"], None
        ),
        "on_batch_start": HookSpec(
            "on_batch_start", "Called before processing a batch", ["trainer", "batch_idx", "data", "targets"], None
        ),
        "on_batch_end": HookSpec(
            "on_batch_end", "Called after processing a batch", ["trainer", "batch_idx", "loss"], None
        ),
        "on_validation_start": HookSpec("on_validation_start", "Called before validation loop", ["trainer"], None),
        "on_validation_end": HookSpec(
            "on_validation_end", "Called after validation loop", ["trainer", "metrics"], None
        ),
        "on_backward": HookSpec("on_backward", "Called after loss.backward()", ["trainer", "loss"], None),
        "on_before_optimizer_step": HookSpec(
            "on_before_optimizer_step", "Called before optimizer.step()", ["trainer", "gradients"], None
        ),
        "on_optimizer_step": HookSpec("on_optimizer_step", "Called after optimizer step", ["trainer"], None),
    }

    def __init__(self, local_hooks: Optional[Dict[str, List[Callable]]] = None):
        self.global_plugins = {}
        self.global_hooks = {name: [] for name in self.HOOKS}
        self.local_hooks = {name: (local_hooks or {}).get(name, []) for name in self.HOOKS}

    def discover_plugins(self, plugin_paths: Optional[List[str]] = None):
        """Discover plugins from specified paths or built-in arch_eval.plugins only.

        Args:
            plugin_paths: List of directories to scan for plugins. If empty or None,
                only scans the built-in arch_eval.plugins subpackage.
        """
        original_path = sys.path.copy()  # Save original sys.path
        if plugin_paths:
            for p in plugin_paths:
                if p not in sys.path:
                    sys.path.insert(0, p)
        # Only scan explicitly provided paths or built-in arch_eval.plugins
        modules_to_scan = []
        if plugin_paths:
            # Scan only user-provided paths
            for path in plugin_paths:
                try:
                    for finder, name, ispkg in pkgutil.iter_modules([path]):
                        if name.startswith("arch_eval_plugin_") or name.endswith("_plugin"):
                            modules_to_scan.append((finder, name, ispkg))
                except Exception as e:
                    logger.warning(f"Failed to scan path {path}: {e}")
        else:
            # Only scan built-in arch_eval.plugins
            try:
                import arch_eval.plugins as plugins_pkg

                plugins_path = os.path.dirname(plugins_pkg.__file__)
                for finder, name, ispkg in pkgutil.iter_modules([plugins_path]):
                    if name.startswith("arch_eval_plugin_") or name.endswith("_plugin"):
                        modules_to_scan.append((finder, name, ispkg))
            except Exception as e:
                logger.warning(f"Failed to scan built-in plugins: {e}")
        for finder, name, ispkg in modules_to_scan:
            try:
                module = importlib.import_module(name)
                self._load_plugin_from_module(module)
            except Exception as e:
                logger.warning(f"Failed to load plugin {name}: {e}")
        # Restore original sys.path to avoid import conflicts
        sys.path[:] = original_path
        logger.debug(f"Discovered {len(self.global_plugins)} global plugins")

    def _load_plugin_from_module(self, module):
        plugin_name = getattr(module, "__plugin_name__", module.__name__)
        version = getattr(module, "__version__", "unknown")
        hooks = {}
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) and hasattr(obj, "_hook_name"):
                hook_name = obj._hook_name
                if hook_name in self.HOOKS:
                    hooks[hook_name] = obj
        if hooks:
            self.global_plugins[plugin_name] = {
                "name": plugin_name,
                "version": version,
                "module": module,
                "hooks": hooks,
            }
            for hname, func in hooks.items():
                self.global_hooks[hname].append(func)
            logger.debug(f"Loaded global plugin: {plugin_name} v{version}")

    def register_local_hook(self, hook_name: str, func: Callable):
        if hook_name not in self.HOOKS:
            raise PluginError(f"Unknown hook: {hook_name}")
        self.local_hooks[hook_name].append(func)

    def execute_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        if hook_name not in self.HOOKS:
            raise PluginError(f"Unknown hook: {hook_name}")
        results = []
        for func in self.global_hooks[hook_name] + self.local_hooks[hook_name]:
            try:
                res = func(*args, **kwargs)
                if res is not None:
                    results.append(res)
            except StopTraining:
                logger.debug(f"Plugin {func.__name__} requested training stop")
                raise
            except Exception as e:
                logger.error(f"Plugin error in {hook_name} ({func.__name__}): {e}\n{traceback.format_exc()}")
        return results

    def get_plugins(self) -> Dict[str, Any]:
        return {
            name: {"version": p["version"], "hooks": list(p["hooks"].keys())} for name, p in self.global_plugins.items()
        }


def hook(hook_name: str):
    """Decorator to mark a function as a plugin hook."""

    def deco(func):
        func._hook_name = hook_name
        return func

    return deco
