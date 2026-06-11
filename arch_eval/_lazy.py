"""Lazy import utilities to reduce import time.

This module provides lazy loading functionality for optional dependencies,
caching imported modules to avoid repeated import overhead.
"""

_cache = {}


def lazy_import(name):
    """Lazily import a module, caching it after first import.

    Args:
        name: The module name to import (e.g., 'pandas', 'wandb').

    Returns:
        The imported module.
    """
    if name not in _cache:
        _cache[name] = __import__(name)
    return _cache[name]
