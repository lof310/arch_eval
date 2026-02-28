"""Configurable logging setup."""

import logging
import sys
from typing import Optional
from logging.handlers import RotatingFileHandler

def setup_logging(level: str = "INFO", log_file: Optional[str] = None, fmt: Optional[str] = None) -> logging.Logger:
    fmt = fmt or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    numeric = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric)
    for h in root.handlers[:]:
        root.removeHandler(h)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(numeric)
    console.setFormatter(logging.Formatter(fmt))
    root.addHandler(console)

    if log_file:
        file = RotatingFileHandler(log_file, maxBytes=10_485_760, backupCount=5)
        file.setLevel(numeric)
        file.setFormatter(logging.Formatter(fmt))
        root.addHandler(file)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    return root


class LoggerAdapter:
    """Adapter for consistent logging across the library."""
    def __init__(self, name: str):
        self.logger = logging.getLogger(f"arch_eval.{name}")

    def debug(self, msg, *a, **kw): self.logger.debug(msg, *a, **kw)
    def info(self, msg, *a, **kw): self.logger.info(msg, *a, **kw)
    def warning(self, msg, *a, **kw): self.logger.warning(msg, *a, **kw)
    def error(self, msg, *a, **kw): self.logger.error(msg, *a, **kw)
    def critical(self, msg, *a, **kw): self.logger.critical(msg, *a, **kw)
    def metric(self, msg, **kw): self.logger.info(f"METRIC: {msg}", extra=kw)
