"""Configurable logging setup."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional


def setup_logging(level: str = "INFO", log_file: Optional[str] = None, fmt: Optional[str] = None, force: bool = False) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to a log file for rotating file handler.
        fmt: Optional custom format string for log messages.
        force: If True, removes all existing handlers. If False, only adds handlers if none exist.
    
    Returns:
        The root logger configured with the specified settings.
    """
    fmt = fmt or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    numeric = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric)
    
    # Only remove existing handlers if force=True
    if force:
        for h in root.handlers[:]:
            root.removeHandler(h)
    elif root.handlers:
        # Handlers already exist, check if we need to add our handlers
        has_console = any(isinstance(h, logging.StreamHandler) and h.stream == sys.stdout for h in root.handlers)
        has_file = log_file and any(
            isinstance(h, RotatingFileHandler) and h.baseFilename == os.path.abspath(log_file) 
            for h in root.handlers
        )
        if has_console and (not log_file or has_file):
            # Already have the handlers we would add, just set level and return
            logging.getLogger("matplotlib").setLevel(logging.WARNING)
            logging.getLogger("PIL").setLevel(logging.WARNING)
            return root
    
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

    def debug(self, msg, *a, **kw):
        self.logger.debug(msg, *a, **kw)

    def info(self, msg, *a, **kw):
        self.logger.info(msg, *a, **kw)

    def warning(self, msg, *a, **kw):
        self.logger.warning(msg, *a, **kw)

    def error(self, msg, *a, **kw):
        self.logger.error(msg, *a, **kw)

    def critical(self, msg, *a, **kw):
        self.logger.critical(msg, *a, **kw)

    def metric(self, msg, **kw):
        self.logger.info(f"METRIC: {msg}", extra=kw)
