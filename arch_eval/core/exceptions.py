"""Custom exceptions for arch_eval."""

class ArchEvalError(Exception):
    """Base exception for all arch_eval errors."""
    pass

class DatasetFormatError(ArchEvalError):
    """Raised when dataset format is not supported."""
    pass

class ConfigurationError(ArchEvalError):
    """Raised when configuration is invalid."""
    pass

class ModelError(ArchEvalError):
    """Raised when model doesn't conform to expected interface."""
    pass

class PluginError(ArchEvalError):
    """Raised when plugin operation fails."""
    pass

class VisualizationError(ArchEvalError):
    """Raised when visualization fails."""
    pass

class StopTraining(ArchEvalError):
    """Exception that can be raised by plugins to stop training gracefully."""
    pass

class DistributedError(ArchEvalError):
    """Raised for distributed training issues."""
    pass
