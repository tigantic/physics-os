"""
Centralized logging configuration for HyperTensor.

This module provides a unified logging setup for the entire library,
with support for different log levels, formatters, and handlers.

Usage:
    from tensornet.logging_config import get_logger, configure_logging
    
    # Get a logger for your module
    logger = get_logger(__name__)
    logger.info("Starting computation")
    
    # Configure global logging level
    configure_logging(level="DEBUG")
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
import warnings

__all__ = [
    "get_logger",
    "configure_logging",
    "set_log_level",
    "HyperTensorLogger",
]

# Default format for log messages
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
SIMPLE_FORMAT = "%(levelname)s: %(message)s"
DEBUG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"

# Root logger name for the library
LIBRARY_ROOT = "tensornet"

# Global state
_configured = False


class HyperTensorFormatter(logging.Formatter):
    """Custom formatter with colors for terminal output."""
    
    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def __init__(self, fmt: Optional[str] = None, use_colors: bool = True):
        super().__init__(fmt or DEFAULT_FORMAT)
        self.use_colors = use_colors and sys.stdout.isatty()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with optional colors."""
        message = super().format(record)
        if self.use_colors and record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            message = f"{color}{message}{self.RESET}"
        return message


class HyperTensorLogger(logging.Logger):
    """Extended logger with HyperTensor-specific methods."""
    
    def computation(self, msg: str, *args, **kwargs):
        """Log computational progress at INFO level."""
        self.info(f"[COMPUTE] {msg}", *args, **kwargs)
    
    def physics(self, msg: str, *args, **kwargs):
        """Log physics-related information at INFO level."""
        self.info(f"[PHYSICS] {msg}", *args, **kwargs)
    
    def convergence(self, iteration: int, value: float, threshold: float = 0.0):
        """Log convergence progress."""
        if threshold > 0:
            self.info(f"[CONV] Iter {iteration}: {value:.2e} (thresh: {threshold:.2e})")
        else:
            self.info(f"[CONV] Iter {iteration}: {value:.2e}")
    
    def tensor_op(self, op_name: str, shape: tuple, dtype: str = "float64"):
        """Log tensor operation details at DEBUG level."""
        self.debug(f"[TENSOR] {op_name}: shape={shape}, dtype={dtype}")


# Register our custom logger class
logging.setLoggerClass(HyperTensorLogger)


def get_logger(name: str) -> HyperTensorLogger:
    """
    Get a logger instance for the given module name.
    
    Args:
        name: The module name (typically __name__)
        
    Returns:
        A configured HyperTensorLogger instance
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting DMRG calculation")
    """
    # Ensure base configuration is done
    if not _configured:
        configure_logging()
    
    # If name doesn't start with library root, prefix it
    if not name.startswith(LIBRARY_ROOT) and name != LIBRARY_ROOT:
        name = f"{LIBRARY_ROOT}.{name}"
    
    return logging.getLogger(name)


def configure_logging(
    level: Union[str, int] = "WARNING",
    format: Optional[str] = None,
    use_colors: bool = True,
    log_file: Optional[Union[str, Path]] = None,
    stream: bool = True,
) -> None:
    """
    Configure logging for the HyperTensor library.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Log message format (None for default)
        use_colors: Whether to use colored output in terminal
        log_file: Optional path to log file
        stream: Whether to output to stdout/stderr
        
    Example:
        >>> configure_logging(level="DEBUG")
        >>> configure_logging(level="INFO", log_file="hypertensor.log")
    """
    global _configured
    
    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.WARNING)
    
    # Get or create the root library logger
    root_logger = logging.getLogger(LIBRARY_ROOT)
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Determine format based on level
    if format is None:
        if level <= logging.DEBUG:
            format = DEBUG_FORMAT
        elif level <= logging.INFO:
            format = DEFAULT_FORMAT
        else:
            format = SIMPLE_FORMAT
    
    # Add stream handler
    if stream:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(HyperTensorFormatter(format, use_colors=use_colors))
        root_logger.addHandler(stream_handler)
    
    # Add file handler if requested
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        # No colors for file output
        file_handler.setFormatter(logging.Formatter(format))
        root_logger.addHandler(file_handler)
    
    # Don't propagate to root logger
    root_logger.propagate = False
    
    _configured = True


def set_log_level(level: Union[str, int]) -> None:
    """
    Set the log level for all HyperTensor loggers.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Example:
        >>> set_log_level("DEBUG")  # Enable debug output
        >>> set_log_level("ERROR")  # Suppress most output
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.WARNING)
    
    logging.getLogger(LIBRARY_ROOT).setLevel(level)


def silence_warnings() -> None:
    """Silence all HyperTensor warnings (not recommended for development)."""
    set_log_level("ERROR")
    warnings.filterwarnings("ignore", module="tensornet.*")


def enable_debug() -> None:
    """Enable debug-level logging for troubleshooting."""
    configure_logging(level="DEBUG", use_colors=True)


# Module-level logger for testing
_module_logger = None


def _get_test_logger() -> HyperTensorLogger:
    """Get a test logger for module testing."""
    global _module_logger
    if _module_logger is None:
        configure_logging(level="DEBUG")
        _module_logger = get_logger("test")
    return _module_logger


# Auto-configure with default settings on import
if not _configured:
    configure_logging()
