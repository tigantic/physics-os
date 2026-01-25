"""
TENSOR GENESIS Logging Infrastructure

Hierarchical, configurable logging for all Genesis primitives with:
- Per-module log levels
- Structured logging with context
- Performance-aware lazy evaluation
- Integration with profiling system

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import IntEnum
from functools import wraps
from typing import Any, Callable, Dict, Iterator, Optional, TypeVar, Union

import numpy as np

# Type variable for generic function decoration
F = TypeVar('F', bound=Callable[..., Any])


class LogLevel(IntEnum):
    """Log levels for Genesis modules."""
    
    TRACE = 5      # Ultra-verbose: every tensor operation
    DEBUG = 10     # Development: shapes, ranks, intermediate values
    INFO = 20      # Normal: algorithm progress, key metrics
    PERF = 25      # Performance: timing, memory, compression ratios
    WARNING = 30   # Warnings: numerical issues, fallbacks
    ERROR = 40     # Errors: recoverable failures
    CRITICAL = 50  # Critical: unrecoverable failures


# Register custom TRACE and PERF levels
logging.addLevelName(LogLevel.TRACE, "TRACE")
logging.addLevelName(LogLevel.PERF, "PERF")


@dataclass
class LogContext:
    """Context information attached to log messages."""
    
    module: str
    layer: Optional[int] = None
    primitive: Optional[str] = None
    operation: Optional[str] = None
    scale: Optional[int] = None
    rank: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for structured logging."""
        result = {"module": self.module}
        if self.layer is not None:
            result["layer"] = self.layer
        if self.primitive is not None:
            result["primitive"] = self.primitive
        if self.operation is not None:
            result["operation"] = self.operation
        if self.scale is not None:
            result["scale"] = self.scale
        if self.rank is not None:
            result["rank"] = self.rank
        result.update(self.extra)
        return result


class GenesisFormatter(logging.Formatter):
    """Custom formatter with Genesis-specific styling."""
    
    COLORS = {
        LogLevel.TRACE: "\033[90m",      # Gray
        LogLevel.DEBUG: "\033[36m",      # Cyan
        LogLevel.INFO: "\033[32m",       # Green
        LogLevel.PERF: "\033[35m",       # Magenta
        LogLevel.WARNING: "\033[33m",    # Yellow
        LogLevel.ERROR: "\033[31m",      # Red
        LogLevel.CRITICAL: "\033[41m",   # Red background
    }
    RESET = "\033[0m"
    
    def __init__(self, use_colors: bool = True):
        """Initialize formatter.
        
        Args:
            use_colors: Whether to use ANSI color codes in output.
        """
        super().__init__()
        self.use_colors = use_colors
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with Genesis styling.
        
        Args:
            record: Log record to format.
            
        Returns:
            Formatted log message string.
        """
        level = record.levelno
        level_name = record.levelname
        
        # Add color if enabled
        if self.use_colors and level in self.COLORS:
            color = self.COLORS[level]
            level_name = f"{color}{level_name:8}{self.RESET}"
        else:
            level_name = f"{level_name:8}"
        
        # Build message
        timestamp = time.strftime("%H:%M:%S")
        module = getattr(record, 'genesis_module', record.name.split('.')[-1])
        
        # Include context if present
        context_str = ""
        if hasattr(record, 'genesis_context'):
            ctx = record.genesis_context
            parts = []
            if ctx.get('primitive'):
                parts.append(f"[{ctx['primitive']}]")
            if ctx.get('operation'):
                parts.append(f"{ctx['operation']}")
            if ctx.get('scale'):
                parts.append(f"N={ctx['scale']}")
            if ctx.get('rank'):
                parts.append(f"r={ctx['rank']}")
            if parts:
                context_str = " " + " ".join(parts)
        
        return f"{timestamp} {level_name} {module:12}{context_str} │ {record.getMessage()}"


class GenesisLogger:
    """Production-grade logger for Genesis primitives.
    
    Provides structured logging with context, lazy evaluation for
    expensive log messages, and integration with the profiling system.
    
    Attributes:
        name: Logger name (typically the module name).
        level: Current log level.
        context: Default context attached to all messages.
        
    Example:
        >>> logger = GenesisLogger("qtt_ot", layer=20, primitive="OT")
        >>> logger.info("Starting Sinkhorn", scale=65536, iterations=100)
        >>> with logger.operation("sinkhorn_iteration"):
        ...     # Code here
        ...     logger.perf("Iteration complete", time_ms=1.5)
    """
    
    def __init__(
        self,
        name: str,
        layer: Optional[int] = None,
        primitive: Optional[str] = None,
        level: Union[int, LogLevel] = LogLevel.INFO,
    ):
        """Initialize Genesis logger.
        
        Args:
            name: Logger name (typically module name).
            layer: Genesis layer number (20-26).
            primitive: Primitive name (e.g., "OT", "SGW", "RKHS").
            level: Initial log level.
        """
        self._logger = logging.getLogger(f"genesis.{name}")
        self._logger.setLevel(level)
        
        self.name = name
        self.context = LogContext(
            module=name,
            layer=layer,
            primitive=primitive,
        )
        
        # Add handler if not already present
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(GenesisFormatter(use_colors=True))
            self._logger.addHandler(handler)
            self._logger.propagate = False
    
    def _log(
        self,
        level: int,
        msg: Union[str, Callable[[], str]],
        **kwargs: Any,
    ) -> None:
        """Internal log method with lazy evaluation.
        
        Args:
            level: Log level.
            msg: Message string or callable that returns message.
            **kwargs: Additional context fields.
        """
        if not self._logger.isEnabledFor(level):
            return
        
        # Lazy evaluation for expensive messages
        if callable(msg):
            msg = msg()
        
        # Build context
        ctx = self.context.to_dict()
        ctx.update(kwargs)
        
        # Create record with context
        record = self._logger.makeRecord(
            self._logger.name,
            level,
            "",  # pathname
            0,   # lineno
            msg,
            (),  # args
            None,  # exc_info
        )
        record.genesis_module = self.name
        record.genesis_context = ctx
        
        self._logger.handle(record)
    
    def trace(self, msg: Union[str, Callable[[], str]], **kwargs: Any) -> None:
        """Log at TRACE level (ultra-verbose).
        
        Args:
            msg: Message or lazy callable.
            **kwargs: Additional context.
        """
        self._log(LogLevel.TRACE, msg, **kwargs)
    
    def debug(self, msg: Union[str, Callable[[], str]], **kwargs: Any) -> None:
        """Log at DEBUG level.
        
        Args:
            msg: Message or lazy callable.
            **kwargs: Additional context.
        """
        self._log(LogLevel.DEBUG, msg, **kwargs)
    
    def info(self, msg: Union[str, Callable[[], str]], **kwargs: Any) -> None:
        """Log at INFO level.
        
        Args:
            msg: Message or lazy callable.
            **kwargs: Additional context.
        """
        self._log(LogLevel.INFO, msg, **kwargs)
    
    def perf(self, msg: Union[str, Callable[[], str]], **kwargs: Any) -> None:
        """Log at PERF level (performance metrics).
        
        Args:
            msg: Message or lazy callable.
            **kwargs: Additional context (typically timing/memory).
        """
        self._log(LogLevel.PERF, msg, **kwargs)
    
    def warning(self, msg: Union[str, Callable[[], str]], **kwargs: Any) -> None:
        """Log at WARNING level.
        
        Args:
            msg: Message or lazy callable.
            **kwargs: Additional context.
        """
        self._log(LogLevel.WARNING, msg, **kwargs)
    
    def error(self, msg: Union[str, Callable[[], str]], **kwargs: Any) -> None:
        """Log at ERROR level.
        
        Args:
            msg: Message or lazy callable.
            **kwargs: Additional context.
        """
        self._log(LogLevel.ERROR, msg, **kwargs)
    
    def critical(self, msg: Union[str, Callable[[], str]], **kwargs: Any) -> None:
        """Log at CRITICAL level.
        
        Args:
            msg: Message or lazy callable.
            **kwargs: Additional context.
        """
        self._log(LogLevel.CRITICAL, msg, **kwargs)
    
    @contextmanager
    def operation(self, name: str, **kwargs: Any) -> Iterator[None]:
        """Context manager for logging operation start/end with timing.
        
        Args:
            name: Operation name.
            **kwargs: Additional context.
            
        Yields:
            None
            
        Example:
            >>> with logger.operation("sinkhorn"):
            ...     result = sinkhorn_qtt(...)
        """
        self.context.operation = name
        start = time.perf_counter()
        self.debug(f"Starting {name}", **kwargs)
        
        try:
            yield
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            self.error(f"{name} failed after {elapsed:.2f}ms: {e}", **kwargs)
            raise
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            self.perf(f"{name} completed in {elapsed:.2f}ms", **kwargs)
            self.context.operation = None
    
    def set_level(self, level: Union[int, LogLevel]) -> None:
        """Set log level.
        
        Args:
            level: New log level.
        """
        self._logger.setLevel(level)
    
    def child(self, name: str, **kwargs: Any) -> 'GenesisLogger':
        """Create a child logger with inherited context.
        
        Args:
            name: Child logger name suffix.
            **kwargs: Additional context to merge.
            
        Returns:
            New GenesisLogger with merged context.
        """
        child = GenesisLogger(
            f"{self.name}.{name}",
            layer=kwargs.get('layer', self.context.layer),
            primitive=kwargs.get('primitive', self.context.primitive),
            level=self._logger.level,
        )
        child.context.extra.update(self.context.extra)
        child.context.extra.update(kwargs)
        return child


# Module-level loggers cache
_loggers: Dict[str, GenesisLogger] = {}


def get_logger(
    name: str,
    layer: Optional[int] = None,
    primitive: Optional[str] = None,
) -> GenesisLogger:
    """Get or create a Genesis logger.
    
    Args:
        name: Logger name (module name).
        layer: Genesis layer number.
        primitive: Primitive name.
        
    Returns:
        Cached or new GenesisLogger instance.
        
    Example:
        >>> logger = get_logger("sinkhorn_qtt", layer=20, primitive="OT")
    """
    if name not in _loggers:
        _loggers[name] = GenesisLogger(name, layer=layer, primitive=primitive)
    return _loggers[name]


def configure_logging(
    level: Union[int, LogLevel] = LogLevel.INFO,
    modules: Optional[Dict[str, Union[int, LogLevel]]] = None,
    use_colors: bool = True,
) -> None:
    """Configure Genesis logging globally.
    
    Args:
        level: Default log level for all Genesis modules.
        modules: Per-module log levels (e.g., {"sinkhorn_qtt": LogLevel.DEBUG}).
        use_colors: Whether to use ANSI colors in output.
        
    Example:
        >>> configure_logging(
        ...     level=LogLevel.INFO,
        ...     modules={"sinkhorn_qtt": LogLevel.DEBUG},
        ... )
    """
    # Configure root genesis logger
    root = logging.getLogger("genesis")
    root.setLevel(level)
    
    # Update all cached loggers
    for logger in _loggers.values():
        logger.set_level(level)
    
    # Apply per-module overrides
    if modules:
        for module_name, module_level in modules.items():
            if module_name in _loggers:
                _loggers[module_name].set_level(module_level)


def logged(
    level: Union[int, LogLevel] = LogLevel.DEBUG,
    include_args: bool = False,
    include_result: bool = False,
) -> Callable[[F], F]:
    """Decorator to log function calls.
    
    Args:
        level: Log level for the messages.
        include_args: Whether to log function arguments.
        include_result: Whether to log return value.
        
    Returns:
        Decorator function.
        
    Example:
        >>> @logged(level=LogLevel.DEBUG, include_args=True)
        ... def my_function(x, y):
        ...     return x + y
    """
    def decorator(func: F) -> F:
        logger = get_logger(func.__module__.split('.')[-1])
        
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Log call
            if include_args:
                args_str = ", ".join(
                    [repr(a)[:50] for a in args] +
                    [f"{k}={repr(v)[:30]}" for k, v in kwargs.items()]
                )
                logger._log(level, f"Calling {func.__name__}({args_str})")
            else:
                logger._log(level, f"Calling {func.__name__}")
            
            # Execute
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000
            
            # Log result
            if include_result:
                result_str = repr(result)[:100]
                logger._log(level, f"{func.__name__} returned {result_str} in {elapsed:.2f}ms")
            else:
                logger._log(level, f"{func.__name__} completed in {elapsed:.2f}ms")
            
            return result
        
        return wrapper  # type: ignore
    
    return decorator
