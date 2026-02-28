"""
TENSOR GENESIS Profiling Infrastructure

Performance profiling decorators and utilities for all Genesis primitives:
- Timing with nanosecond precision
- Memory tracking (peak and current)
- Operation counting
- Hierarchical profiling with call trees
- Integration with logging system

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import functools
import gc
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, TypeVar, Union

import numpy as np

# Type variable for decorated functions
F = TypeVar('F', bound=Callable[..., Any])


@dataclass
class ProfileResult:
    """Result of a profiled operation.
    
    Attributes:
        name: Operation name.
        time_seconds: Wall-clock time in seconds.
        time_cpu_seconds: CPU time in seconds.
        memory_before_bytes: Memory before operation.
        memory_after_bytes: Memory after operation.
        memory_peak_bytes: Peak memory during operation.
        operations: Dictionary of operation counts.
        children: Nested ProfileResults for sub-operations.
        metadata: Additional metadata.
    """
    
    name: str
    time_seconds: float = 0.0
    time_cpu_seconds: float = 0.0
    memory_before_bytes: int = 0
    memory_after_bytes: int = 0
    memory_peak_bytes: int = 0
    operations: Dict[str, int] = field(default_factory=dict)
    children: List['ProfileResult'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def time_ms(self) -> float:
        """Time in milliseconds."""
        return self.time_seconds * 1000
    
    @property
    def memory_delta_bytes(self) -> int:
        """Change in memory usage."""
        return self.memory_after_bytes - self.memory_before_bytes
    
    @property
    def total_time_seconds(self) -> float:
        """Total time including children."""
        return self.time_seconds + sum(c.total_time_seconds for c in self.children)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation.
        """
        return {
            "name": self.name,
            "time_seconds": self.time_seconds,
            "time_ms": self.time_ms,
            "time_cpu_seconds": self.time_cpu_seconds,
            "memory_before_bytes": self.memory_before_bytes,
            "memory_after_bytes": self.memory_after_bytes,
            "memory_peak_bytes": self.memory_peak_bytes,
            "memory_delta_bytes": self.memory_delta_bytes,
            "operations": self.operations,
            "children": [c.to_dict() for c in self.children],
            "metadata": self.metadata,
        }
    
    def summary(self, indent: int = 0) -> str:
        """Generate human-readable summary.
        
        Args:
            indent: Indentation level.
            
        Returns:
            Formatted string.
        """
        prefix = "  " * indent
        lines = [
            f"{prefix}{self.name}: {self.time_ms:.2f}ms, "
            f"Δmem={_format_bytes(self.memory_delta_bytes)}"
        ]
        
        if self.operations:
            ops_str = ", ".join(f"{k}={v}" for k, v in self.operations.items())
            lines.append(f"{prefix}  ops: {ops_str}")
        
        for child in self.children:
            lines.append(child.summary(indent + 1))
        
        return "\n".join(lines)


class PerformanceTracker:
    """Thread-local performance tracker with hierarchical profiling.
    
    Provides a stack-based profiling system where nested operations
    are automatically tracked as children of their parent operations.
    
    Example:
        >>> tracker = PerformanceTracker()
        >>> with tracker.profile("outer"):
        ...     # outer code
        ...     with tracker.profile("inner"):
        ...         # inner code
        ...         tracker.count_op("matmul", 10)
        >>> print(tracker.root.summary())
    """
    
    # Thread-local storage for profile stacks
    _local = threading.local()
    
    def __init__(self, name: str = "root"):
        """Initialize tracker.
        
        Args:
            name: Root operation name.
        """
        self.root = ProfileResult(name=name)
        self._stack: List[ProfileResult] = [self.root]
        self._enabled = True
    
    @property
    def current(self) -> ProfileResult:
        """Get current profile context."""
        return self._stack[-1] if self._stack else self.root
    
    @contextmanager
    def profile(
        self,
        name: str,
        track_memory: bool = True,
        **metadata: Any,
    ) -> Iterator[ProfileResult]:
        """Context manager for profiling an operation.
        
        Args:
            name: Operation name.
            track_memory: Whether to track memory usage.
            **metadata: Additional metadata to attach.
            
        Yields:
            ProfileResult for this operation.
        """
        if not self._enabled:
            yield ProfileResult(name=name)
            return
        
        result = ProfileResult(name=name, metadata=metadata)
        
        # Memory before
        if track_memory:
            gc.collect()
            result.memory_before_bytes = _get_memory_usage()
        
        # Timing
        start_wall = time.perf_counter()
        start_cpu = time.process_time()
        
        # Add to current parent's children BEFORE pushing
        parent = self.current
        parent.children.append(result)
        
        # Push onto stack
        self._stack.append(result)
        
        try:
            yield result
        finally:
            # Pop from stack
            self._stack.pop()
            
            # Record timing
            result.time_seconds = time.perf_counter() - start_wall
            result.time_cpu_seconds = time.process_time() - start_cpu
            
            # Memory after
            if track_memory:
                result.memory_after_bytes = _get_memory_usage()
                result.memory_peak_bytes = max(
                    result.memory_before_bytes,
                    result.memory_after_bytes
                )
    
    def count_op(self, op_name: str, count: int = 1) -> None:
        """Count operations in current context.
        
        Args:
            op_name: Operation type name.
            count: Number of operations.
        """
        if self._enabled:
            current = self.current
            current.operations[op_name] = current.operations.get(op_name, 0) + count
    
    def add_metadata(self, **kwargs: Any) -> None:
        """Add metadata to current context.
        
        Args:
            **kwargs: Metadata key-value pairs.
        """
        if self._enabled:
            self.current.metadata.update(kwargs)
    
    def disable(self) -> None:
        """Disable profiling (for production)."""
        self._enabled = False
    
    def enable(self) -> None:
        """Enable profiling."""
        self._enabled = True
    
    def reset(self) -> None:
        """Reset all profiling data."""
        self.root = ProfileResult(name=self.root.name)
        self._stack = [self.root]
    
    def report(self) -> str:
        """Generate full profiling report.
        
        Returns:
            Formatted report string.
        """
        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║                  PERFORMANCE PROFILE REPORT                  ║",
            "╚══════════════════════════════════════════════════════════════╝",
            "",
        ]
        lines.append(self.root.summary())
        return "\n".join(lines)


# Global tracker instance (thread-safe)
_global_tracker: Optional[PerformanceTracker] = None
_tracker_lock = threading.Lock()


def get_tracker() -> PerformanceTracker:
    """Get or create global performance tracker.
    
    Returns:
        Global PerformanceTracker instance.
    """
    global _global_tracker
    if _global_tracker is None:
        with _tracker_lock:
            if _global_tracker is None:
                _global_tracker = PerformanceTracker("genesis")
    return _global_tracker


# Decorator functions

def profile(
    name: Optional[str] = None,
    track_memory: bool = True,
) -> Callable[[F], F]:
    """Decorator to profile a function.
    
    Args:
        name: Override function name in profile.
        track_memory: Whether to track memory.
        
    Returns:
        Decorator function.
        
    Example:
        >>> @profile(track_memory=True)
        ... def expensive_function(x):
        ...     return x ** 2
    """
    def decorator(func: F) -> F:
        profile_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracker = get_tracker()
            with tracker.profile(profile_name, track_memory=track_memory) as result:
                return_value = func(*args, **kwargs)
                
                # Try to extract shape info from numpy arrays
                if isinstance(return_value, np.ndarray):
                    result.metadata["output_shape"] = return_value.shape
                    result.metadata["output_dtype"] = str(return_value.dtype)
                
                return return_value
        
        return wrapper  # type: ignore
    
    return decorator


def profile_memory(func: F) -> F:
    """Decorator to profile memory usage only.
    
    Lighter weight than full profile() - only tracks memory.
    
    Args:
        func: Function to decorate.
        
    Returns:
        Decorated function.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        gc.collect()
        mem_before = _get_memory_usage()
        
        result = func(*args, **kwargs)
        
        gc.collect()
        mem_after = _get_memory_usage()
        
        # Attach to function for inspection
        wrapper._last_memory_delta = mem_after - mem_before  # type: ignore
        
        return result
    
    wrapper._last_memory_delta = 0  # type: ignore
    return wrapper  # type: ignore


def timed(func: F) -> F:
    """Decorator to time a function with minimal overhead.
    
    Args:
        func: Function to decorate.
        
    Returns:
        Decorated function with timing info.
        
    Example:
        >>> @timed
        ... def my_function():
        ...     time.sleep(0.1)
        >>> my_function()
        >>> print(my_function._last_time_ms)
        100.5
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        
        wrapper._last_time_seconds = elapsed  # type: ignore
        wrapper._last_time_ms = elapsed * 1000  # type: ignore
        
        return result
    
    wrapper._last_time_seconds = 0.0  # type: ignore
    wrapper._last_time_ms = 0.0  # type: ignore
    return wrapper  # type: ignore


def traced(
    log_args: bool = False,
    log_result: bool = False,
    log_time: bool = True,
) -> Callable[[F], F]:
    """Decorator combining profiling with logging.
    
    Args:
        log_args: Log function arguments.
        log_result: Log return value.
        log_time: Log execution time.
        
    Returns:
        Decorator function.
    """
    def decorator(func: F) -> F:
        # Import here to avoid circular dependency
        from ontic.genesis.core.logging import get_logger, LogLevel
        
        logger = get_logger(func.__module__.split('.')[-1])
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Log entry
            if log_args:
                args_summary = _summarize_args(args, kwargs)
                logger.debug(f"→ {func.__name__}({args_summary})")
            else:
                logger.debug(f"→ {func.__name__}")
            
            # Execute with timing
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.error(f"✗ {func.__name__} failed after {elapsed_ms:.2f}ms: {e}")
                raise
            
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            # Log exit
            if log_result:
                result_summary = _summarize_value(result)
                logger.debug(f"← {func.__name__} = {result_summary} ({elapsed_ms:.2f}ms)")
            elif log_time:
                logger.perf(f"← {func.__name__} ({elapsed_ms:.2f}ms)")
            
            return result
        
        return wrapper  # type: ignore
    
    return decorator


# Context managers

@contextmanager
def profile_block(
    name: str,
    track_memory: bool = True,
    **metadata: Any,
) -> Iterator[ProfileResult]:
    """Profile a code block.
    
    Args:
        name: Block name.
        track_memory: Track memory usage.
        **metadata: Additional metadata.
        
    Yields:
        ProfileResult for the block.
        
    Example:
        >>> with profile_block("matrix_operations") as p:
        ...     result = np.dot(A, B)
        >>> print(f"Time: {p.time_ms:.2f}ms")
    """
    tracker = get_tracker()
    with tracker.profile(name, track_memory=track_memory, **metadata) as result:
        yield result


@contextmanager
def timer(name: Optional[str] = None) -> Iterator[Dict[str, float]]:
    """Simple timing context manager.
    
    Args:
        name: Optional name (unused, for documentation).
        
    Yields:
        Dictionary that will contain elapsed_seconds and elapsed_ms.
        
    Example:
        >>> with timer("operation") as t:
        ...     do_work()
        >>> print(f"Took {t['elapsed_ms']:.2f}ms")
    """
    result: Dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        elapsed = time.perf_counter() - start
        result["elapsed_seconds"] = elapsed
        result["elapsed_ms"] = elapsed * 1000


# Utility functions

def _get_memory_usage() -> int:
    """Get current process memory usage in bytes.
    
    Returns:
        Memory usage in bytes.
    """
    try:
        import resource
        # getrusage returns memory in kilobytes on Linux
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except ImportError:
        # Fallback: estimate from sys.getsizeof of gc objects
        return sum(sys.getsizeof(obj) for obj in gc.get_objects()[:1000])


def _format_bytes(n_bytes: int) -> str:
    """Format bytes as human-readable string.
    
    Args:
        n_bytes: Number of bytes.
        
    Returns:
        Formatted string.
    """
    sign = "-" if n_bytes < 0 else ""
    n_bytes = abs(n_bytes)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if n_bytes < 1024:
            return f"{sign}{n_bytes:.1f} {unit}"
        n_bytes //= 1024
    return f"{sign}{n_bytes:.1f} TB"


def _summarize_args(args: tuple, kwargs: dict) -> str:
    """Summarize function arguments.
    
    Args:
        args: Positional arguments.
        kwargs: Keyword arguments.
        
    Returns:
        Summary string.
    """
    parts = []
    for i, arg in enumerate(args[:3]):  # Limit to first 3
        parts.append(_summarize_value(arg))
    if len(args) > 3:
        parts.append("...")
    for k, v in list(kwargs.items())[:3]:
        parts.append(f"{k}={_summarize_value(v)}")
    return ", ".join(parts)


def _summarize_value(value: Any) -> str:
    """Summarize a value for logging.
    
    Args:
        value: Any value.
        
    Returns:
        Summary string.
    """
    if isinstance(value, np.ndarray):
        return f"ndarray{value.shape}"
    elif isinstance(value, (list, tuple)) and len(value) > 5:
        return f"{type(value).__name__}[{len(value)}]"
    elif isinstance(value, dict) and len(value) > 3:
        return f"dict[{len(value)}]"
    else:
        s = repr(value)
        return s if len(s) <= 30 else s[:27] + "..."


# Benchmark utilities

def benchmark(
    func: Callable[..., Any],
    *args: Any,
    n_runs: int = 10,
    warmup: int = 2,
    **kwargs: Any,
) -> Dict[str, float]:
    """Benchmark a function with multiple runs.
    
    Args:
        func: Function to benchmark.
        *args: Function arguments.
        n_runs: Number of timed runs.
        warmup: Number of warmup runs.
        **kwargs: Function keyword arguments.
        
    Returns:
        Dictionary with timing statistics.
    """
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        times.append(time.perf_counter() - start)
    
    times_np = np.array(times)
    return {
        "mean_seconds": float(np.mean(times_np)),
        "std_seconds": float(np.std(times_np)),
        "min_seconds": float(np.min(times_np)),
        "max_seconds": float(np.max(times_np)),
        "median_seconds": float(np.median(times_np)),
        "mean_ms": float(np.mean(times_np) * 1000),
        "n_runs": n_runs,
    }
