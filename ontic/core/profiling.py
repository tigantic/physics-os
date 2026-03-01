"""
Profiling Utilities
====================

Memory and performance profiling decorators per Article VIII.8.2.

Usage:
    from ontic.core.profiling import profile, memory_profile

    @profile
    def my_function():
        ...

    @memory_profile
    def memory_intensive_function():
        ...

Enable profiling by setting environment variable:
    ONTIC_PROFILE=1 python script.py
"""

import functools
import logging
import os
import time
import tracemalloc
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

# Check if profiling is enabled
PROFILING_ENABLED = os.environ.get("ONTIC_PROFILE", "0") == "1"

F = TypeVar("F", bound=Callable[..., Any])


def profile(func: F) -> F:
    """
    Decorator that profiles function execution time.

    Only active when ONTIC_PROFILE=1 environment variable is set.

    Args:
        func: Function to profile

    Returns:
        Wrapped function with timing

    Example:
        >>> @profile
        ... def dmrg_sweep(psi, H):
        ...     ...
    """
    if not PROFILING_ENABLED:
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"[PROFILE] {func.__module__}.{func.__name__}: {elapsed:.4f}s")
        return result

    return wrapper  # type: ignore


def memory_profile(func: F) -> F:
    """
    Decorator that profiles function memory usage.

    Only active when ONTIC_PROFILE=1 environment variable is set.
    Uses tracemalloc to measure peak memory allocation.

    Args:
        func: Function to profile

    Returns:
        Wrapped function with memory tracking

    Example:
        >>> @memory_profile
        ... def build_environments(psi, H):
        ...     ...
    """
    if not PROFILING_ENABLED:
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        start = time.perf_counter()

        result = func(*args, **kwargs)

        elapsed = time.perf_counter() - start
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        logger.info(
            f"[MEMORY] {func.__module__}.{func.__name__}: "
            f"{elapsed:.4f}s, current={current/1024/1024:.2f}MB, peak={peak/1024/1024:.2f}MB"
        )
        return result

    return wrapper  # type: ignore


def profile_block(name: str):
    """
    Context manager for profiling code blocks.

    Only active when ONTIC_PROFILE=1 environment variable is set.

    Args:
        name: Name for the profiled block

    Example:
        >>> with profile_block("SVD computation"):
        ...     U, S, V = torch.linalg.svd(A)
    """

    class ProfileContext:
        def __enter__(self):
            if PROFILING_ENABLED:
                self.start = time.perf_counter()
                tracemalloc.start()
            return self

        def __exit__(self, *args):
            if PROFILING_ENABLED:
                elapsed = time.perf_counter() - self.start
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                logger.info(
                    f"[BLOCK] {name}: {elapsed:.4f}s, peak={peak/1024/1024:.2f}MB"
                )

    return ProfileContext()


class PerformanceReport:
    """
    Collect and report performance statistics.

    Example:
        >>> report = PerformanceReport()
        >>> with report.measure("DMRG sweep"):
        ...     dmrg_sweep(psi, H)
        >>> print(report.summary())
    """

    def __init__(self):
        self.measurements = []

    def measure(self, name: str):
        """Context manager to measure a named operation."""
        report = self

        class MeasureContext:
            def __enter__(self):
                self.start = time.perf_counter()
                tracemalloc.start()
                return self

            def __exit__(self, *args):
                elapsed = time.perf_counter() - self.start
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                report.measurements.append(
                    {
                        "name": name,
                        "time_s": elapsed,
                        "current_mb": current / 1024 / 1024,
                        "peak_mb": peak / 1024 / 1024,
                    }
                )

        return MeasureContext()

    def summary(self) -> str:
        """Generate summary report."""
        if not self.measurements:
            return "No measurements recorded."

        lines = ["Performance Report", "=" * 50]
        total_time = 0
        max_peak = 0

        for m in self.measurements:
            lines.append(
                f"  {m['name']}: {m['time_s']:.4f}s, peak={m['peak_mb']:.2f}MB"
            )
            total_time += m["time_s"]
            max_peak = max(max_peak, m["peak_mb"])

        lines.append("=" * 50)
        lines.append(f"Total time: {total_time:.4f}s, max peak: {max_peak:.2f}MB")

        return "\n".join(lines)
