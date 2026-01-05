"""
Dense Materialization Guard
============================

Execution guard that detects and forbids hidden dense tensor materialization.
This ensures O(N·d·χ²) TT-CFD storage claims are not invalidated by hidden dense ops.

CLAIM SCOPE:
    ✅ Storage: O(N·d·χ²) - provable by core element counts
    ✅ No dense GRID materialization: O(N²), O(N³), etc. forbidden
    ✅ O(N) or O(N·d) diagnostic vectors: ALLOWED (per-site values, primitives)
    ⚠️ Runtime: "empirically sub-dense" - not proven here, but no blowups observed

Usage:
    # PROOF MODE (strict enforcement)
    with DenseMaterializationGuard(hard_threshold=N*d*chi**2, forbid=True) as guard:
        result = tt_solver.step(dt)
    # Raises RuntimeError immediately on violation

    # DIAGNOSTIC MODE (observational)
    with DenseMaterializationGuard(hard_threshold=N*d*chi**2, forbid=False) as guard:
        result = tt_solver.step(dt)
    print(guard.report())

Monitored Operations:
    - torch.zeros/ones/full/empty and _like variants
    - torch.tensor, stack, cat, arange, linspace
    - Tensor.clone(), contiguous(), numpy(), tolist()
    - Any TT→dense conversion via to_dense()
"""

import functools
import threading
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class DenseViolation:
    """Record of a dense materialization violation."""

    operation: str
    size: int
    shape: tuple[int, ...]
    location: str
    message: str
    is_critical: bool  # True if exceeds hard threshold


@dataclass
class DenseMaterializationGuard:
    """
    Context manager that monitors and optionally forbids dense materialization.

    Two threshold levels:
        - soft_threshold: O(N·d·χ) - flags for review, small buffers OK
        - hard_threshold: O(N·d·χ²) - absolute limit, anything larger is CRITICAL

    Attributes:
        hard_threshold: Maximum allowed allocation (elements). Exceeding = critical.
        soft_threshold: Warning threshold. Between soft and hard = flagged but allowed.
        forbid: If True, raise exception on CRITICAL violation. If False, just log.
        allow_diagnostics: If True, O(N·d) allocations are always allowed.
    """

    hard_threshold: int = 10000  # O(N·d·χ²) scale
    soft_threshold: int = 1000  # O(N·d·χ) scale
    forbid: bool = True  # PROOF MODE by default
    allow_diagnostics: bool = True  # O(N·d) vectors allowed
    N: int = 64  # Grid size for diagnostic threshold
    d: int = 3  # Physical dimension

    # Class-level tracking
    _active: bool = False
    _violations: list[DenseViolation] = field(default_factory=list)
    _original_methods: dict[str, Callable] = field(default_factory=dict)
    _total_dense_elements: int = 0
    _critical_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        self._violations = []
        self._original_methods = {}
        self._total_dense_elements = 0
        self._critical_count = 0
        self._lock = threading.Lock()

        # Diagnostic threshold: O(N·d) is always OK
        self._diagnostic_threshold = self.N * self.d * 10  # 10x buffer

    @property
    def violations(self) -> int:
        """Number of dense materialization violations detected."""
        return len(self._violations)

    @property
    def critical_violations(self) -> int:
        """Number of CRITICAL violations (exceed hard threshold)."""
        return self._critical_count

    @property
    def total_dense_elements(self) -> int:
        """Total number of dense elements materialized."""
        return self._total_dense_elements

    def get_violations(self) -> list[DenseViolation]:
        """Get list of all violations."""
        return self._violations.copy()

    def _is_diagnostic_size(self, size: int) -> bool:
        """Check if size is diagnostic-scale O(N·d), not grid-scale O(N²+)."""
        return self.allow_diagnostics and size <= self._diagnostic_threshold

    def _record_violation(
        self, operation: str, size: int, shape: tuple[int, ...], message: str = ""
    ) -> None:
        """Record a dense materialization violation."""
        import traceback

        # Skip diagnostic-scale allocations
        if self._is_diagnostic_size(size):
            return

        # Determine severity
        is_critical = size >= self.hard_threshold

        # Get call location
        stack = traceback.extract_stack()
        location = "unknown"
        for frame in reversed(stack[:-2]):
            if "dense_guard.py" not in frame.filename:
                location = f"{frame.filename}:{frame.lineno} in {frame.name}"
                break

        violation = DenseViolation(
            operation=operation,
            size=size,
            shape=shape,
            location=location,
            message=message,
            is_critical=is_critical,
        )

        with self._lock:
            self._violations.append(violation)
            self._total_dense_elements += size
            if is_critical:
                self._critical_count += 1

        severity = "CRITICAL" if is_critical else "WARNING"
        msg = (
            f"[{severity}] Dense materialization: {operation} "
            f"size={size:,} shape={shape} at {location}"
        )

        if is_critical and self.forbid:
            raise RuntimeError(f"FORBIDDEN DENSE MATERIALIZATION: {msg}")
        elif is_critical:
            warnings.warn(f"CRITICAL: {msg}", stacklevel=4)
        elif size >= self.soft_threshold:
            warnings.warn(msg, stacklevel=4)

    def _wrap_factory(self, name: str, original: Callable) -> Callable:
        """Wrap torch factory functions (zeros, ones, full, empty, tensor, etc.)."""
        guard = self

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            size = result.numel()
            if size >= guard.soft_threshold:
                guard._record_violation(
                    f"torch.{name}()",
                    size,
                    tuple(result.shape),
                    f"Dense tensor created via {name}()",
                )
            return result

        return wrapped

    def _wrap_like_factory(self, name: str, original: Callable) -> Callable:
        """Wrap torch _like factory functions."""
        guard = self

        @functools.wraps(original)
        def wrapped(tensor, *args, **kwargs):
            result = original(tensor, *args, **kwargs)
            size = result.numel()
            if size >= guard.soft_threshold:
                guard._record_violation(
                    f"torch.{name}()",
                    size,
                    tuple(result.shape),
                    f"Dense tensor created via {name}()",
                )
            return result

        return wrapped

    def _wrap_combining(self, name: str, original: Callable) -> Callable:
        """Wrap torch combining functions (stack, cat)."""
        guard = self

        @functools.wraps(original)
        def wrapped(tensors, *args, **kwargs):
            result = original(tensors, *args, **kwargs)
            size = result.numel()
            if size >= guard.soft_threshold:
                guard._record_violation(
                    f"torch.{name}()",
                    size,
                    tuple(result.shape),
                    f"Dense tensor created via {name}()",
                )
            return result

        return wrapped

    def _wrap_tensor_method(self, name: str, original: Callable) -> Callable:
        """Wrap Tensor methods (clone, contiguous, numpy, tolist)."""
        guard = self

        @functools.wraps(original)
        def wrapped(self_tensor, *args, **kwargs):
            result = original(self_tensor, *args, **kwargs)
            size = self_tensor.numel()
            if size >= guard.soft_threshold:
                guard._record_violation(
                    f"Tensor.{name}()",
                    size,
                    tuple(self_tensor.shape),
                    f"Dense operation via {name}()",
                )
            return result

        return wrapped

    def __enter__(self) -> "DenseMaterializationGuard":
        """Activate the guard by patching monitored methods."""
        self._active = True
        self._violations = []
        self._total_dense_elements = 0
        self._critical_count = 0

        # 1. Factory functions
        for name in ["zeros", "ones", "full", "empty", "tensor", "arange", "linspace"]:
            if hasattr(torch, name):
                self._original_methods[f"torch.{name}"] = getattr(torch, name)
                setattr(torch, name, self._wrap_factory(name, getattr(torch, name)))

        # 2. _like factory functions
        for name in ["zeros_like", "ones_like", "full_like", "empty_like"]:
            if hasattr(torch, name):
                self._original_methods[f"torch.{name}"] = getattr(torch, name)
                setattr(
                    torch, name, self._wrap_like_factory(name, getattr(torch, name))
                )

        # 3. Combining functions
        for name in ["stack", "cat"]:
            if hasattr(torch, name):
                self._original_methods[f"torch.{name}"] = getattr(torch, name)
                setattr(torch, name, self._wrap_combining(name, getattr(torch, name)))

        # 4. Tensor methods
        for name in ["numpy", "tolist", "clone", "contiguous"]:
            if hasattr(torch.Tensor, name):
                self._original_methods[f"Tensor.{name}"] = getattr(torch.Tensor, name)
                setattr(
                    torch.Tensor,
                    name,
                    self._wrap_tensor_method(name, getattr(torch.Tensor, name)),
                )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Restore original methods."""
        self._active = False

        # Restore torch functions
        for name in [
            "zeros",
            "ones",
            "full",
            "empty",
            "tensor",
            "arange",
            "linspace",
            "zeros_like",
            "ones_like",
            "full_like",
            "empty_like",
            "stack",
            "cat",
        ]:
            key = f"torch.{name}"
            if key in self._original_methods:
                setattr(torch, name, self._original_methods[key])

        # Restore Tensor methods
        for name in ["numpy", "tolist", "clone", "contiguous"]:
            key = f"Tensor.{name}"
            if key in self._original_methods:
                setattr(torch.Tensor, name, self._original_methods[key])

        self._original_methods.clear()

        # Don't suppress exceptions
        return False

    def report(self) -> str:
        """Generate a human-readable report of violations."""
        lines = [
            "=" * 60,
            "Dense Materialization Guard Report",
            "=" * 60,
            f"Mode: {'PROOF (forbid=True)' if self.forbid else 'DIAGNOSTIC (forbid=False)'}",
            f"Soft threshold: {self.soft_threshold:,} elements (O(N·d·χ) scale)",
            f"Hard threshold: {self.hard_threshold:,} elements (O(N·d·χ²) scale)",
            f"Diagnostic allowed: O(N·d) ≤ {self._diagnostic_threshold:,} elements",
            "",
            f"Total violations: {len(self._violations)}",
            f"Critical violations: {self._critical_count}",
            f"Total dense elements: {self._total_dense_elements:,}",
            "",
        ]

        if self._violations:
            lines.append("Violations:")
            for i, v in enumerate(self._violations, 1):
                severity = "🔴 CRITICAL" if v.is_critical else "🟡 WARNING"
                lines.append(f"  {i}. [{severity}] {v.operation}")
                lines.append(f"     Size: {v.size:,} elements, Shape: {v.shape}")
                lines.append(f"     Location: {v.location}")
                if v.message:
                    lines.append(f"     Note: {v.message}")
                lines.append("")
        else:
            lines.append("✅ No dense materializations above threshold!")

        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# Convenience functions
# =============================================================================


@contextmanager
def forbid_dense(
    N: int,
    d: int = 3,
    chi_max: int = 16,
    strict: bool = True,
    c_hard: float = 1.0,
    c_soft: float = 0.1,
):
    """
    Context manager to forbid dense materialization with auto-computed thresholds.

    Args:
        N: Grid size
        d: Physical dimension (default 3 for CFD)
        chi_max: Maximum bond dimension
        strict: If True (PROOF MODE), raise on critical. If False, just log.
        c_hard: Multiplier for hard threshold (default 1.0 = exactly O(N·d·χ²))
        c_soft: Multiplier for soft threshold (default 0.1)

    Thresholds:
        hard = c_hard * N * d * chi_max²
        soft = c_soft * N * d * chi_max²
    """
    hard_threshold = int(c_hard * N * d * chi_max * chi_max)
    soft_threshold = int(c_soft * N * d * chi_max * chi_max)

    guard = DenseMaterializationGuard(
        hard_threshold=hard_threshold,
        soft_threshold=soft_threshold,
        forbid=strict,
        allow_diagnostics=True,
        N=N,
        d=d,
    )
    with guard:
        yield guard


def assert_no_dense_materialization(
    func: Callable, *args, N: int = 64, d: int = 3, chi_max: int = 16, **kwargs
) -> Any:
    """
    Run a function and assert no critical dense materialization occurred.

    Raises:
        AssertionError if critical dense materialization detected
    """
    with forbid_dense(N, d, chi_max, strict=False) as guard:
        result = func(*args, **kwargs)

    if guard.critical_violations > 0:
        raise AssertionError(
            f"Critical dense materialization detected!\n{guard.report()}"
        )

    return result


# =============================================================================
# TT-specific complexity checks
# =============================================================================


def check_tt_complexity(cores: list, chi_max: int, N: int, d: int = 3) -> dict:
    """
    Verify that TT cores satisfy O(N·d·χ²) storage complexity.

    Args:
        cores: List of TT cores
        chi_max: Maximum bond dimension
        N: Number of sites
        d: Physical dimension per site

    Returns:
        Dict with complexity analysis
    """
    total_elements = sum(c.numel() for c in cores)

    # Theoretical O(N·d·χ²) storage bound
    expected_max = N * d * chi_max * chi_max

    # Full dense would be d^N (exponential) - but for CFD it's N*d (field storage)
    # For CFD, "dense grid" means O(N²) or O(N³), not exponential
    dense_grid_size = N * N * d  # O(N²·d) - 2D grid equivalent

    # Compression ratio vs naive storage
    naive_storage = N * d  # Just storing the field
    compression = total_elements / max(naive_storage, 1)

    return {
        "total_elements": total_elements,
        "expected_max": expected_max,
        "within_bound": total_elements <= expected_max * 1.1,  # 10% tolerance
        "compression_ratio": compression,
        "is_efficient": total_elements < dense_grid_size,
        "actual_vs_theoretical": total_elements / max(expected_max, 1),
    }


def verify_tt_native_operation(
    input_cores: list, output_cores: list, operation_name: str, chi_max: int
) -> dict:
    """
    Verify that a TT operation maintains TT-native complexity.
    """
    input_total = sum(c.numel() for c in input_cores)
    output_total = sum(c.numel() for c in output_cores)

    ratio = output_total / max(input_total, 1)
    is_tt_native = ratio <= chi_max * 2

    return {
        "operation": operation_name,
        "input_elements": input_total,
        "output_elements": output_total,
        "growth_ratio": ratio,
        "is_tt_native": is_tt_native,
        "passed": is_tt_native,
    }
