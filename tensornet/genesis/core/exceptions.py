"""
TENSOR GENESIS Exception Hierarchy

Custom exceptions with informative messages for all Genesis primitives.
Each exception includes:
- Clear description of what went wrong
- Context about the operation that failed
- Suggestions for resolution when applicable

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class GenesisError(Exception):
    """Base exception for all Genesis errors.
    
    All Genesis exceptions inherit from this class, allowing users to
    catch all Genesis-related errors with a single except clause.
    
    Attributes:
        message: Human-readable error description.
        context: Dictionary of contextual information.
        suggestion: Optional suggestion for resolution.
        
    Example:
        >>> try:
        ...     result = sinkhorn_qtt(source, target)
        ... except GenesisError as e:
        ...     print(f"Genesis error: {e}")
        ...     print(f"Context: {e.context}")
    """
    
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
    ):
        """Initialize GenesisError.
        
        Args:
            message: Error description.
            context: Contextual information dictionary.
            suggestion: Resolution suggestion.
        """
        self.message = message
        self.context = context or {}
        self.suggestion = suggestion
        
        # Build full message
        full_message = message
        if context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in context.items())
            full_message += f" (context: {ctx_str})"
        if suggestion:
            full_message += f"\n  Suggestion: {suggestion}"
        
        super().__init__(full_message)


class QTTRankError(GenesisError):
    """Error related to QTT tensor rank issues.
    
    Raised when:
    - Rank exceeds maximum allowed value
    - Rank explosion during computation
    - Truncation leads to unacceptable error
    
    Attributes:
        current_rank: The rank that caused the error.
        max_rank: Maximum allowed rank (if applicable).
        operation: Operation that caused rank explosion.
    """
    
    def __init__(
        self,
        message: str,
        current_rank: Optional[int] = None,
        max_rank: Optional[int] = None,
        operation: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize QTTRankError.
        
        Args:
            message: Error description.
            current_rank: Current tensor rank.
            max_rank: Maximum allowed rank.
            operation: Operation that caused the error.
            **kwargs: Additional context.
        """
        context = kwargs
        if current_rank is not None:
            context["current_rank"] = current_rank
        if max_rank is not None:
            context["max_rank"] = max_rank
        if operation is not None:
            context["operation"] = operation
        
        suggestion = None
        if current_rank and max_rank and current_rank > max_rank:
            suggestion = (
                f"Try increasing max_rank (currently {max_rank}) or "
                "use more aggressive truncation with higher tolerance."
            )
        
        super().__init__(message, context, suggestion)
        self.current_rank = current_rank
        self.max_rank = max_rank
        self.operation = operation


class ConvergenceError(GenesisError):
    """Error when an iterative algorithm fails to converge.
    
    Raised when:
    - Sinkhorn iterations don't converge
    - Optimization fails to reach tolerance
    - Fixed-point iteration doesn't stabilize
    
    Attributes:
        iterations: Number of iterations attempted.
        residual: Final residual/error value.
        tolerance: Target tolerance.
        algorithm: Name of the algorithm.
    """
    
    def __init__(
        self,
        message: str,
        iterations: Optional[int] = None,
        residual: Optional[float] = None,
        tolerance: Optional[float] = None,
        algorithm: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize ConvergenceError.
        
        Args:
            message: Error description.
            iterations: Iterations attempted.
            residual: Final residual value.
            tolerance: Target tolerance.
            algorithm: Algorithm name.
            **kwargs: Additional context.
        """
        context = kwargs
        if iterations is not None:
            context["iterations"] = iterations
        if residual is not None:
            context["residual"] = f"{residual:.2e}"
        if tolerance is not None:
            context["tolerance"] = f"{tolerance:.2e}"
        if algorithm is not None:
            context["algorithm"] = algorithm
        
        suggestion = None
        if residual and tolerance and residual > tolerance:
            ratio = residual / tolerance
            if ratio > 100:
                suggestion = (
                    "Residual is very far from tolerance. Consider: "
                    "(1) increasing max_iterations, "
                    "(2) relaxing tolerance, "
                    "(3) using preconditioning, "
                    "(4) checking input conditioning."
                )
            else:
                suggestion = (
                    f"Residual is {ratio:.1f}× tolerance. "
                    "Try increasing max_iterations or relaxing tolerance."
                )
        
        super().__init__(message, context, suggestion)
        self.iterations = iterations
        self.residual = residual
        self.tolerance = tolerance
        self.algorithm = algorithm


class DimensionMismatchError(GenesisError):
    """Error when tensor/array dimensions don't match.
    
    Raised when:
    - Matrix dimensions incompatible for multiplication
    - QTT cores have mismatched bond dimensions
    - Distribution lengths don't match cost matrix
    
    Attributes:
        expected: Expected shape or dimensions.
        actual: Actual shape or dimensions.
        operand: Name of the operand with wrong dimensions.
    """
    
    def __init__(
        self,
        message: str,
        expected: Optional[Union[Tuple[int, ...], int]] = None,
        actual: Optional[Union[Tuple[int, ...], int]] = None,
        operand: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize DimensionMismatchError.
        
        Args:
            message: Error description.
            expected: Expected dimensions.
            actual: Actual dimensions.
            operand: Name of problematic operand.
            **kwargs: Additional context.
        """
        context = kwargs
        if expected is not None:
            context["expected"] = expected
        if actual is not None:
            context["actual"] = actual
        if operand is not None:
            context["operand"] = operand
        
        suggestion = None
        if expected and actual:
            suggestion = (
                f"Reshape {operand or 'input'} from {actual} to {expected}, "
                "or verify the operation's dimension requirements."
            )
        
        super().__init__(message, context, suggestion)
        self.expected = expected
        self.actual = actual
        self.operand = operand


class NumericalInstabilityError(GenesisError):
    """Error when numerical instability is detected.
    
    Raised when:
    - NaN or Inf values appear in computation
    - Matrix is nearly singular
    - Overflow/underflow detected
    - Condition number too high
    
    Attributes:
        values_affected: Count of problematic values.
        location: Where the instability was detected.
        condition_number: Matrix condition number (if applicable).
    """
    
    def __init__(
        self,
        message: str,
        values_affected: Optional[int] = None,
        location: Optional[str] = None,
        condition_number: Optional[float] = None,
        has_nan: bool = False,
        has_inf: bool = False,
        **kwargs: Any,
    ):
        """Initialize NumericalInstabilityError.
        
        Args:
            message: Error description.
            values_affected: Number of bad values.
            location: Where issue was detected.
            condition_number: Matrix condition number.
            has_nan: Whether NaN values are present.
            has_inf: Whether Inf values are present.
            **kwargs: Additional context.
        """
        context = kwargs
        if values_affected is not None:
            context["values_affected"] = values_affected
        if location is not None:
            context["location"] = location
        if condition_number is not None:
            context["condition_number"] = f"{condition_number:.2e}"
        if has_nan:
            context["has_nan"] = True
        if has_inf:
            context["has_inf"] = True
        
        suggestions = []
        if has_nan or has_inf:
            suggestions.append("Check inputs for extreme values")
        if condition_number and condition_number > 1e10:
            suggestions.append("Add regularization to improve conditioning")
        suggestions.append("Consider using higher precision (float64)")
        suggestion = "; ".join(suggestions) if suggestions else None
        
        super().__init__(message, context, suggestion)
        self.values_affected = values_affected
        self.location = location
        self.condition_number = condition_number
        self.has_nan = has_nan
        self.has_inf = has_inf


class MemoryBudgetExceededError(GenesisError):
    """Error when memory usage exceeds budget.
    
    Raised when:
    - Dense materialization would exceed limit
    - QTT compression insufficient
    - Intermediate results too large
    
    Attributes:
        required_bytes: Memory required for operation.
        budget_bytes: Maximum allowed memory.
        operation: Operation that would exceed budget.
    """
    
    def __init__(
        self,
        message: str,
        required_bytes: Optional[int] = None,
        budget_bytes: Optional[int] = None,
        operation: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize MemoryBudgetExceededError.
        
        Args:
            message: Error description.
            required_bytes: Required memory in bytes.
            budget_bytes: Budget in bytes.
            operation: Operation name.
            **kwargs: Additional context.
        """
        context = kwargs
        if required_bytes is not None:
            context["required"] = _format_bytes(required_bytes)
        if budget_bytes is not None:
            context["budget"] = _format_bytes(budget_bytes)
        if operation is not None:
            context["operation"] = operation
        
        suggestion = None
        if required_bytes and budget_bytes:
            ratio = required_bytes / budget_bytes
            if ratio > 10:
                suggestion = (
                    f"Operation needs {ratio:.0f}× more memory than budget. "
                    "Consider chunked processing or streaming algorithms."
                )
            else:
                suggestion = (
                    f"Operation needs {ratio:.1f}× budget. "
                    "Try increasing budget or using more aggressive compression."
                )
        
        super().__init__(message, context, suggestion)
        self.required_bytes = required_bytes
        self.budget_bytes = budget_bytes
        self.operation = operation


class InvalidInputError(GenesisError):
    """Error for invalid input parameters.
    
    Raised when:
    - Parameter out of valid range
    - Invalid type provided
    - Constraint violation
    
    Attributes:
        parameter: Name of invalid parameter.
        value: The invalid value.
        constraint: Description of valid values.
    """
    
    def __init__(
        self,
        message: str,
        parameter: Optional[str] = None,
        value: Any = None,
        constraint: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize InvalidInputError.
        
        Args:
            message: Error description.
            parameter: Parameter name.
            value: Invalid value.
            constraint: Valid value description.
            **kwargs: Additional context.
        """
        context = kwargs
        if parameter is not None:
            context["parameter"] = parameter
        if value is not None:
            context["value"] = repr(value)[:50]
        if constraint is not None:
            context["constraint"] = constraint
        
        suggestion = None
        if constraint:
            suggestion = f"Ensure {parameter or 'input'} satisfies: {constraint}"
        
        super().__init__(message, context, suggestion)
        self.parameter = parameter
        self.value = value
        self.constraint = constraint


class CompressionError(GenesisError):
    """Error related to QTT compression.
    
    Raised when:
    - Compression fails to achieve target ratio
    - Decompression produces incorrect results
    - Round-trip error exceeds tolerance
    
    Attributes:
        target_compression: Target compression ratio.
        achieved_compression: Actually achieved ratio.
        round_trip_error: Error from compress/decompress.
    """
    
    def __init__(
        self,
        message: str,
        target_compression: Optional[float] = None,
        achieved_compression: Optional[float] = None,
        round_trip_error: Optional[float] = None,
        **kwargs: Any,
    ):
        """Initialize CompressionError.
        
        Args:
            message: Error description.
            target_compression: Target ratio.
            achieved_compression: Achieved ratio.
            round_trip_error: Round-trip error.
            **kwargs: Additional context.
        """
        context = kwargs
        if target_compression is not None:
            context["target_compression"] = f"{target_compression:.1f}×"
        if achieved_compression is not None:
            context["achieved_compression"] = f"{achieved_compression:.1f}×"
        if round_trip_error is not None:
            context["round_trip_error"] = f"{round_trip_error:.2e}"
        
        suggestion = None
        if target_compression and achieved_compression:
            if achieved_compression < target_compression:
                suggestion = (
                    "Try increasing max_rank or reducing truncation tolerance "
                    "to improve compression ratio."
                )
        
        super().__init__(message, context, suggestion)
        self.target_compression = target_compression
        self.achieved_compression = achieved_compression
        self.round_trip_error = round_trip_error


# Layer-specific exceptions

class OTError(GenesisError):
    """Error in Optimal Transport (Layer 20) operations."""
    pass


class SGWError(GenesisError):
    """Error in Spectral Graph Wavelets (Layer 21) operations."""
    pass


class TropicalError(GenesisError):
    """Error in Tropical Geometry (Layer 23) operations."""
    pass


class RKHSError(GenesisError):
    """Error in RKHS/Kernel (Layer 24) operations."""
    pass


class TopologyError(GenesisError):
    """Error in Persistent Homology (Layer 25) operations."""
    pass


class GAError(GenesisError):
    """Error in Geometric Algebra (Layer 26) operations."""
    pass


# Utility functions

def _format_bytes(n_bytes: int) -> str:
    """Format byte count as human-readable string.
    
    Args:
        n_bytes: Number of bytes.
        
    Returns:
        Formatted string (e.g., "1.5 GB").
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


def check_finite(
    arr: np.ndarray,
    name: str = "array",
    location: Optional[str] = None,
) -> None:
    """Check array for NaN/Inf values and raise if found.
    
    Args:
        arr: NumPy array to check.
        name: Name for error messages.
        location: Location description for errors.
        
    Raises:
        NumericalInstabilityError: If non-finite values found.
    """
    if not np.isfinite(arr).all():
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        raise NumericalInstabilityError(
            f"Non-finite values in {name}",
            values_affected=int(n_nan + n_inf),
            location=location,
            has_nan=n_nan > 0,
            has_inf=n_inf > 0,
        )


def check_shape(
    arr: np.ndarray,
    expected_shape: Tuple[int, ...],
    name: str = "array",
) -> None:
    """Check array shape and raise if mismatched.
    
    Args:
        arr: NumPy array to check.
        expected_shape: Expected shape (use -1 for any).
        name: Name for error messages.
        
    Raises:
        DimensionMismatchError: If shapes don't match.
    """
    if len(arr.shape) != len(expected_shape):
        raise DimensionMismatchError(
            f"{name} has wrong number of dimensions",
            expected=expected_shape,
            actual=arr.shape,
            operand=name,
        )
    
    for i, (actual, expected) in enumerate(zip(arr.shape, expected_shape)):
        if expected != -1 and actual != expected:
            raise DimensionMismatchError(
                f"{name} has wrong shape at dimension {i}",
                expected=expected_shape,
                actual=arr.shape,
                operand=name,
            )
