"""
TENSOR GENESIS Input Validation Utilities

Type validation and input checking for Genesis primitives with:
- QTT core structure validation
- Tensor shape verification
- Numerical constraint checking
- Automatic type coercion

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, TypeVar, Union

import numpy as np

from tensornet.genesis.core.exceptions import (
    DimensionMismatchError,
    InvalidInputError,
    NumericalInstabilityError,
    QTTRankError,
)

# Type variables
T = TypeVar('T')
ArrayLike = Union[np.ndarray, List, Tuple]


def validate_qtt_cores(
    cores: List[np.ndarray],
    expected_modes: Optional[int] = None,
    max_rank: Optional[int] = None,
    name: str = "QTT cores",
) -> List[np.ndarray]:
    """Validate QTT tensor cores structure.
    
    Checks that:
    - Cores have correct 3D shape (r_left, mode_size, r_right)
    - Bond dimensions match between adjacent cores
    - First core has r_left = 1
    - Last core has r_right = 1
    - Ranks don't exceed maximum
    
    Args:
        cores: List of QTT cores.
        expected_modes: Expected number of cores (optional).
        max_rank: Maximum allowed rank (optional).
        name: Name for error messages.
        
    Returns:
        Validated cores (same reference).
        
    Raises:
        DimensionMismatchError: If core shapes are invalid.
        QTTRankError: If ranks exceed maximum.
        InvalidInputError: If cores list is empty.
        
    Example:
        >>> cores = [np.random.randn(1, 2, 4), np.random.randn(4, 2, 1)]
        >>> validate_qtt_cores(cores)
    """
    if not cores:
        raise InvalidInputError(
            f"{name} cannot be empty",
            parameter=name,
            value=cores,
            constraint="non-empty list of 3D arrays",
        )
    
    if expected_modes is not None and len(cores) != expected_modes:
        raise DimensionMismatchError(
            f"{name} has wrong number of cores",
            expected=expected_modes,
            actual=len(cores),
            operand=name,
        )
    
    ranks = []
    for i, core in enumerate(cores):
        # Check 3D
        if core.ndim != 3:
            raise DimensionMismatchError(
                f"{name}[{i}] must be 3D (r_left, mode_size, r_right)",
                expected=(None, None, None),
                actual=core.shape,
                operand=f"{name}[{i}]",
            )
        
        r_left, mode_size, r_right = core.shape
        ranks.append(r_right)
        
        # Check boundary conditions
        if i == 0 and r_left != 1:
            raise DimensionMismatchError(
                f"{name}[0] must have r_left=1 (got {r_left})",
                expected=1,
                actual=r_left,
                operand=f"{name}[0] left rank",
            )
        
        if i == len(cores) - 1 and r_right != 1:
            raise DimensionMismatchError(
                f"{name}[-1] must have r_right=1 (got {r_right})",
                expected=1,
                actual=r_right,
                operand=f"{name}[-1] right rank",
            )
        
        # Check adjacent bond dimensions
        if i > 0:
            prev_r_right = cores[i - 1].shape[2]
            if r_left != prev_r_right:
                raise DimensionMismatchError(
                    f"{name}[{i}] left rank doesn't match {name}[{i-1}] right rank",
                    expected=prev_r_right,
                    actual=r_left,
                    operand=f"{name}[{i}] left rank",
                )
        
        # Check max rank
        if max_rank is not None:
            current_max = max(r_left, r_right)
            if current_max > max_rank:
                raise QTTRankError(
                    f"{name}[{i}] rank exceeds maximum",
                    current_rank=current_max,
                    max_rank=max_rank,
                    operation=f"validating {name}",
                )
    
    return cores


def validate_tensor_shape(
    arr: np.ndarray,
    expected_shape: Tuple[int, ...],
    name: str = "tensor",
    allow_broadcast: bool = False,
) -> np.ndarray:
    """Validate tensor shape.
    
    Args:
        arr: Array to validate.
        expected_shape: Expected shape (use -1 for any dimension).
        name: Name for error messages.
        allow_broadcast: Whether to allow broadcastable shapes.
        
    Returns:
        Validated array.
        
    Raises:
        DimensionMismatchError: If shape doesn't match.
        
    Example:
        >>> x = np.zeros((10, 20))
        >>> validate_tensor_shape(x, (10, -1))  # -1 matches any
    """
    if arr.ndim != len(expected_shape):
        raise DimensionMismatchError(
            f"{name} has wrong number of dimensions",
            expected=len(expected_shape),
            actual=arr.ndim,
            operand=name,
        )
    
    for i, (actual, expected) in enumerate(zip(arr.shape, expected_shape)):
        if expected == -1:
            continue  # Wildcard
        
        if allow_broadcast and (actual == 1 or expected == 1):
            continue  # Broadcastable
        
        if actual != expected:
            raise DimensionMismatchError(
                f"{name} has wrong size at dimension {i}",
                expected=expected_shape,
                actual=arr.shape,
                operand=name,
            )
    
    return arr


def validate_positive(
    value: Union[int, float],
    name: str = "value",
    allow_zero: bool = False,
) -> Union[int, float]:
    """Validate that value is positive.
    
    Args:
        value: Value to check.
        name: Name for error messages.
        allow_zero: Whether zero is valid.
        
    Returns:
        Validated value.
        
    Raises:
        InvalidInputError: If value is not positive.
    """
    if allow_zero:
        if value < 0:
            raise InvalidInputError(
                f"{name} must be non-negative",
                parameter=name,
                value=value,
                constraint=">= 0",
            )
    else:
        if value <= 0:
            raise InvalidInputError(
                f"{name} must be positive",
                parameter=name,
                value=value,
                constraint="> 0",
            )
    
    return value


def validate_probability(
    value: Union[float, np.ndarray],
    name: str = "probability",
) -> Union[float, np.ndarray]:
    """Validate probability value(s) in [0, 1].
    
    Args:
        value: Probability or array of probabilities.
        name: Name for error messages.
        
    Returns:
        Validated value.
        
    Raises:
        InvalidInputError: If not in valid range.
    """
    if isinstance(value, np.ndarray):
        if np.any(value < 0) or np.any(value > 1):
            n_invalid = np.sum((value < 0) | (value > 1))
            raise InvalidInputError(
                f"{name} has {n_invalid} values outside [0, 1]",
                parameter=name,
                value=f"array with {n_invalid} invalid values",
                constraint="all values in [0, 1]",
            )
    else:
        if value < 0 or value > 1:
            raise InvalidInputError(
                f"{name} must be in [0, 1]",
                parameter=name,
                value=value,
                constraint="0 <= x <= 1",
            )
    
    return value


def validate_distribution(
    dist: np.ndarray,
    name: str = "distribution",
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Validate probability distribution (sums to 1, non-negative).
    
    Args:
        dist: Distribution array.
        name: Name for error messages.
        tolerance: Tolerance for sum-to-1 check.
        
    Returns:
        Validated (and possibly normalized) distribution.
        
    Raises:
        InvalidInputError: If not a valid distribution.
    """
    if dist.ndim != 1:
        raise DimensionMismatchError(
            f"{name} must be 1D",
            expected=(dist.size,),
            actual=dist.shape,
            operand=name,
        )
    
    if np.any(dist < 0):
        n_negative = np.sum(dist < 0)
        raise InvalidInputError(
            f"{name} has {n_negative} negative values",
            parameter=name,
            constraint="all values >= 0",
        )
    
    total = dist.sum()
    if abs(total - 1.0) > tolerance:
        # Normalize if reasonably close (within 10x of 1)
        if 0.1 < total < 10.0:
            return dist / total
        else:
            raise InvalidInputError(
                f"{name} sums to {total}, not 1",
                parameter=name,
                value=f"sum={total}",
                constraint="sum to 1",
            )
    
    return dist


def validate_dtype(
    arr: np.ndarray,
    expected_dtype: Union[np.dtype, Type, str],
    name: str = "array",
    coerce: bool = False,
) -> np.ndarray:
    """Validate array dtype.
    
    Args:
        arr: Array to check.
        expected_dtype: Expected dtype.
        name: Name for error messages.
        coerce: Whether to convert dtype if mismatched.
        
    Returns:
        Array with correct dtype.
        
    Raises:
        InvalidInputError: If dtype wrong and coerce=False.
    """
    expected = np.dtype(expected_dtype)
    
    if arr.dtype != expected:
        if coerce:
            return arr.astype(expected)
        else:
            raise InvalidInputError(
                f"{name} has wrong dtype",
                parameter=name,
                value=str(arr.dtype),
                constraint=f"dtype={expected}",
            )
    
    return arr


def check_numerical_stability(
    arr: np.ndarray,
    name: str = "array",
    max_value: Optional[float] = None,
    check_finite: bool = True,
    check_condition: bool = False,
) -> np.ndarray:
    """Check array for numerical stability issues.
    
    Args:
        arr: Array to check.
        name: Name for error messages.
        max_value: Maximum allowed absolute value.
        check_finite: Check for NaN/Inf.
        check_condition: Check condition number (for matrices).
        
    Returns:
        Validated array.
        
    Raises:
        NumericalInstabilityError: If stability issues found.
    """
    issues = []
    
    if check_finite:
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        
        if n_nan > 0 or n_inf > 0:
            raise NumericalInstabilityError(
                f"{name} contains non-finite values",
                values_affected=int(n_nan + n_inf),
                location=name,
                has_nan=n_nan > 0,
                has_inf=n_inf > 0,
            )
    
    if max_value is not None:
        n_large = np.sum(np.abs(arr) > max_value)
        if n_large > 0:
            issues.append(f"{n_large} values exceed {max_value}")
    
    if check_condition and arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        try:
            cond = np.linalg.cond(arr)
            if cond > 1e12:
                raise NumericalInstabilityError(
                    f"{name} is ill-conditioned",
                    condition_number=cond,
                    location=name,
                )
        except np.linalg.LinAlgError:
            pass  # Singular matrix - will be caught elsewhere
    
    if issues:
        raise NumericalInstabilityError(
            f"{name} has numerical issues: {'; '.join(issues)}",
            location=name,
        )
    
    return arr


def coerce_array(
    value: ArrayLike,
    dtype: Union[np.dtype, Type, str] = np.float64,
    name: str = "input",
) -> np.ndarray:
    """Convert input to numpy array with specified dtype.
    
    Args:
        value: Input value (array, list, tuple).
        dtype: Target dtype.
        name: Name for error messages.
        
    Returns:
        NumPy array.
        
    Raises:
        InvalidInputError: If conversion fails.
    """
    try:
        arr = np.asarray(value, dtype=dtype)
        return arr
    except (ValueError, TypeError) as e:
        raise InvalidInputError(
            f"Cannot convert {name} to numpy array",
            parameter=name,
            value=type(value).__name__,
            constraint=f"array-like convertible to {dtype}",
        ) from e


def validate_power_of_two(
    value: int,
    name: str = "value",
) -> int:
    """Validate that value is a power of 2.
    
    Args:
        value: Value to check.
        name: Name for error messages.
        
    Returns:
        Validated value.
        
    Raises:
        InvalidInputError: If not power of 2.
    """
    if value <= 0 or (value & (value - 1)) != 0:
        # Find nearest power of 2
        nearest = 1 << (value - 1).bit_length()
        raise InvalidInputError(
            f"{name} must be a power of 2",
            parameter=name,
            value=value,
            constraint=f"power of 2 (nearest: {nearest})",
        )
    return value


def validate_range(
    value: Union[int, float],
    min_val: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
    name: str = "value",
    inclusive: bool = True,
) -> Union[int, float]:
    """Validate value is within specified range.
    
    Args:
        value: Value to check.
        min_val: Minimum value (optional).
        max_val: Maximum value (optional).
        name: Name for error messages.
        inclusive: Whether bounds are inclusive.
        
    Returns:
        Validated value.
        
    Raises:
        InvalidInputError: If out of range.
    """
    if inclusive:
        if min_val is not None and value < min_val:
            raise InvalidInputError(
                f"{name} is below minimum",
                parameter=name,
                value=value,
                constraint=f">= {min_val}",
            )
        if max_val is not None and value > max_val:
            raise InvalidInputError(
                f"{name} is above maximum",
                parameter=name,
                value=value,
                constraint=f"<= {max_val}",
            )
    else:
        if min_val is not None and value <= min_val:
            raise InvalidInputError(
                f"{name} must be greater than {min_val}",
                parameter=name,
                value=value,
                constraint=f"> {min_val}",
            )
        if max_val is not None and value >= max_val:
            raise InvalidInputError(
                f"{name} must be less than {max_val}",
                parameter=name,
                value=value,
                constraint=f"< {max_val}",
            )
    
    return value
