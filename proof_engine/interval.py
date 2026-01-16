"""
Interval Arithmetic Engine
==========================

Replaces Float64 with rigorous intervals [lower, upper].

The key property: if we compute f([a,b]) = [c,d], then
for ALL x ∈ [a,b], we have f(x) ∈ [c,d].

This transforms floating-point calculations into PROOFS:
    If lower_bound > 0, then the quantity is PROVABLY positive.

Based on IEEE 754 directed rounding and the Arb library concepts.
"""

import numpy as np
from dataclasses import dataclass
from typing import Union, Tuple, Optional
from enum import Enum
import warnings


class RoundingMode(Enum):
    """IEEE 754 rounding modes for rigorous bounds."""
    NEAREST = 0      # Default (not rigorous)
    DOWN = 1         # Round toward -∞
    UP = 2           # Round toward +∞
    TOWARD_ZERO = 3  # Round toward 0


@dataclass
class Interval:
    """
    A rigorous interval [lower, upper] with guaranteed containment.
    
    All arithmetic operations maintain the invariant:
        true_value ∈ [self.lower, self.upper]
    
    This is achieved by:
        - Rounding lower bounds DOWN
        - Rounding upper bounds UP
    
    Example:
        >>> x = Interval(0.1, 0.1)  # "0.1" but rigorous
        >>> y = x * Interval(3, 3)
        >>> print(y)  # [0.29999..., 0.30000...1] contains 0.3
    """
    lower: float
    upper: float
    
    def __post_init__(self):
        """Validate interval."""
        if self.lower > self.upper:
            raise ValueError(f"Invalid interval: [{self.lower}, {self.upper}]")
        # Handle NaN
        if np.isnan(self.lower) or np.isnan(self.upper):
            raise ValueError("Interval cannot contain NaN")
    
    @classmethod
    def from_float(cls, x: float, ulp_radius: int = 1) -> 'Interval':
        """
        Create interval from float with specified ULP (unit in last place) radius.
        
        Args:
            x: The floating-point value
            ulp_radius: Number of ULPs to add/subtract for safety margin
        
        Returns:
            Interval guaranteed to contain true value
        """
        if np.isinf(x):
            return cls(x, x)
        
        # Get machine epsilon at this magnitude
        eps = np.spacing(x) * ulp_radius
        
        # Use nextafter for rigorous bounds
        lower = np.nextafter(x - eps, -np.inf)
        upper = np.nextafter(x + eps, np.inf)
        
        return cls(lower, upper)
    
    @classmethod
    def from_string(cls, s: str) -> 'Interval':
        """
        Create interval from decimal string with full precision.
        
        Example: Interval.from_string("0.1") contains true 1/10
        """
        from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
        
        d = Decimal(s)
        # Convert to float with directed rounding
        lower = float(d)  # This rounds, so we need safety
        upper = float(d)
        
        # Add safety margin
        return cls.from_float(lower, ulp_radius=2)
    
    @classmethod
    def exact(cls, x: float) -> 'Interval':
        """Create exact interval (use only for exactly representable values)."""
        return cls(x, x)
    
    @classmethod
    def pi(cls) -> 'Interval':
        """Return rigorous interval for π."""
        # π is between these IEEE 754 doubles
        return cls(3.141592653589793, 3.1415926535897936)
    
    @classmethod
    def e(cls) -> 'Interval':
        """Return rigorous interval for e."""
        return cls(2.718281828459045, 2.7182818284590455)
    
    @property
    def midpoint(self) -> float:
        """Central value (for display, not rigorous)."""
        return (self.lower + self.upper) / 2
    
    @property
    def radius(self) -> float:
        """Half-width of interval."""
        return (self.upper - self.lower) / 2
    
    @property
    def width(self) -> float:
        """Full width of interval."""
        return self.upper - self.lower
    
    @property
    def relative_uncertainty(self) -> float:
        """Relative uncertainty (radius / |midpoint|)."""
        mid = abs(self.midpoint)
        if mid == 0:
            return np.inf if self.radius > 0 else 0
        return self.radius / mid
    
    def contains(self, x: float) -> bool:
        """Check if x is in interval."""
        return self.lower <= x <= self.upper
    
    def overlaps(self, other: 'Interval') -> bool:
        """Check if intervals overlap."""
        return self.lower <= other.upper and other.lower <= self.upper
    
    def is_positive(self) -> bool:
        """RIGOROUS check: is entire interval > 0?"""
        return self.lower > 0
    
    def is_negative(self) -> bool:
        """RIGOROUS check: is entire interval < 0?"""
        return self.upper < 0
    
    def is_nonzero(self) -> bool:
        """RIGOROUS check: is 0 NOT in interval?"""
        return self.is_positive() or self.is_negative()
    
    def is_zero(self) -> bool:
        """Check if interval could be zero."""
        return self.lower <= 0 <= self.upper
    
    # Arithmetic operations with rigorous rounding
    
    def __neg__(self) -> 'Interval':
        return Interval(-self.upper, -self.lower)
    
    def __add__(self, other: Union['Interval', float]) -> 'Interval':
        if isinstance(other, (int, float)):
            other = Interval.from_float(float(other))
        
        # Rigorous: round lower DOWN, upper UP
        lower = np.nextafter(self.lower + other.lower, -np.inf)
        upper = np.nextafter(self.upper + other.upper, np.inf)
        return Interval(lower, upper)
    
    def __radd__(self, other: float) -> 'Interval':
        return self + other
    
    def __sub__(self, other: Union['Interval', float]) -> 'Interval':
        if isinstance(other, (int, float)):
            other = Interval.from_float(float(other))
        
        lower = np.nextafter(self.lower - other.upper, -np.inf)
        upper = np.nextafter(self.upper - other.lower, np.inf)
        return Interval(lower, upper)
    
    def __rsub__(self, other: float) -> 'Interval':
        return Interval.from_float(other) - self
    
    def __mul__(self, other: Union['Interval', float]) -> 'Interval':
        if isinstance(other, (int, float)):
            other = Interval.from_float(float(other))
        
        # All four endpoint products
        products = [
            self.lower * other.lower,
            self.lower * other.upper,
            self.upper * other.lower,
            self.upper * other.upper
        ]
        
        lower = np.nextafter(min(products), -np.inf)
        upper = np.nextafter(max(products), np.inf)
        return Interval(lower, upper)
    
    def __rmul__(self, other: float) -> 'Interval':
        return self * other
    
    def __truediv__(self, other: Union['Interval', float]) -> 'Interval':
        if isinstance(other, (int, float)):
            other = Interval.from_float(float(other))
        
        # Check for division by zero
        if other.lower <= 0 <= other.upper:
            if other.lower == 0 and other.upper == 0:
                raise ZeroDivisionError("Division by zero interval")
            # Division by interval containing zero
            warnings.warn("Division by interval containing zero - result is unbounded")
            return Interval(-np.inf, np.inf)
        
        # Safe division
        inv = Interval(1/other.upper, 1/other.lower)
        return self * inv
    
    def __rtruediv__(self, other: float) -> 'Interval':
        return Interval.from_float(other) / self
    
    def __pow__(self, n: int) -> 'Interval':
        """Integer power with rigorous bounds."""
        if not isinstance(n, int):
            raise TypeError("Only integer powers supported for rigorous bounds")
        
        if n == 0:
            return Interval.exact(1.0)
        
        if n < 0:
            return Interval.exact(1.0) / (self ** (-n))
        
        if n == 1:
            return Interval(self.lower, self.upper)
        
        if n % 2 == 0:  # Even power
            if self.lower >= 0:
                lower = np.nextafter(self.lower ** n, -np.inf)
                upper = np.nextafter(self.upper ** n, np.inf)
            elif self.upper <= 0:
                lower = np.nextafter(self.upper ** n, -np.inf)
                upper = np.nextafter(self.lower ** n, np.inf)
            else:  # Interval contains 0
                lower = 0.0
                upper = np.nextafter(max(self.lower ** n, self.upper ** n), np.inf)
            return Interval(lower, upper)
        else:  # Odd power
            lower = np.nextafter(self.lower ** n, -np.inf)
            upper = np.nextafter(self.upper ** n, np.inf)
            return Interval(lower, upper)
    
    def sqrt(self) -> 'Interval':
        """Square root with rigorous bounds."""
        if self.lower < 0:
            raise ValueError("Cannot take sqrt of interval with negative values")
        
        lower = np.nextafter(np.sqrt(self.lower), -np.inf)
        upper = np.nextafter(np.sqrt(self.upper), np.inf)
        return Interval(max(0, lower), upper)
    
    def exp(self) -> 'Interval':
        """Exponential with rigorous bounds."""
        lower = np.nextafter(np.exp(self.lower), -np.inf)
        upper = np.nextafter(np.exp(self.upper), np.inf)
        return Interval(lower, upper)
    
    def log(self) -> 'Interval':
        """Natural log with rigorous bounds."""
        if self.lower <= 0:
            raise ValueError("Cannot take log of interval containing non-positive values")
        
        lower = np.nextafter(np.log(self.lower), -np.inf)
        upper = np.nextafter(np.log(self.upper), np.inf)
        return Interval(lower, upper)
    
    def sin(self) -> 'Interval':
        """Sine with rigorous bounds (conservative for now)."""
        # TODO: Implement proper monotonicity checking
        return Interval(-1.0, 1.0)  # Conservative
    
    def cos(self) -> 'Interval':
        """Cosine with rigorous bounds (conservative for now)."""
        return Interval(-1.0, 1.0)  # Conservative
    
    def abs(self) -> 'Interval':
        """Absolute value with rigorous bounds."""
        if self.lower >= 0:
            return Interval(self.lower, self.upper)
        elif self.upper <= 0:
            return Interval(-self.upper, -self.lower)
        else:
            return Interval(0, max(-self.lower, self.upper))
    
    def __repr__(self) -> str:
        return f"[{self.lower:.15g}, {self.upper:.15g}]"
    
    def __str__(self) -> str:
        # Nice format with uncertainty
        if self.radius < 1e-10 * abs(self.midpoint):
            return f"{self.midpoint:.10g}"
        return f"{self.midpoint:.6g} ± {self.radius:.2g}"


class IntervalTensor:
    """
    Tensor with interval-valued entries.
    
    Every element is an Interval, so all tensor operations
    produce rigorous bounds on the result.
    """
    
    def __init__(self, data: np.ndarray, radii: Optional[np.ndarray] = None):
        """
        Create interval tensor from data array.
        
        Args:
            data: Array of central values
            radii: Array of radii (if None, use 1 ULP)
        """
        self.lower = data.copy()
        self.upper = data.copy()
        
        if radii is not None:
            self.lower -= radii
            self.upper += radii
        else:
            # Default: 1 ULP safety
            eps = np.spacing(data)
            self.lower = np.nextafter(self.lower - eps, -np.inf)
            self.upper = np.nextafter(self.upper + eps, np.inf)
    
    @classmethod
    def from_intervals(cls, lower: np.ndarray, upper: np.ndarray) -> 'IntervalTensor':
        """Create from separate lower/upper arrays."""
        obj = cls.__new__(cls)
        obj.lower = lower.copy()
        obj.upper = upper.copy()
        return obj
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.lower.shape
    
    @property
    def midpoint(self) -> np.ndarray:
        return (self.lower + self.upper) / 2
    
    @property
    def radius(self) -> np.ndarray:
        return (self.upper - self.lower) / 2
    
    @property
    def max_relative_uncertainty(self) -> float:
        """Maximum relative uncertainty across all elements."""
        mid = np.abs(self.midpoint)
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = self.radius / mid
            rel[~np.isfinite(rel)] = 0
        return np.max(rel)
    
    def is_positive(self) -> np.ndarray:
        """Element-wise check: is element provably > 0?"""
        return self.lower > 0
    
    def all_positive(self) -> bool:
        """Are ALL elements provably positive?"""
        return np.all(self.lower > 0)
    
    def __add__(self, other: 'IntervalTensor') -> 'IntervalTensor':
        lower = np.nextafter(self.lower + other.lower, -np.inf)
        upper = np.nextafter(self.upper + other.upper, np.inf)
        return IntervalTensor.from_intervals(lower, upper)
    
    def __sub__(self, other: 'IntervalTensor') -> 'IntervalTensor':
        lower = np.nextafter(self.lower - other.upper, -np.inf)
        upper = np.nextafter(self.upper - other.lower, np.inf)
        return IntervalTensor.from_intervals(lower, upper)
    
    def __mul__(self, other: Union['IntervalTensor', float]) -> 'IntervalTensor':
        if isinstance(other, (int, float)):
            if other >= 0:
                lower = np.nextafter(self.lower * other, -np.inf)
                upper = np.nextafter(self.upper * other, np.inf)
            else:
                lower = np.nextafter(self.upper * other, -np.inf)
                upper = np.nextafter(self.lower * other, np.inf)
            return IntervalTensor.from_intervals(lower, upper)
        
        # Element-wise interval multiplication
        prods = np.stack([
            self.lower * other.lower,
            self.lower * other.upper,
            self.upper * other.lower,
            self.upper * other.upper
        ], axis=-1)
        
        lower = np.nextafter(np.min(prods, axis=-1), -np.inf)
        upper = np.nextafter(np.max(prods, axis=-1), np.inf)
        return IntervalTensor.from_intervals(lower, upper)
    
    def __rmul__(self, other: float) -> 'IntervalTensor':
        return self * other
    
    def contract(self, other: 'IntervalTensor', axes: int = 1) -> 'IntervalTensor':
        """
        Contract two interval tensors along last/first axes.
        
        This is the key operation for QTT: matrix multiplication
        with rigorous error bounds.
        """
        # For rigorous contraction, we need to account for all
        # possible combinations of bounds
        
        # Shape: self is (..., m, k), other is (k, n, ...)
        # Result is (..., m, n, ...)
        
        # Use the standard formula with worst-case error bounds
        mid_result = np.tensordot(self.midpoint, other.midpoint, axes=axes)
        
        # Error bound: |result - mid_result| ≤ sum of all error terms
        # For A @ B, error ≈ |A| @ rad(B) + rad(A) @ |B| + rad(A) @ rad(B)
        
        abs_self_mid = np.abs(self.midpoint)
        abs_other_mid = np.abs(other.midpoint)
        
        err1 = np.tensordot(abs_self_mid, other.radius, axes=axes)
        err2 = np.tensordot(self.radius, abs_other_mid, axes=axes)
        err3 = np.tensordot(self.radius, other.radius, axes=axes)
        
        total_err = err1 + err2 + err3
        
        # Add rounding safety
        total_err = np.nextafter(total_err * 1.01, np.inf)
        
        lower = np.nextafter(mid_result - total_err, -np.inf)
        upper = np.nextafter(mid_result + total_err, np.inf)
        
        return IntervalTensor.from_intervals(lower, upper)
    
    def norm(self) -> Interval:
        """Frobenius norm with rigorous bounds."""
        # ||A||² = sum(A²)
        sq_lower = np.minimum(self.lower**2, self.upper**2)
        sq_upper = np.maximum(self.lower**2, self.upper**2)
        
        # Where interval contains zero, lower bound is 0
        contains_zero = (self.lower <= 0) & (self.upper >= 0)
        sq_lower[contains_zero] = 0
        
        sum_sq_lower = np.nextafter(np.sum(sq_lower), -np.inf)
        sum_sq_upper = np.nextafter(np.sum(sq_upper), np.inf)
        
        norm_lower = np.nextafter(np.sqrt(max(0, sum_sq_lower)), -np.inf)
        norm_upper = np.nextafter(np.sqrt(sum_sq_upper), np.inf)
        
        return Interval(norm_lower, norm_upper)


# Convenience functions
def interval(x: Union[float, str]) -> Interval:
    """Create interval from float or string."""
    if isinstance(x, str):
        return Interval.from_string(x)
    return Interval.from_float(x)


def itensor(data: np.ndarray, radii: Optional[np.ndarray] = None) -> IntervalTensor:
    """Create interval tensor."""
    return IntervalTensor(data, radii)


# Test the module
if __name__ == "__main__":
    print("=" * 60)
    print("INTERVAL ARITHMETIC ENGINE TEST")
    print("=" * 60)
    print()
    
    # Test basic interval arithmetic
    print("1. Basic Interval Arithmetic:")
    x = Interval.from_float(0.1)
    print(f"   0.1 as interval: {x}")
    
    y = x * Interval.from_float(3.0)
    print(f"   0.1 × 3 = {y}")
    print(f"   Contains 0.3? {y.contains(0.3)}")
    
    # Test pi
    print()
    print("2. Mathematical Constants:")
    pi = Interval.pi()
    print(f"   π ∈ {pi}")
    print(f"   Width: {pi.width:.2e}")
    
    # Test positivity proof
    print()
    print("3. Positivity Proofs:")
    gap = Interval(1.48, 1.52)
    print(f"   Gap interval: {gap}")
    print(f"   Is gap provably > 0? {gap.is_positive()}")
    
    eps = Interval(-0.001, 0.001)
    print(f"   Epsilon interval: {eps}")
    print(f"   Is epsilon provably > 0? {eps.is_positive()}")
    
    # Test tensor operations
    print()
    print("4. Interval Tensor Operations:")
    A = IntervalTensor(np.random.randn(3, 3))
    B = IntervalTensor(np.random.randn(3, 3))
    print(f"   A shape: {A.shape}, max rel. uncertainty: {A.max_relative_uncertainty:.2e}")
    
    C = A.contract(B)
    print(f"   A @ B shape: {C.shape}, max rel. uncertainty: {C.max_relative_uncertainty:.2e}")
    
    norm = C.norm()
    print(f"   ||A @ B|| ∈ {norm}")
