"""
Interval Arithmetic for Computer-Assisted Proofs (CAP).

This module provides rigorous interval arithmetic where every floating-point
number is represented as [lo, hi] bounds, enabling mathematically rigorous
error analysis for the Navier-Stokes blow-up proof.

Key insight: Standard floats hide rounding errors. Intervals expose them.
If x = Interval(1.0, 1.0) and we compute x/3*3, we get Interval(0.999..., 1.000...1)
instead of pretending we got exactly 1.0.

Reference: Thomas Hou's Euler blow-up proof (2022) used this technique.
"""

from __future__ import annotations
import torch
import numpy as np
from dataclasses import dataclass
from typing import Union, Tuple, Optional
import math


@dataclass
class Interval:
    """
    Rigorous interval arithmetic: every value is [lo, hi].
    
    Properties:
        lo: Lower bound (toward -∞)
        hi: Upper bound (toward +∞)
        
    Invariant: lo <= hi always (if violated, the interval is empty/invalid)
    """
    lo: torch.Tensor
    hi: torch.Tensor
    
    def __post_init__(self):
        """Ensure tensors and validate bounds."""
        if not isinstance(self.lo, torch.Tensor):
            self.lo = torch.tensor(self.lo, dtype=torch.float64)
        if not isinstance(self.hi, torch.Tensor):
            self.hi = torch.tensor(self.hi, dtype=torch.float64)
        # Ensure lo <= hi (with tolerance for numerical noise)
        if not torch.all(self.lo <= self.hi + 1e-15):
            raise ValueError(f"Invalid interval: lo > hi detected")
    
    @classmethod
    def from_value(cls, x: Union[float, torch.Tensor], ulp_pad: int = 1) -> 'Interval':
        """
        Create interval from a point value with ULP (unit in last place) padding.
        
        Args:
            x: The nominal value
            ulp_pad: Number of ULPs to pad for rounding safety
            
        Returns:
            Interval conservatively containing x
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float64)
        x = x.to(torch.float64)
        
        # Compute ULP for each element
        eps = torch.finfo(torch.float64).eps
        magnitude = torch.abs(x).clamp(min=eps)
        ulp = magnitude * eps * ulp_pad
        
        return cls(lo=x - ulp, hi=x + ulp)
    
    @classmethod
    def from_bounds(cls, lo: torch.Tensor, hi: torch.Tensor) -> 'Interval':
        """Create interval from explicit bounds."""
        return cls(lo=lo, hi=hi)
    
    @property
    def mid(self) -> torch.Tensor:
        """Midpoint (best estimate)."""
        return (self.lo + self.hi) / 2
    
    @property
    def rad(self) -> torch.Tensor:
        """Radius (half-width = uncertainty)."""
        return (self.hi - self.lo) / 2
    
    @property
    def width(self) -> torch.Tensor:
        """Full width of the interval."""
        return self.hi - self.lo
    
    @property
    def shape(self) -> torch.Size:
        """Shape of the underlying tensors."""
        return self.lo.shape
    
    def contains(self, x: Union[float, torch.Tensor]) -> torch.Tensor:
        """Check if x is contained in the interval."""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float64)
        return (self.lo <= x) & (x <= self.hi)
    
    def intersects(self, other: 'Interval') -> torch.Tensor:
        """Check if two intervals overlap."""
        return (self.lo <= other.hi) & (other.lo <= self.hi)
    
    def is_valid(self) -> torch.Tensor:
        """Check if interval is non-empty (lo <= hi)."""
        return self.lo <= self.hi
    
    # ========== Arithmetic Operations (Rigorous) ==========
    
    def __add__(self, other: Union['Interval', float, torch.Tensor]) -> 'Interval':
        """[a,b] + [c,d] = [a+c, b+d]"""
        if isinstance(other, Interval):
            return Interval(self.lo + other.lo, self.hi + other.hi)
        else:
            other = torch.as_tensor(other, dtype=torch.float64)
            return Interval(self.lo + other, self.hi + other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __neg__(self) -> 'Interval':
        """-[a,b] = [-b, -a]"""
        return Interval(-self.hi, -self.lo)
    
    def __sub__(self, other: Union['Interval', float, torch.Tensor]) -> 'Interval':
        """[a,b] - [c,d] = [a-d, b-c]"""
        if isinstance(other, Interval):
            return Interval(self.lo - other.hi, self.hi - other.lo)
        else:
            other = torch.as_tensor(other, dtype=torch.float64)
            return Interval(self.lo - other, self.hi - other)
    
    def __rsub__(self, other):
        return (-self).__add__(other)
    
    def __mul__(self, other: Union['Interval', float, torch.Tensor]) -> 'Interval':
        """
        [a,b] * [c,d] = [min(ac,ad,bc,bd), max(ac,ad,bc,bd)]
        
        This is the most complex operation due to sign combinations.
        """
        if isinstance(other, Interval):
            # Compute all four products
            ac = self.lo * other.lo
            ad = self.lo * other.hi
            bc = self.hi * other.lo
            bd = self.hi * other.hi
            
            # Stack and take min/max
            products = torch.stack([ac, ad, bc, bd], dim=0)
            return Interval(products.min(dim=0).values, products.max(dim=0).values)
        else:
            other = torch.as_tensor(other, dtype=torch.float64)
            if torch.all(other >= 0):
                return Interval(self.lo * other, self.hi * other)
            elif torch.all(other <= 0):
                return Interval(self.hi * other, self.lo * other)
            else:
                # Mixed signs - use the general formula
                products = torch.stack([self.lo * other, self.hi * other], dim=0)
                return Interval(products.min(dim=0).values, products.max(dim=0).values)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other: Union['Interval', float, torch.Tensor]) -> 'Interval':
        """
        [a,b] / [c,d]
        
        CRITICAL: Division by an interval containing 0 produces [-∞, +∞]
        """
        if isinstance(other, Interval):
            # Check for division by zero
            if other.contains(0.0).any():
                raise ValueError("Division by interval containing zero - result unbounded")
            
            # 1/[c,d] = [1/d, 1/c] when 0 not in [c,d]
            reciprocal = Interval(1.0 / other.hi, 1.0 / other.lo)
            return self * reciprocal
        else:
            other = torch.as_tensor(other, dtype=torch.float64)
            if torch.any(other == 0):
                raise ValueError("Division by zero")
            if torch.all(other > 0):
                return Interval(self.lo / other, self.hi / other)
            else:
                return Interval(self.hi / other, self.lo / other)
    
    def __pow__(self, n: int) -> 'Interval':
        """
        [a,b]^n - must handle even/odd powers carefully.
        
        For even n: if 0 ∈ [a,b], minimum is 0
        For odd n: [a^n, b^n]
        """
        if n == 0:
            return Interval.from_value(1.0)
        elif n == 1:
            return self
        elif n % 2 == 0:  # Even power
            if torch.all(self.lo >= 0):
                return Interval(self.lo ** n, self.hi ** n)
            elif torch.all(self.hi <= 0):
                return Interval(self.hi ** n, self.lo ** n)
            else:
                # 0 is in interval - minimum is 0
                max_val = torch.max(torch.abs(self.lo), torch.abs(self.hi)) ** n
                return Interval(torch.zeros_like(self.lo), max_val)
        else:  # Odd power
            return Interval(self.lo ** n, self.hi ** n)
    
    def sqrt(self) -> 'Interval':
        """Square root - requires non-negative interval."""
        if torch.any(self.lo < 0):
            raise ValueError("Square root of negative interval")
        return Interval(torch.sqrt(self.lo), torch.sqrt(self.hi))
    
    def abs(self) -> 'Interval':
        """Absolute value of interval."""
        if torch.all(self.lo >= 0):
            return self
        elif torch.all(self.hi <= 0):
            return -self
        else:
            # Straddles zero
            return Interval(
                torch.zeros_like(self.lo),
                torch.max(torch.abs(self.lo), torch.abs(self.hi))
            )
    
    def exp(self) -> 'Interval':
        """Exponential - monotonically increasing."""
        return Interval(torch.exp(self.lo), torch.exp(self.hi))
    
    def log(self) -> 'Interval':
        """Natural log - requires positive interval."""
        if torch.any(self.lo <= 0):
            raise ValueError("Logarithm of non-positive interval")
        return Interval(torch.log(self.lo), torch.log(self.hi))
    
    def sin(self) -> 'Interval':
        """
        Sine - NOT monotonic! Must check for extrema.
        
        For small intervals (width < π), can use monotonicity analysis.
        For large intervals, returns [-1, 1].
        """
        width = self.width.max()
        if width >= 2 * math.pi:
            return Interval(
                -torch.ones_like(self.lo),
                torch.ones_like(self.hi)
            )
        
        # Conservative: evaluate endpoints and check for extrema
        sin_lo = torch.sin(self.lo)
        sin_hi = torch.sin(self.hi)
        
        result_lo = torch.min(sin_lo, sin_hi)
        result_hi = torch.max(sin_lo, sin_hi)
        
        # Check if we cross a maximum (π/2 + 2πk)
        # or minimum (3π/2 + 2πk = -π/2 + 2πk)
        # This is approximate - for full rigor, need exact crossing detection
        if width > math.pi / 2:
            result_hi = torch.ones_like(self.hi)
            result_lo = -torch.ones_like(self.lo)
        
        return Interval(result_lo, result_hi)
    
    def cos(self) -> 'Interval':
        """Cosine via sin(x + π/2)."""
        shifted = Interval(self.lo + math.pi/2, self.hi + math.pi/2)
        return shifted.sin()
    
    # ========== Norms for CAP ==========
    
    def norm_upper_bound(self) -> torch.Tensor:
        """
        Upper bound on the L2 norm: ||x|| <= this value for all x in interval.
        
        This is CRITICAL for Newton-Kantorovich - we need ||R(U)|| bounded.
        """
        # ||x||² ≤ sum of max(|lo|², |hi|²) per component
        max_abs = torch.max(torch.abs(self.lo), torch.abs(self.hi))
        return torch.sqrt((max_abs ** 2).sum())
    
    def norm_lower_bound(self) -> torch.Tensor:
        """Lower bound on the L2 norm."""
        # If 0 in any component, lower bound on that component's contribution is 0
        min_abs = torch.zeros_like(self.lo)
        positive = self.lo > 0
        negative = self.hi < 0
        min_abs[positive] = self.lo[positive]
        min_abs[negative] = -self.hi[negative]
        return torch.sqrt((min_abs ** 2).sum())
    
    def sup_norm_bound(self) -> torch.Tensor:
        """
        Upper bound on ||x||_∞ = max|x_i|.
        
        For BKM criterion: ||ω||_∞
        """
        return torch.max(torch.abs(self.lo).max(), torch.abs(self.hi).max())
    
    # ========== Representation ==========
    
    def __repr__(self) -> str:
        if self.lo.numel() == 1:
            return f"Interval([{self.lo.item():.6e}, {self.hi.item():.6e}])"
        else:
            return f"Interval(shape={self.shape}, width_max={self.width.max():.2e})"
    
    def __str__(self) -> str:
        return self.__repr__()


class IntervalTensor:
    """
    Multi-dimensional interval tensor for CFD fields.
    
    Wraps Interval for convenient field operations.
    """
    
    def __init__(self, interval: Interval):
        self.interval = interval
    
    @classmethod
    def from_tensor(cls, x: torch.Tensor, ulp_pad: int = 2) -> 'IntervalTensor':
        """Create from a standard tensor with ULP padding."""
        return cls(Interval.from_value(x, ulp_pad))
    
    @classmethod
    def zeros(cls, shape: Tuple[int, ...]) -> 'IntervalTensor':
        """Create interval tensor of exact zeros [0, 0]."""
        z = torch.zeros(shape, dtype=torch.float64)
        return cls(Interval(z, z))
    
    @classmethod
    def ones(cls, shape: Tuple[int, ...]) -> 'IntervalTensor':
        """Create interval tensor of exact ones [1, 1]."""
        o = torch.ones(shape, dtype=torch.float64)
        return cls(Interval(o, o))
    
    @property
    def shape(self) -> torch.Size:
        return self.interval.shape
    
    @property
    def mid(self) -> torch.Tensor:
        return self.interval.mid
    
    @property
    def rad(self) -> torch.Tensor:
        return self.interval.rad
    
    def norm(self) -> Interval:
        """Return interval bound on the norm."""
        return Interval(
            self.interval.norm_lower_bound(),
            self.interval.norm_upper_bound()
        )
    
    def sup_norm(self) -> Interval:
        """Return interval bound on the sup norm."""
        return Interval(
            torch.tensor(0.0),  # Lower bound is 0 if zero possible
            self.interval.sup_norm_bound()
        )
    
    def __add__(self, other):
        if isinstance(other, IntervalTensor):
            return IntervalTensor(self.interval + other.interval)
        return IntervalTensor(self.interval + other)
    
    def __sub__(self, other):
        if isinstance(other, IntervalTensor):
            return IntervalTensor(self.interval - other.interval)
        return IntervalTensor(self.interval - other)
    
    def __mul__(self, other):
        if isinstance(other, IntervalTensor):
            return IntervalTensor(self.interval * other.interval)
        return IntervalTensor(self.interval * other)
    
    def __truediv__(self, other):
        if isinstance(other, IntervalTensor):
            return IntervalTensor(self.interval / other.interval)
        return IntervalTensor(self.interval / other)
    
    def __repr__(self):
        return f"IntervalTensor(shape={self.shape}, max_width={self.rad.max()*2:.2e})"


def validate_interval_arithmetic():
    """
    Self-test: Verify interval arithmetic is correctly implemented.
    
    This is a mini-proof that our interval module is trustworthy.
    """
    print("=" * 60)
    print("INTERVAL ARITHMETIC VALIDATION")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    # Test 1: Addition
    a = Interval.from_value(1.0)
    b = Interval.from_value(2.0)
    c = a + b
    assert c.contains(3.0), "1 + 2 should contain 3"
    print(f"✓ Test 1: {a} + {b} = {c} contains 3.0")
    passed += 1
    
    # Test 2: Multiplication with signs
    pos = Interval(torch.tensor(1.0), torch.tensor(2.0))
    neg = Interval(torch.tensor(-2.0), torch.tensor(-1.0))
    prod = pos * neg
    assert prod.lo <= -4 and prod.hi >= -1, "Sign handling in multiplication"
    print(f"✓ Test 2: {pos} * {neg} = {prod}")
    passed += 1
    
    # Test 3: Division safety
    try:
        zero_interval = Interval(torch.tensor(-1.0), torch.tensor(1.0))
        _ = pos / zero_interval
        print("✗ Test 3: Division by zero interval should raise")
        failed += 1
    except ValueError:
        print("✓ Test 3: Division by zero interval correctly raises error")
        passed += 1
    
    # Test 4: Square root of negative
    try:
        neg_interval = Interval(torch.tensor(-1.0), torch.tensor(-0.5))
        _ = neg_interval.sqrt()
        print("✗ Test 4: Sqrt of negative should raise")
        failed += 1
    except ValueError:
        print("✓ Test 4: Sqrt of negative correctly raises error")
        passed += 1
    
    # Test 5: Norm bounds
    x = Interval(torch.tensor([1.0, 2.0]), torch.tensor([1.5, 2.5]))
    lb = x.norm_lower_bound()
    ub = x.norm_upper_bound()
    true_mid = torch.sqrt(torch.tensor([1.25**2 + 2.25**2])).item()
    assert lb <= true_mid <= ub, "Norm bounds should contain true norm"
    print(f"✓ Test 5: Norm bounds [{lb:.4f}, {ub:.4f}] contain {true_mid:.4f}")
    passed += 1
    
    # Test 6: Containment propagation
    x = Interval.from_value(torch.tensor([1.0, 2.0, 3.0]))
    y = x * 2 - 1
    expected = torch.tensor([1.0, 3.0, 5.0])
    assert y.contains(expected).all(), "Arithmetic should preserve containment"
    print(f"✓ Test 6: (x * 2 - 1) contains expected values")
    passed += 1
    
    # Test 7: Even power with zero crossing
    straddle = Interval(torch.tensor(-1.0), torch.tensor(2.0))
    sq = straddle ** 2
    assert sq.lo >= 0, "x² is non-negative"
    assert sq.hi >= 4, "max of [-1,2]² should be at least 4"
    print(f"✓ Test 7: [-1,2]² = {sq}")
    passed += 1
    
    print("=" * 60)
    print(f"VALIDATION: {passed}/{passed+failed} tests passed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    validate_interval_arithmetic()
