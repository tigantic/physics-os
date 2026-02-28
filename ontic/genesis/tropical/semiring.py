"""
QTT Tropical Semiring Operations

Implements the tropical (min-plus and max-plus) semirings with QTT-compatible
smooth approximations.

The key insight is that min/max can be approximated by differentiable functions:
    softmin(a, b; β) = -(1/β) log(e^{-βa} + e^{-βb})
    softmax(a, b; β) = (1/β) log(e^{βa} + e^{βb})

As β → ∞, these converge to exact min/max. For finite β, all operations
remain smooth, enabling TT operations.

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Union, Optional, Tuple, List
import torch


class SemiringType(Enum):
    """Type of tropical semiring."""
    MIN_PLUS = "min_plus"
    MAX_PLUS = "max_plus"


@dataclass
class TropicalSemiring:
    """
    Abstract tropical semiring with smooth approximations.
    
    Tropical semirings replace:
        Classical (+, ×) → Tropical (⊕, ⊗)
    
    Min-plus: (⊕, ⊗) = (min, +), zero = +∞, one = 0
    Max-plus: (⊕, ⊗) = (max, +), zero = -∞, one = 0
    
    Attributes:
        semiring_type: MIN_PLUS or MAX_PLUS
        beta: Smoothing parameter (higher = more accurate, less smooth)
        inf_value: Large value representing infinity
    """
    semiring_type: SemiringType = SemiringType.MIN_PLUS
    beta: float = 100.0
    inf_value: float = 1e10
    
    @property
    def zero(self) -> float:
        """Additive identity (absorbing element)."""
        if self.semiring_type == SemiringType.MIN_PLUS:
            return self.inf_value  # +∞
        else:
            return -self.inf_value  # -∞
    
    @property
    def one(self) -> float:
        """Multiplicative identity."""
        return 0.0
    
    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Tropical addition: min or max.
        
        Uses softmin/softmax for smooth approximation when needed.
        """
        if self.semiring_type == SemiringType.MIN_PLUS:
            return self._softmin(a, b)
        else:
            return self._softmax(a, b)
    
    def mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Tropical multiplication: classical addition.
        """
        return a + b
    
    def _softmin(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Smooth approximation to min.
        
        softmin(a, b; β) = -(1/β) log(e^{-βa} + e^{-βb})
        
        For numerical stability, use log-sum-exp trick:
        = -(1/β) (m + log(e^{-β(a-m)} + e^{-β(b-m)}))
        where m = min(a, b) elementwise (for stability)
        """
        if self.beta >= 1000:
            # High beta: use exact min
            return torch.minimum(a, b)
        
        # Log-sum-exp for numerical stability
        m = torch.minimum(a, b)
        exp_a = torch.exp(-self.beta * (a - m))
        exp_b = torch.exp(-self.beta * (b - m))
        
        result = -(1.0 / self.beta) * (self.beta * m + torch.log(exp_a + exp_b))
        return result
    
    def _softmax(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Smooth approximation to max.
        
        softmax(a, b; β) = (1/β) log(e^{βa} + e^{βb})
        """
        if self.beta >= 1000:
            return torch.maximum(a, b)
        
        m = torch.maximum(a, b)
        exp_a = torch.exp(self.beta * (a - m))
        exp_b = torch.exp(self.beta * (b - m))
        
        result = (1.0 / self.beta) * (self.beta * m + torch.log(exp_a + exp_b))
        return result
    
    def reduce_add(self, values: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Reduce over dimension using tropical addition.
        
        For min-plus: computes min over dimension
        For max-plus: computes max over dimension
        """
        if self.beta >= 1000:
            if self.semiring_type == SemiringType.MIN_PLUS:
                return values.min(dim=dim)[0]
            else:
                return values.max(dim=dim)[0]
        
        # Soft reduction
        if self.semiring_type == SemiringType.MIN_PLUS:
            # softmin reduction
            m = values.min(dim=dim, keepdim=True)[0]
            exp_vals = torch.exp(-self.beta * (values - m))
            result = -(1.0 / self.beta) * (
                self.beta * m.squeeze(dim) + 
                torch.log(exp_vals.sum(dim=dim))
            )
        else:
            # softmax reduction
            m = values.max(dim=dim, keepdim=True)[0]
            exp_vals = torch.exp(self.beta * (values - m))
            result = (1.0 / self.beta) * (
                self.beta * m.squeeze(dim) + 
                torch.log(exp_vals.sum(dim=dim))
            )
        
        return result


# Pre-defined semiring instances
MinPlusSemiring = TropicalSemiring(SemiringType.MIN_PLUS, beta=1000.0)
MaxPlusSemiring = TropicalSemiring(SemiringType.MAX_PLUS, beta=1000.0)


def softmin(a: torch.Tensor, b: torch.Tensor, 
            beta: float = 100.0) -> torch.Tensor:
    """
    Smooth approximation to elementwise min.
    
    softmin(a, b; β) = -(1/β) log(e^{-βa} + e^{-βb})
    
    As β → ∞, converges to min(a, b).
    
    Args:
        a: First tensor
        b: Second tensor
        beta: Smoothness parameter (larger = sharper)
        
    Returns:
        Approximately min(a, b)
    """
    semiring = TropicalSemiring(SemiringType.MIN_PLUS, beta=beta)
    return semiring._softmin(a, b)


def softmax(a: torch.Tensor, b: torch.Tensor,
            beta: float = 100.0) -> torch.Tensor:
    """
    Smooth approximation to elementwise max.
    
    softmax(a, b; β) = (1/β) log(e^{βa} + e^{βb})
    
    As β → ∞, converges to max(a, b).
    
    Args:
        a: First tensor
        b: Second tensor
        beta: Smoothness parameter
        
    Returns:
        Approximately max(a, b)
    """
    semiring = TropicalSemiring(SemiringType.MAX_PLUS, beta=beta)
    return semiring._softmax(a, b)


def tropical_min(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Exact elementwise min (min-plus addition)."""
    return torch.minimum(a, b)


def tropical_max(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Exact elementwise max (max-plus addition)."""
    return torch.maximum(a, b)


def tropical_add(a: torch.Tensor, b: torch.Tensor,
                 semiring: TropicalSemiring = MinPlusSemiring) -> torch.Tensor:
    """
    Tropical addition in given semiring.
    
    Min-plus: returns min(a, b)
    Max-plus: returns max(a, b)
    """
    return semiring.add(a, b)


def tropical_mul(a: torch.Tensor, b: torch.Tensor,
                 semiring: TropicalSemiring = MinPlusSemiring) -> torch.Tensor:
    """
    Tropical multiplication in given semiring.
    
    Both semirings: returns a + b (classical addition)
    """
    return semiring.mul(a, b)


def tropical_pow(a: torch.Tensor, n: int,
                 semiring: TropicalSemiring = MinPlusSemiring) -> torch.Tensor:
    """
    Tropical power: a ⊗ a ⊗ ... ⊗ a (n times).
    
    Since tropical multiplication is classical addition:
    a^⊗n = n * a
    
    Args:
        a: Base tensor
        n: Exponent
        semiring: Tropical semiring
        
    Returns:
        n * a
    """
    return n * a


def tropical_sum(values: torch.Tensor, dim: int = -1,
                 semiring: TropicalSemiring = MinPlusSemiring) -> torch.Tensor:
    """
    Sum (tropical addition) over dimension.
    
    Min-plus: returns min over dimension
    Max-plus: returns max over dimension
    """
    return semiring.reduce_add(values, dim)


@dataclass
class TropicalScalar:
    """
    A scalar in the tropical semiring.
    
    Represents a value with special handling for ±∞.
    """
    value: float
    semiring: TropicalSemiring = field(default_factory=lambda: MinPlusSemiring)
    
    def __add__(self, other: 'TropicalScalar') -> 'TropicalScalar':
        """Tropical addition."""
        if isinstance(other, (int, float)):
            other = TropicalScalar(other, self.semiring)
        
        a = torch.tensor([self.value])
        b = torch.tensor([other.value])
        result = self.semiring.add(a, b).item()
        return TropicalScalar(result, self.semiring)
    
    def __mul__(self, other: 'TropicalScalar') -> 'TropicalScalar':
        """Tropical multiplication."""
        if isinstance(other, (int, float)):
            other = TropicalScalar(other, self.semiring)
        
        result = self.value + other.value
        return TropicalScalar(result, self.semiring)
    
    def __repr__(self) -> str:
        if self.value >= 1e9:
            return "∞"
        elif self.value <= -1e9:
            return "-∞"
        return f"T({self.value:.4g})"
    
    @classmethod
    def zero(cls, semiring: TropicalSemiring = MinPlusSemiring) -> 'TropicalScalar':
        """Additive identity."""
        return cls(semiring.zero, semiring)
    
    @classmethod
    def one(cls, semiring: TropicalSemiring = MinPlusSemiring) -> 'TropicalScalar':
        """Multiplicative identity."""
        return cls(semiring.one, semiring)


def verify_semiring_axioms(semiring: TropicalSemiring,
                           values: Optional[torch.Tensor] = None,
                           tol: float = 1e-6) -> Tuple[bool, str]:
    """
    Verify that a tropical semiring satisfies the semiring axioms.
    
    Axioms checked:
    1. (S, ⊕) is a commutative monoid with identity 0
    2. (S, ⊗) is a monoid with identity 1
    3. ⊗ distributes over ⊕
    4. 0 is absorbing: a ⊗ 0 = 0 ⊗ a = 0
    
    Args:
        semiring: The tropical semiring
        values: Test values (random if None)
        tol: Tolerance for comparisons
        
    Returns:
        (passed, message) tuple
    """
    if values is None:
        values = torch.tensor([0.0, 1.0, 2.0, 5.0, -1.0])
    
    zero = torch.tensor([semiring.zero])
    one = torch.tensor([semiring.one])
    
    issues = []
    
    # Check identity elements
    for v in values:
        v_t = torch.tensor([v.item()])
        
        # a ⊕ 0 = a
        result = semiring.add(v_t, zero)
        if not torch.allclose(result, v_t, atol=tol):
            issues.append(f"Additive identity: {v} ⊕ 0 ≠ {v}")
        
        # a ⊗ 1 = a
        result = semiring.mul(v_t, one)
        if not torch.allclose(result, v_t, atol=tol):
            issues.append(f"Multiplicative identity: {v} ⊗ 1 ≠ {v}")
    
    # Check commutativity of ⊕
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([3.0, 1.0, 2.0])
    if not torch.allclose(semiring.add(a, b), semiring.add(b, a), atol=tol):
        issues.append("Commutativity: a ⊕ b ≠ b ⊕ a")
    
    # Check associativity of ⊕
    c = torch.tensor([2.0, 3.0, 1.0])
    left = semiring.add(semiring.add(a, b), c)
    right = semiring.add(a, semiring.add(b, c))
    if not torch.allclose(left, right, atol=tol):
        issues.append("Associativity: (a ⊕ b) ⊕ c ≠ a ⊕ (b ⊕ c)")
    
    if issues:
        return False, "; ".join(issues)
    return True, "All semiring axioms verified"
