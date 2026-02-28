"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                     C O N S T R A I N T S   M O D U L E                                 ║
║                                                                                          ║
║     Type-level constraints that mathematical objects must satisfy.                      ║
║     These are ENFORCED at runtime and checked after every operation.                   ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

Constraints encode mathematical invariants:

    VectorField[R3, Divergence(0)]       # Incompressible flow
    VectorField[R3, Curl(0)]             # Irrotational (potential) flow
    SymplecticForm[R2n, Closed]          # Symplectic structure
    Tensor[R3, Symmetric, Traceless]     # Deviatoric stress

The type system GUARANTEES these constraints are preserved through transformations.
If an operation would violate a constraint, it raises InvariantViolation.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    TypeVar, Generic, Optional, Tuple, Dict, Any,
    Type, Union, Callable, List, Set, Protocol, runtime_checkable
)
from enum import Enum, auto
import torch
from torch import Tensor


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE VARIABLES
# ═══════════════════════════════════════════════════════════════════════════════

T = TypeVar("T")
C = TypeVar("C", bound="Constraint")


# ═══════════════════════════════════════════════════════════════════════════════
# BASE CONSTRAINT
# ═══════════════════════════════════════════════════════════════════════════════

class Constraint(ABC):
    """
    Abstract base class for type-level constraints.
    
    A constraint defines:
    - A verification method that checks if data satisfies the constraint
    - A tolerance for numerical verification
    - Metadata about what the constraint means
    
    Constraints can be composed:
        Divergence(0) & Curl(0)  # Both must hold
        Symmetric | Antisymmetric  # Either can hold
    """
    
    @abstractmethod
    def verify(self, data: Tensor, **context: Any) -> Tuple[bool, str]:
        """
        Verify if data satisfies this constraint.
        
        Args:
            data: The tensor data to check
            **context: Additional context (space, grid, etc.)
            
        Returns:
            (satisfied, message) - whether constraint holds and explanation
        """
        ...
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this constraint."""
        ...
    
    @property
    def tolerance(self) -> float:
        """Numerical tolerance for constraint verification."""
        return 1e-6
    
    def __and__(self, other: "Constraint") -> "AndConstraint":
        """Combine constraints with AND."""
        return AndConstraint(self, other)
    
    def __or__(self, other: "Constraint") -> "OrConstraint":
        """Combine constraints with OR."""
        return OrConstraint(self, other)
    
    def __repr__(self) -> str:
        return self.name


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTRAINT COMBINATORS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AndConstraint(Constraint):
    """Conjunction of constraints - both must hold."""
    left: Constraint
    right: Constraint
    
    def verify(self, data: Tensor, **context: Any) -> Tuple[bool, str]:
        sat_l, msg_l = self.left.verify(data, **context)
        sat_r, msg_r = self.right.verify(data, **context)
        
        if sat_l and sat_r:
            return True, f"({msg_l}) AND ({msg_r})"
        elif not sat_l:
            return False, f"Failed: {msg_l}"
        else:
            return False, f"Failed: {msg_r}"
    
    @property
    def name(self) -> str:
        return f"({self.left.name} ∧ {self.right.name})"


@dataclass
class OrConstraint(Constraint):
    """Disjunction of constraints - at least one must hold."""
    left: Constraint
    right: Constraint
    
    def verify(self, data: Tensor, **context: Any) -> Tuple[bool, str]:
        sat_l, msg_l = self.left.verify(data, **context)
        if sat_l:
            return True, msg_l
        
        sat_r, msg_r = self.right.verify(data, **context)
        if sat_r:
            return True, msg_r
        
        return False, f"Neither: {msg_l} nor {msg_r}"
    
    @property
    def name(self) -> str:
        return f"({self.left.name} ∨ {self.right.name})"


@dataclass
class NotConstraint(Constraint):
    """Negation of a constraint."""
    inner: Constraint
    
    def verify(self, data: Tensor, **context: Any) -> Tuple[bool, str]:
        sat, msg = self.inner.verify(data, **context)
        if sat:
            return False, f"Should not satisfy: {msg}"
        else:
            return True, f"Correctly does not satisfy: {msg}"
    
    @property
    def name(self) -> str:
        return f"¬({self.inner.name})"


# ═══════════════════════════════════════════════════════════════════════════════
# DIFFERENTIAL CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Divergence(Constraint):
    """
    Divergence constraint: ∇·F = value
    
    For value=0, this is the incompressibility constraint.
    """
    value: float = 0.0
    _tolerance: float = 1e-6
    
    def verify(self, data: Tensor, **context: Any) -> Tuple[bool, str]:
        """
        Verify divergence constraint.
        
        Expects data shape: [..., dim] or [..., nx, ny, nz, dim] for field
        Context should include 'dx' for grid spacing.
        """
        dx = context.get("dx", 1.0)
        
        # Compute divergence using finite differences
        if data.dim() == 1:
            # Single vector - divergence not applicable
            return True, f"Single vector, divergence trivial"
        
        if data.dim() == 2:
            # [..., dim] - batch of vectors
            return True, f"Point-wise vectors, divergence trivial"
        
        # Assume [..., *spatial, dim] layout
        dim = data.shape[-1]
        div = torch.zeros(data.shape[:-1], device=data.device, dtype=data.dtype)
        
        for i in range(dim):
            # Central difference for ∂v_i/∂x_i
            v_i = data[..., i]
            dv_i = (torch.roll(v_i, -1, dims=-(dim - i)) - 
                    torch.roll(v_i, 1, dims=-(dim - i))) / (2 * dx)
            div = div + dv_i
        
        max_div = torch.abs(div - self.value).max().item()
        satisfied = max_div < self._tolerance
        
        if satisfied:
            return True, f"∇·F = {self.value} (max deviation: {max_div:.2e})"
        else:
            return False, f"∇·F ≠ {self.value} (max deviation: {max_div:.2e})"
    
    @property
    def name(self) -> str:
        if self.value == 0:
            return "Divergence-Free"
        return f"Divergence={self.value}"
    
    @property
    def tolerance(self) -> float:
        return self._tolerance


@dataclass  
class Curl(Constraint):
    """
    Curl constraint: ∇×F = value
    
    For value=0, this is the irrotational (potential flow) constraint.
    """
    value: float = 0.0
    _tolerance: float = 1e-6
    
    def verify(self, data: Tensor, **context: Any) -> Tuple[bool, str]:
        """
        Verify curl constraint (3D only).
        """
        dx = context.get("dx", 1.0)
        
        if data.shape[-1] != 3:
            return True, "Curl only defined for 3D"
        
        # For 3D vector field [..., nx, ny, nz, 3]
        if data.dim() < 4:
            return True, "Need spatial dimensions for curl"
        
        # Compute curl components
        vx, vy, vz = data[..., 0], data[..., 1], data[..., 2]
        
        # ∂vz/∂y - ∂vy/∂z
        curl_x = ((torch.roll(vz, -1, dims=-3) - torch.roll(vz, 1, dims=-3)) -
                  (torch.roll(vy, -1, dims=-2) - torch.roll(vy, 1, dims=-2))) / (2 * dx)
        
        # ∂vx/∂z - ∂vz/∂x
        curl_y = ((torch.roll(vx, -1, dims=-2) - torch.roll(vx, 1, dims=-2)) -
                  (torch.roll(vz, -1, dims=-4) - torch.roll(vz, 1, dims=-4))) / (2 * dx)
        
        # ∂vy/∂x - ∂vx/∂y
        curl_z = ((torch.roll(vy, -1, dims=-4) - torch.roll(vy, 1, dims=-4)) -
                  (torch.roll(vx, -1, dims=-3) - torch.roll(vx, 1, dims=-3))) / (2 * dx)
        
        curl_mag = torch.sqrt(curl_x**2 + curl_y**2 + curl_z**2)
        max_curl = (curl_mag - self.value).abs().max().item()
        
        satisfied = max_curl < self._tolerance
        
        if satisfied:
            return True, f"|∇×F| = {self.value} (max deviation: {max_curl:.2e})"
        else:
            return False, f"|∇×F| ≠ {self.value} (max deviation: {max_curl:.2e})"
    
    @property
    def name(self) -> str:
        if self.value == 0:
            return "Curl-Free"
        return f"Curl={self.value}"


@dataclass
class Gradient(Constraint):
    """Constraint that field is a gradient of some potential."""
    _tolerance: float = 1e-6
    
    def verify(self, data: Tensor, **context: Any) -> Tuple[bool, str]:
        """A field is a gradient iff it is curl-free."""
        curl_constraint = Curl(0.0, self._tolerance)
        return curl_constraint.verify(data, **context)
    
    @property
    def name(self) -> str:
        return "Gradient-Field"


@dataclass
class Laplacian(Constraint):
    """
    Laplacian constraint: ∇²f = value
    
    For value=0, this is harmonicity.
    """
    value: float = 0.0
    _tolerance: float = 1e-6
    
    def verify(self, data: Tensor, **context: Any) -> Tuple[bool, str]:
        dx = context.get("dx", 1.0)
        
        # Compute Laplacian using central differences
        lap = torch.zeros_like(data)
        for i in range(data.dim()):
            lap = lap + (torch.roll(data, -1, dims=i) + 
                        torch.roll(data, 1, dims=i) - 2 * data) / (dx ** 2)
        
        max_dev = torch.abs(lap - self.value).max().item()
        satisfied = max_dev < self._tolerance
        
        if satisfied:
            return True, f"∇²f = {self.value} (max deviation: {max_dev:.2e})"
        else:
            return False, f"∇²f ≠ {self.value} (max deviation: {max_dev:.2e})"
    
    @property
    def name(self) -> str:
        if self.value == 0:
            return "Harmonic"
        return f"Laplacian={self.value}"


# ═══════════════════════════════════════════════════════════════════════════════
# CONSERVATION CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Conserved(Constraint):
    """
    Conservation constraint: some quantity remains constant.
    
    Examples:
        Conserved("mass")
        Conserved("energy")
        Conserved("momentum")
        Conserved("angular_momentum")
    """
    quantity: str
    _tolerance: float = 1e-6
    _reference: Optional[float] = None
    
    def set_reference(self, value: float) -> None:
        """Set the reference value for conservation check."""
        self._reference = value
    
    def verify(self, data: Tensor, **context: Any) -> Tuple[bool, str]:
        """
        Verify conservation.
        
        Context should include the quantity extractor function.
        """
        extractor = context.get(f"extract_{self.quantity}", None)
        
        if extractor is None:
            # Default extractors
            if self.quantity == "mass":
                current = data.sum().item()
            elif self.quantity == "energy":
                current = (data ** 2).sum().item() * 0.5
            elif self.quantity == "momentum":
                current = data.sum(dim=-1).norm().item()
            else:
                return True, f"No extractor for {self.quantity}"
        else:
            current = extractor(data)
        
        if self._reference is None:
            self._reference = current
            return True, f"{self.quantity} = {current:.6e} (reference set)"
        
        deviation = abs(current - self._reference) / (abs(self._reference) + 1e-12)
        satisfied = deviation < self._tolerance
        
        if satisfied:
            return True, f"{self.quantity} conserved (rel. dev: {deviation:.2e})"
        else:
            return False, f"{self.quantity} not conserved (rel. dev: {deviation:.2e})"
    
    @property
    def name(self) -> str:
        return f"Conserved[{self.quantity}]"


# ═══════════════════════════════════════════════════════════════════════════════
# ALGEBRAIC CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Symplectic(Constraint):
    """
    Symplectic constraint: transformation preserves symplectic form.
    
    For Hamiltonian systems, the flow preserves the canonical 2-form ω = Σ dp_i ∧ dq_i.
    """
    _tolerance: float = 1e-6
    
    def verify(self, data: Tensor, **context: Any) -> Tuple[bool, str]:
        """
        Verify symplecticity of a Jacobian matrix.
        
        A matrix M is symplectic if M^T J M = J where J = [[0, I], [-I, 0]].
        """
        n = data.shape[-1]
        if n % 2 != 0:
            return False, "Symplectic requires even dimension"
        
        half = n // 2
        
        # Construct J matrix
        J = torch.zeros(n, n, device=data.device, dtype=data.dtype)
        J[:half, half:] = torch.eye(half, device=data.device, dtype=data.dtype)
        J[half:, :half] = -torch.eye(half, device=data.device, dtype=data.dtype)
        
        # Check M^T J M = J
        result = data.T @ J @ data
        deviation = (result - J).abs().max().item()
        
        satisfied = deviation < self._tolerance
        
        if satisfied:
            return True, f"Symplectic (max deviation: {deviation:.2e})"
        else:
            return False, f"Not symplectic (max deviation: {deviation:.2e})"
    
    @property
    def name(self) -> str:
        return "Symplectic"


@dataclass
class Unitary(Constraint):
    """
    Unitary constraint: U†U = UU† = I
    
    For complex matrices preserving inner products.
    """
    _tolerance: float = 1e-6
    
    def verify(self, data: Tensor, **context: Any) -> Tuple[bool, str]:
        n = data.shape[-1]
        I = torch.eye(n, device=data.device, dtype=data.dtype)
        
        # U†U - I
        if data.is_complex():
            UdaggerU = data.conj().T @ data
        else:
            UdaggerU = data.T @ data
        
        deviation = (UdaggerU - I).abs().max().item()
        satisfied = deviation < self._tolerance
        
        if satisfied:
            return True, f"Unitary (max deviation: {deviation:.2e})"
        else:
            return False, f"Not unitary (max deviation: {deviation:.2e})"
    
    @property
    def name(self) -> str:
        return "Unitary"


@dataclass
class Orthogonal(Constraint):
    """
    Orthogonal constraint: O^T O = OO^T = I
    
    For real matrices preserving inner products.
    """
    _tolerance: float = 1e-6
    
    def verify(self, data: Tensor, **context: Any) -> Tuple[bool, str]:
        n = data.shape[-1]
        I = torch.eye(n, device=data.device, dtype=data.dtype)
        
        OTO = data.T @ data
        deviation = (OTO - I).abs().max().item()
        
        satisfied = deviation < self._tolerance
        
        if satisfied:
            return True, f"Orthogonal (max deviation: {deviation:.2e})"
        else:
            return False, f"Not orthogonal (max deviation: {deviation:.2e})"
    
    @property
    def name(self) -> str:
        return "Orthogonal"


# ═══════════════════════════════════════════════════════════════════════════════
# TENSOR CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Normalized(Constraint):
    """
    Normalization constraint: ||x|| = value
    """
    value: float = 1.0
    _tolerance: float = 1e-6
    
    def verify(self, data: Tensor, **context: Any) -> Tuple[bool, str]:
        norm = torch.norm(data).item()
        deviation = abs(norm - self.value) / (self.value + 1e-12)
        
        satisfied = deviation < self._tolerance
        
        if satisfied:
            return True, f"||x|| = {norm:.6f} ≈ {self.value}"
        else:
            return False, f"||x|| = {norm:.6f} ≠ {self.value}"
    
    @property
    def name(self) -> str:
        return f"Normalized[{self.value}]"


@dataclass
class Positive(Constraint):
    """
    Positivity constraint: x > 0 (or x ≥ 0 if strict=False)
    """
    strict: bool = False
    _tolerance: float = 1e-10
    
    def verify(self, data: Tensor, **context: Any) -> Tuple[bool, str]:
        if self.strict:
            satisfied = (data > self._tolerance).all().item()
            condition = "> 0"
        else:
            satisfied = (data >= -self._tolerance).all().item()
            condition = "≥ 0"
        
        min_val = data.min().item()
        
        if satisfied:
            return True, f"x {condition} (min: {min_val:.2e})"
        else:
            return False, f"x not {condition} (min: {min_val:.2e})"
    
    @property
    def name(self) -> str:
        if self.strict:
            return "Positive"
        return "NonNegative"


@dataclass
class Symmetric(Constraint):
    """
    Symmetry constraint: A = A^T
    """
    _tolerance: float = 1e-6
    
    def verify(self, data: Tensor, **context: Any) -> Tuple[bool, str]:
        if data.dim() < 2:
            return True, "Scalar is trivially symmetric"
        
        AT = data.transpose(-1, -2)
        deviation = (data - AT).abs().max().item()
        
        satisfied = deviation < self._tolerance
        
        if satisfied:
            return True, f"A = A^T (max deviation: {deviation:.2e})"
        else:
            return False, f"A ≠ A^T (max deviation: {deviation:.2e})"
    
    @property
    def name(self) -> str:
        return "Symmetric"


@dataclass
class Antisymmetric(Constraint):
    """
    Antisymmetry constraint: A = -A^T
    """
    _tolerance: float = 1e-6
    
    def verify(self, data: Tensor, **context: Any) -> Tuple[bool, str]:
        if data.dim() < 2:
            return True, "Scalar is trivially antisymmetric (zero)"
        
        AT = data.transpose(-1, -2)
        deviation = (data + AT).abs().max().item()
        
        satisfied = deviation < self._tolerance
        
        if satisfied:
            return True, f"A = -A^T (max deviation: {deviation:.2e})"
        else:
            return False, f"A ≠ -A^T (max deviation: {deviation:.2e})"
    
    @property
    def name(self) -> str:
        return "Antisymmetric"


@dataclass
class Traceless(Constraint):
    """
    Traceless constraint: Tr(A) = 0
    """
    _tolerance: float = 1e-6
    
    def verify(self, data: Tensor, **context: Any) -> Tuple[bool, str]:
        if data.dim() < 2:
            return True, "Scalar trace is the scalar itself"
        
        trace = torch.diagonal(data, dim1=-2, dim2=-1).sum(dim=-1)
        max_trace = trace.abs().max().item()
        
        satisfied = max_trace < self._tolerance
        
        if satisfied:
            return True, f"Tr(A) = 0 (max: {max_trace:.2e})"
        else:
            return False, f"Tr(A) ≠ 0 (max: {max_trace:.2e})"
    
    @property
    def name(self) -> str:
        return "Traceless"


@dataclass
class Hermitian(Constraint):
    """
    Hermitian constraint: A = A†
    """
    _tolerance: float = 1e-6
    
    def verify(self, data: Tensor, **context: Any) -> Tuple[bool, str]:
        if data.dim() < 2:
            return True, "Scalar is trivially Hermitian"
        
        if data.is_complex():
            Adagger = data.conj().transpose(-1, -2)
        else:
            Adagger = data.transpose(-1, -2)
        
        deviation = (data - Adagger).abs().max().item()
        satisfied = deviation < self._tolerance
        
        if satisfied:
            return True, f"A = A† (max deviation: {deviation:.2e})"
        else:
            return False, f"A ≠ A† (max deviation: {deviation:.2e})"
    
    @property
    def name(self) -> str:
        return "Hermitian"


# ═══════════════════════════════════════════════════════════════════════════════
# SPECIAL CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Bounded(Constraint):
    """
    Boundedness constraint: lower ≤ x ≤ upper
    """
    lower: float = float("-inf")
    upper: float = float("inf")
    _tolerance: float = 1e-10
    
    def verify(self, data: Tensor, **context: Any) -> Tuple[bool, str]:
        min_val = data.min().item()
        max_val = data.max().item()
        
        in_range = (min_val >= self.lower - self._tolerance and 
                   max_val <= self.upper + self._tolerance)
        
        if in_range:
            return True, f"x ∈ [{min_val:.4f}, {max_val:.4f}] ⊂ [{self.lower}, {self.upper}]"
        else:
            return False, f"x ∈ [{min_val:.4f}, {max_val:.4f}] ⊄ [{self.lower}, {self.upper}]"
    
    @property
    def name(self) -> str:
        return f"Bounded[{self.lower}, {self.upper}]"


@dataclass
class Probability(Constraint):
    """
    Probability constraint: x ≥ 0 and Σx = 1
    """
    _tolerance: float = 1e-6
    
    def verify(self, data: Tensor, **context: Any) -> Tuple[bool, str]:
        # Check non-negativity
        min_val = data.min().item()
        if min_val < -self._tolerance:
            return False, f"Negative values (min: {min_val:.2e})"
        
        # Check normalization
        total = data.sum().item()
        if abs(total - 1.0) > self._tolerance:
            return False, f"Not normalized (sum: {total:.6f})"
        
        return True, f"Valid probability (sum: {total:.6f}, min: {min_val:.2e})"
    
    @property
    def name(self) -> str:
        return "Probability"


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTRAINT SETS (common combinations)
# ═══════════════════════════════════════════════════════════════════════════════

# Incompressible flow
INCOMPRESSIBLE = Divergence(0)

# Irrotational flow  
IRROTATIONAL = Curl(0)

# Potential flow (both)
POTENTIAL_FLOW = INCOMPRESSIBLE & IRROTATIONAL

# Stress tensor constraints
DEVIATORIC = Symmetric() & Traceless()

# Rotation matrices
ROTATION = Orthogonal()

# Hamiltonian flow preservation
HAMILTONIAN_FLOW = Symplectic()

# Quantum state
QUANTUM_STATE = Normalized(1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Base
    "Constraint", "AndConstraint", "OrConstraint", "NotConstraint",
    # Differential
    "Divergence", "Curl", "Gradient", "Laplacian",
    # Conservation
    "Conserved",
    # Algebraic
    "Symplectic", "Unitary", "Orthogonal",
    # Tensor
    "Normalized", "Positive", "Symmetric", "Antisymmetric", "Traceless", "Hermitian",
    # Special
    "Bounded", "Probability",
    # Common sets
    "INCOMPRESSIBLE", "IRROTATIONAL", "POTENTIAL_FLOW",
    "DEVIATORIC", "ROTATION", "HAMILTONIAN_FLOW", "QUANTUM_STATE",
]
