"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                         F I E L D S   M O D U L E                                       ║
║                                                                                          ║
║     Type-safe field representations with constraint enforcement.                        ║
║     These are the "what" of geometric computation.                                      ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

Fields are functions on spaces with values in vector spaces:
    f: M → V

Types of fields:
    ScalarField[M]           - M → R
    VectorField[M]           - M → TM (tangent bundle)
    TensorField[M, r, s]     - M → T^r_s M (mixed tensor)
    SpinorField[M]           - M → spinor bundle
    DifferentialForm[M, k]   - M → Λ^k T*M (k-forms)

Each field carries:
    - Its domain space M
    - Its constraints (Divergence=0, etc.)
    - Its QTT representation (the actual data)
    - Methods that preserve constraints
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    TypeVar, Generic, Optional, Tuple, Dict, Any,
    Type, Union, Callable, List, Set, Sequence
)
import torch
from torch import Tensor

from ontic.types.spaces import Space, EuclideanSpace, R3, TangentSpace
from ontic.types.constraints import (
    Constraint, Divergence, Curl, Conserved,
    Symmetric, Antisymmetric, Normalized
)


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE VARIABLES
# ═══════════════════════════════════════════════════════════════════════════════

S = TypeVar("S", bound=Space)
C = TypeVar("C", bound=Constraint)


# ═══════════════════════════════════════════════════════════════════════════════
# INVARIANT VIOLATION EXCEPTION
# ═══════════════════════════════════════════════════════════════════════════════

class InvariantViolation(Exception):
    """
    Raised when an operation would violate a declared constraint.
    
    This is the ENFORCEMENT mechanism. If you declare a VectorField with
    Divergence=0, and an operation produces non-zero divergence, this
    exception is raised.
    """
    
    def __init__(self, constraint: Constraint, message: str, data: Optional[Tensor] = None):
        self.constraint = constraint
        self.data = data
        super().__init__(f"Invariant '{constraint.name}' violated: {message}")


# ═══════════════════════════════════════════════════════════════════════════════
# BASE FIELD CLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Field(ABC, Generic[S]):
    """
    Abstract base class for all fields on a space.
    
    A Field encapsulates:
    - The domain space
    - The data (as QTT or dense tensor)
    - The declared constraints
    - Grid information for discretized fields
    """
    
    space: S
    data: Tensor
    constraints: Tuple[Constraint, ...] = field(default_factory=tuple)
    grid_shape: Optional[Tuple[int, ...]] = None
    dx: float = 1.0  # Grid spacing for discretized fields
    _verified: bool = False
    
    def __post_init__(self):
        """Verify constraints on construction."""
        if self.constraints and not self._verified:
            self.verify_constraints()
            self._verified = True
    
    def verify_constraints(self) -> None:
        """
        Verify all declared constraints are satisfied.
        
        Raises InvariantViolation if any constraint fails.
        """
        context = {
            "space": self.space,
            "dx": self.dx,
            "grid_shape": self.grid_shape,
        }
        
        for constraint in self.constraints:
            satisfied, message = constraint.verify(self.data, **context)
            if not satisfied:
                raise InvariantViolation(constraint, message, self.data)
    
    @abstractmethod
    def evaluate(self, point: Tensor) -> Tensor:
        """
        Evaluate the field at a point or set of points.
        
        Args:
            point: Coordinates, shape [..., dim]
            
        Returns:
            Field value at point(s)
        """
        ...
    
    def with_data(self, new_data: Tensor) -> "Field[S]":
        """
        Create a new field with different data but same constraints.
        
        Verifies that constraints still hold for new data.
        """
        new_field = self.__class__(
            space=self.space,
            data=new_data,
            constraints=self.constraints,
            grid_shape=self.grid_shape,
            dx=self.dx,
            _verified=False  # Will verify in __post_init__
        )
        return new_field
    
    @property
    def device(self) -> torch.device:
        return self.data.device
    
    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype
    
    def to(self, device: torch.device) -> "Field[S]":
        """Move field to a device."""
        return self.with_data(self.data.to(device))
    
    def __repr__(self) -> str:
        constraint_str = ", ".join(c.name for c in self.constraints) if self.constraints else "None"
        return f"{self.__class__.__name__}[{self.space}](constraints=[{constraint_str}])"


# ═══════════════════════════════════════════════════════════════════════════════
# SCALAR FIELD
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScalarField(Field[S]):
    """
    Scalar field f: M → R
    
    A real-valued function on a manifold.
    
    Examples:
        - Temperature field
        - Pressure field
        - Density field
        - Potential (gravitational, electric, etc.)
    """
    
    def evaluate(self, point: Tensor) -> Tensor:
        """
        Evaluate scalar field at a point using trilinear interpolation.
        """
        if self.grid_shape is None:
            # Continuous representation - use the point directly
            raise NotImplementedError("Continuous evaluation not implemented")
        
        # Discretized field - interpolate
        return self._interpolate(point)
    
    def _interpolate(self, point: Tensor) -> Tensor:
        """Trilinear interpolation for grid-based field."""
        # Normalize point to grid coordinates
        grid_point = point / self.dx
        
        # Simple nearest-neighbor for now (TODO: proper interpolation)
        indices = grid_point.long()
        indices = indices.clamp(min=0)
        for i, gs in enumerate(self.grid_shape):
            indices[..., i] = indices[..., i].clamp(max=gs - 1)
        
        # Index into data
        if len(self.grid_shape) == 1:
            return self.data[indices[..., 0]]
        elif len(self.grid_shape) == 2:
            return self.data[indices[..., 0], indices[..., 1]]
        elif len(self.grid_shape) == 3:
            return self.data[indices[..., 0], indices[..., 1], indices[..., 2]]
        else:
            raise NotImplementedError(f"Interpolation for {len(self.grid_shape)}D")
    
    def gradient(self) -> "VectorField[S]":
        """
        Compute gradient of scalar field.
        
        Returns a VectorField that is GUARANTEED to be curl-free.
        """
        dim = self.space.dimension
        grad_data = torch.zeros(*self.data.shape, dim, device=self.device, dtype=self.dtype)
        
        for i in range(dim):
            # Central difference
            grad_data[..., i] = (
                torch.roll(self.data, -1, dims=i) - 
                torch.roll(self.data, 1, dims=i)
            ) / (2 * self.dx)
        
        # Gradient fields are IRROTATIONAL by construction
        return VectorField(
            space=self.space,
            data=grad_data,
            constraints=(Curl(0),),  # Guaranteed!
            grid_shape=self.grid_shape,
            dx=self.dx
        )
    
    def laplacian(self) -> "ScalarField[S]":
        """Compute Laplacian ∇²f."""
        dim = self.space.dimension
        lap_data = torch.zeros_like(self.data)
        
        for i in range(dim):
            lap_data = lap_data + (
                torch.roll(self.data, -1, dims=i) +
                torch.roll(self.data, 1, dims=i) -
                2 * self.data
            ) / (self.dx ** 2)
        
        return ScalarField(
            space=self.space,
            data=lap_data,
            constraints=(),  # No guaranteed constraints
            grid_shape=self.grid_shape,
            dx=self.dx
        )
    
    def __add__(self, other: Union["ScalarField[S]", float]) -> "ScalarField[S]":
        if isinstance(other, ScalarField):
            return self.with_data(self.data + other.data)
        return self.with_data(self.data + other)
    
    def __mul__(self, other: Union["ScalarField[S]", float]) -> "ScalarField[S]":
        if isinstance(other, ScalarField):
            return self.with_data(self.data * other.data)
        return self.with_data(self.data * other)
    
    def __neg__(self) -> "ScalarField[S]":
        return self.with_data(-self.data)


# ═══════════════════════════════════════════════════════════════════════════════
# VECTOR FIELD
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VectorField(Field[S]):
    """
    Vector field V: M → TM
    
    A vector-valued function on a manifold (sections of tangent bundle).
    
    Examples:
        - Velocity field (fluid dynamics)
        - Force field (mechanics)
        - Electric field (electromagnetism)
        - Gradient of a potential
    
    Type-level constraints:
        VectorField[R3, Divergence(0)]   # Incompressible flow
        VectorField[R3, Curl(0)]         # Irrotational flow
        VectorField[R3, Conserved("momentum")]
    """
    
    def evaluate(self, point: Tensor) -> Tensor:
        """Evaluate vector field at a point."""
        if self.grid_shape is None:
            raise NotImplementedError("Continuous evaluation not implemented")
        
        return self._interpolate(point)
    
    def _interpolate(self, point: Tensor) -> Tensor:
        """Interpolate vector field at a point."""
        grid_point = point / self.dx
        indices = grid_point.long()
        indices = indices.clamp(min=0)
        for i, gs in enumerate(self.grid_shape):
            indices[..., i] = indices[..., i].clamp(max=gs - 1)
        
        if len(self.grid_shape) == 2:
            return self.data[indices[..., 0], indices[..., 1]]
        elif len(self.grid_shape) == 3:
            return self.data[indices[..., 0], indices[..., 1], indices[..., 2]]
        else:
            raise NotImplementedError(f"Interpolation for {len(self.grid_shape)}D")
    
    def divergence(self) -> ScalarField[S]:
        """
        Compute divergence ∇·V.
        
        If this field has Divergence(0) constraint, the result is guaranteed
        to be (approximately) zero.
        """
        dim = self.data.shape[-1]
        div_data = torch.zeros(self.data.shape[:-1], device=self.device, dtype=self.dtype)
        
        for i in range(dim):
            # ∂v_i/∂x_i using central difference
            v_i = self.data[..., i]
            dv_i = (
                torch.roll(v_i, -1, dims=i) - 
                torch.roll(v_i, 1, dims=i)
            ) / (2 * self.dx)
            div_data = div_data + dv_i
        
        return ScalarField(
            space=self.space,
            data=div_data,
            constraints=(),
            grid_shape=self.grid_shape,
            dx=self.dx
        )
    
    def curl(self) -> "VectorField[S]":
        """
        Compute curl ∇×V (3D only).
        
        If this field has Curl(0) constraint, the result is guaranteed
        to be (approximately) zero.
        """
        if self.data.shape[-1] != 3:
            raise ValueError("Curl is only defined for 3D vector fields")
        
        vx, vy, vz = self.data[..., 0], self.data[..., 1], self.data[..., 2]
        
        # Compute curl components using central differences
        def partial(f: Tensor, dim: int) -> Tensor:
            return (torch.roll(f, -1, dims=dim) - torch.roll(f, 1, dims=dim)) / (2 * self.dx)
        
        curl_x = partial(vz, 1) - partial(vy, 2)  # ∂vz/∂y - ∂vy/∂z
        curl_y = partial(vx, 2) - partial(vz, 0)  # ∂vx/∂z - ∂vz/∂x
        curl_z = partial(vy, 0) - partial(vx, 1)  # ∂vy/∂x - ∂vx/∂y
        
        curl_data = torch.stack([curl_x, curl_y, curl_z], dim=-1)
        
        # Curl of a vector field is DIVERGENCE-FREE by identity
        return VectorField(
            space=self.space,
            data=curl_data,
            constraints=(Divergence(0),),  # Guaranteed!
            grid_shape=self.grid_shape,
            dx=self.dx
        )
    
    def magnitude(self) -> ScalarField[S]:
        """Compute |V| at each point."""
        mag_data = torch.norm(self.data, dim=-1)
        return ScalarField(
            space=self.space,
            data=mag_data,
            constraints=(),
            grid_shape=self.grid_shape,
            dx=self.dx
        )
    
    def normalize(self) -> "VectorField[S]":
        """Return unit vector field V/|V|."""
        mag = torch.norm(self.data, dim=-1, keepdim=True)
        normalized_data = self.data / (mag + 1e-10)
        return VectorField(
            space=self.space,
            data=normalized_data,
            constraints=(),
            grid_shape=self.grid_shape,
            dx=self.dx
        )
    
    def dot(self, other: "VectorField[S]") -> ScalarField[S]:
        """Inner product V·W at each point."""
        dot_data = (self.data * other.data).sum(dim=-1)
        return ScalarField(
            space=self.space,
            data=dot_data,
            constraints=(),
            grid_shape=self.grid_shape,
            dx=self.dx
        )
    
    def cross(self, other: "VectorField[S]") -> "VectorField[S]":
        """Cross product V×W (3D only)."""
        if self.data.shape[-1] != 3:
            raise ValueError("Cross product is only defined for 3D vectors")
        
        cross_data = torch.cross(self.data, other.data, dim=-1)
        return VectorField(
            space=self.space,
            data=cross_data,
            constraints=(),
            grid_shape=self.grid_shape,
            dx=self.dx
        )
    
    def project_divergence_free(self) -> "VectorField[S]":
        """
        Project to divergence-free component using Helmholtz decomposition.
        
        V = ∇φ + ∇×A
        
        We find φ by solving ∇²φ = ∇·V, then subtract ∇φ.
        
        Returns:
            VectorField with GUARANTEED Divergence(0) constraint.
        """
        # Compute divergence
        div = self.divergence()
        
        # Solve Poisson equation ∇²φ = div(V)
        # Using spectral method (FFT)
        div_fft = torch.fft.fftn(div.data)
        
        # Build Laplacian in Fourier space
        shape = div.data.shape
        k = [torch.fft.fftfreq(s, d=self.dx, device=self.device) for s in shape]
        K = torch.meshgrid(*k, indexing='ij')
        k_squared = sum(ki ** 2 for ki in K)
        k_squared[k_squared == 0] = 1  # Avoid division by zero
        
        # φ in Fourier space
        phi_fft = div_fft / (-4 * math.pi ** 2 * k_squared)
        phi_fft[tuple(0 for _ in shape)] = 0  # Zero mean
        
        # Back to real space
        phi = torch.fft.ifftn(phi_fft).real
        
        # Create scalar field and compute gradient
        phi_field = ScalarField(
            space=self.space,
            data=phi,
            constraints=(),
            grid_shape=self.grid_shape,
            dx=self.dx
        )
        
        # Subtract gradient to get divergence-free part
        grad_phi = phi_field.gradient()
        projected_data = self.data - grad_phi.data
        
        # Return with GUARANTEED constraint
        return VectorField(
            space=self.space,
            data=projected_data,
            constraints=(Divergence(0),),
            grid_shape=self.grid_shape,
            dx=self.dx
        )
    
    def __add__(self, other: "VectorField[S]") -> "VectorField[S]":
        # Addition preserves divergence-free if both are divergence-free
        common_constraints = tuple(
            c for c in self.constraints 
            if c in other.constraints
        )
        return VectorField(
            space=self.space,
            data=self.data + other.data,
            constraints=common_constraints,
            grid_shape=self.grid_shape,
            dx=self.dx
        )
    
    def __mul__(self, scalar: float) -> "VectorField[S]":
        # Scalar multiplication preserves all linear constraints
        return VectorField(
            space=self.space,
            data=self.data * scalar,
            constraints=self.constraints,
            grid_shape=self.grid_shape,
            dx=self.dx
        )
    
    def __neg__(self) -> "VectorField[S]":
        return self * (-1)


# ═══════════════════════════════════════════════════════════════════════════════
# TENSOR FIELD
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TensorField(Field[S]):
    """
    Tensor field T: M → T^r_s M
    
    A tensor-valued function on a manifold.
    
    Examples:
        - Stress tensor (r=0, s=2, Symmetric)
        - Strain tensor
        - Riemann curvature tensor
        - Metric tensor
    
    Type-level constraints:
        TensorField[R3, Symmetric]          # Symmetric tensor
        TensorField[R3, Symmetric, Traceless]  # Deviatoric stress
    """
    
    rank: Tuple[int, int] = (0, 2)  # (contravariant, covariant)
    
    def evaluate(self, point: Tensor) -> Tensor:
        """Evaluate tensor field at a point."""
        if self.grid_shape is None:
            raise NotImplementedError("Continuous evaluation not implemented")
        
        # Simplified nearest-neighbor
        grid_point = point / self.dx
        indices = grid_point.long()
        indices = indices.clamp(min=0)
        for i, gs in enumerate(self.grid_shape):
            indices[..., i] = indices[..., i].clamp(max=gs - 1)
        
        # Index based on spatial dimensions
        n_spatial = len(self.grid_shape)
        if n_spatial == 3:
            return self.data[indices[..., 0], indices[..., 1], indices[..., 2]]
        else:
            raise NotImplementedError(f"Evaluation for {n_spatial}D fields")
    
    def trace(self) -> ScalarField[S]:
        """Compute trace of tensor field."""
        # Trace over last two indices
        trace_data = torch.diagonal(self.data, dim1=-2, dim2=-1).sum(dim=-1)
        return ScalarField(
            space=self.space,
            data=trace_data,
            constraints=(),
            grid_shape=self.grid_shape,
            dx=self.dx
        )
    
    def symmetrize(self) -> "TensorField[S]":
        """Return symmetric part: (T + T^T) / 2."""
        sym_data = (self.data + self.data.transpose(-1, -2)) / 2
        return TensorField(
            space=self.space,
            data=sym_data,
            constraints=(Symmetric(),),
            rank=self.rank,
            grid_shape=self.grid_shape,
            dx=self.dx
        )
    
    def antisymmetrize(self) -> "TensorField[S]":
        """Return antisymmetric part: (T - T^T) / 2."""
        asym_data = (self.data - self.data.transpose(-1, -2)) / 2
        return TensorField(
            space=self.space,
            data=asym_data,
            constraints=(Antisymmetric(),),
            rank=self.rank,
            grid_shape=self.grid_shape,
            dx=self.dx
        )
    
    def deviatoric(self) -> "TensorField[S]":
        """Return traceless (deviatoric) part: T - (Tr(T)/n) I."""
        n = self.data.shape[-1]
        trace = self.trace().data
        identity = torch.eye(n, device=self.device, dtype=self.dtype)
        
        # Expand trace for broadcasting
        trace_expanded = trace.unsqueeze(-1).unsqueeze(-1)
        dev_data = self.data - (trace_expanded / n) * identity
        
        new_constraints = tuple(c for c in self.constraints) + (Symmetric(),)
        return TensorField(
            space=self.space,
            data=dev_data,
            constraints=new_constraints,
            rank=self.rank,
            grid_shape=self.grid_shape,
            dx=self.dx
        )
    
    def contract(self, vector: VectorField[S], index: int = -1) -> VectorField[S]:
        """Contract tensor with a vector: T_ij v^j."""
        if index == -1:
            contracted = torch.einsum("...ij,...j->...i", self.data, vector.data)
        else:
            contracted = torch.einsum("...ij,...i->...j", self.data, vector.data)
        
        return VectorField(
            space=self.space,
            data=contracted,
            constraints=(),
            grid_shape=self.grid_shape,
            dx=self.dx
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SPINOR FIELD
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SpinorField(Field[S]):
    """
    Spinor field ψ: M → spinor bundle
    
    Complex field transforming under spin representation.
    
    Examples:
        - Dirac spinor (4-component)
        - Weyl spinor (2-component)
        - Pauli spinor
    
    Type-level constraints:
        SpinorField[R3, Normalized]  # Quantum state normalization
    """
    
    components: int = 2  # Number of spinor components
    
    def evaluate(self, point: Tensor) -> Tensor:
        """Evaluate spinor field at a point."""
        if self.grid_shape is None:
            raise NotImplementedError("Continuous evaluation not implemented")
        
        grid_point = point / self.dx
        indices = grid_point.long()
        indices = indices.clamp(min=0)
        for i, gs in enumerate(self.grid_shape):
            indices[..., i] = indices[..., i].clamp(max=gs - 1)
        
        if len(self.grid_shape) == 3:
            return self.data[indices[..., 0], indices[..., 1], indices[..., 2]]
        else:
            raise NotImplementedError()
    
    def norm_squared(self) -> ScalarField[S]:
        """Compute |ψ|² at each point (probability density)."""
        if self.data.is_complex():
            norm_sq = (self.data * self.data.conj()).real.sum(dim=-1)
        else:
            norm_sq = (self.data ** 2).sum(dim=-1)
        
        return ScalarField(
            space=self.space,
            data=norm_sq,
            constraints=(),
            grid_shape=self.grid_shape,
            dx=self.dx
        )
    
    def normalize(self) -> "SpinorField[S]":
        """Normalize the spinor field so ∫|ψ|²dV = 1."""
        norm_sq = self.norm_squared()
        total = norm_sq.data.sum() * (self.dx ** len(self.grid_shape))
        normalized_data = self.data / torch.sqrt(total)
        
        return SpinorField(
            space=self.space,
            data=normalized_data,
            constraints=(Normalized(1.0),),
            components=self.components,
            grid_shape=self.grid_shape,
            dx=self.dx
        )
    
    def probability_current(self) -> VectorField[S]:
        """
        Compute probability current j = (ħ/2mi)(ψ*∇ψ - ψ∇ψ*).
        
        Returns a DIVERGENCE-FREE vector field (conservation of probability).
        """
        dim = len(self.grid_shape)
        j_data = torch.zeros(*self.data.shape[:-1], dim, device=self.device)
        
        for i in range(dim):
            dpsi = (
                torch.roll(self.data, -1, dims=i) - 
                torch.roll(self.data, 1, dims=i)
            ) / (2 * self.dx)
            
            if self.data.is_complex():
                # j_i = Im(ψ* ∂ψ/∂x_i)
                j_data[..., i] = (self.data.conj() * dpsi).imag.sum(dim=-1)
            else:
                j_data[..., i] = (self.data * dpsi).sum(dim=-1)
        
        return VectorField(
            space=self.space,
            data=j_data,
            constraints=(Divergence(0),),  # Probability conservation
            grid_shape=self.grid_shape,
            dx=self.dx
        )


# ═══════════════════════════════════════════════════════════════════════════════
# DIFFERENTIAL FORMS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DifferentialForm(Field[S]):
    """
    Differential k-form ω ∈ Ω^k(M)
    
    Antisymmetric covariant tensor field.
    
    Examples:
        - 0-form: function (scalar field)
        - 1-form: covector field (e.g., df)
        - 2-form: e.g., electromagnetic field tensor
        - n-form: volume form
    """
    
    degree: int = 1  # k in k-form
    
    def evaluate(self, point: Tensor) -> Tensor:
        """Evaluate form at a point."""
        if self.grid_shape is None:
            raise NotImplementedError()
        
        grid_point = point / self.dx
        indices = grid_point.long()
        indices = indices.clamp(min=0)
        for i, gs in enumerate(self.grid_shape):
            indices[..., i] = indices[..., i].clamp(max=gs - 1)
        
        if len(self.grid_shape) == 3:
            return self.data[indices[..., 0], indices[..., 1], indices[..., 2]]
        else:
            raise NotImplementedError()
    
    def exterior_derivative(self) -> "DifferentialForm[S]":
        """
        Compute exterior derivative dω.
        
        Key property: d(dω) = 0 (nilpotency).
        """
        if self.degree == 0:
            # d(f) = ∂f/∂x^i dx^i (gradient as 1-form)
            dim = len(self.grid_shape)
            df_data = torch.zeros(*self.data.shape, dim, device=self.device, dtype=self.dtype)
            
            for i in range(dim):
                df_data[..., i] = (
                    torch.roll(self.data, -1, dims=i) -
                    torch.roll(self.data, 1, dims=i)
                ) / (2 * self.dx)
            
            return DifferentialForm(
                space=self.space,
                data=df_data,
                constraints=(),  # Closed forms (dω = 0 for ω = df)
                degree=1,
                grid_shape=self.grid_shape,
                dx=self.dx
            )
        
        elif self.degree == 1:
            # d(ω_i dx^i) = (∂ω_j/∂x^i) dx^i ∧ dx^j
            dim = self.data.shape[-1]
            # Result is antisymmetric 2-form
            dw_data = torch.zeros(*self.data.shape, dim, device=self.device, dtype=self.dtype)
            
            for i in range(dim):
                for j in range(dim):
                    if i < j:
                        # ∂ω_j/∂x^i - ∂ω_i/∂x^j
                        dw_data[..., i, j] = (
                            (torch.roll(self.data[..., j], -1, dims=i) -
                             torch.roll(self.data[..., j], 1, dims=i)) -
                            (torch.roll(self.data[..., i], -1, dims=j) -
                             torch.roll(self.data[..., i], 1, dims=j))
                        ) / (2 * self.dx)
                        dw_data[..., j, i] = -dw_data[..., i, j]
            
            return DifferentialForm(
                space=self.space,
                data=dw_data,
                constraints=(),
                degree=2,
                grid_shape=self.grid_shape,
                dx=self.dx
            )
        
        else:
            raise NotImplementedError(f"Exterior derivative for {self.degree}-forms")
    
    def wedge(self, other: "DifferentialForm[S]") -> "DifferentialForm[S]":
        """
        Wedge product ω ∧ η.
        
        Result has degree = deg(ω) + deg(η).
        """
        new_degree = self.degree + other.degree
        
        if self.degree == 0:
            # f ∧ η = f * η
            return DifferentialForm(
                space=self.space,
                data=self.data.unsqueeze(-1) * other.data,
                constraints=(),
                degree=new_degree,
                grid_shape=self.grid_shape,
                dx=self.dx
            )
        
        elif self.degree == 1 and other.degree == 1:
            # ω ∧ η where both are 1-forms
            dim = self.data.shape[-1]
            wedge_data = torch.zeros(*self.data.shape, dim, device=self.device, dtype=self.dtype)
            
            for i in range(dim):
                for j in range(dim):
                    wedge_data[..., i, j] = (
                        self.data[..., i] * other.data[..., j] -
                        self.data[..., j] * other.data[..., i]
                    )
            
            return DifferentialForm(
                space=self.space,
                data=wedge_data,
                constraints=(Antisymmetric(),),
                degree=2,
                grid_shape=self.grid_shape,
                dx=self.dx
            )
        
        else:
            raise NotImplementedError(f"Wedge of {self.degree}-form with {other.degree}-form")
    
    def hodge_star(self) -> "DifferentialForm[S]":
        """
        Hodge star operator *ω.
        
        Maps k-forms to (n-k)-forms.
        """
        n = len(self.grid_shape)
        new_degree = n - self.degree
        
        if self.degree == 0:
            # *f = f * (volume form data)
            return DifferentialForm(
                space=self.space,
                data=self.data.clone(),
                constraints=(),
                degree=new_degree,
                grid_shape=self.grid_shape,
                dx=self.dx
            )
        
        # For higher forms, need proper implementation with Levi-Civita
        raise NotImplementedError(f"Hodge star for {self.degree}-forms in {n}D")


# Convenient aliases
OneForm = DifferentialForm  # with degree=1
TwoForm = DifferentialForm  # with degree=2
TopForm = DifferentialForm  # with degree=n


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def scalar_field(
    space: S,
    values: Tensor,
    constraints: Sequence[Constraint] = (),
    dx: float = 1.0
) -> ScalarField[S]:
    """Create a scalar field from tensor data."""
    return ScalarField(
        space=space,
        data=values,
        constraints=tuple(constraints),
        grid_shape=values.shape,
        dx=dx
    )


def vector_field(
    space: S,
    values: Tensor,
    constraints: Sequence[Constraint] = (),
    dx: float = 1.0
) -> VectorField[S]:
    """Create a vector field from tensor data."""
    return VectorField(
        space=space,
        data=values,
        constraints=tuple(constraints),
        grid_shape=values.shape[:-1],
        dx=dx
    )


def divergence_free_field(
    space: S,
    values: Tensor,
    dx: float = 1.0
) -> VectorField[S]:
    """
    Create a divergence-free vector field.
    
    Projects the input to the divergence-free subspace.
    """
    temp = VectorField(
        space=space,
        data=values,
        constraints=(),
        grid_shape=values.shape[:-1],
        dx=dx
    )
    return temp.project_divergence_free()


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Exception
    "InvariantViolation",
    # Base
    "Field",
    # Field types
    "ScalarField", "VectorField", "TensorField", "SpinorField",
    # Differential forms
    "DifferentialForm", "OneForm", "TwoForm", "TopForm",
    # Factory functions
    "scalar_field", "vector_field", "divergence_free_field",
]
