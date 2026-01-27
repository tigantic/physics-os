"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                      O P E R A T O R S   M O D U L E                                    ║
║                                                                                          ║
║     Type-safe operators acting on fields.                                               ║
║     These are the "how" of geometric computation.                                       ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

Operators transform fields into fields:
    L: Field[M, V] → Field[M, W]

Types of operators:
    LinearOperator[From, To]         - Linear maps between field spaces
    DifferentialOperator[From, To]   - Involve derivatives
    IntegralOperator[From, To]       - Involve integration
    GreenFunction[L]                 - Inverse of differential operator L
    Propagator[H]                    - Time evolution exp(-iHt)

Key property: operators track how constraints transform.
    If L preserves divergence-free, its type signature shows that.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    TypeVar, Generic, Optional, Tuple, Dict, Any,
    Type, Union, Callable, List
)
import torch
from torch import Tensor

from tensornet.types.spaces import Space, EuclideanSpace, R3
from tensornet.types.constraints import (
    Constraint, Divergence, Curl, Symmetric, Antisymmetric
)
from tensornet.types.fields import (
    Field, ScalarField, VectorField, TensorField, DifferentialForm
)


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE VARIABLES
# ═══════════════════════════════════════════════════════════════════════════════

S = TypeVar("S", bound=Space)
F = TypeVar("F", bound=Field)
G = TypeVar("G", bound=Field)


# ═══════════════════════════════════════════════════════════════════════════════
# BASE OPERATOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Operator(ABC, Generic[F, G]):
    """
    Abstract base class for operators on fields.
    
    An operator L: F → G maps fields of type F to fields of type G.
    
    The type system tracks:
    - What constraints are preserved
    - What constraints are created
    - What constraints are destroyed
    """
    
    # Constraints that the operator preserves
    preserves: Tuple[Type[Constraint], ...] = field(default_factory=tuple)
    
    # Constraints that the operator creates (output always has these)
    creates: Tuple[Constraint, ...] = field(default_factory=tuple)
    
    @abstractmethod
    def apply(self, field: F) -> G:
        """Apply the operator to a field."""
        ...
    
    def __call__(self, field: F) -> G:
        """Shorthand for apply."""
        return self.apply(field)
    
    def compose(self, other: "Operator[G, Any]") -> "ComposedOperator":
        """Compose operators: (L ∘ M)(f) = L(M(f))."""
        return ComposedOperator(self, other)
    
    def __matmul__(self, other: "Operator[G, Any]") -> "ComposedOperator":
        """@ operator for composition."""
        return self.compose(other)


@dataclass
class ComposedOperator(Operator):
    """Composition of two operators."""
    first: Operator = field(default=None)
    second: Operator = field(default=None)
    
    def __post_init__(self):
        if self.first is None or self.second is None:
            raise ValueError("ComposedOperator requires 'first' and 'second' operators")
    
    def apply(self, field: Field) -> Field:
        intermediate = self.first.apply(field)
        return self.second.apply(intermediate)


# ═══════════════════════════════════════════════════════════════════════════════
# LINEAR OPERATORS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LinearOperator(Operator[F, G]):
    """
    Linear operator L satisfying L(αf + βg) = αL(f) + βL(g).
    
    Can be represented by a matrix/kernel in appropriate basis.
    """
    
    matrix: Optional[Tensor] = None  # Matrix representation if finite-dimensional
    kernel: Optional[Callable[[Tensor, Tensor], Tensor]] = None  # Kernel K(x, y)
    
    def apply(self, field: F) -> G:
        if self.matrix is not None:
            # Matrix multiplication
            new_data = torch.einsum("...ij,...j->...i", self.matrix, field.data)
            return field.with_data(new_data)
        elif self.kernel is not None:
            # Integral operator: (Lf)(x) = ∫ K(x,y) f(y) dy
            raise NotImplementedError("Kernel operators require integration")
        else:
            raise ValueError("LinearOperator needs either matrix or kernel")
    
    def adjoint(self) -> "LinearOperator[G, F]":
        """Compute adjoint operator L†."""
        if self.matrix is not None:
            adj_matrix = self.matrix.conj().transpose(-1, -2)
            return LinearOperator(matrix=adj_matrix)
        raise NotImplementedError("Adjoint for kernel operators")
    
    def __add__(self, other: "LinearOperator[F, G]") -> "LinearOperator[F, G]":
        if self.matrix is not None and other.matrix is not None:
            return LinearOperator(matrix=self.matrix + other.matrix)
        raise NotImplementedError("Addition for kernel operators")
    
    def __mul__(self, scalar: float) -> "LinearOperator[F, G]":
        if self.matrix is not None:
            return LinearOperator(matrix=scalar * self.matrix)
        raise NotImplementedError("Scaling for kernel operators")


# ═══════════════════════════════════════════════════════════════════════════════
# DIFFERENTIAL OPERATORS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DifferentialOperator(Operator[F, G]):
    """
    Differential operator involving spatial derivatives.
    
    Examples:
        - Gradient: f → ∇f
        - Divergence: V → ∇·V
        - Curl: V → ∇×V
        - Laplacian: f → ∇²f
        - Navier-Stokes operator
    
    Key property: tracks constraint transformations.
    """
    
    order: int = 1  # Order of highest derivative
    space: Optional[Space] = None
    
    @abstractmethod
    def apply(self, field: F) -> G:
        ...


@dataclass
class GradientOperator(DifferentialOperator[ScalarField, VectorField]):
    """
    Gradient operator: ∇
    
    Maps scalar fields to vector fields.
    OUTPUT IS ALWAYS CURL-FREE.
    """
    
    order: int = 1
    creates: Tuple[Constraint, ...] = field(default_factory=lambda: (Curl(0),))
    
    def apply(self, field: ScalarField) -> VectorField:
        return field.gradient()


@dataclass
class DivergenceOperator(DifferentialOperator[VectorField, ScalarField]):
    """
    Divergence operator: ∇·
    
    Maps vector fields to scalar fields.
    """
    
    order: int = 1
    
    def apply(self, field: VectorField) -> ScalarField:
        return field.divergence()


@dataclass
class CurlOperator(DifferentialOperator[VectorField, VectorField]):
    """
    Curl operator: ∇×
    
    Maps vector fields to vector fields.
    OUTPUT IS ALWAYS DIVERGENCE-FREE.
    """
    
    order: int = 1
    creates: Tuple[Constraint, ...] = field(default_factory=lambda: (Divergence(0),))
    
    def apply(self, field: VectorField) -> VectorField:
        return field.curl()


@dataclass
class LaplacianOperator(DifferentialOperator[ScalarField, ScalarField]):
    """
    Laplacian operator: ∇²
    
    Second-order differential operator.
    """
    
    order: int = 2
    
    def apply(self, field: ScalarField) -> ScalarField:
        return field.laplacian()


@dataclass
class VectorLaplacianOperator(DifferentialOperator[VectorField, VectorField]):
    """
    Vector Laplacian: ∇²V = ∇(∇·V) - ∇×(∇×V)
    
    PRESERVES DIVERGENCE-FREE: If ∇·V = 0, then ∇²V is also divergence-free.
    """
    
    order: int = 2
    preserves: Tuple[Type[Constraint], ...] = field(default_factory=lambda: (Divergence,))
    
    def apply(self, field: VectorField) -> VectorField:
        dim = field.data.shape[-1]
        lap_data = torch.zeros_like(field.data)
        
        for i in range(dim):
            for d in range(len(field.grid_shape)):
                lap_data[..., i] += (
                    torch.roll(field.data[..., i], -1, dims=d) +
                    torch.roll(field.data[..., i], 1, dims=d) -
                    2 * field.data[..., i]
                ) / (field.dx ** 2)
        
        # Preserve divergence-free constraint if input has it
        preserved_constraints = tuple(
            c for c in field.constraints
            if isinstance(c, Divergence)
        )
        
        return VectorField(
            space=field.space,
            data=lap_data,
            constraints=preserved_constraints,
            grid_shape=field.grid_shape,
            dx=field.dx
        )


@dataclass  
class HelmholtzProjector(DifferentialOperator[VectorField, VectorField]):
    """
    Helmholtz projector P: V → V_div_free
    
    Projects any vector field onto its divergence-free component.
    OUTPUT IS ALWAYS DIVERGENCE-FREE.
    
    V = ∇φ + V_df
    P(V) = V_df = V - ∇φ where ∇²φ = ∇·V
    """
    
    order: int = 0  # Actually involves solving Poisson, but conceptually order 0
    creates: Tuple[Constraint, ...] = field(default_factory=lambda: (Divergence(0),))
    
    def apply(self, field: VectorField) -> VectorField:
        return field.project_divergence_free()


# ═══════════════════════════════════════════════════════════════════════════════
# GREEN FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GreenFunction(Operator[ScalarField, ScalarField]):
    """
    Green function G for a differential operator L.
    
    G is the (pseudo-)inverse: LG = GL = I (on appropriate subspace).
    
    (Gf)(x) = ∫ G(x, y) f(y) dy
    
    Examples:
        - Poisson Green function: ∇²G = δ
        - Helmholtz Green function: (∇² + k²)G = δ
    """
    
    differential_op: DifferentialOperator = field(default=None)
    kernel: Optional[Callable[[Tensor, Tensor], Tensor]] = None
    
    def __post_init__(self):
        if self.differential_op is None:
            raise ValueError("GreenFunction requires 'differential_op'")
    
    def apply(self, source: ScalarField) -> ScalarField:
        """
        Solve L(u) = source using the Green function.
        
        This is typically done spectrally (FFT) for translation-invariant kernels.
        """
        # Use FFT-based solver for Laplacian
        if isinstance(self.differential_op, LaplacianOperator):
            return self._solve_poisson(source)
        
        raise NotImplementedError(f"Green function for {type(self.differential_op)}")
    
    def _solve_poisson(self, source: ScalarField) -> ScalarField:
        """Solve ∇²u = f using FFT."""
        f_fft = torch.fft.fftn(source.data)
        
        # Build Laplacian in Fourier space: -4π²k²
        shape = source.data.shape
        k = [torch.fft.fftfreq(s, d=source.dx, device=source.device) for s in shape]
        K = torch.meshgrid(*k, indexing='ij')
        k_squared = sum(ki ** 2 for ki in K)
        
        # Avoid division by zero at k=0 (sets mean to zero)
        k_squared_safe = k_squared.clone()
        k_squared_safe[k_squared_safe == 0] = 1
        
        # u_fft = f_fft / (-4π²k²)
        u_fft = f_fft / (-4 * math.pi ** 2 * k_squared_safe)
        u_fft[tuple(0 for _ in shape)] = 0  # Zero mean
        
        u = torch.fft.ifftn(u_fft).real
        
        return ScalarField(
            space=source.space,
            data=u,
            constraints=(),
            grid_shape=source.grid_shape,
            dx=source.dx
        )


@dataclass
class HelmholtzGreen(GreenFunction):
    """
    Green function for Helmholtz operator (∇² + k²).
    """
    
    wavenumber: float = 1.0
    
    def apply(self, source: ScalarField) -> ScalarField:
        """Solve (∇² + k²)u = f."""
        f_fft = torch.fft.fftn(source.data)
        
        shape = source.data.shape
        k = [torch.fft.fftfreq(s, d=source.dx, device=source.device) for s in shape]
        K = torch.meshgrid(*k, indexing='ij')
        k_squared = sum(ki ** 2 for ki in K)
        
        # Helmholtz operator in Fourier space: -4π²|k|² + k²
        helm = -4 * math.pi ** 2 * k_squared + self.wavenumber ** 2
        helm_safe = helm.clone()
        helm_safe[helm_safe.abs() < 1e-10] = 1
        
        u_fft = f_fft / helm_safe
        u = torch.fft.ifftn(u_fft).real
        
        return ScalarField(
            space=source.space,
            data=u,
            constraints=(),
            grid_shape=source.grid_shape,
            dx=source.dx
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PROPAGATORS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Propagator(Operator[F, F]):
    """
    Time evolution operator U(t) = exp(-iHt/ħ) or exp(-Lt).
    
    Propagators preserve certain structures:
    - Unitary propagators preserve norm
    - Symplectic propagators preserve phase space volume
    - Dissipative propagators contract phase space
    """
    
    generator: Operator = field(default=None)  # H or L
    time: float = 1.0
    method: str = "exponential"  # exponential, runge_kutta, split_step
    
    def __post_init__(self):
        if self.generator is None:
            raise ValueError("Propagator requires 'generator' operator")
    
    def apply(self, initial: F) -> F:
        """Evolve initial condition forward in time."""
        if self.method == "exponential":
            return self._exponential_evolution(initial)
        elif self.method == "runge_kutta":
            return self._runge_kutta(initial)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _exponential_evolution(self, initial: F) -> F:
        """
        Compute exp(-Lt) * initial using matrix exponential.
        
        Only works for finite-dimensional linear operators.
        """
        if not isinstance(self.generator, LinearOperator):
            raise ValueError("Exponential evolution requires LinearOperator generator")
        
        if self.generator.matrix is None:
            raise ValueError("Generator must have matrix representation")
        
        # exp(-L*t) using torch.linalg.matrix_exp
        exp_Lt = torch.linalg.matrix_exp(-self.time * self.generator.matrix)
        
        new_data = torch.einsum("...ij,...j->...i", exp_Lt, initial.data)
        return initial.with_data(new_data)
    
    def _runge_kutta(self, initial: F, dt: float = 0.01) -> F:
        """
        Evolve using 4th-order Runge-Kutta.
        
        du/dt = -L(u)
        """
        n_steps = int(self.time / dt)
        u = initial
        
        for _ in range(n_steps):
            k1 = self.generator.apply(u)
            
            u_half1 = u.with_data(u.data - 0.5 * dt * k1.data)
            k2 = self.generator.apply(u_half1)
            
            u_half2 = u.with_data(u.data - 0.5 * dt * k2.data)
            k3 = self.generator.apply(u_half2)
            
            u_full = u.with_data(u.data - dt * k3.data)
            k4 = self.generator.apply(u_full)
            
            new_data = u.data - (dt / 6) * (k1.data + 2*k2.data + 2*k3.data + k4.data)
            u = u.with_data(new_data)
        
        return u


@dataclass
class HamiltonianPropagator(Propagator):
    """
    Unitary time evolution for quantum mechanics.
    
    U(t) = exp(-iHt/ħ)
    
    PRESERVES NORMALIZATION.
    """
    
    hbar: float = 1.0
    preserves: Tuple[Type[Constraint], ...] = field(default_factory=lambda: ())
    
    def apply(self, initial: F) -> F:
        """Evolve quantum state."""
        # For this to preserve normalization, we need unitary evolution
        # Use split-step or Crank-Nicolson for stability
        return self._split_step(initial)
    
    def _split_step(self, initial: F, dt: float = 0.01) -> F:
        """
        Split-step Fourier method for Schrödinger equation.
        
        Alternates between position and momentum space evolution.
        """
        n_steps = int(self.time / dt)
        psi = initial.data.clone()
        
        # This is a simplified version - full implementation would
        # extract kinetic and potential parts from the Hamiltonian
        
        for _ in range(n_steps):
            # Half step in position space (potential)
            # exp(-iV*dt/2ħ) - placeholder
            psi = psi * torch.exp(-1j * dt / (2 * self.hbar))
            
            # Full step in momentum space (kinetic)
            psi_k = torch.fft.fftn(psi)
            # exp(-iT*dt/ħ) - placeholder
            psi_k = psi_k * torch.exp(-1j * dt / self.hbar)
            psi = torch.fft.ifftn(psi_k)
            
            # Half step in position space (potential)
            psi = psi * torch.exp(-1j * dt / (2 * self.hbar))
        
        return initial.with_data(psi)


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRAL OPERATORS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IntegralOperator(Operator[F, G]):
    """
    Integral operator defined by a kernel.
    
    (Kf)(x) = ∫ K(x, y) f(y) dy
    
    Examples:
        - Convolution operators
        - Green functions
        - Heat kernel
    """
    
    kernel_fn: Callable[[Tensor, Tensor], Tensor] = field(default=None)
    integration_points: Optional[Tensor] = None
    weights: Optional[Tensor] = None  # Quadrature weights
    
    def __post_init__(self):
        if self.kernel_fn is None:
            pass  # Allow None for subclasses like ConvolutionOperator
    
    def apply(self, f: F) -> G:
        """Apply integral operator using quadrature."""
        if self.integration_points is None:
            # Use field's grid points
            raise NotImplementedError("Auto grid not implemented")
        
        # Numerical integration: sum over y
        result = torch.zeros_like(field.data)
        
        for i, y in enumerate(self.integration_points):
            K_xy = self.kernel_fn(field.data, y)
            w = self.weights[i] if self.weights is not None else 1.0
            result = result + w * K_xy * field.evaluate(y)
        
        return field.with_data(result)


@dataclass
class ConvolutionOperator(IntegralOperator):
    """
    Translation-invariant integral operator.
    
    (Kf)(x) = ∫ K(x - y) f(y) dy = (K * f)(x)
    
    Efficiently computed using FFT.
    """
    
    kernel_tensor: Optional[Tensor] = None  # K(x) sampled on grid
    
    def apply(self, field: F) -> G:
        """Compute convolution using FFT."""
        if self.kernel_tensor is None:
            raise ValueError("ConvolutionOperator needs kernel_tensor")
        
        # Convolution via FFT: F(K * f) = F(K) · F(f)
        f_fft = torch.fft.fftn(field.data)
        K_fft = torch.fft.fftn(self.kernel_tensor)
        
        result_fft = K_fft * f_fft
        result = torch.fft.ifftn(result_fft).real
        
        return field.with_data(result)


# ═══════════════════════════════════════════════════════════════════════════════
# OPERATOR FACTORIES
# ═══════════════════════════════════════════════════════════════════════════════

def gradient() -> GradientOperator:
    """Create gradient operator."""
    return GradientOperator()


def divergence() -> DivergenceOperator:
    """Create divergence operator."""
    return DivergenceOperator()


def curl() -> CurlOperator:
    """Create curl operator."""
    return CurlOperator()


def laplacian() -> LaplacianOperator:
    """Create Laplacian operator."""
    return LaplacianOperator()


def vector_laplacian() -> VectorLaplacianOperator:
    """Create vector Laplacian operator."""
    return VectorLaplacianOperator()


def helmholtz_projector() -> HelmholtzProjector:
    """Create Helmholtz projector to divergence-free fields."""
    return HelmholtzProjector()


def poisson_solver() -> GreenFunction:
    """Create solver for ∇²u = f."""
    return GreenFunction(differential_op=LaplacianOperator())


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Base
    "Operator", "ComposedOperator",
    # Linear
    "LinearOperator",
    # Differential
    "DifferentialOperator",
    "GradientOperator", "DivergenceOperator", "CurlOperator",
    "LaplacianOperator", "VectorLaplacianOperator",
    "HelmholtzProjector",
    # Green functions
    "GreenFunction", "HelmholtzGreen",
    # Propagators
    "Propagator", "HamiltonianPropagator",
    # Integral
    "IntegralOperator", "ConvolutionOperator",
    # Factories
    "gradient", "divergence", "curl", "laplacian",
    "vector_laplacian", "helmholtz_projector", "poisson_solver",
]
