"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                      A L G E B R A I C   M O D U L E                                    ║
║                                                                                          ║
║     Lie groups, Lie algebras, fiber bundles, and connections.                          ║
║     The "symmetry" structures in geometric computation.                                 ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

Algebraic structures capture symmetries and gauge theories:

    LieGroup          - Smooth groups (SO(3), SU(2), etc.)
    LieAlgebra        - Infinitesimal generators
    FiberBundle       - E → M with fiber F
    PrincipalBundle   - G-bundle for gauge theories
    Connection        - Parallel transport / gauge field
    Curvature         - Field strength tensor

These are essential for:
    - Yang-Mills gauge theory
    - General relativity
    - Quantum field theory
    - Robotics and control theory
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

from tensornet.types.spaces import Space, EuclideanSpace, R3, Manifold


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE VARIABLES
# ═══════════════════════════════════════════════════════════════════════════════

G = TypeVar("G", bound="LieGroup")
M = TypeVar("M", bound=Space)


# ═══════════════════════════════════════════════════════════════════════════════
# LIE GROUP
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LieGroup(ABC):
    """
    Abstract Lie group.
    
    A Lie group is a smooth manifold that is also a group.
    
    Examples:
        - GL(n): General linear group
        - SL(n): Special linear group (det = 1)
        - O(n): Orthogonal group
        - SO(n): Special orthogonal group (rotations)
        - U(n): Unitary group
        - SU(n): Special unitary group
        - Sp(n): Symplectic group
    """
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimension as a manifold."""
        ...
    
    @property
    @abstractmethod
    def identity(self) -> Tensor:
        """Identity element e."""
        ...
    
    @abstractmethod
    def multiply(self, g1: Tensor, g2: Tensor) -> Tensor:
        """Group multiplication g1 * g2."""
        ...
    
    @abstractmethod
    def inverse(self, g: Tensor) -> Tensor:
        """Group inverse g^{-1}."""
        ...
    
    @abstractmethod
    def exp(self, X: Tensor) -> Tensor:
        """Exponential map from Lie algebra to group."""
        ...
    
    @abstractmethod
    def log(self, g: Tensor) -> Tensor:
        """Logarithm map from group to Lie algebra."""
        ...
    
    @property
    @abstractmethod
    def lie_algebra(self) -> "LieAlgebra":
        """The associated Lie algebra."""
        ...
    
    def adjoint(self, g: Tensor, X: Tensor) -> Tensor:
        """Adjoint action: Ad_g(X) = gXg^{-1}."""
        g_inv = self.inverse(g)
        return g @ X @ g_inv
    
    def left_action(self, g: Tensor, x: Tensor) -> Tensor:
        """Left group action on a vector."""
        return g @ x
    
    def right_action(self, g: Tensor, x: Tensor) -> Tensor:
        """Right group action on a vector."""
        return x @ g


@dataclass
class SO3(LieGroup):
    """
    Special orthogonal group SO(3) - 3D rotations.
    
    Elements are 3×3 orthogonal matrices with det = 1.
    Dimension: 3 (3 rotation angles)
    """
    
    @property
    def dimension(self) -> int:
        return 3
    
    @property
    def identity(self) -> Tensor:
        return torch.eye(3)
    
    def multiply(self, g1: Tensor, g2: Tensor) -> Tensor:
        return g1 @ g2
    
    def inverse(self, g: Tensor) -> Tensor:
        return g.transpose(-1, -2)
    
    def exp(self, X: Tensor) -> Tensor:
        """
        Exponential map using Rodrigues' formula.
        
        exp(X) = I + sin(θ)/θ X + (1-cos(θ))/θ² X²
        where θ = ||X|| (Frobenius norm / sqrt(2))
        """
        # X is a skew-symmetric matrix (element of so(3))
        # Extract angle from ||X||_F
        theta = torch.norm(X, dim=(-2, -1)) / math.sqrt(2)
        theta = theta.unsqueeze(-1).unsqueeze(-1)
        
        # Avoid division by zero
        theta_safe = theta.clamp(min=1e-8)
        
        I = torch.eye(3, device=X.device, dtype=X.dtype)
        
        # Rodrigues formula
        R = (I + 
             torch.sin(theta) / theta_safe * X + 
             (1 - torch.cos(theta)) / (theta_safe ** 2) * (X @ X))
        
        return R
    
    def log(self, g: Tensor) -> Tensor:
        """
        Logarithm map using inverse Rodrigues.
        """
        # Compute rotation angle
        trace = torch.diagonal(g, dim1=-2, dim2=-1).sum(dim=-1)
        theta = torch.acos((trace - 1) / 2).unsqueeze(-1).unsqueeze(-1)
        
        theta_safe = theta.clamp(min=1e-8)
        
        # X = θ/(2sin(θ)) (R - R^T)
        X = theta / (2 * torch.sin(theta_safe)) * (g - g.transpose(-1, -2))
        
        return X
    
    @property
    def lie_algebra(self) -> "LieAlgebra":
        return so3()
    
    @staticmethod
    def from_axis_angle(axis: Tensor, angle: Tensor) -> Tensor:
        """Create rotation from axis and angle."""
        axis = axis / torch.norm(axis, dim=-1, keepdim=True)
        K = SO3._skew(axis)
        I = torch.eye(3, device=axis.device, dtype=axis.dtype)
        angle = angle.unsqueeze(-1).unsqueeze(-1)
        return I + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
    
    @staticmethod
    def from_euler(roll: Tensor, pitch: Tensor, yaw: Tensor) -> Tensor:
        """Create rotation from Euler angles (ZYX convention)."""
        cr, sr = torch.cos(roll), torch.sin(roll)
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        
        R = torch.stack([
            torch.stack([cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr], dim=-1),
            torch.stack([sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr], dim=-1),
            torch.stack([-sp, cp*sr, cp*cr], dim=-1)
        ], dim=-2)
        
        return R
    
    @staticmethod
    def _skew(v: Tensor) -> Tensor:
        """Create skew-symmetric matrix from vector."""
        zero = torch.zeros_like(v[..., 0])
        return torch.stack([
            torch.stack([zero, -v[..., 2], v[..., 1]], dim=-1),
            torch.stack([v[..., 2], zero, -v[..., 0]], dim=-1),
            torch.stack([-v[..., 1], v[..., 0], zero], dim=-1)
        ], dim=-2)


@dataclass
class SU2(LieGroup):
    """
    Special unitary group SU(2).
    
    Elements are 2×2 unitary matrices with det = 1.
    Isomorphic to unit quaternions and double cover of SO(3).
    """
    
    @property
    def dimension(self) -> int:
        return 3
    
    @property
    def identity(self) -> Tensor:
        return torch.eye(2, dtype=torch.complex64)
    
    def multiply(self, g1: Tensor, g2: Tensor) -> Tensor:
        return g1 @ g2
    
    def inverse(self, g: Tensor) -> Tensor:
        return g.conj().transpose(-1, -2)
    
    def exp(self, X: Tensor) -> Tensor:
        """Exponential map for su(2)."""
        return torch.linalg.matrix_exp(X)
    
    def log(self, g: Tensor) -> Tensor:
        """Logarithm for SU(2)."""
        # Use eigendecomposition
        eigvals, eigvecs = torch.linalg.eig(g)
        log_eigvals = torch.log(eigvals)
        return eigvecs @ torch.diag_embed(log_eigvals) @ eigvecs.inverse()
    
    @property
    def lie_algebra(self) -> "LieAlgebra":
        return su2()
    
    @staticmethod
    def from_quaternion(q: Tensor) -> Tensor:
        """Convert unit quaternion [w, x, y, z] to SU(2) matrix."""
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        
        # SU(2) matrix representation
        U = torch.stack([
            torch.stack([w + 1j*z, y + 1j*x], dim=-1),
            torch.stack([-y + 1j*x, w - 1j*z], dim=-1)
        ], dim=-2)
        
        return U


# ═══════════════════════════════════════════════════════════════════════════════
# LIE ALGEBRA
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LieAlgebra(ABC):
    """
    Lie algebra - infinitesimal generators of a Lie group.
    
    A Lie algebra is a vector space with a Lie bracket [X, Y].
    """
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimension of the Lie algebra."""
        ...
    
    @abstractmethod
    def bracket(self, X: Tensor, Y: Tensor) -> Tensor:
        """Lie bracket [X, Y]."""
        ...
    
    @abstractmethod
    def basis(self) -> List[Tensor]:
        """Return a basis for the Lie algebra."""
        ...
    
    @property
    @abstractmethod
    def structure_constants(self) -> Tensor:
        """
        Structure constants f^k_ij where [T_i, T_j] = f^k_ij T_k.
        """
        ...
    
    def killing_form(self, X: Tensor, Y: Tensor) -> Tensor:
        """
        Killing form B(X, Y) = Tr(ad_X ∘ ad_Y).
        
        This is the natural inner product on a semisimple Lie algebra.
        """
        # ad_X(Z) = [X, Z]
        # B(X, Y) = Tr(Z -> [X, [Y, Z]])
        basis = self.basis()
        n = len(basis)
        
        trace = torch.zeros(X.shape[:-2], device=X.device)
        for T in basis:
            ad_Y_T = self.bracket(Y, T)
            ad_X_ad_Y_T = self.bracket(X, ad_Y_T)
            # Inner product with T (trace)
            trace = trace + (ad_X_ad_Y_T * T).sum(dim=(-2, -1))
        
        return trace


@dataclass
class MatrixLieAlgebra(LieAlgebra):
    """
    Lie algebra of matrices with commutator bracket.
    """
    
    matrix_dim: int = 3
    _basis: Optional[List[Tensor]] = None
    
    @property
    def dimension(self) -> int:
        return len(self.basis())
    
    def bracket(self, X: Tensor, Y: Tensor) -> Tensor:
        """Matrix commutator [X, Y] = XY - YX."""
        return X @ Y - Y @ X
    
    def basis(self) -> List[Tensor]:
        if self._basis is not None:
            return self._basis
        raise NotImplementedError("Subclass must provide basis")
    
    @property
    def structure_constants(self) -> Tensor:
        """Compute structure constants from basis and bracket."""
        basis = self.basis()
        n = len(basis)
        f = torch.zeros(n, n, n)
        
        for i in range(n):
            for j in range(n):
                bracket_ij = self.bracket(basis[i], basis[j])
                for k in range(n):
                    # Project onto basis element k
                    f[k, i, j] = (bracket_ij * basis[k]).sum()
        
        return f


def so3() -> MatrixLieAlgebra:
    """
    Lie algebra so(3) of 3D rotations.
    
    Basis: L_x, L_y, L_z (angular momentum generators)
    [L_i, L_j] = ε_ijk L_k
    """
    # Basis elements (skew-symmetric 3x3 matrices)
    L_x = torch.tensor([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float32)
    L_y = torch.tensor([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], dtype=torch.float32)
    L_z = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.float32)
    
    return MatrixLieAlgebra(matrix_dim=3, _basis=[L_x, L_y, L_z])


def su2() -> MatrixLieAlgebra:
    """
    Lie algebra su(2).
    
    Basis: Pauli matrices (times i/2)
    """
    # Pauli matrices times i/2
    sigma_x = 0.5j * torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
    sigma_y = 0.5j * torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
    sigma_z = 0.5j * torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
    
    return MatrixLieAlgebra(matrix_dim=2, _basis=[sigma_x, sigma_y, sigma_z])


# ═══════════════════════════════════════════════════════════════════════════════
# FIBER BUNDLES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FiberBundle(ABC, Generic[M]):
    """
    Fiber bundle E → M with fiber F.
    
    Locally: E ≈ M × F
    But globally may have non-trivial topology.
    
    Examples:
        - Tangent bundle TM
        - Cotangent bundle T*M
        - Vector bundles
        - Principal bundles (gauge theory)
    """
    
    base: M
    fiber_dim: int
    
    @abstractmethod
    def projection(self, e: Tensor) -> Tensor:
        """Project from total space to base: π: E → M."""
        ...
    
    @abstractmethod
    def fiber_at(self, point: Tensor) -> Space:
        """Get the fiber over a point in the base."""
        ...
    
    def section(self, s: Callable[[Tensor], Tensor]) -> "Section":
        """Create a section (cross-section) of the bundle."""
        return Section(bundle=self, section_fn=s)


@dataclass
class Section(Generic[M]):
    """
    Section of a fiber bundle: s: M → E such that π ∘ s = id.
    
    A section assigns to each point in the base a point in the fiber over it.
    """
    
    bundle: FiberBundle[M]
    section_fn: Callable[[Tensor], Tensor]
    
    def __call__(self, point: Tensor) -> Tensor:
        """Evaluate section at a point."""
        return self.section_fn(point)


@dataclass
class TangentBundle(FiberBundle[M]):
    """
    Tangent bundle TM.
    
    Fiber at p: T_p M (tangent space at p).
    """
    
    def projection(self, e: Tensor) -> Tensor:
        """Project (x, v) → x."""
        base_dim = self.base.dimension
        return e[..., :base_dim]
    
    def fiber_at(self, point: Tensor) -> Space:
        """Return tangent space at point."""
        return self.base.tangent_space_at(point)


@dataclass
class PrincipalBundle(FiberBundle[M], Generic[M, G]):
    """
    Principal G-bundle: a fiber bundle where the fiber is a Lie group G
    and G acts freely and transitively on fibers.
    
    This is the foundation of gauge theory.
    
    Examples:
        - Frame bundle (fiber = GL(n))
        - Spin bundle (fiber = Spin(n))
        - Gauge bundle (fiber = gauge group)
    """
    
    structure_group: G
    
    def projection(self, e: Tensor) -> Tensor:
        """Project to base space."""
        base_dim = self.base.dimension
        return e[..., :base_dim]
    
    def fiber_at(self, point: Tensor) -> Space:
        """The fiber is the structure group."""
        # Return the group as a manifold
        return EuclideanSpace(self.structure_group.dimension)
    
    def right_action(self, e: Tensor, g: Tensor) -> Tensor:
        """Right action of structure group: e · g."""
        base_dim = self.base.dimension
        base = e[..., :base_dim]
        fiber = e[..., base_dim:]
        
        # Group multiplication on fiber
        new_fiber = self.structure_group.multiply(
            fiber.reshape(-1, *self.structure_group.identity.shape),
            g
        )
        
        return torch.cat([base, new_fiber.flatten(-2)], dim=-1)


@dataclass
class AssociatedBundle(FiberBundle[M]):
    """
    Associated bundle E = P ×_G F.
    
    Constructed from a principal bundle P and a representation of G on F.
    
    Examples:
        - Vector bundles from frame bundle
        - Spinor bundles from spin bundle
    """
    
    principal_bundle: PrincipalBundle
    representation: Callable[[Tensor, Tensor], Tensor]  # (g, v) → g·v
    
    def projection(self, e: Tensor) -> Tensor:
        return self.principal_bundle.projection(e)
    
    def fiber_at(self, point: Tensor) -> Space:
        return EuclideanSpace(self.fiber_dim)


# ═══════════════════════════════════════════════════════════════════════════════
# CONNECTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Connection(Generic[M]):
    """
    Connection on a fiber bundle - defines parallel transport.
    
    A connection specifies how to compare fibers at different points.
    
    Representations:
        - Connection 1-form ω (on principal bundle)
        - Christoffel symbols Γ^i_jk (on tangent bundle)
        - Gauge field A_μ (in physics)
    """
    
    bundle: FiberBundle[M]
    
    @abstractmethod
    def parallel_transport(
        self,
        vector: Tensor,
        start: Tensor,
        end: Tensor,
        path: Optional[Tensor] = None
    ) -> Tensor:
        """
        Parallel transport a vector along a path.
        
        Args:
            vector: Vector in fiber at start
            start: Starting point on base
            end: Ending point on base
            path: Optional explicit path (default: geodesic)
            
        Returns:
            Vector in fiber at end
        """
        ...
    
    @abstractmethod
    def covariant_derivative(self, section: Section, direction: Tensor) -> Tensor:
        """
        Covariant derivative ∇_X s.
        
        Measures how a section changes as we move along a direction.
        """
        ...
    
    @abstractmethod
    def curvature(self, X: Tensor, Y: Tensor) -> Tensor:
        """
        Curvature 2-form: F(X, Y) = ∇_X ∇_Y - ∇_Y ∇_X - ∇_{[X,Y]}
        
        Measures the failure of parallel transport around infinitesimal loops.
        """
        ...


@dataclass
class LeviCivitaConnection(Connection[M]):
    """
    Levi-Civita connection - the unique torsion-free metric connection.
    
    For Riemannian manifolds, this is THE natural connection.
    Christoffel symbols: Γ^k_ij = ½ g^kl (∂_i g_lj + ∂_j g_il - ∂_l g_ij)
    """
    
    def parallel_transport(
        self,
        vector: Tensor,
        start: Tensor,
        end: Tensor,
        path: Optional[Tensor] = None
    ) -> Tensor:
        """Parallel transport using geodesic."""
        # For flat space, this is just identity
        if isinstance(self.bundle.base, EuclideanSpace):
            return vector
        
        # For curved space, integrate geodesic equation
        # dv^i/dt + Γ^i_jk v^j dx^k/dt = 0
        raise NotImplementedError("Curved parallel transport")
    
    def covariant_derivative(self, section: Section, direction: Tensor) -> Tensor:
        """
        ∇_X s = X^i ∂_i s + Γ^k_ij X^i s^j e_k
        """
        raise NotImplementedError("Covariant derivative")
    
    def curvature(self, X: Tensor, Y: Tensor) -> Tensor:
        """Riemann curvature tensor."""
        return self.bundle.base.riemann_tensor_at(X)


@dataclass  
class GaugeConnection(Connection[M]):
    """
    Gauge connection (Yang-Mills field).
    
    A = A_μ^a T_a dx^μ
    
    where T_a are Lie algebra generators.
    """
    
    gauge_field: Tensor  # A_μ^a(x), shape [..., spacetime_dim, algebra_dim]
    lie_algebra: LieAlgebra
    
    def parallel_transport(
        self,
        vector: Tensor,
        start: Tensor,
        end: Tensor,
        path: Optional[Tensor] = None
    ) -> Tensor:
        """
        Wilson line: P exp(i ∫ A · dx)
        """
        raise NotImplementedError("Wilson line")
    
    def covariant_derivative(self, section: Section, direction: Tensor) -> Tensor:
        """
        D_μ φ = ∂_μ φ + A_μ φ
        """
        raise NotImplementedError("Gauge covariant derivative")
    
    def curvature(self, X: Tensor, Y: Tensor) -> Tensor:
        """
        Field strength: F_μν = ∂_μ A_ν - ∂_ν A_μ + [A_μ, A_ν]
        """
        return self.field_strength()
    
    def field_strength(self) -> Tensor:
        """
        Compute Yang-Mills field strength F_μν.
        """
        A = self.gauge_field
        dim = A.shape[-2]
        
        # F_μν = ∂_μ A_ν - ∂_ν A_μ + [A_μ, A_ν]
        F = torch.zeros(*A.shape[:-1], dim, device=A.device, dtype=A.dtype)
        
        # This is simplified - full implementation needs spatial derivatives
        # and Lie algebra structure
        
        return F


@dataclass
class Curvature:
    """
    Curvature of a connection.
    
    For Levi-Civita: Riemann tensor R^l_ijk
    For gauge: Field strength F_μν^a
    """
    
    tensor: Tensor
    connection: Connection
    
    def ricci(self) -> Tensor:
        """Contract to Ricci tensor R_ij = R^k_ikj."""
        return torch.einsum("...kikj->...ij", self.tensor)
    
    def scalar(self, metric: Tensor) -> Tensor:
        """Ricci scalar R = g^ij R_ij."""
        ricci = self.ricci()
        g_inv = torch.linalg.inv(metric)
        return torch.einsum("...ij,...ij->...", g_inv, ricci)
    
    def weyl(self, metric: Tensor) -> Tensor:
        """Weyl tensor (traceless part of Riemann)."""
        raise NotImplementedError("Weyl tensor")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Lie groups
    "LieGroup", "SO3", "SU2",
    # Lie algebras
    "LieAlgebra", "MatrixLieAlgebra", "so3", "su2",
    # Fiber bundles
    "FiberBundle", "Section", "TangentBundle",
    "PrincipalBundle", "AssociatedBundle",
    # Connections
    "Connection", "LeviCivitaConnection", "GaugeConnection",
    # Curvature
    "Curvature",
]
