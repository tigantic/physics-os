"""
Adjoint Solver for Sensitivity Analysis
========================================

Implements continuous and discrete adjoint methods for computing
sensitivities of objective functions with respect to design variables.

Key Applications:
    - Shape optimization (minimize drag, control heating)
    - Flow control (actuation placement)
    - Uncertainty quantification
    - Inverse problems

The Adjoint Method:
    For objective J(U, α) where U solves R(U, α) = 0:
    
    dJ/dα = ∂J/∂α + ψᵀ ∂R/∂α
    
    where adjoint variable ψ solves:
    
    (∂R/∂U)ᵀ ψ = -(∂J/∂U)ᵀ

Advantages:
    - Cost independent of number of design variables
    - Single adjoint solve gives all sensitivities
    - Enables gradient-based optimization

References:
    [1] Giles & Pierce, "An Introduction to the Adjoint Approach to Design",
        Flow, Turbulence and Combustion 65, 2000
    [2] Jameson, "Aerodynamic Design via Control Theory", J. Sci. Comput. 3, 1988
    [3] Nielsen & Anderson, "Aerodynamic Design Optimization on Unstructured 
        Meshes Using the Navier-Stokes Equations", AIAA J. 37(11), 1999
"""

import torch
import math
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Callable, List
from enum import Enum


class AdjointMethod(Enum):
    """Adjoint formulation type."""
    CONTINUOUS = "continuous"   # Derive adjoint PDEs, then discretize
    DISCRETE = "discrete"       # Differentiate discrete equations directly


class ObjectiveType(Enum):
    """Common objective function types."""
    DRAG = "drag"
    LIFT = "lift"
    HEAT_FLUX = "heat_flux"
    PRESSURE_INTEGRAL = "pressure_integral"
    CUSTOM = "custom"


@dataclass
class AdjointState:
    """
    Adjoint variable state.
    
    For Euler equations, adjoint variables are:
        ψ = [ψ_ρ, ψ_ρu, ψ_ρv, ψ_E]ᵀ
    """
    psi_rho: torch.Tensor       # Adjoint to density
    psi_rhou: torch.Tensor      # Adjoint to x-momentum
    psi_rhov: torch.Tensor      # Adjoint to y-momentum
    psi_E: torch.Tensor         # Adjoint to energy
    
    @property
    def shape(self) -> torch.Size:
        return self.psi_rho.shape
    
    def to_tensor(self) -> torch.Tensor:
        """Stack adjoint variables into (4, Ny, Nx) tensor."""
        return torch.stack([
            self.psi_rho, self.psi_rhou, self.psi_rhov, self.psi_E
        ], dim=0)
    
    @classmethod
    def from_tensor(cls, psi: torch.Tensor) -> 'AdjointState':
        """Create from (4, Ny, Nx) tensor."""
        return cls(
            psi_rho=psi[0],
            psi_rhou=psi[1],
            psi_rhov=psi[2],
            psi_E=psi[3]
        )
    
    @classmethod
    def zeros(cls, shape: Tuple[int, int], dtype=torch.float64) -> 'AdjointState':
        """Create zero-initialized adjoint state."""
        return cls(
            psi_rho=torch.zeros(shape, dtype=dtype),
            psi_rhou=torch.zeros(shape, dtype=dtype),
            psi_rhov=torch.zeros(shape, dtype=dtype),
            psi_E=torch.zeros(shape, dtype=dtype)
        )


@dataclass
class SensitivityResult:
    """Result from sensitivity computation."""
    dJ_dalpha: torch.Tensor     # Sensitivity gradient
    objective_value: float      # Current objective value
    adjoint_state: AdjointState # Final adjoint solution
    converged: bool             # Whether adjoint converged
    iterations: int             # Number of iterations


@dataclass
class AdjointConfig:
    """Configuration for adjoint solver."""
    method: AdjointMethod = AdjointMethod.DISCRETE
    max_iterations: int = 1000
    tolerance: float = 1e-10
    cfl: float = 0.5
    smoothing_iterations: int = 0    # Implicit residual smoothing
    dissipation_coeff: float = 0.5   # Artificial dissipation


class ObjectiveFunction:
    """
    Base class for objective functions.
    
    Subclasses implement:
        - evaluate: Compute J(U)
        - gradient: Compute ∂J/∂U
    """
    
    def evaluate(
        self,
        rho: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Evaluate objective function."""
        from tensornet.core.phase_deferred import PhaseDeferredError
        raise PhaseDeferredError(
            phase="24",
            reason="ObjectiveFunction.evaluate - requires domain-specific implementation",
            depends_on=["stable forward solver", "objective definition"]
        )
    
    def gradient(
        self,
        rho: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute gradient ∂J/∂U in primitive variables.
        
        Returns:
            Tuple of (∂J/∂ρ, ∂J/∂u, ∂J/∂v, ∂J/∂p)
        """
        from tensornet.core.phase_deferred import PhaseDeferredError
        raise PhaseDeferredError(
            phase="24",
            reason="ObjectiveFunction.gradient - adjoint gradient computation",
            depends_on=["stable forward solver", "memory-efficient checkpointing"]
        )


class DragObjective(ObjectiveFunction):
    """
    Drag coefficient objective.
    
    C_D = (1/q_∞ S) ∫ p n_x dS
    
    where n_x is the x-component of surface normal.
    """
    
    def __init__(
        self,
        surface_mask: torch.Tensor,
        normal_x: torch.Tensor,
        normal_y: torch.Tensor,
        q_inf: float,
        S_ref: float
    ):
        """
        Args:
            surface_mask: Boolean mask for surface cells
            normal_x, normal_y: Surface normal components
            q_inf: Freestream dynamic pressure
            S_ref: Reference area
        """
        self.surface_mask = surface_mask
        self.normal_x = normal_x
        self.normal_y = normal_y
        self.q_inf = q_inf
        self.S_ref = S_ref
    
    def evaluate(
        self,
        rho: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        # Pressure drag only (ignoring viscous drag)
        p_surface = p * self.surface_mask
        drag_integrand = p_surface * self.normal_x
        
        return drag_integrand.sum() / (self.q_inf * self.S_ref)
    
    def gradient(
        self,
        rho: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # ∂J/∂p = n_x * mask / (q_inf * S_ref)
        dJ_dp = self.normal_x * self.surface_mask / (self.q_inf * self.S_ref)
        
        return (
            torch.zeros_like(rho),  # ∂J/∂ρ
            torch.zeros_like(u),    # ∂J/∂u
            torch.zeros_like(v),    # ∂J/∂v
            dJ_dp                   # ∂J/∂p
        )


class HeatFluxObjective(ObjectiveFunction):
    """
    Integrated wall heat flux objective.
    
    J = ∫ q_w dS
    
    For hypersonic vehicles, minimizing peak or integrated
    heat flux is critical for TPS design.
    """
    
    def __init__(
        self,
        wall_mask: torch.Tensor,
        dy: float,
        k: torch.Tensor
    ):
        """
        Args:
            wall_mask: Boolean mask for wall-adjacent cells
            dy: Grid spacing normal to wall
            k: Thermal conductivity field
        """
        self.wall_mask = wall_mask
        self.dy = dy
        self.k = k
    
    def evaluate(
        self,
        rho: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
        T: Optional[torch.Tensor] = None,
        T_wall: float = 300.0,
        gamma: float = 1.4,
        R: float = 287.0,
        **kwargs
    ) -> torch.Tensor:
        if T is None:
            T = p / (rho * R)
        
        # q_w = -k dT/dy at wall
        # Approximate with first-order difference
        dT_dy = (T - T_wall) / self.dy
        q_w = -self.k * dT_dy * self.wall_mask
        
        return q_w.sum()
    
    def gradient(
        self,
        rho: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
        T_wall: float = 300.0,
        gamma: float = 1.4,
        R: float = 287.0,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # q_w ∝ T ∝ p/ρ
        # ∂q_w/∂p = -k/(dy*R*ρ) * mask
        # ∂q_w/∂ρ = k*p/(dy*R*ρ²) * mask
        
        coeff = -self.k / (self.dy * R * rho) * self.wall_mask
        
        dJ_dp = coeff
        dJ_drho = -coeff * p / rho
        
        return (
            dJ_drho,
            torch.zeros_like(u),
            torch.zeros_like(v),
            dJ_dp
        )


class AdjointEuler2D:
    """
    Adjoint solver for 2D Euler equations.
    
    Solves the adjoint equations backward in time (for unsteady)
    or to steady state (for steady primal).
    """
    
    def __init__(
        self,
        Nx: int,
        Ny: int,
        dx: float,
        dy: float,
        gamma: float = 1.4,
        config: AdjointConfig = None
    ):
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dy = dy
        self.gamma = gamma
        self.config = config or AdjointConfig()
    
    def flux_jacobian_x(
        self,
        rho: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Jacobian ∂F/∂U for x-flux.
        
        Returns:
            (4, 4, Ny, Nx) tensor of Jacobians at each point
        """
        gamma = self.gamma
        
        E = p / (gamma - 1) + 0.5 * rho * (u**2 + v**2)
        H = (E + p) / rho
        
        # A = ∂F/∂U (4x4 at each point)
        A = torch.zeros(4, 4, *rho.shape, dtype=rho.dtype, device=rho.device)
        
        # Row 0: ∂(ρu)/∂U
        A[0, 1] = torch.ones_like(rho)
        
        # Row 1: ∂(ρu² + p)/∂U
        A[1, 0] = 0.5 * (gamma - 3) * u**2 + 0.5 * (gamma - 1) * v**2
        A[1, 1] = (3 - gamma) * u
        A[1, 2] = -(gamma - 1) * v
        A[1, 3] = (gamma - 1) * torch.ones_like(rho)
        
        # Row 2: ∂(ρuv)/∂U
        A[2, 0] = -u * v
        A[2, 1] = v
        A[2, 2] = u
        
        # Row 3: ∂(u(E+p))/∂U
        A[3, 0] = u * (0.5 * (gamma - 1) * (u**2 + v**2) - H)
        A[3, 1] = H - (gamma - 1) * u**2
        A[3, 2] = -(gamma - 1) * u * v
        A[3, 3] = gamma * u
        
        return A
    
    def flux_jacobian_y(
        self,
        rho: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Jacobian ∂G/∂U for y-flux.
        
        Returns:
            (4, 4, Ny, Nx) tensor of Jacobians at each point
        """
        gamma = self.gamma
        
        E = p / (gamma - 1) + 0.5 * rho * (u**2 + v**2)
        H = (E + p) / rho
        
        B = torch.zeros(4, 4, *rho.shape, dtype=rho.dtype, device=rho.device)
        
        # Row 0: ∂(ρv)/∂U
        B[0, 2] = torch.ones_like(rho)
        
        # Row 1: ∂(ρuv)/∂U
        B[1, 0] = -u * v
        B[1, 1] = v
        B[1, 2] = u
        
        # Row 2: ∂(ρv² + p)/∂U
        B[2, 0] = 0.5 * (gamma - 3) * v**2 + 0.5 * (gamma - 1) * u**2
        B[2, 1] = -(gamma - 1) * u
        B[2, 2] = (3 - gamma) * v
        B[2, 3] = (gamma - 1) * torch.ones_like(rho)
        
        # Row 3: ∂(v(E+p))/∂U
        B[3, 0] = v * (0.5 * (gamma - 1) * (u**2 + v**2) - H)
        B[3, 1] = -(gamma - 1) * u * v
        B[3, 2] = H - (gamma - 1) * v**2
        B[3, 3] = gamma * v
        
        return B
    
    def adjoint_rhs(
        self,
        psi: AdjointState,
        rho: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
        source_term: torch.Tensor
    ) -> AdjointState:
        """
        Compute RHS for adjoint equations.
        
        ∂ψ/∂t = Aᵀ ∂ψ/∂x + Bᵀ ∂ψ/∂y - (∂J/∂U)ᵀ
        
        Note: Adjoint equations are solved backward in time,
        so we negate the spatial terms.
        """
        psi_vec = psi.to_tensor()  # (4, Ny, Nx)
        
        # Flux Jacobians
        A = self.flux_jacobian_x(rho, u, v, p)  # (4, 4, Ny, Nx)
        B = self.flux_jacobian_y(rho, u, v, p)
        
        # Transpose Jacobians (swap first two indices)
        AT = A.permute(1, 0, 2, 3)
        BT = B.permute(1, 0, 2, 3)
        
        # Compute Aᵀ ∂ψ/∂x
        dpsi_dx = torch.zeros_like(psi_vec)
        dpsi_dx[:, :, 1:-1] = (psi_vec[:, :, 2:] - psi_vec[:, :, :-2]) / (2 * self.dx)
        
        AT_dpsi_dx = torch.einsum('ijkl,jkl->ikl', AT, dpsi_dx)
        
        # Compute Bᵀ ∂ψ/∂y
        dpsi_dy = torch.zeros_like(psi_vec)
        dpsi_dy[:, 1:-1, :] = (psi_vec[:, 2:, :] - psi_vec[:, :-2, :]) / (2 * self.dy)
        
        BT_dpsi_dy = torch.einsum('ijkl,jkl->ikl', BT, dpsi_dy)
        
        # Artificial dissipation for stability
        d2psi_dx2 = torch.zeros_like(psi_vec)
        d2psi_dx2[:, :, 1:-1] = (psi_vec[:, :, 2:] - 2*psi_vec[:, :, 1:-1] + psi_vec[:, :, :-2])
        
        d2psi_dy2 = torch.zeros_like(psi_vec)
        d2psi_dy2[:, 1:-1, :] = (psi_vec[:, 2:, :] - 2*psi_vec[:, 1:-1, :] + psi_vec[:, :-2, :])
        
        dissipation = self.config.dissipation_coeff * (d2psi_dx2 + d2psi_dy2)
        
        # RHS = -Aᵀ ∂ψ/∂x - Bᵀ ∂ψ/∂y + dissipation + source
        rhs = -AT_dpsi_dx - BT_dpsi_dy + dissipation + source_term
        
        return AdjointState.from_tensor(rhs)
    
    def solve_steady(
        self,
        rho: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
        objective: ObjectiveFunction
    ) -> SensitivityResult:
        """
        Solve steady adjoint equations.
        
        Iterates until adjoint residual converges.
        """
        config = self.config
        shape = rho.shape
        
        # Initialize adjoint state
        psi = AdjointState.zeros(shape)
        
        # Objective gradient (forcing term)
        dJ = objective.gradient(rho, u, v, p)
        source = torch.stack(dJ, dim=0)  # (4, Ny, Nx)
        
        # Compute timestep
        a = torch.sqrt(self.gamma * p / rho)
        u_max = u.abs().max() + a.max()
        v_max = v.abs().max() + a.max()
        dt = config.cfl * min(self.dx / u_max, self.dy / v_max)
        
        # Iterate to steady state
        for iteration in range(config.max_iterations):
            # Compute RHS
            rhs = self.adjoint_rhs(psi, rho, u, v, p, source)
            
            # Check convergence
            residual = rhs.to_tensor().norm().item()
            
            if residual < config.tolerance:
                break
            
            # Update (forward Euler)
            psi_new = AdjointState(
                psi_rho=psi.psi_rho + dt * rhs.psi_rho,
                psi_rhou=psi.psi_rhou + dt * rhs.psi_rhou,
                psi_rhov=psi.psi_rhov + dt * rhs.psi_rhov,
                psi_E=psi.psi_E + dt * rhs.psi_E
            )
            psi = psi_new
        
        # Compute objective value
        J = objective.evaluate(rho, u, v, p).item()
        
        return SensitivityResult(
            dJ_dalpha=psi.to_tensor(),  # Full sensitivity field
            objective_value=J,
            adjoint_state=psi,
            converged=(residual < config.tolerance),
            iterations=iteration + 1
        )


def compute_shape_sensitivity(
    adjoint_state: AdjointState,
    rho: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    surface_mask: torch.Tensor,
    normal_x: torch.Tensor,
    normal_y: torch.Tensor,
    gamma: float = 1.4
) -> torch.Tensor:
    """
    Compute shape sensitivity dJ/d(surface).
    
    For surface points, sensitivity is:
        dJ/dn = ψᵀ (∂R/∂n)
    
    Args:
        adjoint_state: Converged adjoint solution
        rho, u, v, p: Flow state
        surface_mask: Boolean mask for surface
        normal_x, normal_y: Surface normals
        
    Returns:
        Sensitivity on surface
    """
    psi = adjoint_state.to_tensor()  # (4, Ny, Nx)
    
    # Flux contribution from moving surface
    # F·n = [ρ(u·n), ρu(u·n)+p*nx, ρv(u·n)+p*ny, (E+p)(u·n)]
    u_dot_n = u * normal_x + v * normal_y
    
    E = p / (gamma - 1) + 0.5 * rho * (u**2 + v**2)
    H = (E + p) / rho
    
    F_n = torch.stack([
        rho * u_dot_n,
        rho * u * u_dot_n + p * normal_x,
        rho * v * u_dot_n + p * normal_y,
        (E + p) * u_dot_n
    ], dim=0)
    
    # Sensitivity = ψ · F_n on surface
    sensitivity = (psi * F_n).sum(dim=0) * surface_mask
    
    return sensitivity


def validate_adjoint():
    """
    Run validation tests for adjoint module.
    """
    print("\n" + "=" * 70)
    print("ADJOINT SOLVER VALIDATION")
    print("=" * 70)
    
    # Test 1: AdjointState creation
    print("\n[Test 1] AdjointState Creation")
    print("-" * 40)
    
    psi = AdjointState.zeros((10, 10))
    
    assert psi.shape == (10, 10)
    assert psi.to_tensor().shape == (4, 10, 10)
    
    print("Shape: ", psi.shape)
    print("Tensor shape: ", psi.to_tensor().shape)
    print("✓ PASS")
    
    # Test 2: Flux Jacobian symmetry properties
    print("\n[Test 2] Flux Jacobian Properties")
    print("-" * 40)
    
    solver = AdjointEuler2D(Nx=10, Ny=10, dx=0.1, dy=0.1)
    
    rho = torch.ones(10, 10, dtype=torch.float64)
    u = torch.full((10, 10), 100.0, dtype=torch.float64)
    v = torch.zeros(10, 10, dtype=torch.float64)
    p = torch.full((10, 10), 101325.0, dtype=torch.float64)
    
    A = solver.flux_jacobian_x(rho, u, v, p)
    B = solver.flux_jacobian_y(rho, u, v, p)
    
    print(f"A shape: {A.shape}")
    print(f"B shape: {B.shape}")
    
    # Check that A is well-formed (no NaN/Inf)
    assert not torch.isnan(A).any()
    assert not torch.isinf(A).any()
    print("✓ PASS: Jacobians well-formed")
    
    # Test 3: Objective function
    print("\n[Test 3] Drag Objective Function")
    print("-" * 40)
    
    surface_mask = torch.zeros(10, 10, dtype=torch.float64)
    surface_mask[0, :] = 1.0  # Bottom wall
    
    normal_x = torch.zeros(10, 10, dtype=torch.float64)
    normal_x[0, :] = 1.0  # Normal pointing in x
    normal_y = torch.zeros(10, 10, dtype=torch.float64)
    
    q_inf = 0.5 * 1.2 * 100**2  # Dynamic pressure
    S_ref = 1.0
    
    drag_obj = DragObjective(surface_mask, normal_x, normal_y, q_inf, S_ref)
    
    # Evaluate
    J = drag_obj.evaluate(rho, u, v, p)
    print(f"Drag coefficient: {J.item():.4f}")
    
    # Gradient
    dJ = drag_obj.gradient(rho, u, v, p)
    print(f"∂J/∂p nonzero elements: {(dJ[3] != 0).sum().item()}")
    
    print("✓ PASS")
    
    # Test 4: Adjoint RHS computation
    print("\n[Test 4] Adjoint RHS Computation")
    print("-" * 40)
    
    psi = AdjointState.zeros((10, 10))
    source = torch.zeros(4, 10, 10, dtype=torch.float64)
    
    rhs = solver.adjoint_rhs(psi, rho, u, v, p, source)
    
    print(f"RHS shape: {rhs.shape}")
    print(f"RHS norm: {rhs.to_tensor().norm().item():.2e}")
    
    # Zero psi and source should give zero RHS (approximately)
    assert rhs.to_tensor().norm().item() < 1e-10
    print("✓ PASS: Zero state gives zero RHS")
    
    print("\n" + "=" * 70)
    print("ADJOINT VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    validate_adjoint()
