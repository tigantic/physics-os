"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                      E V O L U T I O N   M O D U L E                                    ║
║                                                                                          ║
║     Time evolution of geometric objects.                                                ║
║     Structure-preserving integrators.                                                   ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

This module provides:

    evolve          - High-level evolution function
    Flow            - Vector field that generates evolution
    Hamiltonian     - Energy function for Hamiltonian systems
    Lagrangian      - Action principle formulation
    
    Integrators:
        - symplectic_euler    - First-order symplectic
        - stormer_verlet      - Second-order symplectic
        - runge_kutta_4       - Classic RK4
        - implicit_midpoint   - Geometric, preserves quadratic invariants
        
The key insight: evolution should PRESERVE constraints.
If you start with Divergence(0), you should end with Divergence(0).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    TypeVar, Generic, Optional, Tuple, Dict, Any,
    Type, Union, Callable, List, Protocol
)
import torch
from torch import Tensor
import math

from ontic.types.spaces import Space
from ontic.types.constraints import Constraint


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE VARIABLES
# ═══════════════════════════════════════════════════════════════════════════════

F = TypeVar("F")
S = TypeVar("S", bound=Space)


# ═══════════════════════════════════════════════════════════════════════════════
# PROTOCOLS
# ═══════════════════════════════════════════════════════════════════════════════

class Evolvable(Protocol):
    """Protocol for objects that can be evolved in time."""
    
    data: Tensor
    constraints: Tuple[Constraint, ...]
    
    def with_data(self, new_data: Tensor) -> "Evolvable":
        """Return a new instance with updated data."""
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# FLOW
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Flow(Generic[F], ABC):
    """
    Abstract flow - a vector field that generates time evolution.
    
    Given a state x, the flow F defines evolution:
        dx/dt = F(x, t)
        
    Examples:
        - Gradient flow: F(x) = -∇V(x)
        - Hamiltonian flow: F(q, p) = (∂H/∂p, -∂H/∂q)
        - Navier-Stokes: F(v) = -(v·∇)v - ∇p + ν∇²v
    """
    
    @abstractmethod
    def __call__(self, state: F, t: float = 0.0) -> F:
        """Evaluate the flow at a state and time."""
        ...
    
    @property
    def preserves(self) -> Tuple[Constraint, ...]:
        """Constraints preserved by this flow."""
        return ()
    
    @property
    def is_autonomous(self) -> bool:
        """Whether the flow is time-independent."""
        return True


@dataclass
class GradientFlow(Flow[F]):
    """
    Gradient flow dx/dt = -∇V(x).
    
    Minimizes the potential V over time.
    All critical points are equilibria.
    """
    
    potential: Callable[[Tensor], Tensor]
    
    def __call__(self, state: F, t: float = 0.0) -> F:
        """Compute -∇V(x)."""
        if hasattr(state, "data"):
            x = state.data.requires_grad_(True)
            V = self.potential(x)
            grad = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
            
            result_data = -grad
            if hasattr(state, "with_data"):
                return state.with_data(result_data)
            return result_data
        
        raise TypeError(f"Cannot apply gradient flow to {type(state)}")


@dataclass
class VectorFieldFlow(Flow[F]):
    """
    Flow defined by a vector field.
    
    dx/dt = V(x)
    """
    
    vector_field: Callable[[Tensor, float], Tensor]
    _preserves: Tuple[Constraint, ...] = ()
    
    def __call__(self, state: F, t: float = 0.0) -> F:
        if hasattr(state, "data"):
            v = self.vector_field(state.data, t)
            if hasattr(state, "with_data"):
                return state.with_data(v)
            return v
        return self.vector_field(state, t)
    
    @property
    def preserves(self) -> Tuple[Constraint, ...]:
        return self._preserves


# ═══════════════════════════════════════════════════════════════════════════════
# HAMILTONIAN MECHANICS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Hamiltonian:
    """
    Hamiltonian H(q, p) for classical mechanics.
    
    Generates evolution via Hamilton's equations:
        dq/dt = ∂H/∂p
        dp/dt = -∂H/∂q
        
    Energy is conserved: dH/dt = 0
    """
    
    energy: Callable[[Tensor, Tensor], Tensor]  # H(q, p)
    
    def __call__(self, q: Tensor, p: Tensor) -> Tensor:
        """Evaluate the Hamiltonian."""
        return self.energy(q, p)
    
    def flow(self, q: Tensor, p: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute Hamilton's equations (dq/dt, dp/dt).
        """
        q_req = q.requires_grad_(True)
        p_req = p.requires_grad_(True)
        
        H = self.energy(q_req, p_req)
        
        dH_dq = torch.autograd.grad(
            H.sum(), q_req, create_graph=True, retain_graph=True
        )[0]
        dH_dp = torch.autograd.grad(
            H.sum(), p_req, create_graph=True
        )[0]
        
        # Hamilton's equations
        dq_dt = dH_dp
        dp_dt = -dH_dq
        
        return dq_dt, dp_dt
    
    def poisson_bracket(
        self,
        f: Callable[[Tensor, Tensor], Tensor],
        g: Callable[[Tensor, Tensor], Tensor],
        q: Tensor,
        p: Tensor
    ) -> Tensor:
        """
        Compute Poisson bracket {f, g} = ∂f/∂q ∂g/∂p - ∂f/∂p ∂g/∂q.
        """
        q_req = q.requires_grad_(True)
        p_req = p.requires_grad_(True)
        
        f_val = f(q_req, p_req)
        g_val = g(q_req, p_req)
        
        df_dq, df_dp = torch.autograd.grad(
            f_val.sum(), [q_req, p_req], create_graph=True, retain_graph=True
        )
        dg_dq, dg_dp = torch.autograd.grad(
            g_val.sum(), [q_req, p_req], create_graph=True
        )
        
        return (df_dq * dg_dp - df_dp * dg_dq).sum(dim=-1)


@dataclass
class HamiltonianFlow(Flow[Tuple[Tensor, Tensor]]):
    """
    Hamiltonian flow in phase space.
    
    State is (q, p).
    Evolution preserves:
        - Energy H(q, p)
        - Symplectic structure (volume in phase space)
        - All Casimir functions
    """
    
    hamiltonian: Hamiltonian
    
    def __call__(
        self,
        state: Tuple[Tensor, Tensor],
        t: float = 0.0
    ) -> Tuple[Tensor, Tensor]:
        q, p = state
        return self.hamiltonian.flow(q, p)
    
    @property
    def preserves(self) -> Tuple[Constraint, ...]:
        from ontic.types.constraints import Symplectic, Conserved
        return (Symplectic(), Conserved("energy"))


# ═══════════════════════════════════════════════════════════════════════════════
# LAGRANGIAN MECHANICS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Lagrangian:
    """
    Lagrangian L(q, q̇) for variational mechanics.
    
    Generates evolution via Euler-Lagrange equations:
        d/dt (∂L/∂q̇) = ∂L/∂q
        
    Related to Hamiltonian by Legendre transform:
        H(q, p) = p·q̇ - L(q, q̇)
        where p = ∂L/∂q̇
    """
    
    lagrangian: Callable[[Tensor, Tensor], Tensor]  # L(q, q_dot)
    
    def __call__(self, q: Tensor, q_dot: Tensor) -> Tensor:
        """Evaluate the Lagrangian."""
        return self.lagrangian(q, q_dot)
    
    def momentum(self, q: Tensor, q_dot: Tensor) -> Tensor:
        """
        Conjugate momentum p = ∂L/∂q̇.
        """
        q_dot_req = q_dot.requires_grad_(True)
        L = self.lagrangian(q, q_dot_req)
        p = torch.autograd.grad(L.sum(), q_dot_req, create_graph=True)[0]
        return p
    
    def to_hamiltonian(self) -> Hamiltonian:
        """
        Legendre transform to Hamiltonian.
        
        Requires ∂²L/∂q̇² to be invertible.
        """
        def H(q: Tensor, p: Tensor) -> Tensor:
            # Solve p = ∂L/∂q̇ for q̇ (Newton iteration)
            q_dot = torch.zeros_like(p)
            
            for _ in range(10):
                q_dot_req = q_dot.requires_grad_(True)
                L = self.lagrangian(q, q_dot_req)
                p_computed = torch.autograd.grad(
                    L.sum(), q_dot_req, create_graph=True
                )[0]
                
                residual = p_computed - p
                if residual.abs().max() < 1e-10:
                    break
                
                # Newton step
                hessian = torch.autograd.grad(
                    p_computed.sum(), q_dot_req, create_graph=True
                )[0]
                q_dot = q_dot - residual / (hessian + 1e-8)
            
            return (p * q_dot).sum(dim=-1) - self.lagrangian(q, q_dot)
        
        return Hamiltonian(energy=H)
    
    def euler_lagrange(
        self,
        q: Tensor,
        q_dot: Tensor,
        dt: float
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute (q̈) from Euler-Lagrange equations.
        
        d/dt(∂L/∂q̇) = ∂L/∂q
        """
        q_req = q.requires_grad_(True)
        q_dot_req = q_dot.requires_grad_(True)
        
        L = self.lagrangian(q_req, q_dot_req)
        
        # ∂L/∂q
        dL_dq = torch.autograd.grad(
            L.sum(), q_req, create_graph=True, retain_graph=True
        )[0]
        
        # ∂L/∂q̇
        dL_dqdot = torch.autograd.grad(
            L.sum(), q_dot_req, create_graph=True
        )[0]
        
        # ∂²L/∂q̇² (mass matrix)
        d2L_dqdot2 = torch.autograd.grad(
            dL_dqdot.sum(), q_dot_req, create_graph=True
        )[0]
        
        # ∂²L/∂q∂q̇
        d2L_dqdqdot = torch.autograd.grad(
            dL_dqdot.sum(), q_req, create_graph=True
        )[0]
        
        # q̈ = (∂²L/∂q̇²)^{-1} (∂L/∂q - ∂²L/∂q∂q̇ q̇)
        rhs = dL_dq - d2L_dqdqdot * q_dot
        q_ddot = rhs / (d2L_dqdot2 + 1e-8)
        
        return q_ddot


@dataclass
class ActionFunctional:
    """
    Action S[q] = ∫ L(q, q̇) dt.
    
    Extremizing the action gives Euler-Lagrange equations.
    """
    
    lagrangian: Lagrangian
    
    def __call__(
        self,
        trajectory: Tensor,
        times: Tensor
    ) -> Tensor:
        """
        Compute action along a trajectory.
        
        Args:
            trajectory: [T, ...] tensor of configurations
            times: [T] tensor of times
            
        Returns:
            Action value
        """
        T = trajectory.shape[0]
        dt = times[1:] - times[:-1]
        
        # Compute velocities
        q_dot = (trajectory[1:] - trajectory[:-1]) / dt.unsqueeze(-1)
        q = trajectory[:-1]
        
        # Integrate L dt
        L = self.lagrangian(q, q_dot)
        action = (L * dt).sum()
        
        return action
    
    def variation(
        self,
        trajectory: Tensor,
        times: Tensor,
        delta_q: Tensor
    ) -> Tensor:
        """
        Compute first variation δS.
        
        δS = 0 ⟹ Euler-Lagrange equations
        """
        trajectory_req = trajectory.requires_grad_(True)
        S = self(trajectory_req, times)
        
        delta_S = torch.autograd.grad(
            S, trajectory_req,
            grad_outputs=delta_q,
            create_graph=True
        )[0]
        
        return delta_S


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATORS
# ═══════════════════════════════════════════════════════════════════════════════

def runge_kutta_4(
    flow: Callable[[Tensor, float], Tensor],
    state: Tensor,
    t: float,
    dt: float
) -> Tensor:
    """
    Classic 4th-order Runge-Kutta integrator.
    
    Good general-purpose integrator.
    Does NOT preserve symplectic structure.
    """
    k1 = flow(state, t)
    k2 = flow(state + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = flow(state + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = flow(state + dt * k3, t + dt)
    
    return state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


def symplectic_euler(
    hamiltonian: Hamiltonian,
    q: Tensor,
    p: Tensor,
    dt: float
) -> Tuple[Tensor, Tensor]:
    """
    Symplectic Euler integrator (first-order).
    
    Preserves symplectic structure.
    Energy may oscillate but bounded.
    """
    q_req = q.requires_grad_(True)
    H = hamiltonian(q_req, p)
    dH_dq = torch.autograd.grad(H.sum(), q_req)[0]
    
    p_new = p - dt * dH_dq
    
    p_req = p_new.requires_grad_(True)
    H = hamiltonian(q, p_req)
    dH_dp = torch.autograd.grad(H.sum(), p_req)[0]
    
    q_new = q + dt * dH_dp
    
    return q_new, p_new


def stormer_verlet(
    hamiltonian: Hamiltonian,
    q: Tensor,
    p: Tensor,
    dt: float
) -> Tuple[Tensor, Tensor]:
    """
    Störmer-Verlet (leapfrog) integrator (second-order).
    
    Symplectic and time-reversible.
    Excellent for long-time integration.
    """
    # Half step in p
    q_req = q.requires_grad_(True)
    H = hamiltonian(q_req, p)
    dH_dq = torch.autograd.grad(H.sum(), q_req)[0]
    p_half = p - 0.5 * dt * dH_dq
    
    # Full step in q
    p_req = p_half.requires_grad_(True)
    H = hamiltonian(q, p_req)
    dH_dp = torch.autograd.grad(H.sum(), p_req)[0]
    q_new = q + dt * dH_dp
    
    # Half step in p
    q_req = q_new.requires_grad_(True)
    H = hamiltonian(q_req, p_half)
    dH_dq = torch.autograd.grad(H.sum(), q_req)[0]
    p_new = p_half - 0.5 * dt * dH_dq
    
    return q_new, p_new


def implicit_midpoint(
    flow: Callable[[Tensor, float], Tensor],
    state: Tensor,
    t: float,
    dt: float,
    max_iter: int = 10,
    tol: float = 1e-10
) -> Tensor:
    """
    Implicit midpoint method.
    
    Preserves all quadratic invariants.
    Good for constrained systems.
    """
    # Solve: y_{n+1} = y_n + dt * f((y_n + y_{n+1})/2, t + dt/2)
    y_new = state.clone()
    
    for _ in range(max_iter):
        midpoint = 0.5 * (state + y_new)
        f_mid = flow(midpoint, t + 0.5 * dt)
        y_next = state + dt * f_mid
        
        if (y_next - y_new).abs().max() < tol:
            break
        
        y_new = y_next
    
    return y_new


# ═══════════════════════════════════════════════════════════════════════════════
# EVOLUTION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvolutionConfig:
    """Configuration for evolution."""
    
    dt: float = 0.01
    method: str = "rk4"  # rk4, symplectic_euler, verlet, implicit_midpoint
    verify_constraints: bool = True
    constraint_tolerance: float = 1e-6
    project_constraints: bool = False
    max_steps: Optional[int] = None


def evolve(
    field: F,
    flow: Flow[F],
    t: float,
    config: Optional[EvolutionConfig] = None
) -> F:
    """
    Evolve a field forward in time.
    
    This is the high-level evolution function.
    It preserves constraints and uses appropriate integrators.
    
    Args:
        field: Initial field (e.g., VectorField)
        flow: The flow generating evolution
        t: Time to evolve to
        config: Evolution configuration
        
    Returns:
        Evolved field with preserved constraints
        
    Example:
        >>> flow = VectorField[R3, Divergence(0)](...)
        >>> evolved = evolve(flow, navier_stokes_flow, t=1.0)
        >>> # evolved is still Divergence(0)!
    """
    if config is None:
        config = EvolutionConfig()
    
    current = field
    current_t = 0.0
    n_steps = int(t / config.dt)
    
    if config.max_steps:
        n_steps = min(n_steps, config.max_steps)
    
    for step in range(n_steps):
        # Get flow direction
        if hasattr(current, "data"):
            direction = flow(current, current_t)
            
            if hasattr(direction, "data"):
                direction_data = direction.data
            else:
                direction_data = direction
            
            # Integrate
            if config.method == "rk4":
                new_data = runge_kutta_4(
                    lambda x, t_: flow(current.__class__(
                        data=x,
                        space=current.space if hasattr(current, "space") else None,
                        constraints=current.constraints if hasattr(current, "constraints") else ()
                    ) if hasattr(current, "__class__") else x, t_).data if hasattr(flow(current.__class__(
                        data=x,
                        space=current.space if hasattr(current, "space") else None,
                        constraints=current.constraints if hasattr(current, "constraints") else ()
                    ) if hasattr(current, "__class__") else x, t_), "data") else flow(x, t_),
                    current.data,
                    current_t,
                    config.dt
                )
            elif config.method == "implicit_midpoint":
                new_data = implicit_midpoint(
                    lambda x, t_: direction_data,
                    current.data,
                    current_t,
                    config.dt
                )
            else:
                # Simple Euler
                new_data = current.data + config.dt * direction_data
            
            # Update field
            if hasattr(current, "with_data"):
                current = current.with_data(new_data)
            elif hasattr(current, "__class__") and hasattr(current, "space"):
                current = current.__class__(
                    data=new_data,
                    space=current.space,
                    constraints=current.constraints if hasattr(current, "constraints") else ()
                )
        
        current_t += config.dt
        
        # Verify constraints periodically
        if config.verify_constraints and step % 10 == 0:
            if hasattr(current, "verify_constraints"):
                try:
                    current.verify_constraints()
                except Exception as e:
                    if config.project_constraints:
                        # Project back onto constraint manifold
                        current = _project_constraints(current)
                    else:
                        raise
    
    # Final constraint check
    if config.verify_constraints and hasattr(current, "verify_constraints"):
        current.verify_constraints()
    
    return current


def _project_constraints(field: F) -> F:
    """
    Project field back onto constraint manifold.
    
    For Divergence(0), this means Helmholtz projection.
    """
    from ontic.types.constraints import Divergence
    from ontic.types.fields import VectorField
    
    if not hasattr(field, "constraints"):
        return field
    
    for constraint in field.constraints:
        if isinstance(constraint, Divergence) and constraint.value == 0:
            if hasattr(field, "project_divergence_free"):
                return field.project_divergence_free()
    
    return field


def evolve_hamiltonian(
    q: Tensor,
    p: Tensor,
    hamiltonian: Hamiltonian,
    t: float,
    dt: float = 0.01,
    method: str = "verlet"
) -> Tuple[Tensor, Tensor]:
    """
    Evolve Hamiltonian system using symplectic integrator.
    
    Returns (q, p) at time t.
    Energy is conserved (up to integrator order).
    """
    n_steps = int(t / dt)
    
    if method == "symplectic_euler":
        integrator = symplectic_euler
    else:
        integrator = stormer_verlet
    
    for _ in range(n_steps):
        q, p = integrator(hamiltonian, q, p, dt)
    
    return q, p


# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICS FLOWS
# ═══════════════════════════════════════════════════════════════════════════════

def heat_flow(alpha: float = 1.0) -> VectorFieldFlow:
    """
    Heat equation flow: ∂u/∂t = α∇²u
    
    Diffuses the field over time.
    """
    def flow_fn(data: Tensor, t: float) -> Tensor:
        # Laplacian via finite differences
        laplacian = torch.zeros_like(data)
        for dim in range(data.ndim):
            laplacian = laplacian + (
                torch.roll(data, 1, dim) +
                torch.roll(data, -1, dim) -
                2 * data
            )
        return alpha * laplacian
    
    return VectorFieldFlow(vector_field=flow_fn)


def wave_equation_flow(c: float = 1.0) -> Flow[Tuple[Tensor, Tensor]]:
    """
    Wave equation: ∂²u/∂t² = c²∇²u
    
    Written as first-order system:
        ∂u/∂t = v
        ∂v/∂t = c²∇²u
    """
    @dataclass
    class WaveFlow(Flow[Tuple[Tensor, Tensor]]):
        speed: float = c
        
        def __call__(
            self,
            state: Tuple[Tensor, Tensor],
            t: float = 0.0
        ) -> Tuple[Tensor, Tensor]:
            u, v = state
            
            # Laplacian
            laplacian = torch.zeros_like(u)
            for dim in range(u.ndim):
                laplacian = laplacian + (
                    torch.roll(u, 1, dim) +
                    torch.roll(u, -1, dim) -
                    2 * u
                )
            
            du_dt = v
            dv_dt = self.speed ** 2 * laplacian
            
            return du_dt, dv_dt
    
    return WaveFlow()


def navier_stokes_flow(
    viscosity: float = 0.01,
    pressure_solver: Optional[Callable[[Tensor], Tensor]] = None
) -> VectorFieldFlow:
    """
    Incompressible Navier-Stokes flow.
    
    ∂v/∂t + (v·∇)v = -∇p + ν∇²v
    ∇·v = 0
    
    Preserves: Divergence(0)
    """
    from ontic.types.constraints import Divergence
    
    def flow_fn(data: Tensor, t: float) -> Tensor:
        # Assume data is velocity field [Nx, Ny, Nz, 3]
        v = data
        
        # Advection: -(v·∇)v
        # Use central differences
        advection = torch.zeros_like(v)
        for i in range(3):
            grad_vi = (torch.roll(v[..., i], -1, i) - torch.roll(v[..., i], 1, i)) / 2
            for j in range(3):
                advection[..., i] = advection[..., i] + v[..., j] * grad_vi
        
        # Diffusion: ν∇²v
        laplacian = torch.zeros_like(v)
        for dim in range(3):
            laplacian = laplacian + (
                torch.roll(v, 1, dim) +
                torch.roll(v, -1, dim) -
                2 * v
            )
        diffusion = viscosity * laplacian
        
        # Pressure projection (Helmholtz decomposition)
        rhs = -advection + diffusion
        
        # Project to divergence-free
        if pressure_solver is not None:
            # Solve ∇²p = ∇·rhs
            div_rhs = sum(
                (torch.roll(rhs[..., i], -1, i) - torch.roll(rhs[..., i], 1, i)) / 2
                for i in range(3)
            )
            p = pressure_solver(div_rhs)
            
            # Subtract ∇p
            for i in range(3):
                grad_p = (torch.roll(p, -1, i) - torch.roll(p, 1, i)) / 2
                rhs[..., i] = rhs[..., i] - grad_p
        
        return rhs
    
    return VectorFieldFlow(
        vector_field=flow_fn,
        _preserves=(Divergence(0),)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Protocols
    "Evolvable",
    # Flows
    "Flow", "GradientFlow", "VectorFieldFlow", "HamiltonianFlow",
    # Mechanics
    "Hamiltonian", "Lagrangian", "ActionFunctional",
    # Integrators
    "runge_kutta_4", "symplectic_euler", "stormer_verlet", "implicit_midpoint",
    # Evolution
    "EvolutionConfig", "evolve", "evolve_hamiltonian",
    # Physics flows
    "heat_flow", "wave_equation_flow", "navier_stokes_flow",
]
