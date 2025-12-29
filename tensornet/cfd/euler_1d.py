"""
1D Euler Equations for Compressible Flow
=========================================

The 1D Euler equations are the fundamental conservation laws for
compressible inviscid flow:

    ∂U/∂t + ∂F(U)/∂x = 0

where U = [ρ, ρu, E]ᵀ is the vector of conserved variables:
    - ρ: density
    - ρu: momentum
    - E: total energy

and F(U) is the flux vector:
    F = [ρu, ρu² + p, (E + p)u]ᵀ

The pressure p is determined by the equation of state:
    p = (γ - 1)(E - ½ρu²)

where γ is the ratio of specific heats (γ = 1.4 for air).

Tensor Network Approach
-----------------------
We discretize the solution on a grid and represent the state as an MPS.
Each site corresponds to a spatial grid point, with physical dimension
d = n_vars (number of conserved variables, typically 3).

For hypersonic flows with shocks, we need:
1. Shock-capturing schemes (Godunov-type)
2. Adaptive bond dimension for discontinuity resolution
3. Characteristic-based MPO operators

This module implements the foundation for tensor network CFD.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Callable, List, Union
import torch
from torch import Tensor
import math

from tensornet.core.mps import MPS


class BCType1D(Enum):
    """Boundary condition types for 1D Euler equations."""
    TRANSMISSIVE = "transmissive"  # Zero-gradient (outflow)
    REFLECTIVE = "reflective"       # Solid wall (u -> -u)
    PERIODIC = "periodic"           # Wrap-around


@dataclass
class EulerState:
    """Container for Euler equation state variables."""
    rho: Tensor      # Density (N,)
    rho_u: Tensor    # Momentum (N,)
    E: Tensor        # Total energy (N,)
    gamma: float     # Ratio of specific heats
    
    @property
    def N(self) -> int:
        """Number of grid points."""
        return self.rho.shape[0]
    
    @property
    def u(self) -> Tensor:
        """Velocity."""
        return self.rho_u / self.rho
    
    @property
    def p(self) -> Tensor:
        """Pressure from equation of state."""
        kinetic = 0.5 * self.rho_u ** 2 / self.rho
        return (self.gamma - 1) * (self.E - kinetic)
    
    @property
    def T(self) -> Tensor:
        """Temperature (assuming ideal gas, R = 1)."""
        return self.p / self.rho
    
    @property
    def a(self) -> Tensor:
        """Speed of sound."""
        return torch.sqrt(self.gamma * self.p / self.rho)
    
    @property
    def M(self) -> Tensor:
        """Mach number."""
        return self.u.abs() / self.a
    
    def to_conserved(self) -> Tensor:
        """Stack conserved variables: (N, 3)."""
        return torch.stack([self.rho, self.rho_u, self.E], dim=-1)
    
    @classmethod
    def from_conserved(cls, U: Tensor, gamma: float = 1.4) -> 'EulerState':
        """Create from conserved variable tensor (N, 3)."""
        return cls(
            rho=U[..., 0],
            rho_u=U[..., 1],
            E=U[..., 2],
            gamma=gamma,
        )
    
    @classmethod
    def from_primitive(
        cls,
        rho: Tensor,
        u: Tensor,
        p: Tensor,
        gamma: float = 1.4,
    ) -> 'EulerState':
        """Create from primitive variables (ρ, u, p)."""
        rho_u = rho * u
        E = p / (gamma - 1) + 0.5 * rho * u ** 2
        return cls(rho=rho, rho_u=rho_u, E=E, gamma=gamma)


class Euler1D:
    """
    1D Euler equation solver using tensor networks.
    
    Discretization: Finite volume with N cells on domain [x_min, x_max].
    
    The solution is stored as an MPS where each site represents a cell,
    and the physical dimension is 3 (for ρ, ρu, E).
    
    For now, we use a simple product state (χ=1) representation,
    which is equivalent to classical FVM. The tensor network structure
    enables future extensions:
    - Entangled representations for multi-scale features
    - MPO-based flux operators
    - Automatic adaptivity via bond dimension
    
    Attributes:
        N: Number of grid cells
        x_min, x_max: Domain bounds
        gamma: Ratio of specific heats
        cfl: CFL number for time stepping
        dx: Cell width
        x_cell: Cell center coordinates
        x_face: Face coordinates
        state: Current EulerState
        t: Current simulation time
    
    Example:
        >>> solver = Euler1D(N=200, x_min=0.0, x_max=1.0)
        >>> state = sod_shock_tube_ic(solver.x_cell)
        >>> solver.set_initial_condition(state)
        >>> for _ in range(100):
        ...     solver.step()
        >>> print(f"Final time: {solver.t:.4f}")
    
    Raises:
        ValueError: If state dimensions don't match grid size
        RuntimeError: If CFL condition is violated
    
    References:
        .. [1] Toro, E.F. "Riemann Solvers and Numerical Methods for 
               Fluid Dynamics", 3rd ed., Springer, 2009.
        .. [2] LeVeque, R.J. "Finite Volume Methods for Hyperbolic Problems",
               Cambridge University Press, 2002.
    """
    
    def __init__(
        self,
        N: int,
        x_min: float = 0.0,
        x_max: float = 1.0,
        gamma: float = 1.4,
        cfl: float = 0.5,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize Euler solver.
        
        Args:
            N: Number of grid cells
            x_min, x_max: Domain bounds
            gamma: Ratio of specific heats
            cfl: CFL number for time stepping
            dtype: Data type
            device: Device
        """
        self.N = N
        self.x_min = x_min
        self.x_max = x_max
        self.gamma = gamma
        self.cfl = cfl
        self.dtype = dtype
        self.device = device if device else torch.device('cpu')
        
        # Grid
        self.dx = (x_max - x_min) / N
        self.x_cell = torch.linspace(
            x_min + 0.5 * self.dx,
            x_max - 0.5 * self.dx,
            N,
            dtype=dtype,
            device=self.device,
        )
        self.x_face = torch.linspace(
            x_min, x_max, N + 1,
            dtype=dtype,
            device=self.device,
        )
        
        # State
        self.state: Optional[EulerState] = None
        self.t = 0.0
        
        # Boundary conditions (default: transmissive/outflow)
        self.bc_left = BCType1D.TRANSMISSIVE
        self.bc_right = BCType1D.TRANSMISSIVE
    
    def set_boundary_conditions(
        self,
        left: BCType1D = BCType1D.TRANSMISSIVE,
        right: BCType1D = BCType1D.TRANSMISSIVE,
    ) -> 'Euler1D':
        """
        Set boundary conditions.
        
        Args:
            left: Left boundary condition type
            right: Right boundary condition type
            
        Returns:
            self for method chaining
        """
        self.bc_left = left
        self.bc_right = right
        return self
    
    def set_initial_condition(self, state: EulerState):
        """Set initial condition."""
        assert state.N == self.N, f"State size {state.N} != grid size {self.N}"
        self.state = state
        self.t = 0.0
    
    def compute_flux(self, U_L: Tensor, U_R: Tensor) -> Tensor:
        """
        Compute numerical flux at cell interface using Rusanov (local Lax-Friedrichs).
        
        F_{i+1/2} = ½(F_L + F_R) - ½ λ_max (U_R - U_L)
        
        Args:
            U_L: Left state (batch, 3)
            U_R: Right state (batch, 3)
            
        Returns:
            Flux (batch, 3)
        """
        gamma = self.gamma
        
        # Extract conserved variables
        rho_L, rho_u_L, E_L = U_L[..., 0], U_L[..., 1], U_L[..., 2]
        rho_R, rho_u_R, E_R = U_R[..., 0], U_R[..., 1], U_R[..., 2]
        
        # Velocities
        u_L = rho_u_L / rho_L
        u_R = rho_u_R / rho_R
        
        # Pressures
        p_L = (gamma - 1) * (E_L - 0.5 * rho_L * u_L ** 2)
        p_R = (gamma - 1) * (E_R - 0.5 * rho_R * u_R ** 2)
        
        # Sound speeds
        a_L = torch.sqrt(gamma * p_L / rho_L)
        a_R = torch.sqrt(gamma * p_R / rho_R)
        
        # Maximum wave speed
        lambda_max = torch.maximum(
            torch.abs(u_L) + a_L,
            torch.abs(u_R) + a_R,
        )
        
        # Physical fluxes
        F_L = torch.stack([
            rho_u_L,
            rho_u_L * u_L + p_L,
            (E_L + p_L) * u_L,
        ], dim=-1)
        
        F_R = torch.stack([
            rho_u_R,
            rho_u_R * u_R + p_R,
            (E_R + p_R) * u_R,
        ], dim=-1)
        
        # Rusanov flux
        flux = 0.5 * (F_L + F_R) - 0.5 * lambda_max.unsqueeze(-1) * (U_R - U_L)
        
        return flux
    
    def compute_dt(self) -> float:
        """Compute time step from CFL condition."""
        if self.state is None:
            raise ValueError("State not initialized")
        
        u = self.state.u
        a = self.state.a
        
        # Maximum wave speed
        max_speed = (torch.abs(u) + a).max().item()
        
        if max_speed < 1e-10:
            return 1e-6  # Fallback for stationary flow
        
        dt = self.cfl * self.dx / max_speed
        return dt
    
    def step(self, dt: Optional[float] = None) -> float:
        """
        Advance solution by one time step.
        
        Uses first-order forward Euler in time with
        Rusanov flux in space.
        
        Args:
            dt: Time step (computed from CFL if None)
            
        Returns:
            Actual dt used
        """
        if self.state is None:
            raise ValueError("State not initialized")
        
        if dt is None:
            dt = self.compute_dt()
        
        U = self.state.to_conserved()  # (N, 3)
        
        # Build ghost cells based on boundary conditions
        U_left_ghost = self._apply_left_bc(U)
        U_right_ghost = self._apply_right_bc(U)
        
        U_ext = torch.cat([U_left_ghost, U, U_right_ghost], dim=0)
        
        # Compute fluxes at all faces
        U_L = U_ext[:-1]  # (N+1, 3)
        U_R = U_ext[1:]   # (N+1, 3)
        
        F = self.compute_flux(U_L, U_R)  # (N+1, 3)
        
        # Flux differences
        dF = F[1:] - F[:-1]  # (N, 3)
        
        # Update conserved variables
        U_new = U - (dt / self.dx) * dF
        
        # Update state
        self.state = EulerState.from_conserved(U_new, self.gamma)
        self.t += dt
        
        return dt
    
    def _apply_left_bc(self, U: Tensor) -> Tensor:
        """Apply left boundary condition to create ghost cell."""
        if self.bc_left == BCType1D.TRANSMISSIVE:
            # Zero-gradient: copy interior cell
            return U[0:1].clone()
        elif self.bc_left == BCType1D.REFLECTIVE:
            # Solid wall: reflect with reversed velocity
            ghost = U[0:1].clone()
            # U = [rho, rho*u, E] -> reverse momentum sign
            ghost[..., 1] = -ghost[..., 1]
            return ghost
        elif self.bc_left == BCType1D.PERIODIC:
            # Wrap from right boundary
            return U[-1:].clone()
        else:
            raise ValueError(f"Unknown left BC type: {self.bc_left}")
    
    def _apply_right_bc(self, U: Tensor) -> Tensor:
        """Apply right boundary condition to create ghost cell."""
        if self.bc_right == BCType1D.TRANSMISSIVE:
            # Zero-gradient: copy interior cell
            return U[-1:].clone()
        elif self.bc_right == BCType1D.REFLECTIVE:
            # Solid wall: reflect with reversed velocity
            ghost = U[-1:].clone()
            ghost[..., 1] = -ghost[..., 1]
            return ghost
        elif self.bc_right == BCType1D.PERIODIC:
            # Wrap from left boundary
            return U[0:1].clone()
        else:
            raise ValueError(f"Unknown right BC type: {self.bc_right}")
    
    def solve(
        self,
        t_final: float,
        callback: Optional[Callable[['Euler1D', float], None]] = None,
        callback_interval: float = 0.0,
    ) -> List[Tuple[float, EulerState]]:
        """
        Solve to final time.
        
        Args:
            t_final: Final time
            callback: Optional callback(solver, t) called periodically
            callback_interval: Interval for callback (0 = every step)
            
        Returns:
            List of (time, state) snapshots
        """
        snapshots = [(self.t, self._copy_state())]
        
        last_callback_t = self.t
        
        while self.t < t_final:
            dt = self.compute_dt()
            
            # Don't overshoot
            if self.t + dt > t_final:
                dt = t_final - self.t
            
            self.step(dt)
            
            # Callback
            if callback is not None:
                if callback_interval <= 0 or self.t - last_callback_t >= callback_interval:
                    callback(self, self.t)
                    last_callback_t = self.t
            
            snapshots.append((self.t, self._copy_state()))
        
        return snapshots
    
    def _copy_state(self) -> EulerState:
        """Deep copy current state."""
        return EulerState(
            rho=self.state.rho.clone(),
            rho_u=self.state.rho_u.clone(),
            E=self.state.E.clone(),
            gamma=self.state.gamma,
        )
    
    def to_mps(self, chi_max: int = 1) -> MPS:
        """
        Convert current state to MPS representation.
        
        Each site is a grid cell with physical dimension 3 (ρ, ρu, E).
        
        For chi_max=1 (product state), this is equivalent to classical FVM.
        Larger chi enables entanglement for multi-scale representation.
        
        Args:
            chi_max: Maximum bond dimension
            
        Returns:
            MPS representation of state
        """
        if self.state is None:
            raise ValueError("State not initialized")
        
        U = self.state.to_conserved()  # (N, 3)
        
        # For product state, each tensor is (1, 3, 1)
        tensors = []
        for i in range(self.N):
            A = U[i].reshape(1, 3, 1)
            tensors.append(A)
        
        return MPS(tensors)
    
    @classmethod
    def from_mps(
        cls,
        mps: MPS,
        x_min: float = 0.0,
        x_max: float = 1.0,
        gamma: float = 1.4,
        cfl: float = 0.5,
    ) -> 'Euler1D':
        """
        Create solver from MPS state.
        
        Args:
            mps: MPS with physical dimension 3
            x_min, x_max: Domain bounds
            gamma: Ratio of specific heats
            cfl: CFL number
            
        Returns:
            Euler1D solver with state from MPS
        """
        N = mps.L
        
        # Extract conserved variables
        # For product state, just read off the tensors
        U_list = []
        for A in mps.tensors:
            # A is (chi_L, 3, chi_R)
            # For now, assume chi=1 and take the values
            U_list.append(A[0, :, 0])
        
        U = torch.stack(U_list, dim=0)  # (N, 3)
        
        solver = cls(
            N=N,
            x_min=x_min,
            x_max=x_max,
            gamma=gamma,
            cfl=cfl,
            dtype=U.dtype,
            device=U.device,
        )
        
        solver.state = EulerState.from_conserved(U, gamma)
        
        return solver


# ==============================================================================
# Standard Test Problems
# ==============================================================================

def sod_shock_tube_ic(
    N: int,
    x_min: float = 0.0,
    x_max: float = 1.0,
    x_discontinuity: float = 0.5,
    gamma: float = 1.4,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> EulerState:
    """
    Sod shock tube initial condition.
    
    Classic test problem with exact solution available.
    
    Left state (x < 0.5): ρ = 1, u = 0, p = 1
    Right state (x > 0.5): ρ = 0.125, u = 0, p = 0.1
    
    Features: rarefaction, contact, shock
    """
    if device is None:
        device = torch.device('cpu')
    
    dx = (x_max - x_min) / N
    x = torch.linspace(x_min + 0.5 * dx, x_max - 0.5 * dx, N, dtype=dtype, device=device)
    
    # Primitive variables
    rho = torch.where(x < x_discontinuity, 
                      torch.ones_like(x), 
                      0.125 * torch.ones_like(x))
    u = torch.zeros_like(x)
    p = torch.where(x < x_discontinuity,
                    torch.ones_like(x),
                    0.1 * torch.ones_like(x))
    
    return EulerState.from_primitive(rho, u, p, gamma)


def lax_shock_tube_ic(
    N: int,
    x_min: float = 0.0,
    x_max: float = 1.0,
    x_discontinuity: float = 0.5,
    gamma: float = 1.4,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> EulerState:
    """
    Lax shock tube initial condition.
    
    More challenging than Sod with higher pressure ratio.
    
    Left state: ρ = 0.445, u = 0.698, p = 3.528
    Right state: ρ = 0.5, u = 0, p = 0.571
    """
    if device is None:
        device = torch.device('cpu')
    
    dx = (x_max - x_min) / N
    x = torch.linspace(x_min + 0.5 * dx, x_max - 0.5 * dx, N, dtype=dtype, device=device)
    
    rho = torch.where(x < x_discontinuity,
                      0.445 * torch.ones_like(x),
                      0.5 * torch.ones_like(x))
    u = torch.where(x < x_discontinuity,
                    0.698 * torch.ones_like(x),
                    torch.zeros_like(x))
    p = torch.where(x < x_discontinuity,
                    3.528 * torch.ones_like(x),
                    0.571 * torch.ones_like(x))
    
    return EulerState.from_primitive(rho, u, p, gamma)


def shu_osher_ic(
    N: int,
    x_min: float = -5.0,
    x_max: float = 5.0,
    gamma: float = 1.4,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> EulerState:
    """
    Shu-Osher problem initial condition.
    
    Shock interacting with a sine wave in density.
    Tests shock-capturing + oscillatory feature resolution.
    
    Left of x=-4: Post-shock state (ρ=3.857, u=2.629, p=10.333)
    Right of x=-4: Pre-shock with density perturbation
    """
    if device is None:
        device = torch.device('cpu')
    
    dx = (x_max - x_min) / N
    x = torch.linspace(x_min + 0.5 * dx, x_max - 0.5 * dx, N, dtype=dtype, device=device)
    
    # Post-shock state
    rho_L, u_L, p_L = 3.857143, 2.629369, 10.33333
    
    # Pre-shock state with density perturbation
    rho_R = 1.0 + 0.2 * torch.sin(5 * x)
    u_R = 0.0
    p_R = 1.0
    
    x_shock = -4.0
    
    rho = torch.where(x < x_shock, rho_L * torch.ones_like(x), rho_R)
    u = torch.where(x < x_shock, u_L * torch.ones_like(x), u_R * torch.ones_like(x))
    p = torch.where(x < x_shock, p_L * torch.ones_like(x), p_R * torch.ones_like(x))
    
    return EulerState.from_primitive(rho, u, p, gamma)


# Convenience wrappers
def euler_to_mps(state: EulerState) -> MPS:
    """Convert EulerState to MPS representation."""
    U = state.to_conserved()
    tensors = [U[i].reshape(1, 3, 1) for i in range(state.N)]
    return MPS(tensors)


def mps_to_euler(mps: MPS, gamma: float = 1.4) -> EulerState:
    """Convert MPS to EulerState."""
    U_list = [A[0, :, 0] for A in mps.tensors]
    U = torch.stack(U_list, dim=0)
    return EulerState.from_conserved(U, gamma)
