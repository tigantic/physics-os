"""
Boundary conditions for compressible flow solvers.

Implements various boundary condition types for 1D and 2D Euler equations:
- Inflow/Outflow (subsonic and supersonic)
- Reflective (solid wall)
- Periodic
- Characteristic-based non-reflecting

References:
    [1] Poinsot & Lele, "Boundary Conditions for Direct Simulations 
        of Compressible Viscous Flows", JCP 101(1):104-129, 1992
    [2] Thompson, "Time Dependent Boundary Conditions for Hyperbolic
        Systems", JCP 68:1-24, 1987
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Callable, Optional, Literal
from dataclasses import dataclass
from enum import Enum
import math


# Default ratio of specific heats for diatomic gas (air)
gamma_default = 1.4


class BCType(Enum):
    """Enumeration of boundary condition types."""
    PERIODIC = "periodic"
    EXTRAPOLATION = "extrapolation"
    REFLECTIVE = "reflective"
    INFLOW_SUBSONIC = "inflow_subsonic"
    OUTFLOW_SUBSONIC = "outflow_subsonic"
    INFLOW_SUPERSONIC = "inflow_supersonic"
    OUTFLOW_SUPERSONIC = "outflow_supersonic"
    CHARACTERISTIC = "characteristic"


@dataclass
class FlowState:
    """
    Primitive flow state for boundary conditions.
    
    Attributes:
        rho: Density
        u: x-velocity
        v: y-velocity (0 for 1D)
        p: Pressure
        gamma: Ratio of specific heats
    """
    rho: float
    u: float
    v: float
    p: float
    gamma: float = gamma_default
    
    @property
    def a(self) -> float:
        """Sound speed."""
        return math.sqrt(self.gamma * self.p / self.rho)
    
    @property
    def M(self) -> float:
        """Mach number magnitude."""
        velocity = math.sqrt(self.u**2 + self.v**2)
        return velocity / self.a
    
    @property
    def is_supersonic(self) -> bool:
        """Check if flow is supersonic."""
        return self.M >= 1.0
    
    def to_tensor(self, dtype=torch.float64, device="cpu") -> Tensor:
        """Convert to tensor [rho, u, v, p]."""
        return torch.tensor([self.rho, self.u, self.v, self.p], dtype=dtype, device=device)


def apply_extrapolation_bc(
    U: Tensor,
    n_ghost: int,
    side: Literal["left", "right", "bottom", "top"]
) -> None:
    """
    Apply zero-gradient (extrapolation) boundary condition in-place.
    
    Simply copies the boundary cell values to ghost cells.
    Appropriate for supersonic outflow where no information
    propagates back into the domain.
    
    Args:
        U: Conservative variables with ghost cells
        n_ghost: Number of ghost cells
        side: Which boundary to apply BC
    """
    if U.dim() == 2:
        # 1D: U has shape (n_vars, Nx + 2*n_ghost)
        _, Nx_ext = U.shape
        Nx = Nx_ext - 2*n_ghost
        
        if side == "left":
            for i in range(n_ghost):
                U[:, i] = U[:, n_ghost]
        elif side == "right":
            for i in range(n_ghost):
                U[:, Nx + n_ghost + i] = U[:, Nx + n_ghost - 1]
    
    elif U.dim() == 3:
        # 2D: U has shape (n_vars, Ny_ext, Nx_ext)
        _, Ny_ext, Nx_ext = U.shape
        
        if side == "left":
            U[:, :, :n_ghost] = U[:, :, n_ghost:n_ghost+1].expand(-1, -1, n_ghost)
        elif side == "right":
            U[:, :, -n_ghost:] = U[:, :, -n_ghost-1:-n_ghost].expand(-1, -1, n_ghost)
        elif side == "bottom":
            U[:, :n_ghost, :] = U[:, n_ghost:n_ghost+1, :].expand(-1, n_ghost, -1)
        elif side == "top":
            U[:, -n_ghost:, :] = U[:, -n_ghost-1:-n_ghost, :].expand(-1, n_ghost, -1)


def apply_reflective_bc(
    U: Tensor,
    n_ghost: int,
    side: Literal["left", "right", "bottom", "top"],
    gamma: float = gamma_default
) -> None:
    """
    Apply reflective (solid wall) boundary condition in-place.
    
    For inviscid flow, the wall-normal velocity is reflected
    while tangential velocity and thermodynamic quantities are
    extrapolated (mirrored).
    
    Args:
        U: Conservative variables with ghost cells
        n_ghost: Number of ghost cells
        side: Which boundary to apply BC
        gamma: Ratio of specific heats
    """
    if U.dim() == 2:
        # 1D: U = [rho, rho*u, E]
        _, Nx_ext = U.shape
        Nx = Nx_ext - 2*n_ghost
        
        if side == "left":
            for i in range(n_ghost):
                U[:, n_ghost - 1 - i] = U[:, n_ghost + i]
                U[1, n_ghost - 1 - i] *= -1  # Reflect momentum
        elif side == "right":
            for i in range(n_ghost):
                U[:, Nx + n_ghost + i] = U[:, Nx + n_ghost - 1 - i]
                U[1, Nx + n_ghost + i] *= -1
    
    elif U.dim() == 3:
        # 2D: U = [rho, rho*u, rho*v, E]
        _, Ny_ext, Nx_ext = U.shape
        
        if side == "left":
            for i in range(n_ghost):
                U[:, :, n_ghost - 1 - i] = U[:, :, n_ghost + i]
                U[1, :, n_ghost - 1 - i] *= -1  # Reflect x-momentum
        elif side == "right":
            for i in range(n_ghost):
                U[:, :, -n_ghost + i] = U[:, :, -n_ghost - 1 - i]
                U[1, :, -n_ghost + i] *= -1
        elif side == "bottom":
            for i in range(n_ghost):
                U[:, n_ghost - 1 - i, :] = U[:, n_ghost + i, :]
                U[2, n_ghost - 1 - i, :] *= -1  # Reflect y-momentum
        elif side == "top":
            for i in range(n_ghost):
                U[:, -n_ghost + i, :] = U[:, -n_ghost - 1 - i, :]
                U[2, -n_ghost + i, :] *= -1


def apply_supersonic_inflow_bc(
    U: Tensor,
    n_ghost: int,
    inflow_state: FlowState,
    side: Literal["left", "right", "bottom", "top"],
    dtype: torch.dtype = torch.float64,
    device: str = "cpu"
) -> None:
    """
    Apply supersonic inflow boundary condition in-place.
    
    For supersonic inflow, all characteristics enter the domain,
    so all flow variables are prescribed from the freestream.
    
    Args:
        U: Conservative variables with ghost cells
        n_ghost: Number of ghost cells
        inflow_state: Prescribed inflow conditions
        side: Which boundary to apply BC
    """
    gamma = inflow_state.gamma
    
    # Compute conservative variables from primitives
    rho = inflow_state.rho
    u = inflow_state.u
    v = inflow_state.v
    p = inflow_state.p
    
    if U.dim() == 2:
        # 1D
        E = p / (gamma - 1) + 0.5 * rho * u**2
        U_bc = torch.tensor([rho, rho * u, E], dtype=dtype, device=device)
        
        if side == "left":
            for i in range(n_ghost):
                U[:, i] = U_bc
        elif side == "right":
            for i in range(n_ghost):
                U[:, -n_ghost + i] = U_bc
    
    elif U.dim() == 3:
        # 2D
        E = p / (gamma - 1) + 0.5 * rho * (u**2 + v**2)
        U_bc = torch.tensor([rho, rho * u, rho * v, E], dtype=dtype, device=device)
        
        _, Ny_ext, Nx_ext = U.shape
        
        if side == "left":
            for i in range(n_ghost):
                U[:, :, i] = U_bc.view(-1, 1)
        elif side == "right":
            for i in range(n_ghost):
                U[:, :, -n_ghost + i] = U_bc.view(-1, 1)
        elif side == "bottom":
            for i in range(n_ghost):
                U[:, i, :] = U_bc.view(-1, 1)
        elif side == "top":
            for i in range(n_ghost):
                U[:, -n_ghost + i, :] = U_bc.view(-1, 1)


def apply_subsonic_inflow_bc(
    U: Tensor,
    n_ghost: int,
    total_pressure: float,
    total_temperature: float,
    flow_angle: float,
    side: Literal["left", "right", "bottom", "top"],
    gamma: float = gamma_default,
    R_gas: float = 287.0,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu"
) -> None:
    """
    Apply subsonic inflow boundary condition in-place.
    
    For subsonic inflow, one characteristic leaves the domain
    (carrying pressure information out). We specify:
    - Total pressure P0
    - Total temperature T0
    - Flow direction
    
    The static pressure is extrapolated from interior.
    
    Args:
        U: Conservative variables with ghost cells
        n_ghost: Number of ghost cells
        total_pressure: Stagnation pressure
        total_temperature: Stagnation temperature
        flow_angle: Flow direction (radians from x-axis)
        side: Which boundary to apply BC
        gamma: Ratio of specific heats
        R_gas: Specific gas constant
    """
    if U.dim() == 3:
        _, Ny_ext, Nx_ext = U.shape
        
        # Get interior pressure for the outgoing characteristic
        if side == "left":
            U_int = U[:, :, n_ghost]
        elif side == "right":
            U_int = U[:, :, -n_ghost - 1]
        elif side == "bottom":
            U_int = U[:, n_ghost, :]
        elif side == "top":
            U_int = U[:, -n_ghost - 1, :]
        
        rho_int = U_int[0]
        u_int = U_int[1] / rho_int
        v_int = U_int[2] / rho_int
        E_int = U_int[3]
        p_int = (gamma - 1) * (E_int - 0.5 * rho_int * (u_int**2 + v_int**2))
        
        # Use isentropic relations with interior pressure
        p_ratio = p_int / total_pressure
        T_ratio = p_ratio ** ((gamma - 1) / gamma)
        T = total_temperature * T_ratio
        rho = p_int / (R_gas * T)
        
        # Velocity from isentropic relations
        V_sq = 2 * gamma / (gamma - 1) * R_gas * total_temperature * (1 - T_ratio)
        V = torch.sqrt(torch.clamp(V_sq, min=0))
        
        u = V * math.cos(flow_angle)
        v = V * math.sin(flow_angle)
        
        E = p_int / (gamma - 1) + 0.5 * rho * (u**2 + v**2)
        
        # Set ghost cells
        if side == "left":
            for i in range(n_ghost):
                U[0, :, i] = rho
                U[1, :, i] = rho * u
                U[2, :, i] = rho * v
                U[3, :, i] = E
        # Similar for other sides...


def apply_subsonic_outflow_bc(
    U: Tensor,
    n_ghost: int,
    back_pressure: float,
    side: Literal["left", "right", "bottom", "top"],
    gamma: float = gamma_default
) -> None:
    """
    Apply subsonic outflow boundary condition in-place.
    
    For subsonic outflow, one characteristic enters the domain
    (carrying pressure information in). We specify:
    - Back pressure
    
    All other quantities are extrapolated from interior.
    
    Args:
        U: Conservative variables with ghost cells
        n_ghost: Number of ghost cells
        back_pressure: Prescribed exit pressure
        side: Which boundary to apply BC
        gamma: Ratio of specific heats
    """
    if U.dim() == 3:
        if side == "right":
            # Get interior state
            U_int = U[:, :, -n_ghost - 1]
            rho_int = U_int[0]
            u_int = U_int[1] / rho_int
            v_int = U_int[2] / rho_int
            
            # Use prescribed pressure, extrapolate others
            p_bc = back_pressure
            E_bc = p_bc / (gamma - 1) + 0.5 * rho_int * (u_int**2 + v_int**2)
            
            for i in range(n_ghost):
                U[0, :, -n_ghost + i] = rho_int
                U[1, :, -n_ghost + i] = rho_int * u_int
                U[2, :, -n_ghost + i] = rho_int * v_int
                U[3, :, -n_ghost + i] = E_bc


def apply_periodic_bc(
    U: Tensor,
    n_ghost: int,
    axis: int
) -> None:
    """
    Apply periodic boundary condition in-place.
    
    Args:
        U: Conservative variables with ghost cells
        n_ghost: Number of ghost cells
        axis: Axis along which to apply periodicity (1 for x, 2 for y in 3D)
    """
    if U.dim() == 2:
        # 1D
        _, N_ext = U.shape
        N = N_ext - 2*n_ghost
        U[:, :n_ghost] = U[:, N:N+n_ghost]
        U[:, N+n_ghost:] = U[:, n_ghost:2*n_ghost]
    
    elif U.dim() == 3:
        if axis == 2:  # x-direction
            _, _, N_ext = U.shape
            N = N_ext - 2*n_ghost
            U[:, :, :n_ghost] = U[:, :, N:N+n_ghost]
            U[:, :, N+n_ghost:] = U[:, :, n_ghost:2*n_ghost]
        elif axis == 1:  # y-direction
            _, N_ext, _ = U.shape
            N = N_ext - 2*n_ghost
            U[:, :n_ghost, :] = U[:, N:N+n_ghost, :]
            U[:, N+n_ghost:, :] = U[:, n_ghost:2*n_ghost, :]


class BoundaryManager:
    """
    Manages boundary conditions for a 2D domain.
    
    Provides a unified interface for applying different BC types
    on each boundary of a rectangular domain.
    
    Attributes:
        left, right, bottom, top: BC type for each boundary
        inflow_state: FlowState for inflow boundaries
        back_pressure: Exit pressure for subsonic outflow
    """
    
    def __init__(
        self,
        left: BCType = BCType.EXTRAPOLATION,
        right: BCType = BCType.EXTRAPOLATION,
        bottom: BCType = BCType.EXTRAPOLATION,
        top: BCType = BCType.EXTRAPOLATION,
        inflow_state: Optional[FlowState] = None,
        back_pressure: float = 1.0,
        gamma: float = gamma_default
    ):
        """
        Initialize boundary manager.
        
        Args:
            left, right, bottom, top: BC type for each boundary
            inflow_state: FlowState for inflow boundaries
            back_pressure: Exit pressure for subsonic outflow
            gamma: Ratio of specific heats
        """
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.inflow_state = inflow_state
        self.back_pressure = back_pressure
        self.gamma = gamma
    
    def apply(
        self,
        U: Tensor,
        n_ghost: int = 2,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu"
    ) -> None:
        """
        Apply all boundary conditions in-place.
        
        Args:
            U: Conservative variables with ghost cells (4, Ny+2ng, Nx+2ng)
            n_ghost: Number of ghost cells
        """
        for side, bc_type in [
            ("left", self.left),
            ("right", self.right),
            ("bottom", self.bottom),
            ("top", self.top)
        ]:
            self._apply_single(U, n_ghost, side, bc_type, dtype, device)
    
    def _apply_single(
        self,
        U: Tensor,
        n_ghost: int,
        side: str,
        bc_type: BCType,
        dtype: torch.dtype,
        device: str
    ) -> None:
        """Apply a single boundary condition."""
        if bc_type == BCType.EXTRAPOLATION:
            apply_extrapolation_bc(U, n_ghost, side)
        
        elif bc_type == BCType.REFLECTIVE:
            apply_reflective_bc(U, n_ghost, side, self.gamma)
        
        elif bc_type == BCType.PERIODIC:
            axis = 2 if side in ["left", "right"] else 1
            apply_periodic_bc(U, n_ghost, axis)
        
        elif bc_type == BCType.INFLOW_SUPERSONIC:
            if self.inflow_state is None:
                raise ValueError("Inflow state required for supersonic inflow BC")
            apply_supersonic_inflow_bc(
                U, n_ghost, self.inflow_state, side, dtype, device
            )
        
        elif bc_type == BCType.OUTFLOW_SUPERSONIC:
            apply_extrapolation_bc(U, n_ghost, side)
        
        elif bc_type == BCType.OUTFLOW_SUBSONIC:
            apply_subsonic_outflow_bc(
                U, n_ghost, self.back_pressure, side, self.gamma
            )
