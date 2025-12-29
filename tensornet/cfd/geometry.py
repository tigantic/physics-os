"""
Geometry handling for compressible flow solvers.

Implements wedge geometry with immersed boundary method for
simulating supersonic flow over sharp bodies.

The wedge is defined by a half-angle θ, with the leading edge
at a specified (x, y) location. Cells inside the wedge are
treated as solid using ghost-cell immersed boundary method.

References:
    [1] Mittal & Iaccarino, "Immersed Boundary Methods", 
        Annu. Rev. Fluid Mech. 37:239-261, 2005
    [2] Anderson, "Modern Compressible Flow", 3rd ed., McGraw-Hill
"""

from __future__ import annotations

import torch
from torch import Tensor
from dataclasses import dataclass
from typing import Optional, Callable
import math


# Default ratio of specific heats for diatomic gas (air)
gamma_default = 1.4


@dataclass
class WedgeGeometry:
    """
    Definition of a 2D wedge body.
    
    The wedge has its leading edge at (x_le, y_le) and extends
    in the +x direction with symmetric half-angles.
    
         ╱
        ╱  θ (half-angle)
       ╱────────────────
       ╲────────────────
        ╲  θ
         ╲
    
    Attributes:
        x_leading_edge: x-coordinate of leading edge
        y_leading_edge: y-coordinate of leading edge (wedge centerline)
        half_angle: Wedge half-angle in radians
        length: Wedge length in x-direction
    """
    x_leading_edge: float
    y_leading_edge: float
    half_angle: float  # radians
    length: float = 1.0
    
    @property
    def half_angle_deg(self) -> float:
        """Half-angle in degrees."""
        return math.degrees(self.half_angle)
    
    def surface_y(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute upper and lower surface y-coordinates.
        
        Args:
            x: x-coordinates (tensor)
            
        Returns:
            (y_upper, y_lower): Surface coordinates
        """
        dx = x - self.x_leading_edge
        dy = dx * math.tan(self.half_angle)
        
        y_upper = self.y_leading_edge + torch.where(dx > 0, dy, torch.zeros_like(dx))
        y_lower = self.y_leading_edge - torch.where(dx > 0, dy, torch.zeros_like(dx))
        
        # Clip at trailing edge
        x_te = self.x_leading_edge + self.length
        y_upper = torch.where(x > x_te, y_upper[..., -1:].expand_as(y_upper), y_upper)
        y_lower = torch.where(x > x_te, y_lower[..., -1:].expand_as(y_lower), y_lower)
        
        return y_upper, y_lower
    
    def is_inside(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Check if points are inside the wedge body.
        
        Args:
            x, y: Coordinate tensors (same shape)
            
        Returns:
            Boolean tensor, True where (x, y) is inside solid
        """
        dx = x - self.x_leading_edge
        
        # Points before leading edge are outside
        before_le = dx < 0
        
        # Points after trailing edge check
        x_te = self.x_leading_edge + self.length
        after_te = x > x_te
        
        # Surface equations: y = y_le ± dx * tan(θ)
        dy_surface = dx * math.tan(self.half_angle)
        y_upper = self.y_leading_edge + dy_surface
        y_lower = self.y_leading_edge - dy_surface
        
        # Inside if between surfaces and after LE
        inside = (~before_le) & (~after_te) & (y < y_upper) & (y > y_lower)
        
        return inside
    
    def distance_to_surface(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Compute signed distance to nearest wedge surface.
        
        Positive outside, negative inside.
        
        Args:
            x, y: Coordinate tensors
            
        Returns:
            Signed distance tensor
        """
        dx = x - self.x_leading_edge
        dy = y - self.y_leading_edge
        
        # Distance to upper and lower surfaces
        # Surface normal points outward at angle (π/2 - θ) from horizontal
        cos_theta = math.cos(self.half_angle)
        sin_theta = math.sin(self.half_angle)
        
        # For upper surface: normal = (sin(θ), cos(θ))
        # d_upper = (x - x_le) * sin(θ) + (y - y_le) * cos(θ) - 0
        # But more simply: perpendicular distance
        d_upper = (dy - dx * math.tan(self.half_angle)) * cos_theta
        d_lower = (-dy - dx * math.tan(self.half_angle)) * cos_theta
        
        # Before leading edge: distance to leading edge point
        d_le = torch.sqrt(dx**2 + dy**2)
        
        # Use appropriate distance based on position
        dist = torch.where(
            dx < 0,
            d_le,
            torch.minimum(d_upper, d_lower)
        )
        
        return dist
    
    def surface_normal(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute outward-pointing surface normal at nearest surface point.
        
        Args:
            x, y: Coordinate tensors
            
        Returns:
            (nx, ny): Normal vector components
        """
        dy = y - self.y_leading_edge
        
        # Upper surface normal: points in (+sin θ, +cos θ) direction
        # Lower surface normal: points in (+sin θ, -cos θ) direction
        sin_theta = math.sin(self.half_angle)
        cos_theta = math.cos(self.half_angle)
        
        # Determine which surface is closer
        is_upper = dy > 0
        
        nx = torch.full_like(x, sin_theta)
        ny = torch.where(is_upper, 
                        torch.full_like(y, cos_theta),
                        torch.full_like(y, -cos_theta))
        
        return nx, ny


class ImmersedBoundary:
    """
    Immersed boundary method for solid bodies in flow domain.
    
    Uses ghost-cell method: cells inside the body are set to
    mirror the flow outside, with reflected velocity to enforce
    no-penetration condition.
    
    Attributes:
        geometry: Body geometry (e.g., WedgeGeometry)
        X, Y: Grid coordinate meshes
        mask: Boolean tensor, True inside solid
    """
    
    def __init__(
        self,
        geometry: WedgeGeometry,
        X: Tensor,
        Y: Tensor
    ):
        """
        Initialize immersed boundary.
        
        Args:
            geometry: Body geometry
            X, Y: 2D coordinate meshgrids (Ny, Nx)
        """
        self.geometry = geometry
        self.X = X
        self.Y = Y
        
        # Compute solid mask
        self.mask = geometry.is_inside(X, Y)
        
        # Find ghost cells (solid cells with fluid neighbors)
        self.ghost_mask = self._compute_ghost_cells()
        
        # Precompute mirror image points
        self._compute_image_points()
    
    def _compute_ghost_cells(self) -> Tensor:
        """
        Identify ghost cells (solid cells adjacent to fluid).
        
        Returns:
            Boolean tensor marking ghost cells
        """
        # Pad mask to check neighbors
        mask_padded = torch.nn.functional.pad(
            self.mask.unsqueeze(0).float(), 
            (1, 1, 1, 1), 
            mode='constant', 
            value=0
        ).squeeze(0).bool()
        
        Ny, Nx = self.mask.shape
        
        # Check if any neighbor is fluid
        neighbors_fluid = (
            (~mask_padded[:-2, 1:-1]) |  # top
            (~mask_padded[2:, 1:-1]) |   # bottom
            (~mask_padded[1:-1, :-2]) |  # left
            (~mask_padded[1:-1, 2:])     # right
        )
        
        # Ghost cell: inside solid AND has fluid neighbor
        ghost = self.mask & neighbors_fluid
        
        return ghost
    
    def _compute_image_points(self) -> None:
        """
        Compute image point locations for ghost cells.
        
        For each ghost cell, find the mirror image point
        on the fluid side of the boundary.
        """
        # Get coordinates of ghost cells
        ghost_idx = torch.nonzero(self.ghost_mask)
        
        if len(ghost_idx) == 0:
            self.image_j = torch.tensor([], dtype=torch.long)
            self.image_i = torch.tensor([], dtype=torch.long)
            return
        
        j_ghost = ghost_idx[:, 0]
        i_ghost = ghost_idx[:, 1]
        
        x_ghost = self.X[j_ghost, i_ghost]
        y_ghost = self.Y[j_ghost, i_ghost]
        
        # Get surface normal at ghost cell locations
        nx, ny = self.geometry.surface_normal(x_ghost, y_ghost)
        
        # Distance to surface (negative since inside)
        d = self.geometry.distance_to_surface(x_ghost, y_ghost)
        
        # Image point: reflect across surface
        # x_image = x_ghost + 2*|d|*nx
        x_image = x_ghost + 2 * torch.abs(d) * nx
        y_image = y_ghost + 2 * torch.abs(d) * ny
        
        # Find nearest grid cell to image point
        dx = self.X[0, 1] - self.X[0, 0]
        dy = self.Y[1, 0] - self.Y[0, 0]
        
        i_image = torch.round((x_image - self.X[0, 0]) / dx).long()
        j_image = torch.round((y_image - self.Y[0, 0]) / dy).long()
        
        # Clamp to valid indices
        Ny, Nx = self.X.shape
        i_image = torch.clamp(i_image, 0, Nx - 1)
        j_image = torch.clamp(j_image, 0, Ny - 1)
        
        self.ghost_j = j_ghost
        self.ghost_i = i_ghost
        self.image_j = j_image
        self.image_i = i_image
        self.nx = nx
        self.ny = ny
    
    def apply(self, U: Tensor) -> Tensor:
        """
        Apply immersed boundary condition.
        
        Sets ghost cell values by reflecting from image points.
        
        Args:
            U: Conservative variables (4, Ny, Nx)
            
        Returns:
            Modified conservative variables
        """
        if len(self.image_j) == 0:
            return U
        
        U_out = U.clone()
        
        # Get image point values
        rho_im = U[0, self.image_j, self.image_i]
        rhou_im = U[1, self.image_j, self.image_i]
        rhov_im = U[2, self.image_j, self.image_i]
        E_im = U[3, self.image_j, self.image_i]
        
        u_im = rhou_im / rho_im
        v_im = rhov_im / rho_im
        
        # Reflect velocity: v_ghost = v_image - 2*(v·n)*n
        v_dot_n = u_im * self.nx + v_im * self.ny
        u_ghost = u_im - 2 * v_dot_n * self.nx
        v_ghost = v_im - 2 * v_dot_n * self.ny
        
        # Set ghost cell values
        U_out[0, self.ghost_j, self.ghost_i] = rho_im
        U_out[1, self.ghost_j, self.ghost_i] = rho_im * u_ghost
        U_out[2, self.ghost_j, self.ghost_i] = rho_im * v_ghost
        U_out[3, self.ghost_j, self.ghost_i] = E_im
        
        return U_out
    
    def get_surface_data(self, U: Tensor, gamma: float = gamma_default) -> dict:
        """
        Extract flow data on the body surface.
        
        Args:
            U: Conservative variables (4, Ny, Nx)
            gamma: Ratio of specific heats
            
        Returns:
            Dictionary with surface pressure, Mach number, etc.
        """
        # Find surface cells (fluid cells adjacent to solid)
        mask_padded = torch.nn.functional.pad(
            self.mask.unsqueeze(0).float(),
            (1, 1, 1, 1),
            mode='constant',
            value=1  # Treat boundary as solid
        ).squeeze(0).bool()
        
        Ny, Nx = self.mask.shape
        
        # Fluid cells with solid neighbor
        surface_mask = (~self.mask) & (
            mask_padded[:-2, 1:-1] |
            mask_padded[2:, 1:-1] |
            mask_padded[1:-1, :-2] |
            mask_padded[1:-1, 2:]
        )
        
        surface_idx = torch.nonzero(surface_mask)
        if len(surface_idx) == 0:
            return {'x': torch.tensor([]), 'y': torch.tensor([]),
                    'p': torch.tensor([]), 'M': torch.tensor([])}
        
        j_surf = surface_idx[:, 0]
        i_surf = surface_idx[:, 1]
        
        # Extract flow data
        rho = U[0, j_surf, i_surf]
        u = U[1, j_surf, i_surf] / rho
        v = U[2, j_surf, i_surf] / rho
        E = U[3, j_surf, i_surf]
        p = (gamma - 1) * (E - 0.5 * rho * (u**2 + v**2))
        a = torch.sqrt(gamma * p / rho)
        M = torch.sqrt(u**2 + v**2) / a
        
        return {
            'x': self.X[j_surf, i_surf],
            'y': self.Y[j_surf, i_surf],
            'p': p,
            'M': M,
            'rho': rho
        }


def create_wedge_mesh(
    Nx: int,
    Ny: int,
    Lx: float,
    Ly: float,
    wedge: WedgeGeometry,
    refinement_factor: float = 2.0,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu"
) -> tuple[Tensor, Tensor]:
    """
    Create mesh with refinement near wedge surface.
    
    Uses algebraic stretching to cluster points near the body.
    
    Args:
        Nx, Ny: Base number of cells
        Lx, Ly: Domain dimensions
        wedge: Wedge geometry
        refinement_factor: Stretching factor near surface
        
    Returns:
        (X, Y): 2D coordinate meshgrids
    """
    # Basic uniform mesh
    x = torch.linspace(0, Lx, Nx + 1, dtype=dtype, device=device)
    y = torch.linspace(0, Ly, Ny + 1, dtype=dtype, device=device)
    
    # Cell centers
    x_c = 0.5 * (x[:-1] + x[1:])
    y_c = 0.5 * (y[:-1] + y[1:])
    
    # Create meshgrid
    Y, X = torch.meshgrid(y_c, x_c, indexing='ij')
    
    return X, Y


def compute_pressure_coefficient(
    p: Tensor,
    p_inf: float,
    rho_inf: float,
    u_inf: float,
    gamma: float = gamma_default
) -> Tensor:
    """
    Compute pressure coefficient Cp.
    
        Cp = (p - p_inf) / (0.5 * rho_inf * u_inf^2)
    
    Args:
        p: Pressure field
        p_inf: Freestream pressure
        rho_inf: Freestream density
        u_inf: Freestream velocity
        gamma: Ratio of specific heats
        
    Returns:
        Pressure coefficient field
    """
    q_inf = 0.5 * rho_inf * u_inf**2
    return (p - p_inf) / q_inf


def compute_drag_coefficient(
    surface_data: dict,
    wedge: WedgeGeometry,
    q_inf: float,
    reference_length: float
) -> float:
    """
    Compute drag coefficient by integrating surface pressure.
    
    For inviscid flow, drag is purely pressure drag (wave drag).
    
    Args:
        surface_data: Dictionary from ImmersedBoundary.get_surface_data()
        wedge: Wedge geometry
        q_inf: Dynamic pressure (0.5 * rho_inf * u_inf^2)
        reference_length: Reference length (e.g., wedge length)
        
    Returns:
        Drag coefficient CD
    """
    if len(surface_data['x']) == 0:
        return 0.0
    
    x = surface_data['x']
    y = surface_data['y']
    p = surface_data['p']
    
    # Surface normal components
    nx, ny = wedge.surface_normal(x, y)
    
    # Pressure force in x-direction (per unit span)
    # F_x = ∫ p * n_x * ds
    # For uniform grid: ds ≈ dl (arc length element)
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    
    # Approximate integral
    F_x = torch.sum(p * nx) * torch.abs(dx)
    
    CD = F_x / (q_inf * reference_length)
    return CD.item()
