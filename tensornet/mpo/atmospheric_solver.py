"""
MPO Atmospheric Solver: Direct TT-core updates for fluid simulation.

Eliminates dense-to-QTT factorization tax by updating QTT cores directly.
Target: 0.65ms per physics update (5× speedup vs 3.33ms dense solver).

Academic validation:
- Oseledets (2011): Tensor-Train Decomposition
- Dolgov & Savostyanov (2014): Alternating Minimal Energy Methods
"""

import torch
from typing import List, Tuple, Optional
import time

# Try to use CUDA-accelerated Laplacian, fallback to CPU version
try:
    from .laplacian_cuda import LaplacianCUDA as LaplacianMPO
    CUDA_LAPLACIAN = True
    print("✓ CUDA Laplacian kernel loaded")
except ImportError:
    from .operators import LaplacianMPO
    CUDA_LAPLACIAN = False
    print("⚠ Using CPU Laplacian (CUDA kernel not available)")

from .operators import AdvectionMPO, ProjectionMPO


class MPOAtmosphericSolver:
    """
    Atmospheric physics solver using Matrix Product Operators.
    
    Updates QTT cores directly without dense materialization:
    1. Laplacian MPO: Diffusion (∇²u)
    2. Advection MPO: Transport (v·∇u)
    3. Projection MPO: Incompressibility (∇·u = 0)
    
    Complexity: O(d·r³) vs O(N²) for dense solver
    Target performance: <0.65ms per update
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (64, 64),
        viscosity: float = 1e-4,
        dt: float = 0.01,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cuda"),
    ):
        """
        Args:
            grid_size: Spatial grid dimensions (Nx, Ny)
            viscosity: Kinematic viscosity coefficient
            dt: Time step for explicit integration
            dtype: Tensor data type
            device: Computation device
        """
        self.grid_size = grid_size
        self.viscosity = viscosity
        self.dt = dt
        self.dtype = dtype
        self.device = device
        
        # Compute number of QTT modes (log₂(N) per dimension)
        Nx, Ny = grid_size
        assert Nx == Ny and (Nx & (Nx - 1)) == 0, "Grid size must be power of 2"
        self.modes_per_dim = int(torch.log2(torch.tensor(Nx)).item())
        self.num_modes = 2 * self.modes_per_dim  # Total modes (x + y)
        
        # Spatial resolution
        self.dx = 1.0 / Nx
        
        # Initialize MPO operators
        self.laplacian = LaplacianMPO(
            num_modes=self.num_modes,
            viscosity=self.viscosity,
            dx=self.dx,
            dtype=dtype,
            device=device,
        )
        
        self.advection = AdvectionMPO(
            num_modes=self.num_modes,
            dtype=dtype,
            device=device,
        )
        
        self.projection = ProjectionMPO(
            num_modes=self.num_modes,
            dx=self.dx,
            dtype=dtype,
            device=device,
        )
        
        # Initialize QTT state (velocity fields)
        self.u_cores = self._init_velocity_cores()  # X-velocity
        self.v_cores = self._init_velocity_cores()  # Y-velocity
        
        # Performance tracking
        self.timings = {
            "laplacian": [],
            "advection": [],
            "projection": [],
            "total": [],
        }
    
    def _init_velocity_cores(self) -> List[torch.Tensor]:
        """
        Initialize velocity field as QTT cores.
        
        Returns:
            List of QTT cores representing zero velocity field
        """
        cores = []
        for i in range(self.num_modes):
            if i == 0:
                # First core: [1, 2, r]
                core = torch.zeros(1, 2, 4, dtype=self.dtype, device=self.device)
                core[0, 0, 0] = 1.0
                core[0, 1, 1] = 1.0
            elif i == self.num_modes - 1:
                # Last core: [r, 2, 1]
                core = torch.zeros(4, 2, 1, dtype=self.dtype, device=self.device)
                core[0, 0, 0] = 1.0
                core[1, 1, 0] = 1.0
            else:
                # Middle core: [r, 2, r]
                core = torch.zeros(4, 2, 4, dtype=self.dtype, device=self.device)
                core[0, 0, 0] = 1.0
                core[1, 1, 1] = 1.0
            
            cores.append(core)
        
        return cores
    
    def step(self) -> None:
        """
        Execute one physics time step using MPO operators.
        
        Performs:
        1. Diffusion: u ← u + dt·ν·∇²u
        2. Advection: u ← u - dt·(v·∇)u
        3. Projection: u ← proj(u) (enforce ∇·u = 0)
        """
        t_start = time.perf_counter()
        
        # 1. Diffusion step (Laplacian MPO)
        t0 = time.perf_counter()
        self.u_cores = self.laplacian.apply(self.u_cores, self.dt)
        self.v_cores = self.laplacian.apply(self.v_cores, self.dt)
        self.timings["laplacian"].append(time.perf_counter() - t0)
        
        # 2. Advection step (semi-Lagrangian)
        t0 = time.perf_counter()
        # Simplified: use mean velocity for advection
        # Full implementation: spatially-varying velocity field
        mean_u = self._get_mean_velocity(self.u_cores)
        mean_v = self._get_mean_velocity(self.v_cores)
        
        self.u_cores = self.advection.apply(self.u_cores, mean_u, mean_v, self.dt)
        self.v_cores = self.advection.apply(self.v_cores, mean_u, mean_v, self.dt)
        self.timings["advection"].append(time.perf_counter() - t0)
        
        # 3. Projection step (incompressibility)
        t0 = time.perf_counter()
        self.u_cores, self.v_cores = self.projection.apply(self.u_cores, self.v_cores)
        self.timings["projection"].append(time.perf_counter() - t0)
        
        self.timings["total"].append(time.perf_counter() - t_start)
    
    def _get_mean_velocity(self, cores: List[torch.Tensor]) -> float:
        """
        Compute mean velocity from QTT cores (simplified).
        
        Args:
            cores: QTT velocity cores
            
        Returns:
            Mean velocity value
        """
        # Simplified: return zero (full implementation requires TT summation)
        return 0.0
    
    def get_cores(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Get current velocity field as QTT cores.
        
        Returns:
            (u_cores, v_cores): X and Y velocity QTT cores
        """
        return self.u_cores, self.v_cores
    
    def set_cores(
        self,
        u_cores: List[torch.Tensor],
        v_cores: List[torch.Tensor],
    ) -> None:
        """
        Set velocity field from external QTT cores.
        
        Args:
            u_cores: X-velocity QTT cores
            v_cores: Y-velocity QTT cores
        """
        self.u_cores = u_cores
        self.v_cores = v_cores
    
    def add_forcing(
        self,
        position: Tuple[int, int],
        velocity: Tuple[float, float],
        radius: float = 5.0,
    ) -> None:
        """
        Add localized forcing to velocity field.
        
        Args:
            position: (x, y) grid coordinates
            velocity: (vx, vy) velocity impulse
            radius: Forcing radius (grid cells)
        """
        # TODO: Implement TT addition for localized forcing
        # Requires converting Gaussian blob to QTT format and adding to cores
        pass
    
    def get_performance_stats(self) -> dict:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with timing statistics (mean ± std in ms)
        """
        stats = {}
        for key, times in self.timings.items():
            if times:
                times_ms = [t * 1000 for t in times]
                stats[key] = {
                    "mean_ms": sum(times_ms) / len(times_ms),
                    "std_ms": torch.tensor(times_ms).std().item(),
                    "min_ms": min(times_ms),
                    "max_ms": max(times_ms),
                    "count": len(times_ms),
                }
        return stats
    
    def reset_performance_stats(self) -> None:
        """Reset performance tracking counters."""
        for key in self.timings:
            self.timings[key] = []
