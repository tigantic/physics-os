"""
N-Dimensional Euler Solver — Compressible Flow at O(log N)

Solves the compressible Euler equations in N dimensions:
    ∂ρ/∂t + ∇·(ρu) = 0
    ∂(ρu)/∂t + ∇·(ρu⊗u + pI) = 0
    ∂E/∂t + ∇·((E+p)u) = 0

where ρ is density, u is velocity, p is pressure, E is total energy.

Example:
    >>> from qtenet.solvers import EulerND, EulerNDConfig
    >>> 
    >>> # 3D Euler on 64^3 grid
    >>> config = EulerNDConfig(qubits_per_dim=6, num_dims=3, max_rank=64)
    >>> solver = EulerND(config)
    >>> state = solver.taylor_green_ic()
    >>> 
    >>> for _ in range(1000):
    ...     state = solver.step(state, dt=0.001)

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from tensornet.cfd.euler_nd_native import (
    EulerND_Native as _EulerND,
    EulerNDConfig as _EulerNDConfig,
    EulerNDState as _EulerNDState,
)


@dataclass
class EulerState:
    """State of compressible Euler equations in QTT format.
    
    Conserved variables: (ρ, ρu, ρv, ρw, E)
    
    Attributes:
        density_cores: QTT cores for density
        momentum_cores: List of QTT cores for each momentum component
        energy_cores: QTT cores for total energy
        num_dims: Number of spatial dimensions
        qubits_per_dim: Qubits per dimension
        time: Current simulation time
        metadata: Additional information
    """
    density_cores: list[Tensor]
    momentum_cores: list[list[Tensor]]
    energy_cores: list[Tensor]
    num_dims: int
    qubits_per_dim: int
    time: float = 0.0
    gamma: float = 1.4
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def total_qubits(self) -> int:
        return self.num_dims * self.qubits_per_dim
    
    @property
    def grid_size(self) -> int:
        return 2 ** self.qubits_per_dim
    
    @property
    def total_points(self) -> int:
        return self.grid_size ** self.num_dims


@dataclass
class EulerNDConfig:
    """Configuration for N-dimensional Euler solver.
    
    Attributes:
        num_dims: Number of spatial dimensions (2 or 3)
        qubits_per_dim: Qubits per dimension
        max_rank: Maximum QTT rank
        gamma: Ratio of specific heats (1.4 for air)
        cfl: CFL number for stability
        device: Torch device
        dtype: Tensor dtype
    """
    num_dims: int = 3
    qubits_per_dim: int = 6
    max_rank: int = 64
    gamma: float = 1.4
    cfl: float = 0.3
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    
    @property
    def grid_size(self) -> int:
        return 2 ** self.qubits_per_dim
    
    @property
    def total_qubits(self) -> int:
        return self.num_dims * self.qubits_per_dim
    
    @property
    def total_points(self) -> int:
        return self.grid_size ** self.num_dims


class EulerND:
    """
    N-dimensional compressible Euler solver in native QTT format.
    
    Uses Strang splitting with Rusanov flux.
    
    Complexity: O(log N × r³) per time step
    where N = grid_size^num_dims, r = max_rank
    
    Example:
        >>> config = EulerNDConfig(num_dims=3, qubits_per_dim=6, max_rank=64)
        >>> solver = EulerND(config)
        >>> state = solver.kelvin_helmholtz_ic()
        >>> 
        >>> for _ in range(1000):
        ...     state = solver.step(state, dt=0.001)
    """
    
    def __init__(self, config: EulerNDConfig):
        self.config = config
        self._solver = _EulerND(_EulerNDConfig(
            num_dims=config.num_dims,
            qubits_per_dim=config.qubits_per_dim,
            max_rank=config.max_rank,
            gamma=config.gamma,
            cfl=config.cfl,
            device=torch.device(config.device),
            dtype=config.dtype,
        ))
    
    def step(self, state: EulerState, dt: float) -> EulerState:
        """Advance one time step."""
        internal = _EulerNDState(
            density_cores=state.density_cores,
            momentum_cores=state.momentum_cores,
            energy_cores=state.energy_cores,
            time=state.time,
        )
        
        new_internal = self._solver.step(internal, dt)
        
        return EulerState(
            density_cores=new_internal.density_cores,
            momentum_cores=new_internal.momentum_cores,
            energy_cores=new_internal.energy_cores,
            num_dims=state.num_dims,
            qubits_per_dim=state.qubits_per_dim,
            time=state.time + dt,
            gamma=state.gamma,
            metadata=state.metadata,
        )


__all__ = ["EulerND", "EulerNDConfig", "EulerState"]
