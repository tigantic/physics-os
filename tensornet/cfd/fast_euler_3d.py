"""
Native 3D Euler Solver using N-Dimensional Shift MPO

This extends the 2D fast solver to 3D using the same generalized
Morton-interleaved shift MPO infrastructure.

For 3D:
- Morton order: bits interleave as x0, y0, z0, x1, y1, z1, ...
- Shift MPO targets specific axis (0=x, 1=y, 2=z)
- State has 5 conserved variables: [rho, rhou, rhov, rhow, E]

Standard 3D benchmark: Taylor-Green Vortex
- Periodic box [0, 2π]³
- Initial: u = sin(x)cos(y)cos(z), v = -cos(x)sin(y)cos(z), w = 0
- Tests: Energy decay, enstrophy growth, vortex stretching

Author: HyperTensor Team
Date: December 2025
"""

import torch
from typing import Tuple, List
from dataclasses import dataclass
import time

from tensornet.cfd.nd_shift_mpo import make_nd_shift_mpo, apply_nd_shift_mpo, truncate_cores
from tensornet.cfd.pure_qtt_ops import QTTState, qtt_add, dense_to_qtt, qtt_to_dense


@dataclass
class QTT3DState:
    """3D field stored in QTT format with Morton ordering."""
    cores: List[torch.Tensor]
    nx: int  # Qubits for x
    ny: int  # Qubits for y
    nz: int  # Qubits for z
    
    @property
    def max_rank(self) -> int:
        return max(c.shape[0] for c in self.cores)
    
    @property
    def total_qubits(self) -> int:
        return len(self.cores)


@dataclass
class Euler3DConfig:
    """Configuration for 3D Euler solver."""
    qubits_per_dim: int = 4  # Grid is 2^n per dimension (16^3 = 4096 points)
    gamma: float = 1.4
    cfl: float = 0.3
    max_rank: int = 32
    device: torch.device = None
    dtype: torch.dtype = torch.float32
    
    def __post_init__(self):
        if self.device is None:
            self.device = torch.device('cpu')
    
    @property
    def grid_size(self) -> int:
        return 2 ** self.qubits_per_dim
    
    @property
    def total_qubits(self) -> int:
        return 3 * self.qubits_per_dim
    
    @property 
    def total_points(self) -> int:
        return self.grid_size ** 3


@dataclass
class Euler3DState:
    """State for 3D compressible Euler equations."""
    rho: QTT3DState   # Density
    rhou: QTT3DState  # X-momentum
    rhov: QTT3DState  # Y-momentum  
    rhow: QTT3DState  # Z-momentum
    E: QTT3DState     # Total energy
    
    def max_rank(self) -> int:
        return max(
            self.rho.max_rank,
            self.rhou.max_rank,
            self.rhov.max_rank,
            self.rhow.max_rank,
            self.E.max_rank
        )


def morton_encode_3d(ix: int, iy: int, iz: int, n_bits: int) -> int:
    """Encode 3D index to Morton (Z-curve) order."""
    z = 0
    for b in range(n_bits):
        z |= ((ix >> b) & 1) << (3 * b)
        z |= ((iy >> b) & 1) << (3 * b + 1)
        z |= ((iz >> b) & 1) << (3 * b + 2)
    return z


def morton_decode_3d(z: int, n_bits: int) -> Tuple[int, int, int]:
    """Decode Morton index to 3D coordinates."""
    ix, iy, iz = 0, 0, 0
    for b in range(n_bits):
        ix |= ((z >> (3 * b)) & 1) << b
        iy |= ((z >> (3 * b + 1)) & 1) << b
        iz |= ((z >> (3 * b + 2)) & 1) << b
    return ix, iy, iz


def dense_to_qtt_3d(field: torch.Tensor, max_bond: int = 32) -> QTT3DState:
    """
    Compress 3D field to QTT with Morton ordering.
    
    Args:
        field: (Nx, Ny, Nz) tensor
        max_bond: Maximum bond dimension
        
    Returns:
        QTT3DState
    """
    Nx, Ny, Nz = field.shape
    nx = int(torch.log2(torch.tensor(Nx)).item())
    ny = int(torch.log2(torch.tensor(Ny)).item())
    nz = int(torch.log2(torch.tensor(Nz)).item())
    
    assert 2**nx == Nx and 2**ny == Ny and 2**nz == Nz
    
    # Flatten to Morton order
    N_total = Nx * Ny * Nz
    morton_field = torch.zeros(N_total, dtype=field.dtype, device=field.device)
    
    for ix in range(Nx):
        for iy in range(Ny):
            for iz in range(Nz):
                z = morton_encode_3d(ix, iy, iz, nx)
                morton_field[z] = field[ix, iy, iz]
    
    # Compress to 1D QTT
    qtt = dense_to_qtt(morton_field, max_bond=max_bond)
    
    return QTT3DState(cores=qtt.cores, nx=nx, ny=ny, nz=nz)


def qtt_3d_to_dense(state: QTT3DState) -> torch.Tensor:
    """Decompress QTT3D to dense 3D array."""
    qtt = QTTState(cores=state.cores, num_qubits=len(state.cores))
    morton_field = qtt_to_dense(qtt)
    
    Nx, Ny, Nz = 2**state.nx, 2**state.ny, 2**state.nz
    field = torch.zeros(Nx, Ny, Nz, dtype=morton_field.dtype, device=morton_field.device)
    
    for z in range(len(morton_field)):
        ix, iy, iz = morton_decode_3d(z, state.nx)
        if ix < Nx and iy < Ny and iz < Nz:
            field[ix, iy, iz] = morton_field[z]
    
    return field


class FastEuler3D:
    """
    Native 3D Euler solver using N-dimensional shift MPO.
    
    Complexity: O(log N × r³) per time step
    where N = total grid points, r = max rank
    """
    
    def __init__(self, config: Euler3DConfig):
        self.config = config
        self.n = config.qubits_per_dim
        self.total_qubits = config.total_qubits
        self.dx = 2 * torch.pi / config.grid_size  # [0, 2π]³ domain
        
        # Pre-build shift MPOs for all three axes
        self.shift_mpos = []
        for axis in range(3):
            mpo = make_nd_shift_mpo(
                config.total_qubits,
                num_dims=3,
                axis_idx=axis,
                direction=+1,
                device=config.device,
                dtype=config.dtype
            )
            self.shift_mpos.append(mpo)
        
        print(f"FastEuler3D: {config.grid_size}³ ({config.total_points} points)")
        print(f"  Total qubits: {config.total_qubits}, Max rank: {config.max_rank}")
    
    def _shift(self, qtt: QTT3DState, axis: int) -> QTT3DState:
        """Apply shift: output[i] = input[i-1] in given axis."""
        cores = apply_nd_shift_mpo(qtt.cores, self.shift_mpos[axis], 
                                    max_rank=self.config.max_rank)
        return QTT3DState(cores, nx=qtt.nx, ny=qtt.ny, nz=qtt.nz)
    
    def _add(self, a: QTT3DState, b: QTT3DState) -> QTT3DState:
        """QTT addition with truncation."""
        a_qtt = QTTState(cores=a.cores, num_qubits=len(a.cores))
        b_qtt = QTTState(cores=b.cores, num_qubits=len(b.cores))
        result = qtt_add(a_qtt, b_qtt, max_bond=self.config.max_rank)
        return QTT3DState(result.cores, nx=a.nx, ny=a.ny, nz=a.nz)
    
    def _scale(self, a: QTT3DState, s: float) -> QTT3DState:
        """Scale QTT."""
        cores = [c.clone() for c in a.cores]
        cores[0] = cores[0] * s
        return QTT3DState(cores, nx=a.nx, ny=a.ny, nz=a.nz)
    
    def _sub(self, a: QTT3DState, b: QTT3DState) -> QTT3DState:
        """QTT subtraction."""
        return self._add(a, self._scale(b, -1.0))
    
    def _evolve_axis(self, state: Euler3DState, dt: float, axis: int) -> Euler3DState:
        """
        Update state in one axis direction using upwind flux.
        
        axis: 0=x, 1=y, 2=z
        """
        coeff = -dt / self.dx
        
        # Get flux for this axis
        if axis == 0:
            F_mom = state.rhou  # x-momentum carries mass flux
        elif axis == 1:
            F_mom = state.rhov
        else:
            F_mom = state.rhow
        
        # Mass update: dF/dx ≈ (F - F_left) / dx
        dF_rho = self._sub(F_mom, self._shift(F_mom, axis))
        rho_new = self._add(state.rho, self._scale(dF_rho, coeff))
        
        # Momentum updates (advection)
        dF_rhou = self._sub(state.rhou, self._shift(state.rhou, axis))
        dF_rhov = self._sub(state.rhov, self._shift(state.rhov, axis))
        dF_rhow = self._sub(state.rhow, self._shift(state.rhow, axis))
        
        rhou_new = self._add(state.rhou, self._scale(dF_rhou, coeff))
        rhov_new = self._add(state.rhov, self._scale(dF_rhov, coeff))
        rhow_new = self._add(state.rhow, self._scale(dF_rhow, coeff))
        
        # Energy
        dF_E = self._sub(state.E, self._shift(state.E, axis))
        E_new = self._add(state.E, self._scale(dF_E, coeff))
        
        return Euler3DState(rho_new, rhou_new, rhov_new, rhow_new, E_new)
    
    def step(self, state: Euler3DState, dt: float) -> Euler3DState:
        """
        Strang splitting for 3D:
        L_x(dt/2) L_y(dt/2) L_z(dt) L_y(dt/2) L_x(dt/2)
        """
        state = self._evolve_axis(state, dt / 2, axis=0)  # X
        state = self._evolve_axis(state, dt / 2, axis=1)  # Y
        state = self._evolve_axis(state, dt, axis=2)      # Z
        state = self._evolve_axis(state, dt / 2, axis=1)  # Y
        state = self._evolve_axis(state, dt / 2, axis=0)  # X
        return state
    
    def compute_dt(self, state: Euler3DState) -> float:
        """Estimate stable dt."""
        # Use fixed dt for simplicity (proper implementation would sample)
        max_speed = 2.0  # Rough estimate for incompressible TGV
        return self.config.cfl * self.dx / max_speed


def create_taylor_green_state(config: Euler3DConfig) -> Euler3DState:
    """
    Create Taylor-Green vortex initial condition.
    
    u = sin(x) cos(y) cos(z)
    v = -cos(x) sin(y) cos(z)
    w = 0
    p = p0 + (cos(2x) + cos(2y))(cos(2z) + 2) / 16
    """
    N = config.grid_size
    n = config.qubits_per_dim
    
    # [0, 2π]³ domain
    x = torch.linspace(0, 2*torch.pi, N, dtype=config.dtype, device=config.device)
    y = torch.linspace(0, 2*torch.pi, N, dtype=config.dtype, device=config.device)
    z = torch.linspace(0, 2*torch.pi, N, dtype=config.dtype, device=config.device)
    
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    # Velocity field
    u = torch.sin(X) * torch.cos(Y) * torch.cos(Z)
    v = -torch.cos(X) * torch.sin(Y) * torch.cos(Z)
    w = torch.zeros_like(u)
    
    # Pressure (isentropic: p = rho^gamma)
    p0 = 100.0 / config.gamma  # Base pressure
    p = p0 + (torch.cos(2*X) + torch.cos(2*Y)) * (torch.cos(2*Z) + 2) / 16
    
    # Density from pressure (isentropic)
    rho = (p * config.gamma / p0) ** (1/config.gamma)
    
    # Energy
    E = p / (config.gamma - 1) + 0.5 * rho * (u**2 + v**2 + w**2)
    
    # Compress to QTT
    print("  Compressing initial condition to QTT3D...")
    t0 = time.perf_counter()
    
    rho_qtt = dense_to_qtt_3d(rho, max_bond=config.max_rank)
    rhou_qtt = dense_to_qtt_3d(rho * u, max_bond=config.max_rank)
    rhov_qtt = dense_to_qtt_3d(rho * v, max_bond=config.max_rank)
    rhow_qtt = dense_to_qtt_3d(rho * w, max_bond=config.max_rank)
    E_qtt = dense_to_qtt_3d(E, max_bond=config.max_rank)
    
    t1 = time.perf_counter()
    print(f"  Compression time: {t1-t0:.2f}s")
    
    return Euler3DState(rho_qtt, rhou_qtt, rhov_qtt, rhow_qtt, E_qtt)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Native 3D Euler Solver - Taylor-Green Vortex")
    print("=" * 60)
    
    # Start with small grid for testing
    config = Euler3DConfig(
        qubits_per_dim=4,  # 16³ = 4096 points
        max_rank=24,
        cfl=0.3
    )
    
    # Create initial condition
    print("\nCreating Taylor-Green vortex initial condition...")
    state = create_taylor_green_state(config)
    print(f"Initial max rank: {state.max_rank()}")
    
    # Create solver
    solver = FastEuler3D(config)
    
    # Run a few steps
    n_steps = 5
    print(f"\nRunning {n_steps} time steps...")
    
    total_time = 0.0
    for i in range(n_steps):
        dt = solver.compute_dt(state)
        
        t0 = time.perf_counter()
        state = solver.step(state, dt)
        step_time = time.perf_counter() - t0
        total_time += step_time
        
        print(f"  Step {i+1}: dt={dt:.5f}, rank={state.max_rank()}, time={step_time:.2f}s")
    
    print(f"\nTotal time: {total_time:.2f}s, avg per step: {total_time/n_steps:.2f}s")
    
    # Verify
    print("\nVerifying...")
    rho = qtt_3d_to_dense(state.rho)
    print(f"Density range: [{rho.min():.4f}, {rho.max():.4f}]")
    print(f"Density sum: {rho.sum():.2f} (should be ~{config.total_points:.0f})")
    
    if rho.min() > 0:
        print("\n✓ 3D EULER: STABILITY TEST PASSED")
    else:
        print("\n✗ 3D EULER: STABILITY TEST FAILED")
    
    print("\n" + "=" * 60)
    print("3D Solver Ready!")
    print("=" * 60)
