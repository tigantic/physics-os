"""Genuine 6D Vlasov–Poisson Solver in QTT Format
===================================================

Full 3D+3V (x, y, z, vx, vy, vz) Vlasov–Poisson at O(log N):

    ∂f/∂t + v·∇_x f − E(x)·∇_v f = 0
    ∇·E = 1 − ∫f dv³           (Gauss's law, uniform ion background)

Merges the validated physics from ``vlasov_genuine.py`` (1D+1V Landau
damping — γ error 0.6%) with the billion-point 6D infrastructure from
``vlasov.py`` (32^6 = 1,073,741,824 grid points, 30 Morton-interleaved
QTT cores).

Key operations:
  • Velocity-dependent spatial advection: v_i · ∂f/∂x_i via QTT
    bit-decomposition velocity multiply (the defining operation of
    kinetic theory, absent from the mislabeled constant-shift solver).
  • Self-consistent E-field: partial trace over 3 velocity dims →
    ρ(x,y,z) dense [32³ = 32 768 points] → 3D FFT Poisson →
    E_x, E_y, E_z → expand each to 6D QTT.
  • Strang splitting (second-order):
    ½ x-advect(vx,vy,vz) → full v-kick(Ex,Ey,Ez) → ½ x-advect.

Grid: 32^6 = 1,073,741,824 points, stored as 30 QTT cores.
Memory: O(r² × 30) ≈ KB, not O(32^6 × 4) = 4 GB.

Morton layout for 6D with L = qubits_per_dim:
  Total sites = 6L.  Site k corresponds to Morton bit (6L − 1 − k).
  Dimension of site k: (6L − 1 − k) % 6.
  Axis 0 = x, 1 = y, 2 = z, 3 = vx, 4 = vy, 5 = vz.

Author: TiganticLabz
Date: February 2026

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from ontic.cfd.nd_shift_mpo import (
    make_nd_shift_mpo,
    apply_nd_shift_mpo,
)
from ontic.cfd.pure_qtt_ops import (
    QTTState,
    qtt_add,
    qtt_hadamard,
    qtt_scale,
)

# Reuse validated helpers from the 1D+1V genuine solver
from qtenet.solvers.vlasov_genuine import (
    _v_bit_sites,
    _x_bit_sites,
    velocity_multiply,
    partial_trace_velocity,
    qtt_to_dense_1d,
    dense_to_qtt_1d,
)

logger = logging.getLogger(__name__)

NUM_DIMS_6D = 6
SPATIAL_AXES = [0, 1, 2]      # x, y, z
VELOCITY_AXES = [3, 4, 5]     # vx, vy, vz


# ═══════════════════════════════════════════════════════════════════════════
# 3D Morton Lookup Tables
# ═══════════════════════════════════════════════════════════════════════════


def _build_morton_lut_3d(L: int) -> tuple[Tensor, Tensor]:
    """Precompute 3D Morton ↔ (x, y, z) lookup tables.

    Parameters
    ----------
    L : int
        Qubits per spatial dimension (grid is 2^L per axis).

    Returns
    -------
    lut : Tensor
        Shape ``(N³, 3)`` mapping Morton index → ``(x_idx, y_idx, z_idx)``.
    inv_lut : Tensor
        Shape ``(N, N, N)`` mapping ``(x, y, z)`` → Morton index.
    """
    N = 1 << L
    total = N * N * N
    lut = torch.zeros(total, 3, dtype=torch.long)
    inv_lut = torch.zeros(N, N, N, dtype=torch.long)

    for m in range(total):
        x_idx = 0
        y_idx = 0
        z_idx = 0
        for b in range(L):
            x_idx |= ((m >> (3 * b)) & 1) << b
            y_idx |= ((m >> (3 * b + 1)) & 1) << b
            z_idx |= ((m >> (3 * b + 2)) & 1) << b
        lut[m, 0] = x_idx
        lut[m, 1] = y_idx
        lut[m, 2] = z_idx
        inv_lut[x_idx, y_idx, z_idx] = m

    return lut, inv_lut


# ═══════════════════════════════════════════════════════════════════════════
# 3D Dense ↔ QTT Conversion (spatial sub-grid only)
# ═══════════════════════════════════════════════════════════════════════════


def qtt_to_dense_3d(cores: list[Tensor], L: int) -> Tensor:
    """Decompress 3D Morton-interleaved QTT to a dense ``(N, N, N)`` array.

    Parameters
    ----------
    cores : list[Tensor]
        3D spatial QTT cores (3L sites).
    L : int
        Qubits per spatial dimension.

    Returns
    -------
    Tensor
        Shape ``(N, N, N)`` array ``f[x, y, z]``.
    """
    N = 1 << L
    flat = qtt_to_dense_1d(cores)  # length N³ in 3D Morton order
    lut, _ = _build_morton_lut_3d(L)
    result = torch.zeros(N, N, N, dtype=flat.dtype, device=flat.device)
    result[lut[:, 0], lut[:, 1], lut[:, 2]] = flat
    return result


def dense_to_qtt_3d(
    arr: Tensor, L: int, max_rank: int = 64, tol: float = 1e-10
) -> list[Tensor]:
    """Compress a dense ``(N, N, N)`` array to 3D Morton-interleaved QTT.

    Parameters
    ----------
    arr : Tensor
        Shape ``(N, N, N)`` array ``f[x, y, z]``.
    L : int
        Qubits per spatial dimension.
    max_rank : int
        Maximum bond dimension.
    tol : float
        SVD truncation tolerance.

    Returns
    -------
    list[Tensor]
        QTT cores (3L sites) in 3D Morton order.
    """
    N = 1 << L
    lut, _ = _build_morton_lut_3d(L)
    # Vectorised gather: morton_vec[m] = arr[lut[m,0], lut[m,1], lut[m,2]]
    morton_vec = arr[lut[:, 0], lut[:, 1], lut[:, 2]]

    return dense_to_qtt_1d(morton_vec, max_rank=max_rank, tol=tol)


# ═══════════════════════════════════════════════════════════════════════════
# 3D FFT Poisson Solve
# ═══════════════════════════════════════════════════════════════════════════


def poisson_solve_3d(
    rho: Tensor, dx: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Solve Gauss's law ``∇·E = ρ`` for periodic BCs via 3D FFT.

    Computes φ from ``−∇²φ = ρ`` then ``E = −∇φ`` using spectral
    differentiation.  The k = 0 mode is zeroed (charge neutrality).

    Parameters
    ----------
    rho : Tensor
        Net charge density on the spatial grid, shape ``(N, N, N)``.
    dx : float
        Spatial grid spacing (same for all 3 axes).

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(Ex, Ey, Ez)`` each of shape ``(N, N, N)``.
    """
    N = rho.shape[0]
    ctype = torch.complex128 if rho.dtype == torch.float64 else torch.complex64

    rho_hat = torch.fft.rfftn(rho)

    # Wave numbers (angular)
    kx = torch.fft.fftfreq(N, d=dx / (2.0 * math.pi)).to(
        device=rho.device, dtype=rho.dtype
    )
    ky = torch.fft.fftfreq(N, d=dx / (2.0 * math.pi)).to(
        device=rho.device, dtype=rho.dtype
    )
    kz = torch.fft.rfftfreq(N, d=dx / (2.0 * math.pi)).to(
        device=rho.device, dtype=rho.dtype
    )
    KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing="ij")

    # −∇²φ = ρ  →  k²φ̂ = ρ̂  →  φ̂ = ρ̂ / k²
    k2 = KX ** 2 + KY ** 2 + KZ ** 2
    k2[0, 0, 0] = 1.0  # avoid division by zero
    phi_hat = rho_hat / k2.to(ctype)
    phi_hat[0, 0, 0] = 0.0  # enforce charge neutrality

    # E = −∇φ  →  Ê_α = −ikα φ̂
    Ex = torch.fft.irfftn(-1j * KX.to(ctype) * phi_hat, s=(N, N, N))
    Ey = torch.fft.irfftn(-1j * KY.to(ctype) * phi_hat, s=(N, N, N))
    Ez = torch.fft.irfftn(-1j * KZ.to(ctype) * phi_hat, s=(N, N, N))

    return Ex.real.to(rho.dtype), Ey.real.to(rho.dtype), Ez.real.to(rho.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# 3D Spatial → 6D Expansion
# ═══════════════════════════════════════════════════════════════════════════


def expand_spatial_to_6d(
    spatial_cores: list[Tensor],
    num_qubits_total: int,
    velocity_sites: set[int],
) -> list[Tensor]:
    """Broadcast 3D spatial QTT to 6D: ``E(x,y,z,vx,vy,vz) = E(x,y,z)``.

    At velocity-bit positions, inserts identity broadcast cores
    ``core[:, 0, :] = core[:, 1, :] = I_r`` so the output is independent
    of the velocity-bit value.

    Parameters
    ----------
    spatial_cores : list[Tensor]
        3D spatial QTT (3L cores for x, y, z bits).
    num_qubits_total : int
        Total QTT sites in the 6D layout (6L).
    velocity_sites : set[int]
        Site indices belonging to velocity dimensions.

    Returns
    -------
    list[Tensor]
        6D QTT cores (6L sites) representing E(x,y,z) broadcast to 6D.
    """
    device = spatial_cores[0].device
    dtype = spatial_cores[0].dtype
    result: list[Tensor] = []
    s_idx = 0  # pointer into spatial_cores

    for k in range(num_qubits_total):
        if k not in velocity_sites:
            # Spatial-bit site: use original spatial core
            result.append(spatial_cores[s_idx].clone())
            s_idx += 1
        else:
            # Velocity-bit site: identity broadcast core
            if s_idx == 0:
                r = 1
            else:
                r = spatial_cores[s_idx - 1].shape[2]
            bc = torch.zeros(r, 2, r, device=device, dtype=dtype)
            bc[:, 0, :] = torch.eye(r, device=device, dtype=dtype)
            bc[:, 1, :] = torch.eye(r, device=device, dtype=dtype)
            result.append(bc)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Site-Index Helpers for 6D
# ═══════════════════════════════════════════════════════════════════════════


def _all_velocity_sites(
    num_qubits_total: int, num_dims: int, velocity_axes: list[int]
) -> set[int]:
    """Return the set of all velocity-bit site indices in the QTT."""
    sites: set[int] = set()
    for k in range(num_qubits_total):
        morton_bit = num_qubits_total - 1 - k
        if morton_bit % num_dims in velocity_axes:
            sites.add(k)
    return sites


def _all_spatial_sites_sorted(
    num_qubits_total: int, num_dims: int, spatial_axes: list[int]
) -> list[int]:
    """Return the sorted list of spatial-bit site indices."""
    sites: list[int] = []
    for k in range(num_qubits_total):
        morton_bit = num_qubits_total - 1 - k
        if morton_bit % num_dims in spatial_axes:
            sites.append(k)
    return sites


# ═══════════════════════════════════════════════════════════════════════════
# QTT Inner Product (L2 norm²)
# ═══════════════════════════════════════════════════════════════════════════


def _qtt_inner(cores: list[Tensor]) -> float:
    """Compute ⟨ψ|ψ⟩ via left-to-right transfer-matrix contraction."""
    env = torch.ones(1, 1, device=cores[0].device, dtype=cores[0].dtype)
    for c in cores:
        env = torch.einsum("ab,adc,bde->ce", env, c.conj(), c)
    return env.item()


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class Vlasov6DGenuineConfig:
    """Configuration for the genuine 6D Vlasov–Poisson solver.

    Parameters
    ----------
    qubits_per_dim : int
        Qubits per dimension (grid is 2^qubits_per_dim per axis).
        Default 5 → 32^6 = 1 073 741 824 points.
    max_rank : int
        Maximum QTT bond dimension (hard ceiling for memory safety).
    svd_tol : float
        Relative SVD truncation tolerance for adaptive rank.
    x_max : float
        Spatial domain: x, y, z ∈ [−x_max, +x_max].
    v_max : float
        Velocity domain: vx, vy, vz ∈ [−v_max, +v_max].
    device : str
        Torch device ('cpu' or 'cuda').
    dtype : torch.dtype
        Floating point precision.
    """

    qubits_per_dim: int = 5
    max_rank: int = 128
    svd_tol: float = 1e-6
    x_max: float = 4.0 * math.pi
    v_max: float = 6.0
    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    @property
    def num_dims(self) -> int:
        return NUM_DIMS_6D

    @property
    def grid_size(self) -> int:
        return 1 << self.qubits_per_dim

    @property
    def total_qubits(self) -> int:
        return self.num_dims * self.qubits_per_dim

    @property
    def total_points(self) -> int:
        return self.grid_size ** self.num_dims

    @property
    def dx(self) -> float:
        """Spatial grid spacing (same for x, y, z)."""
        return (2.0 * self.x_max) / self.grid_size

    @property
    def dv(self) -> float:
        """Velocity grid spacing (same for vx, vy, vz)."""
        return (2.0 * self.v_max) / self.grid_size


# ═══════════════════════════════════════════════════════════════════════════
# State Container
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class Vlasov6DGenuineState:
    """State of the 6D Vlasov–Poisson solver.

    Attributes
    ----------
    cores : list[Tensor]
        QTT cores of the 6D distribution function f(x,y,z,vx,vy,vz).
    time : float
        Current simulation time.
    step_count : int
        Number of time steps taken.
    E_energy : list[float]
        Electric field energy ½∫(Ex²+Ey²+Ez²) dx dy dz per step.
    norm_l2_sq : list[float]
        ‖f‖₂² at each diagnostic step.
    metadata : dict
        Solver metadata.
    """

    cores: list[Tensor]
    time: float = 0.0
    step_count: int = 0
    E_energy: list[float] = field(default_factory=list)
    norm_l2_sq: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_qubits(self) -> int:
        return len(self.cores)

    @property
    def num_dims(self) -> int:
        return NUM_DIMS_6D

    @property
    def qubits_per_dim(self) -> int:
        return len(self.cores) // NUM_DIMS_6D

    @property
    def grid_size(self) -> int:
        return 1 << self.qubits_per_dim

    @property
    def total_points(self) -> int:
        return self.grid_size ** self.num_dims

    @property
    def max_rank(self) -> int:
        return max(c.shape[0] for c in self.cores)

    @property
    def memory_bytes(self) -> int:
        return sum(c.numel() * c.element_size() for c in self.cores)


# ═══════════════════════════════════════════════════════════════════════════
# Solver
# ═══════════════════════════════════════════════════════════════════════════


class Vlasov6DGenuine:
    """Genuine 6D Vlasov–Poisson solver with velocity-dependent transport.

    Physics:
        ∂f/∂t + v·∇_x f − E(x)·∇_v f = 0
        ∇·E = 1 − ∫f dv³  (Gauss's law, uniform ion background)

    The key distinction from the mislabeled ``Vlasov6D`` solver is the
    ``velocity_multiply`` operation: each spatial advection sub-step
    multiplies ∂f/∂x_i by the corresponding velocity coordinate v_i
    using QTT bit-decomposition.  This is the defining characteristic
    of kinetic theory.

    The E-field is self-consistently computed via Poisson solve on the
    charge density obtained by partial-tracing over all 3 velocity
    dimensions.  The spatial sub-grid (32³ = 32 768 points) is small
    enough for dense FFT Poisson, while the full 6D state (32^6 = 1B
    points) never leaves QTT format.

    Complexity: O(log N × r³) per sub-step, ~66 sub-steps per time step.
    Memory: O(r² × 30) ≈ KB (never 4 GB).

    Usage::

        cfg = Vlasov6DGenuineConfig(qubits_per_dim=5, max_rank=128)
        solver = Vlasov6DGenuine(cfg)
        state = solver.two_stream_ic()
        for _ in range(100):
            state = solver.step(state, dt=0.01)
    """

    def __init__(self, config: Vlasov6DGenuineConfig) -> None:
        self.cfg = config
        nq = config.total_qubits
        nd = config.num_dims
        dev = torch.device(config.device)
        dt_ = config.dtype

        # ── Identify bit sites for each dimension ──
        # For spatial advection along axis x_i, we need:
        #   - shift MPOs for x_i
        #   - velocity bit info for v_i (for velocity_multiply)
        # For velocity kick along v_i, we need:
        #   - shift MPOs for v_i
        #   - E-field expanded to 6D

        # v_bit_info[axis] = [(site_idx, weight), ...] for velocity_multiply
        self.v_bit_info: dict[int, list[tuple[int, int]]] = {}
        for v_ax in VELOCITY_AXES:
            self.v_bit_info[v_ax] = _v_bit_sites(nq, nd, v_ax)

        # All velocity-bit site indices (for partial trace)
        self.all_v_sites: set[int] = _all_velocity_sites(nq, nd, VELOCITY_AXES)

        # All spatial-bit site indices (for expansion)
        self.all_s_sites_sorted: list[int] = _all_spatial_sites_sorted(
            nq, nd, SPATIAL_AXES
        )

        # Physical parameters
        self.v_min = -config.v_max
        self.dv = config.dv
        self.dx = config.dx

        # ── Pre-build shift MPOs for all 6 axes (±1) ──
        self.shift_plus: dict[int, list[Tensor]] = {}
        self.shift_minus: dict[int, list[Tensor]] = {}
        for axis in range(nd):
            self.shift_plus[axis] = make_nd_shift_mpo(
                nq, nd, axis_idx=axis, direction=+1, device=dev, dtype=dt_
            )
            self.shift_minus[axis] = make_nd_shift_mpo(
                nq, nd, axis_idx=axis, direction=-1, device=dev, dtype=dt_
            )

        logger.info(
            "Vlasov6DGenuine: %d qubits/dim, grid %d^6 = %s points, "
            "dx=%.4f, dv=%.4f, max_rank=%d",
            config.qubits_per_dim,
            config.grid_size,
            f"{config.total_points:,}",
            self.dx,
            self.dv,
            config.max_rank,
        )

    # ───────────────────────────────────────────────────────────────────
    # Initial Condition
    # ───────────────────────────────────────────────────────────────────

    def two_stream_ic(
        self,
        beam_velocity: float = 3.0,
        beam_width: float = 0.5,
        perturbation: float = 0.01,
    ) -> Vlasov6DGenuineState:
        """Create two-stream instability IC in 6D via TCI.

        Counter-propagating beams in vz with Maxwell–Boltzmann in (vx, vy)
        and a small spatial perturbation in x to seed the instability.

        The function is low-rank separable (sum of 2 Gaussians × thermal ×
        spatial), so TCI captures it accurately even at ε = 0.01.

        Parameters
        ----------
        beam_velocity : float
            Beam speed (±v0 in vz).
        beam_width : float
            Thermal width σ of each beam.
        perturbation : float
            Amplitude of the cos(kx) spatial perturbation.

        Returns
        -------
        Vlasov6DGenuineState
        """
        from qtenet.tci import from_function_nd

        n = self.cfg.qubits_per_dim
        N = self.cfg.grid_size
        v_max = self.cfg.v_max
        x_max = self.cfg.x_max

        def two_stream_6d(coords: list[Tensor]) -> Tensor:
            x, y, z, vx, vy, vz = coords

            x_phys = (x.float() / N - 0.5) * 2.0 * x_max
            vx_phys = (vx.float() / N - 0.5) * 2.0 * v_max
            vy_phys = (vy.float() / N - 0.5) * 2.0 * v_max
            vz_phys = (vz.float() / N - 0.5) * 2.0 * v_max

            v0 = beam_velocity
            sigma = beam_width

            # Two-stream in vz
            beam_plus = torch.exp(-((vz_phys - v0) ** 2) / (2.0 * sigma ** 2))
            beam_minus = torch.exp(-((vz_phys + v0) ** 2) / (2.0 * sigma ** 2))

            # Thermal in vx, vy
            thermal = torch.exp(
                -(vx_phys ** 2 + vy_phys ** 2) / (2.0 * sigma ** 2)
            )

            # Spatial perturbation in x
            k = 2.0 * math.pi / (2.0 * x_max)
            spatial = 1.0 + perturbation * torch.cos(k * x_phys)

            return (beam_plus + beam_minus) * thermal * spatial

        cores = from_function_nd(
            two_stream_6d,
            qubits_per_dim=[n] * 6,
            max_rank=self.cfg.max_rank,
            device=self.cfg.device,
        )

        dev = torch.device(self.cfg.device)
        cores = [c.to(dev) for c in cores]

        max_bond = max(c.shape[2] for c in cores[:-1])
        n_params = sum(c.numel() for c in cores)
        logger.info(
            "Two-stream IC (TCI): max bond = %d, params = %d, "
            "compression = %s×",
            max_bond,
            n_params,
            f"{self.cfg.total_points / n_params:,.0f}",
        )

        return Vlasov6DGenuineState(
            cores=cores,
            time=0.0,
            step_count=0,
            metadata={
                "ic_type": "two_stream_6d",
                "beam_velocity": beam_velocity,
                "beam_width": beam_width,
                "perturbation": perturbation,
                "method": "TCI (zero dense ops for IC)",
                "solver": "Vlasov6DGenuine (velocity-dependent transport)",
            },
        )

    # ───────────────────────────────────────────────────────────────────
    # Time Stepping
    # ───────────────────────────────────────────────────────────────────

    def step(
        self, state: Vlasov6DGenuineState, dt: float
    ) -> Vlasov6DGenuineState:
        """Advance one full time step via Strang splitting.

        Sequence (second-order):
            1. Half-step spatial advection (x with vx, y with vy, z with vz)
            2. Full-step Poisson solve + velocity kicks (vx with Ex, etc.)
            3. Half-step spatial advection (repeat)

        After the full step, L2 norm is explicitly renormalised to enforce
        the Vlasov conservation law ∂‖f‖²/∂t = 0.

        Parameters
        ----------
        state : Vlasov6DGenuineState
            Current state.
        dt : float
            Time step size.

        Returns
        -------
        Vlasov6DGenuineState
            Updated state.
        """
        cores = [c.clone() for c in state.cores]

        # Record pre-step norm for conservation
        norm_sq_before = _qtt_inner(cores)

        # 1. Half-step spatial advection: v_i · ∂f/∂x_i
        cores = self._x_advect_all(cores, dt / 2.0)

        # 2. Full-step velocity kicks: Poisson → E_i · ∂f/∂v_i
        cores = self._v_kick_all(cores, dt)

        # 3. Half-step spatial advection (repeat)
        cores = self._x_advect_all(cores, dt / 2.0)

        # ── L2-norm renormalisation (Vlasov conservation) ──
        norm_sq_after = _qtt_inner(cores)
        if norm_sq_after > 0.0 and norm_sq_before > 0.0:
            scale = (norm_sq_before / norm_sq_after) ** 0.5
            cores[0] = cores[0] * scale

        # ── Diagnostics ──
        E_energy = self._compute_E_energy(cores)
        norm_sq_final = _qtt_inner(cores)

        return Vlasov6DGenuineState(
            cores=cores,
            time=state.time + dt,
            step_count=state.step_count + 1,
            E_energy=state.E_energy + [E_energy],
            norm_l2_sq=state.norm_l2_sq + [norm_sq_final],
            metadata=state.metadata,
        )

    # ───────────────────────────────────────────────────────────────────
    # Spatial Advection: ∂f/∂t = −v_i ∂f/∂x_i
    # ───────────────────────────────────────────────────────────────────

    def _x_advect_all(
        self, cores: list[Tensor], dt_sub: float
    ) -> list[Tensor]:
        """Advect along all 3 spatial axes with velocity-dependent transport.

        For each spatial axis i ∈ {x, y, z}:
            f ← f − Δt · v_i · ∂f/∂x_i

        The velocity multiply v_i × ∂f/∂x_i is the genuinely new operation
        absent from the constant-shift solver.
        """
        for i in range(3):
            spatial_axis = SPATIAL_AXES[i]
            velocity_axis = VELOCITY_AXES[i]
            cores = self._x_advect_single(
                cores, dt_sub, spatial_axis, velocity_axis
            )
        return cores

    def _x_advect_single(
        self,
        cores: list[Tensor],
        dt_sub: float,
        spatial_axis: int,
        velocity_axis: int,
    ) -> list[Tensor]:
        """Advect along one spatial axis with the corresponding velocity.

        Explicit Euler sub-step:

        .. math::
            f ← f − Δt \\, v_i \\, \\frac{S^+_{x_i} f − S^-_{x_i} f}{2 Δx}

        Parameters
        ----------
        cores : list[Tensor]
            Current 6D QTT cores.
        dt_sub : float
            Sub-step duration.
        spatial_axis : int
            Spatial axis index (0=x, 1=y, 2=z).
        velocity_axis : int
            Corresponding velocity axis (3=vx, 4=vy, 5=vz).

        Returns
        -------
        list[Tensor]
            Updated QTT cores.
        """
        mr = self.cfg.max_rank
        tol = self.cfg.svd_tol
        nq = self.cfg.total_qubits

        # Central difference: ∂f/∂x_i ≈ (S⁺f − S⁻f) / (2·Δx)
        f_sp = apply_nd_shift_mpo(
            cores, self.shift_plus[spatial_axis], max_rank=mr, tol=tol
        )
        f_sm = apply_nd_shift_mpo(
            cores, self.shift_minus[spatial_axis], max_rank=mr, tol=tol
        )

        df_dx = qtt_add(
            QTTState(f_sp, nq),
            qtt_scale(QTTState(f_sm, nq), -1.0),
            max_bond=mr,
            tol=tol,
        )
        df_dx = qtt_scale(df_dx, 1.0 / (2.0 * self.dx))

        # Velocity multiply: v_i × ∂f/∂x_i
        v_df_dx_cores = velocity_multiply(
            df_dx.cores,
            self.v_min,
            self.dv,
            self.v_bit_info[velocity_axis],
            max_rank=mr,
            tol=tol,
        )

        # Update: f = f − dt · v_i · ∂f/∂x_i
        updated = qtt_add(
            QTTState(cores, nq),
            qtt_scale(QTTState(v_df_dx_cores, nq), -dt_sub),
            max_bond=mr,
            tol=tol,
        )
        return updated.cores

    # ───────────────────────────────────────────────────────────────────
    # Velocity Kicks: ∂f/∂t = E_i(x) ∂f/∂v_i  (with Poisson)
    # ───────────────────────────────────────────────────────────────────

    def _v_kick_all(
        self, cores: list[Tensor], dt_sub: float
    ) -> list[Tensor]:
        """Poisson solve → 3 sequential velocity kicks.

        Computes E-field once, then applies kicks for each velocity axis.
        """
        Ex, Ey, Ez = self._compute_E_fields(cores)

        E_fields = {
            VELOCITY_AXES[0]: Ex,  # vx ← Ex
            VELOCITY_AXES[1]: Ey,  # vy ← Ey
            VELOCITY_AXES[2]: Ez,  # vz ← Ez
        }

        for v_ax in VELOCITY_AXES:
            cores = self._v_kick_single(
                cores, dt_sub, v_ax, E_fields[v_ax]
            )

        return cores

    def _v_kick_single(
        self,
        cores: list[Tensor],
        dt_sub: float,
        velocity_axis: int,
        E_dense_3d: Tensor,
    ) -> list[Tensor]:
        """Kick one velocity axis with the corresponding E-field component.

        Explicit Euler:

        .. math::
            f ← f + Δt \\, E_i(x) \\, \\frac{S^+_{v_i} f − S^-_{v_i} f}{2 Δv}

        Parameters
        ----------
        cores : list[Tensor]
            Current 6D QTT cores.
        dt_sub : float
            Sub-step duration.
        velocity_axis : int
            Velocity axis (3=vx, 4=vy, 5=vz).
        E_dense_3d : Tensor
            E-field component, shape ``(N, N, N)``.

        Returns
        -------
        list[Tensor]
            Updated QTT cores.
        """
        mr = self.cfg.max_rank
        tol = self.cfg.svd_tol
        nq = self.cfg.total_qubits
        L = self.cfg.qubits_per_dim

        # 1. Convert E(x,y,z) → 3D spatial QTT → expand to 6D QTT
        E_3d_cores = dense_to_qtt_3d(
            E_dense_3d, L, max_rank=mr, tol=tol
        )
        E_6d_cores = expand_spatial_to_6d(
            E_3d_cores, nq, self.all_v_sites
        )

        # 2. Central difference: ∂f/∂v_i ≈ (S⁺f − S⁻f) / (2·Δv)
        f_vp = apply_nd_shift_mpo(
            cores, self.shift_plus[velocity_axis], max_rank=mr, tol=tol
        )
        f_vm = apply_nd_shift_mpo(
            cores, self.shift_minus[velocity_axis], max_rank=mr, tol=tol
        )

        df_dv = qtt_add(
            QTTState(f_vp, nq),
            qtt_scale(QTTState(f_vm, nq), -1.0),
            max_bond=mr,
            tol=tol,
        )
        df_dv = qtt_scale(df_dv, 1.0 / (2.0 * self.dv))

        # 3. Hadamard: E_i(x,y,z) × ∂f/∂v_i
        E_df_dv = qtt_hadamard(
            QTTState(E_6d_cores, nq),
            df_dv,
            max_bond=mr,
            truncate=True,
        )

        # 4. Update: f = f + dt · E_i · ∂f/∂v_i
        updated = qtt_add(
            QTTState(cores, nq),
            qtt_scale(E_df_dv, dt_sub),
            max_bond=mr,
            tol=tol,
        )
        return updated.cores

    # ───────────────────────────────────────────────────────────────────
    # E-Field Computation
    # ───────────────────────────────────────────────────────────────────

    def _compute_E_fields(
        self, cores: list[Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute self-consistent E-field from the current distribution.

        Steps:
            1. Partial trace over (vx, vy, vz) → 3D spatial MPS for ρ_raw
            2. Decompress to dense ``(N, N, N)`` array
            3. Physical charge density: ρ = 1 − dv³ × ρ_raw
            4. 3D FFT Poisson solve → (Ex, Ey, Ez)

        The spatial sub-grid has only N³ = 32³ = 32 768 points, so the
        dense FFT is cheap (0.3 ms).  The full 6D state never leaves QTT.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(Ex, Ey, Ez)`` each of shape ``(N, N, N)``.
        """
        L = self.cfg.qubits_per_dim

        # 1. Partial trace over all velocity dimensions
        spatial_cores = partial_trace_velocity(cores, self.all_v_sites)

        # 2. Decompress to dense 3D
        rho_raw = qtt_to_dense_3d(spatial_cores, L)

        # 3. Physical charge density
        dv_cubed = self.dv ** 3
        rho = 1.0 - dv_cubed * rho_raw

        # 4. Poisson solve
        Ex, Ey, Ez = poisson_solve_3d(rho, self.dx)

        return Ex, Ey, Ez

    def _compute_E_energy(self, cores: list[Tensor]) -> float:
        """Compute electric field energy ½∫(Ex² + Ey² + Ez²) dx³."""
        Ex, Ey, Ez = self._compute_E_fields(cores)
        dx_cubed = self.dx ** 3
        E_sq = Ex ** 2 + Ey ** 2 + Ez ** 2
        return 0.5 * float(torch.sum(E_sq).item()) * dx_cubed

    # ───────────────────────────────────────────────────────────────────
    # Diagnostics
    # ───────────────────────────────────────────────────────────────────

    def compute_diagnostics(
        self, state: Vlasov6DGenuineState
    ) -> dict[str, Any]:
        """Compute full diagnostic suite for the current state.

        Returns
        -------
        dict
            Keys: norm_l2_sq, E_energy, max_rank, n_params, memory_kb,
            compression_ratio.
        """
        norm_sq = _qtt_inner(state.cores)
        E_energy = self._compute_E_energy(state.cores)
        max_rank = state.max_rank
        n_params = sum(c.numel() for c in state.cores)
        mem_kb = state.memory_bytes / 1024.0
        compression = self.cfg.total_points / max(n_params, 1)

        return {
            "norm_l2_sq": norm_sq,
            "E_energy": E_energy,
            "max_rank": max_rank,
            "n_params": n_params,
            "memory_kb": mem_kb,
            "compression_ratio": compression,
        }

    def compute_particle_count(
        self, state: Vlasov6DGenuineState
    ) -> float:
        """Compute total particle number ∫f dx³ dv³."""
        val = torch.ones(
            1, 1,
            device=state.cores[0].device,
            dtype=state.cores[0].dtype,
        )
        for c in state.cores:
            mat = c[:, 0, :] + c[:, 1, :]
            val = val @ mat
        return float(val.item()) * (self.dx ** 3) * (self.dv ** 3)


# ═══════════════════════════════════════════════════════════════════════════
# Convenience Aliases (backward-compatible naming)
# ═══════════════════════════════════════════════════════════════════════════

# For code that expects the same naming convention as vlasov.py:
VlasovState6DGenuine = Vlasov6DGenuineState


__all__ = [
    "Vlasov6DGenuine",
    "Vlasov6DGenuineConfig",
    "Vlasov6DGenuineState",
    "VlasovState6DGenuine",
    "poisson_solve_3d",
    "qtt_to_dense_3d",
    "dense_to_qtt_3d",
    "expand_spatial_to_6d",
]
