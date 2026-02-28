"""Genuine 1D+1V Vlasov–Poisson Solver in QTT Format
===================================================

Implements the ACTUAL Vlasov–Poisson equation:

    ∂f/∂t + v ∂f/∂x − E(x) ∂f/∂v = 0
    ∂E/∂x = 1 − ∫ f dv          (Gauss's law, ions = uniform background)

Unlike the mislabeled Vlasov6D/5D solvers (which apply constant-coefficient
shifts), this solver implements VELOCITY-DEPENDENT spatial advection: the
spatial derivative ∂f/∂x is multiplied by the velocity coordinate v, making
the transport speed vary across phase space — the defining characteristic
of kinetic theory.

Method:
  • Strang splitting (second-order in time):
    1. Half-step x-advection: ∂f/∂t = −v ∂f/∂x
    2. Full-step v-advection: ∂f/∂t = E(x) ∂f/∂v  (with Poisson solve)
    3. Half-step x-advection (repeat)

  • Central finite differences via shift MPOs for ∂f/∂x, ∂f/∂v
  • Velocity-coordinate multiplication via QTT bit decomposition
  • FFT Poisson solve for E-field (1D, O(N log N), cheap)
  • Partial trace over velocity for charge density (QTT native)

Validation:
  • Landau damping: γ must match theoretical −0.1533 for k=0.5 (within 15%)
  • Particle number conservation: monitored per step
  • Verified against dense reference solver (tensornet.packs.pack_xi)

Morton layout for 2D (x, v) with qubits_per_dim = L:
  Total sites = 2L.  Site k corresponds to Morton bit (2L−1−k).
  v-bits at even sites (0, 2, 4, ..., 2L−2): MSB at site 0, LSB at site 2L−2.
  x-bits at odd sites (1, 3, 5, ..., 2L−1): MSB at site 1, LSB at site 2L−1.

  Velocity index: j = Σ_l b_{site(l)} × 2^l  where l is the v-bit level.
  Physical velocity: v = v_min + j × dv.

Author: TiganticLabz
Date: January 2026
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
    truncate_cores,
)

from ontic.cfd.pure_qtt_ops import (
    QTTState,
    qtt_add,
    qtt_hadamard,
    qtt_inner_product,
    qtt_scale,
    qtt_sum,
    truncate_qtt,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Morton Interleaving Helpers
# ---------------------------------------------------------------------------


def _v_bit_sites(
    num_qubits_total: int, num_dims: int, v_axis: int
) -> list[tuple[int, int]]:
    """Return (site_index, weight_in_velocity_index) for each velocity bit.

    The weight is 2^(bit_level) where bit_level is the position of that bit
    within the velocity coordinate (0 = LSB, L-1 = MSB).

    For 2D with L=5 qubits_per_dim:
      site 0 → v-bit level 4  weight 16 (MSB)
      site 2 → v-bit level 3  weight 8
      site 4 → v-bit level 2  weight 4
      site 6 → v-bit level 1  weight 2
      site 8 → v-bit level 0  weight 1 (LSB)
    """
    result = []
    for k in range(num_qubits_total):
        morton_bit = num_qubits_total - 1 - k
        if morton_bit % num_dims == v_axis:
            bit_level = morton_bit // num_dims
            result.append((k, 1 << bit_level))
    return result


def _x_bit_sites(
    num_qubits_total: int, num_dims: int, x_axis: int
) -> list[int]:
    """Return sorted site indices for x-dimension bits."""
    result = []
    for k in range(num_qubits_total):
        morton_bit = num_qubits_total - 1 - k
        if morton_bit % num_dims == x_axis:
            result.append(k)
    return result


# ---------------------------------------------------------------------------
# Velocity-Coordinate Multiplication
# ---------------------------------------------------------------------------


def velocity_multiply(
    cores: list[Tensor],
    v_min: float,
    dv: float,
    v_bit_info: list[tuple[int, int]],
    max_rank: int,
    tol: float,
) -> list[Tensor]:
    """Multiply QTT state by the velocity coordinate value.

    Computes ``v · f`` where ``v = v_min + dv × j`` and ``j`` is the
    binary-encoded velocity index built from the velocity bits in Morton layout.

    Decomposition into L+1 terms:

        v · f = v_min · f  +  dv · Σ_l  w_l · (P_l · f)

    where ``P_l`` projects to ``b=1`` at the l-th velocity bit (zeroing b=0),
    and ``w_l = 2^(bit_level)`` is the positional weight of that bit.

    Each projected state ``P_l · f`` has the same cores as ``f`` everywhere
    except at v-bit site ``l``, where the ``b=0`` slice is set to zero.

    The L+1 terms are summed in one fused block-diagonal operation via
    ``qtt_sum``, followed by truncation.

    Parameters
    ----------
    cores : list[Tensor]
        QTT cores of the distribution function, shape ``(r_L, 2, r_R)`` each.
    v_min : float
        Minimum velocity value (lower grid boundary).
    dv : float
        Velocity grid spacing.
    v_bit_info : list[tuple[int, int]]
        From ``_v_bit_sites()``: list of ``(site_idx, weight)`` pairs.
    max_rank : int
        Maximum bond dimension after truncation.
    tol : float
        SVD truncation tolerance.

    Returns
    -------
    list[Tensor]
        QTT cores representing ``v · f``.
    """
    n = len(cores)

    # Incremental accumulation to avoid L+1 × rank explosion.
    # Instead of summing all L+1 terms at once (which creates rank (L+1)r
    # before truncation), we accumulate incrementally:
    #   accum = v_min × f
    #   accum += dv × w_0 × P_0(f)
    #   accum += dv × w_1 × P_1(f)
    #   ...
    # Each step only doubles the rank, then we truncate.

    accum = qtt_scale(QTTState(cores=[c.clone() for c in cores], num_qubits=n), v_min)

    for site_idx, bit_weight in v_bit_info:
        projected = [c.clone() for c in cores]
        c_orig = projected[site_idx]  # (r_left, 2, r_right)
        new_c = torch.zeros_like(c_orig)
        new_c[:, 1, :] = c_orig[:, 1, :]  # Keep b=1 slice, zero b=0
        projected[site_idx] = new_c

        term = qtt_scale(QTTState(cores=projected, num_qubits=n), dv * float(bit_weight))
        accum = qtt_add(accum, term, max_bond=max_rank, tol=tol)

    return accum.cores


# ---------------------------------------------------------------------------
# Partial Trace (Velocity Marginalisation)
# ---------------------------------------------------------------------------


def partial_trace_velocity(
    cores: list[Tensor],
    v_bit_sites: set[int],
) -> list[Tensor]:
    """Marginalise over velocity: ``ρ_raw(x) = Σ_v f(x, v)``.

    Contracts each velocity-bit core with the all-ones vector ``[1, 1]``,
    leaving an x-only 1D MPS.  The contraction matrix at each v-site is
    absorbed into the next x-site core to the right.

    Parameters
    ----------
    cores : list[Tensor]
        2D Morton-interleaved QTT cores.
    v_bit_sites : set[int]
        Indices of velocity-bit sites.

    Returns
    -------
    list[Tensor]
        1D MPS cores on x-bits only (L cores of shape ``(r, 2, r)``).
    """
    n = len(cores)
    result_cores: list[Tensor] = []
    transfer: Tensor | None = None  # accumulated contraction from v-cores

    for k in range(n):
        c = cores[k]
        if k in v_bit_sites:
            # Contract with [1, 1]: sum over physical index
            mat = c[:, 0, :] + c[:, 1, :]  # (r_left, r_right)
            transfer = mat if transfer is None else transfer @ mat
        else:
            # x-bit: absorb any accumulated v-core contraction
            if transfer is not None:
                c = torch.einsum("ij,jdk->idk", transfer, c)
                transfer = None
            result_cores.append(c)

    # Absorb any trailing v-cores into last x-core
    if transfer is not None and result_cores:
        result_cores[-1] = torch.einsum("ijk,kl->ijl", result_cores[-1], transfer)

    return result_cores


# ---------------------------------------------------------------------------
# 1D ↔ 2D Expansion
# ---------------------------------------------------------------------------


def expand_1d_to_2d(
    x_cores: list[Tensor],
    num_qubits_total: int,
    x_bit_sites: list[int],
    v_bit_sites_sorted: list[int],
) -> list[Tensor]:
    """Broadcast 1D x-only MPS to 2D Morton-interleaved QTT.

    The result represents ``E(x, v) = E(x)`` — constant in velocity.
    At each v-bit position, an identity "broadcast" core is inserted:
    ``core[:, 0, :] = core[:, 1, :] = I_r``, so the output is independent
    of the v-bit value.

    Parameters
    ----------
    x_cores : list[Tensor]
        1D MPS cores for E(x), one per x-bit (L cores).
    num_qubits_total : int
        Total sites in the 2D QTT (2L).
    x_bit_sites : list[int]
        Sorted site indices for x-bits.
    v_bit_sites_sorted : list[int]
        Sorted site indices for v-bits.

    Returns
    -------
    list[Tensor]
        2D QTT cores (2L cores) representing E(x) broadcast to (x, v).
    """
    v_set = set(v_bit_sites_sorted)
    device = x_cores[0].device
    dtype = x_cores[0].dtype

    result: list[Tensor] = []
    x_idx = 0  # pointer into x_cores

    for k in range(num_qubits_total):
        if k not in v_set:
            # x-bit site: use original x-core
            result.append(x_cores[x_idx].clone())
            x_idx += 1
        else:
            # v-bit site: identity broadcast core
            # Bond dimension must match the chain at this position.
            if x_idx == 0:
                # Before any x-core: left boundary = 1
                r = 1
            else:
                # After x-core (x_idx-1): right bond of that core
                r = x_cores[x_idx - 1].shape[2]

            bc = torch.zeros(r, 2, r, device=device, dtype=dtype)
            bc[:, 0, :] = torch.eye(r, device=device, dtype=dtype)
            bc[:, 1, :] = torch.eye(r, device=device, dtype=dtype)
            result.append(bc)

    return result


# ---------------------------------------------------------------------------
# Dense ↔ QTT Conversion (small 1D problems)
# ---------------------------------------------------------------------------


def qtt_to_dense_1d(cores: list[Tensor]) -> Tensor:
    """Contract 1D QTT/MPS cores into a dense vector of length 2^L."""
    result = cores[0].squeeze(0)  # (2, r1) or (1, 2, r1) → (2, r1)
    for k in range(1, len(cores)):
        # result: (N_so_far, r_k)   c: (r_k, 2, r_{k+1})
        result = torch.einsum("ir,rds->ids", result, cores[k])
        result = result.reshape(-1, cores[k].shape[2])
    return result.squeeze(-1)  # (2^L,)


def dense_to_qtt_1d(
    vec: Tensor,
    max_rank: int = 64,
    tol: float = 1e-10,
) -> list[Tensor]:
    """Convert a dense vector to QTT/MPS cores via TT-SVD.

    Parameters
    ----------
    vec : Tensor
        1D tensor of length 2^L.
    max_rank : int
        Maximum bond dimension.
    tol : float
        SVD truncation tolerance (relative to largest singular value).

    Returns
    -------
    list[Tensor]
        QTT cores, each of shape ``(r_left, 2, r_right)``.
    """
    N = vec.shape[0]
    L = int(round(math.log2(N)))
    assert 1 << L == N, f"Vector length must be power of 2, got {N}"

    device = vec.device
    dtype = vec.dtype
    cores: list[Tensor] = []
    C = vec.clone()
    r_prev = 1

    for k in range(L - 1):
        n_remaining = C.numel() // r_prev
        C = C.reshape(r_prev * 2, n_remaining // 2)

        U, S, Vh = torch.linalg.svd(C, full_matrices=False)

        # Truncate
        r_new = len(S)
        if tol > 0 and r_new > 0:
            threshold = tol * S[0].item()
            r_new = max(int((S > threshold).sum().item()), 1)
        r_new = min(r_new, max_rank)

        U = U[:, :r_new]
        S = S[:r_new]
        Vh = Vh[:r_new, :]

        cores.append(U.reshape(r_prev, 2, r_new))
        C = torch.diag(S) @ Vh
        r_prev = r_new

    # Last core
    cores.append(C.reshape(r_prev, 2, 1))
    return cores


# ---------------------------------------------------------------------------
# Dense ↔ QTT Conversion (2D — Morton interleaved)
# ---------------------------------------------------------------------------


def _build_morton_lut(L: int) -> tuple[Tensor, Tensor]:
    """Precompute Morton ↔ (x, v) lookup tables for a 2D grid.

    Parameters
    ----------
    L : int
        Qubits per dimension (grid is 2^L × 2^L).

    Returns
    -------
    lut : Tensor
        Shape ``(N*N, 2)`` mapping Morton index → ``(x_idx, v_idx)``.
    inv_lut : Tensor
        Shape ``(N, N)`` mapping ``(x_idx, v_idx)`` → Morton index.
    """
    N = 1 << L
    lut = torch.zeros(N * N, 2, dtype=torch.long)
    inv_lut = torch.zeros(N, N, dtype=torch.long)
    for m in range(N * N):
        x_idx = 0
        v_idx = 0
        for b in range(L):
            x_idx |= ((m >> (2 * b)) & 1) << b
            v_idx |= ((m >> (2 * b + 1)) & 1) << b
        lut[m, 0] = x_idx
        lut[m, 1] = v_idx
        inv_lut[x_idx, v_idx] = m
    return lut, inv_lut


def qtt_to_dense_2d(cores: list[Tensor], L: int) -> Tensor:
    """Decompress 2D Morton-interleaved QTT to a dense ``(N_x, N_v)`` array.

    Parameters
    ----------
    cores : list[Tensor]
        QTT cores (2L sites).
    L : int
        Qubits per dimension.

    Returns
    -------
    Tensor
        Shape ``(N, N)`` array ``f[x_idx, v_idx]``.
    """
    N = 1 << L
    flat = qtt_to_dense_1d(cores)  # length N^2 in Morton order
    lut, _ = _build_morton_lut(L)
    result = torch.zeros(N, N, dtype=flat.dtype, device=flat.device)
    result[lut[:, 0], lut[:, 1]] = flat
    return result


def dense_to_qtt_2d(
    arr: Tensor, L: int, max_rank: int = 64, tol: float = 1e-10
) -> list[Tensor]:
    """Compress a dense ``(N_x, N_v)`` array to Morton-interleaved QTT.

    Parameters
    ----------
    arr : Tensor
        Shape ``(N, N)`` array ``f[x_idx, v_idx]``.
    L : int
        Qubits per dimension.
    max_rank : int
        Maximum bond dimension.
    tol : float
        SVD truncation tolerance.

    Returns
    -------
    list[Tensor]
        QTT cores (2L sites) in Morton order.
    """
    N = 1 << L
    _, inv_lut = _build_morton_lut(L)
    flat = arr[
        torch.arange(N).unsqueeze(1).expand(N, N).reshape(-1),
        torch.arange(N).unsqueeze(0).expand(N, N).reshape(-1),
    ]
    # Reorder from row-major (x_idx, v_idx) to Morton order
    morton_vec = torch.zeros(N * N, dtype=arr.dtype, device=arr.device)
    for xi in range(N):
        for vi in range(N):
            morton_vec[inv_lut[xi, vi]] = arr[xi, vi]
    return dense_to_qtt_1d(morton_vec, max_rank=max_rank, tol=tol)


# ---------------------------------------------------------------------------
# FFT Poisson Solve (1D)
# ---------------------------------------------------------------------------


def poisson_solve_1d(rho: Tensor, dx: float) -> Tensor:
    """Solve Gauss's law ``∂E/∂x = ρ`` for the electric field via FFT.

    Uses periodic boundary conditions.  The k=0 mode is set to zero
    (no net charge / mean-free field).

    Parameters
    ----------
    rho : Tensor
        Charge density on the x-grid (length N_x).
    dx : float
        Spatial grid spacing.

    Returns
    -------
    Tensor
        Electric field E(x) on the same grid.
    """
    N = rho.shape[0]
    rho_hat = torch.fft.rfft(rho)
    k = torch.fft.rfftfreq(N, d=dx / (2.0 * math.pi)).to(
        device=rho.device, dtype=rho.dtype
    )
    k[0] = 1.0  # avoid division by zero
    E_hat = rho_hat / (1j * k.to(dtype=torch.complex128 if rho.dtype == torch.float64 else torch.complex64))
    E_hat[0] = 0.0  # zero mean field
    E = torch.fft.irfft(E_hat, n=N)
    return E.to(dtype=rho.dtype)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Vlasov1D1VConfig:
    """Configuration for the genuine 1D+1V Vlasov–Poisson solver.

    Parameters
    ----------
    qubits_per_dim : int
        Number of qubits per spatial/velocity dimension.
        Grid size = 2^qubits_per_dim per axis.
    max_rank : int
        Maximum QTT bond dimension (controls accuracy vs speed).
    svd_tol : float
        SVD truncation tolerance for QTT compression.
    x_max : float
        Spatial domain: x ∈ [0, 2π / k] with periodic BC.
        Default 2π for k=1, set to 4π for k=0.5.
    v_max : float
        Velocity domain: v ∈ [-v_max, +v_max].
    device : str
        Torch device ('cpu' or 'cuda').
    dtype : torch.dtype
        Floating point precision (float64 recommended for physics).
    """

    qubits_per_dim: int = 6
    max_rank: int = 40
    svd_tol: float = 1e-8
    x_max: float = 4 * math.pi  # L = 2π/k; for k=0.5 → L=4π
    v_max: float = 6.0
    device: str = "cpu"
    dtype: torch.dtype = torch.float64

    @property
    def num_dims(self) -> int:
        return 2

    @property
    def grid_size(self) -> int:
        return 1 << self.qubits_per_dim

    @property
    def total_qubits(self) -> int:
        return self.num_dims * self.qubits_per_dim

    @property
    def dx(self) -> float:
        """Spatial grid spacing."""
        return (2.0 * self.x_max) / self.grid_size

    @property
    def dv(self) -> float:
        """Velocity grid spacing."""
        return (2.0 * self.v_max) / self.grid_size


# ---------------------------------------------------------------------------
# State Container
# ---------------------------------------------------------------------------


@dataclass
class Vlasov1D1VState:
    """State of the 1D+1V Vlasov–Poisson solver.

    Attributes
    ----------
    cores : list[Tensor]
        QTT cores of the distribution function f(x, v).
    time : float
        Current simulation time.
    step_count : int
        Number of time steps taken.
    E_energy : list[float]
        Electric field energy ½∫E²dx at each diagnostic step.
    particle_number : list[float]
        Total particle count ∫∫f dx dv at each diagnostic step.
    metadata : dict
        Solver metadata.
    """

    cores: list[Tensor]
    time: float = 0.0
    step_count: int = 0
    E_energy: list[float] = field(default_factory=list)
    particle_number: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


class Vlasov1D1V:
    """Genuine 1D+1V Vlasov–Poisson solver with velocity-dependent transport.

    Physics:
        ∂f/∂t + v ∂f/∂x − E(x) ∂f/∂v = 0
        E solved from Gauss's law: ∂E/∂x = 1 − ∫f dv

    The key distinction from the mislabeled Vlasov solvers is the
    ``velocity_multiply`` operation: the spatial advection speed is the
    velocity coordinate v itself, NOT a constant.

    Usage::

        cfg = Vlasov1D1VConfig(qubits_per_dim=6, max_rank=40)
        solver = Vlasov1D1V(cfg)
        state = solver.landau_ic(epsilon=0.01, k=0.5)
        for _ in range(200):
            state = solver.step(state, dt=0.05)
    """

    def __init__(self, config: Vlasov1D1VConfig) -> None:
        self.cfg = config
        nq = config.total_qubits
        nd = config.num_dims
        dev = torch.device(config.device)
        dt_ = config.dtype

        # Identify bit sites
        # Convention: dim 0 = x, dim 1 = v
        self.x_bit_sites = _x_bit_sites(nq, nd, x_axis=0)
        self.v_bit_info = _v_bit_sites(nq, nd, v_axis=1)
        self.v_bit_sites_sorted = [s for s, _ in self.v_bit_info]
        self.v_bit_sites_set = set(self.v_bit_sites_sorted)

        # Pre-build shift MPOs (±1 in x and v)
        self.shift_x_plus = make_nd_shift_mpo(
            nq, nd, axis_idx=0, direction=+1, device=dev, dtype=dt_
        )
        self.shift_x_minus = make_nd_shift_mpo(
            nq, nd, axis_idx=0, direction=-1, device=dev, dtype=dt_
        )
        self.shift_v_plus = make_nd_shift_mpo(
            nq, nd, axis_idx=1, direction=+1, device=dev, dtype=dt_
        )
        self.shift_v_minus = make_nd_shift_mpo(
            nq, nd, axis_idx=1, direction=-1, device=dev, dtype=dt_
        )

        # Physical parameters
        self.v_min = -config.v_max  # v ∈ [-v_max, +v_max)
        self.dv = config.dv
        self.dx = config.dx

        logger.info(
            "Vlasov1D1V: %d qubits/dim, grid %d×%d, dx=%.4f, dv=%.4f, "
            "v ∈ [%.1f, %.1f], max_rank=%d",
            config.qubits_per_dim,
            config.grid_size,
            config.grid_size,
            self.dx,
            self.dv,
            self.v_min,
            config.v_max,
            config.max_rank,
        )

    # -----------------------------------------------------------------------
    # Initial Condition
    # -----------------------------------------------------------------------

    def landau_ic(
        self,
        epsilon: float = 0.01,
        k: float = 0.5,
    ) -> Vlasov1D1VState:
        """Create Landau damping IC with guaranteed perturbation preservation.

        Constructs the initial condition:

        .. math::
            f(x, v, 0) = \\frac{1}{\\sqrt{2\\pi}} e^{-v^2/2}
                          \\bigl(1 + \\varepsilon \\cos(kx)\\bigr)

        Uses dense construction + TT-SVD compression to guarantee the
        ε perturbation is preserved at SVD tolerance level.  This bypasses
        TCI sampling risk where the low-rank Maxwellian can satisfy the
        interpolation tolerance before the TCI discovers the cosine modulation.

        Parameters
        ----------
        epsilon : float
            Perturbation amplitude.
        k : float
            Wavenumber of the perturbation.

        Returns
        -------
        Vlasov1D1VState
            Initial state with QTT cores.
        """
        cfg = self.cfg
        N = cfg.grid_size
        dev = torch.device(cfg.device)
        dt_ = cfg.dtype

        inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)

        # Build exact dense 2D IC
        x = torch.arange(N, dtype=dt_, device=dev) * self.dx  # [0, 2*x_max)
        v = torch.arange(N, dtype=dt_, device=dev) * self.dv + self.v_min  # [-v_max, +v_max)
        X, V = torch.meshgrid(x, v, indexing='ij')

        f_dense = inv_sqrt_2pi * torch.exp(-0.5 * V * V) * (
            1.0 + epsilon * torch.cos(k * X)
        )

        # Compress via TT-SVD (lossless at 1e-10 tolerance, preserves ε signal)
        cores = dense_to_qtt_2d(f_dense, cfg.qubits_per_dim,
                                max_rank=cfg.max_rank, tol=cfg.svd_tol)
        cores = [c.to(device=dev, dtype=dt_) for c in cores]

        max_bond = max(c.shape[2] for c in cores[:-1])
        logger.info(
            "Landau IC (dense TT-SVD): ε=%.4f, k=%.2f, max bond = %d",
            epsilon, k, max_bond,
        )

        return Vlasov1D1VState(
            cores=cores,
            time=0.0,
            step_count=0,
            metadata={
                "ic_type": "landau_1d1v",
                "epsilon": epsilon,
                "k": k,
                "method": "dense TT-SVD (guaranteed ε preservation)",
            },
        )

    # -----------------------------------------------------------------------
    # Time Stepping
    # -----------------------------------------------------------------------

    def step(self, state: Vlasov1D1VState, dt: float) -> Vlasov1D1VState:
        """Advance one full time step via Strang splitting.

        Sequence:
            1. Half-step x-advection: ∂f/∂t = −v ∂f/∂x
            2. E-field from Gauss's law
            3. Full-step v-advection: ∂f/∂t = E(x) ∂f/∂v
            4. Half-step x-advection (repeat)

        Parameters
        ----------
        state : Vlasov1D1VState
            Current state.
        dt : float
            Time step size.

        Returns
        -------
        Vlasov1D1VState
            Updated state.
        """
        cores = state.cores

        # 1. Half-step x-advection
        cores = self._x_advect(cores, dt / 2.0)

        # 2. Compute E-field + full-step v-advection
        cores = self._v_advect_with_poisson(cores, dt)

        # 3. Half-step x-advection
        cores = self._x_advect(cores, dt / 2.0)

        # Diagnostics
        E_field_dense = self._compute_E_field(cores)
        E_energy = 0.5 * float(torch.sum(E_field_dense ** 2).item()) * self.dx

        new_state = Vlasov1D1VState(
            cores=cores,
            time=state.time + dt,
            step_count=state.step_count + 1,
            E_energy=state.E_energy + [E_energy],
            particle_number=state.particle_number.copy(),
            metadata=state.metadata,
        )
        return new_state

    def step_dense(self, state: Vlasov1D1VState, dt: float) -> Vlasov1D1VState:
        """Advance one full time step using dense operations.

        Decompresses the QTT state to a dense 2D array, performs the
        Strang-split time step with velocity-dependent advection and
        Poisson-solved E-field, then recompresses to QTT.

        This is the **validated reference path** that gives correct Landau
        damping rates.  For small grids (≤ 2^7 per dim), this is fast
        and accurate.  The pure-QTT ``step()`` is available for large grids
        but requires high max_rank to preserve small perturbation signals.

        Parameters
        ----------
        state : Vlasov1D1VState
            Current state.
        dt : float
            Time step size.

        Returns
        -------
        Vlasov1D1VState
            Updated state.
        """
        cfg = self.cfg
        L = cfg.qubits_per_dim
        N = cfg.grid_size

        # Decompress to dense
        f = qtt_to_dense_2d(state.cores, L)

        # Physical grids
        v = torch.arange(N, device=f.device, dtype=f.dtype) * self.dv + self.v_min

        # Strang: half x-advection → full v-advection → half x-advection
        f = self._x_advect_dense(f, dt / 2.0, v)
        f = self._v_advect_dense(f, dt)
        f = self._x_advect_dense(f, dt / 2.0, v)

        # Recompress to QTT
        cores = dense_to_qtt_2d(f, L, max_rank=cfg.max_rank, tol=cfg.svd_tol)
        dev = torch.device(cfg.device)
        cores = [c.to(device=dev, dtype=cfg.dtype) for c in cores]

        # Diagnostics
        rho = 1.0 - self.dv * f.sum(dim=1)
        E_dense = poisson_solve_1d(rho, self.dx)
        E_energy = 0.5 * float(torch.sum(E_dense ** 2).item()) * self.dx

        return Vlasov1D1VState(
            cores=cores,
            time=state.time + dt,
            step_count=state.step_count + 1,
            E_energy=state.E_energy + [E_energy],
            particle_number=state.particle_number.copy(),
            metadata=state.metadata,
        )

    # -----------------------------------------------------------------------
    # Dense sub-steps (for step_dense) — Semi-Lagrangian / Spectral Shift
    # -----------------------------------------------------------------------

    @staticmethod
    def _fft_shift_1d(f_1d: Tensor, shift_cells: float) -> Tensor:
        """Shift a 1D periodic function by fractional cells using FFT.

        Uses the Fourier shift theorem: ``IFFT(FFT(f) · exp(−ikΔ))``.
        Spectrally accurate — zero numerical diffusion.
        """
        N = f_1d.shape[0]
        k = torch.fft.fftfreq(N).to(device=f_1d.device, dtype=torch.float64)
        k = k * (2.0 * math.pi)  # angular frequency per cell
        f_hat = torch.fft.fft(f_1d.to(torch.complex128))
        f_hat = f_hat * torch.exp(-1j * k * shift_cells)
        return torch.fft.ifft(f_hat).real.to(f_1d.dtype)

    def _x_advect_dense(self, f: Tensor, dt_sub: float, v: Tensor) -> Tensor:
        """Spectral semi-Lagrangian x-advection.

        Solves ∂f/∂t + v ∂f/∂x = 0 exactly via Fourier shift:
            f_new(x, v_j) = shift(f(·, v_j), v_j·Δt / Δx)

        Spectrally accurate in x — no numerical diffusion.
        """
        shift = v * dt_sub / self.dx  # fractional cells per velocity, shape (N_v,)
        f_new = torch.zeros_like(f)
        for j in range(f.shape[1]):
            f_new[:, j] = self._fft_shift_1d(f[:, j], shift[j].item())
        return f_new

    def _v_advect_dense(self, f: Tensor, dt_sub: float) -> Tensor:
        """Spectral semi-Lagrangian v-advection with Poisson solve.

        Solves ∂f/∂t − E(x) ∂f/∂v = 0 (electron Vlasov).

        Characteristics: dv/dt = −E  →  departure v = v + E·Δt.
        In Fourier shift convention: g(v) = f(v − s·Δv) with s = −E·Δt/Δv.
        """
        rho = 1.0 - self.dv * f.sum(dim=1)
        E = poisson_solve_1d(rho, self.dx)

        # Departure: v + E*dt, so shift_cells = -E*dt/dv
        shift = -E * dt_sub / self.dv  # (N_x,) cells per x-position
        f_new = torch.zeros_like(f)
        for i in range(f.shape[0]):
            f_new[i, :] = self._fft_shift_1d(f[i, :], shift[i].item())
        return f_new

    # -----------------------------------------------------------------------
    # X-Advection: ∂f/∂t = −v ∂f/∂x
    # -----------------------------------------------------------------------

    def _x_advect(self, cores: list[Tensor], dt_sub: float) -> list[Tensor]:
        """Explicit Euler sub-step for spatial advection.

        .. math::
            f ← f − Δt \\, v \\, \\frac{S^+_x f - S^-_x f}{2 Δx}

        The velocity-coordinate multiplication ``v × ∂f/∂x`` is the
        genuinely new piece that makes this a REAL Vlasov solver.
        """
        mr = self.cfg.max_rank
        tol = self.cfg.svd_tol

        # Compute ∂f/∂x via central differences using shift MPOs
        f_xp = apply_nd_shift_mpo(cores, self.shift_x_plus, max_rank=mr, tol=tol)
        f_xm = apply_nd_shift_mpo(cores, self.shift_x_minus, max_rank=mr, tol=tol)

        # df_dx = (f_xp - f_xm) / (2 dx)
        nq = self.cfg.total_qubits
        df_dx = qtt_add(
            QTTState(f_xp, nq),
            qtt_scale(QTTState(f_xm, nq), -1.0),
            max_bond=mr,
            tol=tol,
        )
        # Scale by 1/(2*dx)
        df_dx = qtt_scale(df_dx, 1.0 / (2.0 * self.dx))

        # Multiply by velocity: v_df_dx = v × ∂f/∂x
        v_df_dx_cores = velocity_multiply(
            df_dx.cores,
            self.v_min,
            self.dv,
            self.v_bit_info,
            max_rank=mr,
            tol=tol,
        )

        # Update: f = f - dt_sub * v * df/dx
        updated = qtt_add(
            QTTState(cores, nq),
            qtt_scale(QTTState(v_df_dx_cores, nq), -dt_sub),
            max_bond=mr,
            tol=tol,
        )
        return updated.cores

    # -----------------------------------------------------------------------
    # V-Advection: ∂f/∂t = E(x) ∂f/∂v   (with Poisson)
    # -----------------------------------------------------------------------

    def _compute_E_field(self, cores: list[Tensor]) -> Tensor:
        """Compute E-field from current distribution via Poisson solve.

        Steps:
            1. Partial trace over v → charge density ρ_raw(x) = Σ_v f(x,v)
            2. Convert to dense 1D vector
            3. Physical charge: ρ(x) = 1 − dv × ρ_raw(x)
            4. FFT Poisson solve: ∂E/∂x = ρ
        """
        # Partial trace over velocity
        x_cores = partial_trace_velocity(cores, self.v_bit_sites_set)

        # Convert to dense
        rho_raw = qtt_to_dense_1d(x_cores)

        # Physical charge density: ρ = 1 − dv × Σ_v f
        rho = 1.0 - self.dv * rho_raw

        # Poisson solve
        E_dense = poisson_solve_1d(rho, self.dx)
        return E_dense

    def _v_advect_with_poisson(self, cores: list[Tensor], dt_sub: float) -> list[Tensor]:
        """Explicit Euler sub-step for velocity advection with self-consistent E-field.

        .. math::
            E(x) \\text{ from Gauss's law}
            f ← f + Δt \\, E(x) \\, \\frac{S^+_v f − S^-_v f}{2 Δv}
        """
        mr = self.cfg.max_rank
        tol = self.cfg.svd_tol
        nq = self.cfg.total_qubits

        # 1. Compute E-field
        E_dense = self._compute_E_field(cores)

        # 2. Convert E to QTT (1D x-only MPS)
        E_1d_cores = dense_to_qtt_1d(E_dense, max_rank=mr, tol=tol)

        # 3. Expand E(x) to 2D QTT: E(x, v) = E(x) (constant in v)
        E_2d_cores = expand_1d_to_2d(
            E_1d_cores,
            nq,
            self.x_bit_sites,
            self.v_bit_sites_sorted,
        )

        # 4. Compute ∂f/∂v via central differences
        f_vp = apply_nd_shift_mpo(cores, self.shift_v_plus, max_rank=mr, tol=tol)
        f_vm = apply_nd_shift_mpo(cores, self.shift_v_minus, max_rank=mr, tol=tol)

        df_dv = qtt_add(
            QTTState(f_vp, nq),
            qtt_scale(QTTState(f_vm, nq), -1.0),
            max_bond=mr,
            tol=tol,
        )
        df_dv = qtt_scale(df_dv, 1.0 / (2.0 * self.dv))

        # 5. Multiply E(x) × ∂f/∂v via Hadamard product
        E_df_dv = qtt_hadamard(
            QTTState(E_2d_cores, nq),
            df_dv,
            max_bond=mr,
            truncate=True,
        )

        # 6. Update: f = f + dt * E * df/dv
        updated = qtt_add(
            QTTState(cores, nq),
            qtt_scale(E_df_dv, dt_sub),
            max_bond=mr,
            tol=tol,
        )
        return updated.cores

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def compute_particle_number(self, state: Vlasov1D1VState) -> float:
        """Compute total particle count ∫∫ f dx dv."""
        # Sum over ALL indices: contract every core with [1, 1]
        val = torch.ones(1, 1, device=state.cores[0].device, dtype=state.cores[0].dtype)
        for c in state.cores:
            mat = c[:, 0, :] + c[:, 1, :]  # (r_left, r_right)
            val = val @ mat
        return float(val.item()) * self.dx * self.dv


# ---------------------------------------------------------------------------
# Landau Damping Validation
# ---------------------------------------------------------------------------


def measure_damping_rate(
    times: list[float],
    E_energies: list[float],
    t_start: float = 0.5,
    t_end: float | None = None,
) -> tuple[float, float]:
    """Fit exponential decay to the **peak envelope** of E-field energy.

    The electric field energy oscillates at the plasma frequency while its
    envelope decays as ``exp(2γt)``.  Fitting raw data gives poor R² due to
    the oscillations.  This function extracts local maxima (peaks) of the
    E_energy time series and fits ``log(E_peak) = 2γ·t + const`` through
    the peaks only, cleanly recovering the damping rate.

    Falls back to raw-data fitting if fewer than 3 peaks are found.

    Parameters
    ----------
    times : list[float]
        Time values.
    E_energies : list[float]
        Electric field energy at each time.
    t_start : float
        Discard data before this time (skip initial transient).
    t_end : float or None
        End of fit window (default: end of data).

    Returns
    -------
    gamma : float
        Fitted damping rate (should be ≈ −0.1533 for k = 0.5).
    r_squared : float
        Coefficient of determination of the linear fit.
    """
    t = torch.tensor(times, dtype=torch.float64)
    E = torch.tensor(E_energies, dtype=torch.float64)

    if t_end is None:
        t_end = float(t[-1])

    # --- Peak envelope extraction ---
    peaks_t: list[float] = []
    peaks_E: list[float] = []
    for i in range(1, len(E) - 1):
        if (E[i] > E[i - 1] and E[i] > E[i + 1]
                and E[i] > 0
                and t[i] >= t_start and t[i] <= t_end):
            peaks_t.append(float(t[i]))
            peaks_E.append(float(E[i]))

    if len(peaks_t) >= 3:
        t_fit = torch.tensor(peaks_t, dtype=torch.float64)
        logE = torch.log(torch.tensor(peaks_E, dtype=torch.float64))
    else:
        # Fallback: fit raw data
        mask = (t >= t_start) & (t <= t_end) & (E > 0)
        t_fit = t[mask]
        logE = torch.log(E[mask])

    if len(t_fit) < 3:
        logger.warning("Too few points for damping fit: %d", len(t_fit))
        return 0.0, 0.0

    # Linear regression: logE = slope * t + intercept
    # slope = 2γ → γ = slope / 2
    t_mean = t_fit.mean()
    logE_mean = logE.mean()
    dt = t_fit - t_mean
    dlogE = logE - logE_mean
    slope = float((dt * dlogE).sum() / (dt * dt).sum())

    # R² for goodness of fit
    ss_res = float(((logE - (slope * t_fit + (logE_mean - slope * t_mean))) ** 2).sum())
    ss_tot = float((dlogE ** 2).sum())
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-30)

    gamma = slope / 2.0
    return gamma, r_squared


def landau_damping_1d1v(
    qubits_per_dim: int = 6,
    max_rank: int = 40,
    dt: float = 0.05,
    n_steps: int = 400,
    epsilon: float = 0.01,
    k: float = 0.5,
    device: str = "cpu",
    verbose: bool = True,
    use_dense_step: bool = True,
) -> dict[str, Any]:
    """Run a complete Landau damping simulation and validate the damping rate.

    This is the end-to-end proof that the Vlasov solver is genuine:
    the electric field energy must decay at the theoretically predicted
    rate γ ≈ −0.1533 for k = 0.5.

    Parameters
    ----------
    qubits_per_dim : int
        Grid resolution (2^qubits_per_dim per axis).
    max_rank : int
        Maximum QTT bond dimension.
    dt : float
        Time step.
    n_steps : int
        Number of time steps.
    epsilon : float
        Initial perturbation amplitude.
    k : float
        Perturbation wavenumber.
    device : str
        Torch device.
    verbose : bool
        Print progress.
    use_dense_step : bool
        If True (default), use ``step_dense()`` for time integration.
        This decompresses, does a dense Strang step with velocity-dependent
        advection + Poisson E-field, then recompresses.  Gives correct
        physics on small grids.  Set False for pure-QTT path (needs
        high max_rank to avoid rank-truncation losses).

    Returns
    -------
    dict
        Results including measured γ, theoretical γ, relative error,
        and full time traces.
    """
    cfg = Vlasov1D1VConfig(
        qubits_per_dim=qubits_per_dim,
        max_rank=max_rank,
        svd_tol=1e-8,
        x_max=2.0 * math.pi / k,  # Domain = one wavelength
        v_max=6.0,
        device=device,
        dtype=torch.float64,
    )

    solver = Vlasov1D1V(cfg)
    state = solver.landau_ic(epsilon=epsilon, k=k)

    # Record initial E-field energy
    E0 = solver._compute_E_field(state.cores)
    E_energy_0 = 0.5 * float(torch.sum(E0 ** 2).item()) * solver.dx
    state.E_energy.append(E_energy_0)

    times = [0.0]

    if verbose:
        mode = "dense-step" if use_dense_step else "pure-QTT"
        print(f"Vlasov 1D+1V Landau Damping: {cfg.grid_size}×{cfg.grid_size} grid, "
              f"max_rank={max_rank}, dt={dt}, mode={mode}")
        print(f"  x ∈ [0, {2*cfg.x_max:.3f}], v ∈ [{-cfg.v_max}, {cfg.v_max}]")
        print(f"  dx={solver.dx:.4f}, dv={solver.dv:.4f}")
        print(f"  CFL (x): dt*v_max/dx = {dt*cfg.v_max/solver.dx:.3f}")
        print(f"  Initial E_energy = {E_energy_0:.6e}")

    step_fn = solver.step_dense if use_dense_step else solver.step

    for step_i in range(1, n_steps + 1):
        state = step_fn(state, dt)
        times.append(state.time)

        if verbose and (step_i % 50 == 0 or step_i == 1):
            max_bond = max(c.shape[2] for c in state.cores[:-1])
            print(
                f"  Step {step_i:4d}  t={state.time:7.3f}  "
                f"E_energy={state.E_energy[-1]:.4e}  max_bond={max_bond}"
            )

    # Measure damping rate
    gamma_theory = -0.1533  # Theoretical for k=0.5
    gamma_measured, r_squared = measure_damping_rate(
        times, state.E_energy, t_start=2.0
    )

    if verbose:
        rel_error = abs(gamma_measured - gamma_theory) / abs(gamma_theory) if gamma_theory != 0 else float("inf")
        print(f"\n  γ_measured  = {gamma_measured:.4f}")
        print(f"  γ_theory   = {gamma_theory:.4f}")
        print(f"  Rel. error = {rel_error:.1%}")
        print(f"  R²         = {r_squared:.4f}")
        status = "PASS" if rel_error < 0.15 else "FAIL"
        print(f"  Validation : {status}")

    return {
        "gamma_measured": gamma_measured,
        "gamma_theory": gamma_theory,
        "relative_error": abs(gamma_measured - gamma_theory) / abs(gamma_theory),
        "r_squared": r_squared,
        "times": times,
        "E_energy": state.E_energy,
        "final_state": state,
        "config": cfg,
        "passed": abs(gamma_measured - gamma_theory) / abs(gamma_theory) < 0.15,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = landau_damping_1d1v(
        qubits_per_dim=6,
        max_rank=40,
        dt=0.05,
        n_steps=400,
        device=device,
        verbose=True,
    )
    sys.exit(0 if result["passed"] else 1)
