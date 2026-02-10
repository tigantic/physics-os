"""
Lattice Boltzmann Method (LBM)
==============================

Mesoscopic kinetic solver for fluid dynamics based on the Boltzmann
transport equation discretised on a regular lattice.

Governing Equation (BGK single-relaxation-time):
    f_i(x + c_i Δt, t + Δt) = f_i(x, t)
        - (f_i - f_i^{eq}) / τ + F_i Δt

    where f_i is the distribution function for velocity direction i,
    f_i^{eq} is the Maxwell–Boltzmann equilibrium, and τ is the
    relaxation time related to kinematic viscosity:

        ν = c_s² (τ - 0.5) Δt

Lattice Models:
    - D2Q9  — 2D, 9 velocities
    - D3Q19 — 3D, 19 velocities
    - D3Q27 — 3D, 27 velocities

Collision Operators:
    - BGK (single relaxation time)
    - TRT (two relaxation times)
    - MRT (multiple relaxation times)

References:
    [1] Chen & Doolen, "Lattice Boltzmann Method for Fluid Flows",
        Annu. Rev. Fluid Mech. 30, 1998.
    [2] Krüger et al., "The Lattice Boltzmann Method: Principles and
        Practice", Springer, 2017.
    [3] d'Humières et al., "Multiple–relaxation–time lattice Boltzmann
        models in three dimensions", Phil. Trans. R. Soc. A 360, 2002.

Domain II.1 — Computational Fluid Dynamics / Lattice Boltzmann.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class LatticeModel(Enum):
    """Supported lattice velocity sets."""
    D2Q9 = "D2Q9"
    D3Q19 = "D3Q19"
    D3Q27 = "D3Q27"


class CollisionModel(Enum):
    """Collision operator variants."""
    BGK = "BGK"
    TRT = "TRT"
    MRT = "MRT"


# ---------------------------------------------------------------------------
# Lattice velocity sets
# ---------------------------------------------------------------------------

def _d2q9() -> Tuple[NDArray, NDArray]:
    """Return D2Q9 velocity vectors and weights."""
    c = np.array([
        [0, 0],   # 0
        [1, 0],   # 1
        [0, 1],   # 2
        [-1, 0],  # 3
        [0, -1],  # 4
        [1, 1],   # 5
        [-1, 1],  # 6
        [-1, -1], # 7
        [1, -1],  # 8
    ], dtype=np.float64)
    w = np.array([
        4.0 / 9.0,
        1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    ], dtype=np.float64)
    return c, w


def _d3q19() -> Tuple[NDArray, NDArray]:
    """Return D3Q19 velocity vectors and weights."""
    c = np.array([
        [0, 0, 0],
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1],
        [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
        [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
        [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
    ], dtype=np.float64)
    w = np.zeros(19, dtype=np.float64)
    w[0] = 1.0 / 3.0
    w[1:7] = 1.0 / 18.0
    w[7:19] = 1.0 / 36.0
    return c, w


def _d3q27() -> Tuple[NDArray, NDArray]:
    """Return D3Q27 velocity vectors and weights."""
    vecs = []
    weights = []
    for ix in (-1, 0, 1):
        for iy in (-1, 0, 1):
            for iz in (-1, 0, 1):
                vecs.append([ix, iy, iz])
                csq = ix * ix + iy * iy + iz * iz
                if csq == 0:
                    weights.append(8.0 / 27.0)
                elif csq == 1:
                    weights.append(2.0 / 27.0)
                elif csq == 2:
                    weights.append(1.0 / 54.0)
                else:
                    weights.append(1.0 / 216.0)
    c = np.array(vecs, dtype=np.float64)
    w = np.array(weights, dtype=np.float64)
    return c, w


LATTICE_REGISTRY = {
    LatticeModel.D2Q9: _d2q9,
    LatticeModel.D3Q19: _d3q19,
    LatticeModel.D3Q27: _d3q27,
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class LBMState:
    """
    Distribution-function state for the lattice Boltzmann method.

    Attributes:
        f: Distribution functions — shape ``(Q, *grid_shape)`` where
           *Q* is the number of lattice velocities.
        rho: Macroscopic density — shape ``grid_shape``.
        u: Macroscopic velocity — shape ``(dim, *grid_shape)``.
    """
    f: NDArray
    rho: NDArray
    u: NDArray

    @classmethod
    def from_macroscopic(
        cls,
        rho: NDArray,
        u: NDArray,
        lattice: LatticeModel = LatticeModel.D2Q9,
    ) -> "LBMState":
        """Initialise distributions at equilibrium for given ρ and u."""
        c, w = LATTICE_REGISTRY[lattice]()
        f_eq = _equilibrium(rho, u, c, w)
        return cls(f=f_eq.copy(), rho=rho.copy(), u=u.copy())


# ---------------------------------------------------------------------------
# Core LBM kernels
# ---------------------------------------------------------------------------

def _equilibrium(
    rho: NDArray,
    u: NDArray,
    c: NDArray,
    w: NDArray,
    cs2: float = 1.0 / 3.0,
) -> NDArray:
    r"""
    Maxwell–Boltzmann equilibrium distribution:

    .. math::
        f_i^{eq} = w_i \rho \left[
            1 + \frac{c_i \cdot u}{c_s^2}
              + \frac{(c_i \cdot u)^2}{2 c_s^4}
              - \frac{u \cdot u}{2 c_s^2}
        \right]

    Args:
        rho: Density field ``(*grid_shape,)``.
        u: Velocity field ``(dim, *grid_shape)``.
        c: Lattice velocity vectors ``(Q, dim)``.
        w: Lattice weights ``(Q,)``.
        cs2: Squared speed of sound (1/3 for standard lattices).

    Returns:
        Equilibrium distributions ``(Q, *grid_shape)``.
    """
    Q = c.shape[0]
    dim = c.shape[1]
    grid_shape = rho.shape

    f_eq = np.empty((Q, *grid_shape), dtype=np.float64)
    u_sq = np.sum(u * u, axis=0)  # |u|²

    for i in range(Q):
        cu = sum(c[i, d] * u[d] for d in range(dim))  # c_i · u
        f_eq[i] = w[i] * rho * (
            1.0
            + cu / cs2
            + 0.5 * cu * cu / (cs2 * cs2)
            - 0.5 * u_sq / cs2
        )
    return f_eq


def _stream(f: NDArray, c: NDArray) -> NDArray:
    """
    Streaming step: propagate distributions along lattice links.

    f_i(x + c_i, t+1) = f_i(x, t) — implemented via np.roll.
    """
    Q = f.shape[0]
    dim = c.shape[1]
    f_new = np.empty_like(f)
    for i in range(Q):
        shift = tuple(int(c[i, d]) for d in range(dim))
        axes = tuple(range(1, 1 + dim))
        f_new[i] = np.roll(f[i], shift, axis=axes)
    return f_new


def _macroscopic(f: NDArray, c: NDArray) -> Tuple[NDArray, NDArray]:
    """Compute macroscopic density and velocity from distributions."""
    Q = f.shape[0]
    dim = c.shape[1]
    rho = np.sum(f, axis=0)
    grid_shape = rho.shape
    u = np.zeros((dim, *grid_shape), dtype=np.float64)
    for i in range(Q):
        for d in range(dim):
            u[d] += c[i, d] * f[i]
    rho_safe = np.where(np.abs(rho) < 1e-30, 1e-30, rho)
    u /= rho_safe
    return rho, u


# ---------------------------------------------------------------------------
# Collision operators
# ---------------------------------------------------------------------------

def _collide_bgk(
    f: NDArray,
    rho: NDArray,
    u: NDArray,
    c: NDArray,
    w: NDArray,
    omega: float,
) -> NDArray:
    r"""
    BGK (Bhatnagar–Gross–Krook) single-relaxation-time collision.

    .. math::
        \Omega_i = -\frac{1}{\tau}(f_i - f_i^{eq})

    Args:
        omega: :math:`1/\tau` relaxation frequency.
    """
    f_eq = _equilibrium(rho, u, c, w)
    return f - omega * (f - f_eq)


def _collide_trt(
    f: NDArray,
    rho: NDArray,
    u: NDArray,
    c: NDArray,
    w: NDArray,
    omega_plus: float,
    omega_minus: float,
) -> NDArray:
    r"""
    Two-Relaxation-Time (TRT) collision.

    Symmetric and antisymmetric parts are relaxed independently:

    .. math::
        f_i^+ &= \tfrac12 (f_i + f_{\bar i}),\quad
        f_i^- = \tfrac12 (f_i - f_{\bar i})

    where :math:`\bar i` is the opposite direction of *i*.

    The magic parameter :math:`\Lambda = (\tau^+ - 0.5)(\tau^- - 0.5)`
    controls bounce-back wall location.  :math:`\Lambda = 3/16` gives
    exact mid-grid bounce-back.
    """
    Q = f.shape[0]
    f_eq = _equilibrium(rho, u, c, w)

    # Build opposite-direction map
    opp = _opposite_map(c)

    f_out = np.empty_like(f)
    for i in range(Q):
        j = opp[i]
        f_plus = 0.5 * (f[i] + f[j])
        f_minus = 0.5 * (f[i] - f[j])
        feq_plus = 0.5 * (f_eq[i] + f_eq[j])
        feq_minus = 0.5 * (f_eq[i] - f_eq[j])
        f_out[i] = (
            f[i]
            - omega_plus * (f_plus - feq_plus)
            - omega_minus * (f_minus - feq_minus)
        )
    return f_out


def _collide_mrt(
    f: NDArray,
    rho: NDArray,
    u: NDArray,
    c: NDArray,
    w: NDArray,
    S_diag: NDArray,
    M: NDArray,
    M_inv: NDArray,
) -> NDArray:
    r"""
    Multiple-Relaxation-Time (MRT) collision.

    Transform to moment space, relax independently, then transform back:

    .. math::
        f^{\text{out}} = f - M^{-1} S (m - m^{eq})

    where *M* maps distributions to moments, *S* is a diagonal matrix
    of relaxation rates, and :math:`m^{eq}` are equilibrium moments.
    """
    Q = f.shape[0]
    grid_shape = f.shape[1:]
    n_cells = int(np.prod(grid_shape))

    f_flat = f.reshape(Q, n_cells)  # (Q, N)
    f_eq = _equilibrium(rho, u, c, w).reshape(Q, n_cells)

    m = M @ f_flat         # (Q, N)
    m_eq = M @ f_eq

    dm = S_diag[:, None] * (m - m_eq)
    f_out = f_flat - M_inv @ dm
    return f_out.reshape(f.shape)


def _opposite_map(c: NDArray) -> NDArray:
    """Return array opp where opp[i] is the index of the opposite velocity."""
    Q = c.shape[0]
    opp = np.empty(Q, dtype=np.intp)
    for i in range(Q):
        for j in range(Q):
            if np.allclose(c[j], -c[i]):
                opp[i] = j
                break
        else:
            raise ValueError(f"No opposite found for velocity {i}: {c[i]}")
    return opp


# ---------------------------------------------------------------------------
# MRT transformation matrices (D2Q9)
# ---------------------------------------------------------------------------

def mrt_d2q9_matrices() -> Tuple[NDArray, NDArray, NDArray]:
    """
    Standard D2Q9 MRT transformation matrix and relaxation rates.

    Returns ``(M, M_inv, S_diag)`` where *S_diag* uses
    Lallemand & Luo (2000) recommended values.
    """
    M = np.array([
        [1,  1,  1,  1,  1,  1,  1,  1,  1],   # rho
        [-4, -1, -1, -1, -1,  2,  2,  2,  2],   # e (energy)
        [4,  -2, -2, -2, -2,  1,  1,  1,  1],   # epsilon
        [0,   1,  0, -1,  0,  1, -1, -1,  1],   # j_x
        [0,  -2,  0,  2,  0,  1, -1, -1,  1],   # q_x
        [0,   0,  1,  0, -1,  1,  1, -1, -1],   # j_y
        [0,   0, -2,  0,  2,  1,  1, -1, -1],   # q_y
        [0,   1, -1,  1, -1,  0,  0,  0,  0],   # p_xx
        [0,   0,  0,  0,  0,  1, -1,  1, -1],   # p_xy
    ], dtype=np.float64)
    M_inv = np.linalg.inv(M)
    # Relaxation rates (s_rho and s_j are conserved → set to 0)
    S_diag = np.array([0.0, 1.4, 1.4, 0.0, 1.2, 0.0, 1.2, 1.0, 1.0],
                       dtype=np.float64)
    return M, M_inv, S_diag


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

def bounce_back(f: NDArray, c: NDArray, mask: NDArray) -> NDArray:
    """
    Full-way bounce-back for no-slip walls.

    At solid nodes swap populations: f_i ↔ f_{ī}.

    Args:
        f: Distributions ``(Q, *grid_shape)``.
        c: Lattice velocities.
        mask: Boolean array ``grid_shape`` — True at solid nodes.
    """
    opp = _opposite_map(c)
    Q = f.shape[0]
    f_out = f.copy()
    for i in range(Q):
        j = opp[i]
        f_out[i][mask] = f[j][mask]
    return f_out


def zou_he_velocity_inlet(
    f: NDArray,
    c: NDArray,
    w: NDArray,
    rho_inlet: float,
    u_inlet: NDArray,
    axis: int = 0,
    side: int = 0,
) -> NDArray:
    """
    Zou–He velocity boundary condition (non-equilibrium bounce-back).

    Imposes a prescribed velocity at the boundary and adjusts unknown
    populations to maintain mass conservation.

    Args:
        f: Distributions.
        c: Lattice velocities.
        w: Lattice weights.
        rho_inlet: Prescribed inlet density.
        u_inlet: Prescribed velocity array ``(dim,)``.
        axis: Normal axis of the boundary (0=x, 1=y, 2=z).
        side: 0 = left/bottom, 1 = right/top.
    """
    f_out = f.copy()
    Q = f.shape[0]
    dim = c.shape[1]

    # Select boundary slice
    slc: list = [slice(None)] * (1 + dim)
    slc[axis + 1] = 0 if side == 0 else -1
    boundary_slc = tuple(slc)

    # Compute equilibrium at prescribed conditions
    # Build local rho/u arrays matching boundary shape
    bnd_shape = list(f.shape[1:])
    bnd_shape[axis] = 1

    rho_b = np.full(bnd_shape, rho_inlet, dtype=np.float64)
    u_b = np.zeros((dim, *bnd_shape), dtype=np.float64)
    for d in range(dim):
        u_b[d] = u_inlet[d]

    f_eq = _equilibrium(rho_b.squeeze(axis=axis), u_b.squeeze(axis=axis + 1),
                        c, w)

    # Replace unknown populations with equilibrium + bounce-back correction
    opp = _opposite_map(c)
    for i in range(Q):
        if (side == 0 and c[i, axis] > 0) or (side == 1 and c[i, axis] < 0):
            j = opp[i]
            f_out[i][tuple(slc)[1:]] = (
                f_eq[i] + (f_out[j][tuple(slc)[1:]] - f_eq[j])
            )
    return f_out


# ---------------------------------------------------------------------------
# Forcing (Guo scheme)
# ---------------------------------------------------------------------------

def guo_forcing(
    u: NDArray,
    c: NDArray,
    w: NDArray,
    F_ext: NDArray,
    omega: float,
    cs2: float = 1.0 / 3.0,
) -> NDArray:
    r"""
    Guo's forcing scheme for body forces in LBM.

    .. math::
        F_i = \left(1 - \frac{\omega}{2}\right) w_i
              \left[\frac{c_i - u}{c_s^2}
              + \frac{(c_i \cdot u) c_i}{c_s^4}\right] \cdot \mathbf{F}

    Args:
        u: Velocity ``(dim, *grid_shape)``.
        c: Lattice velocities.
        w: Lattice weights.
        F_ext: External force density ``(dim, *grid_shape)``.
        omega: Relaxation frequency.
    """
    Q = c.shape[0]
    dim = c.shape[1]
    grid_shape = u.shape[1:]
    Fi = np.zeros((Q, *grid_shape), dtype=np.float64)
    prefactor = 1.0 - 0.5 * omega

    for i in range(Q):
        cu = sum(c[i, d] * u[d] for d in range(dim))
        dot = 0.0
        for d in range(dim):
            term = (c[i, d] - u[d]) / cs2 + cu * c[i, d] / (cs2 * cs2)
            dot = dot + term * F_ext[d]
        Fi[i] = prefactor * w[i] * dot
    return Fi


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class LatticeBoltzmannSolver:
    r"""
    Production Lattice Boltzmann solver.

    Supports D2Q9, D3Q19, D3Q27 lattices with BGK, TRT, and MRT
    collision operators.  Includes bounce-back walls, Zou–He inlets,
    and Guo forcing.

    Parameters:
        lattice: Lattice velocity model.
        collision: Collision operator.
        tau: Relaxation time.  Related to kinematic viscosity by
             :math:`\nu = c_s^2 (\tau - 0.5)`.
        grid_shape: Spatial domain dimensions.
        F_ext: Optional constant body force ``(dim,)`` (e.g., gravity).

    Example::

        solver = LatticeBoltzmannSolver(
            lattice=LatticeModel.D2Q9,
            collision=CollisionModel.BGK,
            tau=0.8,
            grid_shape=(128, 64),
        )
        state = solver.initialise(rho0=np.ones((128, 64)),
                                  u0=np.zeros((2, 128, 64)))
        for _ in range(1000):
            state = solver.step(state)
    """

    def __init__(
        self,
        lattice: LatticeModel = LatticeModel.D2Q9,
        collision: CollisionModel = CollisionModel.BGK,
        tau: float = 0.8,
        grid_shape: Tuple[int, ...] = (128, 64),
        F_ext: Optional[NDArray] = None,
    ) -> None:
        if tau <= 0.5:
            raise ValueError(f"τ must be > 0.5 for stability, got {tau}")

        self.lattice = lattice
        self.collision = collision
        self.tau = tau
        self.omega = 1.0 / tau
        self.grid_shape = grid_shape
        self.cs2 = 1.0 / 3.0
        self.nu = self.cs2 * (tau - 0.5)

        self.c, self.w = LATTICE_REGISTRY[lattice]()
        self.Q = self.c.shape[0]
        self.dim = self.c.shape[1]

        if len(grid_shape) != self.dim:
            raise ValueError(
                f"{lattice.value} requires {self.dim}D grid, "
                f"got shape {grid_shape}"
            )

        self.F_ext = F_ext
        self.solid_mask: Optional[NDArray] = None

        # MRT matrices (only for MRT collision)
        self._M: Optional[NDArray] = None
        self._M_inv: Optional[NDArray] = None
        self._S_diag: Optional[NDArray] = None

        if collision == CollisionModel.MRT:
            if lattice == LatticeModel.D2Q9:
                self._M, self._M_inv, self._S_diag = mrt_d2q9_matrices()
                # Replace viscosity-related rates with omega
                self._S_diag[7] = self.omega
                self._S_diag[8] = self.omega
            else:
                raise NotImplementedError(
                    f"MRT not yet implemented for {lattice.value}; "
                    "use BGK or TRT."
                )

        # TRT magic parameter → exact mid-grid bounce-back
        if collision == CollisionModel.TRT:
            self._omega_plus = self.omega
            Lambda = 3.0 / 16.0
            tau_minus = 0.5 + Lambda / (tau - 0.5)
            self._omega_minus = 1.0 / tau_minus

        self._step_count = 0

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def initialise(
        self,
        rho0: Optional[NDArray] = None,
        u0: Optional[NDArray] = None,
    ) -> LBMState:
        """Create initial state at Maxwell–Boltzmann equilibrium."""
        if rho0 is None:
            rho0 = np.ones(self.grid_shape, dtype=np.float64)
        if u0 is None:
            u0 = np.zeros((self.dim, *self.grid_shape), dtype=np.float64)

        if rho0.shape != self.grid_shape:
            raise ValueError(f"rho0 shape {rho0.shape} != grid {self.grid_shape}")
        if u0.shape != (self.dim, *self.grid_shape):
            raise ValueError(f"u0 shape {u0.shape} mismatch")

        return LBMState.from_macroscopic(rho0, u0, self.lattice)

    def set_solid_mask(self, mask: NDArray) -> None:
        """Set boolean mask for solid (no-slip) nodes."""
        if mask.shape != self.grid_shape:
            raise ValueError(f"Mask shape {mask.shape} != grid {self.grid_shape}")
        self.solid_mask = mask.astype(bool)

    def step(self, state: LBMState) -> LBMState:
        """Advance one LBM time step: collide → (force) → stream → BC."""
        f = state.f.copy()

        # --- Collision ---
        if self.collision == CollisionModel.BGK:
            f = _collide_bgk(f, state.rho, state.u, self.c, self.w, self.omega)
        elif self.collision == CollisionModel.TRT:
            f = _collide_trt(
                f, state.rho, state.u, self.c, self.w,
                self._omega_plus, self._omega_minus,
            )
        elif self.collision == CollisionModel.MRT:
            f = _collide_mrt(
                f, state.rho, state.u, self.c, self.w,
                self._S_diag, self._M, self._M_inv,
            )

        # --- Forcing ---
        if self.F_ext is not None:
            F_field = np.zeros((self.dim, *self.grid_shape), dtype=np.float64)
            for d in range(self.dim):
                F_field[d] = self.F_ext[d]
            Fi = guo_forcing(state.u, self.c, self.w, F_field, self.omega)
            f += Fi

        # --- Streaming ---
        f = _stream(f, self.c)

        # --- Bounce-back walls ---
        if self.solid_mask is not None:
            f = bounce_back(f, self.c, self.solid_mask)

        # --- Update macroscopic fields ---
        rho, u = _macroscopic(f, self.c)

        # Guo velocity correction for forcing
        if self.F_ext is not None:
            for d in range(self.dim):
                u[d] += 0.5 * self.F_ext[d] / np.where(
                    np.abs(rho) < 1e-30, 1e-30, rho
                )

        self._step_count += 1
        return LBMState(f=f, rho=rho, u=u)

    def step_n(self, state: LBMState, n_steps: int) -> LBMState:
        """Advance *n_steps* time steps."""
        for _ in range(n_steps):
            state = self.step(state)
        return state

    @property
    def viscosity(self) -> float:
        """Kinematic viscosity in lattice units."""
        return self.nu

    @property
    def steps(self) -> int:
        """Number of time steps completed."""
        return self._step_count

    def reynolds_number(self, U: float, L: float) -> float:
        """Compute Re = U L / ν in lattice units."""
        return U * L / self.nu

    def drag_coefficient(
        self,
        state: LBMState,
        obstacle_mask: NDArray,
        U_inf: float,
        D: float,
    ) -> Tuple[float, float]:
        """
        Compute drag and lift coefficients via momentum-exchange method.

        Args:
            state: Current LBM state.
            obstacle_mask: Boolean mask of obstacle nodes.
            U_inf: Free-stream velocity.
            D: Characteristic length.

        Returns:
            ``(C_D, C_L)`` drag and lift coefficients.
        """
        opp = _opposite_map(self.c)
        Fx = 0.0
        Fy = 0.0

        boundary_nodes = self._boundary_nodes(obstacle_mask)
        for idx in boundary_nodes:
            for i in range(self.Q):
                j = opp[i]
                nb = tuple(int(idx[d] + self.c[i, d]) for d in range(self.dim))
                # Check if neighbour is fluid
                try:
                    if not obstacle_mask[nb]:
                        Fx += self.c[i, 0] * (state.f[i][idx] + state.f[j][nb])
                        if self.dim >= 2:
                            Fy += self.c[i, 1] * (state.f[i][idx] + state.f[j][nb])
                except IndexError:
                    continue

        denom = 0.5 * state.rho.mean() * U_inf ** 2 * D
        if abs(denom) < 1e-30:
            return 0.0, 0.0
        C_D = Fx / denom
        C_L = Fy / denom
        return C_D, C_L

    def _boundary_nodes(self, mask: NDArray) -> list:
        """Return list of solid-node index tuples adjacent to fluid."""
        from itertools import product as cart
        indices = list(zip(*np.where(mask)))
        boundary = []
        for idx in indices:
            for i in range(self.Q):
                nb = tuple(int(idx[d] + self.c[i, d]) for d in range(self.dim))
                try:
                    if not mask[nb]:
                        boundary.append(idx)
                        break
                except IndexError:
                    continue
        return boundary
