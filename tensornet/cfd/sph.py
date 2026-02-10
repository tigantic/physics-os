"""
Smoothed Particle Hydrodynamics (SPH)
======================================

Lagrangian meshfree method for continuum mechanics where the domain
is discretised by a set of particles carrying field quantities.

Governing Equations (weakly compressible):
    Dρ_a/Dt = -ρ_a Σ_b m_b/ρ_b (v_b - v_a)·∇_a W_{ab}

    Dv_a/Dt = -Σ_b m_b (p_a/ρ_a² + p_b/ρ_b² + Π_{ab}) ∇_a W_{ab} + g

    De_a/Dt = (p_a/ρ_a²) Σ_b m_b (v_a - v_b)·∇_a W_{ab}

Kernels:
    - Cubic spline (Monaghan & Lattanzio 1985)
    - Wendland C2 / C4 (Wendland 1995)
    - Quintic spline (Morris 1996)

Equation of State (Tait):
    p = B [(ρ/ρ₀)^γ - 1], B = c₀² ρ₀ / γ

References:
    [1] Monaghan, "Smoothed Particle Hydrodynamics",
        Annu. Rev. Astron. Astrophys. 30, 1992.
    [2] Price, "Smoothed Particle Hydrodynamics and
        Magnetohydrodynamics", J. Comp. Phys. 231, 2012.
    [3] Wendland, "Piecewise polynomial, positive definite and
        compactly supported radial functions of minimal degree",
        Adv. Comput. Math. 4, 1995.

Domain II.1 — Computational Fluid Dynamics / Smoothed Particle Hydrodynamics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Kernel Functions
# ---------------------------------------------------------------------------

class KernelType(Enum):
    """SPH smoothing kernel variants."""
    CUBIC_SPLINE = "cubic-spline"
    WENDLAND_C2 = "wendland-c2"
    WENDLAND_C4 = "wendland-c4"
    QUINTIC_SPLINE = "quintic-spline"


def _sigma(dim: int, kernel: KernelType) -> float:
    """Normalisation constant for kernel in *dim* dimensions."""
    if kernel == KernelType.CUBIC_SPLINE:
        return {1: 2.0 / 3.0, 2: 10.0 / (7.0 * math.pi),
                3: 1.0 / math.pi}[dim]
    if kernel == KernelType.WENDLAND_C2:
        return {1: 5.0 / 8.0, 2: 7.0 / (4.0 * math.pi),
                3: 21.0 / (16.0 * math.pi)}[dim]
    if kernel == KernelType.WENDLAND_C4:
        return {2: 9.0 / (4.0 * math.pi),
                3: 495.0 / (256.0 * math.pi)}.get(dim, 1.0)
    if kernel == KernelType.QUINTIC_SPLINE:
        return {1: 1.0 / 120.0, 2: 7.0 / (478.0 * math.pi),
                3: 1.0 / (120.0 * math.pi)}[dim]
    raise ValueError(f"Unknown kernel {kernel}")


def kernel_eval(
    r: NDArray,
    h: float,
    dim: int,
    kernel: KernelType = KernelType.CUBIC_SPLINE,
) -> NDArray:
    r"""
    Evaluate SPH kernel :math:`W(r, h)`.

    Args:
        r: Distances (≥ 0).
        h: Smoothing length.
        dim: Spatial dimension (1, 2, or 3).
        kernel: Kernel type.

    Returns:
        Kernel values.
    """
    q = np.asarray(r, dtype=np.float64) / h
    sigma = _sigma(dim, kernel) / (h ** dim)

    if kernel == KernelType.CUBIC_SPLINE:
        W = np.where(
            q < 1.0,
            sigma * (1.0 - 1.5 * q ** 2 + 0.75 * q ** 3),
            np.where(q < 2.0,
                     sigma * 0.25 * (2.0 - q) ** 3,
                     0.0),
        )
    elif kernel == KernelType.WENDLAND_C2:
        W = np.where(q < 2.0,
                     sigma * (1.0 - 0.5 * q) ** 4 * (2.0 * q + 1.0),
                     0.0)
    elif kernel == KernelType.WENDLAND_C4:
        W = np.where(
            q < 2.0,
            sigma * (1.0 - 0.5 * q) ** 6 * (35.0 / 12.0 * q ** 2 + 3.0 * q + 1.0),
            0.0,
        )
    elif kernel == KernelType.QUINTIC_SPLINE:
        t1 = np.maximum(3.0 - q, 0.0) ** 5
        t2 = np.maximum(2.0 - q, 0.0) ** 5
        t3 = np.maximum(1.0 - q, 0.0) ** 5
        W = sigma * (t1 - 6.0 * t2 + 15.0 * t3)
    else:
        raise ValueError(f"Unknown kernel {kernel}")
    return W


def kernel_gradient(
    r: NDArray,
    h: float,
    dim: int,
    kernel: KernelType = KernelType.CUBIC_SPLINE,
) -> NDArray:
    r"""
    Evaluate :math:`dW/dr` (radial derivative of the kernel).

    Used to construct the full gradient:
    :math:`\nabla_a W_{ab} = (dW/dr)(r_{ab}/|r_{ab}|)`.
    """
    q = np.asarray(r, dtype=np.float64) / h
    sigma = _sigma(dim, kernel) / (h ** (dim + 1))

    if kernel == KernelType.CUBIC_SPLINE:
        dW = np.where(
            q < 1.0,
            sigma * (-3.0 * q + 2.25 * q ** 2),
            np.where(q < 2.0,
                     sigma * (-0.75 * (2.0 - q) ** 2),
                     0.0),
        )
    elif kernel == KernelType.WENDLAND_C2:
        dW = np.where(
            q < 2.0,
            sigma * (-5.0 * q * (1.0 - 0.5 * q) ** 3),
            0.0,
        )
    elif kernel == KernelType.WENDLAND_C4:
        dW = np.where(
            q < 2.0,
            sigma * (
                -14.0 / 3.0 * q * (1.0 - 0.5 * q) ** 5
                * (4.0 * q + 1.0)
            ),
            0.0,
        )
    elif kernel == KernelType.QUINTIC_SPLINE:
        t1 = -5.0 * np.maximum(3.0 - q, 0.0) ** 4
        t2 = 30.0 * np.maximum(2.0 - q, 0.0) ** 4
        t3 = -75.0 * np.maximum(1.0 - q, 0.0) ** 4
        dW = sigma * (t1 + t2 + t3)
    else:
        raise ValueError(f"Unknown kernel {kernel}")
    return dW


# ---------------------------------------------------------------------------
# Equation of State
# ---------------------------------------------------------------------------

@dataclass
class TaitEOS:
    """
    Tait equation of state for weakly compressible SPH.

    p = B [(ρ/ρ₀)^γ - 1]  with  B = c₀² ρ₀ / γ.
    """
    rho0: float = 1000.0
    c0: float = 40.0
    gamma: float = 7.0

    @property
    def B(self) -> float:
        return self.c0 ** 2 * self.rho0 / self.gamma

    def pressure(self, rho: NDArray) -> NDArray:
        return self.B * ((rho / self.rho0) ** self.gamma - 1.0)

    def sound_speed(self, rho: NDArray) -> NDArray:
        return self.c0 * (rho / self.rho0) ** ((self.gamma - 1.0) / 2.0)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class SPHState:
    """Particle state for SPH simulation.

    Attributes:
        x: Positions ``(N, dim)``.
        v: Velocities ``(N, dim)``.
        rho: Densities ``(N,)``.
        m: Masses ``(N,)``.
        p: Pressures ``(N,)``.
        e: Internal energy ``(N,)`` (optional).
        h: Smoothing lengths ``(N,)`` or scalar.
    """
    x: NDArray
    v: NDArray
    rho: NDArray
    m: NDArray
    p: NDArray
    e: Optional[NDArray] = None
    h_arr: Optional[NDArray] = None

    @property
    def n_particles(self) -> int:
        return self.x.shape[0]

    @property
    def dim(self) -> int:
        return self.x.shape[1]


# ---------------------------------------------------------------------------
# Neighbour search
# ---------------------------------------------------------------------------

class CellList:
    """
    Linked-cell-list neighbour search.

    Builds a uniform grid of cell size ≥ 2h and bins particles for
    O(N) neighbour queries.
    """

    def __init__(self, domain_min: NDArray, domain_max: NDArray, cell_size: float) -> None:
        self.domain_min = np.asarray(domain_min, dtype=np.float64)
        self.domain_max = np.asarray(domain_max, dtype=np.float64)
        self.cell_size = cell_size
        self.dim = len(domain_min)

        extent = self.domain_max - self.domain_min
        self.n_cells_per_dim = np.maximum(
            (extent / cell_size).astype(np.intp), 1
        )
        self.total_cells = int(np.prod(self.n_cells_per_dim))
        self._heads: NDArray = np.full(self.total_cells, -1, dtype=np.intp)
        self._nexts: NDArray = np.array([], dtype=np.intp)

    def build(self, x: NDArray) -> None:
        """Bin *N* particles into cells."""
        N = x.shape[0]
        self._heads[:] = -1
        self._nexts = np.full(N, -1, dtype=np.intp)

        for i in range(N):
            ci = self._cell_index(x[i])
            self._nexts[i] = self._heads[ci]
            self._heads[ci] = i

    def _cell_index(self, pos: NDArray) -> int:
        rel = np.clip(pos - self.domain_min, 0, None)
        idx = np.minimum(
            (rel / self.cell_size).astype(np.intp),
            self.n_cells_per_dim - 1,
        )
        flat = 0
        stride = 1
        for d in range(self.dim - 1, -1, -1):
            flat += int(idx[d]) * stride
            stride *= int(self.n_cells_per_dim[d])
        return flat

    def neighbours(self, pos: NDArray, x: NDArray, cutoff: float) -> list[int]:
        """Return indices of particles within *cutoff* of *pos*."""
        rel = np.clip(pos - self.domain_min, 0, None)
        center = np.minimum(
            (rel / self.cell_size).astype(np.intp),
            self.n_cells_per_dim - 1,
        )
        nbrs: list[int] = []
        cutoff2 = cutoff * cutoff

        # Visit neighbouring cells
        offsets = self._stencil()
        for offset in offsets:
            cell_idx = center + offset
            if np.any(cell_idx < 0) or np.any(cell_idx >= self.n_cells_per_dim):
                continue
            flat = 0
            stride = 1
            for d in range(self.dim - 1, -1, -1):
                flat += int(cell_idx[d]) * stride
                stride *= int(self.n_cells_per_dim[d])

            j = self._heads[flat]
            while j >= 0:
                dx = x[j] - pos
                r2 = np.dot(dx, dx)
                if r2 < cutoff2:
                    nbrs.append(j)
                j = self._nexts[j]
        return nbrs

    def _stencil(self) -> list[NDArray]:
        """3^dim neighbour stencil offsets."""
        from itertools import product
        ranges = [range(-1, 2)] * self.dim
        return [np.array(o, dtype=np.intp) for o in product(*ranges)]


# ---------------------------------------------------------------------------
# Artificial viscosity
# ---------------------------------------------------------------------------

def artificial_viscosity(
    rho_a: float, rho_b: float,
    x_ab: NDArray, v_ab: NDArray,
    h: float, c_a: float, c_b: float,
    alpha: float = 1.0, beta: float = 2.0,
) -> float:
    r"""
    Monaghan artificial viscosity:

    .. math::
        \Pi_{ab} = \begin{cases}
            \frac{-\alpha \bar c \mu_{ab} + \beta \mu_{ab}^2}{\bar\rho}
                & \text{if } v_{ab}\cdot x_{ab} < 0 \\
            0   & \text{otherwise}
        \end{cases}

    where :math:`\mu_{ab} = h\, v_{ab}\cdot x_{ab} /
    (|x_{ab}|^2 + \eta^2)`.
    """
    vx = np.dot(v_ab, x_ab)
    if vx >= 0.0:
        return 0.0

    r2 = np.dot(x_ab, x_ab)
    eta2 = (0.01 * h) ** 2
    mu = h * vx / (r2 + eta2)
    rho_bar = 0.5 * (rho_a + rho_b)
    c_bar = 0.5 * (c_a + c_b)

    return (-alpha * c_bar * mu + beta * mu ** 2) / rho_bar


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class SPHSolver:
    r"""
    Weakly Compressible SPH (WCSPH) solver.

    Implements the standard WCSPH formulation with:
    - Tait equation of state
    - Monaghan artificial viscosity
    - Leapfrog (Kick-Drift-Kick) time integration
    - Linked-cell-list neighbour search
    - Variable smoothing lengths (optional)

    Parameters:
        dim: Spatial dimensions (2 or 3).
        eos: Equation of state.
        kernel: SPH kernel type.
        h: Smoothing length (uniform).
        domain_min: Lower corner of the domain.
        domain_max: Upper corner of the domain.
        alpha_visc: Artificial viscosity parameter α.
        beta_visc: Artificial viscosity parameter β.
        gravity: Gravity vector ``(dim,)``.

    Example::

        solver = SPHSolver(dim=2, h=0.02,
                           domain_min=np.array([0.0, 0.0]),
                           domain_max=np.array([1.0, 1.0]))
        state = solver.create_state(x0, v0, rho0, m0)
        for _ in range(500):
            state = solver.step(state, dt=1e-4)
    """

    def __init__(
        self,
        dim: int = 2,
        eos: Optional[TaitEOS] = None,
        kernel: KernelType = KernelType.CUBIC_SPLINE,
        h: float = 0.02,
        domain_min: Optional[NDArray] = None,
        domain_max: Optional[NDArray] = None,
        alpha_visc: float = 1.0,
        beta_visc: float = 2.0,
        gravity: Optional[NDArray] = None,
    ) -> None:
        self.dim = dim
        self.eos = eos or TaitEOS()
        self.kernel = kernel
        self.h = h
        self.alpha_visc = alpha_visc
        self.beta_visc = beta_visc
        self.cutoff = 2.0 * h  # Kernel support

        if domain_min is None:
            domain_min = np.zeros(dim)
        if domain_max is None:
            domain_max = np.ones(dim)
        self.domain_min = np.asarray(domain_min, dtype=np.float64)
        self.domain_max = np.asarray(domain_max, dtype=np.float64)

        self.gravity = (
            np.asarray(gravity, dtype=np.float64)
            if gravity is not None
            else np.zeros(dim, dtype=np.float64)
        )

        self._cells = CellList(self.domain_min, self.domain_max, self.cutoff)
        self._step_count = 0

    def create_state(
        self,
        x: NDArray,
        v: NDArray,
        rho: NDArray,
        m: NDArray,
    ) -> SPHState:
        """Build initial SPHState from arrays."""
        p = self.eos.pressure(rho)
        return SPHState(
            x=x.copy().astype(np.float64),
            v=v.copy().astype(np.float64),
            rho=rho.copy().astype(np.float64),
            m=m.copy().astype(np.float64),
            p=p,
        )

    def compute_density(self, state: SPHState) -> NDArray:
        """Recompute density from particle distribution (summation density)."""
        N = state.n_particles
        rho = np.zeros(N, dtype=np.float64)
        self._cells.build(state.x)

        for a in range(N):
            nbrs = self._cells.neighbours(state.x[a], state.x, self.cutoff)
            for b in nbrs:
                r_ab = np.linalg.norm(state.x[a] - state.x[b])
                W = kernel_eval(r_ab, self.h, self.dim, self.kernel)
                rho[a] += state.m[b] * float(W)
        return rho

    def compute_acceleration(self, state: SPHState) -> NDArray:
        """Compute acceleration for all particles (momentum equation)."""
        N = state.n_particles
        acc = np.zeros((N, self.dim), dtype=np.float64)
        cs = self.eos.sound_speed(state.rho)
        self._cells.build(state.x)

        for a in range(N):
            nbrs = self._cells.neighbours(state.x[a], state.x, self.cutoff)
            for b in nbrs:
                if a == b:
                    continue
                x_ab = state.x[a] - state.x[b]
                v_ab = state.v[a] - state.v[b]
                r_ab = np.linalg.norm(x_ab)
                if r_ab < 1e-30:
                    continue

                dW = float(kernel_gradient(r_ab, self.h, self.dim, self.kernel))
                e_ab = x_ab / r_ab

                # Pressure gradient
                p_term = state.p[a] / (state.rho[a] ** 2) + state.p[b] / (state.rho[b] ** 2)

                # Artificial viscosity
                Pi_ab = artificial_viscosity(
                    state.rho[a], state.rho[b],
                    x_ab, v_ab, self.h,
                    cs[a], cs[b],
                    self.alpha_visc, self.beta_visc,
                )

                acc[a] -= state.m[b] * (p_term + Pi_ab) * dW * e_ab

            # Gravity
            acc[a] += self.gravity
        return acc

    def step(self, state: SPHState, dt: float) -> SPHState:
        """Advance one time step via Kick-Drift-Kick leapfrog."""
        # Kick (half)
        acc = self.compute_acceleration(state)
        v_half = state.v + 0.5 * dt * acc

        # Drift
        x_new = state.x + dt * v_half

        # Reflective boundary clamp
        for d in range(self.dim):
            mask_lo = x_new[:, d] < self.domain_min[d]
            mask_hi = x_new[:, d] > self.domain_max[d]
            x_new[mask_lo, d] = 2.0 * self.domain_min[d] - x_new[mask_lo, d]
            x_new[mask_hi, d] = 2.0 * self.domain_max[d] - x_new[mask_hi, d]
            v_half[mask_lo, d] *= -1.0
            v_half[mask_hi, d] *= -1.0

        # Update state for second kick
        rho_new = self.compute_density(
            SPHState(x=x_new, v=v_half, rho=state.rho, m=state.m, p=state.p)
        )
        p_new = self.eos.pressure(rho_new)

        state_mid = SPHState(x=x_new, v=v_half, rho=rho_new, m=state.m, p=p_new)
        acc_new = self.compute_acceleration(state_mid)

        # Kick (second half)
        v_new = v_half + 0.5 * dt * acc_new

        self._step_count += 1
        return SPHState(x=x_new, v=v_new, rho=rho_new, m=state.m, p=p_new)

    def step_n(self, state: SPHState, n_steps: int, dt: float) -> SPHState:
        """Advance *n_steps* time steps."""
        for _ in range(n_steps):
            state = self.step(state, dt)
        return state

    @property
    def steps(self) -> int:
        return self._step_count

    def adaptive_dt(self, state: SPHState, cfl: float = 0.25) -> float:
        r"""
        CFL-based adaptive time step.

        .. math::
            \Delta t = C_{\text{CFL}} \min_a \frac{h}{c_a + |v_a|}
        """
        cs = self.eos.sound_speed(state.rho)
        v_mag = np.linalg.norm(state.v, axis=1)
        dt = cfl * self.h / np.max(cs + v_mag + 1e-30)
        return float(dt)

    def kinetic_energy(self, state: SPHState) -> float:
        """Total kinetic energy."""
        return 0.5 * float(np.sum(state.m * np.sum(state.v ** 2, axis=1)))

    def potential_energy(self, state: SPHState) -> float:
        """Gravitational potential energy (relative to domain_min)."""
        g_mag = np.linalg.norm(self.gravity)
        if g_mag < 1e-30:
            return 0.0
        g_dir = self.gravity / g_mag
        heights = np.sum((state.x - self.domain_min) * g_dir, axis=1)
        return -float(np.sum(state.m * g_mag * heights))
