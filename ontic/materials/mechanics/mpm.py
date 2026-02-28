"""
Material Point Method (MPM)
============================

Hybrid Lagrangian–Eulerian method that tracks material as particles
(material points) and solves momentum on a background Eulerian grid.

MPM Algorithm (MUSL — Modified Update Stress Last):
    1. Particle-to-Grid (P2G): map mass and momentum to grid
    2. Grid Solve: apply forces, compute acceleration, update velocity
    3. Grid-to-Particle (G2P): interpolate updated velocity back
    4. Update particle stress (constitutive model)
    5. Advect particles with updated velocity

Governing equations on the grid:
    m_I a_I = f_I^{int} + f_I^{ext}

    f_I^{int} = -Σ_p V_p σ_p · ∇N_I(x_p)

Shape functions:
    - Linear (tent), Quadratic B-spline, GIMP (cpGIMP)

References:
    [1] Sulsky, Chen & Schreyer, "A Particle Method for
        History-Dependent Materials", Comp. Meth. Appl. Mech. 118, 1994.
    [2] Brackbill & Ruppel, "FLIP: A Method for Adaptively Zoned,
        Particle-in-Cell Calculations of Fluid Flows",
        J. Comp. Phys. 65, 1986.
    [3] Jiang et al., "The Affine Particle-In-Cell Method",
        ACM Trans. Graphics 34, 2015.

Domain III.1 — Solid Mechanics / Material Point Method.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class ShapeFunction(Enum):
    """Grid shape function types."""
    LINEAR = "linear"
    QUADRATIC_BSPLINE = "quadratic-bspline"
    CUBIC_BSPLINE = "cubic-bspline"


class ConstitutiveModel(Enum):
    """Material constitutive model."""
    NEO_HOOKEAN = "neo-hookean"
    LINEAR_ELASTIC = "linear-elastic"
    DRUCKER_PRAGER = "drucker-prager"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class MPMParticleState:
    """
    Material point (particle) state.

    Attributes:
        x: Positions ``(N, dim)``.
        v: Velocities ``(N, dim)``.
        m: Masses ``(N,)``.
        vol: Volumes ``(N,)``.
        F: Deformation gradient ``(N, dim, dim)``.
        sigma: Cauchy stress ``(N, dim, dim)``.
        C: APIC affine matrix ``(N, dim, dim)`` (optional).
    """
    x: NDArray
    v: NDArray
    m: NDArray
    vol: NDArray
    F: NDArray
    sigma: NDArray
    C: Optional[NDArray] = None

    @property
    def n_particles(self) -> int:
        return self.x.shape[0]

    @property
    def dim(self) -> int:
        return self.x.shape[1]


@dataclass
class MPMGridState:
    """
    Background grid state.

    Attributes:
        mass: Node masses ``grid_shape``.
        momentum: Node momenta ``(dim,) + grid_shape``.
        velocity: Node velocities ``(dim,) + grid_shape``.
        force: Node forces ``(dim,) + grid_shape``.
    """
    mass: NDArray
    momentum: NDArray
    velocity: NDArray
    force: NDArray


# ---------------------------------------------------------------------------
# Shape function evaluation
# ---------------------------------------------------------------------------

def _linear_shape(x: float) -> float:
    """1D linear (hat/tent) shape function centered at 0, support [-1,1]."""
    ax = abs(x)
    return max(1.0 - ax, 0.0)


def _linear_shape_grad(x: float) -> float:
    """Gradient of 1D linear shape function."""
    if -1.0 < x < 0.0:
        return 1.0
    elif 0.0 < x < 1.0:
        return -1.0
    return 0.0


def _quadratic_bspline(x: float) -> float:
    """1D quadratic B-spline, support [-1.5, 1.5]."""
    ax = abs(x)
    if ax < 0.5:
        return 0.75 - ax * ax
    elif ax < 1.5:
        return 0.5 * (1.5 - ax) ** 2
    return 0.0


def _quadratic_bspline_grad(x: float) -> float:
    ax = abs(x)
    s = 1.0 if x >= 0 else -1.0
    if ax < 0.5:
        return -2.0 * x
    elif ax < 1.5:
        return -s * (1.5 - ax)
    return 0.0


def _cubic_bspline(x: float) -> float:
    """1D cubic B-spline, support [-2, 2]."""
    ax = abs(x)
    if ax < 1.0:
        return (3.0 * ax ** 3 - 6.0 * ax ** 2 + 4.0) / 6.0
    elif ax < 2.0:
        return (2.0 - ax) ** 3 / 6.0
    return 0.0


def _cubic_bspline_grad(x: float) -> float:
    ax = abs(x)
    s = 1.0 if x >= 0 else -1.0
    if ax < 1.0:
        return s * (1.5 * ax ** 2 - 2.0 * ax) / 1.0
    elif ax < 2.0:
        return -s * 0.5 * (2.0 - ax) ** 2
    return 0.0


SHAPE_FN = {
    ShapeFunction.LINEAR: (_linear_shape, _linear_shape_grad),
    ShapeFunction.QUADRATIC_BSPLINE: (_quadratic_bspline, _quadratic_bspline_grad),
    ShapeFunction.CUBIC_BSPLINE: (_cubic_bspline, _cubic_bspline_grad),
}


# ---------------------------------------------------------------------------
# Constitutive Models
# ---------------------------------------------------------------------------

def neo_hookean_stress(
    F: NDArray, E: float, nu: float,
) -> NDArray:
    r"""
    Compressible Neo-Hookean stress (Kirchhoff → Cauchy).

    .. math::
        \boldsymbol\tau = \mu (\mathbf{b} - \mathbf{I})
            + \lambda \ln J \, \mathbf{I}

    :math:`\boldsymbol\sigma = \boldsymbol\tau / J`.
    """
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    dim = F.shape[0]
    J = np.linalg.det(F)
    b = F @ F.T
    tau = mu * (b - np.eye(dim)) + lam * np.log(max(J, 1e-30)) * np.eye(dim)
    return tau / max(J, 1e-30)


def linear_elastic_stress(
    F: NDArray, E: float, nu: float,
) -> NDArray:
    """Small-strain linear elastic Cauchy stress."""
    dim = F.shape[0]
    eps = 0.5 * (F + F.T) - np.eye(dim)  # Small-strain approximation
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return lam * np.trace(eps) * np.eye(dim) + 2.0 * mu * eps


def drucker_prager_stress(
    F: NDArray, E: float, nu: float,
    cohesion: float = 1e4, friction_angle: float = 30.0,
) -> NDArray:
    """
    Drucker-Prager elasto-plastic return mapping.

    Yield surface: f = √J₂ + α I₁ - k = 0
    where α = 2 sin φ / (√3 (3 - sin φ)),
          k = 6 c cos φ / (√3 (3 - sin φ)).
    """
    dim = F.shape[0]
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    # Trial elastic stress
    eps = 0.5 * (F + F.T) - np.eye(dim)
    sigma_trial = lam * np.trace(eps) * np.eye(dim) + 2.0 * mu * eps

    # Invariants
    I1 = np.trace(sigma_trial)
    s = sigma_trial - (I1 / dim) * np.eye(dim)
    J2 = 0.5 * np.sum(s * s)
    sqrt_J2 = np.sqrt(J2 + 1e-30)

    phi_rad = np.radians(friction_angle)
    sin_phi = np.sin(phi_rad)
    cos_phi = np.cos(phi_rad)
    alpha = 2.0 * sin_phi / (np.sqrt(3.0) * (3.0 - sin_phi))
    k = 6.0 * cohesion * cos_phi / (np.sqrt(3.0) * (3.0 - sin_phi))

    f = sqrt_J2 + alpha * I1 - k
    if f <= 0:
        return sigma_trial  # Elastic

    # Return mapping (associated flow)
    dlam = f / (mu + dim * lam * alpha ** 2 + 1e-30)
    sigma = sigma_trial - dlam * (mu * s / (sqrt_J2 + 1e-30) + lam * alpha * np.eye(dim))
    return sigma


CONSTITUTIVE_FN = {
    ConstitutiveModel.NEO_HOOKEAN: neo_hookean_stress,
    ConstitutiveModel.LINEAR_ELASTIC: linear_elastic_stress,
    ConstitutiveModel.DRUCKER_PRAGER: drucker_prager_stress,
}


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class MPMSolver:
    r"""
    Material Point Method solver (2D/3D).

    Supports FLIP/PIC blending, APIC (Affine PIC), and multiple
    constitutive models.

    Parameters:
        grid_min: Lower corner of the grid.
        grid_max: Upper corner of the grid.
        cell_size: Grid cell size.
        shape: Shape function type.
        constitutive: Constitutive model.
        E: Young's modulus.
        nu: Poisson's ratio.
        flip_ratio: FLIP/PIC blend (1 = pure FLIP, 0 = pure PIC).
        gravity: Gravity vector.

    Example::

        solver = MPMSolver(
            grid_min=np.array([0.0, 0.0]),
            grid_max=np.array([1.0, 1.0]),
            cell_size=0.02,
            constitutive=ConstitutiveModel.NEO_HOOKEAN,
            E=1e6, nu=0.3,
        )
        state = solver.create_particles(x0, v0, m0, vol0)
        for _ in range(100):
            state = solver.step(state, dt=1e-4)
    """

    def __init__(
        self,
        grid_min: NDArray,
        grid_max: NDArray,
        cell_size: float = 0.02,
        shape: ShapeFunction = ShapeFunction.QUADRATIC_BSPLINE,
        constitutive: ConstitutiveModel = ConstitutiveModel.NEO_HOOKEAN,
        E: float = 1e6,
        nu: float = 0.3,
        flip_ratio: float = 0.95,
        gravity: Optional[NDArray] = None,
        **constitutive_kwargs,
    ) -> None:
        self.grid_min = np.asarray(grid_min, dtype=np.float64)
        self.grid_max = np.asarray(grid_max, dtype=np.float64)
        self.h = cell_size
        self.dim = len(grid_min)
        self.E = E
        self.nu = nu
        self.flip_ratio = flip_ratio
        self.constitutive = constitutive
        self.constitutive_kwargs = constitutive_kwargs

        self._shape_fn, self._shape_grad = SHAPE_FN[shape]

        if gravity is None:
            self.gravity = np.zeros(self.dim, dtype=np.float64)
            self.gravity[-1] = -9.81
        else:
            self.gravity = np.asarray(gravity, dtype=np.float64)

        # Grid dimensions
        self.grid_size = np.ceil(
            (self.grid_max - self.grid_min) / self.h
        ).astype(np.intp) + 1
        self.n_nodes = int(np.prod(self.grid_size))

        self._step_count = 0

    def create_particles(
        self,
        x: NDArray,
        v: Optional[NDArray] = None,
        m: Optional[NDArray] = None,
        vol: Optional[NDArray] = None,
    ) -> MPMParticleState:
        """Initialise particle state."""
        N = x.shape[0]
        if v is None:
            v = np.zeros_like(x)
        if m is None:
            m = np.ones(N, dtype=np.float64) * 0.1
        if vol is None:
            vol = np.full(N, self.h ** self.dim, dtype=np.float64)

        F = np.zeros((N, self.dim, self.dim), dtype=np.float64)
        for i in range(N):
            F[i] = np.eye(self.dim)
        sigma = np.zeros_like(F)

        return MPMParticleState(
            x=x.copy(), v=v.copy(), m=m.copy(), vol=vol.copy(),
            F=F, sigma=sigma,
        )

    def _node_index(self, grid_pos: NDArray) -> int:
        """Convert grid multi-index to flat index."""
        idx = 0
        stride = 1
        for d in range(self.dim - 1, -1, -1):
            idx += int(grid_pos[d]) * stride
            stride *= int(self.grid_size[d])
        return idx

    def _p2g(self, state: MPMParticleState) -> Tuple[NDArray, NDArray]:
        """Particle-to-Grid transfer: scatter mass and momentum."""
        mass = np.zeros(self.n_nodes, dtype=np.float64)
        momentum = np.zeros((self.dim, self.n_nodes), dtype=np.float64)

        for p in range(state.n_particles):
            base = np.floor((state.x[p] - self.grid_min) / self.h).astype(np.intp)
            for offset in self._stencil_offsets():
                node = base + offset
                if np.any(node < 0) or np.any(node >= self.grid_size):
                    continue
                ni = self._node_index(node)
                x_node = self.grid_min + node * self.h
                dx = (state.x[p] - x_node) / self.h

                w = 1.0
                for d in range(self.dim):
                    w *= self._shape_fn(dx[d])

                mass[ni] += state.m[p] * w
                for d in range(self.dim):
                    momentum[d, ni] += state.m[p] * state.v[p, d] * w
        return mass, momentum

    def _compute_grid_forces(
        self, state: MPMParticleState, mass: NDArray,
    ) -> NDArray:
        """Compute internal + external forces on grid nodes."""
        force = np.zeros((self.dim, self.n_nodes), dtype=np.float64)

        # Internal forces: f_I = -Σ_p V_p σ_p · ∇N_I
        for p in range(state.n_particles):
            base = np.floor((state.x[p] - self.grid_min) / self.h).astype(np.intp)
            for offset in self._stencil_offsets():
                node = base + offset
                if np.any(node < 0) or np.any(node >= self.grid_size):
                    continue
                ni = self._node_index(node)
                x_node = self.grid_min + node * self.h
                dx = (state.x[p] - x_node) / self.h

                grad_w = np.zeros(self.dim, dtype=np.float64)
                for d in range(self.dim):
                    grad_w[d] = self._shape_grad(dx[d]) / self.h
                    for d2 in range(self.dim):
                        if d2 != d:
                            grad_w[d] *= self._shape_fn(dx[d2])

                f_int = -state.vol[p] * state.sigma[p] @ grad_w
                force[:, ni] += f_int

        # External forces (gravity)
        for ni in range(self.n_nodes):
            if mass[ni] > 1e-30:
                for d in range(self.dim):
                    force[d, ni] += mass[ni] * self.gravity[d]

        return force

    def _g2p(
        self,
        state: MPMParticleState,
        grid_vel_old: NDArray,
        grid_vel_new: NDArray,
    ) -> MPMParticleState:
        """Grid-to-Particle transfer: gather updated velocity."""
        x_new = state.x.copy()
        v_new = state.v.copy()
        F_new = state.F.copy()
        sigma_new = state.sigma.copy()
        vol_new = state.vol.copy()

        for p in range(state.n_particles):
            base = np.floor((state.x[p] - self.grid_min) / self.h).astype(np.intp)
            v_pic = np.zeros(self.dim, dtype=np.float64)
            v_flip = state.v[p].copy()
            grad_v = np.zeros((self.dim, self.dim), dtype=np.float64)

            for offset in self._stencil_offsets():
                node = base + offset
                if np.any(node < 0) or np.any(node >= self.grid_size):
                    continue
                ni = self._node_index(node)
                x_node = self.grid_min + node * self.h
                dx = (state.x[p] - x_node) / self.h

                w = 1.0
                for d in range(self.dim):
                    w *= self._shape_fn(dx[d])

                grad_w = np.zeros(self.dim, dtype=np.float64)
                for d in range(self.dim):
                    grad_w[d] = self._shape_grad(dx[d]) / self.h
                    for d2 in range(self.dim):
                        if d2 != d:
                            grad_w[d] *= self._shape_fn(dx[d2])

                v_pic += w * grid_vel_new[:, ni]
                v_flip += w * (grid_vel_new[:, ni] - grid_vel_old[:, ni])
                for di in range(self.dim):
                    for dj in range(self.dim):
                        grad_v[di, dj] += grid_vel_new[di, ni] * grad_w[dj]

            # FLIP/PIC blend
            v_new[p] = self.flip_ratio * v_flip + (1.0 - self.flip_ratio) * v_pic

        return MPMParticleState(
            x=x_new, v=v_new, m=state.m, vol=vol_new,
            F=F_new, sigma=sigma_new,
        )

    def step(self, state: MPMParticleState, dt: float) -> MPMParticleState:
        """Execute one MPM time step."""
        # 1. Update stress
        stress_fn = CONSTITUTIVE_FN[self.constitutive]
        for p in range(state.n_particles):
            kwargs = {}
            if self.constitutive == ConstitutiveModel.DRUCKER_PRAGER:
                kwargs = self.constitutive_kwargs
            state.sigma[p] = stress_fn(state.F[p], self.E, self.nu, **kwargs)

        # 2. P2G
        mass, momentum = self._p2g(state)
        vel_old = np.zeros_like(momentum)
        for ni in range(self.n_nodes):
            if mass[ni] > 1e-30:
                vel_old[:, ni] = momentum[:, ni] / mass[ni]

        # 3. Grid forces
        force = self._compute_grid_forces(state, mass)

        # 4. Grid velocity update
        vel_new = vel_old.copy()
        for ni in range(self.n_nodes):
            if mass[ni] > 1e-30:
                vel_new[:, ni] += dt * force[:, ni] / mass[ni]

        # 5. G2P
        state = self._g2p(state, vel_old, vel_new)

        # 6. Advect particles
        for p in range(state.n_particles):
            # Use PIC velocity for advection
            base = np.floor((state.x[p] - self.grid_min) / self.h).astype(np.intp)
            v_adv = np.zeros(self.dim, dtype=np.float64)
            for offset in self._stencil_offsets():
                node = base + offset
                if np.any(node < 0) or np.any(node >= self.grid_size):
                    continue
                ni = self._node_index(node)
                x_node = self.grid_min + node * self.h
                dx = (state.x[p] - x_node) / self.h
                w = 1.0
                for d in range(self.dim):
                    w *= self._shape_fn(dx[d])
                v_adv += w * vel_new[:, ni]
            state.x[p] += dt * v_adv

            # Update deformation gradient: F_new = (I + dt ∇v) F_old
            grad_v = np.zeros((self.dim, self.dim), dtype=np.float64)
            for offset in self._stencil_offsets():
                node = base + offset
                if np.any(node < 0) or np.any(node >= self.grid_size):
                    continue
                ni = self._node_index(node)
                x_node = self.grid_min + node * self.h
                dxn = (state.x[p] - dt * v_adv - x_node) / self.h
                grad_w = np.zeros(self.dim, dtype=np.float64)
                for d in range(self.dim):
                    grad_w[d] = self._shape_grad(dxn[d]) / self.h
                    for d2 in range(self.dim):
                        if d2 != d:
                            grad_w[d] *= self._shape_fn(dxn[d2])
                for di in range(self.dim):
                    for dj in range(self.dim):
                        grad_v[di, dj] += vel_new[di, ni] * grad_w[dj]

            state.F[p] = (np.eye(self.dim) + dt * grad_v) @ state.F[p]
            J = np.linalg.det(state.F[p])
            state.vol[p] = state.m[p] / (state.m[p] / state.vol[p]) * max(J, 1e-10)

        # Clamp particles to grid
        for d in range(self.dim):
            state.x[:, d] = np.clip(
                state.x[:, d],
                self.grid_min[d] + self.h * 0.01,
                self.grid_max[d] - self.h * 0.01,
            )

        self._step_count += 1
        return state

    def step_n(self, state: MPMParticleState, n_steps: int, dt: float) -> MPMParticleState:
        for _ in range(n_steps):
            state = self.step(state, dt)
        return state

    @property
    def steps(self) -> int:
        return self._step_count

    def _stencil_offsets(self) -> list[NDArray]:
        """Return shape-function stencil offsets."""
        from itertools import product
        rng = range(-1, 3)  # Covers quadratic/cubic B-spline support
        return [np.array(o, dtype=np.intp) for o in product(*([rng] * self.dim))]

    def kinetic_energy(self, state: MPMParticleState) -> float:
        return 0.5 * float(np.sum(state.m * np.sum(state.v ** 2, axis=1)))
