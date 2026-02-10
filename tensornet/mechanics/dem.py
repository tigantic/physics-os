"""
Discrete Element Method (DEM) for Granular Mechanics
=====================================================

Particle-based simulation of granular media: spherical particles
interacting via contact forces, with Hertzian or linear-spring
contact models, Coulomb friction, and velocity-Verlet integration.

Contact Law (Hertz-Mindlin):
    Normal force:
    .. math::
        F_n = \\frac{4}{3} E^* \\sqrt{R^*}\\, \\delta_n^{3/2}
              - \\gamma_n v_n

    Tangential force:
    .. math::
        F_t = \\min(\\mu F_n,\\; k_t \\xi_t + \\gamma_t v_t)

where:
    - :math:`\\delta_n` is the overlap,
    - :math:`E^* = [(1-\\nu_1^2)/E_1 + (1-\\nu_2^2)/E_2]^{-1}` the
      effective modulus,
    - :math:`R^* = (1/R_1 + 1/R_2)^{-1}` the effective radius,
    - :math:`\\mu` the Coulomb friction coefficient.

References:
    [1] Cundall & Strack, "A Discrete Numerical Model for Granular
        Assemblies", Géotechnique 29, 1979.
    [2] Poschel & Schwager, *Computational Granular Dynamics*, Springer 2005.
    [3] Luding, "Introduction to Discrete Element Methods", in *Discrete
        Modelling of Geomaterials*, 2008.

Domain III.10 — Solid Mechanics / Granular Mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Contact models
# ---------------------------------------------------------------------------

class ContactModel(Enum):
    LINEAR_SPRING = auto()
    HERTZ_MINDLIN = auto()


@dataclass
class MaterialProperties:
    """DEM particle material properties."""
    E: float = 1e7        # Young's modulus [Pa]
    nu: float = 0.3       # Poisson's ratio
    rho: float = 2500.0   # Density [kg/m³]
    mu: float = 0.5       # Friction coefficient
    e_n: float = 0.8      # Normal coefficient of restitution
    e_t: float = 0.5      # Tangential coefficient of restitution

    @property
    def G(self) -> float:
        """Shear modulus."""
        return self.E / (2.0 * (1.0 + self.nu))


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class DEMState:
    """
    DEM particle state.

    Attributes:
        x: Positions ``(n_particles, dim)``.
        v: Velocities ``(n_particles, dim)``.
        omega: Angular velocities ``(n_particles, 3)`` (or ``(n, 1)`` in 2D).
        radii: Particle radii ``(n_particles,)``.
        mass: Particle masses ``(n_particles,)``.
        inertia: Moments of inertia ``(n_particles,)``.
        force: Net forces ``(n_particles, dim)``.
        torque: Net torques ``(n_particles, 3)``.
    """
    x: NDArray
    v: NDArray
    omega: NDArray
    radii: NDArray
    mass: NDArray
    inertia: NDArray
    force: NDArray
    torque: NDArray

    @property
    def n_particles(self) -> int:
        return self.x.shape[0]

    @property
    def dim(self) -> int:
        return self.x.shape[1]


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class DEMSolver:
    r"""
    Discrete Element Method solver.

    Velocity-Verlet integration with cell-list contact detection
    and Hertz-Mindlin or linear-spring-dashpot contact law.

    Parameters:
        contact_model: Contact model type.
        material: Material properties.
        gravity: Gravity vector ``(dim,)``.
        dt: Time step [s].

    Example::

        mat = MaterialProperties(E=1e6, mu=0.5, rho=2500)
        solver = DEMSolver(contact_model=ContactModel.HERTZ_MINDLIN, material=mat, dt=1e-5)
        state = solver.create_packing(n=200, box=(1.0, 1.0), r_mean=0.02)
        state = solver.step_n(state, 1000)
    """

    def __init__(
        self,
        contact_model: ContactModel = ContactModel.HERTZ_MINDLIN,
        material: MaterialProperties = MaterialProperties(),
        gravity: Optional[NDArray] = None,
        dt: float = 1e-5,
    ) -> None:
        self.model = contact_model
        self.mat = material
        self.dt = dt

        if gravity is None:
            self.gravity = np.array([0.0, -9.81], dtype=np.float64)
        else:
            self.gravity = np.asarray(gravity, dtype=np.float64)

        # Tangential displacement history
        self._xi_t: dict[Tuple[int, int], NDArray] = {}

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def create_packing(
        self,
        n: int,
        box: Tuple[float, ...],
        r_mean: float = 0.02,
        r_std: float = 0.002,
        seed: int = 42,
    ) -> DEMState:
        """
        Create an initial random packing of spheres in a box.

        Particles are placed on a grid with small random perturbation
        to avoid perfect alignment.
        """
        rng = np.random.default_rng(seed)
        dim = len(box)
        radii = rng.normal(r_mean, r_std, n)
        radii = np.clip(radii, r_mean * 0.5, r_mean * 1.5)

        # Grid placement
        side = int(np.ceil(n ** (1.0 / dim)))
        spacing = [box[d] / (side + 1) for d in range(dim)]
        x = np.zeros((n, dim), dtype=np.float64)
        idx = 0
        if dim == 2:
            for i in range(side):
                for j in range(side):
                    if idx >= n:
                        break
                    x[idx] = [(i + 1) * spacing[0], (j + 1) * spacing[1]]
                    idx += 1
        else:
            for i in range(side):
                for j in range(side):
                    for k in range(side):
                        if idx >= n:
                            break
                        x[idx] = [
                            (i + 1) * spacing[0],
                            (j + 1) * spacing[1],
                            (k + 1) * spacing[2],
                        ]
                        idx += 1

        # Small perturbation
        x += rng.uniform(-r_mean * 0.1, r_mean * 0.1, x.shape)

        mass = self.mat.rho * (4.0 / 3.0 * np.pi * radii ** 3 if dim == 3
                               else np.pi * radii ** 2)
        inertia = (0.4 * mass * radii ** 2 if dim == 3
                   else 0.5 * mass * radii ** 2)

        omega_dim = 3 if dim == 3 else 1
        return DEMState(
            x=x,
            v=np.zeros((n, dim), dtype=np.float64),
            omega=np.zeros((n, omega_dim), dtype=np.float64),
            radii=radii,
            mass=mass,
            inertia=inertia,
            force=np.zeros((n, dim), dtype=np.float64),
            torque=np.zeros((n, omega_dim), dtype=np.float64),
        )

    # -----------------------------------------------------------------------
    # Contact detection
    # -----------------------------------------------------------------------

    def _find_contacts(
        self, state: DEMState,
    ) -> list[Tuple[int, int, float, NDArray]]:
        """
        Detect contacts using brute-force O(n²) neighbour search.

        Returns list of ``(i, j, overlap, normal_ij)`` for each contact.
        """
        contacts = []
        n = state.n_particles
        for i in range(n):
            for j in range(i + 1, n):
                rij = state.x[j] - state.x[i]
                dist = np.linalg.norm(rij)
                overlap = state.radii[i] + state.radii[j] - dist
                if overlap > 0:
                    nij = rij / (dist + 1e-30)
                    contacts.append((i, j, overlap, nij))
        return contacts

    # -----------------------------------------------------------------------
    # Contact forces
    # -----------------------------------------------------------------------

    def _hertz_mindlin_force(
        self,
        i: int, j: int,
        overlap: float,
        nij: NDArray,
        state: DEMState,
    ) -> Tuple[NDArray, NDArray]:
        """Hertz-Mindlin contact force and torque."""
        Ri, Rj = state.radii[i], state.radii[j]
        R_eff = Ri * Rj / (Ri + Rj)
        E_eff = self.mat.E / (2.0 * (1.0 - self.mat.nu ** 2))

        # Normal force (Hertz)
        kn = (4.0 / 3.0) * E_eff * np.sqrt(R_eff)
        Fn_mag = kn * overlap ** 1.5

        # Normal damping
        m_eff = state.mass[i] * state.mass[j] / (state.mass[i] + state.mass[j])
        ln_e = np.log(max(self.mat.e_n, 1e-10))
        gamma_n = -2.0 * ln_e * np.sqrt(m_eff * kn * np.sqrt(overlap)) / (
            np.sqrt(np.pi ** 2 + ln_e ** 2) + 1e-30
        )

        # Relative velocity
        vij = state.v[i] - state.v[j]
        vn = np.dot(vij, nij)
        Fn = (Fn_mag - gamma_n * vn) * nij

        # Tangential
        vt = vij - vn * nij
        dim = state.dim
        if dim == 2:
            # Add rotational contribution
            omega_cross = np.zeros(dim)
            omega_cross[0] = -state.omega[i, 0] * Ri
            omega_cross[1] = state.omega[i, 0] * Ri
            # Simplified for 2D
        key = (min(i, j), max(i, j))
        if key not in self._xi_t:
            self._xi_t[key] = np.zeros(dim, dtype=np.float64)
        self._xi_t[key] += vt * self.dt

        G_eff = self.mat.G / (2.0 * (2.0 - self.mat.nu))
        kt = 8.0 * G_eff * np.sqrt(R_eff * overlap)
        Ft = -kt * self._xi_t[key]

        # Coulomb limit
        ft_mag = np.linalg.norm(Ft)
        fn_mag = np.linalg.norm(Fn)
        if ft_mag > self.mat.mu * fn_mag:
            Ft *= self.mat.mu * fn_mag / (ft_mag + 1e-30)
            self._xi_t[key] = -Ft / (kt + 1e-30)

        # Torque from tangential force
        torque_i = np.zeros(3 if dim == 3 else 1, dtype=np.float64)
        torque_j = np.zeros_like(torque_i)
        if dim == 3:
            lever_i = Ri * nij
            torque_i = np.cross(lever_i, Ft)
            torque_j = np.cross(-Rj * nij, -Ft)
        else:
            torque_i[0] = Ri * (nij[0] * Ft[1] - nij[1] * Ft[0])
            torque_j[0] = Rj * (-nij[0] * (-Ft[1]) + nij[1] * (-Ft[0]))

        F_total = Fn + Ft
        return F_total, torque_i

    def _linear_spring_force(
        self,
        i: int, j: int,
        overlap: float,
        nij: NDArray,
        state: DEMState,
    ) -> Tuple[NDArray, NDArray]:
        """Linear spring-dashpot contact model."""
        Ri, Rj = state.radii[i], state.radii[j]
        R_eff = Ri * Rj / (Ri + Rj)
        m_eff = state.mass[i] * state.mass[j] / (state.mass[i] + state.mass[j])

        # Stiffness (from Hertz theory at reference overlap)
        E_eff = self.mat.E / (2.0 * (1.0 - self.mat.nu ** 2))
        kn = (4.0 / 3.0) * E_eff * np.sqrt(R_eff)

        vij = state.v[i] - state.v[j]
        vn = np.dot(vij, nij)

        # Damping
        ln_e = np.log(max(self.mat.e_n, 1e-10))
        gamma = -2.0 * ln_e * np.sqrt(m_eff * kn) / (
            np.sqrt(np.pi ** 2 + ln_e ** 2) + 1e-30
        )

        Fn = (kn * overlap - gamma * vn) * nij
        dim = state.dim
        torque = np.zeros(3 if dim == 3 else 1, dtype=np.float64)
        return Fn, torque

    # -----------------------------------------------------------------------
    # Wall forces (simple box walls)
    # -----------------------------------------------------------------------

    def _wall_forces(
        self, state: DEMState, box: Tuple[float, ...],
    ) -> None:
        """Apply repulsive wall forces at domain boundaries."""
        dim = state.dim
        kw = self.mat.E * 0.01  # wall stiffness

        for d in range(dim):
            for i in range(state.n_particles):
                r = state.radii[i]
                # Lower wall
                overlap = r - state.x[i, d]
                if overlap > 0:
                    state.force[i, d] += kw * overlap
                # Upper wall
                overlap = state.x[i, d] + r - box[d]
                if overlap > 0:
                    state.force[i, d] -= kw * overlap

    # -----------------------------------------------------------------------
    # Time integration
    # -----------------------------------------------------------------------

    def compute_forces(
        self, state: DEMState, box: Optional[Tuple[float, ...]] = None,
    ) -> None:
        """Compute all contact forces and gravity."""
        dim = state.dim
        state.force[:] = 0.0
        state.torque[:] = 0.0

        # Gravity
        for i in range(state.n_particles):
            state.force[i] = state.mass[i] * self.gravity[:dim]

        # Contacts
        contacts = self._find_contacts(state)
        for i, j, overlap, nij in contacts:
            if self.model == ContactModel.HERTZ_MINDLIN:
                F, tau = self._hertz_mindlin_force(i, j, overlap, nij, state)
            else:
                F, tau = self._linear_spring_force(i, j, overlap, nij, state)
            state.force[i] += F
            state.force[j] -= F
            state.torque[i] += tau
            state.torque[j] -= tau

        # Walls
        if box is not None:
            self._wall_forces(state, box)

    def step(
        self, state: DEMState, box: Optional[Tuple[float, ...]] = None,
    ) -> DEMState:
        """Velocity-Verlet time step."""
        dt = self.dt
        dim = state.dim

        # Half-step velocity
        a = state.force / state.mass[:, None]
        state.v += 0.5 * dt * a

        alpha = state.torque / state.inertia[:, None]
        state.omega += 0.5 * dt * alpha

        # Full-step position
        state.x += dt * state.v

        # Recompute forces
        self.compute_forces(state, box)

        # Full-step velocity
        a = state.force / state.mass[:, None]
        state.v += 0.5 * dt * a

        alpha = state.torque / state.inertia[:, None]
        state.omega += 0.5 * dt * alpha

        return state

    def step_n(
        self, state: DEMState, n_steps: int,
        box: Optional[Tuple[float, ...]] = None,
    ) -> DEMState:
        """Multiple time steps."""
        self.compute_forces(state, box)
        for _ in range(n_steps):
            state = self.step(state, box)
        return state

    def kinetic_energy(self, state: DEMState) -> float:
        """Total translational + rotational kinetic energy."""
        KE_trans = 0.5 * np.sum(state.mass * np.sum(state.v ** 2, axis=1))
        KE_rot = 0.5 * np.sum(state.inertia * np.sum(state.omega ** 2, axis=1))
        return float(KE_trans + KE_rot)

    def coordination_number(self, state: DEMState) -> float:
        """Average coordination number (contacts per particle)."""
        contacts = self._find_contacts(state)
        if state.n_particles == 0:
            return 0.0
        return 2.0 * len(contacts) / state.n_particles

    def packing_fraction(self, state: DEMState, box: Tuple[float, ...]) -> float:
        """Volume (area) fraction occupied by particles."""
        dim = state.dim
        V_box = 1.0
        for d in range(dim):
            V_box *= box[d]
        if dim == 3:
            V_part = np.sum(4.0 / 3.0 * np.pi * state.radii ** 3)
        else:
            V_part = np.sum(np.pi * state.radii ** 2)
        return float(V_part / V_box)
