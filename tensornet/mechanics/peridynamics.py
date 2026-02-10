"""
Peridynamics
============

Non-local continuum mechanics formulation that replaces the PDE-based
stress divergence with an integral operator, naturally handling
discontinuities (cracks) without remeshing.

Equation of Motion (bond-based):
    ρ ü(x, t) = ∫_{H_x} f(u(x', t) - u(x, t), x' - x) dV_{x'} + b(x, t)

where H_x is the neighbourhood (horizon) of radius δ, and f is the
pairwise force density function (bond force).

Bond-based (prototype microelastic):
    f = c s(ξ, η) (ξ + η) / |ξ + η|

    where ξ = x' - x (reference), η = u' - u (displacement),
    s = (|ξ + η| - |ξ|) / |ξ| is bond stretch, and
    c = 18 K / (π δ⁴) (3D) is the micromodulus.

State-based:
    Ordinary state-based extends to full elasticity and plasticity.

References:
    [1] Silling, "Reformulation of elasticity theory for
        discontinuities and long-range forces",
        J. Mech. Phys. Solids 48, 2000.
    [2] Silling & Askari, "A Meshfree Method Based on the
        Peridynamic Model of Solid Mechanics", Comp. & Struct. 83, 2005.
    [3] Madenci & Oterkus, "Peridynamic Theory and Its Applications",
        Springer, 2014.

Domain III.1 — Solid Mechanics / Peridynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class PeridynamicsModel(Enum):
    """Peridynamics formulation variant."""
    BOND_BASED = "bond-based"
    ORDINARY_STATE = "ordinary-state"


@dataclass
class Material:
    """Peridynamic material properties."""
    E: float = 200e9       # Young's modulus [Pa]
    nu: float = 0.25       # Poisson's ratio (bond-based fixes ν = 1/4 in 3D, 1/3 in 2D)
    rho: float = 7800.0    # Density [kg/m³]
    G_c: float = 2700.0    # Critical energy release rate [J/m²]

    @property
    def K(self) -> float:
        """Bulk modulus."""
        return self.E / (3.0 * (1.0 - 2.0 * self.nu))

    @property
    def G(self) -> float:
        """Shear modulus."""
        return self.E / (2.0 * (1.0 + self.nu))


@dataclass
class PeridynamicState:
    """
    Particle state for peridynamic simulation.

    Attributes:
        x0: Reference positions ``(N, dim)``.
        u: Displacements ``(N, dim)``.
        v: Velocities ``(N, dim)``.
        volume: Particle volumes ``(N,)``.
        damage: Damage index ``(N,)`` in [0, 1]; 0 = intact, 1 = broken.
    """
    x0: NDArray
    u: NDArray
    v: NDArray
    volume: NDArray
    damage: NDArray

    @property
    def n_particles(self) -> int:
        return self.x0.shape[0]

    @property
    def dim(self) -> int:
        return self.x0.shape[1]

    @property
    def x(self) -> NDArray:
        """Current (deformed) positions."""
        return self.x0 + self.u


class PeridynamicsSolver:
    r"""
    Bond-based peridynamics solver with damage.

    Micromodulus:
        - 2D plane stress: :math:`c = 9E / (\pi h_t \delta^3)`
        - 3D: :math:`c = 18 K / (\pi \delta^4)`

    where :math:`h_t` is thickness (2D) and :math:`\delta` is the
    horizon radius.

    Bond failure criterion:
        :math:`s > s_0` where :math:`s_0 = \sqrt{5 G_c / (9 K \delta)}`
        (3D) or :math:`s_0 = \sqrt{4 \pi G_c / (9 E \delta)}` (2D).

    Parameters:
        material: Material properties.
        delta: Horizon radius.
        dim: Spatial dimension (2 or 3).
        thickness: Plate thickness for 2D.
        dt: Time step (auto-selected if None).
    """

    def __init__(
        self,
        material: Optional[Material] = None,
        delta: float = 0.003,
        dim: int = 2,
        thickness: float = 0.001,
        dt: Optional[float] = None,
    ) -> None:
        self.mat = material or Material()
        self.delta = delta
        self.dim = dim
        self.thickness = thickness

        # Micromodulus
        if dim == 2:
            self.c = 9.0 * self.mat.E / (np.pi * thickness * delta ** 3)
        else:
            self.c = 18.0 * self.mat.K / (np.pi * delta ** 4)

        # Critical stretch
        if dim == 2:
            self.s0 = np.sqrt(
                4.0 * np.pi * self.mat.G_c / (9.0 * self.mat.E * delta)
            )
        else:
            self.s0 = np.sqrt(
                5.0 * self.mat.G_c / (9.0 * self.mat.K * delta)
            )

        # Stable time step estimate: Δt ≤ √(2ρ / (πδ²c·n_family))
        # Conservative default
        if dt is not None:
            self.dt = dt
        else:
            self.dt = 0.8 * np.sqrt(2.0 * self.mat.rho / (np.pi * delta ** 2 * self.c * 30))

        self._bonds: Optional[list] = None
        self._bond_intact: Optional[NDArray] = None
        self._step_count = 0

    def build_bonds(self, x0: NDArray) -> None:
        """
        Pre-compute bond connectivity: for each particle, find all
        family members within the horizon δ.
        """
        N = x0.shape[0]
        self._bonds = []
        for i in range(N):
            dx = x0 - x0[i]
            dist = np.linalg.norm(dx, axis=1)
            mask = (dist > 0) & (dist <= self.delta)
            self._bonds.append(np.where(mask)[0])

        # Bond status (intact = True)
        total_bonds = sum(len(b) for b in self._bonds)
        self._bond_intact_map: dict[Tuple[int, int], bool] = {}
        for i in range(N):
            for j in self._bonds[i]:
                key = (min(i, j), max(i, j))
                if key not in self._bond_intact_map:
                    self._bond_intact_map[key] = True

    def create_state(
        self,
        x0: NDArray,
        v0: Optional[NDArray] = None,
    ) -> PeridynamicState:
        """Create initial state and build bonds."""
        N = x0.shape[0]
        if v0 is None:
            v0 = np.zeros_like(x0)
        volume = np.full(N, (self.delta / 3.0) ** self.dim, dtype=np.float64)
        # Better volume estimate for uniform grids
        if N > 1:
            # Approximate spacing
            dists = np.linalg.norm(x0[1:] - x0[:-1], axis=1)
            h = np.median(dists[dists > 0])
            volume[:] = h ** self.dim
            if self.dim == 2:
                volume *= self.thickness

        self.build_bonds(x0)
        return PeridynamicState(
            x0=x0.copy(),
            u=np.zeros_like(x0),
            v=v0.copy(),
            volume=volume,
            damage=np.zeros(N, dtype=np.float64),
        )

    def compute_force(self, state: PeridynamicState) -> NDArray:
        """Compute peridynamic force density for all particles."""
        N = state.n_particles
        f = np.zeros((N, self.dim), dtype=np.float64)
        x_cur = state.x  # Deformed positions

        for i in range(N):
            for j in self._bonds[i]:
                key = (min(i, j), max(i, j))
                if not self._bond_intact_map.get(key, True):
                    continue

                xi = state.x0[j] - state.x0[i]      # Reference bond vector
                eta = state.u[j] - state.u[i]         # Relative displacement
                y = xi + eta                           # Deformed bond vector
                xi_norm = np.linalg.norm(xi)
                y_norm = np.linalg.norm(y)

                if y_norm < 1e-30 or xi_norm < 1e-30:
                    continue

                # Bond stretch
                s = (y_norm - xi_norm) / xi_norm

                # Check failure
                if abs(s) > self.s0:
                    self._bond_intact_map[key] = False
                    continue

                # Bond force (scalar)
                f_mag = self.c * s

                # Direction
                e_bond = y / y_norm
                f[i] += f_mag * e_bond * state.volume[j]

        return f

    def compute_damage(self, state: PeridynamicState) -> NDArray:
        """
        Compute damage index φ for each particle:
        φ = 1 - (intact bonds) / (original bonds).
        """
        N = state.n_particles
        damage = np.zeros(N, dtype=np.float64)
        for i in range(N):
            n_total = len(self._bonds[i])
            if n_total == 0:
                continue
            n_intact = 0
            for j in self._bonds[i]:
                key = (min(i, j), max(i, j))
                if self._bond_intact_map.get(key, True):
                    n_intact += 1
            damage[i] = 1.0 - n_intact / n_total
        return damage

    def step(
        self,
        state: PeridynamicState,
        body_force: Optional[NDArray] = None,
    ) -> PeridynamicState:
        """Advance one time step via Velocity-Verlet integration."""
        dt = self.dt
        N = state.n_particles

        # Force at current time
        f = self.compute_force(state)
        if body_force is not None:
            f += body_force

        acc = f / self.mat.rho

        # Half-step velocity
        v_half = state.v + 0.5 * dt * acc

        # Full-step displacement
        u_new = state.u + dt * v_half

        # Force at new position
        state_new = PeridynamicState(
            x0=state.x0, u=u_new, v=v_half,
            volume=state.volume, damage=state.damage,
        )
        f_new = self.compute_force(state_new)
        if body_force is not None:
            f_new += body_force
        acc_new = f_new / self.mat.rho

        # Full-step velocity
        v_new = v_half + 0.5 * dt * acc_new

        # Update damage
        damage_new = self.compute_damage(state_new)

        self._step_count += 1
        return PeridynamicState(
            x0=state.x0, u=u_new, v=v_new,
            volume=state.volume, damage=damage_new,
        )

    def step_n(
        self,
        state: PeridynamicState,
        n_steps: int,
        body_force: Optional[NDArray] = None,
    ) -> PeridynamicState:
        for _ in range(n_steps):
            state = self.step(state, body_force)
        return state

    @property
    def steps(self) -> int:
        return self._step_count

    def strain_energy(self, state: PeridynamicState) -> float:
        """Total elastic strain energy."""
        W = 0.0
        for i in range(state.n_particles):
            for j in self._bonds[i]:
                if j <= i:
                    continue  # Count each bond once
                key = (i, j)
                if not self._bond_intact_map.get(key, True):
                    continue
                xi = state.x0[j] - state.x0[i]
                eta = state.u[j] - state.u[i]
                xi_norm = np.linalg.norm(xi)
                y_norm = np.linalg.norm(xi + eta)
                if xi_norm < 1e-30:
                    continue
                s = (y_norm - xi_norm) / xi_norm
                w_bond = 0.5 * self.c * s ** 2 * xi_norm
                W += w_bond * state.volume[i] * state.volume[j]
        return W

    def kinetic_energy(self, state: PeridynamicState) -> float:
        """Total kinetic energy."""
        vol = state.volume
        if self.dim == 2:
            vol = state.volume  # Already includes thickness if set
        return 0.5 * self.mat.rho * float(
            np.sum(vol * np.sum(state.v ** 2, axis=1))
        )
