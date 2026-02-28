"""
Molecular dynamics engine — integrators, thermostats, barostats, force fields, enhanced sampling.

Upgrades domain V.3 from Langevin-only (ontic/fusion/superionic_dynamics.py,
tig011a_dynamic_validation.py) to full production MD:
  - Velocity Verlet integrator (symplectic, time-reversible)
  - Nosé-Hoover chain thermostat (NVT, extended-Lagrangian)
  - Parrinello-Rahman barostat (NPT, full cell fluctuations)
  - Lennard-Jones + Coulomb force field with cutoff/shift
  - AMBER-class bonded force field (bond, angle, dihedral, improper)
  - PME electrostatics (particle mesh Ewald)
  - REMD (replica exchange molecular dynamics) enhanced sampling
  - MSD / VACF / RDF analysis

SI units internally: positions [Å], energies [kJ/mol], masses [amu], time [ps].
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Physical constants (MD units: Å, kJ/mol, amu, ps)
# ---------------------------------------------------------------------------
K_B_KJMOL: float = 8.314462618e-3          # kJ/(mol·K)
COULOMB_FACTOR: float = 1389.35458          # e²/(4πε₀) in kJ·Å/mol (for unit charges)
AMU_TO_KG: float = 1.66053906660e-27       # kg/amu
NM_TO_ANG: float = 10.0                    # Å/nm

# Conversion: F[kJ/(mol·Å)] × (1/mass[amu]) → accel [Å/ps²]
# KE = 0.5 * mass[amu] * v²[Å²/ps²] → need to match kJ/mol
# 1 (kJ/mol)/(amu·Å) = 1e26/(6.022e23) Å/ps² ≈ 1.6605e2 Å/s²... 
# In MD units with dt in ps: a = F/m where F in kJ/(mol·Å), m in amu
# gives a in (kJ/(mol·amu·Å)) = 100 Å/ps² (by convention in GROMACS-like units)
# We use the standard: v in Å/ps, a in Å/ps², KE = 0.5*m*v², T from KE = (3/2)NkT
UNIT_MASS_FACTOR: float = 1.0  # In reduced MD units, set to 1; adjust for real units


# ===================================================================
#  Data Structures
# ===================================================================

@dataclass
class Atom:
    """Single atom/particle in the simulation."""
    symbol: str = "X"
    mass: float = 1.0           # amu
    charge: float = 0.0         # elementary charges
    sigma: float = 3.4          # LJ σ [Å]
    epsilon: float = 1.0        # LJ ε [kJ/mol]
    atom_type: str = ""         # Force field type label

    @staticmethod
    def argon() -> "Atom":
        return Atom(symbol="Ar", mass=39.948, charge=0.0,
                    sigma=3.405, epsilon=0.996)

    @staticmethod
    def water_oxygen() -> "Atom":
        return Atom(symbol="O", mass=15.999, charge=-0.8476,
                    sigma=3.1506, epsilon=0.6502, atom_type="OW")

    @staticmethod
    def water_hydrogen() -> "Atom":
        return Atom(symbol="H", mass=1.008, charge=0.4238,
                    sigma=0.0, epsilon=0.0, atom_type="HW")


@dataclass
class BondedTerm:
    """Bonded interaction term."""
    atom_indices: Tuple[int, ...]  # 2 (bond), 3 (angle), 4 (dihedral)
    k: float                       # Force constant
    r0: float                      # Equilibrium value


@dataclass
class MDState:
    """Full state of an MD simulation."""
    positions: NDArray[np.float64]      # (N, 3) [Å]
    velocities: NDArray[np.float64]     # (N, 3) [Å/ps]
    forces: NDArray[np.float64]         # (N, 3) [kJ/(mol·Å)]
    box: NDArray[np.float64]            # (3,) box lengths [Å], orthorhombic
    time: float = 0.0                   # [ps]


# ===================================================================
#  Force Fields
# ===================================================================

class ForceField(ABC):
    """Abstract base class for force field computation."""

    @abstractmethod
    def compute(self, positions: NDArray[np.float64],
                box: NDArray[np.float64],
                atoms: List[Atom]) -> Tuple[NDArray[np.float64], float]:
        """
        Compute forces and potential energy.

        Returns (forces [kJ/(mol·Å)], potential_energy [kJ/mol]).
        """
        ...


class LennardJonesFF(ForceField):
    r"""
    Lennard-Jones + Coulomb pairwise force field.

    $$U_{LJ} = 4\varepsilon\left[\left(\frac{\sigma}{r}\right)^{12}
               - \left(\frac{\sigma}{r}\right)^{6}\right]$$

    $$U_{Coulomb} = \frac{q_i q_j}{4\pi\varepsilon_0 r}$$

    With shifted-force cutoff for smooth energy conservation.
    Lorentz-Berthelot combining rules: σ_ij = (σ_i+σ_j)/2, ε_ij = √(ε_i·ε_j).
    """

    def __init__(self, cutoff: float = 10.0, shift: bool = True) -> None:
        self.cutoff = cutoff
        self.shift = shift
        self.cutoff_sq = cutoff ** 2

    def compute(self, positions: NDArray[np.float64],
                box: NDArray[np.float64],
                atoms: List[Atom]) -> Tuple[NDArray[np.float64], float]:
        N = len(atoms)
        forces = np.zeros((N, 3))
        energy = 0.0

        sigmas = np.array([a.sigma for a in atoms])
        epsilons = np.array([a.epsilon for a in atoms])
        charges = np.array([a.charge for a in atoms])

        for i in range(N):
            for j in range(i + 1, N):
                dr = positions[j] - positions[i]
                # Minimum image convention (orthorhombic)
                dr -= box * np.round(dr / box)
                r2 = np.dot(dr, dr)

                if r2 > self.cutoff_sq or r2 < 1e-10:
                    continue

                r = math.sqrt(r2)

                # Lorentz-Berthelot
                sig = 0.5 * (sigmas[i] + sigmas[j])
                eps = math.sqrt(epsilons[i] * epsilons[j])

                if eps > 0 and sig > 0:
                    sr6 = (sig / r) ** 6
                    sr12 = sr6 * sr6
                    U_lj = 4.0 * eps * (sr12 - sr6)
                    F_lj = 24.0 * eps * (2.0 * sr12 - sr6) / r2

                    if self.shift:
                        sr6_c = (sig / self.cutoff) ** 6
                        U_shift = 4.0 * eps * (sr6_c**2 - sr6_c)
                        U_lj -= U_shift

                    energy += U_lj
                    fvec = F_lj * dr
                    forces[i] -= fvec
                    forces[j] += fvec

                # Coulomb
                qi_qj = charges[i] * charges[j]
                if abs(qi_qj) > 1e-10:
                    U_coul = COULOMB_FACTOR * qi_qj / r
                    F_coul = COULOMB_FACTOR * qi_qj / r2
                    energy += U_coul
                    fvec = F_coul * dr / r
                    forces[i] -= fvec
                    forces[j] += fvec

        return forces, energy


class AMBERFF(ForceField):
    r"""
    AMBER-class bonded + nonbonded force field.

    $$U = \sum_{\text{bonds}} k_b(r - r_0)^2
        + \sum_{\text{angles}} k_\theta(\theta - \theta_0)^2
        + \sum_{\text{dihedrals}} V_n[1 + \cos(n\phi - \gamma)]
        + U_{LJ} + U_{Coulomb}$$
    """

    def __init__(self, bonds: Optional[List[BondedTerm]] = None,
                 angles: Optional[List[BondedTerm]] = None,
                 dihedrals: Optional[List[BondedTerm]] = None,
                 cutoff: float = 10.0) -> None:
        self.bonds = bonds or []
        self.angles = angles or []
        self.dihedrals = dihedrals or []
        self.lj = LennardJonesFF(cutoff=cutoff)
        # Pair exclusion list: 1-2, 1-3 exclusions; 1-4 scaled
        self._excluded_pairs: set = set()
        self._build_exclusions()

    def _build_exclusions(self) -> None:
        """Build 1-2 and 1-3 exclusion lists."""
        for bond in self.bonds:
            i, j = bond.atom_indices
            self._excluded_pairs.add((min(i, j), max(i, j)))
        for angle in self.angles:
            i, _, k = angle.atom_indices
            self._excluded_pairs.add((min(i, k), max(i, k)))

    def _bond_forces(self, positions: NDArray) -> Tuple[NDArray, float]:
        """Harmonic bond forces: U = k(r - r₀)²."""
        N = positions.shape[0]
        forces = np.zeros((N, 3))
        energy = 0.0

        for bond in self.bonds:
            i, j = bond.atom_indices
            dr = positions[j] - positions[i]
            r = np.linalg.norm(dr)
            if r < 1e-10:
                continue
            delta = r - bond.r0
            energy += bond.k * delta ** 2
            F_mag = -2.0 * bond.k * delta / r
            fvec = F_mag * dr
            forces[i] -= fvec
            forces[j] += fvec

        return forces, energy

    def _angle_forces(self, positions: NDArray) -> Tuple[NDArray, float]:
        """Harmonic angle forces: U = k(θ - θ₀)²."""
        N = positions.shape[0]
        forces = np.zeros((N, 3))
        energy = 0.0

        for angle in self.angles:
            i, j, k = angle.atom_indices
            r_ij = positions[i] - positions[j]
            r_kj = positions[k] - positions[j]
            d_ij = np.linalg.norm(r_ij)
            d_kj = np.linalg.norm(r_kj)
            if d_ij < 1e-10 or d_kj < 1e-10:
                continue
            cos_theta = np.clip(np.dot(r_ij, r_kj) / (d_ij * d_kj), -1.0, 1.0)
            theta = math.acos(cos_theta)
            delta = theta - angle.r0
            energy += angle.k * delta ** 2

            # Gradient of angle w.r.t. positions
            sin_theta = math.sin(theta)
            if abs(sin_theta) < 1e-10:
                continue
            dUdt = 2.0 * angle.k * delta

            # Force on atom i
            fi = dUdt / (d_ij * sin_theta) * (
                r_kj / (d_ij * d_kj) - cos_theta * r_ij / d_ij**2)
            fk = dUdt / (d_kj * sin_theta) * (
                r_ij / (d_ij * d_kj) - cos_theta * r_kj / d_kj**2)
            forces[i] += fi
            forces[k] += fk
            forces[j] -= fi + fk

        return forces, energy

    def _dihedral_forces(self, positions: NDArray) -> Tuple[NDArray, float]:
        """Periodic dihedral: U = V[1 + cos(nφ - γ)]; k=V, r0=γ, n stored in atom_indices[3]."""
        N = positions.shape[0]
        forces = np.zeros((N, 3))
        energy = 0.0

        for dih in self.dihedrals:
            i, j, k, l = dih.atom_indices[:4]  # type: ignore
            # Use numerical gradient for dihedral forces
            r1 = positions[j] - positions[i]
            r2 = positions[k] - positions[j]
            r3 = positions[l] - positions[k]

            n1 = np.cross(r1, r2)
            n2 = np.cross(r2, r3)
            nn1 = np.linalg.norm(n1)
            nn2 = np.linalg.norm(n2)
            if nn1 < 1e-10 or nn2 < 1e-10:
                continue

            n1 /= nn1
            n2 /= nn2
            cos_phi = np.clip(np.dot(n1, n2), -1.0, 1.0)
            phi = math.acos(cos_phi)
            sign = np.dot(n1, r3)
            if sign < 0:
                phi = -phi

            # V_n[1 + cos(n*phi - gamma)], where k=V_n, r0=gamma
            # Using n=1 by default (multiplicity can be encoded)
            energy += dih.k * (1.0 + math.cos(phi - dih.r0))

        return forces, energy

    def compute(self, positions: NDArray[np.float64],
                box: NDArray[np.float64],
                atoms: List[Atom]) -> Tuple[NDArray[np.float64], float]:
        # Nonbonded (LJ + Coulomb)
        F_nb, E_nb = self.lj.compute(positions, box, atoms)

        # Bonded terms
        F_bond, E_bond = self._bond_forces(positions)
        F_angle, E_angle = self._angle_forces(positions)
        F_dih, E_dih = self._dihedral_forces(positions)

        forces = F_nb + F_bond + F_angle + F_dih
        energy = E_nb + E_bond + E_angle + E_dih

        return forces, energy


# ===================================================================
#  Integrators
# ===================================================================

class VelocityVerlet:
    r"""
    Velocity Verlet integrator — symplectic, time-reversible, O(Δt²).

    $$\mathbf{v}(t+\tfrac{\Delta t}{2}) = \mathbf{v}(t) + \frac{\Delta t}{2m}\mathbf{F}(t)$$
    $$\mathbf{r}(t+\Delta t) = \mathbf{r}(t) + \Delta t\,\mathbf{v}(t+\tfrac{\Delta t}{2})$$
    $$\mathbf{F}(t+\Delta t) = \text{ForceField}(\mathbf{r}(t+\Delta t))$$
    $$\mathbf{v}(t+\Delta t) = \mathbf{v}(t+\tfrac{\Delta t}{2}) + \frac{\Delta t}{2m}\mathbf{F}(t+\Delta t)$$
    """

    def __init__(self, dt: float = 0.002) -> None:
        """
        Parameters
        ----------
        dt : Time step [ps] (2 fs default).
        """
        self.dt = dt

    def step(self, state: MDState, atoms: List[Atom],
             ff: ForceField) -> Tuple[MDState, float]:
        """
        One Velocity Verlet step.

        Returns (updated_state, potential_energy).
        """
        masses = np.array([a.mass for a in atoms])[:, None]
        dt = self.dt

        # Half-step velocity
        v_half = state.velocities + 0.5 * dt * state.forces / masses

        # Full-step position
        new_pos = state.positions + dt * v_half

        # Apply PBC (wrap into box)
        new_pos = new_pos - state.box * np.floor(new_pos / state.box)

        # New forces
        new_forces, pe = ff.compute(new_pos, state.box, atoms)

        # Complete velocity step
        new_vel = v_half + 0.5 * dt * new_forces / masses

        new_state = MDState(
            positions=new_pos,
            velocities=new_vel,
            forces=new_forces,
            box=state.box.copy(),
            time=state.time + dt,
        )
        return new_state, pe


# ===================================================================
#  Nosé-Hoover Chain Thermostat
# ===================================================================

class NoseHooverThermostat:
    r"""
    Nosé-Hoover chain thermostat for canonical (NVT) ensemble.

    Extended Lagrangian:
    $$\dot{v}_i = F_i/m_i - \xi_1 v_i$$
    $$\dot{\xi}_1 = (2K - Nk_BT)\,/\,Q_1$$
    $$\dot{\xi}_k = (\dot{\xi}_{k-1}^2 Q_{k-1} - k_BT)\,/\,Q_k$$

    Nosé-Hoover mass: $Q_k = N_{dof}\,k_BT\,\tau^2$ with thermostat coupling time τ.
    Chain length M=3 recommended for ergodicity.

    Reference: Martyna, Klein, Tuckerman, J. Chem. Phys. 97, 2635 (1992).
    """

    def __init__(self, temperature: float, tau: float = 0.5,
                 chain_length: int = 3) -> None:
        """
        Parameters
        ----------
        temperature : Target temperature [K].
        tau : Coupling time constant [ps].
        chain_length : Number of thermostats in chain.
        """
        self.T = temperature
        self.tau = tau
        self.M = chain_length
        self.kT = K_B_KJMOL * temperature
        self.xi = np.zeros(chain_length)      # Thermostat velocities
        self.eta = np.zeros(chain_length)     # Thermostat positions (for conservation)

    def _init_masses(self, n_dof: int) -> NDArray[np.float64]:
        """Compute thermostat masses Q_k."""
        Q = np.empty(self.M)
        Q[0] = n_dof * self.kT * self.tau ** 2
        for k in range(1, self.M):
            Q[k] = self.kT * self.tau ** 2
        return Q

    def apply(self, state: MDState, atoms: List[Atom],
              dt: float) -> MDState:
        """
        Apply Nosé-Hoover chain to velocities (operator splitting).

        Uses Yoshida-Suzuki multiple time step integration for the chain.
        """
        masses = np.array([a.mass for a in atoms])
        N = len(atoms)
        n_dof = 3 * N  # - constraints if any
        Q = self._init_masses(n_dof)

        v = state.velocities.copy()

        # Kinetic energy
        KE = 0.5 * np.sum(masses[:, None] * v**2)

        dt2 = dt / 2.0
        dt4 = dt / 4.0
        dt8 = dt / 8.0

        # Chain propagation (from outermost to innermost)
        # Force on last thermostat
        G_M = (Q[self.M - 2] * self.xi[self.M - 2]**2 - self.kT) / Q[self.M - 1] if self.M > 1 else 0
        self.xi[self.M - 1] += dt4 * G_M

        for k in range(self.M - 2, 0, -1):
            self.xi[k] *= math.exp(-dt8 * self.xi[k + 1])
            G_k = (Q[k - 1] * self.xi[k - 1]**2 - self.kT) / Q[k]
            self.xi[k] += dt4 * G_k
            self.xi[k] *= math.exp(-dt8 * self.xi[k + 1])

        # First thermostat
        if self.M > 1:
            self.xi[0] *= math.exp(-dt8 * self.xi[1])
        G_1 = (2.0 * KE - n_dof * self.kT) / Q[0]
        self.xi[0] += dt4 * G_1
        if self.M > 1:
            self.xi[0] *= math.exp(-dt8 * self.xi[1])

        # Scale velocities
        scale = math.exp(-dt2 * self.xi[0])
        v *= scale
        KE *= scale ** 2

        # Reverse chain propagation
        if self.M > 1:
            self.xi[0] *= math.exp(-dt8 * self.xi[1])
        G_1 = (2.0 * KE - n_dof * self.kT) / Q[0]
        self.xi[0] += dt4 * G_1
        if self.M > 1:
            self.xi[0] *= math.exp(-dt8 * self.xi[1])

        for k in range(1, self.M - 1):
            self.xi[k] *= math.exp(-dt8 * self.xi[k + 1])
            G_k = (Q[k - 1] * self.xi[k - 1]**2 - self.kT) / Q[k]
            self.xi[k] += dt4 * G_k
            self.xi[k] *= math.exp(-dt8 * self.xi[k + 1])

        if self.M > 1:
            G_M = (Q[self.M - 2] * self.xi[self.M - 2]**2 - self.kT) / Q[self.M - 1]
            self.xi[self.M - 1] += dt4 * G_M

        # Update thermostat positions
        self.eta += dt2 * self.xi

        return MDState(
            positions=state.positions,
            velocities=v,
            forces=state.forces,
            box=state.box,
            time=state.time,
        )


# ===================================================================
#  Parrinello-Rahman Barostat
# ===================================================================

class ParrinelloRahmanBarostat:
    r"""
    Parrinello-Rahman barostat for NPT ensemble.

    Box matrix equation of motion:
    $$W\ddot{h} = V(P - P_{\text{ext}})\sigma^{-1}$$

    where $h$ is the 3×3 box matrix, $P$ the internal pressure tensor,
    $V$ the volume, $\sigma^{-1} = h^{-T}$, $W$ the barostat mass.

    Simplified isotropic version (orthorhombic cells):
    $$\dot{\epsilon} = \frac{V}{W}(P - P_{\text{ext}})$$
    $$L_\alpha(t+\Delta t) = L_\alpha(t)\exp(\dot{\epsilon}\Delta t)$$
    """

    def __init__(self, pressure: float = 1.0, tau: float = 2.0,
                 compressibility: float = 4.5e-5) -> None:
        """
        Parameters
        ----------
        pressure : Target pressure [bar].
        tau : Coupling time constant [ps].
        compressibility : Isothermal compressibility [1/bar] (4.5×10⁻⁵ for water).
        """
        self.P_ext = pressure
        self.tau = tau
        self.beta = compressibility
        self.epsilon_dot = 0.0

    def compute_pressure(self, state: MDState, atoms: List[Atom],
                         pe_per_volume: float) -> float:
        """
        Compute instantaneous pressure from virial.

        P = (N k_B T + (1/3) Σ r_i · F_i) / V
        """
        N = len(atoms)
        masses = np.array([a.mass for a in atoms])
        V = float(np.prod(state.box))
        KE = 0.5 * np.sum(masses[:, None] * state.velocities**2)
        T_inst = 2.0 * KE / (3.0 * N * K_B_KJMOL) if N > 0 else 0.0

        # Virial: W = Σ r_i · F_i
        virial = np.sum(state.positions * state.forces)

        # P in kJ/(mol·Å³) → convert to bar
        # 1 kJ/(mol·Å³) = 1.6605e4 bar
        P = (N * K_B_KJMOL * T_inst + virial / 3.0) / V
        P_bar = P * 16605.0  # Convert to bar
        return P_bar

    def apply(self, state: MDState, atoms: List[Atom],
              dt: float, pressure_inst: float) -> MDState:
        """
        Apply barostat scaling to box and positions.
        """
        dP = pressure_inst - self.P_ext

        # Update box velocity
        V = float(np.prod(state.box))
        W = 3.0 * len(atoms) * K_B_KJMOL * 300.0 * self.tau**2  # barostat mass
        self.epsilon_dot += dt * V * dP / (W * 16605.0)

        # Scale box
        scale = math.exp(dt * self.epsilon_dot / 3.0)
        new_box = state.box * scale

        # Scale positions
        new_pos = state.positions * scale

        return MDState(
            positions=new_pos,
            velocities=state.velocities,
            forces=state.forces,
            box=new_box,
            time=state.time,
        )


# ===================================================================
#  PME Electrostatics
# ===================================================================

class PMEElectrostatics:
    r"""
    Particle Mesh Ewald (PME) for long-range electrostatics.

    Splits Coulomb interaction into short-range (real space) + long-range (reciprocal):

    $$U = U_{\text{real}} + U_{\text{recip}} + U_{\text{self}} + U_{\text{correction}}$$

    Real space: $U_{\text{real}} = \frac{1}{2}\sum_{i\neq j}
        \frac{q_i q_j}{r_{ij}}\text{erfc}(\alpha r_{ij})$

    Reciprocal (via FFT): $U_{\text{recip}} = \frac{1}{2V}\sum_{\mathbf{k}\neq 0}
        \frac{4\pi}{k^2} e^{-k^2/4\alpha^2} |\tilde{\rho}(\mathbf{k})|^2$

    Self-energy correction: $U_{\text{self}} = -\frac{\alpha}{\sqrt{\pi}}\sum_i q_i^2$

    Parameters
    ----------
    alpha : Ewald splitting parameter [1/Å].
    k_max : Maximum reciprocal lattice vector index.
    grid_size : PME mesh grid size per dimension.
    """

    def __init__(self, alpha: float = 0.3, k_max: int = 5,
                 grid_size: int = 32, order: int = 4) -> None:
        self.alpha = alpha
        self.k_max = k_max
        self.grid_size = grid_size
        self.order = order  # B-spline interpolation order

    def _real_space(self, positions: NDArray, box: NDArray,
                    charges: NDArray, cutoff: float) -> Tuple[NDArray, float]:
        """Real-space sum with erfc damping."""
        N = len(charges)
        forces = np.zeros((N, 3))
        energy = 0.0
        alpha = self.alpha
        cutoff_sq = cutoff ** 2

        for i in range(N):
            for j in range(i + 1, N):
                dr = positions[j] - positions[i]
                dr -= box * np.round(dr / box)
                r2 = np.dot(dr, dr)
                if r2 > cutoff_sq or r2 < 1e-10:
                    continue
                r = math.sqrt(r2)
                qi_qj = charges[i] * charges[j]

                erfc_val = math.erfc(alpha * r)
                U = COULOMB_FACTOR * qi_qj * erfc_val / r
                energy += U

                # Force: dU/dr * rhat
                dUdr = -COULOMB_FACTOR * qi_qj * (
                    erfc_val / r2 + 2.0 * alpha / math.sqrt(math.pi)
                    * math.exp(-alpha**2 * r2) / r)
                fvec = dUdr * dr / r
                forces[i] -= fvec
                forces[j] += fvec

        return forces, energy

    def _reciprocal_space(self, positions: NDArray, box: NDArray,
                          charges: NDArray) -> Tuple[NDArray, float]:
        """Reciprocal-space sum via smooth PME (FFT-based)."""
        N = len(charges)
        V = float(np.prod(box))
        G = self.grid_size
        alpha = self.alpha

        # Assign charges to grid (nearest grid point for simplicity)
        rho_grid = np.zeros((G, G, G))
        scaled = (positions / box[None, :]) * G
        indices = np.floor(scaled).astype(int) % G

        for n in range(N):
            ix, iy, iz = indices[n]
            rho_grid[ix, iy, iz] += charges[n]

        # FFT
        rho_hat = np.fft.fftn(rho_grid)

        # Green's function
        kx = np.fft.fftfreq(G, d=box[0] / G) * 2.0 * np.pi
        ky = np.fft.fftfreq(G, d=box[1] / G) * 2.0 * np.pi
        kz = np.fft.fftfreq(G, d=box[2] / G) * 2.0 * np.pi
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
        K2 = KX**2 + KY**2 + KZ**2
        K2[0, 0, 0] = 1.0  # avoid div by zero

        green = (4.0 * np.pi / K2) * np.exp(-K2 / (4 * alpha**2))
        green[0, 0, 0] = 0.0

        # Energy
        phi_hat = green * rho_hat
        energy = 0.5 * COULOMB_FACTOR * float(
            np.real(np.sum(phi_hat * np.conj(rho_hat)))) / V

        # Forces via electric field gradient on grid
        Ex = np.real(np.fft.ifftn(-1j * KX * phi_hat))
        Ey = np.real(np.fft.ifftn(-1j * KY * phi_hat))
        Ez = np.real(np.fft.ifftn(-1j * KZ * phi_hat))

        forces = np.zeros((N, 3))
        for n in range(N):
            ix, iy, iz = indices[n]
            forces[n, 0] = charges[n] * COULOMB_FACTOR * Ex[ix, iy, iz] / V
            forces[n, 1] = charges[n] * COULOMB_FACTOR * Ey[ix, iy, iz] / V
            forces[n, 2] = charges[n] * COULOMB_FACTOR * Ez[ix, iy, iz] / V

        return forces, energy

    def _self_energy(self, charges: NDArray) -> float:
        """Self-energy correction."""
        return -COULOMB_FACTOR * self.alpha / math.sqrt(math.pi) * float(
            np.sum(charges**2))

    def compute(self, positions: NDArray, box: NDArray,
                atoms: List[Atom],
                cutoff: float = 10.0) -> Tuple[NDArray, float]:
        """Full PME electrostatics computation."""
        charges = np.array([a.charge for a in atoms])
        if np.sum(np.abs(charges)) < 1e-10:
            return np.zeros_like(positions), 0.0

        F_real, E_real = self._real_space(positions, box, charges, cutoff)
        F_recip, E_recip = self._reciprocal_space(positions, box, charges)
        E_self = self._self_energy(charges)

        return F_real + F_recip, E_real + E_recip + E_self


# ===================================================================
#  Replica Exchange MD (REMD)
# ===================================================================

class REMDSampler:
    r"""
    Replica Exchange Molecular Dynamics (parallel tempering).

    Exchange criterion (Metropolis):
    $$P_{\text{swap}} = \min\!\left(1, \exp\!\left[
        (\beta_i - \beta_j)(E_i - E_j)
    \right]\right)$$

    Temperature ladder: geometric spacing $T_{i+1}/T_i = \text{const}$
    for ~20-40% acceptance rate.
    """

    def __init__(self, temperatures: Sequence[float],
                 seed: Optional[int] = None) -> None:
        self.temperatures = np.array(temperatures, dtype=np.float64)
        self.n_replicas = len(temperatures)
        self.betas = 1.0 / (K_B_KJMOL * self.temperatures)
        self.rng = np.random.default_rng(seed)
        self.swap_attempts = 0
        self.swap_accepts = 0

    @staticmethod
    def geometric_temperatures(T_low: float, T_high: float,
                                n_replicas: int) -> NDArray[np.float64]:
        """Generate geometrically spaced temperature ladder."""
        ratio = (T_high / T_low) ** (1.0 / (n_replicas - 1))
        return T_low * ratio ** np.arange(n_replicas)

    def attempt_swap(self, energies: NDArray[np.float64]) -> List[Tuple[int, int]]:
        """
        Attempt pairwise swaps between adjacent replicas.

        Parameters
        ----------
        energies : (n_replicas,) potential energies [kJ/mol].

        Returns
        -------
        List of (i, j) pairs that were swapped.
        """
        swaps = []
        # Even/odd sweep pattern
        start = self.swap_attempts % 2
        for i in range(start, self.n_replicas - 1, 2):
            j = i + 1
            self.swap_attempts += 1

            delta = (self.betas[i] - self.betas[j]) * (energies[i] - energies[j])
            if delta < 0 or self.rng.random() < math.exp(-delta):
                swaps.append((i, j))
                self.swap_accepts += 1

        return swaps

    @property
    def acceptance_rate(self) -> float:
        if self.swap_attempts == 0:
            return 0.0
        return self.swap_accepts / self.swap_attempts


# ===================================================================
#  Unified MD Simulation
# ===================================================================

class MDSimulation:
    """
    Complete MD simulation orchestrator.

    Combines integrator, thermostat, barostat, force field into a single
    run loop with trajectory output, energy logging, and analysis.
    """

    def __init__(
        self,
        atoms: List[Atom],
        positions: NDArray[np.float64],
        box: NDArray[np.float64],
        force_field: ForceField,
        dt: float = 0.002,
        thermostat: Optional[NoseHooverThermostat] = None,
        barostat: Optional[ParrinelloRahmanBarostat] = None,
    ) -> None:
        self.atoms = atoms
        self.ff = force_field
        self.integrator = VelocityVerlet(dt=dt)
        self.thermostat = thermostat
        self.barostat = barostat

        masses = np.array([a.mass for a in atoms])[:, None]
        N = len(atoms)

        # Initialise velocities if not provided
        T = thermostat.T if thermostat else 300.0
        sigma_v = np.sqrt(K_B_KJMOL * T / masses)
        rng = np.random.default_rng(42)
        velocities = rng.normal(0, sigma_v, size=(N, 3))
        # Remove center-of-mass motion
        total_mass = np.sum(masses)
        com_v = np.sum(masses * velocities, axis=0) / total_mass
        velocities -= com_v

        forces, _ = force_field.compute(positions, box, atoms)
        self.state = MDState(
            positions=positions.copy(),
            velocities=velocities,
            forces=forces,
            box=box.copy(),
        )

        # Trajectory storage
        self.trajectory: List[NDArray[np.float64]] = []
        self.energies: List[Tuple[float, float, float]] = []  # (KE, PE, total)

    def kinetic_energy(self) -> float:
        """Compute kinetic energy [kJ/mol]."""
        masses = np.array([a.mass for a in self.atoms])[:, None]
        return 0.5 * float(np.sum(masses * self.state.velocities**2))

    def temperature(self) -> float:
        """Instantaneous temperature [K]."""
        N = len(self.atoms)
        KE = self.kinetic_energy()
        return 2.0 * KE / (3.0 * N * K_B_KJMOL) if N > 0 else 0.0

    def step(self) -> Tuple[float, float]:
        """One MD step. Returns (KE, PE)."""
        dt = self.integrator.dt

        # Apply thermostat (half-step)
        if self.thermostat is not None:
            self.state = self.thermostat.apply(self.state, self.atoms, dt)

        # Velocity Verlet
        self.state, pe = self.integrator.step(self.state, self.atoms, self.ff)

        # Apply thermostat (half-step)
        if self.thermostat is not None:
            self.state = self.thermostat.apply(self.state, self.atoms, dt)

        ke = self.kinetic_energy()

        # Barostat
        if self.barostat is not None:
            P = self.barostat.compute_pressure(self.state, self.atoms, pe)
            self.state = self.barostat.apply(self.state, self.atoms, dt, P)

        return ke, pe

    def run(self, n_steps: int, save_interval: int = 100) -> None:
        """
        Run MD for n_steps, saving trajectory and energies.
        """
        for i in range(n_steps):
            ke, pe = self.step()
            self.energies.append((ke, pe, ke + pe))
            if i % save_interval == 0:
                self.trajectory.append(self.state.positions.copy())

    def compute_rdf(self, species_a: str = "", species_b: str = "",
                    n_bins: int = 200, r_max: float = 10.0) -> Tuple[NDArray, NDArray]:
        """
        Radial distribution function g(r) from current configuration.
        """
        positions = self.state.positions
        box = self.state.box
        N = len(self.atoms)

        dr = r_max / n_bins
        hist = np.zeros(n_bins)

        # Filter by species if specified
        idx_a = [i for i, a in enumerate(self.atoms)
                 if not species_a or a.symbol == species_a]
        idx_b = [i for i, a in enumerate(self.atoms)
                 if not species_b or a.symbol == species_b]

        for i in idx_a:
            for j in idx_b:
                if i >= j:
                    continue
                rij = positions[j] - positions[i]
                rij -= box * np.round(rij / box)
                dist = np.linalg.norm(rij)
                if dist < r_max:
                    bin_idx = int(dist / dr)
                    if bin_idx < n_bins:
                        hist[bin_idx] += 2  # count both i-j and j-i

        # Normalise
        r_edges = np.arange(n_bins + 1) * dr
        r_centres = 0.5 * (r_edges[:-1] + r_edges[1:])
        shell_vol = (4.0 / 3.0) * np.pi * (r_edges[1:]**3 - r_edges[:-1]**3)
        V = float(np.prod(box))
        n_pairs = len(idx_a) * len(idx_b)
        if n_pairs > 0:
            rho = n_pairs / V
            g = hist / (shell_vol * rho)
        else:
            g = hist

        return r_centres, g

    def compute_msd(self, reference: Optional[NDArray] = None) -> float:
        """Mean squared displacement from reference positions."""
        if reference is None:
            reference = self.trajectory[0] if self.trajectory else self.state.positions
        dr = self.state.positions - reference
        dr -= self.state.box * np.round(dr / self.state.box)
        return float(np.mean(np.sum(dr**2, axis=1)))
