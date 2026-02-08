"""
Multiscale Methods: FE² concurrent multiscale, computational homogenisation,
quasi-continuum, hierarchical bridging.

Upgrades domain XVIII.7.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Representative Volume Element (RVE) + Homogenisation
# ---------------------------------------------------------------------------

class RVEHomogenisation:
    r"""
    First-order computational homogenisation for heterogeneous materials.

    Macro strain ε̄ applied to RVE boundary (periodic BC):
    $$\mathbf{u}(\mathbf{x}) = \bar{\boldsymbol{\varepsilon}}\cdot\mathbf{x}
      + \tilde{\mathbf{u}}(\mathbf{x})$$

    where ũ is the fluctuation field (periodic).

    Effective stiffness: $\bar{C}_{ijkl} = \frac{1}{|V|}\int_V C_{ijkl}\,dV$
    (Voigt) or from actual RVE BVP solution.

    Bounds:
    - Voigt (upper): uniform strain assumption
    - Reuss (lower): uniform stress assumption
    - Hashin-Shtrikman: tighter bounds using variational principles
    """

    def __init__(self, n_phases: int = 2) -> None:
        self.n_phases = n_phases
        self.C_phases: List[NDArray[np.float64]] = []  # 6×6 stiffness
        self.volume_fractions: List[float] = []

    def add_phase(self, C: NDArray[np.float64], vf: float) -> None:
        """Add a material phase with 6×6 Voigt stiffness and volume fraction."""
        if C.shape != (6, 6):
            raise ValueError("Stiffness must be 6×6 Voigt notation")
        self.C_phases.append(C.copy())
        self.volume_fractions.append(vf)

    def voigt_average(self) -> NDArray[np.float64]:
        """Upper bound: C̄_V = Σ f_i C_i."""
        C_eff = np.zeros((6, 6))
        for C, f in zip(self.C_phases, self.volume_fractions):
            C_eff += f * C
        return C_eff

    def reuss_average(self) -> NDArray[np.float64]:
        """Lower bound: S̄_R = Σ f_i S_i, C̄_R = S̄_R⁻¹."""
        S_eff = np.zeros((6, 6))
        for C, f in zip(self.C_phases, self.volume_fractions):
            S_eff += f * np.linalg.inv(C)
        return np.linalg.inv(S_eff)

    def hill_average(self) -> NDArray[np.float64]:
        """Hill (VRH) average: C̄_H = (C̄_V + C̄_R)/2."""
        return 0.5 * (self.voigt_average() + self.reuss_average())

    def hashin_shtrikman_bounds(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Hashin-Shtrikman bounds for two-phase isotropic composite.

        Returns (C_lower, C_upper) as 6×6 matrices.
        """
        if len(self.C_phases) != 2:
            raise ValueError("HS bounds implemented for 2 phases only")

        def _isotropic_moduli(C: NDArray) -> Tuple[float, float]:
            K = (C[0, 0] + C[1, 1] + C[2, 2]
                 + 2 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9.0
            G = (C[0, 0] + C[1, 1] + C[2, 2]
                 - C[0, 1] - C[0, 2] - C[1, 2]) / 15.0 + (C[3, 3] + C[4, 4] + C[5, 5]) / 5.0
            return K, G

        K1, G1 = _isotropic_moduli(self.C_phases[0])
        K2, G2 = _isotropic_moduli(self.C_phases[1])
        f1, f2 = self.volume_fractions[0], self.volume_fractions[1]

        # Identify stiffer phase (for upper bound, use stiffer as reference)
        if K1 < K2:
            K_soft, K_stiff = K1, K2
            G_soft, G_stiff = G1, G2
            f_soft, f_stiff = f1, f2
        else:
            K_soft, K_stiff = K2, K1
            G_soft, G_stiff = G2, G1
            f_soft, f_stiff = f2, f1

        # Lower bound (softer phase as reference)
        denom_K_lo = 1.0 / (K_stiff - K_soft + 1e-30) + 3 * f_soft / (3 * K_soft + 4 * G_soft)
        K_lo = K_soft + f_stiff / denom_K_lo
        denom_G_lo = 1.0 / (G_stiff - G_soft + 1e-30) + 6 * f_soft * (K_soft + 2 * G_soft) / (5 * G_soft * (3 * K_soft + 4 * G_soft))
        G_lo = G_soft + f_stiff / denom_G_lo

        # Upper bound (stiffer phase as reference)
        denom_K_up = 1.0 / (K_soft - K_stiff + 1e-30) + 3 * f_stiff / (3 * K_stiff + 4 * G_stiff)
        K_up = K_stiff + f_soft / denom_K_up
        denom_G_up = 1.0 / (G_soft - G_stiff + 1e-30) + 6 * f_stiff * (K_stiff + 2 * G_stiff) / (5 * G_stiff * (3 * K_stiff + 4 * G_stiff))
        G_up = G_stiff + f_soft / denom_G_up

        def _build_isotropic_C(K: float, G: float) -> NDArray:
            lam = K - 2 * G / 3
            C = np.zeros((6, 6))
            for i in range(3):
                for j in range(3):
                    C[i, j] = lam
                C[i, i] += 2 * G
            for i in range(3, 6):
                C[i, i] = G
            return C

        return _build_isotropic_C(K_lo, G_lo), _build_isotropic_C(K_up, G_up)


# ---------------------------------------------------------------------------
#  FE² Concurrent Multiscale
# ---------------------------------------------------------------------------

@dataclass
class MicroState:
    """State of a microscale RVE at one macro integration point."""
    strain: NDArray[np.float64]           # macro strain applied
    stress: NDArray[np.float64]           # homogenised stress
    tangent: NDArray[np.float64]          # consistent tangent 6×6
    converged: bool = False


class FE2Solver:
    r"""
    FE² concurrent multiscale: nested FE at macro and micro scales.

    At each macro Gauss point, an RVE problem is solved:
    1. Apply macro strain ε̄ as BC on RVE
    2. Solve micro BVP (micro FE or FFT)
    3. Return volume-averaged stress σ̄ and consistent tangent C̄

    This implementation uses a simple 1D bar model for illustration
    with heterogeneous Young's moduli (composite bar).
    """

    def __init__(self, L_macro: float, n_elem_macro: int,
                 n_elem_micro: int = 20) -> None:
        """
        Parameters
        ----------
        L_macro : Macro domain length (m).
        n_elem_macro : Number of macro elements.
        n_elem_micro : Number of micro elements per RVE.
        """
        self.L_macro = L_macro
        self.n_macro = n_elem_macro
        self.n_micro = n_elem_micro
        self.rve_moduli: NDArray[np.float64] = np.ones(n_elem_micro) * 200e9

    def set_rve_moduli(self, E_array: NDArray[np.float64]) -> None:
        """Set Young's modulus for each micro element."""
        if len(E_array) != self.n_micro:
            raise ValueError(f"Expected {self.n_micro} moduli, got {len(E_array)}")
        self.rve_moduli = E_array.copy()

    def solve_rve(self, macro_strain: float) -> MicroState:
        """Solve 1D RVE under uniform macro strain.

        Periodic BC: u_fluctuation periodic.
        """
        n = self.n_micro
        h = 1.0 / n  # normalised RVE domain [0,1]

        # Assembly: K u = f with periodic BC
        # For 1D: each element has stiffness E_i A / h
        # With uniform strain ε̄: displacement u = ε̄ x + ũ(x)
        # ũ periodic → ũ(0) = ũ(1)

        K = np.zeros((n, n))
        for i in range(n):
            Ei = self.rve_moduli[i]
            ke = Ei / h
            ip = i
            jp = (i + 1) % n  # periodic
            K[ip, ip] += ke
            K[ip, jp] -= ke
            K[jp, ip] -= ke
            K[jp, jp] += ke

        # RHS: force from macro strain
        f = np.zeros(n)
        for i in range(n):
            Ei = self.rve_moduli[i]
            f[i] += Ei * macro_strain  # - Ei * macro_strain for element i to i+1
            f[(i + 1) % n] -= Ei * macro_strain

        # Fix one DOF for uniqueness (ũ₀ = 0)
        K[0, :] = 0
        K[:, 0] = 0
        K[0, 0] = 1.0
        f[0] = 0.0

        u_fluct = np.linalg.solve(K, f)

        # Compute element strains and stresses
        strains = np.zeros(n)
        stresses = np.zeros(n)
        for i in range(n):
            j = (i + 1) % n
            strains[i] = macro_strain + (u_fluct[j] - u_fluct[i]) / h
            stresses[i] = self.rve_moduli[i] * strains[i]

        # Homogenised stress = volume average
        sigma_bar = float(np.mean(stresses))

        # Consistent tangent = dσ̄/dε̄
        # For linear: C̄ = σ̄/ε̄ (if ε̄ ≠ 0)
        C_bar = sigma_bar / macro_strain if abs(macro_strain) > 1e-30 else float(np.mean(self.rve_moduli))

        return MicroState(
            strain=np.array([macro_strain]),
            stress=np.array([sigma_bar]),
            tangent=np.array([[C_bar]]),
            converged=True,
        )

    def solve_macro(self, F_applied: float) -> Tuple[NDArray, NDArray]:
        """Solve macro problem with nested RVE at each element.

        Parameters
        ----------
        F_applied : Applied force at right end (N).

        Returns
        -------
        (displacements, element_stresses).
        """
        n = self.n_macro
        h = self.L_macro / n

        # Newton-Raphson on macro scale
        u = np.zeros(n + 1)
        u[0] = 0.0  # fixed left end

        for newton_iter in range(20):
            K_global = np.zeros((n + 1, n + 1))
            f_int = np.zeros(n + 1)

            for e in range(n):
                # Macro strain in element e
                eps_macro = (u[e + 1] - u[e]) / h

                # Solve RVE
                micro = self.solve_rve(eps_macro if abs(eps_macro) > 1e-30 else 1e-12)
                sigma_e = float(micro.stress[0])
                C_e = float(micro.tangent[0, 0])

                # Element stiffness
                ke = C_e / h
                K_global[e, e] += ke
                K_global[e, e + 1] -= ke
                K_global[e + 1, e] -= ke
                K_global[e + 1, e + 1] += ke

                # Internal force
                f_int[e] -= sigma_e
                f_int[e + 1] += sigma_e

            # External force
            f_ext = np.zeros(n + 1)
            f_ext[n] = F_applied

            # Residual
            R = f_ext - f_int
            R[0] = 0  # BC

            if np.linalg.norm(R) < 1e-10:
                break

            # Solve
            K_global[0, :] = 0
            K_global[:, 0] = 0
            K_global[0, 0] = 1.0
            R[0] = 0

            du = np.linalg.solve(K_global, R)
            u += du

        # Compute final stresses
        stresses = np.zeros(n)
        for e in range(n):
            eps = (u[e + 1] - u[e]) / h
            micro = self.solve_rve(eps if abs(eps) > 1e-12 else 1e-12)
            stresses[e] = float(micro.stress[0])

        return u, stresses


# ---------------------------------------------------------------------------
#  Quasi-Continuum Method
# ---------------------------------------------------------------------------

class QuasiContinuum:
    r"""
    Quasi-continuum method: adaptive atomistic/continuum coupling.

    Full atomistic resolution near defects, coarsened FE mesh far away.

    - Repatoms: representative atoms at FE nodes
    - Summation rules: Cauchy-Born for bulk, full lattice sum near defects
    - Seamless handshake: ghost-force correction

    1D implementation with Lennard-Jones interactions.
    """

    def __init__(self, n_atoms: int, a0: float,
                 epsilon: float = 1.0, sigma_lj: float = 1.0) -> None:
        """
        Parameters
        ----------
        n_atoms : Total number of atoms.
        a0 : Equilibrium lattice spacing.
        epsilon : LJ well depth.
        sigma_lj : LJ length parameter.
        """
        self.n_atoms = n_atoms
        self.a0 = a0
        self.eps = epsilon
        self.sig = sigma_lj
        self.positions = np.arange(n_atoms, dtype=np.float64) * a0
        self.is_repatom = np.ones(n_atoms, dtype=bool)  # all atoms initially

    def set_coarsening(self, repatom_indices: NDArray[np.int64]) -> None:
        """Define repatoms (rest are interpolated)."""
        self.is_repatom[:] = False
        self.is_repatom[repatom_indices] = True

    def adaptive_coarsening(self, defect_center: int,
                              atomistic_radius: int = 10) -> None:
        """Automatic: full atoms near defect, coarsen away."""
        self.is_repatom[:] = False
        for i in range(self.n_atoms):
            dist = abs(i - defect_center)
            if dist <= atomistic_radius:
                self.is_repatom[i] = True
            elif dist <= 2 * atomistic_radius:
                if i % 2 == 0:
                    self.is_repatom[i] = True
            elif dist <= 4 * atomistic_radius:
                if i % 4 == 0:
                    self.is_repatom[i] = True
            else:
                if i % 8 == 0:
                    self.is_repatom[i] = True

        # Always include endpoints
        self.is_repatom[0] = True
        self.is_repatom[-1] = True

    def _lj_force(self, r: float) -> float:
        """LJ pair force: F = -dV/dr = 24ε/r [2(σ/r)¹² - (σ/r)⁶]."""
        sr6 = (self.sig / r)**6
        return 24.0 * self.eps / r * (2.0 * sr6**2 - sr6)

    def _lj_energy(self, r: float) -> float:
        """LJ pair energy."""
        sr6 = (self.sig / r)**6
        return 4.0 * self.eps * (sr6**2 - sr6)

    def _cauchy_born_stress(self, strain: float) -> float:
        """Cauchy-Born stress for 1D LJ chain under uniform strain."""
        r = self.a0 * (1 + strain)
        return self._lj_force(r) if r > 0.5 * self.sig else 0.0

    def total_energy(self) -> float:
        """Total energy: sum over repatoms with appropriate weights."""
        E = 0.0
        rep_indices = np.where(self.is_repatom)[0]

        for idx, i in enumerate(rep_indices[:-1]):
            j = rep_indices[idx + 1]
            r = self.positions[j] - self.positions[i]
            n_interp = j - i  # number of atoms represented

            if abs(j - i) <= 2:
                # Atomistic region: full sum
                E += self._lj_energy(r / n_interp) * n_interp
            else:
                # Continuum region: Cauchy-Born
                strain = (r - n_interp * self.a0) / (n_interp * self.a0)
                E += self._lj_energy(self.a0 * (1 + strain)) * n_interp

        return E

    def relax(self, fixed_left: bool = True, fixed_right: bool = True,
                n_steps: int = 500, dt: float = 0.001) -> float:
        """Relax repatom positions via steepest descent.

        Returns final energy.
        """
        rep_indices = np.where(self.is_repatom)[0]
        n_rep = len(rep_indices)

        for step in range(n_steps):
            forces = np.zeros(n_rep)

            for idx in range(n_rep):
                i = rep_indices[idx]

                # Left neighbour
                if idx > 0:
                    j = rep_indices[idx - 1]
                    r = self.positions[i] - self.positions[j]
                    if r > 0.5 * self.sig:
                        forces[idx] += self._lj_force(r)

                # Right neighbour
                if idx < n_rep - 1:
                    j = rep_indices[idx + 1]
                    r = self.positions[j] - self.positions[i]
                    if r > 0.5 * self.sig:
                        forces[idx] -= self._lj_force(r)

            # Apply BCs
            if fixed_left:
                forces[0] = 0.0
            if fixed_right:
                forces[-1] = 0.0

            # Update positions
            for idx in range(n_rep):
                self.positions[rep_indices[idx]] += dt * forces[idx]

            # Interpolate non-repatom positions
            for idx in range(n_rep - 1):
                i = rep_indices[idx]
                j = rep_indices[idx + 1]
                for k in range(i + 1, j):
                    t = (k - i) / (j - i)
                    self.positions[k] = (1 - t) * self.positions[i] + t * self.positions[j]

            if np.max(np.abs(forces)) < 1e-10:
                break

        return self.total_energy()


# ---------------------------------------------------------------------------
#  Hierarchical Multiscale Bridge
# ---------------------------------------------------------------------------

class HierarchicalBridge:
    r"""
    Hierarchical multiscale: information passing between scales.

    Workflow:
    1. Atomistic → fit interatomic potential parameters
    2. Mesoscale → extract constitutive law
    3. Continuum → run macro FE with fitted constitutive

    This class manages parameter passing between scale levels.
    """

    @dataclass
    class ScaleLevel:
        name: str
        parameters: Dict[str, float] = field(default_factory=dict)
        derived: Dict[str, float] = field(default_factory=dict)

    def __init__(self) -> None:
        self.levels: Dict[str, HierarchicalBridge.ScaleLevel] = {}
        self.bridges: List[Tuple[str, str, Callable]] = []

    def add_level(self, name: str, params: Dict[str, float]) -> None:
        self.levels[name] = self.ScaleLevel(name=name, parameters=params)

    def add_bridge(self, source: str, target: str,
                     transfer_func: Callable[[Dict[str, float]], Dict[str, float]]) -> None:
        """Define parameter transfer function between scales."""
        self.bridges.append((source, target, transfer_func))

    def execute(self) -> Dict[str, Dict[str, float]]:
        """Execute all bridges in order, passing parameters up-scale."""
        for source, target, func in self.bridges:
            if source not in self.levels or target not in self.levels:
                raise ValueError(f"Scale level not found: {source} or {target}")

            src_params = {**self.levels[source].parameters, **self.levels[source].derived}
            transferred = func(src_params)
            self.levels[target].derived.update(transferred)

        return {name: {**lv.parameters, **lv.derived} for name, lv in self.levels.items()}
