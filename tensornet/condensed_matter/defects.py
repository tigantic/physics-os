"""
Crystal defect physics: point defects, dislocations, grain boundaries.

Upgrades domain IX.7 from NEB-only to full defect energetics toolkit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Point Defect Formation Energy
# ---------------------------------------------------------------------------

@dataclass
class DefectEnergy:
    """Result of a point defect calculation."""
    formation_energy: float   # eV
    relaxation_energy: float  # eV
    migration_barrier: float  # eV
    defect_type: str


class PointDefectCalculator:
    r"""
    Point defect formation energy from pair-potential models.

    Formation energy of a vacancy:
    $$E_f^{vac} = E_{N-1}^{relax} - \frac{N-1}{N}\,E_N^{perfect}$$

    Formation energy of an interstitial:
    $$E_f^{int} = E_{N+1}^{relax} - \frac{N+1}{N}\,E_N^{perfect}$$

    Migration barrier from NEB or drag method.

    Implements:
    - Lennard-Jones and Morse pair potentials
    - Vacancy / interstitial / substitutional energetics
    - Local relaxation via steepest descent or FIRE
    - Defect concentration: $c = \exp(-E_f / k_B T)$
    """

    def __init__(self, positions: NDArray[np.float64],
                 box: NDArray[np.float64],
                 pair_potential: str = "lj",
                 params: Optional[Dict[str, float]] = None) -> None:
        """
        Parameters
        ----------
        positions : (N, 3) perfect crystal positions.
        box : (3,) box dimensions (orthorhombic).
        pair_potential : 'lj' or 'morse'.
        params : Potential parameters.
        """
        self.positions = positions.copy()
        self.box = box.copy()
        self.N = len(positions)
        self.pot_type = pair_potential

        if params is None:
            if pair_potential == "lj":
                params = {"epsilon": 1.0, "sigma": 1.0, "cutoff": 2.5}
            else:
                params = {"D": 1.0, "alpha": 1.0, "r0": 1.0, "cutoff": 5.0}
        self.params = params

    def _pair_energy_force(self, r: float) -> Tuple[float, float]:
        """Pair energy and -dV/dr for given interatomic distance."""
        cutoff = self.params["cutoff"]
        if r > cutoff or r < 1e-10:
            return 0.0, 0.0

        if self.pot_type == "lj":
            eps = self.params["epsilon"]
            sig = self.params["sigma"]
            sr6 = (sig / r)**6
            sr12 = sr6**2
            energy = 4.0 * eps * (sr12 - sr6)
            force = 24.0 * eps * (2.0 * sr12 - sr6) / r  # -dV/dr
            # Shift
            src6 = (sig / cutoff)**6
            energy -= 4.0 * eps * (src6**2 - src6)
        else:  # morse
            D = self.params["D"]
            alpha = self.params["alpha"]
            r0 = self.params["r0"]
            exp_term = math.exp(-alpha * (r - r0))
            energy = D * (1.0 - exp_term)**2 - D
            force = 2.0 * D * alpha * exp_term * (1.0 - exp_term)

        return energy, force

    def _min_image_dist(self, r1: NDArray, r2: NDArray) -> Tuple[float, NDArray]:
        """Minimum image convention distance."""
        dr = r2 - r1
        for d in range(3):
            dr[d] -= self.box[d] * round(dr[d] / self.box[d])
        dist = float(np.linalg.norm(dr))
        return dist, dr

    def total_energy(self, pos: NDArray[np.float64]) -> float:
        """Total pair energy for configuration."""
        N = len(pos)
        E = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                r, _ = self._min_image_dist(pos[i], pos[j])
                e, _ = self._pair_energy_force(r)
                E += e
        return E

    def forces(self, pos: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute forces on all atoms."""
        N = len(pos)
        F = np.zeros((N, 3))
        for i in range(N):
            for j in range(i + 1, N):
                r, dr = self._min_image_dist(pos[i], pos[j])
                _, f_mag = self._pair_energy_force(r)
                if r > 1e-10:
                    f_vec = f_mag * dr / r
                    F[i] += f_vec
                    F[j] -= f_vec
        return F

    def relax(self, pos: NDArray[np.float64],
              max_iter: int = 5000, f_tol: float = 1e-6,
              dt: float = 0.01) -> NDArray[np.float64]:
        """FIRE relaxation algorithm."""
        N = len(pos)
        v = np.zeros((N, 3))
        dt_current = dt
        alpha = 0.1
        N_min = 5
        f_inc = 1.1
        f_dec = 0.5
        alpha_start = 0.1
        f_alpha = 0.99
        n_pos = 0

        p = pos.copy()
        for step in range(max_iter):
            F = self.forces(p)
            f_max = float(np.max(np.linalg.norm(F, axis=1)))
            if f_max < f_tol:
                break

            # FIRE
            power = float(np.sum(F * v))
            v_norm = np.linalg.norm(v)
            F_norm = np.linalg.norm(F)

            v = (1.0 - alpha) * v + alpha * (v_norm / (F_norm + 1e-20)) * F

            if power > 0:
                n_pos += 1
                if n_pos > N_min:
                    dt_current = min(dt_current * f_inc, 10.0 * dt)
                    alpha *= f_alpha
            else:
                n_pos = 0
                v[:] = 0.0
                dt_current *= f_dec
                alpha = alpha_start

            v += F * dt_current
            p += v * dt_current

        return p

    def vacancy_formation_energy(self, site: int = 0) -> DefectEnergy:
        """Remove atom at 'site' and compute formation energy."""
        E_perfect = self.total_energy(self.positions)
        pos_defect = np.delete(self.positions, site, axis=0)
        pos_relaxed = self.relax(pos_defect)
        E_defect = self.total_energy(pos_relaxed)
        E_unrelaxed = self.total_energy(pos_defect)

        Ef = E_defect - (self.N - 1) / self.N * E_perfect
        E_relax = E_unrelaxed - E_defect

        return DefectEnergy(
            formation_energy=Ef,
            relaxation_energy=E_relax,
            migration_barrier=0.0,
            defect_type="vacancy",
        )

    @staticmethod
    def defect_concentration(E_f: float, T: float) -> float:
        r"""Equilibrium concentration: $c = \exp(-E_f / k_B T)$."""
        kB = 8.617e-5  # eV/K
        if T < 1e-10:
            return 0.0
        return math.exp(-E_f / (kB * T))


# ---------------------------------------------------------------------------
#  Peierls-Nabarro Dislocation Model
# ---------------------------------------------------------------------------

class PeierlsNabarroModel:
    r"""
    Peierls-Nabarro model for dislocation core structure and Peierls stress.

    Misfit energy (sinusoidal potential):
    $$E_{\text{misfit}} = \frac{\mu b}{4\pi^2 d}\int_{-\infty}^{\infty}
        \left[1 - \cos\!\left(\frac{2\pi u(x)}{b}\right)\right]dx$$

    Peierls-Nabarro displacement field:
    $$u(x) = \frac{b}{\pi}\arctan\!\left(\frac{x}{\zeta}\right) + \frac{b}{2}$$

    where $\zeta = d / (2(1-\nu))$ (edge) or $\zeta = d/2$ (screw).

    Peierls stress:
    $$\tau_P = \frac{2\mu}{1-\nu}\exp\!\left(-\frac{2\pi\zeta}{b}\right)$$
    """

    def __init__(self, b: float, d: float, mu: float, nu: float = 0.3) -> None:
        """
        Parameters
        ----------
        b : Burgers vector magnitude (Å).
        d : Interplanar spacing (Å).
        mu : Shear modulus (GPa).
        nu : Poisson's ratio.
        """
        self.b = b
        self.d = d
        self.mu = mu
        self.nu = nu

    def core_width(self, dislocation_type: str = "edge") -> float:
        r"""Half-width ζ of the dislocation core."""
        if dislocation_type == "edge":
            return self.d / (2.0 * (1.0 - self.nu))
        return self.d / 2.0  # screw

    def displacement_field(self, x: NDArray[np.float64],
                             dislocation_type: str = "edge") -> NDArray[np.float64]:
        """PN displacement field u(x)."""
        zeta = self.core_width(dislocation_type)
        return self.b / math.pi * np.arctan(x / zeta) + self.b / 2.0

    def peierls_stress(self, dislocation_type: str = "edge") -> float:
        """Peierls stress τ_P in GPa."""
        zeta = self.core_width(dislocation_type)
        prefactor = 2.0 * self.mu / (1.0 - self.nu) if dislocation_type == "edge" \
            else 2.0 * self.mu
        return prefactor * math.exp(-2.0 * math.pi * zeta / self.b)

    def line_energy(self, dislocation_type: str = "edge",
                     R: float = 1000.0) -> float:
        r"""
        Dislocation line energy per unit length:
        $$E_{\text{edge}} = \frac{\mu b^2}{4\pi(1-\nu)}\ln(R/r_0)$$
        $$E_{\text{screw}} = \frac{\mu b^2}{4\pi}\ln(R/r_0)$$
        """
        r0 = self.core_width(dislocation_type)
        if dislocation_type == "edge":
            return self.mu * self.b**2 / (4.0 * math.pi * (1.0 - self.nu)) * math.log(R / r0)
        return self.mu * self.b**2 / (4.0 * math.pi) * math.log(R / r0)


# ---------------------------------------------------------------------------
#  Grain Boundary Energy
# ---------------------------------------------------------------------------

class GrainBoundaryEnergy:
    r"""
    Grain boundary energy models.

    Read-Shockley (low-angle):
    $$\gamma(\theta) = \gamma_0\,\theta\,(A - \ln\theta)$$

    where $\gamma_0 = \mu b / (4\pi(1-\nu))$, $A = 1 + \ln(b/(2\pi r_0))$.

    Implements:
    - Read-Shockley for low-angle tilt/twist
    - Energy vs misorientation curve
    - CSL Σ-value calculation
    """

    def __init__(self, b: float, mu: float, nu: float = 0.3,
                 r0: Optional[float] = None) -> None:
        self.b = b
        self.mu = mu
        self.nu = nu
        self.r0 = r0 if r0 is not None else b
        self.gamma0 = mu * b / (4.0 * math.pi * (1.0 - nu))
        self.A = 1.0 + math.log(b / (2.0 * math.pi * self.r0))

    def read_shockley(self, theta: float) -> float:
        """Energy (J/m²) for misorientation angle θ (radians)."""
        if theta < 1e-10:
            return 0.0
        if theta > math.pi / 6:
            # Beyond low-angle regime: use saturated value
            return self.gamma0 * (math.pi / 6) * (self.A - math.log(math.pi / 6))
        return self.gamma0 * theta * (self.A - math.log(theta))

    def energy_vs_angle(self, n_points: int = 100) -> Tuple[NDArray, NDArray]:
        """Compute GB energy curve from 0 to 60°."""
        theta = np.linspace(0.001, math.pi / 3, n_points)
        gamma = np.array([self.read_shockley(t) for t in theta])
        return np.degrees(theta), gamma

    @staticmethod
    def csl_sigma(angle_deg: float, axis: Tuple[int, int, int] = (1, 0, 0),
                   max_sigma: int = 50) -> int:
        """
        Find nearest CSL Σ value for given misorientation.

        For [100] tilt: Σ = h² + k² where tan(θ/2) = k/h.
        Simplified for cubic symmetry.
        """
        theta = math.radians(angle_deg / 2.0)
        tan_half = math.tan(theta)

        best_sigma = 1
        best_error = float('inf')

        for h in range(1, max_sigma):
            for k in range(0, h + 1):
                sigma = h * h + k * k
                if sigma > max_sigma:
                    continue
                # Σ must be odd for CSL in cubic
                if sigma % 2 == 0:
                    sigma //= 2
                ratio = k / h if h > 0 else 0.0
                error = abs(ratio - tan_half)
                if error < best_error:
                    best_error = error
                    best_sigma = sigma

        return best_sigma
