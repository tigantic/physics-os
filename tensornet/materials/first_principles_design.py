"""
First-Principles Materials Design — DFT-based property prediction, phase stability,
phonon spectra, elastic constants, alloy screening.

Domain XIV.1 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Physical Constants
# ---------------------------------------------------------------------------

EV_TO_J: float = 1.602e-19
BOHR_TO_ANG: float = 0.5292
RY_TO_EV: float = 13.606
K_B_EV: float = 8.617e-5  # eV/K


# ---------------------------------------------------------------------------
#  Total Energy & Equation of State
# ---------------------------------------------------------------------------

class BirchMurnaghanEOS:
    r"""
    Birch-Murnaghan equation of state (3rd order).

    $$E(V) = E_0 + \frac{9V_0 B_0}{16}
      \left\{
        \left[\left(\frac{V_0}{V}\right)^{2/3}-1\right]^3 B_0'
        + \left[\left(\frac{V_0}{V}\right)^{2/3}-1\right]^2
        \left[6-4\left(\frac{V_0}{V}\right)^{2/3}\right]
      \right\}$$

    P(V):
    $$P = \frac{3B_0}{2}\left[\left(\frac{V_0}{V}\right)^{7/3}
      - \left(\frac{V_0}{V}\right)^{5/3}\right]
      \left\{1 + \frac{3}{4}(B_0'-4)\left[\left(\frac{V_0}{V}\right)^{2/3}-1\right]\right\}$$
    """

    def __init__(self, V0: float = 75.0, E0: float = -8.5,
                 B0: float = 100.0, B0p: float = 4.0) -> None:
        """
        V0: equilibrium volume (ų/atom).
        E0: energy minimum (eV/atom).
        B0: bulk modulus (GPa).
        B0p: pressure derivative of B0.
        """
        self.V0 = V0
        self.E0 = E0
        self.B0 = B0 * 1e9 * 1e-30 / EV_TO_J  # GPa → eV/ų
        self.B0p = B0p

    def energy(self, V: float) -> float:
        """E(V) in eV/atom."""
        eta = (self.V0 / V)**(2 / 3)
        f = eta - 1
        return self.E0 + 9 * self.V0 * self.B0 / 16 * (
            f**3 * self.B0p + f**2 * (6 - 4 * eta))

    def pressure(self, V: float) -> float:
        """P(V) in eV/ų."""
        eta = (self.V0 / V)**(2 / 3)
        f = eta - 1
        return 1.5 * self.B0 * (eta**(7 / 2) - eta**(5 / 2)) * (
            1 + 0.75 * (self.B0p - 4) * f)

    def fit(self, volumes: NDArray, energies: NDArray) -> Dict[str, float]:
        """Fit E(V) data to 3rd-order Birch-Murnaghan.

        Returns {'V0', 'E0', 'B0_GPa', 'B0p'}.
        """
        from scipy.optimize import curve_fit

        def bm3(V: NDArray, V0: float, E0: float, B0: float, B0p: float) -> NDArray:
            eta = (V0 / V)**(2 / 3)
            f = eta - 1
            return E0 + 9 * V0 * B0 / 16 * (f**3 * B0p + f**2 * (6 - 4 * eta))

        p0 = [volumes[np.argmin(energies)], min(energies), 0.05, 4.0]
        popt, _ = curve_fit(bm3, volumes, energies, p0=p0, maxfev=5000)
        return {
            'V0': popt[0],
            'E0': popt[1],
            'B0_GPa': popt[2] * EV_TO_J / 1e-30 / 1e9,
            'B0p': popt[3],
        }


# ---------------------------------------------------------------------------
#  Elastic Constants from Strain-Energy
# ---------------------------------------------------------------------------

class ElasticConstants:
    r"""
    Compute elastic constants from strain-energy curves (Voigt notation).

    For cubic: C₁₁, C₁₂, C₄₄.

    Bulk modulus: $B = (C_{11} + 2C_{12})/3$
    Shear modulus (Voigt): $G_V = (C_{11} - C_{12} + 3C_{44})/5$
    Zener anisotropy ratio: $A = 2C_{44}/(C_{11} - C_{12})$

    Elastic energy density:
    $$E = \frac{1}{2}C_{ijkl}\varepsilon_{ij}\varepsilon_{kl}$$
    """

    def __init__(self, C11: float = 165.0, C12: float = 65.0,
                 C44: float = 80.0) -> None:
        """Elastic constants in GPa."""
        self.C11 = C11
        self.C12 = C12
        self.C44 = C44

    def bulk_modulus(self) -> float:
        """B = (C₁₁ + 2C₁₂)/3 (GPa)."""
        return (self.C11 + 2 * self.C12) / 3

    def shear_modulus_voigt(self) -> float:
        """G_V = (C₁₁ − C₁₂ + 3C₄₄)/5 (GPa)."""
        return (self.C11 - self.C12 + 3 * self.C44) / 5

    def shear_modulus_reuss(self) -> float:
        """G_R = 5(C₁₁ − C₁₂)C₄₄ / (4C₄₄ + 3(C₁₁ − C₁₂)) (GPa)."""
        d = self.C11 - self.C12
        return 5 * d * self.C44 / (4 * self.C44 + 3 * d)

    def shear_modulus_hill(self) -> float:
        """G_H = (G_V + G_R)/2."""
        return (self.shear_modulus_voigt() + self.shear_modulus_reuss()) / 2

    def youngs_modulus(self) -> float:
        """E = 9BG/(3B + G) (GPa)."""
        B = self.bulk_modulus()
        G = self.shear_modulus_hill()
        return 9 * B * G / (3 * B + G)

    def poisson_ratio(self) -> float:
        """ν = (3B − 2G)/(6B + 2G)."""
        B = self.bulk_modulus()
        G = self.shear_modulus_hill()
        return (3 * B - 2 * G) / (6 * B + 2 * G)

    def zener_ratio(self) -> float:
        """A = 2C₄₄/(C₁₁ − C₁₂)."""
        return 2 * self.C44 / (self.C11 - self.C12)

    def pugh_ratio(self) -> float:
        """B/G — ductile if > 1.75."""
        return self.bulk_modulus() / self.shear_modulus_hill()

    def cauchy_pressure(self) -> float:
        """C₁₂ − C₄₄ — positive indicates metallic bonding."""
        return self.C12 - self.C44

    def stiffness_tensor_cubic(self) -> NDArray:
        """6×6 Voigt stiffness matrix for cubic symmetry."""
        C = np.zeros((6, 6))
        C[0, 0] = C[1, 1] = C[2, 2] = self.C11
        C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = self.C12
        C[3, 3] = C[4, 4] = C[5, 5] = self.C44
        return C


# ---------------------------------------------------------------------------
#  Phase Stability (Convex Hull)
# ---------------------------------------------------------------------------

class ConvexHullStability:
    r"""
    Thermodynamic phase stability via convex hull construction.

    Formation energy:
    $$\Delta E_f = E_{\text{compound}} - \sum_i x_i E_i^{\text{ref}}$$

    A compound is stable if its formation energy lies on the convex hull
    of the composition-energy space. Distance above hull = decomposition energy.
    """

    def __init__(self) -> None:
        self.compositions: List[float] = []
        self.energies: List[float] = []
        self.labels: List[str] = []

    def add_phase(self, x: float, energy: float, label: str = '') -> None:
        """Add a phase at composition x with formation energy."""
        self.compositions.append(x)
        self.energies.append(energy)
        self.labels.append(label)

    def compute_hull(self) -> Tuple[NDArray, NDArray, List[int]]:
        """Compute lower convex hull.

        Returns (hull_x, hull_E, hull_indices).
        """
        n = len(self.compositions)
        x = np.array(self.compositions)
        E = np.array(self.energies)
        idx = np.argsort(x)
        x = x[idx]
        E = E[idx]

        hull_idx = [0]
        for i in range(1, n):
            while len(hull_idx) >= 2:
                j = hull_idx[-2]
                k = hull_idx[-1]
                # Check cross product for lower hull
                if ((x[k] - x[j]) * (E[i] - E[j])
                        - (E[k] - E[j]) * (x[i] - x[j])) <= 0:
                    hull_idx.pop()
                else:
                    break
            hull_idx.append(i)

        return x[hull_idx], E[hull_idx], [int(idx[i]) for i in hull_idx]

    def distance_above_hull(self, x_query: float, E_query: float) -> float:
        """Energy above convex hull at composition x_query."""
        hx, hE, _ = self.compute_hull()
        # Linear interpolation on hull
        if x_query <= hx[0]:
            return E_query - hE[0]
        if x_query >= hx[-1]:
            return E_query - hE[-1]
        for i in range(len(hx) - 1):
            if hx[i] <= x_query <= hx[i + 1]:
                frac = (x_query - hx[i]) / (hx[i + 1] - hx[i])
                E_hull = hE[i] + frac * (hE[i + 1] - hE[i])
                return E_query - E_hull
        return 0.0


# ---------------------------------------------------------------------------
#  Phonon Spectrum (Force-Constants)
# ---------------------------------------------------------------------------

class PhononDispersion1D:
    r"""
    1D phonon dispersion from harmonic force constants.

    Dynamical matrix (monoatomic, nearest-neighbour):
    $$D(q) = \frac{2\Phi_1}{M}\left[1 - \cos(qa)\right]$$

    $$\omega(q) = 2\sqrt{\frac{\Phi_1}{M}}\left|\sin\left(\frac{qa}{2}\right)\right|$$

    For diatomic chain (mass M₁, M₂):
    $$\omega^2 = \frac{\Phi_1}{M_r}\left[1 \pm \sqrt{1 - \frac{4M_r^2}{M_1 M_2}\sin^2(qa/2)}\right]$$
    where $M_r = (M_1 + M_2)/2$.
    """

    def __init__(self, M: float = 28.0, Phi1: float = 5.0,
                 a: float = 5.43) -> None:
        """
        M: atomic mass (amu).
        Phi1: nearest-neighbour force constant (eV/ų).
        a: lattice constant (Å).
        """
        self.M = M * 1.661e-27  # kg
        self.Phi1_SI = Phi1 * EV_TO_J * 1e20  # eV/ų → N/m
        self.a = a * 1e-10  # Å → m

    def monoatomic_dispersion(self, q: NDArray) -> NDArray:
        """ω(q) for 1D monoatomic chain (rad/s)."""
        return 2 * np.sqrt(self.Phi1_SI / self.M) * np.abs(np.sin(q * self.a / 2))

    def diatomic_dispersion(self, q: NDArray, M2_amu: float = 16.0) -> Tuple[NDArray, NDArray]:
        """Acoustic and optical branches for diatomic chain."""
        M2 = M2_amu * 1.661e-27
        M1 = self.M
        sum_M = M1 + M2
        prod_M = M1 * M2

        C = self.Phi1_SI
        sin2 = np.sin(q * self.a / 2)**2

        discriminant = np.sqrt(np.maximum(1 - 4 * prod_M / sum_M**2 * sin2, 0))
        omega_sq_plus = C / (prod_M / sum_M) * (1 + discriminant)
        omega_sq_minus = C / (prod_M / sum_M) * (1 - discriminant)

        return np.sqrt(np.maximum(omega_sq_minus, 0)), np.sqrt(omega_sq_plus)

    def debye_temperature(self) -> float:
        """Θ_D from max phonon frequency.

        Θ_D = ℏ ω_max / k_B
        """
        omega_max = 2 * math.sqrt(self.Phi1_SI / self.M)
        hbar = 1.055e-34
        kB = 1.381e-23
        return hbar * omega_max / kB

    def dos_monoatomic(self, n_bins: int = 200) -> Tuple[NDArray, NDArray]:
        """Phonon density of states g(ω) for monoatomic chain."""
        q = np.linspace(-math.pi / self.a, math.pi / self.a, 10000)
        omega_all = self.monoatomic_dispersion(q)
        omega_max = float(np.max(omega_all))
        bins = np.linspace(0, omega_max * 1.01, n_bins + 1)
        dos, _ = np.histogram(omega_all, bins=bins, density=True)
        omega_centres = 0.5 * (bins[:-1] + bins[1:])
        return omega_centres, dos
