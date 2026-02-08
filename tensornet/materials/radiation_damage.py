"""
Radiation Damage — primary knock-on atom, binary collision approximation,
Frenkel pair production, displacement cascades, Wigner energy.

Domain XIV.5 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Physical Constants
# ---------------------------------------------------------------------------

EV_TO_J: float = 1.602e-19
AMU_KG: float = 1.661e-27
A0_BOHR: float = 0.5292e-10   # m
E_RYDBERG: float = 13.606     # eV


# ---------------------------------------------------------------------------
#  Kinchin-Pease / NRT Displacement Model
# ---------------------------------------------------------------------------

class NRTDisplacements:
    r"""
    NRT (Norgett-Robinson-Torrens) model for displacement production.

    Number of Frenkel pairs:
    $$N_d(E) = \begin{cases}
      0 & E < E_d \\
      1 & E_d \leq E < 2E_d/0.8 \\
      0.8\,E_{\text{dam}} / (2E_d) & E \geq 2E_d/0.8
    \end{cases}$$

    where $E_d$ = threshold displacement energy,
    $E_{\text{dam}} = E / (1 + k_L g(\epsilon))$ = damage energy
    (Lindhard electronic stopping correction).
    """

    def __init__(self, Ed: float = 40.0, Z: int = 26, A: float = 55.845) -> None:
        """
        Ed: displacement energy (eV).
        Z: atomic number.
        A: atomic mass (amu).
        """
        self.Ed = Ed
        self.Z = Z
        self.A = A

    def lindhard_partition(self, E: float) -> float:
        """Lindhard electronic energy loss partition.

        k_L = 0.1337 Z^{1/6} (Z/A)^{1/2}
        ε = E × a_L / (Z² e²)
        """
        Z = self.Z
        A = self.A
        k_L = 0.1337 * Z**(1 / 6) * (Z / A)**0.5

        a_L = 0.8853 * A0_BOHR / Z**(1 / 3)

        epsilon_L = E * a_L / (Z**2 * 14.4)  # 14.4 eV·Å = e²

        g = epsilon_L + 0.40244 * epsilon_L**(3 / 4) + 3.4008 * epsilon_L**(1 / 6)
        return E / (1 + k_L * g)

    def nrt_displacements(self, E: float) -> float:
        """Number of Frenkel pairs per PKA of energy E (eV)."""
        if E < self.Ed:
            return 0.0
        if E < 2 * self.Ed / 0.8:
            return 1.0
        E_dam = self.lindhard_partition(E)
        return 0.8 * E_dam / (2 * self.Ed)

    def athermal_recombination_corrected(self, E: float,
                                            b_arc: float = 0.3) -> float:
        """ARC-corrected NRT (Nordlund et al., 2018).

        N_d^{ARC} = (1 − c_arc) N_d^{NRT}
        ~ reduces defects by ~30% due to in-cascade recombination.
        """
        return (1 - b_arc) * self.nrt_displacements(E)

    def dpa(self, fluence: float, sigma: float, E_avg: float) -> float:
        """Displacements per atom.

        dpa = fluence × σ × N_d(E_avg)
        fluence in n/cm², σ in barn.
        """
        sigma_cm2 = sigma * 1e-24
        return fluence * sigma_cm2 * self.nrt_displacements(E_avg)


# ---------------------------------------------------------------------------
#  Binary Collision Approximation (BCA)
# ---------------------------------------------------------------------------

class BinaryCollisionApproximation:
    r"""
    Binary collision approximation for ion-solid interactions.

    Scattering angle (Rutherford + screened):
    $$\theta = \pi - 2\int_{r_{\min}}^{\infty}
      \frac{b\,dr}{r^2\sqrt{1 - V(r)/E_{\text{cm}} - b^2/r^2}}$$

    Ziegler-Biersack-Littmark (ZBL) universal potential:
    $$V(r) = \frac{Z_1 Z_2 e^2}{r}\phi(r/a_u)$$
    $$\phi(x) = 0.1818 e^{-3.2x} + 0.5099 e^{-0.9423x}
      + 0.2802 e^{-0.4029x} + 0.02817 e^{-0.2016x}$$
    """

    def __init__(self, Z1: int = 26, Z2: int = 26,
                 M1: float = 55.845, M2: float = 55.845) -> None:
        self.Z1 = Z1
        self.Z2 = Z2
        self.M1 = M1
        self.M2 = M2
        self.a_u = 0.8854 * A0_BOHR / (Z1**0.23 + Z2**0.23)

    def zbl_phi(self, x: float) -> float:
        """ZBL universal screening function φ(x)."""
        return (0.1818 * math.exp(-3.2 * x) + 0.5099 * math.exp(-0.9423 * x)
                + 0.2802 * math.exp(-0.4029 * x) + 0.02817 * math.exp(-0.2016 * x))

    def zbl_potential(self, r: float) -> float:
        """ZBL interatomic potential V(r) in eV."""
        if r < 1e-15:
            return 1e10
        x = r / self.a_u
        return self.Z1 * self.Z2 * 14.4 / (r * 1e10) * self.zbl_phi(x)  # 14.4 eV·Å

    def energy_transfer(self, E: float, theta_cm: float) -> float:
        """Energy transferred in elastic collision.

        T = T_max sin²(θ_cm/2)
        T_max = 4 M₁M₂/(M₁+M₂)² × E
        """
        T_max = 4 * self.M1 * self.M2 / (self.M1 + self.M2)**2 * E
        return T_max * math.sin(theta_cm / 2)**2

    def rutherford_cross_section(self, E_cm: float, theta: float) -> float:
        """Rutherford differential cross section (cm²/sr).

        dσ/dΩ = (Z₁Z₂e²/(4E_cm))² / sin⁴(θ/2)
        """
        sin_half = math.sin(theta / 2)
        if abs(sin_half) < 1e-10:
            return 1e30
        a = self.Z1 * self.Z2 * 14.4 / (4 * E_cm)  # Å
        return (a * 1e-8)**2 / sin_half**4

    def thomas_fermi_cross_section(self, E: float) -> float:
        """Total elastic cross section (Thomas-Fermi, cm²).

        σ ≈ π a_u² (Z₁Z₂ e²/(2E a_u))^{2/3}
        """
        reduced_E = self.Z1 * self.Z2 * 14.4 / (2 * E * self.a_u * 1e10)
        return math.pi * (self.a_u * 100)**2 * reduced_E**(2 / 3)


# ---------------------------------------------------------------------------
#  Stopping Power
# ---------------------------------------------------------------------------

class StoppingPower:
    r"""
    Electronic and nuclear stopping power for ions in matter.

    Nuclear stopping (ZBL):
    $$S_n(\epsilon) = \frac{0.5\ln(1+1.1383\epsilon)}{\epsilon + 0.01321\epsilon^{0.21226} + 0.19593\epsilon^{0.5}}$$

    Electronic stopping (low energy, Lindhard-Scharff):
    $$S_e = k_L \sqrt{E}$$

    Projected range (Lindhard):
    $$R_p = \int_0^E \frac{dE'}{S_n(E') + S_e(E')}$$
    """

    def __init__(self, Z1: int = 2, Z2: int = 14,
                 M1: float = 4.0, M2: float = 28.0) -> None:
        self.Z1 = Z1
        self.Z2 = Z2
        self.M1 = M1
        self.M2 = M2
        self.a_u = 0.8854 * A0_BOHR / (Z1**0.23 + Z2**0.23)

    def reduced_energy(self, E: float) -> float:
        """ε = (M₂ E a_u) / (Z₁ Z₂ e² (M₁+M₂))."""
        return (self.M2 * E * self.a_u * 1e10
                / (self.Z1 * self.Z2 * 14.4 * (self.M1 + self.M2)))

    def nuclear_stopping_reduced(self, eps: float) -> float:
        """S_n(ε) — universal ZBL nuclear stopping."""
        if eps < 1e-12:
            return 0.0
        num = 0.5 * math.log(1 + 1.1383 * eps)
        denom = eps + 0.01321 * eps**0.21226 + 0.19593 * eps**0.5
        return num / denom

    def electronic_stopping_LS(self, E: float) -> float:
        """Lindhard-Scharff electronic stopping (eV/Å).

        S_e ≈ k_e √E
        k_e = 0.0793 Z₁^{2/3} Z₂^{1/2} (Z₁^{2/3}+Z₂^{2/3})^{-3/4} × (M₁/M₂)^{1/2}
        """
        Z1, Z2 = self.Z1, self.Z2
        k_e = (0.0793 * Z1**(2 / 3) * Z2**0.5
               * (Z1**(2 / 3) + Z2**(2 / 3))**(-3 / 4)
               * (self.M1 / self.M2)**0.5)
        return k_e * math.sqrt(max(E, 0))

    def projected_range(self, E: float, n_steps: int = 1000) -> float:
        """Projected range (Å) by integration of 1/S_total."""
        dE_step = E / n_steps
        R = 0.0
        for i in range(n_steps):
            Ei = E - i * dE_step
            if Ei <= 0:
                break
            eps = self.reduced_energy(Ei)
            S_n = self.nuclear_stopping_reduced(eps) * 50  # scale factor
            S_e = self.electronic_stopping_LS(Ei)
            S_total = S_n + S_e
            if S_total > 1e-10:
                R += dE_step / S_total
        return R


# ---------------------------------------------------------------------------
#  Frenkel Pair Thermodynamics
# ---------------------------------------------------------------------------

class FrenkelPairThermodynamics:
    r"""
    Thermodynamics of point defects (Frenkel pairs).

    Equilibrium concentration:
    $$c_{\text{eq}} = \exp\left(-\frac{E_f}{2k_BT}\right)$$

    Defect migration: $D = D_0\exp(-E_m/k_B T)$

    Recombination rate: $K_{iv} = 4\pi r_{iv}(D_i + D_v)$

    Wigner energy stored:
    $$E_W \approx N_d \times E_f \quad\text{(up to 2.8 kJ/mol for graphite)}$$
    """

    K_B: float = 8.617e-5  # eV/K

    def __init__(self, E_f_vacancy: float = 1.5, E_f_interstitial: float = 3.0,
                 E_m_vacancy: float = 0.7, E_m_interstitial: float = 0.1) -> None:
        """Energies in eV."""
        self.E_f_v = E_f_vacancy
        self.E_f_i = E_f_interstitial
        self.E_m_v = E_m_vacancy
        self.E_m_i = E_m_interstitial

    def equilibrium_concentration(self, T: float, E_f: float) -> float:
        """c_eq = exp(−E_f / k_BT)."""
        if T < 1:
            return 0.0
        return math.exp(-E_f / (self.K_B * T))

    def diffusion_coefficient(self, T: float, E_m: float,
                                 D0: float = 1e-3) -> float:
        """D = D₀ exp(−E_m/k_BT)  (cm²/s)."""
        if T < 1:
            return 0.0
        return D0 * math.exp(-E_m / (self.K_B * T))

    def recombination_radius(self, T: float) -> float:
        """Recombination radius r_iv (Å).

        r_iv ≈ a₀ × (E_f / k_BT)^{1/3} (approximate, ~few lattice constants)
        """
        return 3.0  # Å (typical for metals)

    def steady_state_concentration(self, T: float, G: float) -> Dict[str, float]:
        """Steady-state defect concentrations under irradiation.

        G: defect production rate (dpa/s).
        Rate equations: dCv/dt = G − K_iv Ci Cv − Dv Cv/L²
        """
        D_v = self.diffusion_coefficient(T, self.E_m_v)
        D_i = self.diffusion_coefficient(T, self.E_m_i)
        r_iv = self.recombination_radius(T) * 1e-8  # cm
        K = 4 * math.pi * r_iv * (D_i + D_v)

        L_sink = 1e-5  # cm (sink spacing)
        k_v = D_v / L_sink**2
        k_i = D_i / L_sink**2

        if K < 1e-30:
            return {'C_v': 0.0, 'C_i': 0.0}

        disc = (k_v + k_i)**2 + 4 * K * G
        C_i = (-(k_v + k_i) + math.sqrt(max(disc, 0))) / (2 * K)
        C_v = G / (K * C_i + k_v) if (K * C_i + k_v) > 1e-30 else 0.0

        return {'C_v': C_v, 'C_i': C_i}

    def wigner_energy(self, N_d: float) -> float:
        """Stored Wigner energy (eV/atom).

        E_W ≈ N_d × (E_f_v + E_f_i) / 2
        """
        return N_d * (self.E_f_v + self.E_f_i) / 2
