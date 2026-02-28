"""
Dusty Plasmas — Yukawa OCP, dust-acoustic waves, grain charging (OML),
dust crystals, Coulomb coupling.

Domain XI.7 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Dusty Plasma Parameters
# ---------------------------------------------------------------------------

@dataclass
class DustyPlasmaParams:
    r"""
    Parameters for a dusty (complex) plasma.

    Key scales:
    - Grain charge: $Q = C_d\phi_s \approx -Z_d e$ where $Z_d \sim 10^3$–$10^5$
    - Dust plasma frequency: $\omega_{pd} = \sqrt{Z_d^2 e^2 n_d / (\varepsilon_0 m_d)}$
    - Debye length: $\lambda_D = \sqrt{\varepsilon_0 k_BT_i / (n_i e^2)}$
    - Wigner-Seitz radius: $a_{WS} = (3/(4\pi n_d))^{1/3}$
    - Coupling parameter: $\Gamma = Q^2/(4\pi\varepsilon_0 a_{WS} k_BT_d)$

    Dust crystals form when $\Gamma > 170$.
    """
    n_i: float = 1e15       # m^-3 (ion density)
    n_e: float = 1e15       # m^-3 (electron density)
    n_d: float = 1e10       # m^-3 (dust density)
    T_i: float = 0.025      # eV (ion temperature ~ 300 K)
    T_e: float = 3.0        # eV (electron temperature)
    T_d: float = 0.025      # eV (dust temperature)
    m_i: float = 6.63e-26   # kg (Ar+)
    m_d: float = 1e-14      # kg (micron-sized grain)
    r_d: float = 1e-6       # m (grain radius)
    Z_d: float = 3000       # (grain charge number, negative)
    e: float = 1.602e-19
    eps_0: float = 8.854e-12
    kB: float = 1.381e-23

    @property
    def Q(self) -> float:
        """Grain charge Q = −Z_d e."""
        return self.Z_d * self.e

    @property
    def lambda_Di(self) -> float:
        """Ion Debye length."""
        return math.sqrt(self.eps_0 * self.T_i * self.e / (self.n_i * self.e**2))

    @property
    def lambda_De(self) -> float:
        """Electron Debye length."""
        return math.sqrt(self.eps_0 * self.T_e * self.e / (self.n_e * self.e**2))

    @property
    def lambda_D(self) -> float:
        """Total Debye length: 1/λ² = 1/λ_Di² + 1/λ_De²."""
        return 1 / math.sqrt(1 / self.lambda_Di**2 + 1 / self.lambda_De**2)

    @property
    def a_ws(self) -> float:
        """Wigner-Seitz radius."""
        return (3 / (4 * math.pi * self.n_d))**(1 / 3)

    @property
    def kappa(self) -> float:
        """Screening parameter: κ = a_WS / λ_D."""
        return self.a_ws / self.lambda_D

    @property
    def Gamma(self) -> float:
        r"""Coulomb coupling parameter: Γ = Q²/(4πε₀ a_WS k_BT_d)."""
        return self.Q**2 / (4 * math.pi * self.eps_0 * self.a_ws * self.T_d * self.e)

    @property
    def is_crystalline(self) -> bool:
        """OCP melting: Γ > 170 (Yukawa shift with κ)."""
        return self.Gamma > 170

    @property
    def omega_pd(self) -> float:
        """Dust plasma frequency."""
        return math.sqrt(self.Z_d**2 * self.e**2 * self.n_d
                          / (self.eps_0 * self.m_d))


# ---------------------------------------------------------------------------
#  OML Grain Charging
# ---------------------------------------------------------------------------

class OMLGrainCharging:
    r"""
    Orbital Motion Limited (OML) theory for grain charging.

    Electron current:
    $$I_e = -\pi r_d^2 n_e e\sqrt{\frac{8k_BT_e}{\pi m_e}}
      \exp\!\left(\frac{e\phi_s}{k_BT_e}\right)\quad (\phi_s < 0)$$

    Ion current:
    $$I_i = \pi r_d^2 n_i e\sqrt{\frac{8k_BT_i}{\pi m_i}}
      \left(1 - \frac{e\phi_s}{k_BT_i}\right)\quad (\phi_s < 0)$$

    Floating potential from $I_e + I_i = 0$.
    """

    def __init__(self, params: Optional[DustyPlasmaParams] = None) -> None:
        self.p = params or DustyPlasmaParams()

    def electron_current(self, phi_s: float) -> float:
        """OML electron current to grain (phi_s < 0)."""
        p = self.p
        m_e = 9.11e-31
        v_te = math.sqrt(8 * p.T_e * p.e / (math.pi * m_e))
        I_e = -math.pi * p.r_d**2 * p.n_e * p.e * v_te

        if phi_s < 0:
            I_e *= math.exp(p.e * phi_s / (p.T_e * p.e))
        return I_e

    def ion_current(self, phi_s: float) -> float:
        """OML ion current to grain (phi_s < 0, repulsive/attractive)."""
        p = self.p
        v_ti = math.sqrt(8 * p.T_i * p.e / (math.pi * p.m_i))
        I_i = math.pi * p.r_d**2 * p.n_i * p.e * v_ti

        if phi_s < 0:
            I_i *= (1 - p.e * phi_s / (p.T_i * p.e))
        else:
            I_i *= math.exp(-p.e * phi_s / (p.T_i * p.e))
        return I_i

    def floating_potential(self, tol: float = 1e-6) -> float:
        """Find φ_s where I_e + I_i = 0 via bisection."""
        phi_lo = -20 * self.p.T_e  # V (normalised)
        phi_hi = 0.0

        for _ in range(200):
            phi_mid = 0.5 * (phi_lo + phi_hi)
            I_total = self.electron_current(phi_mid) + self.ion_current(phi_mid)

            if abs(I_total) < tol * abs(self.ion_current(phi_mid)):
                return phi_mid

            if I_total > 0:
                phi_hi = phi_mid
            else:
                phi_lo = phi_mid

        return 0.5 * (phi_lo + phi_hi)

    def charge_number(self) -> float:
        """Z_d = C |φ_s| / e, where C = 4πε₀r_d."""
        phi_s = abs(self.floating_potential())
        C = 4 * math.pi * self.p.eps_0 * self.p.r_d
        return C * phi_s / self.p.e

    def charging_time(self) -> float:
        """τ_ch ≈ C / (dI/dφ)_{φ=φ_s}."""
        phi_s = self.floating_potential()
        C = 4 * math.pi * self.p.eps_0 * self.p.r_d
        eps = 0.001 * abs(phi_s) + 1e-10
        dI = ((self.electron_current(phi_s + eps) + self.ion_current(phi_s + eps))
              - (self.electron_current(phi_s - eps) + self.ion_current(phi_s - eps))) / (2 * eps)
        return C / (abs(dI) + 1e-30)


# ---------------------------------------------------------------------------
#  Dust-Acoustic Waves (DAW)
# ---------------------------------------------------------------------------

class DustAcousticWave:
    r"""
    Dust-Acoustic Wave (DAW) dispersion — Rao, Shukla, Yu (1990).

    $$\omega^2 = \frac{k^2 C_{DA}^2}{1 + k^2\lambda_D^2}$$

    where $C_{DA} = \omega_{pd}\lambda_D$ = dust-acoustic speed.

    Kinetic damping: Landau damping on ions at short wavelengths.
    """

    def __init__(self, params: Optional[DustyPlasmaParams] = None) -> None:
        self.p = params or DustyPlasmaParams()

    @property
    def C_DA(self) -> float:
        """Dust-acoustic speed: C_DA = ω_pd λ_D."""
        return self.p.omega_pd * self.p.lambda_D

    def dispersion(self, k: NDArray) -> NDArray:
        """ω(k) for DAW."""
        return np.sqrt(k**2 * self.C_DA**2 / (1 + k**2 * self.p.lambda_D**2))

    def phase_velocity(self, k: NDArray) -> NDArray:
        """v_ph = ω/k."""
        omega = self.dispersion(k)
        return omega / (k + 1e-30)

    def group_velocity(self, k: NDArray) -> NDArray:
        """v_g = dω/dk."""
        omega = self.dispersion(k)
        lamD2 = self.p.lambda_D**2
        return self.C_DA**2 * k / (omega * (1 + k**2 * lamD2)**2 + 1e-30)

    def ion_landau_damping(self, k: float) -> float:
        """Ion Landau damping rate (approximate)."""
        omega = float(self.dispersion(np.array([k])))
        v_ti = math.sqrt(self.p.T_i * self.p.e / self.p.m_i)
        xi = omega / (k * v_ti + 1e-30)
        return -math.sqrt(math.pi / 8) * omega * xi**3 * math.exp(-xi**2 / 2)


# ---------------------------------------------------------------------------
#  Yukawa One-Component Plasma (OCP)
# ---------------------------------------------------------------------------

class YukawaOCP:
    r"""
    Yukawa One-Component Plasma — screened Coulomb (Debye-Hückel) model
    for strongly coupled dusty plasmas.

    Pair potential:
    $$\phi(r) = \frac{Q^2}{4\pi\varepsilon_0 r}\exp(-r/\lambda_D)$$

    Phase diagram parameterised by (Γ, κ):
    - Γ: Coulomb coupling at a_WS
    - κ = a_WS/λ_D: screening parameter

    Crystallisation line: Γ_m(κ) ≈ 170 exp(Aκ + Bκ²) (Hamaguchi et al.).
    """

    def __init__(self, params: Optional[DustyPlasmaParams] = None) -> None:
        self.p = params or DustyPlasmaParams()

    def pair_potential(self, r: float) -> float:
        """Yukawa potential φ(r)."""
        if r < 1e-15:
            return float('inf')
        return (self.p.Q**2 / (4 * math.pi * self.p.eps_0 * r)
                * math.exp(-r / self.p.lambda_D))

    def pair_potential_array(self, r: NDArray) -> NDArray:
        """Vectorised Yukawa potential."""
        return (self.p.Q**2 / (4 * math.pi * self.p.eps_0 * r)
                * np.exp(-r / self.p.lambda_D))

    def madelung_energy_bcc(self) -> float:
        """Approximate Madelung energy for BCC Yukawa crystal.

        E_M/N ≈ −0.8959 Q²/(4πε₀ a_WS) exp(−κ).
        """
        return -0.8959 * self.p.Q**2 / (4 * math.pi * self.p.eps_0 * self.p.a_ws) \
            * math.exp(-self.p.kappa)

    def melting_curve(self, kappa: float) -> float:
        """Γ_m(κ) ≈ 170 exp(Aκ + Bκ²) — Hamaguchi fit.

        A ≈ 1.08, B ≈ 0.12 (fitted to MD data).
        """
        A, B = 1.08, 0.12
        return 170 * math.exp(A * kappa + B * kappa**2)

    def einstein_frequency(self) -> float:
        """Einstein frequency for Yukawa crystal oscillation.

        ω_E² = (Q²/(4πε₀ m_d a_WS³)) (1 + κ + κ²/2) exp(−κ).
        """
        p = self.p
        kappa = p.kappa
        prefactor = p.Q**2 / (4 * math.pi * p.eps_0 * p.m_d * p.a_ws**3)
        return math.sqrt(prefactor * (1 + kappa + kappa**2 / 2) * math.exp(-kappa))

    def dust_lattice_wave(self, k: NDArray) -> NDArray:
        """Longitudinal dust lattice wave dispersion (1D chain).

        ω² = (ω_E²/2) [1 − cos(ka)] (Debye model).
        """
        omega_E = self.einstein_frequency()
        return omega_E * np.sqrt(0.5 * (1 - np.cos(k * self.p.a_ws)))

    def phase_state(self) -> str:
        """Determine phase state from (Γ, κ)."""
        Gamma = self.p.Gamma
        kappa = self.p.kappa
        Gamma_m = self.melting_curve(kappa)
        if Gamma > Gamma_m:
            return "crystalline"
        elif Gamma > 1:
            return "strongly_coupled_liquid"
        else:
            return "weakly_coupled_gas"
