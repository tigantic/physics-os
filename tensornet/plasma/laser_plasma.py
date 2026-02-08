"""
Laser-Plasma Interaction: SRS, SBS, relativistic self-focusing,
parametric instability.

Upgrades domain XI.6 to full laser-plasma physics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Physical Constants (CGS-Gaussian for laser-plasma conventions)
# ---------------------------------------------------------------------------

C_LIGHT: float = 2.998e10       # cm/s
E_CHARGE: float = 4.803e-10     # statcoulombs
E_MASS: float = 9.109e-28       # g
PROTON_MASS: float = 1.673e-24  # g
BOLTZMANN_CGS: float = 1.381e-16  # erg/K


# ---------------------------------------------------------------------------
#  Plasma Parameters
# ---------------------------------------------------------------------------

@dataclass
class LaserPlasmaParams:
    """Parameters for laser-plasma interaction."""
    n_e: float            # electron density (cm⁻³)
    T_e: float            # electron temperature (keV)
    T_i: float            # ion temperature (keV)
    Z_ion: int = 1        # ionisation state
    A_ion: int = 1        # atomic mass number
    lambda_0: float = 1.053e-4  # laser wavelength (cm) [1.053 μm = Nd:glass]
    I_laser: float = 1e15       # laser intensity (W/cm²)

    @property
    def omega_0(self) -> float:
        """Laser angular frequency ω₀ = 2πc/λ."""
        return 2.0 * math.pi * C_LIGHT / self.lambda_0

    @property
    def omega_pe(self) -> float:
        """Electron plasma frequency ωₚₑ = √(4π nₑ e² / mₑ)."""
        return math.sqrt(4.0 * math.pi * self.n_e * E_CHARGE**2 / E_MASS)

    @property
    def n_critical(self) -> float:
        """Critical density nₑ = mₑ ω₀² / (4π e²)."""
        return E_MASS * self.omega_0**2 / (4.0 * math.pi * E_CHARGE**2)

    @property
    def n_over_ncrit(self) -> float:
        return self.n_e / self.n_critical

    @property
    def v_osc(self) -> float:
        """Electron quiver velocity vₒₛ = eE₀/(mₑω₀).

        E₀ = √(8πI/c) in Gaussian units.
        """
        E0 = math.sqrt(8.0 * math.pi * self.I_laser * 1e7 / C_LIGHT)  # convert W → erg/s
        return E_CHARGE * E0 / (E_MASS * self.omega_0)

    @property
    def v_te(self) -> float:
        """Electron thermal velocity vₜₑ = √(Tₑ/mₑ) [T in erg]."""
        T_erg = self.T_e * 1.602e-9  # keV → erg
        return math.sqrt(T_erg / E_MASS)

    @property
    def v_ti(self) -> float:
        """Ion thermal velocity."""
        T_erg = self.T_i * 1.602e-9
        m_i = self.A_ion * PROTON_MASS
        return math.sqrt(T_erg / m_i)

    @property
    def debye_length(self) -> float:
        """λ_D = vₜₑ / ωₚₑ."""
        return self.v_te / self.omega_pe if self.omega_pe > 0 else float('inf')


# ---------------------------------------------------------------------------
#  Stimulated Raman Scattering (SRS)
# ---------------------------------------------------------------------------

class StimulatedRamanScattering:
    r"""
    Stimulated Raman Scattering: EM wave (ω₀,k₀) → EM wave (ωₛ,kₛ) + EPW (ω_epw, k_epw).

    Matching conditions:
    $$\omega_0 = \omega_s + \omega_{\text{epw}},\quad k_0 = k_s + k_{\text{epw}}$$

    Growth rate (homogeneous):
    $$\gamma_0 = \frac{k_{\text{epw}} v_{\text{osc}}}{4}
                 \sqrt{\frac{\omega_{pe}^2}{\omega_s \omega_{\text{epw}}}}$$

    Threshold (inhomogeneous): $\gamma_0^2 > \kappa_s \kappa_{\text{epw}} / \tau$
    where $\kappa$ = convective loss rate.
    """

    def __init__(self, params: LaserPlasmaParams) -> None:
        self.p = params

    def frequency_matching(self) -> Tuple[float, float]:
        """Return (ω_scattered, ω_EPW) satisfying matching conditions."""
        omega_0 = self.p.omega_0
        omega_pe = self.p.omega_pe

        # EPW: ω_epw ≈ ωₚₑ √(1 + 3k²λD²)  [at matching k]
        # Scattered: ω_s = ω₀ - ω_epw
        # For backscatter (k_epw ≈ 2k₀): ω_epw ≈ ωₚₑ
        omega_epw = omega_pe
        omega_s = omega_0 - omega_epw
        return omega_s, omega_epw

    def growth_rate(self) -> float:
        r"""Homogeneous temporal growth rate γ₀."""
        omega_s, omega_epw = self.frequency_matching()
        k_epw = omega_epw / self.p.v_te  # approximate
        v_osc = self.p.v_osc
        omega_pe = self.p.omega_pe

        if omega_s <= 0 or omega_epw <= 0:
            return 0.0

        gamma = 0.25 * k_epw * v_osc * math.sqrt(omega_pe**2 / (omega_s * omega_epw))
        return gamma

    def convective_gain(self, L_n: float) -> float:
        """
        Rosenbluth convective gain for inhomogeneous plasma.

        G = 2π γ₀² / (κₛ κ_epw |v'|)
        where v' = d(vₛ·v_epw)/dx at resonance point.
        """
        gamma = self.growth_rate()
        omega_0 = self.p.omega_0
        # Scale length L_n: density gradient
        kappa_s = omega_0 / (C_LIGHT * L_n) if L_n > 0 else 1.0
        kappa_epw = kappa_s  # approximate for backscatter
        v_group_s = C_LIGHT * math.sqrt(max(1.0 - self.p.n_over_ncrit, 0.01))
        v_group_epw = 3.0 * self.p.v_te  # EPW group velocity

        gain = 2.0 * math.pi * gamma**2 * L_n / (abs(v_group_s * v_group_epw) + 1e-30)
        return gain


# ---------------------------------------------------------------------------
#  Stimulated Brillouin Scattering (SBS)
# ---------------------------------------------------------------------------

class StimulatedBrillouinScattering:
    r"""
    SBS: EM (ω₀,k₀) → EM (ωₛ,kₛ) + IAW (ω_iaw, k_iaw).

    Growth rate:
    $$\gamma_0 = \frac{k_{\text{iaw}} v_{\text{osc}}}{4}
                 \sqrt{\frac{\omega_{pe}^2}{\omega_0 \omega_{\text{iaw}}}}$$

    IAW frequency: $\omega_{\text{iaw}} = k c_s$ where $c_s = \sqrt{Z T_e / m_i}$.
    """

    def __init__(self, params: LaserPlasmaParams) -> None:
        self.p = params

    @property
    def sound_speed(self) -> float:
        """Ion-acoustic sound speed cₛ = √(ZTₑ/mᵢ)."""
        T_erg = self.p.T_e * 1.602e-9
        m_i = self.p.A_ion * PROTON_MASS
        return math.sqrt(self.p.Z_ion * T_erg / m_i)

    def frequency_matching(self) -> Tuple[float, float]:
        """(ω_scattered, ω_IAW)."""
        omega_0 = self.p.omega_0
        cs = self.sound_speed
        # backscatter: k_iaw ≈ 2k₀
        k0 = omega_0 / C_LIGHT * math.sqrt(max(1.0 - self.p.n_over_ncrit, 0.01))
        k_iaw = 2.0 * k0
        omega_iaw = k_iaw * cs
        omega_s = omega_0 - omega_iaw
        return omega_s, omega_iaw

    def growth_rate(self) -> float:
        """Homogeneous SBS growth rate γ₀."""
        omega_s, omega_iaw = self.frequency_matching()
        omega_0 = self.p.omega_0
        omega_pe = self.p.omega_pe
        v_osc = self.p.v_osc

        k0 = omega_0 / C_LIGHT * math.sqrt(max(1.0 - self.p.n_over_ncrit, 0.01))
        k_iaw = 2.0 * k0

        if omega_0 <= 0 or omega_iaw <= 0:
            return 0.0

        return 0.25 * k_iaw * v_osc * math.sqrt(omega_pe**2 / (omega_0 * omega_iaw))

    def landau_damping_iaw(self) -> float:
        """IAW Landau damping rate."""
        cs = self.sound_speed
        v_ti = self.p.v_ti
        if v_ti < 1e-10:
            return 0.0
        # ν_L / ω ≈ √(π/8) (cs/v_ti) exp(-cs²/(2v_ti²))
        ratio = cs / v_ti
        return math.sqrt(math.pi / 8.0) * ratio * math.exp(-0.5 * ratio**2)


# ---------------------------------------------------------------------------
#  Relativistic Self-Focusing
# ---------------------------------------------------------------------------

class RelativisticSelfFocusing:
    r"""
    Relativistic self-focusing of intense laser beams.

    Critical power:
    $$P_c = 17.4\frac{n_c}{n_e}\;\text{GW}$$

    Normalised vector potential:
    $$a_0 = \frac{eE_0}{m_e c\omega_0} = 0.855\lambda[\mu m]\sqrt{I_{18}}$$

    Relativistic mass correction → refractive index increase → self-focusing.
    """

    def __init__(self, params: LaserPlasmaParams) -> None:
        self.p = params

    @property
    def a0(self) -> float:
        """Normalised laser vector potential a₀ = eE₀/(mₑcω₀)."""
        E0 = math.sqrt(8.0 * math.pi * self.p.I_laser * 1e7 / C_LIGHT)
        return E_CHARGE * E0 / (E_MASS * C_LIGHT * self.p.omega_0)

    @property
    def critical_power_gw(self) -> float:
        """Critical power for relativistic self-focusing (GW)."""
        return 17.4 * self.p.n_critical / self.p.n_e  # GW

    @property
    def gamma_rel(self) -> float:
        """Relativistic Lorentz factor γ = √(1 + a₀²/2) for circular polarisation."""
        return math.sqrt(1.0 + self.a0**2 / 2.0)

    def beam_power_gw(self, w0: float) -> float:
        """Laser beam power (GW) for given spot size w₀ (cm).

        P = π w₀² I / 2.
        """
        return math.pi * w0**2 * self.p.I_laser / 2e9

    def is_self_focusing(self, w0: float) -> bool:
        """Check if P > Pc for given spot size."""
        return self.beam_power_gw(w0) > self.critical_power_gw

    def effective_plasma_frequency(self) -> float:
        """ωₚₑ,eff = ωₚₑ / √γ due to relativistic mass increase."""
        return self.p.omega_pe / math.sqrt(self.gamma_rel)

    def self_focusing_length(self, w0: float) -> float:
        """Estimated self-focusing length for P > Pc.

        z_sf ≈ z_R / √(P/Pc - 1) where z_R = π w₀² / λ.
        """
        z_R = math.pi * w0**2 / self.p.lambda_0
        P = self.beam_power_gw(w0)
        Pc = self.critical_power_gw
        if P <= Pc:
            return float('inf')
        return z_R / math.sqrt(P / Pc - 1.0)


# ---------------------------------------------------------------------------
#  Parametric Instabilities (General)
# ---------------------------------------------------------------------------

class ParametricInstability:
    r"""
    General parametric instability framework.

    Dispersion relation: $D(\omega, k) = 0$ for coupled three-wave system.

    Two-plasmon decay (TPD):
    $$\omega_0 = \omega_1 + \omega_2,\quad k_0 = k_1 + k_2$$
    where both daughter waves are EPWs at $\omega \approx \omega_{pe}$.

    Threshold: $v_{osc}/v_{te} > (k\lambda_D)$ typically.
    """

    def __init__(self, params: LaserPlasmaParams) -> None:
        self.p = params

    def two_plasmon_decay_growth(self) -> float:
        r"""TPD growth rate at quarter-critical density.

        γ_TPD = √3/4 × (v_osc/v_te) × ωₚₑ × (k λ_D)^{1/3}

        Requires nₑ ≈ nc/4.
        """
        v_osc = self.p.v_osc
        v_te = self.p.v_te
        omega_pe = self.p.omega_pe
        k_lam = self.p.omega_0 / C_LIGHT * self.p.debye_length

        ratio = v_osc / (v_te + 1e-30)
        gamma = math.sqrt(3.0) / 4.0 * ratio * omega_pe * max(k_lam, 1e-6)**(1.0 / 3.0)
        return gamma

    def two_plasmon_decay_threshold(self) -> float:
        """v_osc/v_te threshold for TPD at nₑ = nc/4."""
        k_lam = self.p.omega_0 / C_LIGHT * self.p.debye_length
        return max(k_lam, 1e-6)

    def oscillating_two_stream(self) -> float:
        """OTSI growth rate for pump near plasma frequency.

        γ_OTSI ≈ (v_osc/v_te)^{2/3} × ωₚₑ / 2^{2/3}
        """
        ratio = self.p.v_osc / (self.p.v_te + 1e-30)
        return ratio**(2.0 / 3.0) * self.p.omega_pe / 2.0**(2.0 / 3.0)

    def filamentation_growth(self) -> float:
        """Ponderomotive filamentation growth rate.

        γ_fil ≈ (ωₚₑ²/ω₀) × (v_osc/c)² / 4
        """
        omega_pe = self.p.omega_pe
        omega_0 = self.p.omega_0
        v_osc = self.p.v_osc
        return (omega_pe**2 / omega_0) * (v_osc / C_LIGHT)**2 / 4.0
