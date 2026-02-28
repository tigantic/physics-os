"""
Atmospheric Physics: Chapman ozone photochemistry, Kessler warm-rain
microphysics, radiative-convective equilibrium.

Upgrades domain XIII.4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

K_BOLT: float = 1.381e-23       # J/K
N_AVOGADRO: float = 6.022e23
R_GAS: float = 8.314            # J/(mol·K)
SIGMA_SB: float = 5.670e-8      # W m⁻² K⁻⁴
G_EARTH: float = 9.81           # m/s²
R_EARTH: float = 6.371e6        # m
SOLAR_CONSTANT: float = 1361.0  # W/m²


# ---------------------------------------------------------------------------
#  Chapman Ozone Photochemistry
# ---------------------------------------------------------------------------

class ChapmanOzone:
    r"""
    Chapman oxygen-only photochemistry for stratospheric ozone.

    Reactions:
    1. O₂ + hν → 2O (J₂)
    2. O + O₂ + M → O₃ + M (k₂)
    3. O₃ + hν → O₂ + O (J₃)
    4. O + O₃ → 2O₂ (k₃)

    Steady-state ozone: $[\text{O}_3]_{ss} = \sqrt{\frac{J_2 k_2 [M]}{J_3 k_3}}[\text{O}_2]$
    """

    def __init__(self, T: float = 220.0, M_density: float = 5e18,
                 O2_density: float = 1e18) -> None:
        """
        Parameters
        ----------
        T : Temperature (K).
        M_density : Third-body (N₂+O₂) number density (cm⁻³).
        O2_density : O₂ number density (cm⁻³).
        """
        self.T = T
        self.M = M_density
        self.O2 = O2_density

    @property
    def J2(self) -> float:
        """O₂ photolysis rate (s⁻¹). ~1e-12 at 40 km."""
        return 1e-12

    @property
    def J3(self) -> float:
        """O₃ photolysis rate (s⁻¹). ~1e-3 at 40 km."""
        return 1e-3

    @property
    def k2(self) -> float:
        """3-body recombination O + O₂ + M → O₃ + M (cm⁶/s)."""
        return 6e-34 * (self.T / 300.0)**(-2.3)

    @property
    def k3(self) -> float:
        """O + O₃ → 2O₂ rate (cm³/s)."""
        return 8e-12 * math.exp(-2060.0 / self.T)

    def steady_state_O3(self) -> float:
        """[O₃]ₛₛ from Chapman equilibrium."""
        return math.sqrt(self.J2 * self.k2 * self.M / (self.J3 * self.k3 + 1e-30)) * self.O2

    def steady_state_O(self) -> float:
        """[O]ₛₛ = J₃[O₃] / (k₂[O₂][M])."""
        O3 = self.steady_state_O3()
        return self.J3 * O3 / (self.k2 * self.O2 * self.M + 1e-30)

    def evolve(self, dt: float, n_steps: int,
               O_init: float, O3_init: float) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Integrate Chapman system forward in time.

        Returns (time, [O], [O₃]) arrays.
        """
        t_arr = np.zeros(n_steps + 1)
        O_arr = np.zeros(n_steps + 1)
        O3_arr = np.zeros(n_steps + 1)

        O_arr[0] = O_init
        O3_arr[0] = O3_init

        O, O3 = O_init, O3_init

        for n in range(n_steps):
            # d[O]/dt = 2 J₂[O₂] + J₃[O₃] - k₂[O][O₂][M] - k₃[O][O₃]
            dO = (2.0 * self.J2 * self.O2 + self.J3 * O3
                  - self.k2 * O * self.O2 * self.M - self.k3 * O * O3)
            # d[O₃]/dt = k₂[O][O₂][M] - J₃[O₃] - k₃[O][O₃]
            dO3 = (self.k2 * O * self.O2 * self.M - self.J3 * O3
                   - self.k3 * O * O3)

            O += dO * dt
            O3 += dO3 * dt

            O = max(O, 0.0)
            O3 = max(O3, 0.0)

            t_arr[n + 1] = (n + 1) * dt
            O_arr[n + 1] = O
            O3_arr[n + 1] = O3

        return t_arr, O_arr, O3_arr


# ---------------------------------------------------------------------------
#  Kessler Warm-Rain Microphysics
# ---------------------------------------------------------------------------

class KesslerMicrophysics:
    r"""
    Kessler (1969) warm-rain parameterisation.

    Processes:
    - Autoconversion: cloud → rain when q_c > q_c0
    - Accretion: rain collecting cloud droplets
    - Evaporation: rain evaporation in sub-saturated air

    $$\frac{dq_r}{dt} = \alpha_1(q_c - q_{c0})^+ + \alpha_2 q_c q_r^{0.875}
                       - \alpha_3(q_{vs} - q_v)^+ q_r^{0.525}$$
    """

    def __init__(self, alpha_auto: float = 1e-3,
                 alpha_accr: float = 2.2,
                 alpha_evap: float = 1e-3,
                 qc_threshold: float = 1e-3) -> None:
        """
        Parameters
        ----------
        alpha_auto : Autoconversion rate (s⁻¹).
        alpha_accr : Accretion coefficient.
        alpha_evap : Evaporation coefficient.
        qc_threshold : Cloud water threshold for autoconversion (kg/kg).
        """
        self.alpha_auto = alpha_auto
        self.alpha_accr = alpha_accr
        self.alpha_evap = alpha_evap
        self.qc_threshold = qc_threshold

    def autoconversion(self, qc: float) -> float:
        """dq_r/dt from autoconversion."""
        return self.alpha_auto * max(qc - self.qc_threshold, 0.0)

    def accretion(self, qc: float, qr: float) -> float:
        """dq_r/dt from accretion."""
        return self.alpha_accr * qc * max(qr, 0.0)**0.875

    def evaporation(self, qv: float, qvs: float, qr: float) -> float:
        """dq_r/dt from evaporation (negative = loss of rain)."""
        deficit = max(qvs - qv, 0.0)
        return -self.alpha_evap * deficit * max(qr, 0.0)**0.525

    def tendency(self, qc: float, qr: float,
                  qv: float, qvs: float) -> Tuple[float, float, float]:
        """
        Compute microphysics tendencies.

        Returns (dqc/dt, dqr/dt, dqv/dt).
        """
        auto = self.autoconversion(qc)
        accr = self.accretion(qc, qr)
        evap = self.evaporation(qv, qvs, qr)

        dqr = auto + accr + evap
        dqc = -auto - accr
        dqv = -evap  # evaporated rain adds to vapour

        return dqc, dqr, dqv

    def terminal_velocity(self, qr: float, rho_air: float = 1.0) -> float:
        """Rain terminal fall speed (m/s).

        V_t = 36.34 (ρ₀/ρ)^{0.4} q_r^{0.1346}
        """
        rho_0 = 1.225  # surface density kg/m³
        return 36.34 * (rho_0 / rho_air)**0.4 * max(qr, 0.0)**0.1346


# ---------------------------------------------------------------------------
#  Saturation Vapour Pressure
# ---------------------------------------------------------------------------

class ClausiusClapeyron:
    """Saturation vapour pressure calculations."""

    @staticmethod
    def es_bolton(T: float) -> float:
        """Bolton (1980) formula: eₛ(T) in Pa. T in °C."""
        return 611.2 * math.exp(17.67 * T / (T + 243.5))

    @staticmethod
    def es_tetens(T_K: float) -> float:
        """Tetens formula: eₛ(T) in Pa. T in K."""
        T_C = T_K - 273.15
        return 610.78 * math.exp(17.27 * T_C / (T_C + 237.3))

    @staticmethod
    def mixing_ratio(es: float, p: float, epsilon: float = 0.622) -> float:
        """Saturation mixing ratio qₛ = ε eₛ / (p - eₛ)."""
        return epsilon * es / (p - es + 1e-10)


# ---------------------------------------------------------------------------
#  Radiative-Convective Equilibrium
# ---------------------------------------------------------------------------

class RadiativeConvectiveEquilibrium:
    r"""
    1D radiative-convective equilibrium model.

    Grey atmosphere with:
    - Radiative transfer: dF↑/dτ = F↑ - σT⁴, dF↓/dτ = -(F↓ - σT⁴)
    - Convective adjustment to dry/moist adiabat when lapse rate exceeds threshold

    Surface energy balance:
    $$F_\text{solar}(1-\alpha) = \sigma T_s^4 (2/(2+\tau_\infty))$$
    """

    def __init__(self, n_layers: int = 40, p_surface: float = 1e5,
                 p_top: float = 100.0, S0: float = SOLAR_CONSTANT,
                 albedo: float = 0.3,
                 tau_lw: float = 2.0) -> None:
        """
        Parameters
        ----------
        n_layers : Number of atmospheric layers.
        p_surface : Surface pressure (Pa).
        p_top : Top-of-atmosphere pressure (Pa).
        S0 : Solar constant (W/m²).
        albedo : Planetary albedo.
        tau_lw : Total longwave optical depth.
        """
        self.n = n_layers
        self.p = np.linspace(p_surface, p_top, n_layers)
        self.dp = self.p[0] - self.p[1]
        self.S0 = S0
        self.albedo = albedo
        self.tau_lw = tau_lw

        # Temperature profile
        self.T = np.full(n_layers, 250.0)
        self.T_surface = 288.0

    def optical_depth(self) -> NDArray[np.float64]:
        """τ(p) = τ_∞ × (p/p_s)²."""
        return self.tau_lw * (self.p / self.p[0])**2

    def radiative_equilibrium_temperature(self) -> NDArray[np.float64]:
        """Grey atmosphere analytical solution.

        T⁴(τ) = (F_solar/σ)(1/2)(1 + τ)
        """
        F = self.S0 * (1.0 - self.albedo) / 4.0  # global mean
        tau = self.optical_depth()
        T4 = (F / SIGMA_SB) * 0.5 * (1.0 + tau)
        return T4**0.25

    def convective_adjustment(self, lapse_rate: float = 6.5e-3) -> None:
        """
        Adjust temperature profile to specified lapse rate (K/m)
        from bottom up wherever radiative profile is super-adiabatic.

        Uses hydrostatic: dT/dp = T Γ / (g p) where Γ = Γ_d ≈ g/cp.
        """
        cp = 1004.0  # J/(kg·K)
        gamma_dry = G_EARTH / cp  # ~9.8 K/km

        # Convert lapse rate to dT/dp
        for i in range(self.n - 2, -1, -1):
            # Height difference from hydrostatic: dz = -dp/(ρg) = -dp·R_d·T/(p·g)
            R_d = 287.0
            T_mean = 0.5 * (self.T[i] + self.T[i + 1])
            dz = -R_d * T_mean / (G_EARTH * self.p[i]) * (self.p[i + 1] - self.p[i])

            T_adjusted = self.T[i] + lapse_rate * dz
            if self.T[i + 1] < T_adjusted:
                # Super-adiabatic: adjust
                self.T[i + 1] = T_adjusted

    def iterate_to_equilibrium(self, n_iter: int = 5000,
                                  dt: float = 86400.0) -> NDArray[np.float64]:
        """
        Iterate radiative heating/cooling with convective adjustment.

        Returns final temperature profile.
        """
        tau = self.optical_depth()
        F_solar = self.S0 * (1.0 - self.albedo) / 4.0
        cp = 1004.0

        for _ in range(n_iter):
            # Upward and downward fluxes
            F_up = np.zeros(self.n)
            F_down = np.zeros(self.n)

            # Bottom BC
            F_up[0] = SIGMA_SB * self.T_surface**4

            # Upward sweep
            for i in range(1, self.n):
                dtau = tau[i - 1] - tau[i]
                F_up[i] = F_up[i - 1] * math.exp(-dtau) + SIGMA_SB * self.T[i]**4 * (1 - math.exp(-dtau))

            # Top BC: no downward flux at TOA
            F_down[-1] = 0.0

            # Downward sweep
            for i in range(self.n - 2, -1, -1):
                dtau = tau[i] - tau[i + 1]
                F_down[i] = F_down[i + 1] * math.exp(-dtau) + SIGMA_SB * self.T[i]**4 * (1 - math.exp(-dtau))

            # Radiative heating
            for i in range(1, self.n - 1):
                dtau = (tau[i - 1] - tau[i + 1]) / 2.0
                dp = self.p[i - 1] - self.p[i + 1]
                net_flux_div = (F_up[i + 1] - F_up[i - 1] + F_down[i - 1] - F_down[i + 1]) / (dp + 1e-30)
                dT = -G_EARTH / cp * net_flux_div * dt
                self.T[i] += np.clip(dT, -5.0, 5.0)

            # Surface balance
            self.T_surface += dt * 1e-7 * (F_solar + F_down[0] - SIGMA_SB * self.T_surface**4)

            # Convective adjustment
            self.convective_adjustment()

            # Enforce minimum temperature
            self.T = np.maximum(self.T, 150.0)

        return self.T
