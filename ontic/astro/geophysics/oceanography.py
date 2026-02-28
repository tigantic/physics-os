"""
Physical Oceanography: primitive equations, thermohaline box model,
internal waves, tidal harmonics.

Upgrades domain XIII.5.
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

G_EARTH: float = 9.81            # m/s²
OMEGA_EARTH: float = 7.292e-5    # rad/s
RHO_WATER: float = 1025.0        # kg/m³ (reference)
CP_WATER: float = 3994.0         # J/(kg·K)


# ---------------------------------------------------------------------------
#  Equation of State (UNESCO/TEOS-10 simplified)
# ---------------------------------------------------------------------------

class SeawaterEOS:
    """
    Seawater equation of state: ρ(T, S, p).

    Simplified: linear approximation about reference state.
    ρ = ρ₀(1 - α(T - T₀) + β(S - S₀))
    where α ~ 2e-4 K⁻¹, β ~ 7.5e-4 (PSU)⁻¹.
    """

    def __init__(self, rho_0: float = 1025.0, T_0: float = 10.0,
                 S_0: float = 35.0, alpha_T: float = 2e-4,
                 beta_S: float = 7.5e-4) -> None:
        self.rho_0 = rho_0
        self.T_0 = T_0
        self.S_0 = S_0
        self.alpha_T = alpha_T
        self.beta_S = beta_S

    def density(self, T: float, S: float, p: float = 0.0) -> float:
        """ρ(T, S) in linear approximation."""
        return self.rho_0 * (1.0 - self.alpha_T * (T - self.T_0)
                             + self.beta_S * (S - self.S_0))

    def buoyancy_frequency(self, T_upper: float, T_lower: float,
                             S_upper: float, S_lower: float,
                             dz: float) -> float:
        """Brunt-Väisälä frequency N² = -(g/ρ₀) dρ/dz."""
        rho_upper = self.density(T_upper, S_upper)
        rho_lower = self.density(T_lower, S_lower)
        drho_dz = (rho_upper - rho_lower) / dz  # negative if stable
        return -G_EARTH / self.rho_0 * drho_dz


# ---------------------------------------------------------------------------
#  Shallow Water / Primitive Equations (2D)
# ---------------------------------------------------------------------------

class ShallowWaterEquations:
    r"""
    Rotating shallow water equations on f-plane.

    $$\frac{\partial u}{\partial t} - fv = -g\frac{\partial\eta}{\partial x}$$
    $$\frac{\partial v}{\partial t} + fu = -g\frac{\partial\eta}{\partial y}$$
    $$\frac{\partial\eta}{\partial t} + H\left(\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}\right) = 0$$

    Numerical: Arakawa C-grid with leapfrog time stepping.
    """

    def __init__(self, nx: int = 100, ny: int = 100,
                 Lx: float = 1e6, Ly: float = 1e6,
                 H: float = 4000.0, f0: float = 1e-4) -> None:
        """
        Parameters
        ----------
        nx, ny : Grid size.
        Lx, Ly : Domain size (m).
        H : Mean depth (m).
        f0 : Coriolis parameter (s⁻¹).
        """
        self.nx = nx
        self.ny = ny
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.H = H
        self.f0 = f0

        self.u = np.zeros((nx, ny))
        self.v = np.zeros((nx, ny))
        self.eta = np.zeros((nx, ny))

        # Previous step for leapfrog
        self.u_prev = np.zeros((nx, ny))
        self.v_prev = np.zeros((nx, ny))
        self.eta_prev = np.zeros((nx, ny))

        self._first_step = True

    def rossby_radius(self) -> float:
        """Rossby deformation radius R_d = √(gH) / f₀."""
        return math.sqrt(G_EARTH * self.H) / abs(self.f0)

    def gravity_wave_speed(self) -> float:
        """c = √(gH)."""
        return math.sqrt(G_EARTH * self.H)

    def cfl_timestep(self, cfl: float = 0.5) -> float:
        """CFL condition: dt < cfl × dx / c."""
        c = self.gravity_wave_speed()
        return cfl * min(self.dx, self.dy) / c

    def step(self, dt: float) -> None:
        """One time step (forward Euler for first, leapfrog after)."""
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        g = G_EARTH
        f = self.f0
        H = self.H

        u_new = np.zeros_like(self.u)
        v_new = np.zeros_like(self.v)
        eta_new = np.zeros_like(self.eta)

        # Compute tendencies
        for i in range(nx):
            ip = (i + 1) % nx
            im = (i - 1) % nx
            for j in range(ny):
                jp = (j + 1) % ny
                jm = (j - 1) % ny

                deta_dx = (self.eta[ip, j] - self.eta[im, j]) / (2.0 * dx)
                deta_dy = (self.eta[i, jp] - self.eta[i, jm]) / (2.0 * dy)
                du_dx = (self.u[ip, j] - self.u[im, j]) / (2.0 * dx)
                dv_dy = (self.v[i, jp] - self.v[i, jm]) / (2.0 * dy)

                if self._first_step:
                    u_new[i, j] = self.u[i, j] + dt * (f * self.v[i, j] - g * deta_dx)
                    v_new[i, j] = self.v[i, j] + dt * (-f * self.u[i, j] - g * deta_dy)
                    eta_new[i, j] = self.eta[i, j] - dt * H * (du_dx + dv_dy)
                else:
                    u_new[i, j] = self.u_prev[i, j] + 2.0 * dt * (f * self.v[i, j] - g * deta_dx)
                    v_new[i, j] = self.v_prev[i, j] + 2.0 * dt * (-f * self.u[i, j] - g * deta_dy)
                    eta_new[i, j] = self.eta_prev[i, j] - 2.0 * dt * H * (du_dx + dv_dy)

        self.u_prev = self.u.copy()
        self.v_prev = self.v.copy()
        self.eta_prev = self.eta.copy()

        # Robert-Asselin filter for leapfrog stability
        gamma_ra = 0.05
        if not self._first_step:
            self.u = self.u + gamma_ra * (u_new - 2.0 * self.u + self.u_prev)
            self.v = self.v + gamma_ra * (v_new - 2.0 * self.v + self.v_prev)
            self.eta = self.eta + gamma_ra * (eta_new - 2.0 * self.eta + self.eta_prev)

        self.u = u_new
        self.v = v_new
        self.eta = eta_new
        self._first_step = False


# ---------------------------------------------------------------------------
#  Stommel Thermohaline Box Model
# ---------------------------------------------------------------------------

class StommelBoxModel:
    r"""
    Stommel (1961) two-box thermohaline circulation model.

    Two boxes (equatorial + polar) with temperature T and salinity S.

    Flow: $q = k(\alpha_T \Delta T - \beta_S \Delta S)$

    When $q > 0$: thermally-driven (modern AMOC).
    When $q < 0$: salinity-driven (reversed circulation).

    Multiple equilibria possible → hysteresis.
    """

    def __init__(self, T_eq: float = 25.0, T_pol: float = 5.0,
                 S_eq: float = 36.0, S_pol: float = 34.0,
                 k: float = 1.5e6,
                 alpha_T: float = 2e-4, beta_S: float = 7.5e-4,
                 tau_T: float = 20.0, tau_S: float = 200.0) -> None:
        """
        Parameters
        ----------
        T_eq, T_pol : Restoring temperatures (°C).
        S_eq, S_pol : Restoring salinities (PSU).
        k : Flow coefficient (m³/s equivalent after scaling).
        alpha_T, beta_S : EOS coefficients.
        tau_T, tau_S : Restoring timescales (years).
        """
        self.T_eq_star = T_eq
        self.T_pol_star = T_pol
        self.S_eq_star = S_eq
        self.S_pol_star = S_pol
        self.k = k
        self.alpha = alpha_T
        self.beta = beta_S
        self.tau_T = tau_T
        self.tau_S = tau_S

        # State
        self.T_eq = T_eq
        self.T_pol = T_pol
        self.S_eq = S_eq
        self.S_pol = S_pol

    @property
    def delta_T(self) -> float:
        return self.T_eq - self.T_pol

    @property
    def delta_S(self) -> float:
        return self.S_eq - self.S_pol

    def flow_rate(self) -> float:
        """q = k(α ΔT - β ΔS)."""
        return self.k * (self.alpha * self.delta_T - self.beta * self.delta_S)

    def step(self, dt_years: float, freshwater_forcing: float = 0.0) -> None:
        """Advance one time step.

        Parameters
        ----------
        dt_years : Time step in years.
        freshwater_forcing : Additional freshwater flux to polar box (PSU/yr).
        """
        q = self.flow_rate()
        q_abs = abs(q)

        # Temperature relaxation + advection
        dT_eq = (self.T_eq_star - self.T_eq) / self.tau_T
        dT_pol = (self.T_pol_star - self.T_pol) / self.tau_T

        if q > 0:
            dT_eq -= q_abs * self.delta_T
            dT_pol += q_abs * self.delta_T
        else:
            dT_eq += q_abs * self.delta_T
            dT_pol -= q_abs * self.delta_T

        # Salinity relaxation + advection + forcing
        dS_eq = (self.S_eq_star - self.S_eq) / self.tau_S
        dS_pol = (self.S_pol_star - self.S_pol) / self.tau_S - freshwater_forcing

        if q > 0:
            dS_eq -= q_abs * self.delta_S
            dS_pol += q_abs * self.delta_S
        else:
            dS_eq += q_abs * self.delta_S
            dS_pol -= q_abs * self.delta_S

        self.T_eq += dT_eq * dt_years
        self.T_pol += dT_pol * dt_years
        self.S_eq += dS_eq * dt_years
        self.S_pol += dS_pol * dt_years

    def evolve(self, t_end_years: float, dt: float = 0.1,
               freshwater_forcing: float = 0.0) -> Dict[str, NDArray]:
        """Evolve and return time series."""
        n = int(t_end_years / dt)
        results: Dict[str, List[float]] = {
            "time": [], "T_eq": [], "T_pol": [], "S_eq": [], "S_pol": [], "q": []
        }

        for i in range(n):
            results["time"].append(i * dt)
            results["T_eq"].append(self.T_eq)
            results["T_pol"].append(self.T_pol)
            results["S_eq"].append(self.S_eq)
            results["S_pol"].append(self.S_pol)
            results["q"].append(self.flow_rate())
            self.step(dt, freshwater_forcing)

        return {k: np.array(v) for k, v in results.items()}


# ---------------------------------------------------------------------------
#  Internal Waves
# ---------------------------------------------------------------------------

class InternalWaves:
    r"""
    Internal gravity waves in stratified ocean.

    Dispersion relation:
    $$\omega^2 = N^2\frac{k_h^2}{k_h^2 + m^2} + f^2\frac{m^2}{k_h^2 + m^2}$$

    where N = Brunt-Väisälä, f = Coriolis, kh = horizontal, m = vertical wavenumber.

    Group velocities:
    - Horizontal: c_{gx} = N² m² k_h / (ω(k_h² + m²)²)
    - Vertical: c_{gz} = -N² k_h² m / (ω(k_h² + m²)²)
    """

    def __init__(self, N: float = 1e-3, f: float = 1e-4) -> None:
        """
        Parameters
        ----------
        N : Buoyancy frequency (s⁻¹).
        f : Coriolis parameter (s⁻¹).
        """
        self.N = N
        self.f = f

    def frequency(self, kh: float, m: float) -> float:
        """ω(kh, m) from dispersion relation."""
        k2 = kh**2 + m**2
        if k2 < 1e-30:
            return 0.0
        omega_sq = self.N**2 * kh**2 / k2 + self.f**2 * m**2 / k2
        return math.sqrt(max(omega_sq, 0.0))

    def group_velocity_horizontal(self, kh: float, m: float) -> float:
        """cg_x = ∂ω/∂kh."""
        omega = self.frequency(kh, m)
        k2 = kh**2 + m**2
        if omega < 1e-30 or k2 < 1e-30:
            return 0.0
        return (self.N**2 - self.f**2) * m**2 * kh / (omega * k2**2)

    def group_velocity_vertical(self, kh: float, m: float) -> float:
        """cg_z = ∂ω/∂m."""
        omega = self.frequency(kh, m)
        k2 = kh**2 + m**2
        if omega < 1e-30 or k2 < 1e-30:
            return 0.0
        return -(self.N**2 - self.f**2) * kh**2 * m / (omega * k2**2)

    def vertical_modes(self, H: float, n_modes: int = 5) -> NDArray[np.float64]:
        """Vertical normal mode wavenumbers m_n = nπ/H."""
        return np.array([n * math.pi / H for n in range(1, n_modes + 1)])

    def ray_tracing(self, kh0: float, m0: float, x0: float, z0: float,
                     dt: float, n_steps: int,
                     N_profile: Optional[NDArray] = None) -> Tuple[NDArray, NDArray]:
        """
        Ray tracing for internal waves in uniform stratification.

        Returns (x, z) trajectory arrays.
        """
        x = np.zeros(n_steps + 1)
        z = np.zeros(n_steps + 1)
        x[0], z[0] = x0, z0
        kh, m = kh0, m0

        for i in range(n_steps):
            cgx = self.group_velocity_horizontal(kh, m)
            cgz = self.group_velocity_vertical(kh, m)
            x[i + 1] = x[i] + cgx * dt
            z[i + 1] = z[i] + cgz * dt

        return x, z


# ---------------------------------------------------------------------------
#  Tidal Harmonics
# ---------------------------------------------------------------------------

@dataclass
class TidalConstituent:
    """Single tidal harmonic constituent."""
    name: str
    period_hours: float
    amplitude: float    # m
    phase: float        # radians

    @property
    def frequency(self) -> float:
        """Angular frequency ω (rad/s)."""
        return 2.0 * math.pi / (self.period_hours * 3600.0)


class TidalHarmonics:
    """
    Tidal prediction from harmonic constituents.

    Standard constituents: M2, S2, N2, K1, O1, P1, etc.
    """

    # Standard constituent periods (hours)
    STANDARD: Dict[str, float] = {
        "M2": 12.4206,   # Principal lunar semidiurnal
        "S2": 12.0000,   # Principal solar semidiurnal
        "N2": 12.6583,   # Larger lunar elliptic
        "K2": 11.9672,   # Lunisolar semidiurnal
        "K1": 23.9345,   # Lunisolar diurnal
        "O1": 25.8193,   # Principal lunar diurnal
        "P1": 24.0659,   # Principal solar diurnal
        "Q1": 26.8684,   # Larger lunar elliptic diurnal
        "M4": 6.2103,    # Shallow water quarter-diurnal
        "M6": 4.1402,    # Shallow water sixth-diurnal
    }

    def __init__(self) -> None:
        self.constituents: List[TidalConstituent] = []

    def add_constituent(self, name: str, amplitude: float,
                          phase: float = 0.0) -> None:
        """Add a tidal constituent."""
        if name in self.STANDARD:
            period = self.STANDARD[name]
        else:
            raise ValueError(f"Unknown constituent: {name}")

        self.constituents.append(
            TidalConstituent(name=name, period_hours=period,
                             amplitude=amplitude, phase=phase)
        )

    def predict(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predict tidal height η(t).

        Parameters
        ----------
        t : Time array (seconds from epoch).
        """
        eta = np.zeros_like(t)
        for c in self.constituents:
            eta += c.amplitude * np.cos(c.frequency * t + c.phase)
        return eta

    def form_factor(self) -> float:
        """F = (K1 + O1) / (M2 + S2). Classifies tide type.

        F < 0.25: semidiurnal
        0.25 < F < 1.5: mixed, mainly semidiurnal
        1.5 < F < 3.0: mixed, mainly diurnal
        F > 3.0: diurnal
        """
        K1 = sum(c.amplitude for c in self.constituents if c.name == "K1")
        O1 = sum(c.amplitude for c in self.constituents if c.name == "O1")
        M2 = sum(c.amplitude for c in self.constituents if c.name == "M2")
        S2 = sum(c.amplitude for c in self.constituents if c.name == "S2")

        denom = M2 + S2
        if denom < 1e-10:
            return float('inf')
        return (K1 + O1) / denom

    def harmonic_analysis(self, t: NDArray[np.float64],
                            eta: NDArray[np.float64],
                            constituent_names: List[str]) -> Dict[str, Tuple[float, float]]:
        """
        Fit amplitudes and phases from time series (least squares).

        Returns {name: (amplitude, phase)}.
        """
        n_const = len(constituent_names)
        frequencies = []
        for name in constituent_names:
            period = self.STANDARD.get(name, 12.0)
            frequencies.append(2.0 * math.pi / (period * 3600.0))

        # Build design matrix: [cos(ω₁t), sin(ω₁t), cos(ω₂t), sin(ω₂t), ...]
        A = np.zeros((len(t), 2 * n_const))
        for k, omega in enumerate(frequencies):
            A[:, 2 * k] = np.cos(omega * t)
            A[:, 2 * k + 1] = np.sin(omega * t)

        # Least squares
        coeffs, _, _, _ = np.linalg.lstsq(A, eta, rcond=None)

        result: Dict[str, Tuple[float, float]] = {}
        for k, name in enumerate(constituent_names):
            a_cos = coeffs[2 * k]
            a_sin = coeffs[2 * k + 1]
            amplitude = math.sqrt(a_cos**2 + a_sin**2)
            phase = math.atan2(-a_sin, a_cos)
            result[name] = (amplitude, phase)

        return result
