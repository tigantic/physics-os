"""
Environmental Physics: Gaussian plume dispersion, SCS curve number hydrology,
storm surge shallow-water, fire-atmosphere coupling.

Upgrades domain XX.7.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Gaussian Plume Dispersion
# ---------------------------------------------------------------------------

class GaussianPlume:
    r"""
    Gaussian plume model for atmospheric pollutant dispersion.

    Concentration at (x, y, z):
    $$C = \frac{Q}{2\pi u\sigma_y\sigma_z}
          \exp\left(-\frac{y^2}{2\sigma_y^2}\right)
          \left[\exp\left(-\frac{(z-H)^2}{2\sigma_z^2}\right)
                +\exp\left(-\frac{(z+H)^2}{2\sigma_z^2}\right)\right]$$

    σ_y, σ_z = Pasquill-Gifford dispersion coefficients.
    """

    # Pasquill-Gifford coefficients: σ_y = a x^b, σ_z = c x^d
    # For stability class A-F
    _PG_PARAMS = {
        'A': (0.22, 0.894, 0.20, 0.894),
        'B': (0.16, 0.894, 0.12, 0.894),
        'C': (0.11, 0.894, 0.08, 0.894),
        'D': (0.08, 0.894, 0.06, 0.894),
        'E': (0.06, 0.894, 0.03, 0.894),
        'F': (0.04, 0.894, 0.016, 0.894),
    }

    def __init__(self, Q: float, H: float, u: float,
                 stability: str = 'D') -> None:
        """
        Parameters
        ----------
        Q : Source emission rate (g/s or kg/s).
        H : Effective stack height (m), including plume rise.
        u : Wind speed at stack height (m/s).
        stability : Pasquill stability class ('A'-'F').
        """
        self.Q = Q
        self.H = H
        self.u = u
        self.stability = stability.upper()
        params = self._PG_PARAMS[self.stability]
        self._ay, self._by = params[0], params[1]
        self._az, self._bz = params[2], params[3]

    def sigma_y(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._ay * x**self._by

    def sigma_z(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._az * x**self._bz

    def concentration(self, x: NDArray, y: NDArray,
                        z: NDArray) -> NDArray[np.float64]:
        """C(x, y, z) in same units as Q/(m³·m/s)."""
        x = np.maximum(x, 1.0)  # avoid x=0
        sy = self.sigma_y(x)
        sz = self.sigma_z(x)

        lateral = np.exp(-0.5 * (y / sy)**2)
        vertical = (np.exp(-0.5 * ((z - self.H) / sz)**2)
                     + np.exp(-0.5 * ((z + self.H) / sz)**2))

        return self.Q / (2 * math.pi * self.u * sy * sz) * lateral * vertical

    def ground_level_centreline(self, x: NDArray) -> NDArray[np.float64]:
        """C(x, 0, 0) — maximum ground-level concentration."""
        return self.concentration(x, np.zeros_like(x), np.zeros_like(x))

    def max_ground_concentration(self) -> Tuple[float, float]:
        """Find x_max and C_max at ground level centreline."""
        x_test = np.linspace(100, 50000, 10000)
        C = self.ground_level_centreline(x_test)
        idx = np.argmax(C)
        return float(x_test[idx]), float(C[idx])

    def plume_rise_briggs(self, buoyancy_flux: float,
                            x_downwind: float) -> float:
        """Briggs plume rise formula.

        ΔH = 1.6 F^{1/3} x^{2/3} / u  (buoyant, unstable/neutral)
        F = g V_s d² ΔT / (4 T_s)
        """
        return 1.6 * buoyancy_flux**(1.0 / 3.0) * x_downwind**(2.0 / 3.0) / self.u


# ---------------------------------------------------------------------------
#  SCS Curve Number Hydrology
# ---------------------------------------------------------------------------

class SCSCurveNumber:
    r"""
    SCS (NRCS) Curve Number method for rainfall-runoff.

    $$Q = \frac{(P - I_a)^2}{P - I_a + S}, \quad P > I_a$$

    where:
    - P = precipitation (mm)
    - I_a = 0.2 S (initial abstraction)
    - S = 25400/CN - 254 (maximum retention, mm)
    - CN = curve number (0-100)

    Time of concentration: Kirpich, SCS lag equations.
    """

    def __init__(self, CN: float, area_km2: float = 1.0) -> None:
        """
        Parameters
        ----------
        CN : Curve number (typically 30-98).
        area_km2 : Catchment area (km²).
        """
        self.CN = CN
        self.area = area_km2

    @property
    def S_mm(self) -> float:
        """Maximum retention S (mm)."""
        return 25400.0 / self.CN - 254.0

    @property
    def initial_abstraction(self) -> float:
        """Ia = 0.2 S (mm)."""
        return 0.2 * self.S_mm

    def runoff_depth(self, P: float) -> float:
        """Direct runoff Q (mm) for rainfall P (mm)."""
        Ia = self.initial_abstraction
        if P <= Ia:
            return 0.0
        return (P - Ia)**2 / (P - Ia + self.S_mm)

    def runoff_volume(self, P: float) -> float:
        """Runoff volume (m³)."""
        Q_mm = self.runoff_depth(P)
        return Q_mm * 1e-3 * self.area * 1e6  # mm to m, km² to m²

    def scs_unit_hydrograph(self, dt: float,
                              Tp: float) -> Tuple[NDArray, NDArray]:
        """SCS dimensionless unit hydrograph.

        Tp = time to peak (hours).
        Returns (time_hours, q/q_peak).
        """
        # Dimensionless UH: gamma distribution approximation
        t = np.arange(0, 5 * Tp, dt)
        t_ratio = t / Tp
        # SCS shape: q = (t/Tp)^m exp(m(1 - t/Tp)) where m ≈ 3.7
        m = 3.7
        q = t_ratio**m * np.exp(m * (1.0 - t_ratio))
        q = np.maximum(q, 0)

        return t, q / (np.max(q) + 1e-30)

    @staticmethod
    def adjust_cn_for_moisture(CN_II: float,
                                 condition: str = 'II') -> float:
        """Adjust CN for antecedent moisture condition.

        I = dry, II = normal, III = wet.
        """
        if condition == 'I':
            return 4.2 * CN_II / (10 - 0.058 * CN_II)
        elif condition == 'III':
            return 23 * CN_II / (10 + 0.13 * CN_II)
        return CN_II


# ---------------------------------------------------------------------------
#  Storm Surge (1D Shallow Water)
# ---------------------------------------------------------------------------

class StormSurge1D:
    r"""
    1D storm surge model using shallow water equations with wind and
    pressure forcing.

    $$\frac{\partial\eta}{\partial t} + \frac{\partial(Hu)}{\partial x} = 0$$
    $$\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x}
      = -g\frac{\partial\eta}{\partial x}
        + \frac{\tau_s}{\rho H} - \frac{\tau_b}{\rho H}
        - \frac{1}{\rho}\frac{\partial p_a}{\partial x}$$

    where:
    - τ_s = C_d ρ_a |W| W (wind stress)
    - τ_b = C_f ρ |u| u (bottom friction)
    - p_a = atmospheric pressure
    """

    def __init__(self, L: float, nx: int, h0: float = 10.0,
                 g: float = 9.81) -> None:
        """
        Parameters
        ----------
        L : Domain length (m).
        nx : Number of cells.
        h0 : Undisturbed water depth (m).
        g : Gravity.
        """
        self.L = L
        self.nx = nx
        self.dx = L / nx
        self.g = g
        self.h0 = h0

        self.eta = np.zeros(nx)  # surface elevation
        self.u = np.zeros(nx)    # velocity

        # Bathymetry (positive down)
        self.depth = np.ones(nx) * h0

        # Wind and pressure
        self.wind = np.zeros(nx)  # wind speed (m/s)
        self.p_atm = np.ones(nx) * 101325.0  # Pa

        # Friction
        self.Cd_wind = 1.3e-3
        self.Cf_bottom = 2.5e-3
        self.rho_water = 1025.0
        self.rho_air = 1.225

    def total_depth(self) -> NDArray[np.float64]:
        return self.depth + self.eta

    def step(self, dt: float) -> None:
        """Lax-Friedrichs time step."""
        H = self.total_depth()
        H = np.maximum(H, 0.01)  # prevent drying

        # Fluxes
        Hu = H * self.u

        # Lax-Friedrichs for continuity
        eta_new = np.zeros_like(self.eta)
        for i in range(1, self.nx - 1):
            eta_new[i] = 0.5 * (self.eta[i + 1] + self.eta[i - 1]) - dt / (2 * self.dx) * (Hu[i + 1] - Hu[i - 1])

        # Boundary: wall at x=0, open at x=L
        eta_new[0] = eta_new[1]
        eta_new[-1] = eta_new[-2]

        # Momentum
        u_new = np.zeros_like(self.u)
        for i in range(1, self.nx - 1):
            # Advection + pressure gradient
            u_new[i] = (0.5 * (self.u[i + 1] + self.u[i - 1])
                         - dt / (2 * self.dx) * (
                             self.u[i] * (self.u[i + 1] - self.u[i - 1])
                             + self.g * (self.eta[i + 1] - self.eta[i - 1])
                             + (self.p_atm[i + 1] - self.p_atm[i - 1]) / (self.rho_water * 2 * self.dx)
                         ))

            # Wind stress
            tau_s = self.Cd_wind * self.rho_air * abs(self.wind[i]) * self.wind[i]
            u_new[i] += dt * tau_s / (self.rho_water * H[i])

            # Bottom friction
            tau_b = self.Cf_bottom * self.rho_water * abs(self.u[i]) * self.u[i]
            u_new[i] -= dt * tau_b / (self.rho_water * H[i])

        # BCs
        u_new[0] = 0.0  # wall
        u_new[-1] = u_new[-2]

        self.eta = eta_new
        self.u = u_new

    def set_hurricane_wind(self, x_eye: float, R_max: float,
                             V_max: float) -> None:
        """Holland hurricane wind profile.

        V(r) = V_max (R_max/r)^B exp(1 - (R_max/r)^B) with B ≈ 1.5.
        """
        x = np.arange(self.nx) * self.dx
        r = np.abs(x - x_eye) + 1.0  # avoid r=0
        B = 1.5
        self.wind = V_max * (R_max / r)**B * np.exp((1.0 - (R_max / r)**B) / B)

    def max_surge(self) -> float:
        """Maximum surge elevation (m)."""
        return float(np.max(self.eta))


# ---------------------------------------------------------------------------
#  Fire-Atmosphere Coupling
# ---------------------------------------------------------------------------

class FireAtmosphere:
    r"""
    Simplified fire-atmosphere coupling (Rothermel + plume dynamics).

    Fire spread rate (Rothermel, 1972):
    $$R = R_0(1 + \phi_W + \phi_S)$$

    where:
    - $R_0$ = no-wind, no-slope rate
    - $\phi_W$ = wind factor ∝ U^B
    - $\phi_S$ = slope factor ∝ tan(φ)²

    Plume rise: Byram's fireline intensity I_B = H w R.
    Convection column: entrainment model.
    """

    def __init__(self, nx: int, ny: int, dx: float) -> None:
        self.nx = nx
        self.ny = ny
        self.dx = dx

        self.fuel_load = np.ones((nx, ny)) * 1.0       # kg/m²
        self.moisture = np.ones((nx, ny)) * 0.1         # fraction
        self.slope = np.zeros((nx, ny))                  # radians
        self.wind_speed = np.ones((nx, ny)) * 5.0       # m/s
        self.wind_dir = np.zeros((nx, ny))               # radians from +x
        self.burning = np.zeros((nx, ny), dtype=bool)
        self.burned = np.zeros((nx, ny), dtype=bool)

        # Fuel properties
        self.heat_content = 18000.0  # kJ/kg
        self.extinction_moisture = 0.30

    def spread_rate(self, i: int, j: int) -> float:
        """Rothermel-like spread rate at cell (i,j) in m/s."""
        if self.burned[i, j] or self.fuel_load[i, j] < 0.01:
            return 0.0

        # Moisture damping
        M = self.moisture[i, j]
        M_x = self.extinction_moisture
        if M >= M_x:
            return 0.0
        eta_M = 1.0 - 2.59 * (M / M_x) + 5.11 * (M / M_x)**2 - 3.52 * (M / M_x)**3

        # Base rate (simplified)
        R0 = 0.01 * self.fuel_load[i, j] * eta_M  # m/s baseline

        # Wind factor
        U = self.wind_speed[i, j]
        phi_W = 0.4 * U**0.5

        # Slope factor
        phi_S = 5.275 * math.tan(self.slope[i, j])**2

        return R0 * (1.0 + phi_W + phi_S)

    def fireline_intensity(self, i: int, j: int) -> float:
        """Byram's fireline intensity I_B = H w R (kW/m)."""
        R = self.spread_rate(i, j)
        w = self.fuel_load[i, j]
        return self.heat_content * w * R

    def step(self, dt: float) -> int:
        """Advance fire spread one time step.

        Returns number of newly ignited cells.
        """
        new_ignitions = 0
        new_burning = self.burning.copy()

        for i in range(self.nx):
            for j in range(self.ny):
                if not self.burning[i, j]:
                    continue

                R = self.spread_rate(i, j)
                if R <= 0:
                    continue

                spread_dist = R * dt

                # Check 8-connected neighbours
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.nx and 0 <= nj < self.ny:
                            if not self.burned[ni, nj] and not new_burning[ni, nj]:
                                cell_dist = self.dx * math.sqrt(di**2 + dj**2)
                                if spread_dist >= cell_dist:
                                    new_burning[ni, nj] = True
                                    new_ignitions += 1

        # Update burn state
        for i in range(self.nx):
            for j in range(self.ny):
                if self.burning[i, j]:
                    self.fuel_load[i, j] -= self.fuel_load[i, j] * 0.1 * dt
                    if self.fuel_load[i, j] < 0.01:
                        self.burning[i, j] = False
                        self.burned[i, j] = True

        self.burning = new_burning
        return new_ignitions

    def ignite(self, i: int, j: int) -> None:
        """Ignite cell (i,j)."""
        self.burning[i, j] = True
