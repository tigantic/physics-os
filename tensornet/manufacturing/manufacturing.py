"""
Manufacturing Physics: Goldak welding heat source, Scheil solidification,
Marangoni-driven AM melt pool, Merchant machining model.

Upgrades domain XX.9.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Goldak Welding Heat Source
# ---------------------------------------------------------------------------

class GoldakWeldingSource:
    r"""
    Goldak double-ellipsoid heat source for welding simulation.

    Front ellipsoid:
    $$q_f = \frac{6\sqrt{3}f_f Q}{\pi\sqrt{\pi}a b c_f}
            \exp\left(-3\frac{x^2}{a^2} - 3\frac{y^2}{b^2} - 3\frac{(\xi-v t)^2}{c_f^2}\right)$$

    Rear ellipsoid: same with $c_r$ replacing $c_f$.

    Parameters: fraction $f_f + f_r = 2$, Q = ηVI (power).
    """

    def __init__(self, Q: float, v: float,
                 a: float = 5e-3, b: float = 5e-3,
                 c_f: float = 5e-3, c_r: float = 10e-3,
                 f_f: float = 0.6) -> None:
        """
        Parameters
        ----------
        Q : Net heat input (W) = η V I.
        v : Welding speed (m/s).
        a : Width semi-axis (m).
        b : Depth semi-axis (m).
        c_f, c_r : Front and rear length semi-axes (m).
        f_f : Front fraction (0-2). f_r = 2 - f_f.
        """
        self.Q = Q
        self.v = v
        self.a = a
        self.b = b
        self.c_f = c_f
        self.c_r = c_r
        self.f_f = f_f
        self.f_r = 2.0 - f_f

    def power_density(self, x: NDArray, y: NDArray,
                        xi: NDArray, t: float) -> NDArray[np.float64]:
        """Volumetric heat generation rate (W/m³) at time t.

        xi = coordinate along welding direction.
        """
        z_rel = xi - self.v * t  # relative to source centre

        # Front (z_rel >= 0)
        q_front = (6 * math.sqrt(3) * self.f_f * self.Q
                    / (math.pi * math.sqrt(math.pi) * self.a * self.b * self.c_f)
                    * np.exp(-3 * x**2 / self.a**2
                              - 3 * y**2 / self.b**2
                              - 3 * z_rel**2 / self.c_f**2))

        # Rear (z_rel < 0)
        q_rear = (6 * math.sqrt(3) * self.f_r * self.Q
                   / (math.pi * math.sqrt(math.pi) * self.a * self.b * self.c_r)
                   * np.exp(-3 * x**2 / self.a**2
                             - 3 * y**2 / self.b**2
                             - 3 * z_rel**2 / self.c_r**2))

        return np.where(z_rel >= 0, q_front, q_rear)

    def peak_temperature_estimate(self, k: float, rho: float,
                                     cp: float) -> float:
        """Rosenthal point-source estimate of peak temperature.

        T_peak - T_0 ≈ Q / (2π k R_min) where R_min ~ √(a·b).
        """
        R_min = math.sqrt(self.a * self.b)
        return self.Q / (2 * math.pi * k * R_min)

    def weld_pool_length(self, k: float, rho: float,
                           cp: float, T_melt: float,
                           T_ambient: float = 300.0) -> float:
        """Estimate weld pool length from Rosenthal equation.

        L_pool ≈ Q / (2π k (T_melt - T_0) v) for high-speed limit.
        """
        return self.Q / (2 * math.pi * k * (T_melt - T_ambient)) * 1.0 / (self.v + 1e-10)


# ---------------------------------------------------------------------------
#  1D Welding Heat Transfer
# ---------------------------------------------------------------------------

class WeldingHeatTransfer1D:
    """
    1D transient heat conduction with Goldak source (line heating).

    ∂T/∂t = α ∂²T/∂x² + q(x,t)/(ρc_p)

    Explicit finite difference with phase-change (enthalpy method).
    """

    def __init__(self, L: float, nx: int,
                 rho: float = 7800.0, cp: float = 500.0,
                 k: float = 40.0) -> None:
        self.L = L
        self.nx = nx
        self.dx = L / (nx - 1)
        self.rho = rho
        self.cp = cp
        self.k = k
        self.alpha = k / (rho * cp)

        self.T = np.ones(nx) * 300.0  # ambient
        self.T_melt = 1773.0  # K (steel)
        self.T_liquidus = 1823.0
        self.L_latent = 270e3  # J/kg

    def step(self, dt: float, q_source: NDArray[np.float64]) -> None:
        """Explicit Euler step with latent heat."""
        T_new = self.T.copy()
        for i in range(1, self.nx - 1):
            laplacian = (self.T[i + 1] - 2 * self.T[i] + self.T[i - 1]) / self.dx**2
            dTdt = self.alpha * laplacian + q_source[i] / (self.rho * self.cp)

            # Effective cp for latent heat (enthalpy method)
            if self.T_melt <= self.T[i] <= self.T_liquidus:
                cp_eff = self.cp + self.L_latent / (self.T_liquidus - self.T_melt)
                dTdt = self.k * laplacian / (self.rho * cp_eff) + q_source[i] / (self.rho * cp_eff)

            T_new[i] = self.T[i] + dt * dTdt

        # Natural convection BC at boundaries
        h_conv = 10.0  # W/m²K
        T_new[0] = self.T[0] + dt * (self.alpha * 2 * (self.T[1] - self.T[0]) / self.dx**2
                                       - h_conv * (self.T[0] - 300.0) / (self.rho * self.cp * self.dx))
        T_new[-1] = self.T[-1] + dt * (self.alpha * 2 * (self.T[-2] - self.T[-1]) / self.dx**2
                                         - h_conv * (self.T[-1] - 300.0) / (self.rho * self.cp * self.dx))

        self.T = T_new


# ---------------------------------------------------------------------------
#  Scheil Solidification Model
# ---------------------------------------------------------------------------

class ScheilSolidification:
    r"""
    Scheil-Gulliver solidification model (non-equilibrium).

    $$C_L = C_0 (1 - f_s)^{k_p - 1}$$

    where C_L = liquid composition, f_s = solid fraction, k_p = partition coefficient.

    Temperature:
    $$T = T_m + m_L C_L$$

    Solidification range, microsegregation, eutectic fraction.
    """

    def __init__(self, C0: float, k_p: float,
                 m_L: float, T_melt: float,
                 T_eutectic: float = 0.0,
                 C_eutectic: float = float('inf')) -> None:
        """
        Parameters
        ----------
        C0 : Nominal composition (wt%).
        k_p : Partition coefficient (<1 for most binary alloys).
        m_L : Liquidus slope (K/wt%).
        T_melt : Pure solvent melting temperature (K).
        T_eutectic : Eutectic temperature (K).
        C_eutectic : Eutectic composition (wt%).
        """
        self.C0 = C0
        self.k_p = k_p
        self.m_L = m_L
        self.T_m = T_melt
        self.T_eut = T_eutectic
        self.C_eut = C_eutectic

    def liquidus_composition(self, fs: NDArray[np.float64]) -> NDArray[np.float64]:
        """C_L(f_s)."""
        return self.C0 * (1 - fs)**(self.k_p - 1)

    def solid_composition(self, fs: NDArray[np.float64]) -> NDArray[np.float64]:
        """C_S = k_p C_L at interface."""
        return self.k_p * self.liquidus_composition(fs)

    def temperature(self, fs: NDArray[np.float64]) -> NDArray[np.float64]:
        """T(f_s) = T_m + m_L C_L."""
        CL = self.liquidus_composition(fs)
        T = self.T_m + self.m_L * CL
        # Cap at eutectic
        if self.T_eut > 0:
            T = np.maximum(T, self.T_eut)
        return T

    def solidification_curve(self, n_pts: int = 200) -> Tuple[NDArray, NDArray, NDArray]:
        """Full Scheil curve.

        Returns (fs, T, CL).
        """
        fs = np.linspace(0, 0.999, n_pts)
        CL = self.liquidus_composition(fs)
        T = self.temperature(fs)

        # Find eutectic fraction
        if self.T_eut > 0:
            eutectic_idx = np.searchsorted(-T, -self.T_eut)
            fs = fs[:eutectic_idx + 1]
            T = T[:eutectic_idx + 1]
            CL = CL[:eutectic_idx + 1]

        return fs, T, CL

    def eutectic_fraction(self) -> float:
        """Fraction of eutectic formed: f_eut = 1 - f_s at T_eut."""
        if self.T_eut <= 0 or self.C_eut >= 1e10:
            return 0.0
        # C_L = C_eut → f_s = 1 - (C_eut/C0)^(1/(k_p-1))
        fs_eut = 1.0 - (self.C_eut / self.C0)**(1.0 / (self.k_p - 1.0))
        return max(1.0 - fs_eut, 0.0)

    def solidification_range(self) -> float:
        """T_liquidus - T_solidus (or T_eutectic)."""
        T_liq = self.T_m + self.m_L * self.C0
        T_sol = self.T_eut if self.T_eut > 0 else self.T_m + self.m_L * self.C0 / self.k_p
        return T_liq - T_sol


# ---------------------------------------------------------------------------
#  Marangoni-Driven Additive Manufacturing Melt Pool
# ---------------------------------------------------------------------------

class MarangoniMeltPool:
    r"""
    Marangoni (thermocapillary) convection in laser/e-beam AM melt pool.

    Free-surface condition:
    $$\mu\frac{\partial u}{\partial z}\bigg|_s = \frac{\partial\gamma}{\partial T}\frac{\partial T}{\partial x}$$

    Key numbers:
    - Marangoni: $Ma = -\frac{\partial\gamma}{\partial T}\frac{\Delta T L}{\mu\alpha}$
    - Péclet: $Pe = uL/\alpha$

    Velocity scale: $u \sim |\partial\gamma/\partial T|\Delta T / \mu$
    """

    def __init__(self, power: float, velocity: float,
                 spot_radius: float = 50e-6) -> None:
        """
        Parameters
        ----------
        power : Laser/beam power (W).
        velocity : Scan speed (m/s).
        spot_radius : Beam spot radius (m).
        """
        self.P = power
        self.v_scan = velocity
        self.r_beam = spot_radius

        # Material (SS316L default)
        self.rho = 7900.0
        self.cp = 500.0
        self.k_th = 20.0
        self.mu = 6e-3         # dynamic viscosity (Pa·s)
        self.alpha_th = self.k_th / (self.rho * self.cp)
        self.dgamma_dT = -4.3e-4  # surface tension temperature coefficient (N/m·K)
        self.T_melt = 1700.0
        self.T_boil = 3100.0
        self.absorptivity = 0.35

    @property
    def power_density(self) -> float:
        """Peak power density (W/m²)."""
        return self.absorptivity * self.P / (math.pi * self.r_beam**2)

    def melt_pool_depth(self) -> float:
        """Estimate depth from Rosenthal line source.

        d ~ √(α P_abs / (π k v ΔT))
        """
        P_abs = self.absorptivity * self.P
        delta_T = self.T_melt - 300.0
        return math.sqrt(self.alpha_th * P_abs / (math.pi * self.k_th * self.v_scan * delta_T + 1e-30))

    def melt_pool_length(self) -> float:
        """Estimate length: L ~ P_abs / (2π k ΔT)."""
        P_abs = self.absorptivity * self.P
        delta_T = self.T_melt - 300.0
        return P_abs / (2 * math.pi * self.k_th * delta_T + 1e-30)

    def marangoni_number(self) -> float:
        """Ma = |dγ/dT| ΔT L / (μ α)."""
        delta_T = self.T_boil - self.T_melt
        L = self.melt_pool_length()
        return abs(self.dgamma_dT) * delta_T * L / (self.mu * self.alpha_th)

    def surface_velocity(self) -> float:
        """Estimate: u ~ |dγ/dT| ΔT / μ."""
        delta_T = self.T_boil - self.T_melt
        return abs(self.dgamma_dT) * delta_T / self.mu

    def peclet_number(self) -> float:
        """Pe = u L / α."""
        u = self.surface_velocity()
        L = self.melt_pool_length()
        return u * L / self.alpha_th

    def keyhole_threshold(self) -> float:
        """Approximate threshold power density for keyhole formation (W/m²).

        ~ ρ L_v v √(α v) where L_v is latent heat of vaporisation.
        """
        L_vap = 6.3e6  # J/kg for steel
        return self.rho * L_vap * math.sqrt(self.alpha_th * self.v_scan)


# ---------------------------------------------------------------------------
#  Merchant Machining Model
# ---------------------------------------------------------------------------

class MerchantMachining:
    r"""
    Ernst Merchant's metal cutting model (orthogonal cutting).

    Shear angle from minimum energy:
    $$\phi = \frac{\pi}{4} - \frac{\beta - \alpha}{2}$$

    where β = friction angle = arctan(μ), α = rake angle.

    Cutting force: $F_c = \frac{\tau_s w t}{\cos(\beta-\alpha)\sin\phi}$
    """

    def __init__(self, tau_s: float, mu_friction: float,
                 alpha_rake: float = 0.0) -> None:
        """
        Parameters
        ----------
        tau_s : Shear yield stress of workpiece (Pa).
        mu_friction : Friction coefficient at tool-chip interface.
        alpha_rake : Tool rake angle (radians).
        """
        self.tau_s = tau_s
        self.mu = mu_friction
        self.alpha = alpha_rake
        self.beta = math.atan(mu_friction)

    @property
    def shear_angle(self) -> float:
        """Merchant's shear angle φ (radians)."""
        return math.pi / 4.0 - (self.beta - self.alpha) / 2.0

    @property
    def chip_thickness_ratio(self) -> float:
        """r = t/t_c = sin(φ)/cos(φ-α)."""
        phi = self.shear_angle
        return math.sin(phi) / math.cos(phi - self.alpha)

    def cutting_force(self, width: float, depth: float) -> float:
        """F_c (N) = τ_s w t / [cos(β-α) sin(φ)]."""
        phi = self.shear_angle
        return (self.tau_s * width * depth
                / (math.cos(self.beta - self.alpha) * math.sin(phi)))

    def thrust_force(self, width: float, depth: float) -> float:
        """F_t (N) = F_c tan(β - α)."""
        Fc = self.cutting_force(width, depth)
        return Fc * math.tan(self.beta - self.alpha)

    def specific_cutting_energy(self, width: float, depth: float) -> float:
        """u = F_c / (w t) (J/m³)."""
        return self.cutting_force(width, depth) / (width * depth)

    def maximum_temperature_rise(self, V: float, width: float,
                                    depth: float, rho: float = 7800.0,
                                    cp: float = 500.0) -> float:
        """Estimate chip temperature rise.

        ΔT ≈ (1-Γ) F_c V / (ρ c_p V_chip)
        Γ ≈ 0.1 (fraction of heat to workpiece).
        """
        Fc = self.cutting_force(width, depth)
        Gamma = 0.1
        V_chip = V * self.chip_thickness_ratio  # chip velocity
        Q_chip = (1 - Gamma) * Fc * V
        mass_rate = rho * width * depth * V
        return Q_chip / (mass_rate * cp) if mass_rate > 0 else 0.0

    def taylor_tool_life(self, V: float, C: float = 300.0,
                           n: float = 0.25) -> float:
        """Taylor tool life equation: V T^n = C.

        T = (C/V)^(1/n) in minutes.
        """
        return (C / V)**(1.0 / n)
