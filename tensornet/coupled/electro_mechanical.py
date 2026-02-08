"""
Electro-Mechanical Coupling — Piezoelectric, MEMS pull-in, electrostatic actuator.

Domain XVIII.3 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Piezoelectric Constitutive
# ---------------------------------------------------------------------------

class PiezoelectricSolver:
    r"""
    Linear piezoelectric constitutive equations (stress form):

    $$\sigma_{ij} = C_{ijkl}^E \varepsilon_{kl} - e_{kij} E_k$$
    $$D_i = e_{ikl}\varepsilon_{kl} + \kappa_{ij}^\varepsilon E_j$$

    Reduced 1D model for a piezoelectric beam actuator:
    $$\sigma = C^E\varepsilon - e_{31}E_3$$
    $$D_3 = e_{31}\varepsilon + \kappa_{33}^\varepsilon E_3$$

    Material: PZT-5A typical values.
    """

    def __init__(self, n_elem: int = 50, L: float = 0.05,
                 width: float = 0.01, thickness: float = 0.5e-3,
                 C_E: float = 66e9, e31: float = -12.3,
                 kappa33: float = 1700 * 8.854e-12,
                 rho: float = 7750.0) -> None:
        self.n = n_elem
        self.L = L
        self.w = width
        self.h = thickness
        self.dx = L / n_elem
        self.C_E = C_E
        self.e31 = e31
        self.kappa33 = kappa33
        self.rho = rho
        self.A = width * thickness

        self.u = np.zeros(n_elem + 1)  # axial displacement
        self.phi = np.zeros(n_elem + 1)  # electric potential
        self.x = np.linspace(0, L, n_elem + 1)

    def strain(self) -> NDArray:
        """Axial strain ε = du/dx (element-centered)."""
        return np.diff(self.u) / self.dx

    def electric_field(self) -> NDArray:
        """E = −dφ/dx (element-centered, but for thickness-mode: E = V/h)."""
        return -np.diff(self.phi) / self.dx

    def stress(self) -> NDArray:
        """σ = C^E ε − e₃₁ E."""
        eps = self.strain()
        E = self.electric_field()
        return self.C_E * eps - self.e31 * E

    def electric_displacement(self) -> NDArray:
        """D = e₃₁ ε + κ₃₃ E."""
        eps = self.strain()
        E = self.electric_field()
        return self.e31 * eps + self.kappa33 * E

    def solve_static(self, V_applied: float = 100.0,
                       fixed_end: str = 'left') -> NDArray:
        """Solve coupled piezoelectric beam (thickness-mode actuation).

        Apply voltage across thickness → uniform E₃ = V/h → axial strain.
        Returns tip displacement.
        """
        E3 = V_applied / self.h
        free_strain = self.e31 * E3 / self.C_E  # d₃₁ · E₃

        # Free extension: u(x) = free_strain * x (clamped left)
        self.u = free_strain * self.x
        if fixed_end == 'left':
            self.u -= self.u[0]
        elif fixed_end == 'right':
            self.u -= self.u[-1]

        self.phi = np.linspace(0, V_applied, self.n + 1)
        return self.u

    def coupling_coefficient(self) -> float:
        r"""Electromechanical coupling: $k_{31}^2 = e_{31}^2/(C^E\kappa_{33})$."""
        return self.e31**2 / (self.C_E * self.kappa33)

    def resonance_frequency(self) -> float:
        """Fundamental longitudinal resonance: f = 1/(2L)√(C^E/ρ)."""
        return 0.5 / self.L * math.sqrt(self.C_E / self.rho)

    def energy_harvested(self, strain_amplitude: float = 1e-4) -> float:
        """Harvested electrical energy per cycle per unit volume.

        W = ½ k² C^E ε² (open circuit).
        """
        k2 = self.coupling_coefficient()
        return 0.5 * k2 * self.C_E * strain_amplitude**2


# ---------------------------------------------------------------------------
#  MEMS Electrostatic Pull-In
# ---------------------------------------------------------------------------

class MEMSPullIn:
    r"""
    Parallel-plate MEMS electrostatic actuator with pull-in instability.

    $$m\ddot{x} + c\dot{x} + kx = \frac{\varepsilon_0 A V^2}{2(g_0-x)^2}$$

    Static pull-in: $V_{\text{PI}} = \sqrt{\frac{8kg_0^3}{27\varepsilon_0 A}}$
    at $x_{\text{PI}} = g_0/3$.

    Dynamic pull-in: $V_{\text{PI,dyn}} \approx 0.92 V_{\text{PI,static}}$.
    """

    def __init__(self, k: float = 1.0, m: float = 1e-10,
                 c: float = 1e-8, g0: float = 2e-6,
                 area: float = 1e-8,
                 eps0: float = 8.854e-12) -> None:
        self.k = k
        self.m = m
        self.c = c
        self.g0 = g0
        self.A = area
        self.eps0 = eps0

    def pull_in_voltage_static(self) -> float:
        """Static pull-in voltage."""
        return math.sqrt(8 * self.k * self.g0**3 / (27 * self.eps0 * self.A))

    def pull_in_voltage_dynamic(self) -> float:
        """Dynamic pull-in (step voltage): ≈ 0.92 V_PI_static."""
        return 0.9196 * self.pull_in_voltage_static()

    def electrostatic_force(self, x: float, V: float) -> float:
        """F_e = ε₀AV²/(2(g₀−x)²)."""
        gap = self.g0 - x
        if gap <= 0:
            return float('inf')
        return self.eps0 * self.A * V**2 / (2 * gap**2)

    def static_equilibrium(self, V: float, tol: float = 1e-10,
                             max_iter: int = 100) -> float:
        """Find static equilibrium displacement for given voltage.

        Newton-Raphson on: kx − ε₀AV²/(2(g₀−x)²) = 0.
        Returns displacement or g0 if pull-in occurs.
        """
        x = 0.0
        for _ in range(max_iter):
            gap = self.g0 - x
            if gap <= 1e-12:
                return self.g0  # pull-in
            F = self.k * x - self.eps0 * self.A * V**2 / (2 * gap**2)
            dF = self.k + self.eps0 * self.A * V**2 / (gap**3)
            dx = -F / dF
            x += dx
            if abs(dx) < tol:
                break
            if x >= self.g0:
                return self.g0
        return x

    def simulate(self, V: float, t_end: float = 1e-3,
                   dt: float = 1e-7) -> Tuple[NDArray, NDArray]:
        """Time-domain simulation.

        Returns (time, displacement).
        """
        n_steps = int(t_end / dt)
        t = np.zeros(n_steps)
        x = np.zeros(n_steps)
        v = np.zeros(n_steps)

        for i in range(n_steps - 1):
            t[i + 1] = t[i] + dt
            gap = self.g0 - x[i]
            if gap <= 1e-12:
                x[i + 1:] = self.g0
                break
            Fe = self.eps0 * self.A * V**2 / (2 * gap**2)
            a = (Fe - self.c * v[i] - self.k * x[i]) / self.m
            v[i + 1] = v[i] + a * dt
            x[i + 1] = x[i] + v[i + 1] * dt
            x[i + 1] = min(x[i + 1], self.g0)

        return t, x

    def squeeze_film_damping(self, mu: float = 1.85e-5,
                                L: float = 100e-6,
                                W: float = 100e-6) -> float:
        """Squeeze-film damping coefficient for rectangular plate.

        c ≈ μLW³/(g₀³) for narrow gap.
        """
        return mu * L * W**3 / self.g0**3


# ---------------------------------------------------------------------------
#  Electrostriction
# ---------------------------------------------------------------------------

class ElectrostrictiveMaterial:
    r"""
    Electrostriction (quadratic electromechanical coupling):

    $$\varepsilon_{ij} = M_{ijkl} E_k E_l$$

    Unlike piezoelectricity, electrostriction exists in all dielectrics
    and is proportional to E².

    Effective strain: $\varepsilon = M_{33} E^2$ (uniaxial).
    """

    def __init__(self, M33: float = 2e-18, E_max: float = 1e7,
                 C11: float = 100e9, kappa: float = 1000 * 8.854e-12) -> None:
        self.M33 = M33  # electrostriction coefficient (m²/V²)
        self.E_max = E_max
        self.C11 = C11
        self.kappa = kappa

    def strain(self, E: float) -> float:
        """Electrostrictive strain ε = M₃₃ E²."""
        return self.M33 * E**2

    def stress(self, E: float, mechanical_strain: float = 0.0) -> float:
        """σ = C(ε_mech − ε_ES) where ε_ES = M₃₃E²."""
        return self.C11 * (mechanical_strain - self.strain(E))

    def maxwell_stress(self, E: float) -> float:
        """Maxwell stress: σ_M = ½κE²."""
        return 0.5 * self.kappa * E**2

    def effective_d33(self, E_bias: float) -> float:
        """Effective piezoelectric coefficient under DC bias.

        d₃₃_eff = 2 M₃₃ E_bias (linearisation around bias field).
        """
        return 2 * self.M33 * E_bias


# ---------------------------------------------------------------------------
#  Capacitive Comb-Drive Actuator
# ---------------------------------------------------------------------------

class CombDriveActuator:
    r"""
    MEMS lateral comb-drive actuator.

    Force along finger overlap direction (x):
    $$F_x = n\frac{\varepsilon_0 h}{g}V^2$$

    where n = number of finger pairs, h = finger height, g = gap.
    Force is independent of displacement → linear force-voltage relation.
    """

    def __init__(self, n_fingers: int = 100, finger_height: float = 20e-6,
                 gap: float = 2e-6, k: float = 0.5,
                 mass: float = 1e-9, eps0: float = 8.854e-12) -> None:
        self.n = n_fingers
        self.h = finger_height
        self.g = gap
        self.k = k
        self.m = mass
        self.eps0 = eps0

    def force(self, V: float) -> float:
        """Electrostatic drive force."""
        return self.n * self.eps0 * self.h / self.g * V**2

    def static_displacement(self, V: float) -> float:
        """Static displacement: x = F/k."""
        return self.force(V) / self.k

    def resonance_frequency(self) -> float:
        """f₀ = (1/2π)√(k/m)."""
        return math.sqrt(self.k / self.m) / (2 * math.pi)

    def voltage_for_displacement(self, x_target: float) -> float:
        """Required voltage for target displacement."""
        F_needed = self.k * x_target
        return math.sqrt(F_needed * self.g / (self.n * self.eps0 * self.h))

    def capacitance(self, x_overlap: float) -> float:
        """C(x) = 2n ε₀ h (L₀ + x) / g."""
        L0 = 10e-6  # initial overlap
        return 2 * self.n * self.eps0 * self.h * (L0 + x_overlap) / self.g
