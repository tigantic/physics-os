"""
Tribology: Friction, Wear & Lubrication
=========================================

Computational tribology models for contact mechanics, dry sliding,
thin-film lubrication, and wear prediction.

Models:
    1. **Archard wear law**: :math:`V = k F L / H`
    2. **Reynolds lubrication**: 1D thin-film pressure equation
    3. **Greenwood-Williamson** (GW): statistical rough-surface contact
    4. **Rate-and-state friction**: Dieterich-Ruina formulation

References:
    [1] Archard, "Contact and Rubbing of Flat Surfaces",
        J. Appl. Phys. 24, 1953.
    [2] Greenwood & Williamson, Proc. Roy. Soc. London A 295, 1966.
    [3] Hamrock, Schmid & Jacobson, *Fundamentals of Fluid Film
        Lubrication*, 2nd ed., CRC (2004).
    [4] Dieterich, "Modeling of Rock Friction", JGR 84, 1979.

Domain III.14 — Solid Mechanics / Tribology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Archard wear model
# ---------------------------------------------------------------------------

@dataclass
class ArchardWear:
    """
    Archard wear law: :math:`V = k F L / H`.

    Attributes:
        k: Dimensionless wear coefficient (typically 1e-7 to 1e-3).
        H: Hardness of the softer material [Pa].
    """
    k: float = 1e-5
    H: float = 1e9

    def wear_volume(self, F: float, L: float) -> float:
        """Wear volume for normal force F and sliding distance L."""
        return self.k * F * L / self.H

    def wear_depth(self, F: float, L: float, A_contact: float) -> float:
        """Wear depth = V / A_contact."""
        return self.wear_volume(F, L) / (A_contact + 1e-30)

    def sliding_simulation(
        self,
        F: float,
        v: float,
        t_total: float,
        dt: float,
    ) -> Tuple[NDArray, NDArray]:
        """
        Time-resolved wear depth accumulation.

        Returns (times, cumulative_depth).
        """
        n_steps = int(t_total / dt)
        times = np.linspace(0, t_total, n_steps + 1)
        depth = np.zeros(n_steps + 1)
        for i in range(1, n_steps + 1):
            dL = v * dt
            depth[i] = depth[i - 1] + self.k * F * dL / self.H
        return times, depth


# ---------------------------------------------------------------------------
# Greenwood-Williamson rough contact
# ---------------------------------------------------------------------------

@dataclass
class GWContact:
    """
    Greenwood-Williamson statistical rough-surface contact model.

    Assumes asperities with Gaussian height distribution and
    Hertzian contact at each asperity.

    Attributes:
        E_star: Effective elastic modulus [Pa].
        R_asp: Asperity tip radius [m].
        sigma_s: RMS asperity height [m].
        eta: Asperity density [1/m²].
        A_nom: Nominal contact area [m²].
    """
    E_star: float = 1e11
    R_asp: float = 1e-5
    sigma_s: float = 1e-6
    eta: float = 1e10
    A_nom: float = 1e-4

    def contact_pressure_and_area(
        self,
        d_sep: float,
        n_points: int = 200,
    ) -> Tuple[float, float]:
        """
        Compute real contact area and mean pressure at separation d_sep.

        Integration over the asperity height PDF from d_sep to ∞.

        Returns (total_load, real_contact_area).
        """
        z_max = d_sep + 5.0 * self.sigma_s
        z_arr = np.linspace(d_sep, z_max, n_points)
        dz = z_arr[1] - z_arr[0]

        # Gaussian distribution of asperity heights
        phi = (1.0 / (self.sigma_s * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * (z_arr / self.sigma_s) ** 2
        )

        delta = z_arr - d_sep  # overlap for each asperity

        # Hertzian force per asperity: F = (4/3) E* sqrt(R) δ^(3/2)
        F_asp = (4.0 / 3.0) * self.E_star * np.sqrt(self.R_asp) * delta ** 1.5
        # Contact area per asperity: a = π R δ
        A_asp = np.pi * self.R_asp * delta

        total_load = self.eta * self.A_nom * np.trapz(F_asp * phi, z_arr)
        real_area = self.eta * self.A_nom * np.trapz(A_asp * phi, z_arr)

        return float(total_load), float(real_area)

    def load_vs_separation(
        self,
        d_range: Tuple[float, float],
        n_d: int = 100,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Load and real-area curves vs separation.

        Returns (d_array, load_array, area_array).
        """
        d_arr = np.linspace(d_range[0], d_range[1], n_d)
        load = np.zeros(n_d)
        area = np.zeros(n_d)
        for i, d in enumerate(d_arr):
            load[i], area[i] = self.contact_pressure_and_area(d)
        return d_arr, load, area


# ---------------------------------------------------------------------------
# 1D Reynolds lubrication
# ---------------------------------------------------------------------------

@dataclass
class ReynoldsLubrication1D:
    r"""
    1D Reynolds equation for thin-film hydrodynamic lubrication:

    .. math::
        \frac{\partial}{\partial x}\left(\frac{h^3}{12\mu}
        \frac{\partial p}{\partial x}\right)
        = \frac{U}{2} \frac{\partial h}{\partial x}

    Discretised with finite differences on a uniform grid.

    Attributes:
        L: Bearing length [m].
        h_func: Film thickness function h(x).
        U: Sliding velocity [m/s].
        mu: Dynamic viscosity [Pa·s].
        p_inlet: Inlet pressure [Pa].
        p_outlet: Outlet pressure [Pa].
    """
    L: float = 0.1
    U: float = 1.0
    mu: float = 0.01
    p_inlet: float = 0.0
    p_outlet: float = 0.0

    def solve(
        self,
        h_func,
        nx: int = 200,
    ) -> Tuple[NDArray, NDArray]:
        """
        Solve 1D Reynolds equation on [0, L] with given h(x).

        Returns (x, p).
        """
        dx = self.L / nx
        x = np.linspace(0.5 * dx, self.L - 0.5 * dx, nx)
        h = np.array([h_func(xi) for xi in x])

        # Build tri-diagonal system
        # A p = b
        A = np.zeros((nx, nx), dtype=np.float64)
        b = np.zeros(nx, dtype=np.float64)

        for i in range(1, nx - 1):
            h_e = 0.5 * (h[i] + h[i + 1])
            h_w = 0.5 * (h[i] + h[i - 1])
            coeff_e = h_e ** 3 / (12.0 * self.mu * dx ** 2)
            coeff_w = h_w ** 3 / (12.0 * self.mu * dx ** 2)

            A[i, i - 1] = coeff_w
            A[i, i] = -(coeff_e + coeff_w)
            A[i, i + 1] = coeff_e

            # RHS: (U/2) dh/dx
            b[i] = self.U / 2.0 * (h[i + 1] - h[i - 1]) / (2.0 * dx)

        # BCs
        A[0, 0] = 1.0
        b[0] = self.p_inlet
        A[-1, -1] = 1.0
        b[-1] = self.p_outlet

        p = np.linalg.solve(A, b)

        # Enforce Reynolds cavitation condition (p ≥ 0)
        p = np.maximum(p, 0.0)

        return x, p

    def load_capacity(self, x: NDArray, p: NDArray) -> float:
        """Integrated load capacity."""
        return float(np.trapz(p, x))

    def friction_force(
        self,
        x: NDArray,
        p: NDArray,
        h_func,
    ) -> float:
        """Viscous friction force on the sliding surface."""
        h = np.array([h_func(xi) for xi in x])
        dp_dx = np.gradient(p, x)
        # Shear stress: τ = μU/h + h/2 dp/dx
        tau = self.mu * self.U / h + h / 2.0 * dp_dx
        return float(np.trapz(tau, x))


# ---------------------------------------------------------------------------
# Rate-and-state friction (Dieterich-Ruina)
# ---------------------------------------------------------------------------

@dataclass
class RateStateFriction:
    r"""
    Dieterich-Ruina rate-and-state friction.

    .. math::
        \mu = \mu_0 + a \ln(V / V_0) + b \ln(V_0 \theta / D_c)

    State evolution (ageing law):

    .. math::
        \dot{\theta} = 1 - V \theta / D_c

    Attributes:
        mu_0: Reference friction coefficient.
        a: Direct effect parameter.
        b: Evolution effect parameter.
        D_c: Critical slip distance [m].
        V_0: Reference velocity [m/s].
    """
    mu_0: float = 0.6
    a: float = 0.01
    b: float = 0.015
    D_c: float = 1e-5
    V_0: float = 1e-6

    def friction_coefficient(self, V: float, theta: float) -> float:
        """Instantaneous friction coefficient."""
        V_safe = max(abs(V), 1e-20)
        theta_safe = max(theta, 1e-20)
        return (self.mu_0
                + self.a * np.log(V_safe / self.V_0)
                + self.b * np.log(self.V_0 * theta_safe / self.D_c))

    def state_evolution(self, V: float, theta: float) -> float:
        """Ageing law: dθ/dt."""
        return 1.0 - V * theta / self.D_c

    def simulate(
        self,
        V_func,
        t_total: float,
        dt: float = 1e-3,
        theta_0: float | None = None,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Time-integrate rate-and-state friction.

        Parameters:
            V_func: Callable returning velocity at time t.
            t_total: Total simulation time.
            dt: Time step.
            theta_0: Initial state variable (default: D_c / V_0).

        Returns:
            (times, mu_array, theta_array).
        """
        n = int(t_total / dt)
        times = np.linspace(0, t_total, n + 1)
        theta = np.zeros(n + 1)
        mu_arr = np.zeros(n + 1)

        theta[0] = theta_0 if theta_0 is not None else self.D_c / self.V_0
        V0 = V_func(0.0)
        mu_arr[0] = self.friction_coefficient(V0, theta[0])

        for i in range(n):
            V = V_func(times[i])
            # RK4 for theta
            k1 = self.state_evolution(V, theta[i])
            k2 = self.state_evolution(V, theta[i] + 0.5 * dt * k1)
            k3 = self.state_evolution(V, theta[i] + 0.5 * dt * k2)
            k4 = self.state_evolution(V, theta[i] + dt * k3)
            theta[i + 1] = theta[i] + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            theta[i + 1] = max(theta[i + 1], 1e-20)

            V_next = V_func(times[i + 1])
            mu_arr[i + 1] = self.friction_coefficient(V_next, theta[i + 1])

        return times, mu_arr, theta
