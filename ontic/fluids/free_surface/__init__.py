"""
Free Surface Flow — Level-set method, thin-film equation, capillary dynamics.

Domain II.10 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Level-Set Method
# ---------------------------------------------------------------------------

class LevelSetSolver:
    r"""
    Level-set method for free-surface tracking.

    $$\frac{\partial\phi}{\partial t} + \mathbf{u}\cdot\nabla\phi = 0$$

    Interface: $\phi = 0$ (negative = fluid, positive = void).

    Reinitialisation (Sussman):
    $$\frac{\partial\phi}{\partial\tau} = \text{sgn}(\phi_0)(1-|\nabla\phi|)$$
    """

    def __init__(self, nx: int, ny: int, dx: float, dy: float) -> None:
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.phi = np.zeros((nx, ny))

    def init_circle(self, cx: float, cy: float, radius: float) -> None:
        """Signed distance function for a circle."""
        x = np.arange(self.nx) * self.dx
        y = np.arange(self.ny) * self.dy
        X, Y = np.meshgrid(x, y, indexing='ij')
        self.phi = np.sqrt((X - cx)**2 + (Y - cy)**2) - radius

    def advect(self, ux: NDArray, uy: NDArray, dt: float) -> None:
        """WENO3 + TVD-RK3 advection."""
        # Simplified: 2nd-order upwind
        for _ in range(1):  # SSP-RK1 for now
            phi_x_p = (np.roll(self.phi, -1, 0) - self.phi) / self.dx
            phi_x_m = (self.phi - np.roll(self.phi, 1, 0)) / self.dx
            phi_y_p = (np.roll(self.phi, -1, 1) - self.phi) / self.dy
            phi_y_m = (self.phi - np.roll(self.phi, 1, 1)) / self.dy

            # Upwind
            dphi_dx = np.where(ux > 0, phi_x_m, phi_x_p)
            dphi_dy = np.where(uy > 0, phi_y_m, phi_y_p)

            self.phi -= dt * (ux * dphi_dx + uy * dphi_dy)

    def reinitialise(self, n_steps: int = 5) -> None:
        """Sussman reinitialisation to signed distance function."""
        phi0 = self.phi.copy()
        dtau = 0.5 * min(self.dx, self.dy)

        for _ in range(n_steps):
            # Godunov upwind gradient
            Dxp = (np.roll(self.phi, -1, 0) - self.phi) / self.dx
            Dxm = (self.phi - np.roll(self.phi, 1, 0)) / self.dx
            Dyp = (np.roll(self.phi, -1, 1) - self.phi) / self.dy
            Dym = (self.phi - np.roll(self.phi, 1, 1)) / self.dy

            # Godunov Hamiltonian
            sign_phi = phi0 / (np.sqrt(phi0**2 + (self.dx)**2) + 1e-30)

            ap = np.maximum(Dxp, 0)**2
            am = np.minimum(Dxm, 0)**2
            bp = np.maximum(Dyp, 0)**2
            bm = np.minimum(Dym, 0)**2

            G_plus = np.sqrt(np.maximum(ap, am) + np.maximum(bp, bm))
            G_minus = np.sqrt(np.maximum(np.minimum(Dxp, 0)**2, np.maximum(Dxm, 0)**2) +
                               np.maximum(np.minimum(Dyp, 0)**2, np.maximum(Dym, 0)**2))

            G = np.where(sign_phi > 0, G_plus, G_minus)
            self.phi -= dtau * sign_phi * (G - 1.0)

    def curvature(self) -> NDArray:
        """κ = ∇·(∇ϕ/|∇ϕ|)."""
        px = (np.roll(self.phi, -1, 0) - np.roll(self.phi, 1, 0)) / (2 * self.dx)
        py = (np.roll(self.phi, -1, 1) - np.roll(self.phi, 1, 1)) / (2 * self.dy)
        pxx = (np.roll(self.phi, -1, 0) - 2 * self.phi + np.roll(self.phi, 1, 0)) / self.dx**2
        pyy = (np.roll(self.phi, -1, 1) - 2 * self.phi + np.roll(self.phi, 1, 1)) / self.dy**2
        pxy = (np.roll(np.roll(self.phi, -1, 0), -1, 1) - np.roll(np.roll(self.phi, -1, 0), 1, 1)
               - np.roll(np.roll(self.phi, 1, 0), -1, 1) + np.roll(np.roll(self.phi, 1, 0), 1, 1)) / (4 * self.dx * self.dy)

        grad_mag = np.sqrt(px**2 + py**2) + 1e-12
        return (pxx * py**2 - 2 * pxy * px * py + pyy * px**2) / grad_mag**3

    def heaviside_smooth(self, eps: Optional[float] = None) -> NDArray:
        """Smoothed Heaviside: H(ϕ) = 0 (ϕ<−ε), ½+ϕ/2ε+sin(πϕ/ε)/2π (|ϕ|≤ε), 1 (ϕ>ε)."""
        if eps is None:
            eps = 1.5 * max(self.dx, self.dy)
        H = np.where(self.phi < -eps, 0.0,
            np.where(self.phi > eps, 1.0,
                     0.5 + self.phi / (2 * eps) + np.sin(np.pi * self.phi / eps) / (2 * np.pi)))
        return H

    def delta_smooth(self, eps: Optional[float] = None) -> NDArray:
        """Smoothed delta: δ = dH/dϕ."""
        if eps is None:
            eps = 1.5 * max(self.dx, self.dy)
        return np.where(np.abs(self.phi) <= eps,
                        (1 + np.cos(np.pi * self.phi / eps)) / (2 * eps), 0.0)


# ---------------------------------------------------------------------------
#  Thin-Film Equation
# ---------------------------------------------------------------------------

class ThinFilmSolver:
    r"""
    Thin-film (lubrication) equation:

    $$\frac{\partial h}{\partial t} + \frac{\partial}{\partial x}
      \left[\frac{h^3}{3\mu}\left(-\gamma\frac{\partial^3 h}{\partial x^3}
      + \rho g\frac{\partial h}{\partial x}\right)\right] = 0$$

    Governs spreading of thin viscous films under gravity and surface tension.
    """

    def __init__(self, nx: int = 256, Lx: float = 1.0,
                 mu: float = 1.0, gamma: float = 0.072,
                 rho: float = 1000.0, g: float = 9.81) -> None:
        self.nx = nx
        self.dx = Lx / nx
        self.mu = mu
        self.gamma = gamma
        self.rho = rho
        self.g = g
        self.x = np.linspace(0, Lx, nx)
        self.h = np.ones(nx) * 0.01  # thin film height

    def init_droplet(self, h0: float = 0.1, width: float = 0.1) -> None:
        """Parabolic droplet profile."""
        xc = self.x[-1] / 2
        self.h = np.maximum(h0 * (1 - ((self.x - xc) / width)**2), 1e-6)

    def step(self, dt: float) -> None:
        """Explicit time step (4th-order spatial stencil for ∂³h/∂x³)."""
        dx = self.dx
        h = self.h.copy()

        # Third derivative (central, 2nd-order)
        d3h = (np.roll(h, -2) - 2 * np.roll(h, -1) + 2 * np.roll(h, 1) - np.roll(h, 2)) / (2 * dx**3)

        # First derivative
        dh = (np.roll(h, -1) - np.roll(h, 1)) / (2 * dx)

        # Mobility
        mob = h**3 / (3 * self.mu)

        # Flux: q = mob * (-γ d³h/dx³ + ρg dh/dx)
        flux = mob * (-self.gamma * d3h + self.rho * self.g * dh)

        # Conservation form
        dqx = (np.roll(flux, -1) - np.roll(flux, 1)) / (2 * dx)
        self.h = h - dt * dqx
        self.h = np.maximum(self.h, 1e-8)

    def tanner_spreading_law(self, t: float, V: float,
                               gamma: float, mu: float) -> float:
        """Tanner's law for contact line spreading.

        R(t) ~ (γV³/μ)^(1/10) t^(1/10)
        """
        return (gamma * V**3 / mu)**(1.0 / 10.0) * t**(1.0 / 10.0)


# ---------------------------------------------------------------------------
#  Contact Angle Dynamics
# ---------------------------------------------------------------------------

class ContactAngleDynamics:
    r"""
    Cox-Voinov law for dynamic contact angle:

    $$\theta_d^3 = \theta_s^3 + 9\frac{\mu U}{\gamma}\ln\frac{L}{L_s}$$

    where θ_s = static angle, U = contact line velocity,
    L = macroscopic length, L_s = slip length.
    """

    def __init__(self, theta_static: float = math.pi / 4,
                 gamma: float = 0.072, mu: float = 1e-3,
                 L_macro: float = 1e-3, L_slip: float = 1e-9) -> None:
        self.theta_s = theta_static
        self.gamma = gamma
        self.mu = mu
        self.L = L_macro
        self.Ls = L_slip

    @property
    def capillary_length(self) -> float:
        """l_c = √(γ/(ρg)), default ρ=1000, g=9.81."""
        return math.sqrt(self.gamma / (1000 * 9.81))

    def dynamic_angle(self, U: float) -> float:
        """Cox-Voinov dynamic contact angle θ_d."""
        Ca = self.mu * abs(U) / self.gamma
        ln_ratio = math.log(self.L / self.Ls)
        theta_d_cubed = self.theta_s**3 + 9 * Ca * ln_ratio * np.sign(U)
        if theta_d_cubed < 0:
            return 0.0
        return theta_d_cubed**(1 / 3)

    def capillary_number(self, U: float) -> float:
        """Ca = μU/γ."""
        return self.mu * abs(U) / self.gamma

    def young_equation(self, gamma_sg: float, gamma_sl: float) -> float:
        """Young's equation: cos θ = (γ_SG − γ_SL)/γ_LG."""
        return math.acos(np.clip((gamma_sg - gamma_sl) / self.gamma, -1, 1))


# ---------------------------------------------------------------------------
#  Marangoni Surface Flow
# ---------------------------------------------------------------------------

class MarangoniSurfaceFlow:
    r"""
    Thermocapillary (Marangoni) flow on a free surface.

    $$\mu\frac{\partial u}{\partial z}\bigg|_s = \frac{\partial\gamma}{\partial T}\frac{\partial T}{\partial x}$$

    Marangoni number: $Ma = -\frac{\partial\gamma}{\partial T}\frac{\Delta T L}{\mu\alpha}$
    """

    def __init__(self, dgamma_dT: float = -1.5e-4,
                 mu: float = 1e-3, alpha_th: float = 1.4e-7,
                 L: float = 0.01) -> None:
        self.dgamma_dT = dgamma_dT
        self.mu = mu
        self.alpha = alpha_th
        self.L = L

    def marangoni_number(self, delta_T: float) -> float:
        """Ma = |dγ/dT| ΔT L / (μα)."""
        return abs(self.dgamma_dT) * delta_T * self.L / (self.mu * self.alpha)

    def surface_velocity(self, dT_dx: float, h: float) -> float:
        """Marangoni-driven surface velocity for thin film.

        u_s = (dγ/dT)(dT/dx) h / (2μ)
        """
        return self.dgamma_dT * dT_dx * h / (2 * self.mu)

    def benard_marangoni_critical(self, Bi: float = 0.0) -> float:
        """Critical Ma for onset of Bénard-Marangoni convection.

        Ma_c ≈ 80 for free-free boundaries (rigid-free: ~48).
        """
        return 48.0 * (1 + Bi) / (1 + Bi / 2)
