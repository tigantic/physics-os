"""
Thermo-Mechanical Coupling — Thermal stress, buckling, casting, welding residual stress.

Domain XVIII.2 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Thermoelastic Stress Solver (2D Plane Stress)
# ---------------------------------------------------------------------------

class ThermoelasticSolver:
    r"""
    2D thermoelastic plane-stress:

    $$\sigma_{ij} = C_{ijkl}(\varepsilon_{kl} - \alpha\Delta T\delta_{kl})$$

    Displacement formulation (Navier-Cauchy with thermal body force):
    $$(\lambda+\mu)\nabla(\nabla\cdot\mathbf{u}) + \mu\nabla^2\mathbf{u}
      = (3\lambda+2\mu)\alpha\nabla T$$
    """

    def __init__(self, nx: int = 50, ny: int = 50,
                 Lx: float = 1.0, Ly: float = 1.0,
                 E: float = 200e9, nu: float = 0.3,
                 alpha_th: float = 12e-6) -> None:
        self.nx = nx
        self.ny = ny
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.E = E
        self.nu = nu
        self.alpha_th = alpha_th

        # Lamé parameters
        self.lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.mu_lame = E / (2 * (1 + nu))

        self.ux = np.zeros((nx, ny))
        self.uy = np.zeros((nx, ny))
        self.T = np.zeros((nx, ny))

    def set_temperature(self, T_field: NDArray) -> None:
        """Set temperature distribution."""
        self.T = T_field.copy()

    def solve(self, n_iter: int = 3000, tol: float = 1e-6) -> int:
        """Iterative displacement solver with thermal loading.

        Fixed bottom (u=0), free top/sides.
        Returns iteration count.
        """
        lam = self.lam
        mu = self.mu_lame
        alpha = self.alpha_th
        dx = self.dx
        dy = self.dy
        beta_th = (3 * lam + 2 * mu) * alpha

        dTdx = (np.roll(self.T, -1, 0) - np.roll(self.T, 1, 0)) / (2 * dx)
        dTdy = (np.roll(self.T, -1, 1) - np.roll(self.T, 1, 1)) / (2 * dy)

        for iteration in range(n_iter):
            ux_old = self.ux.copy()
            uy_old = self.uy.copy()

            for i in range(1, self.nx - 1):
                for j in range(1, self.ny - 1):
                    # ∇²ux
                    lap_ux = ((self.ux[i + 1, j] - 2 * self.ux[i, j] + self.ux[i - 1, j]) / dx**2
                              + (self.ux[i, j + 1] - 2 * self.ux[i, j] + self.ux[i, j - 1]) / dy**2)

                    # ∂(∇·u)/∂x
                    div_x = ((self.ux[i + 1, j] - self.ux[i - 1, j]) / (2 * dx) +
                              (self.uy[i, j + 1] - self.uy[i, j - 1]) / (2 * dy))
                    d_div_dx = ((self.ux[i + 1, j] - 2 * self.ux[i, j] + self.ux[i - 1, j]) / dx**2
                                + (self.uy[i + 1, j + 1] - self.uy[i + 1, j - 1]
                                   - self.uy[i - 1, j + 1] + self.uy[i - 1, j - 1]) / (4 * dx * dy))

                    rhs_x = beta_th * dTdx[i, j] - (lam + mu) * d_div_dx
                    coeff = 2 * mu / dx**2 + 2 * mu / dy**2
                    neighbor = mu * ((self.ux[i + 1, j] + self.ux[i - 1, j]) / dx**2
                                     + (self.ux[i, j + 1] + self.ux[i, j - 1]) / dy**2)
                    self.ux[i, j] = (neighbor - rhs_x) / coeff if coeff > 0 else 0

                    # Similarly for uy
                    lap_uy = ((self.uy[i + 1, j] - 2 * self.uy[i, j] + self.uy[i - 1, j]) / dx**2
                              + (self.uy[i, j + 1] - 2 * self.uy[i, j] + self.uy[i, j - 1]) / dy**2)

                    d_div_dy = ((self.ux[i + 1, j + 1] - self.ux[i - 1, j + 1]
                                 - self.ux[i + 1, j - 1] + self.ux[i - 1, j - 1]) / (4 * dx * dy)
                                + (self.uy[i, j + 1] - 2 * self.uy[i, j] + self.uy[i, j - 1]) / dy**2)

                    rhs_y = beta_th * dTdy[i, j] - (lam + mu) * d_div_dy
                    neighbor_y = mu * ((self.uy[i + 1, j] + self.uy[i - 1, j]) / dx**2
                                       + (self.uy[i, j + 1] + self.uy[i, j - 1]) / dy**2)
                    self.uy[i, j] = (neighbor_y - rhs_y) / coeff if coeff > 0 else 0

            # BCs: fixed bottom
            self.ux[:, 0] = 0
            self.uy[:, 0] = 0
            # Free top: zero traction (Neumann)
            self.ux[:, -1] = self.ux[:, -2]
            self.uy[:, -1] = self.uy[:, -2]
            # Free sides
            self.ux[0, :] = self.ux[1, :]
            self.ux[-1, :] = self.ux[-2, :]
            self.uy[0, :] = self.uy[1, :]
            self.uy[-1, :] = self.uy[-2, :]

            res = max(float(np.max(np.abs(self.ux - ux_old))),
                      float(np.max(np.abs(self.uy - uy_old))))
            if res < tol:
                return iteration + 1

        return n_iter

    def stress_field(self) -> Tuple[NDArray, NDArray, NDArray]:
        """Compute σ_xx, σ_yy, σ_xy including thermal."""
        dux_dx = (np.roll(self.ux, -1, 0) - np.roll(self.ux, 1, 0)) / (2 * self.dx)
        duy_dy = (np.roll(self.uy, -1, 1) - np.roll(self.uy, 1, 1)) / (2 * self.dy)
        dux_dy = (np.roll(self.ux, -1, 1) - np.roll(self.ux, 1, 1)) / (2 * self.dy)
        duy_dx = (np.roll(self.uy, -1, 0) - np.roll(self.uy, 1, 0)) / (2 * self.dx)

        eps_th = self.alpha_th * self.T
        sigma_xx = (self.lam + 2 * self.mu_lame) * dux_dx + self.lam * duy_dy - (3 * self.lam + 2 * self.mu_lame) * eps_th
        sigma_yy = self.lam * dux_dx + (self.lam + 2 * self.mu_lame) * duy_dy - (3 * self.lam + 2 * self.mu_lame) * eps_th
        sigma_xy = self.mu_lame * (dux_dy + duy_dx)
        return sigma_xx, sigma_yy, sigma_xy

    def von_mises(self) -> NDArray:
        """Von Mises equivalent stress."""
        sxx, syy, sxy = self.stress_field()
        return np.sqrt(sxx**2 + syy**2 - sxx * syy + 3 * sxy**2)


# ---------------------------------------------------------------------------
#  Thermal Buckling
# ---------------------------------------------------------------------------

class ThermalBuckling:
    r"""
    Critical temperature for thermal buckling of a plate.

    Simply-supported rectangular plate, uniform ΔT:
    $$\Delta T_{cr} = \frac{\pi^2}{12\alpha(1+\nu)}
      \left(\frac{h}{b}\right)^2\left(m^2\omega^2 + n^2\right)^{-1}
      \left[\frac{(m^2\omega^2+n^2)^2}{m^2\omega^2+n^2}\right]$$

    where ω = b/a (aspect ratio), m, n = half-wave numbers.

    Simplified uniaxial:
    $$\Delta T_{cr} = \frac{\pi^2}{\alpha(1-\nu^2)}\frac{D}{N_T a^2}$$
    $$N_T = \frac{E\alpha\Delta T h}{1-\nu}$$
    """

    def __init__(self, a: float = 1.0, b: float = 1.0,
                 h: float = 0.01, E: float = 200e9,
                 nu: float = 0.3, alpha: float = 12e-6) -> None:
        self.a = a
        self.b = b
        self.h = h
        self.E = E
        self.nu = nu
        self.alpha = alpha

        self.D = E * h**3 / (12 * (1 - nu**2))

    def critical_temperature(self, m: int = 1, n: int = 1) -> float:
        """Critical buckling ΔT for mode (m,n)."""
        omega = self.b / self.a
        k = (m**2 / self.a**2 + n**2 / self.b**2)**2
        N_cr = self.D * math.pi**2 * k / (m**2 / self.a**2 + n**2 / self.b**2)
        return N_cr * (1 - self.nu) / (self.E * self.alpha * self.h)

    def minimum_critical_temperature(self, m_max: int = 5,
                                        n_max: int = 5) -> Tuple[float, int, int]:
        """Find minimum ΔT_cr over all modes."""
        best_T = float('inf')
        best_m, best_n = 1, 1
        for m in range(1, m_max + 1):
            for n in range(1, n_max + 1):
                dT = self.critical_temperature(m, n)
                if dT < best_T:
                    best_T = dT
                    best_m, best_n = m, n
        return best_T, best_m, best_n


# ---------------------------------------------------------------------------
#  Welding Residual Stress (Goldak Double-Ellipsoid)
# ---------------------------------------------------------------------------

class WeldingResidualStress:
    r"""
    Goldak double-ellipsoid heat source for welding simulation.

    $$q_f(x,y,z) = \frac{6\sqrt{3}f_f Q}{\pi\sqrt{\pi}a_f b c}
      \exp\!\left(-\frac{3x^2}{a_f^2}-\frac{3y^2}{b^2}-\frac{3z^2}{c^2}\right)$$

    Q = ηVI (total heat input), a_f/a_r = front/rear semi-axes.
    """

    def __init__(self, nx: int = 100, ny: int = 50,
                 Lx: float = 0.3, Ly: float = 0.1,
                 rho: float = 7800.0, cp: float = 500.0,
                 k_th: float = 40.0) -> None:
        self.nx = nx
        self.ny = ny
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.rho = rho
        self.cp = cp
        self.k_th = k_th
        self.alpha_th = k_th / (rho * cp)

        self.T = np.ones((nx, ny)) * 300.0  # K, initial ambient
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)

    def goldak_source(self, x0: float, y0: float,
                        Q: float = 1000.0,
                        af: float = 0.005, ar: float = 0.01,
                        b: float = 0.003, c: float = 0.005,
                        ff: float = 0.6) -> NDArray:
        """Goldak double-ellipsoid heat source at (x0, y0).

        Returns volumetric source q (W/m³).
        """
        fr = 1.0 - ff
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        xi = X - x0
        eta = Y - y0

        q_front = (6 * math.sqrt(3) * ff * Q
                    / (af * b * c * math.pi * math.sqrt(math.pi))
                    * np.exp(-3 * xi**2 / af**2 - 3 * eta**2 / b**2))
        q_rear = (6 * math.sqrt(3) * fr * Q
                   / (ar * b * c * math.pi * math.sqrt(math.pi))
                   * np.exp(-3 * xi**2 / ar**2 - 3 * eta**2 / b**2))

        return np.where(xi >= 0, q_front, q_rear)

    def step(self, dt: float, source: NDArray) -> None:
        """Explicit heat conduction with volumetric source."""
        T = self.T
        lap_T = ((np.roll(T, -1, 0) - 2 * T + np.roll(T, 1, 0)) / self.dx**2
                  + (np.roll(T, -1, 1) - 2 * T + np.roll(T, 1, 1)) / self.dy**2)
        self.T += dt * (self.alpha_th * lap_T + source / (self.rho * self.cp))

    def simulate_pass(self, speed: float = 0.005, Q: float = 1000.0,
                        dt: float = 0.01) -> NDArray:
        """Simulate a single welding pass from left to right.

        Returns peak temperature field.
        """
        peak_T = self.T.copy()
        x_torch = 0.0
        while x_torch < self.x[-1]:
            src = self.goldak_source(x_torch, self.y[self.ny // 2], Q=Q)
            self.step(dt, src)
            peak_T = np.maximum(peak_T, self.T)
            x_torch += speed * dt
        return peak_T


# ---------------------------------------------------------------------------
#  Casting Solidification Stress
# ---------------------------------------------------------------------------

class CastingSolidificationStress:
    r"""
    Simplified casting solidification stress model.

    Temperature: Stefan problem with phase change.
    Stress: thermoelastic with temperature-dependent yield.

    $$\rho c_{\text{eff}}\frac{\partial T}{\partial t} = k\nabla^2 T$$
    $$c_{\text{eff}} = c_p + L\frac{\partial f_s}{\partial T}$$

    $f_s$ = solid fraction (linear between $T_L$ and $T_S$).
    """

    def __init__(self, nx: int = 100, Lx: float = 0.1,
                 T_pour: float = 1600.0, T_mold: float = 300.0,
                 T_liquidus: float = 1540.0, T_solidus: float = 1480.0,
                 k_th: float = 30.0, rho: float = 7200.0,
                 cp: float = 700.0, L_latent: float = 270e3) -> None:
        self.nx = nx
        self.dx = Lx / nx
        self.T_L = T_liquidus
        self.T_S = T_solidus
        self.k_th = k_th
        self.rho = rho
        self.cp = cp
        self.L = L_latent

        self.T = np.ones(nx) * T_pour
        self.T[0] = T_mold
        self.T[-1] = T_mold

    def solid_fraction(self, T: NDArray) -> NDArray:
        """Linear solidification: f_s = (T_L − T)/(T_L − T_S), clamped [0,1]."""
        return np.clip((self.T_L - T) / (self.T_L - self.T_S), 0, 1)

    def effective_cp(self, T: NDArray) -> NDArray:
        """Effective heat capacity including latent heat."""
        dfs_dT = np.zeros_like(T)
        mask = (T >= self.T_S) & (T <= self.T_L)
        dfs_dT[mask] = -1.0 / (self.T_L - self.T_S)
        return self.cp - self.L * dfs_dT

    def step(self, dt: float) -> None:
        """Heat conduction with effective cp."""
        T = self.T
        cp_eff = self.effective_cp(T)
        alpha_eff = self.k_th / (self.rho * cp_eff + 1e-10)

        lap_T = (np.roll(T, -1) - 2 * T + np.roll(T, 1)) / self.dx**2
        self.T += dt * alpha_eff * lap_T

        # Mold BCs
        self.T[0] = 300.0
        self.T[-1] = 300.0

    def solidification_time(self, dt: float = 0.01) -> float:
        """Time until entire domain is solid."""
        t = 0.0
        while np.any(self.T > self.T_S):
            self.step(dt)
            t += dt
            if t > 1e6:
                break
        return t
