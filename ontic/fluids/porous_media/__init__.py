"""
Porous Media Flow — Darcy, Brinkman, Richards unsaturated, Buckley-Leverett.

Domain II.9 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Darcy Flow Solver (2D Steady)
# ---------------------------------------------------------------------------

class DarcySolver:
    r"""
    Darcy flow in porous media: $\mathbf{u} = -(k/\mu)\nabla p$.

    Pressure equation (incompressible):
    $$\nabla\cdot\left(\frac{k}{\mu}\nabla p\right) = q$$

    Solved on 2D Cartesian grid via iterative Laplace-type equation.
    """

    def __init__(self, nx: int, ny: int, Lx: float = 1.0,
                 Ly: float = 1.0, mu: float = 1e-3) -> None:
        self.nx = nx
        self.ny = ny
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.mu = mu

        self.perm = np.ones((nx, ny)) * 1e-12  # permeability (m²)
        self.pressure = np.zeros((nx, ny))
        self.source = np.zeros((nx, ny))

    def set_random_permeability(self, mean_log: float = -12.0,
                                  std_log: float = 1.0,
                                  seed: int = 42) -> None:
        """Log-normal random permeability field."""
        rng = np.random.default_rng(seed)
        log_k = mean_log + std_log * rng.standard_normal((self.nx, self.ny))
        self.perm = 10.0**log_k

    def solve(self, p_left: float = 1e5, p_right: float = 0.0,
              n_iter: int = 5000, tol: float = 1e-6) -> int:
        """Solve pressure equation with Dirichlet BCs on left/right.

        Returns iteration count.
        """
        dx2 = self.dx**2
        dy2 = self.dy**2

        # Harmonic mean permeability at faces
        for iteration in range(n_iter):
            p_old = self.pressure.copy()

            for i in range(1, self.nx - 1):
                for j in range(1, self.ny - 1):
                    kx_p = 2 * self.perm[i, j] * self.perm[i + 1, j] / (self.perm[i, j] + self.perm[i + 1, j] + 1e-30)
                    kx_m = 2 * self.perm[i, j] * self.perm[i - 1, j] / (self.perm[i, j] + self.perm[i - 1, j] + 1e-30)
                    ky_p = 2 * self.perm[i, j] * self.perm[i, j + 1] / (self.perm[i, j] + self.perm[i, j + 1] + 1e-30)
                    ky_m = 2 * self.perm[i, j] * self.perm[i, j - 1] / (self.perm[i, j] + self.perm[i, j - 1] + 1e-30)

                    coeff = (kx_p + kx_m) / dx2 + (ky_p + ky_m) / dy2
                    rhs = (kx_p * self.pressure[i + 1, j] + kx_m * self.pressure[i - 1, j]) / dx2 \
                        + (ky_p * self.pressure[i, j + 1] + ky_m * self.pressure[i, j - 1]) / dy2 \
                        + self.mu * self.source[i, j]

                    self.pressure[i, j] = rhs / (coeff + 1e-30)

            # BCs
            self.pressure[0, :] = p_left
            self.pressure[-1, :] = p_right
            # Neumann top/bottom
            self.pressure[:, 0] = self.pressure[:, 1]
            self.pressure[:, -1] = self.pressure[:, -2]

            residual = float(np.max(np.abs(self.pressure - p_old)))
            if residual < tol:
                return iteration + 1

        return n_iter

    def velocity_field(self) -> Tuple[NDArray, NDArray]:
        """Darcy velocity u = -(k/μ)∇p."""
        dpdx = (np.roll(self.pressure, -1, 0) - np.roll(self.pressure, 1, 0)) / (2 * self.dx)
        dpdy = (np.roll(self.pressure, -1, 1) - np.roll(self.pressure, 1, 1)) / (2 * self.dy)
        ux = -self.perm * dpdx / self.mu
        uy = -self.perm * dpdy / self.mu
        return ux, uy

    def flow_rate(self) -> float:
        """Total flow rate across right boundary."""
        ux, _ = self.velocity_field()
        return float(np.sum(ux[-2, :]) * self.dy)


# ---------------------------------------------------------------------------
#  Richards Equation (Unsaturated Flow)
# ---------------------------------------------------------------------------

class RichardsSolver:
    r"""
    Richards equation for unsaturated flow in variably-saturated porous media.

    Mixed form:
    $$\frac{\partial\theta}{\partial t} = \nabla\cdot[K(\psi)(\nabla\psi + \hat{z})]$$

    Van Genuchten soil water retention:
    $$S_e = \frac{\theta-\theta_r}{\theta_s-\theta_r}
          = [1 + |\alpha_v\psi|^n]^{-m}, \quad m = 1-1/n$$

    Hydraulic conductivity (Mualem):
    $$K(S_e) = K_s S_e^{1/2}[1-(1-S_e^{1/m})^m]^2$$
    """

    def __init__(self, nz: int = 100, Lz: float = 1.0) -> None:
        self.nz = nz
        self.dz = Lz / nz
        self.z = np.linspace(0, Lz, nz)

        # Van Genuchten parameters (loam)
        self.theta_r = 0.078
        self.theta_s = 0.43
        self.alpha_vg = 3.6    # 1/m
        self.n_vg = 1.56
        self.m_vg = 1 - 1 / self.n_vg
        self.Ks = 2.89e-6      # m/s saturated conductivity

        # State
        self.psi = np.ones(nz) * (-1.0)  # pressure head (m), negative = unsaturated

    def effective_saturation(self, psi: NDArray) -> NDArray:
        """Van Genuchten Se(ψ)."""
        Se = np.where(
            psi < 0,
            (1 + np.abs(self.alpha_vg * psi)**self.n_vg)**(- self.m_vg),
            1.0
        )
        return np.clip(Se, 0, 1)

    def water_content(self, psi: NDArray) -> NDArray:
        """θ(ψ) = θ_r + (θ_s − θ_r)Se."""
        Se = self.effective_saturation(psi)
        return self.theta_r + (self.theta_s - self.theta_r) * Se

    def conductivity(self, psi: NDArray) -> NDArray:
        """Mualem-Van Genuchten K(ψ)."""
        Se = self.effective_saturation(psi)
        return self.Ks * Se**0.5 * (1 - (1 - Se**(1 / self.m_vg))**self.m_vg)**2

    def capacity(self, psi: NDArray) -> NDArray:
        """Specific moisture capacity C(ψ) = dθ/dψ."""
        C = np.zeros_like(psi)
        mask = psi < 0
        abs_apsi = np.abs(self.alpha_vg * psi[mask])
        term = (1 + abs_apsi**self.n_vg)
        C[mask] = (self.alpha_vg * self.m_vg * self.n_vg
                   * (self.theta_s - self.theta_r)
                   * abs_apsi**(self.n_vg - 1)
                   * term**(-(self.m_vg + 1)))
        return C

    def step(self, dt: float, q_top: float = -1e-6) -> None:
        """Implicit Picard iteration step.

        q_top: infiltration flux at top (m/s), negative = inflow.
        """
        dz = self.dz
        dz2 = dz**2

        for _ in range(10):  # Picard iterations
            K = self.conductivity(self.psi)
            C = self.capacity(self.psi)
            theta = self.water_content(self.psi)

            psi_new = self.psi.copy()

            for i in range(1, self.nz - 1):
                K_half_up = 0.5 * (K[i] + K[i + 1])
                K_half_dn = 0.5 * (K[i] + K[i - 1])

                a = K_half_dn / dz2
                c = K_half_up / dz2
                b = -(a + c) - C[i] / dt
                rhs = -C[i] * self.psi[i] / dt - (K_half_up - K_half_dn) / dz

                psi_new[i] = (rhs - a * psi_new[i - 1] - c * self.psi[i + 1]) / b

            # Top BC: flux
            psi_new[-1] = psi_new[-2] + q_top * dz / K[-1]
            # Bottom BC: free drainage
            psi_new[0] = psi_new[1]

            if np.max(np.abs(psi_new - self.psi)) < 1e-6:
                self.psi = psi_new
                break
            self.psi = psi_new

    def cumulative_infiltration(self, psi_initial: NDArray) -> float:
        """Total water storage change."""
        theta_now = self.water_content(self.psi)
        theta_init = self.water_content(psi_initial)
        return float(np.sum(theta_now - theta_init) * self.dz)


# ---------------------------------------------------------------------------
#  Brinkman Equation (Intermediate Regime)
# ---------------------------------------------------------------------------

class BrinkmanSolver:
    r"""
    Brinkman equation: viscous flow in porous media where both Darcy drag
    and viscous shear matter.

    $$-\tilde{\mu}\nabla^2\mathbf{u} + \frac{\mu}{k}\mathbf{u} + \nabla p = 0$$
    $$\nabla\cdot\mathbf{u} = 0$$

    Reduces to Darcy for $\text{Da} = k/L^2 \to 0$
    and Stokes for $\text{Da} \to \infty$.

    1D channel-in-porous-medium benchmark (analytical solution available).
    """

    def __init__(self, ny: int = 100, Ly: float = 1.0,
                 mu: float = 1e-3, perm: float = 1e-10,
                 mu_eff: Optional[float] = None) -> None:
        self.ny = ny
        self.dy = Ly / ny
        self.Ly = Ly
        self.mu = mu
        self.k = perm
        self.mu_eff = mu_eff if mu_eff is not None else mu

        self.y = np.linspace(0, Ly, ny)
        self.u = np.zeros(ny)

    @property
    def darcy_number(self) -> float:
        return self.k / self.Ly**2

    def solve_1d(self, dpdx: float = -1.0) -> None:
        """Solve 1D Brinkman: −μ̃ d²u/dy² + (μ/k)u = −dp/dx.

        Dirichlet u=0 at y=0 and y=Ly.
        """
        dy = self.dy
        n = self.ny
        alpha2 = self.mu / (self.mu_eff * self.k)

        # Tridiagonal: −μ̃(u_{i-1} − 2u_i + u_{i+1})/dy² + (μ/k)u_i = −dp/dx
        a = np.zeros(n)  # lower
        b = np.zeros(n)  # diagonal
        c = np.zeros(n)  # upper
        d = np.zeros(n)  # RHS

        for i in range(1, n - 1):
            a[i] = -self.mu_eff / dy**2
            c[i] = -self.mu_eff / dy**2
            b[i] = 2 * self.mu_eff / dy**2 + self.mu / self.k
            d[i] = -dpdx

        b[0] = 1.0
        d[0] = 0.0
        b[-1] = 1.0
        d[-1] = 0.0

        # Thomas
        for i in range(1, n):
            m = a[i] / (b[i - 1] + 1e-30)
            b[i] -= m * c[i - 1]
            d[i] -= m * d[i - 1]

        self.u[-1] = d[-1] / b[-1]
        for i in range(n - 2, -1, -1):
            self.u[i] = (d[i] - c[i] * self.u[i + 1]) / (b[i] + 1e-30)

    def analytical_1d(self, dpdx: float = -1.0) -> NDArray:
        """Analytical Brinkman solution for channel between two walls.

        u(y) = (−dp/dx)(k/μ)[1 − cosh(α(y−Ly/2))/cosh(αLy/2)]
        where α² = μ/(μ̃k).
        """
        alpha = math.sqrt(self.mu / (self.mu_eff * self.k))
        u_darcy = -dpdx * self.k / self.mu
        return u_darcy * (1 - np.cosh(alpha * (self.y - self.Ly / 2)) / math.cosh(alpha * self.Ly / 2))


# ---------------------------------------------------------------------------
#  Buckley-Leverett (Two-Phase Displacement)
# ---------------------------------------------------------------------------

class BuckleyLeverett:
    r"""
    Buckley-Leverett equation for immiscible two-phase displacement in 1D.

    $$\frac{\partial S_w}{\partial t} + \frac{u_T}{\phi}\frac{\partial f_w}{\partial x} = 0$$

    Fractional flow:
    $$f_w(S_w) = \frac{k_{rw}/\mu_w}{k_{rw}/\mu_w + k_{ro}/\mu_o}$$

    Relative permeabilities (Corey):
    $$k_{rw} = k_{rw}^{\max} S_e^{n_w}, \quad
      k_{ro} = k_{ro}^{\max}(1-S_e)^{n_o}$$
    """

    def __init__(self, nx: int = 200, Lx: float = 1.0,
                 phi: float = 0.2, mu_w: float = 1e-3,
                 mu_o: float = 5e-3) -> None:
        self.nx = nx
        self.dx = Lx / nx
        self.Lx = Lx
        self.phi = phi
        self.mu_w = mu_w
        self.mu_o = mu_o

        # Corey parameters
        self.Sw_irr = 0.2
        self.So_res = 0.2
        self.nw = 2.0
        self.no = 2.0
        self.krw_max = 0.3
        self.kro_max = 1.0

        self.x = np.linspace(0, Lx, nx)
        self.Sw = np.ones(nx) * self.Sw_irr  # initial: irreducible water

    def _Se(self, Sw: NDArray) -> NDArray:
        return np.clip((Sw - self.Sw_irr) / (1 - self.Sw_irr - self.So_res), 0, 1)

    def rel_perm(self, Sw: NDArray) -> Tuple[NDArray, NDArray]:
        """Corey relative permeabilities."""
        Se = self._Se(Sw)
        krw = self.krw_max * Se**self.nw
        kro = self.kro_max * (1 - Se)**self.no
        return krw, kro

    def fractional_flow(self, Sw: NDArray) -> NDArray:
        """f_w(Sw)."""
        krw, kro = self.rel_perm(Sw)
        mob_w = krw / self.mu_w
        mob_o = kro / self.mu_o
        return mob_w / (mob_w + mob_o + 1e-30)

    def solve(self, uT: float = 1e-5, t_end: float = 1e4,
              dt: float = 10.0) -> NDArray:
        """Godunov (upwind) scheme.

        uT: total Darcy velocity (m/s).
        Returns final Sw profile.
        """
        n_steps = int(t_end / dt)
        Sw = self.Sw.copy()

        for _ in range(n_steps):
            fw = self.fractional_flow(Sw)
            # Upwind flux
            flux = np.zeros(self.nx + 1)
            for i in range(self.nx + 1):
                if i == 0:
                    flux[i] = uT * 1.0  # injection: fw = 1 (pure water)
                else:
                    flux[i] = uT * fw[min(i, self.nx - 1)]

            for i in range(self.nx):
                Sw[i] -= dt / (self.phi * self.dx) * (flux[i + 1] - flux[i])

            np.clip(Sw, self.Sw_irr, 1 - self.So_res, out=Sw)

        self.Sw = Sw
        return Sw

    def shock_velocity(self) -> float:
        """Welge tangent construction for shock front velocity.

        v_s = f_w(Sw_f) / (Sw_f - Sw_irr) where tangent from (Sw_irr, 0).
        """
        Sw_arr = np.linspace(self.Sw_irr + 0.001, 1 - self.So_res, 1000)
        fw = self.fractional_flow(Sw_arr)
        # Slope from origin (Sw_irr, 0)
        slopes = fw / (Sw_arr - self.Sw_irr + 1e-30)
        idx = np.argmax(slopes)
        return float(slopes[idx]) / self.phi
