"""
Multiphase Flow — Cahn-Hilliard phase-field coupled to Navier-Stokes, VOF
advection, surface tension (CSF), Rayleigh-Taylor instability benchmark.

Domain II.4 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Cahn-Hilliard Phase-Field Solver (2D)
# ---------------------------------------------------------------------------

class CahnHilliardSolver:
    r"""
    Cahn-Hilliard equation for diffuse-interface multiphase modelling.

    $$\frac{\partial\phi}{\partial t} + \mathbf{u}\cdot\nabla\phi
      = M\nabla^2\mu$$

    Chemical potential:
    $$\mu = \phi^3 - \phi - \varepsilon^2\nabla^2\phi$$

    Free energy:
    $$F[\phi] = \int\left[\frac{1}{4}(\phi^2-1)^2
                + \frac{\varepsilon^2}{2}|\nabla\phi|^2\right]dx$$

    Semi-implicit time integration: implicit for the linear diffusion/
    biharmonic term, explicit for the nonlinear cubic.
    """

    def __init__(self, nx: int, ny: int, Lx: float = 1.0, Ly: float = 1.0,
                 epsilon: float = 0.01, mobility: float = 1.0) -> None:
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.epsilon = epsilon
        self.M = mobility

        # Wavenumbers for spectral solve (periodic BC)
        kx = np.fft.fftfreq(nx, d=self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(ny, d=self.dy) * 2 * np.pi
        self.KX, self.KY = np.meshgrid(kx, ky, indexing='ij')
        self.K2 = self.KX**2 + self.KY**2

        # Phase field
        self.phi = np.zeros((nx, ny))

    def init_random(self, mean: float = 0.0, amplitude: float = 0.05,
                      seed: int = 42) -> None:
        """Random initial perturbation around mean composition."""
        rng = np.random.default_rng(seed)
        self.phi = mean + amplitude * (2 * rng.random((self.nx, self.ny)) - 1)

    def init_circle(self, cx: float = 0.5, cy: float = 0.5,
                      radius: float = 0.25) -> None:
        """Circular droplet: φ = +1 inside, −1 outside (tanh profile)."""
        x = np.linspace(0, self.Lx, self.nx, endpoint=False)
        y = np.linspace(0, self.Ly, self.ny, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing='ij')
        r = np.sqrt((X - cx)**2 + (Y - cy)**2)
        self.phi = np.tanh((radius - r) / (np.sqrt(2) * self.epsilon))

    def step(self, dt: float, velocity: Optional[Tuple[NDArray, NDArray]] = None) -> None:
        """Semi-implicit spectral time step.

        Linear terms treated implicitly, nonlinear (ϕ³) explicitly.
        """
        phi_hat = np.fft.fft2(self.phi)

        # Advection (explicit upwind in Fourier)
        if velocity is not None:
            ux, uy = velocity
            adv = ux * np.real(np.fft.ifft2(1j * self.KX * phi_hat)) + \
                  uy * np.real(np.fft.ifft2(1j * self.KY * phi_hat))
            adv_hat = np.fft.fft2(adv)
        else:
            adv_hat = np.zeros_like(phi_hat)

        # Nonlinear term: ϕ³
        nl_hat = np.fft.fft2(self.phi**3)

        # Semi-implicit: (1 + dt·M·K²(1 + ε²K²)) φ̂^{n+1}
        #   = φ̂^n + dt·M·K²·(nl̂ - φ̂^n) - dt·adv_hat
        # Rearranged for stability
        denom = 1.0 + dt * self.M * self.K2 * (1.0 + self.epsilon**2 * self.K2)
        numer = phi_hat - dt * adv_hat - dt * self.M * self.K2 * (nl_hat)
        # Correct form: chemical potential μ = ϕ³ - ϕ - ε²∇²ϕ
        # → ∇²μ = ∇²(ϕ³) - ∇²ϕ - ε²∇⁴ϕ
        # Semi-implicit: treat -∇²ϕ - ε²∇⁴ϕ implicitly, ∇²(ϕ³) explicitly
        numer = phi_hat - dt * adv_hat + dt * self.M * self.K2 * nl_hat

        phi_hat_new = numer / denom
        self.phi = np.real(np.fft.ifft2(phi_hat_new))

    def free_energy(self) -> float:
        """Total free energy F[ϕ]."""
        grad_x = np.roll(self.phi, -1, axis=0) - self.phi
        grad_y = np.roll(self.phi, -1, axis=1) - self.phi
        grad_sq = (grad_x / self.dx)**2 + (grad_y / self.dy)**2

        bulk = 0.25 * (self.phi**2 - 1)**2
        interface = 0.5 * self.epsilon**2 * grad_sq

        return float(np.sum(bulk + interface) * self.dx * self.dy)

    def interface_width(self) -> float:
        """Effective interface width from gradient magnitude."""
        grad_x = (np.roll(self.phi, -1, axis=0) - np.roll(self.phi, 1, axis=0)) / (2 * self.dx)
        grad_y = (np.roll(self.phi, -1, axis=1) - np.roll(self.phi, 1, axis=1)) / (2 * self.dy)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        max_grad = np.max(grad_mag)
        if max_grad < 1e-15:
            return float('inf')
        return 2.0 / max_grad  # width ~ 2/max|∇ϕ|


# ---------------------------------------------------------------------------
#  VOF (Volume of Fluid) Advection
# ---------------------------------------------------------------------------

class VOFAdvection:
    r"""
    Volume of Fluid advection for sharp interface tracking.

    $$\frac{\partial\alpha}{\partial t} + \nabla\cdot(\alpha\mathbf{u}) = 0$$

    where α ∈ [0,1] is the volume fraction.

    PLIC (Piecewise Linear Interface Calculation) reconstruction
    with operator-split directional advection.
    """

    def __init__(self, nx: int, ny: int, dx: float, dy: float) -> None:
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.alpha = np.zeros((nx, ny))

    def init_circle(self, cx: float, cy: float, radius: float) -> None:
        """Analytical VOF initialisation for a circle."""
        for i in range(self.nx):
            for j in range(self.ny):
                x = (i + 0.5) * self.dx
                y = (j + 0.5) * self.dy
                dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                if dist < radius - 0.5 * self.dx:
                    self.alpha[i, j] = 1.0
                elif dist < radius + 0.5 * self.dx:
                    # Linear approximation in interface cells
                    self.alpha[i, j] = max(0, min(1, (radius - dist + 0.5 * self.dx) / self.dx))
                else:
                    self.alpha[i, j] = 0.0

    def interface_normal(self) -> Tuple[NDArray, NDArray]:
        """Youngs' method: interface normal from ∇α."""
        nx = (np.roll(self.alpha, -1, axis=0) - np.roll(self.alpha, 1, axis=0)) / (2 * self.dx)
        ny = (np.roll(self.alpha, -1, axis=1) - np.roll(self.alpha, 1, axis=1)) / (2 * self.dy)
        mag = np.sqrt(nx**2 + ny**2) + 1e-30
        return nx / mag, ny / mag

    def step_x(self, ux: NDArray, dt: float) -> None:
        """Operator-split advection in x-direction (donor-acceptor)."""
        flux = np.zeros_like(self.alpha)
        for i in range(self.nx):
            ip = (i + 1) % self.nx
            im = (i - 1) % self.nx
            for j in range(self.ny):
                if ux[i, j] >= 0:
                    flux[i, j] = ux[i, j] * self.alpha[i, j] * dt / self.dx
                else:
                    flux[i, j] = ux[i, j] * self.alpha[ip, j] * dt / self.dx

        for i in range(self.nx):
            ip = (i + 1) % self.nx
            im = (i - 1) % self.nx
            self.alpha[i, :] -= (flux[i, :] - flux[im, :])

        np.clip(self.alpha, 0, 1, out=self.alpha)

    def step_y(self, uy: NDArray, dt: float) -> None:
        """Operator-split advection in y-direction."""
        flux = np.zeros_like(self.alpha)
        for j in range(self.ny):
            jp = (j + 1) % self.ny
            jm = (j - 1) % self.ny
            for i in range(self.nx):
                if uy[i, j] >= 0:
                    flux[i, j] = uy[i, j] * self.alpha[i, j] * dt / self.dy
                else:
                    flux[i, j] = uy[i, j] * self.alpha[i, jp] * dt / self.dy

        for j in range(self.ny):
            jp = (j + 1) % self.ny
            jm = (j - 1) % self.ny
            self.alpha[:, j] -= (flux[:, j] - flux[:, jm])

        np.clip(self.alpha, 0, 1, out=self.alpha)

    def step(self, ux: NDArray, uy: NDArray, dt: float) -> None:
        """Strang-split advection step."""
        self.step_x(ux, dt / 2)
        self.step_y(uy, dt)
        self.step_x(ux, dt / 2)

    def total_volume(self) -> float:
        """Total fluid volume (should be conserved)."""
        return float(np.sum(self.alpha) * self.dx * self.dy)


# ---------------------------------------------------------------------------
#  Surface Tension — Continuum Surface Force (CSF)
# ---------------------------------------------------------------------------

class SurfaceTensionCSF:
    r"""
    Brackbill's Continuum Surface Force model for surface tension.

    $$\mathbf{f}_s = \gamma\kappa\nabla\alpha\,\delta_s$$

    Curvature from volume fraction:
    $$\kappa = -\nabla\cdot\hat{\mathbf{n}}, \quad \hat{\mathbf{n}} = \frac{\nabla\alpha}{|\nabla\alpha|}$$

    Spurious current suppression via Laplace pressure benchmark.
    """

    def __init__(self, gamma: float = 0.072, dx: float = 1e-3,
                 dy: float = 1e-3) -> None:
        self.gamma = gamma
        self.dx = dx
        self.dy = dy

    def curvature(self, alpha: NDArray) -> NDArray:
        """Compute curvature κ = −∇·n̂."""
        # Smoothed gradient
        gx = (np.roll(alpha, -1, axis=0) - np.roll(alpha, 1, axis=0)) / (2 * self.dx)
        gy = (np.roll(alpha, -1, axis=1) - np.roll(alpha, 1, axis=1)) / (2 * self.dy)
        mag = np.sqrt(gx**2 + gy**2) + 1e-12

        nx = gx / mag
        ny = gy / mag

        # Divergence of normal
        dnx = (np.roll(nx, -1, axis=0) - np.roll(nx, 1, axis=0)) / (2 * self.dx)
        dny = (np.roll(ny, -1, axis=1) - np.roll(ny, 1, axis=1)) / (2 * self.dy)

        return -(dnx + dny)

    def force(self, alpha: NDArray) -> Tuple[NDArray, NDArray]:
        """CSF body force: fₓ, f_y = γκ ∂α/∂x, γκ ∂α/∂y."""
        kappa = self.curvature(alpha)
        gx = (np.roll(alpha, -1, axis=0) - np.roll(alpha, 1, axis=0)) / (2 * self.dx)
        gy = (np.roll(alpha, -1, axis=1) - np.roll(alpha, 1, axis=1)) / (2 * self.dy)
        return self.gamma * kappa * gx, self.gamma * kappa * gy

    def laplace_pressure_error(self, alpha: NDArray, radius: float) -> float:
        """Spurious current benchmark: Δp = γ/R (2D circle)."""
        kappa = self.curvature(alpha)
        # Mean curvature in interface band (0.05 < α < 0.95)
        mask = (alpha > 0.05) & (alpha < 0.95)
        if np.sum(mask) == 0:
            return float('inf')
        kappa_mean = float(np.mean(np.abs(kappa[mask])))
        kappa_exact = 1.0 / radius
        return abs(kappa_mean - kappa_exact) / kappa_exact


# ---------------------------------------------------------------------------
#  Rayleigh-Taylor Instability Setup
# ---------------------------------------------------------------------------

@dataclass
class RayleighTaylorSetup:
    r"""
    Rayleigh-Taylor instability benchmark.

    Heavy fluid (ρ_h) atop light fluid (ρ_l) with gravity g.

    Linear growth rate:
    $$\gamma = \sqrt{A_t g k}$$

    where Atwood number $A_t = (\rho_h - \rho_l)/(\rho_h + \rho_l)$.

    Nonlinear bubble/spike velocity (Goncharov):
    $$v_b \approx \sqrt{\frac{2 A_t g}{(1+A_t)k C_d}}$$
    """

    rho_heavy: float = 3.0
    rho_light: float = 1.0
    g: float = 9.81
    Lx: float = 1.0
    Ly: float = 4.0
    perturbation_amp: float = 0.05

    @property
    def atwood(self) -> float:
        return (self.rho_heavy - self.rho_light) / (self.rho_heavy + self.rho_light)

    def linear_growth_rate(self, k: float) -> float:
        """σ = √(At g k) for inviscid, infinite-depth."""
        return math.sqrt(self.atwood * self.g * k)

    def init_vof(self, nx: int, ny: int) -> NDArray:
        """Initial volume fraction with sinusoidal perturbation at interface."""
        alpha = np.zeros((nx, ny))
        dx = self.Lx / nx
        dy = self.Ly / ny
        k = 2 * math.pi / self.Lx  # single mode

        for i in range(nx):
            x = (i + 0.5) * dx
            y_interface = self.Ly / 2.0 + self.perturbation_amp * math.cos(k * x)
            for j in range(ny):
                y = (j + 0.5) * dy
                if y < y_interface - 0.5 * dy:
                    alpha[i, j] = 1.0  # heavy
                elif y < y_interface + 0.5 * dy:
                    alpha[i, j] = (y_interface + 0.5 * dy - y) / dy
                else:
                    alpha[i, j] = 0.0

        return alpha

    def init_density(self, alpha: NDArray) -> NDArray:
        """ρ = α ρ_h + (1 − α) ρ_l."""
        return alpha * self.rho_heavy + (1.0 - alpha) * self.rho_light

    def mixing_width(self, alpha: NDArray, ny: int,
                       Ly: float) -> float:
        """Integral mixing width: W = ∫<α>(1 − <α>) dy."""
        # Column-average alpha
        alpha_avg = np.mean(alpha, axis=0)
        dy = Ly / ny
        return float(np.sum(alpha_avg * (1.0 - alpha_avg)) * dy)

    def bubble_velocity_goncharov(self, k: float, Cd: float = 6.0) -> float:
        """Nonlinear bubble velocity (Goncharov model)."""
        At = self.atwood
        return math.sqrt(2 * At * self.g / ((1 + At) * k * Cd))


# ---------------------------------------------------------------------------
#  Two-Phase Navier-Stokes (Projection Method + VOF)
# ---------------------------------------------------------------------------

class TwoPhaseNavierStokes:
    r"""
    Two-phase incompressible Navier-Stokes with VOF interface tracking.

    $$\rho(\partial_t\mathbf{u} + \mathbf{u}\cdot\nabla\mathbf{u})
      = -\nabla p + \nabla\cdot(\mu\nabla\mathbf{u})
      + \rho\mathbf{g} + \mathbf{f}_s$$

    Chorin projection method:
    1. Predictor: $\mathbf{u}^* = \mathbf{u}^n + \Delta t(\text{RHS without ∇p})$
    2. Pressure Poisson: $\nabla^2 p^{n+1} = \rho\nabla\cdot\mathbf{u}^*/\Delta t$
    3. Corrector: $\mathbf{u}^{n+1} = \mathbf{u}^* - \Delta t\nabla p^{n+1}/\rho$
    """

    def __init__(self, nx: int, ny: int, Lx: float, Ly: float,
                 rho1: float = 1.0, rho2: float = 1000.0,
                 mu1: float = 1e-3, mu2: float = 1e-3,
                 sigma: float = 0.072, g: float = -9.81) -> None:
        self.nx = nx
        self.ny = ny
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.rho1 = rho1
        self.rho2 = rho2
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma = sigma
        self.grav = g

        self.u = np.zeros((nx, ny))
        self.v = np.zeros((nx, ny))
        self.p = np.zeros((nx, ny))
        self.alpha = np.zeros((nx, ny))  # volume fraction of fluid 2

        self.csf = SurfaceTensionCSF(sigma, self.dx, self.dy)

    @property
    def rho(self) -> NDArray:
        """Mixture density."""
        return self.alpha * self.rho2 + (1 - self.alpha) * self.rho1

    @property
    def mu(self) -> NDArray:
        """Mixture viscosity."""
        return self.alpha * self.mu2 + (1 - self.alpha) * self.mu1

    def _laplacian(self, f: NDArray) -> NDArray:
        d2x = (np.roll(f, -1, 0) - 2 * f + np.roll(f, 1, 0)) / self.dx**2
        d2y = (np.roll(f, -1, 1) - 2 * f + np.roll(f, 1, 1)) / self.dy**2
        return d2x + d2y

    def _divergence(self, u: NDArray, v: NDArray) -> NDArray:
        dudx = (np.roll(u, -1, 0) - np.roll(u, 1, 0)) / (2 * self.dx)
        dvdy = (np.roll(v, -1, 1) - np.roll(v, 1, 1)) / (2 * self.dy)
        return dudx + dvdy

    def _pressure_poisson(self, rhs: NDArray, n_iter: int = 200) -> NDArray:
        """Jacobi iteration for pressure Poisson equation."""
        p = self.p.copy()
        dx2 = self.dx**2
        dy2 = self.dy**2
        coeff = 2 * (1 / dx2 + 1 / dy2)

        for _ in range(n_iter):
            p_new = ((np.roll(p, -1, 0) + np.roll(p, 1, 0)) / dx2
                      + (np.roll(p, -1, 1) + np.roll(p, 1, 1)) / dy2
                      - rhs) / coeff
            # Neumann BC approximation (zero gradient)
            p_new[0, :] = p_new[1, :]
            p_new[-1, :] = p_new[-2, :]
            p_new[:, 0] = p_new[:, 1]
            p_new[:, -1] = p_new[:, -2]
            p = p_new

        return p

    def step(self, dt: float) -> None:
        """One projection-method time step."""
        rho = self.rho
        mu = self.mu

        # Surface tension force
        fsx, fsy = self.csf.force(self.alpha)

        # Advection (upwind)
        dudx = self.u * (np.roll(self.u, -1, 0) - np.roll(self.u, 1, 0)) / (2 * self.dx)
        dudy = self.v * (np.roll(self.u, -1, 1) - np.roll(self.u, 1, 1)) / (2 * self.dy)
        dvdx = self.u * (np.roll(self.v, -1, 0) - np.roll(self.v, 1, 0)) / (2 * self.dx)
        dvdy = self.v * (np.roll(self.v, -1, 1) - np.roll(self.v, 1, 1)) / (2 * self.dy)

        # Viscous diffusion
        visc_u = mu * self._laplacian(self.u)
        visc_v = mu * self._laplacian(self.v)

        # Predictor
        u_star = self.u + dt * (-dudx - dudy + visc_u / rho + fsx / rho)
        v_star = self.v + dt * (-dvdx - dvdy + visc_v / rho + fsy / rho + self.grav)

        # Pressure Poisson
        div = self._divergence(u_star, v_star)
        rhs = rho * div / dt
        self.p = self._pressure_poisson(rhs)

        # Corrector
        dpdx = (np.roll(self.p, -1, 0) - np.roll(self.p, 1, 0)) / (2 * self.dx)
        dpdy = (np.roll(self.p, -1, 1) - np.roll(self.p, 1, 1)) / (2 * self.dy)
        self.u = u_star - dt * dpdx / rho
        self.v = v_star - dt * dpdy / rho

        # VOF advection (simple first-order)
        dalpha_dx = self.u * (np.roll(self.alpha, -1, 0) - np.roll(self.alpha, 1, 0)) / (2 * self.dx)
        dalpha_dy = self.v * (np.roll(self.alpha, -1, 1) - np.roll(self.alpha, 1, 1)) / (2 * self.dy)
        self.alpha -= dt * (dalpha_dx + dalpha_dy)
        np.clip(self.alpha, 0, 1, out=self.alpha)

    def kinetic_energy(self) -> float:
        """Total KE = ½∫ρ|u|² dV."""
        rho = self.rho
        return 0.5 * float(np.sum(rho * (self.u**2 + self.v**2)) * self.dx * self.dy)
