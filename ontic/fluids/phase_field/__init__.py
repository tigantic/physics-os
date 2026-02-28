"""
Phase-Field Methods — Cahn-Hilliard, Allen-Cahn, dendritic solidification,
spinodal decomposition.

Domain XIV.3 — NEW.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Allen-Cahn Equation
# ---------------------------------------------------------------------------

class AllenCahnSolver:
    r"""
    Allen-Cahn equation for phase ordering:

    $$\frac{\partial\phi}{\partial t} = -\frac{1}{\tau}(f'(\phi) - \varepsilon^2\nabla^2\phi)$$

    Double-well: $f(\phi) = \frac{1}{4}(\phi^2-1)^2$, so $f' = \phi^3 - \phi$.

    Unlike Cahn-Hilliard, Allen-Cahn does NOT conserve the order parameter.
    """

    def __init__(self, nx: int, ny: int, Lx: float = 1.0,
                 Ly: float = 1.0, epsilon: float = 0.02,
                 tau: float = 1.0) -> None:
        self.nx = nx
        self.ny = ny
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.epsilon = epsilon
        self.tau = tau
        self.phi = np.zeros((nx, ny))

    def init_random(self, mean: float = 0.0, amp: float = 0.1,
                      seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.phi = mean + amp * (2 * rng.random((self.nx, self.ny)) - 1)

    def _laplacian(self, f: NDArray) -> NDArray:
        return ((np.roll(f, -1, 0) - 2 * f + np.roll(f, 1, 0)) / self.dx**2
                + (np.roll(f, -1, 1) - 2 * f + np.roll(f, 1, 1)) / self.dy**2)

    def step(self, dt: float) -> None:
        """Explicit Euler step."""
        lap = self._laplacian(self.phi)
        self.phi += dt / self.tau * (-(self.phi**3 - self.phi) + self.epsilon**2 * lap)

    def interface_length(self) -> float:
        """Approximate total interface length: ∫|∇ϕ|dx."""
        gx = (np.roll(self.phi, -1, 0) - np.roll(self.phi, 1, 0)) / (2 * self.dx)
        gy = (np.roll(self.phi, -1, 1) - np.roll(self.phi, 1, 1)) / (2 * self.dy)
        return float(np.sum(np.sqrt(gx**2 + gy**2)) * self.dx * self.dy)


# ---------------------------------------------------------------------------
#  Dendritic Solidification (Phase-Field)
# ---------------------------------------------------------------------------

class DendriticSolidification:
    r"""
    Phase-field model for dendritic solidification (Kobayashi 1993).

    $$\tau(\hat{n})\frac{\partial\phi}{\partial t}
      = \nabla\cdot[\varepsilon(\hat{n})^2\nabla\phi]
      + \phi(1-\phi)\left(\phi - \frac{1}{2} + m(T)\right)$$

    $$\frac{\partial T}{\partial t} = D_T\nabla^2 T + \frac{L}{c_p}\frac{\partial\phi}{\partial t}$$

    Anisotropy: $\varepsilon(\hat{n}) = \bar{\varepsilon}(1 + \delta\cos(j(\theta-\theta_0)))$

    where $\theta = \arctan(\partial_y\phi / \partial_x\phi)$, j = mode (typically 4 for cubic).
    """

    def __init__(self, nx: int = 300, ny: int = 300,
                 dx: float = 0.03, aniso_strength: float = 0.05,
                 aniso_mode: int = 4) -> None:
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dx

        # Phase-field parameters
        self.eps_bar = 0.01
        self.delta = aniso_strength
        self.j = aniso_mode
        self.tau0 = 1.0
        self.alpha_pf = 0.9     # coupling strength
        self.gamma_pf = 10.0    # undercooling -> driving force

        # Thermal
        self.D_T = 2.25
        self.L_over_cp = 1.8    # latent heat / cp

        # Fields
        self.phi = np.zeros((nx, ny))
        self.T = np.ones((nx, ny)) * (-0.5)  # undercooling

    def init_seed(self, cx: Optional[int] = None, cy: Optional[int] = None,
                    radius: int = 3) -> None:
        """Place a solid seed."""
        if cx is None:
            cx = self.nx // 2
        if cy is None:
            cy = self.ny // 2
        for i in range(max(0, cx - radius), min(self.nx, cx + radius + 1)):
            for j in range(max(0, cy - radius), min(self.ny, cy + radius + 1)):
                if (i - cx)**2 + (j - cy)**2 <= radius**2:
                    self.phi[i, j] = 1.0

    def _anisotropy(self) -> Tuple[NDArray, NDArray]:
        """Compute ε(θ) and τ(θ) from interface normal angle."""
        px = (np.roll(self.phi, -1, 0) - np.roll(self.phi, 1, 0)) / (2 * self.dx)
        py = (np.roll(self.phi, -1, 1) - np.roll(self.phi, 1, 1)) / (2 * self.dy)
        theta = np.arctan2(py, px + 1e-20)

        eps = self.eps_bar * (1 + self.delta * np.cos(self.j * theta))
        tau = self.tau0 * eps**2 / self.eps_bar**2  # τ ∝ ε²
        return eps, tau

    def step(self, dt: float) -> None:
        """One time step with anisotropic phase-field + heat equation."""
        eps, tau = self._anisotropy()

        # Gradients
        px = (np.roll(self.phi, -1, 0) - np.roll(self.phi, 1, 0)) / (2 * self.dx)
        py = (np.roll(self.phi, -1, 1) - np.roll(self.phi, 1, 1)) / (2 * self.dy)

        # ∇·(ε²∇ϕ) with variable ε
        eps2 = eps**2
        d_eps2_px = (np.roll(eps2, -1, 0) * (np.roll(self.phi, -1, 0) - self.phi)
                      - eps2 * (self.phi - np.roll(self.phi, 1, 0))) / self.dx**2
        d_eps2_py = (np.roll(eps2, -1, 1) * (np.roll(self.phi, -1, 1) - self.phi)
                      - eps2 * (self.phi - np.roll(self.phi, 1, 1))) / self.dy**2
        div_eps2_grad = d_eps2_px + d_eps2_py

        # Driving force
        m = self.alpha_pf / math.pi * np.arctan(self.gamma_pf * self.T)
        reaction = self.phi * (1 - self.phi) * (self.phi - 0.5 + m)

        # Phase-field evolution
        dphi_dt = (div_eps2_grad + reaction) / (tau + 1e-15)
        self.phi += dt * dphi_dt
        np.clip(self.phi, 0, 1, out=self.phi)

        # Heat equation with latent heat release
        lap_T = ((np.roll(self.T, -1, 0) - 2 * self.T + np.roll(self.T, 1, 0)) / self.dx**2
                  + (np.roll(self.T, -1, 1) - 2 * self.T + np.roll(self.T, 1, 1)) / self.dy**2)
        self.T += dt * (self.D_T * lap_T + self.L_over_cp * dphi_dt)

    def solid_fraction(self) -> float:
        return float(np.mean(self.phi))

    def tip_position(self) -> Tuple[int, int]:
        """Find dendrite tip (furthest solid pixel from centre)."""
        cx, cy = self.nx // 2, self.ny // 2
        solid = np.argwhere(self.phi > 0.5)
        if len(solid) == 0:
            return cx, cy
        dist = (solid[:, 0] - cx)**2 + (solid[:, 1] - cy)**2
        idx = np.argmax(dist)
        return int(solid[idx, 0]), int(solid[idx, 1])

    def tip_velocity(self, pos_prev: Tuple[int, int],
                       pos_now: Tuple[int, int], dt_total: float) -> float:
        """Tip velocity from two positions."""
        dist = math.sqrt((pos_now[0] - pos_prev[0])**2 + (pos_now[1] - pos_prev[1])**2)
        return dist * self.dx / dt_total


# ---------------------------------------------------------------------------
#  Spinodal Decomposition (Cahn-Hilliard with Thermal Noise)
# ---------------------------------------------------------------------------

class SpinodalDecomposition:
    r"""
    Spinodal decomposition via Cahn-Hilliard with Flory-Huggins free energy.

    $$f(c) = c\ln c + (1-c)\ln(1-c) + \chi c(1-c)$$

    Spinodal region: $\chi > 2$ for symmetric mixture.
    Growth rate:
    $$\omega(k) = -Mk^2[\chi(1-2c_0)^{-1} - 2\chi + \varepsilon^2 k^2]$$

    Dominant wavelength: $\lambda_m = 2\pi/k_m = 2\pi\sqrt{2}\varepsilon / \sqrt{2\chi - (1-2c_0)^{-1}}$
    """

    def __init__(self, nx: int = 256, ny: int = 256,
                 Lx: float = 1.0, Ly: float = 1.0,
                 chi: float = 3.0, epsilon: float = 0.01,
                 mobility: float = 1.0) -> None:
        self.nx = nx
        self.ny = ny
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.chi = chi
        self.epsilon = epsilon
        self.M = mobility

        # Wavenumbers
        kx = np.fft.fftfreq(nx, d=self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(ny, d=self.dy) * 2 * np.pi
        self.KX, self.KY = np.meshgrid(kx, ky, indexing='ij')
        self.K2 = self.KX**2 + self.KY**2

        self.c = np.ones((nx, ny)) * 0.5  # composition

    def init_random(self, c0: float = 0.5, amp: float = 0.01,
                      seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.c = c0 + amp * (2 * rng.random((self.nx, self.ny)) - 1)

    def chemical_potential(self, c: NDArray) -> NDArray:
        """μ = df/dc − ε²∇²c.

        Flory-Huggins: df/dc = ln(c/(1−c)) + χ(1−2c).
        """
        c_safe = np.clip(c, 1e-8, 1 - 1e-8)
        df = np.log(c_safe / (1 - c_safe)) + self.chi * (1 - 2 * c_safe)
        lap_c = ((np.roll(c, -1, 0) - 2 * c + np.roll(c, 1, 0)) / self.dx**2
                  + (np.roll(c, -1, 1) - 2 * c + np.roll(c, 1, 1)) / self.dy**2)
        return df - self.epsilon**2 * lap_c

    def step_spectral(self, dt: float) -> None:
        """Semi-implicit spectral: implicit biharmonic, explicit nonlinear."""
        c_hat = np.fft.fft2(self.c)

        # Nonlinear: df/dc in real space
        c_safe = np.clip(self.c, 1e-8, 1 - 1e-8)
        df = np.log(c_safe / (1 - c_safe)) + self.chi * (1 - 2 * c_safe)
        df_hat = np.fft.fft2(df)

        # Semi-implicit
        denom = 1.0 + dt * self.M * self.epsilon**2 * self.K2**2
        c_hat_new = (c_hat - dt * self.M * self.K2 * df_hat) / denom

        self.c = np.real(np.fft.ifft2(c_hat_new))
        np.clip(self.c, 1e-8, 1 - 1e-8, out=self.c)

    def dominant_wavelength(self, c0: float = 0.5) -> float:
        """Theoretical dominant wavelength from linear stability."""
        denom = 2 * self.chi - 1.0 / (c0 * (1 - c0) + 1e-10)
        if denom <= 0:
            return float('inf')
        return 2 * math.pi * math.sqrt(2) * self.epsilon / math.sqrt(denom)

    def growth_rate(self, k: float, c0: float = 0.5) -> float:
        """Linear growth rate ω(k)."""
        d2f = 1.0 / (c0 * (1 - c0) + 1e-10) - 2 * self.chi
        return -self.M * k**2 * (d2f + self.epsilon**2 * k**2)

    def structure_factor(self) -> Tuple[NDArray, NDArray]:
        """Radially averaged structure factor S(k)."""
        c_fluct = self.c - np.mean(self.c)
        S = np.abs(np.fft.fft2(c_fluct))**2 / (self.nx * self.ny)

        # Radial average
        kx = np.fft.fftfreq(self.nx, d=self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(self.ny, d=self.dy) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        K = np.sqrt(KX**2 + KY**2)

        k_bins = np.linspace(0, np.max(K) / 2, 50)
        S_avg = np.zeros(len(k_bins) - 1)
        k_avg = np.zeros(len(k_bins) - 1)

        for b in range(len(k_bins) - 1):
            mask = (K >= k_bins[b]) & (K < k_bins[b + 1])
            if np.any(mask):
                S_avg[b] = float(np.mean(S[mask]))
                k_avg[b] = 0.5 * (k_bins[b] + k_bins[b + 1])

        return k_avg, S_avg
