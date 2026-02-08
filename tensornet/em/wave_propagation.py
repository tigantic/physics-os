"""
EM Wave Propagation — FDTD, plane waves, waveguides, transmission-line model,
Mie scattering.

Domain III.5 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

C_LIGHT: float = 2.998e8
EPS_0: float = 8.854e-12
MU_0: float = 4 * math.pi * 1e-7
ETA_0: float = math.sqrt(MU_0 / EPS_0)


# ---------------------------------------------------------------------------
#  1D FDTD — EM Wave Propagation
# ---------------------------------------------------------------------------

class FDTD1D:
    r"""
    1D Finite-Difference Time-Domain for EM wave propagation.

    Maxwell's equations (1D, TEM, $E_x$, $H_y$):
    $$\frac{\partial E_x}{\partial t} = -\frac{1}{\epsilon}\frac{\partial H_y}{\partial z}
      - \frac{\sigma}{\epsilon}E_x$$
    $$\frac{\partial H_y}{\partial t} = -\frac{1}{\mu}\frac{\partial E_x}{\partial z}$$

    Yee discretisation (leapfrog):
    $$E_x^{n+1}(k) = C_a E_x^n(k) + C_b[H_y^n(k) - H_y^n(k-1)]$$
    $$H_y^{n+1/2}(k) = H_y^{n-1/2}(k) - \frac{\Delta t}{\mu\Delta z}[E_x^n(k+1) - E_x^n(k)]$$

    CFL: $\Delta t \leq \Delta z / c$
    """

    def __init__(self, nz: int = 1000, dz: float = 1e-3,
                 n_steps: int = 3000) -> None:
        self.nz = nz
        self.dz = dz
        self.dt = 0.99 * dz / C_LIGHT  # Courant number ~1
        self.n_steps = n_steps

        self.eps_r = np.ones(nz)
        self.mu_r = np.ones(nz)
        self.sigma = np.zeros(nz)

        self.Ex = np.zeros(nz)
        self.Hy = np.zeros(nz)

    def set_material(self, i_start: int, i_end: int,
                        eps_r: float = 1.0, sigma: float = 0.0) -> None:
        """Set material properties in region [i_start, i_end)."""
        self.eps_r[i_start:i_end] = eps_r
        self.sigma[i_start:i_end] = sigma

    def add_pml(self, n_pml: int = 20, sigma_max: float = 1.0) -> None:
        """Add absorbing boundary via conductivity grading."""
        for i in range(n_pml):
            s = sigma_max * ((n_pml - i) / n_pml)**3
            self.sigma[i] = s
            self.sigma[self.nz - 1 - i] = s

    def run(self, source_pos: int = 50,
               freq: float = 1e9) -> Dict[str, NDArray]:
        """Run FDTD simulation.

        Returns final Ex, Hy and time history at source.
        """
        Ca = (1 - self.sigma * self.dt / (2 * EPS_0 * self.eps_r)) / \
             (1 + self.sigma * self.dt / (2 * EPS_0 * self.eps_r))
        Cb = self.dt / (EPS_0 * self.eps_r * self.dz) / \
             (1 + self.sigma * self.dt / (2 * EPS_0 * self.eps_r))

        Ex_history = np.zeros(self.n_steps)

        for n in range(self.n_steps):
            # Update H
            self.Hy[:-1] -= self.dt / (MU_0 * self.mu_r[:-1] * self.dz) * (
                self.Ex[1:] - self.Ex[:-1])

            # Gaussian pulse source
            t = n * self.dt
            t0 = 30 * self.dt
            spread = 10 * self.dt
            pulse = math.exp(-0.5 * ((t - t0) / spread)**2) * math.sin(
                2 * math.pi * freq * t)
            self.Hy[source_pos] += pulse

            # Update E
            self.Ex[1:] = Ca[1:] * self.Ex[1:] + Cb[1:] * (
                self.Hy[1:] - self.Hy[:-1])

            Ex_history[n] = self.Ex[source_pos + 50]

        return {
            'Ex': self.Ex.copy(),
            'Hy': self.Hy.copy(),
            'time_history': Ex_history,
            'z': np.arange(self.nz) * self.dz,
        }


# ---------------------------------------------------------------------------
#  2D FDTD — TM Polarisation
# ---------------------------------------------------------------------------

class FDTD2D_TM:
    r"""
    2D FDTD for TM polarisation ($E_z$, $H_x$, $H_y$).

    $$\frac{\partial H_x}{\partial t} = -\frac{1}{\mu}\frac{\partial E_z}{\partial y}$$
    $$\frac{\partial H_y}{\partial t} = \frac{1}{\mu}\frac{\partial E_z}{\partial x}$$
    $$\frac{\partial E_z}{\partial t} = \frac{1}{\epsilon}\left(
      \frac{\partial H_y}{\partial x} - \frac{\partial H_x}{\partial y}\right)$$

    CFL: $\Delta t \leq \frac{1}{c\sqrt{1/\Delta x^2 + 1/\Delta y^2}}$
    """

    def __init__(self, nx: int = 200, ny: int = 200,
                 dx: float = 1e-3, dy: float = 1e-3) -> None:
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.dt = 0.99 / (C_LIGHT * math.sqrt(1 / dx**2 + 1 / dy**2))

        self.eps_r = np.ones((nx, ny))
        self.Ez = np.zeros((nx, ny))
        self.Hx = np.zeros((nx, ny))
        self.Hy = np.zeros((nx, ny))

    def set_permittivity(self, eps_r: NDArray) -> None:
        self.eps_r = eps_r.copy()

    def step(self) -> None:
        """Advance one FDTD timestep."""
        # Update H
        self.Hx[:, :-1] -= self.dt / (MU_0 * self.dy) * (
            self.Ez[:, 1:] - self.Ez[:, :-1])
        self.Hy[:-1, :] += self.dt / (MU_0 * self.dx) * (
            self.Ez[1:, :] - self.Ez[:-1, :])

        # Update E
        self.Ez[1:-1, 1:-1] += self.dt / (EPS_0 * self.eps_r[1:-1, 1:-1]) * (
            (self.Hy[1:-1, 1:-1] - self.Hy[:-2, 1:-1]) / self.dx
            - (self.Hx[1:-1, 1:-1] - self.Hx[1:-1, :-2]) / self.dy)

    def run(self, n_steps: int = 500, source_pos: Tuple[int, int] = (100, 100),
               freq: float = 10e9) -> NDArray:
        """Run 2D FDTD with point source."""
        for n in range(n_steps):
            self.step()
            t = n * self.dt
            self.Ez[source_pos[0], source_pos[1]] += math.sin(
                2 * math.pi * freq * t)
        return self.Ez.copy()


# ---------------------------------------------------------------------------
#  Mie Scattering
# ---------------------------------------------------------------------------

class MieScattering:
    r"""
    Mie theory for scattering by a dielectric sphere.

    Size parameter: $x = ka = 2\pi a n_m / \lambda$

    Scattering coefficients:
    $$a_n = \frac{m\psi_n(mx)\psi_n'(x) - \psi_n(x)\psi_n'(mx)}{m\psi_n(mx)\xi_n'(x) - \xi_n(x)\psi_n'(mx)}$$
    $$b_n = \frac{\psi_n(mx)\psi_n'(x) - m\psi_n(x)\psi_n'(mx)}{\psi_n(mx)\xi_n'(x) - m\xi_n(x)\psi_n'(mx)}$$

    Cross sections:
    $$Q_{\text{sca}} = \frac{2}{x^2}\sum_n(2n+1)(|a_n|^2+|b_n|^2)$$
    $$Q_{\text{ext}} = \frac{2}{x^2}\sum_n(2n+1)\,\text{Re}(a_n+b_n)$$
    """

    def __init__(self, radius: float = 500e-9, n_sphere: complex = 1.5,
                 n_medium: float = 1.0) -> None:
        self.a = radius
        self.n_s = n_sphere
        self.n_m = n_medium
        self.m = n_sphere / n_medium

    def size_parameter(self, wavelength: float) -> float:
        """x = 2πa n_m/λ."""
        return 2 * math.pi * self.a * self.n_m / wavelength

    def _psi(self, n: int, z: complex) -> complex:
        """Riccati-Bessel ψ_n(z) = z j_n(z)."""
        # j_0(z) = sin(z)/z, j_1(z) = sin(z)/z² - cos(z)/z
        if abs(z) < 1e-15:
            return complex(0)

        if n == 0:
            if isinstance(z, complex):
                val = np.sin(z) / z
            else:
                val = math.sin(z) / z
            return z * val
        elif n == 1:
            if isinstance(z, complex):
                val = np.sin(z) / z**2 - np.cos(z) / z
            else:
                val = math.sin(z) / z**2 - math.cos(z) / z
            return z * val
        else:
            # Upward recurrence for j_n
            j_prev = np.sin(z) / z if isinstance(z, complex) else complex(math.sin(z) / z)
            j_curr = np.sin(z) / z**2 - np.cos(z) / z if isinstance(z, complex) else complex(math.sin(z) / z**2 - math.cos(z) / z)
            for k in range(2, n + 1):
                j_next = (2 * k - 1) / z * j_curr - j_prev
                j_prev = j_curr
                j_curr = j_next
            return z * j_curr

    def _dpsi(self, n: int, z: complex) -> complex:
        """Derivative ψ_n'(z) via recurrence."""
        return self._psi(n - 1, z) - n / z * self._psi(n, z) if n > 0 else complex(np.cos(z))

    def coefficients(self, wavelength: float,
                        n_max: int = 0) -> Tuple[NDArray, NDArray]:
        """Compute Mie coefficients a_n, b_n.

        n_max: number of terms (0 = auto).
        """
        x = self.size_parameter(wavelength)
        mx = self.m * x

        if n_max == 0:
            n_max = int(x + 4 * x**(1 / 3) + 2)

        an = np.zeros(n_max, dtype=complex)
        bn = np.zeros(n_max, dtype=complex)

        for n in range(1, n_max + 1):
            psi_mx = self._psi(n, mx)
            psi_x = self._psi(n, x)
            dpsi_mx = self._dpsi(n, mx)
            dpsi_x = self._dpsi(n, x)

            # xi_n = psi_n + i chi_n (approximation)
            xi_x = psi_x + 1j * self._psi(n, complex(0, -x.imag if isinstance(x, complex) else 0) + x)

            num_a = self.m * psi_mx * dpsi_x - psi_x * dpsi_mx
            den_a = self.m * psi_mx * dpsi_x - xi_x * dpsi_mx
            an[n - 1] = num_a / den_a if abs(den_a) > 1e-30 else 0

            num_b = psi_mx * dpsi_x - self.m * psi_x * dpsi_mx
            den_b = psi_mx * dpsi_x - self.m * xi_x * dpsi_mx
            bn[n - 1] = num_b / den_b if abs(den_b) > 1e-30 else 0

        return an, bn

    def efficiencies(self, wavelength: float) -> Dict[str, float]:
        """Scattering, extinction, absorption efficiencies."""
        x = self.size_parameter(wavelength)
        an, bn = self.coefficients(wavelength)
        n_terms = len(an)
        ns = np.arange(1, n_terms + 1)

        Q_sca = 2 / x**2 * float(np.sum((2 * ns + 1) * (np.abs(an)**2 + np.abs(bn)**2)))
        Q_ext = 2 / x**2 * float(np.sum((2 * ns + 1) * np.real(an + bn)))
        Q_abs = Q_ext - Q_sca

        return {'Q_sca': Q_sca, 'Q_ext': Q_ext, 'Q_abs': Q_abs}
