"""
Frequency-Domain Electromagnetics — FDFD, Method of Moments, 2D scattering,
radar cross-section, impedance boundary conditions.

Domain III.4 — NEW.
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

C_LIGHT: float = 2.998e8     # m/s
EPS_0: float = 8.854e-12     # F/m
MU_0: float = 4 * math.pi * 1e-7  # H/m
ETA_0: float = math.sqrt(MU_0 / EPS_0)  # ~377 Ω


# ---------------------------------------------------------------------------
#  Finite Difference Frequency Domain (FDFD) — 2D TM
# ---------------------------------------------------------------------------

class FDFD2D_TM:
    r"""
    2D Finite-Difference Frequency-Domain for TM polarisation ($E_z$, $H_x$, $H_y$).

    $$\nabla\times\nabla\times\mathbf{E} - k_0^2\epsilon_r\mathbf{E} = -i\omega\mu_0\mathbf{J}$$

    Discretised:
    $$\left(\mathbf{D}_x\mu_r^{-1}\mathbf{D}_x + \mathbf{D}_y\mu_r^{-1}\mathbf{D}_y
      + k_0^2\epsilon_r\right)E_z = -i\omega\mu_0 J_z$$

    where $\mathbf{D}_x$, $\mathbf{D}_y$ are discrete derivative operators.

    PML via stretched coordinates: $\tilde{x} = \int_0^x s_x(x')dx'$
    """

    def __init__(self, nx: int = 100, ny: int = 100,
                 Lx: float = 1.0, Ly: float = 1.0,
                 freq: float = 1e9) -> None:
        """
        Lx, Ly: domain size (m).
        freq: frequency (Hz).
        """
        self.nx = nx
        self.ny = ny
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.freq = freq
        self.omega = 2 * math.pi * freq
        self.k0 = self.omega / C_LIGHT

        self.eps_r = np.ones((nx, ny))
        self.mu_r = np.ones((nx, ny))
        self.Jz = np.zeros((nx, ny), dtype=complex)

    def set_permittivity(self, eps_r: NDArray) -> None:
        """Set relative permittivity profile."""
        self.eps_r = eps_r.copy()

    def set_source(self, ix: int, iy: int, amplitude: complex = 1.0) -> None:
        """Place point source at grid location."""
        self.Jz[ix, iy] = amplitude

    def build_system(self) -> Tuple[NDArray, NDArray]:
        """Build sparse-like system matrix A and RHS b.

        A Ez = b, for the vectorised Ez field.
        Returns dense matrix (N×N) and RHS (N,).
        """
        N = self.nx * self.ny
        A = np.zeros((N, N), dtype=complex)
        b = np.zeros(N, dtype=complex)

        dx2 = self.dx**2
        dy2 = self.dy**2

        def idx(i: int, j: int) -> int:
            return i * self.ny + j

        for i in range(self.nx):
            for j in range(self.ny):
                n = idx(i, j)

                # PEC boundary
                if i == 0 or i == self.nx - 1 or j == 0 or j == self.ny - 1:
                    A[n, n] = 1.0
                    b[n] = 0.0
                    continue

                # Laplacian + k²ε_r
                A[n, idx(i + 1, j)] = 1.0 / dx2
                A[n, idx(i - 1, j)] = 1.0 / dx2
                A[n, idx(i, j + 1)] = 1.0 / dy2
                A[n, idx(i, j - 1)] = 1.0 / dy2
                A[n, n] = -2.0 / dx2 - 2.0 / dy2 + self.k0**2 * self.eps_r[i, j]

                b[n] = -1j * self.omega * MU_0 * self.Jz[i, j]

        return A, b

    def solve(self) -> NDArray:
        """Solve for Ez field."""
        A, b = self.build_system()
        Ez_vec = np.linalg.solve(A, b)
        return Ez_vec.reshape(self.nx, self.ny)


# ---------------------------------------------------------------------------
#  Method of Moments (2D TM Scattering)
# ---------------------------------------------------------------------------

class MethodOfMoments2D:
    r"""
    Method of Moments for 2D TM scattering from a PEC cylinder.

    EFIE (2D):
    $$E_z^{\text{inc}}(\boldsymbol{\rho}) = -\frac{k\eta}{4}\oint
      J_z(\boldsymbol{\rho}')H_0^{(2)}(k|\boldsymbol{\rho}-\boldsymbol{\rho}'|)\,dl'$$

    Discretised with pulse basis and point matching:
    $$[Z_{mn}][I_n] = [V_m]$$

    $Z_{mn} = -\frac{k\eta\Delta\ell}{4}H_0^{(2)}(k|\mathbf{r}_m-\mathbf{r}_n|)$

    Self-term: $Z_{nn} = -\frac{k\eta\Delta\ell}{4}[1-j(2/\pi)\ln(k\Delta\ell\gamma_e/4)]$
    """

    def __init__(self, freq: float = 1e9, radius: float = 0.1,
                 n_segments: int = 60) -> None:
        self.freq = freq
        self.omega = 2 * math.pi * freq
        self.k = self.omega / C_LIGHT
        self.eta = ETA_0
        self.radius = radius
        self.n_seg = n_segments
        self.gamma_e = 1.7811  # exp(Euler-Mascheroni)

        # Boundary discretisation
        theta = np.linspace(0, 2 * math.pi, n_segments, endpoint=False)
        self.x = radius * np.cos(theta)
        self.y = radius * np.sin(theta)
        self.dl = 2 * math.pi * radius / n_segments

    def impedance_matrix(self) -> NDArray:
        """Build MoM impedance matrix Z."""
        Z = np.zeros((self.n_seg, self.n_seg), dtype=complex)

        for m in range(self.n_seg):
            for n in range(self.n_seg):
                if m == n:
                    Z[m, n] = (-self.k * self.eta * self.dl / 4 *
                               (1 - 2j / math.pi *
                                math.log(self.k * self.dl * self.gamma_e / 4)))
                else:
                    dx = self.x[m] - self.x[n]
                    dy = self.y[m] - self.y[n]
                    dist = math.sqrt(dx**2 + dy**2)
                    kr = self.k * dist
                    # H₀⁽²⁾(kr) ≈ J₀(kr) − iY₀(kr) using small-argument approx
                    H0 = self._hankel2_0(kr)
                    Z[m, n] = -self.k * self.eta * self.dl / 4 * H0

        return Z

    def _hankel2_0(self, x: float) -> complex:
        """Hankel function H₀⁽²⁾(x) = J₀(x) − iY₀(x) via series."""
        # J₀(x) Bessel series
        J0 = 0.0
        for k in range(20):
            J0 += (-1)**k * (x / 2)**(2 * k) / (math.factorial(k))**2

        # Y₀(x) Neumann series (large/small argument)
        if x < 1e-10:
            Y0 = -1e10
        else:
            Y0 = 2 / math.pi * (math.log(x / 2) + 0.5772) * J0
            S = 0.0
            Hk = 0.0
            for k in range(1, 20):
                Hk += 1 / k
                S += (-1)**(k + 1) * (x / 2)**(2 * k) / (math.factorial(k))**2 * Hk
            Y0 += 2 / math.pi * S

        return complex(J0, -Y0)

    def excitation_vector(self, theta_inc: float = 0.0) -> NDArray:
        """Plane-wave excitation V_m = E_z^inc(r_m).

        theta_inc: angle of incidence (radians).
        """
        kx = self.k * math.cos(theta_inc)
        ky = self.k * math.sin(theta_inc)
        V = np.exp(-1j * (kx * self.x + ky * self.y))
        return V

    def solve(self, theta_inc: float = 0.0) -> NDArray:
        """Solve for surface currents J."""
        Z = self.impedance_matrix()
        V = self.excitation_vector(theta_inc)
        return np.linalg.solve(Z, V)

    def rcs(self, J: NDArray, n_angles: int = 360) -> Tuple[NDArray, NDArray]:
        """Bistatic radar cross-section (2D).

        σ₂D(φ) = (2π/k)|Σ J_n exp(jk r̂·rₙ) Δℓ|²
        """
        phi = np.linspace(0, 2 * math.pi, n_angles)
        sigma = np.zeros(n_angles)

        for ip in range(n_angles):
            kx = self.k * math.cos(phi[ip])
            ky = self.k * math.sin(phi[ip])
            S = np.sum(J * np.exp(1j * (kx * self.x + ky * self.y)) * self.dl)
            sigma[ip] = 2 * math.pi / self.k * abs(S)**2

        return np.degrees(phi), sigma
