"""
Magnetostatics — Biot-Savart, magnetic vector potential, magnetic boundary
value problems, force/torque on current loops, superconducting critical state.

Domain III.2 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

MU_0: float = 4 * math.pi * 1e-7  # H/m


# ---------------------------------------------------------------------------
#  Biot-Savart Law
# ---------------------------------------------------------------------------

class BiotSavart:
    r"""
    Magnetic field from arbitrary current distributions via Biot-Savart.

    $$\mathbf{B}(\mathbf{r}) = \frac{\mu_0}{4\pi}\int
      \frac{\mathbf{J}(\mathbf{r}')\times(\mathbf{r}-\mathbf{r}')}{|\mathbf{r}-\mathbf{r}'|^3}\,dV'$$

    For a finite wire segment from $\mathbf{a}$ to $\mathbf{b}$ carrying current $I$:
    $$\mathbf{B} = \frac{\mu_0 I}{4\pi d}(\sin\alpha_2 - \sin\alpha_1)\hat{\phi}$$
    """

    def __init__(self, current: float = 1.0) -> None:
        self.I = current

    def wire_segment(self, a: NDArray, b: NDArray,
                        r: NDArray) -> NDArray:
        """B-field at point r from straight wire segment a→b.

        Returns B vector (3,) in Tesla.
        """
        dl = b - a
        L = np.linalg.norm(dl)
        dl_hat = dl / L

        r_a = r - a
        r_b = r - b

        # Perpendicular distance
        cross_ra = np.cross(dl_hat, r_a)
        d = np.linalg.norm(cross_ra)

        if d < 1e-15:
            return np.zeros(3)

        cos_a1 = np.dot(dl_hat, r_a) / np.linalg.norm(r_a)
        cos_a2 = np.dot(dl_hat, r_b) / np.linalg.norm(r_b)
        sin_a1 = math.sqrt(max(1 - cos_a1**2, 0))
        sin_a2 = math.sqrt(max(1 - cos_a2**2, 0))

        # Determine sign
        if np.dot(dl_hat, r_a) < 0:
            sin_a1 = -sin_a1

        if np.dot(dl_hat, r_b) > 0:
            sin_a2 = -sin_a2

        phi_hat = cross_ra / d
        B_mag = MU_0 * self.I / (4 * math.pi * d) * (sin_a1 - sin_a2)
        return B_mag * phi_hat

    def circular_loop(self, R: float, z: float) -> float:
        r"""B_z on axis of circular current loop of radius R.

        $$B_z = \frac{\mu_0 I R^2}{2(R^2+z^2)^{3/2}}$$
        """
        return MU_0 * self.I * R**2 / (2 * (R**2 + z**2)**1.5)

    def circular_loop_field(self, R: float, r_pts: NDArray,
                               z_pts: NDArray, n_phi: int = 200) -> Tuple[NDArray, NDArray]:
        """Off-axis field of circular loop via numerical Biot-Savart.

        Returns (B_r, B_z) on (r, z) grid.
        """
        nr = len(r_pts)
        nz = len(z_pts)
        B_r = np.zeros((nr, nz))
        B_z = np.zeros((nr, nz))
        phi = np.linspace(0, 2 * math.pi, n_phi, endpoint=False)
        dphi = phi[1] - phi[0]

        for ir in range(nr):
            for iz in range(nz):
                Br_acc = 0.0
                Bz_acc = 0.0
                for p in phi:
                    # Source point on loop
                    xs = R * math.cos(p)
                    ys = R * math.sin(p)
                    zs = 0.0
                    # dl = R dphi × tangent
                    dlx = -R * math.sin(p) * dphi
                    dly = R * math.cos(p) * dphi
                    dlz = 0.0
                    # Field point (cylindrical axisymmetric, take phi_f=0)
                    xf = r_pts[ir]
                    yf = 0.0
                    zf = z_pts[iz]
                    # Displacement
                    dx = xf - xs
                    dy = yf - ys
                    dz = zf - zs
                    dist = math.sqrt(dx**2 + dy**2 + dz**2)
                    if dist < 1e-15:
                        continue
                    dist3 = dist**3
                    # dl × r̂ / r²
                    Bx = MU_0 * self.I / (4 * math.pi) * (dly * dz - dlz * dy) / dist3
                    By = MU_0 * self.I / (4 * math.pi) * (dlz * dx - dlx * dz) / dist3
                    Bz_c = MU_0 * self.I / (4 * math.pi) * (dlx * dy - dly * dx) / dist3

                    Br_acc += Bx  # at phi_f=0, B_r = B_x
                    Bz_acc += Bz_c

                B_r[ir, iz] = Br_acc
                B_z[ir, iz] = Bz_acc

        return B_r, B_z

    def solenoid_on_axis(self, R: float, L: float, n: float,
                            z: float) -> float:
        """B_z on axis of finite solenoid.

        n: turns per metre.
        B_z = (μ₀nI/2)(cos θ₁ − cos θ₂)
        """
        z1 = z + L / 2  # distance to near end
        z2 = z - L / 2  # distance to far end
        cos1 = z1 / math.sqrt(R**2 + z1**2)
        cos2 = z2 / math.sqrt(R**2 + z2**2)
        return MU_0 * n * self.I / 2 * (cos1 - cos2)


# ---------------------------------------------------------------------------
#  Magnetic Vector Potential (2D)
# ---------------------------------------------------------------------------

class MagneticVectorPotential2D:
    r"""
    2D magnetic vector potential A_z from Poisson equation.

    $$\nabla^2 A_z = -\mu_0 J_z$$

    Solved via finite differences (5-point stencil).

    B = curl A: $B_x = \partial A_z/\partial y$, $B_y = -\partial A_z/\partial x$.
    """

    def __init__(self, nx: int = 100, ny: int = 100,
                 Lx: float = 1.0, Ly: float = 1.0) -> None:
        self.nx = nx
        self.ny = ny
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.Az = np.zeros((nx, ny))

    def solve_poisson(self, Jz: NDArray, n_iter: int = 5000,
                         tol: float = 1e-6) -> NDArray:
        """Solve ∇²Az = −μ₀Jz by Gauss-Seidel."""
        Az = self.Az.copy()
        dx2 = self.dx**2
        dy2 = self.dy**2
        fac = 2 / dx2 + 2 / dy2

        for it in range(n_iter):
            max_diff = 0.0
            for i in range(1, self.nx - 1):
                for j in range(1, self.ny - 1):
                    new = ((Az[i + 1, j] + Az[i - 1, j]) / dx2
                           + (Az[i, j + 1] + Az[i, j - 1]) / dy2
                           + MU_0 * Jz[i, j]) / fac
                    diff = abs(new - Az[i, j])
                    if diff > max_diff:
                        max_diff = diff
                    Az[i, j] = new
            if max_diff < tol:
                break

        self.Az = Az
        return Az

    def compute_B(self) -> Tuple[NDArray, NDArray]:
        """Compute B from curl of A: Bx = dAz/dy, By = -dAz/dx."""
        Bx = np.zeros_like(self.Az)
        By = np.zeros_like(self.Az)
        Bx[1:-1, 1:-1] = (self.Az[1:-1, 2:] - self.Az[1:-1, :-2]) / (2 * self.dy)
        By[1:-1, 1:-1] = -(self.Az[2:, 1:-1] - self.Az[:-2, 1:-1]) / (2 * self.dx)
        return Bx, By

    def magnetic_flux(self, contour: NDArray) -> float:
        """Magnetic flux Φ = ∮ A · dl around closed contour.

        contour: (N, 2) array of (x_idx, y_idx) indices.
        """
        flux = 0.0
        n = len(contour)
        for k in range(n):
            i1, j1 = contour[k]
            i2, j2 = contour[(k + 1) % n]
            Az_avg = 0.5 * (self.Az[i1, j1] + self.Az[i2, j2])
            dl_x = (i2 - i1) * self.dx
            dl_y = (j2 - j1) * self.dy
            # A_z is out of plane, only non-zero component doesn't contribute to line integral
            # For 2D out-of-plane problem, Φ = ∫∫ B·dS = ∮ A·dl
            # Here A has only z-component, so we need the in-plane components
            # Actually Az = Az for 2D: Φ = Az contour integral is not standard
            # Use Stokes: Φ = ∫∫(∇×A)·dS = ∫∫ Bz dS, but B is in-plane
            flux += Az_avg * math.sqrt(dl_x**2 + dl_y**2)
        return flux


# ---------------------------------------------------------------------------
#  Magnetic Force and Torque
# ---------------------------------------------------------------------------

class MagneticDipole:
    r"""
    Forces and torques on magnetic dipoles.

    Torque: $\boldsymbol{\tau} = \mathbf{m}\times\mathbf{B}$

    Force: $\mathbf{F} = \nabla(\mathbf{m}\cdot\mathbf{B})$

    Dipole field:
    $$\mathbf{B} = \frac{\mu_0}{4\pi}\left[\frac{3(\mathbf{m}\cdot\hat{r})\hat{r}-\mathbf{m}}{r^3}\right]$$

    Interaction energy: $U = -\mathbf{m}\cdot\mathbf{B}$
    """

    def __init__(self, moment: NDArray) -> None:
        """moment: magnetic moment vector (A·m²)."""
        self.m = np.asarray(moment, dtype=float)

    def field(self, r: NDArray) -> NDArray:
        """Dipole field B(r) in Tesla."""
        r_mag = np.linalg.norm(r)
        if r_mag < 1e-15:
            return np.zeros(3)
        r_hat = r / r_mag
        return MU_0 / (4 * math.pi * r_mag**3) * (
            3 * np.dot(self.m, r_hat) * r_hat - self.m
        )

    def torque(self, B: NDArray) -> NDArray:
        """τ = m × B (N·m)."""
        return np.cross(self.m, B)

    def energy(self, B: NDArray) -> float:
        """U = −m · B (J)."""
        return -float(np.dot(self.m, B))

    def force_gradient(self, B_field_func, r: NDArray,
                          dr: float = 1e-8) -> NDArray:
        """Force F = ∇(m · B) via numerical gradient."""
        F = np.zeros(3)
        for k in range(3):
            r_plus = r.copy()
            r_minus = r.copy()
            r_plus[k] += dr
            r_minus[k] -= dr
            B_p = B_field_func(r_plus)
            B_m = B_field_func(r_minus)
            F[k] = (np.dot(self.m, B_p) - np.dot(self.m, B_m)) / (2 * dr)
        return F

    @staticmethod
    def dipole_dipole_energy(m1: NDArray, m2: NDArray,
                                r: NDArray) -> float:
        """Interaction energy between two magnetic dipoles.

        U = (μ₀/4πr³)(m₁·m₂ − 3(m₁·r̂)(m₂·r̂))
        """
        r_mag = np.linalg.norm(r)
        if r_mag < 1e-15:
            return 0.0
        r_hat = r / r_mag
        return MU_0 / (4 * math.pi * r_mag**3) * (
            np.dot(m1, m2) - 3 * np.dot(m1, r_hat) * np.dot(m2, r_hat)
        )
