"""
Geodynamo — magnetic induction equation, kinematic dynamo, mean-field αω dynamo,
Elsasser number, magnetic diffusion.

Domain XIII.3 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Physical Constants (Earth's Core)
# ---------------------------------------------------------------------------

MU_0: float = 4 * math.pi * 1e-7   # H/m
SIGMA_CORE: float = 1e6             # S/m (electrical conductivity)
ETA_MAG: float = 1.0 / (MU_0 * SIGMA_CORE)  # ~0.8 m²/s (magnetic diffusivity)
R_CORE: float = 3.48e6              # m (outer core radius)
OMEGA_EARTH: float = 7.27e-5        # rad/s


# ---------------------------------------------------------------------------
#  Magnetic Induction Equation
# ---------------------------------------------------------------------------

class MagneticInduction2D:
    r"""
    2D magnetic induction equation for kinematic dynamo.

    $$\frac{\partial\mathbf{B}}{\partial t}
      = \nabla\times(\mathbf{v}\times\mathbf{B}) + \eta\nabla^2\mathbf{B}$$

    In 2D with $B_z$ out of plane:
    $$\frac{\partial B_z}{\partial t}
      = -\nabla\cdot(B_z\mathbf{v}) + \eta\nabla^2 B_z$$

    Magnetic Reynolds number: $Rm = vL/\eta$
    """

    def __init__(self, nx: int = 128, ny: int = 128,
                 Lx: float = 1.0, Ly: float = 1.0,
                 eta: float = 0.01) -> None:
        self.nx = nx
        self.ny = ny
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.eta = eta

        self.Bz = np.zeros((nx, ny))

    def laplacian(self, f: NDArray) -> NDArray:
        """2D Laplacian (periodic BC)."""
        return ((np.roll(f, 1, 0) + np.roll(f, -1, 0) - 2 * f) / self.dx**2
                + (np.roll(f, 1, 1) + np.roll(f, -1, 1) - 2 * f) / self.dy**2)

    def advection(self, Bz: NDArray, vx: NDArray, vy: NDArray) -> NDArray:
        """−∇·(B v) = −∂(Bz vx)/∂x − ∂(Bz vy)/∂y."""
        flux_x = Bz * vx
        flux_y = Bz * vy
        div = ((np.roll(flux_x, -1, 0) - np.roll(flux_x, 1, 0)) / (2 * self.dx)
               + (np.roll(flux_y, -1, 1) - np.roll(flux_y, 1, 1)) / (2 * self.dy))
        return -div

    def step(self, vx: NDArray, vy: NDArray, dt: float) -> NDArray:
        """Advance B by one timestep."""
        dBdt = self.advection(self.Bz, vx, vy) + self.eta * self.laplacian(self.Bz)
        self.Bz += dBdt * dt
        return self.Bz

    def magnetic_energy(self) -> float:
        """E_B = ∫ B²/(2μ₀) dV (non-dim: ⟨B²⟩/2)."""
        return 0.5 * float(np.mean(self.Bz**2))

    @staticmethod
    def magnetic_reynolds(v: float, L: float, eta: float) -> float:
        """Rm = vL/η."""
        return v * L / eta


# ---------------------------------------------------------------------------
#  Mean-Field αω Dynamo
# ---------------------------------------------------------------------------

class AlphaOmegaDynamo:
    r"""
    Mean-field αω dynamo model (1D in radius).

    Toroidal field equation:
    $$\frac{\partial B_\phi}{\partial t} = s(\mathbf{B}_p\cdot\nabla)\Omega
      + \eta_T\left(\nabla^2 - \frac{1}{s^2}\right)B_\phi$$

    Poloidal field (through α-effect):
    $$\frac{\partial A}{\partial t} = \alpha B_\phi
      + \eta_T\left(\nabla^2 - \frac{1}{s^2}\right)A$$

    where $A$ = vector potential for poloidal field, $s = r\sin\theta$.

    Dynamo number: $D = \alpha_0\Omega_0 d^3/\eta_T^2$
    Critical: $|D| > D_c \approx 10^2$.
    """

    def __init__(self, nr: int = 100, r_inner: float = 0.35,
                 r_outer: float = 1.0, eta_T: float = 0.01,
                 alpha_0: float = 1.0, omega_0: float = 100.0) -> None:
        self.nr = nr
        self.r = np.linspace(r_inner, r_outer, nr)
        self.dr = (r_outer - r_inner) / (nr - 1)
        self.eta_T = eta_T
        self.alpha_0 = alpha_0
        self.omega_0 = omega_0

        self.A = np.sin(math.pi * (self.r - r_inner) / (r_outer - r_inner)) * 1e-3
        self.B_phi = np.zeros(nr)

    def alpha_profile(self) -> NDArray:
        """α(r) = α₀ cos(θ) — simplified radial profile."""
        r_mid = (self.r[0] + self.r[-1]) / 2
        return self.alpha_0 * np.cos(math.pi * (self.r - r_mid) / (self.r[-1] - self.r[0]))

    def omega_shear(self) -> NDArray:
        """d Ω/dr profile (differential rotation)."""
        return -self.omega_0 * np.ones(self.nr)

    def laplacian_1d(self, f: NDArray) -> NDArray:
        """∇² f − f/r² in spherical (simplified 1D)."""
        lap = np.zeros(self.nr)
        lap[1:-1] = (f[2:] - 2 * f[1:-1] + f[:-2]) / self.dr**2
        return lap

    def step(self, dt: float = 1e-4) -> Tuple[NDArray, NDArray]:
        """Advance A and B_φ by one timestep."""
        alpha = self.alpha_profile()
        dOmega_dr = self.omega_shear()

        lap_A = self.laplacian_1d(self.A)
        lap_B = self.laplacian_1d(self.B_phi)

        dA_dt = alpha * self.B_phi + self.eta_T * lap_A
        dB_dt = self.r * dOmega_dr * np.gradient(self.A, self.dr) + self.eta_T * lap_B

        self.A += dA_dt * dt
        self.B_phi += dB_dt * dt

        self.A[0] = 0
        self.A[-1] = 0
        self.B_phi[0] = 0
        self.B_phi[-1] = 0

        return self.A, self.B_phi

    def dynamo_number(self) -> float:
        """D = α₀ Ω₀ d³ / η_T²."""
        d = self.r[-1] - self.r[0]
        return self.alpha_0 * self.omega_0 * d**3 / self.eta_T**2

    def magnetic_energy(self) -> Tuple[float, float]:
        """E_tor = ⟨B_φ²⟩, E_pol = ⟨(∂A/∂r)²⟩."""
        E_tor = float(np.mean(self.B_phi**2))
        dAdr = np.gradient(self.A, self.dr)
        E_pol = float(np.mean(dAdr**2))
        return E_tor, E_pol


# ---------------------------------------------------------------------------
#  Elsasser Number & Dynamo Criteria
# ---------------------------------------------------------------------------

class DynamoParameters:
    r"""
    Dimensionless parameters governing core dynamics.

    Elsasser number:
    $$\Lambda = \frac{B^2}{\rho\mu_0\eta\Omega}$$

    $\Lambda \sim 1$ indicates magnetostrophic balance (Lorentz ∼ Coriolis).

    Magnetic Ekman number:
    $$Ek_m = \frac{\eta}{\Omega L^2}$$

    Rossby number:
    $$Ro = \frac{v}{\Omega L}$$

    Modified Rayleigh number:
    $$Ra_q = \frac{\alpha g q d^2}{\kappa\Omega}$$
    """

    def __init__(self, B: float = 2e-3, rho: float = 1.1e4,
                 Omega: float = OMEGA_EARTH, L: float = R_CORE) -> None:
        self.B = B
        self.rho = rho
        self.Omega = Omega
        self.L = L

    def elsasser(self) -> float:
        """Λ = B²/(ρ μ₀ η Ω)."""
        return self.B**2 / (self.rho * MU_0 * ETA_MAG * self.Omega)

    def magnetic_ekman(self) -> float:
        """Ek_m = η/(Ω L²)."""
        return ETA_MAG / (self.Omega * self.L**2)

    def rossby(self, v: float = 5e-4) -> float:
        """Ro = v/(Ω L)."""
        return v / (self.Omega * self.L)

    def magnetic_reynolds_core(self, v: float = 5e-4) -> float:
        """Rm = v L / η."""
        return v * self.L / ETA_MAG

    def magnetic_prandtl(self, nu: float = 1e-6) -> float:
        """Pm = ν/η."""
        return nu / ETA_MAG

    def summary(self, v: float = 5e-4) -> Dict[str, float]:
        """Return all dimensionless numbers."""
        return {
            'Elsasser': self.elsasser(),
            'Magnetic_Ekman': self.magnetic_ekman(),
            'Rossby': self.rossby(v),
            'Rm': self.magnetic_reynolds_core(v),
            'Pm': self.magnetic_prandtl(),
        }
