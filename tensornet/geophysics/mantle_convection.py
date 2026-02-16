"""
Mantle Convection — Stokes flow, Boussinesq convection, thermal boundary layers,
Rayleigh number scaling, plate-like behaviour.

Domain XIII.2 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Physical Parameters (Earth's Mantle)
# ---------------------------------------------------------------------------

RHO_0: float = 3300.0        # kg/m³ (reference density)
ALPHA_TH: float = 3e-5       # K⁻¹ (thermal expansion)
G_EARTH: float = 9.81        # m/s²
ETA_MANTLE: float = 1e21     # Pa·s (reference viscosity)
KAPPA_TH: float = 1e-6       # m²/s (thermal diffusivity)
CP_MANTLE: float = 1250.0    # J/(kg·K)
D_MANTLE: float = 2.89e6     # m (mantle thickness)


# ---------------------------------------------------------------------------
#  Stokes Flow Solver (2D)
# ---------------------------------------------------------------------------

class StokesFlow2D:
    r"""
    2D Stokes flow for mantle convection (Boussinesq approximation).

    $$\nabla\cdot\boldsymbol{\sigma} + \rho_0\alpha\Delta T\,g\hat{z} = 0$$
    $$\nabla\cdot\mathbf{v} = 0$$

    Stream function formulation: $v_x = \partial\psi/\partial z$,
    $v_z = -\partial\psi/\partial x$.

    $$\nabla^4\psi = -\frac{\rho_0\alpha g}{\eta}\frac{\partial T}{\partial x}$$
    """

    def __init__(self, nx: int = 64, nz: int = 64,
                 Lx: float = 1.0, Lz: float = 1.0) -> None:
        self.nx = nx
        self.nz = nz
        self.Lx = Lx
        self.Lz = Lz
        self.dx = Lx / (nx - 1)
        self.dz = Lz / (nz - 1)

    def biharmonic_solve(self, T: NDArray, Ra: float) -> NDArray:
        """Solve ∇⁴ψ = −Ra ∂T/∂x via spectral method.

        Uses discrete sine transform (free-slip BCs).
        """
        dTdx = np.gradient(T, self.dx, axis=0)
        rhs = -Ra * dTdx

        from scipy.fft import dstn, idstn

        rhs_hat = dstn(rhs, type=1)

        kx = np.arange(1, self.nx + 1) * math.pi / self.Lx
        kz = np.arange(1, self.nz + 1) * math.pi / self.Lz
        KX, KZ = np.meshgrid(kx, kz, indexing='ij')
        K4 = (KX**2 + KZ**2)**2
        K4 = np.maximum(K4, 1e-10)

        psi_hat = rhs_hat / K4
        psi = idstn(psi_hat, type=1) / (4 * self.nx * self.nz)

        return psi

    def velocity_from_stream(self, psi: NDArray) -> Tuple[NDArray, NDArray]:
        """vx = ∂ψ/∂z, vz = −∂ψ/∂x."""
        vx = np.gradient(psi, self.dz, axis=1)
        vz = -np.gradient(psi, self.dx, axis=0)
        return vx, vz


# ---------------------------------------------------------------------------
#  Thermal Convection Solver
# ---------------------------------------------------------------------------

class MantleConvection2D:
    r"""
    2D thermal convection in a Boussinesq fluid.

    Energy equation (non-dimensional):
    $$\frac{\partial T}{\partial t} + \mathbf{v}\cdot\nabla T = \nabla^2 T + H$$

    where $H$ is internal heating rate.

    Rayleigh number:
    $$Ra = \frac{\rho_0\alpha g\Delta T d^3}{\eta\kappa}$$

    Critical Rayleigh number (free-slip): $Ra_c = 657.5$.
    """

    def __init__(self, nx: int = 64, nz: int = 64,
                 Ra: float = 1e4, H: float = 0.0) -> None:
        self.nx = nx
        self.nz = nz
        self.Ra = Ra
        self.H = H
        self.dx = 1.0 / (nx - 1)
        self.dz = 1.0 / (nz - 1)

        self.T = np.zeros((nx, nz))
        x = np.linspace(0, 1, nx)
        z = np.linspace(0, 1, nz)
        X, Z = np.meshgrid(x, z, indexing='ij')
        self.T = 1.0 - Z + 0.01 * np.sin(math.pi * X) * np.sin(math.pi * Z)

        self.stokes = StokesFlow2D(nx, nz)

    def diffusion_step(self, T: NDArray, dt: float) -> NDArray:
        """Explicit diffusion: T += dt ∇²T."""
        lap = (np.roll(T, 1, 0) + np.roll(T, -1, 0) - 2 * T) / self.dx**2
        lap += (np.roll(T, 1, 1) + np.roll(T, -1, 1) - 2 * T) / self.dz**2
        return T + dt * lap + dt * self.H

    def advection_step(self, T: NDArray, vx: NDArray, vz: NDArray,
                          dt: float) -> NDArray:
        """Upwind advection."""
        dTdx = np.where(vx > 0,
                        T - np.roll(T, 1, axis=0),
                        np.roll(T, -1, axis=0) - T) / self.dx
        dTdz = np.where(vz > 0,
                        T - np.roll(T, 1, axis=1),
                        np.roll(T, -1, axis=1) - T) / self.dz
        return T - dt * (vx * dTdx + vz * dTdz)

    def apply_bc(self, T: NDArray) -> NDArray:
        """T(z=0) = 1 (base), T(z=1) = 0 (surface)."""
        T[:, 0] = 1.0
        T[:, -1] = 0.0
        return T

    def step(self, dt: float = 1e-5) -> Tuple[NDArray, NDArray, NDArray]:
        """Full timestep: Stokes solve + advection-diffusion.

        Returns (T, vx, vz).
        """
        psi = self.stokes.biharmonic_solve(self.T, self.Ra)
        vx, vz = self.stokes.velocity_from_stream(psi)

        self.T = self.advection_step(self.T, vx, vz, dt)
        self.T = self.diffusion_step(self.T, dt)
        self.T = self.apply_bc(self.T)

        return self.T, vx, vz

    def nusselt_number(self) -> float:
        """Surface Nusselt number: Nu = −⟨∂T/∂z⟩|_{z=1}.

        Nu > 1 indicates convective heat transport.
        """
        dTdz_surface = -(self.T[:, -1] - self.T[:, -2]) / self.dz
        return float(np.mean(dTdz_surface))

    @staticmethod
    def rayleigh_number(delta_T: float = 2500.0, d: float = D_MANTLE,
                           eta: float = ETA_MANTLE) -> float:
        """Ra = ρ₀αgΔTd³/(ηκ)."""
        return RHO_0 * ALPHA_TH * G_EARTH * delta_T * d**3 / (eta * KAPPA_TH)


# ---------------------------------------------------------------------------
#  Viscosity Models
# ---------------------------------------------------------------------------

class MantleViscosity:
    r"""
    Temperature- and pressure-dependent viscosity for mantle flow.

    Arrhenius rheology:
    $$\eta(T,P) = A^{-1/n}\dot{\varepsilon}^{(1-n)/n}
      \exp\left(\frac{E^* + PV^*}{nRT}\right)$$

    Diffusion creep (n=1):
    $$\eta_{\text{diff}} = \frac{d^m}{A_{\text{diff}}}
      \exp\left(\frac{E^*_{\text{diff}} + PV^*_{\text{diff}}}{RT}\right)$$

    Dislocation creep (n≈3.5):
    $$\eta_{\text{disl}} = A_{\text{disl}}^{-1/n}\dot{\varepsilon}^{(1-n)/n}
      \exp\left(\frac{E^*_{\text{disl}} + PV^*_{\text{disl}}}{nRT}\right)$$
    """

    R_GAS: float = 8.314  # J/(mol·K)

    def __init__(self) -> None:
        # Olivine diffusion creep (Hirth & Kohlstedt, 2003)
        self.E_diff: float = 375e3   # J/mol
        self.V_diff: float = 6e-6    # m³/mol
        self.A_diff: float = 1.5e9   # Pa·s (pre-exponential)
        self.grain_exp: float = 3.0  # grain size exponent

        # Olivine dislocation creep
        self.E_disl: float = 530e3
        self.V_disl: float = 14e-6
        self.A_disl: float = 3.5e22   # 1/Pa^n/s
        self.n_disl: float = 3.5

    def diffusion_creep(self, T: float, P: float,
                           d_grain: float = 1e-3) -> float:
        """Diffusion creep viscosity (Pa·s).

        d_grain in metres.
        """
        if T < 1.0:
            return 1e30
        return (self.A_diff * d_grain**self.grain_exp
                * math.exp((self.E_diff + P * self.V_diff) / (self.R_GAS * T)))

    def dislocation_creep(self, T: float, P: float,
                             strain_rate: float = 1e-15) -> float:
        """Dislocation creep viscosity (Pa·s)."""
        if T < 1.0:
            return 1e30
        n = self.n_disl
        return (self.A_disl**(-1 / n) * strain_rate**((1 - n) / n)
                * math.exp((self.E_disl + P * self.V_disl) / (n * self.R_GAS * T)))

    def composite(self, T: float, P: float, strain_rate: float = 1e-15,
                     d_grain: float = 1e-3) -> float:
        """Composite viscosity: 1/η = 1/η_diff + 1/η_disl."""
        eta_d = self.diffusion_creep(T, P, d_grain)
        eta_n = self.dislocation_creep(T, P, strain_rate)
        return 1.0 / (1.0 / eta_d + 1.0 / eta_n)
