"""
Microstructure Evolution — phase field, grain growth, Cahn-Hilliard,
Allen-Cahn, nucleation, Ostwald ripening.

Domain XIV.4 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Cahn-Hilliard Equation (Spinodal Decomposition)
# ---------------------------------------------------------------------------

class CahnHilliard2D:
    r"""
    Cahn-Hilliard equation for spinodal decomposition.

    $$\frac{\partial c}{\partial t} = M\nabla^2\left(\frac{\partial f}{\partial c}
      - \kappa\nabla^2 c\right)$$

    Double-well free energy density:
    $$f(c) = W c^2(1 - c)^2$$

    Chemical potential:
    $$\mu = \frac{\partial f}{\partial c} - \kappa\nabla^2 c
      = W[2c(1-c)^2 - 2c^2(1-c)] - \kappa\nabla^2 c$$

    Interfacial energy: $\gamma = \frac{1}{6}\sqrt{2\kappa W}$
    Interface width: $\ell \sim \sqrt{\kappa/W}$
    """

    def __init__(self, nx: int = 128, ny: int = 128,
                 dx: float = 1.0, M: float = 1.0,
                 kappa: float = 0.5, W: float = 1.0) -> None:
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.M = M
        self.kappa = kappa
        self.W = W

        self.c = 0.5 + 0.02 * np.random.randn(nx, ny)
        self.c = np.clip(self.c, 0, 1)

    def laplacian(self, f: NDArray) -> NDArray:
        """Periodic Laplacian."""
        return ((np.roll(f, 1, 0) + np.roll(f, -1, 0)
                + np.roll(f, 1, 1) + np.roll(f, -1, 1)
                - 4 * f) / self.dx**2)

    def df_dc(self, c: NDArray) -> NDArray:
        """∂f/∂c for double-well potential."""
        return self.W * (2 * c * (1 - c)**2 - 2 * c**2 * (1 - c))

    def chemical_potential(self, c: NDArray) -> NDArray:
        """μ = ∂f/∂c − κ∇²c."""
        return self.df_dc(c) - self.kappa * self.laplacian(c)

    def step(self, dt: float = 0.01) -> NDArray:
        """Forward Euler timestep."""
        mu = self.chemical_potential(self.c)
        self.c += dt * self.M * self.laplacian(mu)
        return self.c

    def free_energy(self) -> float:
        """Total free energy: F = ∫[f(c) + (κ/2)|∇c|²] dA."""
        f_bulk = self.W * self.c**2 * (1 - self.c)**2
        dcx = (np.roll(self.c, -1, 0) - np.roll(self.c, 1, 0)) / (2 * self.dx)
        dcy = (np.roll(self.c, -1, 1) - np.roll(self.c, 1, 1)) / (2 * self.dx)
        f_grad = 0.5 * self.kappa * (dcx**2 + dcy**2)
        return float(np.sum(f_bulk + f_grad) * self.dx**2)

    def interfacial_energy(self) -> float:
        """γ = (1/6)√(2κW)."""
        return math.sqrt(2 * self.kappa * self.W) / 6


# ---------------------------------------------------------------------------
#  Allen-Cahn Equation (Order-Disorder Transitions)
# ---------------------------------------------------------------------------

class AllenCahn2D:
    r"""
    Allen-Cahn equation for non-conserved order parameter.

    $$\frac{\partial\phi}{\partial t} = -L\left(\frac{\partial f}{\partial\phi}
      - \kappa\nabla^2\phi\right)$$

    where $L$ is kinetic coefficient.

    Curvature-driven motion: $v_n = -L\kappa\,\mathcal{K}$
    where $\mathcal{K}$ is interface curvature.
    """

    def __init__(self, nx: int = 128, ny: int = 128,
                 dx: float = 1.0, L: float = 1.0,
                 kappa: float = 0.5, W: float = 1.0) -> None:
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.L = L
        self.kappa = kappa
        self.W = W

        self.phi = np.zeros((nx, ny))

    def laplacian(self, f: NDArray) -> NDArray:
        return ((np.roll(f, 1, 0) + np.roll(f, -1, 0)
                + np.roll(f, 1, 1) + np.roll(f, -1, 1)
                - 4 * f) / self.dx**2)

    def df_dphi(self, phi: NDArray) -> NDArray:
        """∂f/∂φ for double well: 4W φ(φ² − 1)."""
        return 4 * self.W * phi * (phi**2 - 1)

    def step(self, dt: float = 0.01) -> NDArray:
        """Forward Euler step."""
        rhs = -self.L * (self.df_dphi(self.phi) - self.kappa * self.laplacian(self.phi))
        self.phi += dt * rhs
        return self.phi


# ---------------------------------------------------------------------------
#  Grain Growth (Multi-Phase Field)
# ---------------------------------------------------------------------------

class MultiPhaseFieldGrainGrowth:
    r"""
    Multi-phase-field model for polycrystalline grain growth (Fan-Chen, 1997).

    $$\frac{\partial\eta_i}{\partial t} = -L\left[\sum_{j\neq i}
      \left(\varepsilon_{ij}^2\eta_j^2\eta_i - \ldots\right)
      - \kappa_i\nabla^2\eta_i\right]$$

    Simplified form for $N_g$ grains:
    $$\frac{\partial\eta_i}{\partial t} = -L\left[-\eta_i + \eta_i^3
      + 2\gamma\eta_i\sum_{j\neq i}\eta_j^2 - \kappa\nabla^2\eta_i\right]$$

    Normal grain growth: $\langle R\rangle^2 = \langle R_0\rangle^2 + k\,t$.
    """

    def __init__(self, nx: int = 64, ny: int = 64,
                 n_grains: int = 4, dx: float = 1.0,
                 L: float = 1.0, kappa: float = 0.5,
                 gamma: float = 1.5) -> None:
        self.nx = nx
        self.ny = ny
        self.n_grains = n_grains
        self.dx = dx
        self.L = L
        self.kappa = kappa
        self.gamma = gamma

        self.eta = self._initialize_grains()

    def _initialize_grains(self) -> List[NDArray]:
        """Voronoi initialisation of grain order parameters."""
        eta = [np.zeros((self.nx, self.ny)) for _ in range(self.n_grains)]
        centers = np.random.randint(0, self.nx, (self.n_grains, 2))
        x, y = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing='ij')

        dist = np.full((self.nx, self.ny), np.inf)
        labels = np.zeros((self.nx, self.ny), dtype=int)

        for g in range(self.n_grains):
            d = np.sqrt((x - centers[g, 0])**2 + (y - centers[g, 1])**2)
            mask = d < dist
            dist[mask] = d[mask]
            labels[mask] = g

        for g in range(self.n_grains):
            eta[g][labels == g] = 1.0

        return eta

    def laplacian(self, f: NDArray) -> NDArray:
        return ((np.roll(f, 1, 0) + np.roll(f, -1, 0)
                + np.roll(f, 1, 1) + np.roll(f, -1, 1)
                - 4 * f) / self.dx**2)

    def step(self, dt: float = 0.01) -> List[NDArray]:
        """Advance all order parameters by one timestep."""
        sum_eta_sq = sum(e**2 for e in self.eta)

        for i in range(self.n_grains):
            others_sq = sum_eta_sq - self.eta[i]**2
            rhs = (-self.eta[i] + self.eta[i]**3
                   + 2 * self.gamma * self.eta[i] * others_sq
                   - self.kappa * self.laplacian(self.eta[i]))
            self.eta[i] -= dt * self.L * rhs

        return self.eta

    def grain_count(self, threshold: float = 0.5) -> int:
        """Count active grains."""
        return sum(1 for e in self.eta if np.max(e) > threshold)

    def mean_grain_area(self) -> float:
        """Average grain area in grid units."""
        total = self.nx * self.ny
        n = max(self.grain_count(), 1)
        return total / n


# ---------------------------------------------------------------------------
#  Classical Nucleation Theory
# ---------------------------------------------------------------------------

class ClassicalNucleation:
    r"""
    Classical nucleation theory (CNT).

    Free energy of nucleus:
    $$\Delta G(r) = -\frac{4}{3}\pi r^3 \Delta g_v + 4\pi r^2 \gamma$$

    Critical radius: $r^* = \frac{2\gamma}{\Delta g_v}$

    Nucleation barrier: $\Delta G^* = \frac{16\pi\gamma^3}{3(\Delta g_v)^2}$

    Nucleation rate:
    $$J = J_0\exp\left(-\frac{\Delta G^*}{k_BT}\right)$$
    """

    def __init__(self, gamma: float = 0.5, delta_gv: float = 0.1) -> None:
        """
        gamma: interface energy (J/m²).
        delta_gv: volume free energy driving force (J/m³).
        """
        self.gamma = gamma
        self.delta_gv = delta_gv

    def critical_radius(self) -> float:
        """r* = 2γ/Δg_v (metres)."""
        return 2 * self.gamma / self.delta_gv

    def nucleation_barrier(self) -> float:
        """ΔG* = 16πγ³/(3Δg_v²) (Joules)."""
        return 16 * math.pi * self.gamma**3 / (3 * self.delta_gv**2)

    def nucleation_rate(self, T: float, J0: float = 1e30) -> float:
        """J = J₀ exp(−ΔG*/(k_BT)) (m⁻³ s⁻¹)."""
        kBT = 1.381e-23 * T
        dG_star = self.nucleation_barrier()
        return J0 * math.exp(-dG_star / kBT)

    def free_energy_profile(self, r_max: float = None,
                               n_pts: int = 200) -> Tuple[NDArray, NDArray]:
        """ΔG(r) curve."""
        if r_max is None:
            r_max = 5 * self.critical_radius()
        r = np.linspace(0, r_max, n_pts)
        dG = -4 / 3 * math.pi * r**3 * self.delta_gv + 4 * math.pi * r**2 * self.gamma
        return r, dG
