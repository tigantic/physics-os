"""
Phase-Field Crystal (PFC) Model
=================================

Continuum model for crystalline microstructure evolution on diffusive
time scales.  The density field is periodic (crystal) and the free
energy is minimised by the crystal structure.

Free energy functional (Swift-Hohenberg type):
    F[ψ] = ∫ [ ψ/2 (r + (q₀² + ∇²)²) ψ + ψ⁴/4 ] dx

where ψ is the dimensionless density deviation, r < 0 drives the
uniform → crystal instability, and q₀ = 2π/a is the crystal
wave number.

Dynamics (conserved, Model B):
    ∂ψ/∂t = ∇² (δF/δψ) = ∇² [ (r + (q₀² + ∇²)²) ψ + ψ³ ]

Spectral implementation:
    ψ̂_t = -k² [ (r + (q₀² - k²)²) ψ̂ + F̂{ψ³} ]

    with semi-implicit:  linear part treated implicitly, nonlinear
    explicitly.

References:
    [1] Elder & Grant, "Modeling Elastic and Plastic Deformations
        in Nonequilibrium Processing Using Phase Field Crystals",
        Phys. Rev. E 70, 2004.
    [2] Elder et al., "Phase-Field Crystal Modeling and Classical
        Density Functional Theory of Freezing",
        Phys. Rev. B 75, 2007.
    [3] Emmerich et al., "Phase-field-crystal models for condensed
        matter dynamics on atomic length and diffusive time scales",
        Adv. Phys. 61, 2012.

Domain III.4 — Phase-Field / Phase-Field Crystal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class PFCState:
    """
    Phase-field crystal state.

    Attributes:
        psi: Density deviation field — ``(Nx, Ny)``.
        t: Current simulation time.
    """
    psi: NDArray
    t: float = 0.0

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.psi.shape

    @property
    def mean_density(self) -> float:
        return float(np.mean(self.psi))


class PFCSolver:
    r"""
    2D Phase-Field Crystal solver (spectral, semi-implicit).

    The linear operator in Fourier space is:

    .. math::
        L(\mathbf{k}) = -|\mathbf{k}|^2
            \left[ r + (q_0^2 - |\mathbf{k}|^2)^2 \right]

    Semi-implicit time stepping:

    .. math::
        \hat\psi^{n+1} = \frac{\hat\psi^n
            - \Delta t \, |\mathbf{k}|^2 \widehat{(\psi^n)^3}}
            {1 - \Delta t \, L(\mathbf{k})}

    Parameters:
        Nx, Ny: Grid dimensions (should be powers of 2 for FFT).
        Lx, Ly: Physical domain size.
        r: Control parameter (r < 0 for crystal instability).
        q0: Crystal wave number (2π/a).
        dt: Time step.

    Example::

        solver = PFCSolver(Nx=256, Ny=256, Lx=64.0, Ly=64.0, r=-0.3)
        state = solver.initialise(psi_bar=-0.3, noise=0.01)
        for _ in range(5000):
            state = solver.step(state)
    """

    def __init__(
        self,
        Nx: int = 256,
        Ny: int = 256,
        Lx: float = 64.0,
        Ly: float = 64.0,
        r: float = -0.3,
        q0: float = 1.0,
        dt: float = 0.5,
    ) -> None:
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.r = r
        self.q0 = q0
        self.dt = dt

        # Wavenumber grids
        kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
        ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=Ly / Ny)
        self.KX, self.KY = np.meshgrid(kx, ky, indexing="ij")
        self.K2 = self.KX ** 2 + self.KY ** 2

        # Linear operator: L(k) = -k² [r + (q0² - k²)²]
        self.L = -self.K2 * (r + (q0 ** 2 - self.K2) ** 2)

        # Semi-implicit denominator: 1 - dt * L
        self._denom = 1.0 - dt * self.L

        self._step_count = 0

    def initialise(
        self,
        psi_bar: float = -0.3,
        noise: float = 0.01,
        seed: Optional[int] = None,
    ) -> PFCState:
        """Create initial state: uniform density + small random noise."""
        rng = np.random.default_rng(seed)
        psi = psi_bar + noise * rng.standard_normal((self.Nx, self.Ny))
        return PFCState(psi=psi, t=0.0)

    def initialise_hexagonal(
        self,
        psi_bar: float = -0.3,
        A: float = 0.2,
    ) -> PFCState:
        """Initialise with hexagonal crystal seed."""
        x = np.linspace(0, self.Lx, self.Nx, endpoint=False)
        y = np.linspace(0, self.Ly, self.Ny, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing="ij")
        q = self.q0

        # Three reciprocal lattice vectors for triangular lattice
        psi = psi_bar + A * (
            np.cos(q * X)
            + np.cos(q * (-0.5 * X + np.sqrt(3) / 2 * Y))
            + np.cos(q * (-0.5 * X - np.sqrt(3) / 2 * Y))
        )
        return PFCState(psi=psi, t=0.0)

    def chemical_potential(self, psi: NDArray) -> NDArray:
        r"""
        Compute chemical potential:
        :math:`\mu = \delta F/\delta\psi = (r + (q_0^2 + \nabla^2)^2)\psi + \psi^3`.

        Computed in Fourier space for the linear part.
        """
        psi_hat = np.fft.fft2(psi)
        # Linear part: [r + (q0² - k²)²] ψ̂
        lin = (self.r + (self.q0 ** 2 - self.K2) ** 2) * psi_hat
        mu_hat = lin + np.fft.fft2(psi ** 3)
        return np.real(np.fft.ifft2(mu_hat))

    def free_energy(self, state: PFCState) -> float:
        """Compute total free energy."""
        psi = state.psi
        psi_hat = np.fft.fft2(psi)

        # (q0² + ∇²) ψ in Fourier space: (q0² - k²) ψ̂
        op_psi_hat = (self.q0 ** 2 - self.K2) * psi_hat
        op_psi = np.real(np.fft.ifft2(op_psi_hat))

        # (q0² + ∇²)² ψ
        op2_psi = np.real(np.fft.ifft2(
            (self.q0 ** 2 - self.K2) ** 2 * psi_hat
        ))

        dx = self.Lx / self.Nx
        dy = self.Ly / self.Ny
        f_density = 0.5 * psi * (self.r * psi + op2_psi) + 0.25 * psi ** 4
        return float(np.sum(f_density) * dx * dy)

    def step(self, state: PFCState) -> PFCState:
        """Advance one semi-implicit time step."""
        psi = state.psi
        psi_hat = np.fft.fft2(psi)

        # Nonlinear term: ψ³
        nl_hat = np.fft.fft2(psi ** 3)

        # Semi-implicit update
        psi_hat_new = (psi_hat - self.dt * self.K2 * nl_hat) / self._denom
        psi_new = np.real(np.fft.ifft2(psi_hat_new))

        self._step_count += 1
        return PFCState(psi=psi_new, t=state.t + self.dt)

    def step_n(self, state: PFCState, n_steps: int) -> PFCState:
        for _ in range(n_steps):
            state = self.step(state)
        return state

    @property
    def steps(self) -> int:
        return self._step_count

    def structure_factor(self, state: PFCState) -> Tuple[NDArray, NDArray]:
        """Compute radially averaged structure factor S(k)."""
        psi_hat = np.fft.fft2(state.psi - state.mean_density)
        S2d = np.abs(psi_hat) ** 2 / (self.Nx * self.Ny)

        k_mag = np.sqrt(self.K2)
        k_max = float(np.max(k_mag)) / 2
        n_bins = min(self.Nx, self.Ny) // 2
        k_bins = np.linspace(0, k_max, n_bins + 1)
        k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
        S_avg = np.zeros(n_bins, dtype=np.float64)

        for i in range(n_bins):
            mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i + 1])
            if np.any(mask):
                S_avg[i] = np.mean(S2d[mask])
        return k_centers, S_avg
