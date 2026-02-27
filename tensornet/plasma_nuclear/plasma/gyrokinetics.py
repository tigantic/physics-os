"""
Gyrokinetics — 5D gyrokinetic Vlasov equation, ITG/TEM/ETG instabilities,
tokamak microinstability growth rates.

Domain XI.4 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Gyrokinetic Plasma Parameters
# ---------------------------------------------------------------------------

@dataclass
class GKParameters:
    r"""
    Gyrokinetic ordering parameters for tokamak core turbulence.

    Small parameter: $\rho^* = \rho_i / a \ll 1$

    where $\rho_i = v_{ti}/\Omega_i$ is the ion gyroradius,
    a is the minor radius, $v_{ti} = \sqrt{T_i/m_i}$.

    Key dimensionless numbers:
    - $k_\perp\rho_i \sim 1$ (resolved scale)
    - $\omega / \Omega_i \sim \rho^*$ (low-frequency)
    - $k_\parallel / k_\perp \sim \rho^*$ (field-aligned)
    """
    B0: float = 3.0          # T (magnetic field)
    n0: float = 3e19         # m^-3 (density)
    T_i: float = 5e3         # eV (ion temperature)
    T_e: float = 5e3         # eV (electron temperature)
    m_i: float = 3.34e-27    # kg (deuterium mass)
    m_e: float = 9.11e-31    # kg
    q_i: float = 1.602e-19   # C
    R0: float = 1.7          # m (major radius)
    a: float = 0.5           # m (minor radius)
    q_safety: float = 1.5    # safety factor
    s_hat: float = 0.8       # magnetic shear

    # Density / temperature gradients
    R_Ln: float = 3.0        # R/L_n (inverse density gradient length)
    R_LTi: float = 9.0       # R/L_Ti (ion temperature gradient)
    R_LTe: float = 9.0       # R/L_Te (electron temperature gradient)

    @property
    def v_ti(self) -> float:
        """Ion thermal velocity: v_ti = sqrt(T_i/m_i)."""
        return math.sqrt(self.T_i * self.q_i / self.m_i)

    @property
    def v_te(self) -> float:
        """Electron thermal velocity."""
        return math.sqrt(self.T_e * self.q_i / self.m_e)

    @property
    def omega_ci(self) -> float:
        """Ion cyclotron frequency: Ω_i = qB/m."""
        return self.q_i * self.B0 / self.m_i

    @property
    def rho_i(self) -> float:
        """Ion gyroradius: ρ_i = v_ti / Ω_i."""
        return self.v_ti / self.omega_ci

    @property
    def rho_s(self) -> float:
        """Sound gyroradius: ρ_s = c_s / Ω_i."""
        cs = math.sqrt(self.T_e * self.q_i / self.m_i)
        return cs / self.omega_ci

    @property
    def rho_star(self) -> float:
        """ρ* = ρ_i / a."""
        return self.rho_i / self.a

    @property
    def beta(self) -> float:
        """Plasma beta: β = 2μ₀nT/B²."""
        mu0 = 4 * math.pi * 1e-7
        return 2 * mu0 * self.n0 * self.T_i * self.q_i / self.B0**2

    @property
    def omega_star_i(self) -> float:
        r"""Ion diamagnetic frequency: ω*_i = k_y ρ_i v_ti R_Ln / R0."""
        return self.v_ti * self.R_Ln / self.R0

    @property
    def omega_star_e(self) -> float:
        return self.v_te * self.R_Ln / self.R0


# ---------------------------------------------------------------------------
#  ITG Dispersion Relation
# ---------------------------------------------------------------------------

class ITGDispersion:
    r"""
    Ion Temperature Gradient (ITG) mode — the dominant electrostatic
    microinstability in tokamak ion scale turbulence.

    Fluid ITG dispersion relation (simplified):
    $$\omega(\omega-\omega_{*i}^T) = -k_\parallel^2 c_s^2 \frac{\tau}{1+\tau}$$

    where $\omega_{*i}^T = \omega_{*i}[1+\eta_i(v^2/(2v_{ti}^2)-3/2)]$,
    $\eta_i = L_n/L_{Ti}$, $\tau = T_e/T_i$.

    Critical gradient: $\eta_{i,\text{crit}} \approx 2/3$ for flat density.
    """

    def __init__(self, params: Optional[GKParameters] = None) -> None:
        self.p = params or GKParameters()

    def eta_i(self) -> float:
        """η_i = L_n/L_Ti = R_LTi/R_Ln."""
        return self.p.R_LTi / (self.p.R_Ln + 1e-10)

    def growth_rate_fluid(self, ky_rho: float = 0.3,
                             kpar_norm: float = 0.1) -> complex:
        r"""Fluid ITG growth rate.

        Dispersion: ω(ω − ω*T) = −k‖²c_s²τ/(1+τ)

        ky_rho: k_y ρ_i.
        kpar_norm: k_‖ qR (normalised).
        Returns complex frequency: ω = ω_r + iγ.
        """
        tau = self.p.T_e / self.p.T_i
        eta = self.eta_i()
        cs = math.sqrt(self.p.T_e * self.p.q_i / self.p.m_i)

        omega_star = ky_rho * cs * self.p.R_Ln / (self.p.R0 * self.p.rho_i)
        omega_star_T = omega_star * (1 + eta)

        k_par = kpar_norm / (self.p.q_safety * self.p.R0)
        rhs = -k_par**2 * cs**2 * tau / (1 + tau)

        # Quadratic: ω² - ω*T ω - rhs = 0
        discriminant = omega_star_T**2 + 4 * rhs
        if discriminant < 0:
            sqrt_disc = 1j * math.sqrt(-discriminant)
        else:
            sqrt_disc = math.sqrt(discriminant)

        omega = 0.5 * (omega_star_T + sqrt_disc)
        return complex(omega)

    def critical_gradient(self) -> float:
        r"""Critical η_i for ITG onset (flat density limit).

        η_{i,crit} ≈ 2/3 (Romanelli 1989).
        More generally: η_{i,crit} = max(2/3, 10/(R/L_n)).
        """
        return max(2.0 / 3.0, 10.0 / self.p.R_Ln)

    def scan_growth_rates(self, ky_rho_range: Optional[NDArray] = None,
                              kpar_norm: float = 0.1) -> Dict[str, NDArray]:
        """Scan growth rate vs k_y ρ_i."""
        if ky_rho_range is None:
            ky_rho_range = np.linspace(0.05, 2.0, 40)

        gammas = np.zeros(len(ky_rho_range))
        omegas = np.zeros(len(ky_rho_range))

        for i, ky in enumerate(ky_rho_range):
            w = self.growth_rate_fluid(ky, kpar_norm)
            omegas[i] = w.real
            gammas[i] = w.imag

        return {'ky_rho': ky_rho_range, 'gamma': gammas, 'omega_r': omegas}


# ---------------------------------------------------------------------------
#  TEM Dispersion
# ---------------------------------------------------------------------------

class TEMDispersion:
    r"""
    Trapped Electron Mode (TEM) — driven by electron temperature
    or density gradients via trapped electron dynamics.

    Bounce-averaged drift-kinetic response:
    $$\omega - \omega_{de} = \omega_{be}\sqrt{\epsilon}$$

    where $\omega_{de}$ = precessional drift frequency,
    $\omega_{be}$ = bounce frequency, ε = inverse aspect ratio.

    TEM is dominant when $\eta_e > 2/3$ and k_y ρ_s ~ 0.1–0.5.
    """

    def __init__(self, params: Optional[GKParameters] = None) -> None:
        self.p = params or GKParameters()

    def trapped_fraction(self) -> float:
        """f_t ≈ √(2ε/(1+ε)) where ε = a/R."""
        eps = self.p.a / self.p.R0
        return math.sqrt(2 * eps / (1 + eps))

    def precession_drift_frequency(self, ky_rho: float = 0.2) -> float:
        """ω_d = k_y ρ_s c_s v_d / (R0 Ω_i)."""
        cs = math.sqrt(self.p.T_e * self.p.q_i / self.p.m_i)
        eps = self.p.a / self.p.R0
        omega_d = ky_rho * cs * eps / (self.p.R0 * self.p.q_safety)
        return omega_d

    def growth_rate_estimate(self, ky_rho: float = 0.2) -> float:
        """Simplified TEM growth rate.

        γ ∝ f_t √ε ω_d (η_e − 2/3).
        """
        eta_e = self.p.R_LTe / (self.p.R_Ln + 1e-10)
        ft = self.trapped_fraction()
        eps = self.p.a / self.p.R0
        omega_d = self.precession_drift_frequency(ky_rho)

        if eta_e < 2.0 / 3.0:
            return 0.0

        return ft * math.sqrt(eps) * omega_d * (eta_e - 2.0 / 3.0)


# ---------------------------------------------------------------------------
#  ETG Dispersion
# ---------------------------------------------------------------------------

class ETGDispersion:
    r"""
    Electron Temperature Gradient (ETG) mode — electron-scale turbulence.

    Similar physics to ITG but at electron scale:
    $k_\perp\rho_e \sim 1$, $k_\perp\rho_i \gg 1$.

    Growth rate (fluid):
    $$\gamma_{\text{ETG}} \approx \frac{v_{te}}{qR}\sqrt{\frac{R}{L_{Te}}
      - \frac{R}{L_{Te,crit}}}$$

    for $R/L_{Te} > R/L_{Te,crit}$.
    """

    def __init__(self, params: Optional[GKParameters] = None) -> None:
        self.p = params or GKParameters()

    @property
    def rho_e(self) -> float:
        """Electron gyroradius."""
        omega_ce = self.p.q_i * self.p.B0 / self.p.m_e
        return self.p.v_te / omega_ce

    def growth_rate(self, ky_rho_e: float = 0.3) -> float:
        r"""ETG growth rate.

        γ = (v_te / qR) √(R/L_Te − R/L_Te,crit)
        with R/L_Te,crit ≈ 5.
        """
        R_LTe_crit = 5.0
        if self.p.R_LTe <= R_LTe_crit:
            return 0.0

        v_te = self.p.v_te
        return v_te / (self.p.q_safety * self.p.R0) * math.sqrt(
            self.p.R_LTe - R_LTe_crit)


# ---------------------------------------------------------------------------
#  Gyrokinetic Vlasov Solver (1D simplified)
# ---------------------------------------------------------------------------

class GyrokineticVlasov1D:
    r"""
    Simplified 1D electrostatic gyrokinetic Vlasov solver.

    $$\frac{\partial f}{\partial t} + v_\parallel\frac{\partial f}{\partial z}
      + \frac{q}{m}E_\parallel\frac{\partial f}{\partial v_\parallel} = 0$$

    Coupled to gyrokinetic Poisson:
    $$\frac{e^2 n_0}{T_e}\phi - \frac{n_0 e^2}{T_i}\langle\phi\rangle_\alpha
      = e\left(\int f_i d^3v - n_e\right)$$

    For flux-tube geometry with periodic parallel BC.
    """

    def __init__(self, nz: int = 64, nv: int = 64,
                 Lz: float = 2 * math.pi,
                 v_max: float = 5.0) -> None:
        self.nz = nz
        self.nv = nv
        self.dz = Lz / nz
        self.dv = 2 * v_max / nv

        self.z = np.linspace(0, Lz, nz, endpoint=False)
        self.v = np.linspace(-v_max, v_max, nv, endpoint=False)

        # Distribution function f(z, v) — init to Maxwellian
        ZZ, VV = np.meshgrid(self.z, self.v, indexing='ij')
        self.f = np.exp(-VV**2 / 2) / math.sqrt(2 * math.pi)

        # Add perturbation
        self.f *= 1 + 0.01 * np.cos(2 * math.pi * ZZ / Lz)

    def density(self) -> NDArray:
        """n(z) = ∫ f dv."""
        return np.sum(self.f, axis=1) * self.dv

    def solve_poisson(self, n: NDArray, tau: float = 1.0) -> NDArray:
        """Gyrokinetic quasi-neutrality in Fourier space.

        φ_k = (n_k − n0) / (k²ρ_s² + τ/(1+τ))
        """
        n0 = np.mean(n)
        n_hat = np.fft.fft(n - n0)
        k = 2 * math.pi * np.fft.fftfreq(self.nz, self.dz)

        rho_s = 1.0  # normalised
        denom = k**2 * rho_s**2 + tau / (1 + tau) + 1e-10
        phi_hat = n_hat / denom
        phi_hat[0] = 0

        return np.real(np.fft.ifft(phi_hat))

    def electric_field(self, phi: NDArray) -> NDArray:
        """E = −∂φ/∂z via spectral."""
        phi_hat = np.fft.fft(phi)
        k = 2 * math.pi * np.fft.fftfreq(self.nz, self.dz)
        E_hat = -1j * k * phi_hat
        return np.real(np.fft.ifft(E_hat))

    def step(self, dt: float) -> float:
        """Split-operator advection step.

        Returns total kinetic energy.
        """
        # z-advection: df/dt + v df/dz = 0
        f_hat = np.fft.fft(self.f, axis=0)
        kz = 2 * math.pi * np.fft.fftfreq(self.nz, self.dz)
        for j in range(self.nv):
            f_hat[:, j] *= np.exp(-1j * kz * self.v[j] * dt)
        self.f = np.real(np.fft.ifft(f_hat, axis=0))

        # Poisson + E-field
        n = self.density()
        phi = self.solve_poisson(n)
        E = self.electric_field(phi)

        # v-advection: df/dt + (q/m)E df/dv = 0
        f_hat_v = np.fft.fft(self.f, axis=1)
        kv = 2 * math.pi * np.fft.fftfreq(self.nv, self.dv)
        for i in range(self.nz):
            f_hat_v[i, :] *= np.exp(-1j * kv * E[i] * dt)
        self.f = np.real(np.fft.ifft(f_hat_v, axis=1))

        return 0.5 * float(np.sum(n * phi)) * self.dz
