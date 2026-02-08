"""
Laser Physics — rate equations, gain media, cavity modes, mode-locking,
Gaussian beams, saturation.

Domain IV.3 — NEW.
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

H_PLANCK: float = 6.626e-34    # J·s
C_LIGHT: float = 2.998e8       # m/s
K_B: float = 1.381e-23         # J/K


# ---------------------------------------------------------------------------
#  Laser Rate Equations (4-Level)
# ---------------------------------------------------------------------------

class FourLevelLaser:
    r"""
    Rate equations for a four-level laser system.

    $$\frac{dN_2}{dt} = R_p - \frac{N_2}{\tau_{21}} - \sigma_e\frac{c}{n}N_2\Phi$$
    $$\frac{d\Phi}{dt} = V_a\sigma_e\frac{c}{n}N_2\Phi - \frac{\Phi}{\tau_c} + \beta\frac{N_2}{\tau_{21}}$$

    where $N_2$ = upper level population, $\Phi$ = photon number,
    $R_p$ = pump rate, $\tau_c = 2Ln/(c(1-R_1R_2))$ = cavity lifetime,
    $\beta$ = spontaneous emission fraction.

    Threshold condition: $N_{2,\text{th}} = \frac{1}{\sigma_e c\tau_c/n}$

    Slope efficiency: $\eta_s = \frac{h\nu_L}{h\nu_p}\frac{T}{T+L_i}$
    """

    def __init__(self, sigma_e: float = 3e-19, tau_21: float = 230e-6,
                 n_refr: float = 1.82, L_cav: float = 0.1,
                 R1: float = 0.999, R2: float = 0.95,
                 V_mode: float = 1e-9) -> None:
        """
        sigma_e: emission cross section (cm²).
        tau_21: upper state lifetime (s).
        n_refr: refractive index of gain medium.
        L_cav: cavity length (m).
        R1, R2: mirror reflectivities.
        V_mode: mode volume (m³).
        """
        self.sigma_e = sigma_e * 1e-4  # cm² → m²
        self.tau_21 = tau_21
        self.n = n_refr
        self.L = L_cav
        self.R1 = R1
        self.R2 = R2
        self.V_mode = V_mode

        loss = -math.log(R1 * R2) / (2 * L_cav)
        self.tau_c = n_refr / (C_LIGHT * loss)
        self.beta = 1e-6  # spontaneous emission coupling

    def threshold_population(self) -> float:
        """N₂,th = n/(σ_e c τ_c) (m⁻³)."""
        return self.n / (self.sigma_e * C_LIGHT * self.tau_c)

    def threshold_pump_rate(self) -> float:
        """R_p,th = N₂,th / τ₂₁ (m⁻³ s⁻¹)."""
        return self.threshold_population() / self.tau_21

    def slope_efficiency(self, lambda_laser: float = 1064e-9,
                            lambda_pump: float = 808e-9,
                            internal_loss: float = 0.01) -> float:
        """η_s = (hν_L/hν_p) × T/(T + L_i)."""
        T = 1 - self.R2
        quantum_eff = lambda_pump / lambda_laser
        return quantum_eff * T / (T + internal_loss)

    def evolve(self, pump_rate: float, dt: float = 1e-9,
                  n_steps: int = 100000) -> Dict[str, NDArray]:
        """Integrate rate equations.

        pump_rate: R_p (m⁻³ s⁻¹).
        Returns time series of N₂ and Φ.
        """
        N2 = np.zeros(n_steps)
        Phi = np.zeros(n_steps)
        t = np.arange(n_steps) * dt

        N2[0] = 0.0
        Phi[0] = 1.0  # seed photon

        c_n = C_LIGHT / self.n
        for i in range(1, n_steps):
            stim = self.sigma_e * c_n * N2[i - 1] * Phi[i - 1]
            dN2 = pump_rate - N2[i - 1] / self.tau_21 - stim
            dPhi = (self.V_mode * self.sigma_e * c_n * N2[i - 1] * Phi[i - 1]
                    - Phi[i - 1] / self.tau_c
                    + self.beta * N2[i - 1] / self.tau_21)

            N2[i] = max(N2[i - 1] + dN2 * dt, 0)
            Phi[i] = max(Phi[i - 1] + dPhi * dt, 0)

        return {'t': t, 'N2': N2, 'Phi': Phi}


# ---------------------------------------------------------------------------
#  Gaussian Beam Optics
# ---------------------------------------------------------------------------

class GaussianBeam:
    r"""
    Paraxial Gaussian beam propagation.

    $$w(z) = w_0\sqrt{1+\left(\frac{z}{z_R}\right)^2}$$

    Rayleigh range: $z_R = \frac{\pi w_0^2 n}{\lambda}$

    Complex beam parameter: $\frac{1}{q} = \frac{1}{R(z)} - \frac{i\lambda}{\pi w(z)^2}$

    ABCD law: $q' = \frac{Aq + B}{Cq + D}$
    """

    def __init__(self, w0: float = 50e-6, wavelength: float = 1064e-9,
                 n: float = 1.0) -> None:
        """
        w0: beam waist radius (m).
        wavelength: λ (m).
        n: refractive index.
        """
        self.w0 = w0
        self.lam = wavelength
        self.n = n
        self.zR = math.pi * w0**2 * n / wavelength

    def waist(self, z: float) -> float:
        """w(z) — beam radius at distance z from waist."""
        return self.w0 * math.sqrt(1 + (z / self.zR)**2)

    def radius_of_curvature(self, z: float) -> float:
        """R(z) — wavefront radius of curvature."""
        if abs(z) < 1e-20:
            return float('inf')
        return z * (1 + (self.zR / z)**2)

    def gouy_phase(self, z: float) -> float:
        """Gouy phase ζ(z) = arctan(z/z_R)."""
        return math.atan(z / self.zR)

    def intensity_profile(self, r: NDArray, z: float = 0.0) -> NDArray:
        """I(r, z) = I₀ (w₀/w)² exp(−2r²/w²)."""
        w = self.waist(z)
        I0 = 1.0  # normalised
        return I0 * (self.w0 / w)**2 * np.exp(-2 * r**2 / w**2)

    def complex_beam_parameter(self, z: float) -> complex:
        """q(z) = z + i z_R."""
        return complex(z, self.zR)

    def abcd_transform(self, q: complex, A: float, B: float,
                          C: float, D: float) -> complex:
        """ABCD law: q' = (Aq + B)/(Cq + D)."""
        return (A * q + B) / (C * q + D)

    def thin_lens(self, q: complex, f: float) -> complex:
        """Transform through thin lens of focal length f."""
        return self.abcd_transform(q, 1, 0, -1 / f, 1)

    def free_space(self, q: complex, d: float) -> complex:
        """Free-space propagation by distance d."""
        return self.abcd_transform(q, 1, d, 0, 1)

    def divergence(self) -> float:
        """Far-field half-angle divergence θ = λ/(π w₀ n) (radians)."""
        return self.lam / (math.pi * self.w0 * self.n)

    def M_squared(self, w0_measured: float, divergence_measured: float) -> float:
        """M² = π w₀ θ / λ."""
        return math.pi * w0_measured * divergence_measured / self.lam


# ---------------------------------------------------------------------------
#  Optical Cavity (Fabry-Perot)
# ---------------------------------------------------------------------------

class FabryPerotCavity:
    r"""
    Optical Fabry-Perot cavity.

    Free spectral range: $\text{FSR} = \frac{c}{2nL}$

    Finesse: $\mathcal{F} = \frac{\pi\sqrt{R}}{1-R}$

    Linewidth: $\delta\nu = \frac{\text{FSR}}{\mathcal{F}}$

    Transmission (Airy function):
    $$T = \frac{1}{1 + F\sin^2(\delta/2)}$$
    $F = 4R/(1-R)^2$, $\delta = 4\pi n L\nu/c$.

    Stability: $0 \leq g_1 g_2 \leq 1$ where $g_i = 1 - L/R_i$.
    """

    def __init__(self, L: float = 0.1, R: float = 0.99,
                 n: float = 1.0) -> None:
        self.L = L
        self.R = R
        self.n = n

    def fsr(self) -> float:
        """Free spectral range (Hz)."""
        return C_LIGHT / (2 * self.n * self.L)

    def finesse(self) -> float:
        """Cavity finesse."""
        return math.pi * math.sqrt(self.R) / (1 - self.R)

    def linewidth(self) -> float:
        """Cavity linewidth δν (Hz)."""
        return self.fsr() / self.finesse()

    def quality_factor(self, nu: float) -> float:
        """Q = ν/δν."""
        return nu / self.linewidth()

    def photon_lifetime(self) -> float:
        """τ_ph = 1/(2π δν) (seconds)."""
        return 1 / (2 * math.pi * self.linewidth())

    def transmission(self, nu: NDArray) -> NDArray:
        """Airy transmission function T(ν)."""
        F_coeff = 4 * self.R / (1 - self.R)**2
        delta = 4 * math.pi * self.n * self.L * nu / C_LIGHT
        return 1 / (1 + F_coeff * np.sin(delta / 2)**2)

    def stability_parameter(self, R1: float, R2: float) -> float:
        """g₁g₂ for curved mirrors. Stable if 0 ≤ g₁g₂ ≤ 1."""
        g1 = 1 - self.L / R1 if abs(R1) > 1e-10 else 1.0
        g2 = 1 - self.L / R2 if abs(R2) > 1e-10 else 1.0
        return g1 * g2
