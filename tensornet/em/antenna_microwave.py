"""
Antenna & Microwave Engineering — dipole antennas, antenna arrays, microstrip
patches, transmission lines, S-parameters.

Domain III.7 — NEW.
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

C_LIGHT: float = 2.998e8
EPS_0: float = 8.854e-12
MU_0: float = 4 * math.pi * 1e-7
ETA_0: float = math.sqrt(MU_0 / EPS_0)


# ---------------------------------------------------------------------------
#  Dipole Antenna
# ---------------------------------------------------------------------------

class DipoleAntenna:
    r"""
    Half-wave and arbitrary-length dipole antenna radiation.

    Far-field pattern (thin wire):
    $$E_\theta \propto \frac{\cos(kL\cos\theta/2) - \cos(kL/2)}{\sin\theta}$$

    Input impedance (half-wave): $Z_{in} \approx 73 + j42.5\,\Omega$

    Directivity: $D = \frac{4\pi U_{\max}}{P_{\text{rad}}}$

    Radiation resistance: $R_r = \frac{2P_{\text{rad}}}{|I_0|^2}$
    """

    def __init__(self, length: float = 0.5, freq: float = 300e6) -> None:
        """
        length: dipole length in wavelengths.
        freq: operating frequency (Hz).
        """
        self.L_lambda = length
        self.freq = freq
        self.wavelength = C_LIGHT / freq
        self.L = length * self.wavelength
        self.k = 2 * math.pi / self.wavelength

    def pattern(self, theta: NDArray) -> NDArray:
        """Normalised electric field pattern F(θ)."""
        kL2 = self.k * self.L / 2
        num = np.cos(kL2 * np.cos(theta)) - math.cos(kL2)
        den = np.sin(theta) + 1e-30
        F = np.abs(num / den)
        F /= np.max(F) + 1e-30
        return F

    def directivity(self, n_theta: int = 1000) -> float:
        """Directivity in dBi."""
        theta = np.linspace(1e-6, math.pi - 1e-6, n_theta)
        F = self.pattern(theta)
        U = F**2
        P_rad = 2 * math.pi * float(np.trapz(U * np.sin(theta), theta))
        D = 4 * math.pi * float(np.max(U)) / P_rad
        return 10 * math.log10(D)

    def radiation_resistance(self) -> float:
        """Radiation resistance R_r (Ω) via numerical integration."""
        n_theta = 1000
        theta = np.linspace(1e-6, math.pi - 1e-6, n_theta)
        kL2 = self.k * self.L / 2
        F = (np.cos(kL2 * np.cos(theta)) - math.cos(kL2)) / (np.sin(theta) + 1e-30)
        integrand = F**2 * np.sin(theta)
        P_norm = float(np.trapz(integrand, theta))
        return ETA_0 * P_norm / math.pi

    def input_impedance_halfwave(self) -> complex:
        """Approximate input impedance for half-wave dipole."""
        return complex(73.1, 42.5)

    def effective_area(self, directivity_linear: float) -> float:
        """A_e = D λ²/(4π) (m²)."""
        return directivity_linear * self.wavelength**2 / (4 * math.pi)


# ---------------------------------------------------------------------------
#  Uniform Linear Array
# ---------------------------------------------------------------------------

class UniformLinearArray:
    r"""
    Uniform linear array (ULA) of isotropic elements.

    Array factor:
    $$AF(\theta) = \sum_{n=0}^{N-1} w_n e^{jnkd\cos\theta}$$

    Uniform weights: $AF = \frac{\sin(N\psi/2)}{N\sin(\psi/2)}$
    where $\psi = kd\cos\theta + \beta$, $\beta$ = progressive phase.

    Beamwidth (broadside): $\Delta\theta \approx 0.886\lambda/(Nd)$
    """

    def __init__(self, n_elements: int = 8, spacing: float = 0.5,
                 freq: float = 1e9) -> None:
        """
        spacing: element spacing in wavelengths.
        """
        self.N = n_elements
        self.wavelength = C_LIGHT / freq
        self.d = spacing * self.wavelength
        self.k = 2 * math.pi / self.wavelength
        self.weights = np.ones(n_elements, dtype=complex)
        self.beta = 0.0  # progressive phase

    def set_scan_angle(self, theta_0: float) -> None:
        """Set beam steering angle (radians from broadside)."""
        self.beta = -self.k * self.d * math.cos(theta_0)

    def set_weights(self, weights: NDArray) -> None:
        """Set complex element weights."""
        self.weights = np.asarray(weights, dtype=complex)

    def array_factor(self, theta: NDArray) -> NDArray:
        """Compute array factor AF(θ)."""
        AF = np.zeros(len(theta), dtype=complex)
        for n in range(self.N):
            psi = self.k * self.d * np.cos(theta) + self.beta
            AF += self.weights[n] * np.exp(1j * n * psi)
        return np.abs(AF) / self.N

    def chebyshev_weights(self, sll_dB: float = -30.0) -> NDArray:
        """Dolph-Chebyshev weights for specified sidelobe level."""
        R = 10**(-sll_dB / 20)
        x0 = math.cosh(math.acosh(R) / (self.N - 1))
        weights = np.zeros(self.N)

        for n in range(self.N):
            for m in range(self.N):
                psi = math.pi * (2 * m + 1) / (2 * self.N)
                T = np.polynomial.chebyshev.chebval(x0 * math.cos(psi), [0] * (self.N - 1) + [1])
                weights[n] += T * math.cos(2 * math.pi * n * m / self.N)
            weights[n] /= self.N

        self.weights = weights / np.max(np.abs(weights))
        return self.weights

    def beamwidth(self) -> float:
        """3 dB beamwidth (degrees)."""
        return math.degrees(0.886 * self.wavelength / (self.N * self.d))

    def grating_lobe_condition(self) -> float:
        """Maximum spacing (in λ) to avoid grating lobes: d < λ/(1+|sin θ₀|)."""
        return 1.0 / (1 + abs(math.sin(math.acos(-self.beta / (self.k * self.d + 1e-30)))) + 1e-30)


# ---------------------------------------------------------------------------
#  Microstrip Patch Antenna
# ---------------------------------------------------------------------------

class MicrostripPatch:
    r"""
    Rectangular microstrip patch antenna design.

    Resonant frequency (dominant TM₁₀ mode):
    $$f_r = \frac{c}{2L_{\text{eff}}\sqrt{\epsilon_r}}$$

    Effective length: $L_{\text{eff}} = L + 2\Delta L$

    Fringing extension:
    $$\Delta L = 0.412h\frac{(\epsilon_{\text{eff}}+0.3)(W/h+0.264)}{(\epsilon_{\text{eff}}-0.258)(W/h+0.8)}$$

    Effective permittivity:
    $$\epsilon_{\text{eff}} = \frac{\epsilon_r+1}{2} + \frac{\epsilon_r-1}{2}\frac{1}{\sqrt{1+12h/W}}$$
    """

    def __init__(self, eps_r: float = 4.4, h: float = 1.6e-3,
                 freq: float = 2.4e9) -> None:
        """
        eps_r: substrate relative permittivity.
        h: substrate height (m).
        freq: target resonant frequency (Hz).
        """
        self.eps_r = eps_r
        self.h = h
        self.freq = freq
        self.wavelength = C_LIGHT / freq

    def patch_width(self) -> float:
        """Optimal patch width W (m)."""
        return C_LIGHT / (2 * self.freq) * math.sqrt(2 / (self.eps_r + 1))

    def effective_permittivity(self) -> float:
        """ε_eff for microstrip."""
        W = self.patch_width()
        return ((self.eps_r + 1) / 2
                + (self.eps_r - 1) / 2 / math.sqrt(1 + 12 * self.h / W))

    def fringing_extension(self) -> float:
        """ΔL — fringing field extension (m)."""
        W = self.patch_width()
        eps_eff = self.effective_permittivity()
        return (0.412 * self.h
                * (eps_eff + 0.3) * (W / self.h + 0.264)
                / ((eps_eff - 0.258) * (W / self.h + 0.8)))

    def patch_length(self) -> float:
        """Physical patch length L (m)."""
        eps_eff = self.effective_permittivity()
        L_eff = C_LIGHT / (2 * self.freq * math.sqrt(eps_eff))
        return L_eff - 2 * self.fringing_extension()

    def input_impedance(self) -> float:
        """Edge input impedance Z_in ≈ 90 ε_r²/(ε_r−1) × (L/W)² (Ω)."""
        W = self.patch_width()
        L = self.patch_length()
        return 90 * self.eps_r**2 / (self.eps_r - 1) * (L / W)**2

    def bandwidth(self, Q: float = 0.0) -> float:
        """Approximate bandwidth (fractional).

        BW ≈ (ε_r − 1)/(ε_r²) × (h/λ₀) × 3.77
        """
        return (self.eps_r - 1) / self.eps_r**2 * (self.h / self.wavelength) * 3.77

    def design_summary(self) -> Dict[str, float]:
        """Complete design parameters."""
        return {
            'width_mm': self.patch_width() * 1e3,
            'length_mm': self.patch_length() * 1e3,
            'eps_eff': self.effective_permittivity(),
            'delta_L_mm': self.fringing_extension() * 1e3,
            'Z_in_ohm': self.input_impedance(),
            'bandwidth_pct': self.bandwidth() * 100,
        }


# ---------------------------------------------------------------------------
#  Transmission Line
# ---------------------------------------------------------------------------

class TransmissionLine:
    r"""
    Transmission line analysis.

    Characteristic impedance: $Z_0 = \sqrt{(R+j\omega L)/(G+j\omega C)}$

    Propagation constant: $\gamma = \sqrt{(R+j\omega L)(G+j\omega C)}$

    Input impedance: $Z_{in} = Z_0\frac{Z_L + Z_0\tanh(\gamma\ell)}{Z_0 + Z_L\tanh(\gamma\ell)}$

    VSWR: $\text{VSWR} = \frac{1+|\Gamma|}{1-|\Gamma|}$

    Smith chart: $\Gamma = \frac{Z_L - Z_0}{Z_L + Z_0}$
    """

    def __init__(self, Z0: float = 50.0, gamma: complex = 0.0,
                 length: float = 1.0) -> None:
        """
        Z0: characteristic impedance (Ω).
        gamma: propagation constant (1/m).
        length: line length (m).
        """
        self.Z0 = Z0
        self.gamma = gamma
        self.length = length

    @classmethod
    def lossless(cls, Z0: float, freq: float, eps_eff: float = 1.0,
                    length: float = 1.0) -> 'TransmissionLine':
        """Create lossless line from Z0 and frequency."""
        beta = 2 * math.pi * freq * math.sqrt(eps_eff) / C_LIGHT
        return cls(Z0=Z0, gamma=complex(0, beta), length=length)

    def input_impedance(self, ZL: complex) -> complex:
        """Z_in = Z0 (ZL + Z0 tanh(γl))/(Z0 + ZL tanh(γl))."""
        gl = self.gamma * self.length
        tanh_gl = np.tanh(gl)
        return self.Z0 * (ZL + self.Z0 * tanh_gl) / (self.Z0 + ZL * tanh_gl)

    def reflection_coefficient(self, ZL: complex) -> complex:
        """Γ = (ZL − Z0)/(ZL + Z0)."""
        return (ZL - self.Z0) / (ZL + self.Z0)

    def vswr(self, ZL: complex) -> float:
        """Voltage standing wave ratio."""
        G = abs(self.reflection_coefficient(ZL))
        if G >= 1:
            return float('inf')
        return (1 + G) / (1 - G)

    def return_loss(self, ZL: complex) -> float:
        """Return loss (dB): −20 log₁₀|Γ|."""
        G = abs(self.reflection_coefficient(ZL))
        if G < 1e-30:
            return 100.0
        return -20 * math.log10(G)

    def quarter_wave_transformer(self, Z_in: float,
                                    Z_out: float) -> float:
        """Z0 for quarter-wave matching: Z0 = √(Z_in × Z_out)."""
        return math.sqrt(Z_in * Z_out)
