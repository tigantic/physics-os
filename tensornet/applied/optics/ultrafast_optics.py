"""
Ultrafast Optics — pulse propagation, chirped pulses, nonlinear envelope equation,
self-phase modulation, autocorrelation, pulse compression.

Domain IV.4 — NEW.
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

C_LIGHT: float = 2.998e8    # m/s
EPS_0: float = 8.854e-12    # F/m
H_PLANCK: float = 6.626e-34 # J·s


# ---------------------------------------------------------------------------
#  Ultrafast Pulse Representations
# ---------------------------------------------------------------------------

class UltrafastPulse:
    r"""
    Temporal and spectral representation of ultrafast laser pulses.

    Gaussian pulse:
    $$E(t) = E_0\exp\left(-\frac{t^2}{2\tau_p^2}\right)
      \exp\left(-i\omega_0 t + i\frac{C t^2}{2\tau_p^2}\right)$$

    Transform-limited duration:
    $$\tau_{\text{TL}} = \frac{2\ln 2}{\pi\Delta\nu}$$
    (Gaussian TBP = 0.4413)

    Chirped pulse: $\tau_p(\phi_2) = \tau_0\sqrt{1+(4\ln 2\,\phi_2/\tau_0^2)^2}$
    where $\phi_2$ = group delay dispersion (GDD in fs²).
    """

    def __init__(self, tau_fwhm: float = 30.0, wavelength: float = 800e-9,
                 chirp: float = 0.0, energy: float = 1e-3) -> None:
        """
        tau_fwhm: pulse duration FWHM (fs).
        wavelength: central wavelength (m).
        chirp: C parameter (dimensionless chirp).
        energy: pulse energy (J).
        """
        self.tau_fwhm = tau_fwhm * 1e-15  # s
        self.tau_p = self.tau_fwhm / (2 * math.sqrt(math.log(2)))  # 1/e² half-width
        self.omega_0 = 2 * math.pi * C_LIGHT / wavelength
        self.wavelength = wavelength
        self.chirp = chirp
        self.energy = energy

    def temporal_field(self, t: NDArray) -> NDArray:
        """E(t) — complex electric field envelope."""
        E0 = math.sqrt(2 * self.energy / (self.tau_p * math.sqrt(math.pi)))
        return (E0 * np.exp(-t**2 / (2 * self.tau_p**2))
                * np.exp(-1j * self.omega_0 * t
                         + 1j * self.chirp * t**2 / (2 * self.tau_p**2)))

    def spectral_field(self, t: NDArray) -> NDArray:
        """Ẽ(ω) via FFT."""
        E_t = self.temporal_field(t)
        E_w = np.fft.fftshift(np.fft.fft(E_t))
        return E_w

    def peak_power(self) -> float:
        """P_peak = 0.94 × E / τ_FWHM (for Gaussian)."""
        return 0.94 * self.energy / self.tau_fwhm

    def peak_intensity(self, beam_radius: float = 50e-6) -> float:
        """I_peak = P_peak / (π w²) (W/m²)."""
        return self.peak_power() / (math.pi * beam_radius**2)

    def tbp(self) -> float:
        """Time-bandwidth product: Δt × Δν.

        TL Gaussian = 0.4413.
        """
        delta_t = self.tau_fwhm
        delta_nu = 1 / (math.pi * self.tau_p) * math.sqrt(1 + self.chirp**2)
        return delta_t * delta_nu

    def chirped_duration(self, gdd: float) -> float:
        """Pulse duration after GDD (fs²).

        τ_out = τ₀ √(1 + (4 ln2 × GDD / τ₀²)²)
        """
        tau0 = self.tau_fwhm
        gdd_s = gdd * 1e-30  # fs² → s²
        return tau0 * math.sqrt(1 + (4 * math.log(2) * gdd_s / tau0**2)**2) / 1e-15  # fs


# ---------------------------------------------------------------------------
#  Nonlinear Pulse Propagation (Split-Step Fourier)
# ---------------------------------------------------------------------------

class SplitStepFourier:
    r"""
    Split-step Fourier method for nonlinear Schrödinger equation.

    $$i\frac{\partial A}{\partial z} + \frac{\beta_2}{2}\frac{\partial^2 A}{\partial t^2}
      = -\gamma|A|^2 A$$

    where $\beta_2$ = GVD (fs²/mm), $\gamma = n_2\omega_0/(cA_{\text{eff}})$
    = nonlinear coefficient (1/(W·m)).

    Split-step: $A(z+h) = e^{i\hat{N}h}\mathcal{F}^{-1}[e^{i\hat{D}h}\tilde{A}]$

    Soliton condition: $N^2 = \gamma P_0 T_0^2/|\beta_2| = 1$
    """

    def __init__(self, n_t: int = 2**14, t_window: float = 10.0,
                 beta2: float = -20.0, gamma: float = 1.0,
                 n_z: int = 1000, z_max: float = 1.0) -> None:
        """
        t_window: temporal window (ps).
        beta2: GVD (ps²/km).
        gamma: nonlinear coefficient (1/(W·km)).
        z_max: propagation distance (km).
        """
        self.n_t = n_t
        self.dt = t_window / n_t  # ps
        self.t = np.linspace(-t_window / 2, t_window / 2, n_t)
        self.dz = z_max / n_z
        self.n_z = n_z
        self.beta2 = beta2
        self.gamma = gamma

        # Frequency grid
        self.omega = 2 * math.pi * np.fft.fftfreq(n_t, self.dt)

    def dispersion_operator(self) -> NDArray:
        """exp(i D̂ dz/2) where D̂ = (β₂/2) ω²."""
        return np.exp(1j * 0.5 * self.beta2 * self.omega**2 * self.dz / 2)

    def propagate(self, A0: NDArray) -> Tuple[NDArray, NDArray]:
        """Propagate pulse A0(t) through fibre.

        Returns (z_array, A_out).
        """
        A = A0.copy().astype(complex)
        D_half = self.dispersion_operator()
        A_history = np.zeros((self.n_z + 1, self.n_t), dtype=complex)
        A_history[0] = A

        for step in range(self.n_z):
            # Half-step dispersion
            A = np.fft.ifft(D_half * np.fft.fft(A))
            # Full-step nonlinearity
            A = A * np.exp(1j * self.gamma * np.abs(A)**2 * self.dz)
            # Half-step dispersion
            A = np.fft.ifft(D_half * np.fft.fft(A))

            A_history[step + 1] = A

        z = np.linspace(0, self.dz * self.n_z, self.n_z + 1)
        return z, A_history[-1]

    def soliton_number(self, P0: float, T0: float) -> float:
        """N² = γ P₀ T₀² / |β₂|."""
        return math.sqrt(abs(self.gamma * P0 * T0**2 / self.beta2))

    def soliton_input(self, P0: float = 1.0, T0: float = 0.1) -> NDArray:
        """Fundamental soliton: A(t) = √P₀ sech(t/T₀)."""
        return math.sqrt(P0) / np.cosh(self.t / T0)


# ---------------------------------------------------------------------------
#  Self-Phase Modulation
# ---------------------------------------------------------------------------

class SelfPhaseModulation:
    r"""
    Self-phase modulation (SPM) in a Kerr medium.

    Phase shift:
    $$\phi_{\text{SPM}}(t) = -\gamma P(t) L_{\text{eff}}$$

    where $L_{\text{eff}} = (1 - e^{-\alpha L})/\alpha$ and
    $P(t) = |A(t)|^2$ is instantaneous power.

    Maximum SPM phase: $\phi_{\max} = \gamma P_0 L_{\text{eff}}$

    Spectral broadening: new frequencies generated on pulse edges.
    """

    def __init__(self, gamma: float = 1.0, L: float = 1.0,
                 alpha: float = 0.0) -> None:
        """
        gamma: nonlinear coefficient (1/(W·km)).
        L: fibre length (km).
        alpha: attenuation (1/km).
        """
        self.gamma = gamma
        self.L = L
        self.alpha = alpha
        self.L_eff = (1 - math.exp(-alpha * L)) / alpha if alpha > 0 else L

    def nlphase(self, power: NDArray) -> NDArray:
        """φ_SPM(t) = γ P(t) L_eff."""
        return self.gamma * power * self.L_eff

    def max_phase(self, P0: float) -> float:
        """Maximum nonlinear phase (radians)."""
        return self.gamma * P0 * self.L_eff

    def broadened_spectrum(self, E_t: NDArray, dt: float) -> Tuple[NDArray, NDArray]:
        """Apply SPM and compute broadened spectrum.

        E_t: electric field envelope.
        dt: time step.
        """
        power = np.abs(E_t)**2
        phase = self.nlphase(power)
        E_out = E_t * np.exp(1j * phase)

        n = len(E_out)
        freq = np.fft.fftshift(np.fft.fftfreq(n, dt))
        spectrum = np.abs(np.fft.fftshift(np.fft.fft(E_out)))**2

        return freq, spectrum


# ---------------------------------------------------------------------------
#  Autocorrelation
# ---------------------------------------------------------------------------

class Autocorrelation:
    r"""
    Intensity autocorrelation for ultrafast pulse characterisation.

    $$A^{(2)}(\tau) = \int_{-\infty}^{\infty} I(t)I(t-\tau)\,dt$$

    Deconvolution factors:
    - Gaussian: τ_pulse = τ_AC / √2
    - Sech²: τ_pulse = τ_AC / 1.543

    Interferometric (fringe-resolved) autocorrelation:
    $$G^{(2)}(\tau) = \int |E(t)+E(t-\tau)|^4\,dt$$

    Peak-to-background ratio = 8:1 for ideal pulses.
    """

    @staticmethod
    def intensity_autocorrelation(I: NDArray, dt: float) -> Tuple[NDArray, NDArray]:
        """Compute intensity autocorrelation via FFT convolution."""
        n = len(I)
        I_fft = np.fft.fft(I, 2 * n)
        AC = np.real(np.fft.ifft(I_fft * np.conj(I_fft)))[:n]
        AC /= np.max(AC)
        tau = np.arange(n) * dt
        tau = tau - tau[n // 2]
        return tau[:n], AC

    @staticmethod
    def deconvolve_gaussian(tau_ac_fwhm: float) -> float:
        """τ_pulse = τ_AC / √2 (for Gaussian pulses)."""
        return tau_ac_fwhm / math.sqrt(2)

    @staticmethod
    def deconvolve_sech2(tau_ac_fwhm: float) -> float:
        """τ_pulse = τ_AC / 1.543 (for sech² pulses)."""
        return tau_ac_fwhm / 1.543

    @staticmethod
    def interferometric_autocorrelation(E: NDArray, dt: float) -> Tuple[NDArray, NDArray]:
        """Fringe-resolved autocorrelation G⁽²⁾(τ).

        G(τ) = ∫|E(t) + E(t−τ)|⁴ dt
        """
        n = len(E)
        n_tau = n // 2
        G = np.zeros(n_tau)

        for j in range(n_tau):
            E_shifted = np.roll(E, j)
            G[j] = float(np.sum(np.abs(E + E_shifted)**4)) * dt

        G /= np.max(G) if np.max(G) > 0 else 1
        tau = np.arange(n_tau) * dt
        return tau, G
