"""
Gravitational Wave Physics — waveform generation, inspiral dynamics,
post-Newtonian approximation, matched filtering, ringdown.

Domain XII.3 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Physical Constants
# ---------------------------------------------------------------------------

G_SI: float = 6.674e-11          # m³ kg⁻¹ s⁻²
C_SI: float = 2.998e8            # m/s
M_SUN_KG: float = 1.989e30       # kg
MPC_M: float = 3.086e22          # metres per Megaparsec
H_PLANCK: float = 6.626e-34      # J s


# ---------------------------------------------------------------------------
#  Post-Newtonian Inspiral Waveform
# ---------------------------------------------------------------------------

class PostNewtonianInspiral:
    r"""
    Post-Newtonian (PN) compact binary inspiral waveform.

    Orbital frequency evolution (0PN):
    $$\frac{d\omega}{dt} = \frac{96}{5}\frac{(G\mathcal{M})^{5/3}}{c^5}\omega^{11/3}$$

    Chirp mass: $\mathcal{M} = \frac{(m_1 m_2)^{3/5}}{(m_1+m_2)^{1/5}}$

    GW strain (quadrupole, leading order):
    $$h_+(t) = \frac{4(G\mathcal{M})^{5/3}}{c^4 D_L}(\pi f)^{2/3}\frac{1+\cos^2\iota}{2}\cos\Phi(t)$$
    $$h_\times(t) = \frac{4(G\mathcal{M})^{5/3}}{c^4 D_L}(\pi f)^{2/3}\cos\iota\sin\Phi(t)$$

    2.5PN energy loss includes spin-orbit and tail terms.
    """

    def __init__(self, m1: float = 30.0, m2: float = 30.0,
                 D_L: float = 410.0, iota: float = 0.0) -> None:
        """
        m1, m2: component masses in solar masses.
        D_L: luminosity distance in Mpc.
        iota: inclination angle (radians).
        """
        self.m1 = m1 * M_SUN_KG
        self.m2 = m2 * M_SUN_KG
        self.M = self.m1 + self.m2
        self.eta = self.m1 * self.m2 / self.M**2
        self.Mc = self.M * self.eta**(3 / 5)
        self.D_L = D_L * MPC_M
        self.iota = iota

    def chirp_mass_solar(self) -> float:
        """Chirp mass in solar masses."""
        return self.Mc / M_SUN_KG

    def isco_frequency(self) -> float:
        """Innermost stable circular orbit frequency (Schwarzschild).

        f_ISCO = c³ / (6^{3/2} π G M)
        """
        return C_SI**3 / (6**1.5 * math.pi * G_SI * self.M)

    def time_to_coalescence(self, f_start: float) -> float:
        """Time from f_start to coalescence (0PN).

        τ = (5/256)(πf)^{-8/3} (GMc/c³)^{-5/3}
        """
        v = (math.pi * G_SI * self.Mc * f_start / C_SI**3)**(1 / 3)
        return 5 / (256 * self.eta) * G_SI * self.M / C_SI**3 * v**(-8)

    def frequency_evolution(self, t: NDArray, f0: float = 10.0) -> NDArray:
        """f(t) — 0PN frequency evolution.

        f(t) = (1/π)(5/256)^{3/8} (GMc/c³)^{-5/8} τ^{-3/8}
        """
        tau_0 = self.time_to_coalescence(f0)
        tau = tau_0 - t
        tau = np.maximum(tau, 1e-10)

        prefactor = (5 / 256)**(3 / 8) / math.pi
        Mc_scaled = G_SI * self.Mc / C_SI**3
        return prefactor * Mc_scaled**(-5 / 8) * tau**(-3 / 8)

    def phase_evolution(self, t: NDArray, f0: float = 10.0) -> NDArray:
        """Φ(t) = 2π ∫ f(t′) dt′."""
        f_t = self.frequency_evolution(t, f0)
        dt = t[1] - t[0] if len(t) > 1 else 1e-4
        return 2 * math.pi * np.cumsum(f_t) * dt

    def strain_plus(self, t: NDArray, f0: float = 10.0) -> NDArray:
        """h_+(t) — plus polarisation."""
        f_t = self.frequency_evolution(t, f0)
        Phi = self.phase_evolution(t, f0)

        Mc_s = G_SI * self.Mc / C_SI**3
        A = 4 * C_SI / self.D_L * (math.pi * Mc_s)**(2 / 3)

        return A * f_t**(2 / 3) * (1 + math.cos(self.iota)**2) / 2 * np.cos(Phi)

    def strain_cross(self, t: NDArray, f0: float = 10.0) -> NDArray:
        """h_×(t) — cross polarisation."""
        f_t = self.frequency_evolution(t, f0)
        Phi = self.phase_evolution(t, f0)

        Mc_s = G_SI * self.Mc / C_SI**3
        A = 4 * C_SI / self.D_L * (math.pi * Mc_s)**(2 / 3)

        return A * f_t**(2 / 3) * math.cos(self.iota) * np.sin(Phi)


# ---------------------------------------------------------------------------
#  Quasi-Normal Mode Ringdown
# ---------------------------------------------------------------------------

class QuasiNormalRingdown:
    r"""
    Black hole quasi-normal mode (QNM) ringdown.

    $$h(t) = A e^{-t/\tau}\cos(2\pi f_{\text{QNM}} t + \phi_0)$$

    For Kerr BH (l=2, m=2, n=0):
    $$f_{\text{QNM}} \approx \frac{c^3}{2\pi G M_f}\left[1.5251 - 1.1568(1-a_f)^{0.1292}\right]$$
    $$Q = \frac{\pi f_{\text{QNM}} \tau}{1} \approx 0.7 + 1.4187(1-a_f)^{-0.4990}$$

    Final mass and spin from NR fits:
    $M_f \approx M(1 - 0.0559\eta^2 - 0.0857\eta)$
    $a_f \approx \sqrt{12}\eta - 3.871\eta^2 + 4.028\eta^3$
    """

    def __init__(self, M_total: float = 60.0, eta: float = 0.25) -> None:
        """
        M_total: total mass (solar masses).
        eta: symmetric mass ratio.
        """
        self.M_total = M_total
        self.eta = eta
        self.M_f, self.a_f = self._final_mass_spin()

    def _final_mass_spin(self) -> Tuple[float, float]:
        """NR-calibrated final mass and spin."""
        eta = self.eta
        M_f = self.M_total * (1 - 0.0559 * eta**2 - 0.0857 * eta)
        a_f = math.sqrt(12) * eta - 3.871 * eta**2 + 4.028 * eta**3
        a_f = min(a_f, 0.998)
        return M_f, a_f

    def qnm_frequency(self) -> float:
        """f_QNM for (l=2, m=2, n=0) mode (Hz)."""
        M_kg = self.M_f * M_SUN_KG
        f = (C_SI**3 / (2 * math.pi * G_SI * M_kg)
             * (1.5251 - 1.1568 * (1 - self.a_f)**0.1292))
        return f

    def quality_factor(self) -> float:
        """Quality factor Q."""
        return 0.7 + 1.4187 * (1 - self.a_f)**(-0.4990)

    def damping_time(self) -> float:
        """τ = Q / (π f_QNM)."""
        f = self.qnm_frequency()
        Q = self.quality_factor()
        return Q / (math.pi * f)

    def ringdown_waveform(self, t: NDArray, amplitude: float = 1e-21,
                             phi0: float = 0.0) -> NDArray:
        """h(t) = A exp(−t/τ) cos(2πf t + φ₀)."""
        f = self.qnm_frequency()
        tau = self.damping_time()
        return amplitude * np.exp(-t / tau) * np.cos(2 * math.pi * f * t + phi0)


# ---------------------------------------------------------------------------
#  Matched Filtering for GW Detection
# ---------------------------------------------------------------------------

class MatchedFilter:
    r"""
    Matched filtering for gravitational wave signal extraction.

    SNR:
    $$\text{SNR}^2 = 4\text{Re}\int_0^\infty
      \frac{\tilde{h}^*(f)\tilde{s}(f)}{S_n(f)}df$$

    Overlap (normalised inner product):
    $$\langle a|b\rangle = 4\text{Re}\int_0^\infty
      \frac{\tilde{a}^*(f)\tilde{b}(f)}{S_n(f)}df$$
    """

    def __init__(self, sample_rate: float = 4096.0) -> None:
        self.fs = sample_rate

    @staticmethod
    def aLIGO_psd(f: NDArray) -> NDArray:
        """Approximate aLIGO design sensitivity PSD (1/Hz).

        Simplified fit: S_n(f) = S_0[(f_0/f)^4 + 2(1 + (f/f_0)²)]
        """
        S0 = 1e-49  # Hz⁻¹
        f0 = 150.0  # Hz
        f_safe = np.maximum(f, 1.0)
        return S0 * ((f0 / f_safe)**4 + 2 * (1 + (f_safe / f0)**2))

    def inner_product(self, a: NDArray, b: NDArray, psd: NDArray,
                         df: float) -> float:
        """Noise-weighted inner product."""
        psd_safe = np.maximum(psd, 1e-60)
        integrand = np.conj(a) * b / psd_safe
        return 4 * float(np.real(np.sum(integrand) * df))

    def snr(self, signal: NDArray, template: NDArray) -> float:
        """Compute optimal matched-filter SNR."""
        N = len(signal)
        df = self.fs / N

        s_fft = np.fft.rfft(signal)
        h_fft = np.fft.rfft(template)
        freqs = np.fft.rfftfreq(N, 1.0 / self.fs)

        psd = self.aLIGO_psd(freqs)

        sigma_sq = self.inner_product(h_fft, h_fft, psd, df)
        if sigma_sq < 1e-60:
            return 0.0

        matched = self.inner_product(s_fft, h_fft, psd, df)
        return matched / math.sqrt(sigma_sq)

    def time_series_snr(self, data: NDArray, template: NDArray) -> NDArray:
        """SNR time series via inverse FFT."""
        N = len(data)
        df = self.fs / N
        freqs = np.fft.rfftfreq(N, 1.0 / self.fs)

        d_fft = np.fft.rfft(data)
        h_fft = np.fft.rfft(template)
        psd = self.aLIGO_psd(freqs)
        psd_safe = np.maximum(psd, 1e-60)

        integrand = np.conj(h_fft) * d_fft / psd_safe
        snr_t = np.fft.irfft(integrand) * 4 * df

        sigma = math.sqrt(self.inner_product(h_fft, h_fft, psd, df))
        if sigma > 0:
            snr_t /= sigma

        return snr_t


# ---------------------------------------------------------------------------
#  Gravitational Wave Energy & Luminosity
# ---------------------------------------------------------------------------

class GWEnergy:
    r"""
    Gravitational-wave energy and luminosity.

    Quadrupole formula:
    $$L_{\text{GW}} = \frac{G}{5c^5}\langle\dddot{I}_{ij}\dddot{I}^{ij}\rangle$$

    Total energy radiated (binary inspiral):
    $$E_{\text{rad}} = \eta M c^2 \left[\frac{v^2}{2} - \frac{v^4}{12}(9+\eta) + \ldots\right]$$

    where $v = (G M \omega / c^3)^{1/3}$ is the PN velocity parameter.

    Peak luminosity (NR fit):
    $$L_{\text{peak}} \approx 1.0\times10^{-2}(c^5/G)\,\eta^2(1+\alpha\eta+\beta\eta^2)$$
    """

    def __init__(self, m1: float = 30.0, m2: float = 30.0) -> None:
        self.m1 = m1 * M_SUN_KG
        self.m2 = m2 * M_SUN_KG
        self.M = self.m1 + self.m2
        self.eta = self.m1 * self.m2 / self.M**2

    def quadrupole_luminosity(self, omega: float) -> float:
        """L_GW from circular orbit (leading order).

        L = (32/5)(G⁴/c⁵) M³ η² ω^{10/3} / (GM)^{10/3}
        → L = (32/5)(η² c⁵/G)(v^{10})
        """
        v = (G_SI * self.M * omega / C_SI**3)**(1 / 3)
        return 32 / 5 * self.eta**2 * C_SI**5 / G_SI * v**10

    def energy_radiated_0pn(self, f_final: float) -> float:
        """Total radiated energy up to frequency f_final (0PN)."""
        v = (math.pi * G_SI * self.M * f_final / C_SI**3)**(1 / 3)
        return 0.5 * self.eta * self.M * C_SI**2 * v**2

    def peak_luminosity(self) -> float:
        """NR-fit peak luminosity (Watts)."""
        L_0 = C_SI**5 / G_SI  # ~3.6×10⁵² W
        return 1e-2 * L_0 * self.eta**2

    def characteristic_strain(self, f: float, D_L_mpc: float = 100.0) -> float:
        """Characteristic strain h_c(f).

        h_c = (2f²/ḟ)^{1/2} × h₀
        """
        D_L = D_L_mpc * MPC_M
        Mc_s = G_SI * self.M * self.eta**(3 / 5) / C_SI**3
        h0 = 4 * C_SI / D_L * (math.pi * Mc_s * f)**(2 / 3)
        f_dot = 96 / 5 * math.pi**(8 / 3) * Mc_s**(5 / 3) * f**(11 / 3)
        if f_dot < 1e-30:
            return h0
        return h0 * math.sqrt(2 * f / f_dot)
