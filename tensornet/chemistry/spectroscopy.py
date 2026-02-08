"""
Computational Spectroscopy — IR/Raman, UV-Vis, NMR chemical shifts,
Franck-Condon factors, rotational spectroscopy.

Domain XV.7 — NEW.
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
HBAR: float = 1.055e-34        # J·s
C_LIGHT: float = 2.998e8       # m/s
K_B: float = 1.381e-23         # J/K
AMU_KG: float = 1.661e-27      # kg
BOHR_M: float = 5.292e-11      # m
EV_J: float = 1.602e-19        # J
CM1_TO_J: float = 1.986e-23    # J per cm⁻¹


# ---------------------------------------------------------------------------
#  Vibrational Spectroscopy (IR / Raman)
# ---------------------------------------------------------------------------

class VibrationalSpectroscopy:
    r"""
    IR and Raman vibrational spectroscopy from force constants.

    Harmonic frequency:
    $$\nu = \frac{1}{2\pi}\sqrt{\frac{k}{\mu}}, \quad
      \tilde{\nu} = \frac{\nu}{c} \text{ (cm⁻¹)}$$

    IR intensity ∝ $|\partial\mu/\partial Q|^2$

    Raman intensity ∝ $|\partial\alpha/\partial Q|^2$ (polarisability derivative)

    Anharmonic correction (Morse):
    $$\tilde{\nu}_0 = \omega_e - 2\omega_e x_e, \quad
      x_e = \frac{h\nu}{4D_e}$$
    """

    def __init__(self) -> None:
        self.modes: List[Dict] = []

    def add_mode(self, k: float, mu: float, ir_intensity: float = 1.0,
                    raman_intensity: float = 1.0, label: str = '') -> None:
        """Add vibrational mode.

        k: force constant (N/m).
        mu: reduced mass (amu).
        ir_intensity: |dμ/dQ|² (km/mol).
        raman_intensity: |dα/dQ|² (ų⁴/amu).
        """
        mu_kg = mu * AMU_KG
        nu_hz = 1 / (2 * math.pi) * math.sqrt(k / mu_kg)
        nu_cm1 = nu_hz / C_LIGHT / 100  # cm⁻¹
        self.modes.append({
            'frequency_cm1': nu_cm1,
            'frequency_hz': nu_hz,
            'force_constant': k,
            'reduced_mass': mu,
            'ir_intensity': ir_intensity,
            'raman_intensity': raman_intensity,
            'label': label,
        })

    def ir_spectrum(self, x_range: Tuple[float, float] = (400, 4000),
                       sigma: float = 10.0,
                       n_pts: int = 1000) -> Tuple[NDArray, NDArray]:
        """Simulated IR absorption spectrum with Lorentzian broadening."""
        x = np.linspace(x_range[0], x_range[1], n_pts)
        spectrum = np.zeros(n_pts)
        for mode in self.modes:
            nu0 = mode['frequency_cm1']
            A = mode['ir_intensity']
            spectrum += A * sigma**2 / ((x - nu0)**2 + sigma**2)
        return x, spectrum

    def raman_spectrum(self, x_range: Tuple[float, float] = (100, 4000),
                          sigma: float = 10.0,
                          n_pts: int = 1000) -> Tuple[NDArray, NDArray]:
        """Simulated Raman spectrum."""
        x = np.linspace(x_range[0], x_range[1], n_pts)
        spectrum = np.zeros(n_pts)
        for mode in self.modes:
            nu0 = mode['frequency_cm1']
            A = mode['raman_intensity']
            spectrum += A * sigma**2 / ((x - nu0)**2 + sigma**2)
        return x, spectrum

    @staticmethod
    def anharmonic_fundamental(omega_e: float, De_eV: float) -> float:
        """Anharmonic fundamental frequency (cm⁻¹).

        ν₀ = ωe(1 − 2xe), xe = ωe/(4De)
        """
        De_cm1 = De_eV * EV_J / CM1_TO_J
        xe = omega_e / (4 * De_cm1)
        return omega_e * (1 - 2 * xe)


# ---------------------------------------------------------------------------
#  Electronic Spectroscopy (UV-Vis)
# ---------------------------------------------------------------------------

class ElectronicSpectroscopy:
    r"""
    UV-Vis absorption spectrum from electronic transition data.

    Oscillator strength:
    $$f_{0k} = \frac{2m_e\omega_{0k}}{3\hbar e^2}|\langle 0|\hat{\mu}|k\rangle|^2$$

    Absorption cross section:
    $$\sigma(\omega) = \frac{\pi e^2}{m_e c \epsilon_0}\sum_k f_{0k}\,L(\omega - \omega_{0k})$$

    Beer-Lambert: $A = \epsilon c l$ where $\epsilon = N_A\sigma/(1000\ln 10)$
    """

    def __init__(self) -> None:
        self.transitions: List[Dict] = []

    def add_transition(self, energy_eV: float, osc_strength: float,
                          label: str = '') -> None:
        """Add electronic transition."""
        wavelength_nm = 1239.8 / energy_eV if energy_eV > 0 else 0
        self.transitions.append({
            'energy_eV': energy_eV,
            'wavelength_nm': wavelength_nm,
            'osc_strength': osc_strength,
            'label': label,
        })

    def absorption_spectrum(self, lambda_range: Tuple[float, float] = (200, 800),
                               sigma_nm: float = 20.0,
                               n_pts: int = 1000) -> Tuple[NDArray, NDArray]:
        """Simulated UV-Vis spectrum (Gaussian broadening).

        Returns (wavelength_nm, molar_absorptivity).
        """
        lam = np.linspace(lambda_range[0], lambda_range[1], n_pts)
        spectrum = np.zeros(n_pts)

        for tr in self.transitions:
            lam0 = tr['wavelength_nm']
            f = tr['osc_strength']
            # Molar absorptivity peak ≈ 2.175×10⁸ f / σ (L/(mol·cm))
            eps_max = 2.175e8 * f / sigma_nm
            spectrum += eps_max * np.exp(-0.5 * ((lam - lam0) / sigma_nm)**2)

        return lam, spectrum

    @staticmethod
    def oscillator_strength_from_dipole(energy_eV: float,
                                           dipole_au: float) -> float:
        """f = (2/3) ΔE |μ|² (in atomic units)."""
        dE_au = energy_eV / 27.211  # eV → Hartree
        return 2 / 3 * dE_au * dipole_au**2


# ---------------------------------------------------------------------------
#  Franck-Condon Factors
# ---------------------------------------------------------------------------

class FranckCondonFactors:
    r"""
    Franck-Condon factors for vibronic transitions.

    $$\text{FC}_{m,n} = |\langle\chi_m'|\chi_n\rangle|^2$$

    For displaced harmonic oscillators (equal frequency):
    $$\text{FC}_{0,n} = \frac{S^n}{n!}e^{-S}$$

    Huang-Rhys factor: $S = \frac{1}{2}\mu\omega(\Delta Q)^2/\hbar$
    """

    def __init__(self, omega: float = 1400.0, delta_Q: float = 0.1,
                 mu: float = 12.0) -> None:
        """
        omega: vibrational frequency (cm⁻¹).
        delta_Q: equilibrium displacement (Å).
        mu: reduced mass (amu).
        """
        omega_si = omega * 100 * C_LIGHT * 2 * math.pi  # rad/s
        mu_kg = mu * AMU_KG
        dQ_m = delta_Q * 1e-10
        self.S = 0.5 * mu_kg * omega_si * dQ_m**2 / HBAR
        self.omega = omega

    def huang_rhys_factor(self) -> float:
        """S = λ/ℏω (dimensionless)."""
        return self.S

    def fc_factor(self, n: int) -> float:
        """FC_{0,n} = S^n exp(−S)/n!."""
        return self.S**n * math.exp(-self.S) / math.factorial(n)

    def fc_spectrum(self, n_max: int = 20) -> Tuple[NDArray, NDArray]:
        """Franck-Condon progression from 0→n.

        Returns (n, FC_0n).
        """
        ns = np.arange(n_max)
        fcs = np.array([self.fc_factor(int(n)) for n in ns])
        return ns, fcs

    def vibronic_spectrum(self, E_00: float = 3.0, n_max: int = 20,
                             sigma: float = 0.02,
                             n_pts: int = 500) -> Tuple[NDArray, NDArray]:
        """Vibronic absorption spectrum.

        E_00: 0-0 transition energy (eV).
        """
        hw = self.omega * CM1_TO_J / EV_J  # eV
        E_range = np.linspace(E_00 - 0.5, E_00 + n_max * hw + 0.5, n_pts)
        spectrum = np.zeros(n_pts)

        for n in range(n_max):
            E_n = E_00 + n * hw
            fc = self.fc_factor(n)
            spectrum += fc * np.exp(-0.5 * ((E_range - E_n) / sigma)**2)

        return E_range, spectrum


# ---------------------------------------------------------------------------
#  Rotational Spectroscopy
# ---------------------------------------------------------------------------

class RotationalSpectroscopy:
    r"""
    Rigid rotor spectroscopy for diatomic/linear molecules.

    Energy levels:
    $$E_J = B J(J+1) - D J^2(J+1)^2$$

    Rotational constant: $B = \hbar/(4\pi I c)$ (cm⁻¹)
    Centrifugal distortion: $D = 4B^3/\omega_e^2$

    Selection rule: $\Delta J = \pm 1$
    Transition frequencies: $\tilde{\nu} = 2B(J+1) - 4D(J+1)^3$
    """

    def __init__(self, B: float = 1.923, D: float = 6e-6) -> None:
        """
        B: rotational constant (cm⁻¹).
        D: centrifugal distortion constant (cm⁻¹).
        """
        self.B = B
        self.D = D

    @classmethod
    def from_bond(cls, mu_amu: float, r_eq_ang: float,
                     omega_e: float = 2000.0) -> 'RotationalSpectroscopy':
        """Create from reduced mass and bond length.

        B = ℏ/(4πIc), I = μ r²
        """
        mu_kg = mu_amu * AMU_KG
        r_m = r_eq_ang * 1e-10
        I = mu_kg * r_m**2
        B = HBAR / (4 * math.pi * I * C_LIGHT * 100)  # cm⁻¹
        D = 4 * B**3 / omega_e**2
        return cls(B=B, D=D)

    def energy(self, J: int) -> float:
        """E(J) in cm⁻¹."""
        return self.B * J * (J + 1) - self.D * J**2 * (J + 1)**2

    def transition_frequency(self, J: int) -> float:
        """ν̃(J → J+1) in cm⁻¹."""
        return 2 * self.B * (J + 1) - 4 * self.D * (J + 1)**3

    def population(self, J: int, T: float = 300.0) -> float:
        """Boltzmann population of level J.

        P(J) ∝ (2J+1) exp(−E(J) hc/(k_BT))
        """
        E_J = self.energy(J) * CM1_TO_J
        return (2 * J + 1) * math.exp(-E_J / (K_B * T))

    def spectrum(self, J_max: int = 30, T: float = 300.0) -> Tuple[NDArray, NDArray]:
        """Rotational absorption spectrum.

        Returns (frequencies_cm1, intensities).
        """
        freqs = np.array([self.transition_frequency(J) for J in range(J_max)])
        intensities = np.array([self.population(J, T) * (J + 1)
                               for J in range(J_max)])
        # Normalise
        intensities /= (np.max(intensities) + 1e-30)
        return freqs, intensities


# ---------------------------------------------------------------------------
#  NMR Chemical Shifts
# ---------------------------------------------------------------------------

class NMRChemicalShift:
    r"""
    NMR chemical shift calculation.

    Shielding tensor:
    $$\sigma_{ij} = \sigma_{ij}^{\text{dia}} + \sigma_{ij}^{\text{para}}$$

    Isotropic shielding: $\sigma_{\text{iso}} = (\sigma_{xx}+\sigma_{yy}+\sigma_{zz})/3$

    Chemical shift: $\delta = \sigma_{\text{ref}} - \sigma_{\text{iso}}$ (ppm)

    Diamagnetic (Lamb):
    $$\sigma^{\text{dia}} = \frac{e^2}{3m_e c^2}\left\langle\frac{1}{r}\right\rangle$$

    Paramagnetic (Ramsey):
    $$\sigma^{\text{para}} = -\frac{e^2}{m_e c^2}\sum_{n\neq 0}
      \frac{\langle 0|\hat{L}|n\rangle\langle n|\hat{L}/r^3|0\rangle}{E_n - E_0}$$
    """

    def __init__(self) -> None:
        self.nuclei: List[Dict] = []

    def add_nucleus(self, label: str, sigma_iso: float,
                       sigma_ref: float = 31.7) -> None:
        """Add nucleus with isotropic shielding.

        sigma_iso: isotropic shielding (ppm).
        sigma_ref: reference shielding (ppm) — TMS for ¹H.
        """
        delta = sigma_ref - sigma_iso
        self.nuclei.append({
            'label': label,
            'sigma_iso': sigma_iso,
            'delta_ppm': delta,
        })

    def get_spectrum(self) -> List[Dict]:
        """Return all chemical shifts."""
        return sorted(self.nuclei, key=lambda x: x['delta_ppm'])

    @staticmethod
    def lamb_diamagnetic(Z: int, r_avg_inv: float) -> float:
        """Lamb diamagnetic shielding σ_dia (ppm).

        σ_dia = (e²/3m_e c²) ⟨1/r⟩ × 10⁶
        r_avg_inv in atomic units (1/a₀).
        """
        e = 1.602e-19
        me = 9.109e-31
        c = C_LIGHT
        a0 = BOHR_M
        sigma = e**2 / (3 * me * c**2) * r_avg_inv / a0 * 1e6
        return sigma
