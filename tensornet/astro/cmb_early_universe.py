"""
CMB & Early Universe — angular power spectrum, recombination,
inflation, Boltzmann hierarchy, CMB lensing.

Domain XII.5 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Physical Constants
# ---------------------------------------------------------------------------

T_CMB: float = 2.7255          # K (CMB temperature today)
K_B_EV: float = 8.617e-5      # eV/K
H0_S: float = 2.184e-18       # H₀ in s⁻¹ (67.4 km/s/Mpc)
SIGMA_T: float = 6.652e-25    # cm² (Thomson cross section)
M_E_EV: float = 0.511e6       # eV (electron mass)
ETA_B: float = 6.1e-10        # baryon-to-photon ratio


# ---------------------------------------------------------------------------
#  Recombination (Saha / Peebles)
# ---------------------------------------------------------------------------

class Recombination:
    r"""
    Hydrogen recombination in the early universe.

    Saha equation:
    $$\frac{n_e n_p}{n_H} = \left(\frac{m_e k_BT}{2\pi\hbar^2}\right)^{3/2}
      \exp\left(-\frac{B_1}{k_BT}\right)$$

    where B₁ = 13.6 eV (hydrogen binding energy).

    Free electron fraction: $X_e = n_e / n_H$.

    Peebles 3-level atom (more accurate):
    $$\frac{dX_e}{dt} = -C\left[\alpha_B n_H X_e^2 - 4\beta_B(1-X_e)
      \exp(-B_1/(4k_BT))\right]$$

    $C = (1 + K\Lambda_{2s}n_H(1-X_e))/(1 + K(\Lambda_{2s}+\beta_B)n_H(1-X_e))$
    """

    B1: float = 13.6  # eV (hydrogen ionisation energy)

    def __init__(self, Omega_b: float = 0.049, h: float = 0.674) -> None:
        self.Omega_b = Omega_b
        self.h = h

    def saha_Xe(self, T: float) -> float:
        """Saha equation free electron fraction X_e(T).

        T in Kelvin.
        """
        kT = K_B_EV * T
        if kT < 1e-4:
            return 0.0

        n_b = 2.5e-7 * self.Omega_b * self.h**2 * (T / T_CMB)**3  # cm⁻³

        S = (M_E_EV * kT / (2 * math.pi))**1.5 * math.exp(-self.B1 / kT) / n_b

        if S > 1e10:
            return 1.0
        discriminant = S**2 + 4 * S
        Xe = (-S + math.sqrt(max(discriminant, 0))) / 2
        return min(max(Xe, 0.0), 1.0)

    def peebles_recombination(self, T_start: float = 6000.0,
                                 T_end: float = 500.0,
                                 n_steps: int = 5000) -> Tuple[NDArray, NDArray]:
        """Peebles 3-level atom recombination.

        Returns (T_array, X_e_array).
        """
        T_arr = np.linspace(T_start, T_end, n_steps)
        Xe = np.ones(n_steps)
        Xe[0] = self.saha_Xe(T_start)

        for i in range(1, n_steps):
            T = T_arr[i]
            kT = K_B_EV * T
            if kT < 1e-6:
                Xe[i] = Xe[i - 1]
                continue

            n_H = 2.5e-7 * self.Omega_b * self.h**2 * (T / T_CMB)**3

            # Case B recombination coefficient
            T4 = T / 1e4
            alpha_B = 2.6e-13 * T4**(-0.8)  # cm³/s

            # Photoionisation from n=2
            beta_B = alpha_B * (M_E_EV * kT / (2 * math.pi))**1.5 * math.exp(-self.B1 / (4 * kT))

            # 2s→1s two-photon rate
            Lambda_2s = 8.227  # s⁻¹

            K = 1.0 / (n_H * 6e-20)  # cosmological redshifting factor

            C_pb = ((1 + K * Lambda_2s * n_H * (1 - Xe[i - 1]))
                    / (1 + K * (Lambda_2s + beta_B) * n_H * (1 - Xe[i - 1])))

            dXe = -C_pb * (alpha_B * n_H * Xe[i - 1]**2
                          - 4 * beta_B * (1 - Xe[i - 1])
                          * math.exp(-self.B1 / (4 * kT)))

            dT = T_arr[1] - T_arr[0] if i < n_steps - 1 else T_arr[i] - T_arr[i - 1]
            dt = abs(dT) / (T * H0_S * 1e5 / 3.086e24)
            Xe[i] = min(max(Xe[i - 1] + dXe * dt, 1e-6), 1.0)

        return T_arr, Xe

    def last_scattering_temperature(self) -> float:
        """T_* where X_e = 0.5 ≈ 3000 K."""
        T_arr, Xe = self.peebles_recombination()
        idx = np.argmin(np.abs(Xe - 0.5))
        return float(T_arr[idx])


# ---------------------------------------------------------------------------
#  CMB Angular Power Spectrum (Sachs-Wolfe + Acoustic)
# ---------------------------------------------------------------------------

class CMBPowerSpectrum:
    r"""
    CMB temperature angular power spectrum C_ℓ.

    Sachs-Wolfe plateau (low ℓ):
    $$\frac{\ell(\ell+1)C_\ell}{2\pi} \approx \frac{A_s}{9\pi}$$

    Acoustic peaks (simplified):
    $$\Theta_\ell = \frac{1}{3}\cos(k r_s) \cdot e^{-k^2/k_D^2}
      \cdot T(k)\cdot j_\ell(k\chi_*)$$

    Sound horizon at last scattering:
    $$r_s = \int_0^{t_*}\frac{c_s}{a}\,dt, \quad c_s = \frac{c}{\sqrt{3(1+R)}}$$
    where $R = 3\rho_b/(4\rho_\gamma)$.
    """

    def __init__(self, A_s: float = 2.1e-9, ns: float = 0.965) -> None:
        self.A_s = A_s
        self.ns = ns
        self.r_s = 144.4  # Mpc (sound horizon, Planck 2018)
        self.chi_star = 14000.0  # Mpc (comoving distance to last scattering)
        self.k_D = 0.14  # Mpc⁻¹ (damping scale)

    def sachs_wolfe(self, ell: int) -> float:
        """Sachs-Wolfe (ISW) contribution at low ℓ.

        ℓ(ℓ+1)C_ℓ/(2π) ≈ A_s/9π for large scales.
        """
        return self.A_s / (9 * math.pi) * 2 * math.pi / (ell * (ell + 1))

    def acoustic_peak_position(self, n: int) -> int:
        """Position of n-th acoustic peak: ℓ_n ≈ n π χ* / r_s."""
        return int(n * math.pi * self.chi_star / self.r_s)

    def cl_spectrum(self, ell_max: int = 2500) -> Tuple[NDArray, NDArray]:
        """Compute C_ℓ spectrum (simplified analytic model).

        Returns (ell, D_ell) where D_ℓ = ℓ(ℓ+1)C_ℓ/(2π).
        """
        ells = np.arange(2, ell_max + 1)
        D_ell = np.zeros(len(ells))

        for i, ell in enumerate(ells):
            k_ell = ell / self.chi_star

            # Primordial
            P_prim = self.A_s * (k_ell / 0.05)**((self.ns - 1))

            # SW plateau
            sw = P_prim / 9

            # Acoustic oscillation
            cos_arg = k_ell * self.r_s
            acoustic = (1 / 3 * math.cos(cos_arg))**2

            # Silk damping
            damping = math.exp(-2 * (k_ell / self.k_D)**2)

            D_ell[i] = (sw + 6 * P_prim * acoustic * damping) * 1e10

        return ells, D_ell

    def peak_ratio(self) -> float:
        """Ratio of first to second peak heights ∝ Ω_b.

        Higher baryon content → enhanced odd peaks.
        """
        ell1 = self.acoustic_peak_position(1)
        ell2 = self.acoustic_peak_position(2)
        _, D = self.cl_spectrum(max(ell1, ell2) + 100)
        return float(D[ell1 - 2]) / (float(D[ell2 - 2]) + 1e-30)


# ---------------------------------------------------------------------------
#  Inflation (Slow-Roll)
# ---------------------------------------------------------------------------

class SlowRollInflation:
    r"""
    Single-field slow-roll inflation.

    Slow-roll parameters:
    $$\epsilon = \frac{M_{\text{Pl}}^2}{2}\left(\frac{V'}{V}\right)^2,
      \quad
      \eta = M_{\text{Pl}}^2\frac{V''}{V}$$

    Number of e-folds:
    $$N = \int_{\phi_{\text{end}}}^{\phi_*}
      \frac{V}{M_{\text{Pl}}^2 V'}\,d\phi \approx 50$$-$60$

    Observables:
    - Scalar spectral index: $n_s = 1 - 6\epsilon + 2\eta$
    - Tensor-to-scalar ratio: $r = 16\epsilon$
    - Scalar amplitude: $A_s = V/(24\pi^2 M_{\text{Pl}}^4 \epsilon)$
    """

    M_PL: float = 2.435e18  # GeV (reduced Planck mass)

    def __init__(self, model: str = 'quadratic') -> None:
        self.model = model

    def potential(self, phi: float) -> float:
        """V(φ) for selected inflationary model."""
        if self.model == 'quadratic':
            m = 6e-6 * self.M_PL
            return 0.5 * m**2 * phi**2
        elif self.model == 'starobinsky':
            Lambda = 3e-3 * self.M_PL
            return Lambda**4 * (1 - math.exp(-math.sqrt(2 / 3) * phi / self.M_PL))**2
        return 0.5 * phi**2

    def potential_derivative(self, phi: float, dp: float = 1e-3) -> float:
        """V'(φ) via finite difference."""
        return (self.potential(phi + dp) - self.potential(phi - dp)) / (2 * dp)

    def potential_second_derivative(self, phi: float, dp: float = 1e-3) -> float:
        """V''(φ) via finite difference."""
        return (self.potential(phi + dp) - 2 * self.potential(phi) + self.potential(phi - dp)) / dp**2

    def epsilon(self, phi: float) -> float:
        """First slow-roll parameter ε."""
        V = self.potential(phi)
        Vp = self.potential_derivative(phi)
        if V < 1e-60:
            return 1.0
        return 0.5 * self.M_PL**2 * (Vp / V)**2

    def eta_sr(self, phi: float) -> float:
        """Second slow-roll parameter η."""
        V = self.potential(phi)
        Vpp = self.potential_second_derivative(phi)
        if V < 1e-60:
            return 0.0
        return self.M_PL**2 * Vpp / V

    def spectral_index(self, phi: float) -> float:
        """n_s = 1 − 6ε + 2η."""
        return 1 - 6 * self.epsilon(phi) + 2 * self.eta_sr(phi)

    def tensor_to_scalar(self, phi: float) -> float:
        """r = 16ε."""
        return 16 * self.epsilon(phi)

    def efolds(self, phi_start: float, phi_end: float,
                  n_steps: int = 1000) -> float:
        """N = ∫ V/(M_Pl² V') dφ."""
        phi_arr = np.linspace(phi_start, phi_end, n_steps)
        dphi = phi_arr[1] - phi_arr[0]
        N = 0.0
        for phi in phi_arr:
            Vp = self.potential_derivative(phi)
            V = self.potential(phi)
            if abs(Vp) > 1e-60:
                N += V / (self.M_PL**2 * Vp) * dphi
        return abs(N)


# ---------------------------------------------------------------------------
#  Boltzmann Hierarchy (Simplified)
# ---------------------------------------------------------------------------

class BoltzmannHierarchy:
    r"""
    Simplified Boltzmann hierarchy for CMB photon perturbations.

    $$\dot{\Theta}_0 = -k\Theta_1 - \dot{\Phi}$$
    $$\dot{\Theta}_1 = \frac{k}{3}\Theta_0 - \frac{2k}{3}\Theta_2 + \frac{k}{3}\Psi
      + \dot{\tau}[\Theta_1 - v_b/3]$$
    $$\dot{\Theta}_\ell = \frac{k}{2\ell+1}[\ell\Theta_{\ell-1}
      - (\ell+1)\Theta_{\ell+1}] - \dot{\tau}\Theta_\ell
      \quad (\ell \geq 2)$$

    where $\dot{\tau} = n_e \sigma_T a$ is the optical depth derivative.
    """

    def __init__(self, k: float = 0.01, ell_max: int = 10) -> None:
        """
        k: wavenumber (Mpc⁻¹).
        ell_max: truncation multipole.
        """
        self.k = k
        self.ell_max = ell_max
        self.Theta = np.zeros(ell_max + 1)

    def evolve(self, n_steps: int = 1000, tau_dot: float = -1.0,
                  Phi_dot: float = 0.0, Psi: float = 0.0,
                  v_b: float = 0.0, dt: float = 0.01) -> NDArray:
        """Evolve Boltzmann hierarchy."""
        k = self.k

        for step in range(n_steps):
            Theta_new = self.Theta.copy()

            # ℓ = 0
            Theta_new[0] = self.Theta[0] + dt * (-k * self.Theta[1] - Phi_dot)

            # ℓ = 1
            if self.ell_max >= 1:
                Theta_new[1] = self.Theta[1] + dt * (
                    k / 3 * self.Theta[0] - 2 * k / 3 * (self.Theta[2] if self.ell_max >= 2 else 0)
                    + k / 3 * Psi + tau_dot * (self.Theta[1] - v_b / 3))

            # ℓ ≥ 2
            for ell in range(2, self.ell_max + 1):
                prev = self.Theta[ell - 1]
                next_val = self.Theta[ell + 1] if ell < self.ell_max else 0.0
                Theta_new[ell] = self.Theta[ell] + dt * (
                    k / (2 * ell + 1) * (ell * prev - (ell + 1) * next_val)
                    - tau_dot * self.Theta[ell])

            self.Theta = Theta_new

        return self.Theta

    def source_function(self, tau_dot: float, Phi: float, Psi: float,
                           v_b: float) -> float:
        """CMB source function S(k, τ).

        S = g(τ)[Θ₀ + Ψ + (1/4)Π] + e^{-τ}(Φ̇ + Ψ̇) − (1/k)d(g v_b)/dτ
        Simplified: S ≈ g(Θ₀ + Ψ)
        """
        g = -tau_dot * math.exp(tau_dot)  # visibility function (simplified)
        return g * (self.Theta[0] + Psi)
