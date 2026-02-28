"""
Computational Photonics — photonic band structure, coupled-mode theory,
waveguide modes, photonic crystal slabs, transfer matrix.

Domain III.6 — NEW.
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


# ---------------------------------------------------------------------------
#  Transfer Matrix Method (1D Photonic Crystal)
# ---------------------------------------------------------------------------

class TransferMatrix1D:
    r"""
    Transfer matrix method for 1D layered photonic structures.

    Each layer transfer matrix:
    $$M_j = \begin{pmatrix}\cos\delta_j & -i\sin\delta_j/\eta_j \\
      -i\eta_j\sin\delta_j & \cos\delta_j\end{pmatrix}$$

    $\delta_j = k_0 n_j d_j\cos\theta_j$ = phase thickness.
    $\eta_j = n_j\cos\theta_j$ (TE) or $n_j/\cos\theta_j$ (TM).

    Reflectance: $r = \frac{(M_{11}+M_{12}\eta_s)\eta_0 - (M_{21}+M_{22}\eta_s)}{(M_{11}+M_{12}\eta_s)\eta_0 + (M_{21}+M_{22}\eta_s)}$
    """

    def __init__(self, n_substrate: float = 1.5, n_superstrate: float = 1.0,
                 theta0: float = 0.0) -> None:
        """
        theta0: angle of incidence (radians).
        """
        self.n_sub = n_substrate
        self.n_sup = n_superstrate
        self.theta0 = theta0
        self.layers: List[Tuple[float, float]] = []  # (n, d) pairs

    def add_layer(self, n: float, d: float) -> None:
        """Add layer with refractive index n and thickness d (metres)."""
        self.layers.append((n, d))

    def add_bragg_stack(self, n1: float, n2: float,
                           d1: float, d2: float, N: int = 10) -> None:
        """Add N periods of alternating layers."""
        for _ in range(N):
            self.layers.append((n1, d1))
            self.layers.append((n2, d2))

    def compute(self, wavelength: float,
                   polarisation: str = 'TE') -> Dict[str, float]:
        """Compute reflectance, transmittance at given wavelength.

        polarisation: 'TE' or 'TM'.
        """
        k0 = 2 * math.pi / wavelength

        # Snell's law angles
        def snell_angle(n: float) -> complex:
            sin_t = self.n_sup * math.sin(self.theta0) / n
            if abs(sin_t) > 1:
                return complex(math.pi / 2, math.acosh(abs(sin_t)))
            return math.asin(sin_t)

        def admittance(n: float, theta: complex) -> complex:
            cos_t = np.cos(theta) if isinstance(theta, complex) else math.cos(theta)
            if polarisation == 'TE':
                return n * cos_t
            else:
                return n / cos_t

        # Build total transfer matrix
        M = np.eye(2, dtype=complex)

        for n_j, d_j in self.layers:
            theta_j = snell_angle(n_j)
            cos_t = np.cos(theta_j) if isinstance(theta_j, complex) else math.cos(theta_j)
            delta = k0 * n_j * d_j * cos_t
            eta_j = admittance(n_j, theta_j)

            layer_M = np.array([
                [np.cos(delta), -1j * np.sin(delta) / eta_j],
                [-1j * eta_j * np.sin(delta), np.cos(delta)]
            ], dtype=complex)

            M = M @ layer_M

        eta_0 = admittance(self.n_sup, self.theta0)
        theta_s = snell_angle(self.n_sub)
        eta_s = admittance(self.n_sub, theta_s)

        num = (M[0, 0] + M[0, 1] * eta_s) * eta_0 - (M[1, 0] + M[1, 1] * eta_s)
        den = (M[0, 0] + M[0, 1] * eta_s) * eta_0 + (M[1, 0] + M[1, 1] * eta_s)

        r = num / den
        R = float(abs(r)**2)
        T = max(0, 1 - R)

        return {'R': R, 'T': T, 'r': complex(r)}

    def band_structure(self, n1: float, n2: float, d1: float, d2: float,
                          n_k: int = 200,
                          n_bands: int = 5) -> Tuple[NDArray, NDArray]:
        """Photonic band structure for 1D crystal (Bloch theorem).

        cos(Ka) = (M₁₁ + M₂₂)/2, a = d₁ + d₂.
        Returns (k, omega).
        """
        a = d1 + d2
        omega_max = 2 * math.pi * C_LIGHT / (min(d1, d2) * 0.5)
        n_omega = 500

        omegas = np.linspace(1e8, omega_max, n_omega)
        K_vals: List[List[float]] = [[] for _ in range(n_omega)]

        for io, omega in enumerate(omegas):
            k0 = omega / C_LIGHT
            delta1 = k0 * n1 * d1
            delta2 = k0 * n2 * d2

            M1 = np.array([
                [np.cos(delta1), -1j * np.sin(delta1) / n1],
                [-1j * n1 * np.sin(delta1), np.cos(delta1)]
            ], dtype=complex)
            M2 = np.array([
                [np.cos(delta2), -1j * np.sin(delta2) / n2],
                [-1j * n2 * np.sin(delta2), np.cos(delta2)]
            ], dtype=complex)

            M = M1 @ M2
            cos_Ka = float(np.real((M[0, 0] + M[1, 1]) / 2))

            if abs(cos_Ka) <= 1:
                K = math.acos(cos_Ka) / a
                K_vals[io].append(K)

        return omegas, K_vals


# ---------------------------------------------------------------------------
#  Coupled-Mode Theory
# ---------------------------------------------------------------------------

class CoupledModeTheory:
    r"""
    Temporal coupled-mode theory for resonator-waveguide systems.

    $$\frac{da}{dt} = \left(i\omega_0 - \frac{1}{\tau_0} - \frac{1}{\tau_e}\right)a
      + \sqrt{\frac{2}{\tau_e}}s_+$$

    Output: $s_- = -s_+ + \sqrt{\frac{2}{\tau_e}}a$

    Transmission:
    $$t = \frac{s_-}{s_+} = \frac{i(\omega-\omega_0)+1/\tau_0-1/\tau_e}{i(\omega-\omega_0)+1/\tau_0+1/\tau_e}$$

    Add/drop filter with two ports:
    $$t = \frac{i(\omega-\omega_0)+1/\tau_0}{i(\omega-\omega_0)+1/\tau_0+2/\tau_e}$$
    $$d = \frac{-2/\tau_e}{i(\omega-\omega_0)+1/\tau_0+2/\tau_e}$$
    """

    def __init__(self, omega_0: float = 2e15, tau_0: float = 1e-11,
                 tau_e: float = 1e-12) -> None:
        """
        omega_0: resonance frequency (rad/s).
        tau_0: intrinsic loss lifetime (s).
        tau_e: external coupling lifetime (s).
        """
        self.omega_0 = omega_0
        self.tau_0 = tau_0
        self.tau_e = tau_e

    def quality_factor(self) -> Dict[str, float]:
        """Q factors: Q_0 (intrinsic), Q_e (external), Q_total."""
        Q_0 = self.omega_0 * self.tau_0 / 2
        Q_e = self.omega_0 * self.tau_e / 2
        Q_t = 1 / (1 / Q_0 + 1 / Q_e)
        return {'Q_0': Q_0, 'Q_e': Q_e, 'Q_total': Q_t}

    def through_transmission(self, omega: NDArray) -> NDArray:
        """Through-port transmission |t|².

        All-pass configuration.
        """
        delta = omega - self.omega_0
        num = 1j * delta + 1 / self.tau_0 - 1 / self.tau_e
        den = 1j * delta + 1 / self.tau_0 + 1 / self.tau_e
        return np.abs(num / den)**2

    def drop_transmission(self, omega: NDArray) -> NDArray:
        """Drop-port transmission |d|² for add-drop filter."""
        delta = omega - self.omega_0
        num = -2 / self.tau_e
        den = 1j * delta + 1 / self.tau_0 + 2 / self.tau_e
        return np.abs(num / den)**2

    def group_delay(self, omega: NDArray) -> NDArray:
        """Group delay τ_g = −dφ/dω (numerical derivative)."""
        t = self._through_complex(omega)
        phase = np.unwrap(np.angle(t))
        d_omega = omega[1] - omega[0]
        tau_g = -np.gradient(phase, d_omega)
        return tau_g

    def _through_complex(self, omega: NDArray) -> NDArray:
        delta = omega - self.omega_0
        num = 1j * delta + 1 / self.tau_0 - 1 / self.tau_e
        den = 1j * delta + 1 / self.tau_0 + 1 / self.tau_e
        return num / den


# ---------------------------------------------------------------------------
#  Slab Waveguide Mode Solver
# ---------------------------------------------------------------------------

class SlabWaveguide:
    r"""
    Symmetric slab waveguide mode solver.

    Dispersion relation (TE):
    $$\kappa d/2 = m\pi/2 + \arctan(\gamma/\kappa)$$

    where $\kappa = \sqrt{k_0^2 n_f^2 - \beta^2}$, $\gamma = \sqrt{\beta^2 - k_0^2 n_s^2}$,
    $\beta = k_0 n_{\text{eff}}$.

    V-number: $V = k_0 d\sqrt{n_f^2 - n_s^2}/2$
    """

    def __init__(self, n_film: float = 3.5, n_sub: float = 1.5,
                 d: float = 300e-9) -> None:
        """
        d: slab thickness (m).
        """
        self.n_f = n_film
        self.n_s = n_sub
        self.d = d

    def v_number(self, wavelength: float) -> float:
        """V-parameter."""
        k0 = 2 * math.pi / wavelength
        return k0 * self.d / 2 * math.sqrt(self.n_f**2 - self.n_s**2)

    def n_modes(self, wavelength: float) -> int:
        """Number of guided modes (TE + TM)."""
        V = self.v_number(wavelength)
        return max(1, int(2 * V / math.pi) + 1)

    def find_modes(self, wavelength: float,
                      n_search: int = 1000) -> List[float]:
        """Find effective indices of guided TE modes.

        n_eff ∈ [n_s, n_f].
        """
        k0 = 2 * math.pi / wavelength
        n_eff_range = np.linspace(self.n_s + 1e-6, self.n_f - 1e-6, n_search)

        modes: List[float] = []
        prev_sign = 0

        for n_eff in n_eff_range:
            beta = k0 * n_eff
            kappa_sq = k0**2 * self.n_f**2 - beta**2
            gamma_sq = beta**2 - k0**2 * self.n_s**2

            if kappa_sq <= 0 or gamma_sq <= 0:
                continue

            kappa = math.sqrt(kappa_sq)
            gamma = math.sqrt(gamma_sq)

            # Dispersion equation: kappa*d/2 - arctan(gamma/kappa) = m*pi/2
            lhs = kappa * self.d / 2 - math.atan(gamma / kappa)
            sign = 1 if math.sin(lhs) > 0 else -1

            if prev_sign != 0 and sign != prev_sign:
                modes.append(float(n_eff))

            prev_sign = sign

        return modes

    def confinement_factor(self, n_eff: float,
                              wavelength: float) -> float:
        """Fraction of power in the core: Γ = ∫_core |E|²dx / ∫|E|²dx.

        Analytical for symmetric slab TE₀.
        """
        k0 = 2 * math.pi / wavelength
        beta = k0 * n_eff
        kappa = math.sqrt(max(k0**2 * self.n_f**2 - beta**2, 0))
        gamma = math.sqrt(max(beta**2 - k0**2 * self.n_s**2, 0))

        if kappa < 1e-15 or gamma < 1e-15:
            return 0.0

        # Γ = 1 − 1/(γd + 1) for fundamental mode approximation
        return 1 - 1 / (gamma * self.d + 1)
