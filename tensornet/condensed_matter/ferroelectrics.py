"""
Ferroelectrics & Piezoelectrics — Landau-Devonshire theory, polarisation
switching, piezoelectric coupling, domain formation, pyroelectric effect.

Domain IX.8 — NEW.
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

EPS_0: float = 8.854e-12   # F/m
K_B: float = 1.381e-23     # J/K
EV_J: float = 1.602e-19    # J
E_CHARGE: float = 1.602e-19


# ---------------------------------------------------------------------------
#  Landau-Devonshire Theory
# ---------------------------------------------------------------------------

class LandauDevonshire:
    r"""
    Landau-Devonshire theory for ferroelectric phase transitions.

    Free energy (1D, uniaxial):
    $$G(P,T) = \frac{\alpha}{2}P^2 + \frac{\beta}{4}P^4 + \frac{\gamma}{6}P^6 - EP$$

    Temperature dependence: $\alpha = \alpha_0(T - T_C)$ (Curie-Weiss)

    Second-order: β > 0 → continuous transition at T_C.
    First-order: β < 0, γ > 0 → discontinuous transition at T₀ > T_C.

    Spontaneous polarisation: $P_s = \sqrt{-\alpha/\beta}$ (2nd order, T < T_C)

    Dielectric susceptibility: $\chi = 1/\alpha$ (T > T_C, Curie-Weiss law)
    """

    def __init__(self, alpha_0: float = 3.8e5, Tc: float = 393.0,
                 beta: float = -7.6e7, gamma: float = 2.6e8) -> None:
        """
        alpha_0: Curie constant (V·m/(C·K)).
        Tc: Curie temperature (K).
        beta: 4th-order coefficient (V·m⁵/C³).
        gamma: 6th-order coefficient (V·m⁹/C⁵).
        """
        self.alpha_0 = alpha_0
        self.Tc = Tc
        self.beta = beta
        self.gamma = gamma

    def alpha(self, T: float) -> float:
        """α(T) = α₀(T − T_C)."""
        return self.alpha_0 * (T - self.Tc)

    def free_energy(self, P: NDArray, T: float,
                       E: float = 0.0) -> NDArray:
        """G(P, T, E) (J/m³)."""
        a = self.alpha(T)
        return (a / 2 * P**2 + self.beta / 4 * P**4
                + self.gamma / 6 * P**6 - E * P)

    def spontaneous_polarisation(self, T: float) -> float:
        """P_s(T) for second-order transition (β > 0).

        P_s = √(−α/β) for T < T_C.
        """
        a = self.alpha(T)
        if a >= 0 or self.beta == 0:
            return 0.0
        if self.beta > 0:
            return math.sqrt(-a / self.beta)
        # First-order: solve cubic numerically
        # αP + βP³ + γP⁵ = 0 → P² = (−β ± √(β²−4αγ))/(2γ)
        disc = self.beta**2 - 4 * a * self.gamma
        if disc < 0:
            return 0.0
        P2 = (-self.beta + math.sqrt(disc)) / (2 * self.gamma)
        return math.sqrt(max(P2, 0))

    def susceptibility(self, T: float) -> float:
        """χ = 1/α (T > T_C), ε_r = 1 + χ/ε₀."""
        a = self.alpha(T)
        if abs(a) < 1e-10:
            return 1e15
        return 1 / a

    def hysteresis_loop(self, T: float, E_max: float = 1e8,
                           n_pts: int = 500) -> Tuple[NDArray, NDArray]:
        """P(E) hysteresis loop from dG/dP = 0.

        Numerical: for each E, find stable P.
        """
        E_up = np.linspace(-E_max, E_max, n_pts)
        E_down = np.linspace(E_max, -E_max, n_pts)
        E_full = np.concatenate([E_up, E_down])
        P_full = np.zeros(2 * n_pts)

        a = self.alpha(T)
        P = 0.0

        for i, E in enumerate(E_full):
            # Newton iteration: dG/dP = αP + βP³ + γP⁵ − E = 0
            for _ in range(50):
                f = a * P + self.beta * P**3 + self.gamma * P**5 - E
                df = a + 3 * self.beta * P**2 + 5 * self.gamma * P**4
                if abs(df) < 1e-30:
                    break
                P -= f / df
            P_full[i] = P

        return E_full, P_full


# ---------------------------------------------------------------------------
#  Piezoelectric Coupling
# ---------------------------------------------------------------------------

class PiezoelectricCoupling:
    r"""
    Linear piezoelectric constitutive relations.

    $$S_i = s_{ij}^E T_j + d_{mi}E_m$$
    $$D_m = d_{mi}T_i + \epsilon_{mn}^T E_n$$

    Alternatively:
    $$T_i = c_{ij}^E S_j - e_{mi}E_m$$
    $$D_m = e_{mi}S_i + \epsilon_{mn}^S E_n$$

    Coupling coefficient: $k^2 = d^2/(s^E\epsilon^T) = e^2/(c^E\epsilon^S)$

    Piezoelectric constants: $e = d \cdot c^E$, $d = e \cdot s^E$
    """

    def __init__(self, d33: float = 300e-12, eps_33: float = 1700.0,
                 s33: float = 20e-12) -> None:
        """
        d33: piezoelectric charge constant (C/N or m/V).
        eps_33: relative permittivity.
        s33: elastic compliance (m²/N).
        """
        self.d33 = d33
        self.eps_33_abs = eps_33 * EPS_0
        self.s33 = s33

    def coupling_coefficient(self) -> float:
        """Electromechanical coupling k² = d²/(s·ε)."""
        return self.d33**2 / (self.s33 * self.eps_33_abs)

    def e33(self) -> float:
        """Piezoelectric stress constant e₃₃ = d₃₃/s₃₃ (C/m²)."""
        return self.d33 / self.s33

    def g33(self) -> float:
        """Voltage constant g₃₃ = d₃₃/ε₃₃ (Vm/N)."""
        return self.d33 / self.eps_33_abs

    def stress_from_field(self, E_field: float) -> float:
        """Stress T = e₃₃ × E (Pa)."""
        return self.e33() * E_field

    def strain_from_field(self, E_field: float) -> float:
        """Strain S = d₃₃ × E (dimensionless)."""
        return self.d33 * E_field

    def voltage_from_stress(self, T: float, thickness: float) -> float:
        """Open-circuit voltage V = g₃₃ × T × t (V)."""
        return self.g33() * T * thickness

    def energy_density(self, E_field: float) -> float:
        """Stored energy density u = (1/2) ε E² + (1/2) d E T (J/m³)."""
        T = self.stress_from_field(E_field)
        return 0.5 * self.eps_33_abs * E_field**2 + 0.5 * self.d33 * E_field * T

    def resonant_frequency(self, thickness: float,
                              density: float = 7800.0) -> float:
        """Thickness-mode resonance f_r = 1/(2t)√(1/(ρ s₃₃)) (Hz)."""
        return 1 / (2 * thickness) * math.sqrt(1 / (density * self.s33))


# ---------------------------------------------------------------------------
#  Ferroelectric Domain Switching
# ---------------------------------------------------------------------------

class DomainSwitching:
    r"""
    Ferroelectric domain switching dynamics.

    Nucleation-limited switching (KAI model):
    $$P(t) = 2P_s\left[1 - \exp\left(-\left(\frac{t}{t_0}\right)^n\right)\right]$$

    Merz's law for switching time:
    $$t_s = t_\infty \exp(E_a/E)$$

    Coercive field estimation: $E_c \approx \frac{2\beta P_s^3 + 4\gamma P_s^5}{3}$
    """

    def __init__(self, Ps: float = 0.26, t0: float = 1e-9,
                 n_exponent: float = 2.0) -> None:
        """
        Ps: spontaneous polarisation (C/m²).
        t0: characteristic switching time (s).
        n_exponent: Avrami exponent.
        """
        self.Ps = Ps
        self.t0 = t0
        self.n = n_exponent

    def kai_switching(self, t: NDArray) -> NDArray:
        """KAI model: P(t)/Ps = 1 − exp(−(t/t₀)^n)."""
        return self.Ps * (1 - np.exp(-(t / self.t0)**self.n))

    def merz_switching_time(self, E: float, Ea: float = 5e6,
                               t_inf: float = 1e-12) -> float:
        """Merz's law: t_s = t_∞ exp(E_a/E).

        E: applied field (V/m).
        Ea: activation field (V/m).
        """
        if abs(E) < 1e-10:
            return float('inf')
        return t_inf * math.exp(Ea / abs(E))

    def fatigue_model(self, n_cycles: NDArray, P0: float = 0.26,
                         n_half: float = 1e6) -> NDArray:
        """Fatigue: P(N) = P₀ / (1 + N/N₁/₂).

        n_half: cycles for 50% degradation.
        """
        return P0 / (1 + n_cycles / n_half)

    def retention_loss(self, t: NDArray, tau: float = 1e8) -> NDArray:
        """Retention: P(t) = P_s exp(−t/τ).

        tau: retention time constant (s).
        """
        return self.Ps * np.exp(-t / tau)


# ---------------------------------------------------------------------------
#  Pyroelectric Effect
# ---------------------------------------------------------------------------

class PyroelectricEffect:
    r"""
    Pyroelectric effect in ferroelectric materials.

    Pyroelectric coefficient: $p = \frac{dP_s}{dT}$

    Current density: $J_p = p\frac{dT}{dt}$

    Figure of merit (current): $F_I = p / (\epsilon c_V)$
    where $c_V$ = volumetric heat capacity.

    Figure of merit (voltage): $F_V = p / (\epsilon c_V \sqrt{\epsilon})$
    """

    def __init__(self, p: float = -2.5e-4, eps_r: float = 200.0,
                 cv: float = 2.5e6) -> None:
        """
        p: pyroelectric coefficient (C/(m²·K)).
        eps_r: relative permittivity.
        cv: volumetric heat capacity (J/(m³·K)).
        """
        self.p = p
        self.eps_r = eps_r
        self.cv = cv

    def current_density(self, dTdt: float) -> float:
        """J_p = p × dT/dt (A/m²)."""
        return self.p * dTdt

    def figure_of_merit_current(self) -> float:
        """F_I = p/(ε₀ε_r c_V) (m²/C)."""
        return abs(self.p) / (EPS_0 * self.eps_r * self.cv)

    def figure_of_merit_voltage(self) -> float:
        """F_V = p/(c_V √(ε₀ε_r)) (m³/J)^{1/2}."""
        return abs(self.p) / (self.cv * math.sqrt(EPS_0 * self.eps_r))

    def voltage_response(self, dTdt: float, area: float,
                            thickness: float) -> float:
        """Open-circuit voltage V = p × dT/dt × t / (ε₀ ε_r) (V)."""
        return abs(self.p) * dTdt * thickness / (EPS_0 * self.eps_r)

    def nea_response(self, wavelength: float = 10e-6,
                        absorptivity: float = 0.8,
                        area: float = 1e-6,
                        Gth: float = 1e-3) -> float:
        """Noise-equivalent power for pyroelectric detector.

        NEP = √(4kTG_th) / (p × A × ω / G_th) — simplified.
        """
        return math.sqrt(4 * K_B * 300 * Gth) * Gth / (abs(self.p) * area)
