"""
Nonadiabatic Dynamics — surface hopping, Landau-Zener, spin-boson,
diabatic/adiabatic representations, decoherence.

Domain XV.4 — NEW.
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

HBAR: float = 1.055e-34      # J·s
EV_J: float = 1.602e-19      # J
FS_S: float = 1e-15          # s
AMU_KG: float = 1.661e-27    # kg


# ---------------------------------------------------------------------------
#  Landau-Zener Model
# ---------------------------------------------------------------------------

class LandauZener:
    r"""
    Landau-Zener nonadiabatic transition at an avoided crossing.

    Diabatic energies: $E_1(x) = F_1 x$, $E_2(x) = -F_2 x + \Delta$

    Transition probability (single pass):
    $$P_{\text{LZ}} = \exp\left(-\frac{2\pi |V_{12}|^2}{\hbar v |\Delta F|}\right)$$

    where $v$ = nuclear velocity at crossing, $\Delta F = |F_1 - F_2|$,
    $V_{12}$ = diabatic coupling.

    Double-pass (Stückelberg):
    $$P_{\text{total}} = 2P_{\text{LZ}}(1-P_{\text{LZ}})$$
    """

    def __init__(self, V12: float = 0.01, delta_F: float = 0.1) -> None:
        """
        V12: diabatic coupling (eV).
        delta_F: slope difference (eV/Å).
        """
        self.V12 = V12 * EV_J
        self.delta_F = delta_F * EV_J / 1e-10  # eV/Å → J/m

    def transition_probability(self, velocity: float) -> float:
        """P_LZ = exp(−2π|V₁₂|²/(ℏv|ΔF|)).

        velocity in m/s.
        """
        if abs(velocity) < 1e-20 or abs(self.delta_F) < 1e-30:
            return 0.0
        arg = 2 * math.pi * self.V12**2 / (HBAR * abs(velocity) * abs(self.delta_F))
        return math.exp(-arg)

    def stuckelberg_probability(self, velocity: float) -> float:
        """Double-pass LZ: P = 2 P_LZ (1 − P_LZ)."""
        p = self.transition_probability(velocity)
        return 2 * p * (1 - p)

    def adiabatic_gap(self) -> float:
        """Minimum adiabatic energy gap: 2|V₁₂| (eV)."""
        return 2 * self.V12 / EV_J

    def adiabatic_surfaces(self, x: NDArray, F1: float = 0.05,
                              F2: float = 0.05) -> Tuple[NDArray, NDArray]:
        """Adiabatic potential energy surfaces (eV).

        x in Ångström.
        """
        V12_eV = self.V12 / EV_J
        E1 = F1 * x
        E2 = -F2 * x
        delta = E1 - E2
        gap = np.sqrt(delta**2 + 4 * V12_eV**2)
        E_plus = 0.5 * (E1 + E2 + gap)
        E_minus = 0.5 * (E1 + E2 - gap)
        return E_minus, E_plus


# ---------------------------------------------------------------------------
#  Tully's Fewest Switches Surface Hopping (FSSH)
# ---------------------------------------------------------------------------

class FewestSwitchesSurfaceHopping:
    r"""
    Tully's fewest-switches surface hopping (FSSH) for nonadiabatic dynamics.

    Classical nuclear motion on active surface $k$:
    $$M\ddot{R} = -\nabla E_k(R)$$

    Electronic amplitudes:
    $$i\hbar\dot{c}_j = \sum_k c_k\left(E_k\delta_{jk}
      - i\hbar\dot{R}\cdot\mathbf{d}_{jk}\right)$$

    Hopping probability (k → j):
    $$g_{k\to j} = \frac{2\,\text{Re}(c_j^* c_k\,\dot{R}\cdot\mathbf{d}_{jk})}{|c_k|^2}\Delta t$$

    $\mathbf{d}_{jk} = \langle\phi_j|\nabla_R\phi_k\rangle$ = nonadiabatic coupling vector.
    """

    def __init__(self, n_states: int = 2, mass: float = 2000.0,
                 dt: float = 0.5) -> None:
        """
        mass: nuclear mass (amu).
        dt: timestep (fs).
        """
        self.n_states = n_states
        self.mass = mass * AMU_KG
        self.dt = dt * FS_S

        self.R: float = -5.0e-10  # position (m)
        self.V: float = 0.0       # velocity (m/s)
        self.active: int = 0
        self.c = np.zeros(n_states, dtype=complex)
        self.c[0] = 1.0

    def set_initial_conditions(self, R0: float, V0: float,
                                  active: int = 0) -> None:
        """Set initial nuclear position (Å), velocity (Å/fs), and state."""
        self.R = R0 * 1e-10
        self.V = V0 * 1e-10 / FS_S
        self.active = active
        self.c = np.zeros(self.n_states, dtype=complex)
        self.c[active] = 1.0

    def propagate_nuclei(self, force: float) -> None:
        """Velocity Verlet step for nuclei."""
        accel = force / self.mass
        self.R += self.V * self.dt + 0.5 * accel * self.dt**2
        self.V += accel * self.dt

    def propagate_electronics(self, energies: NDArray,
                                 coupling: NDArray) -> None:
        """Propagate electronic amplitudes via RK4.

        energies: (n_states,) adiabatic energies (J).
        coupling: (n_states, n_states) d_{jk} · Ṙ (s⁻¹).
        """
        def dcdt(c: NDArray) -> NDArray:
            dc = np.zeros_like(c)
            for j in range(self.n_states):
                for k in range(self.n_states):
                    if j == k:
                        dc[j] -= 1j / HBAR * energies[j] * c[j]
                    else:
                        dc[j] -= c[k] * coupling[j, k]
            return dc

        k1 = dcdt(self.c) * self.dt
        k2 = dcdt(self.c + 0.5 * k1) * self.dt
        k3 = dcdt(self.c + 0.5 * k2) * self.dt
        k4 = dcdt(self.c + k3) * self.dt
        self.c += (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def hopping_probability(self, coupling: NDArray) -> NDArray:
        """Compute g_{k→j} for all states j ≠ active."""
        g = np.zeros(self.n_states)
        k = self.active
        ck_sq = abs(self.c[k])**2
        if ck_sq < 1e-30:
            return g

        for j in range(self.n_states):
            if j != k:
                g[j] = max(0, 2 * np.real(
                    np.conj(self.c[j]) * self.c[k] * coupling[j, k]
                ) / ck_sq * self.dt / FS_S)
        return g

    def attempt_hop(self, g: NDArray, energies: NDArray) -> bool:
        """Stochastic hop with energy conservation (velocity rescaling)."""
        xi = np.random.random()
        cum = 0.0
        for j in range(self.n_states):
            if j == self.active:
                continue
            cum += g[j]
            if xi < cum:
                dE = energies[j] - energies[self.active]
                KE = 0.5 * self.mass * self.V**2
                if KE >= dE:
                    new_KE = KE - dE
                    self.V = math.copysign(math.sqrt(2 * new_KE / self.mass), self.V)
                    self.active = j
                    return True
                return False  # frustrated hop
        return False


# ---------------------------------------------------------------------------
#  Spin-Boson Model
# ---------------------------------------------------------------------------

class SpinBosonModel:
    r"""
    Spin-boson model for two-state systems coupled to a bath.

    $$H = \frac{\epsilon}{2}\sigma_z + \frac{\Delta}{2}\sigma_x
      + \sum_k \omega_k b_k^\dagger b_k
      + \frac{\sigma_z}{2}\sum_k c_k(b_k^\dagger + b_k)$$

    Spectral density (Ohmic with exponential cutoff):
    $$J(\omega) = \frac{\pi}{2}\alpha\omega\,e^{-\omega/\omega_c}$$

    Reorganisation energy: $\lambda = \frac{2}{\pi}\int_0^\infty\frac{J(\omega)}{\omega}d\omega = \alpha\omega_c$

    Marcus rate:
    $$k_{\text{Marcus}} = \frac{\Delta^2}{2\hbar}\sqrt{\frac{\pi}{\lambda k_BT}}
      \exp\left(-\frac{(\epsilon-\lambda)^2}{4\lambda k_BT}\right)$$
    """

    K_B: float = 1.381e-23

    def __init__(self, epsilon: float = 0.0, Delta: float = 0.01,
                 alpha: float = 0.1, omega_c: float = 1.0) -> None:
        """
        epsilon: bias (eV).
        Delta: coupling (eV).
        alpha: dimensionless Kondo parameter.
        omega_c: cutoff frequency (eV).
        """
        self.epsilon = epsilon * EV_J
        self.Delta = Delta * EV_J
        self.alpha = alpha
        self.omega_c = omega_c * EV_J / HBAR  # rad/s
        self.lambda_reorg = alpha * omega_c * EV_J

    def spectral_density(self, omega: float) -> float:
        """J(ω) = (π/2) α ω exp(−ω/ω_c)."""
        return 0.5 * math.pi * self.alpha * omega * math.exp(-omega / self.omega_c)

    def marcus_rate(self, T: float) -> float:
        """Marcus rate k (s⁻¹)."""
        kBT = self.K_B * T
        lam = self.lambda_reorg
        if kBT < 1e-30 or lam < 1e-30:
            return 0.0
        return (self.Delta**2 / (2 * HBAR)
                * math.sqrt(math.pi / (lam * kBT))
                * math.exp(-(self.epsilon - lam)**2 / (4 * lam * kBT)))

    def redfield_dynamics(self, T: float, t_max: float = 100.0,
                             n_steps: int = 1000) -> Tuple[NDArray, NDArray]:
        """Simplified Redfield dynamics for population P₁(t).

        dp₁/dt = −(k_f + k_b)p₁ + k_b
        """
        kf = self.marcus_rate(T)
        kb = kf * math.exp(-self.epsilon / (self.K_B * T)) if T > 0 else 0

        t = np.linspace(0, t_max * FS_S, n_steps)
        dt = t[1] - t[0]
        p1 = np.zeros(n_steps)
        p1[0] = 1.0

        for i in range(1, n_steps):
            p1[i] = p1[i - 1] + dt * (-(kf + kb) * p1[i - 1] + kb)

        return t / FS_S, p1
