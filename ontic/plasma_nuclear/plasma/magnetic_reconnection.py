"""
Magnetic Reconnection — Sweet-Parker, Petschek, plasmoid instability,
tearing mode stability.

Domain XI.5 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Sweet-Parker Reconnection
# ---------------------------------------------------------------------------

class SweetParkerReconnection:
    r"""
    Sweet-Parker model of steady-state magnetic reconnection.

    Aspect ratio: $\delta/L = S^{-1/2}$
    Reconnection rate: $v_{\text{in}} = v_A / \sqrt{S}$
    Outflow: $v_{\text{out}} = v_A$

    where Lundquist number $S = \mu_0 L v_A / \eta$ and
    $v_A = B/\sqrt{\mu_0\rho}$ is the Alfvén speed.

    Sweet-Parker is *slow* reconnection: rate ~ S^{−1/2}.
    For solar corona S ~ 10^{12}, rate ~ 10^{−6} v_A — far too slow.
    """

    def __init__(self, B0: float = 1e-3, n: float = 1e18,
                 eta: float = 1e-4, L: float = 1.0,
                 m_i: float = 1.67e-27) -> None:
        self.B0 = B0     # T
        self.n = n        # m^-3
        self.eta = eta    # Ohm·m (resistivity)
        self.L = L        # m (current sheet half-length)
        self.m_i = m_i
        self.mu0 = 4 * math.pi * 1e-7
        self.rho = n * m_i

    @property
    def v_A(self) -> float:
        """Alfvén speed: v_A = B/√(μ₀ρ)."""
        return self.B0 / math.sqrt(self.mu0 * self.rho)

    @property
    def lundquist_number(self) -> float:
        """S = μ₀ L v_A / η."""
        return self.mu0 * self.L * self.v_A / self.eta

    @property
    def current_sheet_thickness(self) -> float:
        """δ = L / √S."""
        return self.L / math.sqrt(self.lundquist_number)

    @property
    def inflow_velocity(self) -> float:
        """v_in = v_A / √S."""
        return self.v_A / math.sqrt(self.lundquist_number)

    @property
    def reconnection_rate(self) -> float:
        """M_A = v_in/v_A = 1/√S."""
        return 1.0 / math.sqrt(self.lundquist_number)

    @property
    def reconnection_electric_field(self) -> float:
        """E_rec = v_in B₀ = B₀ v_A / √S ≈ η j."""
        return self.inflow_velocity * self.B0

    @property
    def energy_release_rate(self) -> float:
        """P = (B²/μ₀) v_in L (per unit length in z)."""
        return self.B0**2 / self.mu0 * self.inflow_velocity * self.L

    def time_scale(self) -> float:
        """τ_SP = L / v_in = L √S / v_A = √(τ_A τ_η)."""
        tau_A = self.L / self.v_A
        tau_eta = self.mu0 * self.L**2 / self.eta
        return math.sqrt(tau_A * tau_eta)


# ---------------------------------------------------------------------------
#  Petschek Reconnection
# ---------------------------------------------------------------------------

class PetschekReconnection:
    r"""
    Petschek model: *fast* reconnection with slow-mode shocks.

    Reconnection rate: $M_A \sim \pi/(8\ln S)$

    Key features:
    - Localised diffusion region (much smaller than L).
    - Slow-mode standing shocks emanating from X-point.
    - Rate essentially independent of S → fast.

    Outflow: $v_{\text{out}} = v_A$.
    """

    def __init__(self, B0: float = 1e-3, n: float = 1e18,
                 eta: float = 1e-4, L: float = 1.0,
                 m_i: float = 1.67e-27) -> None:
        self.B0 = B0
        self.n = n
        self.eta = eta
        self.L = L
        self.mu0 = 4 * math.pi * 1e-7
        self.rho = n * m_i

    @property
    def v_A(self) -> float:
        return self.B0 / math.sqrt(self.mu0 * self.rho)

    @property
    def lundquist_number(self) -> float:
        return self.mu0 * self.L * self.v_A / self.eta

    @property
    def reconnection_rate(self) -> float:
        """M_A ≈ π / (8 ln S)."""
        S = self.lundquist_number
        return math.pi / (8 * math.log(S + 1))

    @property
    def diffusion_region_length(self) -> float:
        """l ~ L / S^{1/2}  (inner SP-like region)."""
        return self.L / math.sqrt(self.lundquist_number)

    @property
    def shock_opening_angle(self) -> float:
        """θ ≈ M_A = v_in/v_A (radians)."""
        return self.reconnection_rate

    def compare_sweet_parker(self) -> Dict[str, float]:
        """Compare Petschek vs Sweet-Parker rates."""
        S = self.lundquist_number
        sp_rate = 1 / math.sqrt(S)
        pet_rate = self.reconnection_rate
        return {
            'S': S,
            'sweet_parker_rate': sp_rate,
            'petschek_rate': pet_rate,
            'speedup': pet_rate / sp_rate,
        }


# ---------------------------------------------------------------------------
#  Plasmoid Instability
# ---------------------------------------------------------------------------

class PlasmoidInstability:
    r"""
    Plasmoid (tearing) instability of Sweet-Parker current sheets.

    For $S > S_c \approx 10^4$, the SP sheet fragments into plasmoids.

    Number of plasmoids: $N \sim S^{3/8}$ (Loureiro et al., 2007)
    Growth rate: $\gamma \tau_A \sim S^{1/4}$
    Plasmoid width: $w \sim L S^{-3/8}$

    This transitions reconnection from slow SP ($S^{-1/2}$)
    to fast ($\sim 0.01 v_A$, nearly S-independent).
    """

    def __init__(self, S: float = 1e6) -> None:
        self.S = S
        self.S_c = 1e4  # critical Lundquist number

    @property
    def is_unstable(self) -> bool:
        return self.S > self.S_c

    def number_of_plasmoids(self) -> float:
        """N ~ S^{3/8}."""
        return self.S**(3 / 8)

    def growth_rate_normalised(self) -> float:
        """γ τ_A ~ S^{1/4}."""
        return self.S**(1 / 4)

    def plasmoid_width(self, L: float = 1.0) -> float:
        """w ~ L S^{-3/8}."""
        return L * self.S**(-3 / 8)

    def reconnection_rate_fast(self) -> float:
        """In plasmoid-dominated regime, M_A ~ 0.01."""
        if self.S > self.S_c:
            return 0.01
        return 1 / math.sqrt(self.S)

    def transition_analysis(self) -> Dict[str, float]:
        """Full transition analysis between SP and plasmoid regimes."""
        return {
            'S': self.S,
            'S_c': self.S_c,
            'is_unstable': float(self.is_unstable),
            'N_plasmoids': self.number_of_plasmoids(),
            'growth_rate_norm': self.growth_rate_normalised(),
            'sp_rate': 1 / math.sqrt(self.S),
            'plasmoid_rate': self.reconnection_rate_fast(),
        }


# ---------------------------------------------------------------------------
#  Tearing Mode (resistive MHD)
# ---------------------------------------------------------------------------

class TearingMode:
    r"""
    Resistive tearing mode instability of a current sheet.

    Harris current sheet: $\mathbf{B} = B_0\tanh(x/a)\hat{y}$.

    Tearing stability parameter:
    $$\Delta' = \lim_{\epsilon\to 0^+}\left[\frac{\psi'(x_s+\epsilon)}{\psi(x_s+\epsilon)}
      - \frac{\psi'(x_s-\epsilon)}{\psi(x_s-\epsilon)}\right]$$

    Instability criterion: $\Delta' > 0$.

    Growth rate (Furth-Killeen-Rosenbluth 1963):
    $$\gamma = \left(\frac{k^2 B_0^2}{\mu_0\rho}\right)^{2/5}
      \left(\frac{\eta}{\mu_0}\right)^{3/5} \Delta'^{4/5}$$

    for the constant-ψ regime.
    """

    def __init__(self, B0: float = 1.0, a: float = 1.0,
                 eta: float = 1e-3, rho: float = 1.0) -> None:
        self.B0 = B0
        self.a = a
        self.eta = eta
        self.rho = rho
        self.mu0 = 1.0  # normalised units

    def harris_field(self, x: NDArray) -> NDArray:
        """B_y(x) = B₀ tanh(x/a)."""
        return self.B0 * np.tanh(x / self.a)

    def delta_prime(self, k: float) -> float:
        """Δ'(k) for Harris sheet: Δ' = 2(1/(ka) − ka)."""
        ka = k * self.a
        if ka >= 1.0:
            return 0.0  # stable
        return 2 * (1 / ka - ka)

    def growth_rate_fkr(self, k: float) -> float:
        """FKR growth rate (constant-ψ regime).

        γ = (k²B₀²/(μ₀ρ))^{2/5} · (η/μ₀)^{3/5} · Δ'^{4/5}
        """
        dp = self.delta_prime(k)
        if dp <= 0:
            return 0.0

        return (k**2 * self.B0**2 / (self.mu0 * self.rho))**(2 / 5) \
            * (self.eta / self.mu0)**(3 / 5) \
            * dp**(4 / 5)

    def growth_rate_scan(self, k_range: Optional[NDArray] = None) -> Dict[str, NDArray]:
        """Scan tearing growth rate vs wavenumber."""
        if k_range is None:
            k_range = np.linspace(0.01, 1.5, 100) / self.a

        gamma = np.array([self.growth_rate_fkr(k) for k in k_range])
        delta_p = np.array([self.delta_prime(k) for k in k_range])

        return {
            'k': k_range,
            'ka': k_range * self.a,
            'gamma': gamma,
            'delta_prime': delta_p,
        }

    def inner_layer_width(self, k: float) -> float:
        r"""Inner resistive layer width: δ ~ a (η τ_A)^{1/4}."""
        tau_A = self.a / (self.B0 / math.sqrt(self.mu0 * self.rho))
        return self.a * (self.eta * tau_A)**0.25

    def reconnected_flux(self, gamma: float, t: float,
                            psi0: float = 1e-5) -> float:
        """Reconnected flux: ψ(t) = ψ₀ exp(γt) (linear growth phase)."""
        return psi0 * math.exp(gamma * min(t, 100))
