"""
Ceramics & High-Temperature Materials: sintering kinetics,
UHTC oxidation, thermal barrier coating.

Upgrades domain XIV.7.
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

R_GAS: float = 8.314       # J/(mol·K)
K_BOLT: float = 1.381e-23  # J/K


# ---------------------------------------------------------------------------
#  Sintering Kinetics
# ---------------------------------------------------------------------------

class SinteringModel:
    r"""
    Sintering models for ceramic densification.

    Initial-stage (two-sphere model):
    $$\frac{x^n}{a^m} = \frac{B t}{k_B T a^p}$$

    where x = neck radius, a = particle radius, B depends on mechanism:
    - Surface diffusion: n=7, m=3, p=4
    - Volume diffusion: n=5, m=2, p=3
    - Grain boundary diffusion: n=6, m=2, p=4
    - Evaporation-condensation: n=3, m=1, p=2

    Master Sintering Curve (MSC):
    $$\Theta(t, T) = \int_0^t \frac{1}{T}\exp\left(-\frac{Q}{RT}\right)dt$$
    """

    @dataclass
    class Mechanism:
        name: str
        n: int       # neck growth exponent
        m: int       # geometry exponent
        p: int       # size exponent
        B0: float    # prefactor
        Q: float     # activation energy (J/mol)

    SURFACE_DIFFUSION = Mechanism("surface_diffusion", 7, 3, 4, 1e-4, 300e3)
    VOLUME_DIFFUSION = Mechanism("volume_diffusion", 5, 2, 3, 1e-5, 400e3)
    GB_DIFFUSION = Mechanism("gb_diffusion", 6, 2, 4, 1e-6, 350e3)
    EVAP_CONDENSATION = Mechanism("evap_condensation", 3, 1, 2, 1e-3, 200e3)

    def __init__(self, mechanism: Optional["SinteringModel.Mechanism"] = None,
                 a: float = 1e-6) -> None:
        """
        Parameters
        ----------
        mechanism : Sintering mechanism. Default: volume diffusion.
        a : Particle radius (m).
        """
        self.mech = mechanism or self.VOLUME_DIFFUSION
        self.a = a

    def neck_ratio(self, t: float, T: float) -> float:
        """x/a neck-to-particle ratio at time t (s), temperature T (K)."""
        B = self.mech.B0 * math.exp(-self.mech.Q / (R_GAS * T))
        x_n = B * t / (K_BOLT * T * self.a**self.mech.p)
        ratio = (x_n * self.a**self.mech.m)**(1.0 / self.mech.n) / self.a
        return min(ratio, 1.0)  # Can't exceed 1

    def densification_rate(self, rho: float, T: float,
                             D: float = 1e-15) -> float:
        """Intermediate/final stage densification.

        dρ/dt = A D Ω γ / (k_B T G³)
        Simplified Coble-type grain boundary mechanism.
        """
        gamma_s = 1.0  # J/m²
        Omega = 2e-29  # atomic volume m³
        G = self.a * 5  # assume grain ~ 5× particle
        return 150 * D * Omega * gamma_s / (K_BOLT * T * G**3 + 1e-60)

    def master_sintering_curve_theta(self, t_arr: NDArray[np.float64],
                                       T_arr: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute MSC work-of-sintering Θ(t, T) from heating profile."""
        n_pts = len(t_arr)
        theta = np.zeros(n_pts)
        Q = self.mech.Q

        for i in range(1, n_pts):
            dt = t_arr[i] - t_arr[i - 1]
            T_avg = 0.5 * (T_arr[i] + T_arr[i - 1])
            if T_avg > 0:
                theta[i] = theta[i - 1] + dt / T_avg * math.exp(-Q / (R_GAS * T_avg))

        return theta


# ---------------------------------------------------------------------------
#  UHTC Oxidation
# ---------------------------------------------------------------------------

class UHTCOxidation:
    r"""
    Ultra-high temperature ceramic oxidation kinetics.

    ZrB₂ + 5/2 O₂ → ZrO₂ + B₂O₃

    Parabolic oxidation: $\Delta m^2 = k_p t$

    $$k_p = k_0 \exp(-Q/RT)$$

    Above ~1200°C: B₂O₃ evaporates → linear-parabolic transition.
    Active-passive transition at low pO₂.
    """

    def __init__(self, k0_parabolic: float = 1e4,
                 Q_parabolic: float = 200e3,
                 k0_linear: float = 1e2,
                 Q_linear: float = 150e3,
                 T_transition: float = 1500.0) -> None:
        """
        Parameters
        ----------
        k0_parabolic : Parabolic rate pre-exponential (mg² cm⁻⁴ s⁻¹).
        Q_parabolic : Parabolic activation energy (J/mol).
        k0_linear : Linear rate pre-exponential (mg cm⁻² s⁻¹).
        Q_linear : Linear activation energy (J/mol).
        T_transition : B₂O₃ evaporation transition temperature (K).
        """
        self.k0_p = k0_parabolic
        self.Q_p = Q_parabolic
        self.k0_l = k0_linear
        self.Q_l = Q_linear
        self.T_trans = T_transition

    def kp(self, T: float) -> float:
        """Parabolic rate constant (mg² cm⁻⁴ s⁻¹)."""
        return self.k0_p * math.exp(-self.Q_p / (R_GAS * T))

    def kl(self, T: float) -> float:
        """Linear rate constant (mg cm⁻² s⁻¹)."""
        return self.k0_l * math.exp(-self.Q_l / (R_GAS * T))

    def mass_gain(self, t: float, T: float) -> float:
        """Mass gain Δm (mg/cm²) at time t (s) and temperature T (K)."""
        if T < self.T_trans:
            # Pure parabolic
            return math.sqrt(self.kp(T) * t)
        else:
            # Linear-parabolic (Deal-Grove like)
            kp = self.kp(T)
            kl = self.kl(T)
            # x² + (kp/kl)x = kp·t  → quadratic
            a_coeff = 1.0
            b_coeff = kp / (kl + 1e-30)
            c_coeff = -kp * t
            disc = b_coeff**2 - 4 * a_coeff * c_coeff
            return (-b_coeff + math.sqrt(max(disc, 0))) / (2.0 * a_coeff)

    def oxide_thickness(self, delta_m: float,
                          rho_oxide: float = 5.68,
                          M_oxide: float = 123.2,
                          M_O2_consumed: float = 80.0) -> float:
        """Convert mass gain to oxide thickness (μm).

        δ = Δm · M_oxide / (M_O₂ × ρ_oxide)
        """
        return delta_m * 1e-3 * M_oxide / (M_O2_consumed * rho_oxide * 1e-4)


# ---------------------------------------------------------------------------
#  Thermal Barrier Coating (TBC)
# ---------------------------------------------------------------------------

class ThermalBarrierCoating:
    r"""
    Thermal barrier coating system: bond coat + TGO + ceramic top coat.

    Steady-state temperature drop:
    $$\Delta T_{TBC} = q \cdot t_{TBC} / k_{TBC}$$

    TGO growth: $h_{TGO}^2 = k_p^{TGO} t$ (parabolic).

    Spallation criterion: stored energy $U = \sigma^2 t / (2E')$
    exceeds fracture energy $G_c$.
    """

    @dataclass
    class Layer:
        name: str
        thickness: float  # m
        k: float          # thermal conductivity W/(m·K)
        E: float          # Young's modulus (GPa)
        alpha_CTE: float  # CTE (K⁻¹)

    def __init__(self, T_gas: float = 1600.0, T_sub: float = 1000.0) -> None:
        """
        Parameters
        ----------
        T_gas : Gas-side temperature (K).
        T_sub : Substrate temperature (K).
        """
        self.T_gas = T_gas
        self.T_sub = T_sub

        # Default 7YSZ / TGO / NiCoCrAlY / superalloy
        self.layers = [
            self.Layer("YSZ_topcoat", 300e-6, 1.5, 200.0, 10e-6),
            self.Layer("TGO", 1e-6, 5.0, 380.0, 8e-6),
            self.Layer("bond_coat", 150e-6, 16.0, 200.0, 14e-6),
        ]

    def temperature_profile(self) -> List[Tuple[str, float]]:
        """Compute T at each interface assuming 1D steady conduction."""
        q = self._heat_flux()
        T = self.T_gas
        profile = [("gas_side", T)]

        for layer in self.layers:
            T -= q * layer.thickness / layer.k
            profile.append((layer.name, T))

        return profile

    def _heat_flux(self) -> float:
        """Heat flux q through TBC stack (W/m²)."""
        R_total = sum(layer.thickness / layer.k for layer in self.layers)
        return (self.T_gas - self.T_sub) / R_total

    def tgo_thickness(self, t_hours: float, T_bc: float = 1050.0,
                        kp: float = 1e-17) -> float:
        """TGO thickness (m) after t hours of oxidation.

        Parameters
        ----------
        t_hours : Exposure time (hours).
        T_bc : Bond coat temperature (K).
        kp : Parabolic rate constant (m²/s).
        """
        t_s = t_hours * 3600.0
        kp_eff = kp * math.exp(-200e3 / (R_GAS * T_bc))
        return math.sqrt(kp_eff * t_s)

    def cte_mismatch_stress(self, delta_T: float,
                              layer_idx: int = 0) -> float:
        """Thermal stress from CTE mismatch (MPa).

        σ = E' × Δα × ΔT, E' = E/(1-ν).
        """
        layer = self.layers[layer_idx]
        # Reference: substrate/bond coat CTE
        alpha_sub = 14e-6
        delta_alpha = abs(layer.alpha_CTE - alpha_sub)
        E_prime = layer.E / (1 - 0.2)  # assume ν ≈ 0.2
        return E_prime * 1e3 * delta_alpha * delta_T  # MPa

    def spallation_lifetime(self, Gc: float = 20.0) -> float:
        """Estimated TBC spallation lifetime (hours).

        Based on TGO-induced stresses reaching fracture energy Gc (J/m²).
        """
        E_tgo = self.layers[1].E * 1e9  # Pa
        sigma_crit = math.sqrt(2.0 * Gc * E_tgo / (self.layers[1].thickness))
        # Time for TGO to grow thick enough
        # Conservative: when h_TGO reaches critical thickness
        h_crit = self.layers[1].thickness * 10  # ~10 μm
        kp = 1e-17 * math.exp(-200e3 / (R_GAS * 1050.0))
        t_s = h_crit**2 / (kp + 1e-60)
        return t_s / 3600.0
