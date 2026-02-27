"""
Surfaces & Interfaces — surface energy, adsorption isotherms, surface diffusion,
Schottky barriers, heterostructure band alignment.

Domain IX.6 — NEW.
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

K_B: float = 1.381e-23       # J/K
EV_J: float = 1.602e-19      # J
EPS_0: float = 8.854e-12     # F/m
E_CHARGE: float = 1.602e-19  # C
M_E: float = 9.109e-31       # kg
HBAR: float = 1.055e-34      # J·s


# ---------------------------------------------------------------------------
#  Surface Energy & Wulff Construction
# ---------------------------------------------------------------------------

class SurfaceEnergy:
    r"""
    Surface and interface energy calculations.

    Surface energy: $\gamma = \frac{1}{2A}(E_{\text{slab}} - N E_{\text{bulk}})$

    Work of adhesion: $W_{ad} = \gamma_1 + \gamma_2 - \gamma_{12}$

    Wulff shape: minimise $\sum_i \gamma_i A_i$ subject to constant volume.
    Wulff condition: $\gamma_i / h_i = \text{const}$ where $h_i$ is the distance
    from centre to face $i$.
    """

    def __init__(self) -> None:
        self.facets: List[Dict] = []

    def add_facet(self, miller: Tuple[int, int, int],
                     gamma: float, normal: NDArray) -> None:
        """Add a crystal facet.

        gamma: surface energy (J/m²).
        normal: outward normal (3,).
        """
        self.facets.append({
            'miller': miller,
            'gamma': gamma,
            'normal': np.asarray(normal, float) / np.linalg.norm(normal),
        })

    def slab_surface_energy(self, E_slab: float, E_bulk_per_atom: float,
                               N_atoms: int, area: float) -> float:
        """γ = (E_slab − N·E_bulk)/(2A).

        All energies in eV, area in Ų. Returns J/m².
        """
        return (E_slab - N_atoms * E_bulk_per_atom) * EV_J / (2 * area * 1e-20)

    def work_of_adhesion(self, gamma1: float, gamma2: float,
                            gamma12: float) -> float:
        """W_ad = γ₁ + γ₂ − γ₁₂ (J/m²)."""
        return gamma1 + gamma2 - gamma12

    def wulff_distances(self) -> List[Dict]:
        """Compute Wulff construction distances h_i ∝ γ_i."""
        if not self.facets:
            return []
        gamma_min = min(f['gamma'] for f in self.facets)
        result = []
        for f in self.facets:
            result.append({
                'miller': f['miller'],
                'h_ratio': f['gamma'] / gamma_min,
                'gamma': f['gamma'],
            })
        return result


# ---------------------------------------------------------------------------
#  Adsorption Isotherms
# ---------------------------------------------------------------------------

class AdsorptionIsotherms:
    r"""
    Classical adsorption isotherms.

    Langmuir: $\theta = \frac{KP}{1+KP}$, $K = K_0\exp(-E_a/k_BT)$

    BET: $\frac{P/P_0}{V(1-P/P_0)} = \frac{1}{V_m C} + \frac{C-1}{V_m C}\frac{P}{P_0}$

    Freundlich: $\theta = K_F P^{1/n}$

    Temkin: $\theta = \frac{k_BT}{\Delta H}\ln(K_T P)$
    """

    @staticmethod
    def langmuir(P: NDArray, K: float) -> NDArray:
        """Langmuir coverage θ(P)."""
        return K * P / (1 + K * P)

    @staticmethod
    def langmuir_K(Ea: float, T: float, K0: float = 1e-5) -> float:
        """Langmuir equilibrium constant K(T).

        Ea: adsorption energy (eV).
        """
        return K0 * math.exp(-Ea * EV_J / (K_B * T))

    @staticmethod
    def bet(P_P0: NDArray, Vm: float, C: float) -> NDArray:
        """BET volume adsorbed V(P/P₀).

        Vm: monolayer volume.
        C: BET constant.
        """
        x = P_P0
        return Vm * C * x / ((1 - x) * (1 + (C - 1) * x))

    @staticmethod
    def freundlich(P: NDArray, K_F: float, n: float) -> NDArray:
        """Freundlich isotherm θ = K_F P^(1/n)."""
        return K_F * np.power(P, 1 / n)

    @staticmethod
    def temkin(P: NDArray, K_T: float, T: float,
                  delta_H: float) -> NDArray:
        """Temkin isotherm θ = (kT/ΔH) ln(K_T P).

        delta_H in eV.
        """
        return K_B * T / (delta_H * EV_J) * np.log(K_T * P + 1e-30)

    @staticmethod
    def isosteric_heat(P1: float, T1: float,
                          P2: float, T2: float) -> float:
        """Clausius-Clapeyron isosteric heat of adsorption.

        Q_st = R T₁T₂/(T₂−T₁) ln(P₂/P₁) (kJ/mol)
        """
        R = 8.314  # J/(mol·K)
        return R * T1 * T2 / (T2 - T1) * math.log(P2 / P1) / 1000


# ---------------------------------------------------------------------------
#  Schottky Barrier
# ---------------------------------------------------------------------------

class SchottkyBarrier:
    r"""
    Metal-semiconductor Schottky barrier.

    Barrier height:
    - n-type: $\phi_{Bn} = \phi_m - \chi_s$
    - p-type: $\phi_{Bp} = E_g - (\phi_m - \chi_s)$

    With Fermi-level pinning (MIGS model):
    $$\phi_B = S(\phi_m - \phi_{\text{CNL}}) + (\phi_{\text{CNL}} - \chi_s)$$
    $S$ = interface pinning parameter.

    Depletion width: $W = \sqrt{\frac{2\epsilon_s(\phi_B - V)}{eN_D}}$

    Current: $J = A^* T^2 \exp(-e\phi_B/k_BT)[\exp(eV/k_BT)-1]$
    """

    def __init__(self, phi_m: float = 4.8, chi_s: float = 4.05,
                 Eg: float = 1.12, eps_r: float = 11.7,
                 Nd: float = 1e22) -> None:
        """
        phi_m: metal work function (eV).
        chi_s: semiconductor electron affinity (eV).
        Eg: band gap (eV).
        eps_r: relative permittivity.
        Nd: donor concentration (m⁻³).
        """
        self.phi_m = phi_m
        self.chi_s = chi_s
        self.Eg = Eg
        self.eps_r = eps_r
        self.Nd = Nd

    def barrier_height_n(self) -> float:
        """φ_Bn = φ_m − χ_s (eV)."""
        return self.phi_m - self.chi_s

    def barrier_height_p(self) -> float:
        """φ_Bp = E_g − (φ_m − χ_s) (eV)."""
        return self.Eg - (self.phi_m - self.chi_s)

    def barrier_with_pinning(self, S: float = 0.1,
                                phi_cnl: float = 4.5) -> float:
        """Barrier with interface pinning: φ_B = S(φ_m − φ_CNL) + (φ_CNL − χ_s)."""
        return S * (self.phi_m - phi_cnl) + (phi_cnl - self.chi_s)

    def depletion_width(self, V: float = 0.0) -> float:
        """W = √(2ε_s(φ_B − V)/(eN_D)) (m)."""
        phi_B = self.barrier_height_n()
        eps_s = self.eps_r * EPS_0
        val = 2 * eps_s * max(phi_B - V, 0) * EV_J / (E_CHARGE * self.Nd)
        return math.sqrt(val) if val > 0 else 0

    def capacitance(self, V: float = 0.0, area: float = 1e-8) -> float:
        """Depletion capacitance C = ε_s A / W (F)."""
        W = self.depletion_width(V)
        if W < 1e-15:
            return 0.0
        return self.eps_r * EPS_0 * area / W

    def iv_characteristic(self, V: NDArray, T: float = 300.0,
                             A_star: float = 1.12e6) -> NDArray:
        """Thermionic emission current density J(V) (A/m²).

        A* = Richardson constant (A/(m²·K²)).
        """
        phi_B = self.barrier_height_n()
        J_s = A_star * T**2 * math.exp(-E_CHARGE * phi_B / (K_B * T))
        return J_s * (np.exp(E_CHARGE * V / (K_B * T)) - 1)


# ---------------------------------------------------------------------------
#  Heterostructure Band Alignment
# ---------------------------------------------------------------------------

class HeterostructureBandAlignment:
    r"""
    Band alignment at semiconductor heterointerfaces.

    Anderson's rule:
    $$\Delta E_c = \chi_1 - \chi_2$$
    $$\Delta E_v = (E_{g2} - E_{g1}) - \Delta E_c$$

    Type I (straddling): both offsets same sign.
    Type II (staggered): offsets opposite sign.
    Type III (broken gap): VB of one above CB of other.
    """

    def __init__(self, material1: Dict[str, float],
                 material2: Dict[str, float]) -> None:
        """Each material: {'chi': electron affinity (eV), 'Eg': band gap (eV)}."""
        self.mat1 = material1
        self.mat2 = material2

    def band_offsets(self) -> Dict[str, float]:
        """Anderson's rule offsets."""
        dEc = self.mat1['chi'] - self.mat2['chi']
        dEv = (self.mat2['Eg'] - self.mat1['Eg']) - dEc
        return {'delta_Ec': dEc, 'delta_Ev': dEv}

    def alignment_type(self) -> str:
        """Determine alignment type (I, II, or III)."""
        offsets = self.band_offsets()
        dEc = offsets['delta_Ec']
        dEv = offsets['delta_Ev']

        # Type I: electron and hole both confined in same layer
        if dEc * dEv > 0:
            return 'Type I (straddling)'
        # Type III: broken gap
        Ec1 = -self.mat1['chi']
        Ev1 = Ec1 - self.mat1['Eg']
        Ec2 = -self.mat2['chi']
        Ev2 = Ec2 - self.mat2['Eg']
        if Ev1 > Ec2 or Ev2 > Ec1:
            return 'Type III (broken gap)'
        return 'Type II (staggered)'

    def quantum_well_levels(self, well_width: float,
                               m_star: float = 0.067,
                               n_max: int = 5) -> NDArray:
        """Infinite well approximation for confined states.

        E_n = n²π²ℏ²/(2m* L²)

        well_width: in nm.
        Returns energies in eV.
        """
        L = well_width * 1e-9
        m = m_star * M_E
        ns = np.arange(1, n_max + 1)
        return ns**2 * math.pi**2 * HBAR**2 / (2 * m * L**2) / EV_J
