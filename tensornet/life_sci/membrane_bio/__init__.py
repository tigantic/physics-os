"""
Membrane Biophysics — Coarse-grained lipid bilayer, electroporation,
pore nucleation free energy.

Domain XVI.3 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Lipid Bilayer (CG Model)
# ---------------------------------------------------------------------------

class CoarseGrainedBilayer:
    r"""
    Coarse-grained lipid bilayer model (Martini-style).

    Each lipid: head (hydrophilic) + tail beads (hydrophobic).
    Interaction: LJ + harmonic bonds + angle potentials.

    Bilayer properties:
    - Area per lipid: $A_L \approx 0.6\text{--}0.7\;\text{nm}^2$
    - Thickness: $d \approx 4\text{--}5\;\text{nm}$
    - Bending modulus: $\kappa_c \approx 10\text{--}20\;k_BT$
    """

    def __init__(self, n_lipids: int = 100, n_beads_per_lipid: int = 4,
                 Lx: float = 10.0, Ly: float = 10.0) -> None:
        self.n_lipids = n_lipids
        self.n_beads = n_beads_per_lipid
        self.Lx = Lx
        self.Ly = Ly

        # Bead parameters
        self.sigma_head = 0.47  # nm (LJ sigma)
        self.sigma_tail = 0.47
        self.eps_head = 5.0     # kJ/mol
        self.eps_tail = 3.5
        self.eps_cross = 1.0    # head-tail repulsion

        # Bond/angle
        self.k_bond = 1250.0    # kJ/mol/nm²
        self.r0_bond = 0.47     # nm
        self.k_angle = 25.0     # kJ/mol
        self.theta0 = math.pi   # straight

        # State: positions (n_lipids * n_beads, 3)
        self.positions = self._init_bilayer()

    def _init_bilayer(self) -> NDArray:
        """Initialize a flat bilayer in the xy-plane."""
        n = self.n_lipids
        n_per_side = n // 2
        side = int(math.ceil(math.sqrt(n_per_side)))

        spacing_x = self.Lx / side
        spacing_y = self.Ly / side

        positions = []
        bead_spacing = 0.35  # nm between beads along z

        for leaf in [0, 1]:  # upper, lower leaflet
            z_sign = 1 if leaf == 0 else -1
            for i in range(n_per_side):
                ix = i % side
                iy = i // side
                x = (ix + 0.5) * spacing_x
                y = (iy + 0.5) * spacing_y

                for b in range(self.n_beads):
                    z = z_sign * (1.0 + b * bead_spacing)
                    positions.append([x, y, z])

        return np.array(positions[:n * self.n_beads])

    def area_per_lipid(self) -> float:
        """A_L = 2 * Lx * Ly / n_lipids (factor 2 for two leaflets)."""
        return 2 * self.Lx * self.Ly / self.n_lipids

    def thickness(self) -> float:
        """Bilayer thickness from head-group z-positions."""
        head_z = []
        for i in range(self.n_lipids):
            z = self.positions[i * self.n_beads, 2]
            head_z.append(abs(z))
        return 2 * float(np.mean(head_z))

    def bending_modulus_helfrich(self, kappa_c: float = 15.0) -> float:
        """Helfrich bending energy per unit area for a sphere of radius R.

        E/A = 2κ_c/R² (sphere) + κ̄/R² (Gaussian).
        """
        return kappa_c

    def lj_energy(self) -> float:
        """Total LJ energy (simplified, pairwise between all beads)."""
        n_total = len(self.positions)
        E = 0.0
        for i in range(n_total):
            for j in range(i + 1, min(i + 20, n_total)):  # truncated
                dr = self.positions[j] - self.positions[i]
                # PBC in x, y
                dr[0] -= self.Lx * round(dr[0] / self.Lx)
                dr[1] -= self.Ly * round(dr[1] / self.Ly)
                r = math.sqrt(float(np.sum(dr**2)))
                if r < 1.2:
                    sr6 = (self.sigma_tail / r)**6
                    E += 4 * self.eps_tail * (sr6**2 - sr6)
        return E


# ---------------------------------------------------------------------------
#  Electroporation Model
# ---------------------------------------------------------------------------

class ElectroporationModel:
    r"""
    Electroporation: formation of pores in lipid bilayer under electric field.

    Pore nucleation energy (cylindrical pore):
    $$\Delta G(r) = 2\pi\gamma r - \pi\sigma_{\text{eff}} r^2
      + C_{\text{hydrophobic}} \cdot \frac{1}{r}$$

    where:
    - γ = pore line tension (N)
    - σ_eff = effective membrane tension (N/m)
    - Third term: steric correction.

    Critical pore radius: $r^* = \gamma / \sigma_{\text{eff}}$.
    Nucleation rate: $k \propto \exp(-\Delta G^* / k_B T)$.
    """

    def __init__(self, gamma_line: float = 1e-11,      # N (line tension)
                 sigma_tension: float = 1e-3,  # N/m (membrane tension)
                 T: float = 300.0,
                 d_membrane: float = 5e-9) -> None:
        self.gamma = gamma_line
        self.sigma = sigma_tension
        self.T = T
        self.kB = 1.381e-23
        self.d = d_membrane

        # Dielectric constants
        self.eps_water = 80.0
        self.eps_lipid = 2.0
        self.eps_0 = 8.854e-12

    def pore_energy(self, r: float, V: float = 0.0) -> float:
        r"""Free energy of a cylindrical pore of radius r under voltage V.

        ΔG(r) = 2πγr − πσ_eff r² − ½ C_pore V²

        C_pore contribution from Maxwell stress on pore edge.
        """
        E_line = 2 * math.pi * self.gamma * r
        E_tension = -math.pi * self.sigma * r**2

        # Electrostatic: capacitor model
        # ΔC/ΔA = Δε/d for pore area vs lipid area
        delta_eps = self.eps_0 * (self.eps_water - self.eps_lipid) / self.d
        E_electric = -0.5 * delta_eps * math.pi * r**2 * V**2

        return E_line + E_tension + E_electric

    def critical_radius(self, V: float = 0.0) -> float:
        """Critical pore radius: dΔG/dr = 0."""
        delta_eps = self.eps_0 * (self.eps_water - self.eps_lipid) / self.d
        sigma_eff = self.sigma + 0.5 * delta_eps * V**2
        if sigma_eff <= 0:
            return float('inf')
        return self.gamma / sigma_eff

    def barrier_height(self, V: float = 0.0) -> float:
        """ΔG* = ΔG(r*)."""
        r_star = self.critical_radius(V)
        return self.pore_energy(r_star, V)

    def nucleation_rate(self, V: float = 0.0,
                          k0: float = 1e9) -> float:
        """Kramers rate: k = k₀ exp(−ΔG*/k_BT)."""
        dG = self.barrier_height(V)
        return k0 * math.exp(-dG / (self.kB * self.T))

    def threshold_voltage(self, target_rate: float = 1e6) -> float:
        """Find V where nucleation rate reaches target."""
        for V in np.linspace(0, 5.0, 5000):
            if self.nucleation_rate(V) >= target_rate:
                return float(V)
        return 5.0


# ---------------------------------------------------------------------------
#  Helfrich Membrane Mechanics
# ---------------------------------------------------------------------------

class HelfrichMembrane:
    r"""
    Helfrich theory of membrane elasticity:

    $$F = \int\left[\frac{\kappa}{2}(2H-c_0)^2 + \bar{\kappa}K\right]dA + \int\sigma\,dA$$

    H = mean curvature, K = Gaussian curvature, c₀ = spontaneous curvature.
    κ = bending rigidity, κ̄ = Gaussian rigidity, σ = surface tension.

    1D membrane profile: $h(x)$ → curvature $\kappa \approx h''/(1+h'^2)^{3/2}$.
    """

    def __init__(self, kappa: float = 20.0, kappa_bar: float = -10.0,
                 sigma: float = 0.0, c0: float = 0.0) -> None:
        # kappa, kappa_bar in units of kBT
        self.kappa = kappa
        self.kappa_bar = kappa_bar
        self.sigma = sigma
        self.c0 = c0

    def bending_energy_1d(self, h: NDArray, dx: float) -> float:
        """1D bending energy for membrane profile h(x)."""
        hp = np.gradient(h, dx)
        hpp = np.gradient(hp, dx)
        curvature = hpp / (1 + hp**2)**1.5
        return 0.5 * self.kappa * float(np.sum((curvature - self.c0)**2)) * dx

    def tension_energy_1d(self, h: NDArray, dx: float) -> float:
        """Surface tension energy for 1D profile."""
        hp = np.gradient(h, dx)
        ds = np.sqrt(1 + hp**2)
        return self.sigma * float(np.sum(ds - 1)) * dx

    def fluctuation_spectrum(self, q: NDArray) -> NDArray:
        r"""Thermal fluctuation spectrum for a planar membrane.

        $\langle|h_q|^2\rangle = \frac{k_BT}{\sigma q^2 + \kappa q^4}$
        """
        return 1.0 / (self.sigma * q**2 + self.kappa * q**4 + 1e-30)

    def persistence_length(self) -> float:
        """Persistence length: l_p = a exp(4πκ/(3k_BT)).

        For kBT = 1: l_p = a exp(4πκ/3).
        """
        a = 1.0  # molecular scale
        return a * math.exp(4 * math.pi * self.kappa / 3)

    def vesicle_shape_parameter(self, V: float, A: float) -> float:
        """Reduced volume: v = V / (4π/3)(A/4π)^{3/2}.

        v = 1 for sphere, < 1 for deflated shapes (prolate, oblate, stomatocyte).
        """
        R_eff = math.sqrt(A / (4 * math.pi))
        V_sphere = 4 * math.pi * R_eff**3 / 3
        return V / V_sphere

    def min_energy_shape(self, reduced_volume: float) -> str:
        """Approximate shape classification from reduced volume.

        Based on Seifert (1997) phase diagram.
        """
        v = reduced_volume
        if v > 0.95:
            return "sphere"
        elif v > 0.65:
            return "prolate"
        elif v > 0.59:
            return "oblate"
        elif v > 0.5:
            return "stomatocyte"
        else:
            return "highly_deflated"
