"""
Relativistic Electronic Structure — ZORA, spin-orbit coupling (SOC),
Douglas-Kroll-Hess, 4-component Dirac.

Domain VIII.6 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Constants (atomic units unless stated)
# ---------------------------------------------------------------------------

SPEED_OF_LIGHT: float = 137.036  # in atomic units


# ---------------------------------------------------------------------------
#  ZORA (Zeroth-Order Regular Approximation)
# ---------------------------------------------------------------------------

class ZORAHamiltonian:
    r"""
    Zeroth-Order Regular Approximation to the Dirac equation.

    $$\hat{H}_{\text{ZORA}} = V + \mathbf{p}\cdot\frac{c^2}{2c^2 - V}\mathbf{p}$$

    This captures ~95% of scalar relativistic effects with an operator
    that acts on 2-component (Pauli) spinors.

    Kinetic energy operator:
    $$T_{\text{ZORA}} = \mathbf{p}\cdot K(r)\mathbf{p}$$
    where $K(r) = c^2/(2c^2 - V(r))$.
    """

    def __init__(self, V: NDArray, grid: NDArray) -> None:
        """
        V: (n,) potential on grid.
        grid: (n,) radial or 1D grid.
        """
        self.V = V
        self.grid = grid
        self.n = len(grid)
        self.c = SPEED_OF_LIGHT

    def zora_kinetic_factor(self) -> NDArray:
        """K(r) = c²/(2c² − V(r))."""
        return self.c**2 / (2 * self.c**2 - self.V)

    def hamiltonian_matrix(self) -> NDArray:
        """Build ZORA Hamiltonian on grid using finite differences.

        H_ZORA = −(1/2)(d/dx)(K(x))(d/dx) + V(x)
        """
        dx = self.grid[1] - self.grid[0]
        K = self.zora_kinetic_factor()

        H = np.zeros((self.n, self.n))
        for i in range(self.n):
            H[i, i] = self.V[i]

        # Kinetic with position-dependent mass: T = −½ ∂_x K ∂_x
        for i in range(1, self.n - 1):
            K_plus = 0.5 * (K[i] + K[i + 1])
            K_minus = 0.5 * (K[i] + K[i - 1])
            H[i, i] += (K_plus + K_minus) / dx**2
            H[i, i - 1] -= K_minus / dx**2
            H[i, i + 1] -= K_plus / dx**2

        return H

    def solve(self) -> Tuple[NDArray, NDArray]:
        """Diagonalise ZORA Hamiltonian."""
        H = self.hamiltonian_matrix()
        return np.linalg.eigh(H)

    def scalar_relativistic_shift(self, V_nuc: NDArray) -> NDArray:
        """Compute scalar relativistic energy shift δε_n = ε_ZORA − ε_NR."""
        H_zora = self.hamiltonian_matrix()

        # Non-relativistic: K → 1/2
        dx = self.grid[1] - self.grid[0]
        H_nr = np.diag(self.V.copy())
        for i in range(1, self.n - 1):
            H_nr[i, i] += 1.0 / dx**2
            H_nr[i, i - 1] -= 0.5 / dx**2
            H_nr[i, i + 1] -= 0.5 / dx**2

        evals_zora = np.linalg.eigvalsh(H_zora)
        evals_nr = np.linalg.eigvalsh(H_nr)
        return evals_zora - evals_nr


# ---------------------------------------------------------------------------
#  Spin-Orbit Coupling (SOC)
# ---------------------------------------------------------------------------

class SpinOrbitCoupling:
    r"""
    Spin-orbit coupling operator within the Pauli/ZORA framework.

    $$\hat{H}_{\text{SOC}} = \frac{1}{2c^2}\frac{1}{r}\frac{dV}{dr}\hat{\mathbf{L}}\cdot\hat{\mathbf{S}}$$

    In matrix form (2-spinor basis: spin-up, spin-down for each spatial orbital):
    $$\langle l,m_l,\sigma|H_{\text{SOC}}|l,m_l',\sigma'\rangle
      = \xi_{nl}\langle lm_l\sigma|\mathbf{L}\cdot\mathbf{S}|lm_l'\sigma'\rangle$$

    where $\xi_{nl}$ is the SOC constant.
    """

    def __init__(self, V: NDArray, grid: NDArray) -> None:
        self.V = V
        self.grid = grid
        self.c = SPEED_OF_LIGHT

    def soc_constant(self, psi_radial: NDArray, l: int) -> float:
        r"""$\xi_{nl} = \frac{1}{2c^2}\int |\psi|^2 \frac{1}{r}\frac{dV}{dr} r^2 dr$."""
        r = self.grid
        dr = r[1] - r[0]
        r_safe = np.maximum(r, 1e-10)

        dV_dr = np.gradient(self.V, dr)
        integrand = psi_radial**2 * dV_dr / r_safe * r**2
        xi = float(np.trapz(integrand, r)) / (2 * self.c**2)
        return xi

    @staticmethod
    def ls_matrix(l: int) -> NDArray:
        """L·S matrix in |l,m_l,σ⟩ basis.

        Dimension: 2(2l+1) × 2(2l+1).

        L·S = L_z S_z + ½(L_+ S_- + L_- S_+)
        """
        dim_orb = 2 * l + 1
        dim = 2 * dim_orb

        ml_vals = list(range(-l, l + 1))
        LS = np.zeros((dim, dim))

        for im, ml in enumerate(ml_vals):
            # spin-up = im, spin-down = im + dim_orb
            up = im
            dn = im + dim_orb

            # L_z S_z
            LS[up, up] += 0.5 * ml
            LS[dn, dn] -= 0.5 * ml

            # L_+ S_- : raises ml by 1, flips down→up
            if ml + 1 <= l:
                factor = math.sqrt(l * (l + 1) - ml * (ml + 1))
                LS[im + 1, dn] += 0.5 * factor  # |ml+1,up⟩ ← |ml,dn⟩

            # L_- S_+ : lowers ml by 1, flips up→down
            if ml - 1 >= -l:
                factor = math.sqrt(l * (l + 1) - ml * (ml - 1))
                LS[im - 1 + dim_orb, up] += 0.5 * factor  # |ml-1,dn⟩ ← |ml,up⟩

        return LS

    def soc_hamiltonian(self, l: int, psi_radial: NDArray) -> NDArray:
        """Full SOC Hamiltonian: H_SOC = ξ L·S."""
        xi = self.soc_constant(psi_radial, l)
        LS = self.ls_matrix(l)
        return xi * LS

    @staticmethod
    def jj_eigenvalues(l: int, xi: float) -> Dict[str, float]:
        """Analytical j-j splitting: E(j=l±½) = ξ/2 [j(j+1) − l(l+1) − 3/4]."""
        if l == 0:
            return {'j=1/2': 0.0}
        jp = l + 0.5
        jm = l - 0.5
        e_plus = xi / 2 * (jp * (jp + 1) - l * (l + 1) - 0.75)
        e_minus = xi / 2 * (jm * (jm + 1) - l * (l + 1) - 0.75)
        return {f'j={jp}': e_plus, f'j={jm}': e_minus}


# ---------------------------------------------------------------------------
#  Douglas-Kroll-Hess Transformation
# ---------------------------------------------------------------------------

class DouglasKrollHess:
    r"""
    Douglas-Kroll-Hess (DKH) scalar relativistic transformation.

    The DKH transformation decouples the large and small components of
    the Dirac equation order by order.

    DKH1:
    $$H_{\text{DKH1}} = E_p + A_p(V + R_p V R_p)A_p$$

    where $E_p = c\sqrt{p^2 + c^2}$, $A_p = \sqrt{(E_p + c^2)/(2E_p)}$,
    $R_p = cp/(E_p + c^2)$.

    DKH2 adds an additional even-even transformation.
    """

    def __init__(self, c: float = SPEED_OF_LIGHT) -> None:
        self.c = c

    def free_particle_energy(self, p: NDArray) -> NDArray:
        """E_p = c√(p² + c²) — free particle energy (positive branch)."""
        return self.c * np.sqrt(p**2 + self.c**2)

    def kinematic_factors(self, p: NDArray) -> Tuple[NDArray, NDArray]:
        """A_p and R_p factors for DKH."""
        Ep = self.free_particle_energy(p)
        Ap = np.sqrt((Ep + self.c**2) / (2 * Ep))
        Rp = self.c * p / (Ep + self.c**2)
        return Ap, Rp

    def dkh1_hamiltonian(self, p_grid: NDArray, V_matrix: NDArray) -> NDArray:
        """DKH1 Hamiltonian in momentum-space representation.

        H_DKH1 = E_p δ_{pp′} + A_p(V_{pp'} + R_p V_{pp'} R_{p'})A_{p'}
        """
        n = len(p_grid)
        Ep = self.free_particle_energy(p_grid)
        Ap, Rp = self.kinematic_factors(p_grid)

        H = np.diag(Ep)
        for i in range(n):
            for j in range(n):
                H[i, j] += Ap[i] * (V_matrix[i, j] + Rp[i] * V_matrix[i, j] * Rp[j]) * Ap[j]

        return H

    def dkh2_correction(self, p_grid: NDArray, V_matrix: NDArray) -> NDArray:
        """DKH2 even-even correction (W₁ term).

        W₁ = −½ [O₁, [O₁, E_p]] where O₁ is the DKH1 odd operator.
        Simplified: perturbative estimate.
        """
        n = len(p_grid)
        Ep = self.free_particle_energy(p_grid)
        Ap, Rp = self.kinematic_factors(p_grid)

        # Odd operator O₁ in DKH1
        O1 = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                O1[i, j] = Ap[i] * (Rp[i] * V_matrix[i, j] - V_matrix[i, j] * Rp[j]) * Ap[j]

        # W₁ = −½ [O₁, [O₁, diag(E_p)]]
        E_diag = np.diag(Ep)
        comm1 = O1 @ E_diag - E_diag @ O1
        W1 = -0.5 * (O1 @ comm1 - comm1 @ O1)
        return W1


# ---------------------------------------------------------------------------
#  4-Component Dirac (Minimal Implementation)
# ---------------------------------------------------------------------------

class Dirac4Component:
    r"""
    4-component Dirac equation (radial, hydrogen-like).

    $$\begin{pmatrix} V + c^2 & c\hat{\sigma}\cdot\hat{\mathbf{p}} \\
    c\hat{\sigma}\cdot\hat{\mathbf{p}} & V - c^2 \end{pmatrix}
    \begin{pmatrix} \psi_L \\ \psi_S \end{pmatrix}
    = E\begin{pmatrix} \psi_L \\ \psi_S \end{pmatrix}$$

    For hydrogen-like atoms with nuclear charge Z:
    $$E_{n\kappa} = c^2\left[\left(1 + \frac{(Z\alpha)^2}{(n-|\kappa|+\gamma)^2}\right)^{-1/2} - 1\right]$$
    where $\gamma = \sqrt{\kappa^2 - (Z\alpha)^2}$.
    """

    def __init__(self, Z: int = 1) -> None:
        self.Z = Z
        self.alpha = 1.0 / SPEED_OF_LIGHT  # fine structure constant

    def exact_energy(self, n: int, kappa: int) -> float:
        """Exact Dirac eigenvalue for hydrogen-like atom.

        n: principal quantum number.
        kappa: relativistic quantum number (−1 for 1s, +1 for 2p₁/₂, etc.).
        """
        Za = self.Z * self.alpha
        gamma = math.sqrt(kappa**2 - Za**2)
        nr = n - abs(kappa)  # radial quantum number
        denom = (nr + gamma)**2
        E = SPEED_OF_LIGHT**2 * ((1 + Za**2 / denom)**(-0.5) - 1)
        return E

    def fine_structure_splitting(self, n: int) -> float:
        """ΔE = E(n, j=l+1/2) − E(n, j=l-1/2) for p states."""
        if n < 2:
            return 0.0
        E_j_plus = self.exact_energy(n, kappa=-(n - 1))  # j = l + 1/2
        E_j_minus = self.exact_energy(n, kappa=(n - 1))   # j = l - 1/2
        return abs(E_j_plus - E_j_minus)

    def lamb_shift_estimate(self, n: int) -> float:
        """Leading-order Lamb shift (Welton estimate):
        ΔE_Lamb ≈ (α/π)(Zα)⁴ mc² × (ln(1/(Zα)²) − const)
        for s states.
        """
        Za = self.Z * self.alpha
        return (self.alpha / math.pi) * Za**4 * SPEED_OF_LIGHT**2 * (
            math.log(1 / Za**2) - 1) / n**3

    def radial_dirac_matrix(self, r: NDArray, V: NDArray, kappa: int) -> NDArray:
        """Build radial Dirac equation matrix on grid.

        Coupled ODEs for large (P) and small (Q) components:
        dP/dr = −(κ/r)P + (E − V + 2c²)Q/c
        dQ/dr = +(κ/r)Q − (E − V)P/c
        """
        n = len(r)
        dr = r[1] - r[0]
        H = np.zeros((2 * n, 2 * n))

        for i in range(n):
            ri = max(r[i], 1e-10)
            # P block
            H[i, i] = V[i] + SPEED_OF_LIGHT**2  # V + c² on diagonal (large comp)
            # Q block
            H[n + i, n + i] = V[i] - SPEED_OF_LIGHT**2  # V − c² (small comp)

            # Coupling: c σ·p
            if i > 0:
                H[i, n + i] = SPEED_OF_LIGHT * kappa / ri
                H[n + i, i] = SPEED_OF_LIGHT * kappa / ri
            if i < n - 1:
                H[i, n + i + 1] = SPEED_OF_LIGHT / (2 * dr)
                H[n + i + 1, i] = -SPEED_OF_LIGHT / (2 * dr)

        return H
