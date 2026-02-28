"""
Polymer & Soft-Matter Physics — chain statistics, Flory-Huggins,
self-consistent field theory, reptation, rubber elasticity.

Domain XIV.6 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Ideal Chain Statistics
# ---------------------------------------------------------------------------

class IdealChainStatistics:
    r"""
    Gaussian chain model for polymer conformations.

    End-to-end distance:
    $$\langle R^2 \rangle = N b^2$$

    Radius of gyration:
    $$\langle R_g^2 \rangle = \frac{Nb^2}{6}$$

    Probability distribution:
    $$P(\mathbf{R}, N) = \left(\frac{3}{2\pi N b^2}\right)^{3/2}
      \exp\left(-\frac{3R^2}{2Nb^2}\right)$$

    Kuhn length $b$, number of Kuhn segments $N$.
    """

    def __init__(self, N: int = 1000, b: float = 1.0) -> None:
        """
        N: number of Kuhn segments.
        b: Kuhn length (nm).
        """
        self.N = N
        self.b = b

    def end_to_end_rms(self) -> float:
        """√⟨R²⟩ = b√N."""
        return self.b * math.sqrt(self.N)

    def radius_of_gyration(self) -> float:
        """R_g = b√(N/6)."""
        return self.b * math.sqrt(self.N / 6)

    def end_distribution(self, R: float) -> float:
        """P(R) = (3/(2πNb²))^{3/2} exp(−3R²/(2Nb²))."""
        Nb2 = self.N * self.b**2
        prefactor = (3 / (2 * math.pi * Nb2))**1.5
        return prefactor * math.exp(-3 * R**2 / (2 * Nb2))

    def structure_factor(self, q: NDArray) -> NDArray:
        """Debye function: g(x) = 2(e^{−x} − 1 + x)/x² where x = q²R_g²."""
        Rg2 = self.N * self.b**2 / 6
        x = q**2 * Rg2
        x_safe = np.maximum(x, 1e-10)
        return 2 * (np.exp(-x_safe) - 1 + x_safe) / x_safe**2

    def generate_walk(self, rng: Optional[np.random.Generator] = None) -> NDArray:
        """Generate 3D random walk (N+1 positions)."""
        if rng is None:
            rng = np.random.default_rng()
        steps = rng.normal(0, self.b / math.sqrt(3), (self.N, 3))
        positions = np.zeros((self.N + 1, 3))
        positions[1:] = np.cumsum(steps, axis=0)
        return positions


# ---------------------------------------------------------------------------
#  Flory-Huggins Theory
# ---------------------------------------------------------------------------

class FloryHuggins:
    r"""
    Flory-Huggins lattice theory for polymer blends.

    Free energy of mixing per lattice site:
    $$\frac{f_{\text{mix}}}{k_BT} = \frac{\phi}{N_A}\ln\phi
      + \frac{1-\phi}{N_B}\ln(1-\phi) + \chi\phi(1-\phi)$$

    Spinodal condition: $\partial^2 f / \partial\phi^2 = 0$
    $$\frac{1}{N_A\phi} + \frac{1}{N_B(1-\phi)} = 2\chi$$

    Critical point: $\chi_c = \frac{1}{2}\left(\frac{1}{\sqrt{N_A}}+\frac{1}{\sqrt{N_B}}\right)^2$
    """

    def __init__(self, N_A: int = 100, N_B: int = 100, chi: float = 0.1) -> None:
        self.N_A = N_A
        self.N_B = N_B
        self.chi = chi

    def free_energy_mixing(self, phi: float) -> float:
        """f_mix / k_BT per site."""
        if phi <= 0 or phi >= 1:
            return 0.0
        return (phi / self.N_A * math.log(phi)
                + (1 - phi) / self.N_B * math.log(1 - phi)
                + self.chi * phi * (1 - phi))

    def chemical_potential(self, phi: float) -> float:
        """μ_A = ∂f/∂φ."""
        if phi <= 0 or phi >= 1:
            return 0.0
        return ((1 + math.log(phi)) / self.N_A
                - (1 + math.log(1 - phi)) / self.N_B
                + self.chi * (1 - 2 * phi))

    def spinodal(self, n_pts: int = 200) -> Tuple[NDArray, NDArray]:
        """Spinodal curve χ_s(φ).

        1/(N_A φ) + 1/(N_B(1−φ)) = 2χ_s
        """
        phi = np.linspace(0.01, 0.99, n_pts)
        chi_s = 0.5 * (1.0 / (self.N_A * phi) + 1.0 / (self.N_B * (1 - phi)))
        return phi, chi_s

    def critical_point(self) -> Tuple[float, float]:
        """(φ_c, χ_c) — critical composition and interaction parameter."""
        sqrt_A = math.sqrt(self.N_A)
        sqrt_B = math.sqrt(self.N_B)
        phi_c = sqrt_A / (sqrt_A + sqrt_B)  # for symmetric, ~0.5
        chi_c = 0.5 * (1 / sqrt_A + 1 / sqrt_B)**2
        return phi_c, chi_c

    def binodal_symmetric(self) -> Tuple[float, float]:
        """Binodal for symmetric blend (N_A = N_B = N).

        φ tanh(χN(1−2φ)/2) = ... solved numerically.
        """
        from scipy.optimize import brentq

        N = self.N_A

        def equation(phi: float) -> float:
            if phi <= 0.01 or phi >= 0.49:
                return 1.0
            return (math.log(phi) - math.log(1 - phi)
                    + self.chi * N * (1 - 2 * phi))

        phi_low = brentq(equation, 0.01, 0.49)
        return phi_low, 1 - phi_low


# ---------------------------------------------------------------------------
#  Self-Consistent Field Theory (1D)
# ---------------------------------------------------------------------------

class SCFT1D:
    r"""
    Self-consistent field theory for diblock copolymer (AB).

    Modified diffusion equation:
    $$\frac{\partial q}{\partial s} = \frac{b^2}{6}\nabla^2 q - w(r)q$$

    Self-consistency:
    $$w_A = \chi N \phi_B + \xi, \quad w_B = \chi N \phi_A + \xi$$
    $$\phi_A + \phi_B = 1$$

    Edwards propagator $q(r, s)$ solved on spatial grid.
    """

    def __init__(self, n_grid: int = 64, L: float = 10.0,
                 N: int = 100, f: float = 0.5, chi_N: float = 20.0) -> None:
        """
        f: A-block fraction.
        chi_N: Flory-Huggins parameter × N.
        """
        self.n = n_grid
        self.L = L
        self.dx = L / n_grid
        self.N = N
        self.f = f
        self.chi_N = chi_N
        self.ds = 1.0 / N

        self.wA = np.zeros(n_grid)
        self.wB = np.zeros(n_grid)
        self.phiA = np.full(n_grid, f)
        self.phiB = np.full(n_grid, 1 - f)

    def propagate(self, w: NDArray) -> NDArray:
        """Solve forward diffusion equation for propagator q(r, s).

        Returns q(r, s=1) by Crank-Nicolson in s.
        """
        n = self.n
        q = np.ones(n)
        b2_6 = 1.0 / 6  # b = 1 in non-dim
        alpha = b2_6 * self.ds / self.dx**2

        for _ in range(self.N):
            lap = (np.roll(q, 1) + np.roll(q, -1) - 2 * q) / self.dx**2
            q = q + self.ds * (b2_6 * lap - w * q)
            q = np.maximum(q, 1e-15)

        return q

    def compute_densities(self) -> Tuple[NDArray, NDArray]:
        """Compute φ_A, φ_B from propagators."""
        qf = self.propagate(self.wA)
        qr = self.propagate(self.wB)

        Q = float(np.mean(qf * qr))
        if Q < 1e-30:
            Q = 1e-30

        self.phiA = qf * qr / Q * self.f
        self.phiB = 1 - self.phiA
        self.phiB = np.maximum(self.phiB, 0)
        return self.phiA, self.phiB

    def update_fields(self, lambda_mix: float = 0.1) -> None:
        """Update w_A, w_B via simple mixing."""
        wA_new = self.chi_N * self.phiB
        wB_new = self.chi_N * self.phiA
        xi = 0.5 * (wA_new + wB_new)
        self.wA = (1 - lambda_mix) * self.wA + lambda_mix * (wA_new - xi)
        self.wB = (1 - lambda_mix) * self.wB + lambda_mix * (wB_new - xi)

    def iterate(self, max_iter: int = 200, tol: float = 1e-5) -> int:
        """SCFT iteration loop.

        Returns number of iterations.
        """
        for it in range(max_iter):
            phiA_old = self.phiA.copy()
            self.compute_densities()
            self.update_fields()
            delta = float(np.max(np.abs(self.phiA - phiA_old)))
            if delta < tol:
                return it + 1
        return max_iter


# ---------------------------------------------------------------------------
#  Reptation & Tube Model
# ---------------------------------------------------------------------------

class ReptationModel:
    r"""
    Doi-Edwards reptation model for entangled polymer dynamics.

    Tube diameter: $a \propto N_e^{1/2} b$

    Reptation time (disengagement):
    $$\tau_d = \frac{\zeta N^3 b^2}{p^2 k_B T N_e}$$

    Diffusion coefficient:
    $$D_{\text{rep}} = \frac{R_g^2}{\tau_d} \propto N^{-2}$$

    Viscosity: $\eta_0 \propto N^{3.4}$ (experimentally 3.4, theory gives 3).

    Dynamic moduli (Maxwell model):
    $$G'(\omega) = G_N^0\sum_p \frac{(\omega\tau_p)^2}{1+(\omega\tau_p)^2}$$
    $$G''(\omega) = G_N^0\sum_p \frac{\omega\tau_p}{1+(\omega\tau_p)^2}$$

    $G_N^0 = \rho R T / M_e$ = plateau modulus.
    """

    def __init__(self, N: int = 1000, N_e: int = 50, b: float = 1.0,
                 zeta: float = 1e-11, T: float = 400.0) -> None:
        """
        N: chain length (Kuhn segments).
        N_e: entanglement length.
        b: Kuhn length (nm).
        zeta: monomeric friction coefficient (N·s/m).
        T: temperature (K).
        """
        self.N = N
        self.N_e = N_e
        self.b = b * 1e-9  # m
        self.zeta = zeta
        self.T = T
        self.kBT = 1.381e-23 * T

    def tube_diameter(self) -> float:
        """a = b √N_e (metres)."""
        return self.b * math.sqrt(self.N_e)

    def reptation_time(self) -> float:
        """τ_d = ζ N³ b² / (π² k_BT N_e) (seconds)."""
        return (self.zeta * self.N**3 * self.b**2
                / (math.pi**2 * self.kBT * self.N_e))

    def diffusion_coefficient(self) -> float:
        """D_rep = R_g² / τ_d (m²/s)."""
        Rg2 = self.N * self.b**2 / 6
        return Rg2 / self.reptation_time()

    def plateau_modulus(self, rho: float = 1000.0, M0: float = 0.1) -> float:
        """G_N⁰ = ρ k_BT / (N_e M₀) (Pa).

        rho: density (kg/m³), M0: monomer mass (kg/mol).
        """
        return rho * self.kBT / (self.N_e * M0 / 6.022e23)

    def dynamic_moduli(self, omega: NDArray, n_modes: int = 50) -> Tuple[NDArray, NDArray]:
        """G'(ω) and G''(ω) from reptation spectrum."""
        tau_d = self.reptation_time()
        G0 = self.plateau_modulus()

        Gp = np.zeros_like(omega)
        Gpp = np.zeros_like(omega)

        for p in range(1, n_modes + 1, 2):
            tau_p = tau_d / p**2
            ot = omega * tau_p
            Gp += 8 / (p**2 * math.pi**2) * ot**2 / (1 + ot**2)
            Gpp += 8 / (p**2 * math.pi**2) * ot / (1 + ot**2)

        return G0 * Gp, G0 * Gpp

    def zero_shear_viscosity(self) -> float:
        """η₀ = G_N⁰ × τ_d × (π²/12) (Pa·s)."""
        return self.plateau_modulus() * self.reptation_time() * math.pi**2 / 12


# ---------------------------------------------------------------------------
#  Rubber Elasticity
# ---------------------------------------------------------------------------

class RubberElasticity:
    r"""
    Rubber elasticity — affine and phantom network models.

    Affine network:
    $$\sigma_{\text{eng}} = \nu k_BT\left(\lambda - \frac{1}{\lambda^2}\right)$$

    where $\nu$ = crosslink density, $\lambda$ = stretch ratio.

    Neo-Hookean strain energy:
    $$W = \frac{G}{2}(I_1 - 3), \quad I_1 = \lambda_1^2+\lambda_2^2+\lambda_3^2$$

    Mooney-Rivlin:
    $$W = C_1(I_1 - 3) + C_2(I_2 - 3)$$
    """

    def __init__(self, nu: float = 1e23, T: float = 300.0) -> None:
        """
        nu: crosslink density (m⁻³).
        T: temperature (K).
        """
        self.nu = nu
        self.T = T
        self.kBT = 1.381e-23 * T
        self.G = nu * self.kBT  # shear modulus

    def engineering_stress(self, lam: float) -> float:
        """σ_eng = νk_BT(λ − 1/λ²)  (Pa)."""
        return self.G * (lam - 1.0 / lam**2)

    def true_stress(self, lam: float) -> float:
        """σ_true = νk_BT(λ² − 1/λ)  (Pa)."""
        return self.G * (lam**2 - 1.0 / lam)

    def neo_hookean_energy(self, lam1: float, lam2: float, lam3: float) -> float:
        """W = (G/2)(I₁ − 3)."""
        I1 = lam1**2 + lam2**2 + lam3**2
        return 0.5 * self.G * (I1 - 3)

    def mooney_rivlin(self, lam: float, C1: float = None,
                         C2: float = None) -> float:
        """Mooney-Rivlin uniaxial stress.

        σ = 2(C₁ + C₂/λ)(λ − 1/λ²)
        """
        if C1 is None:
            C1 = self.G / 2
        if C2 is None:
            C2 = 0.0
        return 2 * (C1 + C2 / lam) * (lam - 1 / lam**2)

    def swelling_ratio(self, chi: float, N_s: int = 100) -> float:
        """Equilibrium swelling (Flory-Rehner).

        ln(1−φ_p) + φ_p + χφ_p² = −ν V_s (φ_p^{1/3} − φ_p/2)
        Solved iteratively for polymer volume fraction φ_p.
        """
        from scipy.optimize import brentq

        V_s = 1e-4  # molar volume of solvent (m³/mol, approximate)

        def equation(phi_p: float) -> float:
            if phi_p <= 0.01 or phi_p >= 0.99:
                return 1.0
            return (math.log(1 - phi_p) + phi_p + chi * phi_p**2
                    + self.nu * V_s / 6.022e23 * (phi_p**(1 / 3) - phi_p / 2))

        phi_p = brentq(equation, 0.01, 0.99)
        return 1.0 / phi_p  # Q = V/V₀ = 1/φ_p
