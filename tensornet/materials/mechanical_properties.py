"""
Mechanical Properties of Materials: elastic tensor, Frenkel theoretical
strength, Griffith fracture, Paris fatigue, creep models.

Upgrades domain XIV.2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Elastic Stiffness Tensor
# ---------------------------------------------------------------------------

class ElasticTensor:
    r"""
    Full 6×6 Voigt-notation elastic stiffness tensor $C_{ij}$.

    Cubic symmetry: 3 independent constants (C₁₁, C₁₂, C₄₄).
    Hexagonal: 5 constants (C₁₁, C₁₂, C₁₃, C₃₃, C₄₄).
    Isotropic: 2 constants (λ, μ Lamé parameters).

    Derived:
    - Young's modulus E, Poisson ratio ν, shear modulus G, bulk modulus K
    - Voigt-Reuss-Hill averages for polycrystals
    - Zener anisotropy ratio A = 2C₄₄/(C₁₁ - C₁₂)
    """

    def __init__(self, C: NDArray[np.float64]) -> None:
        """
        Parameters
        ----------
        C : 6×6 elastic stiffness matrix (GPa) in Voigt notation.
        """
        assert C.shape == (6, 6), "C must be 6×6"
        self.C = C.copy()
        self.S = np.linalg.inv(C)  # Compliance

    @classmethod
    def from_cubic(cls, C11: float, C12: float, C44: float) -> "ElasticTensor":
        """Construct from cubic symmetry constants (GPa)."""
        C = np.zeros((6, 6))
        C[0, 0] = C[1, 1] = C[2, 2] = C11
        C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = C12
        C[3, 3] = C[4, 4] = C[5, 5] = C44
        return cls(C)

    @classmethod
    def from_isotropic(cls, E: float, nu: float) -> "ElasticTensor":
        """Construct from Young's modulus E (GPa) and Poisson ratio ν."""
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        C = np.zeros((6, 6))
        C[0, 0] = C[1, 1] = C[2, 2] = lam + 2 * mu
        C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = lam
        C[3, 3] = C[4, 4] = C[5, 5] = mu
        return cls(C)

    def voigt_bulk_modulus(self) -> float:
        """Voigt (upper bound) bulk modulus K_V."""
        C = self.C
        return ((C[0, 0] + C[1, 1] + C[2, 2])
                + 2 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9.0

    def reuss_bulk_modulus(self) -> float:
        """Reuss (lower bound) bulk modulus K_R."""
        S = self.S
        return 1.0 / ((S[0, 0] + S[1, 1] + S[2, 2])
                       + 2 * (S[0, 1] + S[0, 2] + S[1, 2]))

    def voigt_shear_modulus(self) -> float:
        """Voigt (upper bound) shear modulus G_V."""
        C = self.C
        return ((C[0, 0] + C[1, 1] + C[2, 2])
                - (C[0, 1] + C[0, 2] + C[1, 2])
                + 3 * (C[3, 3] + C[4, 4] + C[5, 5])) / 15.0

    def reuss_shear_modulus(self) -> float:
        """Reuss (lower bound) shear modulus G_R."""
        S = self.S
        num = 15.0
        den = (4 * (S[0, 0] + S[1, 1] + S[2, 2])
               - 4 * (S[0, 1] + S[0, 2] + S[1, 2])
               + 3 * (S[3, 3] + S[4, 4] + S[5, 5]))
        return num / den if abs(den) > 1e-30 else 0.0

    def hill_averages(self) -> Dict[str, float]:
        """Voigt-Reuss-Hill averages."""
        K_V = self.voigt_bulk_modulus()
        K_R = self.reuss_bulk_modulus()
        G_V = self.voigt_shear_modulus()
        G_R = self.reuss_shear_modulus()
        K_H = 0.5 * (K_V + K_R)
        G_H = 0.5 * (G_V + G_R)
        E_H = 9 * K_H * G_H / (3 * K_H + G_H) if (3 * K_H + G_H) > 0 else 0
        nu_H = (3 * K_H - 2 * G_H) / (6 * K_H + 2 * G_H) if (6 * K_H + 2 * G_H) > 0 else 0
        return {
            "K_Voigt": K_V, "K_Reuss": K_R, "K_Hill": K_H,
            "G_Voigt": G_V, "G_Reuss": G_R, "G_Hill": G_H,
            "E_Hill": E_H, "nu_Hill": nu_H,
        }

    def zener_anisotropy(self) -> float:
        """A = 2C₄₄/(C₁₁ - C₁₂). A=1 for isotropic."""
        diff = self.C[0, 0] - self.C[0, 1]
        if abs(diff) < 1e-30:
            return float('inf')
        return 2.0 * self.C[3, 3] / diff

    def christoffel_velocities(self, n: NDArray[np.float64],
                                  rho: float) -> NDArray[np.float64]:
        """
        Phase velocities from Christoffel equation.

        Γ_ik = C_ijkl n_j n_l / ρ, eigenvalues → v² = λ.
        """
        # Build Christoffel matrix from Voigt C
        # Full index mapping
        voigt_map = {(0, 0): 0, (1, 1): 1, (2, 2): 2,
                     (1, 2): 3, (2, 1): 3, (0, 2): 4,
                     (2, 0): 4, (0, 1): 5, (1, 0): 5}

        Gamma = np.zeros((3, 3))
        for i in range(3):
            for k in range(3):
                for j in range(3):
                    for l_idx in range(3):
                        ij = voigt_map.get((i, j), -1)
                        kl = voigt_map.get((k, l_idx), -1)
                        if ij >= 0 and kl >= 0:
                            Gamma[i, k] += self.C[ij, kl] * n[j] * n[l_idx]

        Gamma /= rho
        eigenvalues = np.linalg.eigvalsh(Gamma)
        return np.sqrt(np.maximum(eigenvalues, 0.0))


# ---------------------------------------------------------------------------
#  Frenkel Theoretical Strength
# ---------------------------------------------------------------------------

class FrenkelStrength:
    r"""
    Frenkel (1926) theoretical shear strength.

    $$\tau_{\max} = \frac{G}{2\pi}\approx \frac{G}{5\text{--}30}$$

    More refined: $\tau_{\max} = \frac{Gb}{2\pi d}$
    where b = Burgers vector, d = interplanar spacing.

    Theoretical tensile strength:
    $$\sigma_{\max} = \sqrt{\frac{E\gamma_s}{d}}$$
    where γₛ = surface energy.
    """

    @staticmethod
    def shear_strength(G: float, b: Optional[float] = None,
                        d: Optional[float] = None) -> float:
        """Theoretical shear strength (GPa).

        Parameters: G in GPa, b,d in Å (optional for refined estimate).
        """
        if b is not None and d is not None:
            return G * b / (2.0 * math.pi * d)
        return G / (2.0 * math.pi)

    @staticmethod
    def tensile_strength(E: float, gamma_s: float, d: float) -> float:
        """Theoretical tensile (cleavage) strength.

        Parameters: E (GPa), γₛ (J/m²), d (Å → converted to m).
        """
        d_m = d * 1e-10
        return math.sqrt(E * 1e9 * gamma_s / d_m) / 1e9  # GPa


# ---------------------------------------------------------------------------
#  Griffith Fracture Mechanics
# ---------------------------------------------------------------------------

class GriffithFracture:
    r"""
    Griffith (1921) energy-balance fracture criterion.

    Critical stress: $\sigma_c = \sqrt{\frac{2E\gamma_s}{\pi a}}$

    Stress intensity factor: $K_I = \sigma\sqrt{\pi a}$

    Fracture toughness: $K_{Ic} = \sqrt{2E\gamma_s}$ (plane stress)

    Irwin: $G_c = K_{Ic}^2 / E$ (plane stress) or $K_{Ic}^2(1-\nu^2)/E$ (plane strain).
    """

    def __init__(self, E: float, gamma_s: float,
                 nu: float = 0.3) -> None:
        """
        Parameters
        ----------
        E : Young's modulus (GPa).
        gamma_s : Surface energy (J/m²).
        nu : Poisson's ratio.
        """
        self.E = E * 1e9  # Pa
        self.gamma_s = gamma_s
        self.nu = nu

    def critical_stress(self, a: float) -> float:
        """σ_c for central crack of half-length a (m). Returns MPa."""
        return math.sqrt(2.0 * self.E * self.gamma_s / (math.pi * a)) / 1e6

    def stress_intensity(self, sigma: float, a: float) -> float:
        """K_I = σ√(πa). sigma in MPa, a in m. Returns MPa·√m."""
        return sigma * math.sqrt(math.pi * a)

    def fracture_toughness_plane_stress(self) -> float:
        """K_Ic = √(2Eγₛ). Returns MPa·√m."""
        return math.sqrt(2.0 * self.E * self.gamma_s) / 1e6

    def fracture_toughness_plane_strain(self) -> float:
        """K_Ic for plane strain. Returns MPa·√m."""
        return math.sqrt(2.0 * self.E * self.gamma_s / (1.0 - self.nu**2)) / 1e6

    def energy_release_rate(self, K: float) -> float:
        """G = K² / E (plane stress). K in MPa√m, returns J/m²."""
        return (K * 1e6)**2 / self.E

    def plastic_zone_size(self, K: float, sigma_y: float) -> float:
        """Irwin plastic zone: r_p = (1/2π)(K/σ_y)². Returns m."""
        return (K / sigma_y)**2 / (2.0 * math.pi)


# ---------------------------------------------------------------------------
#  Paris Fatigue Law
# ---------------------------------------------------------------------------

class ParisFatigue:
    r"""
    Paris-Erdogan fatigue crack growth law.

    $$\frac{da}{dN} = C(\Delta K)^m$$

    where ΔK = stress intensity range, C and m are material constants.

    Typical: m ≈ 2-4, C depends on units.

    Integration: $N_f = \int_{a_0}^{a_f} \frac{da}{C(\Delta K)^m}$
    """

    def __init__(self, C: float = 1e-12, m: float = 3.0) -> None:
        """
        Parameters
        ----------
        C : Paris coefficient (m/cycle per (MPa√m)^m).
        m : Paris exponent.
        """
        self.C = C
        self.m = m

    def growth_rate(self, delta_K: float) -> float:
        """da/dN (m/cycle) for given ΔK (MPa√m)."""
        return self.C * delta_K**self.m

    def cycles_to_failure(self, a0: float, af: float,
                            sigma_range: float,
                            Y: float = 1.0,
                            n_steps: int = 10000) -> float:
        """
        Integrate Paris law from a₀ to a_f.

        ΔK = Y Δσ √(πa), Y = geometry factor.
        """
        da = (af - a0) / n_steps
        a = a0
        N = 0.0

        for _ in range(n_steps):
            delta_K = Y * sigma_range * math.sqrt(math.pi * a)
            if delta_K < 1e-30:
                break
            dN = da / (self.C * delta_K**self.m)
            N += dN
            a += da
            if a >= af:
                break

        return N

    def crack_growth_curve(self, a0: float, sigma_range: float,
                             N_max: int = 100000,
                             Y: float = 1.0) -> Tuple[NDArray, NDArray]:
        """a(N) crack length vs cycles."""
        a_arr = [a0]
        N_arr = [0]
        a = a0

        for i in range(1, N_max):
            delta_K = Y * sigma_range * math.sqrt(math.pi * a)
            da = self.C * delta_K**self.m
            a += da
            a_arr.append(a)
            N_arr.append(i)

            if a > 10.0 * a0:  # Arbitrary runaway limit
                break

        return np.array(N_arr), np.array(a_arr)


# ---------------------------------------------------------------------------
#  Creep Models
# ---------------------------------------------------------------------------

class CreepModel:
    r"""
    Creep deformation models.

    Power-law (Dorn/Norton): $\dot\varepsilon = A\sigma^n\exp(-Q/RT)$

    Coble (diffusion): $\dot\varepsilon = A_{Coble}\sigma\Omega D_b/(k_B T d^3)$

    Nabarro-Herring: $\dot\varepsilon = A_{NH}\sigma\Omega D_v/(k_B T d^2)$
    """

    @staticmethod
    def norton_power_law(sigma: float, T: float,
                           A: float = 1e-10, n: float = 5.0,
                           Q: float = 250e3) -> float:
        """Norton power-law creep rate ε̇ (s⁻¹).

        Parameters
        ----------
        sigma : Stress (MPa).
        T : Temperature (K).
        A : Pre-exponential (s⁻¹ MPa⁻ⁿ).
        n : Stress exponent.
        Q : Activation energy (J/mol).
        """
        R = 8.314
        return A * sigma**n * math.exp(-Q / (R * T))

    @staticmethod
    def nabarro_herring(sigma: float, T: float,
                          d: float, D_v: float,
                          Omega: float = 1.2e-29) -> float:
        """Nabarro-Herring (lattice diffusion) creep rate.

        Parameters
        ----------
        sigma : Stress (Pa).
        T : Temperature (K).
        d : Grain size (m).
        D_v : Volume diffusivity (m²/s).
        Omega : Atomic volume (m³).
        """
        k_B = 1.381e-23
        return 14.0 * sigma * Omega * D_v / (k_B * T * d**2)

    @staticmethod
    def coble(sigma: float, T: float,
              d: float, D_b: float, delta_b: float = 5e-10,
              Omega: float = 1.2e-29) -> float:
        """Coble (grain boundary diffusion) creep rate.

        Parameters
        ----------
        sigma : Stress (Pa).
        T : Temperature (K).
        d : Grain size (m).
        D_b : GB diffusivity (m²/s).
        delta_b : GB width (m).
        Omega : Atomic volume (m³).
        """
        k_B = 1.381e-23
        return 47.0 * sigma * Omega * D_b * delta_b / (k_B * T * d**3)

    @staticmethod
    def deformation_mechanism_map(sigma_range: NDArray[np.float64],
                                    T: float, d: float,
                                    D_v: float, D_b: float,
                                    A_pl: float = 1e-10,
                                    n_pl: float = 5.0,
                                    Q_pl: float = 250e3) -> Dict[str, NDArray]:
        """
        Compute strain rates for all mechanisms.
        Returns dict of arrays for plotting mechanism maps.
        """
        nh = np.array([CreepModel.nabarro_herring(s * 1e6, T, d, D_v) for s in sigma_range])
        cb = np.array([CreepModel.coble(s * 1e6, T, d, D_b) for s in sigma_range])
        pl = np.array([CreepModel.norton_power_law(s, T, A_pl, n_pl, Q_pl) for s in sigma_range])
        total = nh + cb + pl

        return {
            "nabarro_herring": nh,
            "coble": cb,
            "power_law": pl,
            "total": total,
        }
