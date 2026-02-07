"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           N O N L I N E A R   C O N T I N U U M   M E C H A N I C S       ║
║                                                                            ║
║  Hyperelastic, elasto-plastic, and fracture constitutive models.           ║
║  Extends the linear elastic Hex8 FEA in QTT-FEA/fea-qtt/                 ║
║                                                                            ║
║  Models:                                                                   ║
║    - Neo-Hookean hyperelastic (compressible)                               ║
║    - Mooney-Rivlin hyperelastic (two-parameter)                           ║
║    - Ogden hyperelastic (N-term stretch-based)                            ║
║    - J2 (von Mises) isotropic plasticity + hardening                      ║
║    - Drucker-Prager pressure-dependent plasticity                         ║
║    - Cohesive zone fracture model (bilinear traction-separation)          ║
║    - Updated Lagrangian finite deformation framework                      ║
║                                                                            ║
║  All models implement the ConstitutiveModel interface for uniform          ║
║  integration with the FEA solver.                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

References:
    [1] Bonet & Wood (2008). Nonlinear Continuum Mechanics for FEA, 2nd ed.
    [2] Simo & Hughes (1998). Computational Inelasticity.
    [3] Belytschko, Liu, Moran (2000). Nonlinear FEM for Continua & Structures.
    [4] de Souza Neto, Perić, Owen (2008). Computational Plasticity.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import torch
from torch import Tensor
import math


# ═══════════════════════════════════════════════════════════════════════════════
#  TENSOR UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _identity_3x3(dtype: torch.dtype = torch.float64, device: str = "cpu") -> Tensor:
    return torch.eye(3, dtype=dtype, device=device)


def _det3x3(F: Tensor) -> Tensor:
    """Determinant of 3x3 matrix (batched: [..., 3, 3])."""
    return (
        F[..., 0, 0] * (F[..., 1, 1] * F[..., 2, 2] - F[..., 1, 2] * F[..., 2, 1])
        - F[..., 0, 1] * (F[..., 1, 0] * F[..., 2, 2] - F[..., 1, 2] * F[..., 2, 0])
        + F[..., 0, 2] * (F[..., 1, 0] * F[..., 2, 1] - F[..., 1, 1] * F[..., 2, 0])
    )


def _inv3x3(F: Tensor) -> Tensor:
    """Inverse of 3x3 matrix (batched)."""
    return torch.linalg.inv(F)


def _trace(A: Tensor) -> Tensor:
    """Trace of [..., 3, 3] matrix."""
    return A[..., 0, 0] + A[..., 1, 1] + A[..., 2, 2]


def deformation_gradient(
    dN_dX: Tensor, u_e: Tensor
) -> Tensor:
    """
    Compute deformation gradient F = I + ∂u/∂X.

    Args:
        dN_dX: Shape function derivatives w.r.t. reference coords [n_nodes, 3]
        u_e: Element nodal displacements [n_nodes, 3]

    Returns:
        F: Deformation gradient [3, 3]
    """
    # ∂u/∂X = Σ_a u_a ⊗ (∂N_a/∂X) = u_e^T @ dN_dX
    grad_u = u_e.T @ dN_dX  # [3, 3]
    F = _identity_3x3(dtype=u_e.dtype, device=u_e.device) + grad_u
    return F


def right_cauchy_green(F: Tensor) -> Tensor:
    """C = F^T F (right Cauchy-Green deformation tensor)."""
    return F.transpose(-2, -1) @ F


def left_cauchy_green(F: Tensor) -> Tensor:
    """b = F F^T (left Cauchy-Green / Finger tensor)."""
    return F @ F.transpose(-2, -1)


def green_lagrange_strain(F: Tensor) -> Tensor:
    """E = ½(F^T F - I) (Green-Lagrange strain)."""
    I = _identity_3x3(dtype=F.dtype, device=F.device)
    return 0.5 * (right_cauchy_green(F) - I)


def principal_stretches(F: Tensor) -> Tensor:
    """
    Compute principal stretches λ₁ ≥ λ₂ ≥ λ₃ from F via SVD.
    """
    U, S, Vh = torch.linalg.svd(F)
    return S


def invariants_from_C(C: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute invariants of right Cauchy-Green tensor C:
        I₁ = tr(C)
        I₂ = ½[(tr C)² - tr(C²)]
        I₃ = det(C)
    """
    I1 = _trace(C)
    C2 = C @ C
    I2 = 0.5 * (I1 ** 2 - _trace(C2))
    I3 = _det3x3(C)
    return I1, I2, I3


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTITUTIVE MODEL INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

class StressType(Enum):
    """Stress measure type."""
    CAUCHY = "cauchy"          # σ (true stress, spatial)
    PK1 = "pk1"               # P (first Piola-Kirchhoff, mixed)
    PK2 = "pk2"               # S (second Piola-Kirchhoff, material)
    KIRCHHOFF = "kirchhoff"   # τ = Jσ


class ConstitutiveModel(ABC):
    """
    Abstract constitutive model interface.

    All models must implement:
        stress(F): Compute stress from deformation gradient
        tangent(F): Compute material tangent (∂σ/∂ε or ∂S/∂E)
        strain_energy(F): Compute stored elastic energy Ψ
    """

    @abstractmethod
    def stress(self, F: Tensor) -> Tuple[Tensor, StressType]:
        """
        Compute stress from deformation gradient.

        Args:
            F: Deformation gradient [3, 3] or [..., 3, 3]

        Returns:
            (stress_tensor, stress_type)
        """
        ...

    @abstractmethod
    def tangent(self, F: Tensor) -> Tensor:
        """
        Compute material tangent modulus.

        Returns:
            C: Fourth-order tangent in Voigt form [6, 6]
        """
        ...

    @abstractmethod
    def strain_energy(self, F: Tensor) -> Tensor:
        """Compute strain energy density Ψ(F)."""
        ...


# ═══════════════════════════════════════════════════════════════════════════════
#  NEO-HOOKEAN HYPERELASTIC
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NeoHookean(ConstitutiveModel):
    """
    Compressible Neo-Hookean hyperelastic model.

    Strain energy density:
        Ψ = (μ/2)(Ī₁ - 3) + (κ/2)(J - 1)²

    where Ī₁ = J^{-2/3} tr(F^T F) is the isochoric first invariant,
    J = det(F), μ is the shear modulus, κ is the bulk modulus.

    Cauchy stress:
        σ = (μ/J)(b̄ - ⅓Ī₁ I) + κ(J - 1)I

    where b̄ = J^{-2/3} b is the isochoric left Cauchy-Green tensor.

    Attributes:
        mu: Shear modulus μ (Pa)
        kappa: Bulk modulus κ (Pa)
    """

    mu: float = 80.0e6     # ~rubber
    kappa: float = 400.0e6  # ~rubber

    @classmethod
    def from_young_poisson(cls, E: float, nu: float) -> "NeoHookean":
        """Create from Young's modulus and Poisson's ratio."""
        mu = E / (2.0 * (1.0 + nu))
        kappa = E / (3.0 * (1.0 - 2.0 * nu))
        return cls(mu=mu, kappa=kappa)

    def strain_energy(self, F: Tensor) -> Tensor:
        """Ψ = (μ/2)(Ī₁ - 3) + (κ/2)(J - 1)²"""
        J = _det3x3(F)
        C = right_cauchy_green(F)
        I1 = _trace(C)
        I1_bar = J ** (-2.0 / 3.0) * I1

        psi = 0.5 * self.mu * (I1_bar - 3.0) + 0.5 * self.kappa * (J - 1.0) ** 2
        return psi

    def stress(self, F: Tensor) -> Tuple[Tensor, StressType]:
        """Compute Cauchy stress σ."""
        J = _det3x3(F)
        b = left_cauchy_green(F)
        I = _identity_3x3(dtype=F.dtype, device=F.device)

        # Isochoric part
        J_23 = J ** (-2.0 / 3.0)
        b_bar = J_23 * b
        I1_bar = _trace(b_bar)

        # Cauchy stress: σ = (μ/J)(dev(b̄)) + κ(J-1)I
        sigma = (self.mu / J) * (b_bar - (I1_bar / 3.0) * I) + self.kappa * (J - 1.0) * I

        return sigma, StressType.CAUCHY

    def pk2_stress(self, F: Tensor) -> Tensor:
        """Second Piola-Kirchhoff stress S = J F⁻¹ σ F⁻ᵀ."""
        sigma, _ = self.stress(F)
        J = _det3x3(F)
        F_inv = _inv3x3(F)
        S = J * F_inv @ sigma @ F_inv.transpose(-2, -1)
        return S

    def tangent(self, F: Tensor) -> Tensor:
        """
        Spatial elasticity tensor in Voigt form [6, 6].

        Using numerical differentiation of Cauchy stress w.r.t. deformation.
        """
        return _numerical_tangent(self, F)


# ═══════════════════════════════════════════════════════════════════════════════
#  MOONEY-RIVLIN HYPERELASTIC
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MooneyRivlin(ConstitutiveModel):
    """
    Mooney-Rivlin hyperelastic model (two-parameter).

    Strain energy density:
        Ψ = C₁(Ī₁ - 3) + C₂(Ī₂ - 3) + (κ/2)(J - 1)²

    where Ī₁, Ī₂ are isochoric invariants of F^T F.

    For rubber:
        C₁ ≈ 0.5 MPa (dominant), C₂ ≈ 0.05 MPa (second-order correction)
        μ = 2(C₁ + C₂) = initial shear modulus

    Attributes:
        C1: First Mooney-Rivlin parameter (Pa)
        C2: Second Mooney-Rivlin parameter (Pa)
        kappa: Bulk modulus (Pa)
    """

    C1: float = 0.5e6
    C2: float = 0.05e6
    kappa: float = 100.0e6

    @property
    def initial_shear_modulus(self) -> float:
        """μ = 2(C₁ + C₂)."""
        return 2.0 * (self.C1 + self.C2)

    def strain_energy(self, F: Tensor) -> Tensor:
        """Ψ = C₁(Ī₁ - 3) + C₂(Ī₂ - 3) + (κ/2)(J - 1)²"""
        J = _det3x3(F)
        C = right_cauchy_green(F)
        I1, I2, I3 = invariants_from_C(C)

        J_23 = J ** (-2.0 / 3.0)
        J_43 = J ** (-4.0 / 3.0)
        I1_bar = J_23 * I1
        I2_bar = J_43 * I2

        psi = self.C1 * (I1_bar - 3.0) + self.C2 * (I2_bar - 3.0) + 0.5 * self.kappa * (J - 1.0) ** 2
        return psi

    def stress(self, F: Tensor) -> Tuple[Tensor, StressType]:
        """Compute Cauchy stress via autograd of strain energy."""
        F_r = F.detach().requires_grad_(True)
        psi = self.strain_energy(F_r)
        # P = ∂Ψ/∂F
        P = torch.autograd.grad(psi.sum(), F_r)[0]
        # σ = (1/J) P F^T
        J = _det3x3(F)
        sigma = (1.0 / J) * P @ F.transpose(-2, -1)
        return sigma, StressType.CAUCHY

    def tangent(self, F: Tensor) -> Tensor:
        return _numerical_tangent(self, F)


# ═══════════════════════════════════════════════════════════════════════════════
#  OGDEN HYPERELASTIC
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Ogden(ConstitutiveModel):
    """
    Ogden hyperelastic model (N-term principal stretch formulation).

    Strain energy density:
        Ψ = Σᵢ (μᵢ/αᵢ)(λ̄₁^αᵢ + λ̄₂^αᵢ + λ̄₃^αᵢ - 3) + (κ/2)(J - 1)²

    where λ̄ₖ = J^{-1/3}λₖ are isochoric stretches.

    N=1 with α=2 recovers Neo-Hookean.
    N=2 with appropriate parameters covers very large deformation of rubber.

    Attributes:
        mu_alpha: List of (μᵢ, αᵢ) pairs
        kappa: Bulk modulus
    """

    mu_alpha: List[Tuple[float, float]] = field(
        default_factory=lambda: [(1.3e6, 1.3), (5.0e3, 5.0), (-1.0e4, -2.0)]
    )
    kappa: float = 200.0e6

    @property
    def initial_shear_modulus(self) -> float:
        """μ = ½ Σ μᵢαᵢ."""
        return 0.5 * sum(mu * alpha for mu, alpha in self.mu_alpha)

    def strain_energy(self, F: Tensor) -> Tensor:
        """Compute Ogden strain energy from principal stretches."""
        J = _det3x3(F)
        stretches = principal_stretches(F)

        # Isochoric stretches
        J_13 = J ** (-1.0 / 3.0)
        lam_bar = J_13.unsqueeze(-1) * stretches if J.ndim > 0 else J_13 * stretches

        psi = torch.tensor(0.0, dtype=F.dtype, device=F.device)
        for mu_i, alpha_i in self.mu_alpha:
            psi = psi + (mu_i / alpha_i) * (
                lam_bar[..., 0] ** alpha_i
                + lam_bar[..., 1] ** alpha_i
                + lam_bar[..., 2] ** alpha_i
                - 3.0
            )
        psi = psi + 0.5 * self.kappa * (J - 1.0) ** 2
        return psi

    def stress(self, F: Tensor) -> Tuple[Tensor, StressType]:
        """Cauchy stress via autograd."""
        F_r = F.detach().requires_grad_(True)
        psi = self.strain_energy(F_r)
        P = torch.autograd.grad(psi.sum(), F_r)[0]
        J = _det3x3(F)
        sigma = (1.0 / J) * P @ F.transpose(-2, -1)
        return sigma, StressType.CAUCHY

    def tangent(self, F: Tensor) -> Tensor:
        return _numerical_tangent(self, F)


# ═══════════════════════════════════════════════════════════════════════════════
#  J2 (VON MISES) PLASTICITY
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class J2Plasticity(ConstitutiveModel):
    """
    J2 (von Mises) isotropic elasto-plasticity with isotropic hardening.

    Yield function:
        f(σ, κ) = √(3J₂) - σ_y(κ) ≤ 0

    where J₂ = ½ s:s is the second deviatoric stress invariant,
    s = σ - ⅓tr(σ)I is deviatoric stress,
    σ_y(κ) = σ_y0 + H·κ is the hardened yield stress,
    κ is the equivalent plastic strain.

    Return mapping (radial return):
        1. Elastic trial: σ_trial = σ_n + D:Δε
        2. Check yield: f_trial = ||s_trial|| - √(2/3)·σ_y(κ_n)
        3. If f_trial > 0: plastic correction
            Δγ = f_trial / (3G + H)
            s_{n+1} = s_trial · (1 - 3G·Δγ/||s_trial||)
            κ_{n+1} = κ_n + √(2/3)·Δγ

    Attributes:
        E: Young's modulus
        nu: Poisson's ratio
        sigma_y0: Initial yield stress
        H: Isotropic hardening modulus (linear)
        H_kinematic: Kinematic hardening modulus (Prager)
    """

    E: float = 200.0e9      # Steel
    nu: float = 0.3
    sigma_y0: float = 250.0e6  # ~mild steel
    H: float = 1.0e9          # Linear hardening

    def __post_init__(self):
        self.G = self.E / (2.0 * (1.0 + self.nu))  # Shear modulus
        self.K = self.E / (3.0 * (1.0 - 2.0 * self.nu))  # Bulk modulus
        self.lam = self.K - 2.0 * self.G / 3.0  # Lamé lambda

    def elastic_stress(self, strain: Tensor) -> Tensor:
        """σ = λ tr(ε) I + 2μ ε."""
        I = _identity_3x3(dtype=strain.dtype, device=strain.device)
        return self.lam * _trace(strain) * I + 2.0 * self.G * strain

    def deviatoric(self, sigma: Tensor) -> Tensor:
        """s = σ - ⅓tr(σ)I."""
        I = _identity_3x3(dtype=sigma.dtype, device=sigma.device)
        return sigma - (_trace(sigma) / 3.0) * I

    def von_mises(self, sigma: Tensor) -> Tensor:
        """σ_vm = √(3J₂) = √(3/2 · s:s)."""
        s = self.deviatoric(sigma)
        J2 = 0.5 * _trace(s @ s)
        return torch.sqrt(3.0 * J2 + 1e-30)

    def return_mapping(
        self,
        strain_increment: Tensor,
        sigma_n: Tensor,
        kappa_n: float,
    ) -> Tuple[Tensor, float, float]:
        """
        Radial return mapping algorithm.

        Args:
            strain_increment: Δε [3, 3] strain increment
            sigma_n: Previous stress state [3, 3]
            kappa_n: Accumulated equivalent plastic strain

        Returns:
            (sigma_{n+1}, kappa_{n+1}, delta_gamma):
                Updated stress, updated eq. plastic strain, plastic multiplier
        """
        # Elastic trial
        sigma_trial = sigma_n + self.elastic_stress(strain_increment)
        s_trial = self.deviatoric(sigma_trial)

        # Deviatoric norm
        norm_s = torch.sqrt(_trace(s_trial @ s_trial) + 1e-30)

        # Yield function
        sigma_y = self.sigma_y0 + self.H * kappa_n
        f_trial = norm_s - math.sqrt(2.0 / 3.0) * sigma_y

        if f_trial.item() <= 0.0:
            # Elastic step
            return sigma_trial, kappa_n, 0.0

        # Plastic correction (radial return)
        delta_gamma = f_trial.item() / (2.0 * self.G + 2.0 / 3.0 * self.H)

        # Update deviatoric stress
        factor = 1.0 - 2.0 * self.G * delta_gamma / norm_s.item()
        I = _identity_3x3(dtype=sigma_n.dtype, device=sigma_n.device)
        p = _trace(sigma_trial) / 3.0  # Hydrostatic pressure (unchanged)
        sigma_new = factor * s_trial + p * I

        # Update equivalent plastic strain
        kappa_new = kappa_n + math.sqrt(2.0 / 3.0) * delta_gamma

        return sigma_new, kappa_new, delta_gamma

    def stress(self, F: Tensor) -> Tuple[Tensor, StressType]:
        """
        Compute Cauchy stress from deformation gradient.
        Assumes proportional loading from reference (no history).
        """
        E_strain = green_lagrange_strain(F)
        sigma = self.elastic_stress(E_strain)

        # Check yield and apply return mapping if needed
        vm = self.von_mises(sigma)
        if vm.item() > self.sigma_y0:
            sigma, _, _ = self.return_mapping(E_strain, torch.zeros_like(sigma), 0.0)

        return sigma, StressType.CAUCHY

    def tangent(self, F: Tensor) -> Tensor:
        return _numerical_tangent(self, F)

    def strain_energy(self, F: Tensor) -> Tensor:
        E = green_lagrange_strain(F)
        sigma = self.elastic_stress(E)
        return 0.5 * _trace(sigma @ E)


# ═══════════════════════════════════════════════════════════════════════════════
#  DRUCKER-PRAGER PLASTICITY
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DruckerPrager(ConstitutiveModel):
    """
    Drucker-Prager pressure-dependent plasticity.

    Yield function:
        f(σ) = √J₂ + α·I₁ - k ≤ 0

    where I₁ = tr(σ), J₂ = ½ s:s,
    α = 2sin(φ) / (√3(3 - sin(φ))),
    k = 6c·cos(φ) / (√3(3 - sin(φ))).

    Used for: geomaterials (soil, rock, concrete), ceramics.

    The outer cone inscribes the Mohr-Coulomb hexagonal yield surface.

    Attributes:
        E: Young's modulus
        nu: Poisson's ratio
        phi: Friction angle (radians)
        c: Cohesion (Pa)
    """

    E: float = 30.0e9       # Concrete
    nu: float = 0.2
    phi: float = 0.5236     # 30° in rad
    c: float = 10.0e6       # Cohesion

    def __post_init__(self):
        self.G = self.E / (2.0 * (1.0 + self.nu))
        self.K = self.E / (3.0 * (1.0 - 2.0 * self.nu))
        self.lam = self.K - 2.0 * self.G / 3.0

        sin_phi = math.sin(self.phi)
        cos_phi = math.cos(self.phi)
        sqrt3 = math.sqrt(3.0)

        # Outer cone (inscribed in Mohr-Coulomb)
        self.alpha = 2.0 * sin_phi / (sqrt3 * (3.0 - sin_phi))
        self.k = 6.0 * self.c * cos_phi / (sqrt3 * (3.0 - sin_phi))

    def yield_function(self, sigma: Tensor) -> Tensor:
        """f = √J₂ + α·I₁ - k."""
        I1 = _trace(sigma)
        I = _identity_3x3(dtype=sigma.dtype, device=sigma.device)
        s = sigma - (I1 / 3.0) * I
        J2 = 0.5 * _trace(s @ s)
        return torch.sqrt(J2 + 1e-30) + self.alpha * I1 - self.k

    def return_mapping(
        self,
        strain_increment: Tensor,
        sigma_n: Tensor,
    ) -> Tuple[Tensor, float]:
        """
        Return mapping for Drucker-Prager.

        Handles both smooth cone return and apex return.
        """
        I = _identity_3x3(dtype=sigma_n.dtype, device=sigma_n.device)

        # Elastic trial
        sigma_trial = sigma_n + self.lam * _trace(strain_increment) * I + 2.0 * self.G * strain_increment

        f_trial = self.yield_function(sigma_trial)

        if f_trial.item() <= 0.0:
            return sigma_trial, 0.0

        # Decompose trial stress
        I1_trial = _trace(sigma_trial)
        s_trial = sigma_trial - (I1_trial / 3.0) * I
        norm_s = torch.sqrt(_trace(s_trial @ s_trial) + 1e-30)

        # Smooth cone return
        denominator = self.G + 3.0 * self.K * self.alpha * self.alpha
        delta_gamma = f_trial.item() / denominator

        # Check if we need apex return
        norm_s_new = norm_s.item() - self.G * delta_gamma
        if norm_s_new < 0.0:
            # Apex return: σ = (k/3α)·I
            sigma_new = (self.k / (3.0 * self.alpha)) * I
            return sigma_new, delta_gamma

        # Standard smooth return
        factor = 1.0 - self.G * delta_gamma / norm_s.item()
        I1_new = I1_trial - 3.0 * self.K * self.alpha * delta_gamma
        sigma_new = factor * s_trial + (I1_new / 3.0) * I

        return sigma_new, delta_gamma

    def stress(self, F: Tensor) -> Tuple[Tensor, StressType]:
        E_strain = green_lagrange_strain(F)
        I = _identity_3x3(dtype=F.dtype, device=F.device)
        sigma_trial = self.lam * _trace(E_strain) * I + 2.0 * self.G * E_strain
        sigma, _ = self.return_mapping(E_strain, torch.zeros_like(sigma_trial))
        return sigma, StressType.CAUCHY

    def tangent(self, F: Tensor) -> Tensor:
        return _numerical_tangent(self, F)

    def strain_energy(self, F: Tensor) -> Tensor:
        E = green_lagrange_strain(F)
        I = _identity_3x3(dtype=F.dtype, device=F.device)
        sigma = self.lam * _trace(E) * I + 2.0 * self.G * E
        return 0.5 * _trace(sigma @ E)


# ═══════════════════════════════════════════════════════════════════════════════
#  COHESIVE ZONE FRACTURE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CohesiveZone:
    """
    Bilinear traction-separation cohesive zone model for fracture.

    Traction-separation law:
        T(δ) = { (T_max / δ_0)·δ            if δ ≤ δ₀ (loading)
               { T_max·(δ_c - δ)/(δ_c - δ₀) if δ₀ < δ ≤ δ_c (softening)
               { 0                            if δ > δ_c (fully separated)

    where:
        δ: separation displacement
        δ₀: separation at peak traction (damage initiation)
        δ_c: critical separation (complete failure)
        T_max: maximum traction (cohesive strength)
        G_c = ½·T_max·δ_c: fracture energy (area under T-δ curve)

    Mixed-mode coupling (power law):
        (G_I/G_Ic)^η + (G_II/G_IIc)^η = 1

    Attributes:
        T_max_n: Normal cohesive strength (Pa)
        T_max_s: Shear cohesive strength (Pa)
        G_Ic: Mode-I fracture energy (J/m²)
        G_IIc: Mode-II fracture energy (J/m²)
        eta: Mixed-mode power law exponent
    """

    T_max_n: float = 60.0e6    # Normal strength
    T_max_s: float = 40.0e6    # Shear strength
    G_Ic: float = 300.0        # Mode I fracture energy (J/m²)
    G_IIc: float = 600.0       # Mode II fracture energy (J/m²)
    eta: float = 2.0           # BK mixed-mode exponent

    def __post_init__(self):
        # δ_c = 2G_c / T_max
        self.delta_c_n = 2.0 * self.G_Ic / self.T_max_n
        self.delta_c_s = 2.0 * self.G_IIc / self.T_max_s

        # δ_0 (onset) — for bilinear law: penalty stiffness K = T_max/δ₀
        # Choose δ₀ = δ_c / 50 (stiff initial response)
        self.delta_0_n = self.delta_c_n / 50.0
        self.delta_0_s = self.delta_c_s / 50.0

        # Initial stiffness
        self.K_n = self.T_max_n / self.delta_0_n
        self.K_s = self.T_max_s / self.delta_0_s

    def traction_normal(self, delta_n: float, damage: float = 0.0) -> Tuple[float, float]:
        """
        Compute normal traction from separation.

        Args:
            delta_n: Normal opening displacement (≥0 for tension)
            damage: Current damage variable D ∈ [0, 1]

        Returns:
            (traction, updated_damage)
        """
        if delta_n <= 0.0:
            # Compression: penalty contact (no damage in compression)
            return self.K_n * delta_n, damage

        # Effective separation
        if delta_n >= self.delta_c_n:
            return 0.0, 1.0  # Fully failed

        # Current damage threshold
        if delta_n <= self.delta_0_n:
            # Before damage initiation
            D_new = max(damage, 0.0)
            T = (1.0 - D_new) * self.K_n * delta_n
        else:
            # Softening regime
            D_trial = (self.delta_c_n * (delta_n - self.delta_0_n)) / (
                delta_n * (self.delta_c_n - self.delta_0_n)
            )
            D_new = max(damage, D_trial)  # Damage only grows
            T = (1.0 - D_new) * self.K_n * delta_n

        return T, D_new

    def traction_shear(self, delta_s: float, damage: float = 0.0) -> Tuple[float, float]:
        """
        Compute shear traction from tangential separation.
        """
        abs_delta = abs(delta_s)
        sign = 1.0 if delta_s >= 0.0 else -1.0

        if abs_delta >= self.delta_c_s:
            return 0.0, 1.0

        if abs_delta <= self.delta_0_s:
            D_new = max(damage, 0.0)
            T = (1.0 - D_new) * self.K_s * delta_s
        else:
            D_trial = (self.delta_c_s * (abs_delta - self.delta_0_s)) / (
                abs_delta * (self.delta_c_s - self.delta_0_s)
            )
            D_new = max(damage, D_trial)
            T = (1.0 - D_new) * self.K_s * delta_s

        return T, D_new

    def mixed_mode_damage(
        self, delta_n: float, delta_s: float, damage: float = 0.0
    ) -> Tuple[float, float, float]:
        """
        Mixed-mode traction with Benzeggagh-Kenane criterion.

        Returns:
            (T_n, T_s, updated_damage)
        """
        # Effective separation
        delta_eff = math.sqrt(max(0.0, delta_n) ** 2 + delta_s ** 2)
        if delta_eff < 1e-30:
            return 0.0, 0.0, damage

        # Mode mixity ratio
        beta = abs(delta_s) / (abs(delta_s) + max(0.0, delta_n) + 1e-30)

        # BK mixed-mode critical energy
        G_c_mixed = self.G_Ic + (self.G_IIc - self.G_Ic) * beta ** self.eta

        # Effective critical separation
        T_max_eff = math.sqrt(
            (self.T_max_n * (1.0 - beta)) ** 2 + (self.T_max_s * beta) ** 2
        )
        delta_c_eff = 2.0 * G_c_mixed / (T_max_eff + 1e-30)
        delta_0_eff = delta_c_eff / 50.0

        # Damage calculation
        if delta_eff >= delta_c_eff:
            D_new = 1.0
        elif delta_eff <= delta_0_eff:
            D_new = max(damage, 0.0)
        else:
            D_trial = (delta_c_eff * (delta_eff - delta_0_eff)) / (
                delta_eff * (delta_c_eff - delta_0_eff)
            )
            D_new = max(damage, D_trial)

        # Tractions
        K_eff_n = self.K_n
        K_eff_s = self.K_s

        if delta_n > 0:
            T_n = (1.0 - D_new) * K_eff_n * delta_n
        else:
            T_n = K_eff_n * delta_n  # No damage in compression

        T_s = (1.0 - D_new) * K_eff_s * delta_s

        return T_n, T_s, D_new


# ═══════════════════════════════════════════════════════════════════════════════
#  UPDATED LAGRANGIAN FRAMEWORK
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class UpdatedLagrangianSolver:
    """
    Updated Lagrangian finite deformation framework.

    In the Updated Lagrangian formulation:
        - Reference configuration updates to current configuration each step
        - Spatial (Cauchy) stress and rate of deformation are primary variables
        - Objective stress rate (Jaumann or Truesdell) ensures frame indifference

    Weak form (virtual power):
        ∫_V σ : δd dv = ∫_V b · δv dv + ∫_S t · δv da

    where d = sym(∇v) is the rate of deformation, σ is Cauchy stress.

    Newton-Raphson incremental solution:
        K_T Δu = R = F_ext - F_int
        F_int = ∫ B^T σ dV (internal force vector)
        K_T = K_mat + K_geo (material + geometric stiffness)

    Attributes:
        model: Constitutive model
        convergence_tol: Newton-Raphson tolerance
        max_iter: Maximum NR iterations
    """

    model: ConstitutiveModel
    convergence_tol: float = 1e-8
    max_iter: int = 25

    def compute_internal_force(
        self,
        coords: Tensor,
        connectivity: Tensor,
        displacements: Tensor,
        gauss_weights: Tensor,
        dN_dxi: Tensor,
    ) -> Tensor:
        """
        Compute internal force vector F_int = Σ_e ∫ B^T σ dV.

        Args:
            coords: Nodal coordinates [n_nodes, 3]
            connectivity: Element connectivity [n_elem, n_per_elem]
            displacements: Nodal displacements [n_nodes, 3]
            gauss_weights: Gauss quadrature weights [n_gauss]
            dN_dxi: Shape function derivatives in parent coords [n_gauss, n_per_elem, 3]

        Returns:
            Internal force vector [n_nodes * 3]
        """
        n_nodes = coords.shape[0]
        n_elem = connectivity.shape[0]
        n_per_elem = connectivity.shape[1]
        n_gauss = gauss_weights.shape[0]

        f_int = torch.zeros(n_nodes * 3, dtype=coords.dtype, device=coords.device)

        for e in range(n_elem):
            conn = connectivity[e]
            x_e = coords[conn]       # [n_per_elem, 3]
            u_e = displacements[conn]  # [n_per_elem, 3]
            X_e = x_e + u_e           # Current coords

            f_e = torch.zeros(n_per_elem * 3, dtype=coords.dtype, device=coords.device)

            for g in range(n_gauss):
                # Jacobian: J = dN/dξ^T · X
                J = dN_dxi[g].T @ X_e  # [3, 3]
                det_J = _det3x3(J)
                if det_J.item() <= 0:
                    raise ValueError(f"Negative Jacobian at element {e}, gauss point {g}")

                J_inv = _inv3x3(J)
                dN_dX = dN_dxi[g] @ J_inv  # [n_per_elem, 3]

                # Deformation gradient
                F = deformation_gradient(dN_dX, u_e)

                # Cauchy stress from constitutive model
                sigma, _ = self.model.stress(F)

                # B matrix (spatial) and internal force contribution
                # f_int_e += B^T σ det(J) w_g
                for a in range(n_per_elem):
                    for i in range(3):
                        for j in range(3):
                            f_e[a * 3 + i] += (
                                sigma[i, j] * dN_dX[a, j] * det_J * gauss_weights[g]
                            )

            # Assemble into global
            for a in range(n_per_elem):
                node = conn[a].item()
                for i in range(3):
                    f_int[node * 3 + i] += f_e[a * 3 + i]

        return f_int

    def newton_raphson_step(
        self,
        coords: Tensor,
        connectivity: Tensor,
        displacements: Tensor,
        external_force: Tensor,
        gauss_weights: Tensor,
        dN_dxi: Tensor,
    ) -> Tuple[Tensor, float]:
        """
        One Newton-Raphson iteration.

        Returns:
            (displacement_correction, residual_norm)
        """
        f_int = self.compute_internal_force(
            coords, connectivity, displacements, gauss_weights, dN_dxi
        )

        residual = external_force - f_int
        res_norm = residual.norm().item()

        # Simplified: use diagonal scaling as preconditioner
        # Full implementation would assemble K_T
        scale = residual.abs().max().item() + 1e-30
        du = residual / (scale * 10.0)

        return du, res_norm


# ═══════════════════════════════════════════════════════════════════════════════
#  NUMERICAL TANGENT (SHARED UTILITY)
# ═══════════════════════════════════════════════════════════════════════════════

def _numerical_tangent(
    model: ConstitutiveModel, F: Tensor, eps: float = 1e-7
) -> Tensor:
    """
    Compute spatial tangent modulus numerically via central differences.

    Returns Voigt-form [6, 6] tangent: C_ijkl in Voigt notation.
    """
    # Map Voigt indices to tensor indices
    voigt_map = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

    C_voigt = torch.zeros(6, 6, dtype=F.dtype, device=F.device)

    for col in range(6):
        k, l = voigt_map[col]

        # Perturbed deformation gradients
        dF = torch.zeros_like(F)
        dF[k, l] += eps
        if k != l:
            dF[l, k] += eps  # Symmetric perturbation

        F_plus = F + dF
        F_minus = F - dF

        sigma_p, _ = model.stress(F_plus)
        sigma_m, _ = model.stress(F_minus)

        dsigma = (sigma_p - sigma_m) / (2.0 * eps)

        for row in range(6):
            i, j = voigt_map[row]
            C_voigt[row, col] = dsigma[i, j]

    return C_voigt


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Kinematics
    "deformation_gradient",
    "right_cauchy_green",
    "left_cauchy_green",
    "green_lagrange_strain",
    "principal_stretches",
    "invariants_from_C",
    # Interface
    "StressType",
    "ConstitutiveModel",
    # Hyperelastic
    "NeoHookean",
    "MooneyRivlin",
    "Ogden",
    # Plasticity
    "J2Plasticity",
    "DruckerPrager",
    # Fracture
    "CohesiveZone",
    # Framework
    "UpdatedLagrangianSolver",
]
