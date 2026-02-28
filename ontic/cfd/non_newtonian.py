"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          N O N - N E W T O N I A N   F L U I D   M O D E L S              ║
║                                                                            ║
║  Viscoelastic and yield-stress constitutive models for domain II.8.        ║
║                                                                            ║
║  Viscosity models (generalized Newtonian):                                 ║
║    - Power-law                                                             ║
║    - Carreau-Yasuda                                                        ║
║    - Herschel-Bulkley (yield-stress + power-law)                           ║
║    - Bingham plastic (Papanastasiou regularized)                           ║
║                                                                            ║
║  Viscoelastic models (conformation-tensor based):                          ║
║    - Oldroyd-B with log-conformation tensor                                ║
║    - FENE-P (finitely extensible, Peterlin closure)                        ║
║    - Giesekus (anisotropic drag)                                           ║
║                                                                            ║
║  All viscoelastic models solve the conformation tensor transport:          ║
║    ∂C/∂t + (u·∇)C = C·∇u + (∇u)ᵀ·C + (1/λ)(g(C)·C - I)               ║
║                                                                            ║
║  Log-conformation approach (Fattal & Kupferman 2004):                      ║
║    Ψ = log(C),  evolve Ψ instead of C for numerical stability.            ║
║                                                                            ║
║  References:                                                               ║
║    [1] Bird, Armstrong, Hassager (1987). Dynamics of Polymeric Liquids.   ║
║    [2] Fattal & Kupferman (2004). J. Non-Newton. Fluid Mech. 123, 281.   ║
║    [3] Papanastasiou (1987). J. Rheol. 31, 385.                          ║
║    [4] Hulsen, Fattal, Kupferman (2005). J. Non-Newton. Fluid Mech. 127. ║
║    [5] Giesekus (1982). J. Non-Newton. Fluid Mech. 11, 69.               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from enum import Enum

import torch
from torch import Tensor
import math


# ═══════════════════════════════════════════════════════════════════════════════
#  ABSTRACT INTERFACES
# ═══════════════════════════════════════════════════════════════════════════════

class RheologyModel(ABC):
    """Abstract base for generalized Newtonian models: τ = μ(γ̇)·γ̇."""

    @abstractmethod
    def effective_viscosity(self, shear_rate: Tensor) -> Tensor:
        """
        Compute effective viscosity μ_eff(γ̇).

        Args:
            shear_rate: [... ] scalar shear rate γ̇ = √(2 D:D) (s⁻¹)

        Returns:
            μ_eff: [...] effective viscosity (Pa·s)
        """
        ...

    def extra_stress(self, strain_rate: Tensor) -> Tensor:
        """
        Compute extra stress tensor τ = 2 μ_eff D.

        Args:
            strain_rate: [..., 3, 3] strain rate tensor D = ½(∇u + ∇uᵀ)

        Returns:
            τ: [..., 3, 3] deviatoric stress
        """
        # Scalar shear rate: γ̇ = √(2 D:D)
        D2 = torch.sum(strain_rate * strain_rate, dim=(-2, -1))
        gamma_dot = torch.sqrt(2.0 * D2).clamp(min=1e-30)

        mu = self.effective_viscosity(gamma_dot)
        return 2.0 * mu.unsqueeze(-1).unsqueeze(-1) * strain_rate


class ViscoelasticModel(ABC):
    """
    Abstract base for viscoelastic models evolving conformation tensor C.

    The polymer stress is:
        τ_p = (η_p / λ) (f(C) · C - I)

    where f(C) depends on the model (identity for Oldroyd-B, Peterlin
    for FENE-P, etc.)
    """

    @abstractmethod
    def relaxation_source(self, C: Tensor) -> Tensor:
        """
        Compute the relaxation source term S(C) such that:
            DC/Dt = ... + (1/λ) S(C)

        Args:
            C: [..., 3, 3] conformation tensor (symmetric positive definite)

        Returns:
            S: [..., 3, 3] source (includes -C back to I)
        """
        ...

    @abstractmethod
    def polymer_stress(self, C: Tensor) -> Tensor:
        """
        Compute polymer stress τ_p from conformation tensor.

        Args:
            C: [..., 3, 3] conformation tensor

        Returns:
            τ_p: [..., 3, 3] polymer contribution to extra stress
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════════
#  GENERALIZED NEWTONIAN MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PowerLaw(RheologyModel):
    """
    Power-law (Ostwald de Waele) model.

        μ(γ̇) = K γ̇^(n-1)

    n < 1: shear-thinning (pseudoplastic)
    n = 1: Newtonian
    n > 1: shear-thickening (dilatant)

    Attributes:
        K: Consistency index (Pa·s^n)
        n: Power-law exponent
        mu_min: Minimum viscosity cutoff (Pa·s)
        mu_max: Maximum viscosity cutoff (Pa·s)
    """

    K: float = 1.0
    n: float = 0.5
    mu_min: float = 1e-4
    mu_max: float = 1e4

    def effective_viscosity(self, shear_rate: Tensor) -> Tensor:
        gamma = shear_rate.clamp(min=1e-30)
        mu = self.K * gamma ** (self.n - 1.0)
        return mu.clamp(min=self.mu_min, max=self.mu_max)


@dataclass
class CarreauYasuda(RheologyModel):
    """
    Carreau-Yasuda generalized Newtonian model.

        μ(γ̇) = μ_∞ + (μ₀ - μ_∞) [1 + (λγ̇)^a]^((n-1)/a)

    Parameters:
        mu_0: Zero-shear viscosity (Pa·s)
        mu_inf: Infinite-shear viscosity (Pa·s)
        lam: Relaxation time (s)
        n: Power-law exponent (< 1 for shear-thinning)
        a: Yasuda exponent (a=2 → standard Carreau)
    """

    mu_0: float = 0.056        # Blood default
    mu_inf: float = 0.00345
    lam: float = 3.313         # s
    n: float = 0.3568
    a: float = 2.0

    @classmethod
    def blood(cls) -> "CarreauYasuda":
        """Human blood at 37°C."""
        return cls(mu_0=0.056, mu_inf=0.00345, lam=3.313, n=0.3568, a=2.0)

    @classmethod
    def polymer_solution(cls, concentration: float = 0.01) -> "CarreauYasuda":
        """Typical polymer solution (xanthan gum-like)."""
        return cls(
            mu_0=10.0 * concentration,
            mu_inf=0.001,
            lam=1.0,
            n=0.3,
            a=2.0,
        )

    def effective_viscosity(self, shear_rate: Tensor) -> Tensor:
        gamma = shear_rate.clamp(min=1e-30)
        factor = (1.0 + (self.lam * gamma) ** self.a) ** ((self.n - 1.0) / self.a)
        return self.mu_inf + (self.mu_0 - self.mu_inf) * factor


@dataclass
class HerschelBulkley(RheologyModel):
    """
    Herschel-Bulkley yield-stress model with Papanastasiou regularization.

    Original (singular):
        τ = τ_y + K γ̇^n     if |τ| > τ_y
        γ̇ = 0                if |τ| ≤ τ_y

    Regularized (Papanastasiou 1987):
        μ(γ̇) = K γ̇^(n-1) + τ_y (1 - exp(-m γ̇)) / γ̇

    where m is the regularization parameter (large m → sharp yield).

    Special cases:
        τ_y = 0: Power-law
        n = 1: Bingham plastic
        τ_y = 0, n = 1: Newtonian

    Attributes:
        tau_y: Yield stress (Pa)
        K: Consistency (Pa·s^n)
        n: Flow index
        m: Papanastasiou regularization parameter (s)
    """

    tau_y: float = 10.0      # Pa
    K: float = 1.0            # Pa·s^n
    n: float = 1.0            # Bingham-like
    m: float = 1000.0         # Regularization

    @classmethod
    def bingham(cls, tau_y: float, mu_p: float, m: float = 1000.0) -> "HerschelBulkley":
        """Bingham plastic: τ = τ_y + μ_p γ̇."""
        return cls(tau_y=tau_y, K=mu_p, n=1.0, m=m)

    @classmethod
    def cement_paste(cls) -> "HerschelBulkley":
        """Typical cement paste."""
        return cls(tau_y=50.0, K=10.0, n=0.5, m=1000.0)

    @classmethod
    def toothpaste(cls) -> "HerschelBulkley":
        return cls(tau_y=200.0, K=5.0, n=0.4, m=500.0)

    def effective_viscosity(self, shear_rate: Tensor) -> Tensor:
        gamma = shear_rate.clamp(min=1e-30)

        # Power-law contribution
        mu_pl = self.K * gamma ** (self.n - 1.0)

        # Yield stress contribution (regularized)
        mu_yield = self.tau_y * (1.0 - torch.exp(-self.m * gamma)) / gamma

        return mu_pl + mu_yield

    def is_yielded(self, stress_magnitude: Tensor) -> Tensor:
        """Check if material has yielded: |τ| > τ_y."""
        return stress_magnitude > self.tau_y


@dataclass
class BinghamPlastic(RheologyModel):
    """
    Bingham plastic model (convenience wrapper on Herschel-Bulkley with n=1).

        μ_eff(γ̇) = μ_p + τ_y / γ̇            (exact)
        μ_eff(γ̇) = μ_p + τ_y(1-e^{-mγ̇})/γ̇  (regularized)

    Common materials: mud, concrete, paint, mayonnaise.

    Attributes:
        tau_y: Yield stress (Pa)
        mu_p: Plastic viscosity (Pa·s)
        m: Regularization parameter
    """

    tau_y: float = 10.0
    mu_p: float = 0.1
    m: float = 1000.0

    def effective_viscosity(self, shear_rate: Tensor) -> Tensor:
        gamma = shear_rate.clamp(min=1e-30)
        mu_yield = self.tau_y * (1.0 - torch.exp(-self.m * gamma)) / gamma
        return self.mu_p + mu_yield


# ═══════════════════════════════════════════════════════════════════════════════
#  VISCOELASTIC MODELS (CONFORMATION TENSOR)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OldroydB(ViscoelasticModel):
    """
    Oldroyd-B viscoelastic model with log-conformation tensor.

    The Oldroyd-B constitutive equation:
        τ_p + λ τ̊_p = 2η_p D

    where τ̊ is the upper-convected derivative:
        τ̊ = ∂τ/∂t + (u·∇)τ - (∇u)ᵀ·τ - τ·∇u

    In conformation tensor form:
        ∂C/∂t + (u·∇)C = C·(∇u)ᵀ + ∇u·C + (1/λ)(I - C)

    Polymer stress: τ_p = (η_p/λ)(C - I)

    Log-conformation (Fattal & Kupferman 2004):
        Ψ = log(C),  C = exp(Ψ)
        ∂Ψ/∂t + (u·∇)Ψ = Ω·Ψ - Ψ·Ω + 2B + (1/λ)(exp(-Ψ) - I)

    where Ω and B come from the decomposition:
        ∇u = Ω + B + N·C⁻¹  (Ω antisymmetric, B symmetric on eigenspace of C)

    Attributes:
        eta_s: Solvent viscosity (Pa·s)
        eta_p: Polymer viscosity (Pa·s)
        lam: Relaxation time (s)
    """

    eta_s: float = 0.01     # Solvent
    eta_p: float = 0.99     # Polymer
    lam: float = 1.0        # Relaxation time

    @property
    def eta_total(self) -> float:
        return self.eta_s + self.eta_p

    @property
    def beta(self) -> float:
        """Viscosity ratio β = η_s / (η_s + η_p)."""
        return self.eta_s / (self.eta_s + self.eta_p)

    def weissenberg(self, shear_rate: float) -> float:
        """Weissenberg number Wi = λ γ̇."""
        return self.lam * shear_rate

    def relaxation_source(self, C: Tensor) -> Tensor:
        """
        S(C) = I - C  (relaxation toward equilibrium C = I).
        """
        I = torch.eye(3, dtype=C.dtype, device=C.device)
        if C.dim() > 2:
            I = I.expand_as(C)
        return I - C

    def polymer_stress(self, C: Tensor) -> Tensor:
        """τ_p = (η_p / λ)(C - I)."""
        I = torch.eye(3, dtype=C.dtype, device=C.device)
        if C.dim() > 2:
            I = I.expand_as(C)
        return (self.eta_p / self.lam) * (C - I)

    def log_conformation_decompose(
        self, grad_u: Tensor, C: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Decompose ∇u into Ω (antisymmetric) and B (extension rate) on
        the eigenspace of C, per Fattal-Kupferman decomposition.

        ∇u = Ω + B + N C⁻¹

        Ω: rotation rate relative to C principal axes
        B: extension in C eigendirections

        Args:
            grad_u: [..., 3, 3] velocity gradient
            C: [..., 3, 3] conformation tensor (SPD)

        Returns:
            (Omega, B): Antisymmetric rotation and symmetric extension
        """
        # Eigen decomposition of C
        eigenvalues, R = torch.linalg.eigh(C)
        eigenvalues = eigenvalues.clamp(min=1e-30)
        log_eigenvalues = torch.log(eigenvalues)

        # Transform grad_u to principal axes of C
        # M = Rᵀ ∇u R
        M = torch.einsum("...ji,...jk,...kl->...il", R, grad_u, R)

        Omega_M = torch.zeros_like(M)
        B_M = torch.zeros_like(M)

        # Diagonal of B = diagonal of M (extension rates)
        for i in range(3):
            B_M[..., i, i] = M[..., i, i]

        # Off-diagonal: distribute between Ω and B
        for i in range(3):
            for j in range(i + 1, 3):
                li = eigenvalues[..., i]
                lj = eigenvalues[..., j]

                # log(λ_j) - log(λ_i) / (λ_j - λ_i)
                diff = lj - li
                log_diff = log_eigenvalues[..., j] - log_eigenvalues[..., i]

                # Avoid division by zero when eigenvalues are equal
                safe = diff.abs() > 1e-10
                zeta = torch.where(safe, log_diff / diff, 1.0 / li)

                # B off-diagonal
                sym = 0.5 * (M[..., i, j] + M[..., j, i])
                asym = 0.5 * (M[..., i, j] - M[..., j, i])

                B_M[..., i, j] = zeta * (lj * sym + li * asym) / (0.5 * (li + lj) + 1e-30)
                B_M[..., j, i] = B_M[..., i, j]

                Omega_M[..., i, j] = M[..., i, j] - B_M[..., i, j]
                Omega_M[..., j, i] = -Omega_M[..., i, j]

        # Transform back to global frame
        Omega = torch.einsum("...ij,...jk,...lk->...il", R, Omega_M, R)
        B = torch.einsum("...ij,...jk,...lk->...il", R, B_M, R)

        return Omega, B

    def log_conformation_rhs(
        self, Psi: Tensor, grad_u: Tensor
    ) -> Tensor:
        """
        Right-hand side for log-conformation evolution:

            ∂Ψ/∂t = -u·∇Ψ + (Ω·Ψ - Ψ·Ω) + 2B + (1/λ)(exp(-Ψ) - I)

        This returns the non-advective terms (advection handled by upstream).

        Args:
            Psi: [..., 3, 3] log-conformation tensor
            grad_u: [..., 3, 3] velocity gradient

        Returns:
            dPsi_dt: [..., 3, 3] time derivative (without advection)
        """
        C = _matrix_exp(Psi)
        Omega, B = self.log_conformation_decompose(grad_u, C)

        I = torch.eye(3, dtype=Psi.dtype, device=Psi.device)
        if Psi.dim() > 2:
            I = I.expand_as(Psi)

        # Rotation term: Ω·Ψ - Ψ·Ω
        rotation = Omega @ Psi - Psi @ Omega

        # Extension: 2B
        extension = 2.0 * B

        # Relaxation: (1/λ)(exp(-Ψ) - I)
        exp_neg_Psi = _matrix_exp(-Psi)
        relaxation = (1.0 / self.lam) * (exp_neg_Psi - I)

        return rotation + extension + relaxation

    def conformation_rhs(
        self, C: Tensor, grad_u: Tensor
    ) -> Tensor:
        """
        Standard (non-log) conformation tensor evolution RHS.

            ∂C/∂t = -(u·∇)C + C·(∇u)ᵀ + ∇u·C + (1/λ)(I - C)

        Returns non-advective part.
        """
        I = torch.eye(3, dtype=C.dtype, device=C.device)
        if C.dim() > 2:
            I = I.expand_as(C)

        grad_u_T = grad_u.transpose(-1, -2)

        # Upper-convected terms
        convection = C @ grad_u_T + grad_u @ C

        # Relaxation
        relaxation = (1.0 / self.lam) * (I - C)

        return convection + relaxation


@dataclass
class FENE_P(ViscoelasticModel):
    """
    FENE-P (finitely extensible nonlinear elastic — Peterlin closure).

    The FENE spring force diverges at maximum extension L:
        F(r) = H r / (1 - r²/L²)

    Peterlin closure approximation:
        f(C) = (L² - 3) / (L² - tr(C))

    Conformation evolution:
        ∂C/∂t + (u·∇)C = C·(∇u)ᵀ + ∇u·C + (1/λ)(I - f(C)C)

    Polymer stress:
        τ_p = (η_p/λ)(f(C)C - I)

    As L → ∞, recovers Oldroyd-B.

    Attributes:
        eta_s: Solvent viscosity
        eta_p: Polymer viscosity
        lam: Relaxation time
        L2: Maximum extensibility squared (L²)
    """

    eta_s: float = 0.01
    eta_p: float = 0.99
    lam: float = 1.0
    L2: float = 100.0   # L² = 100 → max extension L = 10

    def peterlin(self, C: Tensor) -> Tensor:
        """
        Peterlin function: f(C) = (L² - 3) / (L² - tr(C)).

        Clamp tr(C) < L² to prevent blowup.
        """
        tr_C = C[..., 0, 0] + C[..., 1, 1] + C[..., 2, 2]
        tr_C = tr_C.clamp(max=self.L2 - 1e-6)
        return (self.L2 - 3.0) / (self.L2 - tr_C)

    def relaxation_source(self, C: Tensor) -> Tensor:
        """S(C) = I - f(C)·C."""
        I = torch.eye(3, dtype=C.dtype, device=C.device)
        if C.dim() > 2:
            I = I.expand_as(C)
        f = self.peterlin(C)
        return I - f.unsqueeze(-1).unsqueeze(-1) * C

    def polymer_stress(self, C: Tensor) -> Tensor:
        """τ_p = (η_p/λ)(f(C)·C - I)."""
        I = torch.eye(3, dtype=C.dtype, device=C.device)
        if C.dim() > 2:
            I = I.expand_as(C)
        f = self.peterlin(C)
        return (self.eta_p / self.lam) * (f.unsqueeze(-1).unsqueeze(-1) * C - I)

    def conformation_rhs(self, C: Tensor, grad_u: Tensor) -> Tensor:
        """
        FENE-P conformation evolution (non-advective RHS):
            C·(∇u)ᵀ + ∇u·C + (1/λ)(I - f(C)·C)
        """
        grad_u_T = grad_u.transpose(-1, -2)
        convection = C @ grad_u_T + grad_u @ C
        relaxation = (1.0 / self.lam) * self.relaxation_source(C)
        return convection + relaxation

    def max_extension_check(self, C: Tensor) -> Tensor:
        """Check if tr(C) approaches L² (chain near full extension)."""
        tr_C = C[..., 0, 0] + C[..., 1, 1] + C[..., 2, 2]
        return tr_C / self.L2


@dataclass
class Giesekus(ViscoelasticModel):
    """
    Giesekus model with anisotropic drag.

    Adds a quadratic stress term to Oldroyd-B:
        τ_p + λ τ̊_p + (α λ / η_p) τ_p² = 2 η_p D

    In conformation form:
        ∂C/∂t + (u·∇)C = C·(∇u)ᵀ + ∇u·C - (1/λ)[(1-α)(C-I) + α(C-I)²]

    The mobility parameter α ∈ [0,1]:
        α = 0: Oldroyd-B
        α = 0.5: Strong shear-thinning
        α = 1: Upper-convected Maxwell (UCM)

    Attributes:
        eta_s: Solvent viscosity
        eta_p: Polymer viscosity
        lam: Relaxation time
        alpha: Giesekus mobility parameter
    """

    eta_s: float = 0.01
    eta_p: float = 0.99
    lam: float = 1.0
    alpha: float = 0.01   # Small: close to Oldroyd-B

    def relaxation_source(self, C: Tensor) -> Tensor:
        """
        S(C) = -[(1-α)(C-I) + α(C-I)²]

        The quadratic term introduces shear-thinning.
        """
        I = torch.eye(3, dtype=C.dtype, device=C.device)
        if C.dim() > 2:
            I = I.expand_as(C)

        CmI = C - I
        CmI2 = CmI @ CmI

        return -((1.0 - self.alpha) * CmI + self.alpha * CmI2)

    def polymer_stress(self, C: Tensor) -> Tensor:
        """τ_p = (η_p/λ)(C - I)."""
        I = torch.eye(3, dtype=C.dtype, device=C.device)
        if C.dim() > 2:
            I = I.expand_as(C)
        return (self.eta_p / self.lam) * (C - I)

    def conformation_rhs(self, C: Tensor, grad_u: Tensor) -> Tensor:
        """Full non-advective RHS for Giesekus."""
        grad_u_T = grad_u.transpose(-1, -2)
        convection = C @ grad_u_T + grad_u @ C
        relaxation = (1.0 / self.lam) * self.relaxation_source(C)
        return convection + relaxation

    def steady_shear_viscosity(self, shear_rate: float) -> float:
        """
        Analytical steady-shear viscosity for Giesekus model.

        At steady shear with rate γ̇:
            η(γ̇) = η_s + η_p f(Wi²)

        where Wi = λγ̇ and f is an algebraic function of α.
        """
        Wi2 = (self.lam * shear_rate) ** 2

        if self.alpha < 1e-10:
            return self.eta_s + self.eta_p  # Oldroyd-B: constant viscosity

        # Giesekus shear-thinning formula
        f_sq = (1.0 + 16.0 * self.alpha * (1.0 - self.alpha) * Wi2)
        f_val = (1.0 - math.sqrt(f_sq)) / (2.0 * self.alpha * Wi2 + 1e-30) if Wi2 > 1e-20 else 1.0
        # Clamp for very high Wi
        f_val = max(f_val, 0.0)

        return self.eta_s + self.eta_p * (1.0 - f_val)


# ═══════════════════════════════════════════════════════════════════════════════
#  MATRIX EXPONENTIAL UTILITY (for log-conformation)
# ═══════════════════════════════════════════════════════════════════════════════

def _matrix_exp(M: Tensor) -> Tensor:
    """
    Matrix exponential for symmetric 3×3 matrices via eigendecomposition.

    exp(M) = R diag(exp(λ₁), exp(λ₂), exp(λ₃)) Rᵀ

    More numerically stable than Padé for SPD matrices.
    """
    if M.dim() == 2:
        eigenvalues, R = torch.linalg.eigh(M)
        return R @ torch.diag(torch.exp(eigenvalues)) @ R.T
    else:
        eigenvalues, R = torch.linalg.eigh(M)
        exp_diag = torch.diag_embed(torch.exp(eigenvalues))
        return R @ exp_diag @ R.transpose(-1, -2)


def _matrix_log(M: Tensor) -> Tensor:
    """
    Matrix logarithm for SPD 3×3 matrices.

    log(M) = R diag(log(λ₁), log(λ₂), log(λ₃)) Rᵀ
    """
    if M.dim() == 2:
        eigenvalues, R = torch.linalg.eigh(M)
        eigenvalues = eigenvalues.clamp(min=1e-30)
        return R @ torch.diag(torch.log(eigenvalues)) @ R.T
    else:
        eigenvalues, R = torch.linalg.eigh(M)
        eigenvalues = eigenvalues.clamp(min=1e-30)
        log_diag = torch.diag_embed(torch.log(eigenvalues))
        return R @ log_diag @ R.transpose(-1, -2)


# ═══════════════════════════════════════════════════════════════════════════════
#  VISCOELASTIC FLOW INTEGRATOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ViscoelasticIntegrator:
    """
    Time integrator for viscoelastic flow with conformation tensor.

    Splits the velocity–pressure–conformation update:

        1. Solve momentum: ρ(∂u/∂t + u·∇u) = -∇p + η_s ∇²u + ∇·τ_p
        2. Continuity: ∇·u = 0 (pressure Poisson)
        3. Evolve conformation: ∂C/∂t + (u·∇)C = RHS(C, ∇u)

    Step 3 uses the log-conformation approach when available
    for numerical stability at high Weissenberg numbers.

    Attributes:
        model: Viscoelastic constitutive model
        use_log: Whether to use log-conformation transform
    """

    model: ViscoelasticModel
    use_log: bool = True

    def step_conformation(
        self,
        C: Tensor,
        grad_u: Tensor,
        dt: float,
    ) -> Tensor:
        """
        Advance conformation tensor by one timestep (non-advective).

        Uses forward Euler or RK2 depending on stability needs.

        Args:
            C: [..., 3, 3] current conformation tensor
            grad_u: [..., 3, 3] velocity gradient at cell centers
            dt: Timestep

        Returns:
            C_new: [..., 3, 3] updated conformation tensor (SPD guaranteed)
        """
        if self.use_log and isinstance(self.model, OldroydB):
            return self._step_log_conformation(C, grad_u, dt)
        else:
            return self._step_standard(C, grad_u, dt)

    def _step_standard(
        self, C: Tensor, grad_u: Tensor, dt: float,
    ) -> Tensor:
        """Standard explicit RK2 step for conformation tensor."""
        if hasattr(self.model, "conformation_rhs"):
            rhs = self.model.conformation_rhs(C, grad_u)
        else:
            grad_u_T = grad_u.transpose(-1, -2)
            rhs = (
                C @ grad_u_T + grad_u @ C
                + (1.0 / getattr(self.model, "lam", 1.0)) * self.model.relaxation_source(C)
            )

        # Heun's method (RK2)
        C_tilde = C + dt * rhs

        # Ensure SPD
        C_tilde = _ensure_spd(C_tilde)

        if hasattr(self.model, "conformation_rhs"):
            rhs_tilde = self.model.conformation_rhs(C_tilde, grad_u)
        else:
            rhs_tilde = (
                C_tilde @ grad_u.transpose(-1, -2) + grad_u @ C_tilde
                + (1.0 / getattr(self.model, "lam", 1.0)) * self.model.relaxation_source(C_tilde)
            )

        C_new = C + 0.5 * dt * (rhs + rhs_tilde)
        return _ensure_spd(C_new)

    def _step_log_conformation(
        self, C: Tensor, grad_u: Tensor, dt: float,
    ) -> Tensor:
        """
        Log-conformation time step (Fattal-Kupferman).

        Ψ = log(C)
        Ψ_{n+1} = Ψ_n + dt × RHS_log(Ψ_n, ∇u)
        C_{n+1} = exp(Ψ_{n+1})

        This is unconditionally stable for SPD — no eigenvalue clamping needed.
        """
        Psi = _matrix_log(C)

        dPsi_dt = self.model.log_conformation_rhs(Psi, grad_u)

        Psi_new = Psi + dt * dPsi_dt

        return _matrix_exp(Psi_new)


def _ensure_spd(C: Tensor, min_eigenvalue: float = 1e-8) -> Tensor:
    """
    Project tensor to nearest SPD matrix (clamp eigenvalues).

    C_spd = R max(Λ, ε I) Rᵀ
    """
    eigenvalues, R = torch.linalg.eigh(C)
    eigenvalues = eigenvalues.clamp(min=min_eigenvalue)
    if C.dim() == 2:
        return R @ torch.diag(eigenvalues) @ R.T
    else:
        return R @ torch.diag_embed(eigenvalues) @ R.transpose(-1, -2)


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Abstract
    "RheologyModel",
    "ViscoelasticModel",
    # Generalized Newtonian
    "PowerLaw",
    "CarreauYasuda",
    "HerschelBulkley",
    "BinghamPlastic",
    # Viscoelastic
    "OldroydB",
    "FENE_P",
    "Giesekus",
    # Integrator
    "ViscoelasticIntegrator",
    # Utilities
    "_matrix_exp",
    "_matrix_log",
    "_ensure_spd",
]
