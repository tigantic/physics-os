"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              S T R U C T U R A L   M E C H A N I C S                       ║
║                                                                            ║
║  Beam, plate, and shell elements with modal analysis.                      ║
║  Covers I.4 of the 140-domain taxonomy.                                    ║
║                                                                            ║
║  Elements:                                                                 ║
║    - Euler-Bernoulli beam (2-node, 4 DOF/node in 2D)                      ║
║    - Timoshenko beam (2-node, shear-deformable)                            ║
║    - Mindlin-Reissner plate (4-node quad, 3 DOF/node)                     ║
║                                                                            ║
║  Analysis types:                                                           ║
║    - Static (KΔu = F)                                                      ║
║    - Eigenvalue buckling (K + λK_g)φ = 0                                   ║
║    - Modal analysis (K - ω²M)φ = 0 via Lanczos                           ║
║    - Composite CLT with Tsai-Wu failure criterion                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

References:
    [1] Cook, Malkus, Plesha, Witt (2002). Concepts & Applications of FEA, 4th ed.
    [2] Bathe, K.J. (2014). Finite Element Procedures, 2nd ed.
    [3] Reddy, J.N. (2004). Mechanics of Laminated Composite Plates, 2nd ed.
    [4] Hughes, T.J.R. (2000). The Finite Element Method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import torch
from torch import Tensor
import math


# ═══════════════════════════════════════════════════════════════════════════════
#  BEAM ELEMENTS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TimoshenkoBeam:
    """
    Timoshenko beam element (2-node, shear-deformable).

    DOFs per node (2D): [w, θ] (transverse displacement, rotation)
    DOFs per node (3D): [u, v, w, θx, θy, θz] (6 DOF)

    Stiffness includes both bending and shear:
        K = K_bending + K_shear

    where:
        K_bending = EI/L³ [...] (cubic Hermite)
        K_shear = κGA/L [...] (shear correction)

    For κ = 5/6 (rectangular), κ = 6/7 (circular).

    The Timoshenko beam reduces to Euler-Bernoulli as G → ∞.

    Attributes:
        E: Young's modulus (Pa)
        G: Shear modulus (Pa)
        A: Cross-section area (m²)
        I: Second moment of area (m⁴)
        kappa: Shear correction factor
        rho: Density (kg/m³)
    """

    E: float = 200.0e9        # Steel
    G: float = 76.9e9
    A: float = 0.01            # 100 mm × 100 mm
    I: float = 8.33e-6         # bh³/12
    kappa: float = 5.0 / 6.0   # Rectangular
    rho: float = 7850.0

    @classmethod
    def rectangular(
        cls, E: float, nu: float, b: float, h: float, rho: float = 7850.0
    ) -> "TimoshenkoBeam":
        """Create from rectangular cross-section b × h."""
        G = E / (2.0 * (1.0 + nu))
        A = b * h
        I = b * h ** 3 / 12.0
        return cls(E=E, G=G, A=A, I=I, kappa=5.0 / 6.0, rho=rho)

    @classmethod
    def circular(
        cls, E: float, nu: float, r: float, rho: float = 7850.0
    ) -> "TimoshenkoBeam":
        """Create from circular cross-section radius r."""
        G = E / (2.0 * (1.0 + nu))
        A = math.pi * r ** 2
        I = math.pi * r ** 4 / 4.0
        return cls(E=E, G=G, A=A, I=I, kappa=6.0 / 7.0, rho=rho)

    def element_stiffness(self, L: float) -> Tensor:
        """
        4×4 Timoshenko beam element stiffness matrix (2D, per element).

        DOFs: [w₁, θ₁, w₂, θ₂]

        K_Tim = (EI/L³(1+Φ)) × [...]

        where Φ = 12EI / (κGA L²) is the shear parameter.
        Φ → 0 gives the Euler-Bernoulli beam.
        """
        EI = self.E * self.I
        Phi = 12.0 * EI / (self.kappa * self.G * self.A * L ** 2)
        denom = L ** 3 * (1.0 + Phi)

        # Stiffness coefficients
        k11 = 12.0 * EI / denom
        k12 = 6.0 * L * EI / denom
        k22 = (4.0 + Phi) * L ** 2 * EI / denom
        k23 = (2.0 - Phi) * L ** 2 * EI / denom

        K = torch.zeros(4, 4, dtype=torch.float64)
        # Row 0: w₁
        K[0, 0] = k11
        K[0, 1] = k12
        K[0, 2] = -k11
        K[0, 3] = k12
        # Row 1: θ₁
        K[1, 0] = k12
        K[1, 1] = k22
        K[1, 2] = -k12
        K[1, 3] = k23
        # Row 2: w₂
        K[2, 0] = -k11
        K[2, 1] = -k12
        K[2, 2] = k11
        K[2, 3] = -k12
        # Row 3: θ₂
        K[3, 0] = k12
        K[3, 1] = k23
        K[3, 2] = -k12
        K[3, 3] = k22

        return K

    def element_mass(self, L: float, consistent: bool = True) -> Tensor:
        """
        4×4 mass matrix for Timoshenko beam element.

        Args:
            L: Element length
            consistent: If True, consistent mass; if False, lumped mass
        """
        if not consistent:
            # Lumped mass
            m_total = self.rho * self.A * L
            M = torch.zeros(4, 4, dtype=torch.float64)
            M[0, 0] = m_total / 2.0
            M[1, 1] = m_total * L ** 2 / 24.0
            M[2, 2] = m_total / 2.0
            M[3, 3] = m_total * L ** 2 / 24.0
            return M

        # Consistent mass (Euler-Bernoulli component)
        m = self.rho * self.A * L / 420.0
        M = torch.zeros(4, 4, dtype=torch.float64)

        M[0, 0] = 156.0 * m
        M[0, 1] = 22.0 * L * m
        M[0, 2] = 54.0 * m
        M[0, 3] = -13.0 * L * m

        M[1, 0] = M[0, 1]
        M[1, 1] = 4.0 * L ** 2 * m
        M[1, 2] = 13.0 * L * m
        M[1, 3] = -3.0 * L ** 2 * m

        M[2, 0] = M[0, 2]
        M[2, 1] = M[1, 2]
        M[2, 2] = 156.0 * m
        M[2, 3] = -22.0 * L * m

        M[3, 0] = M[0, 3]
        M[3, 1] = M[1, 3]
        M[3, 2] = M[2, 3]
        M[3, 3] = 4.0 * L ** 2 * m

        return M

    def element_geometric_stiffness(self, L: float, N: float) -> Tensor:
        """
        4×4 geometric stiffness matrix K_g for buckling analysis.

        K_g accounts for the effect of axial load N on lateral stiffness.
        Buckling load: det(K + λ K_g) = 0.

        Args:
            L: Element length
            N: Axial force (positive = tension)
        """
        Kg = torch.zeros(4, 4, dtype=torch.float64)

        c = N / (30.0 * L)

        Kg[0, 0] = 36.0 * c
        Kg[0, 1] = 3.0 * L * c
        Kg[0, 2] = -36.0 * c
        Kg[0, 3] = 3.0 * L * c

        Kg[1, 0] = 3.0 * L * c
        Kg[1, 1] = 4.0 * L ** 2 * c
        Kg[1, 2] = -3.0 * L * c
        Kg[1, 3] = -L ** 2 * c

        Kg[2, 0] = -36.0 * c
        Kg[2, 1] = -3.0 * L * c
        Kg[2, 2] = 36.0 * c
        Kg[2, 3] = -3.0 * L * c

        Kg[3, 0] = 3.0 * L * c
        Kg[3, 1] = -L ** 2 * c
        Kg[3, 2] = -3.0 * L * c
        Kg[3, 3] = 4.0 * L ** 2 * c

        return Kg


# ═══════════════════════════════════════════════════════════════════════════════
#  MINDLIN-REISSNER PLATE ELEMENT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MindlinReissnerPlate:
    """
    4-node Mindlin-Reissner plate element (Quad4, shear-deformable).

    DOFs per node: [w, θx, θy] (deflection + 2 rotations)
    Total: 12 DOF per element.

    Strain-displacement:
        κ = {∂θx/∂x, ∂θy/∂y, ∂θx/∂y + ∂θy/∂x}    (curvatures)
        γ = {∂w/∂x - θx, ∂w/∂y - θy}                (transverse shear)

    Constitutive:
        M = D_b κ    (moment resultants)
        Q = D_s γ    (shear resultants)

    where:
        D_b = (Eh³/12(1-ν²)) × [[1,ν,0],[ν,1,0],[0,0,(1-ν)/2]]
        D_s = κGh × [[1,0],[0,1]]

    Selective reduced integration prevents shear locking:
        - Bending: 2×2 Gauss quadrature (full)
        - Shear: 1×1 Gauss quadrature (reduced)

    Attributes:
        E: Young's modulus
        nu: Poisson's ratio
        h: Plate thickness
        kappa: Shear correction factor (5/6)
        rho: Density
    """

    E: float = 200.0e9
    nu: float = 0.3
    h: float = 0.01         # 10 mm plate
    kappa: float = 5.0 / 6.0
    rho: float = 7850.0

    def __post_init__(self):
        self.G = self.E / (2.0 * (1.0 + self.nu))

    @property
    def bending_stiffness_matrix(self) -> Tensor:
        """D_b = Eh³/12(1-ν²) × [...] (3×3 bending)."""
        D_factor = self.E * self.h ** 3 / (12.0 * (1.0 - self.nu ** 2))
        Db = torch.zeros(3, 3, dtype=torch.float64)
        Db[0, 0] = D_factor
        Db[0, 1] = D_factor * self.nu
        Db[1, 0] = D_factor * self.nu
        Db[1, 1] = D_factor
        Db[2, 2] = D_factor * (1.0 - self.nu) / 2.0
        return Db

    @property
    def shear_stiffness_matrix(self) -> Tensor:
        """D_s = κGh × I₂ (2×2 shear)."""
        Ds_factor = self.kappa * self.G * self.h
        Ds = torch.zeros(2, 2, dtype=torch.float64)
        Ds[0, 0] = Ds_factor
        Ds[1, 1] = Ds_factor
        return Ds

    @staticmethod
    def _shape_functions(xi: float, eta: float) -> Tensor:
        """Bilinear shape functions N₁..N₄ at (ξ, η) ∈ [-1,1]²."""
        return 0.25 * torch.tensor([
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ], dtype=torch.float64)

    @staticmethod
    def _shape_derivatives(xi: float, eta: float) -> Tensor:
        """Derivatives [∂N/∂ξ, ∂N/∂η] at (ξ, η). Shape: [4, 2]."""
        return 0.25 * torch.tensor([
            [-(1.0 - eta), -(1.0 - xi)],
            [ (1.0 - eta), -(1.0 + xi)],
            [ (1.0 + eta),  (1.0 + xi)],
            [-(1.0 + eta),  (1.0 - xi)],
        ], dtype=torch.float64)

    def element_stiffness(self, node_coords: Tensor) -> Tensor:
        """
        12×12 element stiffness matrix with selective reduced integration.

        Args:
            node_coords: [4, 2] nodal coordinates (x, y)

        Returns:
            K: [12, 12] element stiffness
        """
        Db = self.bending_stiffness_matrix
        Ds = self.shear_stiffness_matrix

        K = torch.zeros(12, 12, dtype=torch.float64)

        # Gauss points
        gp2 = 1.0 / math.sqrt(3.0)
        # 2×2 for bending
        gauss_full = [(-gp2, -gp2), (gp2, -gp2), (gp2, gp2), (-gp2, gp2)]
        # 1×1 for shear (reduced to prevent locking)
        gauss_reduced = [(0.0, 0.0)]

        # Bending contribution (2×2 integration)
        for xi, eta in gauss_full:
            dN_dxi = self._shape_derivatives(xi, eta)
            J = dN_dxi.T @ node_coords  # [2, 2]
            det_J = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            J_inv = torch.tensor([
                [J[1, 1], -J[0, 1]],
                [-J[1, 0], J[0, 0]]
            ], dtype=torch.float64) / det_J

            dN_dx = dN_dxi @ J_inv  # [4, 2]

            # Bending B-matrix: κ = B_b × a_e
            # a_e = [w₁,θx₁,θy₁, w₂,θx₂,θy₂, w₃,θx₃,θy₃, w₄,θx₄,θy₄]
            Bb = torch.zeros(3, 12, dtype=torch.float64)
            for i in range(4):
                # κ_xx = ∂θx/∂x
                Bb[0, 3 * i + 1] = dN_dx[i, 0]
                # κ_yy = ∂θy/∂y
                Bb[1, 3 * i + 2] = dN_dx[i, 1]
                # κ_xy = ∂θx/∂y + ∂θy/∂x
                Bb[2, 3 * i + 1] = dN_dx[i, 1]
                Bb[2, 3 * i + 2] = dN_dx[i, 0]

            K += Bb.T @ Db @ Bb * det_J  # weight = 1 for each 2×2 point

        # Shear contribution (1×1 reduced integration)
        for xi, eta in gauss_reduced:
            N = self._shape_functions(xi, eta)
            dN_dxi = self._shape_derivatives(xi, eta)
            J = dN_dxi.T @ node_coords
            det_J = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            J_inv = torch.tensor([
                [J[1, 1], -J[0, 1]],
                [-J[1, 0], J[0, 0]]
            ], dtype=torch.float64) / det_J

            dN_dx = dN_dxi @ J_inv

            # Shear B-matrix: γ = B_s × a_e
            Bs = torch.zeros(2, 12, dtype=torch.float64)
            for i in range(4):
                # γ_xz = ∂w/∂x - θx
                Bs[0, 3 * i] = dN_dx[i, 0]
                Bs[0, 3 * i + 1] = -N[i]
                # γ_yz = ∂w/∂y - θy
                Bs[1, 3 * i] = dN_dx[i, 1]
                Bs[1, 3 * i + 2] = -N[i]

            K += Bs.T @ Ds @ Bs * det_J * 4.0  # weight = 4 for 1-point integration

        return K

    def element_mass(self, node_coords: Tensor) -> Tensor:
        """
        12×12 consistent mass matrix.

        M = ∫ ρh N^T N dA + ∫ ρh³/12 N_θ^T N_θ dA
        """
        M = torch.zeros(12, 12, dtype=torch.float64)

        gp2 = 1.0 / math.sqrt(3.0)
        gauss_pts = [(-gp2, -gp2), (gp2, -gp2), (gp2, gp2), (-gp2, gp2)]

        for xi, eta in gauss_pts:
            N = self._shape_functions(xi, eta)
            dN_dxi = self._shape_derivatives(xi, eta)
            J = dN_dxi.T @ node_coords
            det_J = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]

            for i in range(4):
                for j in range(4):
                    # Translational mass (w-w coupling)
                    M[3 * i, 3 * j] += self.rho * self.h * N[i] * N[j] * det_J

                    # Rotary inertia (θx-θx, θy-θy coupling)
                    rotary = self.rho * self.h ** 3 / 12.0
                    M[3 * i + 1, 3 * j + 1] += rotary * N[i] * N[j] * det_J
                    M[3 * i + 2, 3 * j + 2] += rotary * N[i] * N[j] * det_J

        return M


# ═══════════════════════════════════════════════════════════════════════════════
#  MODAL ANALYSIS (LANCZOS EIGENSOLVER)
# ═══════════════════════════════════════════════════════════════════════════════

def lanczos_eigensolver(
    K: Tensor,
    M: Tensor,
    n_modes: int = 10,
    sigma: float = 0.0,
    max_iter: int = 300,
    tol: float = 1e-10,
) -> Tuple[Tensor, Tensor]:
    """
    Lanczos eigensolver for the generalized eigenvalue problem:

        K φ = ω² M φ

    Using shift-and-invert spectral transformation:
        (K - σM)⁻¹ M φ = θ φ,  where θ = 1/(ω² - σ)

    The Lanczos algorithm builds a Krylov subspace and extracts
    eigenvalues from a tridiagonal matrix — optimal for sparse systems.

    Args:
        K: Global stiffness matrix [N, N] (symmetric positive semi-definite)
        M: Global mass matrix [N, N] (symmetric positive definite)
        n_modes: Number of eigenvalues/eigenvectors to compute
        sigma: Shift value (target frequency squared neighborhood)
        max_iter: Maximum Lanczos iterations
        tol: Convergence tolerance

    Returns:
        (eigenvalues, eigenvectors):
            eigenvalues: [n_modes] natural frequencies squared ω²
            eigenvectors: [N, n_modes] mode shapes
    """
    N = K.shape[0]
    n_modes = min(n_modes, N)

    # Shift-and-invert operator: Op = (K - σM)⁻¹ M
    Kshift = K - sigma * M

    # LU factorization for solving (K - σM) x = M v
    try:
        L, U = torch.linalg.lu_factor(Kshift)
        use_lu = True
    except Exception:
        use_lu = False

    def op_multiply(v: Tensor) -> Tensor:
        """Apply (K - σM)⁻¹ M v."""
        rhs = M @ v
        if use_lu:
            return torch.linalg.lu_solve(L, U, rhs.unsqueeze(-1)).squeeze(-1)
        else:
            return torch.linalg.solve(Kshift, rhs)

    # Lanczos iteration
    m = min(max_iter, N)
    Q = torch.zeros(N, m + 1, dtype=K.dtype, device=K.device)
    alpha = torch.zeros(m, dtype=K.dtype, device=K.device)
    beta = torch.zeros(m, dtype=K.dtype, device=K.device)

    # Initial vector (random, M-orthonormalize)
    q = torch.randn(N, dtype=K.dtype, device=K.device)
    Mq = M @ q
    q_norm = torch.sqrt(q @ Mq)
    q = q / q_norm
    Q[:, 0] = q

    for j in range(m):
        # w = Op @ q_j
        w = op_multiply(Q[:, j])

        # Orthogonalize
        alpha[j] = w @ M @ Q[:, j]
        w = w - alpha[j] * Q[:, j]
        if j > 0:
            w = w - beta[j - 1] * Q[:, j - 1]

        # Reorthogonalization (full, for numerical stability)
        for k in range(j + 1):
            Mq_k = M @ Q[:, k]
            h = w @ Mq_k
            w = w - h * Q[:, k]

        Mw = M @ w
        beta_val = torch.sqrt(abs(w @ Mw))
        beta[j] = beta_val

        if beta_val < tol:
            m = j + 1
            break

        Q[:, j + 1] = w / beta_val

        # Check convergence every 10 steps
        if (j + 1) >= n_modes and (j + 1) % 10 == 0:
            T = _build_tridiagonal(alpha[:j + 1], beta[:j])
            evals, evecs = torch.linalg.eigh(T)
            # Largest eigenvalues of (K-σM)⁻¹M correspond to closest to σ
            if j + 1 >= 2 * n_modes:
                break

    # Extract eigenvalues from tridiagonal matrix
    T = _build_tridiagonal(alpha[:m], beta[:m - 1] if m > 1 else torch.tensor([]))
    theta, S = torch.linalg.eigh(T)

    # Convert back: ω² = σ + 1/θ
    # Sort by largest θ (closest to shift)
    sorted_idx = torch.argsort(theta.abs(), descending=True)
    theta = theta[sorted_idx]
    S = S[:, sorted_idx]

    eigenvalues = sigma + 1.0 / (theta[:n_modes] + 1e-30)

    # Ritz vectors
    eigenvectors = Q[:, :m] @ S[:, :n_modes]

    # Normalize eigenvectors: φ^T M φ = 1
    for i in range(n_modes):
        Mphi = M @ eigenvectors[:, i]
        norm = torch.sqrt(abs(eigenvectors[:, i] @ Mphi))
        if norm > 1e-30:
            eigenvectors[:, i] /= norm

    # Sort by ascending frequency
    sort_idx = torch.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    return eigenvalues, eigenvectors


def _build_tridiagonal(alpha: Tensor, beta: Tensor) -> Tensor:
    """Build symmetric tridiagonal matrix from diagonal and sub-diagonal."""
    m = alpha.shape[0]
    T = torch.diag(alpha)
    if beta.numel() > 0:
        T += torch.diag(beta[:m - 1], 1) + torch.diag(beta[:m - 1], -1)
    return T


def eigenvalue_buckling(
    K: Tensor, K_geo: Tensor, n_modes: int = 5
) -> Tuple[Tensor, Tensor]:
    """
    Linear eigenvalue buckling analysis.

    Solve: (K + λ K_g) φ = 0

    The smallest positive eigenvalue λ_cr is the critical buckling
    load factor. The corresponding eigenvector φ is the buckling mode.

    For a column with Euler buckling:
        P_cr = π²EI/L²  (fixed-free)
        P_cr = 4π²EI/L² (fixed-fixed)

    Args:
        K: Global stiffness matrix [N, N]
        K_geo: Global geometric stiffness matrix [N, N]
        n_modes: Number of buckling modes

    Returns:
        (load_factors, mode_shapes): Critical load factors and modes
    """
    # Solve K φ = -λ K_g φ  →  K_g⁻¹ K φ = -λ φ
    # Using generalized eigenvalue: K φ = μ (-K_g) φ, λ = -μ
    # Since K_g is signed, we solve directly
    evals, evecs = torch.linalg.eigh(
        torch.linalg.solve(K, -K_geo) if K.shape[0] < 5000
        else _iterative_eigen(-K_geo, K, n_modes)
    )

    # Buckling load factors: take smallest positive eigenvalues
    pos_mask = evals > 1e-10
    if pos_mask.any():
        pos_evals = evals[pos_mask]
        pos_evecs = evecs[:, pos_mask]
        sorted_idx = torch.argsort(pos_evals)[:n_modes]
        return pos_evals[sorted_idx], pos_evecs[:, sorted_idx]

    # Fallback: return all eigenvalues sorted by magnitude
    sorted_idx = torch.argsort(evals.abs())[:n_modes]
    return evals[sorted_idx], evecs[:, sorted_idx]


def _iterative_eigen(A: Tensor, B: Tensor, n: int) -> Tuple[Tensor, Tensor]:
    """Fallback eigenvalue computation for large systems."""
    return torch.linalg.eigh(torch.linalg.solve(B, A))


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPOSITE CLASSICAL LAMINATION THEORY (CLT)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CompositeLamina:
    """
    Single orthotropic lamina (ply) properties.

    Material coordinates: 1 = fiber direction, 2 = transverse.

    Reduced stiffness (plane stress):
        Q₁₁ = E₁/(1-ν₁₂ν₂₁)
        Q₂₂ = E₂/(1-ν₁₂ν₂₁)
        Q₁₂ = ν₁₂E₂/(1-ν₁₂ν₂₁)
        Q₆₆ = G₁₂
    """

    E1: float = 140.0e9    # Fiber direction (carbon/epoxy)
    E2: float = 10.0e9     # Transverse
    G12: float = 5.0e9     # In-plane shear
    nu12: float = 0.28     # Major Poisson's ratio
    thickness: float = 0.125e-3  # 0.125 mm ply

    # Strength values for Tsai-Wu
    Xt: float = 1500.0e6   # Tensile strength, fiber dir
    Xc: float = 1200.0e6   # Compressive strength, fiber dir
    Yt: float = 50.0e6     # Tensile strength, transverse
    Yc: float = 200.0e6    # Compressive strength, transverse
    S12: float = 70.0e6    # Shear strength

    @property
    def nu21(self) -> float:
        """Minor Poisson's ratio: ν₂₁ = ν₁₂ E₂/E₁."""
        return self.nu12 * self.E2 / self.E1

    def Q_matrix(self) -> Tensor:
        """Reduced stiffness matrix [Q] in material coordinates (3×3)."""
        nu21 = self.nu21
        denom = 1.0 - self.nu12 * nu21

        Q = torch.zeros(3, 3, dtype=torch.float64)
        Q[0, 0] = self.E1 / denom
        Q[0, 1] = self.nu12 * self.E2 / denom
        Q[1, 0] = Q[0, 1]
        Q[1, 1] = self.E2 / denom
        Q[2, 2] = self.G12
        return Q

    def Q_bar(self, theta: float) -> Tensor:
        """
        Transformed reduced stiffness [Q̄] at angle θ (radians).

        [Q̄] = [T]⁻¹ [Q] [T]⁻ᵀ

        where T is the stress transformation matrix.
        """
        c = math.cos(theta)
        s = math.sin(theta)
        c2 = c * c
        s2 = s * s
        cs = c * s

        Q = self.Q_matrix()
        Q11, Q12, Q22, Q66 = Q[0, 0], Q[0, 1], Q[1, 1], Q[2, 2]

        Qbar = torch.zeros(3, 3, dtype=torch.float64)
        Qbar[0, 0] = Q11 * c2**2 + 2.0 * (Q12 + 2.0 * Q66) * c2 * s2 + Q22 * s2**2
        Qbar[0, 1] = (Q11 + Q22 - 4.0 * Q66) * c2 * s2 + Q12 * (c2**2 + s2**2)
        Qbar[1, 0] = Qbar[0, 1]
        Qbar[1, 1] = Q11 * s2**2 + 2.0 * (Q12 + 2.0 * Q66) * c2 * s2 + Q22 * c2**2
        Qbar[0, 2] = (Q11 - Q12 - 2.0 * Q66) * c * s * c2 + (Q12 - Q22 + 2.0 * Q66) * c * s * s2
        Qbar[2, 0] = Qbar[0, 2]
        Qbar[1, 2] = (Q11 - Q12 - 2.0 * Q66) * c * s * s2 + (Q12 - Q22 + 2.0 * Q66) * c * s * c2
        Qbar[2, 1] = Qbar[1, 2]
        Qbar[2, 2] = (Q11 + Q22 - 2.0 * Q12 - 2.0 * Q66) * c2 * s2 + Q66 * (c2**2 + s2**2)

        return Qbar

    def tsai_wu_failure(self, sigma_1: float, sigma_2: float, tau_12: float) -> float:
        """
        Tsai-Wu failure index (FI < 1 means safe).

        F₁σ₁ + F₂σ₂ + F₁₁σ₁² + F₂₂σ₂² + F₆₆τ₁₂² + 2F₁₂σ₁σ₂ = FI

        where:
            F₁ = 1/Xt - 1/Xc,  F₂ = 1/Yt - 1/Yc
            F₁₁ = 1/(Xt·Xc),   F₂₂ = 1/(Yt·Yc)
            F₆₆ = 1/S₁₂²
            F₁₂ = -½√(F₁₁·F₂₂)  (Tsai-Hahn interaction term)
        """
        F1 = 1.0 / self.Xt - 1.0 / self.Xc
        F2 = 1.0 / self.Yt - 1.0 / self.Yc
        F11 = 1.0 / (self.Xt * self.Xc)
        F22 = 1.0 / (self.Yt * self.Yc)
        F66 = 1.0 / (self.S12 ** 2)
        F12 = -0.5 * math.sqrt(F11 * F22)

        FI = (
            F1 * sigma_1
            + F2 * sigma_2
            + F11 * sigma_1 ** 2
            + F22 * sigma_2 ** 2
            + F66 * tau_12 ** 2
            + 2.0 * F12 * sigma_1 * sigma_2
        )
        return FI


@dataclass
class CompositeLaminate:
    """
    Classical Lamination Theory (CLT) for composite layup.

    ABD matrix:
        {N}   [A  B] {ε⁰}
        {M} = [B  D] {κ}

    where:
        A_ij = Σ Q̄_ij^(k) (z_k - z_{k-1})           (extensional)
        B_ij = ½ Σ Q̄_ij^(k) (z_k² - z_{k-1}²)      (coupling)
        D_ij = ⅓ Σ Q̄_ij^(k) (z_k³ - z_{k-1}³)      (bending)

    Symmetric laminates have B = 0 (no bending-extension coupling).

    Attributes:
        plies: List of (CompositeLamina, angle_in_radians) pairs
    """

    plies: List[Tuple[CompositeLamina, float]]

    @property
    def total_thickness(self) -> float:
        return sum(ply.thickness for ply, _ in self.plies)

    def ABD_matrix(self) -> Tensor:
        """Compute 6×6 ABD stiffness matrix."""
        n = len(self.plies)
        h_total = self.total_thickness

        # z-coordinates of ply interfaces (from bottom)
        z = [-h_total / 2.0]
        for ply, _ in self.plies:
            z.append(z[-1] + ply.thickness)

        A = torch.zeros(3, 3, dtype=torch.float64)
        B = torch.zeros(3, 3, dtype=torch.float64)
        D = torch.zeros(3, 3, dtype=torch.float64)

        for k, (ply, angle) in enumerate(self.plies):
            Qbar = ply.Q_bar(angle)
            z_bot = z[k]
            z_top = z[k + 1]

            A += Qbar * (z_top - z_bot)
            B += 0.5 * Qbar * (z_top ** 2 - z_bot ** 2)
            D += (1.0 / 3.0) * Qbar * (z_top ** 3 - z_bot ** 3)

        ABD = torch.zeros(6, 6, dtype=torch.float64)
        ABD[:3, :3] = A
        ABD[:3, 3:] = B
        ABD[3:, :3] = B
        ABD[3:, 3:] = D

        return ABD

    def ply_stresses(
        self, mid_strain: Tensor, curvature: Tensor
    ) -> List[Dict[str, float]]:
        """
        Compute stress in each ply (top and bottom of each layer).

        Args:
            mid_strain: [ε_xx, ε_yy, γ_xy] mid-plane strains
            curvature: [κ_xx, κ_yy, κ_xy] curvatures

        Returns:
            List of dicts with 'sigma_1', 'sigma_2', 'tau_12', 'FI' per ply face
        """
        h_total = self.total_thickness
        z = [-h_total / 2.0]
        for ply, _ in self.plies:
            z.append(z[-1] + ply.thickness)

        results: List[Dict[str, float]] = []

        for k, (ply, angle) in enumerate(self.plies):
            Qbar = ply.Q_bar(angle)

            for z_val in [z[k], z[k + 1]]:
                # Global strain at this z
                eps_global = mid_strain + z_val * curvature
                # Global stress
                sigma_global = Qbar @ eps_global

                # Transform to material coordinates
                c = math.cos(angle)
                s = math.sin(angle)
                T = torch.tensor([
                    [c * c, s * s, 2 * c * s],
                    [s * s, c * c, -2 * c * s],
                    [-c * s, c * s, c * c - s * s],
                ], dtype=torch.float64)
                sigma_mat = T @ sigma_global

                sigma_1 = sigma_mat[0].item()
                sigma_2 = sigma_mat[1].item()
                tau_12 = sigma_mat[2].item()
                FI = ply.tsai_wu_failure(sigma_1, sigma_2, tau_12)

                results.append({
                    "ply": k,
                    "z": z_val,
                    "angle_deg": math.degrees(angle),
                    "sigma_1": sigma_1,
                    "sigma_2": sigma_2,
                    "tau_12": tau_12,
                    "tsai_wu_FI": FI,
                    "failed": FI >= 1.0,
                })

        return results


# ═══════════════════════════════════════════════════════════════════════════════
#  BEAM ASSEMBLY HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def assemble_beam_system(
    beam: TimoshenkoBeam,
    n_elements: int,
    total_length: float,
    fixed_dofs: Optional[List[int]] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Assemble global stiffness and mass matrices for a uniform beam.

    Args:
        beam: Beam section properties
        n_elements: Number of elements
        total_length: Total beam length (m)
        fixed_dofs: List of DOF indices to fix (boundary conditions)

    Returns:
        (K_global, M_global, K_geo_global) assembled matrices
    """
    L_e = total_length / n_elements
    n_dof = 2 * (n_elements + 1)  # [w, θ] per node

    K = torch.zeros(n_dof, n_dof, dtype=torch.float64)
    M = torch.zeros(n_dof, n_dof, dtype=torch.float64)
    Kg = torch.zeros(n_dof, n_dof, dtype=torch.float64)

    Ke = beam.element_stiffness(L_e)
    Me = beam.element_mass(L_e)
    Kge = beam.element_geometric_stiffness(L_e, 1.0)  # Unit load

    for e in range(n_elements):
        idx = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        for i in range(4):
            for j in range(4):
                K[idx[i], idx[j]] += Ke[i, j]
                M[idx[i], idx[j]] += Me[i, j]
                Kg[idx[i], idx[j]] += Kge[i, j]

    # Apply boundary conditions (penalty method)
    if fixed_dofs is not None:
        penalty = K.abs().max() * 1e8
        for dof in fixed_dofs:
            K[dof, dof] += penalty
            M[dof, dof] += penalty * 1e-12  # Small mass for constrained DOF

    return K, M, Kg


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Beam elements
    "TimoshenkoBeam",
    # Plate elements
    "MindlinReissnerPlate",
    # Eigensolvers
    "lanczos_eigensolver",
    "eigenvalue_buckling",
    # Composites
    "CompositeLamina",
    "CompositeLaminate",
    # Assembly
    "assemble_beam_system",
]
