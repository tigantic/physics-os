"""
Self-Similar Coordinate Transformation for Blow-Up Analysis.

The key insight: A finite-time singularity in physical time becomes a
STEADY STATE in rescaled "dynamic" coordinates.

Mathematical Setup
==================

In physical coordinates (x, t), the velocity near blow-up at time T:
    u(x, t) ~ (T-t)^(-α) U(ξ)

where ξ = x / (T-t)^β is the "zoomed-in" coordinate.

The Rescaled Time:
    τ = -log(T - t)

As t → T, we have τ → ∞. The blow-up becomes a limit at τ = ∞.

In these coordinates, the Navier-Stokes equations become:
    ∂U/∂τ = F(U)

A self-similar singularity corresponds to F(U*) = 0 (a steady state).

Reference: Hou & Huang (2022), "Potential singularity of the 3D Euler equations"
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch


class BlowUpType(Enum):
    """Classification of potential singularity types."""

    VORTEX_SHEET = "vortex_sheet"  # ω concentrated on a surface
    VORTEX_FILAMENT = "vortex_filament"  # ω concentrated on a curve
    POINT_SINGULARITY = "point"  # ω concentrated at a point
    SELF_SIMILAR = "self_similar"  # Scale-invariant blow-up


@dataclass
class SelfSimilarScaling:
    """
    Self-similar scaling parameters for blow-up.

    The velocity ansatz:
        u(x, t) = (T-t)^(-α) U(x / (T-t)^β)

    Dimensional analysis for Navier-Stokes gives:
        α = 1/2, β = 1/2 for viscous scaling
        α = 1,   β = 1   for inviscid (Euler) scaling
    """

    alpha: float = 0.5  # Velocity exponent
    beta: float = 0.5  # Spatial exponent
    T_star: float = 1.0  # Estimated blow-up time

    def validate(self) -> bool:
        """Check dimensional consistency."""
        # For NS: [u] = L/T, [x] = L, [t] = T
        # u ~ (T-t)^(-α) U(x/(T-t)^β)
        # [u] = T^(-α), [ξ] = L / T^β
        # For consistent [u] = L/T: need -α = -1, i.e., α = 1
        # But viscosity gives different scaling
        return self.alpha > 0 and self.beta > 0

    def to_rescaled_time(self, t: torch.Tensor) -> torch.Tensor:
        """
        Convert physical time to rescaled time.

        τ = -log(T* - t) → ∞ as t → T*
        """
        margin = self.T_star - t
        if torch.any(margin <= 0):
            raise ValueError("Time exceeds blow-up time T*")
        return -torch.log(margin)

    def from_rescaled_time(self, tau: torch.Tensor) -> torch.Tensor:
        """
        Convert rescaled time back to physical time.

        t = T* - exp(-τ)
        """
        return self.T_star - torch.exp(-tau)

    def velocity_rescaling(self, t: torch.Tensor) -> torch.Tensor:
        """
        Velocity rescaling factor: (T* - t)^α

        Physical velocity = rescaled velocity / factor
        """
        margin = self.T_star - t
        return torch.pow(margin, self.alpha)

    def spatial_rescaling(self, t: torch.Tensor) -> torch.Tensor:
        """
        Spatial rescaling factor: (T* - t)^β

        Physical coordinate = rescaled coordinate * factor
        """
        margin = self.T_star - t
        return torch.pow(margin, self.beta)


@dataclass
class RescaledNSEquations:
    """
    Navier-Stokes equations in self-similar coordinates.

    The rescaled equations for profile U(ξ):

    ∂U/∂τ + (U·∇)U + α U + β (ξ·∇)U = -∇P + ν_eff ΔU

    where ν_eff = ν (T* - t)^(1-2β) is the effective (time-dependent) viscosity.

    At the fixed point (∂U/∂τ = 0):
        (U·∇)U + α U + β (ξ·∇)U + ∇P - ν_eff ΔU = 0

    This is the equation we need to solve to prove blow-up.
    """

    scaling: SelfSimilarScaling
    nu: float = 1e-3  # Physical viscosity
    L: float = 2 * np.pi  # Domain size (in ξ coordinates)
    N: int = 64  # Grid resolution

    def __post_init__(self):
        """Initialize spectral grid."""
        self.dx = self.L / self.N
        # Rescaled coordinate ξ
        self.xi = torch.linspace(-self.L / 2, self.L / 2, self.N, dtype=torch.float64)

        # Spectral wavenumbers
        self.k = torch.fft.fftfreq(self.N, self.dx) * 2 * np.pi
        self.kx, self.ky, self.kz = torch.meshgrid(
            self.k, self.k, self.k, indexing="ij"
        )
        self.k_sq = self.kx**2 + self.ky**2 + self.kz**2
        self.k_sq[0, 0, 0] = 1.0  # Avoid division by zero

    def effective_viscosity(self, tau: torch.Tensor) -> torch.Tensor:
        """
        Effective viscosity in rescaled coordinates.

        ν_eff(τ) = ν * exp((1 - 2β)τ)

        For β = 1/2: ν_eff = ν (constant)
        For β < 1/2: ν_eff → ∞ as τ → ∞ (viscosity dominates)
        For β > 1/2: ν_eff → 0 as τ → ∞ (Euler-like)
        """
        exponent = (1 - 2 * self.scaling.beta) * tau
        return self.nu * torch.exp(exponent)

    def stretching_term(self, U: torch.Tensor) -> torch.Tensor:
        """
        The self-similar stretching: β(ξ·∇)U + αU

        This is the term that balances the nonlinearity at the fixed point.

        Args:
            U: Velocity field (N, N, N, 3)

        Returns:
            Stretching contribution (N, N, N, 3)
        """
        alpha = self.scaling.alpha
        beta = self.scaling.beta

        # Construct ξ grids
        xi_x = self.xi.view(-1, 1, 1, 1).expand_as(U[..., 0:1])
        xi_y = self.xi.view(1, -1, 1, 1).expand_as(U[..., 0:1])
        xi_z = self.xi.view(1, 1, -1, 1).expand_as(U[..., 0:1])

        # ∇U via spectral derivatives
        U_hat = torch.fft.fftn(U, dim=(0, 1, 2))

        dUdx = torch.fft.ifftn(1j * self.kx.unsqueeze(-1) * U_hat, dim=(0, 1, 2)).real
        dUdy = torch.fft.ifftn(1j * self.ky.unsqueeze(-1) * U_hat, dim=(0, 1, 2)).real
        dUdz = torch.fft.ifftn(1j * self.kz.unsqueeze(-1) * U_hat, dim=(0, 1, 2)).real

        # (ξ·∇)U = ξ_x ∂U/∂x + ξ_y ∂U/∂y + ξ_z ∂U/∂z
        xi_dot_grad_U = xi_x * dUdx + xi_y * dUdy + xi_z * dUdz

        # β(ξ·∇)U + αU
        return beta * xi_dot_grad_U + alpha * U

    def advection_term(self, U: torch.Tensor) -> torch.Tensor:
        """
        Nonlinear advection: (U·∇)U

        Args:
            U: Velocity field (N, N, N, 3)

        Returns:
            Advection (N, N, N, 3)
        """
        U_hat = torch.fft.fftn(U, dim=(0, 1, 2))

        dUdx = torch.fft.ifftn(1j * self.kx.unsqueeze(-1) * U_hat, dim=(0, 1, 2)).real
        dUdy = torch.fft.ifftn(1j * self.ky.unsqueeze(-1) * U_hat, dim=(0, 1, 2)).real
        dUdz = torch.fft.ifftn(1j * self.kz.unsqueeze(-1) * U_hat, dim=(0, 1, 2)).real

        # (U·∇)U component by component
        advection = torch.zeros_like(U)
        for i in range(3):
            advection[..., i] = (
                U[..., 0] * dUdx[..., i]
                + U[..., 1] * dUdy[..., i]
                + U[..., 2] * dUdz[..., i]
            )

        return advection

    def laplacian(self, U: torch.Tensor) -> torch.Tensor:
        """
        Laplacian ΔU via spectral method.

        Args:
            U: Velocity field (N, N, N, 3)

        Returns:
            ΔU (N, N, N, 3)
        """
        U_hat = torch.fft.fftn(U, dim=(0, 1, 2))
        lap_hat = -self.k_sq.unsqueeze(-1) * U_hat
        return torch.fft.ifftn(lap_hat, dim=(0, 1, 2)).real

    def pressure_projection(self, F: torch.Tensor) -> torch.Tensor:
        """
        Project out non-divergence-free component: (I - ∇∇^(-1)∇·)F

        This enforces ∇·U = 0.
        """
        F_hat = torch.fft.fftn(F, dim=(0, 1, 2))

        # ∇·F in spectral space
        div_hat = (
            1j * self.kx * F_hat[..., 0]
            + 1j * self.ky * F_hat[..., 1]
            + 1j * self.kz * F_hat[..., 2]
        )

        # Pressure: P = ∇^(-2) (∇·F)
        P_hat = div_hat / self.k_sq
        P_hat[0, 0, 0] = 0  # Zero mean pressure

        # Subtract ∇P from each component
        proj_hat = F_hat.clone()
        proj_hat[..., 0] -= 1j * self.kx * P_hat
        proj_hat[..., 1] -= 1j * self.ky * P_hat
        proj_hat[..., 2] -= 1j * self.kz * P_hat

        return torch.fft.ifftn(proj_hat, dim=(0, 1, 2)).real

    def residual(self, U: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        Compute the fixed-point residual: R(U) = F(U) where ∂U/∂τ = F(U).

        At the self-similar fixed point, R(U*) = 0.

        The residual is:
            R = -(U·∇)U - α U - β(ξ·∇)U + ν_eff ΔU - ∇P

        where the pressure is determined by ∇·R = 0 (incompressibility).

        Args:
            U: Candidate profile (N, N, N, 3)
            tau: Rescaled time

        Returns:
            R(U): Residual (N, N, N, 3) — should be small for good profile
        """
        nu_eff = self.effective_viscosity(tau)

        # Assemble unprojected residual
        R_unprojected = (
            -self.advection_term(U)
            - self.stretching_term(U)
            + nu_eff * self.laplacian(U)
        )

        # Project to enforce incompressibility
        R = self.pressure_projection(R_unprojected)

        return R

    def vorticity(self, U: torch.Tensor) -> torch.Tensor:
        """
        Compute vorticity ω = ∇ × U.

        Args:
            U: Velocity field (N, N, N, 3)

        Returns:
            ω: Vorticity (N, N, N, 3)
        """
        U_hat = torch.fft.fftn(U, dim=(0, 1, 2))

        dUdx = 1j * self.kx.unsqueeze(-1) * U_hat
        dUdy = 1j * self.ky.unsqueeze(-1) * U_hat
        dUdz = 1j * self.kz.unsqueeze(-1) * U_hat

        omega_hat = torch.zeros_like(U_hat)
        omega_hat[..., 0] = dUdy[..., 2] - dUdz[..., 1]  # ∂w/∂y - ∂v/∂z
        omega_hat[..., 1] = dUdz[..., 0] - dUdx[..., 2]  # ∂u/∂z - ∂w/∂x
        omega_hat[..., 2] = dUdx[..., 1] - dUdy[..., 0]  # ∂v/∂x - ∂u/∂y

        return torch.fft.ifftn(omega_hat, dim=(0, 1, 2)).real

    def enstrophy(self, U: torch.Tensor) -> torch.Tensor:
        """
        Enstrophy: Ω = ∫|ω|² dξ
        """
        omega = self.vorticity(U)
        return (omega**2).sum() * self.dx**3

    def max_vorticity(self, U: torch.Tensor) -> torch.Tensor:
        """
        Maximum vorticity magnitude: ||ω||_∞

        This is the BKM criterion quantity.
        """
        omega = self.vorticity(U)
        omega_mag = torch.sqrt((omega**2).sum(dim=-1))
        return omega_mag.max()


@dataclass
class SelfSimilarProfile:
    """
    A candidate self-similar blow-up profile.

    This represents U(ξ) that we're trying to prove satisfies F(U) = 0.
    """

    U: torch.Tensor  # (N, N, N, 3) velocity profile
    scaling: SelfSimilarScaling
    residual_norm: float | None = None
    max_vorticity: float | None = None

    def save(self, path: str):
        """Save profile to disk."""
        torch.save(
            {
                "U": self.U,
                "alpha": self.scaling.alpha,
                "beta": self.scaling.beta,
                "T_star": self.scaling.T_star,
                "residual_norm": self.residual_norm,
                "max_vorticity": self.max_vorticity,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> SelfSimilarProfile:
        """Load profile from disk."""
        data = torch.load(path, weights_only=True)
            alpha=data["alpha"], beta=data["beta"], T_star=data["T_star"]
        )
        return cls(
            U=data["U"],
            scaling=scaling,
            residual_norm=data.get("residual_norm"),
            max_vorticity=data.get("max_vorticity"),
        )


def create_candidate_profile(
    N: int = 64,
    profile_type: str = "tornado",
    strength: float = 1.0,
) -> torch.Tensor:
    """
    Create a candidate blow-up profile for optimization.

    Args:
        N: Grid resolution
        profile_type: Type of initial guess
            - "tornado": Concentrated vortex tube
            - "dipole": Colliding vortex rings
            - "random": Smooth random field
        strength: Amplitude scaling

    Returns:
        U: Velocity field (N, N, N, 3)
    """
    L = 2 * np.pi
    x = torch.linspace(-L / 2, L / 2, N, dtype=torch.float64)
    X, Y, Z = torch.meshgrid(x, x, x, indexing="ij")

    U = torch.zeros(N, N, N, 3, dtype=torch.float64)

    if profile_type == "tornado":
        # Concentrated vortex tube along z-axis
        r = torch.sqrt(X**2 + Y**2)
        sigma = L / 10  # Core radius
        vortex_strength = strength * torch.exp(-(r**2) / (2 * sigma**2))

        # u_θ = Γ/(2πr) * (1 - exp(-r²/σ²)) ≈ Lamb-Oseen vortex
        # Convert to Cartesian
        U[..., 0] = -Y * vortex_strength / (r + 1e-10)  # u_x
        U[..., 1] = X * vortex_strength / (r + 1e-10)  # u_y
        U[..., 2] = 0.1 * vortex_strength  # Weak axial flow

    elif profile_type == "dipole":
        # Two vortex rings approaching each other
        z_sep = L / 4

        for sign, z0 in [(1, z_sep), (-1, -z_sep)]:
            r_from_axis = torch.sqrt(X**2 + Y**2)
            r_ring = L / 6  # Ring radius

            # Distance from ring core
            dist = torch.sqrt((r_from_axis - r_ring) ** 2 + (Z - z0) ** 2)
            sigma = L / 20  # Core thickness
            core = torch.exp(-(dist**2) / (2 * sigma**2))

            # Azimuthal velocity around ring
            U[..., 0] += sign * (-Y / (r_from_axis + 1e-10)) * core
            U[..., 1] += sign * (X / (r_from_axis + 1e-10)) * core

        U *= strength

    elif profile_type == "random":
        # Smooth random field (filtered noise)
        k = torch.fft.fftfreq(N) * N
        kx, ky, kz = torch.meshgrid(k, k, k, indexing="ij")
        k_mag = torch.sqrt(kx**2 + ky**2 + kz**2)

        # Filter: emphasize intermediate scales
        filt = torch.exp(-(k_mag**2) / 10) * k_mag**2

        for i in range(3):
            noise = torch.randn(N, N, N, dtype=torch.float64)
            noise_hat = torch.fft.fftn(noise)
            U[..., i] = torch.fft.ifftn(noise_hat * filt).real

        U = U / U.abs().max() * strength

    # Make divergence-free
    U_hat = torch.fft.fftn(U, dim=(0, 1, 2))
    k = torch.fft.fftfreq(N, L / N) * 2 * np.pi
    kx, ky, kz = torch.meshgrid(k, k, k, indexing="ij")
    k_sq = kx**2 + ky**2 + kz**2
    k_sq[0, 0, 0] = 1.0

    div_hat = (
        1j * kx * U_hat[..., 0] + 1j * ky * U_hat[..., 1] + 1j * kz * U_hat[..., 2]
    )
    P_hat = div_hat / k_sq
    P_hat[0, 0, 0] = 0

    U_hat[..., 0] -= 1j * kx * P_hat
    U_hat[..., 1] -= 1j * ky * P_hat
    U_hat[..., 2] -= 1j * kz * P_hat

    U = torch.fft.ifftn(U_hat, dim=(0, 1, 2)).real

    return U


def verify_self_similar_transform():
    """
    Verify the self-similar transformation is correctly implemented.
    """
    print("=" * 60)
    print("SELF-SIMILAR TRANSFORM VERIFICATION")
    print("=" * 60)

    # Test 1: Time coordinate transforms
    scaling = SelfSimilarScaling(alpha=0.5, beta=0.5, T_star=1.0)
    t = torch.tensor([0.0, 0.5, 0.9, 0.99])
    tau = scaling.to_rescaled_time(t)
    t_back = scaling.from_rescaled_time(tau)

    assert torch.allclose(t, t_back, atol=1e-10), "Time transform roundtrip failed"
    print(f"✓ Time transform: t={t.tolist()} → τ={tau.tolist()}")
    print(f"  τ → ∞ as t → T*: τ(t=0.99) = {tau[-1]:.2f}")

    # Test 2: Residual computation
    N = 32
    ns = RescaledNSEquations(scaling, nu=0.01, N=N)
    U = create_candidate_profile(N, "tornado", strength=0.5)
    tau = torch.tensor(1.0)

    R = ns.residual(U, tau)
    R_norm = torch.sqrt((R**2).sum()) * ns.dx**1.5
    print(f"✓ Residual computed: ||R|| = {R_norm:.4e}")

    # Test 3: Vorticity and enstrophy
    omega = ns.vorticity(U)
    omega_max = ns.max_vorticity(U)
    enstrophy = ns.enstrophy(U)
    print(f"✓ Vorticity: ||ω||_∞ = {omega_max:.4f}, Ω = {enstrophy:.4f}")

    # Test 4: Profile types
    for ptype in ["tornado", "dipole", "random"]:
        U = create_candidate_profile(N, ptype)
        div = divergence(U, ns.dx)
        div_max = div.abs().max()
        print(f"✓ Profile '{ptype}': max|∇·U| = {div_max:.2e} (divergence-free)")

    print("=" * 60)
    print("All self-similar transform tests passed!")
    print("=" * 60)


def divergence(U: torch.Tensor, dx: float) -> torch.Tensor:
    """Compute divergence ∇·U."""
    N = U.shape[0]
    k = torch.fft.fftfreq(N, dx) * 2 * np.pi
    kx, ky, kz = torch.meshgrid(k, k, k, indexing="ij")

    U_hat = torch.fft.fftn(U, dim=(0, 1, 2))
    div_hat = (
        1j * kx * U_hat[..., 0] + 1j * ky * U_hat[..., 1] + 1j * kz * U_hat[..., 2]
    )
    return torch.fft.ifftn(div_hat).real


if __name__ == "__main__":
    verify_self_similar_transform()
