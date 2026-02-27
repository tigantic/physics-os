"""
Tokamak Magnetic Geometry — Field Configurations for Fusion

Tokamak magnetic fields confine plasma in a toroidal (donut) shape.
The field has both toroidal (long way around) and poloidal (short way) components.

Key parameters:
- R0: Major radius (center of torus to center of plasma)
- a: Minor radius (radius of plasma cross-section)
- B0: Magnetic field at magnetic axis
- q: Safety factor (field line pitch)

The magnetic field in a tokamak:
    B_φ = B0 × R0 / R        (toroidal, 1/R dependence)
    B_θ = B_φ × r / (q × R)  (poloidal, from plasma current)

Coordinate systems:
    (R, Z, φ): Cylindrical (R=major radius, Z=vertical, φ=toroidal angle)
    (r, θ, φ): Flux coordinates (r=minor radius, θ=poloidal angle)

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from torch import Tensor


@dataclass
class TokamakConfig:
    """Configuration for tokamak magnetic geometry.
    
    Attributes:
        R0: Major radius [m]
        a: Minor radius [m]
        B0: Toroidal field at magnetic axis [T]
        Ip: Plasma current [MA]
        kappa: Elongation (vertical stretch)
        delta: Triangularity (D-shape)
        q0: Central safety factor
        q_edge: Edge safety factor
        n_e0: Central electron density [10^19 m^-3]
        T_e0: Central electron temperature [keV]
    """
    R0: float = 6.2          # ITER-like
    a: float = 2.0
    B0: float = 5.3
    Ip: float = 15.0
    kappa: float = 1.7
    delta: float = 0.33
    q0: float = 1.0
    q_edge: float = 3.5
    n_e0: float = 10.0
    T_e0: float = 10.0
    
    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio R0/a."""
        return self.R0 / self.a
    
    @property
    def inverse_aspect(self) -> float:
        """Inverse aspect ratio ε = a/R0."""
        return self.a / self.R0
    
    @property
    def beta_t(self) -> float:
        """Approximate toroidal beta (thermal/magnetic pressure ratio)."""
        # β_t ≈ 2μ0 × n × T / B^2
        mu0 = 4 * math.pi * 1e-7
        n = self.n_e0 * 1e19  # m^-3
        T = self.T_e0 * 1e3 * 1.6e-19  # Joules
        p = n * T
        return 2 * mu0 * p / (self.B0 ** 2)
    
    @property
    def normalized_beta(self) -> float:
        """Normalized beta β_N = β_t × a × B0 / Ip."""
        return self.beta_t * self.a * self.B0 / self.Ip


class TokamakGeometry:
    """
    Tokamak magnetic field geometry.
    
    Provides methods for:
    - Magnetic field components B_R, B_Z, B_φ
    - Safety factor profile q(r)
    - Flux surface geometry
    - Coordinate transformations
    
    Example:
        >>> config = TokamakConfig(R0=6.2, a=2.0, B0=5.3)
        >>> geom = TokamakGeometry(config)
        >>> 
        >>> # Get field at a point
        >>> R, Z, phi = 7.0, 0.5, 0.0
        >>> B_R, B_Z, B_phi = geom.magnetic_field(R, Z, phi)
        >>> 
        >>> # Get safety factor
        >>> r = 1.0  # minor radius
        >>> q = geom.safety_factor(r)
    """
    
    def __init__(self, config: TokamakConfig):
        self.config = config
    
    def safety_factor(self, r: Tensor | float) -> Tensor | float:
        """
        Safety factor profile q(r).
        
        Uses a simple parabolic profile:
            q(r) = q0 + (q_edge - q0) × (r/a)²
        
        Args:
            r: Minor radius coordinate
        
        Returns:
            Safety factor at r
        """
        cfg = self.config
        rho = r / cfg.a if isinstance(r, (int, float)) else r / cfg.a
        return cfg.q0 + (cfg.q_edge - cfg.q0) * rho ** 2
    
    def magnetic_field(
        self, R: Tensor, Z: Tensor, phi: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Magnetic field components in cylindrical coordinates.
        
        Uses large aspect ratio approximation with Shafranov shift.
        
        Args:
            R: Major radius coordinate
            Z: Vertical coordinate
            phi: Toroidal angle
        
        Returns:
            (B_R, B_Z, B_phi): Field components [T]
        """
        cfg = self.config
        
        # Convert to flux coordinates
        r, theta = self._to_flux_coords(R, Z)
        
        # Safety factor at this radius
        q = self.safety_factor(r)
        
        # Toroidal field (1/R dependence)
        B_phi = cfg.B0 * cfg.R0 / R
        
        # Poloidal field magnitude
        # B_theta = r × B_phi / (q × R)
        B_theta = r * B_phi / (q * R)
        
        # Convert poloidal field to (R, Z) components
        # B_R = -B_theta × sin(theta)
        # B_Z = B_theta × cos(theta)
        B_R = -B_theta * torch.sin(theta)
        B_Z = B_theta * torch.cos(theta)
        
        return B_R, B_Z, B_phi
    
    def _to_flux_coords(
        self, R: Tensor, Z: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Convert (R, Z) to flux coordinates (r, θ).
        
        Uses Miller parameterization for shaped plasmas.
        """
        cfg = self.config
        
        # Simple circular approximation
        # r = sqrt((R - R0)^2 + Z^2)
        # theta = atan2(Z, R - R0)
        
        dR = R - cfg.R0
        
        # Include elongation in Z direction
        Z_scaled = Z / cfg.kappa
        
        r = torch.sqrt(dR ** 2 + Z_scaled ** 2)
        theta = torch.atan2(Z_scaled, dR)
        
        return r, theta
    
    def flux_surface(
        self, r: float, n_points: int = 100
    ) -> Tuple[Tensor, Tensor]:
        """
        Generate (R, Z) points on a flux surface.
        
        Uses Miller parameterization for D-shaped plasmas.
        
        Args:
            r: Minor radius of flux surface
            n_points: Number of points around surface
        
        Returns:
            (R, Z): Arrays of coordinates
        """
        cfg = self.config
        theta = torch.linspace(0, 2 * math.pi, n_points)
        
        # Miller parameterization
        # R = R0 + r × cos(θ + δ × sin(θ))
        # Z = κ × r × sin(θ)
        
        R = cfg.R0 + r * torch.cos(theta + cfg.delta * torch.sin(theta))
        Z = cfg.kappa * r * torch.sin(theta)
        
        return R, Z
    
    def grad_B(
        self, R: Tensor, Z: Tensor, phi: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Gradient of |B| (for drift calculations).
        
        The ∇B drift velocity is:
            v_∇B = (m v_⊥² / 2qB³) × B × ∇B
        """
        cfg = self.config
        
        # |B| ≈ B0 × R0 / R (dominated by toroidal field)
        B_mag = cfg.B0 * cfg.R0 / R
        
        # ∇|B| ≈ -B0 × R0 / R² × R̂
        dB_dR = -cfg.B0 * cfg.R0 / (R ** 2)
        dB_dZ = torch.zeros_like(Z)
        dB_dphi = torch.zeros_like(phi)
        
        return dB_dR, dB_dZ, dB_dphi
    
    def curvature(
        self, R: Tensor, Z: Tensor, phi: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Magnetic field line curvature vector κ = b̂·∇b̂.
        
        For tokamak: κ ≈ -1/R × R̂ (bad curvature on outboard side)
        """
        cfg = self.config
        
        # Curvature magnitude
        kappa_mag = 1.0 / R
        
        # Direction: points away from center (radially outward)
        kappa_R = -kappa_mag * torch.ones_like(R)
        kappa_Z = torch.zeros_like(Z)
        kappa_phi = torch.zeros_like(phi)
        
        return kappa_R, kappa_Z, kappa_phi
    
    def connection_length(self, r: float) -> float:
        """
        Field line connection length (half period).
        
        L_c = π × q × R0
        
        This is the distance a particle travels along the field
        before returning to the same poloidal position.
        """
        cfg = self.config
        q = self.safety_factor(r)
        return math.pi * q * cfg.R0
    
    def banana_width(self, r: float, v_parallel: float) -> float:
        """
        Banana orbit width for trapped particles.
        
        Δr_b ≈ q × ρ_i / √ε
        
        where ρ_i is the ion gyroradius and ε = r/R0.
        """
        cfg = self.config
        q = self.safety_factor(r)
        epsilon = r / cfg.R0
        
        # Ion gyroradius (hydrogen, typical thermal velocity)
        m_p = 1.67e-27  # proton mass
        e = 1.6e-19
        B = cfg.B0
        v_perp = abs(v_parallel)  # Approximate
        rho_i = m_p * v_perp / (e * B)
        
        return q * rho_i / math.sqrt(epsilon)


class VelocitySpaceGeometry:
    """
    Velocity space geometry for 6D Vlasov.
    
    Coordinates:
        (v_∥, v_⊥, ξ): Parallel velocity, perpendicular velocity, gyrophase
    
    Or equivalently:
        (v_∥, μ, ξ): Parallel velocity, magnetic moment, gyrophase
    
    where μ = m v_⊥² / (2B) is the adiabatic invariant.
    """
    
    def __init__(self, tokamak: TokamakGeometry):
        self.tokamak = tokamak
        self.config = tokamak.config
    
    def magnetic_moment(self, v_perp: Tensor, B: Tensor, mass: float = 1.67e-27) -> Tensor:
        """
        Magnetic moment μ = m v_⊥² / (2B).
        
        This is the first adiabatic invariant (conserved in slow field changes).
        """
        return mass * v_perp ** 2 / (2 * B)
    
    def pitch_angle(self, v_parallel: Tensor, v_perp: Tensor) -> Tensor:
        """
        Pitch angle ξ = v_∥ / v.
        
        ξ = ±1: passing particles (move along field)
        ξ ≈ 0: trapped particles (bounce in magnetic wells)
        """
        v = torch.sqrt(v_parallel ** 2 + v_perp ** 2)
        return v_parallel / (v + 1e-15)
    
    def trapped_passing_boundary(
        self, r: float, energy: float, mass: float = 1.67e-27
    ) -> float:
        """
        Critical pitch angle for trapping.
        
        Particles with |ξ| < ξ_c are trapped.
        
        ξ_c = √(2ε / (1 + ε)) where ε = r/R0
        """
        epsilon = r / self.config.R0
        return math.sqrt(2 * epsilon / (1 + epsilon))
    
    def bounce_frequency(
        self, r: float, v: float, pitch_angle: float, mass: float = 1.67e-27
    ) -> float:
        """
        Bounce frequency for trapped particles.
        
        ω_b ≈ √(ε) × v / (q × R0)
        """
        cfg = self.config
        epsilon = r / cfg.R0
        q = self.tokamak.safety_factor(r)
        
        return math.sqrt(epsilon) * v / (q * cfg.R0)
    
    def transit_frequency(
        self, r: float, v_parallel: float
    ) -> float:
        """
        Transit frequency for passing particles.
        
        ω_t = v_∥ / (q × R0)
        """
        cfg = self.config
        q = self.tokamak.safety_factor(r)
        
        return abs(v_parallel) / (q * cfg.R0)


def create_iter_geometry() -> TokamakGeometry:
    """Create ITER-like tokamak geometry."""
    config = TokamakConfig(
        R0=6.2,
        a=2.0,
        B0=5.3,
        Ip=15.0,
        kappa=1.7,
        delta=0.33,
        q0=1.0,
        q_edge=3.5,
        n_e0=10.0,
        T_e0=10.0,
    )
    return TokamakGeometry(config)


def create_sparc_geometry() -> TokamakGeometry:
    """Create SPARC-like (compact high-field) tokamak geometry."""
    config = TokamakConfig(
        R0=1.85,
        a=0.57,
        B0=12.2,  # High-field superconducting
        Ip=8.7,
        kappa=1.8,
        delta=0.4,
        q0=1.0,
        q_edge=4.0,
        n_e0=30.0,
        T_e0=20.0,
    )
    return TokamakGeometry(config)


def create_jet_geometry() -> TokamakGeometry:
    """Create JET-like tokamak geometry."""
    config = TokamakConfig(
        R0=2.96,
        a=1.25,
        B0=3.45,
        Ip=4.0,
        kappa=1.68,
        delta=0.27,
        q0=0.9,
        q_edge=3.0,
        n_e0=5.0,
        T_e0=5.0,
    )
    return TokamakGeometry(config)


if __name__ == "__main__":
    print("Tokamak Geometry Test")
    print("=" * 60)
    
    # Create ITER geometry
    geom = create_iter_geometry()
    cfg = geom.config
    
    print(f"\nITER Parameters:")
    print(f"  R0 = {cfg.R0} m")
    print(f"  a = {cfg.a} m")
    print(f"  B0 = {cfg.B0} T")
    print(f"  Ip = {cfg.Ip} MA")
    print(f"  Aspect ratio = {cfg.aspect_ratio:.2f}")
    print(f"  β_t ≈ {cfg.beta_t:.2%}")
    
    print(f"\nSafety factor profile:")
    for rho in [0.0, 0.25, 0.5, 0.75, 1.0]:
        r = rho * cfg.a
        q = geom.safety_factor(r)
        print(f"  r/a = {rho:.2f}: q = {q:.2f}")
    
    print(f"\nMagnetic field at midplane (Z=0):")
    R = torch.tensor([cfg.R0 - cfg.a, cfg.R0, cfg.R0 + cfg.a])
    Z = torch.zeros_like(R)
    phi = torch.zeros_like(R)
    B_R, B_Z, B_phi = geom.magnetic_field(R, Z, phi)
    
    for i, label in enumerate(["Inboard", "Axis", "Outboard"]):
        B_total = torch.sqrt(B_R[i]**2 + B_Z[i]**2 + B_phi[i]**2)
        print(f"  {label}: B = {B_total:.2f} T (B_φ = {B_phi[i]:.2f} T)")
    
    print("\n" + "=" * 60)
    print("Geometry module ready for tokamak simulations")
