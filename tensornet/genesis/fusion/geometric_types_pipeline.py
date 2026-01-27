#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                          ║
║          G E O M E T R I C   T Y P E S   P I P E L I N E                                ║
║                                                                                          ║
║                   TYPE-SAFE GEOMETRY • CONSTRAINT PRESERVATION                          ║
║                                                                                          ║
║     Types:  VectorField → Measure → Manifold → Spinor                                   ║
║     Ops:    OT → SGW → RKHS → PH → GA                                                   ║
║                                                                                          ║
║     Each geometric type has INVARIANTS that must be preserved through operations        ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

GEOMETRIC TYPES:
    VectorField  — ∇·v = 0 (divergence-free, incompressible flow)
    Measure      — ∫dμ = 1 (normalized probability measure)
    Manifold     — g_ij symmetric, det(g) > 0 (valid metric)
    Spinor       — |ψ|² = 1 (unit norm, quantum state)

GENESIS PRIMITIVES:
    OT   — Optimal Transport (moves measures)
    SGW  — Spectral Graph Wavelets (multi-scale analysis)
    RKHS — Kernel Methods (divergence computation)
    PH   — Persistent Homology (topological features)
    GA   — Geometric Algebra (rotations, transformations)

Author: HyperTensor Genesis Protocol
Date: January 27, 2026
"""

import torch
import torch.fft as fft
import time
import math
import json
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod

# ═══════════════════════════════════════════════════════════════════════════════
# GPU VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

assert torch.cuda.is_available(), "CUDA REQUIRED"
DEVICE = torch.device("cuda")

props = torch.cuda.get_device_properties(0)
print(f"\n✓ GPU: {props.name}")
print(f"✓ VRAM: {props.total_memory / 1e9:.1f} GB")


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTRAINT VIOLATION EXCEPTION
# ═══════════════════════════════════════════════════════════════════════════════

class ConstraintViolation(Exception):
    """Raised when a geometric invariant is violated."""
    def __init__(self, type_name: str, constraint: str, expected: float, actual: float):
        self.type_name = type_name
        self.constraint = constraint
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"CONSTRAINT VIOLATION in {type_name}:\n"
            f"  {constraint}\n"
            f"  Expected: {expected:.6e}\n"
            f"  Actual:   {actual:.6e}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# GEOMETRIC TYPE 1: VECTOR FIELD
# Constraint: ∇·v = 0 (divergence-free)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VectorField:
    """
    Divergence-free vector field on a 3D grid.
    
    Invariant: ∇·v = 0 (spectral divergence < tolerance)
    
    This represents incompressible fluid flow, magnetic fields, etc.
    """
    data: torch.Tensor  # Shape: [Nx, Ny, Nz, 3]
    dx: float
    tolerance: float = 1e-6
    
    def __post_init__(self):
        assert self.data.device.type == 'cuda', "VectorField must be on GPU"
        assert len(self.data.shape) == 4 and self.data.shape[-1] == 3
        self.verify_constraint("construction")
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self.data.shape[:3])
    
    def divergence(self) -> torch.Tensor:
        """Compute spectral divergence ∇·v."""
        Nx, Ny, Nz = self.shape
        kx = fft.fftfreq(Nx, d=self.dx, device=DEVICE).to(torch.float64) * 2 * math.pi
        ky = fft.fftfreq(Ny, d=self.dx, device=DEVICE).to(torch.float64) * 2 * math.pi
        kz = fft.fftfreq(Nz, d=self.dx, device=DEVICE).to(torch.float64) * 2 * math.pi
        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
        
        vx_hat = fft.fftn(self.data[..., 0])
        vy_hat = fft.fftn(self.data[..., 1])
        vz_hat = fft.fftn(self.data[..., 2])
        
        # k·v_hat (spectral divergence) - should be ~0 for curl fields
        kdotv = KX * vx_hat + KY * vy_hat + KZ * vz_hat
        return kdotv.abs().max().item()
    
    def verify_constraint(self, context: str = "") -> float:
        """Verify ∇·v = 0."""
        div = self.divergence()
        if div > self.tolerance:
            raise ConstraintViolation("VectorField", "∇·v = 0", self.tolerance, div)
        return div
    
    def project_divergence_free(self) -> "VectorField":
        """Helmholtz projection onto divergence-free subspace."""
        Nx, Ny, Nz = self.shape
        kx = fft.fftfreq(Nx, d=self.dx, device=DEVICE).to(torch.float64) * 2 * math.pi
        ky = fft.fftfreq(Ny, d=self.dx, device=DEVICE).to(torch.float64) * 2 * math.pi
        kz = fft.fftfreq(Nz, d=self.dx, device=DEVICE).to(torch.float64) * 2 * math.pi
        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
        K2 = KX**2 + KY**2 + KZ**2
        K2[0, 0, 0] = 1.0  # Avoid div by zero
        
        vx_hat = fft.fftn(self.data[..., 0])
        vy_hat = fft.fftn(self.data[..., 1])
        vz_hat = fft.fftn(self.data[..., 2])
        
        kdotv = KX * vx_hat + KY * vy_hat + KZ * vz_hat
        factor = kdotv / K2
        factor[0, 0, 0] = 0
        
        proj_data = torch.zeros_like(self.data)
        proj_data[..., 0] = fft.ifftn(vx_hat - KX * factor).real
        proj_data[..., 1] = fft.ifftn(vy_hat - KY * factor).real
        proj_data[..., 2] = fft.ifftn(vz_hat - KZ * factor).real
        
        return VectorField(proj_data, self.dx, self.tolerance)
    
    @classmethod
    def random_solenoidal(cls, shape: Tuple[int, int, int], dx: float = 0.1) -> "VectorField":
        """Create a random divergence-free vector field via curl of potential."""
        Nx, Ny, Nz = shape
        
        # Create random potential field
        psi = torch.randn(Nx, Ny, Nz, 3, device=DEVICE, dtype=torch.float64)
        
        # Wavenumbers
        kx = fft.fftfreq(Nx, d=dx, device=DEVICE).to(torch.float64) * 2 * math.pi
        ky = fft.fftfreq(Ny, d=dx, device=DEVICE).to(torch.float64) * 2 * math.pi
        kz = fft.fftfreq(Nz, d=dx, device=DEVICE).to(torch.float64) * 2 * math.pi
        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
        
        # FFT of potential
        psi_x_hat = fft.fftn(psi[..., 0])
        psi_y_hat = fft.fftn(psi[..., 1])
        psi_z_hat = fft.fftn(psi[..., 2])
        
        # Curl in Fourier: v = ∇×ψ = ik×ψ_hat
        vx_hat = 1j * (KY * psi_z_hat - KZ * psi_y_hat)
        vy_hat = 1j * (KZ * psi_x_hat - KX * psi_z_hat)
        vz_hat = 1j * (KX * psi_y_hat - KY * psi_x_hat)
        
        # Zero Nyquist modes for real output
        nyquist_mask = torch.zeros((Nx, Ny, Nz), dtype=torch.bool, device=DEVICE)
        if Nx % 2 == 0:
            nyquist_mask[Nx // 2, :, :] = True
        if Ny % 2 == 0:
            nyquist_mask[:, Ny // 2, :] = True
        if Nz % 2 == 0:
            nyquist_mask[:, :, Nz // 2] = True
        
        vx_hat[nyquist_mask] = 0
        vy_hat[nyquist_mask] = 0
        vz_hat[nyquist_mask] = 0
        
        data = torch.zeros(Nx, Ny, Nz, 3, device=DEVICE, dtype=torch.float64)
        data[..., 0] = fft.ifftn(vx_hat).real
        data[..., 1] = fft.ifftn(vy_hat).real
        data[..., 2] = fft.ifftn(vz_hat).real
        
        # Normalize to reasonable magnitude
        max_val = data.abs().max()
        if max_val > 0:
            data = data / max_val
        
        return cls(data, dx)


# ═══════════════════════════════════════════════════════════════════════════════
# GEOMETRIC TYPE 2: MEASURE
# Constraint: ∫dμ = 1 (normalized probability)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Measure:
    """
    Probability measure on a grid.
    
    Invariant: ∫dμ = 1, μ ≥ 0 (normalized, non-negative)
    """
    density: torch.Tensor  # Shape: [N] or [Nx, Ny, ...]
    dx: float
    tolerance: float = 1e-6
    
    def __post_init__(self):
        assert self.density.device.type == 'cuda', "Measure must be on GPU"
        self.verify_constraint("construction")
    
    @property
    def total_mass(self) -> float:
        """Compute ∫dμ."""
        return (self.density.sum() * self.dx ** self.density.dim()).item()
    
    @property
    def is_positive(self) -> bool:
        """Check μ ≥ 0."""
        return (self.density >= -self.tolerance).all().item()
    
    def verify_constraint(self, context: str = "") -> float:
        """Verify ∫dμ = 1 and μ ≥ 0."""
        mass = self.total_mass
        if abs(mass - 1.0) > self.tolerance:
            raise ConstraintViolation("Measure", "∫dμ = 1", 1.0, mass)
        if not self.is_positive:
            min_val = self.density.min().item()
            raise ConstraintViolation("Measure", "μ ≥ 0", 0.0, min_val)
        return abs(mass - 1.0)
    
    def normalize(self) -> "Measure":
        """Renormalize to unit mass."""
        mass = self.density.sum() * self.dx ** self.density.dim()
        new_density = self.density / mass
        return Measure(new_density, self.dx, self.tolerance)
    
    @classmethod
    def gaussian(cls, n: int, mean: float = 0.0, std: float = 1.0, 
                 bounds: Tuple[float, float] = (-5.0, 5.0)) -> "Measure":
        """Create a Gaussian probability measure."""
        dx = (bounds[1] - bounds[0]) / n
        x = torch.linspace(bounds[0], bounds[1], n, device=DEVICE, dtype=torch.float64)
        density = torch.exp(-0.5 * ((x - mean) / std) ** 2)
        density = density / (density.sum() * dx)  # Normalize
        return cls(density, dx)


# ═══════════════════════════════════════════════════════════════════════════════
# GEOMETRIC TYPE 3: MANIFOLD (Riemannian Metric)
# Constraint: g_ij = g_ji, det(g) > 0
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass  
class Manifold:
    """
    Riemannian manifold represented by metric tensor field.
    
    Invariants:
    - g_ij = g_ji (symmetry)
    - det(g) > 0 (positive definite)
    """
    metric: torch.Tensor  # Shape: [N_points, dim, dim]
    points: torch.Tensor  # Shape: [N_points, dim]
    tolerance: float = 1e-6
    
    def __post_init__(self):
        assert self.metric.device.type == 'cuda', "Manifold must be on GPU"
        self.verify_constraint("construction")
    
    @property
    def dim(self) -> int:
        return self.metric.shape[-1]
    
    @property
    def n_points(self) -> int:
        return self.metric.shape[0]
    
    def verify_constraint(self, context: str = "") -> Dict[str, float]:
        """Verify symmetry and positive definiteness."""
        results = {}
        
        # Symmetry: g - g^T = 0
        asymmetry = (self.metric - self.metric.transpose(-1, -2)).abs().max().item()
        results["asymmetry"] = asymmetry
        if asymmetry > self.tolerance:
            raise ConstraintViolation("Manifold", "g_ij = g_ji", self.tolerance, asymmetry)
        
        # Positive definite: det(g) > 0
        dets = torch.linalg.det(self.metric)
        min_det = dets.min().item()
        results["min_det"] = min_det
        if min_det <= 0:
            raise ConstraintViolation("Manifold", "det(g) > 0", self.tolerance, min_det)
        
        return results
    
    def geodesic_distance(self, i: int, j: int) -> float:
        """Approximate geodesic distance between points i and j."""
        # For small distances, use metric at midpoint
        diff = self.points[j] - self.points[i]
        mid_metric = (self.metric[i] + self.metric[j]) / 2
        return torch.sqrt(diff @ mid_metric @ diff).item()
    
    @classmethod
    def euclidean(cls, points: torch.Tensor) -> "Manifold":
        """Create Euclidean manifold (flat metric)."""
        n, dim = points.shape
        metric = torch.eye(dim, device=DEVICE, dtype=torch.float64).unsqueeze(0).expand(n, -1, -1).clone()
        return cls(metric, points)
    
    @classmethod
    def sphere_embedded(cls, n_points: int, radius: float = 1.0) -> "Manifold":
        """Create metric on a sphere (2D surface in 3D)."""
        # Points on sphere
        torch.manual_seed(42)
        theta = torch.rand(n_points, device=DEVICE, dtype=torch.float64) * math.pi
        phi = torch.rand(n_points, device=DEVICE, dtype=torch.float64) * 2 * math.pi
        
        # Spherical coordinates
        points = torch.stack([theta, phi], dim=1)
        
        # Induced metric g = diag(R², R²sin²θ)
        metric = torch.zeros(n_points, 2, 2, device=DEVICE, dtype=torch.float64)
        metric[:, 0, 0] = radius ** 2
        metric[:, 1, 1] = radius ** 2 * torch.sin(theta) ** 2
        # Ensure positive definite by adding small regularization at poles
        metric[:, 1, 1] = torch.clamp(metric[:, 1, 1], min=1e-6)
        
        return cls(metric, points)


# ═══════════════════════════════════════════════════════════════════════════════
# GEOMETRIC TYPE 4: SPINOR
# Constraint: |ψ|² = 1 (unit norm)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Spinor:
    """
    Spinor field (e.g., quantum state, Dirac spinor).
    
    Invariant: ⟨ψ|ψ⟩ = 1 (unit normalization)
    """
    components: torch.Tensor  # Shape: [N, n_components] complex
    tolerance: float = 1e-6
    
    def __post_init__(self):
        assert self.components.device.type == 'cuda', "Spinor must be on GPU"
        if not self.components.is_complex():
            self.components = self.components.to(torch.complex128)
        self.verify_constraint("construction")
    
    @property
    def n_states(self) -> int:
        return self.components.shape[0]
    
    @property
    def n_components(self) -> int:
        return self.components.shape[1] if len(self.components.shape) > 1 else 1
    
    def norm_squared(self) -> torch.Tensor:
        """Compute |ψ|² for each state."""
        return (self.components.conj() * self.components).real.sum(dim=-1)
    
    def verify_constraint(self, context: str = "") -> float:
        """Verify ⟨ψ|ψ⟩ = 1."""
        norms = self.norm_squared()
        max_deviation = (norms - 1.0).abs().max().item()
        if max_deviation > self.tolerance:
            raise ConstraintViolation("Spinor", "|ψ|² = 1", 1.0, norms.mean().item())
        return max_deviation
    
    def normalize(self) -> "Spinor":
        """Renormalize to unit norm."""
        norms = torch.sqrt(self.norm_squared()).unsqueeze(-1)
        new_components = self.components / norms
        return Spinor(new_components, self.tolerance)
    
    def inner_product(self, other: "Spinor") -> torch.Tensor:
        """Compute ⟨self|other⟩."""
        return (self.components.conj() * other.components).sum(dim=-1)
    
    @classmethod
    def random_normalized(cls, n_states: int, n_components: int = 2) -> "Spinor":
        """Create random unit-normalized spinors."""
        real = torch.randn(n_states, n_components, device=DEVICE, dtype=torch.float64)
        imag = torch.randn(n_states, n_components, device=DEVICE, dtype=torch.float64)
        components = torch.complex(real, imag)
        norms = torch.sqrt((components.conj() * components).real.sum(dim=-1, keepdim=True))
        components = components / norms
        return cls(components)


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE STAGES - PRIMITIVES OPERATING ON GEOMETRIC TYPES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TypeStageResult:
    """Result from a geometric type stage."""
    stage: int
    type_name: str
    primitive: str
    constraint_before: float
    constraint_after: float
    constraint_preserved: bool
    time_seconds: float
    details: Dict[str, Any]


def stage1_vectorfield_sgw(v: VectorField) -> Tuple[TypeStageResult, Dict]:
    """
    Apply Spectral Graph Wavelets to VectorField.
    
    Constraint: ∇·v = 0 must be preserved
    """
    print("━━━ STAGE 1: VectorField + SGW ━━━")
    print(f"  Input: VectorField {v.shape}, ∇·v = 0")
    
    start = time.perf_counter()
    div_before = v.divergence()
    print(f"  Divergence before: {div_before:.2e}")
    
    # Apply wavelet-like multi-scale decomposition
    # Low-pass filter (preserves divergence-free)
    Nx, Ny, Nz = v.shape
    kx = fft.fftfreq(Nx, d=v.dx, device=DEVICE) * 2 * math.pi
    ky = fft.fftfreq(Ny, d=v.dx, device=DEVICE) * 2 * math.pi
    kz = fft.fftfreq(Nz, d=v.dx, device=DEVICE) * 2 * math.pi
    KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
    K = torch.sqrt(KX**2 + KY**2 + KZ**2)
    
    # Mexican hat wavelet in Fourier: k² exp(-k²/2σ²)
    sigma = 2.0
    wavelet = K**2 * torch.exp(-K**2 / (2 * sigma**2))
    wavelet[0, 0, 0] = 0  # Zero DC
    
    # Apply to each component
    filtered = torch.zeros_like(v.data)
    for i in range(3):
        v_hat = fft.fftn(v.data[..., i])
        filtered[..., i] = fft.ifftn(v_hat * wavelet).real
    
    # Project back to divergence-free (wavelet may introduce small divergence)
    v_filtered = VectorField(filtered, v.dx, v.tolerance * 10)  # Relaxed for filter
    v_out = v_filtered.project_divergence_free()
    
    div_after = v_out.divergence()
    elapsed = time.perf_counter() - start
    
    print(f"  Divergence after:  {div_after:.2e}")
    print(f"  ✓ Constraint preserved: {div_after < v.tolerance}")
    print(f"  ✓ Time: {elapsed:.3f}s")
    print("")
    
    result = TypeStageResult(
        stage=1,
        type_name="VectorField",
        primitive="SGW",
        constraint_before=div_before,
        constraint_after=div_after,
        constraint_preserved=div_after < v.tolerance,
        time_seconds=elapsed,
        details={"wavelet_sigma": sigma, "energy_ratio": (filtered**2).sum().item() / (v.data**2).sum().item()}
    )
    
    return result, {"vectorfield": v_out}


def stage2_measure_ot(mu: Measure, nu: Measure) -> Tuple[TypeStageResult, Dict]:
    """
    Apply Optimal Transport between two Measures.
    
    Constraint: ∫dμ = 1 must be preserved for both
    """
    print("━━━ STAGE 2: Measure + OT ━━━")
    print(f"  Input: Measure μ (mass={mu.total_mass:.6f}), ν (mass={nu.total_mass:.6f})")
    
    start = time.perf_counter()
    mass_before_mu = mu.total_mass
    mass_before_nu = nu.total_mass
    
    # Compute barycenter (midpoint in Wasserstein space)
    # Simplified: arithmetic mean with renormalization
    alpha = 0.5
    bary_density = alpha * mu.density + (1 - alpha) * nu.density
    bary = Measure(bary_density, mu.dx, mu.tolerance)
    bary = bary.normalize()
    
    # Compute Wasserstein-like distance (simplified 1D)
    x = torch.linspace(-5, 5, len(mu.density), device=DEVICE, dtype=torch.float64)
    mean_mu = (x * mu.density).sum() * mu.dx
    mean_nu = (x * nu.density).sum() * nu.dx
    var_mu = ((x - mean_mu)**2 * mu.density).sum() * mu.dx
    var_nu = ((x - mean_nu)**2 * nu.density).sum() * nu.dx
    
    # W2 for Gaussians: |m1-m2|² + (σ1-σ2)²
    w2_approx = (mean_mu - mean_nu)**2 + (torch.sqrt(var_mu) - torch.sqrt(var_nu))**2
    
    mass_after = bary.total_mass
    elapsed = time.perf_counter() - start
    
    print(f"  Barycenter mass: {mass_after:.6f}")
    print(f"  W₂ distance: {w2_approx.item():.6f}")
    print(f"  ✓ Constraint preserved: {abs(mass_after - 1.0) < mu.tolerance}")
    print(f"  ✓ Time: {elapsed:.3f}s")
    print("")
    
    result = TypeStageResult(
        stage=2,
        type_name="Measure",
        primitive="OT",
        constraint_before=abs(mass_before_mu - 1.0),
        constraint_after=abs(mass_after - 1.0),
        constraint_preserved=abs(mass_after - 1.0) < mu.tolerance,
        time_seconds=elapsed,
        details={"w2_distance": w2_approx.item(), "alpha": alpha}
    )
    
    return result, {"measure": bary, "w2": w2_approx.item()}


def stage3_manifold_ph(M: Manifold) -> Tuple[TypeStageResult, Dict]:
    """
    Apply Persistent Homology to analyze Manifold structure.
    
    Constraint: g_ij symmetric, det(g) > 0 must be preserved
    """
    print("━━━ STAGE 3: Manifold + PH ━━━")
    print(f"  Input: Manifold with {M.n_points} points, dim={M.dim}")
    
    start = time.perf_counter()
    constraints_before = M.verify_constraint("before PH")
    
    # Compute geodesic distance matrix
    n = M.n_points
    dist_matrix = torch.zeros(n, n, device=DEVICE, dtype=torch.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = M.geodesic_distance(i, j)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    
    # Simple connected component analysis (H₀)
    # Count components at different thresholds
    thresholds = torch.linspace(0, dist_matrix.max().item(), 20, device=DEVICE)
    betti_0 = []
    
    for eps in thresholds:
        # Adjacency at threshold eps
        adj = (dist_matrix <= eps.item()) & (dist_matrix > 0)
        adj = adj | adj.T | torch.eye(n, device=DEVICE, dtype=torch.bool)
        
        # Count connected components via matrix power convergence
        conn = adj.float()
        for _ in range(int(math.log2(n)) + 1):
            conn = (conn @ conn > 0).float()
        n_components = len(torch.unique(conn.argmax(dim=1)))
        betti_0.append(n_components)
    
    # Manifold unchanged (PH is read-only analysis)
    constraints_after = M.verify_constraint("after PH")
    elapsed = time.perf_counter() - start
    
    print(f"  Betti₀ range: [{min(betti_0)}, {max(betti_0)}]")
    print(f"  Metric symmetry: {constraints_after['asymmetry']:.2e}")
    print(f"  Min det(g): {constraints_after['min_det']:.6f}")
    print(f"  ✓ Constraint preserved: True (read-only)")
    print(f"  ✓ Time: {elapsed:.3f}s")
    print("")
    
    result = TypeStageResult(
        stage=3,
        type_name="Manifold",
        primitive="PH",
        constraint_before=constraints_before['asymmetry'],
        constraint_after=constraints_after['asymmetry'],
        constraint_preserved=True,
        time_seconds=elapsed,
        details={"betti_0_final": betti_0[-1], "diameter": dist_matrix.max().item()}
    )
    
    return result, {"manifold": M, "distance_matrix": dist_matrix, "betti_0": betti_0}


def stage4_spinor_ga(psi: Spinor) -> Tuple[TypeStageResult, Dict]:
    """
    Apply Geometric Algebra rotation to Spinor.
    
    Constraint: |ψ|² = 1 must be preserved (unitary transformation)
    """
    print("━━━ STAGE 4: Spinor + GA ━━━")
    print(f"  Input: Spinor with {psi.n_states} states, {psi.n_components} components")
    
    start = time.perf_counter()
    norm_before = psi.norm_squared().mean().item()
    
    # SU(2) rotation (spinor transformation)
    # R = exp(-iθσ/2) where σ is Pauli matrix
    theta = math.pi / 4  # 45° rotation
    
    # Pauli-Z rotation: diag(e^{-iθ/2}, e^{iθ/2})
    phase = torch.exp(torch.tensor(-1j * theta / 2, device=DEVICE))
    phase_conj = torch.exp(torch.tensor(1j * theta / 2, device=DEVICE))
    
    # Apply to 2-component spinors
    if psi.n_components == 2:
        rotated = torch.zeros_like(psi.components)
        rotated[:, 0] = psi.components[:, 0] * phase
        rotated[:, 1] = psi.components[:, 1] * phase_conj
    else:
        # General case: just phase rotation
        rotated = psi.components * phase
    
    psi_out = Spinor(rotated, psi.tolerance)
    norm_after = psi_out.norm_squared().mean().item()
    
    # Verify unitarity: inner products preserved
    inner_before = (psi.components[0].conj() @ psi.components[1]).abs().item() if psi.n_states > 1 else 0
    inner_after = (psi_out.components[0].conj() @ psi_out.components[1]).abs().item() if psi.n_states > 1 else 0
    
    elapsed = time.perf_counter() - start
    
    print(f"  Rotation angle: {theta:.4f} rad ({math.degrees(theta):.1f}°)")
    print(f"  Norm before: {norm_before:.10f}")
    print(f"  Norm after:  {norm_after:.10f}")
    print(f"  ✓ Constraint preserved: {abs(norm_after - 1.0) < psi.tolerance}")
    print(f"  ✓ Time: {elapsed:.3f}s")
    print("")
    
    result = TypeStageResult(
        stage=4,
        type_name="Spinor",
        primitive="GA",
        constraint_before=abs(norm_before - 1.0),
        constraint_after=abs(norm_after - 1.0),
        constraint_preserved=abs(norm_after - 1.0) < psi.tolerance,
        time_seconds=elapsed,
        details={"rotation_angle": theta, "inner_product_preserved": abs(inner_before - inner_after) < 1e-10}
    )
    
    return result, {"spinor": psi_out, "rotation_angle": theta}


def stage5_measure_rkhs(mu: Measure, nu: Measure) -> Tuple[TypeStageResult, Dict]:
    """
    Compute RKHS divergence (MMD) between Measures.
    
    Constraint: Measures remain normalized
    """
    print("━━━ STAGE 5: Measure + RKHS ━━━")
    print(f"  Input: Two Measures for MMD computation")
    
    start = time.perf_counter()
    mass_mu = mu.total_mass
    mass_nu = nu.total_mass
    
    # RBF kernel MMD
    n = len(mu.density)
    x = torch.linspace(-5, 5, n, device=DEVICE, dtype=torch.float64)
    
    # Kernel matrix K(x_i, x_j) = exp(-|x_i - x_j|² / 2σ²)
    sigma = 1.0
    diff = x.unsqueeze(0) - x.unsqueeze(1)
    K = torch.exp(-diff**2 / (2 * sigma**2))
    
    # MMD² = E[K(X,X')] + E[K(Y,Y')] - 2E[K(X,Y)]
    # Using density weighting
    dx = mu.dx
    
    term_xx = (mu.density.unsqueeze(0) * K * mu.density.unsqueeze(1)).sum() * dx**2
    term_yy = (nu.density.unsqueeze(0) * K * nu.density.unsqueeze(1)).sum() * dx**2
    term_xy = (mu.density.unsqueeze(0) * K * nu.density.unsqueeze(1)).sum() * dx**2
    
    mmd_squared = term_xx + term_yy - 2 * term_xy
    mmd = torch.sqrt(torch.clamp(mmd_squared, min=0)).item()
    
    elapsed = time.perf_counter() - start
    
    print(f"  Kernel: RBF (σ={sigma})")
    print(f"  MMD²: {mmd_squared.item():.6f}")
    print(f"  MMD:  {mmd:.6f}")
    print(f"  ✓ Masses unchanged: μ={mass_mu:.6f}, ν={mass_nu:.6f}")
    print(f"  ✓ Time: {elapsed:.3f}s")
    print("")
    
    result = TypeStageResult(
        stage=5,
        type_name="Measure",
        primitive="RKHS",
        constraint_before=abs(mass_mu - 1.0),
        constraint_after=abs(mass_mu - 1.0),  # Read-only
        constraint_preserved=True,
        time_seconds=elapsed,
        details={"mmd": mmd, "mmd_squared": mmd_squared.item(), "kernel_sigma": sigma}
    )
    
    return result, {"mmd": mmd}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_geometric_types_pipeline(grid_size: int = 32) -> Dict:
    """
    Run the Geometric Types Pipeline.
    
    Creates each geometric type and passes through relevant primitives,
    verifying constraints are preserved at each step.
    """
    print("")
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║                                                                              ║")
    print("║     G E O M E T R I C   T Y P E S   P I P E L I N E                         ║")
    print("║                                                                              ║")
    print("║     VectorField → Measure → Manifold → Spinor                               ║")
    print("║     ∇·v=0         ∫dμ=1     g=gᵀ,det>0   |ψ|²=1                              ║")
    print("║                                                                              ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print("")
    
    results: List[TypeStageResult] = []
    start_total = time.perf_counter()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Create Geometric Types
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("Creating geometric types...")
    
    # VectorField: divergence-free
    v = VectorField.random_solenoidal((grid_size, grid_size, grid_size))
    print(f"  ✓ VectorField: {v.shape}, ∇·v = {v.divergence():.2e}")
    
    # Measures: normalized probability
    mu = Measure.gaussian(256, mean=-0.5, std=0.8)
    nu = Measure.gaussian(256, mean=0.5, std=1.2)
    print(f"  ✓ Measure μ: mass = {mu.total_mass:.6f}")
    print(f"  ✓ Measure ν: mass = {nu.total_mass:.6f}")
    
    # Manifold: valid metric
    M = Manifold.sphere_embedded(64)
    print(f"  ✓ Manifold: {M.n_points} points, dim={M.dim}")
    
    # Spinor: unit norm
    psi = Spinor.random_normalized(32, n_components=2)
    print(f"  ✓ Spinor: {psi.n_states} states, |ψ|² = {psi.norm_squared().mean():.10f}")
    print("")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Stage 1: VectorField + SGW
    # ═══════════════════════════════════════════════════════════════════════════
    
    result1, data1 = stage1_vectorfield_sgw(v)
    results.append(result1)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Stage 2: Measure + OT
    # ═══════════════════════════════════════════════════════════════════════════
    
    result2, data2 = stage2_measure_ot(mu, nu)
    results.append(result2)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Stage 3: Manifold + PH
    # ═══════════════════════════════════════════════════════════════════════════
    
    result3, data3 = stage3_manifold_ph(M)
    results.append(result3)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Stage 4: Spinor + GA
    # ═══════════════════════════════════════════════════════════════════════════
    
    result4, data4 = stage4_spinor_ga(psi)
    results.append(result4)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Stage 5: Measure + RKHS
    # ═══════════════════════════════════════════════════════════════════════════
    
    result5, data5 = stage5_measure_rkhs(mu, nu)
    results.append(result5)
    
    total_time = time.perf_counter() - start_total
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║                                                                              ║")
    print("║              G E O M E T R I C   T Y P E S   S U M M A R Y                  ║")
    print("║                                                                              ║")
    print("╠══════════════════════════════════════════════════════════════════════════════╣")
    print("║  Stage │ Type        │ Primitive │ Constraint  │ Time                       ║")
    print("║  ──────┼─────────────┼───────────┼─────────────┼────────                    ║")
    
    all_preserved = True
    for r in results:
        status = "✓ PRESERVED" if r.constraint_preserved else "✗ VIOLATED"
        line = f"  {r.stage}     │ {r.type_name:11s} │ {r.primitive:9s} │ {status:11s} │ {r.time_seconds:.3f}s"
        print(f"║{line}".ljust(79) + "║")
        all_preserved = all_preserved and r.constraint_preserved
    
    print("╠══════════════════════════════════════════════════════════════════════════════╣")
    print(f"║  Total Time: {total_time:.3f}s".ljust(79) + "║")
    
    if all_preserved:
        print("║                                                                              ║")
        print("║  ★★★ ALL GEOMETRIC CONSTRAINTS PRESERVED ★★★                              ║")
        print("║                                                                              ║")
    else:
        print("║                                                                              ║")
        print("║  ⚠ SOME CONSTRAINTS VIOLATED                                               ║")
        print("║                                                                              ║")
    
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    
    # Attestation
    attestation = {
        "pipeline": "GEOMETRIC TYPES PIPELINE",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "types": ["VectorField", "Measure", "Manifold", "Spinor"],
        "primitives": ["SGW", "OT", "PH", "GA", "RKHS"],
        "total_time_seconds": total_time,
        "all_constraints_preserved": all_preserved,
        "stages": [
            {
                "stage": r.stage,
                "type": r.type_name,
                "primitive": r.primitive,
                "constraint_before": r.constraint_before,
                "constraint_after": r.constraint_after,
                "preserved": r.constraint_preserved,
                "time": r.time_seconds,
            }
            for r in results
        ]
    }
    
    attestation_str = json.dumps(attestation, indent=2)
    sha256_hash = hashlib.sha256(attestation_str.encode()).hexdigest()
    attestation["sha256"] = sha256_hash
    
    with open("GEOMETRIC_TYPES_PIPELINE_ATTESTATION.json", "w") as f:
        json.dump(attestation, f, indent=2)
    
    print("")
    print(f"  ✓ Attestation: GEOMETRIC_TYPES_PIPELINE_ATTESTATION.json")
    print(f"    SHA256: {sha256_hash[:32]}...")
    print("")
    
    return {
        "results": results,
        "total_time": total_time,
        "all_preserved": all_preserved,
        "attestation": attestation
    }


if __name__ == "__main__":
    run_geometric_types_pipeline()
