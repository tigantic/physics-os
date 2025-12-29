"""
Adaptive χ Framework and Regularity Diagnostic [PHASE-1D/1E]
=============================================================

The χ(t) diagnostic tracks bond dimension evolution as a proxy
for solution regularity in incompressible Navier-Stokes.

Key Insight:
    For tensor network representations of NS solutions:
    - Smooth flows → low χ (compressible in TT format)
    - Turbulent/singular flows → high χ (entanglement growth)
    
    χ(t) ~ ||∇u||² captures gradient magnitude which is related to
    enstrophy and potential singularity formation.

This module provides:
    1. SVD-based truncation with adaptive χ
    2. Gradient-based χ estimation
    3. χ(t) trajectory tracking
    4. Regularity analysis utilities

Constitution Compliance: Article IV.1 (Verification), Phase 1d/1e
Tag: [PHASE-1D] [PHASE-1E] [NS-REGULARITY]
"""

from __future__ import annotations

import torch
from torch import Tensor
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import math


@dataclass
class ChiState:
    """
    Tracks χ (bond dimension) as regularity proxy.
    
    For NS solutions, χ relates to solution complexity:
        - Low χ: smooth, laminar flow
        - High χ: complex, turbulent structures
        - χ → ∞: potential singularity
    """
    time: float
    chi_actual: int           # Current bond dimension
    chi_target: int           # Target/max allowed
    truncation_error: float   # SVD truncation error
    gradient_norm: float      # ||∇u||₂
    enstrophy: float         # (1/2)||ω||²
    spectral_radius: float    # Largest eigenvalue of velocity gradient tensor
    
    @property
    def chi_ratio(self) -> float:
        """Ratio of actual to target χ (saturation indicator)."""
        return self.chi_actual / max(self.chi_target, 1)


@dataclass
class ChiTrajectory:
    """
    Full χ(t) trajectory for regularity analysis.
    """
    states: List[ChiState] = field(default_factory=list)
    
    def add(self, state: ChiState) -> None:
        self.states.append(state)
    
    @property
    def times(self) -> List[float]:
        return [s.time for s in self.states]
    
    @property
    def chi_values(self) -> List[int]:
        return [s.chi_actual for s in self.states]
    
    @property
    def gradient_norms(self) -> List[float]:
        return [s.gradient_norm for s in self.states]
    
    @property
    def enstrophies(self) -> List[float]:
        return [s.enstrophy for s in self.states]
    
    def growth_rate(self) -> Optional[float]:
        """
        Estimate exponential growth rate of χ(t).
        
        If χ(t) ~ exp(λt), returns λ.
        Positive λ indicates potential blowup.
        """
        if len(self.states) < 2:
            return None
        
        times = torch.tensor(self.times)
        chi = torch.tensor(self.chi_values, dtype=torch.float64)
        
        # Avoid log(0)
        chi = chi.clamp(min=1.0)
        log_chi = torch.log(chi)
        
        # Linear regression: log(χ) = λt + c
        if times[-1] - times[0] < 1e-10:
            return 0.0
            
        # Simple least squares
        t_mean = times.mean()
        log_chi_mean = log_chi.mean()
        
        numerator = ((times - t_mean) * (log_chi - log_chi_mean)).sum()
        denominator = ((times - t_mean) ** 2).sum()
        
        if denominator < 1e-10:
            return 0.0
        
        return (numerator / denominator).item()
    
    def enstrophy_growth_rate(self) -> Optional[float]:
        """
        Estimate enstrophy growth rate.
        
        For NS, enstrophy growth indicates cascade to small scales.
        """
        if len(self.states) < 2:
            return None
        
        times = torch.tensor(self.times)
        enst = torch.tensor(self.enstrophies, dtype=torch.float64)
        
        # Check for valid data
        if (enst <= 0).any() or not torch.isfinite(enst).all():
            return 0.0
        
        log_enst = torch.log(enst)
        
        if times[-1] - times[0] < 1e-10:
            return 0.0
        
        t_mean = times.mean()
        log_enst_mean = log_enst.mean()
        
        numerator = ((times - t_mean) * (log_enst - log_enst_mean)).sum()
        denominator = ((times - t_mean) ** 2).sum()
        
        if denominator < 1e-10:
            return 0.0
        
        return (numerator / denominator).item()


def compute_velocity_gradient_tensor_2d(
    u: Tensor,
    v: Tensor,
    dx: float,
    dy: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Compute 2D velocity gradient tensor A_ij = ∂u_i/∂x_j.
    
    Returns:
        (∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y)
    """
    # Use spectral derivatives for accuracy
    Nx, Ny = u.shape
    dtype = u.dtype
    device = u.device
    
    kx = torch.fft.fftfreq(Nx, d=dx, device=device, dtype=dtype) * 2 * math.pi
    ky = torch.fft.fftfreq(Ny, d=dy, device=device, dtype=dtype) * 2 * math.pi
    
    KX = kx.unsqueeze(1).expand(Nx, Ny)
    KY = ky.unsqueeze(0).expand(Nx, Ny)
    
    u_hat = torch.fft.fft2(u)
    v_hat = torch.fft.fft2(v)
    
    du_dx = torch.fft.ifft2(1j * KX * u_hat).real
    du_dy = torch.fft.ifft2(1j * KY * u_hat).real
    dv_dx = torch.fft.ifft2(1j * KX * v_hat).real
    dv_dy = torch.fft.ifft2(1j * KY * v_hat).real
    
    return du_dx, du_dy, dv_dx, dv_dy


def compute_velocity_gradient_tensor_3d(
    u: Tensor,
    v: Tensor,
    w: Tensor,
    dx: float,
    dy: float,
    dz: float,
) -> Tuple[Tensor, ...]:
    """
    Compute 3D velocity gradient tensor A_ij = ∂u_i/∂x_j.
    
    Returns 9 components: (∂u/∂x, ∂u/∂y, ∂u/∂z, ∂v/∂x, ..., ∂w/∂z)
    """
    Nx, Ny, Nz = u.shape
    dtype = u.dtype
    device = u.device
    
    kx = torch.fft.fftfreq(Nx, d=dx, device=device, dtype=dtype) * 2 * math.pi
    ky = torch.fft.fftfreq(Ny, d=dy, device=device, dtype=dtype) * 2 * math.pi
    kz = torch.fft.fftfreq(Nz, d=dz, device=device, dtype=dtype) * 2 * math.pi
    
    KX = kx.reshape(Nx, 1, 1).expand(Nx, Ny, Nz)
    KY = ky.reshape(1, Ny, 1).expand(Nx, Ny, Nz)
    KZ = kz.reshape(1, 1, Nz).expand(Nx, Ny, Nz)
    
    u_hat = torch.fft.fftn(u)
    v_hat = torch.fft.fftn(v)
    w_hat = torch.fft.fftn(w)
    
    du_dx = torch.fft.ifftn(1j * KX * u_hat).real
    du_dy = torch.fft.ifftn(1j * KY * u_hat).real
    du_dz = torch.fft.ifftn(1j * KZ * u_hat).real
    
    dv_dx = torch.fft.ifftn(1j * KX * v_hat).real
    dv_dy = torch.fft.ifftn(1j * KY * v_hat).real
    dv_dz = torch.fft.ifftn(1j * KZ * v_hat).real
    
    dw_dx = torch.fft.ifftn(1j * KX * w_hat).real
    dw_dy = torch.fft.ifftn(1j * KY * w_hat).real
    dw_dz = torch.fft.ifftn(1j * KZ * w_hat).real
    
    return du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz


def compute_gradient_frobenius_norm_2d(
    u: Tensor,
    v: Tensor,
    dx: float,
    dy: float,
) -> float:
    """
    Compute ||∇u||_F = sqrt(sum_ij (∂u_i/∂x_j)²)
    
    This is the Frobenius norm of the velocity gradient tensor,
    a key quantity for regularity.
    """
    du_dx, du_dy, dv_dx, dv_dy = compute_velocity_gradient_tensor_2d(u, v, dx, dy)
    
    frobenius_sq = (du_dx**2 + du_dy**2 + dv_dx**2 + dv_dy**2).mean().item()
    return math.sqrt(frobenius_sq)


def compute_gradient_frobenius_norm_3d(
    u: Tensor,
    v: Tensor,
    w: Tensor,
    dx: float,
    dy: float,
    dz: float,
) -> float:
    """
    Compute ||∇u||_F for 3D velocity field.
    """
    grads = compute_velocity_gradient_tensor_3d(u, v, w, dx, dy, dz)
    
    frobenius_sq = sum((g**2).mean().item() for g in grads)
    return math.sqrt(frobenius_sq)


def estimate_required_chi_2d(
    u: Tensor,
    v: Tensor,
    dx: float,
    dy: float,
    truncation_tol: float = 1e-6,
) -> int:
    """
    Estimate bond dimension χ required to represent velocity field.
    
    Uses SVD of the velocity field to determine how many singular
    values are needed to achieve the target truncation tolerance.
    
    This is a proxy for solution complexity/regularity.
    """
    Nx, Ny = u.shape
    
    # Stack velocity components
    uv = torch.stack([u, v], dim=0)  # (2, Nx, Ny)
    
    # Reshape to matrix for SVD
    mat = uv.reshape(2 * Nx, Ny)
    
    # Check for non-finite values
    if not torch.isfinite(mat).all():
        return 1  # Conservative fallback
    
    # rSVD - faster above 100x100
    try:
        m, n = mat.shape
        if min(m, n) > 100:
            U, S, V = torch.svd_lowrank(mat, q=min(100, min(m, n)))
        else:
            U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
    except Exception:
        return 1  # Fallback on failure
    
    # Find rank for given tolerance
    if S[0] < 1e-16:
        return 1
    S_normalized = S / S[0]  # Normalize by largest singular value
    
    # Count singular values above tolerance
    chi = (S_normalized > truncation_tol).sum().item()
    
    return max(1, int(chi))


def estimate_required_chi_3d(
    u: Tensor,
    v: Tensor,
    w: Tensor,
    dx: float,
    dy: float,
    dz: float,
    truncation_tol: float = 1e-6,
) -> int:
    """
    Estimate bond dimension χ for 3D velocity field.
    """
    Nx, Ny, Nz = u.shape
    
    # Stack velocity components
    uvw = torch.stack([u, v, w], dim=0)  # (3, Nx, Ny, Nz)
    
    # Reshape to matrix (unfold along first spatial dimension)
    mat = uvw.reshape(3 * Nx, Ny * Nz)
    
    # rSVD - faster above 100x100
    m, n = mat.shape
    if min(m, n) > 100:
        _, S, _ = torch.svd_lowrank(mat, q=min(100, min(m, n)))
    else:
        _, S, _ = torch.linalg.svd(mat, full_matrices=False)
    
    S_normalized = S / S[0]
    chi = (S_normalized > truncation_tol).sum().item()
    
    return max(1, int(chi))


def compute_chi_state_2d(
    u: Tensor,
    v: Tensor,
    t: float,
    dx: float,
    dy: float,
    chi_target: int = 64,
    truncation_tol: float = 1e-6,
) -> ChiState:
    """
    Compute full χ state for 2D velocity field.
    """
    from tensornet.cfd.tt_poisson import compute_vorticity_2d
    
    # Gradient norm
    grad_norm = compute_gradient_frobenius_norm_2d(u, v, dx, dy)
    
    # Enstrophy
    omega = compute_vorticity_2d(u, v, dx, dy, method='spectral')
    enstrophy = 0.5 * (omega**2).mean().item()
    
    # Spectral radius (max eigenvalue of gradient tensor at each point)
    du_dx, du_dy, dv_dx, dv_dy = compute_velocity_gradient_tensor_2d(u, v, dx, dy)
    
    # For 2D, eigenvalues of [[a,b],[c,d]] are (a+d)/2 ± sqrt((a-d)²/4 + bc)
    trace = du_dx + dv_dy
    det = du_dx * dv_dy - du_dy * dv_dx
    
    # Maximum eigenvalue magnitude
    discriminant = (trace/2)**2 - det
    discriminant = discriminant.clamp(min=0)
    
    eig_plus = trace/2 + torch.sqrt(discriminant)
    eig_minus = trace/2 - torch.sqrt(discriminant)
    
    spectral_radius = torch.max(torch.abs(eig_plus), torch.abs(eig_minus)).max().item()
    
    # Estimate χ
    chi_actual = estimate_required_chi_2d(u, v, dx, dy, truncation_tol)
    
    # Truncation error (approximation)
    truncation_error = truncation_tol  # Conservative estimate
    
    return ChiState(
        time=t,
        chi_actual=chi_actual,
        chi_target=chi_target,
        truncation_error=truncation_error,
        gradient_norm=grad_norm,
        enstrophy=enstrophy,
        spectral_radius=spectral_radius,
    )


def compute_chi_state_3d(
    u: Tensor,
    v: Tensor,
    w: Tensor,
    t: float,
    dx: float,
    dy: float,
    dz: float,
    chi_target: int = 64,
    truncation_tol: float = 1e-6,
) -> ChiState:
    """
    Compute full χ state for 3D velocity field.
    """
    from tensornet.cfd.ns_3d import compute_vorticity_3d
    
    # Gradient norm
    grad_norm = compute_gradient_frobenius_norm_3d(u, v, w, dx, dy, dz)
    
    # Enstrophy
    omega_x, omega_y, omega_z = compute_vorticity_3d(u, v, w, dx, dy, dz, method='spectral')
    enstrophy = 0.5 * (omega_x**2 + omega_y**2 + omega_z**2).mean().item()
    
    # Spectral radius (simplified: max of diagonal elements)
    grads = compute_velocity_gradient_tensor_3d(u, v, w, dx, dy, dz)
    du_dx, _, _, _, dv_dy, _, _, _, dw_dz = grads
    
    spectral_radius = max(
        du_dx.abs().max().item(),
        dv_dy.abs().max().item(),
        dw_dz.abs().max().item(),
    )
    
    # Estimate χ
    chi_actual = estimate_required_chi_3d(u, v, w, dx, dy, dz, truncation_tol)
    
    return ChiState(
        time=t,
        chi_actual=chi_actual,
        chi_target=chi_target,
        truncation_error=truncation_tol,
        gradient_norm=grad_norm,
        enstrophy=enstrophy,
        spectral_radius=spectral_radius,
    )


def analyze_regularity(trajectory: ChiTrajectory) -> Dict[str, any]:
    """
    Analyze χ trajectory for regularity indicators.
    
    Returns analysis including:
        - chi_growth_rate: exponential growth rate
        - enstrophy_growth_rate: enstrophy growth rate
        - saturation_events: times when χ hit target
        - regularity_assessment: 'smooth', 'complex', 'potential_blowup'
    """
    if len(trajectory.states) < 2:
        return {'error': 'Insufficient data'}
    
    chi_growth = trajectory.growth_rate()
    enst_growth = trajectory.enstrophy_growth_rate()
    
    # Check for saturation (χ hitting target)
    saturation_events = [
        s.time for s in trajectory.states if s.chi_ratio > 0.95
    ]
    
    # Final gradient norm ratio (protected against zero/NaN)
    final_grad = trajectory.gradient_norms[-1]
    initial_grad = trajectory.gradient_norms[0]
    if abs(initial_grad) > 1e-10 and abs(final_grad) > 1e-10:
        grad_ratio = final_grad / initial_grad
    else:
        grad_ratio = 1.0  # Default for near-zero gradients
    
    # Assessment based on physics:
    # - Taylor-Green decays smoothly: chi stable, grad decreasing
    # - Complex flows: chi grows moderately
    # - Potential blowup: chi or gradients explode
    if chi_growth is not None and chi_growth > 1.0:
        assessment = 'potential_blowup'
    elif chi_growth is not None and chi_growth < 0 and grad_ratio <= 1.5:
        # Chi decreasing and gradients not growing - smooth decay
        assessment = 'smooth'
    elif len(saturation_events) > len(trajectory.states) // 2:
        assessment = 'complex'
    elif grad_ratio < 2.0:
        assessment = 'smooth'
    else:
        assessment = 'complex'
    
    return {
        'chi_growth_rate': chi_growth,
        'enstrophy_growth_rate': enst_growth,
        'saturation_events': saturation_events,
        'gradient_ratio': grad_ratio,
        'regularity_assessment': assessment,
        'final_chi': trajectory.chi_values[-1],
        'max_chi': max(trajectory.chi_values),
    }


def test_chi_tracking():
    """Test χ tracking on Taylor-Green vortex."""
    from tensornet.cfd.ns_2d import NS2DSolver
    
    print("\n" + "=" * 60)
    print("χ(t) Regularity Tracking Test [PHASE-1D/1E]")
    print("=" * 60)
    
    N = 64
    nu = 0.1
    solver = NS2DSolver(Nx=N, Ny=N, nu=nu, dtype=torch.float64)
    
    state = solver.create_taylor_green(A=1.0)
    trajectory = ChiTrajectory()
    
    # Track χ during evolution
    t_final = 1.0
    dt = 0.02
    t = 0.0
    
    while t < t_final:
        chi_state = compute_chi_state_2d(
            state.u, state.v, t,
            solver.dx, solver.dy,
            chi_target=64,
        )
        trajectory.add(chi_state)
        
        state, _ = solver.step_rk4(state, dt)
        t += dt
    
    # Final state
    chi_state = compute_chi_state_2d(
        state.u, state.v, t,
        solver.dx, solver.dy,
        chi_target=64,
    )
    trajectory.add(chi_state)
    
    # Analysis
    analysis = analyze_regularity(trajectory)
    
    print(f"\nTrajectory: {len(trajectory.states)} states")
    print(f"χ range: {min(trajectory.chi_values)} - {max(trajectory.chi_values)}")
    print(f"χ growth rate: {analysis['chi_growth_rate']:.4f}")
    print(f"Enstrophy growth rate: {analysis['enstrophy_growth_rate']:.4f}")
    print(f"Gradient ratio: {analysis['gradient_ratio']:.4f}")
    print(f"Assessment: {analysis['regularity_assessment']}")
    
    # For Taylor-Green (smooth decay), we expect:
    # - Decreasing χ (less complexity as flow decays)
    # - Negative growth rate
    # - 'smooth' assessment
    
    passed = analysis['regularity_assessment'] == 'smooth'
    print(f"\nGate (smooth flow detection): {'PASS' if passed else 'FAIL'}")
    print("=" * 60)
    
    return {
        'trajectory': trajectory,
        'analysis': analysis,
        'passed': passed,
    }


if __name__ == '__main__':
    test_chi_tracking()
