#!/usr/bin/env python3
"""
QTT-Spectral Poisson Solver
===========================

Production-grade spectral Poisson solver for 3D QTT turbulence.

PROBLEM:
    Solve ∇²ψ = f with periodic BCs in 3D
    
SOLUTION:
    In Fourier space: ψ̂(k) = f̂(k) / |k|² (where k ≠ 0)
    
APPROACH:
    QTT-Hybrid: Convert to dense → FFT solve → back to QTT
    
    This is EXACT (machine precision) unlike Jacobi iteration.
    The memory cost is O(N³) temporarily, but:
    1. Only 3 fields needed (one per component)
    2. At 128³, this is 8 MB per field (trivial vs 8GB GPU)
    3. FFT is O(N³ log N) but faster than O(N³) Jacobi iterations
    
    For 256³, memory is 64 MB per field — still fits easily.
    
    Future: Pure QTT spectral via QTT-FFT (more complex, same accuracy).

VALIDATION:
    Error < 1e-6 for analytical solutions on 32³, 64³, 128³

Author: HyperTensor Team
Date: 2026-02-05
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class SpectralPoissonConfig:
    """Configuration for spectral Poisson solver."""
    n_bits: int                     # Bits per dimension (N = 2^n_bits)
    L: float = 2 * math.pi          # Domain size
    device: str = "cuda"            # Device
    dtype: torch.dtype = torch.float32  # Data type
    regularization: float = 1e-12   # Regularization for k=0 mode


@dataclass
class SpectralPoissonResult:
    """Result of spectral Poisson solve."""
    solution: Tensor            # ψ in dense format (N, N, N)
    residual: float            # ||∇²ψ - f|| / ||f||
    k_max_used: float          # Maximum wavenumber used


def build_wavenumber_grid(
    N: int,
    L: float,
    device: str,
    dtype: torch.dtype,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Build 3D wavenumber grid for spectral methods.
    
    Args:
        N: Grid points per dimension
        L: Domain size
        device: Target device
        dtype: Data type
    
    Returns:
        kx, ky, kz: Wavenumber components (N, N, N)
        k_sq: |k|² (N, N, N)
    """
    # fftfreq returns frequencies n/N where n = 0, 1, ..., N/2-1, -N/2, ..., -1
    # For domain [0, L], actual wavenumber is k = 2πn/L
    # fftfreq with d=1/N gives n directly, then scale by 2π/L
    k = torch.fft.fftfreq(N, d=1.0/N, device=device, dtype=dtype)
    k = k * (2 * math.pi / L)
    
    # 3D grid
    kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
    
    # |k|²
    k_sq = kx**2 + ky**2 + kz**2
    
    return kx, ky, kz, k_sq


def spectral_poisson_solve(
    f: Tensor,
    L: float = 2 * math.pi,
    regularization: float = 1e-12,
) -> SpectralPoissonResult:
    """
    Solve ∇²ψ = f via FFT spectral method.
    
    The solution in Fourier space is:
        ψ̂(k) = f̂(k) / |k|²
    
    For k = 0, we set ψ̂(0) = 0 (zero-mean constraint).
    
    Args:
        f: RHS field (N, N, N)
        L: Domain size (assumes periodic [0, L]³)
        regularization: Small value for k=0 stability
    
    Returns:
        SpectralPoissonResult with solution and diagnostics
    """
    N = f.shape[0]
    device = f.device
    dtype = f.dtype
    
    # Build wavenumber grid
    kx, ky, kz, k_sq = build_wavenumber_grid(N, L, device, dtype)
    
    # Forward FFT
    f_hat = torch.fft.fftn(f)
    
    # Solve: ψ̂ = -f̂ / |k|²
    # Because ∇² in Fourier space is multiplication by -|k|²
    # So ∇²ψ = f → -|k|²ψ̂ = f̂ → ψ̂ = -f̂ / |k|²
    # Handle k=0 mode: set to 0 (zero-mean solution)
    k_sq_reg = k_sq.clone()
    k_sq_reg[0, 0, 0] = 1.0  # Avoid division by zero
    
    psi_hat = -f_hat / k_sq_reg
    psi_hat[0, 0, 0] = 0.0  # Zero-mean constraint
    
    # Inverse FFT
    psi = torch.fft.ifftn(psi_hat).real
    
    # Compute residual: ||∇²ψ - f|| / ||f||
    lap_psi_hat = -k_sq * psi_hat
    lap_psi = torch.fft.ifftn(lap_psi_hat).real
    
    residual_norm = torch.norm(lap_psi - f).item()
    f_norm = torch.norm(f).item()
    residual = residual_norm / (f_norm + 1e-15)
    
    k_max = math.sqrt(k_sq.max().item())
    
    return SpectralPoissonResult(
        solution=psi,
        residual=residual,
        k_max_used=k_max,
    )


def spectral_biot_savart(
    omega: List[Tensor],
    L: float = 2 * math.pi,
) -> List[Tensor]:
    """
    Recover velocity from vorticity via Biot-Savart in spectral space.
    
    Given ω = ∇×u and ∇·u = 0, we can write:
        û(k) = i k × ω̂(k) / |k|²
    
    Args:
        omega: Vorticity components [ωx, ωy, ωz], each (N, N, N)
        L: Domain size
    
    Returns:
        Velocity components [ux, uy, uz], each (N, N, N)
    """
    N = omega[0].shape[0]
    device = omega[0].device
    dtype = omega[0].dtype
    
    # Build wavenumber grid
    kx, ky, kz, k_sq = build_wavenumber_grid(N, L, device, dtype)
    
    # Regularize k=0
    k_sq_reg = k_sq.clone()
    k_sq_reg[0, 0, 0] = 1.0
    
    # Forward FFT of vorticity
    omega_hat = [torch.fft.fftn(omega[i]) for i in range(3)]
    
    # Biot-Savart in Fourier space:
    # û = i k × ω̂ / |k|²
    # 
    # (k × ω)_x = ky * ωz - kz * ωy
    # (k × ω)_y = kz * ωx - kx * ωz
    # (k × ω)_z = kx * ωy - ky * ωx
    
    cross_x = ky * omega_hat[2] - kz * omega_hat[1]
    cross_y = kz * omega_hat[0] - kx * omega_hat[2]
    cross_z = kx * omega_hat[1] - ky * omega_hat[0]
    
    # û = i * cross / |k|² = cross / |k|² * i
    # Note: multiplying by i in Fourier space → 90° phase shift
    u_hat_x = 1j * cross_x / k_sq_reg
    u_hat_y = 1j * cross_y / k_sq_reg
    u_hat_z = 1j * cross_z / k_sq_reg
    
    # Zero mean
    u_hat_x[0, 0, 0] = 0.0
    u_hat_y[0, 0, 0] = 0.0
    u_hat_z[0, 0, 0] = 0.0
    
    # Inverse FFT
    ux = torch.fft.ifftn(u_hat_x).real
    uy = torch.fft.ifftn(u_hat_y).real
    uz = torch.fft.ifftn(u_hat_z).real
    
    return [ux, uy, uz]


def spectral_curl(
    f: List[Tensor],
    L: float = 2 * math.pi,
) -> List[Tensor]:
    """
    Compute curl of vector field in spectral space.
    
    ∇×f = (∂fz/∂y - ∂fy/∂z, ∂fx/∂z - ∂fz/∂x, ∂fy/∂x - ∂fx/∂y)
    
    In Fourier: ∂/∂x → i*kx
    
    Args:
        f: Vector field components [fx, fy, fz], each (N, N, N)
        L: Domain size
    
    Returns:
        Curl components [curlx, curly, curlz], each (N, N, N)
    """
    N = f[0].shape[0]
    device = f[0].device
    dtype = f[0].dtype
    
    kx, ky, kz, _ = build_wavenumber_grid(N, L, device, dtype)
    
    # Forward FFT
    f_hat = [torch.fft.fftn(f[i]) for i in range(3)]
    
    # Curl in Fourier: (ik) × f
    curl_hat_x = 1j * ky * f_hat[2] - 1j * kz * f_hat[1]
    curl_hat_y = 1j * kz * f_hat[0] - 1j * kx * f_hat[2]
    curl_hat_z = 1j * kx * f_hat[1] - 1j * ky * f_hat[0]
    
    # Inverse FFT
    curl_x = torch.fft.ifftn(curl_hat_x).real
    curl_y = torch.fft.ifftn(curl_hat_y).real
    curl_z = torch.fft.ifftn(curl_hat_z).real
    
    return [curl_x, curl_y, curl_z]


def spectral_laplacian(
    f: Tensor,
    L: float = 2 * math.pi,
) -> Tensor:
    """
    Compute Laplacian of scalar field in spectral space.
    
    ∇²f = -|k|² f̂  (in Fourier)
    
    Args:
        f: Scalar field (N, N, N)
        L: Domain size
    
    Returns:
        Laplacian ∇²f (N, N, N)
    """
    N = f.shape[0]
    device = f.device
    dtype = f.dtype
    
    _, _, _, k_sq = build_wavenumber_grid(N, L, device, dtype)
    
    f_hat = torch.fft.fftn(f)
    lap_f_hat = -k_sq * f_hat
    lap_f = torch.fft.ifftn(lap_f_hat).real
    
    return lap_f


# ═══════════════════════════════════════════════════════════════════════════════════════
# QTT-HYBRID INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════════════

def qtt_to_dense_3d(
    cores: List[Tensor],
    n_bits: int,
) -> Tensor:
    """
    Convert QTT cores to dense 3D tensor.
    
    QTT format: 3*n_bits cores, each (r_left, 2, r_right)
    Morton interleaved: x0, y0, z0, x1, y1, z1, ...
    
    Dense format: (N, N, N) where N = 2^n_bits
    
    Args:
        cores: List of QTT cores
        n_bits: Bits per dimension
    
    Returns:
        Dense tensor (N, N, N)
    """
    n_cores = 3 * n_bits
    assert len(cores) == n_cores, f"Expected {n_cores} cores, got {len(cores)}"
    
    N = 2 ** n_bits
    device = cores[0].device
    dtype = cores[0].dtype
    
    # Contract cores to dense tensor
    # Result shape: (2, 2, 2, ..., 2) with 3*n_bits dimensions
    result = cores[0].squeeze(0)  # (2, r_1)
    
    for i in range(1, n_cores):
        # result: (..., r_i)
        # cores[i]: (r_i, 2, r_{i+1})
        result = torch.einsum('...r,rjR->...jR', result, cores[i])
    
    # result: (2, 2, ..., 2, 1) → squeeze last dim
    result = result.squeeze(-1)  # (2,) * 3*n_bits
    
    # Reshape from Morton order to (N, N, N)
    # Morton: i0, j0, k0, i1, j1, k1, ...
    # Need to transpose to: i0, i1, ..., j0, j1, ..., k0, k1, ...
    
    # Permutation: group all x-bits, then y-bits, then z-bits
    perm = []
    for d in range(3):  # x, y, z
        for b in range(n_bits):
            perm.append(3 * b + d)
    
    result = result.permute(*perm)
    result = result.reshape(N, N, N)
    
    return result


def solver_qtt_to_dense_3d(
    cores: List[Tensor],
    n_bits: int,
) -> Tensor:
    """
    Convert QTT cores (solver format) to dense 3D tensor.
    
    The solver uses row-major (C-order) TT decomposition:
    - Flat index = i * N² + j * N + k
    - Bits in MSB-first order (NOT Morton)
    
    This differs from qtt_to_dense_3d which uses Morton interleaving.
    
    Args:
        cores: List of QTT cores from TurboNS3DSolver
        n_bits: Bits per dimension
    
    Returns:
        Dense tensor (N, N, N)
    """
    n_cores = 3 * n_bits
    assert len(cores) == n_cores, f"Expected {n_cores} cores, got {len(cores)}"
    
    N = 2 ** n_bits
    
    # Contract cores to flat tensor
    # Result shape: (2, 2, 2, ..., 2) with 3*n_bits dimensions = (2^(3*n_bits),) = (N³,)
    result = cores[0].squeeze(0)  # (2, r_1)
    
    for i in range(1, n_cores):
        # result: (..., r_i)
        # cores[i]: (r_i, 2, r_{i+1})
        result = torch.einsum('...r,rjR->...jR', result, cores[i])
    
    # result: (2, 2, ..., 2, 1) → squeeze last dim
    result = result.squeeze(-1)  # (2,) * 3*n_bits
    
    # Flatten to (N³,) then reshape to (N, N, N)
    # The solver uses flat index = i * N² + j * N + k
    # So we just reshape directly
    result = result.reshape(N, N, N)
    
    return result


def dense_to_solver_qtt_3d(
    tensor: Tensor,
    n_bits: int,
    max_rank: int = 16,
    tol: float = 1e-8,
) -> List[Tensor]:
    """
    Convert dense 3D tensor to QTT format (solver compatible).
    
    Uses row-major (C-order) TT decomposition, matching TurboNS3DSolver._dense_to_qtt.
    
    Args:
        tensor: Dense tensor (N, N, N)
        n_bits: Bits per dimension
        max_rank: Maximum bond dimension
        tol: Truncation tolerance
    
    Returns:
        List of QTT cores in solver format
    """
    N = 2 ** n_bits
    n_cores = 3 * n_bits
    device = tensor.device
    dtype = tensor.dtype
    
    assert tensor.shape == (N, N, N), f"Expected ({N}, {N}, {N}), got {tensor.shape}"
    
    # Flatten to 1D (row-major order, matching solver)
    flat = tensor.reshape(-1)  # (N³,)
    
    # TT-SVD decomposition
    cores = []
    work = flat.reshape(2, -1)  # (2, N³/2)
    r_left = 1
    
    for i in range(n_cores - 1):
        m, n = work.shape
        
        # SVD
        U, S, Vh = torch.linalg.svd(work, full_matrices=False)
        
        # Truncate
        k = min(max_rank, len(S))
        if tol > 0 and len(S) > 0 and S[0] > 1e-14:
            rel_s = S / S[0]
            k = max(1, min(k, (rel_s > tol).sum().item()))
        
        U = U[:, :k]
        S = S[:k]
        Vh = Vh[:k, :]
        
        # Create core: reshape U from (m, k) to (r_left, 2, k)
        core = U.reshape(r_left, 2, k)
        cores.append(core)
        
        # Prepare next iteration
        work = torch.diag(S) @ Vh  # (k, n)
        
        # Reshape for next site
        if i < n_cores - 2:
            remaining = work.shape[1]
            if remaining >= 2:
                work = work.reshape(k * 2, remaining // 2)
        
        r_left = k
    
    # Last core
    cores.append(work.reshape(r_left, 2, 1))
    
    return cores


def dense_to_qtt_3d(
    tensor: Tensor,
    n_bits: int,
    max_rank: int = 16,
    tol: float = 1e-8,
) -> List[Tensor]:
    """
    Convert dense 3D tensor to QTT format with SVD-based truncation.
    
    Args:
        tensor: Dense tensor (N, N, N)
        n_bits: Bits per dimension
        max_rank: Maximum bond dimension
        tol: Truncation tolerance
    
    Returns:
        List of QTT cores in Morton order
    """
    N = 2 ** n_bits
    n_cores = 3 * n_bits
    device = tensor.device
    dtype = tensor.dtype
    
    assert tensor.shape == (N, N, N), f"Expected ({N}, {N}, {N}), got {tensor.shape}"
    
    # Reshape to Morton order
    # From (N, N, N) → (2^n, 2^n, 2^n) → (2, 2, ..., 2) interleaved
    
    # First reshape each dimension to binary
    t = tensor.reshape(*([2] * n_bits), *([2] * n_bits), *([2] * n_bits))
    
    # Permute to Morton interleaved order
    # From: i0, i1, ..., j0, j1, ..., k0, k1, ...
    # To: i0, j0, k0, i1, j1, k1, ...
    perm = []
    for b in range(n_bits):
        perm.extend([b, n_bits + b, 2 * n_bits + b])
    
    t = t.permute(*perm)
    
    # Flatten to (2^n_cores,) for TT decomposition
    t = t.reshape(-1)
    
    # TT-SVD decomposition
    cores = []
    current = t.reshape(1, -1)  # (1, 2^n_cores)
    
    for i in range(n_cores - 1):
        r_left = current.shape[0]
        remaining = 2 ** (n_cores - i - 1)
        
        # Reshape for SVD
        mat = current.reshape(r_left * 2, remaining)
        
        # SVD
        U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
        
        # Truncate based on tolerance and max_rank
        S_sum = S.sum()
        S_cumsum = torch.cumsum(S, dim=0)
        rank = min(
            max_rank,
            int((S_cumsum / S_sum < 1 - tol).sum().item()) + 1,
            len(S),
        )
        rank = max(rank, 1)
        
        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]
        
        # Core: (r_left, 2, rank)
        core = U.reshape(r_left, 2, rank)
        cores.append(core)
        
        # Prepare next iteration
        current = torch.diag(S) @ Vh
    
    # Last core: (r_left, 2, 1)
    last_core = current.reshape(-1, 2, 1)
    cores.append(last_core)
    
    return cores


class SpectralPoissonQTT:
    """
    QTT-Spectral Poisson solver.
    
    Provides exact spectral Poisson solve with QTT interface:
    1. Convert QTT vorticity to dense
    2. Solve Poisson/Biot-Savart in spectral space (exact)
    3. Convert result back to QTT
    
    Memory: O(N³) temporarily for FFT
    Accuracy: Machine precision (no iteration error)
    Speed: O(N³ log N) for FFT, plus O(N³ log N) for QTT conversion
    """
    
    def __init__(self, config: SpectralPoissonConfig):
        self.config = config
        self.N = 2 ** config.n_bits
        self.n_bits = config.n_bits
        
        # Pre-build wavenumber grid
        self.kx, self.ky, self.kz, self.k_sq = build_wavenumber_grid(
            self.N, config.L, config.device, config.dtype
        )
        
        # Regularized k²
        self.k_sq_reg = self.k_sq.clone()
        self.k_sq_reg[0, 0, 0] = 1.0
    
    def solve_poisson(
        self,
        f_qtt: List[Tensor],
        max_rank: int = 16,
    ) -> List[Tensor]:
        """
        Solve ∇²ψ = f with QTT input/output.
        
        Args:
            f_qtt: RHS in QTT format
            max_rank: Max rank for output QTT
        
        Returns:
            Solution ψ in QTT format
        """
        # QTT → Dense
        f_dense = qtt_to_dense_3d(f_qtt, self.n_bits)
        
        # Spectral solve
        result = spectral_poisson_solve(
            f_dense,
            L=self.config.L,
            regularization=self.config.regularization,
        )
        
        # Dense → QTT
        psi_qtt = dense_to_qtt_3d(
            result.solution,
            self.n_bits,
            max_rank=max_rank,
        )
        
        return psi_qtt
    
    def biot_savart(
        self,
        omega_qtt: List[List[Tensor]],
        max_rank: int = 16,
    ) -> List[List[Tensor]]:
        """
        Recover velocity from vorticity via Biot-Savart.
        
        Args:
            omega_qtt: Vorticity [ωx, ωy, ωz], each in QTT format
            max_rank: Max rank for output QTT
        
        Returns:
            Velocity [ux, uy, uz], each in QTT format
        """
        # QTT → Dense for each component
        omega_dense = [
            qtt_to_dense_3d(omega_qtt[i], self.n_bits)
            for i in range(3)
        ]
        
        # Spectral Biot-Savart
        u_dense = spectral_biot_savart(omega_dense, L=self.config.L)
        
        # Dense → QTT for each component
        u_qtt = [
            dense_to_qtt_3d(u_dense[i], self.n_bits, max_rank=max_rank)
            for i in range(3)
        ]
        
        return u_qtt


# ═══════════════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════════════

def validate_spectral_poisson(
    n_bits: int = 5,
    device: str = "cuda",
) -> Tuple[float, bool]:
    """
    Validate spectral Poisson solver against analytical solution.
    
    Test problem:
        f(x,y,z) = -3 * sin(x) * sin(y) * sin(z)
        ψ(x,y,z) = sin(x) * sin(y) * sin(z)
        ∇²ψ = -3 * sin(x) * sin(y) * sin(z) = f  ✓
    
    Args:
        n_bits: Bits per dimension
        device: Target device
    
    Returns:
        (error, passed): Relative L2 error and whether gate passed
    """
    N = 2 ** n_bits
    L = 2 * math.pi
    
    # Grid
    x = torch.linspace(0, L, N + 1, device=device)[:-1]  # Periodic: exclude endpoint
    X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')
    
    # Analytical solution
    psi_exact = torch.sin(X) * torch.sin(Y) * torch.sin(Z)
    
    # RHS: ∇²ψ = -3 sin(x) sin(y) sin(z)
    f = -3.0 * torch.sin(X) * torch.sin(Y) * torch.sin(Z)
    
    # Solve
    result = spectral_poisson_solve(f, L=L)
    
    # Compare (up to constant shift - both have zero mean due to spectral solve)
    psi_numerical = result.solution
    
    # Remove mean from both for fair comparison
    psi_exact_zm = psi_exact - psi_exact.mean()
    psi_numerical_zm = psi_numerical - psi_numerical.mean()
    
    error = torch.norm(psi_numerical_zm - psi_exact_zm) / torch.norm(psi_exact_zm)
    error = error.item()
    
    # Gate: error < 1e-6
    passed = error < 1e-6
    
    return error, passed


if __name__ == "__main__":
    print("=" * 70)
    print("SPECTRAL POISSON SOLVER VALIDATION")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    grids = [5, 6, 7]  # 32³, 64³, 128³
    
    all_passed = True
    for n_bits in grids:
        N = 2 ** n_bits
        error, passed = validate_spectral_poisson(n_bits, device)
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {N}³: error = {error:.2e}  {status}")
        all_passed = all_passed and passed
    
    print()
    if all_passed:
        print("✓✓✓ ALL GRIDS PASSED — SPECTRAL POISSON VALIDATED ✓✓✓")
    else:
        print("✗✗✗ SOME GRIDS FAILED ✗✗✗")
    
    exit(0 if all_passed else 1)
