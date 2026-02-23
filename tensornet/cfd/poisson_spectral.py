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


def _eigh_svd_full(
    mat: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    SVD via eigendecomposition of the Gram matrix.

    GPU-stable alternative to ``torch.linalg.svd`` which can trigger
    cuSOLVER convergence failures on near-singular matrices.

    Only the small Gram matrix (m×m or n×n) is promoted to float64
    for stability — the large matrix ``mat`` stays in its original
    dtype throughout, avoiding expensive whole-array float64 copies.

    Returns
    -------
    U : (m, k)
    S : (k,)    singular values in descending order
    Vh : (k, n)
    where k = min(m, n).
    """
    m, n = mat.shape
    orig_dtype = mat.dtype

    if m <= n:
        # Gram matrix in caller dtype, then promote for eigh
        G = (mat @ mat.T).to(torch.float64)               # (m, m)
        evals, evecs = torch.linalg.eigh(G)
        del G
        evals = evals.flip(0).clamp(min=0)
        evecs = evecs.flip(1)
        S = torch.sqrt(evals).to(orig_dtype)
        U = evecs.to(orig_dtype)                           # (m, m)
        mask = S > 1e-7
        inv_S = torch.zeros_like(S)
        inv_S[mask] = 1.0 / S[mask]
        # Vh reconstruction in caller dtype (the big multiply)
        Vh = inv_S.unsqueeze(1) * (U.T @ mat)             # (m, n)
    else:
        # Gram matrix in caller dtype, then promote for eigh
        G = (mat.T @ mat).to(torch.float64)                # (n, n)
        evals, evecs = torch.linalg.eigh(G)
        del G
        evals = evals.flip(0).clamp(min=0)
        evecs = evecs.flip(1)
        S = torch.sqrt(evals).to(orig_dtype)
        Vh = evecs.T.to(orig_dtype)                        # (n, n)
        mask = S > 1e-7
        inv_S = torch.zeros_like(S)
        inv_S[mask] = 1.0 / S[mask]
        U = mat @ evecs.to(orig_dtype) * inv_S.unsqueeze(0)  # (m, n)

    return U, S, Vh


def qtt_to_dense_3d(
    cores: List[Tensor],
    n_bits: int,
) -> Tensor:
    """
    Convert QTT cores to dense 3D tensor.

    QTT format: 3*n_bits cores, each (r_left, 2, r_right)
    Morton interleaved in MSB-first order per triplet:
        core 0,1,2 → (x_{n-1}, y_{n-1}, z_{n-1})  — most significant bits
        core 3,4,5 → (x_{n-2}, y_{n-2}, z_{n-2})
        ...
        core 3(n-1), 3(n-1)+1, 3(n-1)+2 → (x_0, y_0, z_0) — least significant

    Dense format: (N, N, N) where N = 2^n_bits.

    Uses incremental 3-axis build so the working tensor never exceeds
    5 dimensions (Nx, Ny, Nz, new_bit, bond) — safe for n_bits up to
    any value without hitting PyTorch's 25-dim limit.

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

    # Working tensor: (Nx, Ny, Nz, bond_dim)
    # Start at (1, 1, 1, 1) — a single scalar with trivial bond.
    result = torch.ones(1, 1, 1, 1, device=device, dtype=dtype)

    for b in range(n_bits):
        cx = cores[3 * b]          # x bit: (r_in, 2, r_mid1)
        cy = cores[3 * b + 1]     # y bit: (r_mid1, 2, r_mid2)
        cz = cores[3 * b + 2]     # z bit: (r_mid2, 2, r_out)

        Nx, Ny, Nz = result.shape[0], result.shape[1], result.shape[2]

        # --- x bit ----------------------------------------------------------
        # (Nx, Ny, Nz, r) × (r, 2, r') → (Nx, Ny, Nz, 2, r')
        result = torch.einsum('xyzr,rdR->xyzdR', result, cx)
        # Solver uses LSB-first ordering (core 0 = x₀ = LSB).
        # New bit is higher-weight → place it BEFORE old bits (MSB end).
        # permute: (2_new, Nx_old, Ny, Nz, r') → reshape (2*Nx, Ny, Nz, r')
        result = result.permute(3, 0, 1, 2, 4).reshape(Nx * 2, Ny, Nz, -1)

        Nx *= 2

        # --- y bit ----------------------------------------------------------
        result = torch.einsum('xyzr,rdR->xyzdR', result, cy)
        # (X, 2_new, Ny_old, Nz, r') → reshape (X, 2*Ny, Nz, r')
        result = result.permute(0, 3, 1, 2, 4).reshape(Nx, Ny * 2, Nz, -1)

        Ny *= 2

        # --- z bit ----------------------------------------------------------
        result = torch.einsum('xyzr,rdR->xyzdR', result, cz)
        # (X, Y, 2_new, Nz_old, r') → reshape (X, Y, 2*Nz, r')
        result = result.permute(0, 1, 3, 2, 4).reshape(Nx, Ny, Nz * 2, -1)

    # (N, N, N, 1) → (N, N, N)
    return result.squeeze(-1)


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

    Uses iterative matrix contraction to stay within PyTorch's 25-dim
    limit for large n_bits.

    Args:
        cores: List of QTT cores from TurboNS3DSolver
        n_bits: Bits per dimension

    Returns:
        Dense tensor (N, N, N)
    """
    n_cores = 3 * n_bits
    assert len(cores) == n_cores, f"Expected {n_cores} cores, got {len(cores)}"

    N = 2 ** n_bits

    # Contract cores iteratively using 2-D working matrix.
    # result shape: (flat_extent, bond_dim)
    result = cores[0].squeeze(0)  # (2, r_1)

    for i in range(1, n_cores):
        flat, r_in = result.shape
        r_in2, d, r_out = cores[i].shape
        # (flat, r_in) × (r_in, 2, r_out) → (flat, 2, r_out) → (flat*2, r_out)
        result = torch.einsum('fr,rdR->fdR', result, cores[i])
        result = result.reshape(flat * 2, r_out)

    # (N³, 1) → (N, N, N)
    result = result.squeeze(-1).reshape(N, N, N)

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

        # SVD via eigh (GPU-stable, no cuSOLVER gesvd)
        U, S, Vh = _eigh_svd_full(work)

        # Truncate
        k = min(max_rank, len(S))
        if tol > 0 and len(S) > 0 and S[0] > 1e-14:
            rel_s = S / S[0]
            k = max(1, min(k, int((rel_s > tol).sum().item())))

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


def _morton_reindex_3d(tensor: Tensor, n_bits: int) -> Tensor:
    """
    Reindex a (N, N, N) tensor from row-major order to a flat vector in
    Morton (Z-curve) bit-interleaved order.

    Morton flat-index bit layout (MSB-first per triplet)::

        bit 3b+2 → x_b,   bit 3b+1 → y_b,   bit 3b → z_b

    where b = 0 (LSB) … n_bits − 1 (MSB).

    Uses int32 arithmetic so the temporary index buffer is only N³ × 4 B
    (536 MB at 512³).

    Args:
        tensor: Dense tensor (N, N, N) in C-order.
        n_bits: Bits per dimension (N = 2^n_bits, n_bits ≤ 10).

    Returns:
        Flat vector of length N³ in Morton order.
    """
    N = 2 ** n_bits
    device = tensor.device
    flat = tensor.reshape(-1)  # C-order: flat[i*N²+j*N+k]

    # For each Morton position m, extract the (i, j, k) it maps to and
    # compute the C-order index.
    #
    # Solver uses LSB-first Morton: core 0 = x₀ (LSB).
    # In TT-SVD, core 0 = MSB of flat index.  So x₀ (LSB of x) sits
    # at the MSB position of the Morton flat index:
    #
    #   morton bit (3(n-1-b)+2) = x_b,   bit (3(n-1-b)+1) = y_b,
    #   bit (3(n-1-b))         = z_b
    #
    m = torch.arange(N ** 3, device=device, dtype=torch.int32)

    i_val = torch.zeros_like(m)
    j_val = torch.zeros_like(m)
    k_val = torch.zeros_like(m)
    for b in range(n_bits):
        bp = 3 * (n_bits - 1 - b)          # base bit position for level b
        i_val |= ((m >> (bp + 2)) & 1) << b
        j_val |= ((m >> (bp + 1)) & 1) << b
        k_val |= ((m >> bp)       & 1) << b

    c_idx = (i_val * (N * N) + j_val * N + k_val).long()
    del m, i_val, j_val, k_val

    return flat[c_idx]


def dense_to_qtt_3d(
    tensor: Tensor,
    n_bits: int,
    max_rank: int = 16,
    tol: float = 1e-8,
) -> List[Tensor]:
    """
    Convert dense 3D tensor to QTT format with SVD-based truncation.

    Uses int32 Morton reindexing instead of a high-dimensional permute,
    so it works for any n_bits without hitting PyTorch's 25-dim limit.

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

    # Reindex to Morton flat order (avoids >25-dim permute)
    t = _morton_reindex_3d(tensor, n_bits)
    
    # TT-SVD decomposition
    cores = []
    current = t.reshape(1, -1)  # (1, 2^n_cores)
    
    for i in range(n_cores - 1):
        r_left = current.shape[0]
        remaining = 2 ** (n_cores - i - 1)
        
        # Reshape for SVD
        mat = current.reshape(r_left * 2, remaining)

        # SVD via eigh (GPU-stable, no cuSOLVER gesvd)
        U, S, Vh = _eigh_svd_full(mat)

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


# ═══════════════════════════════════════════════════════════════════════════════════════
# SPECTRAL POISSON PRECONDITIONER
# ═══════════════════════════════════════════════════════════════════════════════════════


def spectral_poisson_precond(
    rhs_cores: List[Tensor],
    n_bits: int,
    L: float = 2 * math.pi,
    precond_rank: int = 12,
    tol: float = 1e-6,
) -> List[Tensor]:
    """
    Spectral preconditioner for the pressure Poisson equation.

    Approximates z = (∇²)⁻¹ r by applying the exact spectral inverse
    Laplacian in dense space, then compressing back to QTT at low rank.
    The low-rank truncation makes this an *approximate* inverse —
    sufficient for preconditioning CG but not for a standalone solve.

    Steps
    -----
    1. QTT → dense 3D (incremental build, ~0.2 s at 512³)
    2. rFFT → divide by discrete Laplacian eigenvalues → irFFT (~0.1 s)
    3. Dense → QTT at precond_rank (Morton reindex + TT-SVD, ~1 s at rank 12)

    Total cost: ~1.3 s per application at 512³, precond_rank = 12.
    VRAM spike: ~1.3 GB (dense 512³ + rFFT workspace).

    The discrete Laplacian eigenvalues match the finite-difference stencil
    used in the CG solver::

        λ_k = (2 / dx²)(cos(2πk/N) − 1)   per axis
        Λ_{k1,k2,k3} = λ_{k1} + λ_{k2} + λ_{k3}

    Parameters
    ----------
    rhs_cores : List[Tensor]
        QTT cores of the CG residual (3 × n_bits cores, Morton order).
    n_bits : int
        Bits per spatial dimension (N = 2^n_bits).
    L : float
        Physical domain size (default: 2π periodic box).
    precond_rank : int
        Maximum QTT rank for the output.  Lower = faster but rougher
        approximation.  12 is a good balance at 512³.
    tol : float
        SVD truncation tolerance.

    Returns
    -------
    List[Tensor]
        QTT cores of the approximate solution z ≈ (∇²)⁻¹ r.
    """
    N = 2 ** n_bits
    device = rhs_cores[0].device
    dtype = rhs_cores[0].dtype
    dx = L / N

    # ── QTT → dense ────────────────────────────────────────────────────
    rhs_dense = qtt_to_dense_3d(rhs_cores, n_bits)

    # ── Build eigenvalues of discrete 3D Laplacian ─────────────────────
    # 1D:  λ_k = (2/dx²)(cos(2πν_k) − 1),   ν_k = fftfreq(N)
    freq = torch.fft.fftfreq(N, device=device, dtype=dtype)
    freq_r = torch.fft.rfftfreq(N, device=device, dtype=dtype)

    lam_full = (2.0 / dx ** 2) * (torch.cos(2.0 * math.pi * freq) - 1.0)
    lam_half = (2.0 / dx ** 2) * (torch.cos(2.0 * math.pi * freq_r) - 1.0)

    # 3D eigenvalues (additive separability of the Laplacian)
    lam3d = (
        lam_full[:, None, None]
        + lam_full[None, :, None]
        + lam_half[None, None, :]
    )
    lam3d[0, 0, 0] = 1.0  # avoid division by zero at the zero mode

    # ── FFT solve ──────────────────────────────────────────────────────
    rhs_hat = torch.fft.rfftn(rhs_dense, dim=(0, 1, 2))
    del rhs_dense

    z_hat = rhs_hat / lam3d
    z_hat[0, 0, 0] = 0.0  # enforce mean-zero pressure
    del rhs_hat, lam3d

    z_dense = torch.fft.irfftn(z_hat, s=(N, N, N), dim=(0, 1, 2))
    del z_hat

    # ── Dense → QTT at low rank ────────────────────────────────────────
    z_cores = dense_to_qtt_3d(z_dense, n_bits, max_rank=precond_rank, tol=tol)
    del z_dense

    torch.cuda.empty_cache()
    return z_cores


# ═══════════════════════════════════════════════════════════════════════════════════════
# SPECTRAL LERAY PROJECTION
# ═══════════════════════════════════════════════════════════════════════════════════════


def spectral_leray_project_qtt(
    ux_cores: List[Tensor],
    uy_cores: List[Tensor],
    uz_cores: List[Tensor],
    n_bits: int,
    L: float = 2 * math.pi,
    max_rank: int = 48,
    tol: float = 1e-8,
) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
    """
    Leray-Helmholtz projection via spectral FFT — correction-only path.

    Instead of converting the full projected velocity through
    dense→QTT (which destroys the existing QTT structure), this
    function computes only the *correction* Δu = −∇p in dense, converts
    that low-rank correction to QTT via TT-SVD, and returns it.

    The caller adds the correction to u* in QTT format::

        u_proj = u* + Δu     (QTT addition, preserves structure)

    Because the correction is the gradient of a smooth scalar pressure
    field, it has very low QTT rank (typically ≤ 16) and compresses
    with negligible loss.

    Algorithm
    ---------
    1. u* QTT → dense → rFFT
    2. Compute divergence analytically: k · û
    3. Pressure: p̂ = −(k · û) / |k|²
    4. **Correction**: Δû_i = −ik_i · p̂  (= k_i · (k · û) / |k|²)
       Wait — the Leray correction is Δu = u_proj − u* = −∇p.
       In Fourier: Δû_i = −k_i · (k · û) / |k|²
       (no imaginary factor needed since FFT of ∂/∂x already includes
       the conventional i-factor via the irfftn/rfftn pair).
    5. irFFT → dense correction
    6. dense → QTT (low rank, fast TT-SVD)

    Returns the *correction* cores, not the projected velocity.

    Parameters
    ----------
    ux_cores, uy_cores, uz_cores : List[Tensor]
        QTT cores (Morton order) for u*.
    n_bits, L, max_rank, tol :
        Grid / rank / tolerance parameters.

    Returns
    -------
    (dx_qtt, dy_qtt, dz_qtt) : tuple of List[Tensor]
        Correction QTT cores such that u_proj = u* + Δu.
    """
    N = 2 ** n_bits
    device = ux_cores[0].device
    dtype = ux_cores[0].dtype

    # ── Phase 1: QTT → dense → rFFT ────────────────────────────────────
    ux_dense = qtt_to_dense_3d(ux_cores, n_bits)
    ux_hat = torch.fft.rfftn(ux_dense)
    del ux_dense

    uy_dense = qtt_to_dense_3d(uy_cores, n_bits)
    uy_hat = torch.fft.rfftn(uy_dense)
    del uy_dense

    uz_dense = qtt_to_dense_3d(uz_cores, n_bits)
    uz_hat = torch.fft.rfftn(uz_dense)
    del uz_dense

    # ── Phase 2: Correction scalar in Fourier space ────────────────────
    k_full = torch.fft.fftfreq(N, d=1.0 / N, device=device, dtype=dtype)
    k_full = k_full * (2.0 * math.pi / L)
    k_half = torch.fft.rfftfreq(N, d=1.0 / N, device=device, dtype=dtype)
    k_half = k_half * (2.0 * math.pi / L)

    kx, ky, kz = torch.meshgrid(k_full, k_full, k_half, indexing="ij")
    del k_full, k_half

    k_sq = kx ** 2 + ky ** 2 + kz ** 2
    k_sq_safe = k_sq.clone()
    k_sq_safe[0, 0, 0] = 1.0

    # Divergence: k · û
    k_dot_u = kx * ux_hat + ky * uy_hat + kz * uz_hat
    del ux_hat, uy_hat, uz_hat

    # Correction factor: (k · û) / |k|²
    correction = k_dot_u / k_sq_safe
    correction[0, 0, 0] = 0.0          # zero-mean pressure
    del k_dot_u, k_sq, k_sq_safe

    # ── Phase 3: Per-component correction → dense → QTT ───────────────
    torch.cuda.empty_cache()

    results: List[List[Tensor]] = []
    for ki in [kx, ky, kz]:
        # Δû_i = −k_i · correction
        delta_hat = -(ki * correction)
        delta_dense = torch.fft.irfftn(delta_hat, s=(N, N, N))
        del delta_hat
        delta_qtt = dense_to_qtt_3d(
            delta_dense, n_bits, max_rank=max_rank, tol=tol,
        )
        del delta_dense
        results.append(delta_qtt)

    del kx, ky, kz, correction
    torch.cuda.empty_cache()

    return results[0], results[1], results[2]


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
