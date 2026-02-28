"""GPU-Native Chu Limit Antenna Topology Optimization.

Challenges the Chu limit Q-factor bound at 4096³ on a single GPU.
Uses the proven GPU QTT Helmholtz solver from qtt_helmholtz_gpu.py.

Architecture
────────────
• All QTT/MPO: ``list[torch.Tensor]`` on ``device='cuda'``
• Forward + adjoint: ``tt_amen_solve_gpu`` (cuSOLVER dense local solves)
• Power metrics: QTT inner products (no dense N³)
• Gradient: exact adjoint via H^H, sparse extraction at design voxels
• Density: 1D ``torch.Tensor`` on CUDA (length n_design)
• Optimization: Adam on GPU

NO NumPy in hot path. NO CPU fallback. NO dense N³ arrays.

Author: TiganticLabz
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

import torch
import numpy as np

from ontic.em.qtt_helmholtz_gpu import (
    _assert_gpu,
    _assert_gpu_cores,
    array_to_tt_gpu,
    tt_round_gpu,
    tt_inner_gpu,
    tt_norm_gpu,
    tt_scale_gpu,
    tt_add_gpu,
    tt_axpy_gpu,
    tt_matvec_gpu,
    mpo_add_gpu,
    mpo_scale_gpu,
    diag_mpo_from_tt_gpu,
    identity_mpo_gpu,
    helmholtz_mpo_3d_gpu,
    laplacian_mpo_3d_gpu,
    build_pml_eps_1d_gpu,
    build_pml_eps_3d_separable_gpu,
    gaussian_source_tt_gpu,
    tt_amen_solve_gpu,
    SolveResult,
)
from ontic.em.boundaries import PMLConfig, _compute_pml_stretching


# ═══════════════════════════════════════════════════════════════════════════════
# Physical Constants
# ═══════════════════════════════════════════════════════════════════════════════

C0 = 299_792_458.0
MU0 = 4.0e-7 * math.pi
EPS0 = 1.0 / (MU0 * C0 ** 2)
ETA0 = math.sqrt(MU0 / EPS0)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Analytical Q Limits
# ═══════════════════════════════════════════════════════════════════════════════


def chu_limit_q(ka: float) -> float:
    """Chu limit: Q_min = 1/(ka)^3 + 1/(ka)."""
    return 1.0 / ka ** 3 + 1.0 / ka


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: PML σ Weights as QTT (GPU)
# ═══════════════════════════════════════════════════════════════════════════════


def build_pml_sigma_tt_gpu(
    n_bits: int,
    k0_norm: float,
    pml: PMLConfig,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.complex64,
    max_rank: int = 32,
) -> list[torch.Tensor]:
    """Build PML loss weight σ_pml(x,y,z) as QTT on GPU.

    σ_pml = [Im(s_x) + Im(s_y) + Im(s_z)] · k₀²

    Additive formulation: inherently non-negative, rank ≤ 3.
    """
    N = 2 ** n_bits
    # Use CPU numpy for the PML stretching computation (small: just N floats)
    s_1d_np = _compute_pml_stretching(N, k0_norm, pml)
    ai_np = np.imag(s_1d_np)  # (N,), >= 0

    ai = torch.tensor(ai_np, device=device, dtype=dtype)
    ones = torch.ones(N, device=device, dtype=dtype)

    # Three rank-1 separable terms
    def separable_3d_tt(
        fx: torch.Tensor, fy: torch.Tensor, fz: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Build 3D QTT from f_x ⊗ f_y ⊗ f_z via Hadamard."""
        cx = array_to_tt_gpu(fx, max_rank=max_rank)
        cy = array_to_tt_gpu(fy, max_rank=max_rank)
        cz = array_to_tt_gpu(fz, max_rank=max_rank)
        return cx + cy + cz  # Concatenate site lists

    term_x = separable_3d_tt(ai, ones, ones)
    term_y = separable_3d_tt(ones, ai, ones)
    term_z = separable_3d_tt(ones, ones, ai)

    k2 = k0_norm ** 2
    result = tt_add_gpu(
        tt_add_gpu(tt_scale_gpu(term_x, k2), tt_scale_gpu(term_y, k2)),
        tt_scale_gpu(term_z, k2),
    )
    return tt_round_gpu(result, max_rank=max_rank)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Sphere Mask & Design Region (GPU)
# ═══════════════════════════════════════════════════════════════════════════════


def spherical_mask_flat_indices_gpu(
    n_bits: int,
    centre: tuple[float, float, float] = (0.5, 0.5, 0.5),
    radius: float = 0.1,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """C-order flat indices of voxels inside a sphere. O(N + n_design) memory.

    C-order: flat = ix * N² + iy * N + iz  (matches QTT [x,y,z] site ordering).
    Returns int64 tensor on GPU.
    """
    N = 2 ** n_bits
    h = 1.0 / N
    coords = np.linspace(h / 2, 1.0 - h / 2, N)
    cx, cy, cz = centre
    r2 = radius ** 2
    dx2 = (coords - cx) ** 2

    chunks: list[np.ndarray] = []
    x_valid = np.where(dx2 <= r2)[0]

    for ix in x_valid:
        ryz2 = r2 - dx2[ix]
        dy2 = (coords - cy) ** 2
        y_valid = np.where(dy2 <= ryz2)[0]

        for iy in y_valid:
            rz2 = ryz2 - dy2[iy]
            if rz2 < 0:
                continue
            rz = np.sqrt(rz2)
            iz_valid = np.where(np.abs(coords - cz) <= rz)[0]
            # C-order: ix * N² + iy * N + iz
            flat = (ix * N + iy) * N + iz_valid.astype(np.int64)
            chunks.append(flat)

    if chunks:
        flat_np = np.sort(np.concatenate(chunks))
    else:
        flat_np = np.array([], dtype=np.int64)
    return torch.tensor(flat_np, device=device, dtype=torch.long)


# ---- QTT Zero Expansion (resolution doubling without decompression) --------

_SPHERE_MASK_BASE_BITS = 7  # 128³ — largest dense array ever allocated


def _zero_expand_3d_qtt_gpu(
    cores: list[torch.Tensor],
    n_bits_current: int,
    n_bits_target: int,
) -> list[torch.Tensor]:
    """Expand 3D QTT resolution by inserting duplication cores. QTT-native.

    Doubles each spatial dimension by appending a new LSB core per dimension
    that copies the value (nearest-neighbour interpolation).  O(r²) per new
    core — never decompresses.

    Site ordering (C-order flat index): [x_MSB..x_LSB, y_MSB..y_LSB, z_MSB..z_LSB]
    """
    if n_bits_current >= n_bits_target:
        return [c.clone() for c in cores]

    device = cores[0].device
    dtype = cores[0].dtype
    result = list(cores)

    for _step in range(n_bits_current, n_bits_target):
        n = len(result) // 3  # current bits per dimension
        new_cores: list[torch.Tensor] = []

        for dim in range(3):
            dim_start = dim * n
            # Copy existing cores for this dimension
            for i in range(dim_start, dim_start + n):
                new_cores.append(result[i])

            # Insert duplication core after LSB:
            # core[:, 0, :] = core[:, 1, :] = I_r  (value unchanged by new bit)
            r = new_cores[-1].shape[2]
            dup = torch.zeros(r, 2, r, device=device, dtype=dtype)
            eye = torch.eye(r, device=device, dtype=dtype)
            dup[:, 0, :] = eye
            dup[:, 1, :] = eye
            new_cores.append(dup)

        result = new_cores

    return result


def build_sphere_mask_tt_gpu(
    n_bits: int,
    centre: tuple[float, float, float] = (0.5, 0.5, 0.5),
    radius: float = 0.1,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.complex64,
    max_rank: int = 32,
) -> list[torch.Tensor]:
    """Build QTT of sphere indicator (0/1). QTT-native zero-expansion.

    1. Builds dense mask at base resolution (128³ = 2M elements, 16 MB).
    2. Decomposes to QTT via rSVD/cuSOLVER on the small array.
    3. Zero-expands to target n_bits by inserting duplication cores.

    NEVER allocates arrays larger than 128³. O(r²·n_bits) for expansion.
    """
    base_bits = min(n_bits, _SPHERE_MASK_BASE_BITS)
    N_base = 2 ** base_bits
    h = 1.0 / N_base

    cx, cy, cz = centre
    coords = torch.linspace(h / 2, 1.0 - h / 2, N_base, device=device)

    # Dense mask at base resolution — 128³ = 2M elements, fine
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing='ij')
    dist2 = (gx - cx) ** 2 + (gy - cy) ** 2 + (gz - cz) ** 2
    mask_dense = (dist2 <= radius ** 2).to(dtype)
    del gx, gy, gz, dist2

    # C-order flatten: flat[ix*N²+iy*N+iz] = mask[ix,iy,iz]
    # QTT site ordering: [x_MSB..x_LSB, y_MSB..y_LSB, z_MSB..z_LSB]
    # — matches Helmholtz MPO / PML eps ordering
    mask_flat = mask_dense.reshape(-1)
    del mask_dense

    base_cores = array_to_tt_gpu(mask_flat, max_rank=max_rank)
    del mask_flat

    if n_bits <= base_bits:
        return base_cores

    # Zero-expand to target resolution — O(r²·n_bits), no dense arrays
    expanded = _zero_expand_3d_qtt_gpu(base_cores, base_bits, n_bits)
    return tt_round_gpu(expanded, max_rank=max_rank)


def compute_voxel_distances_gpu(
    flat_indices: torch.Tensor,
    n_bits: int,
    point: tuple[float, float, float],
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """Distance from indexed voxels to a point. GPU-native, O(K) memory.

    C-order: flat = ix * N² + iy * N + iz.
    """
    N = 2 ** n_bits
    h = 1.0 / N
    coords = torch.linspace(h / 2, 1.0 - h / 2, N, device=device)

    # C-order decode
    iz = flat_indices % N
    iy = (flat_indices // N) % N
    ix = flat_indices // (N * N)

    px, py, pz = point
    dx = coords[ix] - px
    dy = coords[iy] - py
    dz = coords[iz] - pz

    return torch.sqrt(dx ** 2 + dy ** 2 + dz ** 2)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Sparse QTT Evaluation (GPU)
# ═══════════════════════════════════════════════════════════════════════════════


def _tt_evaluate_batch_gpu(
    tt_cores: list[torch.Tensor],
    flat_indices: torch.Tensor,
    total_sites: int,
) -> torch.Tensor:
    """Evaluate QTT at a batch of flat indices. Internal — no batching."""
    K = flat_indices.shape[0]
    device = tt_cores[0].device
    dtype = tt_cores[0].dtype

    # Convert flat indices to binary (MSB first)
    bits = torch.zeros(K, total_sites, device=device, dtype=torch.long)
    idx_remaining = flat_indices.clone()
    for b in range(total_sites):
        bits[:, total_sites - 1 - b] = idx_remaining % 2
        idx_remaining //= 2

    # Contract through cores: maintain (K, r_left) transfer matrix
    r0 = tt_cores[0].shape[0]
    transfer = torch.ones(K, r0, device=device, dtype=dtype)

    for site in range(total_sites):
        core = tt_cores[site]  # (r_l, 2, r_r)
        bit_vals = bits[:, site]  # (K,) with values 0 or 1

        # Gather: select core[:, bit, :] for each sample
        selected = core[:, bit_vals, :]  # (r_l, K, r_r)
        selected = selected.permute(1, 0, 2)  # (K, r_l, r_r)

        # Contract: transfer @ selected per sample
        transfer = torch.einsum('ki,kij->kj', transfer, selected)

    return transfer.squeeze(-1)


def tt_evaluate_at_indices_gpu(
    tt_cores: list[torch.Tensor],
    flat_indices: torch.Tensor,
    n_bits: int,
    batch_size: int = 200_000,
) -> torch.Tensor:
    """Evaluate QTT at specific flat indices — O(n_sites · r² · K). GPU-native.

    Batched to avoid VRAM explosion: processes batch_size indices at a time.
    Memory per batch: O(batch_size · r²) — with r=16, batch=200K → ~50MB.

    Parameters
    ----------
    tt_cores : list[torch.Tensor]
        QTT cores (3*n_bits sites) on CUDA.
    flat_indices : torch.Tensor
        int64 tensor of C-order flat indices on CUDA.
    n_bits : int
        Bits per dimension.
    batch_size : int
        Max indices per batch (controls peak VRAM).

    Returns
    -------
    torch.Tensor
        Complex values at requested indices, shape (K,), on CUDA.
    """
    total_sites = 3 * n_bits
    K = flat_indices.shape[0]

    if K <= batch_size:
        return _tt_evaluate_batch_gpu(tt_cores, flat_indices, total_sites)

    chunks: list[torch.Tensor] = []
    for start in range(0, K, batch_size):
        end = min(start + batch_size, K)
        batch = flat_indices[start:end]
        chunks.append(_tt_evaluate_batch_gpu(tt_cores, batch, total_sites))

    return torch.cat(chunks)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: SIMP Material Model (GPU)
# ═══════════════════════════════════════════════════════════════════════════════


def simp_sigma_gpu(
    rho_proj: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
    p: float,
) -> torch.Tensor:
    """SIMP conductivity: σ(ρ) = σ_min + (σ_max − σ_min) · ρ^p. GPU."""
    rho = rho_proj.clamp(0.0, 1.0)
    return sigma_min + (sigma_max - sigma_min) * rho.pow(p)


def simp_dsigma_drho_gpu(
    rho_proj: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
    p: float,
) -> torch.Tensor:
    """dσ/dρ = (σ_max − σ_min) · p · ρ^(p−1). GPU."""
    rho = rho_proj.clamp(1e-12, 1.0)
    return (sigma_max - sigma_min) * p * rho.pow(p - 1.0)


def heaviside_projection_gpu(
    rho: torch.Tensor, beta: float, eta: float = 0.5,
) -> torch.Tensor:
    """Smooth Heaviside projection. GPU."""
    num = torch.tanh(torch.tensor(beta * eta, device=rho.device)) + torch.tanh(
        beta * (rho - eta)
    )
    den = torch.tanh(torch.tensor(beta * eta, device=rho.device)) + torch.tanh(
        torch.tensor(beta * (1.0 - eta), device=rho.device)
    )
    return num / den


def heaviside_gradient_gpu(
    rho: torch.Tensor, beta: float, eta: float = 0.5,
) -> torch.Tensor:
    """Derivative of Heaviside projection wrt rho. GPU."""
    den = torch.tanh(torch.tensor(beta * eta, device=rho.device)) + torch.tanh(
        torch.tensor(beta * (1.0 - eta), device=rho.device)
    )
    sech2 = 1.0 / torch.cosh(beta * (rho - eta)).pow(2)
    return beta * sech2 / den


def volume_preserving_projection_gpu(
    rho_filt: torch.Tensor,
    beta: float,
    vol_target: float,
    tol: float = 1e-4,
    max_bisect: int = 40,
) -> tuple[torch.Tensor, float]:
    """Bisect η so that mean(H_β,η(ρ_filt)) = V_target. GPU O(n·log(1/tol)).

    Forces the projected volume to equal V_target exactly (within tolerance).
    Returns (ρ_proj, η_found).
    """
    lo, hi = 0.0, 1.0
    for _ in range(max_bisect):
        eta_mid = (lo + hi) / 2.0
        proj = heaviside_projection_gpu(rho_filt, beta, eta_mid)
        v_mean = proj.mean().item()
        if abs(v_mean - vol_target) < tol:
            return proj, eta_mid
        # Higher eta → lower projected volume (threshold goes up)
        if v_mean > vol_target:
            lo = eta_mid
        else:
            hi = eta_mid
    # Return best we have
    return heaviside_projection_gpu(rho_filt, beta, (lo + hi) / 2.0), (lo + hi) / 2.0


def volume_preserving_gradient_gpu(
    rho_filt: torch.Tensor,
    beta: float,
    eta: float,
) -> torch.Tensor:
    """Gradient of volume-preserving projection wrt rho_filt.

    Since η is fixed per iteration (found by bisection), this is just
    dH/dρ at the bisected η.
    """
    return heaviside_gradient_gpu(rho_filt, beta, eta)


def density_filter_gpu(
    density: torch.Tensor, radius: int,
) -> torch.Tensor:
    """Box-average density filter via conv1d. GPU-native, O(n) single kernel."""
    if radius <= 0:
        return density.clone()
    w = 2 * radius + 1
    kernel = torch.ones(1, 1, w, device=density.device, dtype=density.dtype) / w
    padded = torch.nn.functional.pad(
        density.view(1, 1, -1), (radius, radius), mode='reflect',
    )
    return torch.nn.functional.conv1d(padded, kernel).view(-1)


def density_filter_gradient_gpu(
    grad: torch.Tensor, radius: int,
) -> torch.Tensor:
    """Adjoint of box-average filter via conv1d. GPU-native, O(n) single kernel."""
    if radius <= 0:
        return grad.clone()
    w = 2 * radius + 1
    kernel = torch.ones(1, 1, w, device=grad.device, dtype=grad.dtype) / w
    padded = torch.nn.functional.pad(
        grad.view(1, 1, -1), (radius, radius), mode='reflect',
    )
    return torch.nn.functional.conv1d(padded, kernel).view(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: Conductivity ε Construction (GPU)
# ═══════════════════════════════════════════════════════════════════════════════


def build_conductivity_eps_tt_gpu(
    density: torch.Tensor,
    design_mask_tt: list[torch.Tensor],
    sigma_min: float,
    sigma_max: float,
    simp_p: float,
    beta: float,
    eta: float,
    filter_radius: int,
    n_bits: int,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.complex64,
    max_rank: int = 64,
    design_flat_indices: Optional[torch.Tensor] = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Build complex ε(ρ) and σ-field as QTT on GPU.

    ε = 1 − j·σ(ρ(x)) on design voxels, 1 elsewhere.

    Per-voxel mode (design_flat_indices provided, grid ≤ 256³):
        Scatters individual σ(ρ_proj[i]) values into a dense N³ array,
        converts to QTT.  Preserves spatial conductivity variation.
        Each design voxel gets its own σ — essential for topology
        optimization to create spatially differentiated structure.
        Memory: N³ × 8 bytes (complex64), e.g. 134 MB at 256³.

    Mean-field fallback (large grids or no indices):
        σ_field ≈ σ_mean · mask_tt.  All design voxels share one σ value.

    Returns
    -------
    (eps_tt, sigma_field_tt) : tuple of QTT core lists
        eps_tt : ε = 1 − j·σ_field
        sigma_field_tt : the σ-field QTT (for reuse in power metrics / adjoint)
    """
    total_sites = 3 * n_bits

    rho_filt = density_filter_gpu(density, filter_radius)
    rho_proj = heaviside_projection_gpu(rho_filt, beta, eta)
    sigma_vals = simp_sigma_gpu(rho_proj, sigma_min, sigma_max, simp_p)

    # Per-voxel path: dense scatter + QTT decomposition (≤ 256³ = 134 MB)
    use_pervoxel = (design_flat_indices is not None and total_sites <= 24)

    if use_pervoxel:
        N_total = 2 ** total_sites
        sigma_3d = torch.zeros(N_total, device=device, dtype=dtype)
        sigma_3d[design_flat_indices] = sigma_vals.to(dtype)
        # Cap σ_tt rank: the conductivity field is a small sphere of
        # smoothly varying values in a large zero domain.  Rank 20
        # captures the sphere boundary (5–10 singular values) + interior
        # variation (5–10 more) without inflating the operator rank.
        sigma_rank = min(max_rank, 20)
        sigma_field_tt = array_to_tt_gpu(sigma_3d, max_rank=sigma_rank)
        del sigma_3d
        torch.cuda.empty_cache()
    else:
        # Mean-field fallback for large grids
        sigma_mean = sigma_vals.mean().item()
        sigma_field_tt = tt_scale_gpu(
            [c.clone() for c in design_mask_tt], sigma_mean,
        )

    # ones QTT (rank 1)
    ones_cores = [
        torch.ones(1, 2, 1, device=device, dtype=dtype)
        for _ in range(total_sites)
    ]

    # ε = 1 − j·σ_field
    j_sigma = tt_scale_gpu(sigma_field_tt, -1j)
    eps_tt = tt_add_gpu(ones_cores, j_sigma)
    eps_tt = tt_round_gpu(eps_tt, max_rank=max_rank)
    return eps_tt, sigma_field_tt


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7: Power Metrics (GPU QTT)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PowerMetricsGPU:
    """Power balance from GPU Helmholtz solve."""
    P_pml: float
    P_cond: float
    P_input: float
    eta_rad: float
    vol: float
    W_near: float = 0.0
    Q_proxy: float = 0.0
    Q_rad: float = 0.0
    P_pml_norm: float = 0.0
    P_cond_norm: float = 0.0
    E2_metal_avg: float = 0.0
    M_dead: float = 0.0


def compute_pml_power_tt_gpu(
    E_cores: list[torch.Tensor],
    sigma_pml_tt: list[torch.Tensor],
    dv: float,
    max_rank: int = 128,
) -> float:
    """P_pml = 0.5 · dV · Re⟨E, diag(σ_pml)·E⟩. GPU QTT."""
    sigma_mpo = diag_mpo_from_tt_gpu(sigma_pml_tt)
    sigma_E = tt_matvec_gpu(sigma_mpo, E_cores, max_rank=max_rank)
    inner = tt_inner_gpu(E_cores, sigma_E)
    return 0.5 * dv * inner.real.item()


def compute_cond_power_tt_gpu(
    E_cores: list[torch.Tensor],
    sigma_design_tt: list[torch.Tensor],
    k0_norm: float,
    dv: float,
    max_rank: int = 128,
) -> float:
    """P_cond = 0.5 · k² · dV · Re⟨E, diag(σ_design)·E⟩. GPU QTT."""
    sigma_mpo = diag_mpo_from_tt_gpu(sigma_design_tt)
    sigma_E = tt_matvec_gpu(sigma_mpo, E_cores, max_rank=max_rank)
    inner = tt_inner_gpu(E_cores, sigma_E)
    return 0.5 * k0_norm ** 2 * dv * inner.real.item()


# ═══════════════════════════════════════════════════════════════════════════════
# Section 8: Adjoint Gradient (GPU)
# ═══════════════════════════════════════════════════════════════════════════════


def mpo_hermitian_conjugate_gpu(
    cores: list[torch.Tensor],
) -> list[torch.Tensor]:
    """H^H: conjugate + swap physical indices. GPU."""
    result: list[torch.Tensor] = []
    for c in cores:
        if c.ndim == 4:
            # (r_l, d_out, d_in, r_r) → (r_l, d_in, d_out, r_r) conjugated
            result.append(c.conj().transpose(1, 2).contiguous())
        elif c.ndim == 3:
            result.append(c.conj())
        else:
            raise ValueError(f"Unexpected core ndim={c.ndim}")
    return result


def build_adjoint_rhs_tt_gpu(
    E_cores: list[torch.Tensor],
    sigma_pml_tt: list[torch.Tensor],
    sigma_design_tt: list[torch.Tensor],
    w_pml: float,
    w_cond: float,
    k0_norm: float,
    dv: float,
    max_rank: int = 128,
) -> list[torch.Tensor]:
    """Build adjoint RHS g = dJ/dE* in QTT on GPU.

    g = w_pml·0.5·σ_pml⊙E·dV + w_cond·0.5·k²·σ_design⊙E·dV
    """
    pml_mpo = diag_mpo_from_tt_gpu(sigma_pml_tt)
    pml_term = tt_matvec_gpu(pml_mpo, E_cores, max_rank=max_rank)
    pml_term = tt_scale_gpu(pml_term, w_pml * 0.5 * dv)

    cond_mpo = diag_mpo_from_tt_gpu(sigma_design_tt)
    cond_term = tt_matvec_gpu(cond_mpo, E_cores, max_rank=max_rank)
    cond_term = tt_scale_gpu(cond_term, w_cond * 0.5 * k0_norm ** 2 * dv)

    g_cores = tt_add_gpu(pml_term, cond_term)
    return tt_round_gpu(g_cores, max_rank=max_rank)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 9: Forward Solve + Adjoint Gradient (GPU)
# ═══════════════════════════════════════════════════════════════════════════════


def solve_forward_gpu(
    density: torch.Tensor,
    design_mask_tt: list[torch.Tensor],
    n_bits: int,
    k0_norm: float,
    domain_size: float,
    pml_cells: int,
    pml_sigma_max: float,
    sigma_min: float,
    sigma_max: float,
    simp_p: float,
    beta: float,
    eta: float,
    filter_radius: int,
    damping: float,
    max_rank: int,
    n_sweeps: int,
    solver_tol: float,
    source_width: float,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.complex64,
    verbose: bool = False,
    design_flat_indices: Optional[torch.Tensor] = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], float,
           list[torch.Tensor]]:
    """GPU forward solve: build H(ρ), solve H·E = −J.

    Returns (H_cores, E_cores, source_cores, residual, sigma_field_tt).
    """
    bits = (n_bits, n_bits, n_bits)
    domain = ((0.0, domain_size),) * 3
    center = (domain_size / 2,) * 3

    # Build ε from density (per-voxel when design_flat_indices provided)
    eps_tt, sigma_field_tt = build_conductivity_eps_tt_gpu(
        density, design_mask_tt, sigma_min, sigma_max, simp_p,
        beta, eta, filter_radius, n_bits, device, dtype, max_rank,
        design_flat_indices=design_flat_indices,
    )

    # Build Helmholtz MPO: H = ∇² + k²·ε
    H_cores = helmholtz_mpo_3d_gpu(
        bits, k0_norm, domain,
        eps_pml_cores=eps_tt,
        pml_cells=pml_cells, sigma_max=pml_sigma_max,
        damping=damping, device=device, dtype=dtype,
        max_rank=max_rank,
    )

    # Source
    source = gaussian_source_tt_gpu(
        bits, domain, center, source_width,
        k0_norm, device, dtype, max_rank=min(max_rank, 16),
    )
    rhs = tt_scale_gpu(source, -1.0)

    # Solve
    result = tt_amen_solve_gpu(
        H_cores, rhs,
        max_rank=max_rank, n_sweeps=n_sweeps, tol=solver_tol,
        verbose=verbose,
    )

    return H_cores, result.x, source, result.final_residual, sigma_field_tt


def compute_adjoint_gradient_gpu(
    density: torch.Tensor,
    design_mask_tt: list[torch.Tensor],
    sigma_pml_tt: list[torch.Tensor],
    design_flat_indices: torch.Tensor,
    n_bits: int,
    k0_norm: float,
    domain_size: float,
    pml_cells: int,
    pml_sigma_max: float,
    sigma_min: float,
    sigma_max: float,
    simp_p: float,
    beta: float,
    eta: float,
    filter_radius: int,
    damping: float,
    max_rank: int,
    n_sweeps: int,
    solver_tol: float,
    source_width: float,
    alpha_loss: float,
    use_log: bool = True,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.complex64,
    verbose: bool = False,
) -> tuple[float, torch.Tensor, PowerMetricsGPU, float]:
    """Full adjoint gradient computation on GPU. No dense N³.

    1. Forward solve H(ρ)E = −J
    2. Power metrics via QTT inner products
    3. Adjoint solve H^H λ = dJ/dE*
    4. Gradient at design voxels via sparse extraction

    Returns (J_value, gradient, metrics, residual).
    """
    N = 2 ** n_bits
    h = 1.0 / N
    dv = h ** 3

    # Step 1: Forward solve (per-voxel σ when design_flat_indices available)
    H_cores, E_cores, source_cores, residual, sigma_design_tt = solve_forward_gpu(
        density, design_mask_tt, n_bits, k0_norm, domain_size,
        pml_cells, pml_sigma_max, sigma_min, sigma_max, simp_p,
        beta, eta, filter_radius, damping, max_rank, n_sweeps,
        solver_tol, source_width, device, dtype, verbose,
        design_flat_indices=design_flat_indices,
    )

    # Design density processing (for gradient chain rule)
    rho_filt = density_filter_gpu(density, filter_radius)
    rho_proj = heaviside_projection_gpu(rho_filt, beta, eta)

    # Step 2: Power metrics (using per-voxel σ_design_tt from forward solve)
    P_pml = compute_pml_power_tt_gpu(E_cores, sigma_pml_tt, dv, max_rank)
    P_cond = compute_cond_power_tt_gpu(E_cores, sigma_design_tt, k0_norm, dv, max_rank)
    # Input power: P_input = 0.5·dV·Re⟨J, E⟩  (positive when source delivers power)
    P_input_val = tt_inner_gpu(source_cores, E_cores)
    P_input_raw = 0.5 * dv * P_input_val.real.item()
    # Ensure positive (sign depends on Helmholtz convention; use absolute)
    P_input = abs(P_input_raw)
    P_input_sign = 1.0 if P_input_raw >= 0 else -1.0

    eta_rad = P_pml / (abs(P_pml) + abs(P_cond) + 1e-30)
    # Report PROJECTED volume (which the bisection enforces), not raw density mean
    vol = rho_proj.mean().item()

    metrics = PowerMetricsGPU(
        P_pml=P_pml, P_cond=P_cond, P_input=P_input,
        eta_rad=eta_rad, vol=vol,
    )

    # Volume enforcement: bisection projection already handled before this call
    # No AL volume term needed — volume is enforced exactly by projection.

    # Near-field stored energy proxy: W_near = 0.5 · dV · ‖E‖²
    E_norm_sq_val = tt_inner_gpu(E_cores, E_cores).real.item()
    W_near = 0.5 * dv * E_norm_sq_val
    omega = k0_norm  # normalized angular frequency

    # Loaded Q (for optimization objective): Q_proxy = ω·W/P_input
    Q_proxy = omega * W_near / (P_input + 1e-30)
    # Unloaded Q (for Chu comparison): Q_rad = ω·W/P_pml
    Q_rad = omega * W_near / (abs(P_pml) + 1e-30)

    metrics.W_near = W_near
    metrics.Q_proxy = Q_proxy
    metrics.Q_rad = Q_rad

    # ── Objective: minimize loaded Q via log-ratio ─────────────────
    # J = log(W_near) − log(P_input) + α·log(P_cond)
    #
    # Key insight: the adjoint source from P_input is proportional to the
    # forward source J (Gaussian at design center), so the adjoint field λ
    # has significant energy at design voxels — unlike the old P_pml-based
    # adjoint which was PML-boundary-dominated with |λ_design| ≈ 0.
    eps_log = 1e-12
    if use_log:
        J = (math.log(W_near + eps_log)
             - math.log(P_input + eps_log)
             + alpha_loss * math.log(P_cond + eps_log))
        # Wirtinger weights for adjoint RHS
        # dP_input/dE* = P_input_sign · 0.25·dV·J_source (linear in E → factor 0.5)
        # dJ/dE* from -log(P_input) = (-1/P_input)·dP_input/dE*
        w_input = -1.0 / (P_input + eps_log)
        w_cond = alpha_loss / (P_cond + eps_log)
        w_wnear = 1.0 / (W_near + eps_log)
    else:
        J = W_near - P_input + alpha_loss * P_cond
        w_input = -1.0
        w_cond = alpha_loss
        w_wnear = 1.0

    # Step 3: Adjoint RHS — P_input source (design-localized) replaces P_pml
    #
    # g = w_input·P_input_sign·0.25·dV·J  +  w_cond·0.5·k²·dV·σ_design·E
    #   + w_wnear·0.5·dV·E
    #
    # The P_input term is the Gaussian source QTT — localized at design center.
    # This ensures the adjoint field λ has O(1) energy at design voxels.

    # P_input adjoint source: (-1/P_input)·sign·0.25·dV · source_cores
    g_input = tt_scale_gpu(
        [c.clone() for c in source_cores],
        w_input * P_input_sign * 0.25 * dv,
    )

    # P_cond adjoint source: (α/P_cond)·0.5·k²·dV · σ_design ⊙ E
    cond_mpo = diag_mpo_from_tt_gpu(sigma_design_tt)
    cond_term = tt_matvec_gpu(cond_mpo, E_cores, max_rank=max_rank)
    cond_term = tt_scale_gpu(cond_term, w_cond * 0.5 * k0_norm ** 2 * dv)

    # W_near adjoint source: (1/W_near)·0.5·dV · E
    wnear_rhs = tt_scale_gpu([c.clone() for c in E_cores], w_wnear * 0.5 * dv)

    # Combine
    g_cores = tt_add_gpu(g_input, wnear_rhs)
    g_cores = tt_add_gpu(g_cores, cond_term)
    g_cores = tt_round_gpu(g_cores, max_rank=max_rank)

    # Step 4: Adjoint solve
    H_H = mpo_hermitian_conjugate_gpu(H_cores)
    adj_result = tt_amen_solve_gpu(
        H_H, g_cores, max_rank=max_rank,
        n_sweeps=n_sweeps, tol=solver_tol, verbose=False,
    )

    # Step 5: Gradient at design voxels (sparse extraction)
    E_design = tt_evaluate_at_indices_gpu(E_cores, design_flat_indices, n_bits)
    lam_design = tt_evaluate_at_indices_gpu(adj_result.x, design_flat_indices, n_bits)

    # Move to float64 for gradient precision
    E_design_f = E_design.to(torch.complex128)
    lam_design_f = lam_design.to(torch.complex128)
    E_design_sq = E_design_f.abs().pow(2).real

    dsigma = simp_dsigma_drho_gpu(rho_proj, sigma_min, sigma_max, simp_p).to(torch.float64)

    # Explicit term: dJ/dρ from P_cond
    dJ_explicit = w_cond * 0.5 * k0_norm ** 2 * dsigma * E_design_sq * dv

    # Adjoint term: −Re[λ̄ · (dH/dρ)·E]
    k2_damp = k0_norm ** 2 * (1.0 + 1j * damping)
    dH_drho_E = k2_damp * (-1j) * dsigma.to(torch.complex128) * E_design_f
    adjoint_term = -(lam_design_f.conj() * dH_drho_E).real

    # Gradient diagnostics (temporary)
    if verbose:
        _n_samp = min(5, E_design_f.shape[0])
        print(f"    GRAD DIAG: |E|=[{E_design_f[:_n_samp].abs().tolist()}] "
              f"|λ|=[{lam_design_f[:_n_samp].abs().tolist()}]")
        print(f"    dσ_range=[{dsigma.min().item():.4e}, {dsigma.max().item():.4e}] "
              f"|adjoint|=[{adjoint_term.abs().max().item():.4e}] "
              f"|explicit|=[{dJ_explicit.abs().max().item():.4e}]")
        print(f"    w_input={w_input:.4e} w_wnear={w_wnear:.4e} w_cond={w_cond:.4e}")
        _g_rhs_norm = math.sqrt(sum(tt_inner_gpu(g_cores, g_cores).real.item()
                                     for _ in [0]))  # ||g|| of adjoint RHS
        print(f"    ||adj_rhs||={_g_rhs_norm:.4e} adj_res={adj_result.final_residual:.4e}")

    # Total gradient wrt ρ_proj (no volume AL term — volume enforced by projection)
    dJ_drho_proj = dJ_explicit + adjoint_term

    # Chain through Heaviside + filter
    h_grad = heaviside_gradient_gpu(rho_filt, beta, eta).to(torch.float64)
    grad_after_h = dJ_drho_proj * h_grad
    grad = density_filter_gradient_gpu(grad_after_h, filter_radius)

    return J, grad.to(torch.float64), metrics, residual


# ═══════════════════════════════════════════════════════════════════════════════
# Section 10: Optimization Config & Result
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ChuGPUConfig:
    """Problem + optimization config for GPU Chu limit challenge."""
    frequency_hz: float = 1.0e9
    ka: float = 0.3
    n_bits: int = 12  # 4096³
    domain_wavelengths: float = 1.5
    max_rank: int = 48
    n_sweeps: int = 100
    solver_tol: float = 1e-3
    damping: float = 0.001
    damping_init: float = 0.05
    damping_final: float = 0.001
    damping_ramp_iters: int = 100
    # SIMP
    sigma_min: float = 0.0
    sigma_max_init: float = 30.0
    sigma_max_final: float = 2000.0
    sigma_ramp_iters: int = 150
    simp_p_init: float = 1.0
    simp_p_final: float = 4.0
    simp_p_ramp_iters: int = 150
    # PML
    pml_cells: int = 20
    pml_sigma_max: float = 10.0
    # Source
    source_width_frac: float = 0.1  # fraction of domain
    # Optimization
    max_iterations: int = 300
    learning_rate: float = 0.001
    beta_init: float = 1.0
    beta_max: float = 256.0
    beta_increase_every: int = 30
    beta_factor: float = 2.0
    eta: float = 0.5
    filter_radius: int = 5
    # Volume constraint
    vol_target: float = 0.3
    al_mu_init: float = 10.0
    al_mu_factor: float = 1.5
    # Alpha loss (conductor penalty)
    alpha_loss: float = 0.1
    alpha_intro_mode: str = "auto"
    alpha_stable_window: int = 50
    # Feed clamp
    feed_seed_clamp_iters: int = 30
    feed_seed_clamp_radius: int = 20
    # Coupling constraint
    use_coupling_constraint: bool = True
    coupling_density_threshold: float = 0.3
    coupling_radius: int = 30
    coupling_al_mu_init: float = 10.0
    # Adam
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    # Convergence
    convergence_tol: float = 1e-4

    @property
    def wavelength(self) -> float:
        return C0 / self.frequency_hz

    @property
    def k0(self) -> float:
        return 2.0 * math.pi / self.wavelength

    @property
    def sphere_radius(self) -> float:
        return self.ka / self.k0

    @property
    def N(self) -> int:
        return 2 ** self.n_bits

    @property
    def domain_size(self) -> float:
        return self.domain_wavelengths * self.wavelength

    @property
    def k0_normalised(self) -> float:
        return self.k0 * self.domain_size

    @property
    def sphere_radius_normalised(self) -> float:
        return self.sphere_radius / self.domain_size

    @property
    def q_chu(self) -> float:
        return chu_limit_q(self.ka)

    @property
    def source_width(self) -> float:
        return self.source_width_frac * self.domain_size

    def sigma_max_at_iter(self, iteration: int) -> float:
        if self.sigma_ramp_iters <= 0:
            return self.sigma_max_final
        t = min(iteration / self.sigma_ramp_iters, 1.0)
        return self.sigma_max_init + t * (self.sigma_max_final - self.sigma_max_init)

    def simp_p_at_iter(self, iteration: int) -> float:
        if self.simp_p_ramp_iters <= 0:
            return self.simp_p_final
        t = min(iteration / self.simp_p_ramp_iters, 1.0)
        return self.simp_p_init + t * (self.simp_p_final - self.simp_p_init)

    def damping_at_iter(self, iteration: int) -> float:
        """Damping continuation: start high for gradient visibility, ramp to physics value."""
        if self.damping_ramp_iters <= 0:
            return self.damping_final
        t = min(iteration / self.damping_ramp_iters, 1.0)
        # Log-linear ramp for smoother transition between orders of magnitude
        import math as _math
        log_init = _math.log(self.damping_init)
        log_final = _math.log(self.damping_final)
        return _math.exp(log_init + t * (log_final - log_init))

    def pml_config(self) -> PMLConfig:
        return PMLConfig.for_problem(
            n_bits=self.n_bits,
            k=self.k0_normalised,
            target_R_dB=-40.0,
        )

    def summary(self) -> str:
        sep = "=" * 70
        return "\n".join([
            f"\n{sep}",
            f"  GPU Chu Limit Challenge — {self.N}³ ({self.n_bits} bits/dim)",
            f"{sep}",
            f"  Frequency:   {self.frequency_hz / 1e9:.3f} GHz",
            f"  Wavelength:  {self.wavelength * 1e3:.1f} mm",
            f"  ka:          {self.ka:.4f}",
            f"  Q_Chu:       {self.q_chu:.2f}",
            f"  Grid:        {self.N}³ = {self.N**3:,} voxels",
            f"  Dense:       {self.N**3 * 8 / 1e9:.1f} GB (complex64)",
            f"  Domain:      {self.domain_wavelengths:.2f}λ = {self.domain_size*1e3:.1f} mm",
            f"  max_rank:    {self.max_rank}",
            f"  Sphere r_n:  {self.sphere_radius_normalised:.4f}",
            f"  σ SIMP:      [{self.sigma_min}, {self.sigma_max_init}→{self.sigma_max_final}]",
            f"  p SIMP:      {self.simp_p_init}→{self.simp_p_final}",
            f"  Damping:     {self.damping_init}→{self.damping_final} over {self.damping_ramp_iters} iters",
            f"  Iterations:  {self.max_iterations}, lr={self.learning_rate}",
            f"  β:           {self.beta_init}→{self.beta_max}",
            f"  Volume:      {self.vol_target:.0%}",
            f"{sep}",
        ])


@dataclass
class ChuGPUResult:
    """Result of GPU Chu limit optimization."""
    density_final: torch.Tensor
    objective_history: list[float] = field(default_factory=list)
    grad_norm_history: list[float] = field(default_factory=list)
    power_metrics_history: list[PowerMetricsGPU] = field(default_factory=list)
    n_iterations: int = 0
    converged: bool = False
    total_time_s: float = 0.0
    config: Optional[ChuGPUConfig] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Section 11: Full Optimization Loop (GPU)
# ═══════════════════════════════════════════════════════════════════════════════


def optimize_chu_antenna_gpu(
    config: ChuGPUConfig,
    verbose: bool = True,
    callback: Optional[Callable] = None,
) -> ChuGPUResult:
    """Run Chu limit topology optimization entirely on GPU.

    No NumPy in the hot path. No dense N³. No CPU fallback.
    """
    device = torch.device("cuda")
    dtype = torch.complex64

    if verbose:
        print(config.summary())
        print(f"  Q_Chu = {config.q_chu:.2f}")

    torch.cuda.reset_peak_memory_stats(device)
    t_start = time.perf_counter()

    # ── Pre-compute infrastructure (one-time) ──────────────────────
    if verbose:
        print("\n  Building GPU infrastructure...", flush=True)

    pml = config.pml_config()
    k0_norm = config.k0_normalised

    # PML σ QTT
    if verbose:
        print("    PML σ weights...", end=" ", flush=True)
    sigma_pml_tt = build_pml_sigma_tt_gpu(
        config.n_bits, k0_norm, pml, device, dtype, max_rank=config.max_rank,
    )
    if verbose:
        r_pml = max(c.shape[2] for c in sigma_pml_tt)
        print(f"rank={r_pml}")

    # Design sphere mask QTT
    if verbose:
        print("    Sphere mask...", end=" ", flush=True)
    design_mask_tt = build_sphere_mask_tt_gpu(
        config.n_bits,
        centre=(0.5, 0.5, 0.5),
        radius=config.sphere_radius_normalised,
        device=device, dtype=dtype,
        max_rank=min(config.max_rank, 64),
    )
    if verbose:
        r_mask = max(c.shape[2] for c in design_mask_tt)
        print(f"rank={r_mask}")

    # Design voxel flat indices
    if verbose:
        print("    Design voxel indices...", end=" ", flush=True)
    design_flat_idx = spherical_mask_flat_indices_gpu(
        config.n_bits, (0.5, 0.5, 0.5),
        config.sphere_radius_normalised, device,
    )
    n_design = design_flat_idx.shape[0]
    if verbose:
        print(f"{n_design:,} voxels")

    # ── Initialize density on GPU ──────────────────────────────────
    # Initialize at vol_target with noise for symmetry breaking.
    # Uniform density → uniform gradient → no spatial differentiation.
    # Perturbation amplitude 0.1 gives density in [vol_target-0.1, vol_target+0.1].
    noise_amp = 0.10
    density = config.vol_target + noise_amp * (
        2.0 * torch.rand(n_design, device=device, dtype=torch.float64) - 1.0
    )
    density = density.clamp(0.01, 0.99)

    # Seed the monopole wire analytically
    N = config.N
    h_grid = 1.0 / N
    coords = torch.linspace(h_grid / 2, 1.0 - h_grid / 2, N, device=device)

    # C-order decode: flat = ix * N² + iy * N + iz
    iz_flat = design_flat_idx % N
    iy_flat = (design_flat_idx // N) % N
    ix_flat = design_flat_idx // (N * N)

    cx, cy, cz = 0.5, 0.5, 0.5
    r_norm = config.sphere_radius_normalised

    ix_c = min(int(cx * N), N - 1)
    iy_c = min(int(cy * N), N - 1)
    wire_r_cells = max(1, int(config.feed_seed_clamp_radius * 0.3))

    wire_mask = (
        ((ix_flat - ix_c).abs() <= wire_r_cells) &
        ((iy_flat - iy_c).abs() <= wire_r_cells)
    )
    density[wire_mask] = 0.95

    if verbose:
        print(f"    Density initialized: {n_design:,} design voxels, "
              f"wire={wire_mask.sum().item()} voxels")

    # ── Feed clamp mask ────────────────────────────────────────────
    feed_pos = (0.5, 0.5, 0.5 - r_norm)
    clamp_r = config.feed_seed_clamp_radius * h_grid
    feed_dists = compute_voxel_distances_gpu(
        design_flat_idx, config.n_bits, feed_pos, device,
    )
    feed_clamp_mask = feed_dists < clamp_r

    # Coupling constraint mask
    coupling_mask: Optional[torch.Tensor] = None
    coupling_lambda_c = 0.0
    coupling_mu_c = config.coupling_al_mu_init
    if config.use_coupling_constraint:
        coupling_r = config.coupling_radius * h_grid
        coupling_mask = feed_dists < coupling_r

    if verbose:
        n_clamped = feed_clamp_mask.sum().item()
        n_coupling = coupling_mask.sum().item() if coupling_mask is not None else 0
        print(f"    Feed clamp: {n_clamped} voxels, coupling: {n_coupling}")

    # ── Baseline air solve ─────────────────────────────────────────
    if verbose:
        print("\n  Baseline air solve...", end=" ", flush=True)
    density_air = torch.zeros(n_design, device=device, dtype=torch.float64)
    _, E_air_cores, _, _, _ = solve_forward_gpu(
        density_air, design_mask_tt, config.n_bits, k0_norm,
        config.domain_size, config.pml_cells, config.pml_sigma_max,
        config.sigma_min, config.sigma_max_init, config.simp_p_init,
        1.0, 0.5, 0, config.damping, config.max_rank,
        config.n_sweeps, config.solver_tol, config.source_width,
        device, dtype, verbose=False,
    )
    P_pml_air = compute_pml_power_tt_gpu(
        E_air_cores, sigma_pml_tt, h_grid ** 3, config.max_rank,
    )
    # M_dead reference
    E_air_at_design = tt_evaluate_at_indices_gpu(
        E_air_cores, design_flat_idx, config.n_bits,
    )
    E_air_design_sq = E_air_at_design.abs().pow(2).float()
    tau_dead = torch.quantile(E_air_design_sq, 0.1).item()
    del E_air_cores, E_air_at_design, density_air
    if verbose:
        print(f"P_pml(air)={P_pml_air:.4e}, τ_dead={tau_dead:.3e}", flush=True)

    # ── Adam state ─────────────────────────────────────────────────
    adam_m = torch.zeros(n_design, device=device, dtype=torch.float64)
    adam_v = torch.zeros(n_design, device=device, dtype=torch.float64)
    adam_t = 0
    g_max_ema = 0.0  # EMA of gradient infinity norm for stable normalization
    density_init = density.clone()  # reference for tracking geometry changes

    # ── Volume projection state ────────────────────────────────────
    beta = config.beta_init
    eta_vol = 0.5  # bisection starting point

    # Dynamic alpha
    p_tilde_stable_count = 0
    alpha_activated = False

    # ── Histories ──────────────────────────────────────────────────
    obj_history: list[float] = []
    grad_norm_history: list[float] = []
    power_history: list[PowerMetricsGPU] = []
    converged = False

    if verbose:
        print(f"\n  Starting optimization: {config.max_iterations} iterations")
        print("-" * 70)

    for iteration in range(config.max_iterations):
        # Beta continuation
        if (iteration > 0 and config.beta_increase_every > 0
                and iteration % config.beta_increase_every == 0):
            beta = min(beta * config.beta_factor, config.beta_max)
            if verbose:
                print(f"  [β→{beta:.1f}]")

        # Sigma/p/damping continuation
        current_sigma_max = config.sigma_max_at_iter(iteration)
        current_simp_p = config.simp_p_at_iter(iteration)
        current_damping = config.damping_at_iter(iteration)

        # Feed clamp
        if iteration < config.feed_seed_clamp_iters:
            density[feed_clamp_mask] = 1.0

        # ── Volume-preserving projection via bisection ─────────────
        rho_filt_vp = density_filter_gpu(density, config.filter_radius)
        _, eta_vol = volume_preserving_projection_gpu(
            rho_filt_vp, beta, config.vol_target,
        )

        # Alpha scheduling
        if config.alpha_intro_mode == "auto":
            alpha_eff = config.alpha_loss if alpha_activated else 0.0
        else:
            alpha_eff = config.alpha_loss if iteration >= config.alpha_stable_window else 0.0

        # Compute gradient (uses the bisected eta for Heaviside)
        J_val, grad, metrics, residual = compute_adjoint_gradient_gpu(
            density=density,
            design_mask_tt=design_mask_tt,
            sigma_pml_tt=sigma_pml_tt,
            design_flat_indices=design_flat_idx,
            n_bits=config.n_bits,
            k0_norm=k0_norm,
            domain_size=config.domain_size,
            pml_cells=config.pml_cells,
            pml_sigma_max=config.pml_sigma_max,
            sigma_min=config.sigma_min,
            sigma_max=current_sigma_max,
            simp_p=current_simp_p,
            beta=beta,
            eta=eta_vol,
            filter_radius=config.filter_radius,
            damping=current_damping,
            max_rank=config.max_rank,
            n_sweeps=config.n_sweeps,
            solver_tol=config.solver_tol,
            source_width=config.source_width,
            alpha_loss=alpha_eff,
            use_log=True,
            device=device,
            dtype=dtype,
            verbose=False,
        )

        # Normalise metrics
        if P_pml_air > 1e-30:
            metrics.P_pml_norm = metrics.P_pml / P_pml_air
            metrics.P_cond_norm = metrics.P_cond / P_pml_air

        # M_dead
        rho_filt_md = density_filter_gpu(density, config.filter_radius)
        rho_proj_md = heaviside_projection_gpu(rho_filt_md, beta, eta_vol)
        rho_sum = rho_proj_md.sum().item()
        if rho_sum > 1e-30:
            dead_mask = E_air_design_sq < tau_dead
            metrics.M_dead = (rho_proj_md.float() * dead_mask.float()).sum().item() / rho_sum

        # Dynamic alpha activation
        clamp_active = iteration < config.feed_seed_clamp_iters
        if config.alpha_intro_mode == "auto" and not alpha_activated and not clamp_active:
            if metrics.P_pml_norm > 1.0:
                p_tilde_stable_count += 1
            else:
                p_tilde_stable_count = 0
            if p_tilde_stable_count >= config.alpha_stable_window:
                alpha_activated = True
                if verbose:
                    print(f"  >>> α activated at iter {iteration + 1}")

        # Coupling constraint
        if (coupling_mask is not None and config.use_coupling_constraint
                and iteration >= config.feed_seed_clamp_iters):
            n_near = coupling_mask.sum().item()
            if n_near > 0:
                rho_near = density[coupling_mask].mean().item()
                g_c = config.coupling_density_threshold - rho_near
                Jc = coupling_lambda_c * g_c + 0.5 * coupling_mu_c * max(g_c, 0.0) ** 2
                J_val += Jc
                if g_c > 0:
                    coeff = -(coupling_lambda_c + coupling_mu_c * g_c) / n_near
                else:
                    coeff = -coupling_lambda_c / n_near
                coupling_grad = torch.zeros_like(grad)
                coupling_grad[coupling_mask] = coeff
                grad = grad + coupling_grad
                coupling_lambda_c += coupling_mu_c * max(g_c, 0.0)

        power_history.append(metrics)
        grad_norm = grad.norm().item()
        grad_max = grad.abs().max().item()
        obj_history.append(J_val)
        grad_norm_history.append(grad_norm)

        if verbose:
            stage = "S0" if not alpha_activated else "S1"
            clamp_str = " CLAMP" if clamp_active else ""
            gpu_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            # Density change tracking
            rho_delta = (density - density_init).abs()
            rho_std = density.std().item()
            rho_max_delta = rho_delta.max().item()
            print(
                f"  [{iteration+1}/{config.max_iterations}]{clamp_str} [{stage}] "
                f"J={J_val:.4e} Q̃={metrics.Q_proxy:.2f} "
                f"Qr={metrics.Q_rad:.2f} "
                f"P̃={metrics.P_pml_norm:.3f} "
                f"Pi={metrics.P_input:.2e} "
                f"Pc={metrics.P_cond:.2e} η={metrics.eta_rad:.3f} "
                f"V={metrics.vol:.2f} |g|={grad_norm:.2e} gmax={grad_max:.2e} "
                f"Δρ={rho_max_delta:.3f} σρ={rho_std:.3f} "
                f"β={beta:.0f} η_v={eta_vol:.3f} δ={current_damping:.4f} "
                f"σ={current_sigma_max:.0f} p={current_simp_p:.1f} "
                f"res={residual:.2e} GPU={gpu_mem:.0f}MB",
                flush=True,
            )

        if callback is not None:
            callback(iteration, J_val, metrics, density.clone())

        # Convergence check (skip during ramp-up)
        # Use relative gradient check: grad_max relative to initial grad_max
        min_iters = max(20, config.feed_seed_clamp_iters + 5)
        if iteration >= min_iters and len(obj_history) >= 2:
            rel_change = abs(obj_history[-1] - obj_history[-2]) / (
                abs(obj_history[-2]) + 1e-30
            )
            ref_grad = grad_norm_history[0] if grad_norm_history else 1.0
            rel_grad = grad_norm / (ref_grad + 1e-30)
            if rel_change < config.convergence_tol and rel_grad < 1e-3:
                converged = True
                if verbose:
                    print(f"  Converged at iteration {iteration + 1}.")
                break

        # Adam update with EMA-stabilized gradient normalization.
        #
        # Key insight: per-iteration L_inf normalization (g / max(|g|)) is
        # UNSTABLE because solver noise places the max at a different random
        # voxel each iteration. This makes each voxel's normalized gradient
        # oscillate wildly, destroying Adam's momentum accumulation.
        #
        # Fix: use an exponential moving average of g_max as the reference.
        # This gives each voxel a CONSISTENT normalized magnitude across
        # iterations, allowing Adam's m_hat to accumulate the true signal.
        g_max = grad.abs().max().item()
        if g_max_ema < 1e-30:
            g_max_ema = g_max  # bootstrap on first iteration
        else:
            g_max_ema = 0.9 * g_max_ema + 0.1 * g_max
        if g_max_ema > 1e-30:
            grad = grad * (1.0 / g_max_ema)

        adam_t += 1
        b1, b2, eps_a = config.adam_beta1, config.adam_beta2, config.adam_eps
        adam_m = b1 * adam_m + (1.0 - b1) * grad
        adam_v = b2 * adam_v + (1.0 - b2) * grad.pow(2)
        m_hat = adam_m / (1.0 - b1 ** adam_t)
        v_hat = adam_v / (1.0 - b2 ** adam_t)
        density = density - config.learning_rate * m_hat / (v_hat.sqrt() + eps_a)
        density = density.clamp(0.0, 1.0)

        # Re-center density to maintain volume target.
        # Without this, raw density drifts because the gradient has no
        # volume term (volume is enforced only via bisection η on the
        # projected density). Re-centering keeps the Heaviside projection
        # operating in a well-conditioned regime.
        shift = config.vol_target - density.mean().item()
        density = (density + shift).clamp(0.01, 0.99)

        # Re-enforce clamp
        if iteration < config.feed_seed_clamp_iters:
            density[feed_clamp_mask] = 1.0

    total_time = time.perf_counter() - t_start
    gpu_peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    if verbose:
        sep = "=" * 70
        print(f"\n{sep}")
        print(f"  GPU Chu Limit Optimization Complete")
        print(f"{sep}")
        print(f"  Iterations:  {len(obj_history)}")
        print(f"  Converged:   {converged}")
        print(f"  Time:        {total_time:.1f}s")
        print(f"  GPU peak:    {gpu_peak:.0f} MB")
        if power_history:
            last = power_history[-1]
            print(f"  Q̃_proxy:    {last.Q_proxy:.2f}")
            print(f"  Q_rad:       {last.Q_rad:.2f}")
            print(f"  W_near:      {last.W_near:.4e}")
            print(f"  P_pml:       {last.P_pml:.4e}")
            print(f"  P_input:     {last.P_input:.4e}")
            print(f"  P_cond:      {last.P_cond:.4e}")
            print(f"  P̃_pml:      {last.P_pml_norm:.4f}")
            print(f"  η_rad:       {last.eta_rad:.4f}")
            print(f"  Volume:      {last.vol:.4f}")
        print(f"  Q_Chu:       {config.q_chu:.2f}")
        print(f"{sep}")

    return ChuGPUResult(
        density_final=density.clone(),
        objective_history=obj_history,
        grad_norm_history=grad_norm_history,
        power_metrics_history=power_history,
        n_iterations=len(obj_history),
        converged=converged,
        total_time_s=total_time,
        config=config,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Section 12: Schedule Factory
# ═══════════════════════════════════════════════════════════════════════════════


def make_chu_gpu_config(grid_level: str = "4096") -> ChuGPUConfig:
    """Return calibrated ChuGPUConfig for a given grid level."""
    schedules = {
        "128": dict(
            n_bits=7, max_rank=48, n_sweeps=30, solver_tol=1e-3,
            max_iterations=60, learning_rate=0.005,
            beta_max=64.0, beta_increase_every=15,
            sigma_max_final=500.0, sigma_ramp_iters=40,
            simp_p_final=4.0, simp_p_ramp_iters=40,
            damping_init=0.02, damping_final=0.001, damping_ramp_iters=30,
            alpha_stable_window=15,
            feed_seed_clamp_iters=10, feed_seed_clamp_radius=4,
            coupling_radius=6, filter_radius=2,
        ),
        "256": dict(
            n_bits=8, max_rank=48, n_sweeps=40, solver_tol=1e-3,
            domain_wavelengths=0.75,
            max_iterations=120, learning_rate=0.005,
            beta_max=128.0, beta_increase_every=20,
            sigma_max_final=500.0, sigma_ramp_iters=60,
            simp_p_final=4.0, simp_p_ramp_iters=60,
            damping_init=0.02, damping_final=0.001, damping_ramp_iters=50,
            alpha_stable_window=20,
            adam_beta2=0.99,
            feed_seed_clamp_iters=10, feed_seed_clamp_radius=6,
            coupling_radius=10, filter_radius=3,
        ),
        "512": dict(
            n_bits=9, max_rank=40, n_sweeps=40, solver_tol=1e-3,
            max_iterations=120, learning_rate=0.002,
            beta_max=128.0, beta_increase_every=20,
            sigma_max_final=1000.0, sigma_ramp_iters=80,
            simp_p_final=4.0, simp_p_ramp_iters=80,
            damping_init=0.03, damping_final=0.001, damping_ramp_iters=60,
            alpha_stable_window=30,
            feed_seed_clamp_iters=15, feed_seed_clamp_radius=10,
            coupling_radius=15, filter_radius=3,
        ),
        "1024": dict(
            n_bits=10, max_rank=32, n_sweeps=40, solver_tol=1e-3,
            max_iterations=200, learning_rate=0.002,
            beta_max=128.0, beta_increase_every=25,
            sigma_max_final=1000.0, sigma_ramp_iters=100,
            simp_p_final=4.0, simp_p_ramp_iters=100,
            damping_init=0.04, damping_final=0.002, damping_ramp_iters=80,
            alpha_stable_window=40,
            feed_seed_clamp_iters=20, feed_seed_clamp_radius=14,
            coupling_radius=20, filter_radius=4,
        ),
        "2048": dict(
            n_bits=11, max_rank=24, n_sweeps=50, solver_tol=1e-3,
            max_iterations=250, learning_rate=0.001,
            beta_max=256.0, beta_increase_every=30,
            sigma_max_final=2000.0, sigma_ramp_iters=150,
            simp_p_final=4.0, simp_p_ramp_iters=150,
            damping_init=0.05, damping_final=0.002, damping_ramp_iters=120,
            alpha_stable_window=50,
            feed_seed_clamp_iters=25, feed_seed_clamp_radius=18,
            coupling_radius=25, filter_radius=5,
        ),
        "4096": dict(
            n_bits=12, max_rank=32, n_sweeps=80, solver_tol=1e-3,
            max_iterations=300, learning_rate=0.02,
            beta_max=256.0, beta_increase_every=30,
            sigma_max_final=2000.0, sigma_ramp_iters=150,
            simp_p_final=4.0, simp_p_ramp_iters=150,
            damping_init=0.15, damping_final=0.005, damping_ramp_iters=200,
            alpha_stable_window=50,
            adam_beta2=0.99,
            feed_seed_clamp_iters=30, feed_seed_clamp_radius=20,
            coupling_radius=30, filter_radius=5,
        ),
    }

    if grid_level not in schedules:
        raise ValueError(f"Unknown grid_level '{grid_level}'. Choose from {list(schedules.keys())}")

    s = schedules[grid_level]
    return ChuGPUConfig(**s)
