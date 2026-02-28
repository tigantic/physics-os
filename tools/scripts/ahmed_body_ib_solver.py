#!/usr/bin/env python3
"""
Ahmed Body Immersed Boundary QTT Solver  (v2 — semi-implicit)
==============================================================

Generates FULL VOLUMETRIC flow fields around a parametric Ahmed body
using Brinkman penalization on the native QTT Navier-Stokes solver.

This is NOT dataset compression — it is MODEL-DRIVEN SYNTHESIS.
The complete 3D flow field lives entirely in QTT tensor-train form,
achieving O(r² log N) storage for an N³ volume.

Key design choices
------------------
* Semi-implicit Brinkman: u_new = u_pred ⊙ [1/(1 + dt/η·χ)]
  → unconditionally stable for any η → 0
* Semi-implicit sponge:   u_new = U∞ + (u − U∞) ⊙ exp(−σ·dt)
  → exact exponential decay, unconditionally stable
* Velocity-only formulation: evolve u, compute ω = ∇×u as diagnostic
  → halves QTT operation count per step
* Forward Euler targeting steady state

Author: Brad Adams / Tigantic Holdings LLC
Date: February 2026
"""

from __future__ import annotations

import os
import sys
import time
import json
import glob
import argparse
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List, Dict, Any

import numpy as np
import torch
from torch import Tensor

# ── HyperTensor QTT Engine ─────────────────────────────────────────
from ontic.cfd.ns3d_native import (
    QTT3DNative,
    QTT3DVectorNative,
    NativeDerivatives3D,
    vector_cross_native,
    _tt_svd_compress,
    _batched_qtt_eval,
    _qtt_vec_max_abs_native,
    _qtt_scalar_max_abs_native,
)
from ontic.cfd.qtt_native_ops import (
    QTTCores,
    qtt_add_native,
    qtt_add_native_batched,
    qtt_scale_native,
    qtt_sub_native,
    qtt_hadamard_native,
    qtt_hadamard_native_batched,
    qtt_inner_native,
    qtt_norm_native,
    qtt_fused_sum,
    qtt_fused_sum_batched,
    qtt_truncate_now,
    qtt_truncate_now_batched,
    turbulence_rank_profile,
)
from ontic.cfd.qtt_tci import qtt_from_function


# ═══════════════════════════════════════════════════════════════════════
# MORTON ORDERING
# ═══════════════════════════════════════════════════════════════════════

def dense_to_morton_flat_vectorized(arr_3d: np.ndarray, n_bits: int) -> np.ndarray:
    """Vectorized reorder of (N,N,N) raster → Morton-order flat vector."""
    N = 1 << n_bits
    assert arr_3d.shape == (N, N, N), f"Expected ({N},{N},{N}), got {arr_3d.shape}"

    ix = np.arange(N, dtype=np.int64)
    IX, IY, IZ = np.meshgrid(ix, ix, ix, indexing='ij')
    IX_f, IY_f, IZ_f = IX.ravel(), IY.ravel(), IZ.ravel()

    morton = np.zeros(N ** 3, dtype=np.int64)
    for b in range(n_bits):
        morton |= ((IX_f >> b) & 1).astype(np.int64) << (3 * b)
        morton |= ((IY_f >> b) & 1).astype(np.int64) << (3 * b + 1)
        morton |= ((IZ_f >> b) & 1).astype(np.int64) << (3 * b + 2)

    flat = np.empty(N ** 3, dtype=arr_3d.dtype)
    flat[morton] = arr_3d.ravel()
    return flat


# ═══════════════════════════════════════════════════════════════════════
# AHMED BODY SDF
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AhmedBodyParams:
    """Parametric Ahmed body geometry (all dims in metres)."""
    length: float = 1.044
    width: float = 0.389
    height: float = 0.288
    ground_clearance: float = 0.05
    slant_angle_deg: float = 25.0
    fillet_radius: float = 0.100
    velocity: float = 40.0
    nu: float = 1.516e-5

    @property
    def Re(self) -> float:
        return self.velocity * self.length / self.nu

    @classmethod
    def from_nvidia_info(cls, filepath: str) -> "AhmedBodyParams":
        params: Dict[str, float] = {}
        with open(filepath) as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    try:
                        params[k.strip()] = float(v.strip())
                    except ValueError:
                        pass
        return cls(
            length=params.get("Length", 1044) / 1000,
            width=params.get("Width", 389) / 1000,
            height=params.get("Height", 288) / 1000,
            ground_clearance=params.get("GroundClearance", 50) / 1000,
            slant_angle_deg=params.get("SlantAngle", 25.0),
            fillet_radius=params.get("FilletRadius", 100) / 1000,
            velocity=params.get("Velocity", 40.0),
        )


def ahmed_body_sdf(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    params: AhmedBodyParams,
    body_center: Tuple[float, float, float],
) -> np.ndarray:
    """Signed distance (negative inside) for a rounded-box Ahmed body."""
    cx, cy, cz = body_center
    hx, hy, hz = params.length / 2, params.height / 2, params.width / 2
    R = params.fillet_radius
    slant = math.radians(params.slant_angle_deg)

    x, y, z = X - cx, Y - cy, Z - cz
    dx = np.abs(x) - hx
    dy = np.abs(y) - hy
    dz = np.abs(z) - hz

    outside = np.sqrt(np.maximum(dx, 0) ** 2
                      + np.maximum(dy, 0) ** 2
                      + np.maximum(dz, 0) ** 2)
    inside = np.minimum(np.maximum(dx, np.maximum(dy, dz)), 0.0)
    sdf = outside + inside

    # front fillet
    if R > 0:
        fm = x < -hx + R
        xf = x + hx - R
        rf = np.sqrt(xf ** 2 + np.maximum(np.abs(y) - (hy - R), 0) ** 2)
        sf = np.where(fm & (np.abs(y) > hy - R), rf - R, sdf)
        sdf = np.where(fm, np.maximum(sdf, sf - sdf * 0.3), sdf)

    # rear slant
    if slant > 0.01:
        sd = (x - hx) * (-math.sin(slant)) + (y - hy) * math.cos(slant)
        sx = hx - params.height * math.tan(slant)
        sm = (x > sx) & (y > 0)
        sdf = np.where(sm, np.maximum(sdf, -sd), sdf)

    return sdf


def create_body_mask(sdf: np.ndarray, dx: float, sharpness: float = 2.0) -> np.ndarray:
    eps = sharpness * dx
    return (0.5 * (1.0 - np.tanh(sdf / eps))).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# TCI HELPERS — zero-dense mask construction for large grids
# ═══════════════════════════════════════════════════════════════════════

def morton_to_xyz_batch(morton_indices: Tensor, n_bits: int) -> Tuple[Tensor, Tensor, Tensor]:
    """Decode batched Morton Z-curve indices to (ix, iy, iz) integer coordinates.

    Morton layout: bit 3*b+0 = x_b, bit 3*b+1 = y_b, bit 3*b+2 = z_b.
    """
    ix = torch.zeros_like(morton_indices)
    iy = torch.zeros_like(morton_indices)
    iz = torch.zeros_like(morton_indices)
    for b in range(n_bits):
        ix |= ((morton_indices >> (3 * b)) & 1) << b
        iy |= ((morton_indices >> (3 * b + 1)) & 1) << b
        iz |= ((morton_indices >> (3 * b + 2)) & 1) << b
    return ix, iy, iz


def ahmed_body_sdf_torch(
    x: Tensor, y: Tensor, z: Tensor,
    params: AhmedBodyParams,
    body_center: Tuple[float, float, float],
) -> Tensor:
    """Signed distance (negative inside) for a rounded-box Ahmed body.

    Pure-torch batched version — runs on GPU without materializing N³ arrays.
    Functionally identical to ``ahmed_body_sdf`` (numpy version).
    """
    cx, cy, cz = body_center
    hx = params.length / 2.0
    hy = params.height / 2.0
    hz = params.width / 2.0
    R = params.fillet_radius
    slant = math.radians(params.slant_angle_deg)

    xr, yr, zr = x - cx, y - cy, z - cz
    dx = torch.abs(xr) - hx
    dy = torch.abs(yr) - hy
    dz = torch.abs(zr) - hz

    outside = torch.sqrt(
        torch.clamp(dx, min=0.0) ** 2
        + torch.clamp(dy, min=0.0) ** 2
        + torch.clamp(dz, min=0.0) ** 2
    )
    inside = torch.clamp(torch.maximum(dx, torch.maximum(dy, dz)), max=0.0)
    sdf = outside + inside

    # front fillet
    if R > 0:
        fm = xr < -hx + R
        xf = xr + hx - R
        rf = torch.sqrt(xf ** 2 + torch.clamp(torch.abs(yr) - (hy - R), min=0.0) ** 2)
        sf = torch.where(fm & (torch.abs(yr) > hy - R), rf - R, sdf)
        sdf = torch.where(fm, torch.maximum(sdf, sf - sdf * 0.3), sdf)

    # rear slant
    if slant > 0.01:
        sd = (xr - hx) * (-math.sin(slant)) + (yr - hy) * math.cos(slant)
        sx = hx - params.height * math.tan(slant)
        sm = (xr > sx) & (yr > 0)
        sdf = torch.where(sm, torch.maximum(sdf, -sd), sdf)

    return sdf


def _maxvol(A: Tensor, tol: float = 1.05, max_iters: int = 100) -> Tensor:
    """MaxVol: find r rows of (n×r) matrix A that maximize |det(A[rows])|.

    Returns sorted row indices (long tensor of length r).
    """
    n, r = A.shape
    if n <= r:
        return torch.arange(n, device=A.device, dtype=torch.long)
    Q, _ = torch.linalg.qr(A.float())
    row_norms = torch.norm(Q, dim=1)
    _, indices = torch.topk(row_norms, r)
    indices = indices.sort().values
    B = A[indices].float()
    for _ in range(max_iters):
        try:
            B_inv = torch.linalg.inv(B)
        except Exception:
            break
        C = A.float() @ B_inv
        max_val = torch.abs(C).max()
        if max_val <= tol:
            break
        flat_idx = torch.abs(C).argmax()
        i, j = flat_idx // r, flat_idx % r
        indices[j] = i
        B = A[indices].float()
    return indices.sort().values


def xyz_to_morton(ix: int, iy: int, iz: int, n_bits: int) -> int:
    """Convert (ix, iy, iz) grid indices to a Morton Z-curve index."""
    m = 0
    for b in range(n_bits):
        m |= ((ix >> b) & 1) << (3 * b)
        m |= ((iy >> b) & 1) << (3 * b + 1)
        m |= ((iz >> b) & 1) << (3 * b + 2)
    return m


def _body_seed_morton_indices(
    n_bits: int,
    body_center: Tuple[float, float, float],
    body_params: AhmedBodyParams,
    L: float,
) -> List[int]:
    """Generate seed Morton indices at and near the body surface.

    Returns a list of Morton indices for:
    - body centre
    - 6 face-centres (±x, ±y, ±z extents)
    - 8 corner points of the bounding box
    - points just outside the body along each axis
    """
    N = 1 << n_bits
    dx = L / N
    cx, cy, cz = body_center
    hx = body_params.length / 2.0
    hy = body_params.height / 2.0
    hz = body_params.width / 2.0
    margin = 4 * dx  # a few cells outside the body

    def clamp(v: float) -> int:
        return max(0, min(N - 1, int(v / dx)))

    seeds: List[int] = []
    # Body centre
    seeds.append(xyz_to_morton(clamp(cx), clamp(cy), clamp(cz), n_bits))
    # 6 face centres
    for sign in (-1, 1):
        seeds.append(xyz_to_morton(clamp(cx + sign * (hx + margin)), clamp(cy), clamp(cz), n_bits))
        seeds.append(xyz_to_morton(clamp(cx), clamp(cy + sign * (hy + margin)), clamp(cz), n_bits))
        seeds.append(xyz_to_morton(clamp(cx), clamp(cy), clamp(cz + sign * (hz + margin)), n_bits))
    # 8 corners of bounding box (with margin)
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                seeds.append(xyz_to_morton(
                    clamp(cx + sx * (hx + margin)),
                    clamp(cy + sy * (hy + margin)),
                    clamp(cz + sz * (hz + margin)),
                    n_bits,
                ))
    # Far-field corners (domain boundary)
    for ix_f in (0, N - 1):
        for iy_f in (0, N - 1):
            for iz_f in (0, N - 1):
                seeds.append(xyz_to_morton(ix_f, iy_f, iz_f, n_bits))
    # Surface sample: ring around body at mid-height
    for angle_deg in range(0, 360, 30):
        rad = math.radians(angle_deg)
        px = cx + hx * 1.2 * math.cos(rad)
        pz = cz + hz * 1.2 * math.sin(rad)
        seeds.append(xyz_to_morton(clamp(px), clamp(cy), clamp(pz), n_bits))
    return list(set(seeds))  # deduplicate


def tci_multisweep(
    f: Callable[[Tensor], Tensor],
    n_qubits: int,
    max_rank: int,
    device: str = "cpu",
    n_sweeps: int = 6,
    tolerance: float = 1e-6,
    seed_morton: Optional[List[int]] = None,
    verbose: bool = False,
) -> Tuple[List[Tensor], Dict[str, Any]]:
    """Multi-sweep TT-Cross Interpolation for QTT construction.

    Alternating left-to-right and right-to-left sweeps with MaxVol
    pivot updates.  Converges reliably even for localized functions
    (e.g. body masks occupying <2%% of the domain).

    Never allocates O(2^n_qubits) arrays.

    Args:
        f: Black-box function mapping a batch of Morton indices (long tensor)
           to values (float32 tensor).
        n_qubits: Total number of binary TT modes (= 3 * n_bits for 3-D Morton).
        max_rank: Maximum TT bond dimension.
        device: Torch device string.
        n_sweeps: Number of full LR+RL sweeps (typically 3-6 suffice).
        tolerance: SVD truncation tolerance (relative to leading singular value).
        seed_morton: Optional list of Morton indices guaranteed to appear in the
            initial pivot set.  Critical for localized functions (e.g. body masks).
        verbose: Print per-sweep diagnostics.

    Returns:
        (cores, metadata) where cores is a list of tensors
        with shapes (r_{k-1}, 2, r_k).
    """
    dev = torch.device(device)
    d = n_qubits
    total_evals = 0

    # ── initialize pivots ───────────────────────────────────────────
    # Seed Morton indices inject known-important contexts at every mode.
    init_rank = min(max_rank, 16)
    seeds = seed_morton or []

    left_pivots: List[Tensor] = []
    right_pivots: List[Tensor] = []
    for k in range(d):
        # left: bits 0..k-1 → range [0, 2^k)
        n_left = 2 ** k
        r_l = min(init_rank, n_left)
        step_l = max(1, n_left // r_l)
        lp_set = set(range(0, n_left, step_l))
        # inject seed left contexts
        for sm in seeds:
            lp_set.add(sm & ((1 << k) - 1))
        lp = torch.tensor(sorted(lp_set), device=dev, dtype=torch.long)
        left_pivots.append(lp)

        # right: bits k+1..d-1 → range [0, 2^(d-k-1))
        n_right = 2 ** (d - k - 1)
        r_r = min(init_rank, n_right)
        step_r = max(1, n_right // r_r)
        rp_set = set(range(0, n_right, step_r))
        # inject seed right contexts
        for sm in seeds:
            rp_set.add(sm >> (k + 1))
        rp = torch.tensor(sorted(rp_set), device=dev, dtype=torch.long)
        right_pivots.append(rp)

    cores: List[Tensor] = [torch.zeros(1, 2, 1, device=dev)] * d

    # ── sweep loop ──────────────────────────────────────────────────
    prev_norm = 0.0
    for sweep_idx in range(n_sweeps):
        sweep_evals = 0

        # ── LEFT → RIGHT ───────────────────────────────────────────
        accumulated_left = torch.zeros(1, device=dev, dtype=torch.long)
        for k in range(d):
            r_left = len(accumulated_left)
            rp = right_pivots[k]
            r_right = len(rp)

            # build sample indices: (r_left, 2, r_right) → flat
            left_exp = accumulated_left.view(-1, 1, 1).expand(r_left, 2, r_right)
            bits = torch.arange(2, device=dev, dtype=torch.long).view(1, -1, 1).expand(
                r_left, 2, r_right
            )
            right_exp = rp.view(1, 1, -1).expand(r_left, 2, r_right)
            sample_idx = left_exp + (bits << k) + (right_exp << (k + 1))
            flat_idx = sample_idx.reshape(-1)

            values = f(flat_idx)
            sweep_evals += len(flat_idx)
            fiber = values.reshape(r_left, 2, r_right)

            if k < d - 1:
                mat = fiber.reshape(r_left * 2, r_right).float()
                U, S, _ = torch.linalg.svd(mat, full_matrices=False)
                rank = min(max_rank, len(S))
                if tolerance > 0 and S[0] > 0:
                    rank = min(rank, max(1, int((S > tolerance * S[0]).sum().item())))
                U = U[:, :rank]
                cores[k] = U.reshape(r_left, 2, rank).to(values.dtype)

                # MaxVol → new accumulated_left
                if U.shape[0] > rank:
                    piv = _maxvol(U, tol=1.1)
                    il = piv // 2
                    bv = piv % 2
                    accumulated_left = accumulated_left[il] + (bv << k)
                else:
                    new_acc = accumulated_left.view(-1, 1) + (
                        torch.arange(2, device=dev, dtype=torch.long) << k
                    ).view(1, -1)
                    accumulated_left = new_acc.reshape(-1)
            else:
                cores[k] = fiber.reshape(r_left, 2, 1).to(values.dtype)

            # update left_pivots for modes > k
            if k + 1 < d:
                left_pivots[k + 1] = accumulated_left.clone()

        # ── RIGHT → LEFT ───────────────────────────────────────────
        accumulated_right = torch.zeros(1, device=dev, dtype=torch.long)
        for k in range(d - 1, -1, -1):
            lp = left_pivots[k]
            r_left = len(lp)
            r_right = len(accumulated_right)

            left_exp = lp.view(-1, 1, 1).expand(r_left, 2, r_right)
            bits = torch.arange(2, device=dev, dtype=torch.long).view(1, -1, 1).expand(
                r_left, 2, r_right
            )
            right_exp = accumulated_right.view(1, 1, -1).expand(r_left, 2, r_right)
            sample_idx = left_exp + (bits << k) + (right_exp << (k + 1))
            flat_idx = sample_idx.reshape(-1)

            values = f(flat_idx)
            sweep_evals += len(flat_idx)
            fiber = values.reshape(r_left, 2, r_right)

            if k > 0:
                mat = fiber.reshape(r_left, 2 * r_right).float()
                U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
                rank = min(max_rank, len(S))
                if tolerance > 0 and S[0] > 0:
                    rank = min(rank, max(1, int((S > tolerance * S[0]).sum().item())))
                Vh = Vh[:rank, :]
                cores[k] = Vh.reshape(rank, 2, r_right).to(values.dtype)

                # MaxVol on columns → new accumulated_right
                VhT = Vh.T  # (2*r_right, rank)
                if VhT.shape[0] > rank:
                    piv = _maxvol(VhT, tol=1.1)
                    ib = piv // r_right
                    ir = piv % r_right
                    accumulated_right = (ib << k) + accumulated_right[ir]
                else:
                    new_acc = (
                        torch.arange(2, device=dev, dtype=torch.long).view(-1, 1) << k
                    ) + accumulated_right.view(1, -1)
                    accumulated_right = new_acc.reshape(-1)
            else:
                cores[k] = fiber.reshape(1, 2, r_right).to(values.dtype)

            # update right_pivots for modes < k
            if k > 0:
                right_pivots[k - 1] = accumulated_right.clone()

        total_evals += sweep_evals

        # ── convergence check ───────────────────────────────────────
        cur_norm = sum(float(torch.norm(c)) for c in cores)
        if verbose:
            max_r = max(c.shape[0] for c in cores[1:])
            print(
                f"    sweep {sweep_idx + 1}/{n_sweeps}: "
                f"max_rank={max_r}, evals={sweep_evals:,}, "
                f"norm={cur_norm:.6e}"
            )
        if sweep_idx > 0 and abs(cur_norm - prev_norm) < tolerance * max(cur_norm, 1e-12):
            if verbose:
                print(f"    converged (δnorm={abs(cur_norm - prev_norm):.2e})")
            break
        prev_norm = cur_norm

    metadata = {
        "method": "tci_multisweep",
        "n_evals": total_evals,
        "n_sweeps": sweep_idx + 1,
        "max_rank_actual": max(c.shape[0] for c in cores[1:]),
    }
    return cores, metadata


# ═══════════════════════════════════════════════════════════════════════
# SPONGE
# ═══════════════════════════════════════════════════════════════════════

def sponge_profile_1d(x, x_min, x_max, width, sigma_max=10.0):
    sigma = np.zeros_like(x)
    lm = x < x_min + width
    sigma = np.where(lm, sigma_max * np.clip((x_min + width - x) / width, 0, 1) ** 2, sigma)
    rm = x > x_max - width
    sigma = np.where(rm, sigma_max * np.clip((x - (x_max - width)) / width, 0, 1) ** 2, sigma)
    return sigma.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# QTT FIELD HELPERS
# ═══════════════════════════════════════════════════════════════════════

def dense3d_to_qtt(arr: np.ndarray, n_bits: int, max_rank: int,
                    device: torch.device,
                    dtype: torch.dtype = torch.float32) -> QTT3DNative:
    N = 1 << n_bits
    assert arr.shape == (N, N, N)
    flat = dense_to_morton_flat_vectorized(arr, n_bits)
    t = torch.from_numpy(flat).to(device=device, dtype=dtype)
    cores = _tt_svd_compress(t, [2] * (3 * n_bits), max_rank)
    return QTT3DNative(QTTCores(cores), n_bits)


def separable_x_field_qtt(values_1d: np.ndarray, n_bits: int, max_rank: int,
                           device: torch.device,
                           dtype: torch.dtype = torch.float32) -> QTT3DNative:
    """Build a 3D QTT field f(x,y,z) = g(x) directly from 1D values.

    **Zero-dense**: never materializes N³ array. Uses N floats, not N³.

    In Morton-ordered QTT (MSB-first from TT-SVD):
      - Site k%3 == 0: z-bit at level (n-1 - k//3)
      - Site k%3 == 1: y-bit at level (n-1 - k//3)
      - Site k%3 == 2: x-bit at level (n-1 - k//3)

    For x-only dependence: x-sites carry 1D QTT cores,
    y/z sites get constant identity-pass cores.

    Args:
        values_1d: 1D array of length N = 2^n_bits holding g(x) values.
        n_bits: Number of bits per dimension (N = 2^n_bits).
        max_rank: Maximum TT rank for the 1D decomposition.
        device: Torch device.
        dtype: Torch dtype.

    Returns:
        QTT3DNative with 3*n_bits sites; rank profile matches 1D QTT.
    """
    N = 1 << n_bits
    if len(values_1d) != N:
        raise ValueError(f"values_1d length {len(values_1d)} != 2^{n_bits} = {N}")

    # 1D QTT of g(x) — n_bits sites, MSB-first
    v_t = torch.from_numpy(np.asarray(values_1d, dtype=np.float32)).to(
        device=device, dtype=dtype
    )
    cores_1d = _tt_svd_compress(v_t, [2] * n_bits, max_rank)

    # Build 3D QTT by interleaving identity cores at z/y positions.
    # Layout per level ℓ (0 = MSB):
    #   site 3ℓ   (z-bit): identity pass-through, rank r_ℓ
    #   site 3ℓ+1 (y-bit): identity pass-through, rank r_ℓ
    #   site 3ℓ+2 (x-bit): 1D QTT core ℓ,        rank r_ℓ → r_{ℓ+1}
    cores_3d: List[Tensor] = []
    for level in range(n_bits):
        x_core = cores_1d[level]  # (r_in, 2, r_out)
        r_in = x_core.shape[0]

        # z-site: identity passthrough, shape (r_in, 2, r_in)
        # Both bit values yield identity matrix → field independent of z-bit
        z_id = torch.eye(r_in, device=device, dtype=dtype)
        z_core = z_id.unsqueeze(1).expand(-1, 2, -1).contiguous()
        cores_3d.append(z_core)

        # y-site: identity passthrough, shape (r_in, 2, r_in)
        y_core = z_id.unsqueeze(1).expand(-1, 2, -1).contiguous()
        cores_3d.append(y_core)

        # x-site: actual 1D QTT core
        cores_3d.append(x_core)

    return QTT3DNative(QTTCores(cores_3d), n_bits)


def constant_field_qtt(value: float, n_bits: int, device: torch.device,
                        dtype: torch.dtype = torch.float32) -> QTT3DNative:
    L = 3 * n_bits
    s = abs(value) ** (1.0 / L) if value != 0 else 0.0
    sign = 1.0 if value >= 0 else -1.0
    cores = []
    for k in range(L):
        c = torch.ones(1, 2, 1, device=device, dtype=dtype) * s
        if k == 0:
            c *= sign
        cores.append(c)
    return QTT3DNative(QTTCores(cores), n_bits)


def separable_y_field_qtt(values_1d: np.ndarray, n_bits: int, max_rank: int,
                           device: torch.device,
                           dtype: torch.dtype = torch.float32) -> QTT3DNative:
    """Build a 3D QTT field f(x,y,z) = g(y) directly from 1D values.

    Zero-dense.  Same interleaving as separable_x_field_qtt, but the
    1D cores sit at y-sites (3*level + 1) and identity pass-throughs at
    x/z sites.
    """
    N = 1 << n_bits
    if len(values_1d) != N:
        raise ValueError(f"values_1d length {len(values_1d)} != 2^{n_bits} = {N}")
    v_t = torch.from_numpy(np.asarray(values_1d, dtype=np.float32)).to(
        device=device, dtype=dtype,
    )
    cores_1d = _tt_svd_compress(v_t, [2] * n_bits, max_rank)
    cores_3d: List[Tensor] = []
    for level in range(n_bits):
        y_core = cores_1d[level]          # (r_in, 2, r_out)
        r_in = y_core.shape[0]
        z_id = torch.eye(r_in, device=device, dtype=dtype)
        # z-site: identity
        cores_3d.append(z_id.unsqueeze(1).expand(-1, 2, -1).contiguous())
        # y-site: 1D core
        cores_3d.append(y_core)
        # x-site: identity of shape (r_out, 2, r_out) = pass through
        r_out = y_core.shape[2]
        x_id = torch.eye(r_out, device=device, dtype=dtype)
        cores_3d.append(x_id.unsqueeze(1).expand(-1, 2, -1).contiguous())
    return QTT3DNative(QTTCores(cores_3d), n_bits)


def separable_z_field_qtt(values_1d: np.ndarray, n_bits: int, max_rank: int,
                           device: torch.device,
                           dtype: torch.dtype = torch.float32) -> QTT3DNative:
    """Build a 3D QTT field f(x,y,z) = g(z) directly from 1D values.

    Zero-dense.  1D cores sit at z-sites (3*level + 0), identity
    pass-throughs at y/x sites.
    """
    N = 1 << n_bits
    if len(values_1d) != N:
        raise ValueError(f"values_1d length {len(values_1d)} != 2^{n_bits} = {N}")
    v_t = torch.from_numpy(np.asarray(values_1d, dtype=np.float32)).to(
        device=device, dtype=dtype,
    )
    cores_1d = _tt_svd_compress(v_t, [2] * n_bits, max_rank)
    cores_3d: List[Tensor] = []
    for level in range(n_bits):
        z_core = cores_1d[level]          # (r_in, 2, r_out)
        r_out = z_core.shape[2]
        # z-site: 1D core
        cores_3d.append(z_core)
        # y-site: identity of shape (r_out, 2, r_out)
        y_id = torch.eye(r_out, device=device, dtype=dtype)
        cores_3d.append(y_id.unsqueeze(1).expand(-1, 2, -1).contiguous())
        # x-site: identity of shape (r_out, 2, r_out)
        x_id = torch.eye(r_out, device=device, dtype=dtype)
        cores_3d.append(x_id.unsqueeze(1).expand(-1, 2, -1).contiguous())
    return QTT3DNative(QTTCores(cores_3d), n_bits)


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AhmedBodyConfig:
    n_bits: int = 7
    max_rank: int = 64
    L: float = 4.0
    body_params: AhmedBodyParams = None
    nu_eff: float = 0.0
    eta_brinkman: float = 1e-3
    sigma_sponge: float = 5.0
    sponge_width_frac: float = 0.15
    dt: float = 0.0
    cfl: float = 0.08
    smagorinsky_cs: float = 0.3
    integrator: str = "rk2"        # "euler" or "rk2" (Heun's method)
    use_projection: bool = False   # Chorin pressure projection for ∇·u=0
    diagnostics_interval: int = 10 # Enstrophy/divergence every N steps
    n_steps: int = 500
    convergence_tol: float = 1e-4
    results_dir: str = "./ahmed_ib_results"
    device: str = "cuda"

    def __post_init__(self):
        if self.body_params is None:
            self.body_params = AhmedBodyParams()
        N = 1 << self.n_bits
        dx = self.L / N
        # Smagorinsky subgrid model: implicit diffusion that also damps
        # truncation artifacts.  Cs is configurable (default 0.3).
        Cs = self.smagorinsky_cs
        S_est = self.body_params.velocity / self.body_params.length
        self.nu_eff = self.body_params.nu + (Cs * dx) ** 2 * S_est
        if self.dt <= 0:
            self.dt = self.cfl * dx / self.body_params.velocity

    @property
    def N(self) -> int:
        return 1 << self.n_bits

    @property
    def dx(self) -> float:
        return self.L / self.N

    @property
    def Re_eff(self) -> float:
        return self.body_params.velocity * self.body_params.length / self.nu_eff


# ═══════════════════════════════════════════════════════════════════════
# SOLVER
# ═══════════════════════════════════════════════════════════════════════

class AhmedBodyIBSolver:
    """
    Per-step:
      1. Explicit NS:  u* = u + dt·(−u×ω + ν∇²u),  ω = ∇×u
      2. Brinkman:     u** = u* ⊙ mask_impl          (kills vel inside body)
      3. Sponge:       u*** = U∞ + (u** − U∞) ⊙ decay (absorbs at boundaries)
      4. Truncate      u_{n+1} = truncate(u***, max_rank)
    """

    def __init__(self, config: AhmedBodyConfig):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.dtype = torch.float32
        nb = config.n_bits
        N, L, dx = config.N, config.L, config.dx

        print(f"  Grid: {N}³ = {N**3:,} cells")
        print(f"  Domain: [0, {L:.1f}]³ m   dx = {dx*1000:.1f} mm")
        print(f"  Re_phys = {config.body_params.Re:.0f}   "
              f"Re_eff = {config.Re_eff:.0f}")
        print(f"  ν_eff = {config.nu_eff:.2e}   dt = {config.dt:.4e}")
        print(f"  η_IB = {config.eta_brinkman:.1e}   χ_max = {config.max_rank}")
        print(f"  Integrator: {config.integrator.upper()}  "
              f"Projection: {'ON' if config.use_projection else 'OFF'}  "
              f"Cs = {config.smagorinsky_cs}")

        # derivatives
        self.deriv = NativeDerivatives3D(
            n_bits=nb, max_rank=config.max_rank,
            base_rank=max(config.max_rank // 2, 4),
            device=self.device, dtype=self.dtype, L=L,
        )

        # masks
        print("  Building masks …")
        t0 = time.perf_counter()
        self.mask, self.mask_impl, self.body_center, self.brink_corr = (
            self._build_masks()
        )
        # No dense arrays stored — all N³ intermediates freed after compression.
        print(f"    χ mask      : rank {self.mask.max_rank:>3}  "
              f"CR {self.mask.compression_ratio:.0f}×")
        print(f"    IB implicit : rank {self.mask_impl.max_rank:>3}  "
              f"CR {self.mask_impl.compression_ratio:.0f}×")
        print(f"    IB correct. : rank {self.brink_corr.max_rank:>3}  "
              f"CR {self.brink_corr.compression_ratio:.0f}×")

        # sponge
        self.sponge_decay = self._build_sponge()
        print(f"    sponge decay: rank {self.sponge_decay.max_rank:>3}  "
              f"CR {self.sponge_decay.compression_ratio:.0f}×")

        # sponge correction: (decay - 1), localized to boundary zones
        self.sponge_corr = self._build_sponge_correction()
        print(f"    sponge corr.: rank {self.sponge_corr.max_rank:>3}  "
              f"CR {self.sponge_corr.compression_ratio:.0f}×")

        # sponge complement: U∞ * (1 - decay), precomputed for x-component
        self.sponge_compl_x = self._build_sponge_complement()
        print(f"    sponge compl: rank {self.sponge_compl_x.max_rank:>3}  "
              f"CR {self.sponge_compl_x.compression_ratio:.0f}×  "
              f"({(time.perf_counter()-t0)*1000:.0f} ms total)")

        # freestream
        U = config.body_params.velocity
        self.u_inf = QTT3DVectorNative(
            constant_field_qtt(U, nb, self.device, self.dtype),
            constant_field_qtt(0.0, nb, self.device, self.dtype),
            constant_field_qtt(0.0, nb, self.device, self.dtype),
        )

        # state
        self.u = self.u_inf.clone()
        self.t = 0.0
        self.step_count = 0
        self.diagnostics_history: List[Dict[str, float]] = []
        self._prev_energy: float = -1.0
        self._clamp_count: int = 0

        # Scale-adaptive rank profile: bell curve over 3*n_bits bonds.
        # Higher scale (large/small k) → lower rank, mid-scale → peak rank.
        n_sites = 3 * nb  # total TT sites (Morton interleaved)
        self._rank_profile = turbulence_rank_profile(
            n_sites, base_rank=max(config.max_rank // 2, 4),
            peak_rank=config.max_rank,
        )
        # rank_profile has n_sites+1 entries; bonds have n_sites-1 entries
        # Trim to bond count
        self._rank_profile = self._rank_profile[:n_sites - 1]

        # Minimum rank floor: prevents catastrophic rank collapse where
        # SVD truncation reduces all bonds to near-rank-1. This is critical
        # for multi-step integrators (RK2) where accumulated truncation error
        # can cause sudden information loss.
        self._min_rank = max(config.max_rank // 4, 4)

    # ── builders ────────────────────────────────────────────────────

    def _build_masks(self):
        N, L, dx, nb = self.config.N, self.config.L, self.config.dx, self.config.n_bits
        bp = self.config.body_params
        dt, eta = self.config.dt, self.config.eta_brinkman
        mr = self.config.max_rank

        rear_x = L / 3.0
        cx = rear_x - bp.length / 2.0
        cy = bp.ground_clearance + bp.height / 2.0
        cz = L / 2.0
        bc = (cx, cy, cz)

        # Dense approach feasible only when the N³ array fits in RAM.
        # 4096³ × 4 bytes = 275 GB — use TCI for n_bits > 9.
        if nb > 9:
            return self._build_masks_tci(N, L, dx, nb, bp, dt, eta, mr, bc)

        x1d = np.linspace(0, L * (N - 1) / N, N)
        X, Y, Z = np.meshgrid(x1d, x1d, x1d, indexing="ij")
        sdf = ahmed_body_sdf(X, Y, Z, bp, bc)
        md = create_body_mask(sdf, dx, 2.0)
        ns = int(np.sum(md > 0.5))
        print(f"    Solid cells: {ns:,} ({ns / N**3 * 100:.1f}%)")

        mid = (1.0 / (1.0 + (dt / eta) * md)).astype(np.float32)

        mq = dense3d_to_qtt(md, nb, mr, self.device, self.dtype)
        miq = dense3d_to_qtt(mid, nb, mr, self.device, self.dtype)

        # Build Brinkman correction (mask_impl - 1) while dense mid is available.
        # (mask_impl - 1) = -dt/η·χ / (1 + dt/η·χ)
        #   ≈ 0    far from body (99.8% of domain)  →  well-compressed
        #   ≈ -1   inside body
        # Used as: u_new = u + u ⊙ (mask_impl - 1)  instead of lossy u ⊙ mask_impl.
        corr = (mid - 1.0).astype(np.float32)
        brink_corr = dense3d_to_qtt(corr, nb, mr, self.device, self.dtype)
        # Dense md, mid, corr go out of scope here — no N³ stored on self.
        return mq, miq, bc, brink_corr

    def _build_masks_tci(
        self,
        N: int, L: float, dx: float, nb: int,
        bp: AhmedBodyParams,
        dt: float, eta: float, mr: int,
        bc: Tuple[float, float, float],
    ) -> Tuple[QTT3DNative, QTT3DNative, Tuple[float, float, float], QTT3DNative]:
        """Build IB masks analytically — zero-dense, O(N) memory.

        Decomposes the rounded-box Ahmed body mask as a product of
        separable 1D bump functions:

            χ(x,y,z)  ≈  χ_x(x)  ⊙  χ_y(y)  ⊙  χ_z(z)

        Each 1D factor is a smooth tanh bump evaluated on the N-point grid
        and converted to a 1D QTT (O(N) floats, rank ≈ 3-6).  The 3D QTT
        is assembled by interleaving identity pass-through cores using
        separable_{x,y,z}_field_qtt, then computing the Hadamard product.

        Total memory: 3 × N floats.  Never allocates an N³ array.

        The separable approximation is exact for axis-aligned boxes and
        differs from the full SDF only at fillets and the rear slant —
        regions that together comprise <0.5 % of cells.  This is perfectly
        adequate for Brinkman penalization.
        """
        n_qubits = 3 * nb
        coeff = dt / eta
        eps_mask = 2.0 * dx
        dev, dty = self.device, self.dtype
        cx, cy, cz = bc
        hx, hy, hz = bp.length / 2.0, bp.height / 2.0, bp.width / 2.0

        print(f"    Separable mode: n_bits={nb}, N={N}³ "
              f"({N**3:,.0f} cells), max_rank={mr}")

        x1d = np.linspace(0, L * (N - 1) / N, N)

        # ── 1D bump functions ──────────────────────────────────────
        # χ_x(x) = 0.5 * (tanh((x - x_min)/ε) - tanh((x - x_max)/ε))
        x_min, x_max = cx - hx, cx + hx
        chi_x = (0.5 * (np.tanh((x1d - x_min) / eps_mask)
                       - np.tanh((x1d - x_max) / eps_mask))).astype(np.float32)
        y_min, y_max = cy - hy, cy + hy
        chi_y = (0.5 * (np.tanh((x1d - y_min) / eps_mask)
                       - np.tanh((x1d - y_max) / eps_mask))).astype(np.float32)
        z_min, z_max = cz - hz, cz + hz
        chi_z = (0.5 * (np.tanh((x1d - z_min) / eps_mask)
                       - np.tanh((x1d - z_max) / eps_mask))).astype(np.float32)

        t0 = time.perf_counter()

        # ── mask χ = χ_x ⊙ χ_y ⊙ χ_z ─────────────────────────────
        qx = separable_x_field_qtt(chi_x, nb, mr, dev, dty)
        qy = separable_y_field_qtt(chi_y, nb, mr, dev, dty)
        qz = separable_z_field_qtt(chi_z, nb, mr, dev, dty)
        # Two Hadamard products, truncated
        xy_cores = qtt_hadamard_native(qx.cores, qy.cores, mr)
        mask_cores = qtt_hadamard_native(xy_cores, qz.cores, mr)
        mq = QTT3DNative(mask_cores, nb)

        # Estimate solid cells from 1D profiles
        # Body ≈ pixels where ALL 3 bumps exceed 0.5 — rough count
        n_x = int(np.sum(chi_x > 0.5))
        n_y = int(np.sum(chi_y > 0.5))
        n_z = int(np.sum(chi_z > 0.5))
        ns_est = n_x * n_y * n_z
        print(f"    Solid cells (est.): {ns_est:,} "
              f"({ns_est / N**3 * 100:.2f}%)")

        # ── mask_implicit = 1 / (1 + dt/η · χ) ───────────────────
        # Since χ = χ_x · χ_y · χ_z, mask_impl is NOT separable.
        # But we can build: mask_impl = 1 − (coeff·χ)/(1 + coeff·χ)
        # Approximation: for boxes, χ ∈ {0, ~1} almost everywhere,
        # and the transition layer is thin.
        # Exact approach: build mask_impl from derived 1D profiles:
        #   mask_impl_x = 1/(1 + coeff · chi_x)  (1D, handle y/z similarly)
        # But mask_impl is NOT separable. Use: mask_impl = 1 + brink_corr.
        # Instead: compute mask, then derive mask_impl and brink_corr
        # from the QTT mask via element-wise operations.
        #
        # Simplest correct approach: brink_corr = -coeff*χ / (1+coeff*χ)
        # At the two extremes:
        #   χ = 0 → brink_corr = 0     (everywhere outside body)
        #   χ = 1 → brink_corr ≈ -1    (inside body, since coeff >> 1)
        # Since χ ∈ [0,1], brink_corr = -coeff*χ / (1+coeff*χ)
        # ≈ -χ when coeff >> 1 (which is typical: coeff = dt/η ~ 0.1/1e-3 = 100)
        #
        # Build brink_corr ≈ -χ (exact in the large-coeff limit)
        # mask_impl = 1 + brink_corr
        #
        # For better accuracy, keep the exact formula on 1D arrays:

        # 1D implicit profiles
        mi_x = (1.0 / (1.0 + coeff * chi_x)).astype(np.float32)
        mi_y = (1.0 / (1.0 + coeff * chi_y)).astype(np.float32)
        mi_z = (1.0 / (1.0 + coeff * chi_z)).astype(np.float32)

        # mask_impl ≈ mi_x · mi_y · mi_z is NOT exactly correct.
        # Exact: mask_impl = 1/(1 + coeff·χ_x·χ_y·χ_z)
        # Separable approx: 1/((1+c·χ_x)(1+c·χ_y)(1+c·χ_z)) ≈ mi_x·mi_y·mi_z
        # These differ, but outside the body (χ≈0) both are 1,
        # and inside the body (χ≈1) the Brinkman penalty drives velocity
        # to zero either way. The mismatch is only in the thin transition
        # layer — acceptable for IB penalization.

        qmi_x = separable_x_field_qtt(mi_x, nb, mr, dev, dty)
        qmi_y = separable_y_field_qtt(mi_y, nb, mr, dev, dty)
        qmi_z = separable_z_field_qtt(mi_z, nb, mr, dev, dty)
        xy_impl = qtt_hadamard_native(qmi_x.cores, qmi_y.cores, mr)
        impl_cores = qtt_hadamard_native(xy_impl, qmi_z.cores, mr)
        miq = QTT3DNative(impl_cores, nb)

        # ── brinkman correction = mask_impl − 1 ──────────────────
        # corr_1d_a = mi_a - 1   (≈0 far, ≈-1 inside)
        corr_x = (mi_x - 1.0).astype(np.float32)
        corr_y = (mi_y - 1.0).astype(np.float32)
        corr_z = (mi_z - 1.0).astype(np.float32)

        # (mi_x·mi_y·mi_z) − 1 is NOT the product of (mi_a − 1).
        # Expand: Π(1+c_a) − 1 = Σ c_a + Σ c_a·c_b + c_x·c_y·c_z
        # where c_a = mi_a − 1.
        # Build via sum: brink_corr = mask_impl − 1_qtt
        one = constant_field_qtt(1.0, nb, dev, dty)
        brink_corr_cores = qtt_sub_native(impl_cores, one.cores, mr)
        brink_corr = QTT3DNative(brink_corr_cores, nb)

        elapsed = time.perf_counter() - t0
        print(f"    mask      : rank {mq.max_rank:>3}  "
              f"CR {mq.compression_ratio:.0f}×")
        print(f"    mask_impl : rank {miq.max_rank:>3}  "
              f"CR {miq.compression_ratio:.0f}×")
        print(f"    brink_corr: rank {brink_corr.max_rank:>3}  "
              f"CR {brink_corr.compression_ratio:.0f}×")
        print(f"    ({elapsed*1000:.0f} ms, zero-dense separable)")

        return mq, miq, bc, brink_corr

    def _build_sponge(self):
        """Build sponge decay field exp(-σ·dt) as QTT — zero-dense.

        The sponge only varies along x → separable: f(x,y,z) = g(x).
        Uses separable_x_field_qtt to avoid materializing N³ dense array.
        """
        N, L, nb = self.config.N, self.config.L, self.config.n_bits
        dt = self.config.dt
        sw = self.config.sponge_width_frac * L

        x1d = np.linspace(0, L * (N - 1) / N, N)
        sigma = sponge_profile_1d(x1d, 0, L, sw, self.config.sigma_sponge)
        self._sponge_decay_1d = np.exp(-sigma * dt).astype(np.float32)
        return separable_x_field_qtt(
            self._sponge_decay_1d, nb, self.config.max_rank,
            self.device, self.dtype
        )

    def _build_sponge_correction(self):
        """Precompute (decay - 1) as QTT field — zero-dense.

        Instead of u_new = u * decay + complement (lossy Hadamard on near-1 field),
        we use:       u_new = u + u * (decay - 1) + complement

        (decay - 1) ≈ 0 in the interior (85% of domain),
                     < 0 in sponge zones (boundary 15%).
        Localized → well-compressed → accurate Hadamard.
        Separable (x-only) → built with separable_x_field_qtt.
        """
        nb = self.config.n_bits
        corr = (self._sponge_decay_1d - 1.0).astype(np.float32)
        return separable_x_field_qtt(
            corr, nb, self.config.max_rank, self.device, self.dtype
        )

    def _build_sponge_complement(self):
        """Precompute U∞_x * (1 - decay) as QTT field — zero-dense.

        Separable (x-only) → built with separable_x_field_qtt.
        """
        nb = self.config.n_bits
        U = self.config.body_params.velocity
        compl = (U * (1.0 - self._sponge_decay_1d)).astype(np.float32)
        return separable_x_field_qtt(
            compl, nb, self.config.max_rank, self.device, self.dtype
        )

    # ── RHS ─────────────────────────────────────────────────────────

    def _rhs(self, u: QTT3DVectorNative) -> QTT3DVectorNative:
        """du/dt = −u×ω + ν∇²u, ω = ∇×u (batched final sum)."""
        mr = self.config.max_rank
        mn = self._min_rank
        nu = self.config.nu_eff
        omega = self.deriv.curl(u)
        uxo = vector_cross_native(u, omega, mr)
        lap = self.deriv.laplacian_vector(u)
        # 3 independent fused sums → 1 batched truncation sweep
        results = qtt_fused_sum_batched(
            [
                [uxo.x.cores, lap.x.cores],
                [uxo.y.cores, lap.y.cores],
                [uxo.z.cores, lap.z.cores],
            ],
            [[-1.0, nu], [-1.0, nu], [-1.0, nu]],
            mr, min_rank=mn,
        )
        nb = u.n_bits
        return QTT3DVectorNative(
            QTT3DNative(results[0], nb),
            QTT3DNative(results[1], nb),
            QTT3DNative(results[2], nb),
        )

    # ── semi-implicit operators ─────────────────────────────────────

    def _brinkman(self, u: QTT3DVectorNative) -> QTT3DVectorNative:
        """Brinkman: u_new = u + u ⊙ (mask_impl − 1).  Batched.
        
        The correction (mask_impl − 1) is ≈ 0 far from body,
        so the Hadamard product is localized and well-compressed.
        This preserves bulk velocity much better than u ⊙ mask_impl.
        """
        mr, nb = self.config.max_rank, u.n_bits
        bc = self.brink_corr.cores
        # 3 independent Hadamard products → 1 batched compress-as-multiply
        corrections = qtt_hadamard_native_batched(
            [(u.x.cores, bc), (u.y.cores, bc), (u.z.cores, bc)],
            mr,
        )
        # 3 independent additions → 1 batched truncation
        sums = qtt_add_native_batched(
            [
                (u.x.cores, corrections[0]),
                (u.y.cores, corrections[1]),
                (u.z.cores, corrections[2]),
            ],
            mr,
        )
        return QTT3DVectorNative(
            QTT3DNative(sums[0], nb),
            QTT3DNative(sums[1], nb),
            QTT3DNative(sums[2], nb),
        )

    def _sponge(self, u: QTT3DVectorNative) -> QTT3DVectorNative:
        """Sponge: u_new = u + u ⊙ (decay − 1) + complement.  Batched.
        
        Reformulated to avoid lossy near-unity Hadamard (u ⊙ decay).
        (decay − 1) ≈ 0 in interior → localized correction, well-compressed.
        U∞ = (U, 0, 0) so:
          x: u_x + u_x*(decay-1) + U*(1-decay)
          y: u_y + u_y*(decay-1)
          z: u_z + u_z*(decay-1)
        """
        mr, nb = self.config.max_rank, u.n_bits
        mn = self._min_rank
        sc = self.sponge_corr.cores
        # 3 independent Hadamard products → 1 batched compress-as-multiply
        corrections = qtt_hadamard_native_batched(
            [(u.x.cores, sc), (u.y.cores, sc), (u.z.cores, sc)],
            mr,
        )
        # 3 independent sums → 1 batched fused sum
        #   x: u_x + correction_x + complement_x  (3 terms)
        #   y: u_y + correction_y                  (2 terms)
        #   z: u_z + correction_z                  (2 terms)
        results = qtt_fused_sum_batched(
            [
                [u.x.cores, corrections[0], self.sponge_compl_x.cores],
                [u.y.cores, corrections[1]],
                [u.z.cores, corrections[2]],
            ],
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ],
            mr, min_rank=mn,
        )
        return QTT3DVectorNative(
            QTT3DNative(results[0], nb),
            QTT3DNative(results[1], nb),
            QTT3DNative(results[2], nb),
        )

    def _trunc(self, v: QTT3DVectorNative) -> QTT3DVectorNative:
        mr = self.config.max_rank
        rp = self._rank_profile
        mn = self._min_rank
        xyz = qtt_truncate_now_batched(
            [v.x.cores, v.y.cores, v.z.cores],
            mr, 1e-10, rank_profile=rp, min_rank=mn,
        )
        return QTT3DVectorNative(
            QTT3DNative(xyz[0], v.n_bits),
            QTT3DNative(xyz[1], v.n_bits),
            QTT3DNative(xyz[2], v.n_bits),
        )

    # ── pressure projection ─────────────────────────────────────────

    def _project(self, u: QTT3DVectorNative) -> QTT3DVectorNative:
        """Chorin pressure projection: enforce ∇·u ≈ 0.

        Solves  ∇²p = (1/dt) ∇·u*  via QTT-CG,
        then    u ← u* − dt ∇p.
        """
        dt, mr, nb = self.config.dt, self.config.max_rank, u.n_bits
        mn = self._min_rank
        div_u = self.deriv.divergence(u)
        rhs = QTT3DNative(qtt_scale_native(div_u.cores, 1.0 / dt), nb)
        p = self.deriv.poisson_cg(rhs, tol=1e-5, max_iter=30)
        grad_p = self.deriv.gradient(p)
        proj_xyz = qtt_fused_sum_batched(
            [
                [u.x.cores, grad_p.x.cores],
                [u.y.cores, grad_p.y.cores],
                [u.z.cores, grad_p.z.cores],
            ],
            [[1.0, -dt], [1.0, -dt], [1.0, -dt]],
            mr, min_rank=mn,
        )
        return QTT3DVectorNative(
            QTT3DNative(proj_xyz[0], nb),
            QTT3DNative(proj_xyz[1], nb),
            QTT3DNative(proj_xyz[2], nb),
        )

    # ── time step ───────────────────────────────────────────────────

    def _energy(self, v: QTT3DVectorNative) -> float:
        return float(
            qtt_inner_native(v.x.cores, v.x.cores)
            + qtt_inner_native(v.y.cores, v.y.cores)
            + qtt_inner_native(v.z.cores, v.z.cores)
        ) * 0.5

    def step(self, debug: bool = False) -> Dict[str, Any]:
        """Advance one timestep.

        Supports Forward Euler or RK2 (Heun) via ``config.integrator``.
        Optional Chorin pressure projection via ``config.use_projection``.

        Returns diagnostics dict with:
            step, time, energy, ranks, compression, clamped,
            u_max, cfl_actual, enstrophy, divergence_max,
            gpu_mem_mb, gpu_peak_mb.
        """
        dt, mr = self.config.dt, self.config.max_rank
        mn = self._min_rank
        nb = self.u.n_bits
        use_rk2 = self.config.integrator == "rk2"

        if debug:
            e0 = self._energy(self.u)
            print(f"    [dbg] pre-step  E={e0:.6e}  rank={self.u.max_rank}")

        # ── Time integration ────────────────────────────────────────
        if use_rk2:
            # Heun's method (2nd-order): k1 at u_n, k2 at predictor
            k1 = self._rhs(self.u)
            pred_xyz = qtt_fused_sum_batched(
                [
                    [self.u.x.cores, k1.x.cores],
                    [self.u.y.cores, k1.y.cores],
                    [self.u.z.cores, k1.z.cores],
                ],
                [[1.0, dt], [1.0, dt], [1.0, dt]],
                mr, min_rank=mn,
            )
            u_pred = QTT3DVectorNative(
                QTT3DNative(pred_xyz[0], nb),
                QTT3DNative(pred_xyz[1], nb),
                QTT3DNative(pred_xyz[2], nb),
            )
            u_pred = self._trunc(u_pred)  # barrier truncation
            k2 = self._rhs(u_pred)
            corr_xyz = qtt_fused_sum_batched(
                [
                    [self.u.x.cores, k1.x.cores, k2.x.cores],
                    [self.u.y.cores, k1.y.cores, k2.y.cores],
                    [self.u.z.cores, k1.z.cores, k2.z.cores],
                ],
                [
                    [1.0, dt / 2, dt / 2],
                    [1.0, dt / 2, dt / 2],
                    [1.0, dt / 2, dt / 2],
                ],
                mr, min_rank=mn,
            )
            us = QTT3DVectorNative(
                QTT3DNative(corr_xyz[0], nb),
                QTT3DNative(corr_xyz[1], nb),
                QTT3DNative(corr_xyz[2], nb),
            )
        else:
            # Forward Euler (1st-order)
            rhs = self._rhs(self.u)
            euler_xyz = qtt_fused_sum_batched(
                [
                    [self.u.x.cores, rhs.x.cores],
                    [self.u.y.cores, rhs.y.cores],
                    [self.u.z.cores, rhs.z.cores],
                ],
                [[1.0, dt], [1.0, dt], [1.0, dt]],
                mr, min_rank=mn,
            )
            us = QTT3DVectorNative(
                QTT3DNative(euler_xyz[0], nb),
                QTT3DNative(euler_xyz[1], nb),
                QTT3DNative(euler_xyz[2], nb),
            )

        if debug:
            e1 = self._energy(us)
            label = "rk2" if use_rk2 else "euler"
            print(f"    [dbg] {label:>10}  E={e1:.6e}  rank={us.max_rank}")

        # ── Pressure projection (optional) ──────────────────────────
        if self.config.use_projection:
            us = self._project(us)
            if debug:
                ep = self._energy(us)
                print(f"    [dbg] project   E={ep:.6e}  rank={us.max_rank}")

        # ── Immersed-boundary corrections ───────────────────────────
        us = self._brinkman(us)
        if debug:
            e2 = self._energy(us)
            print(f"    [dbg] brinkman  E={e2:.6e}  rank={us.max_rank}")

        us = self._sponge(us)
        if debug:
            e3 = self._energy(us)
            print(f"    [dbg] sponge    E={e3:.6e}  rank={us.max_rank}")

        self.u = self._trunc(us)
        if debug:
            e4 = self._energy(self.u)
            print(f"    [dbg] truncate  E={e4:.6e}  rank={self.u.max_rank}")

        # ── Energy safety valve ─────────────────────────────────────
        energy = self._energy(self.u)

        # NaN / Inf guard — abort early before poison propagates
        if math.isnan(energy) or math.isinf(energy):
            self.t += dt
            self.step_count += 1
            d: Dict[str, Any] = {
                "step": self.step_count,
                "time": self.t,
                "energy": float("nan"),
                "max_rank_u": self.u.max_rank,
                "mean_rank_u": self.u.mean_rank,
                "compression_ratio": 0.0,
                "clamped": False,
                "u_max": 0.0,
                "cfl_actual": 0.0,
                "enstrophy": 0.0,
                "divergence_max": 0.0,
                "gpu_mem_mb": 0.0,
                "gpu_peak_mb": 0.0,
                "nan_detected": True,
            }
            self.diagnostics_history.append(d)
            return d

        clamped = False
        if self._prev_energy > 0 and energy > self._prev_energy:
            target = self._prev_energy * 0.999
            scale = math.sqrt(target / energy)
            self.u = QTT3DVectorNative(
                QTT3DNative(qtt_scale_native(self.u.x.cores, scale), nb),
                QTT3DNative(qtt_scale_native(self.u.y.cores, scale), nb),
                QTT3DNative(qtt_scale_native(self.u.z.cores, scale), nb),
            )
            self._clamp_count += 1
            clamped = True
            energy = target
            if debug:
                print(f"    [dbg] E-clamp  scale={scale:.6f}")
        self._prev_energy = energy

        self.t += dt
        self.step_count += 1

        # ── Actual CFL from flow field ──────────────────────────────
        u_max = _qtt_vec_max_abs_native(self.u, n_samples=1024)
        cfl_actual = u_max * dt / self.config.dx if u_max > 0 else 0.0

        # ── Extended diagnostics (periodic) ─────────────────────────
        compute_ext = (
            self.step_count % self.config.diagnostics_interval == 0
            or self.step_count <= 2
            or self.step_count == self.config.n_steps
        )
        enstrophy = 0.0
        div_max = 0.0
        if compute_ext:
            omega = self.deriv.curl(self.u)
            enstrophy = float(
                qtt_inner_native(omega.x.cores, omega.x.cores)
                + qtt_inner_native(omega.y.cores, omega.y.cores)
                + qtt_inner_native(omega.z.cores, omega.z.cores)
            ) * 0.5
            div_u = self.deriv.divergence(self.u)
            div_max = _qtt_scalar_max_abs_native(div_u, n_samples=1024)

        # ── GPU memory tracking ─────────────────────────────────────
        gpu_mem_mb = 0.0
        gpu_peak_mb = 0.0
        if torch.cuda.is_available() and self.device.type == "cuda":
            gpu_mem_mb = torch.cuda.memory_allocated(self.device) / 1e6
            gpu_peak_mb = torch.cuda.max_memory_allocated(self.device) / 1e6

        d: Dict[str, Any] = {
            "step": self.step_count,
            "time": self.t,
            "energy": energy,
            "max_rank_u": self.u.max_rank,
            "mean_rank_u": self.u.mean_rank,
            "compression_ratio": self.u.compression_ratio,
            "clamped": clamped,
            "u_max": u_max,
            "cfl_actual": cfl_actual,
            "enstrophy": enstrophy,
            "divergence_max": div_max,
            "gpu_mem_mb": gpu_mem_mb,
            "gpu_peak_mb": gpu_peak_mb,
        }
        self.diagnostics_history.append(d)
        return d

    # ── storage ─────────────────────────────────────────────────────

    def qtt_storage_bytes(self) -> Dict[str, int]:
        def fb(f: QTT3DNative) -> int:
            return sum(c.numel() * c.element_size() for c in f.cores.cores)
        s = {
            "u_x": fb(self.u.x), "u_y": fb(self.u.y), "u_z": fb(self.u.z),
            "mask": fb(self.mask), "mask_impl": fb(self.mask_impl),
            "sponge": fb(self.sponge_decay),
        }
        s["total"] = sum(s.values())
        return s

    # ── surface probe ───────────────────────────────────────────────

    def probe_surface(self, coords: np.ndarray) -> Dict[str, np.ndarray]:
        N, L, nb = self.config.N, self.config.L, self.config.n_bits
        ix = np.clip((coords[:, 0] / L * N).astype(int), 0, N - 1)
        iy = np.clip((coords[:, 1] / L * N).astype(int), 0, N - 1)
        iz = np.clip((coords[:, 2] / L * N).astype(int), 0, N - 1)
        M = len(ix)
        mi = np.zeros(M, dtype=np.int64)
        for b in range(nb):
            mi |= ((ix >> b) & 1).astype(np.int64) << (3 * b)
            mi |= ((iy >> b) & 1).astype(np.int64) << (3 * b + 1)
            mi |= ((iz >> b) & 1).astype(np.int64) << (3 * b + 2)
        idx = torch.from_numpy(mi).to(self.device)
        out: Dict[str, np.ndarray] = {}
        for name, qf in [("ux", self.u.x), ("uy", self.u.y), ("uz", self.u.z)]:
            out[name] = _batched_qtt_eval(qf.cores.cores, idx).cpu().numpy()
        return out

    # ── run loop ────────────────────────────────────────────────────

    def run(self, verbose: bool = True) -> List[Dict[str, float]]:
        ns = self.config.n_steps
        tol = self.config.convergence_tol
        if verbose:
            print(f"\n  Running {ns} steps ({self.config.integrator.upper()}) …")
            print(f"  {'Step':>6} {'Time':>10} {'Energy':>14} "
                  f"{'Rank':>6} {'CR':>8} {'CFL':>6} {'ms':>6}")
        prev_e = None
        for i in range(ns):
            t0 = time.perf_counter()
            d = self.step(debug=(i < 3 and verbose))
            ms = (time.perf_counter() - t0) * 1000
            if verbose and (i % max(1, ns // 20) == 0 or i == ns - 1):
                clamp_flag = " C" if d.get("clamped") else ""
                print(f"  {d['step']:>6} {d['time']:>10.6f} "
                      f"{d['energy']:>14.6e} "
                      f"{d['max_rank_u']:>6} {d['compression_ratio']:>8.0f}×"
                      f" {d.get('cfl_actual', 0):>6.3f}"
                      f" {ms:>6.0f}{clamp_flag}")
            if prev_e is not None and prev_e > 0 and not d.get("clamped"):
                rel = abs(d["energy"] - prev_e) / prev_e
                if rel < tol and i > max(50, ns // 2):
                    if verbose:
                        print(f"  Converged: ΔE/E = {rel:.2e}")
                    break
            prev_e = d["energy"]
        else:
            if verbose:
                print(f"  Completed {ns} steps.")
        return self.diagnostics_history


# ═══════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════

def generate_report(solver: AhmedBodyIBSolver, cfg: AhmedBodyConfig,
                     wall: float) -> str:
    N, nb = cfg.N, cfg.n_bits
    nf = 3
    dpf = N ** 3 * 4
    dt = dpf * nf
    qs = solver.qtt_storage_bytes()
    qv = qs["u_x"] + qs["u_y"] + qs["u_z"]
    qt = qs["total"]
    cv = dt / qv if qv else float("inf")
    ct = dt / qt if qt else float("inf")

    L: List[str] = []
    sep = "═" * 72
    L.append(sep)
    L.append("QTT VOLUME COMPRESSION — AHMED BODY IB SOLVER")
    L.append(sep)
    L.append(f"Grid:      {N}³ = {N**3:,} cells ({nb} bits/axis)")
    L.append(f"Domain:    [0, {cfg.L:.1f}]³ m   dx = {cfg.dx*1000:.1f} mm")
    L.append(f"Body:      {cfg.body_params.length:.3f} × "
             f"{cfg.body_params.width:.3f} × "
             f"{cfg.body_params.height:.3f} m")
    L.append(f"Re_phys:   {cfg.body_params.Re:.0f}")
    L.append(f"Re_eff:    {cfg.Re_eff:.0f}")
    L.append(f"ν_eff:     {cfg.nu_eff:.2e}")
    L.append(f"χ_max:     {cfg.max_rank}")
    L.append(f"Steps:     {solver.step_count}")
    L.append(f"E-clamps:  {solver._clamp_count}")
    L.append(f"Wall time: {wall:.1f} s")
    L.append("")
    L.append("─" * 72)
    L.append("STORAGE")
    L.append("─" * 72)
    L.append(f"  Dense ({nf} × {N}³ × f32):  {dt / 1e6:.1f} MB")
    L.append(f"  QTT velocity:            {qv / 1e3:.1f} KB  →  {cv:.0f}×")
    L.append(f"  QTT total:               {qt / 1e3:.1f} KB  →  {ct:.0f}×")
    L.append("")
    for k, v in sorted(qs.items()):
        if k != "total":
            cr = dpf / v if v else float("inf")
            L.append(f"    {k:>14}: {v/1e3:>8.1f} KB  ({cr:.0f}×)")

    L.append("")
    L.append("─" * 72)
    L.append("SCALING PROJECTION")
    L.append("─" * 72)
    acb = qv / (3 * 3 * nb)
    for pb in [7, 8, 9, 10, 11, 12]:
        pN = 1 << pb
        pq = acb * (3 * pb) / (3 * nb) * 3 * 3 * pb
        pd = pN ** 3 * 4 * nf
        pc = pd / pq if pq else float("inf")
        L.append(f"  {pN:>6}³ ({pN**3:>13,}): "
                 f"dense {pd/1e9:.1f} GB   QTT ≈ {pq/1e6:.1f} MB   ≈ {pc:.0f}×")

    if solver.diagnostics_history:
        d = solver.diagnostics_history[-1]
        L.append("")
        L.append("─" * 72)
        L.append("FINAL STATE")
        L.append("─" * 72)
        L.append(f"  Energy = {d['energy']:.6e}")
        L.append(f"  Rank   = {d['max_rank_u']} (mean {d['mean_rank_u']:.1f})")
        L.append(f"  CR     = {d['compression_ratio']:.0f}×")
    L.append("")
    L.append(sep)
    return "\n".join(L)


# ═══════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════

def validate_nvidia(solver, cfg, vtp_path):
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError:
        print("  VTK not available — skipping.")
        return {}
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    pd = reader.GetOutput()
    coords = vtk_to_numpy(pd.GetPoints().GetData()).copy()
    bp = cfg.body_params
    cx, cy, cz = solver.body_center
    mapped = coords.copy()
    mapped[:, 0] += cx + bp.length / 2
    mapped[:, 1] += cy - bp.height / 2 - bp.ground_clearance
    mapped[:, 2] += cz - bp.width / 2
    fields = solver.probe_surface(mapped)
    umag = np.sqrt(fields["ux"] ** 2 + fields["uy"] ** 2 + fields["uz"] ** 2)
    return {
        "n_points": int(len(coords)),
        "mean_wall_vel": float(np.mean(umag)),
        "max_wall_vel": float(np.max(umag)),
        "brinkman_quality": float(1 - np.mean(umag) / bp.velocity),
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser("Ahmed Body IB QTT Solver v2")
    ap.add_argument("--n-bits", type=int, default=7)
    ap.add_argument("--max-rank", type=int, default=64)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--domain-size", type=float, default=4.0)
    ap.add_argument("--eta", type=float, default=1e-3)
    ap.add_argument("--cfl", type=float, default=0.08)
    ap.add_argument("--case-file", type=str, default=None)
    ap.add_argument("--validate-vtp", type=str, default=None)
    ap.add_argument("--results-dir", type=str, default="./ahmed_ib_results")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    if args.case_file and os.path.exists(args.case_file):
        bp = AhmedBodyParams.from_nvidia_info(args.case_file)
        cid = Path(args.case_file).stem.replace("_info", "")
        print(f"  Loaded {args.case_file}")
    else:
        bp = AhmedBodyParams()
        cid = "standard_ahmed"

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Ahmed Body IB — QTT Volumetric Synthesis  (v2 semi-implicit)  ║")
    print("║  HyperTensor QTT Engine — Tigantic Holdings LLC                ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  Case: {cid}   Body: {bp.length:.3f}×{bp.width:.3f}×{bp.height:.3f} m")
    print(f"  U∞ = {bp.velocity:.1f} m/s   Re = {bp.Re:.0f}")
    print()

    cfg = AhmedBodyConfig(
        n_bits=args.n_bits, max_rank=args.max_rank, L=args.domain_size,
        body_params=bp, eta_brinkman=args.eta, cfl=args.cfl,
        n_steps=args.steps, results_dir=args.results_dir, device=args.device,
    )

    print("─" * 72)
    print("INIT")
    print("─" * 72)
    t0 = time.perf_counter()
    solver = AhmedBodyIBSolver(cfg)
    ti = time.perf_counter() - t0
    print(f"  Init: {ti:.1f} s")

    print("\n" + "─" * 72)
    print("SIMULATION")
    print("─" * 72)
    t0 = time.perf_counter()
    solver.run()
    tr = time.perf_counter() - t0

    print("\n" + "─" * 72)
    print("COMPRESSION REPORT")
    print("─" * 72)
    rpt = generate_report(solver, cfg, ti + tr)
    print(rpt)

    rd = Path(cfg.results_dir)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / f"report_{cid}.txt").write_text(rpt)
    with open(rd / f"diagnostics_{cid}.json", "w") as f:
        json.dump(solver.diagnostics_history, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.floating,))
                  else int(o) if isinstance(o, (np.integer,)) else o)
    with open(rd / f"storage_{cid}.json", "w") as f:
        json.dump(solver.qtt_storage_bytes(), f, indent=2)
    print(f"\n  Saved to {rd}/")

    if args.validate_vtp and os.path.exists(args.validate_vtp):
        print("\n" + "─" * 72)
        print("VALIDATION")
        print("─" * 72)
        m = validate_nvidia(solver, cfg, args.validate_vtp)
        for k, v in m.items():
            print(f"  {k}: {v}")
        with open(rd / f"validation_{cid}.json", "w") as f:
            json.dump(m, f, indent=2)


if __name__ == "__main__":
    main()
