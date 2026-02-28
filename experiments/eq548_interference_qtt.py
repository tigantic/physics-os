#!/usr/bin/env python3
"""
Rigorous QTT compression test for Eq. 5.48 (Badui dissertation).

The quadratic form I = |A V|² generates interference cross-terms when squaring
a coherent sum of partial waves.  This experiment proves that the resulting
intensity distribution—including all beat frequencies—remains low-rank in
QTT format.

Physics:
    A matrix: Wigner-D matrices D^j_{m,λ}(θ,φ) for spin-1/2 baryon system
              (half-integer j from 1/2 to 7/2, helicity λ = 1/2)
    V vector: Complex production amplitudes (16-state basis)
    Intensity: I(θ,φ) = |Σ_b A_b V_b|² — coherent quadratic form
    The cross-terms A_b V_b (A_c V_c)* create beat frequencies at L±L'

QTT proof:
    We decompose I(θ,φ) into QTT format via TT-SVD and measure:
    1. Bond dimensions at every bond (after tolerance-based truncation)
    2. Full SVD spectrum showing exponential singular value decay
    3. Compression ratio: dense storage / QTT storage
    4. Frobenius-norm reconstruction error

Hardware:  NVIDIA GeForce RTX 5070 Laptop GPU (CUDA)
Authority: Adams (2026), physics-os Platform V2.0.0
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor
from scipy.special import factorial

# ─── matplotlib setup (must precede pyplot import) ───────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
from matplotlib.lines import Line2D

# ─── JCP house style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.7",
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.3,
    "ytick.minor.width": 0.3,
    "lines.linewidth": 1.2,
    "lines.markersize": 5,
    "axes.grid": False,
})

# Okabe-Ito accessible palette
C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_RED = "#D55E00"
C_PURPLE = "#CC79A7"
C_CYAN = "#56B4E9"
C_BLACK = "#000000"
C_GREY = "#999999"

SINGLE_COL = (3.5, 2.8)
DOUBLE_COL = (7.0, 3.5)

OUTPUT_DIR = Path(__file__).parent.parent / "paper" / "figures"


# ════════════════════════════════════════════════════════════════════════════════
# WIGNER-D MATRIX COMPUTATION
# ════════════════════════════════════════════════════════════════════════════════

def wigner_small_d(
    j: float,
    m: float,
    mp: float,
    beta: np.ndarray,
) -> np.ndarray:
    """
    Wigner small-d matrix element d^j_{m',m}(β) via explicit summation.

    Uses the standard formula (Sakurai Eq. 3.8.33 / Rose convention):

        d^j_{m'm}(β) = Σ_k  (-1)^k √[(j+m)!(j-m)!(j+m')!(j-m')!]
                        / [(j+m-k)!(j-m'-k)! k! (k+m'-m)!]
                        × cos(β/2)^{2j-2k-m'+m} × sin(β/2)^{2k+m'-m}

    where the sum runs over all k such that factorials are non-negative.

    Parameters
    ----------
    j : float
        Total angular momentum quantum number (half-integer or integer).
    m : float
        Magnetic quantum number (projection along z).
    mp : float
        Helicity / second projection index.
    beta : ndarray
        Polar angle(s) in radians, shape (N,).

    Returns
    -------
    ndarray
        d^j_{m',m}(β) evaluated at each β, shape (N,).
    """
    k_min = int(max(0, m - mp))
    k_max = int(min(j + m, j - mp))

    # Pre-compute the factorials (all arguments guaranteed non-negative in valid range)
    fj_pm = factorial(j + m, exact=False)
    fj_mm = factorial(j - m, exact=False)
    fj_pmp = factorial(j + mp, exact=False)
    fj_mmp = factorial(j - mp, exact=False)
    prefactor = np.sqrt(fj_pm * fj_mm * fj_pmp * fj_mmp)

    half_beta = beta / 2.0
    cos_hb = np.cos(half_beta)
    sin_hb = np.sin(half_beta)

    result = np.zeros_like(beta, dtype=np.float64)

    for k in range(k_min, k_max + 1):
        sign = (-1.0) ** k
        denom = (
            factorial(j + m - k, exact=False)
            * factorial(j - mp - k, exact=False)
            * factorial(k, exact=False)
            * factorial(k + mp - m, exact=False)
        )
        cos_exp = int(2 * j - 2 * k - mp + m)
        sin_exp = int(2 * k + mp - m)

        # Handle zero^0 = 1 edge cases via safe power
        cos_term = np.where(cos_exp == 0, 1.0, cos_hb ** cos_exp)
        sin_term = np.where(sin_exp == 0, 1.0, sin_hb ** sin_exp)

        result += sign * (prefactor / denom) * cos_term * sin_term

    return result


def wigner_D(
    j: float,
    m: float,
    mp: float,
    theta: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """
    Full Wigner-D matrix element D^j_{m,λ}(φ,θ,0).

    D^j_{m,λ}(α,β,γ) = e^{-i m α} d^j_{m,λ}(β) e^{-i λ γ}

    We set γ = 0 (standard PWA convention — azimuthal phase only).

    Parameters
    ----------
    j, m, mp : float
        Angular momentum quantum numbers.
    theta : ndarray
        Polar angle array (β), shape (N,).
    phi : ndarray
        Azimuthal angle array (α), shape (N,).

    Returns
    -------
    ndarray
        Complex D^j_{m,mp}(φ,θ,0), shape (N,).
    """
    d_val = wigner_small_d(j, m, mp, theta)
    phase = np.exp(-1j * m * phi)
    return phase * d_val


# ════════════════════════════════════════════════════════════════════════════════
# QTT DECOMPOSITION WITH SVD DUMP
# ════════════════════════════════════════════════════════════════════════════════

def dense_to_qtt_2d_with_svd_dump(
    dense_2d: Tensor,
    n_bits: int,
    max_rank: int,
    tol: float = 1e-10,
) -> Tuple[List[Tensor], Dict[int, np.ndarray], List[int]]:
    """
    TT-SVD decomposition of a dense 2D field (θ × φ) in QTT format,
    returning cores, full SVD spectra at every bond, and bond dimensions.

    Uses Morton interleaving of θ-bits and φ-bits for spatial locality.

    Parameters
    ----------
    dense_2d : Tensor
        Dense (N_θ, N_φ) field on CUDA or CPU. Will be cast to float64.
    n_bits : int
        Number of bits per axis (N = 2^n_bits).
    max_rank : int
        Maximum bond dimension.
    tol : float
        Relative truncation tolerance.

    Returns
    -------
    cores : list[Tensor]
        QTT cores, each shape (r_{k-1}, 2, r_k).
    svd_spectra : dict[int, ndarray]
        Bond index → full singular value vector (before truncation).
    bond_dims : list[int]
        Bond dimension at each bond after truncation.
    """
    total_q = 2 * n_bits

    # Reshape to (2, 2, ..., 2) — n_bits for θ, n_bits for φ
    reshaped = dense_2d.reshape([2] * n_bits + [2] * n_bits)

    # Morton interleave: θ-bits and φ-bits → interleaved
    perm = []
    for i in range(n_bits):
        perm.extend([i, i + n_bits])
    morton = reshaped.permute(perm).reshape(2**total_q)

    cores: List[Tensor] = []
    svd_spectra: Dict[int, np.ndarray] = {}
    bond_dims: List[int] = []

    current = morton.reshape(1, -1)

    for k in range(total_q - 1):
        r_left = current.shape[0]
        current = current.reshape(r_left * 2, -1)

        U, S, Vh = torch.linalg.svd(current, full_matrices=False)

        # Store FULL spectrum before truncation
        svd_spectra[k] = S.detach().cpu().numpy().copy()

        # Truncate: relative threshold + hard cap
        s_max = S[0].item()
        if s_max > 0:
            keep = int((S > s_max * tol).sum().item())
        else:
            keep = 1
        r = max(1, min(max_rank, keep))

        U_trunc = U[:, :r]
        S_trunc = S[:r]
        Vh_trunc = Vh[:r, :]

        cores.append(U_trunc.reshape(r_left, 2, r))
        bond_dims.append(r)
        current = torch.diag(S_trunc) @ Vh_trunc

    # Final core
    cores.append(current.reshape(-1, 2, 1))
    bond_dims.append(1)

    return cores, svd_spectra, bond_dims


def qtt_reconstruct_2d(
    cores: List[Tensor],
    n_bits: int,
) -> Tensor:
    """
    Reconstruct the dense 2D field from QTT cores (inverse of decomposition).

    Contracts all cores via sequential bond contraction, un-permutes
    Morton interleaving, and reshapes to (N, N).

    Uses the standard MPS contraction: absorb each core left-to-right,
    keeping the physical indices as a flat product.
    """
    total_q = 2 * n_bits
    N = 2 ** n_bits

    # Start: cores[0] has shape (1, 2, r1)
    # We treat it as (2, r1) since left boundary dim is 1.
    result = cores[0].squeeze(0)  # (2, r1)

    for k in range(1, len(cores)):
        # result: (2^k, r_k),  cores[k]: (r_k, 2, r_{k+1})
        r_k = result.shape[-1]
        core_k = cores[k]  # (r_k, 2, r_{k+1})
        r_next = core_k.shape[-1]
        # Reshape core: (r_k, 2*r_next)
        core_flat = core_k.reshape(r_k, 2 * r_next)
        # Contract: (2^k, r_k) @ (r_k, 2*r_next) = (2^k, 2*r_next)
        result = result @ core_flat
        # Reshape to (2^(k+1), r_next)
        result = result.reshape(-1, r_next)

    # result: (2^total_q, 1) → flatten to (2^total_q,)
    vec = result.squeeze(-1)

    # Reshape to QTT shape (2, 2, ..., 2) with Morton ordering
    vec = vec.reshape([2] * total_q)

    # Inverse Morton permutation
    perm = []
    for i in range(n_bits):
        perm.extend([i, i + n_bits])
    inv_perm = [0] * total_q
    for new_pos, old_pos in enumerate(perm):
        inv_perm[old_pos] = new_pos
    vec = vec.permute(inv_perm)

    return vec.reshape(N, N)


# ════════════════════════════════════════════════════════════════════════════════
# PHYSICS SIMULATION
# ════════════════════════════════════════════════════════════════════════════════

def build_intensity_field(
    N_grid: int,
    j_values: List[float],
    helicity: float,
    seed: int,
    device: torch.device,
) -> Tuple[Tensor, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Build the Eq. 5.48 intensity field I(θ,φ) = |Σ_b A_b V_b|²
    from Wigner-D partial wave amplitudes.

    Parameters
    ----------
    N_grid : int
        Number of grid points per axis (must be power of 2).
    j_values : list[float]
        Total angular momentum quantum numbers to include.
    helicity : float
        Helicity quantum number λ (typically 0.5 for spin-1/2 baryon).
    seed : int
        RNG seed for production amplitudes V.
    device : torch.device
        Target device (cuda or cpu).

    Returns
    -------
    intensity : Tensor
        (N_grid, N_grid) real-valued intensity field on device, dtype float64.
    theta : ndarray
        Polar angle array (N_grid,).
    phi : ndarray
        Azimuthal angle array (N_grid,).
    V_vector : ndarray
        Complex production amplitudes.
    amplitude_grid : ndarray
        Complex amplitude field (N_grid, N_grid) for diagnostics.
    n_states : int
        Total number of quantum states.
    """
    theta = np.linspace(0.0, np.pi, N_grid, endpoint=True)
    phi = np.linspace(0.0, 2.0 * np.pi, N_grid, endpoint=False)
    THETA, PHI = np.meshgrid(theta, phi, indexing="ij")
    theta_flat = THETA.ravel()
    phi_flat = PHI.ravel()
    n_pixels = N_grid * N_grid

    # Enumerate quantum states: for each j, m ranges -j to +j
    states: List[Tuple[float, float]] = []
    for j in j_values:
        for m_val in np.arange(-j, j + 1, 1.0):
            states.append((j, m_val))
    n_states = len(states)

    # Production amplitudes V — complex, normalized
    rng = np.random.default_rng(seed)
    V_real = rng.standard_normal(n_states)
    V_imag = rng.standard_normal(n_states)
    V_vector = V_real + 1j * V_imag
    V_vector /= np.linalg.norm(V_vector)  # Unit-normalize

    # Build A matrix: (n_pixels, n_states) complex128
    A_matrix = np.zeros((n_pixels, n_states), dtype=np.complex128)
    for idx, (j, m) in enumerate(states):
        A_matrix[:, idx] = wigner_D(j, m, helicity, theta_flat, phi_flat)

    # Coherent sum: amplitude = Σ_b A_b V_b — the key physics
    total_amplitude = A_matrix @ V_vector  # (n_pixels,)

    # Quadratic form: I = |amplitude|²  — this creates interference cross-terms
    intensity_np = np.abs(total_amplitude) ** 2
    amplitude_grid = total_amplitude.reshape(N_grid, N_grid)
    intensity_grid = intensity_np.reshape(N_grid, N_grid)

    # Transfer to torch for QTT decomposition
    intensity_tensor = torch.tensor(
        intensity_grid, dtype=torch.float64, device=device
    )

    return intensity_tensor, theta, phi, V_vector, amplitude_grid, n_states


# ════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT DRIVER
# ════════════════════════════════════════════════════════════════════════════════

def run_experiment(
    N_grid: int = 256,
    max_rank: int = 128,
    tol: float = 1e-12,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run the full Eq. 5.48 interference experiment.

    Parameters
    ----------
    N_grid : int
        Grid resolution per axis (power of 2).
    max_rank : int
        Maximum QTT bond dimension.
    tol : float
        SVD truncation tolerance.
    seed : int
        RNG seed.

    Returns
    -------
    results : dict
        All experimental data, metrics, and arrays for figure generation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_bits = int(math.log2(N_grid))
    assert 2**n_bits == N_grid, f"N_grid must be power of 2, got {N_grid}"

    # Half-integer spin system: j = 1/2, 3/2, 5/2, 7/2
    j_values = [0.5, 1.5, 2.5, 3.5]
    helicity = 0.5

    print("=" * 70)
    print("RIGOROUS QTT TEST — Eq. 5.48 Interference (Badui Dissertation)")
    print(f"  Grid:       {N_grid}×{N_grid} (n_bits={n_bits})")
    print(f"  J values:   {j_values}")
    print(f"  Helicity:   λ = {helicity}")
    print(f"  max_rank:   {max_rank}")
    print(f"  tol:        {tol}")
    print(f"  Device:     {device}")
    print("=" * 70)

    # ── Phase 1: Build intensity field ────────────────────────────────────────
    t0 = time.perf_counter()
    intensity, theta, phi, V_vector, amplitude_grid, n_states = build_intensity_field(
        N_grid=N_grid,
        j_values=j_values,
        helicity=helicity,
        seed=seed,
        device=device,
    )
    t_physics = time.perf_counter() - t0

    print(f"\nPhase 1: Intensity field constructed ({n_states} quantum states)")
    print(f"  I_max  = {intensity.max().item():.6f}")
    print(f"  I_min  = {intensity.min().item():.6e}")
    print(f"  I_mean = {intensity.mean().item():.6f}")
    print(f"  Time:    {t_physics:.3f}s")

    # Verify coherent vs incoherent — the cross-terms matter
    incoherent = np.sum(
        np.abs(amplitude_grid.reshape(-1, 1)) ** 2
    )  # This is total energy either way
    # For diagnostics: compute incoherent sum (no cross-terms)
    A_flat = np.zeros((N_grid * N_grid, n_states), dtype=np.complex128)
    theta_flat = np.tile(theta, N_grid)
    phi_flat = np.repeat(phi, N_grid)
    THETA, PHI = np.meshgrid(theta, phi, indexing="ij")
    theta_flat = THETA.ravel()
    phi_flat = PHI.ravel()
    states_list = []
    for j in j_values:
        for m_val in np.arange(-j, j + 1, 1.0):
            states_list.append((j, m_val))
    for idx, (j, m) in enumerate(states_list):
        A_flat[:, idx] = wigner_D(j, m, helicity, theta_flat, phi_flat)
    incoherent_intensity = np.sum(np.abs(A_flat) ** 2 * np.abs(V_vector) ** 2, axis=1)
    coherent_intensity = np.abs(A_flat @ V_vector) ** 2
    cross_term_fraction = 1.0 - np.sum(incoherent_intensity) / np.sum(coherent_intensity)
    print(f"  Cross-term fraction: {cross_term_fraction:.4f}")
    print(f"  (0 = no interference, ≠0 = interference present)")

    # ── Phase 2: QTT decomposition ───────────────────────────────────────────
    t0 = time.perf_counter()
    cores, svd_spectra, bond_dims = dense_to_qtt_2d_with_svd_dump(
        dense_2d=intensity,
        n_bits=n_bits,
        max_rank=max_rank,
        tol=tol,
    )
    t_qtt = time.perf_counter() - t0

    chi_max = max(bond_dims)
    chi_mean = np.mean(bond_dims)
    total_qubits = 2 * n_bits

    # QTT storage: Σ_k r_{k-1} × 2 × r_k
    qtt_storage = sum(c.numel() for c in cores)
    dense_storage = N_grid * N_grid
    compression_ratio = dense_storage / qtt_storage

    print(f"\nPhase 2: QTT decomposition complete")
    print(f"  Total qubits:     {total_qubits}")
    print(f"  χ_max:            {chi_max}")
    print(f"  χ_mean:           {chi_mean:.1f}")
    print(f"  Bond dimensions:  {bond_dims}")
    print(f"  Dense storage:    {dense_storage:,} floats")
    print(f"  QTT storage:      {qtt_storage:,} floats")
    print(f"  Compression:      {compression_ratio:.1f}×")
    print(f"  Time:             {t_qtt:.3f}s")

    # ── Phase 3: Reconstruction error ────────────────────────────────────────
    t0 = time.perf_counter()
    reconstructed = qtt_reconstruct_2d(cores, n_bits)
    t_recon = time.perf_counter() - t0

    error_abs = torch.norm(intensity - reconstructed).item()
    error_rel = error_abs / torch.norm(intensity).item()

    print(f"\nPhase 3: Reconstruction verification")
    print(f"  ||I - I_QTT||_F:  {error_abs:.6e}")
    print(f"  Relative error:   {error_rel:.6e}")
    print(f"  Time:             {t_recon:.3f}s")

    # ── Phase 4: Multi-resolution sweep ──────────────────────────────────────
    print(f"\nPhase 4: Multi-resolution sweep")
    resolution_results: List[Dict[str, Any]] = []
    for n_b in [5, 6, 7, 8, 9]:
        ng = 2**n_b
        I_sweep, _, _, _, _, _ = build_intensity_field(
            N_grid=ng, j_values=j_values, helicity=helicity,
            seed=seed, device=device,
        )
        _, _, bd_sweep = dense_to_qtt_2d_with_svd_dump(
            I_sweep, n_b, max_rank=max_rank, tol=tol,
        )
        chi_max_sweep = max(bd_sweep)
        dense_s = ng * ng
        qtt_s = sum(
            bd_sweep[max(0, k - 1)] * 2 * bd_sweep[k]
            if k < len(bd_sweep) - 1
            else bd_sweep[k - 1] * 2 * 1
            for k in range(len(bd_sweep))
        )
        # Recompute properly from actual cores
        c_sweep, _, _ = dense_to_qtt_2d_with_svd_dump(
            I_sweep, n_b, max_rank=max_rank, tol=tol,
        )
        qtt_s = sum(c.numel() for c in c_sweep)
        cr = dense_s / qtt_s
        resolution_results.append({
            "n_bits": n_b,
            "N_grid": ng,
            "chi_max": chi_max_sweep,
            "dense": dense_s,
            "qtt": qtt_s,
            "compression": cr,
        })
        print(f"  {ng:>5}×{ng:<5}  χ_max={chi_max_sweep:>3}  "
              f"CR={cr:>8.1f}×")

    # ── Phase 5: J-sweep (increasing angular complexity) ─────────────────────
    print(f"\nPhase 5: Complexity sweep (increasing J_max)")
    j_sweep_results: List[Dict[str, Any]] = []
    j_sweep_configs = [
        [0.5],
        [0.5, 1.5],
        [0.5, 1.5, 2.5],
        [0.5, 1.5, 2.5, 3.5],
        [0.5, 1.5, 2.5, 3.5, 4.5],
        [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
    ]
    for j_cfg in j_sweep_configs:
        n_st = sum(int(2 * j + 1) for j in j_cfg)
        I_j, _, _, _, _, _ = build_intensity_field(
            N_grid=N_grid, j_values=j_cfg, helicity=helicity,
            seed=seed, device=device,
        )
        c_j, svd_j, bd_j = dense_to_qtt_2d_with_svd_dump(
            I_j, n_bits, max_rank=max_rank, tol=tol,
        )
        chi_j = max(bd_j)
        qtt_j = sum(c.numel() for c in c_j)
        cr_j = dense_storage / qtt_j

        # Find most entangled bond
        max_bond_idx = np.argmax([len(s) for s in svd_j.values()])

        j_sweep_results.append({
            "j_max": max(j_cfg),
            "n_states": n_st,
            "chi_max": chi_j,
            "compression": cr_j,
            "svd_spectrum_peak": svd_j[max_bond_idx],
        })
        print(f"  J_max={max(j_cfg):>4.1f}  states={n_st:>3}  "
              f"χ_max={chi_j:>3}  CR={cr_j:>7.1f}×")

    # ── Collate results ──────────────────────────────────────────────────────
    results = {
        "N_grid": N_grid,
        "n_bits": n_bits,
        "n_states": n_states,
        "j_values": j_values,
        "helicity": helicity,
        "max_rank": max_rank,
        "tol": tol,
        "seed": seed,
        "device": str(device),
        "chi_max": chi_max,
        "chi_mean": float(chi_mean),
        "bond_dims": bond_dims,
        "dense_storage": dense_storage,
        "qtt_storage": qtt_storage,
        "compression_ratio": compression_ratio,
        "error_abs": error_abs,
        "error_rel": error_rel,
        "cross_term_fraction": cross_term_fraction,
        "intensity": intensity.detach().cpu().numpy(),
        "reconstructed": reconstructed.detach().cpu().numpy(),
        "theta": theta,
        "phi": phi,
        "amplitude_grid": amplitude_grid,
        "svd_spectra": svd_spectra,
        "resolution_results": resolution_results,
        "j_sweep_results": j_sweep_results,
        "t_physics": t_physics,
        "t_qtt": t_qtt,
        "t_recon": t_recon,
    }

    print(f"\n{'=' * 70}")
    print(f"RESULT: χ_max = {chi_max} out of {max_rank} allowed")
    print(f"        Compression = {compression_ratio:.1f}×")
    print(f"        Relative error = {error_rel:.2e}")
    print(f"        Interference present: cross-term fraction = {cross_term_fraction:.4f}")
    print(f"{'=' * 70}")

    return results


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE GENERATION
# ════════════════════════════════════════════════════════════════════════════════

def figure_intensity_map(results: Dict[str, Any]) -> None:
    """
    Fig A: Intensity I(θ,φ) — the interference pattern itself.
    Shows the complex angular structure from coherent partial wave sum.
    """
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COL)

    I = results["intensity"]
    theta = np.degrees(results["theta"])
    phi = np.degrees(results["phi"])

    # Left: full intensity map
    ax = axes[0]
    im = ax.imshow(
        I,
        extent=[phi[0], phi[-1], theta[-1], theta[0]],
        aspect="auto",
        cmap="inferno",
        interpolation="bilinear",
    )
    ax.set_xlabel(r"$\phi$ (deg)")
    ax.set_ylabel(r"$\theta$ (deg)")
    ax.set_title(r"$I(\theta,\phi) = |\sum_b A_b V_b|^2$")
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("Intensity", fontsize=8)
    ax.tick_params(direction="in", which="both")

    # Right: θ cross-section at equator
    ax = axes[1]
    mid_theta = I.shape[0] // 2
    ax.plot(phi, I[mid_theta, :], color=C_BLUE, linewidth=1.0)
    ax.set_xlabel(r"$\phi$ (deg)")
    ax.set_ylabel(r"$I(\theta=90°,\phi)$")
    ax.set_title("Equatorial cross-section")
    ax.tick_params(direction="in", which="both")

    fig.tight_layout(w_pad=2.0)
    for fmt in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"eq548_intensity_map.{fmt}")
    plt.close(fig)
    print("  Fig A saved: eq548_intensity_map.pdf")


def figure_svd_spectrum(results: Dict[str, Any]) -> None:
    """
    Fig B: SVD spectrum at the most-entangled bond.
    Proves the interference pattern has rapidly decaying singular values.
    """
    fig, ax = plt.subplots(figsize=SINGLE_COL)

    svd_spectra = results["svd_spectra"]
    bond_dims = results["bond_dims"]

    # Find the most entangled bond (largest χ)
    peak_bond = int(np.argmax(bond_dims[:-1]))  # Exclude final trivial bond
    S_peak = svd_spectra[peak_bond]
    S_norm = S_peak / S_peak[0]  # Normalize to leading SV

    ax.semilogy(
        np.arange(1, len(S_norm) + 1),
        S_norm,
        "o-",
        color=C_BLUE,
        markersize=3,
        linewidth=0.8,
        label=f"Bond {peak_bond}",
    )

    # Also show a few other bonds for context
    n_bonds = len(svd_spectra)
    other_bonds = [0, n_bonds // 4, n_bonds // 2, 3 * n_bonds // 4]
    colors_other = [C_ORANGE, C_GREEN, C_RED, C_PURPLE]
    for bond_idx, color in zip(other_bonds, colors_other):
        if bond_idx == peak_bond or bond_idx not in svd_spectra:
            continue
        S_b = svd_spectra[bond_idx]
        S_b_norm = S_b / S_b[0] if S_b[0] > 0 else S_b
        ax.semilogy(
            np.arange(1, len(S_b_norm) + 1),
            S_b_norm,
            "s-",
            color=color,
            markersize=2,
            linewidth=0.6,
            alpha=0.7,
            label=f"Bond {bond_idx}",
        )

    # Mark the effective rank (truncation threshold)
    ax.axhline(
        results["tol"], color=C_GREY, linestyle="--", linewidth=0.6, alpha=0.8
    )
    ax.text(
        len(S_norm) * 0.7,
        results["tol"] * 2,
        f"tol = {results['tol']:.0e}",
        fontsize=7,
        color=C_GREY,
    )

    # Mark effective rank on peak bond
    eff_rank = results["bond_dims"][peak_bond]
    ax.axvline(eff_rank, color=C_RED, linestyle=":", linewidth=0.6)
    ax.text(
        eff_rank + 1, 0.3, f"χ = {eff_rank}",
        fontsize=7, color=C_RED,
    )

    ax.set_xlabel("Singular value index")
    ax.set_ylabel(r"$\sigma_i / \sigma_1$")
    ax.set_title("SVD spectrum (most-entangled bond)")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(bottom=1e-16, top=2.0)
    ax.tick_params(direction="in", which="both")

    fig.tight_layout()
    for fmt in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"eq548_svd_spectrum.{fmt}")
    plt.close(fig)
    print("  Fig B saved: eq548_svd_spectrum.pdf")


def figure_bond_dimensions(results: Dict[str, Any]) -> None:
    """
    Fig C: Bond dimension profile across all bonds.
    Shows spatial structure of entanglement in the QTT representation.
    """
    fig, ax = plt.subplots(figsize=SINGLE_COL)

    bd = results["bond_dims"]
    n_bonds = len(bd)
    ax.bar(
        range(n_bonds),
        bd,
        color=C_BLUE,
        edgecolor=C_BLACK,
        linewidth=0.3,
        alpha=0.85,
    )
    ax.axhline(
        results["max_rank"], color=C_RED, linestyle="--", linewidth=0.8,
        label=f"max_rank = {results['max_rank']}",
    )
    ax.set_xlabel("Bond index")
    ax.set_ylabel(r"Bond dimension $\chi$")
    ax.set_title(f"QTT bond profile — {results['N_grid']}×{results['N_grid']} grid")
    ax.legend(fontsize=7)
    ax.set_xlim(-0.5, n_bonds - 0.5)
    ax.tick_params(direction="in", which="both")

    fig.tight_layout()
    for fmt in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"eq548_bond_dimensions.{fmt}")
    plt.close(fig)
    print("  Fig C saved: eq548_bond_dimensions.pdf")


def figure_compression_vs_grid(results: Dict[str, Any]) -> None:
    """
    Fig D: Compression ratio vs grid size — log-linear scaling proves O(log N).
    """
    fig, ax = plt.subplots(figsize=SINGLE_COL)

    res = results["resolution_results"]
    grids = [r["N_grid"] for r in res]
    crs = [r["compression"] for r in res]
    chis = [r["chi_max"] for r in res]

    ax.semilogy(
        [f"{g}²" for g in grids],
        crs,
        "o-",
        color=C_BLUE,
        linewidth=1.2,
        markersize=6,
        zorder=3,
    )

    # Annotate χ_max at each point
    for i, (g, cr, chi) in enumerate(zip(grids, crs, chis)):
        ax.annotate(
            f"χ={chi}",
            (i, cr),
            textcoords="offset points",
            xytext=(0, 10),
            fontsize=6,
            ha="center",
            color=C_GREY,
        )

    ax.set_xlabel("Grid resolution")
    ax.set_ylabel("Compression ratio")
    ax.set_title(r"QTT compression: $I(\theta,\phi) = |\sum A_b V_b|^2$")
    ax.tick_params(direction="in", which="both")

    fig.tight_layout()
    for fmt in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"eq548_compression_vs_grid.{fmt}")
    plt.close(fig)
    print("  Fig D saved: eq548_compression_vs_grid.pdf")


def figure_chi_vs_jmax(results: Dict[str, Any]) -> None:
    """
    Fig E: χ_max vs J_max (angular complexity).
    Proves rank grows slowly even as interference pattern complexity increases.
    """
    fig, ax = plt.subplots(figsize=SINGLE_COL)

    j_res = results["j_sweep_results"]
    j_max_vals = [r["j_max"] for r in j_res]
    chi_vals = [r["chi_max"] for r in j_res]
    n_states_vals = [r["n_states"] for r in j_res]

    ax.plot(
        j_max_vals, chi_vals,
        "o-", color=C_BLUE, linewidth=1.2, markersize=6,
        label=r"$\chi_{\max}$",
    )

    # Secondary axis: number of quantum states
    ax2 = ax.twinx()
    ax2.plot(
        j_max_vals, n_states_vals,
        "s--", color=C_ORANGE, linewidth=1.0, markersize=5,
        label="# states",
    )
    ax2.set_ylabel("Number of quantum states", color=C_ORANGE, fontsize=9)
    ax2.tick_params(axis="y", labelcolor=C_ORANGE, direction="in")

    ax.set_xlabel(r"$J_{\max}$")
    ax.set_ylabel(r"$\chi_{\max}$", color=C_BLUE)
    ax.set_title("Bond dimension vs angular complexity")
    ax.tick_params(axis="y", labelcolor=C_BLUE, direction="in")
    ax.tick_params(axis="x", direction="in")

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")

    fig.tight_layout()
    for fmt in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"eq548_chi_vs_jmax.{fmt}")
    plt.close(fig)
    print("  Fig E saved: eq548_chi_vs_jmax.pdf")


def figure_reconstruction_error(results: Dict[str, Any]) -> None:
    """
    Fig F: Pointwise reconstruction error map.
    Proves the QTT approximation is faithful everywhere, not just on average.
    """
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COL)

    I_orig = results["intensity"]
    I_recon = results["reconstructed"]
    diff = np.abs(I_orig - I_recon)
    theta = np.degrees(results["theta"])
    phi = np.degrees(results["phi"])

    # Left: reconstructed intensity
    ax = axes[0]
    im = ax.imshow(
        I_recon,
        extent=[phi[0], phi[-1], theta[-1], theta[0]],
        aspect="auto",
        cmap="inferno",
        interpolation="bilinear",
    )
    ax.set_xlabel(r"$\phi$ (deg)")
    ax.set_ylabel(r"$\theta$ (deg)")
    ax.set_title("QTT reconstruction")
    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    ax.tick_params(direction="in", which="both")

    # Right: absolute error
    ax = axes[1]
    if diff.max() > 0:
        im = ax.imshow(
            diff,
            extent=[phi[0], phi[-1], theta[-1], theta[0]],
            aspect="auto",
            cmap="hot",
            interpolation="bilinear",
        )
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label(r"$|I - I_{\mathrm{QTT}}|$", fontsize=8)
    else:
        ax.text(
            0.5, 0.5, "Error = 0\n(exact)",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=12, color=C_GREY,
        )
    ax.set_xlabel(r"$\phi$ (deg)")
    ax.set_ylabel(r"$\theta$ (deg)")
    ax.set_title(
        f"Pointwise error (rel = {results['error_rel']:.2e})"
    )
    ax.tick_params(direction="in", which="both")

    fig.tight_layout(w_pad=2.0)
    for fmt in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"eq548_reconstruction_error.{fmt}")
    plt.close(fig)
    print("  Fig F saved: eq548_reconstruction_error.pdf")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    print(f"\n{'=' * 70}")
    print("Eq. 5.48 INTERFERENCE QTT EXPERIMENT — Adams (2026)")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Time:   {timestamp}")
    print(f"{'=' * 70}\n")

    # Run the experiment
    results = run_experiment(
        N_grid=256,
        max_rank=128,
        tol=1e-12,
        seed=42,
    )

    # Generate all figures
    print(f"\n{'=' * 70}")
    print("Figure generation (600 DPI vector PDF)")
    print("-" * 70)

    figure_intensity_map(results)
    figure_svd_spectrum(results)
    figure_bond_dimensions(results)
    figure_compression_vs_grid(results)
    figure_chi_vs_jmax(results)
    figure_reconstruction_error(results)

    # Write metadata
    metadata = {
        "experiment": "eq548_interference_qtt",
        "timestamp": timestamp,
        "N_grid": results["N_grid"],
        "n_states": results["n_states"],
        "j_values": results["j_values"],
        "helicity": results["helicity"],
        "max_rank": results["max_rank"],
        "tol": results["tol"],
        "chi_max": results["chi_max"],
        "chi_mean": results["chi_mean"],
        "bond_dims": results["bond_dims"],
        "compression_ratio": results["compression_ratio"],
        "error_rel": results["error_rel"],
        "cross_term_fraction": results["cross_term_fraction"],
        "dense_storage": results["dense_storage"],
        "qtt_storage": results["qtt_storage"],
        "device": results["device"],
        "resolution_sweep": results["resolution_results"],
        "j_sweep": [
            {
                "j_max": r["j_max"],
                "n_states": r["n_states"],
                "chi_max": r["chi_max"],
                "compression": r["compression"],
            }
            for r in results["j_sweep_results"]
        ],
        "timings": {
            "physics_s": results["t_physics"],
            "qtt_s": results["t_qtt"],
            "reconstruct_s": results["t_recon"],
        },
    }

    meta_path = OUTPUT_DIR / "eq548_metadata.json"
    meta_json = json.dumps(metadata, indent=2)
    meta_path.write_text(meta_json)
    sha = hashlib.sha256(meta_json.encode()).hexdigest()[:16]

    t_total = results["t_physics"] + results["t_qtt"] + results["t_recon"]
    print(f"\n{'=' * 70}")
    print("COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Figures:     eq548_*.{{pdf,png}} in {OUTPUT_DIR}")
    print(f"  Metadata:    eq548_metadata.json (SHA-256: {sha}...)")
    print(f"  Wall:        {t_total:.1f}s")
    print(f"\n  KEY RESULT: χ_max = {results['chi_max']} → "
          f"Compression = {results['compression_ratio']:.1f}×")
    print(f"  INTERFERENCE CROSS-TERMS: {results['cross_term_fraction']:.4f} "
          f"fraction of total energy")
    print(f"  CONCLUSION: {_conclusion(results)}")


def _conclusion(results: Dict[str, Any]) -> str:
    chi = results["chi_max"]
    mr = results["max_rank"]
    cr = results["compression_ratio"]
    err = results["error_rel"]
    if chi < mr and cr > 2.0 and err < 1e-6:
        return (
            f"QTT PROVES LOW-RANK. Even with full Eq. 5.48 interference, "
            f"the intensity field compresses {cr:.0f}× with χ_max={chi} "
            f"(rel error {err:.1e}). Tensor networks handle the quadratic "
            f"form cross-terms."
        )
    elif chi < mr:
        return (
            f"Interference pattern is compressible (χ_max={chi} < max_rank={mr}), "
            f"compression = {cr:.1f}×, error = {err:.1e}."
        )
    else:
        return (
            f"Bond dimension hit cap (χ_max={chi} = max_rank={mr}). "
            f"Increase max_rank to determine true rank."
        )


if __name__ == "__main__":
    main()
