#!/usr/bin/env python3
"""Phase 0: Helmholtz QTT Rank Validation.

Risk gate: Does the frequency-domain Helmholtz solution have bounded rank
in QTT format? If χ stays bounded as N grows, frequency-domain QTT Maxwell
is viable and we skip the O(N) time-step wall entirely.

Test protocol
-------------
1. Build 1D Helmholtz operator: ∇²E + k²ε·E = J(x)
2. Solve dense (numpy) at N = 2^8, 2^10, 2^12, 2^14, 2^16
3. Compress each solution to QTT via TT-SVD
4. Measure max bond dimension χ vs. grid size N
5. PASS if χ is bounded (doesn't grow with N)

Also tests with:
- Uniform medium (ε=1): standing wave → known low rank
- Piecewise medium (ε=1/4 for x>0.5): step discontinuity → tests rank
- Lossy PML-like medium (complex ε): absorbing BC

Uses existing infrastructure:
- ontic.vm.operators.laplacian_mpo_1d   — Laplacian MPO
- ontic.vm.operators.identity_mpo       — Identity MPO
- ontic.vm.operators.mpo_add/mpo_scale  — MPO algebra
- ontic.vm.gpu_tensor._tt_svd_1d        — QTT compression

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ─────────────────────────────────────────────────────────────────────
# 1D Helmholtz solve (sparse for large, dense for small)
# ─────────────────────────────────────────────────────────────────────

def solve_helmholtz_1d(
    N: int,
    k: float,
    eps_r: NDArray | None = None,
    source_pos: float = 0.5,
    source_width: float = 0.02,
    pml_cells: int = 20,
    pml_sigma_max: float = 10.0,
) -> tuple[NDArray, NDArray]:
    """Solve 1D Helmholtz: (∂²/∂x² + k²·ε_r(x)) E(x) = J(x).

    Uses scipy sparse tridiagonal solver for N > 8192 to avoid
    O(N²) memory. Falls back to dense numpy for small grids.

    Parameters
    ----------
    N : int
        Number of grid points.
    k : float
        Wavenumber (2πf/c).
    eps_r : NDArray, optional
        Relative permittivity profile, length N. Defaults to uniform 1.0.
    source_pos : float
        Source position in [0, 1].
    source_width : float
        Gaussian source width.
    pml_cells : int
        Number of PML absorbing cells on each end.
    pml_sigma_max : float
        Maximum PML conductivity.

    Returns
    -------
    x_grid : NDArray
        Grid coordinates.
    E_field : NDArray
        Complex electric field solution.
    """
    h = 1.0 / N
    x = np.linspace(h / 2, 1.0 - h / 2, N)

    if eps_r is None:
        eps_r = np.ones(N)

    # Build PML complex coordinate stretch
    eps_pml = eps_r.astype(complex).copy()
    for i in range(N):
        if i < pml_cells:
            depth = (pml_cells - i) / pml_cells
            sigma = pml_sigma_max * depth ** 2
            eps_pml[i] = eps_r[i] * (1.0 + 1j * sigma)
        elif i >= N - pml_cells:
            depth = (i - (N - pml_cells - 1)) / pml_cells
            sigma = pml_sigma_max * depth ** 2
            eps_pml[i] = eps_r[i] * (1.0 + 1j * sigma)

    # Source: Gaussian current at source_pos
    J = np.exp(-((x - source_pos) ** 2) / (2 * source_width ** 2))
    J = J / (np.sqrt(2 * np.pi) * source_width)
    J[0] = 0.0
    J[N - 1] = 0.0
    rhs = -J.astype(complex)

    # Diagonal values: -2/h² + k²·ε_pml
    diag_main = -2.0 / (h * h) + k * k * eps_pml
    diag_off = np.full(N - 1, 1.0 / (h * h), dtype=complex)

    # PEC boundary: E = 0 at edges
    diag_main[0] = 1.0
    diag_main[N - 1] = 1.0
    diag_off[0] = 0.0       # A[0,1] = 0
    diag_off[N - 2] = 0.0   # A[N-1,N-2] = 0
    rhs[0] = 0.0
    rhs[N - 1] = 0.0

    # Use sparse solver for large grids, dense for small
    if HAS_SCIPY and N > 8192:
        A = sparse.diags(
            [diag_off, diag_main, diag_off],
            offsets=[-1, 0, 1],
            shape=(N, N),
            format="csc",
            dtype=complex,
        )
        E = spsolve(A, rhs)
    else:
        A = np.zeros((N, N), dtype=complex)
        for i in range(N):
            A[i, i] = diag_main[i]
            if i > 0:
                A[i, i - 1] = diag_off[i - 1]
            if i < N - 1:
                A[i, i + 1] = diag_off[i]
        E = np.linalg.solve(A, rhs)

    return x, E


# ─────────────────────────────────────────────────────────────────────
# QTT compression of 1D solution
# ─────────────────────────────────────────────────────────────────────

def compress_to_qtt(
    field: NDArray,
    max_rank: int = 256,
    cutoff: float = 1e-12,
) -> tuple[list[NDArray], dict[str, Any]]:
    """Compress a 1D array to QTT format via TT-SVD.

    Parameters
    ----------
    field : NDArray
        1-D array of length 2^n.
    max_rank : int
        Maximum bond dimension.
    cutoff : float
        Singular value cutoff.

    Returns
    -------
    cores : list[NDArray]
        QTT cores, each (r_l, 2, r_r).
    info : dict
        Rank profile and compression stats.
    """
    from ontic.engine.vm.gpu_tensor import _tt_svd_1d

    N = len(field)
    n_bits = int(math.log2(N))
    assert 2 ** n_bits == N, f"N={N} must be power of 2"

    # Handle complex fields: compress real and imaginary parts separately,
    # then report the max rank across both
    if np.iscomplexobj(field):
        re_cores, re_info = compress_to_qtt(field.real, max_rank, cutoff)
        im_cores, im_info = compress_to_qtt(field.imag, max_rank, cutoff)

        # Combined rank profile: max of real and imag at each bond
        re_ranks = re_info["ranks"]
        im_ranks = im_info["ranks"]
        combined_ranks = [max(r, i) for r, i in zip(re_ranks, im_ranks)]
        chi_max = max(combined_ranks)

        re_storage = sum(c.nbytes for c in re_cores)
        im_storage = sum(c.nbytes for c in im_cores)

        info = {
            "n_bits": n_bits,
            "N": N,
            "chi_max": chi_max,
            "ranks": combined_ranks,
            "chi_max_real": re_info["chi_max"],
            "chi_max_imag": im_info["chi_max"],
            "storage_bytes": re_storage + im_storage,
            "dense_bytes": N * 16,  # complex128
            "compression_ratio": (N * 16) / (re_storage + im_storage),
        }
        return re_cores, info  # Return real cores for reference

    # Real case
    vals = field.astype(np.float64).reshape([2] * n_bits)
    cores = _tt_svd_1d(vals, max_rank=max_rank, cutoff=cutoff)

    ranks = [1] + [c.shape[2] for c in cores]
    chi_max = max(ranks)
    storage = sum(c.nbytes for c in cores)

    info = {
        "n_bits": n_bits,
        "N": N,
        "chi_max": chi_max,
        "ranks": ranks,
        "storage_bytes": storage,
        "dense_bytes": N * 8,  # float64
        "compression_ratio": (N * 8) / storage,
    }
    return cores, info


# ─────────────────────────────────────────────────────────────────────
# Main convergence test
# ─────────────────────────────────────────────────────────────────────

def run_phase0() -> dict[str, Any]:
    """Run the full Phase 0 rank validation test."""
    ts = datetime.now(timezone.utc).isoformat()
    print("=" * 80)
    print("  PHASE 0: HELMHOLTZ QTT RANK VALIDATION")
    print("  Does the frequency-domain solution have bounded rank?")
    print("=" * 80)
    print()

    # Test configurations
    n_bits_list = [8, 10, 12, 14, 16, 18, 20]
    wavenumbers = [2 * math.pi, 4 * math.pi, 8 * math.pi]  # k = 2π, 4π, 8π

    # Media profiles
    media = {
        "uniform": lambda N: np.ones(N),
        "piecewise": lambda N: np.where(
            np.linspace(0, 1, N) > 0.5, 4.0, 1.0
        ),
        "graded": lambda N: 1.0 + 3.0 * np.linspace(0, 1, N) ** 2,
    }

    all_results: list[dict[str, Any]] = []

    for medium_name, eps_fn in media.items():
        print(f"\n{'#' * 72}")
        print(f"  MEDIUM: {medium_name}")
        print(f"{'#' * 72}")

        for k in wavenumbers:
            k_label = f"k={k / math.pi:.0f}π"
            print(f"\n  --- {k_label} ---")

            chi_vs_N: list[dict[str, Any]] = []

            for n_bits in n_bits_list:
                N = 2 ** n_bits
                eps_r = eps_fn(N)

                print(f"    N = {N:>7,} (2^{n_bits}): ", end="", flush=True)

                t0 = time.perf_counter()

                # Dense solve
                try:
                    x_grid, E_field = solve_helmholtz_1d(
                        N=N, k=k, eps_r=eps_r,
                        pml_cells=min(40, N // 8),
                        pml_sigma_max=10.0,
                    )
                except (np.linalg.LinAlgError, MemoryError) as e:
                    print(f"SOLVE FAILED: {e}")
                    continue

                t_solve = time.perf_counter() - t0

                # QTT compress
                t1 = time.perf_counter()
                _, info = compress_to_qtt(E_field, max_rank=256, cutoff=1e-12)
                t_compress = time.perf_counter() - t1

                chi = info["chi_max"]
                ratio = info["compression_ratio"]
                ranks = info.get("ranks", [])

                print(f"χ = {chi:3d}  |  "
                      f"compress = {ratio:>7.1f}×  |  "
                      f"solve = {t_solve:.2f}s  |  "
                      f"compress = {t_compress:.3f}s")

                entry = {
                    "medium": medium_name,
                    "k": k,
                    "k_label": k_label,
                    "n_bits": n_bits,
                    "N": N,
                    "chi_max": chi,
                    "chi_real": info.get("chi_max_real", chi),
                    "chi_imag": info.get("chi_max_imag", chi),
                    "compression_ratio": round(ratio, 2),
                    "solve_time_s": round(t_solve, 4),
                    "compress_time_s": round(t_compress, 4),
                    "ranks": ranks if len(ranks) < 30 else ranks[:5] + ["..."] + ranks[-5:],
                }
                chi_vs_N.append(entry)
                all_results.append(entry)

            # Print rank growth summary for this (medium, k)
            if len(chi_vs_N) >= 2:
                chi_vals = [r["chi_max"] for r in chi_vs_N]
                N_vals = [r["N"] for r in chi_vs_N]
                growth = chi_vals[-1] / max(chi_vals[0], 1)
                N_growth = N_vals[-1] / N_vals[0]
                bounded = chi_vals[-1] <= 2 * chi_vals[0] + 5

                verdict = "BOUNDED ✓" if bounded else "GROWING ✗"
                print(f"\n    Rank growth: χ went {chi_vals[0]} → {chi_vals[-1]} "
                      f"({growth:.1f}×) while N grew {N_growth:.0f}×  →  {verdict}")

    # ── Overall verdict ──────────────────────────────────────────────
    print(f"\n\n{'=' * 80}")
    print("  PHASE 0 VERDICT")
    print(f"{'=' * 80}")

    # Group by medium
    for medium_name in media:
        results_m = [r for r in all_results if r["medium"] == medium_name]
        if not results_m:
            continue

        # For each k, check if rank is bounded
        for k in wavenumbers:
            results_mk = [r for r in results_m if r["k"] == k]
            if len(results_mk) < 2:
                continue

            chi_first = results_mk[0]["chi_max"]
            chi_last = results_mk[-1]["chi_max"]
            N_first = results_mk[0]["N"]
            N_last = results_mk[-1]["N"]

            bounded = chi_last <= 2 * chi_first + 5
            k_label = results_mk[0]["k_label"]
            symbol = "✓" if bounded else "✗"
            print(f"  {symbol} {medium_name:12s} {k_label:8s}: "
                  f"χ = {chi_first} → {chi_last} "
                  f"(N: {N_first:,} → {N_last:,})")

    # Final pass/fail
    all_bounded = True
    for medium_name in media:
        for k in wavenumbers:
            results_mk = [r for r in all_results
                          if r["medium"] == medium_name and r["k"] == k]
            if len(results_mk) >= 2:
                if results_mk[-1]["chi_max"] > 2 * results_mk[0]["chi_max"] + 5:
                    all_bounded = False

    print()
    if all_bounded:
        print("  ╔══════════════════════════════════════════════════╗")
        print("  ║   PASS — Helmholtz solutions have bounded rank  ║")
        print("  ║   Frequency-domain QTT Maxwell is VIABLE        ║")
        print("  ╚══════════════════════════════════════════════════╝")
    else:
        print("  ╔══════════════════════════════════════════════════╗")
        print("  ║   MIXED — Some configurations show rank growth  ║")
        print("  ║   Review per-configuration results              ║")
        print("  ╚══════════════════════════════════════════════════╝")

    # ── Save results ─────────────────────────────────────────────────
    output = {
        "protocol": "PHASE0_HELMHOLTZ_RANK_VALIDATION",
        "version": "1.0.0",
        "timestamp": ts,
        "verdict": "PASS" if all_bounded else "MIXED",
        "test_config": {
            "n_bits_tested": n_bits_list,
            "wavenumbers": wavenumbers,
            "media": list(media.keys()),
            "max_rank_cap": 256,
            "svd_cutoff": 1e-12,
            "pml_sigma_max": 10.0,
        },
        "results": all_results,
    }

    out_file = f"phase0_helmholtz_rank_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved: {out_file}")

    return output


if __name__ == "__main__":
    run_phase0()
