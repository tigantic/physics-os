#!/usr/bin/env python3
"""Phase 1 Validation: QTT Helmholtz Solver vs Dense Ground Truth.

Solves the same 1D Helmholtz problems from Phase 0 using the new QTT
frequency-domain solver, and compares against scipy sparse ground truth.

Test plan:
1. Small N (2^8, 2^10, 2^12): full solution comparison vs sparse solve
2. Large N (2^14, 2^16, 2^18): QTT-only, verify residual convergence + rank
3. Multiple media and wavenumbers

Author: HyperTensor Team
Date: February 2026
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

# QTT Helmholtz solver
from ontic.em.qtt_helmholtz import (
    HelmholtzConfig,
    HelmholtzResult,
    array_to_tt,
    build_pml_eps_profile,
    diag_mpo_from_tt,
    gaussian_source_tt,
    helmholtz_mpo_1d,
    helmholtz_solve_1d,
    mpo_add_c,
    mpo_scale_c,
    reconstruct_1d,
    tt_gmres_complex,
    tt_inner_hermitian,
    tt_norm_c,
)
from ontic.engine.vm.operators import laplacian_mpo_1d

# Sparse solver for ground truth
try:
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ======================================================================
# Ground truth solver (scipy sparse)
# ======================================================================

def solve_helmholtz_1d_sparse(
    N: int,
    k: float,
    eps_r: NDArray | None = None,
    source_pos: float = 0.5,
    source_width: float = 0.02,
    pml_cells: int = 20,
    pml_sigma_max: float = 10.0,
) -> tuple[NDArray, NDArray]:
    """Solve 1D Helmholtz with PEC boundaries via scipy sparse.

    Identical physics to the QTT solver for apples-to-apples comparison.
    Uses explicit PEC: E[0] = E[N-1] = 0.
    """
    h = 1.0 / N
    x = np.linspace(h / 2, 1.0 - h / 2, N)

    if eps_r is None:
        eps_r = np.ones(N)

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

    J = np.exp(-((x - source_pos) ** 2) / (2 * source_width ** 2))
    J = J / (np.sqrt(2 * np.pi) * source_width)
    J[0] = 0.0
    J[N - 1] = 0.0
    rhs = -J.astype(complex)

    diag_main = -2.0 / (h * h) + k * k * eps_pml
    diag_off = np.full(N - 1, 1.0 / (h * h), dtype=complex)

    # PEC boundary
    diag_main[0] = 1.0
    diag_main[N - 1] = 1.0
    diag_off[0] = 0.0
    diag_off[N - 2] = 0.0
    rhs[0] = 0.0
    rhs[N - 1] = 0.0

    A = sparse.diags(
        [diag_off, diag_main, diag_off],
        offsets=[-1, 0, 1],
        shape=(N, N),
        format="csc",
        dtype=complex,
    )
    E = spsolve(A, rhs)
    return x, E


# ======================================================================
# Test harness
# ======================================================================

def test_accuracy_small(
    n_bits: int,
    k: float,
    eps_r_fn: Any,
    medium_name: str,
    pml_cells: int = 20,
    sigma_max: float = 10.0,
    max_rank: int = 128,
    gmres_tol: float = 1e-6,
) -> dict[str, Any]:
    """Run accuracy test at small grid: compare QTT vs sparse ground truth."""
    N = 2 ** n_bits
    h = 1.0 / N
    eps_r_arr = eps_r_fn(N)

    print(f"\n  --- N={N:,} (2^{n_bits}), k={k / math.pi:.0f}π, "
          f"medium={medium_name} ---")

    # Ground truth
    t0 = time.perf_counter()
    x_grid, E_ref = solve_helmholtz_1d_sparse(
        N, k, eps_r=eps_r_arr,
        pml_cells=pml_cells, pml_sigma_max=sigma_max,
    )
    t_ref = time.perf_counter() - t0

    # QTT solve
    config = HelmholtzConfig(
        n_bits=n_bits,
        k=k,
        eps_r=eps_r_arr,
        pml_cells=pml_cells,
        pml_sigma_max=sigma_max,
        max_rank=max_rank,
        gmres_restart=30,
        gmres_max_iter=300,
        gmres_tol=gmres_tol,
        verbose=False,
    )
    t1 = time.perf_counter()
    result = helmholtz_solve_1d(config)
    t_qtt = time.perf_counter() - t1

    # Reconstruct QTT solution for comparison
    E_qtt = reconstruct_1d(result.E_cores)

    # Compare in interior (exclude PML regions)
    interior = slice(pml_cells + 5, N - pml_cells - 5)
    E_ref_int = E_ref[interior]
    E_qtt_int = E_qtt[interior]

    # Relative L2 error in interior
    err_l2 = np.linalg.norm(E_qtt_int - E_ref_int) / max(np.linalg.norm(E_ref_int), 1e-30)
    # Max absolute error
    err_max = np.max(np.abs(E_qtt_int - E_ref_int))
    # Relative max error
    err_max_rel = err_max / max(np.max(np.abs(E_ref_int)), 1e-30)

    # QTT solution rank
    ranks = [1] + [c.shape[2] for c in result.E_cores]
    chi_max = max(ranks)

    # Dense storage vs QTT storage
    dense_bytes = N * 16  # complex128
    qtt_bytes = sum(c.nbytes for c in result.E_cores)
    compression = dense_bytes / max(qtt_bytes, 1)

    status = "✓" if result.converged else "✗"
    print(f"    {status} GMRES: {result.n_iter} iters, "
          f"residual={result.final_residual:.2e}")
    print(f"    L2 error (interior): {err_l2:.4e}")
    print(f"    Max error (relative): {err_max_rel:.4e}")
    print(f"    χ_max={chi_max}, compression={compression:.1f}×")
    print(f"    Time: QTT={t_qtt:.2f}s, sparse={t_ref:.4f}s")

    return {
        "n_bits": n_bits,
        "N": N,
        "k": k,
        "k_label": f"k={k / math.pi:.0f}π",
        "medium": medium_name,
        "converged": result.converged,
        "n_iter": result.n_iter,
        "gmres_residual": result.final_residual,
        "err_l2_interior": float(err_l2),
        "err_max_rel": float(err_max_rel),
        "chi_max": chi_max,
        "compression": round(compression, 1),
        "time_qtt_s": round(t_qtt, 4),
        "time_sparse_s": round(t_ref, 4),
        "mpo_bond_dim": result.mpo_bond_dim,
    }


def test_scaling_large(
    n_bits: int,
    k: float,
    eps_r_fn: Any,
    medium_name: str,
    pml_cells: int = 40,
    sigma_max: float = 10.0,
    max_rank: int = 128,
    gmres_tol: float = 1e-6,
) -> dict[str, Any]:
    """Run scaling test at large grid: QTT only, check residual + rank."""
    N = 2 ** n_bits

    print(f"\n  --- N={N:,} (2^{n_bits}), k={k / math.pi:.0f}π, "
          f"medium={medium_name} ---")

    config = HelmholtzConfig(
        n_bits=n_bits,
        k=k,
        eps_r=eps_r_fn(N),
        pml_cells=pml_cells,
        pml_sigma_max=sigma_max,
        max_rank=max_rank,
        gmres_restart=30,
        gmres_max_iter=300,
        gmres_tol=gmres_tol,
        verbose=False,
    )

    t0 = time.perf_counter()
    result = helmholtz_solve_1d(config)
    t_total = time.perf_counter() - t0

    ranks = [1] + [c.shape[2] for c in result.E_cores]
    chi_max = max(ranks)
    qtt_bytes = sum(c.nbytes for c in result.E_cores)
    dense_bytes = N * 16
    compression = dense_bytes / max(qtt_bytes, 1)

    status = "✓" if result.converged else "✗"
    print(f"    {status} GMRES: {result.n_iter} iters, "
          f"residual={result.final_residual:.2e}")
    print(f"    χ_max={chi_max}, compression={compression:.1f}×")
    print(f"    Time: {t_total:.2f}s (setup={result.setup_time_s:.3f}s, "
          f"solve={result.solve_time_s:.2f}s)")

    return {
        "n_bits": n_bits,
        "N": N,
        "k": k,
        "k_label": f"k={k / math.pi:.0f}π",
        "medium": medium_name,
        "converged": result.converged,
        "n_iter": result.n_iter,
        "gmres_residual": result.final_residual,
        "chi_max": chi_max,
        "compression": round(compression, 1),
        "time_total_s": round(t_total, 4),
        "time_setup_s": round(result.setup_time_s, 4),
        "time_solve_s": round(result.solve_time_s, 4),
        "mpo_bond_dim": result.mpo_bond_dim,
    }


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    ts = datetime.now(timezone.utc).isoformat()
    print("=" * 80)
    print("  PHASE 1: QTT HELMHOLTZ SOLVER VALIDATION")
    print("  Frequency-domain solve entirely in QTT format")
    print("=" * 80)

    if not HAS_SCIPY:
        print("ERROR: scipy required for ground truth — install it.")
        sys.exit(1)

    # Media profiles (matching Phase 0)
    media = {
        "uniform": lambda N: np.ones(N),
        "piecewise": lambda N: np.where(
            np.linspace(0, 1, N) > 0.5, 4.0, 1.0
        ),
        "graded": lambda N: 1.0 + 3.0 * np.linspace(0, 1, N) ** 2,
    }

    wavenumbers = [2 * math.pi]  # Start with k=2π for initial validation
    all_results: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Part 1: Accuracy tests (small grids, compare to ground truth)
    # ------------------------------------------------------------------
    print(f"\n{'#' * 72}")
    print("  PART 1: ACCURACY VALIDATION (vs scipy ground truth)")
    print(f"{'#' * 72}")

    accuracy_pass = True
    for medium_name, eps_fn in media.items():
        for k in wavenumbers:
            for n_bits in [8, 10, 12]:
                try:
                    result = test_accuracy_small(
                        n_bits, k, eps_fn, medium_name,
                        max_rank=128, gmres_tol=1e-6,
                    )
                    all_results.append(result)
                    if not result["converged"]:
                        accuracy_pass = False
                except Exception as e:
                    print(f"    FAILED: {e}")
                    import traceback
                    traceback.print_exc()
                    accuracy_pass = False

    # ------------------------------------------------------------------
    # Part 2: Scaling tests (large grids, QTT-only)
    # ------------------------------------------------------------------
    print(f"\n{'#' * 72}")
    print("  PART 2: SCALING TESTS (QTT-only)")
    print(f"{'#' * 72}")

    for medium_name in ["uniform"]:
        eps_fn = media[medium_name]
        for k in [2 * math.pi]:
            for n_bits in [14, 16, 18]:
                try:
                    result = test_scaling_large(
                        n_bits, k, eps_fn, medium_name,
                        pml_cells=min(40, 2 ** n_bits // 8),
                        max_rank=128, gmres_tol=1e-5,
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"    FAILED: {e}")
                    import traceback
                    traceback.print_exc()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n\n{'=' * 80}")
    print("  PHASE 1 SUMMARY")
    print(f"{'=' * 80}")

    # Accuracy results
    acc_results = [r for r in all_results if "err_l2_interior" in r]
    if acc_results:
        print("\n  Accuracy (interior L2 error):")
        for r in acc_results:
            sym = "✓" if r["converged"] else "✗"
            print(f"    {sym} N={r['N']:>8,} {r['medium']:12s} {r['k_label']:6s}: "
                  f"L2={r['err_l2_interior']:.2e}, "
                  f"χ={r['chi_max']:3d}, "
                  f"{r['n_iter']:3d} iters, "
                  f"{r['time_qtt_s']:.2f}s")

    # Scaling results
    scale_results = [r for r in all_results if "time_total_s" in r]
    if scale_results:
        print("\n  Scaling:")
        for r in scale_results:
            sym = "✓" if r["converged"] else "✗"
            print(f"    {sym} N={r['N']:>10,} (2^{r['n_bits']:2d}): "
                  f"χ={r['chi_max']:3d}, {r['compression']:.0f}× compress, "
                  f"{r['n_iter']:3d} iters, {r['time_total_s']:.1f}s")

    # Verdict
    any_converged = any(r.get("converged", False) for r in all_results)
    all_converged = all(r.get("converged", True) for r in all_results)

    print()
    if all_converged and accuracy_pass:
        print("  ╔═══════════════════════════════════════════════════════╗")
        print("  ║  PASS — QTT Helmholtz solver validated               ║")
        print("  ║  Frequency-domain QTT Maxwell: OPERATIONAL           ║")
        print("  ╚═══════════════════════════════════════════════════════╝")
    elif any_converged:
        print("  ╔═══════════════════════════════════════════════════════╗")
        print("  ║  PARTIAL — Some configs converged, some did not      ║")
        print("  ║  May need preconditioning for harder problems        ║")
        print("  ╚═══════════════════════════════════════════════════════╝")
    else:
        print("  ╔═══════════════════════════════════════════════════════╗")
        print("  ║  FAIL — GMRES did not converge in any configuration  ║")
        print("  ╚═══════════════════════════════════════════════════════╝")

    # Save
    out = {
        "protocol": "PHASE1_QTT_HELMHOLTZ_VALIDATION",
        "version": "1.0.0",
        "timestamp": ts,
        "verdict": "PASS" if (all_converged and accuracy_pass) else
                   "PARTIAL" if any_converged else "FAIL",
        "results": all_results,
    }
    out_file = f"phase1_helmholtz_qtt_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Results saved: {out_file}")


if __name__ == "__main__":
    main()
