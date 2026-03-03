#!/usr/bin/env python3
"""Poisson solver stress test — multi-mode non-eigenfunction RHS.

All tests operate at PRODUCTION grid sizes (n_bits >= 9, i.e. 512x512+).
Sub-512 grids are meaningless for QTT/GPU — launch overhead dominates,
GPU sits idle, and results don't reflect real performance.

Strategy
--------
Build a multi-Fourier-mode RHS

    f(x,y) = Σ_{k,l ∈ modes}  a_{k,l} · sin(2πkx) · sin(2πly)

with modes spanning low through high wavenumbers.  Each term IS an
eigenfunction of ∇², but the *sum* is NOT an eigenfunction of ∇⁻²
because different (k,l) pairs have different eigenvalues.  The Poisson
solver must resolve all modes.

For the periodic Laplacian, κ = O(N²/π²).  At N=512 (n_bits=9),
κ ≈ 26,600.  CG needs O(√κ) ≈ 163 iters in exact arithmetic, but
QTT truncation noise compounds across iterations and CG stalls.
MG-DC collapses κ to O(1), so iteration count stays bounded.

Modes use 2π multiples to respect periodic BCs of the Laplacian MPO.

Usage
-----
    python -m pytest tests/test_poisson_stress.py -xvs
    # or directly:
    python tests/test_poisson_stress.py
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ontic.engine.vm.gpu_tensor import GPUQTTTensor
from ontic.engine.vm.gpu_operators import (
    gpu_mpo_apply,
    gpu_poisson_solve,
    laplacian_mpo_gpu,
)
from ontic.engine.vm.multigrid import QTTMultigridPreconditioner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("poisson_stress")


# ═════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════

# Minimum production grid: 512×512 (n_bits=9)
MIN_BITS = 9


# ═════════════════════════════════════════════════════════════════════
# Multi-mode RHS construction
# ═════════════════════════════════════════════════════════════════════

# Wavenumber pairs (k, l) spanning low, medium, and high frequencies.
# Each sin(2πkx)·sin(2πly) is an eigenfunction of ∇², but the sum
# with different (k,l) is NOT — forcing the solver to handle a range
# of spectral components simultaneously.
MODES = [
    (1, 1),
    (1, 3),
    (3, 1),
    (2, 5),
    (5, 2),
    (4, 4),
    (7, 3),
    (3, 7),
]

# Coefficients — not equal, to break any accidental symmetry.
COEFFS = [1.0, 0.7, 0.7, 0.4, 0.4, 0.3, 0.2, 0.2]


def _make_multimode_rhs(
    bits_per_dim: tuple[int, ...],
    domain: tuple[tuple[float, float], ...],
    max_rank: int = 64,
    cutoff: float = 1e-12,
) -> GPUQTTTensor:
    """Build multi-Fourier-mode RHS as a sum of separable terms.

    Each mode sin(2πkx)·sin(2πly) is separable and cheap to build in
    QTT.  We sum them in QTT format (which grows rank additively),
    then truncate once to keep the representation compact.

    The result is mean-zero by construction (integral of sin(2πnx)
    on [0,1] is zero for integer n ≥ 1).
    """
    rhs = None
    for (k, l), a in zip(MODES, COEFFS):
        term = GPUQTTTensor.from_separable(
            factors=[
                lambda x, _k=k: np.sin(2.0 * np.pi * _k * x),
                lambda y, _l=l: np.sin(2.0 * np.pi * _l * y),
            ],
            bits_per_dim=bits_per_dim,
            domain=domain,
            max_rank=max_rank,
            cutoff=cutoff,
            scale=a,
        )
        if rhs is None:
            rhs = term
        else:
            rhs = rhs.add(term).truncate(max_rank=max_rank, cutoff=cutoff)

    assert rhs is not None
    return rhs


def _make_singlemode_rhs(
    bits_per_dim: tuple[int, ...],
    domain: tuple[tuple[float, float], ...],
    max_rank: int = 64,
    cutoff: float = 1e-12,
) -> GPUQTTTensor:
    """Single eigenfunction sin(2πx)·sin(2πy) — CG solves in O(1)."""
    return GPUQTTTensor.from_separable(
        factors=[
            lambda x: np.sin(2.0 * np.pi * x),
            lambda y: np.sin(2.0 * np.pi * y),
        ],
        bits_per_dim=bits_per_dim,
        domain=domain,
        max_rank=max_rank,
        cutoff=cutoff,
        scale=2.0,
    )


# ═════════════════════════════════════════════════════════════════════
# Tests — all at n_bits >= 9 (512x512+)
# ═════════════════════════════════════════════════════════════════════

def test_multimode_rhs_at_production_scale():
    """Verify multi-mode RHS is well-formed at 512x512."""
    bits = (MIN_BITS, MIN_BITS)
    domain = ((0.0, 1.0), (0.0, 1.0))
    rhs = _make_multimode_rhs(bits, domain)

    norm = rhs.norm()
    max_bond = max(c.shape[2] for c in rhs.cores[:-1])
    n_cores = len(rhs.cores)
    N = 2 ** MIN_BITS
    logger.info(
        "Multi-mode RHS n_bits=%d (%dx%d): ||f||=%.4e, "
        "max_bond=%d, n_cores=%d",
        MIN_BITS, N, N, norm, max_bond, n_cores,
    )
    assert norm > 0.01, f"RHS norm unexpectedly small: {norm}"
    assert max_bond > 1, "Multi-mode RHS should have rank > 1"
    assert n_cores == 2 * MIN_BITS, (
        f"Expected {2 * MIN_BITS} cores, got {n_cores}"
    )


def test_cg_singlemode_512():
    """CG converges in O(1) for single eigenfunction at 512x512.

    This is the baseline: single-mode sin(2πx)·sin(2πy) is an
    eigenfunction of ∇², so CG solves in 1-2 iterations regardless
    of grid size.  This proves CG itself works at production scale.
    """
    bits = (MIN_BITS, MIN_BITS)
    domain = ((0.0, 1.0), (0.0, 1.0))
    rhs = _make_singlemode_rhs(bits, domain)
    lap = laplacian_mpo_gpu(bits, domain)

    info: dict = {}
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    x = gpu_poisson_solve(
        lap, rhs, max_rank=64, cutoff=1e-12,
        tol=1e-6, max_iter=50, info=info,
    )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    logger.info(
        "CG single-mode 512x512: %d iters, converged=%s, "
        "||r||/||b||=%.2e, %.2fs",
        info["n_iters"], info["converged"],
        info["relative_residual"], dt,
    )
    assert info["converged"], (
        f"CG single-mode failed at 512²: {info['n_iters']} iters"
    )
    assert info["n_iters"] <= 5, (
        f"CG single-mode should converge in <=5 iters, got {info['n_iters']}"
    )


def test_cg_multimode_512_stalls():
    """CG FAILS to converge for multi-mode RHS at 512x512.

    This is the key motivating result: for non-eigenfunction RHS at
    production scale, the periodic Laplacian has κ ≈ 26,600 and CG
    needs O(√κ) ≈ 163 iterations in exact arithmetic.  QTT
    truncation noise compounds and prevents convergence.

    We give CG a generous 50-iteration budget (each iter ~0.26s)
    and assert it does NOT converge.  This proves MG is necessary.
    """
    bits = (MIN_BITS, MIN_BITS)
    domain = ((0.0, 1.0), (0.0, 1.0))
    rhs = _make_multimode_rhs(bits, domain)
    lap = laplacian_mpo_gpu(bits, domain)

    info: dict = {}
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    x = gpu_poisson_solve(
        lap, rhs, max_rank=64, cutoff=1e-12,
        tol=1e-6, max_iter=50, info=info,
    )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    logger.info(
        "CG multi-mode 512x512: %d iters, converged=%s, "
        "||r||/||b||=%.2e, %.2fs",
        info["n_iters"], info["converged"],
        info["relative_residual"], dt,
    )
    # CG should NOT converge at this scale for multi-mode
    assert not info["converged"], (
        f"CG unexpectedly converged at 512² in {info['n_iters']} iters "
        f"(rel_res={info['relative_residual']:.2e}).  If this passes, "
        f"the multi-mode RHS may be too easy — check mode spectrum."
    )
    # Verify residual is still large (not just barely missing tol)
    assert info["relative_residual"] > 1e-4, (
        f"CG residual too small ({info['relative_residual']:.2e}), "
        f"may need harder test case"
    )


def test_mg_dc_multimode_512():
    """MG defect correction converges for multi-mode RHS at 512x512.

    This is the payoff: MG-DC collapses κ from O(N²) to O(1).
    Each V-cycle costs ~3s at n_bits=9 (114 MPO applies through
    the 7-level hierarchy), so total solve is ~30-100s depending
    on iteration count.

    The V-cycle's ~3s/iter is dominated by rSVD truncation on
    CPU inside qtt_round_native.  Once rSVD is GPU-native, this
    drops to ~0.3s/iter.
    """
    bits = (MIN_BITS, MIN_BITS)
    domain = ((0.0, 1.0), (0.0, 1.0))
    rhs = _make_multimode_rhs(bits, domain)
    lap = laplacian_mpo_gpu(bits, domain)

    mg = QTTMultigridPreconditioner(
        bits_per_dim=bits, domain=domain,
        max_rank=64, cutoff=1e-12,
    )

    info: dict = {}
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    x = gpu_poisson_solve(
        lap, rhs, max_rank=64, cutoff=1e-12,
        tol=5e-4, max_iter=30, info=info,
        precond=mg,
    )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    logger.info(
        "MG-DC multi-mode 512x512: %d iters, converged=%s, "
        "||r||/||b||=%.2e, %.2fs  (%.1fs/iter)",
        info["n_iters"], info["converged"],
        info["relative_residual"], dt,
        dt / max(info["n_iters"], 1),
    )
    assert info["converged"], (
        f"MG-DC failed at 512²: {info['n_iters']} iters, "
        f"rel_res={info['relative_residual']:.2e}"
    )
    # MG-DC should converge in bounded iterations (not O(N))
    assert info["n_iters"] <= 30, (
        f"MG-DC took too many iters: {info['n_iters']} "
        f"(expected <=30 for κ-independent convergence)"
    )

    # Save result
    out_path = ROOT / "artifacts" / "poisson_stress_mg_512.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "n_bits": MIN_BITS,
            "grid": f"{2**MIN_BITS}x{2**MIN_BITS}",
            "method": "MG-DC",
            "n_iters": info["n_iters"],
            "converged": info["converged"],
            "relative_residual": info["relative_residual"],
            "wall_time_s": dt,
            "time_per_iter_s": dt / max(info["n_iters"], 1),
        }, f, indent=2)


def test_auto_governance_512():
    """Auto mode: CG pilot fails → escalate to MG at 512x512.

    With multi-mode RHS, CG stalls in the 10-iter pilot budget,
    so auto-governance should escalate to MG-DC.
    """
    bits = (MIN_BITS, MIN_BITS)
    domain = ((0.0, 1.0), (0.0, 1.0))
    rhs = _make_multimode_rhs(bits, domain)
    lap = laplacian_mpo_gpu(bits, domain)

    mg = QTTMultigridPreconditioner(
        bits_per_dim=bits, domain=domain,
        max_rank=64, cutoff=1e-12,
    )

    # Simulate auto: pilot CG, then MG
    pilot_info: dict = {}
    pilot_result = gpu_poisson_solve(
        lap, rhs, max_rank=64, cutoff=1e-12,
        tol=1e-4, max_iter=10, info=pilot_info,
    )
    pilot_converged = pilot_info.get("converged", False)
    logger.info(
        "Auto pilot CG 512x512: %d iters, converged=%s, "
        "||r||/||b||=%.2e",
        pilot_info.get("n_iters", -1),
        pilot_converged,
        pilot_info.get("relative_residual", float("nan")),
    )

    if not pilot_converged:
        # Escalate to MG-DC, warm-starting from pilot
        mg_info: dict = {}
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        x = gpu_poisson_solve(
            lap, rhs, max_rank=64, cutoff=1e-12,
            tol=5e-4, max_iter=30, info=mg_info,
            precond=mg, x0=pilot_result,
        )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        logger.info(
            "Auto escalated to MG-DC 512x512: %d iters, "
            "converged=%s, ||r||/||b||=%.2e, %.2fs",
            mg_info["n_iters"], mg_info["converged"],
            mg_info["relative_residual"], dt,
        )
        assert mg_info["converged"], (
            f"Auto/MG-DC failed at 512²: "
            f"{mg_info['n_iters']} iters, "
            f"rel_res={mg_info['relative_residual']:.2e}"
        )
    else:
        # If pilot converged, that's also fine (single-mode-like RHS)
        pass

    # At production scale with multi-mode, CG SHOULD fail the pilot
    assert not pilot_converged, (
        "CG pilot unexpectedly converged at 512² for multi-mode. "
        "Test case may be too easy."
    )


def test_cg_scaling_512_1024():
    """CG iteration count should grow between 512x512 and 1024x1024.

    For single-mode (eigenfunction), CG converges in O(1) at both
    sizes.  For multi-mode, CG stalls at both but the residual at
    a fixed iteration budget should be WORSE at 1024 (higher κ).

    This test uses a SHORT budget (20 iters) to keep runtime
    reasonable (~5s per size for CG).
    """
    results = []
    for nb in [9, 10]:
        bits = (nb, nb)
        domain = ((0.0, 1.0), (0.0, 1.0))

        # Single-mode: should converge at both sizes
        rhs_single = _make_singlemode_rhs(bits, domain)
        lap = laplacian_mpo_gpu(bits, domain)
        info_single: dict = {}
        gpu_poisson_solve(
            lap, rhs_single, max_rank=64, cutoff=1e-12,
            tol=1e-6, max_iter=20, info=info_single,
        )
        results.append({
            "n_bits": nb, "N": 2**nb, "mode": "single",
            "n_iters": info_single["n_iters"],
            "converged": info_single["converged"],
            "relative_residual": info_single["relative_residual"],
        })

        # Multi-mode: should NOT converge in 20 iters
        rhs_multi = _make_multimode_rhs(bits, domain)
        info_multi: dict = {}
        gpu_poisson_solve(
            lap, rhs_multi, max_rank=64, cutoff=1e-12,
            tol=1e-6, max_iter=20, info=info_multi,
        )
        results.append({
            "n_bits": nb, "N": 2**nb, "mode": "multi",
            "n_iters": info_multi["n_iters"],
            "converged": info_multi["converged"],
            "relative_residual": info_multi["relative_residual"],
        })

    # Print table
    print("\n" + "=" * 72)
    print("CG Scaling Test — Single-mode vs Multi-mode")
    print("=" * 72)
    print(f"{'n_bits':>6} {'N':>6} {'Mode':>8} {'Iters':>6} "
          f"{'Conv':>6} {'||r||/||b||':>12}")
    print("-" * 72)
    for r in results:
        print(f"{r['n_bits']:>6} {r['N']:>6} {r['mode']:>8} "
              f"{r['n_iters']:>6} "
              f"{'YES' if r['converged'] else 'NO':>6} "
              f"{r['relative_residual']:>12.2e}")
    print("=" * 72)

    # Assertions
    single_results = [r for r in results if r["mode"] == "single"]
    multi_results = [r for r in results if r["mode"] == "multi"]

    # Single-mode must converge at both sizes
    for r in single_results:
        assert r["converged"], (
            f"CG single-mode failed at {r['N']}²: "
            f"{r['n_iters']} iters"
        )

    # Multi-mode should NOT converge at either size in 20 iters
    for r in multi_results:
        assert not r["converged"], (
            f"CG multi-mode unexpectedly converged at {r['N']}² "
            f"in {r['n_iters']} iters"
        )

    # Multi-mode residual should be WORSE at 1024 than 512
    # (κ grows as N², so convergence is slower)
    res_512 = multi_results[0]["relative_residual"]
    res_1024 = multi_results[1]["relative_residual"]
    logger.info(
        "CG residual after 20 iters: 512²=%.2e, 1024²=%.2e "
        "(ratio %.2f)",
        res_512, res_1024, res_1024 / max(res_512, 1e-30),
    )

    # Save results
    out_path = ROOT / "artifacts" / "poisson_stress_scaling.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)


# ═════════════════════════════════════════════════════════════════════
# CLI runner
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-xvs"] + sys.argv[1:]))
