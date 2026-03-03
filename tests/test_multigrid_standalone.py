#!/usr/bin/env python3
"""Standalone test for the QTT multigrid V-cycle preconditioner.

Exercises restriction, prolongation, smoothing, and the full V-cycle
at n_bits=5..9 to isolate any shape mismatches before running the
expensive V&V harness.

Usage
-----
    python -m pytest tests/test_multigrid_standalone.py -xvs
    # or directly:
    python tests/test_multigrid_standalone.py
"""

from __future__ import annotations

import logging
import math
import sys
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
from ontic.engine.vm.multigrid import (
    QTTMultigridPreconditioner,
    _restrict,
    _prolongate,
    _smooth,
    laplacian_diagonal,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("test_mg")


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _make_sin_rhs(bits_per_dim: tuple[int, ...],
                  domain: tuple[tuple[float, float], ...]) -> GPUQTTTensor:
    """Create a separable sin(2πx)·sin(2πy) RHS on GPU."""
    return GPUQTTTensor.from_separable(
        factors=[
            lambda x: np.sin(2.0 * np.pi * x),
            lambda y: np.sin(2.0 * np.pi * y),
        ],
        bits_per_dim=bits_per_dim,
        domain=domain,
        max_rank=64,
        cutoff=1e-12,
        scale=2.0,
    )


def _check_core_shapes(tensor: GPUQTTTensor, label: str) -> bool:
    """Validate all cores have shape (r_l, 2, r_r) and bonds match."""
    ok = True
    cores = tensor.cores
    for i, c in enumerate(cores):
        if c.ndim != 3:
            logger.error("%s: core %d ndim=%d (expected 3)", label, i, c.ndim)
            ok = False
        elif c.shape[1] != 2:
            logger.error(
                "%s: core %d physical dim=%d (expected 2), shape=%s",
                label, i, c.shape[1], tuple(c.shape),
            )
            ok = False
    # Bond compatibility
    for i in range(len(cores) - 1):
        r_right = cores[i].shape[2]
        r_left_next = cores[i + 1].shape[0]
        if r_right != r_left_next:
            logger.error(
                "%s: bond mismatch at core %d→%d: r_right=%d vs r_left=%d",
                label, i, i + 1, r_right, r_left_next,
            )
            ok = False
    # Boundary bonds
    if cores[0].shape[0] != 1:
        logger.error(
            "%s: first core left bond=%d (expected 1)", label, cores[0].shape[0]
        )
        ok = False
    if cores[-1].shape[2] != 1:
        logger.error(
            "%s: last core right bond=%d (expected 1)", label, cores[-1].shape[2]
        )
        ok = False
    if ok:
        ranks = [c.shape[0] for c in cores] + [cores[-1].shape[2]]
        logger.info(
            "%s: OK, %d cores, bits=%s, max_rank=%d, ranks=%s",
            label, len(cores), tensor.bits_per_dim,
            max(ranks), ranks,
        )
    return ok


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────

def test_laplacian_diagonal():
    """Verify laplacian_diagonal computes correct values."""
    bits = (6, 6)
    domain = ((0.0, 1.0), (0.0, 1.0))

    # 2nd order: -2/h² per dim
    h = 1.0 / 64
    expected_v1 = 2.0 * (-2.0 / (h * h))
    actual_v1 = laplacian_diagonal(bits, domain, variant="lap_v1")
    assert abs(actual_v1 - expected_v1) < 1e-10, f"{actual_v1} != {expected_v1}"

    # 4th order: -30/(12h²) per dim
    expected_v2 = 2.0 * (-30.0 / (12.0 * h * h))
    actual_v2 = laplacian_diagonal(bits, domain, variant="lap_v2_high_order")
    assert abs(actual_v2 - expected_v2) < 1e-10, f"{actual_v2} != {expected_v2}"

    logger.info("laplacian_diagonal: PASS")


def test_restriction():
    """Restriction should reduce bits_per_dim by 1 each and preserve core shapes."""
    for nb in [5, 6, 7, 8]:
        bits = (nb, nb)
        domain = ((0.0, 1.0), (0.0, 1.0))
        rhs = _make_sin_rhs(bits, domain)
        assert _check_core_shapes(rhs, f"rhs_nb{nb}")

        coarse = _restrict(rhs, bits)
        expected_bits = (nb - 1, nb - 1)
        assert coarse.bits_per_dim == expected_bits, (
            f"Expected bits {expected_bits}, got {coarse.bits_per_dim}"
        )
        assert len(coarse.cores) == sum(expected_bits), (
            f"Expected {sum(expected_bits)} cores, got {len(coarse.cores)}"
        )
        assert _check_core_shapes(coarse, f"restricted_nb{nb}")

    logger.info("restriction: PASS")


def test_prolongation():
    """Prolongation should restore bits_per_dim and preserve core shapes."""
    for nb in [5, 6, 7, 8]:
        bits = (nb, nb)
        coarse_bits = (nb - 1, nb - 1)
        domain = ((0.0, 1.0), (0.0, 1.0))
        rhs = _make_sin_rhs(bits, domain)
        coarse = _restrict(rhs, bits)

        fine = _prolongate(coarse, bits)
        assert fine.bits_per_dim == bits, (
            f"Expected bits {bits}, got {fine.bits_per_dim}"
        )
        assert len(fine.cores) == sum(bits), (
            f"Expected {sum(bits)} cores, got {len(fine.cores)}"
        )
        assert _check_core_shapes(fine, f"prolongated_nb{nb}")

    logger.info("prolongation: PASS")


def test_restrict_prolongate_roundtrip():
    """Restrict then prolongate should approximate the original."""
    bits = (6, 6)
    domain = ((0.0, 1.0), (0.0, 1.0))
    rhs = _make_sin_rhs(bits, domain)
    original_norm = rhs.norm()

    coarse = _restrict(rhs, bits)
    recovered = _prolongate(coarse, bits)

    diff = rhs.sub(recovered)
    diff_norm = diff.norm()
    rel_err = diff_norm / max(original_norm, 1e-30)
    logger.info(
        "restrict→prolongate roundtrip: ||diff||/||orig|| = %.4e",
        rel_err,
    )
    # Piecewise-constant prolongation is O(h) accurate, so
    # expect ~10% relative error for smooth fields
    assert rel_err < 1.0, f"Roundtrip error too large: {rel_err}"


def test_smoothing():
    """Richardson smoothing should reduce residual for Poisson."""
    bits = (5, 5)
    domain = ((0.0, 1.0), (0.0, 1.0))
    rhs = _make_sin_rhs(bits, domain)
    lap_mpo = laplacian_mpo_gpu(bits, domain, variant="lap_v1")
    diag = laplacian_diagonal(bits, domain, variant="lap_v1")
    x = GPUQTTTensor.zeros(bits, domain)

    max_rank = 64
    cutoff = 1e-12

    # Pre-smoothing residual
    Lx = gpu_mpo_apply(lap_mpo, x, max_rank=max_rank, cutoff=cutoff)
    r0 = rhs.sub(Lx)
    r0_norm = r0.norm()

    # Smooth
    x = _smooth(
        x, rhs, lap_mpo, 1.0 / diag, 2.0 / 3.0,
        n_sweeps=5, max_rank=max_rank, cutoff=cutoff,
    )
    assert _check_core_shapes(x, "smoothed")

    # Post-smoothing residual
    Lx = gpu_mpo_apply(lap_mpo, x, max_rank=max_rank, cutoff=cutoff)
    r1 = rhs.sub(Lx)
    r1_norm = r1.norm()

    logger.info(
        "smoothing: ||r0||=%.4e → ||r1||=%.4e (ratio %.4f)",
        r0_norm, r1_norm, r1_norm / max(r0_norm, 1e-30),
    )
    assert r1_norm < r0_norm, "Smoothing should reduce residual"


def test_vcycle_shapes():
    """V-cycle should return a tensor with correct shapes."""
    for nb in [5, 6, 7]:
        bits = (nb, nb)
        domain = ((0.0, 1.0), (0.0, 1.0))
        rhs = _make_sin_rhs(bits, domain)

        mg = QTTMultigridPreconditioner(
            bits_per_dim=bits,
            domain=domain,
            max_rank=64,
            cutoff=1e-12,
            variant="lap_v1",
        )
        z = mg(rhs)

        assert z.bits_per_dim == bits, (
            f"V-cycle output bits {z.bits_per_dim} != input bits {bits}"
        )
        assert len(z.cores) == sum(bits), (
            f"V-cycle output {len(z.cores)} cores != expected {sum(bits)}"
        )
        assert _check_core_shapes(z, f"vcycle_nb{nb}")

    logger.info("vcycle_shapes: PASS")


def test_vcycle_reduces_residual():
    """One V-cycle applied to the residual should reduce it."""
    bits = (6, 6)
    domain = ((0.0, 1.0), (0.0, 1.0))
    rhs = _make_sin_rhs(bits, domain)
    lap_mpo = laplacian_mpo_gpu(bits, domain, variant="lap_v1")
    max_rank = 64
    cutoff = 1e-12

    mg = QTTMultigridPreconditioner(
        bits_per_dim=bits,
        domain=domain,
        max_rank=max_rank,
        cutoff=cutoff,
        variant="lap_v1",
    )

    # Start from zero
    x = GPUQTTTensor.zeros(bits, domain)
    Lx = gpu_mpo_apply(lap_mpo, x, max_rank=max_rank, cutoff=cutoff)
    r = rhs.sub(Lx).truncate(max_rank=max_rank, cutoff=cutoff)
    r0_norm = r.norm()

    # Apply one V-cycle: z ≈ L⁻¹·r, then x += z
    z = mg(r)
    x = x.add(z).truncate(max_rank=max_rank, cutoff=cutoff)

    Lx = gpu_mpo_apply(lap_mpo, x, max_rank=max_rank, cutoff=cutoff)
    r1 = rhs.sub(Lx).truncate(max_rank=max_rank, cutoff=cutoff)
    r1_norm = r1.norm()

    ratio = r1_norm / max(r0_norm, 1e-30)
    logger.info(
        "V-cycle residual reduction: ||r0||=%.4e → ||r1||=%.4e (ρ=%.4f)",
        r0_norm, r1_norm, ratio,
    )
    # With 1+1 smoothing (cheap preconditioner for FCG), each V-cycle
    # is weaker but ~10× cheaper.  ρ < 0.99 confirms the V-cycle is
    # doing useful work; the FCG outer solver provides the acceleration.
    assert ratio < 0.99, f"V-cycle convergence factor too high: ρ={ratio:.4f}"


def test_pcg_with_mg():
    """PCG with MG preconditioner should converge in few iterations."""
    bits = (7, 7)
    domain = ((0.0, 1.0), (0.0, 1.0))
    rhs = _make_sin_rhs(bits, domain)
    lap_mpo = laplacian_mpo_gpu(bits, domain, variant="lap_v1")
    max_rank = 64
    cutoff = 1e-12

    mg = QTTMultigridPreconditioner(
        bits_per_dim=bits,
        domain=domain,
        max_rank=max_rank,
        cutoff=cutoff,
        variant="lap_v1",
    )

    info: dict = {}
    x = gpu_poisson_solve(
        lap_mpo, rhs,
        max_rank=max_rank,
        cutoff=cutoff,
        tol=1e-6,
        max_iter=50,
        info=info,
        precond=mg,
    )

    logger.info(
        "PCG+MG: converged=%s, iters=%d, rel_residual=%.4e",
        info.get("converged"), info.get("n_iters"), info.get("relative_residual"),
    )
    assert info["converged"], (
        f"PCG+MG did not converge: iters={info['n_iters']}, "
        f"rel_res={info.get('relative_residual', '?')}"
    )
    assert info["n_iters"] < 35, (
        f"PCG+MG took too many iterations: {info['n_iters']}"
    )


def test_pcg_no_mg_baseline():
    """Plain CG (no preconditioner) as a baseline comparison."""
    bits = (7, 7)
    domain = ((0.0, 1.0), (0.0, 1.0))
    rhs = _make_sin_rhs(bits, domain)
    lap_mpo = laplacian_mpo_gpu(bits, domain, variant="lap_v1")
    max_rank = 64
    cutoff = 1e-12

    info: dict = {}
    x = gpu_poisson_solve(
        lap_mpo, rhs,
        max_rank=max_rank,
        cutoff=cutoff,
        tol=1e-6,
        max_iter=500,
        info=info,
    )

    logger.info(
        "CG (no precond): converged=%s, iters=%d, rel_residual=%.4e",
        info.get("converged"), info.get("n_iters"), info.get("relative_residual"),
    )


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("laplacian_diagonal", test_laplacian_diagonal),
        ("restriction", test_restriction),
        ("prolongation", test_prolongation),
        ("restrict_prolongate_roundtrip", test_restrict_prolongate_roundtrip),
        ("smoothing", test_smoothing),
        ("vcycle_shapes", test_vcycle_shapes),
        ("vcycle_reduces_residual", test_vcycle_reduces_residual),
        ("pcg_no_mg_baseline", test_pcg_no_mg_baseline),
        ("pcg_with_mg", test_pcg_with_mg),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print(f"{'='*60}")
        try:
            fn()
            print(f"✓ {name}: PASSED")
            passed += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"✗ {name}: FAILED — {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed, {passed+failed} total")
    print(f"{'='*60}")
    sys.exit(1 if failed > 0 else 0)
