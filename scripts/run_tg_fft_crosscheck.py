#!/usr/bin/env python3
"""Scenario 2 — Dense FFT cross-check: QTT vs dense @ 256².

Run ID: TG_XCHECK_256_QTT_vs_DENSEFDFFT

Proves the QTT simulation tracks a high-accuracy dense reference for the
same discrete PDE (same grid, same 5-point periodic Laplacian), isolating
TT truncation and QTT Poisson floor effects.

The dense reference uses FFT diagonalization — an exact solve for the
discrete Laplacian, not a different PDE.

Note on grid size: This scenario runs at 256² (n_bits=8) by design.
The dense reference requires materializing N×N arrays in float64.
The 256² grid keeps this tractable (~0.5 GB) while giving meaningful
truncation error signal.  The n_bits ≥ 9 policy applies to performance
and production benchmarks; this is a discrete-exact reference cross-check.

Spec: docs/reports/NS2D_SCHEDULED_VV_SCENARIOS.md §2

Usage:
    python3 scripts/run_tg_fft_crosscheck.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import time
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("tg_fft_xcheck")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch

from physics_os.core.executor import ExecutionConfig, execute
from physics_os.core.physics_qoi import extract_physics_qoi

# ═══════════════════════════════════════════════════════════════════════
# Locked configuration — matches spec exactly
# ═══════════════════════════════════════════════════════════════════════
CASE_ID = "TG_XCHECK_256_QTT_vs_DENSEFDFFT"
N_BITS = 8
GRID = 2 ** N_BITS  # 256
VISCOSITY = 0.01
MAX_RANK = 64
TRUNC_TOL = 1e-10
PRODUCTION_TOL = 1e-3
POISSON_MAX_ITERS = 80

# dt = 4× the 512² dt (safe for 256² diffusion)
_h = 1.0 / GRID
DT = 4.0 * (0.25 * (1.0 / 512) ** 2 / (2.0 * VISCOSITY))  # 1.9073486328125e-04
T_FINAL = 0.05
N_STEPS = round(T_FINAL / DT)  # 262

# Pass criteria
REL_L2_OMEGA_MAX = 5e-3
ENSTROPHY_REL_DIFF_MAX = 1e-4
OMEGA_L2_REL_DIFF_MAX = 1e-4


# ═══════════════════════════════════════════════════════════════════════
# JSON helpers — compact series + full-precision QoIs
# ═══════════════════════════════════════════════════════════════════════

def _summarize_series(
    values: list[float],
    name: str,
    *,
    decimation: int = 10,
    precision: int = 8,
) -> dict[str, Any]:
    """Compress a per-step series into stats + decimated samples.

    Returns a compact dict with min/max/mean/std/first/last plus
    every *decimation*-th sample (always including first and last).
    """
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=np.float64)
    # Decimated indices: first, every Nth, last
    indices = sorted(set([0] + list(range(0, len(arr), decimation)) + [len(arr) - 1]))
    return {
        "n": len(arr),
        "min": round(float(np.min(arr)), precision),
        "max": round(float(np.max(arr)), precision),
        "mean": round(float(np.mean(arr)), precision),
        "std": round(float(np.std(arr)), precision),
        "first": round(float(arr[0]), precision),
        "last": round(float(arr[-1]), precision),
        "samples": {
            "decimation": decimation,
            "step_indices": indices,
            "values": [round(float(arr[i]), precision) for i in indices],
        },
    }


def _full_precision_qoi(result: Any) -> dict[str, float]:
    """Compute enstrophy & omega_l2 at full float64 from QTT inner product.

    Bypasses the 7-sig-fig rounding in extract_physics_qoi so the
    cross-check comparison uses maximum available precision.
    """
    omega = result.fields.get("omega")
    if omega is None:
        return {"enstrophy": float("nan"), "omega_l2": float("nan")}
    hx = omega.grid_spacing(0)
    hy = omega.grid_spacing(1) if omega.n_dims > 1 else 1.0
    dA = hx * hy
    omega_sq_integral = dA * omega.inner(omega)
    return {
        "enstrophy": float(0.5 * omega_sq_integral),
        "omega_l2": float(math.sqrt(omega_sq_integral)),
    }


# ═══════════════════════════════════════════════════════════════════════
# Taylor-Green initial condition on dense grid
# ═══════════════════════════════════════════════════════════════════════

def _taylor_green_omega_dense(N: int) -> np.ndarray:
    """ω₀(x,y) = 2·sin(2πx)·sin(2πy) on [0,1]² with N×N grid (node-centered).

    Uses the same grid convention as the QTT runtime:
        x_i = i·h, i = 0, …, N-1  (= np.linspace(0, 1, N, endpoint=False))
    """
    h = 1.0 / N
    x = np.arange(N) * h  # node centers — matches QTT linspace(0,1,N,endpoint=False)
    y = np.arange(N) * h
    X, Y = np.meshgrid(x, y, indexing="ij")
    omega = 2.0 * np.sin(2.0 * np.pi * X) * np.sin(2.0 * np.pi * Y)
    return omega


# ═══════════════════════════════════════════════════════════════════════
# Dense FFT Poisson solver for the 5-point periodic Laplacian
# ═══════════════════════════════════════════════════════════════════════

def _fft_poisson_eigenvalues(N: int) -> np.ndarray:
    """Eigenvalues of the 5-point periodic Laplacian on an N×N grid.

    (Δ_h ψ)_{i,j} = (ψ_{i+1,j} + ψ_{i-1,j} + ψ_{i,j+1} + ψ_{i,j-1} - 4ψ_{i,j}) / h²

    In Fourier: λ(kx,ky) = (2cos(2πkx/N) + 2cos(2πky/N) - 4) / h²
    """
    h = 1.0 / N
    kx = np.arange(N)
    ky = np.arange(N)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    lam = (2.0 * np.cos(2.0 * np.pi * KX / N)
           + 2.0 * np.cos(2.0 * np.pi * KY / N)
           - 4.0) / (h * h)
    return lam


def fft_poisson_solve(omega: np.ndarray) -> np.ndarray:
    """Solve Δ_h ψ = ω via FFT diagonalization (exact for discrete Laplacian).

    - Compute ω̂(k) via FFT
    - Set ψ̂(0,0) = 0  (zero-mean)
    - ψ̂(k) = ω̂(k) / λ(k)  for k ≠ (0,0)
    - Inverse FFT → ψ
    """
    N = omega.shape[0]
    lam = _fft_poisson_eigenvalues(N)
    omega_hat = np.fft.fft2(omega)

    psi_hat = np.zeros_like(omega_hat)
    # λ(0,0) = 0 → skip (zero-mean constraint: ψ̂(0,0) = 0)
    mask = np.abs(lam) > 1e-30
    psi_hat[mask] = omega_hat[mask] / lam[mask]

    psi = np.real(np.fft.ifft2(psi_hat))
    return psi


# ═══════════════════════════════════════════════════════════════════════
# Dense NS2D time stepper (explicit Euler, vorticity-streamfunction)
# ═══════════════════════════════════════════════════════════════════════

def _dense_gradient_periodic(f: np.ndarray, axis: int) -> np.ndarray:
    """Central-difference gradient with periodic BCs. ∂f/∂x = (f_{i+1} - f_{i-1}) / (2h)."""
    N = f.shape[0]
    h = 1.0 / N
    return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2.0 * h)


def _dense_laplacian_periodic(f: np.ndarray) -> np.ndarray:
    """5-point periodic Laplacian."""
    N = f.shape[0]
    h = 1.0 / N
    lap = (
        np.roll(f, -1, axis=0) + np.roll(f, 1, axis=0)
        + np.roll(f, -1, axis=1) + np.roll(f, 1, axis=1)
        - 4.0 * f
    ) / (h * h)
    return lap


def dense_ns2d_step(omega: np.ndarray, dt: float, nu: float) -> np.ndarray:
    """One explicit Euler step of vorticity-streamfunction NS2D.

    ω_t + u·∇ω = ν·Δω
    where u = (∂ψ/∂y, -∂ψ/∂x),  Δψ = ω

    Steps:
      1. Solve Poisson: Δψ = ω  (FFT exact)
      2. Compute velocity: u = ∂ψ/∂y, v = -∂ψ/∂x
      3. Compute advection: u·∇ω = u·∂ω/∂x + v·∂ω/∂y
      4. Compute diffusion: ν·Δω
      5. Forward Euler: ω_new = ω + dt·(-advection + diffusion)
    """
    # 1. Poisson solve (exact discrete)
    psi = fft_poisson_solve(omega)

    # 2. Velocity from streamfunction
    u_vel = _dense_gradient_periodic(psi, axis=1)    # ∂ψ/∂y
    v_vel = -_dense_gradient_periodic(psi, axis=0)   # -∂ψ/∂x

    # 3. Advection: u·∂ω/∂x + v·∂ω/∂y
    domega_dx = _dense_gradient_periodic(omega, axis=0)
    domega_dy = _dense_gradient_periodic(omega, axis=1)
    advection = u_vel * domega_dx + v_vel * domega_dy

    # 4. Diffusion
    diffusion = nu * _dense_laplacian_periodic(omega)

    # 5. Forward Euler update
    omega_new = omega + dt * (-advection + diffusion)
    return omega_new


def run_dense_reference() -> dict[str, Any]:
    """Run the full dense NS2D simulation and return QoIs."""
    logger.info("=" * 64)
    logger.info("DENSE REFERENCE: %d×%d, dt=%.6e, %d steps", GRID, GRID, DT, N_STEPS)
    logger.info("=" * 64)

    omega = _taylor_green_omega_dense(GRID)
    h = 1.0 / GRID
    dA = h * h

    t0 = time.perf_counter()
    for step in range(N_STEPS):
        omega = dense_ns2d_step(omega, DT, VISCOSITY)
        if (step + 1) % 50 == 0 or step == N_STEPS - 1:
            enst = 0.5 * dA * np.sum(omega ** 2)
            ol2 = math.sqrt(dA * np.sum(omega ** 2))
            logger.info("  step %d/%d  enstrophy=%.8e  omega_l2=%.8e",
                         step + 1, N_STEPS, enst, ol2)
    wall = time.perf_counter() - t0

    # Final QoIs
    enstrophy = 0.5 * dA * float(np.sum(omega ** 2))
    omega_l2 = math.sqrt(dA * float(np.sum(omega ** 2)))

    logger.info("Dense done: %.1fs  enstrophy=%.8e  omega_l2=%.8e", wall, enstrophy, omega_l2)

    return {
        "wall_time_s": round(wall, 2),
        "enstrophy": enstrophy,
        "omega_l2": omega_l2,
        "omega_field": omega,  # kept for field comparison, stripped before JSON
    }


# ═══════════════════════════════════════════════════════════════════════
# QTT run at 256²
# ═══════════════════════════════════════════════════════════════════════

def run_qtt(seed: int = 42) -> dict[str, Any]:
    """Execute the QTT NS2D solver at 256² and return QoIs + omega field."""
    logger.info("=" * 64)
    logger.info("QTT RUN: %d×%d, n_bits=%d, dt=%.6e, %d steps, seed=%d",
                GRID, GRID, N_BITS, DT, N_STEPS, seed)
    logger.info("=" * 64)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    params: dict[str, Any] = {
        "viscosity": VISCOSITY,
        "ic_type": "taylor_green",
        "ic_n_modes": 4,
        "poisson_precond": "mg",
        "poisson_tol": PRODUCTION_TOL,
        "poisson_max_iters": POISSON_MAX_ITERS,
    }

    config = ExecutionConfig(
        domain="navier_stokes_2d",
        n_bits=N_BITS,
        n_steps=N_STEPS,
        max_rank=MAX_RANK,
        truncation_tol=TRUNC_TOL,
        parameters=params,
        dt=DT,
    )

    t0 = time.perf_counter()
    result = execute(config)
    wall = time.perf_counter() - t0
    logger.info("QTT done: %.1fs, success=%s", wall, result.success)

    # Extract QoIs — full float64 precision from QTT inner product
    fp_qoi = _full_precision_qoi(result)
    enstrophy = fp_qoi["enstrophy"]
    omega_l2 = fp_qoi["omega_l2"]

    # Also get Poisson stats from the standard extractor
    ctx = {"n_bits": N_BITS, "n_steps": N_STEPS, "Re": 1.0 / VISCOSITY}
    qoi = extract_physics_qoi(result, "navier_stokes_2d", ctx)

    p_dict = qoi.get("poisson", {})
    poisson_max_res = p_dict.get("max_relative_residual", float("nan"))
    poisson_mean_iters = p_dict.get("mean_cg_iters", float("nan"))
    poisson_max_iters = p_dict.get("cg_iters_max", 0)

    probes = getattr(result, "probes", {})
    per_step_residual = probes.get("poisson_relative_residual", [])
    per_step_iters = probes.get("poisson_cg_iters", [])

    # Divergence
    div_rel = float("nan")
    try:
        from ontic.engine.vm.gpu_operators import GPUOperatorCache, gpu_mpo_apply

        psi = result.fields.get("psi")
        if psi is not None:
            cache = GPUOperatorCache()
            bpd = psi.bits_per_dim
            dom = psi.domain

            grad_x_mpo = cache.get_gradient(0, bpd, dom, variant="grad_v1")
            grad_y_mpo = cache.get_gradient(1, bpd, dom, variant="grad_v1")

            u = gpu_mpo_apply(grad_y_mpo, psi, max_rank=MAX_RANK, cutoff=TRUNC_TOL)
            v_pos = gpu_mpo_apply(grad_x_mpo, psi, max_rank=MAX_RANK, cutoff=TRUNC_TOL)

            du_dx = gpu_mpo_apply(grad_x_mpo, u, max_rank=MAX_RANK, cutoff=TRUNC_TOL)
            dv_pos_dy = gpu_mpo_apply(grad_y_mpo, v_pos, max_rank=MAX_RANK, cutoff=TRUNC_TOL)

            div_tt = du_dx.sub(dv_pos_dy).truncate(max_rank=MAX_RANK, cutoff=TRUNC_TOL)

            hx = psi.grid_spacing(0)
            hy = psi.grid_spacing(1)
            dA_div = hx * hy

            div_l2 = math.sqrt(max(dA_div * div_tt.inner(div_tt), 0.0))
            u_l2_sq = dA_div * u.inner(u)
            v_neg = v_pos.negate()
            v_l2_sq = dA_div * v_neg.inner(v_neg)
            vel_l2 = math.sqrt(max(u_l2_sq + v_l2_sq, 0.0))
            div_rel = div_l2 / vel_l2 if vel_l2 > 1e-30 else float("nan")
    except Exception as exc:
        logger.warning("Divergence computation failed: %s", exc)

    # Extract omega as dense field for cross-check comparison

    # QTT-EXCEPTION: Rule 5 — Decompression Kills QTT
    # Why: We need the dense omega field for field-level L2 comparison
    #      against the dense reference solver.  This is a V&V diagnostic,
    #      not part of the execution loop.
    # Cost: O(2^(2*n_bits)) = 64K elements materialized; negligible vs run.
    # Fix: Not applicable — field comparison requires dense access.
    omega_field = None
    try:
        omega_tt = result.fields.get("omega")
        if omega_tt is not None and hasattr(omega_tt, "to_dense"):
            omega_dense_flat = omega_tt.to_dense()
            if hasattr(omega_dense_flat, "cpu"):
                omega_dense_flat = omega_dense_flat.cpu().numpy()
            omega_field = omega_dense_flat.reshape(GRID, GRID)
    except Exception as exc:
        logger.warning("Could not extract dense omega from QTT: %s", exc)

    return {
        "wall_time_s": round(wall, 2),
        "enstrophy": enstrophy,
        "omega_l2": omega_l2,
        "div_relative_to_vel": div_rel,
        "poisson": {
            "max_residual": poisson_max_res,
            "mean_iters": poisson_mean_iters,
            "max_iters": poisson_max_iters,
        },
        "series": {
            "poisson_residual": _summarize_series(
                per_step_residual, "poisson_residual", decimation=20,
            ),
            "poisson_iters": _summarize_series(
                [float(i) for i in per_step_iters], "poisson_iters",
                decimation=20, precision=0,
            ),
        },
        "omega_field": omega_field,  # stripped before JSON
    }


# ═══════════════════════════════════════════════════════════════════════
# Comparison
# ═══════════════════════════════════════════════════════════════════════

def compare_fields(qtt_omega: np.ndarray | None, dense_omega: np.ndarray) -> dict[str, float]:
    """Compute field-level L2 relative error ‖ω_qtt − ω_dense‖₂ / ‖ω_dense‖₂."""
    if qtt_omega is None:
        return {"omega_rel_l2": float("nan"), "available": False}

    diff = qtt_omega - dense_omega
    h = 1.0 / dense_omega.shape[0]
    dA = h * h

    diff_l2 = math.sqrt(dA * float(np.sum(diff ** 2)))
    dense_l2 = math.sqrt(dA * float(np.sum(dense_omega ** 2)))
    rel = diff_l2 / dense_l2 if dense_l2 > 1e-30 else float("nan")

    return {"omega_rel_l2": rel, "available": True}


def analytical_ref(t: float, nu: float = VISCOSITY) -> dict[str, float]:
    """Exact Taylor-Green analytical QoIs at time t."""
    decay = math.exp(-8.0 * math.pi ** 2 * nu * t)
    return {
        "enstrophy": 0.5 * decay ** 2,
        "omega_l2": abs(decay),
    }


# ═══════════════════════════════════════════════════════════════════════
# Validation checks
# ═══════════════════════════════════════════════════════════════════════

def validate(
    qtt: dict[str, Any],
    dense: dict[str, Any],
    comparison: dict[str, Any],
) -> tuple[bool, list[dict[str, Any]]]:
    """Run pass/fail checks per the spec."""
    checks: list[dict[str, Any]] = []

    # Field agreement: rel_L2(omega) ≤ 5e-3
    omega_rel = comparison.get("omega_rel_l2", float("nan"))
    checks.append({
        "name": "field_omega_rel_l2",
        "passed": omega_rel <= REL_L2_OMEGA_MAX,
        "value": omega_rel,
        "threshold": REL_L2_OMEGA_MAX,
        "failure_severity": "error",
    })

    # Enstrophy match: |E_qtt - E_dense| / E_dense ≤ 1e-4
    e_qtt = qtt.get("enstrophy", float("nan"))
    e_dense = dense.get("enstrophy", float("nan"))
    e_diff = abs(e_qtt - e_dense) / abs(e_dense) if abs(e_dense) > 1e-30 else float("nan")
    checks.append({
        "name": "enstrophy_rel_diff",
        "passed": e_diff <= ENSTROPHY_REL_DIFF_MAX,
        "value": e_diff,
        "threshold": ENSTROPHY_REL_DIFF_MAX,
        "failure_severity": "error",
    })

    # Omega L2 match
    o_qtt = qtt.get("omega_l2", float("nan"))
    o_dense = dense.get("omega_l2", float("nan"))
    o_diff = abs(o_qtt - o_dense) / abs(o_dense) if abs(o_dense) > 1e-30 else float("nan")
    checks.append({
        "name": "omega_l2_rel_diff",
        "passed": o_diff <= OMEGA_L2_REL_DIFF_MAX,
        "value": o_diff,
        "threshold": OMEGA_L2_REL_DIFF_MAX,
        "failure_severity": "error",
    })

    all_passed = all(c["passed"] for c in checks)
    return all_passed, checks


# ═══════════════════════════════════════════════════════════════════════
# Claims
# ═══════════════════════════════════════════════════════════════════════

def generate_claims(
    qtt: dict[str, Any],
    dense: dict[str, Any],
    comparison: dict[str, Any],
    passed: bool,
) -> list[dict[str, str]]:
    """Generate provenance claim tags.

    Three semantically distinct claims:
    - CONVERGENCE: Poisson solver meets tol with warm-start efficiency.
    - VALIDATION:  Discrete-exact FFT reference agreement (the headline).
    - STABILITY:   Constraint leakage (div) stays small over the run.
    """
    poisson = qtt.get("poisson", {})
    return [
        {
            "tag": "CONVERGENCE",
            "claim": (
                f"Poisson solve meets tol={PRODUCTION_TOL} with warm-start; "
                f"max iters {poisson.get('max_iters', '?')}, "
                f"mean {poisson.get('mean_iters', '?')}"
            ),
        },
        {
            "tag": "VALIDATION",
            "claim": (
                f"Discrete-exact FFT reference match at {GRID}\u00b2: "
                f"rel_L2(\u03c9) = {comparison.get('omega_rel_l2', float('nan')):.2e}, "
                f"enstrophy diff = {comparison.get('enstrophy_rel_diff', float('nan')):.2e}"
            ),
        },
        {
            "tag": "STABILITY",
            "claim": (
                f"Constraint leakage: div_rel_vel = {qtt.get('div_relative_to_vel', float('nan')):.2e} "
                f"over t = {T_FINAL}"
            ),
        },
    ]


# ═══════════════════════════════════════════════════════════════════════
# Git SHA
# ═══════════════════════════════════════════════════════════════════════

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# ═══════════════════════════════════════════════════════════════════════
# JSON helpers
# ═══════════════════════════════════════════════════════════════════════

def _strip_fields(d: dict) -> dict:
    """Remove non-serializable keys (dense arrays, etc.)."""
    return {k: v for k, v in d.items() if not isinstance(v, np.ndarray) and k != "omega_field"}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Inf"
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return str(obj)


# ═══════════════════════════════════════════════════════════════════════
# Main entry point (callable from ns2d_evidence.py or standalone)
# ═══════════════════════════════════════════════════════════════════════

def run_scenario() -> dict[str, Any]:
    """Execute the full cross-check scenario and return the result dict."""
    t_total = time.perf_counter()

    # Run both solvers
    qtt_data = run_qtt(seed=42)
    dense_data = run_dense_reference()

    # Field comparison
    comparison = compare_fields(qtt_data.get("omega_field"), dense_data["omega_field"])

    # QoI comparison
    e_qtt = qtt_data["enstrophy"]
    e_dense = dense_data["enstrophy"]
    o_qtt = qtt_data["omega_l2"]
    o_dense = dense_data["omega_l2"]

    comparison["enstrophy_rel_diff"] = abs(e_qtt - e_dense) / abs(e_dense) if abs(e_dense) > 1e-30 else float("nan")
    comparison["omega_l2_rel_diff"] = abs(o_qtt - o_dense) / abs(o_dense) if abs(o_dense) > 1e-30 else float("nan")

    # Analytical reference at t_final
    t_actual = N_STEPS * DT
    ref = analytical_ref(t_actual)
    analytical = {
        "enstrophy": ref["enstrophy"],
        "omega_l2": ref["omega_l2"],
        "qtt_enstrophy_error_rel": abs(e_qtt - ref["enstrophy"]) / ref["enstrophy"] if ref["enstrophy"] > 0 else float("nan"),
        "dense_enstrophy_error_rel": abs(e_dense - ref["enstrophy"]) / ref["enstrophy"] if ref["enstrophy"] > 0 else float("nan"),
    }

    passed, checks = validate(qtt_data, dense_data, comparison)
    claims = generate_claims(qtt_data, dense_data, comparison, passed)

    total_wall = time.perf_counter() - t_total

    result = {
        "case_id": CASE_ID,
        "commit": _git_sha(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "config": {
            "n_bits": N_BITS,
            "grid": f"{GRID}x{GRID}",
            "dt": DT,
            "n_steps": N_STEPS,
            "t_final": T_FINAL,
            "nu": VISCOSITY,
            "qtt_poisson": "MG-DC",
            "qtt_poisson_tol": PRODUCTION_TOL,
            "qtt_rank": MAX_RANK,
            "dense_poisson": "FFT diagonalization (exact discrete)",
        },
        "qtt": _strip_fields(qtt_data),
        "dense": _strip_fields(dense_data),
        "comparison": comparison,
        "analytical": analytical,
        "total_wall_time_s": round(total_wall, 2),
        "validation": {
            "passed": passed,
            "checks": checks,
        },
        "claims": claims,
    }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="TG FFT cross-check @ 256²")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    args = parser.parse_args()

    if args.dry_run:
        print(f"Case ID:      {CASE_ID}")
        print(f"Grid:         {GRID}x{GRID} (n_bits={N_BITS})")
        print(f"dt:           {DT}")
        print(f"n_steps:      {N_STEPS}")
        print(f"t_final:      {N_STEPS * DT:.6f}")
        print(f"QTT Poisson:  MG-DC, tol={PRODUCTION_TOL}, rank={MAX_RANK}")
        print(f"Dense Poisson: FFT diagonalization (exact)")
        return

    result = run_scenario()
    passed = result["validation"]["passed"]

    # Write JSON
    out_dir = os.path.join(ROOT, "scenario_output", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{CASE_ID.lower()}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=_json_default)
    sz = os.path.getsize(out_path) / 1024
    logger.info("Result written: %s (%.1f KB)", out_path, sz)

    # Summary
    comp = result["comparison"]
    print("\n" + "=" * 64)
    print(f"SCENARIO 2: {CASE_ID}")
    print(f"  PASSED: {passed}")
    print(f"  Total wall time: {result['total_wall_time_s']:.0f}s")
    print(f"  Field rel_L2(ω): {comp.get('omega_rel_l2', 'N/A')}")
    print(f"  Enstrophy diff:  {comp.get('enstrophy_rel_diff', 'N/A')}")
    print(f"  Omega L2 diff:   {comp.get('omega_l2_rel_diff', 'N/A')}")
    for c in result["validation"]["checks"]:
        status = "✓" if c["passed"] else "✗"
        print(f"  {status} {c['name']}: {c.get('value', '?')}")
    print(f"  Output: {out_path}")
    print("=" * 64)

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
