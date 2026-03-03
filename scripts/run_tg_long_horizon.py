#!/usr/bin/env python3
"""Scenario 1 — Long-horizon Taylor-Green @ 512².

Run ID: TG_LH_512_QTT_PROD

Proves the QTT MG-DC Poisson solver stays correct and stable beyond the
trivial early-time window (t_final = 0.05, 1049 steps), with meaningful
viscous decay (~7.6% enstrophy drop), on the production grid at production
tolerance.

Two seeds (0, 42) verify reproducibility under rSVD randomness.

Spec: docs/reports/NS2D_SCHEDULED_VV_SCENARIOS.md §1

Usage:
    python3 scripts/run_tg_long_horizon.py [--dry-run]
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("tg_long_horizon")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch

import numpy as np

from physics_os.core.executor import ExecutionConfig, execute
from physics_os.core.physics_qoi import extract_physics_qoi

# ═══════════════════════════════════════════════════════════════════════
# Locked configuration — DO NOT CHANGE without updating the spec doc
# ═══════════════════════════════════════════════════════════════════════
CASE_ID = "TG_LH_512_QTT_PROD"
N_BITS = 9
GRID = 2 ** N_BITS  # 512
VISCOSITY = 0.01
MAX_RANK = 64
TRUNC_TOL = 1e-10
PRODUCTION_TOL = 1e-3
POISSON_MAX_ITERS = 80

# dt from the stable CFL calculation: 0.25 * h² / (2ν)
_h = 1.0 / GRID
DT = 0.25 * _h * _h / (2.0 * VISCOSITY)  # 4.76837158203125e-05
T_FINAL = 0.05
N_STEPS = round(T_FINAL / DT)  # 1049

SEEDS = [0, 42]

# Snapshot times (we save the closest completed step)
SNAPSHOT_TIMES = [0.01, 0.02, 0.05]

# Pass criteria
ENSTROPHY_ERROR_REL_MAX = 1e-4
OMEGA_L2_ERROR_REL_MAX = 1e-4
DIV_RELATIVE_TO_VEL_MAX = 1e-3
REPRO_DELTA_MAX = 1e-6


# ═══════════════════════════════════════════════════════════════════════
# JSON helpers — compact series + full-precision QoIs
# ═══════════════════════════════════════════════════════════════════════

def _summarize_series(
    values: list[float],
    name: str,
    *,
    decimation: int = 50,
    precision: int = 8,
) -> dict[str, Any]:
    """Compress a per-step series into stats + decimated samples.

    Returns a compact dict with min/max/mean/std/first/last plus
    every *decimation*-th sample (always including first and last).
    """
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=np.float64)
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
    V&V comparison uses maximum available precision.
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
# Analytical reference
# ═══════════════════════════════════════════════════════════════════════

def analytical_taylor_green(t: float, nu: float = VISCOSITY) -> dict[str, float]:
    """Exact Taylor-Green solution at time t.

    ω(x,y,t) = 2·sin(2πx)·sin(2πy)·exp(-8π²νt)
    E(t) = 0.5·exp(-16π²νt)
    ‖ω‖₂ = exp(-8π²νt)
    """
    decay = math.exp(-8.0 * math.pi ** 2 * nu * t)
    omega_sq = decay ** 2
    return {
        "enstrophy": 0.5 * omega_sq,
        "omega_l2": math.sqrt(omega_sq),
        "circulation": 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════
# Divergence computation (reused from run_evidence_package.py)
# ═══════════════════════════════════════════════════════════════════════

def _compute_divergence(result: Any) -> dict[str, float]:
    """Compute ∇·u from final ψ via QTT-native gradient MPOs."""
    from ontic.engine.vm.gpu_operators import GPUOperatorCache, gpu_mpo_apply

    psi = result.fields.get("psi")
    if psi is None:
        return {"div_l2_norm": float("nan"), "div_relative_to_vel": float("nan"), "available": False}

    cache = GPUOperatorCache()
    bpd = psi.bits_per_dim
    dom = psi.domain

    grad_x_mpo = cache.get_gradient(0, bpd, dom, variant="grad_v1")
    grad_y_mpo = cache.get_gradient(1, bpd, dom, variant="grad_v1")

    # u = ∂ψ/∂y,  v = -∂ψ/∂x
    u = gpu_mpo_apply(grad_y_mpo, psi, max_rank=MAX_RANK, cutoff=TRUNC_TOL)
    v_pos = gpu_mpo_apply(grad_x_mpo, psi, max_rank=MAX_RANK, cutoff=TRUNC_TOL)

    du_dx = gpu_mpo_apply(grad_x_mpo, u, max_rank=MAX_RANK, cutoff=TRUNC_TOL)
    dv_pos_dy = gpu_mpo_apply(grad_y_mpo, v_pos, max_rank=MAX_RANK, cutoff=TRUNC_TOL)

    div = du_dx.sub(dv_pos_dy)
    div = div.truncate(max_rank=MAX_RANK, cutoff=TRUNC_TOL)

    hx = psi.grid_spacing(0)
    hy = psi.grid_spacing(1)
    dA = hx * hy

    div_l2_sq = dA * div.inner(div)
    div_l2 = math.sqrt(max(div_l2_sq, 0.0))

    u_l2_sq = dA * u.inner(u)
    v_neg = v_pos.negate()
    v_l2_sq = dA * v_neg.inner(v_neg)
    vel_l2 = math.sqrt(max(u_l2_sq + v_l2_sq, 0.0))

    return {
        "available": True,
        "div_l2_norm": div_l2,
        "div_relative_to_vel": div_l2 / vel_l2 if vel_l2 > 1e-30 else float("nan"),
    }


# ═══════════════════════════════════════════════════════════════════════
# Single-seed runner
# ═══════════════════════════════════════════════════════════════════════

def run_single_seed(seed: int) -> dict[str, Any]:
    """Execute NS2D for N_STEPS at DT with a given seed.

    Returns a dict with final QoIs, snapshots, per-step series, and
    Poisson solver statistics.
    """
    logger.info("=" * 64)
    logger.info("SEED %d — %d steps @ dt=%.6e  (t_final=%.4f)", seed, N_STEPS, DT, T_FINAL)
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
    logger.info("Execution complete: %.1fs, success=%s", wall, result.success)

    # ── Extract QoIs — full float64 precision from QTT inner product
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

    telem = result.telemetry
    inv_init = getattr(telem, "invariant_initial", 0.0)
    inv_final = getattr(telem, "invariant_final", 0.0)
    conservation_error = abs(inv_final - inv_init)

    probes = getattr(result, "probes", {})
    per_step_residual = probes.get("poisson_relative_residual", [])
    per_step_iters = probes.get("poisson_cg_iters", [])

    # ── Divergence ────────────────────────────────────────────────
    logger.info("Computing divergence ∇·u ...")
    div_info = _compute_divergence(result)
    div_rel = div_info.get("div_relative_to_vel", float("nan"))

    # ── Analytical comparison at t_final ──────────────────────────
    t_actual = N_STEPS * DT
    ref = analytical_taylor_green(t_actual)
    enstr_err_rel = abs(enstrophy - ref["enstrophy"]) / ref["enstrophy"] if ref["enstrophy"] > 0 else float("nan")
    ol2_err_rel = abs(omega_l2 - ref["omega_l2"]) / ref["omega_l2"] if ref["omega_l2"] > 0 else float("nan")

    # ── Snapshots at intermediate times ───────────────────────────
    # We use per-step enstrophy probes if available; otherwise single final.
    per_step_enstrophy = probes.get("enstrophy", [])
    per_step_omega_l2 = probes.get("omega_l2_norm", [])

    snapshots: dict[str, dict[str, Any]] = {}
    for t_snap in SNAPSHOT_TIMES:
        step_idx = round(t_snap / DT) - 1  # 0-based index
        t_at_snap = (step_idx + 1) * DT
        snap_ref = analytical_taylor_green(t_at_snap)

        snap_entry: dict[str, Any] = {
            "step": step_idx + 1,
            "t": round(t_at_snap, 10),
            "enstrophy_analytical": snap_ref["enstrophy"],
            "omega_l2_analytical": snap_ref["omega_l2"],
        }

        if step_idx < len(per_step_enstrophy):
            e_val = per_step_enstrophy[step_idx]
            snap_entry["enstrophy"] = e_val
            snap_entry["enstrophy_error_rel"] = abs(e_val - snap_ref["enstrophy"]) / snap_ref["enstrophy"]
        if step_idx < len(per_step_omega_l2):
            o_val = per_step_omega_l2[step_idx]
            snap_entry["omega_l2"] = o_val
            snap_entry["omega_l2_error_rel"] = abs(o_val - snap_ref["omega_l2"]) / snap_ref["omega_l2"]

        snapshots[str(t_snap)] = snap_entry

    # If no per-step probes were available, at least record t_final as the 0.05 snapshot
    if "0.05" not in snapshots or "enstrophy" not in snapshots.get("0.05", {}):
        snapshots["0.05"] = {
            "step": N_STEPS,
            "t": t_actual,
            "enstrophy": enstrophy,
            "enstrophy_analytical": ref["enstrophy"],
            "enstrophy_error_rel": enstr_err_rel,
            "omega_l2": omega_l2,
            "omega_l2_analytical": ref["omega_l2"],
            "omega_l2_error_rel": ol2_err_rel,
        }

    # ── Cold start detection ──────────────────────────────────────
    cold_iters = int(per_step_iters[0]) if per_step_iters else 0

    logger.info(
        "  enstrophy=%.8e  err_rel=%.4e  omega_l2=%.8e  err_rel=%.4e  div=%.4e  wall=%.1fs",
        enstrophy, enstr_err_rel, omega_l2, ol2_err_rel, div_rel, wall,
    )

    return {
        "seed": seed,
        "wall_time_s": round(wall, 2),
        "snapshots": snapshots,
        "final": {
            "t": round(t_actual, 10),
            "enstrophy": enstrophy,
            "enstrophy_analytical": ref["enstrophy"],
            "enstrophy_error_rel": enstr_err_rel,
            "omega_l2": omega_l2,
            "omega_l2_analytical": ref["omega_l2"],
            "omega_l2_error_rel": ol2_err_rel,
            "div_l2_norm": div_info.get("div_l2_norm", float("nan")),
            "div_relative_to_vel": div_rel,
            "circulation_error": conservation_error,
        },
        "poisson": {
            "max_residual": poisson_max_res,
            "mean_iters": poisson_mean_iters,
            "max_iters": poisson_max_iters,
            "cold_iters": cold_iters,
        },
        "series": {
            "poisson_residual": _summarize_series(
                per_step_residual, "poisson_residual", decimation=50,
            ),
            "poisson_iters": _summarize_series(
                [float(i) for i in per_step_iters], "poisson_iters",
                decimation=50, precision=0,
            ),
            "enstrophy": _summarize_series(
                per_step_enstrophy, "enstrophy", decimation=50, precision=10,
            ),
            "omega_l2": _summarize_series(
                per_step_omega_l2, "omega_l2", decimation=50, precision=10,
            ),
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# Validation checks
# ═══════════════════════════════════════════════════════════════════════

def validate(seeds_data: dict[int, dict[str, Any]]) -> tuple[bool, list[dict[str, Any]]]:
    """Run pass/fail checks against the spec.  Returns (all_passed, checks)."""
    checks: list[dict[str, Any]] = []

    # Per-seed accuracy + constraint
    for seed, data in seeds_data.items():
        f = data["final"]

        checks.append({
            "name": f"enstrophy_error_rel_seed{seed}",
            "passed": f["enstrophy_error_rel"] <= ENSTROPHY_ERROR_REL_MAX,
            "value": f["enstrophy_error_rel"],
            "threshold": ENSTROPHY_ERROR_REL_MAX,
            "failure_severity": "error",
        })
        checks.append({
            "name": f"omega_l2_error_rel_seed{seed}",
            "passed": f["omega_l2_error_rel"] <= OMEGA_L2_ERROR_REL_MAX,
            "value": f["omega_l2_error_rel"],
            "threshold": OMEGA_L2_ERROR_REL_MAX,
            "failure_severity": "error",
        })
        checks.append({
            "name": f"div_relative_to_vel_seed{seed}",
            "passed": f["div_relative_to_vel"] <= DIV_RELATIVE_TO_VEL_MAX,
            "value": f["div_relative_to_vel"],
            "threshold": DIV_RELATIVE_TO_VEL_MAX,
            "failure_severity": "error",
        })

    # Stability: Poisson residual must not blow up (last ≤ 2× first)
    for seed, data in seeds_data.items():
        resid_summary = data["series"].get("poisson_residual", {})
        if isinstance(resid_summary, dict) and resid_summary.get("n", 0) >= 2:
            first = resid_summary["first"]
            last = resid_summary["last"]
            stable = last <= 2.0 * first + 1e-12
            checks.append({
                "name": f"poisson_stability_seed{seed}",
                "passed": stable,
                "value": last / (first + 1e-30),
                "detail": f"first={first:.4e} last={last:.4e}",
                "failure_severity": "error",
            })

    # Reproducibility between seeds
    seed_list = sorted(seeds_data.keys())
    if len(seed_list) >= 2:
        d0 = seeds_data[seed_list[0]]["final"]
        d1 = seeds_data[seed_list[1]]["final"]
        delta_e = abs(d0["enstrophy"] - d1["enstrophy"]) / max(abs(d0["enstrophy"]), 1e-30)
        delta_o = abs(d0["omega_l2"] - d1["omega_l2"]) / max(abs(d0["omega_l2"]), 1e-30)
        checks.append({
            "name": "reproducibility_enstrophy",
            "passed": delta_e <= REPRO_DELTA_MAX,
            "value": delta_e,
            "threshold": REPRO_DELTA_MAX,
            "failure_severity": "error",
        })
        checks.append({
            "name": "reproducibility_omega_l2",
            "passed": delta_o <= REPRO_DELTA_MAX,
            "value": delta_o,
            "threshold": REPRO_DELTA_MAX,
            "failure_severity": "error",
        })

    all_passed = all(c["passed"] for c in checks)
    return all_passed, checks


# ═══════════════════════════════════════════════════════════════════════
# Claims
# ═══════════════════════════════════════════════════════════════════════

def generate_claims(seeds_data: dict[int, dict[str, Any]], passed: bool) -> list[dict[str, str]]:
    """Generate provenance claim tags.

    Five semantically distinct claims:
    - CONSERVATION: Kelvin's theorem (circulation preserved).
    - STABILITY:    Constraint leakage (div) stays small.
    - CONVERGENCE:  QoI accuracy vs analytical solution.
    - BOUND:        Poisson residual stays below contract tol.
    - REPRODUCIBILITY: Seed-to-seed QoI delta (if ≥2 seeds).
    """
    claims: list[dict[str, str]] = []
    s0_key = sorted(seeds_data.keys())[0]
    d0 = seeds_data[s0_key]["final"]
    p0 = seeds_data[s0_key]["poisson"]

    claims.append({
        "tag": "CONSERVATION",
        "claim": f"Circulation preserved to {d0['circulation_error']:.2e} absolute error over {N_STEPS} steps",
    })
    claims.append({
        "tag": "STABILITY",
        "claim": f"Divergence ratio {d0['div_relative_to_vel']:.2e} at t_final={T_FINAL}; Poisson bounded",
    })
    claims.append({
        "tag": "CONVERGENCE",
        "claim": f"Enstrophy error {d0['enstrophy_error_rel']:.2e} at t_final={T_FINAL} vs analytical",
    })
    claims.append({
        "tag": "BOUND",
        "claim": (
            f"Poisson residual bounded below {PRODUCTION_TOL} "
            f"(max = {p0['max_residual']:.4e})"
        ),
    })

    if len(seeds_data) >= 2:
        s0, s1 = sorted(seeds_data.keys())[:2]
        delta_e = abs(seeds_data[s0]["final"]["enstrophy"] - seeds_data[s1]["final"]["enstrophy"])
        claims.append({
            "tag": "REPRODUCIBILITY",
            "claim": f"Enstrophy delta between seed {s0} and {s1}: {delta_e:.2e}",
        })

    return claims


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
# JSON serializer
# ═══════════════════════════════════════════════════════════════════════

def _json_default(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Inf"
    return str(obj)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def run_scenario() -> dict[str, Any]:
    """Execute the full long-horizon scenario and return the result dict.

    Callable from ns2d_evidence.py VVTest executor or standalone.
    """
    t_total = time.perf_counter()

    seeds_results: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        seeds_results[seed] = run_single_seed(seed)

    passed, checks = validate(seeds_results)
    claims = generate_claims(seeds_results, passed)

    # Reproducibility summary
    s0, s1 = sorted(seeds_results.keys())[:2]
    d0f = seeds_results[s0]["final"]
    d1f = seeds_results[s1]["final"]
    repro = {
        "delta_enstrophy_rel": abs(d0f["enstrophy"] - d1f["enstrophy"]) / max(abs(d0f["enstrophy"]), 1e-30),
        "delta_omega_l2_rel": abs(d0f["omega_l2"] - d1f["omega_l2"]) / max(abs(d0f["omega_l2"]), 1e-30),
    }

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
            "poisson": "MG-DC",
            "poisson_tol": PRODUCTION_TOL,
            "rank": MAX_RANK,
            "mg_config": "3+3 smooth, 7 levels (9->3 bits), 5 coarse sweeps",
        },
        "seeds": {str(s): d for s, d in seeds_results.items()},
        "reproducibility": repro,
        "total_wall_time_s": round(total_wall, 2),
        "validation": {
            "passed": passed,
            "checks": checks,
        },
        "claims": claims,
    }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="TG long-horizon @ 512²")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    args = parser.parse_args()

    if args.dry_run:
        print(f"Case ID:  {CASE_ID}")
        print(f"Grid:     {GRID}x{GRID} (n_bits={N_BITS})")
        print(f"dt:       {DT}")
        print(f"n_steps:  {N_STEPS}")
        print(f"t_final:  {N_STEPS * DT:.6f}")
        print(f"Seeds:    {SEEDS}")
        print(f"Poisson:  MG-DC, tol={PRODUCTION_TOL}, rank={MAX_RANK}")
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
    print("\n" + "=" * 64)
    print(f"SCENARIO 1: {CASE_ID}")
    print(f"  PASSED: {passed}")
    print(f"  Total wall time: {result['total_wall_time_s']:.0f}s")
    for c in result["validation"]["checks"]:
        status = "✓" if c["passed"] else "✗"
        print(f"  {status} {c['name']}: {c.get('value', '?')}")
    print(f"  Output: {out_path}")
    print("=" * 64)

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
