#!/usr/bin/env python3
"""Run multi-mode NS 2D at 1024² to stress the Poisson MG solver.

Usage:
    python3 scripts/run_multimode_qoi.py [--steps N] [--n-bits B] [--modes K] [--precond mg|none]

Pipeline:
  1. Execute NS 2D with multi-mode IC (broadband ω₀) on GPU
  2. Extract physics QoI from RAW result (before sanitization)
  3. Sanitize result (IP boundary)
  4. Inject QoI into sanitized dict
  5. Generate validation report + claims
  6. Write slim JSON

The multi-mode IC injects Fourier content at wavenumbers 1..K per
dimension with Kolmogorov-like 1/(k²+m²) amplitude decay.  This
stresses the Poisson solver by requiring resolution of modes with
widely separated eigenvalues — unlike the single Taylor-Green mode
which converges in 1 CG iteration.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("multimode_qoi")

# ── Project root ────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from physics_os.core.executor import ExecutionConfig, execute
from physics_os.core.sanitizer import sanitize_result
from physics_os.core.evidence import generate_validation_report, generate_claims
from physics_os.core.physics_qoi import extract_physics_qoi


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-mode Poisson stress run")
    parser.add_argument("--steps", type=int, default=256, help="Time steps")
    parser.add_argument("--n-bits", type=int, default=10, help="Bits per dim (grid = 2^n)")
    parser.add_argument("--modes", type=int, default=4, help="Fourier modes per dim")
    parser.add_argument("--precond", type=str, default="mg",
                        choices=["mg", "none"], help="Poisson preconditioner")
    parser.add_argument("--max-rank", type=int, default=64, help="Max QTT rank")
    parser.add_argument("--poisson-tol", type=float, default=1e-6,
                        help="CG convergence tolerance")
    parser.add_argument("--poisson-max-iters", type=int, default=80,
                        help="Max CG iterations per step")
    args = parser.parse_args()

    n_bits = args.n_bits
    n_steps = args.steps
    ic_n_modes = args.modes
    precond = args.precond
    domain_key = "navier_stokes_2d"
    grid_size = 2 ** n_bits

    # Moderate Reynolds — not creeping flow, not turbulent.
    # Re ~ 1 gives nonlinear advection (multi-mode interaction) while
    # remaining stable with explicit Euler at the CFL-limited dt.
    viscosity = 0.01
    Re_est = 1.0 / viscosity  # U ~ 1 from IC, L = 1

    execution_context = {
        "n_bits": n_bits,
        "n_steps": n_steps,
        "Re": Re_est,
        "ic_type": "multi_mode",
        "ic_n_modes": ic_n_modes,
        "precond": precond,
    }

    params = {
        "viscosity": viscosity,
        "ic_type": "multi_mode",
        "ic_n_modes": ic_n_modes,
        "poisson_precond": precond,
        "poisson_tol": args.poisson_tol,
        "poisson_max_iters": args.poisson_max_iters,
    }

    config = ExecutionConfig(
        domain=domain_key,
        n_bits=n_bits,
        n_steps=n_steps,
        max_rank=args.max_rank,
        truncation_tol=1e-10,
        parameters=params,
    )

    logger.info(
        "Starting: %d² grid, %d steps, ν=%.4f (Re≈%.0f), "
        "%d modes/dim, precond=%s, max_rank=%d",
        grid_size, n_steps, viscosity, Re_est,
        ic_n_modes, precond, args.max_rank,
    )
    t0 = time.perf_counter()
    raw_result = execute(config)
    wall = time.perf_counter() - t0
    logger.info("Execution: %.2fs, success=%s", wall, raw_result.success)

    # ── Tier 2 QoI extraction (BEFORE sanitization) ─────────────
    logger.info("Extracting physics QoIs from raw result...")
    physics_qoi = extract_physics_qoi(
        raw_result, domain_key, execution_context,
    )
    logger.info("Physics QoI keys: %s", list(physics_qoi.keys()))
    for key, val in physics_qoi.items():
        avail = val.get("available", False) if isinstance(val, dict) else "?"
        logger.info("  %s: available=%s", key, avail)

    # ── Sanitize (IP boundary) ──────────────────────────────────
    san = sanitize_result(
        raw_result,
        domain_key,
        include_fields=False,
        include_coordinates=False,
        execution_context=execution_context,
    )

    # ── Inject QoI into sanitized result ────────────────────────
    san["physics_qoi"] = physics_qoi

    # ── Validation report + claims ──────────────────────────────
    val_report = generate_validation_report(san, domain_key)
    claims = generate_claims(san, domain_key)

    # ── Build slim output ───────────────────────────────────────
    grid_raw = san.get("grid", {})
    cons_raw = san.get("conservation", {})
    perf_raw = san.get("performance", {})

    output = {
        "scenario": f"Multi-mode NS 2D ({grid_size}², {ic_n_modes}² modes, precond={precond})",
        "domain": domain_key,
        "grid": {
            "dimensions": grid_raw.get("dimensions"),
            "resolution": grid_raw.get("resolution"),
            "domain_bounds": grid_raw.get("domain_bounds"),
        },
        "execution": {
            "n_bits": n_bits,
            "n_steps": n_steps,
            "viscosity": viscosity,
            "Re_estimate": Re_est,
            "ic_type": "multi_mode",
            "ic_n_modes": ic_n_modes,
            "precond": precond,
            "max_rank": args.max_rank,
            "truncation_tol": config.truncation_tol,
            "wall_time_s": perf_raw.get("wall_time_s"),
            "throughput_gp_per_s": perf_raw.get("throughput_gp_per_s"),
        },
        "conservation": {
            "quantity": cons_raw.get("quantity"),
            "absolute_error": cons_raw.get("absolute_error"),
            "status": cons_raw.get("status"),
            "tier": cons_raw.get("resolution_tier"),
            "threshold": cons_raw.get("tier_threshold"),
        },
        "physics_qoi": physics_qoi,
        "validation": val_report,
        "claims": claims,
    }

    # ── Write JSON ──────────────────────────────────────────────
    out_dir = os.path.join(ROOT, "scenario_output", "data")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"multimode_{grid_size}_{n_steps}_{precond}"
    out_path = os.path.join(out_dir, f"{tag}_qoi.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Output written: %s (%.1f KB)", out_path, os.path.getsize(out_path) / 1024)

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"MULTI-MODE NS 2D — {grid_size}² × {n_steps} steps")
    print(f"  IC: {ic_n_modes}² Fourier modes, Kolmogorov spectrum")
    print(f"  Poisson preconditioner: {precond}")
    print("=" * 64)
    wt = san.get("performance", {}).get("wall_time_s", 0)
    print(f"Wall time:     {wt:.2f} s")
    print(f"Validation:    valid={val_report.get('valid')}")
    for check in val_report.get("checks", []):
        status = "PASS" if check["passed"] else "FAIL"
        sev = check.get("failure_severity", "?")
        print(f"  [{status}] ({sev}) {check['name']}: {check['detail']}")
    print(f"\nClaims ({len(claims)}):")
    for c in claims:
        sat = "✓" if c["satisfied"] else "✗"
        print(f"  [{sat}] {c['tag']}: {c['claim']}")

    # Physics QoI detail
    p = physics_qoi.get("poisson", {})
    if p.get("available"):
        print(f"\nPoisson solver:")
        print(f"  max ||Aψ−ω||/||ω|| = {p['max_relative_residual']:.2e}")
        print(f"  final residual      = {p['final_relative_residual']:.2e}")
        print(f"  converged fraction  = {p.get('converged_fraction', 'N/A')}")
        print(f"  mean CG iters       = {p.get('mean_cg_iters', 'N/A')}")
        print(f"  CG iters p95        = {p.get('cg_iters_p95', 'N/A')}")
        print(f"  CG iters max        = {p.get('cg_iters_max', 'N/A')}")

    e = physics_qoi.get("enstrophy", {})
    if e.get("available"):
        print(f"\nEnstrophy:")
        print(f"  E = {e['enstrophy']:.4e}")
        print(f"  ||ω||₂ = {e['omega_l2_norm']:.4e}")

    s = physics_qoi.get("stokes_drag", {})
    if s.get("available"):
        print(f"\nStokes drag reference:")
        print(f"  Re = {s['Re']:.2e}")
        print(f"  Lamb (1911) C_d = {s['lamb_cd']:.2f}")
    elif Re_est >= 10.0:
        print(f"\n(Stokes drag: skipped, Re≈{Re_est:.0f} > 10)")


if __name__ == "__main__":
    main()
