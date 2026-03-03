#!/usr/bin/env python3
"""Run Stokes 512² at 1000 steps with Tier 2 physics QoI extraction.

Pipeline:
  1. Execute NS 2D (glycerol, cylinder, Re ≈ 0.09) on GPU
  2. Extract physics QoI from RAW result (before sanitization)
  3. Sanitize result (IP boundary)
  4. Inject QoI into sanitized dict
  5. Generate validation report + claims
  6. Write slim JSON
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("stokes_qoi")

# ── Project root ────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from physics_os.core.executor import ExecutionConfig, execute
from physics_os.core.sanitizer import sanitize_result
from physics_os.core.evidence import generate_validation_report, generate_claims
from physics_os.core.physics_qoi import extract_physics_qoi


def main() -> None:
    n_bits = 9  # 512 = 2^9
    n_steps = 1000
    domain_key = "navier_stokes_2d"

    # Glycerol past a 10 mm cylinder at 0.01 m/s  →  Re ≈ 0.09
    Re = 0.09
    params = {
        "Re": Re,
        "inlet_velocity": 0.01,
        "cylinder_radius": 0.005,
    }

    execution_context = {
        "n_bits": n_bits,
        "n_steps": n_steps,
        "Re": Re,
    }

    config = ExecutionConfig(
        domain=domain_key,
        n_bits=n_bits,
        n_steps=n_steps,
        max_rank=64,
        truncation_tol=1e-10,
        parameters=params,
    )

    logger.info("Starting: %d² grid, %d steps, Re=%.2f", 2**n_bits, n_steps, Re)
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
        execution_context=execution_context,
    )

    # ── Inject QoI into sanitized result ────────────────────────
    san["physics_qoi"] = physics_qoi

    # ── Validation report + claims ──────────────────────────────
    val_report = generate_validation_report(san, domain_key)
    claims = generate_claims(san, domain_key)

    # ── Build output ────────────────────────────────────────────
    output = {
        "scenario": "stokes_512_1000_qoi",
        "domain": domain_key,
        "grid": san.get("grid", {}),
        "execution_params": {
            "n_bits": n_bits,
            "n_steps": n_steps,
            "Re": Re,
            "max_rank": config.max_rank,
            "truncation_tol": config.truncation_tol,
        },
        "performance": san.get("performance", {}),
        "conservation": san.get("conservation", {}),
        "physics_qoi": physics_qoi,
        "validation": val_report,
        "claims": claims,
    }

    # ── Write JSON ──────────────────────────────────────────────
    out_dir = os.path.join(ROOT, "scenario_output", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "stokes_512_1000_qoi.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Output written: %s (%.1f KB)", out_path, os.path.getsize(out_path) / 1024)

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STOKES 512² × 1000 steps — Tier 2 QoI Report")
    print("=" * 60)
    print(f"Wall time:     {san.get('performance', {}).get('wall_time_s', 0):.2f} s")
    print(f"Validation:    valid={val_report.get('valid')}")
    for check in val_report.get("checks", []):
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  [{status}] {check['name']}: {check['detail']}")
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
        print(f"  enstrophy bounded = {s['enstrophy_bounded']}")

    print("=" * 60)


if __name__ == "__main__":
    main()
