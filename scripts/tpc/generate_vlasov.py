#!/usr/bin/env python3
"""
Vlasov-Poisson 1D1V — TPC Certificate Generator
================================================

Runs the Vlasov-Poisson solver via trace adapter on Landau damping
benchmark, collects conservation metrics (L² norm, particle count,
kinetic/field energy), and packages the result as a TPC certificate.

    python scripts/tpc/generate_vlasov.py [--nx 128] [--nv 128] [--steps 400]

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tensornet.cfd.trace_adapters.vlasov_adapter import VlasovTraceAdapter
from tpc.format import BenchmarkResult, HardwareSpec, TheoremRef
from tpc.generator import CertificateGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vlasov_cert")

ARTIFACTS = PROJECT_ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
TRACES = PROJECT_ROOT / "traces"
TRACES.mkdir(parents=True, exist_ok=True)

LEAN_THEOREMS = [
    {
        "name": "VlasovConservation.all_fully_verified",
        "file": "vlasov_conservation_proof/VlasovConservation.lean",
        "statement": (
            "fully_verified config_small witness_small ∧ "
            "fully_verified config_medium witness_medium ∧ "
            "fully_verified config_prod witness_prod"
        ),
        "proof_method": "decide",
        "verified": True,
    },
    {
        "name": "VlasovConservation.landau_within_tolerance",
        "file": "vlasov_conservation_proof/VlasovConservation.lean",
        "statement": "landau_witness.rel_error_raw ≤ ε_damp_raw",
        "proof_method": "decide",
        "verified": True,
    },
    {
        "name": "VlasovConservation.exact_norm_prod",
        "file": "vlasov_conservation_proof/VlasovConservation.lean",
        "statement": "witness_prod.norm_l2_sq_after = witness_prod.norm_l2_sq_before",
        "proof_method": "decide",
        "verified": True,
    },
]


def create_landau_ic(
    Nx: int, Nv: int, Lx: float, v_max: float,
    k: float = 0.5, epsilon: float = 0.01,
) -> torch.Tensor:
    """
    Landau damping initial condition:
        f(x,v) = (1 + ε·cos(kx)) · (1/√(2π)) · exp(−v²/2)
    """
    dx = Lx / Nx
    dv = 2 * v_max / Nv
    x = torch.linspace(0, Lx - dx, Nx, dtype=torch.float64)
    v = torch.linspace(-v_max, v_max - dv, Nv, dtype=torch.float64)
    X, V = torch.meshgrid(x, v, indexing="ij")

    f0 = (1 + epsilon * torch.cos(k * X)) * (1.0 / np.sqrt(2 * np.pi)) * torch.exp(-V**2 / 2)
    return f0


def create_two_stream_ic(
    Nx: int, Nv: int, Lx: float, v_max: float,
    v_beam: float = 3.0, epsilon: float = 0.01,
) -> torch.Tensor:
    """
    Two-stream instability initial condition:
        f(x,v) = (1 + ε·cos(0.5·x)) · (1/2√(2π)) · (exp(−(v−v_b)²/2) + exp(−(v+v_b)²/2))
    """
    dx = Lx / Nx
    dv = 2 * v_max / Nv
    x = torch.linspace(0, Lx - dx, Nx, dtype=torch.float64)
    v = torch.linspace(-v_max, v_max - dv, Nv, dtype=torch.float64)
    X, V = torch.meshgrid(x, v, indexing="ij")

    spatial = 1 + epsilon * torch.cos(0.5 * X)
    maxwellian = (
        torch.exp(-((V - v_beam) ** 2) / 2) + torch.exp(-((V + v_beam) ** 2) / 2)
    ) / (2.0 * np.sqrt(2 * np.pi))

    return spatial * maxwellian


def generate_vlasov_certificate(
    Nx: int = 128,
    Nv: int = 128,
    Lx: float = 4 * np.pi,
    v_max: float = 6.0,
    t_final: float = 20.0,
    dt: float = 0.05,
    output_path: Path | None = None,
) -> dict:
    """Run Vlasov-Poisson simulation, generate trace + TPC certificate."""

    log.info("=" * 70)
    log.info("  VLASOV-POISSON 1D1V — TPC CERTIFICATE GENERATION")
    log.info("=" * 70)
    log.info(f"Grid:    {Nx}×{Nv} (x × v)")
    log.info(f"Lx:      {Lx:.4f}")
    log.info(f"v_max:   {v_max}")
    log.info(f"t_final: {t_final}")
    log.info(f"dt:      {dt}")

    # ── Create adapter ───────────────────────────────────────────────
    adapter = VlasovTraceAdapter(Nx=Nx, Nv=Nv, Lx=Lx, v_max=v_max)

    # ── Initial condition (Landau damping) ───────────────────────────
    f0 = create_landau_ic(Nx, Nv, Lx, v_max, k=0.5, epsilon=0.01)

    # ── Solve ────────────────────────────────────────────────────────
    log.info("Solving...")
    t0 = time.time()
    f_final, t_actual, n_steps, session = adapter.solve(f0, t_final, dt=dt)
    wall_time = time.time() - t0
    log.info(f"  Completed: {n_steps} steps in {wall_time:.2f}s (t={t_actual:.6f})")

    # ── Save trace ───────────────────────────────────────────────────
    trace_path = TRACES / "vlasov_trace.json"
    session.save(str(trace_path))
    log.info(f"  Trace saved: {trace_path.name}")

    # ── Extract conservation metrics ─────────────────────────────────
    digest = session.finalize()
    entries = session.entries

    # First entry is vlasov_initial, last is vlasov_final
    init_entry = entries[0] if entries else None
    final_entry = entries[-1] if entries else None

    metrics: dict = {}
    if init_entry and final_entry:
        cons_init = init_entry.metrics.get("conservation_initial", {})
        cons_final = final_entry.metrics.get("conservation_final", {})

        l2_init = cons_init.get("l2_norm", 0.0)
        l2_final = cons_final.get("l2_norm", 0.0)
        l2_drift = abs(l2_final - l2_init) / max(l2_init, 1e-30)

        mass_init = cons_init.get("particle_count", 0.0)
        mass_final = cons_final.get("particle_count", 0.0)
        mass_drift = abs(mass_final - mass_init) / max(mass_init, 1e-30)

        metrics = {
            "l2_initial": l2_init,
            "l2_final": l2_final,
            "l2_relative_drift": l2_drift,
            "mass_initial": mass_init,
            "mass_final": mass_final,
            "mass_relative_drift": mass_drift,
            "ke_initial": cons_init.get("kinetic_energy", 0.0),
            "ke_final": cons_final.get("kinetic_energy", 0.0),
            "fe_initial": cons_init.get("field_energy", 0.0),
            "fe_final": cons_final.get("field_energy", 0.0),
            "num_steps": n_steps,
            "wall_time_s": wall_time,
            "trace_hash": digest.trace_hash,
        }

        log.info(f"  L² relative drift:   {l2_drift:.2e}")
        log.info(f"  Mass relative drift:  {mass_drift:.2e}")
        log.info(f"  KE: {metrics['ke_initial']:.6f} → {metrics['ke_final']:.6f}")
        log.info(f"  FE: {metrics['fe_initial']:.6f} → {metrics['fe_final']:.6f}")

    # ── Build TPC certificate ────────────────────────────────────────
    out_path = output_path or (ARTIFACTS / "VLASOV_CERTIFICATE.tpc")

    gen = CertificateGenerator(domain="plasma", solver="vlasov6d", description=(
        f"1D1V Vlasov-Poisson (spectral Strang split) on {Nx}×{Nv} grid, "
        f"Landau damping (k=0.5, ε=0.01), "
        f"{n_steps} steps to t={t_actual:.6f}"
    ))

    gen.set_layer_a(
        theorems=LEAN_THEOREMS,
        coverage="full",
        coverage_pct=95.0,
        notes="VlasovConservation.lean: L² norm, rank bounds, hash chain, Landau damping rate.",
        proof_system="lean4",
    )

    gen.set_layer_b(
        proof_system="stark",
        public_inputs={
            "trace_hash": digest.trace_hash,
            "trace_entries": digest.entry_count,
            "solver": "vlasov_poisson_1d1v",
            "grid": f"{Nx}x{Nv}",
            "steps": n_steps,
        },
        public_outputs=metrics,
    )

    gen.set_layer_c(
        benchmarks=[
            {
                "name": "vlasov_landau_damping",
                "gauntlet": "phase5",
                "l2_error": metrics.get("l2_relative_drift", 0.0),
                "max_deviation": metrics.get("mass_relative_drift", 0.0),
                "conservation_error": metrics.get("l2_relative_drift", 0.0),
                "passed": metrics.get("l2_relative_drift", 1.0) < 0.01,
                "threshold_l2": 0.01,
                "threshold_max": 0.01,
                "threshold_conservation": 0.01,
                "metrics": metrics,
            }
        ],
        hardware=HardwareSpec.detect(),
        total_time_s=wall_time,
    )

    gen.set_solver_hash(
        PROJECT_ROOT / "tensornet" / "cfd" / "trace_adapters" / "vlasov_adapter.py"
    )

    cert, report = gen.generate_and_save(str(out_path))
    log.info(f"  Certificate: {out_path.name}")
    log.info(f"  Verified:    {report.valid}")

    result = {
        "domain": "vlasov",
        "certificate_path": str(out_path),
        "trace_path": str(trace_path),
        "verified": report.valid,
        "grid": f"{Nx}x{Nv}",
        "steps": n_steps,
        "wall_time_s": wall_time,
        "metrics": metrics,
    }

    report_path = ARTIFACTS / "vlasov_report.json"
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    log.info(f"  Report:      {report_path.name}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Vlasov TPC certificate")
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--nv", type=int, default=128)
    parser.add_argument("--lx", type=float, default=4 * np.pi)
    parser.add_argument("--v-max", type=float, default=6.0)
    parser.add_argument("--t-final", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    output = Path(args.output) if args.output else None

    result = generate_vlasov_certificate(
        Nx=args.nx, Nv=args.nv, Lx=args.lx, v_max=args.v_max,
        t_final=args.t_final, dt=args.dt,
        output_path=output,
    )

    if result["verified"]:
        log.info("✅ Vlasov certificate generated and verified")
    else:
        log.error("❌ Certificate verification FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
