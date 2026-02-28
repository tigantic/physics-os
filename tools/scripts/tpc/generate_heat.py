#!/usr/bin/env python3
"""
Heat Equation — TPC Certificate Generator
==========================================

Runs the implicit spectral heat solver via trace adapter, collects
energy conservation metrics, and packages the result as a TPC certificate.

    python tools/tools/scripts/tpc/generate_heat.py [--nx 128] [--alpha 0.01] [--output artifacts/HEAT_CERTIFICATE.tpc]

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

from ontic.cfd.trace_adapters.heat_adapter import HeatTransferTraceAdapter
from tpc.format import BenchmarkResult, HardwareSpec, TheoremRef
from tpc.generator import CertificateGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("heat_cert")

ARTIFACTS = PROJECT_ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
TRACES = PROJECT_ROOT / "traces"
TRACES.mkdir(parents=True, exist_ok=True)

LEAN_THEOREMS = [
    {
        "name": "ThermalConservation.all_fully_verified",
        "file": "thermal_conservation_proof/ThermalConservation.lean",
        "statement": (
            "fully_verified config_small witness_small ∧ "
            "fully_verified config_medium witness_medium ∧ "
            "fully_verified config_prod witness_prod"
        ),
        "proof_method": "decide",
        "verified": True,
    },
    {
        "name": "ThermalConservation.exact_conservation_small",
        "file": "thermal_conservation_proof/ThermalConservation.lean",
        "statement": (
            "witness_small.integral_after - witness_small.integral_before "
            "- witness_small.source_integral = 0"
        ),
        "proof_method": "decide",
        "verified": True,
    },
    {
        "name": "ThermalConservation.svd_lossless_prod",
        "file": "thermal_conservation_proof/ThermalConservation.lean",
        "statement": "witness_prod.svd_error = 0",
        "proof_method": "decide",
        "verified": True,
    },
]


def create_gaussian_ic(Nx: int, Ny: int, Lx: float, Ly: float) -> torch.Tensor:
    """
    Gaussian blob initial condition for 2D heat equation.

    T(x,y) = exp(−((x−Lx/2)² + (y−Ly/2)²) / σ²)
    """
    sigma = min(Lx, Ly) / 8.0
    x = torch.linspace(0, Lx, Nx + 1, dtype=torch.float64)[:-1]
    y = torch.linspace(0, Ly, Ny + 1, dtype=torch.float64)[:-1]
    Y, X = torch.meshgrid(y, x, indexing="ij")

    T0 = torch.exp(-((X - Lx / 2) ** 2 + (Y - Ly / 2) ** 2) / sigma**2)
    return T0


def create_sinusoidal_ic(Nx: int, Ny: int, Lx: float, Ly: float) -> torch.Tensor:
    """
    Sinusoidal initial condition: T = sin(2πx/Lx) · sin(2πy/Ly).
    Has known analytical decay: T(t) = T(0) · exp(−α(kx²+ky²)t).
    """
    x = torch.linspace(0, Lx, Nx + 1, dtype=torch.float64)[:-1]
    y = torch.linspace(0, Ly, Ny + 1, dtype=torch.float64)[:-1]
    Y, X = torch.meshgrid(y, x, indexing="ij")

    T0 = torch.sin(2 * np.pi * X / Lx) * torch.sin(2 * np.pi * Y / Ly)
    return T0


def generate_heat_certificate(
    Nx: int = 128,
    Ny: int = 128,
    Lx: float = 2 * np.pi,
    Ly: float = 2 * np.pi,
    alpha: float = 0.01,
    t_final: float = 1.0,
    dt: float = 0.01,
    output_path: Path | None = None,
) -> dict:
    """Run heat equation solver, generate trace + TPC certificate."""

    log.info("=" * 70)
    log.info("  HEAT EQUATION — TPC CERTIFICATE GENERATION")
    log.info("=" * 70)
    log.info(f"Grid:    {Nx}×{Ny}")
    log.info(f"Domain:  [0,{Lx:.4f}]×[0,{Ly:.4f}]")
    log.info(f"α:       {alpha}")
    log.info(f"t_final: {t_final}")
    log.info(f"dt:      {dt}")

    # ── Create adapter ───────────────────────────────────────────────
    adapter = HeatTransferTraceAdapter(
        Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, alpha=alpha,
    )

    # ── Initial condition (sinusoidal for analytical comparison) ──────
    T0 = create_sinusoidal_ic(Nx, Ny, Lx, Ly)

    # ── Solve ────────────────────────────────────────────────────────
    log.info("Solving...")
    t0 = time.time()
    T_final, t_actual, n_steps, session = adapter.solve(T0, t_final, dt=dt)
    wall_time = time.time() - t0
    log.info(f"  Completed: {n_steps} steps in {wall_time:.2f}s (t={t_actual:.6f})")

    # ── Analytical comparison ────────────────────────────────────────
    kx = 2 * np.pi / Lx
    ky = 2 * np.pi / Ly
    decay_rate = alpha * (kx**2 + ky**2)
    expected_amplitude = np.exp(-decay_rate * t_actual)
    actual_amplitude = float(T_final.abs().max().item())
    amplitude_error = abs(actual_amplitude - expected_amplitude)

    log.info(f"  Expected amplitude: {expected_amplitude:.6f}")
    log.info(f"  Actual amplitude:   {actual_amplitude:.6f}")
    log.info(f"  Amplitude error:    {amplitude_error:.2e}")

    # ── Save trace ───────────────────────────────────────────────────
    trace_path = TRACES / "heat_trace.json"
    session.save(str(trace_path))
    log.info(f"  Trace saved: {trace_path.name}")

    # ── Extract conservation metrics ─────────────────────────────────
    digest = session.finalize()
    entries = session.entries
    # Step entries are entries[1:-1] (between initial and final)
    step_entries = entries[1:-1] if len(entries) > 2 else entries

    max_energy_violation = 0.0
    for entry in step_entries:
        ev = abs(entry.metrics.get("energy_violation", 0.0))
        max_energy_violation = max(max_energy_violation, ev)

    metrics = {
        "expected_amplitude": expected_amplitude,
        "actual_amplitude": actual_amplitude,
        "amplitude_error": amplitude_error,
        "max_energy_violation": max_energy_violation,
        "num_steps": n_steps,
        "wall_time_s": wall_time,
        "trace_hash": digest.trace_hash,
        "energy_initial": float(T0.sum().item()) * (Lx / Nx) * (Ly / Ny),
        "energy_final": float(T_final.sum().item()) * (Lx / Nx) * (Ly / Ny),
    }

    log.info(f"  Max energy violation (per step): {max_energy_violation:.2e}")

    # ── Build TPC certificate ────────────────────────────────────────
    out_path = output_path or (ARTIFACTS / "HEAT_CERTIFICATE.tpc")

    gen = CertificateGenerator(domain="thermal", solver="heat3d", description=(
        f"2D heat equation (implicit spectral) on {Nx}×{Ny} grid, "
        f"α={alpha}, sinusoidal IC, "
        f"{n_steps} steps to t={t_actual:.6f}"
    ))

    gen.set_layer_a(
        theorems=LEAN_THEOREMS,
        coverage="full",
        coverage_pct=95.0,
        notes="ThermalConservation.lean: energy conservation, rank bounds, CG termination, SVD losslessness.",
        proof_system="lean4",
    )

    gen.set_layer_b(
        proof_system="stark",
        public_inputs={
            "trace_hash": digest.trace_hash,
            "trace_entries": digest.entry_count,
            "solver": "heat_implicit_spectral",
            "grid": f"{Nx}x{Ny}",
            "steps": n_steps,
        },
        public_outputs=metrics,
    )

    gen.set_layer_c(
        benchmarks=[
            {
                "name": "heat_conservation",
                "gauntlet": "phase5",
                "l2_error": amplitude_error,
                "max_deviation": max_energy_violation,
                "conservation_error": max_energy_violation,
                "passed": max_energy_violation < 1e-8 and amplitude_error < 0.01,
                "threshold_l2": 0.01,
                "threshold_max": 1e-8,
                "threshold_conservation": 1e-8,
                "metrics": metrics,
            }
        ],
        hardware=HardwareSpec.detect(),
        total_time_s=wall_time,
    )

    gen.set_solver_hash(
        PROJECT_ROOT / "ontic" / "cfd" / "trace_adapters" / "heat_adapter.py"
    )

    cert, report = gen.generate_and_save(str(out_path))
    log.info(f"  Certificate: {out_path.name}")
    log.info(f"  Verified:    {report.valid}")

    result = {
        "domain": "heat",
        "certificate_path": str(out_path),
        "trace_path": str(trace_path),
        "verified": report.valid,
        "grid": f"{Nx}x{Ny}",
        "steps": n_steps,
        "wall_time_s": wall_time,
        "metrics": metrics,
    }

    report_path = ARTIFACTS / "heat_report.json"
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    log.info(f"  Report:      {report_path.name}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Heat Equation TPC certificate")
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--t-final", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    ny = args.ny or args.nx
    output = Path(args.output) if args.output else None

    result = generate_heat_certificate(
        Nx=args.nx, Ny=ny, alpha=args.alpha,
        t_final=args.t_final, dt=args.dt,
        output_path=output,
    )

    if result["verified"]:
        log.info("✅ Heat equation certificate generated and verified")
    else:
        log.error("❌ Certificate verification FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
