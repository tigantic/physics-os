#!/usr/bin/env python3
"""
Trustless Ahmed Body — Full ZK-Proof Run
==========================================

Runs the QTT Navier-Stokes Ahmed Body solver with complete
cryptographic proof generation:

    • Per-timestep SHA-256 state commitments
    • 7 physics invariants verified at every step
    • Hash-chain linking all timesteps
    • Merkle tree for O(log n) step verification
    • Run-level convergence + conservation proofs
    • Self-verifying certificate with SHA-256 seal

The output certificate can be verified offline without
re-running the simulation or having access to GPU hardware.

Usage:
    cd HyperTensor-VM-main
    PYTHONPATH="$PWD:$PYTHONPATH" python3 tools/tools/scripts/run_trustless_ahmed.py \\
        --n-bits 7 --max-rank 48 --steps 200 --cfl 0.08

    # Verify the certificate (no GPU needed):
    PYTHONPATH="$PWD:$PYTHONPATH" python3 tools/tools/scripts/run_trustless_ahmed.py \\
        --verify ahmed_ib_results/trustless_certificate.json

    # Verify a specific step:
    PYTHONPATH="$PWD:$PYTHONPATH" python3 tools/tools/scripts/run_trustless_ahmed.py \\
        --verify ahmed_ib_results/trustless_certificate.json --step 42

Author: Brad Adams / Tigantic Holdings LLC
Date: February 2026
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ── Ensure repo root is on PYTHONPATH ──────────────────────────────
_tools_dir = Path(__file__).resolve().parent.parent   # tools/
ROOT = _tools_dir.parent                              # repo root
sys.path.insert(0, str(_tools_dir))
sys.path.insert(0, str(ROOT))

from scripts.ahmed_body_ib_solver import (
    AhmedBodyConfig,
    AhmedBodyIBSolver,
    AhmedBodyParams,
    generate_report,
)
from scripts.trustless_physics import (
    TrustlessCertificate,
    TrustlessPhysicsProver,
    generate_proof_report,
    verify_certificate,
    verify_step,
)


def run_trustless(args: argparse.Namespace) -> None:
    """Run solver with full trustless proof generation."""
    n_bits = args.n_bits
    N = 1 << n_bits

    print(f"\n{'═' * 72}")
    print(f"  TRUSTLESS PHYSICS — QTT AHMED BODY SOLVER")
    print(f"  Resolution: {N}³ ({N**3:,} cells)  |  Max rank: {args.max_rank}")
    print(f"  CFL: {args.cfl}  |  Integrator: {args.integrator}  |  Projection: {args.projection}")
    print(f"  Max steps: {args.steps}  |  Cs: {args.cs}")
    print(f"{'\u2550' * 72}")

    # Build configuration
    body = AhmedBodyParams(velocity=40.0)
    cfg = AhmedBodyConfig(
        n_bits=n_bits,
        max_rank=args.max_rank,
        n_steps=args.steps,
        cfl=args.cfl,
        body_params=body,
        convergence_tol=1e-4,
        smagorinsky_cs=args.cs,
        integrator=args.integrator,
        use_projection=args.projection,
    )

    # Build solver
    print(f"\n  Initializing solver …")
    t0 = time.perf_counter()
    solver = AhmedBodyIBSolver(cfg)
    t_init = time.perf_counter() - t0
    print(f"  Solver initialized in {t_init:.1f} s")

    # Save all outputs
    out_dir = ROOT / "ahmed_ib_results"
    out_dir.mkdir(exist_ok=True)

    # Create prover and run with proof (incremental JSONL)
    prover = TrustlessPhysicsProver(solver)
    incremental = out_dir / "step_proofs.jsonl" if not args.no_incremental else None
    certificate = prover.run_with_proof(verbose=True, incremental_path=incremental)

    wall_time = certificate.wall_time_s

    # Save certificate
    cert_path = out_dir / "trustless_certificate.json"
    certificate.save(cert_path)
    cert_size = cert_path.stat().st_size
    print(f"\n  ⛓  Certificate saved: {cert_path}")
    print(f"     Size: {cert_size / 1024:.1f} KB")

    # Generate proof report
    report_path = generate_proof_report(certificate, out_dir)
    print(f"  ⛓  Proof report:     {report_path}")

    # Save solver report
    solver_report = generate_report(solver, cfg, wall_time)
    report_txt = out_dir / f"{N}_trustless_report.txt"
    report_txt.write_text(solver_report, encoding="utf-8")
    print(f"  ⛓  Solver report:    {report_txt}")

    # Summary stats
    n_invariants = sum(len(sp["invariants"]) for sp in certificate.step_proofs)
    n_passed = sum(
        sum(1 for inv in sp["invariants"] if inv["satisfied"])
        for sp in certificate.step_proofs
    )

    print(f"\n  {'═' * 72}")
    print(f"  ⛓  TRUSTLESS PHYSICS SUMMARY")
    print(f"  {'═' * 72}")
    print(f"  Steps:                {certificate.total_steps}")
    print(f"  Invariant checks:     {n_invariants}")
    print(f"  Passed:               {n_passed} ({n_passed/max(n_invariants,1)*100:.1f}%)")
    print(f"  Merkle depth:         {certificate.merkle_depth}")
    print(f"  Hash-chain intact:    {'✓' if certificate.chain_intact else '✗'}")
    print(f"  Run proofs passed:    {sum(1 for rp in certificate.run_proofs if rp['satisfied'])}/{len(certificate.run_proofs)}")
    print(f"  Certificate hash:     {certificate.certificate_hash[:32]}…")
    print(f"  Wall time:            {wall_time:.1f} s")
    verdict = "✓ ALL PROOFS PASSED" if certificate.all_invariants_satisfied else "✗ SOME PROOFS FAILED"
    print(f"  VERDICT:              {verdict}")
    print(f"  {'═' * 72}")

    # Immediately verify as a self-test
    print(f"\n  ⛓  Self-verification …")
    valid = verify_certificate(cert_path, verbose=True)
    if not valid:
        print(f"\n  ✗ SELF-VERIFICATION FAILED!")
        sys.exit(1)
    print(f"\n  ✓ Self-verification passed — certificate is independently verifiable.")


def do_verify(args: argparse.Namespace) -> None:
    """Verify a certificate offline (no GPU needed)."""
    cert_path = Path(args.verify)
    if not cert_path.exists():
        print(f"  ✗ Certificate not found: {cert_path}")
        sys.exit(1)

    if args.step is not None:
        # Verify a specific step
        valid, merkle_path = verify_step(cert_path, args.step, verbose=True)
        sys.exit(0 if valid else 1)
    else:
        # Verify full certificate
        valid = verify_certificate(cert_path, verbose=True)
        sys.exit(0 if valid else 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trustless Physics — QTT Ahmed Body Solver with ZK Proof",
    )
    parser.add_argument("--n-bits", type=int, default=7, help="Bits per axis (7=128³)")
    parser.add_argument("--max-rank", type=int, default=48, help="Max TT rank χ")
    parser.add_argument("--steps", type=int, default=200, help="Max solver steps")
    parser.add_argument("--cfl", type=float, default=0.08, help="CFL number")
    parser.add_argument("--integrator", type=str, default="rk2",
                        choices=["euler", "rk2"],
                        help="Time integrator (euler or rk2)")
    parser.add_argument("--projection", action="store_true",
                        help="Enable Chorin pressure projection")
    parser.add_argument("--cs", type=float, default=0.3,
                        help="Smagorinsky constant Cs")
    parser.add_argument("--no-incremental", action="store_true",
                        help="Disable incremental JSONL step proofs")
    parser.add_argument("--verify", type=str, default=None,
                        help="Path to certificate JSON to verify (no GPU needed)")
    parser.add_argument("--step", type=int, default=None,
                        help="Specific step to verify (with --verify)")
    args = parser.parse_args()

    if args.verify:
        do_verify(args)
    else:
        run_trustless(args)


if __name__ == "__main__":
    main()
