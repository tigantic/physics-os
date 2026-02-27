#!/usr/bin/env python3
"""QTT Physics VM — Resolution sweep.

Runs all 5 domain compilers at n_bits = 6, 8, 10, 12, 14 to demonstrate
that χ_max remains bounded as grid resolution N = 2^n grows.

This is the constructive proof of resolution-independence: if χ does not
grow with N, the VM's memory cost is O(n · χ²) regardless of domain.

Usage::

    python tools/scripts/research/vm_resolution_sweep.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tensornet.engine.vm.runtime import QTTRuntime
from tensornet.engine.vm.rank_governor import RankGovernor, TruncationPolicy
from tensornet.engine.vm.compilers import (
    BurgersCompiler,
    MaxwellCompiler,
    SchrodingerCompiler,
    DiffusionCompiler,
    VlasovPoissonCompiler,
)

# ── Configuration ───────────────────────────────────────────────────
BITS = [6, 8, 10, 12, 14]
N_STEPS = 50          # enough to see rank stabilize
MAX_RANK = 128        # generous ceiling
REL_TOL = 1e-10

# Vlasov uses bits_x + bits_v cores; cap at 10b per dim to keep runtime sane
VLASOV_MAX_BITS = 10


def make_compiler(domain: str, n_bits: int):
    """Instantiate the right compiler for a domain and bit count."""
    if domain == "burgers":
        return BurgersCompiler(n_bits=n_bits, n_steps=N_STEPS)
    elif domain == "maxwell":
        return MaxwellCompiler(n_bits=n_bits, n_steps=N_STEPS)
    elif domain == "schrodinger":
        return SchrodingerCompiler(n_bits=n_bits, n_steps=N_STEPS)
    elif domain == "diffusion":
        return DiffusionCompiler(n_bits=n_bits, n_steps=N_STEPS)
    elif domain == "vlasov":
        # Vlasov uses 2D phase space; bits_x = bits_v = n_bits - 2
        bpd = max(4, n_bits - 2)
        return VlasovPoissonCompiler(bits_x=bpd, bits_v=bpd, n_steps=N_STEPS)
    else:
        raise ValueError(f"Unknown domain: {domain}")


def main() -> None:
    domains = ["burgers", "maxwell", "schrodinger", "diffusion", "vlasov"]
    results: list[dict] = []

    governor = RankGovernor(TruncationPolicy(max_rank=MAX_RANK, rel_tol=REL_TOL))
    runtime = QTTRuntime(governor=governor)

    hdr = (
        f"{'Domain':<20s} {'bits':>4s} {'N':>8s} {'χ_max':>6s} "
        f"{'Compress':>10s} {'Time':>8s} {'Δinv':>12s} {'Class':>5s}"
    )
    print("=" * 80)
    print("  QTT PHYSICS VM — RESOLUTION INDEPENDENCE SWEEP")
    print("=" * 80)
    print()
    print(hdr)
    print("─" * len(hdr))

    for domain in domains:
        for n_bits in BITS:
            # Skip Vlasov above cap
            if domain == "vlasov" and n_bits > VLASOV_MAX_BITS:
                continue

            compiler = make_compiler(domain, n_bits)
            program = compiler.compile()

            governor.reset()
            t0 = time.perf_counter()
            result = runtime.execute(program)
            wall = time.perf_counter() - t0

            tel = result.telemetry
            N = 2 ** n_bits

            row = {
                "domain": domain,
                "n_bits": n_bits,
                "N": N,
                "chi_max": tel.chi_max,
                "chi_mean": round(tel.chi_mean, 2),
                "compression_ratio": round(tel.compression_ratio_final, 1),
                "wall_time_s": round(wall, 2),
                "invariant_error": tel.invariant_error,
                "scaling_class": tel.scaling_class,
                "success": result.success,
            }
            results.append(row)

            status = "✓" if result.success else "✗"
            print(
                f"  {status} {domain:<17s} {n_bits:>4d} {N:>8d} "
                f"{tel.chi_max:>6d} {tel.compression_ratio_final:>10.1f}× "
                f"{wall:>7.2f}s {tel.invariant_error:>12.2e} "
                f"{tel.scaling_class:>5s}"
            )

    print("─" * len(hdr))

    # ── Resolution-independence test ────────────────────────────────
    print()
    print("  RESOLUTION-INDEPENDENCE CHECK")
    print("  ─────────────────────────────")
    all_bounded = True
    for domain in domains:
        domain_rows = [r for r in results if r["domain"] == domain]
        if len(domain_rows) < 2:
            continue
        chi_vals = [r["chi_max"] for r in domain_rows]
        bits_vals = [r["n_bits"] for r in domain_rows]
        # Check: does χ grow with N?
        ratio = chi_vals[-1] / max(chi_vals[0], 1)
        N_ratio = 2 ** (bits_vals[-1] - bits_vals[0])
        badge = "✓ BOUNDED" if ratio < 4.0 else "⚠ GROWING"
        if ratio >= 4.0:
            all_bounded = False
        print(
            f"  {domain:<17s}  χ: {chi_vals[0]} → {chi_vals[-1]}  "
            f"(×{ratio:.1f} while N grew ×{N_ratio})  {badge}"
        )

    print()
    if all_bounded:
        print("  VERDICT: χ bounded across all domains and resolutions")
        print("  Resolution-independence confirmed — the backend is the product")
    else:
        print("  VERDICT: Some domains show rank growth with resolution")

    print("=" * 80)

    # ── Save ────────────────────────────────────────────────────────
    out_path = Path("data/vm_resolution_sweep.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
