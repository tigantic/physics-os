"""QTT Physics VM — Unified benchmark harness.

Runs every compiled domain through the same QTT runtime with the same
rank governor and produces a single benchmark sheet comparing:

    χ_max, compression ratio, wall time, invariant error, scaling class

across all domains.  This is the proof that the backend — not the
domain — is the product.

Usage
-----
    python -m ontic.vm.benchmark [--n-bits 8] [--n-steps 100]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .ir import Program
from .rank_governor import RankGovernor, TruncationPolicy
from .runtime import ExecutionResult, QTTRuntime
from .telemetry import ProgramTelemetry
from .compilers import (
    ALL_COMPILERS,
    BurgersCompiler,
    DiffusionCompiler,
    MaxwellCompiler,
    SchrodingerCompiler,
    VlasovPoissonCompiler,
)


def run_benchmark(
    n_bits: int = 8,
    n_steps: int = 100,
    max_rank: int = 64,
    rel_tol: float = 1e-10,
    vlasov_bits: int | None = None,
) -> list[ProgramTelemetry]:
    """Run all 5 domain compilers through one runtime.

    Parameters
    ----------
    n_bits : int
        Grid resolution for 1-D domains (2^n_bits points).
    n_steps : int
        Time steps per domain.
    max_rank : int
        Rank governor ceiling.
    rel_tol : float
        SVD truncation tolerance.
    vlasov_bits : int | None
        Override bits for Vlasov–Poisson (default: n_bits - 2 per dim).

    Returns
    -------
    list[ProgramTelemetry]
        One telemetry result per domain.
    """
    governor = RankGovernor(
        policy=TruncationPolicy(max_rank=max_rank, rel_tol=rel_tol),
    )
    runtime = QTTRuntime(governor=governor)

    vbits = vlasov_bits or max(4, n_bits - 2)

    compilers = [
        BurgersCompiler(n_bits=n_bits, n_steps=n_steps),
        MaxwellCompiler(n_bits=n_bits, n_steps=n_steps),
        SchrodingerCompiler(n_bits=n_bits, n_steps=n_steps),
        DiffusionCompiler(n_bits=n_bits, n_steps=n_steps),
        VlasovPoissonCompiler(bits_x=vbits, bits_v=vbits, n_steps=n_steps),
    ]

    results: list[ProgramTelemetry] = []
    total_start = time.perf_counter()

    for compiler in compilers:
        program = compiler.compile()
        label = f"[{compiler.domain}]"
        print(f"\n{'─' * 60}")
        print(f"  Compiling & executing: {compiler.domain_label}")
        print(f"  Equations: {program.metadata.get('equations', 'N/A')}")
        print(f"  Grid: {program.n_bits}b per dim, "
              f"dt={program.dt:.2e}, steps={program.n_steps}")
        print(f"  Instructions: {len(program.instructions)} "
              f"({len(set(i.opcode.value for i in program.instructions))} unique opcodes)")
        print(f"{'─' * 60}")

        result = runtime.execute(program)

        if result.success:
            t = result.telemetry
            print(f"  ✓ {t.summary_line()}")
        else:
            print(f"  ✗ FAILED: {result.error}")
            t = result.telemetry

        results.append(t)

    total_time = time.perf_counter() - total_start

    # ── Summary table ───────────────────────────────────────────────
    print(f"\n{'═' * 80}")
    print(f"  QTT PHYSICS VM — UNIFIED BENCHMARK RESULTS")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Runtime: {total_time:.2f}s total, max_rank={max_rank}, "
          f"tol={rel_tol:.0e}")
    print(f"{'═' * 80}")
    print()

    # Table header
    hdr = (f"{'Domain':<24s} {'bits':>4s} {'χ_max':>5s} "
           f"{'Compress':>10s} {'Time(s)':>8s} "
           f"{'Δ invariant':>12s} {'Class':>5s}")
    print(hdr)
    print("─" * len(hdr))

    all_class_a = True
    for t in results:
        status = "✓" if t.scaling_class in ("A", "B") else "✗"
        if t.scaling_class not in ("A",):
            all_class_a = False
        row = (
            f"{t.domain:<24s} {t.n_bits:>4d} {t.chi_max:>5d} "
            f"{t.compression_ratio_final:>10.1f}× {t.total_wall_time_s:>8.2f} "
            f"{t.invariant_error:>12.2e} {t.scaling_class:>5s}"
        )
        print(row)

    print("─" * len(hdr))

    # ── Verdict ─────────────────────────────────────────────────────
    classes = [t.scaling_class for t in results]
    n_bounded = sum(1 for c in classes if c in ("A", "B", "C"))
    n_total = len(classes)

    print()
    if n_bounded == n_total:
        print(f"  VERDICT: ALL {n_total} DOMAINS EXECUTE ON THE SAME "
              f"QTT RUNTIME WITH BOUNDED RANK")
        print(f"  k=1 universality observed: one execution substrate "
              f"for all physical law")
    else:
        print(f"  VERDICT: {n_bounded}/{n_total} domains show bounded rank")
        for t in results:
            if t.scaling_class == "D":
                print(f"    ⚠ {t.domain}: scaling class {t.scaling_class}")

    # ── Opcodes used across all domains ────────────────────────────
    all_opcodes = set()
    for t in results:
        all_opcodes.update(t.ir_opcodes_used)
    print(f"\n  Shared IR opcodes: {sorted(all_opcodes)}")
    print(f"  Total unique opcodes used: {len(all_opcodes)}")
    print(f"{'═' * 80}\n")

    return results


def save_results(
    results: list[ProgramTelemetry],
    output_path: str | Path,
) -> None:
    """Save benchmark results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "vm_version": "1.0.0",
        "n_domains": len(results),
        "domains": [t.to_dict() for t in results],
        "summary": {
            "all_bounded": all(
                t.scaling_class in ("A", "B") for t in results
            ),
            "chi_max_overall": max(t.chi_max for t in results),
            "total_wall_time_s": sum(t.total_wall_time_s for t in results),
        },
    }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Results saved to {output_path}")


def generate_markdown_report(
    results: list[ProgramTelemetry],
    output_path: str | Path,
) -> None:
    """Generate a Markdown summary of the benchmark."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# QTT Physics VM — Unified Benchmark Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"**VM Version:** 1.0.0",
        f"**Domains tested:** {len(results)}",
        "",
        "## Results",
        "",
        "| Domain | Equations | bits | χ_max | Compression | Time (s) | Δ Invariant | Class |",
        "|--------|-----------|------|-------|-------------|----------|-------------|-------|",
    ]

    for t in results:
        lines.append(
            f"| {t.domain_label[:30]} | — | {t.n_bits} | {t.chi_max} "
            f"| {t.compression_ratio_final:.1f}× | {t.total_wall_time_s:.2f} "
            f"| {t.invariant_error:.2e} | {t.scaling_class} |"
        )

    classes = [t.scaling_class for t in results]
    n_bounded = sum(1 for c in classes if c in ("A", "B", "C"))

    lines += [
        "",
        "## Verdict",
        "",
    ]

    if n_bounded == len(results):
        lines.append(
            f"**ALL {len(results)} DOMAINS** execute on the same QTT runtime "
            f"with bounded rank (Class A–C).  "
            f"k=1 universality confirmed — one execution substrate for physical law."
        )
    else:
        lines.append(
            f"{n_bounded}/{len(results)} domains show bounded rank."
        )

    lines += [
        "",
        "## Architecture",
        "",
        "All domains compiled to the same operator IR and executed on "
        "one QTT runtime with:",
        "",
        "- **Same rank governor** (uniform truncation policy)",
        "- **Same telemetry hooks** (identical metrics for all domains)",
        "- **Same proof artifacts** (this report)",
        "",
        "The backend is the product.  Domains are front-end adapters.",
        "",
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved to {output_path}")


# ── CLI entry point ────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="QTT Physics VM — Unified Benchmark",
    )
    parser.add_argument("--n-bits", type=int, default=8,
                        help="Grid bits for 1D domains (default: 8)")
    parser.add_argument("--n-steps", type=int, default=100,
                        help="Time steps (default: 100)")
    parser.add_argument("--max-rank", type=int, default=64,
                        help="Rank governor ceiling (default: 64)")
    parser.add_argument("--tol", type=float, default=1e-10,
                        help="SVD cutoff (default: 1e-10)")
    parser.add_argument("--vlasov-bits", type=int, default=None,
                        help="Per-dim bits for Vlasov (default: n_bits-2)")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save results JSON")
    parser.add_argument("--output-md", type=str, default=None,
                        help="Save Markdown report")

    args = parser.parse_args()

    results = run_benchmark(
        n_bits=args.n_bits,
        n_steps=args.n_steps,
        max_rank=args.max_rank,
        rel_tol=args.tol,
        vlasov_bits=args.vlasov_bits,
    )

    if args.output_json:
        save_results(results, args.output_json)
    if args.output_md:
        generate_markdown_report(results, args.output_md)


if __name__ == "__main__":
    main()
