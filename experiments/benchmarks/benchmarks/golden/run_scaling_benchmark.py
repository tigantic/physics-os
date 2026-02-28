#!/usr/bin/env python3
"""QTT Cubic Scaling Benchmark — O(log N) Validation • GPU-Native.

Runs Maxwell 3D at resolutions 128³ through 4096³ on GPU using
GPURuntime with Triton/CUDA kernels. Extracts raw QTT telemetry
(chi_max, compression ratio, QTT bytes vs dense bytes, wall time)
to validate the O(log N × r²) scaling claim.

THE RULES (from TOOLBOX.md / triton_ops.py):
1. QTT stays Native — GPU-resident torch.Tensor cores
2. SVD = rSVD (via triton_ops.qtt_round_native, NEVER full SVD)
3. Python loops = Triton Kernels (where applicable)
4. Higher scale = higher compression = lower rank (adaptive)
5. NEVER decompress to dense — kills QTT
6. Triton kernels (L2 cache optimized)
7. Adaptive rank (not fixed)
8. Start at scale

This benchmark bypasses the IP sanitizer to capture internal QTT
metrics that are stripped from the public API. It provides the
empirical evidence for:

    Dense storage:  O(N)           = N × 8 bytes per field
    QTT storage:    O(n_bits × r²) = (d × n_bits) × r² × 8 bytes per field
    Speedup:        O(N) / O(log N × r²)

Usage:
    # Full sweep (128³ → 4096³)
    python run_scaling_benchmark.py --output scaling_results.json

    # Quick test (128³ → 512³ only)
    python run_scaling_benchmark.py --n-bits 7 8 9 --output quick.json

Exit codes:
    0 — Benchmark completed, O(log N) scaling confirmed
    1 — Benchmark completed, scaling anomaly detected
    2 — Runtime error

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# n_bits → per-axis resolution → cubic grid
# 7  →  128³ =     2,097,152  (2.1M)
# 8  →  256³ =    16,777,216  (16.8M)
# 9  →  512³ =   134,217,728  (134M)
# 10 → 1024³ = 1,073,741,824  (1.07B)
# 11 → 2048³ = 8,589,934,592  (8.59B)
# 12 → 4096³ = 68,719,476,736 (68.7B)

DEFAULT_N_BITS = [7, 8, 9, 10, 11, 12]
DEFAULT_N_STEPS = 10  # Enough to prove physics, fast enough for large grids
DEFAULT_MAX_RANK = 64
N_FIELDS = 6  # Ex, Ey, Ez, Bx, By, Bz
BYTES_PER_FLOAT = 8  # float64

JOB_TIMEOUT_S = 600  # 10 min max per scale point

BANNER = r"""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║     Q T T   C U B I C   S C A L I N G   B E N C H M A R K      ║
║                                                                  ║
║            HyperTensor VM • Maxwell 3D • GPU-Native             ║
║                                                                  ║
║     128³ → 4096³ • O(log N) • Triton/CUDA • rSVD • Adaptive    ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────────────────────────────────────
# Timeout
# ─────────────────────────────────────────────────────────────────────────────


def _timeout_handler(signum: int, frame: Any) -> None:
    raise TimeoutError("Scaling benchmark job exceeded hard timeout")


# ─────────────────────────────────────────────────────────────────────────────
# Single scale-point measurement
# ─────────────────────────────────────────────────────────────────────────────


def run_scale_point(
    n_bits: int,
    n_steps: int,
    max_rank: int,
    timeout_s: int = JOB_TIMEOUT_S,
) -> dict[str, Any]:
    """Execute Maxwell 3D at a single scale point on GPU and extract telemetry.

    GPU-native execution path:
    1. Compile program (CPU — one-time)
    2. GPURuntime.execute() — all ops on CUDA via Triton/CUDA kernels
    3. Extract QTT metrics from GPUQTTTensor cores (shape reads, no data xfer)

    Parameters
    ----------
    n_bits : int
        Bits per axis (grid = (2^n_bits)³).
    n_steps : int
        Time-integration steps.
    max_rank : int
        Rank truncation ceiling (adaptive governor may use lower).
    timeout_s : int
        Hard timeout per measurement.

    Returns
    -------
    dict
        Full measurement record with QTT internals exposed.
    """
    from ontic.engine.vm.gpu_runtime import GPURuntime, GPURankGovernor
    from ontic.engine.vm.compilers import Maxwell3DCompiler

    N_per_axis = 2 ** n_bits
    N_total = N_per_axis ** 3
    n_sites = 3 * n_bits  # 3D → 3 × n_bits QTT sites per field

    dense_bytes_per_field = N_total * BYTES_PER_FLOAT
    dense_bytes_total = dense_bytes_per_field * N_FIELDS

    ts = datetime.now(timezone.utc).isoformat()
    job_id = f"scaling-maxwell3d-{N_per_axis}cube-{n_steps}steps-gpu"

    # GPU device info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_vram_bytes = torch.cuda.get_device_properties(0).total_memory

    record: dict[str, Any] = {
        "domain": "maxwell_3d",
        "backend": "gpu",
        "gpu_device": gpu_name,
        "gpu_vram_bytes": gpu_vram_bytes,
        "gpu_vram_human": _human_bytes(gpu_vram_bytes),
        "n_bits": n_bits,
        "N_per_axis": N_per_axis,
        "grid_label": f"{N_per_axis}³",
        "N_total": N_total,
        "n_steps": n_steps,
        "max_rank": max_rank,
        "n_sites": n_sites,
        "n_fields": N_FIELDS,
        "dense_bytes_per_field": dense_bytes_per_field,
        "dense_bytes_total": dense_bytes_total,
        "dense_human": _human_bytes(dense_bytes_total),
        "timestamp": ts,
        "job_id": job_id,
    }

    # Arm SIGALRM
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_s)

    try:
        # ── Compile (CPU, one-time) ─────────────────────────────────
        t_compile_start = time.monotonic()
        compiler = Maxwell3DCompiler(
            n_bits=n_bits,
            n_steps=n_steps,
        )
        program = compiler.compile()
        t_compile = time.monotonic() - t_compile_start

        # ── Execute on GPU ──────────────────────────────────────────
        # GPURankGovernor uses adaptive rank + rSVD (NEVER full SVD)
        governor = GPURankGovernor(
            max_rank=max_rank,
            rel_tol=1e-10,
            adaptive=True,
            base_rank=max_rank,
            min_rank=4,
        )
        runtime = GPURuntime(governor=governor)

        # CUDA sync for accurate timing
        torch.cuda.synchronize()
        t_exec_start = time.monotonic()
        result = runtime.execute(program)
        torch.cuda.synchronize()
        t_exec = time.monotonic() - t_exec_start

        if not result.success:
            raise RuntimeError(f"GPU execution failed: {result.error}")

        # ── Extract telemetry (below sanitizer) ─────────────────────
        # Fields are GPUQTTTensor — read shapes only, no GPU→CPU transfer
        telem = result.telemetry

        # Per-field QTT metrics (read from GPU tensor core shapes)
        field_metrics: list[dict[str, Any]] = []
        qtt_bytes_total = 0

        for fname, gpu_tensor in result.fields.items():
            ranks = gpu_tensor.ranks
            numel_qtt = gpu_tensor.numel_compressed
            qtt_bytes = numel_qtt * BYTES_PER_FLOAT
            qtt_bytes_total += qtt_bytes
            compression = gpu_tensor.compression_ratio

            field_metrics.append({
                "name": fname,
                "n_cores": gpu_tensor.n_cores,
                "ranks": ranks,
                "max_rank": max(ranks),
                "mean_rank": round(sum(ranks) / len(ranks), 2),
                "numel_compressed": numel_qtt,
                "qtt_bytes": qtt_bytes,
                "dense_bytes": dense_bytes_per_field,
                "compression_ratio": round(compression, 1),
                "device": str(gpu_tensor.device),
            })

        # Aggregate QTT metrics
        all_max_ranks = [fm["max_rank"] for fm in field_metrics]
        chi_max = max(all_max_ranks) if all_max_ranks else 0
        chi_mean = round(sum(all_max_ranks) / len(all_max_ranks), 2) if all_max_ranks else 0

        compression_ratio_total = (
            dense_bytes_total / qtt_bytes_total
            if qtt_bytes_total > 0 else float("inf")
        )

        # Conservation
        cons_err = telem.invariant_error
        cons_initial = telem.invariant_initial
        cons_final = telem.invariant_final

        # Throughput
        throughput = N_total * n_steps / (t_exec + 1e-30)

        # Scaling analysis
        # QTT storage theory: O(d × n_bits × r² × 8) per field
        # where d=3 (3D), r=chi_max
        theoretical_qtt_per_field = n_sites * (chi_max ** 2) * BYTES_PER_FLOAT * 2
        theoretical_qtt_total = theoretical_qtt_per_field * N_FIELDS

        # GPU memory usage
        gpu_mem_allocated = torch.cuda.memory_allocated()
        gpu_mem_reserved = torch.cuda.memory_reserved()

        # Adaptive rank info
        governor_peak = governor.peak_rank
        governor_mean = round(governor.mean_rank, 2)
        governor_truncations = governor.n_truncations
        governor_saturation = round(governor.saturation_rate, 4)

        record.update({
            "success": True,
            "compile_time_s": round(t_compile, 4),
            "wall_time_s": round(t_exec, 4),
            "total_time_s": round(t_compile + t_exec, 4),
            "throughput_gp_per_s": round(throughput, 1),

            # QTT metrics (the core claim)
            "chi_max": chi_max,
            "chi_mean": chi_mean,
            "qtt_bytes_total": qtt_bytes_total,
            "qtt_human": _human_bytes(qtt_bytes_total),
            "compression_ratio": round(compression_ratio_total, 1),
            "theoretical_qtt_bytes": theoretical_qtt_total,

            # Scaling evidence
            "log2_N": round(math.log2(N_total), 2),
            "dense_vs_qtt_ratio": round(dense_bytes_total / max(qtt_bytes_total, 1), 1),

            # Conservation
            "conservation_quantity": "em_energy",
            "conservation_initial": round(cons_initial, 10),
            "conservation_final": round(cons_final, 10),
            "conservation_relative_error": float(f"{cons_err:.2e}"),

            # Telemetry internals
            "scaling_class": telem.scaling_class,
            "saturation_rate": round(telem.saturation_rate, 4),
            "total_truncations": telem.total_truncations,

            # GPU-specific metrics
            "gpu_mem_allocated_bytes": gpu_mem_allocated,
            "gpu_mem_allocated_human": _human_bytes(gpu_mem_allocated),
            "gpu_mem_reserved_bytes": gpu_mem_reserved,
            "gpu_mem_reserved_human": _human_bytes(gpu_mem_reserved),

            # Adaptive rank governor
            "governor_peak_rank": governor_peak,
            "governor_mean_rank": governor_mean,
            "governor_truncations": governor_truncations,
            "governor_saturation_rate": governor_saturation,

            # Per-field breakdown
            "field_metrics": field_metrics,
        })

    except Exception as exc:
        record.update({
            "success": False,
            "error": str(exc),
            "wall_time_s": 0.0,
        })

    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        # Free GPU memory between measurements
        torch.cuda.empty_cache()

    return record


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _human_bytes(n: int | float) -> str:
    """Convert bytes to human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} EB"


def _scaling_analysis(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze scaling behavior across all scale points.

    Fits:
        wall_time vs N_total    → should be sub-linear (ideally O(log N))
        qtt_bytes vs N_total    → should be O(log N) if rank bounded
        chi_max vs n_bits       → should be bounded (constant)

    Returns analysis dict with fit results and verdict.
    """
    successful = [r for r in results if r.get("success")]
    if len(successful) < 2:
        return {"verdict": "INSUFFICIENT_DATA", "n_points": len(successful)}

    n_bits_arr = np.array([r["n_bits"] for r in successful], dtype=np.float64)
    N_total_arr = np.array([r["N_total"] for r in successful], dtype=np.float64)
    log_N_arr = np.log2(N_total_arr)
    wall_arr = np.array([r["wall_time_s"] for r in successful], dtype=np.float64)
    qtt_arr = np.array([r["qtt_bytes_total"] for r in successful], dtype=np.float64)
    chi_arr = np.array([r["chi_max"] for r in successful], dtype=np.float64)
    comp_arr = np.array([r["compression_ratio"] for r in successful], dtype=np.float64)

    analysis: dict[str, Any] = {
        "n_points": len(successful),
        "n_bits_range": [int(n_bits_arr.min()), int(n_bits_arr.max())],
        "N_range": [int(N_total_arr.min()), int(N_total_arr.max())],
    }

    # ── 1. Rank boundedness ─────────────────────────────────────────
    # If chi_max stays constant ± small growth, rank is bounded
    chi_slope = np.polyfit(n_bits_arr, chi_arr, 1)[0] if len(n_bits_arr) >= 2 else 0
    chi_growth_rate = abs(chi_slope) / max(chi_arr.mean(), 1)
    rank_bounded = chi_growth_rate < 0.1  # < 10% growth per bit

    analysis["chi_max_values"] = chi_arr.tolist()
    analysis["chi_slope_per_bit"] = round(float(chi_slope), 4)
    analysis["chi_growth_rate"] = round(chi_growth_rate, 4)
    analysis["rank_bounded"] = rank_bounded

    # ── 2. QTT memory scaling ──────────────────────────────────────
    # Fit: qtt_bytes = a + b * log2(N)  [O(log N) claim]
    # vs:  qtt_bytes = a + b * N         [O(N) null hypothesis]
    if len(log_N_arr) >= 2:
        qtt_log_fit = np.polyfit(log_N_arr, qtt_arr, 1)  # linear in log N
        qtt_log_residual = np.sum((qtt_arr - np.polyval(qtt_log_fit, log_N_arr)) ** 2)

        qtt_lin_fit = np.polyfit(N_total_arr, qtt_arr, 1)  # linear in N
        qtt_lin_residual = np.sum((qtt_arr - np.polyval(qtt_lin_fit, N_total_arr)) ** 2)

        # Power law fit: log(qtt) = a + b * log(N) → scaling exponent b
        log_qtt = np.log(qtt_arr + 1)
        log_N = np.log(N_total_arr + 1)
        if len(log_N) >= 2:
            power_fit = np.polyfit(log_N, log_qtt, 1)
            scaling_exponent = float(power_fit[0])
        else:
            scaling_exponent = 0.0

        analysis["qtt_scaling_exponent"] = round(scaling_exponent, 4)
        analysis["qtt_scales_sublinear"] = scaling_exponent < 0.5
        analysis["qtt_scales_log_N"] = scaling_exponent < 0.1
    else:
        analysis["qtt_scaling_exponent"] = None
        analysis["qtt_scales_sublinear"] = None
        analysis["qtt_scales_log_N"] = None

    # ── 3. Wall time scaling ────────────────────────────────────────
    if len(log_N_arr) >= 2:
        log_wall = np.log(wall_arr + 1e-30)
        time_power_fit = np.polyfit(log_N, log_wall, 1)
        time_scaling_exponent = float(time_power_fit[0])

        analysis["time_scaling_exponent"] = round(time_scaling_exponent, 4)
        analysis["time_scales_sublinear"] = time_scaling_exponent < 0.5
    else:
        analysis["time_scaling_exponent"] = None

    # ── 4. Compression growth ───────────────────────────────────────
    analysis["compression_ratios"] = comp_arr.tolist()
    analysis["compression_grows"] = bool(comp_arr[-1] > comp_arr[0] * 1.5) if len(comp_arr) >= 2 else None

    # ── 5. Overall verdict ──────────────────────────────────────────
    if rank_bounded and analysis.get("qtt_scales_sublinear"):
        if analysis.get("qtt_scales_log_N"):
            analysis["verdict"] = "O(log N) CONFIRMED"
        else:
            analysis["verdict"] = "SUB-LINEAR CONFIRMED (polylog)"
    elif rank_bounded:
        analysis["verdict"] = "RANK BOUNDED (scaling inconclusive)"
    else:
        analysis["verdict"] = "RANK GROWTH DETECTED"

    return analysis


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────


def print_scaling_table(results: list[dict[str, Any]]) -> None:
    """Print the scaling evidence table."""
    print("\n" + "=" * 120)
    print("QTT CUBIC SCALING EVIDENCE — Maxwell 3D — GPU-Native")
    print("=" * 120)
    print(
        f"{'Grid':<10} {'N_total':>14} {'Dense':>12} {'QTT':>12} "
        f"{'Comprn':>10} {'χ_max':>6} {'Wall(s)':>9} {'Thru(GP/s)':>14} "
        f"{'GPU_mem':>10} {'Cons Err':>10}"
    )
    print("-" * 120)

    for r in results:
        if not r.get("success"):
            print(f"{r.get('grid_label', '?'):<10} FAILED: {r.get('error', '?')}")
            continue

        print(
            f"{r['grid_label']:<10} "
            f"{r['N_total']:>14,} "
            f"{r['dense_human']:>12} "
            f"{r['qtt_human']:>12} "
            f"{r['compression_ratio']:>9.1f}× "
            f"{r['chi_max']:>6} "
            f"{r['wall_time_s']:>9.3f} "
            f"{r['throughput_gp_per_s']:>14,.0f} "
            f"{r.get('gpu_mem_allocated_human', '?'):>10} "
            f"{r['conservation_relative_error']:>10.2e}"
        )

    print("=" * 120)


def print_analysis(analysis: dict[str, Any]) -> None:
    """Print the scaling analysis."""
    print("\n" + "=" * 80)
    print("SCALING ANALYSIS")
    print("=" * 80)

    print(f"\n  Scale points:          {analysis['n_points']}")

    if "N_range" not in analysis:
        print(f"\n  VERDICT: {analysis['verdict']}")
        print("=" * 80)
        return

    print(f"  N range:               {analysis['N_range'][0]:,} → {analysis['N_range'][1]:,}")

    print(f"\n  Rank boundedness:")
    print(f"    χ_max values:        {analysis['chi_max_values']}")
    print(f"    Growth rate:         {analysis['chi_growth_rate']:.4f} (< 0.1 = bounded)")
    print(f"    Bounded:             {'✓ YES' if analysis['rank_bounded'] else '✗ NO'}")

    if analysis.get("qtt_scaling_exponent") is not None:
        print(f"\n  QTT memory scaling:")
        print(f"    Power-law exponent:  {analysis['qtt_scaling_exponent']:.4f}")
        print(f"    Sub-linear (< 0.5):  {'✓ YES' if analysis['qtt_scales_sublinear'] else '✗ NO'}")
        print(f"    O(log N) (< 0.1):    {'✓ YES' if analysis['qtt_scales_log_N'] else '✗ NO'}")

    if analysis.get("time_scaling_exponent") is not None:
        print(f"\n  Wall-time scaling:")
        print(f"    Power-law exponent:  {analysis['time_scaling_exponent']:.4f}")
        print(f"    Sub-linear (< 0.5):  {'✓ YES' if analysis['time_scales_sublinear'] else '✗ NO'}")

    if analysis.get("compression_grows") is not None:
        print(f"\n  Compression ratios:    {analysis['compression_ratios']}")
        print(f"    Grows with scale:    {'✓ YES' if analysis['compression_grows'] else '✗ NO'}")

    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  VERDICT: {analysis['verdict']:<30s}│")
    print(f"  └─────────────────────────────────────────┘")
    print("=" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="QTT Cubic Scaling Benchmark — O(log N) Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scale points (3D Maxwell):
  n_bits=7  →  128³ =     2,097,152 points  (dense:  96 MB)
  n_bits=8  →  256³ =    16,777,216 points  (dense: 768 MB)
  n_bits=9  →  512³ =   134,217,728 points  (dense:   6 GB)
  n_bits=10 → 1024³ = 1,073,741,824 points  (dense:  48 GB)
  n_bits=11 → 2048³ = 8,589,934,592 points  (dense: 384 GB)
  n_bits=12 → 4096³ = 68,719,476,736 points (dense:   3 TB)
        """,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scaling_results.json",
        help="Output JSON file (default: scaling_results.json)",
    )
    parser.add_argument(
        "--n-bits",
        nargs="*",
        type=int,
        default=None,
        help=f"Resolution values (default: {DEFAULT_N_BITS})",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=DEFAULT_N_STEPS,
        help=f"Time steps per measurement (default: {DEFAULT_N_STEPS})",
    )
    parser.add_argument(
        "--max-rank",
        type=int,
        default=DEFAULT_MAX_RANK,
        help=f"Rank truncation ceiling (default: {DEFAULT_MAX_RANK})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=JOB_TIMEOUT_S,
        help=f"Per-job timeout in seconds (default: {JOB_TIMEOUT_S})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-measurement output",
    )
    args = parser.parse_args()

    n_bits_list = args.n_bits if args.n_bits else DEFAULT_N_BITS

    # ── Banner ──────────────────────────────────────────────────────
    print(BANNER)
    print("=" * 70)
    print("QTT Cubic Scaling Benchmark v2.0 — GPU-Native")
    print("=" * 70)

    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_memory
        print(f"GPU:        {gpu_name}")
        print(f"VRAM:       {_human_bytes(gpu_vram)}")
        print(f"CUDA:       {torch.version.cuda}")
    else:
        print("WARNING:    No CUDA device detected — will fail")

    print(f"Backend:    Triton/CUDA • rSVD • Adaptive rank")
    print(f"Domain:     Maxwell 3D (6 fields: Ex,Ey,Ez,Bx,By,Bz)")
    print(f"Scales:     {', '.join(f'{2**nb}³' for nb in n_bits_list)}")
    print(f"Steps:      {args.n_steps}")
    print(f"Max rank:   {args.max_rank} (adaptive governor)")
    print(f"Timeout:    {args.timeout}s per point")
    print(f"Output:     {args.output}")
    print("=" * 70)

    # ── Run sweep ───────────────────────────────────────────────────
    results: list[dict[str, Any]] = []
    t_total_start = time.monotonic()

    for idx, nb in enumerate(n_bits_list, 1):
        N_per = 2 ** nb
        N_tot = N_per ** 3
        dense_total = N_tot * N_FIELDS * BYTES_PER_FLOAT

        print(
            f"\n[{idx}/{len(n_bits_list)}] Running {N_per}³ "
            f"(N={N_tot:,}, dense={_human_bytes(dense_total)})..."
        )

        t0 = time.monotonic()
        record = run_scale_point(
            n_bits=nb,
            n_steps=args.n_steps,
            max_rank=args.max_rank,
            timeout_s=args.timeout,
        )
        elapsed = time.monotonic() - t0

        results.append(record)

        if record.get("success"):
            print(
                f"  ✓ {N_per}³: wall={record['wall_time_s']:.3f}s, "
                f"χ_max={record['chi_max']}, "
                f"QTT={record['qtt_human']}, "
                f"compression={record['compression_ratio']:.1f}×, "
                f"cons_err={record['conservation_relative_error']:.2e}, "
                f"GPU_mem={record.get('gpu_mem_allocated_human', '?')} "
                f"({elapsed:.1f}s)"
            )
        else:
            print(f"  ✗ {N_per}³: FAILED — {record.get('error', '?')} ({elapsed:.1f}s)")

    t_total = time.monotonic() - t_total_start

    # ── Print results table ─────────────────────────────────────────
    print_scaling_table(results)

    # ── Scaling analysis ────────────────────────────────────────────
    analysis = _scaling_analysis(results)
    print_analysis(analysis)

    # ── Write results ───────────────────────────────────────────────
    output_path = Path(args.output)
    output_data = {
        "_meta": {
            "benchmark": "QTT Cubic Scaling Benchmark v2.0 — GPU-Native",
            "domain": "maxwell_3d",
            "backend": "gpu",
            "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
            "gpu_vram_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
            "cuda_version": torch.version.cuda or "none",
            "n_fields": N_FIELDS,
            "n_steps": args.n_steps,
            "max_rank": args.max_rank,
            "rank_governor": "GPURankGovernor (adaptive + rSVD)",
            "n_bits_sweep": n_bits_list,
            "total_time_s": round(t_total, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "measurements": results,
        "scaling_analysis": analysis,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nTotal time: {t_total:.1f}s")
    print(f"Results:    {output_path}")

    # Exit code
    if analysis.get("verdict", "").startswith("O(log N)"):
        return 0
    elif "CONFIRMED" in analysis.get("verdict", ""):
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
