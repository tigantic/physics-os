#!/usr/bin/env python3
"""Rank sweep: find minimum TT rank for res < 0.05 at each grid scale.

Tests forward + adjoint solves at 128³, 256³, 512³ with varying ranks.
Reports: residual, gradient magnitude, P_pml, P_input, W_near, Q_proxy, GPU memory, time.

Usage:
    python3 experiments/benchmarks/benchmarks/rank_sweep.py                    # full sweep
    python3 experiments/benchmarks/benchmarks/rank_sweep.py --scales 256       # single scale
    python3 experiments/benchmarks/benchmarks/rank_sweep.py --scales 256 512   # two scales
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ontic.em.chu_limit_gpu import (
    ChuGPUConfig,
    PowerMetricsGPU,
    build_pml_sigma_tt_gpu,
    build_sphere_mask_tt_gpu,
    compute_adjoint_gradient_gpu,
    make_chu_gpu_config,
    spherical_mask_flat_indices_gpu,
)


def run_single_solve(
    n_bits: int,
    max_rank: int,
    n_sweeps: int,
    damping: float,
    sigma_max: float = 30.0,
    simp_p: float = 1.0,
    verbose: bool = True,
) -> dict:
    """Run a single forward + adjoint solve and return diagnostics."""
    device = torch.device("cuda")
    dtype = torch.complex64

    N = 2 ** n_bits
    config = ChuGPUConfig(
        n_bits=n_bits,
        max_rank=max_rank,
        n_sweeps=n_sweeps,
        damping=damping,
        sigma_max_init=sigma_max,
        sigma_max_final=sigma_max,
        sigma_ramp_iters=0,
        simp_p_init=simp_p,
        simp_p_final=simp_p,
        simp_p_ramp_iters=0,
        damping_init=damping,
        damping_final=damping,
        damping_ramp_iters=0,
    )

    k0_norm = config.k0_normalised
    pml = config.pml_config()

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()

    # Build PML σ
    sigma_pml_tt = build_pml_sigma_tt_gpu(
        n_bits, k0_norm, pml, device, dtype, max_rank=max_rank,
    )

    # Build design sphere mask
    design_mask_tt = build_sphere_mask_tt_gpu(
        n_bits,
        centre=(0.5, 0.5, 0.5),
        radius=config.sphere_radius_normalised,
        device=device, dtype=dtype,
        max_rank=min(max_rank, 64),
    )

    # Design flat indices
    design_flat_idx = spherical_mask_flat_indices_gpu(
        n_bits, (0.5, 0.5, 0.5),
        config.sphere_radius_normalised, device,
    )
    n_design = design_flat_idx.shape[0]

    # Initialize density: vol_target with ±0.10 noise (matches optimizer)
    noise_amp = 0.10
    density = config.vol_target + noise_amp * (
        2.0 * torch.rand(n_design, device=device, dtype=torch.float64) - 1.0
    )
    density = density.clamp(0.01, 0.99)

    # Seed wire along z-axis
    h_grid = 1.0 / N
    iz_flat = design_flat_idx % N
    iy_flat = (design_flat_idx // N) % N
    ix_flat = design_flat_idx // (N * N)
    ix_c = min(int(0.5 * N), N - 1)
    iy_c = min(int(0.5 * N), N - 1)
    wire_r_cells = max(1, int(config.feed_seed_clamp_radius * 0.3))
    wire_mask = (
        ((ix_flat - ix_c).abs() <= wire_r_cells)
        & ((iy_flat - iy_c).abs() <= wire_r_cells)
    )
    density[wire_mask] = 0.95

    # Run full adjoint gradient computation (forward + adjoint solve)
    J_val, grad, metrics, residual = compute_adjoint_gradient_gpu(
        density=density,
        design_mask_tt=design_mask_tt,
        sigma_pml_tt=sigma_pml_tt,
        design_flat_indices=design_flat_idx,
        n_bits=n_bits,
        k0_norm=k0_norm,
        domain_size=config.domain_size,
        pml_cells=config.pml_cells,
        pml_sigma_max=config.pml_sigma_max,
        sigma_min=config.sigma_min,
        sigma_max=sigma_max,
        simp_p=simp_p,
        beta=1.0,
        eta=0.5,
        filter_radius=min(2, max(1, n_bits - 6)),  # scale filter with grid
        damping=damping,
        max_rank=max_rank,
        n_sweeps=n_sweeps,
        solver_tol=1e-4,
        source_width=config.source_width,
        alpha_loss=0.0,  # no conductor penalty for rank sweep
        use_log=True,
        device=device,
        dtype=dtype,
        verbose=verbose,
    )

    elapsed = time.perf_counter() - t0
    gpu_peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    grad_max = grad.abs().max().item()
    grad_norm = grad.norm().item()

    result = {
        "n_bits": n_bits,
        "N": N,
        "max_rank": max_rank,
        "n_sweeps": n_sweeps,
        "damping": damping,
        "sigma_max": sigma_max,
        "simp_p": simp_p,
        "n_design": n_design,
        "residual": residual,
        "J": J_val,
        "grad_max": grad_max,
        "grad_norm": grad_norm,
        "P_pml": metrics.P_pml,
        "P_input": metrics.P_input,
        "P_cond": metrics.P_cond,
        "W_near": metrics.W_near,
        "Q_proxy": metrics.Q_proxy,
        "Q_rad": metrics.Q_rad,
        "eta_rad": metrics.eta_rad,
        "time_s": elapsed,
        "gpu_peak_MB": gpu_peak,
    }

    if verbose:
        print(
            f"  {N}³  rank={max_rank:3d}  sweeps={n_sweeps:3d}  "
            f"δ={damping:.3f}  res={residual:.4f}  "
            f"gmax={grad_max:.2e}  |g|={grad_norm:.2e}  "
            f"J={J_val:.3f}  Q̃={metrics.Q_proxy:.2f}  Qr={metrics.Q_rad:.2f}  "
            f"Pi={metrics.P_input:.2e}  Pp={metrics.P_pml:.2e}  "
            f"t={elapsed:.1f}s  GPU={gpu_peak:.0f}MB"
        )

    # Free GPU memory
    del density, grad, design_mask_tt, sigma_pml_tt, design_flat_idx
    torch.cuda.empty_cache()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="TT Rank Sweep for Helmholtz solver")
    parser.add_argument(
        "--scales", nargs="+", default=["128", "256", "512"],
        help="Grid scales to test (default: 128 256 512)",
    )
    parser.add_argument(
        "--ranks", nargs="+", type=int, default=None,
        help="Override rank list (default: scale-dependent)",
    )
    parser.add_argument(
        "--damping", type=float, default=None,
        help="Override damping (default: scale-dependent)",
    )
    parser.add_argument(
        "--output", type=str, default="artifacts/rank_sweep.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    # Scale-dependent configurations
    scale_configs = {
        "128": {
            "n_bits": 7, "n_sweeps": 40,
            "damping": 0.02,
            "ranks": [16, 24, 32, 48, 64, 96],
        },
        "256": {
            "n_bits": 8, "n_sweeps": 50,
            "damping": 0.02,
            "ranks": [16, 24, 32, 48, 64, 96, 128],
        },
        "512": {
            "n_bits": 9, "n_sweeps": 60,
            "damping": 0.03,
            "ranks": [16, 24, 32, 48, 64, 96],
        },
        "1024": {
            "n_bits": 10, "n_sweeps": 60,
            "damping": 0.04,
            "ranks": [24, 32, 48, 64],
        },
    }

    print("=" * 80)
    print("  TT Rank Sweep — Helmholtz Solver Convergence Analysis")
    print("=" * 80)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 80)

    all_results: list[dict] = []

    for scale_str in args.scales:
        if scale_str not in scale_configs:
            print(f"  WARNING: no config for scale {scale_str}, skipping")
            continue

        sc = scale_configs[scale_str]
        n_bits = sc["n_bits"]
        n_sweeps = sc["n_sweeps"]
        damping = args.damping if args.damping is not None else sc["damping"]
        ranks = args.ranks if args.ranks is not None else sc["ranks"]

        N = 2 ** n_bits
        print(f"\n{'─' * 80}")
        print(f"  Scale: {N}³ ({n_bits} bits/dim), damping={damping}, sweeps={n_sweeps}")
        print(f"  Ranks: {ranks}")
        print(f"{'─' * 80}")

        for rank in ranks:
            try:
                result = run_single_solve(
                    n_bits=n_bits,
                    max_rank=rank,
                    n_sweeps=n_sweeps,
                    damping=damping,
                    verbose=True,
                )
                all_results.append(result)

                # Early termination: if res < 0.01, skip higher ranks
                if result["residual"] < 0.01:
                    print(f"    → res < 0.01, skipping higher ranks for {N}³")
                    break
            except Exception as e:
                print(f"    FAILED rank={rank}: {e}")
                all_results.append({
                    "n_bits": n_bits, "N": N, "max_rank": rank,
                    "error": str(e),
                })

    # Summary table
    print(f"\n{'=' * 80}")
    print("  SUMMARY: Residual vs Rank")
    print(f"{'=' * 80}")
    print(f"  {'N':>6s}  {'rank':>5s}  {'res':>8s}  {'gmax':>10s}  {'Q̃':>6s}  {'Qr':>6s}  {'time':>7s}  {'GPU':>7s}")
    print(f"  {'─'*6}  {'─'*5}  {'─'*8}  {'─'*10}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*7}")
    for r in all_results:
        if "error" in r:
            print(f"  {r['N']:>6d}  {r['max_rank']:>5d}  {'ERROR':>8s}")
            continue
        res_str = f"{r['residual']:.4f}"
        marker = " ✓" if r["residual"] < 0.05 else " ✗" if r["residual"] > 0.20 else ""
        print(
            f"  {r['N']:>6d}  {r['max_rank']:>5d}  {res_str:>8s}{marker}"
            f"  {r['grad_max']:>10.2e}"
            f"  {r['Q_proxy']:>6.2f}"
            f"  {r['Q_rad']:>6.2f}"
            f"  {r['time_s']:>6.1f}s"
            f"  {r['gpu_peak_MB']:>6.0f}M"
        )

    # Find minimum rank for res < 0.05 at each scale
    print(f"\n  MINIMUM RANK FOR res < 0.05:")
    scales_seen = sorted(set(r.get("N", 0) for r in all_results if "error" not in r))
    for N in scales_seen:
        scale_results = [r for r in all_results if r.get("N") == N and "error" not in r]
        passing = [r for r in scale_results if r["residual"] < 0.05]
        if passing:
            best = min(passing, key=lambda x: x["max_rank"])
            print(f"    {N}³: rank ≥ {best['max_rank']} (res={best['residual']:.4f}, GPU={best['gpu_peak_MB']:.0f}MB)")
        else:
            worst = min(scale_results, key=lambda x: x["residual"]) if scale_results else None
            if worst:
                print(f"    {N}³: rank > {worst['max_rank']} needed (best res={worst['residual']:.4f})")
            else:
                print(f"    {N}³: no valid results")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gpu": torch.cuda.get_device_name(0),
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"\n  Results → {args.output}")


if __name__ == "__main__":
    main()
