#!/usr/bin/env python3
"""GPU Chu Limit Challenge — CLI Runner.

Usage:
    python3 experiments/benchmarks/benchmarks/gpu_chu_limit.py --scale 128     # smoke test
    python3 experiments/benchmarks/benchmarks/gpu_chu_limit.py --scale 4096    # full 4096³ run
    python3 experiments/benchmarks/benchmarks/gpu_chu_limit.py                 # default: 4096
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import hashlib
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tensornet.em.chu_limit_gpu import (
    make_chu_gpu_config,
    optimize_chu_antenna_gpu,
    ChuGPUConfig,
    ChuGPUResult,
    chu_limit_q,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU Chu Limit Challenge")
    parser.add_argument("--scale", default="4096",
                        choices=["128", "256", "512", "1024", "2048", "4096"],
                        help="Grid scale (N³)")
    parser.add_argument("--max-iter", type=int, default=None,
                        help="Override max iterations")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run (20 iterations)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config = make_chu_gpu_config(args.scale)

    if args.quick:
        config.max_iterations = 20
        config.n_sweeps = min(config.n_sweeps, 20)
        config.feed_seed_clamp_iters = 5
        config.beta_increase_every = 5
        config.sigma_ramp_iters = 10
        config.simp_p_ramp_iters = 10
        config.alpha_stable_window = 5

    if args.max_iter is not None:
        config.max_iterations = args.max_iter

    verbose = not args.quiet

    if verbose:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    result = optimize_chu_antenna_gpu(config, verbose=verbose)

    # Generate attestation
    attestation = {
        "type": "GPU_CHU_LIMIT_CHALLENGE",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
        "grid": f"{config.N}³",
        "n_bits": config.n_bits,
        "ka": config.ka,
        "Q_Chu": round(config.q_chu, 2),
        "iterations": result.n_iterations,
        "converged": result.converged,
        "time_s": round(result.total_time_s, 1),
        "gpu_peak_MB": round(
            torch.cuda.max_memory_allocated() / (1024 ** 2), 0
        ),
    }

    if result.power_metrics_history:
        last = result.power_metrics_history[-1]
        attestation.update({
            "Q_proxy": last.Q_proxy,
            "Q_rad": last.Q_rad,
            "W_near": last.W_near,
            "P_pml": last.P_pml,
            "P_input": last.P_input,
            "P_cond": last.P_cond,
            "P_pml_norm": last.P_pml_norm,
            "P_cond_norm": last.P_cond_norm,
            "eta_rad": last.eta_rad,
            "volume": last.vol,
            "M_dead": last.M_dead,
        })

    if result.objective_history:
        attestation["J_final"] = result.objective_history[-1]
        attestation["J_history_first5"] = result.objective_history[:5]
        attestation["J_history_last5"] = result.objective_history[-5:]

    attestation["density_hash"] = hashlib.sha256(
        result.density_final.cpu().numpy().tobytes()
    ).hexdigest()[:16]

    out_path = args.output or f"artifacts/gpu_chu_limit_{args.scale}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(attestation, f, indent=2, default=str)
    print(f"\n  Attestation → {out_path}")


if __name__ == "__main__":
    main()
