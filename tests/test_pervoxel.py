#!/usr/bin/env python3
"""Quick test: per-voxel σ vs mean-field σ at 128³.

Runs 5 optimization iterations and checks whether J moves.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from tensornet.em.chu_limit_gpu import (
    make_chu_gpu_config,
    optimize_chu_antenna_gpu,
)


def main():
    config = make_chu_gpu_config("128")
    config.max_iterations = 5
    print(f"Config: n_bits={config.n_bits}, max_rank={config.max_rank}, "
          f"lr={config.learning_rate}, sweeps={config.n_sweeps}")
    print(f"Per-voxel σ enabled: total_sites={3*config.n_bits} ≤ 24 → "
          f"{'YES' if 3*config.n_bits <= 24 else 'NO (mean-field fallback)'}")

    result = optimize_chu_antenna_gpu(config, verbose=True)
    print(f"\nFinal J={result.objective:.6f}")
    print(f"Done — {len(result.objective_history)} iters completed")


if __name__ == "__main__":
    main()
