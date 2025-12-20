#!/usr/bin/env python
"""
Compare tensornet DMRG against TeNPy.

Usage:
    python benchmarks/compare_tenpy.py [--save]
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

try:
    from tensornet import dmrg, heisenberg_mpo, MPS
except ImportError:
    print("tensornet not installed")
    sys.exit(1)

try:
    import tenpy
    from tenpy.networks.mps import MPS as TenPyMPS
    from tenpy.models.xxz_chain import XXZChain
    from tenpy.algorithms import dmrg as tenpy_dmrg
    HAS_TENPY = True
except ImportError:
    HAS_TENPY = False
    print("TeNPy not installed, skipping comparison")


def run_tensornet(L: int, chi: int) -> tuple[float, float]:
    """Run tensornet DMRG, return (energy, time)."""
    torch.manual_seed(42)
    H = heisenberg_mpo(L=L, J=1.0, h=0.0)
    psi = MPS.random(L=L, d=2, chi=chi)
    
    t0 = time.time()
    _, E, _ = dmrg(psi, H, num_sweeps=20, chi_max=chi, tol=1e-10)
    elapsed = time.time() - t0
    
    return float(E), elapsed


def run_tenpy(L: int, chi: int) -> tuple[float, float]:
    """Run TeNPy DMRG, return (energy, time)."""
    if not HAS_TENPY:
        return None, None
    
    model_params = {
        'L': L,
        'Jxx': 1.0,
        'Jz': 1.0,
        'hz': 0.0,
        'bc_MPS': 'finite',
    }
    model = XXZChain(model_params)
    
    psi = TenPyMPS.from_product_state(model.lat.mps_sites(), ['up'] * L, bc='finite')
    
    dmrg_params = {
        'mixer': True,
        'max_E_err': 1e-10,
        'trunc_params': {'chi_max': chi, 'svd_min': 1e-10},
        'max_sweeps': 20,
    }
    
    t0 = time.time()
    info = tenpy_dmrg.run(psi, model, dmrg_params)
    elapsed = time.time() - t0
    
    return info['E'], elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    
    systems = [
        (10, 32),
        (20, 64),
        (50, 128),
    ]
    
    results = []
    
    print("=" * 70)
    print("TENSORNET vs TENPY COMPARISON")
    print("=" * 70)
    print(f"{'L':>4} {'χ':>4} {'TensorNet E':>14} {'TeNPy E':>14} {'Diff':>12} {'TN time':>8} {'TP time':>8}")
    print("-" * 70)
    
    for L, chi in systems:
        tn_E, tn_time = run_tensornet(L, chi)
        tp_E, tp_time = run_tenpy(L, chi) if HAS_TENPY else (None, None)
        
        if tp_E is not None:
            diff = abs(tn_E - tp_E)
            print(f"{L:>4} {chi:>4} {tn_E:>14.8f} {tp_E:>14.8f} {diff:>12.2e} {tn_time:>7.2f}s {tp_time:>7.2f}s")
        else:
            print(f"{L:>4} {chi:>4} {tn_E:>14.8f} {'N/A':>14} {'N/A':>12} {tn_time:>7.2f}s {'N/A':>8}")
        
        results.append({
            "L": L,
            "chi": chi,
            "tensornet_E": tn_E,
            "tenpy_E": tp_E,
            "diff": abs(tn_E - tp_E) if tp_E else None,
            "tensornet_time": tn_time,
            "tenpy_time": tp_time,
        })
    
    print("=" * 70)
    
    if args.save:
        Path("results").mkdir(exist_ok=True)
        output = {
            "timestamp": datetime.now().isoformat(),
            "tensornet_version": "0.1.0",
            "tenpy_version": tenpy.__version__ if HAS_TENPY else None,
            "results": results,
        }
        with open("results/comparison_tenpy.json", "w") as f:
            json.dump(output, f, indent=2)
        print("Saved to results/comparison_tenpy.json")


if __name__ == "__main__":
    main()
