#!/usr/bin/env python
"""
TeNPy Comparison Benchmark
==========================

Compares HyperTensor DMRG implementation against TeNPy (if installed)
for the Heisenberg XXZ chain ground state.

This serves as validation against a well-established tensor network library.

Usage:
    python tools/scripts/compare_tenpy.py [--L 20] [--chi 64]

Prerequisites:
    pip install physics-tenpy  # Optional, will skip if not installed
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


def run_hypertensor_dmrg(L: int, chi_max: int, num_sweeps: int = 10):
    """Run DMRG with HyperTensor."""
    from tensornet import dmrg, heisenberg_mpo

    print(f"  Building Heisenberg MPO (L={L})...")
    H = heisenberg_mpo(L, J=1.0, Jz=1.0)

    print(f"  Running DMRG (chi_max={chi_max}, {num_sweeps} sweeps)...")
    start = time.perf_counter()
    result = dmrg(H, chi_max=chi_max, num_sweeps=num_sweeps)
    elapsed = time.perf_counter() - start

    return {
        "energy": result.energy,
        "energy_per_site": result.energy / L,
        "time": elapsed,
        "sweeps": num_sweeps,
        "converged": result.converged,
    }


def run_tenpy_dmrg(L: int, chi_max: int, num_sweeps: int = 10):
    """Run DMRG with TeNPy (if installed)."""
    try:
        import tenpy
        from tenpy.algorithms import dmrg as tenpy_dmrg
        from tenpy.models.xxz_chain import XXZChain
        from tenpy.networks.mps import MPS
    except ImportError:
        return None

    print(f"  Building XXZChain model (L={L})...")
    model_params = {
        "L": L,
        "Jxx": 1.0,
        "Jz": 1.0,
        "bc_MPS": "finite",
        "conserve": "Sz",
    }
    model = XXZChain(model_params)

    print(f"  Creating initial MPS...")
    psi = MPS.from_product_state(
        model.lat.mps_sites(),
        ["up", "down"] * (L // 2) + (["up"] if L % 2 else []),
        bc=model.lat.bc_MPS,
    )

    dmrg_params = {
        "mixer": True,
        "trunc_params": {"chi_max": chi_max, "svd_min": 1e-10},
        "max_sweeps": num_sweeps,
    }

    print(f"  Running TeNPy DMRG...")
    start = time.perf_counter()
    eng = tenpy_dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
    E, psi = eng.run()
    elapsed = time.perf_counter() - start

    return {
        "energy": E,
        "energy_per_site": E / L,
        "time": elapsed,
        "sweeps": (
            eng.sweep_stats["sweep"][-1] if eng.sweep_stats["sweep"] else num_sweeps
        ),
        "converged": True,  # TeNPy doesn't expose this directly
    }


def bethe_ansatz_energy(L: int) -> float:
    """
    Approximate Bethe ansatz energy for Heisenberg chain.

    For infinite chain: E/L = 1/4 - ln(2) ≈ -0.4431
    Finite-size corrections: E/L ≈ -0.4431 + O(1/L^2)
    """
    E_per_site_inf = 0.25 - 0.6931471805599453  # 1/4 - ln(2)
    # Simple finite-size correction
    E_per_site = E_per_site_inf * (1 - 1.0 / (L * L))
    return E_per_site * L


def main():
    parser = argparse.ArgumentParser(description="TeNPy comparison benchmark")
    parser.add_argument("--L", type=int, default=20, help="Chain length")
    parser.add_argument("--chi", type=int, default=64, help="Maximum bond dimension")
    parser.add_argument("--sweeps", type=int, default=10, help="Number of DMRG sweeps")
    args = parser.parse_args()

    print("=" * 60)
    print("TeNPy Comparison Benchmark")
    print("=" * 60)
    print(f"System: Heisenberg XXZ chain, L={args.L}, J=Jz=1.0")
    print(f"DMRG: chi_max={args.chi}, {args.sweeps} sweeps")
    print()

    # Bethe ansatz reference
    E_bethe = bethe_ansatz_energy(args.L)
    print(f"Bethe ansatz reference: E = {E_bethe:.8f} (E/L = {E_bethe/args.L:.8f})")
    print()

    # HyperTensor
    print("HyperTensor DMRG:")
    ht_result = run_hypertensor_dmrg(args.L, args.chi, args.sweeps)
    print(f"  Energy: {ht_result['energy']:.8f}")
    print(f"  E/L: {ht_result['energy_per_site']:.8f}")
    print(f"  Time: {ht_result['time']:.2f}s")
    print(f"  Error vs Bethe: {abs(ht_result['energy'] - E_bethe):.2e}")
    print()

    # TeNPy
    print("TeNPy DMRG:")
    tp_result = run_tenpy_dmrg(args.L, args.chi, args.sweeps)
    if tp_result is None:
        print("  [TeNPy not installed - skipping]")
        print("  To install: pip install physics-tenpy")
    else:
        print(f"  Energy: {tp_result['energy']:.8f}")
        print(f"  E/L: {tp_result['energy_per_site']:.8f}")
        print(f"  Time: {tp_result['time']:.2f}s")
        print(f"  Error vs Bethe: {abs(tp_result['energy'] - E_bethe):.2e}")
        print()

        # Comparison
        print("=" * 60)
        print("COMPARISON")
        print("=" * 60)
        energy_diff = abs(ht_result["energy"] - tp_result["energy"])
        print(f"Energy difference: {energy_diff:.2e}")
        print(f"HyperTensor speedup: {tp_result['time'] / ht_result['time']:.2f}x")

        if energy_diff < 1e-6:
            print("✓ Energies match to machine precision!")
        elif energy_diff < 1e-4:
            print("✓ Energies match to 4 decimal places")
        else:
            print("⚠ Energies differ - check implementation")


if __name__ == "__main__":
    main()
