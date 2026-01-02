"""
Benchmark: TCI vs Hybrid Approach

Compares:
1. Dense: O(N) evaluation, O(N) flux, O(N) memory
2. Hybrid: O(N) decompress + O(N) flux + O(N) recompress
3. TCI: O(r² × n) sampling + O(log N × r⁵) construction

This is the key benchmark to determine if TCI is worth the complexity.
"""

import sys
import time

import torch

sys.path.insert(0, ".")

from tensornet.cfd.qtt_eval import dense_to_qtt_cores, qtt_eval_batch
from tensornet.cfd.qtt_tci import qtt_from_function, qtt_rusanov_flux_tci
from tensornet.cfd.tci_flux import rusanov_flux


def benchmark_dense(N: int, n_steps: int = 10):
    """Benchmark dense Rusanov flux."""
    gamma = 1.4
    x = torch.linspace(0, 1, N)

    # Sod IC
    rho = torch.where(x < 0.5, torch.ones_like(x), 0.125 * torch.ones_like(x))
    u = torch.zeros(N)
    p = torch.where(x < 0.5, torch.ones_like(x), 0.1 * torch.ones_like(x))
    rhou = rho * u
    E = p / (gamma - 1) + 0.5 * rho * u**2

    # Warmup
    _ = rusanov_flux(
        rho,
        rhou,
        E,
        torch.roll(rho, -1),
        torch.roll(rhou, -1),
        torch.roll(E, -1),
        gamma,
    )

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(n_steps):
        F_rho, F_rhou, F_E = rusanov_flux(
            rho,
            rhou,
            E,
            torch.roll(rho, -1),
            torch.roll(rhou, -1),
            torch.roll(E, -1),
            gamma,
        )
    t_total = time.perf_counter() - t0

    return {
        "method": "dense",
        "N": N,
        "time_per_step": t_total / n_steps,
        "steps": n_steps,
    }


def benchmark_hybrid(N: int, n_qubits: int, max_rank: int = 64, n_steps: int = 5):
    """Benchmark hybrid: decompress → flux → recompress."""
    gamma = 1.4
    x = torch.linspace(0, 1, N)

    # Sod IC
    rho = torch.where(x < 0.5, torch.ones_like(x), 0.125 * torch.ones_like(x))
    u = torch.zeros(N)
    p = torch.where(x < 0.5, torch.ones_like(x), 0.1 * torch.ones_like(x))
    rhou = rho * u
    E = p / (gamma - 1) + 0.5 * rho * u**2

    # Convert to QTT
    rho_cores = dense_to_qtt_cores(rho, max_rank=max_rank)
    rhou_cores = dense_to_qtt_cores(rhou, max_rank=max_rank)
    E_cores = dense_to_qtt_cores(E, max_rank=max_rank)

    all_idx = torch.arange(N)

    # Warmup
    _ = qtt_eval_batch(rho_cores, all_idx)

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(n_steps):
        # Decompress
        rho_d = qtt_eval_batch(rho_cores, all_idx)
        rhou_d = qtt_eval_batch(rhou_cores, all_idx)
        E_d = qtt_eval_batch(E_cores, all_idx)

        # Flux
        F_rho, F_rhou, F_E = rusanov_flux(
            rho_d,
            rhou_d,
            E_d,
            torch.roll(rho_d, -1),
            torch.roll(rhou_d, -1),
            torch.roll(E_d, -1),
            gamma,
        )

        # Recompress
        F_rho_cores = dense_to_qtt_cores(F_rho, max_rank=max_rank)
        F_rhou_cores = dense_to_qtt_cores(F_rhou, max_rank=max_rank)
        F_E_cores = dense_to_qtt_cores(F_E, max_rank=max_rank)

    t_total = time.perf_counter() - t0

    return {
        "method": "hybrid",
        "N": N,
        "max_rank": max_rank,
        "time_per_step": t_total / n_steps,
        "steps": n_steps,
    }


def benchmark_tci(N: int, n_qubits: int, max_rank: int = 64, n_steps: int = 3):
    """Benchmark TCI flux construction."""
    gamma = 1.4
    x = torch.linspace(0, 1, N)

    # Sod IC
    rho = torch.where(x < 0.5, torch.ones_like(x), 0.125 * torch.ones_like(x))
    u = torch.zeros(N)
    p = torch.where(x < 0.5, torch.ones_like(x), 0.1 * torch.ones_like(x))
    rhou = rho * u
    E = p / (gamma - 1) + 0.5 * rho * u**2

    # Convert to QTT
    rho_cores = dense_to_qtt_cores(rho, max_rank=max_rank)
    rhou_cores = dense_to_qtt_cores(rhou, max_rank=max_rank)
    E_cores = dense_to_qtt_cores(E, max_rank=max_rank)

    # Benchmark
    total_evals = 0
    t0 = time.perf_counter()
    for _ in range(n_steps):
        F_rho, F_rhou, F_E, meta = qtt_rusanov_flux_tci(
            rho_cores,
            rhou_cores,
            E_cores,
            gamma=gamma,
            max_rank=max_rank,
            verbose=False,
        )
        total_evals += meta["total_evals"]
    t_total = time.perf_counter() - t0

    return {
        "method": "tci",
        "N": N,
        "max_rank": max_rank,
        "time_per_step": t_total / n_steps,
        "steps": n_steps,
        "evals_per_step": total_evals / n_steps,
        "compression": 3 * N / (total_evals / n_steps),
    }


def main():
    print("=" * 70)
    print("BENCHMARK: TCI vs HYBRID vs DENSE")
    print("=" * 70)
    print()

    results = []

    # Test different grid sizes
    for n_qubits in [10, 12, 14]:
        N = 2**n_qubits
        print(f"\nGrid size N = 2^{n_qubits} = {N:,}")
        print("-" * 50)

        # Dense
        r_dense = benchmark_dense(N, n_steps=20)
        results.append(r_dense)
        print(f"  Dense:  {r_dense['time_per_step']*1000:.2f} ms/step")

        # Hybrid
        r_hybrid = benchmark_hybrid(N, n_qubits, max_rank=64, n_steps=10)
        results.append(r_hybrid)
        print(f"  Hybrid: {r_hybrid['time_per_step']*1000:.2f} ms/step")

        # TCI (fewer steps - slower)
        r_tci = benchmark_tci(N, n_qubits, max_rank=64, n_steps=3)
        results.append(r_tci)
        print(
            f"  TCI:    {r_tci['time_per_step']*1000:.2f} ms/step ({r_tci['evals_per_step']:.0f} evals, {r_tci['compression']:.1f}x compression)"
        )

    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()
    print("Current Status:")
    print("  - TCI uses Python fallback (Rust TCI not yet wired to skeleton→TT)")
    print("  - TCI advantage comes from O(r² × n) sampling vs O(N) dense")
    print("  - For N > 2^16, TCI should win on memory alone")
    print()
    print("Expected with Rust TCI:")
    print("  - 10-50x speedup from native code")
    print("  - True O(log N) complexity")
    print("  - Memory: O(n × r²) vs O(N)")
    print()
    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
