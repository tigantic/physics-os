"""
Benchmark: Rust TCI vs Python TCI

Compares performance and sample counts between the Rust TCI sampler
and the pure-Python fallback implementation.
"""

import time
import numpy as np
import torch
import tensornet.cfd.qtt_tci as qtt_mod
from tensornet.cfd.qtt_tci import qtt_from_function


def test_fn(idx):
    """Test function: smooth sinusoidal with Gaussian envelope."""
    x = idx.float() / 65536.0
    return torch.sin(2 * np.pi * x) * torch.exp(-x**2)


def benchmark():
    print("=" * 60)
    print("Rust TCI vs Python TCI Benchmark")
    print("=" * 60)
    print()

    results = []

    for n_qubits in [12, 14, 16]:
        N = 2**n_qubits
        print(f"--- N = {N:,} (2^{n_qubits}) ---")

        # Python TCI
        qtt_mod.RUST_AVAILABLE = False
        t0 = time.perf_counter()
        cores_py, meta_py = qtt_from_function(test_fn, n_qubits, max_rank=16)
        py_time = time.perf_counter() - t0
        py_evals = meta_py.get("n_evals", N)

        # Rust TCI
        qtt_mod.RUST_AVAILABLE = True
        t0 = time.perf_counter()
        cores_rust, meta_rust = qtt_from_function(test_fn, n_qubits, max_rank=16)
        rust_time = time.perf_counter() - t0
        rust_evals = meta_rust.get("n_evals", N)

        speedup = py_time / rust_time if rust_time > 0 else float("inf")
        eval_ratio = py_evals / rust_evals if rust_evals > 0 else float("inf")

        print(f"  Python: {py_time:.3f}s  ({py_evals:,} evals)")
        print(f"  Rust:   {rust_time:.3f}s  ({rust_evals:,} evals)")
        print(f"  Speedup: {speedup:.2f}x time, {eval_ratio:.2f}x fewer evals")
        print()

        results.append({
            "n_qubits": n_qubits,
            "N": N,
            "py_time": py_time,
            "py_evals": py_evals,
            "rust_time": rust_time,
            "rust_evals": rust_evals,
            "speedup": speedup,
        })

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for r in results:
        print(f"  2^{r['n_qubits']}: {r['speedup']:.1f}x faster")


if __name__ == "__main__":
    benchmark()
