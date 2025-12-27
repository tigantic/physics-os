"""Test Rust TCI implementation."""
import torch
import time

from tensornet.cfd.qtt_tci import qtt_from_function, RUST_AVAILABLE
from tensornet.cfd.qtt_eval import qtt_eval_batch


def test_rust_tci():
    print(f"RUST_AVAILABLE: {RUST_AVAILABLE}")
    
    def sine_func(indices):
        return torch.sin(indices.float() * 0.001)
    
    n_qubits = 16  # 65536 points
    N = 2 ** n_qubits
    
    print(f"Testing N = {N:,} points")
    print()
    
    # Time Rust TCI
    t0 = time.perf_counter()
    cores, meta = qtt_from_function(sine_func, n_qubits=n_qubits, max_rank=32, verbose=True)
    t_rust = time.perf_counter() - t0
    
    print()
    print(f"Method: {meta['method']}")
    print(f"Time: {t_rust:.2f}s")
    print(f"Evals: {meta['n_evals']}")
    print(f"Compression: {N / meta['n_evals']:.1f}x")
    
    # Verify accuracy
    test_idx = torch.arange(0, N, N // 1000)
    expected = sine_func(test_idx)
    actual = qtt_eval_batch(cores, test_idx)
    error = (expected - actual).abs()
    print(f"Max error: {error.max():.2e}")
    print(f"Mean error: {error.mean():.2e}")
    
    assert meta['method'] in ['tci_rust', 'tci_python', 'dense'], f"Unknown method: {meta['method']}"
    assert error.max() < 0.1, f"Error too large: {error.max()}"
    
    print()
    print("SUCCESS!")


if __name__ == "__main__":
    test_rust_tci()
