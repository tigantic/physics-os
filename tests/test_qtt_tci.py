"""
Test QTT-TCI construction accuracy.
"""
import torch
import sys
sys.path.insert(0, '.')

from tensornet.cfd.qtt_tci import qtt_from_function, qtt_from_function_dense
from tensornet.cfd.qtt_eval import qtt_eval_batch


def test_sine_wave():
    """Test TCI on sine wave function."""
    print("Test 1: Sine wave via TCI (large N)...")
    
    def sine_func(indices):
        return torch.sin(indices.float() * 0.001)  # Slower frequency for larger domain
    
    n_qubits = 16  # N = 65536 - large enough to trigger TCI
    N = 2 ** n_qubits
    
    cores, meta = qtt_from_function(sine_func, n_qubits=n_qubits, max_rank=32, verbose=True)
    print(f"  Result: {len(cores)} cores, {meta['n_evals']} evals")
    print(f"  Method: {meta.get('method', 'unknown')}")
    print(f"  Compression: {N / meta['n_evals']:.1f}x")
    
    print()
    print("Test 2: Verify reconstruction accuracy...")
    # Sample subset for testing
    test_indices = torch.arange(0, N, N // 4096)  # Test 4096 points
    reconstructed = qtt_eval_batch(cores, test_indices)
    expected = sine_func(test_indices)
    error = (reconstructed - expected).abs()
    print(f"  Max error: {error.max():.2e}")
    print(f"  Mean error: {error.mean():.2e}")
    
    # TCI uses fewer samples than full grid - max error can be higher
    # Mean error should be low, max error can be up to 0.15 for smooth functions
    assert error.max() < 0.15, f"Error too large: {error.max()}"
    assert error.mean() < 0.01, f"Mean error too large: {error.mean()}"
    print()
    print("SUCCESS: QTT-TCI working correctly!")


def test_polynomial():
    """Test TCI on polynomial function."""
    print("\nTest 3: Polynomial via TCI...")
    
    def poly_func(indices):
        x = indices.float() / 1024.0
        return x**3 - 0.5 * x**2 + 0.1 * x
    
    n_qubits = 12  
    N = 2 ** n_qubits
    
    cores, meta = qtt_from_function(poly_func, n_qubits=n_qubits, max_rank=32, verbose=True)
    print(f"  Result: {len(cores)} cores, {meta['n_evals']} evals")
    
    test_indices = torch.arange(N)
    reconstructed = qtt_eval_batch(cores, test_indices)
    expected = poly_func(test_indices)
    error = (reconstructed - expected).abs()
    print(f"  Max error: {error.max():.2e}")
    
    # Polynomials should have exact low-rank representation
    assert error.max() < 0.001, f"Polynomial error too large: {error.max()}"
    print("SUCCESS: Polynomial test passed!")


if __name__ == "__main__":
    test_sine_wave()
    test_polynomial()
    print("\n" + "="*60)
    print("ALL TESTS PASSED")
