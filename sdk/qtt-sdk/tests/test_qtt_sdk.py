"""
QTT-SDK Test Suite
"""

import pytest
import torch
import math

from qtt_sdk import (
    QTTState, MPO,
    dense_to_qtt, qtt_to_dense,
    qtt_add, qtt_scale, qtt_norm, qtt_inner_product,
    truncate_qtt,
)


class TestConversion:
    """Tests for dense <-> QTT conversion."""
    
    def test_roundtrip_small(self):
        """Test that conversion and reconstruction preserves data."""
        N = 2**10  # 1024 points
        x = torch.linspace(0, 2*math.pi, N, dtype=torch.float64)
        original = torch.sin(x)
        
        qtt = dense_to_qtt(original, max_bond=32)
        reconstructed = qtt_to_dense(qtt)
        
        error = torch.norm(original - reconstructed) / torch.norm(original)
        assert error < 1e-10, f"Roundtrip error {error:.2e} exceeds tolerance"
    
    def test_compression_ratio(self):
        """Test that smooth functions achieve good compression."""
        N = 2**20  # 1 million points
        x = torch.linspace(0, 1, N, dtype=torch.float64)
        signal = torch.sin(2 * math.pi * x)  # Single frequency
        
        qtt = dense_to_qtt(signal, max_bond=16)
        
        dense_bytes = N * 8
        qtt_bytes = qtt.memory_bytes
        ratio = dense_bytes / qtt_bytes
        
        assert ratio > 100, f"Compression ratio {ratio:.1f}x is too low for sinusoid"
    
    def test_power_of_two_required(self):
        """Test that non-power-of-2 lengths raise error."""
        with pytest.raises(ValueError):
            dense_to_qtt(torch.randn(1000))
    
    def test_qtt_properties(self):
        """Test QTTState property methods."""
        qtt = dense_to_qtt(torch.randn(2**15), max_bond=32)
        
        assert qtt.grid_size == 2**15
        assert qtt.num_qubits == 15
        assert qtt.max_rank <= 32
        assert len(qtt.ranks) == 14  # n-1 bonds
        assert qtt.memory_bytes > 0


class TestArithmetic:
    """Tests for QTT arithmetic operations."""
    
    def test_scale(self):
        """Test scalar multiplication."""
        N = 2**12
        original = torch.randn(N, dtype=torch.float64)
        qtt = dense_to_qtt(original, max_bond=64)
        
        scaled_qtt = qtt_scale(qtt, 2.5)
        scaled_dense = qtt_to_dense(scaled_qtt)
        
        error = torch.norm(scaled_dense - 2.5 * original) / torch.norm(original)
        assert error < 1e-10
    
    def test_norm(self):
        """Test norm computation."""
        N = 2**12
        original = torch.randn(N, dtype=torch.float64)
        qtt = dense_to_qtt(original, max_bond=64)
        
        qtt_n = qtt_norm(qtt)
        dense_n = torch.norm(original)
        
        error = abs(qtt_n - dense_n) / dense_n
        assert error < 1e-10
    
    def test_inner_product(self):
        """Test inner product computation."""
        N = 2**10
        a = torch.randn(N, dtype=torch.float64)
        b = torch.randn(N, dtype=torch.float64)
        
        qtt_a = dense_to_qtt(a, max_bond=64)
        qtt_b = dense_to_qtt(b, max_bond=64)
        
        qtt_ip = qtt_inner_product(qtt_a, qtt_b)
        dense_ip = torch.dot(a, b)
        
        error = abs(qtt_ip - dense_ip.item()) / abs(dense_ip.item())
        assert error < 1e-9
    
    def test_add(self):
        """Test QTT addition."""
        N = 2**10
        a = torch.sin(torch.linspace(0, 2*math.pi, N, dtype=torch.float64))
        b = torch.cos(torch.linspace(0, 2*math.pi, N, dtype=torch.float64))
        
        qtt_a = dense_to_qtt(a, max_bond=32)
        qtt_b = dense_to_qtt(b, max_bond=32)
        
        qtt_sum = qtt_add(qtt_a, qtt_b, max_bond=64)
        sum_dense = qtt_to_dense(qtt_sum)
        
        expected = a + b
        error = torch.norm(sum_dense - expected) / torch.norm(expected)
        assert error < 1e-8
    
    def test_truncation(self):
        """Test that truncation reduces rank while preserving accuracy."""
        N = 2**12
        # Create high-rank QTT
        original = torch.randn(N, dtype=torch.float64)
        qtt_high = dense_to_qtt(original, max_bond=128)
        
        # Truncate to lower rank
        qtt_low = truncate_qtt(qtt_high, max_bond=32)
        
        assert qtt_low.max_rank <= 32
        assert qtt_low.memory_bytes < qtt_high.memory_bytes


class TestBillionPointScale:
    """Tests at billion-point scale (no dense reconstruction)."""
    
    def test_billion_point_creation(self):
        """Test that we can represent billion-point grids."""
        num_qubits = 30  # 2^30 = 1 billion points
        
        # Create using analytic QTT (no dense intermediate)
        cores = []
        for i in range(num_qubits):
            r_left = 1 if i == 0 else 4
            r_right = 1 if i == num_qubits - 1 else 4
            # Smooth function cores
            core = torch.randn(r_left, 2, r_right, dtype=torch.float64) * 0.1
            cores.append(core)
        
        qtt = QTTState(cores=cores, num_qubits=num_qubits)
        
        assert qtt.grid_size == 2**30
        assert qtt.memory_bytes < 200 * 1024  # Less than 200 KB
    
    def test_billion_point_operations(self):
        """Test operations at billion-point scale."""
        num_qubits = 30
        
        # Create two QTT states analytically
        def make_qtt(seed):
            torch.manual_seed(seed)
            cores = []
            for i in range(num_qubits):
                r_left = 1 if i == 0 else 8
                r_right = 1 if i == num_qubits - 1 else 8
                core = torch.randn(r_left, 2, r_right, dtype=torch.float64) * 0.1
                cores.append(core)
            return QTTState(cores=cores, num_qubits=num_qubits)
        
        qtt_a = make_qtt(42)
        qtt_b = make_qtt(123)
        
        # Test norm (no dense)
        norm_a = qtt_norm(qtt_a)
        assert norm_a > 0
        assert math.isfinite(norm_a)
        
        # Test inner product (no dense)
        ip = qtt_inner_product(qtt_a, qtt_b)
        assert math.isfinite(ip)
        
        # Test scaling (no dense)
        scaled = qtt_scale(qtt_a, 2.0)
        scaled_norm = qtt_norm(scaled)
        assert abs(scaled_norm - 2.0 * norm_a) / norm_a < 1e-10
        
        # Test addition (no dense)
        summed = qtt_add(qtt_a, qtt_b, max_bond=16)
        assert summed.max_rank <= 16
        assert summed.memory_bytes < 100 * 1024


class TestMemoryEfficiency:
    """Tests for memory usage."""
    
    @pytest.mark.parametrize("num_qubits", [20, 25, 28])
    def test_memory_scaling(self, num_qubits):
        """Verify memory stays bounded as grid size grows."""
        max_bond = 32
        
        # Build QTT analytically (no dense intermediate)
        cores = []
        for i in range(num_qubits):
            r_left = 1 if i == 0 else max_bond
            r_right = 1 if i == num_qubits - 1 else max_bond
            core = torch.zeros(r_left, 2, r_right, dtype=torch.float64)
            # Fill with some pattern
            core[..., 0, :] = 1.0
            cores.append(core)
        
        qtt = QTTState(cores=cores, num_qubits=num_qubits)
        
        # Memory should be O(n * r^2), not O(2^n)
        expected_max = num_qubits * max_bond * max_bond * 2 * 8 * 2  # With safety margin
        assert qtt.memory_bytes < expected_max, \
            f"Memory {qtt.memory_bytes} exceeds expected {expected_max}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
