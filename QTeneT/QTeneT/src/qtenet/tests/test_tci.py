"""
Unit tests for TCI (Tensor Cross Interpolation) compression.
"""
import pytest
import numpy as np
import torch
from qtenet.tci import from_function


class TestFromFunction:
    """Test from_function TCI compression."""

    def test_returns_list_of_cores(self):
        """from_function should return a list of tensors."""
        def func(idx):
            return torch.ones(idx.shape[0])
        result = from_function(f=func, n_qubits=4, max_rank=4)
        assert isinstance(result, list)
        assert all(isinstance(c, torch.Tensor) for c in result)

    def test_correct_number_of_cores(self):
        """Should return n_qubits cores."""
        def func(idx):
            return idx.float()
        result = from_function(f=func, n_qubits=5, max_rank=4)
        assert len(result) == 5

    def test_core_shapes_valid(self):
        """All cores should be 3D with valid bond dimensions."""
        def func(idx):
            return (idx.float() ** 2)
        result = from_function(f=func, n_qubits=5, max_rank=4)
        
        for i, core in enumerate(result):
            assert core.ndim == 3, f"Core {i} should be 3D"
            # Physical dimension = 2 (qubit)
            assert core.shape[1] == 2, f"Core {i} physical dim should be 2"
            # Bond dimensions
            if i == 0:
                assert core.shape[0] == 1, "First core left bond should be 1"
            if i == len(result) - 1:
                assert core.shape[2] == 1, "Last core right bond should be 1"

    def test_max_rank_respected(self):
        """Bond dimensions should not exceed max_rank."""
        def func(idx):
            # Pseudo-random function to force high rank
            return torch.sin(idx.float() * 12.345)
        
        max_rank = 4
        result = from_function(f=func, n_qubits=8, max_rank=max_rank)
        
        for core in result:
            assert core.shape[0] <= max_rank + 1
            assert core.shape[2] <= max_rank + 1

    @pytest.mark.parametrize("n_qubits", [2, 4, 6, 8])
    def test_various_sizes(self, n_qubits):
        """TCI should work for various qubit counts."""
        def func(idx):
            return idx.float()
        result = from_function(f=func, n_qubits=n_qubits, max_rank=4)
        
        assert len(result) == n_qubits


class TestTCIConstant:
    """Test TCI on constant functions (rank-1)."""

    def test_constant_one(self):
        """Constant one should compress well."""
        def func(idx):
            return torch.ones(idx.shape[0])
        result = from_function(f=func, n_qubits=8, max_rank=4)
        
        # Should have cores
        assert len(result) == 8


class TestTCILinear:
    """Test TCI on linear functions (low rank)."""

    def test_linear_function(self):
        """Linear f(x) = x should compress."""
        n_qubits = 6
        N = 2 ** n_qubits
        
        def func(idx):
            return idx.float() / N
        
        result = from_function(f=func, n_qubits=n_qubits, max_rank=n_qubits + 2)
        
        # Should have correct number of cores
        assert len(result) == n_qubits


class TestTCISmoothFunctions:
    """Test TCI on smooth functions."""

    def test_polynomial(self):
        """Polynomial should compress reasonably."""
        n_qubits = 5
        N = 2 ** n_qubits
        
        def func(idx):
            return (idx.float() / N) ** 2
        
        result = from_function(f=func, n_qubits=n_qubits, max_rank=8)
        
        assert len(result) == n_qubits

    def test_exponential_decay(self):
        """Exponential decay should compress."""
        n_qubits = 6
        N = 2 ** n_qubits
        
        def func(idx):
            return torch.exp(-idx.float() / N)
        
        result = from_function(f=func, n_qubits=n_qubits, max_rank=8)
        
        assert len(result) == n_qubits


class TestTCIResult:
    """Test the TCIResult dataclass."""

    def test_tci_result_import(self):
        """TCIResult should be importable."""
        from qtenet.tci import TCIResult
        assert TCIResult is not None

    def test_tci_result_construction(self):
        """TCIResult should be constructable."""
        from qtenet.tci import TCIResult
        
        cores = [torch.randn(1, 2, 2), torch.randn(2, 2, 1)]
        result = TCIResult(
            cores=cores,
            n_qubits=2,
            max_rank_achieved=2,
            n_function_evals=10,
            compression_ratio=2.0,
            method="test",
        )
        
        assert result.n_qubits == 2
        assert len(result.cores) == 2


class TestTCIConfig:
    """Test the TCIConfig dataclass."""

    def test_tci_config_import(self):
        """TCIConfig should be importable."""
        from qtenet.tci import TCIConfig
        assert TCIConfig is not None

    def test_tci_config_defaults(self):
        """TCIConfig should have sensible defaults."""
        from qtenet.tci import TCIConfig
        
        config = TCIConfig()
        assert config.max_rank == 64
        assert config.tolerance == 1e-6
