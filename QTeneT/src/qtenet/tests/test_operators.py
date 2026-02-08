"""
Unit tests for QTeneT operators: shift_nd and apply_shift.
"""
import pytest
import numpy as np
from qtenet.operators import shift_nd, apply_shift


class TestShiftND:
    """Test N-dimensional shift operator construction."""

    def test_shift_nd_returns_list(self):
        """shift_nd should return a list of cores."""
        total_qubits = 6
        shift = shift_nd(total_qubits=total_qubits, num_dims=3, axis=0, direction=1)
        assert isinstance(shift, list)
        assert len(shift) == total_qubits

    def test_shift_nd_core_shapes(self):
        """Each core should be 4D MPO tensor."""
        total_qubits = 8
        shift = shift_nd(total_qubits=total_qubits, num_dims=4, axis=1, direction=1)
        
        for i, core in enumerate(shift):
            assert core.ndim == 4, f"Core {i} should be 4D"
            # Physical dimensions should be 2 (qubit)
            assert core.shape[1] == 2
            assert core.shape[2] == 2

    @pytest.mark.parametrize("num_dims", [2, 3, 5, 6])
    def test_shift_nd_various_dims(self, num_dims):
        """shift_nd should work for various dimension counts."""
        n_qubits_per_dim = 2
        total_qubits = num_dims * n_qubits_per_dim
        
        shift = shift_nd(total_qubits=total_qubits, num_dims=num_dims, axis=0, direction=1)
        assert len(shift) == total_qubits

    @pytest.mark.parametrize("direction", [-1, 1])
    def test_shift_nd_directions(self, direction):
        """Both shift directions should work."""
        shift = shift_nd(total_qubits=6, num_dims=3, axis=0, direction=direction)
        assert shift is not None
        assert len(shift) == 6


class TestApplyShift:
    """Test applying shift to QTT cores."""

    def test_apply_shift_preserves_core_count(self):
        """Shifting should preserve number of cores."""
        from qtenet.tci import from_function
        import torch
        
        total_qubits = 6
        num_dims = 3
        
        # Create simple QTT
        def func(idx):
            return idx.float()
        
        cores = from_function(f=func, n_qubits=total_qubits, max_rank=4)
        
        # Apply shift
        shift = shift_nd(total_qubits=total_qubits, num_dims=num_dims, axis=0, direction=1)
        shifted = apply_shift(cores, shift, max_rank=16)
        
        assert len(shifted) == len(cores)

    def test_apply_shift_bounded_rank(self):
        """Shifted QTT should respect max_rank."""
        from qtenet.tci import from_function
        import torch
        
        total_qubits = 8
        num_dims = 4
        max_rank = 8
        
        def func(idx):
            return torch.sin(idx.float() * 0.1)
        
        cores = from_function(f=func, n_qubits=total_qubits, max_rank=4)
        shift = shift_nd(total_qubits=total_qubits, num_dims=num_dims, axis=0, direction=1)
        shifted = apply_shift(cores, shift, max_rank=max_rank)
        
        for core in shifted:
            assert core.shape[0] <= max_rank + 1
            assert core.shape[-1] <= max_rank + 1


class TestShiftNDCreation:
    """Test shift operator creation for multi-dimensional cases."""

    def test_6d_vlasov_shifts(self):
        """Create shift operators for 6D Vlasov (the Holy Grail)."""
        n_qubits_per_dim = 2  # Small for test
        total_qubits = 6 * n_qubits_per_dim
        
        # Position shifts (axes 0, 1, 2)
        shift_x = shift_nd(total_qubits=total_qubits, num_dims=6, axis=0, direction=1)
        shift_y = shift_nd(total_qubits=total_qubits, num_dims=6, axis=1, direction=1)
        shift_z = shift_nd(total_qubits=total_qubits, num_dims=6, axis=2, direction=1)
        
        # Velocity shifts (axes 3, 4, 5)
        shift_vx = shift_nd(total_qubits=total_qubits, num_dims=6, axis=3, direction=1)
        shift_vy = shift_nd(total_qubits=total_qubits, num_dims=6, axis=4, direction=1)
        shift_vz = shift_nd(total_qubits=total_qubits, num_dims=6, axis=5, direction=1)
        
        for shift in [shift_x, shift_y, shift_z, shift_vx, shift_vy, shift_vz]:
            assert len(shift) == total_qubits

    def test_5d_vlasov_shifts(self):
        """Create shift operators for 5D Vlasov."""
        n_qubits_per_dim = 3
        total_qubits = 5 * n_qubits_per_dim
        
        # All 5 axes
        for axis in range(5):
            shift = shift_nd(total_qubits=total_qubits, num_dims=5, axis=axis, direction=1)
            assert len(shift) == total_qubits
