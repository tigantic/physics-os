"""
Unit tests for QTeneT demos.
"""
import pytest


class TestDemoImports:
    """Test that all demos are importable."""

    def test_holy_grail_6d_import(self):
        """holy_grail_6d should be importable."""
        from qtenet.demos import holy_grail_6d
        assert callable(holy_grail_6d)

    def test_holy_grail_5d_import(self):
        """holy_grail_5d should be importable."""
        from qtenet.demos import holy_grail_5d
        assert callable(holy_grail_5d)

    def test_two_stream_instability_import(self):
        """two_stream_instability should be importable."""
        from qtenet.demos import two_stream_instability
        assert callable(two_stream_instability)

    def test_result_types_import(self):
        """Result types should be importable."""
        from qtenet.demos import TwoStreamResult, HolyGrailResult
        assert TwoStreamResult is not None
        assert HolyGrailResult is not None


class TestHolyGrail:
    """Test Holy Grail demo execution."""

    @pytest.mark.slow
    def test_6d_minimal_run(self):
        """Holy Grail 6D should run with minimal parameters."""
        from qtenet.demos import holy_grail_6d, HolyGrailResult
        
        result = holy_grail_6d(
            qubits_per_dim=2,  # Minimal: 4^6 = 4096 points
            max_rank=2,
            n_steps=1,
            dt=0.01,
            verbose=False,
        )
        
        assert isinstance(result, HolyGrailResult)
        assert result.dims == 6
        assert result.n_steps == 1

    @pytest.mark.slow
    def test_5d_minimal_run(self):
        """Holy Grail 5D should run with minimal parameters."""
        from qtenet.demos import holy_grail_5d, HolyGrailResult
        
        result = holy_grail_5d(
            qubits_per_dim=2,  # Minimal: 4^5 = 1024 points
            max_rank=2,
            n_steps=1,
            dt=0.01,
            verbose=False,
        )
        
        assert isinstance(result, HolyGrailResult)
        assert result.dims == 5

    def test_result_summary(self):
        """HolyGrailResult should have a summary method."""
        from qtenet.demos import HolyGrailResult
        
        # Create a mock result
        result = HolyGrailResult(
            dims=6,
            qubits_per_dim=5,
            grid_size=32,
            total_points=1073741824,
            n_steps=10,
            max_rank=64,
            qtt_parameters=100000,
            memory_kb=400.0,
            dense_memory_gb=4.0,
            compression_ratio=10000.0,
            construction_time_s=1.0,
            total_time_s=10.0,
            time_per_step_ms=1000.0,
            final_rank=64,
            energy_conservation=0.001,
        )
        
        summary = result.summary()
        assert "HOLY GRAIL" in summary
        assert "BROKEN" in summary


class TestTwoStream:
    """Test two-stream instability demo."""

    @pytest.mark.slow
    def test_two_stream_runs(self):
        """Two-stream instability should run."""
        from qtenet.demos import two_stream_instability, TwoStreamResult
        
        result = two_stream_instability(
            qubits_per_dim=2,
            max_rank=2,
            n_steps=1,
            dt=0.01,
            verbose=False,
        )
        
        assert isinstance(result, TwoStreamResult)


class TestGenesisExports:
    """Test that genesis module exports everything."""

    def test_genesis_import(self):
        """Genesis should import without error."""
        from qtenet import genesis
        assert genesis is not None

    def test_genesis_has_layer_modules(self):
        """Genesis should have the 7 layer submodules."""
        from qtenet import genesis
        
        # The 7 Genesis layers
        assert hasattr(genesis, 'ot')        # Layer 20: Optimal Transport
        assert hasattr(genesis, 'sgw')       # Layer 21: Spectral Graph Wavelets
        assert hasattr(genesis, 'rmt')       # Layer 22: Random Matrix Theory
        assert hasattr(genesis, 'tropical')  # Layer 23: Tropical Geometry
        assert hasattr(genesis, 'rkhs')      # Layer 24: Kernel Methods
        assert hasattr(genesis, 'topology')  # Layer 25: Persistent Homology
        assert hasattr(genesis, 'ga')        # Layer 26: Geometric Algebra

    def test_genesis_has_qtt_primitives(self):
        """Genesis should have QTT primitive classes."""
        from qtenet import genesis
        
        # Core QTT classes
        assert hasattr(genesis, 'QTTDistribution')
        assert hasattr(genesis, 'QTTSinkhorn')
        assert hasattr(genesis, 'QTTLaplacian')
        assert hasattr(genesis, 'QTTEnsemble')

    def test_genesis_export_count(self):
        """Genesis should have many exports."""
        from qtenet import genesis
        
        # Count public exports
        exports = [x for x in dir(genesis) if not x.startswith('_')]
        assert len(exports) >= 20, f"Expected >=20 exports, got {len(exports)}"
