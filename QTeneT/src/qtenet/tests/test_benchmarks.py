"""
Unit tests for QTeneT benchmarks.
"""
import pytest


class TestBenchmarkImports:
    """Test that benchmarks are importable."""

    def test_curse_of_dimensionality_import(self):
        """curse_of_dimensionality should be importable."""
        from qtenet.benchmarks import curse_of_dimensionality
        assert callable(curse_of_dimensionality)

    def test_dimension_scaling_import(self):
        """dimension_scaling should be importable."""
        from qtenet.benchmarks import dimension_scaling
        assert callable(dimension_scaling)

    def test_rank_scaling_import(self):
        """rank_scaling should be importable."""
        from qtenet.benchmarks import rank_scaling
        assert callable(rank_scaling)

    def test_result_types_import(self):
        """Result types should be importable."""
        from qtenet.benchmarks import CurseScalingResult, BenchmarkSuite
        assert CurseScalingResult is not None
        assert BenchmarkSuite is not None


class TestBenchmarkResults:
    """Test benchmark result structures."""

    def test_dimension_scaling_returns_result(self):
        """dimension_scaling should return results."""
        from qtenet.benchmarks import dimension_scaling, BenchmarkSuite
        
        result = dimension_scaling(
            max_dims=3,  # Small for fast test
            qubits_per_dim=2,
            max_rank=2,
        )
        
        assert isinstance(result, BenchmarkSuite)

    def test_rank_scaling_returns_result(self):
        """rank_scaling should return results."""
        from qtenet.benchmarks import rank_scaling
        
        result = rank_scaling(
            dims=2,
            qubits_per_dim=2,
            ranks=[2, 4],
        )
        
        assert isinstance(result, list)
        assert len(result) > 0


class TestBenchmarkExecutables:
    """Test that benchmark scripts are executable."""

    def test_run_dimension_scaling_minimal(self):
        """Should be able to run dimension_scaling with minimal params."""
        from qtenet.benchmarks import dimension_scaling, BenchmarkSuite
        
        # Very small test
        result = dimension_scaling(
            max_dims=2,
            qubits_per_dim=2,
            max_rank=2,
        )
        
        assert isinstance(result, BenchmarkSuite)

    def test_run_rank_scaling_minimal(self):
        """Should be able to run rank_scaling with minimal params."""
        from qtenet.benchmarks import rank_scaling
        
        result = rank_scaling(
            dims=2,
            qubits_per_dim=2,
            ranks=[2],
        )
        
        assert isinstance(result, list)


class TestResultStructures:
    """Test result dataclass structures."""

    def test_curse_scaling_result_has_summary(self):
        """CurseScalingResult should have a summary method."""
        from qtenet.benchmarks import CurseScalingResult
        
        result = CurseScalingResult(
            dims=3,
            qubits_per_dim=4,
            total_qubits=12,
            grid_size=16,
            total_points=4096,
            max_rank=8,
            qtt_parameters=1000,
            dense_parameters=4096,
            compression_ratio=4.096,
            qtt_memory_bytes=4000,
            dense_memory_bytes=16384,
            construction_time_s=0.1,
            operation_time_s=0.001,
            reconstruction_error=0.01,
            theoretical_speedup=100.0,
        )
        
        summary = result.summary()
        assert "3D" in summary
        assert "Compression" in summary

    def test_benchmark_suite_has_add(self):
        """BenchmarkSuite should have an add method."""
        from qtenet.benchmarks import BenchmarkSuite
        
        suite = BenchmarkSuite()
        assert hasattr(suite, 'add')
        assert hasattr(suite, 'results')
        assert isinstance(suite.results, list)
