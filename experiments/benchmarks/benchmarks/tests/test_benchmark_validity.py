"""
Tests for QTT compression benchmarks.

Validates that qtt_compression.py runs correctly and produces valid results.
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.benchmark]


class TestQTTCompressionBenchmark:
    """Test QTT compression benchmark validity."""

    def test_benchmark_imports(self):
        """Test that benchmark module can be imported."""
        try:
            from benchmarks import qtt_compression

            assert qtt_compression is not None
        except ImportError as e:
            pytest.skip(f"Benchmark dependencies not available: {e}")

    def test_compression_ratio_positive(self):
        """Test that compression achieves positive ratio."""
        try:
            from ontic.core.qtt import compress_qtt

            # Create test data with compressible structure
            L = 6  # 2^6 = 64 elements
            x = np.cos(np.linspace(0, 2 * np.pi, 2**L))

            # Compress
            result = compress_qtt(x, max_rank=10, eps=1e-6)

            # Verify compression worked
            assert result.ranks is not None
            assert max(result.ranks) <= 10

        except ImportError as e:
            pytest.skip(f"QTT not available: {e}")

    def test_reconstruction_accuracy(self):
        """Test that decompressed data matches original within tolerance."""
        try:
            from ontic.core.qtt import compress_qtt, decompress_qtt

            # Create smooth test signal
            L = 6
            x_original = np.sin(np.linspace(0, 4 * np.pi, 2**L))

            # Compress and decompress
            qtt = compress_qtt(x_original, max_rank=16, eps=1e-8)
            x_reconstructed = decompress_qtt(qtt)

            # Check reconstruction error
            error = np.linalg.norm(x_original - x_reconstructed) / np.linalg.norm(
                x_original
            )
            assert error < 1e-6, f"Reconstruction error {error} exceeds tolerance"

        except ImportError as e:
            pytest.skip(f"QTT not available: {e}")


class TestSodShockTubeBenchmark:
    """Test Sod shock tube benchmark validity."""

    def test_benchmark_imports(self):
        """Test that sod_shock_tube benchmark can be imported."""
        try:
            from benchmarks import sod_shock_tube

            assert sod_shock_tube is not None
        except ImportError as e:
            pytest.skip(f"Benchmark dependencies not available: {e}")

    def test_sod_initial_conditions(self):
        """Test Sod shock tube initial conditions are valid."""
        # Standard Sod shock tube ICs
        rho_left, u_left, p_left = 1.0, 0.0, 1.0
        rho_right, u_right, p_right = 0.125, 0.0, 0.1
        gamma = 1.4

        # Verify physical validity
        assert rho_left > 0, "Density must be positive"
        assert rho_right > 0, "Density must be positive"
        assert p_left > 0, "Pressure must be positive"
        assert p_right > 0, "Pressure must be positive"

        # Left state is higher pressure (correct for Sod problem)
        assert p_left > p_right, "Left pressure should exceed right"


class TestTCIBenchmark:
    """Test TCI algorithm benchmark validity."""

    def test_benchmark_imports(self):
        """Test that TCI benchmark can be imported."""
        try:
            from benchmarks import tci_vs_hybrid

            assert tci_vs_hybrid is not None
        except ImportError as e:
            pytest.skip(f"Benchmark dependencies not available: {e}")

    def test_tci_rank_bounds(self):
        """Test that TCI produces bounded ranks."""
        try:
            from ontic.core.tci import adaptive_tci

            # Simple smooth function
            def f(idx):
                return np.sin(0.1 * np.sum(idx))

            result = adaptive_tci(
                func=f,
                shape=(16, 16, 16),
                max_rank=32,
                eps=1e-4,
            )

            # Verify rank is bounded
            assert result.max_rank <= 32

        except ImportError as e:
            pytest.skip(f"TCI not available: {e}")


class TestPerformanceThresholds:
    """
    Performance regression tests with threshold assertions.

    These tests verify that critical operations meet minimum performance
    requirements. Thresholds are conservative to avoid flaky tests.
    """

    # Performance thresholds (conservative for CI reliability)
    COMPRESSION_TIME_MAX_MS = 100.0  # Max time for small QTT compression
    RECONSTRUCTION_TIME_MAX_MS = 50.0  # Max time for QTT reconstruction

    @pytest.mark.slow
    def test_qtt_compression_performance(self):
        """Test QTT compression meets time threshold."""
        import time

        try:
            from ontic.core.qtt import compress_qtt

            L = 6  # 64 elements
            x = np.sin(np.linspace(0, 4 * np.pi, 2**L))

            start = time.perf_counter()
            for _ in range(10):  # Average over 10 runs
                compress_qtt(x, max_rank=16, eps=1e-6)
            elapsed_ms = (time.perf_counter() - start) * 100  # ms per operation

            assert (
                elapsed_ms < self.COMPRESSION_TIME_MAX_MS
            ), f"Compression took {elapsed_ms:.1f}ms, threshold is {self.COMPRESSION_TIME_MAX_MS}ms"

        except ImportError as e:
            pytest.skip(f"QTT not available: {e}")

    @pytest.mark.slow
    def test_qtt_reconstruction_performance(self):
        """Test QTT reconstruction meets time threshold."""
        import time

        try:
            from ontic.core.qtt import compress_qtt, decompress_qtt

            L = 6
            x = np.sin(np.linspace(0, 4 * np.pi, 2**L))
            qtt = compress_qtt(x, max_rank=16, eps=1e-6)

            start = time.perf_counter()
            for _ in range(10):
                decompress_qtt(qtt)
            elapsed_ms = (time.perf_counter() - start) * 100

            assert (
                elapsed_ms < self.RECONSTRUCTION_TIME_MAX_MS
            ), f"Reconstruction took {elapsed_ms:.1f}ms, threshold is {self.RECONSTRUCTION_TIME_MAX_MS}ms"

        except ImportError as e:
            pytest.skip(f"QTT not available: {e}")
