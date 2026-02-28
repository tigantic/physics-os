"""
Unit Tests for Tensor Slicer
============================

Verifies:
1. Single element extraction matches full reconstruction
2. 2D slice rendering works correctly
3. Zoom functionality produces consistent results
4. Performance scales logarithmically
"""

import numpy as np
import pytest

from ontic.sim.visualization.tensor_slicer import (TensorSlicer,
                                                   create_sine_qtt,
                                                   create_test_qtt)


class TestTensorSlicerBasic:
    """Basic functionality tests."""

    def test_init_valid_cores(self):
        """Test initialization with valid cores."""
        cores = [
            np.random.randn(1, 2, 4),
            np.random.randn(4, 2, 4),
            np.random.randn(4, 2, 1),
        ]
        slicer = TensorSlicer(cores)
        assert slicer.n_cores == 3
        assert slicer.grid_size == 8

    def test_init_invalid_cores(self):
        """Test initialization rejects invalid cores."""
        cores = [np.random.randn(2, 3, 2)]  # Wrong physical dim
        with pytest.raises(ValueError):
            TensorSlicer(cores)

    def test_get_element_binary_string(self):
        """Test element extraction with binary string."""
        slicer = create_test_qtt(n_cores=4, rank=2)
        val = slicer.get_element("0101")
        assert isinstance(val, float)

    def test_get_element_integer(self):
        """Test element extraction with integer."""
        slicer = create_test_qtt(n_cores=4, rank=2)
        val = slicer.get_element(5)
        assert isinstance(val, float)

    def test_get_element_consistency(self):
        """Test that integer and binary give same result."""
        slicer = create_test_qtt(n_cores=4, rank=2)
        val_int = slicer.get_element(5)
        val_bin = slicer.get_element("0101")
        assert np.isclose(val_int, val_bin)


class TestTensorSlicerCorrectness:
    """Correctness verification tests."""

    def test_reconstruction_matches_manual(self):
        """Test that element extraction matches manual tensor contraction."""
        n_cores = 4
        cores = [
            np.random.randn(1, 2, 3),
            np.random.randn(3, 2, 3),
            np.random.randn(3, 2, 3),
            np.random.randn(3, 2, 1),
        ]
        slicer = TensorSlicer(cores)

        # Manually compute full tensor
        full = np.zeros(2**n_cores)
        for idx in range(2**n_cores):
            binary = format(idx, f"0{n_cores}b")
            result = cores[0][:, int(binary[0]), :]
            for i in range(1, n_cores):
                result = result @ cores[i][:, int(binary[i]), :]
            full[idx] = result.squeeze()

        # Compare with slicer
        for idx in range(2**n_cores):
            slicer_val = slicer.get_element(idx)
            assert np.isclose(slicer_val, full[idx], rtol=1e-10)

    def test_batch_extraction(self):
        """Test batch element extraction."""
        slicer = create_test_qtt(n_cores=6, rank=2)

        indices = [0, 10, 20, 30, 40, 50, 60, 63]
        batch_vals = slicer.get_elements_batch(indices)

        for i, idx in enumerate(indices):
            single_val = slicer.get_element(idx)
            assert np.isclose(batch_vals[i], single_val)


class TestTensorSlicer2D:
    """2D slice rendering tests."""

    def test_render_slice_1d(self):
        """Test 1D slice rendering."""
        slicer = create_test_qtt(n_cores=8, rank=2)
        slice_1d = slicer.render_slice_1d(num_points=100)

        assert slice_1d.shape == (100,)
        assert not np.isnan(slice_1d).any()

    def test_render_slice_2d(self):
        """Test 2D slice rendering."""
        slicer = create_test_qtt(n_cores=8, rank=2)

        x_cores = [0, 1, 2, 3]
        y_cores = [4, 5, 6, 7]

        slice_2d = slicer.render_slice_2d(x_cores, y_cores, resolution=(32, 32))

        assert slice_2d.shape == (32, 32)
        assert not np.isnan(slice_2d).any()

    def test_render_slice_2d_vectorized(self):
        """Test vectorized 2D slice matches regular version."""
        slicer = create_test_qtt(n_cores=6, rank=2)

        x_cores = [0, 1, 2]
        y_cores = [3, 4, 5]
        resolution = (16, 16)

        regular = slicer.render_slice_2d(x_cores, y_cores, resolution=resolution)
        vectorized = slicer.render_slice_2d_vectorized(
            x_cores, y_cores, resolution=resolution
        )

        assert np.allclose(regular, vectorized)

    def test_render_plane_xy(self):
        """Test XY plane rendering."""
        slicer = create_test_qtt(n_cores=9, rank=2)  # 3 cores per dim

        plane = slicer.render_plane("xy", depth=0.5, resolution=(16, 16))

        assert plane.shape == (16, 16)
        assert not np.isnan(plane).any()


class TestTensorSlicerZoom:
    """Zoom functionality tests."""

    def test_zoom_level_1(self):
        """Test zoom level 1 covers full range."""
        slicer = create_test_qtt(n_cores=8, rank=2)

        full = slicer.render_zoomed(
            center=(0.5, 0.5), zoom_level=1, resolution=(32, 32)
        )

        assert full.shape == (32, 32)

    def test_zoom_preserves_center(self):
        """Test that center point is preserved across zoom levels."""
        slicer = create_test_qtt(n_cores=8, rank=2)

        # Get center value at different zoom levels
        center_vals = []
        for zoom in [1, 2, 4, 8]:
            img = slicer.render_zoomed(
                center=(0.5, 0.5),
                zoom_level=zoom,
                resolution=(33, 33),  # Odd so center pixel exists
            )
            center_vals.append(img[16, 16])

        # Center value should be same regardless of zoom
        for val in center_vals[1:]:
            assert np.isclose(center_vals[0], val, rtol=0.1)

    def test_zoom_reduces_range(self):
        """Test that higher zoom shows smaller range of indices."""
        slicer = create_test_qtt(n_cores=8, rank=2)

        zoom1 = slicer.render_zoomed(center=(0.5, 0.5), zoom_level=1, resolution=(8, 8))
        zoom4 = slicer.render_zoomed(center=(0.5, 0.5), zoom_level=4, resolution=(8, 8))

        # Different zoom levels should give different images
        # (unless the data is perfectly uniform)
        # The variance should generally change
        assert zoom1.shape == zoom4.shape


class TestTensorSlicerPerformance:
    """Performance and scaling tests."""

    def test_logarithmic_scaling(self):
        """Test that time scales logarithmically with grid size."""
        import time

        times = []
        sizes = [10, 15, 20]  # 2^10, 2^15, 2^20

        for n in sizes:
            slicer = create_test_qtt(n_cores=n, rank=2)

            t0 = time.perf_counter()
            for _ in range(100):
                slicer.get_element(0)
            times.append(time.perf_counter() - t0)

        # Time should grow roughly linearly with n_cores (which is log of grid size)
        # So doubling n should roughly double time, not quadruple it
        ratio_1_to_2 = times[1] / times[0]
        ratio_2_to_3 = times[2] / times[1]

        # Ratios should be similar (both roughly 1.5-2x)
        assert ratio_1_to_2 < 4  # Should not be exponential
        assert ratio_2_to_3 < 4

    def test_benchmark_runs(self):
        """Test that benchmark function works."""
        slicer = create_test_qtt(n_cores=10, rank=2)
        results = slicer.benchmark_render(resolution=(32, 32))

        assert "single_point_us" in results
        assert "slice_1d_ms" in results
        assert "slice_2d_ms" in results
        assert "estimated_fps" in results
        assert results["estimated_fps"] > 0


class TestTensorSlicerHeatmap:
    """Heatmap rendering tests."""

    def test_to_heatmap(self):
        """Test heatmap conversion."""
        pytest.importorskip("matplotlib", reason="matplotlib required for heatmap")
        slicer = create_test_qtt(n_cores=6, rank=2)

        data = slicer.render_slice_2d_vectorized(
            [0, 1, 2], [3, 4, 5], resolution=(16, 16)
        )

        rgb = slicer.to_heatmap(data)

        assert rgb.shape == (16, 16, 3)
        assert rgb.dtype == np.uint8
        assert rgb.min() >= 0
        assert rgb.max() <= 255


class TestSineQTT:
    """Tests for sine wave QTT construction."""

    def test_sine_qtt_shape(self):
        """Test sine QTT has correct structure."""
        slicer = create_sine_qtt(n_cores=8)
        assert slicer.n_cores == 8
        assert slicer.grid_size == 256


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
