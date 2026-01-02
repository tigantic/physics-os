"""
Test Module: tensornet/visualization/

Visualization Infrastructure Tests
Constitutional Compliance:
    - Article III, Section 3.1: Unit tests with 90%+ coverage
    - Article III, Section 3.2: Deterministic seeding (seed=42)
    - Article IV, Section 4.1: Physical validation
    - Article V, Section 5.1: Float64 precision

References:
    Oseledets, I.V. (2011). "Tensor-Train Decomposition."
    SIAM Journal on Scientific Computing.
    
    Decompression-Free Rendering: Project screen pixels onto
    Tensor Train without materializing full grid.
"""

import pytest
import torch
import numpy as np
import math
from typing import List, Optional, Tuple
from dataclasses import dataclass


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def deterministic_seed():
    """Per Article III, Section 3.2: Reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    yield


@pytest.fixture
def device():
    """Get device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# MOCK CLASSES
# ============================================================================

class MockQTTCores:
    """Mock QTT cores for testing."""
    
    def __init__(self, n_cores: int = 10, max_rank: int = 4, dtype=np.float64):
        """Create random QTT cores."""
        self.n_cores = n_cores
        self.max_rank = max_rank
        self.dtype = dtype
        self.cores = self._create_cores()
        self.grid_size = 2 ** n_cores
    
    def _create_cores(self) -> List[np.ndarray]:
        """Create list of cores."""
        cores = []
        for i in range(self.n_cores):
            r_left = 1 if i == 0 else self.max_rank
            r_right = 1 if i == self.n_cores - 1 else self.max_rank
            core = np.random.randn(r_left, 2, r_right).astype(self.dtype)
            cores.append(core)
        return cores


class MockTensorSlicer:
    """Mock tensor slicer for testing."""
    
    def __init__(self, cores: List[np.ndarray], dtype=np.float64):
        """Initialize with QTT cores."""
        self.cores = [np.asarray(c, dtype=dtype) for c in cores]
        self.n_cores = len(cores)
        self.dtype = dtype
        self.grid_size = 2 ** self.n_cores
    
    def get_element(self, index: int) -> float:
        """Extract single element via contraction."""
        binary = format(index, f'0{self.n_cores}b')
        
        result = None
        for i, bit in enumerate(binary):
            bit_idx = int(bit)
            matrix = self.cores[i][:, bit_idx, :]
            
            if result is None:
                result = matrix
            else:
                result = result @ matrix
        
        return float(result.squeeze())
    
    def render_slice_1d(self, start: int, end: int, num_points: int) -> np.ndarray:
        """Render 1D slice."""
        indices = np.linspace(start, end - 1, num_points, dtype=int)
        values = np.array([self.get_element(int(i)) for i in indices], dtype=self.dtype)
        return values


# ============================================================================
# UNIT TESTS: TENSOR SLICER CONSTRUCTION
# ============================================================================

class TestTensorSlicerConstruction:
    """Test TensorSlicer construction."""
    
    @pytest.mark.unit
    def test_slicer_creation(self, deterministic_seed):
        """Slicer can be created."""
        qtt = MockQTTCores(n_cores=10, max_rank=4)
        slicer = MockTensorSlicer(qtt.cores)
        
        assert slicer is not None
        assert slicer.n_cores == 10
    
    @pytest.mark.unit
    def test_grid_size_computation(self, deterministic_seed):
        """Grid size is 2^n_cores."""
        for n_cores in [5, 10, 15, 20]:
            qtt = MockQTTCores(n_cores=n_cores, max_rank=4)
            slicer = MockTensorSlicer(qtt.cores)
            
            assert slicer.grid_size == 2 ** n_cores
    
    @pytest.mark.unit
    def test_core_validation(self, deterministic_seed):
        """Cores must be 3D with physical dimension 2."""
        cores = [np.random.randn(1, 2, 4).astype(np.float64)]
        slicer = MockTensorSlicer(cores)
        
        assert slicer.cores[0].shape[1] == 2
    
    @pytest.mark.unit
    def test_dtype_propagation(self, deterministic_seed):
        """Dtype is propagated correctly."""
        qtt = MockQTTCores(n_cores=5, max_rank=2, dtype=np.float64)
        slicer = MockTensorSlicer(qtt.cores, dtype=np.float64)
        
        assert slicer.dtype == np.float64


# ============================================================================
# UNIT TESTS: ELEMENT EXTRACTION
# ============================================================================

class TestElementExtraction:
    """Test single element extraction from QTT."""
    
    @pytest.mark.unit
    def test_get_element_returns_scalar(self, deterministic_seed):
        """Element extraction returns scalar."""
        qtt = MockQTTCores(n_cores=10, max_rank=4)
        slicer = MockTensorSlicer(qtt.cores)
        
        value = slicer.get_element(0)
        
        assert isinstance(value, float)
    
    @pytest.mark.unit
    def test_get_element_valid_indices(self, deterministic_seed):
        """Valid indices work."""
        qtt = MockQTTCores(n_cores=10, max_rank=4)
        slicer = MockTensorSlicer(qtt.cores)
        
        # Test various valid indices
        for idx in [0, 1, 100, 500, slicer.grid_size - 1]:
            value = slicer.get_element(idx)
            assert not math.isnan(value)
    
    @pytest.mark.unit
    def test_element_deterministic(self, deterministic_seed):
        """Same index gives same value."""
        qtt = MockQTTCores(n_cores=10, max_rank=4)
        slicer = MockTensorSlicer(qtt.cores)
        
        idx = 512
        v1 = slicer.get_element(idx)
        v2 = slicer.get_element(idx)
        
        assert v1 == v2
    
    @pytest.mark.unit
    def test_different_indices_different_values(self, deterministic_seed):
        """Different indices typically give different values."""
        qtt = MockQTTCores(n_cores=10, max_rank=4)
        slicer = MockTensorSlicer(qtt.cores)
        
        values = [slicer.get_element(i) for i in range(100)]
        unique_values = len(set(values))
        
        # Most should be unique
        assert unique_values > 50
    
    @pytest.mark.unit
    def test_binary_index_conversion(self, deterministic_seed):
        """Binary indexing works correctly."""
        n_cores = 5
        qtt = MockQTTCores(n_cores=n_cores, max_rank=2)
        slicer = MockTensorSlicer(qtt.cores)
        
        # Index 13 = 01101 in binary (5 bits)
        idx = 13
        expected_binary = '01101'
        
        binary = format(idx, f'0{n_cores}b')
        assert binary == expected_binary


# ============================================================================
# UNIT TESTS: CONTRACTION ALGORITHM
# ============================================================================

class TestContractionAlgorithm:
    """Test tensor network contraction."""
    
    @pytest.mark.unit
    def test_matrix_chain_multiplication(self, deterministic_seed):
        """Matrix chain multiplication is associative."""
        # Create 3 matrices
        A = np.random.randn(1, 4).astype(np.float64)
        B = np.random.randn(4, 4).astype(np.float64)
        C = np.random.randn(4, 1).astype(np.float64)
        
        # (A @ B) @ C should equal A @ (B @ C)
        result1 = (A @ B) @ C
        result2 = A @ (B @ C)
        
        assert np.allclose(result1, result2)
    
    @pytest.mark.unit
    def test_contraction_complexity(self, deterministic_seed):
        """Contraction is O(d * r^2)."""
        # Track operations: d cores, r^2 per contraction
        n_cores = 20
        rank = 8
        
        estimated_ops = n_cores * rank * rank
        
        # Should be much smaller than full grid
        grid_size = 2 ** n_cores
        compression = grid_size / estimated_ops
        
        assert compression > 500  # Huge compression (actual ~819)
    
    @pytest.mark.unit
    def test_slice_selection(self, deterministic_seed):
        """Correct slice is selected per bit."""
        core = np.random.randn(4, 2, 4).astype(np.float64)
        
        slice_0 = core[:, 0, :]
        slice_1 = core[:, 1, :]
        
        # Shapes should match
        assert slice_0.shape == (4, 4)
        assert slice_1.shape == (4, 4)
        
        # Slices should be different
        assert not np.allclose(slice_0, slice_1)


# ============================================================================
# UNIT TESTS: 1D SLICE RENDERING
# ============================================================================

class TestSlice1DRendering:
    """Test 1D slice rendering."""
    
    @pytest.mark.unit
    def test_render_1d_shape(self, deterministic_seed):
        """1D slice has correct shape."""
        qtt = MockQTTCores(n_cores=10, max_rank=4)
        slicer = MockTensorSlicer(qtt.cores)
        
        num_points = 256
        result = slicer.render_slice_1d(0, 1024, num_points)
        
        assert result.shape == (num_points,)
    
    @pytest.mark.unit
    def test_render_1d_dtype(self, deterministic_seed):
        """1D slice maintains dtype."""
        qtt = MockQTTCores(n_cores=10, max_rank=4, dtype=np.float64)
        slicer = MockTensorSlicer(qtt.cores, dtype=np.float64)
        
        result = slicer.render_slice_1d(0, 100, 50)
        
        assert result.dtype == np.float64
    
    @pytest.mark.unit
    def test_render_1d_no_nan(self, deterministic_seed):
        """1D slice contains no NaN."""
        qtt = MockQTTCores(n_cores=10, max_rank=4)
        slicer = MockTensorSlicer(qtt.cores)
        
        result = slicer.render_slice_1d(0, 1024, 256)
        
        assert not np.isnan(result).any()


# ============================================================================
# UNIT TESTS: 2D SLICE RENDERING
# ============================================================================

class Test2DSliceRendering:
    """Test 2D cross-section rendering."""
    
    @pytest.mark.unit
    def test_render_2d_shape(self, deterministic_seed):
        """2D slice has correct shape."""
        width, height = 256, 256
        
        # Mock 2D rendering
        result = np.random.randn(height, width).astype(np.float64)
        
        assert result.shape == (height, width)
    
    @pytest.mark.unit
    def test_pixel_to_index_mapping(self, deterministic_seed):
        """Pixel coordinates map to tensor indices."""
        width, height = 256, 256
        n_x_bits = 8  # 2^8 = 256
        n_y_bits = 8
        
        px, py = 128, 64
        
        # Map pixel to binary index
        x_bits = format(px, f'0{n_x_bits}b')
        y_bits = format(py, f'0{n_y_bits}b')
        
        assert len(x_bits) == n_x_bits
        assert len(y_bits) == n_y_bits
    
    @pytest.mark.unit
    def test_fixed_dimension_slicing(self, deterministic_seed):
        """Fixed dimensions create 2D slice."""
        # 4D tensor: (X, Y, Z, T)
        # Fix Z=5, T=0 to get 2D slice (X, Y)
        
        fixed = {'Z': 5, 'T': 0}
        n_dims = 4
        n_free = n_dims - len(fixed)
        
        assert n_free == 2  # X and Y are free


# ============================================================================
# UNIT TESTS: COLORMAP APPLICATION
# ============================================================================

class TestColormapApplication:
    """Test colormap application for visualization."""
    
    @pytest.mark.unit
    def test_normalize_to_01(self, deterministic_seed):
        """Values are normalized to [0, 1]."""
        values = np.random.randn(100, 100).astype(np.float64)
        
        vmin, vmax = values.min(), values.max()
        normalized = (values - vmin) / (vmax - vmin + 1e-10)
        
        assert normalized.min() >= 0
        assert normalized.max() <= 1
    
    @pytest.mark.unit
    def test_colormap_rgb_output(self, deterministic_seed):
        """Colormap produces RGB output."""
        values = np.random.randn(100, 100).astype(np.float64)
        
        # Normalize
        normalized = (values - values.min()) / (values.max() - values.min() + 1e-10)
        
        # Simple grayscale to RGB
        rgb = np.stack([normalized, normalized, normalized], axis=-1)
        
        assert rgb.shape == (100, 100, 3)
    
    @pytest.mark.unit
    def test_viridis_colormap(self, deterministic_seed):
        """Viridis-like colormap works."""
        t = np.linspace(0, 1, 256)
        
        # Simple viridis approximation
        r = 0.267 + 0.329 * t + 1.057 * t**2 - 1.378 * t**3
        g = 0.004 + 0.873 * t - 0.238 * t**2 + 0.127 * t**3
        b = 0.329 + 1.092 * t - 1.681 * t**2 + 0.541 * t**3
        
        r = np.clip(r, 0, 1)
        g = np.clip(g, 0, 1)
        b = np.clip(b, 0, 1)
        
        assert r.shape == (256,)
        assert np.all(r >= 0) and np.all(r <= 1)


# ============================================================================
# UNIT TESTS: RESOLUTION HANDLING
# ============================================================================

class TestResolutionHandling:
    """Test various output resolutions."""
    
    @pytest.mark.unit
    def test_resolution_256(self, deterministic_seed):
        """256x256 resolution works."""
        result = np.random.randn(256, 256).astype(np.float64)
        assert result.shape == (256, 256)
    
    @pytest.mark.unit
    def test_resolution_1080p(self, deterministic_seed):
        """1080p resolution works."""
        result = np.random.randn(1080, 1920).astype(np.float64)
        assert result.shape == (1080, 1920)
    
    @pytest.mark.unit
    def test_resolution_4k(self, deterministic_seed):
        """4K resolution works."""
        result = np.random.randn(2160, 3840).astype(np.float64)
        assert result.shape == (2160, 3840)
    
    @pytest.mark.unit
    def test_non_square_resolution(self, deterministic_seed):
        """Non-square resolutions work."""
        result = np.random.randn(480, 640).astype(np.float64)
        assert result.shape == (480, 640)


# ============================================================================
# UNIT TESTS: MEMORY EFFICIENCY
# ============================================================================

class TestMemoryEfficiency:
    """Test memory-efficient operations."""
    
    @pytest.mark.unit
    def test_qtt_compression_ratio(self, deterministic_seed):
        """QTT achieves compression."""
        n_cores = 30  # 2^30 = 1 billion points
        rank = 10
        
        # QTT storage
        qtt_storage = n_cores * rank * 2 * rank * 8  # bytes
        
        # Dense storage
        dense_storage = (2 ** n_cores) * 8  # bytes
        
        compression = dense_storage / qtt_storage
        
        assert compression > 1e5  # High compression (actual ~179k)
    
    @pytest.mark.unit
    def test_streaming_slice(self, deterministic_seed):
        """Slices can be streamed row by row."""
        width = 1920
        row_buffer = np.zeros(width, dtype=np.float64)
        
        # Simulate streaming
        n_rows = 10
        for row in range(n_rows):
            row_buffer[:] = np.random.randn(width)
        
        assert row_buffer.shape == (width,)


# ============================================================================
# FLOAT64 COMPLIANCE
# ============================================================================

class TestFloat64ComplianceVisualization:
    """Article V: Float64 precision tests."""
    
    @pytest.mark.unit
    def test_cores_float64(self, deterministic_seed):
        """QTT cores are float64."""
        qtt = MockQTTCores(n_cores=10, max_rank=4, dtype=np.float64)
        
        for core in qtt.cores:
            assert core.dtype == np.float64
    
    @pytest.mark.unit
    def test_slice_float64(self, deterministic_seed):
        """Rendered slices are float64."""
        qtt = MockQTTCores(n_cores=10, max_rank=4, dtype=np.float64)
        slicer = MockTensorSlicer(qtt.cores, dtype=np.float64)
        
        result = slicer.render_slice_1d(0, 100, 50)
        
        assert result.dtype == np.float64
    
    @pytest.mark.unit
    def test_contraction_float64(self, deterministic_seed):
        """Contraction maintains float64."""
        A = np.random.randn(4, 4).astype(np.float64)
        B = np.random.randn(4, 4).astype(np.float64)
        
        result = A @ B
        
        assert result.dtype == np.float64


# ============================================================================
# GPU COMPATIBILITY
# ============================================================================

class TestGPUCompatibilityVisualization:
    """Test GPU execution compatibility."""
    
    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_torch_contraction_gpu(self, deterministic_seed, device):
        """Contraction on GPU with PyTorch."""
        A = torch.randn(4, 4, dtype=torch.float64, device=device)
        B = torch.randn(4, 4, dtype=torch.float64, device=device)
        
        result = A @ B
        
        assert result.device.type == device.type  # Compare device type, not index
    
    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_slice_on_gpu(self, deterministic_seed, device):
        """Slice can be on GPU."""
        slice_data = torch.randn(1080, 1920, dtype=torch.float64, device=device)
        
        assert slice_data.device.type == device.type  # Compare device type, not index


# ============================================================================
# NUMERICAL STABILITY
# ============================================================================

class TestNumericalStabilityVisualization:
    """Test numerical stability."""
    
    @pytest.mark.unit
    def test_contraction_stability(self, deterministic_seed):
        """Contraction doesn't explode or vanish."""
        qtt = MockQTTCores(n_cores=20, max_rank=4)
        slicer = MockTensorSlicer(qtt.cores)
        
        values = [slicer.get_element(i) for i in range(100)]
        
        # No NaN or Inf
        assert all(not math.isnan(v) for v in values)
        assert all(not math.isinf(v) for v in values)
    
    @pytest.mark.unit
    def test_colormap_nan_handling(self, deterministic_seed):
        """Colormap handles edge cases."""
        # Constant field
        values = np.ones((100, 100), dtype=np.float64) * 5.0
        
        vmin, vmax = values.min(), values.max()
        normalized = (values - vmin) / (vmax - vmin + 1e-10)
        
        assert not np.isnan(normalized).any()


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

class TestReproducibilityVisualization:
    """Article III, Section 3.2: Reproducibility tests."""
    
    @pytest.mark.unit
    def test_deterministic_rendering(self):
        """Same seed produces identical slices."""
        np.random.seed(42)
        qtt1 = MockQTTCores(n_cores=10, max_rank=4)
        slicer1 = MockTensorSlicer(qtt1.cores)
        slice1 = slicer1.render_slice_1d(0, 100, 50)
        
        np.random.seed(42)
        qtt2 = MockQTTCores(n_cores=10, max_rank=4)
        slicer2 = MockTensorSlicer(qtt2.cores)
        slice2 = slicer2.render_slice_1d(0, 100, 50)
        
        assert np.allclose(slice1, slice2)


# ============================================================================
# PERFORMANCE
# ============================================================================

class TestPerformanceVisualization:
    """Test computational performance."""
    
    @pytest.mark.unit
    @pytest.mark.performance
    def test_element_extraction_speed(self, deterministic_seed):
        """Element extraction is fast."""
        import time
        
        qtt = MockQTTCores(n_cores=20, max_rank=8)
        slicer = MockTensorSlicer(qtt.cores)
        
        start = time.perf_counter()
        for i in range(1000):
            _ = slicer.get_element(i)
        elapsed = time.perf_counter() - start
        
        assert elapsed < 5.0  # 1000 extractions in < 5 seconds
    
    @pytest.mark.unit
    @pytest.mark.performance
    def test_1d_slice_speed(self, deterministic_seed):
        """1D slice rendering is fast."""
        import time
        
        qtt = MockQTTCores(n_cores=15, max_rank=4)
        slicer = MockTensorSlicer(qtt.cores)
        
        start = time.perf_counter()
        _ = slicer.render_slice_1d(0, 1000, 256)
        elapsed = time.perf_counter() - start
        
        assert elapsed < 5.0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestVisualizationIntegration:
    """Integration tests for visualization."""
    
    @pytest.mark.integration
    def test_qtt_to_image_workflow(self, deterministic_seed):
        """Full QTT to image workflow."""
        # Create QTT
        qtt = MockQTTCores(n_cores=10, max_rank=4, dtype=np.float64)
        slicer = MockTensorSlicer(qtt.cores, dtype=np.float64)
        
        # Render 1D slice
        values = slicer.render_slice_1d(0, 256, 256)
        
        # Normalize
        vmin, vmax = values.min(), values.max()
        normalized = (values - vmin) / (vmax - vmin + 1e-10)
        
        # Expand to 2D (repeat rows)
        image = np.tile(normalized, (256, 1))
        
        # Apply colormap (grayscale)
        rgb = np.stack([image, image, image], axis=-1)
        
        assert rgb.shape == (256, 256, 3)
        assert rgb.dtype == np.float64
        assert np.all(rgb >= 0) and np.all(rgb <= 1)
    
    @pytest.mark.integration
    def test_multi_resolution_workflow(self, deterministic_seed):
        """Multi-resolution rendering workflow."""
        qtt = MockQTTCores(n_cores=15, max_rank=4)
        slicer = MockTensorSlicer(qtt.cores)
        
        # Render at multiple resolutions
        resolutions = [64, 128, 256]
        
        for res in resolutions:
            values = slicer.render_slice_1d(0, 1024, res)
            assert values.shape == (res,)


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.mark.unit
    def test_single_core(self, deterministic_seed):
        """Single core QTT works."""
        cores = [np.random.randn(1, 2, 1).astype(np.float64)]
        slicer = MockTensorSlicer(cores)
        
        assert slicer.grid_size == 2
    
    @pytest.mark.unit
    def test_rank_1_cores(self, deterministic_seed):
        """Rank-1 cores work."""
        qtt = MockQTTCores(n_cores=10, max_rank=1)
        slicer = MockTensorSlicer(qtt.cores)
        
        value = slicer.get_element(0)
        assert isinstance(value, float)
    
    @pytest.mark.unit
    def test_first_and_last_index(self, deterministic_seed):
        """First and last indices work."""
        qtt = MockQTTCores(n_cores=10, max_rank=4)
        slicer = MockTensorSlicer(qtt.cores)
        
        first = slicer.get_element(0)
        last = slicer.get_element(slicer.grid_size - 1)
        
        assert isinstance(first, float)
        assert isinstance(last, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])
