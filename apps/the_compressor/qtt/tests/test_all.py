"""
QTT Test Suite
==============

Comprehensive tests for QTT library.

Run with:
    pytest qtt/tests/ -v
    
Or for quick smoke test:
    python -m qtt.tests.test_all
"""

import numpy as np
import tempfile
from pathlib import Path
import pytest


class TestSpatialCompression:
    """Tests for TT-SVD spatial compression."""
    
    def test_tt_svd_basic(self):
        """Test basic TT-SVD decomposition."""
        from qtt.spatial import tt_svd, tt_reconstruct
        
        # Create smooth test data
        x = np.linspace(0, 1, 32)
        y = np.linspace(0, 1, 32)
        z = np.linspace(0, 1, 32)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        data = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) * Z
        data = data.astype(np.float32)
        
        # Compress
        cores = tt_svd(data, max_rank=16)
        
        # Verify cores structure
        assert len(cores) == 3
        assert cores[0].shape[0] == 1  # First rank is 1
        assert cores[-1].shape[2] == 1  # Last rank is 1
        
        # Reconstruct
        recon = tt_reconstruct(cores)
        
        # Check error
        rel_error = np.linalg.norm(recon - data) / np.linalg.norm(data)
        assert rel_error < 0.01  # Less than 1% error
    
    def test_tt_element_access(self):
        """Test random element access."""
        from qtt.spatial import tt_svd, tt_reconstruct_element
        
        np.random.seed(42)
        data = np.random.randn(16, 16, 16).astype(np.float32)
        
        cores = tt_svd(data, max_rank=64)
        
        # Test several random elements
        for _ in range(10):
            i, j, k = np.random.randint(16, size=3)
            expected = data[i, j, k]
            actual = tt_reconstruct_element(cores, (i, j, k))
            assert abs(actual - expected) < 0.01
    
    def test_spatial_compressor(self):
        """Test SpatialCompressor class."""
        from qtt.spatial import SpatialCompressor
        
        data = np.random.randn(20, 20, 20).astype(np.float32)
        
        compressor = SpatialCompressor(max_rank=32)
        compressor.compress(data)
        
        # Check stats
        assert compressor.stats.compression_ratio > 1
        assert compressor.stats.relative_error < 1.0
        
        # Test element access
        value = compressor.reconstruct_element((10, 10, 10))
        expected = data[10, 10, 10]
        assert abs(value - expected) < 0.1


class TestSemanticCompression:
    """Tests for Product Quantization semantic compression."""
    
    def test_product_quantizer(self):
        """Test PQ training and encoding."""
        from qtt.semantic import ProductQuantizer
        
        # Random embeddings
        np.random.seed(42)
        embeddings = np.random.randn(1000, 384).astype(np.float32)
        
        pq = ProductQuantizer(n_subvectors=12, n_centroids=256)
        pq.train(embeddings)
        
        # Encode
        codes = pq.encode(embeddings)
        assert codes.shape == (1000, 12)
        assert codes.dtype == np.uint8
        
        # Compute distances
        query = embeddings[0]
        distances = pq.compute_distances(query, codes)
        
        # Query should be closest to itself
        assert np.argmin(distances) == 0
    
    def test_semantic_index(self):
        """Test SemanticIndex search."""
        from qtt.semantic import SemanticIndex
        
        texts = [
            "Quantum mechanics describes particle behavior",
            "Machine learning is a type of AI",
            "The Roman Empire was ancient",
            "DNA contains genetic information",
            "Climate change affects weather patterns"
        ]
        
        index = SemanticIndex.from_texts(texts, show_progress=False)
        
        # Search
        result = index.search("quantum physics", top_k=3)
        
        # First result should be about quantum
        assert "quantum" in texts[result.matches[0].document_id].lower()


class TestContainer:
    """Tests for QTT container format."""
    
    def test_semantic_container(self):
        """Test creating and reading semantic container."""
        from qtt import QTTContainer
        
        texts = [
            "Physics is the study of matter and energy",
            "Chemistry explores molecular interactions",
            "Biology studies living organisms",
        ]
        
        # Create
        container = QTTContainer.from_text_corpus(
            texts, 
            n_subvectors=12,
            n_centroids=256,
            show_progress=False
        )
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.qtt', delete=False) as f:
            path = f.name
        
        try:
            container.save(path)
            
            # Verify file exists and has content
            assert Path(path).stat().st_size > 0
            
            # Open and query
            with QTTContainer.open(path) as reader:
                assert reader.is_semantic
                
                result = reader.slice(query="physics", top_k=2)
                assert len(result.matches) == 2
                
                # Read text
                text = reader.read_text(result.matches[0])
                assert len(text) > 0
        finally:
            Path(path).unlink()
    
    def test_spatial_container(self):
        """Test creating and reading spatial container."""
        from qtt import QTTContainer
        
        # Create test data
        np.random.seed(42)
        x = np.linspace(0, 1, 32)
        y = np.linspace(0, 1, 32)
        X, Y = np.meshgrid(x, y, indexing='ij')
        data = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        data = data.astype(np.float32)
        
        # Create container
        container = QTTContainer.from_spatial_data(data, max_rank=16, show_progress=False)
        
        # Save
        with tempfile.NamedTemporaryFile(suffix='.qtt', delete=False) as f:
            path = f.name
        
        try:
            container.save(path)
            
            # Open and slice
            with QTTContainer.open(path) as reader:
                assert reader.is_spatial
                
                # Test center point
                value = reader.slice(coords=(16, 16))
                expected = data[16, 16]
                assert abs(value - expected) < 0.01
        finally:
            Path(path).unlink()
    
    def test_header_packing(self):
        """Test header serialization."""
        from qtt.container import QTTHeader, QTT_MAGIC
        
        header = QTTHeader(
            metadata_offset=64,
            metadata_size=100,
            index_offset=164,
            index_size=1000,
            payload_offset=1164,
            payload_size=5000,
            footer_offset=6164
        )
        
        packed = header.pack()
        assert len(packed) == QTTHeader.SIZE
        
        unpacked = QTTHeader.unpack(packed)
        assert unpacked.magic == QTT_MAGIC
        assert unpacked.metadata_offset == 64
        assert unpacked.index_size == 1000


class TestSlicer:
    """Tests for QTTSlicer high-level API."""
    
    def test_spatial_slicer(self):
        """Test slicer with spatial data."""
        from qtt.slicer import QTTSlicer
        
        data = np.random.randn(16, 16, 16).astype(np.float32)
        
        slicer = QTTSlicer.from_spatial(data, max_rank=32)
        
        assert slicer.mode == 'spatial'
        
        value = slicer.slice(coords=(8, 8, 8))
        expected = data[8, 8, 8]
        assert abs(value - expected) < 0.1
    
    def test_semantic_slicer(self):
        """Test slicer with text corpus."""
        from qtt.slicer import QTTSlicer
        
        texts = [
            "Artificial intelligence and machine learning",
            "Quantum computing research",
            "Deep neural networks"
        ]
        
        slicer = QTTSlicer.from_corpus(texts, show_progress=False)
        
        assert slicer.mode == 'semantic'
        
        matches = slicer.slice(query="AI", top_k=2)
        assert len(matches) == 2
        assert matches[0].text is not None


def run_smoke_test():
    """Quick smoke test."""
    print("Running QTT smoke tests...")
    
    # Test imports
    from qtt import QTTContainer, QTTSlicer
    from qtt.spatial import tt_svd, SpatialCompressor
    from qtt.semantic import ProductQuantizer, SemanticIndex
    print("✓ All imports successful")
    
    # Test spatial
    data = np.random.randn(16, 16, 16).astype(np.float32)
    cores = tt_svd(data, max_rank=8)
    print(f"✓ TT-SVD: {len(cores)} cores")
    
    # Test semantic (minimal)
    pq = ProductQuantizer(n_subvectors=4, n_centroids=16)
    embeddings = np.random.randn(100, 32).astype(np.float32)
    pq.train(embeddings)
    codes = pq.encode(embeddings)
    print(f"✓ PQ: encoded {len(codes)} vectors")
    
    # Test container
    container = QTTContainer.from_spatial_data(data, max_rank=8, show_progress=False)
    print(f"✓ Container: spatial mode")
    
    print("\nAll smoke tests passed!")


if __name__ == '__main__':
    run_smoke_test()
