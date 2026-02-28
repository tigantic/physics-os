"""
TCI-LLM Unit Tests.

Run with: pytest tests/ (if pytest available)
Or: python tests/test_tci_llm.py (standalone)
"""

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    
import numpy as np
import torch
import sys
from pathlib import Path

# Add parent to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tci_llm import TCI_LLM, qtt_from_function_dense, qtt_eval_batch, qtt_eval_at_index


class TestQTT:
    """Test QTT construction and evaluation."""
    
    def test_dense_to_qtt_identity(self):
        """Test that QTT can represent identity function."""
        n_qubits = 6
        N = 2**n_qubits
        
        def identity(indices):
            return indices.float()
        
        cores = qtt_from_function_dense(identity, n_qubits=n_qubits, max_rank=64)
        
        # Evaluate at all points
        indices = torch.arange(N)
        result = qtt_eval_batch(cores, indices)
        expected = identity(indices)
        
        # Should be exact for identity
        assert torch.allclose(result, expected, atol=1e-4)
    
    def test_dense_to_qtt_constant(self):
        """Test QTT for constant function."""
        n_qubits = 5
        N = 2**n_qubits
        
        def constant(indices):
            return torch.ones_like(indices, dtype=torch.float32) * 42.0
        
        cores = qtt_from_function_dense(constant, n_qubits=n_qubits, max_rank=8)
        
        indices = torch.arange(N)
        result = qtt_eval_batch(cores, indices)
        
        assert torch.allclose(result, torch.full((N,), 42.0), atol=1e-4)
    
    def test_qtt_eval_single_vs_batch(self):
        """Test that single and batch evaluation agree."""
        n_qubits = 5
        
        def func(indices):
            return (indices % 7).float()
        
        cores = qtt_from_function_dense(func, n_qubits=n_qubits, max_rank=32)
        
        # Compare single vs batch
        for idx in [0, 1, 15, 31]:
            single_result = qtt_eval_at_index(cores, idx)
            batch_result = qtt_eval_batch(cores, torch.tensor([idx]))[0]
            assert torch.allclose(single_result, batch_result, atol=1e-4)


class TestTCI_LLM:
    """Test main TCI_LLM class."""
    
    def test_from_text_basic(self):
        """Test basic model creation from text."""
        text = "Hello world! Hello universe! Hello everyone!"
        model = TCI_LLM.from_text(text, context_length=4)
        
        assert model.n_contexts > 0
        assert model.params > 0
        assert model.build_time > 0
    
    def test_predict_next(self):
        """Test next-byte prediction."""
        text = "aaaa" * 100  # Simple pattern
        model = TCI_LLM.from_text(text, context_length=4)
        
        # Should predict 'a' after 'aaaa'
        predicted = model.predict_next(b"aaaa")
        assert predicted == ord('a')
    
    def test_generate(self):
        """Test text generation."""
        text = "Hello Hello Hello Hello Hello"
        model = TCI_LLM.from_text(text, context_length=4)
        
        output = model.generate(b"Hell", n_tokens=10)
        
        # Should be bytes
        assert isinstance(output, bytes)
        # Should have seed + generated
        assert len(output) == 4 + 10
    
    def test_benchmark(self):
        """Test benchmarking function."""
        text = "Test text for benchmarking purposes."
        model = TCI_LLM.from_text(text, context_length=4)
        
        throughput = model.benchmark(n_iterations=10, tokens_per_iter=10)
        
        # Should be positive
        assert throughput > 0
    
    def test_repr(self):
        """Test string representation."""
        text = "Simple test text."
        model = TCI_LLM.from_text(text, context_length=4)
        
        repr_str = repr(model)
        assert "TCI_LLM" in repr_str
        assert "contexts" in repr_str
    
    def test_empty_context_handling(self):
        """Test handling of unknown contexts."""
        text = "abcdefgh"
        model = TCI_LLM.from_text(text, context_length=4)
        
        # Unknown context should return default
        predicted = model.predict_next(b"zzzz")
        assert predicted == 32  # Space
    
    def test_seed_reproducibility(self):
        """Test that same seed gives same results."""
        text = "Reproducibility test text."
        
        model1 = TCI_LLM.from_text(text, context_length=4, seed=42)
        model2 = TCI_LLM.from_text(text, context_length=4, seed=42)
        
        # Should have same parameters
        assert model1.params == model2.params
        
        # Should generate same output
        out1 = model1.generate(b"Repr", n_tokens=20)
        out2 = model2.generate(b"Repr", n_tokens=20)
        assert out1 == out2


class TestIntegration:
    """Integration tests with realistic data."""
    
    def test_constitution_snippet(self):
        """Test with a snippet similar to CONSTITUTION.md."""
        text = """
# ONTIC_ENGINE CONSTITUTION

## Article I: Scientific Integrity

All claims must be:
1. Reproducible with fixed seeds
2. Quantitatively measured
3. Documented with methodology
        """
        
        model = TCI_LLM.from_text(text, context_length=4, max_rank=128)
        
        # Should build successfully
        assert model.n_contexts > 10
        
        # Should generate coherent-ish output
        output = model.generate(b"All ", n_tokens=20)
        assert len(output) == 24
    
    def test_large_rank(self):
        """Test with large max_rank."""
        text = "a" * 100 + "b" * 100 + "c" * 100
        model = TCI_LLM.from_text(text, context_length=4, max_rank=256)
        
        assert model.params > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
