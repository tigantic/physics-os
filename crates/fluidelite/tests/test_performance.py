"""
Performance Regression Tests for FluidElite
============================================

Tests that enforce performance baselines per Article II.4:
"Any regression beyond 10% from baseline shall block merge until resolved."

These tests measure:
1. Throughput (tok/s) - Must maintain >= 200 tok/s
2. Latency (ms/token) - Must maintain <= 5.0 ms/token
3. Memory usage - Must stay bounded regardless of sequence length
4. Learning capability - Must achieve >= 70% on digit prediction

Run with: pytest tests/test_performance.py -v

Constitutional Compliance:
    - Article II.4: Performance regression tests
    - Article VI.3: Representative workloads
    - Article VII.4: Demonstration requirement
"""

import gc
import time
import warnings
from dataclasses import dataclass
from typing import Optional

import pytest
import torch

# Skip all tests if CUDA not available (performance tests require GPU)
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Performance tests require CUDA"
)


# =============================================================================
# Performance Baselines (Article II.4)
# =============================================================================

@dataclass
class PerformanceBaseline:
    """Performance baselines that must be maintained."""
    
    # Throughput: tokens per second
    min_throughput: float = 200.0  # tok/s (Phase 3 target)
    
    # Latency: milliseconds per token
    max_latency: float = 5.0  # ms/token
    
    # Memory: bytes per sequence (bounded memory guarantee)
    max_memory_per_seq: float = 50 * 1024 * 1024  # 50 MB total
    
    # Learning: minimum accuracy on digit prediction task
    min_accuracy: float = 0.70  # 70%
    
    # Regression tolerance (Article II.4)
    tolerance: float = 0.10  # 10%
    
    def throughput_threshold(self) -> float:
        """Minimum acceptable throughput with tolerance."""
        return self.min_throughput * (1 - self.tolerance)
    
    def latency_threshold(self) -> float:
        """Maximum acceptable latency with tolerance."""
        return self.max_latency * (1 + self.tolerance)
    
    def memory_threshold(self) -> float:
        """Maximum acceptable memory with tolerance."""
        return self.max_memory_per_seq * (1 + self.tolerance)
    
    def accuracy_threshold(self) -> float:
        """Minimum acceptable accuracy with tolerance."""
        return self.min_accuracy * (1 - self.tolerance)


BASELINE = PerformanceBaseline()


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def model():
    """Create FluidElite model for testing."""
    from fluidelite.llm.fluid_elite import FluidElite
    
    model = FluidElite(
        num_sites=16,
        rank=128,
        mpo_rank=1,  # Optimized setting from Phase 3
        vocab_size=50000,
        truncate_every=20,  # Optimized setting from Phase 3
        dtype=torch.float32
    )
    model.cuda()
    model.eval()
    
    # Warmup
    with torch.no_grad():
        ctx = model.embed(42)
        for _ in range(10):
            ctx = model.step(ctx, 1)
    
    torch.cuda.synchronize()
    
    yield model
    
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture
def memory_baseline():
    """Get baseline memory before test."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda.memory_allocated()


# =============================================================================
# Throughput Tests
# =============================================================================

class TestThroughput:
    """Throughput regression tests."""
    
    def test_single_token_throughput(self, model):
        """Test throughput for single token processing."""
        num_tokens = 100
        
        with torch.no_grad():
            ctx = model.embed(0)
            
            # Warmup
            for i in range(10):
                ctx = model.step(ctx, i % 10)
            torch.cuda.synchronize()
            
            # Measure
            start = time.perf_counter()
            for i in range(num_tokens):
                ctx = model.step(ctx, i % 10)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        
        throughput = num_tokens / elapsed
        threshold = BASELINE.throughput_threshold()
        
        print(f"\nThroughput: {throughput:.1f} tok/s (threshold: {threshold:.1f})")
        
        assert throughput >= threshold, (
            f"Throughput regression: {throughput:.1f} tok/s < {threshold:.1f} tok/s "
            f"(baseline: {BASELINE.min_throughput:.1f}, tolerance: {BASELINE.tolerance*100:.0f}%)"
        )
    
    def test_sustained_throughput(self, model):
        """Test sustained throughput over 1000 tokens."""
        num_tokens = 1000
        
        with torch.no_grad():
            ctx = model.embed(0)
            
            # Warmup
            for i in range(50):
                ctx = model.step(ctx, i % 10)
            torch.cuda.synchronize()
            
            # Measure sustained throughput
            start = time.perf_counter()
            for i in range(num_tokens):
                ctx = model.step(ctx, i % 10)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        
        throughput = num_tokens / elapsed
        threshold = BASELINE.throughput_threshold()
        
        print(f"\nSustained throughput: {throughput:.1f} tok/s (threshold: {threshold:.1f})")
        
        assert throughput >= threshold, (
            f"Sustained throughput regression: {throughput:.1f} tok/s < {threshold:.1f} tok/s"
        )


# =============================================================================
# Latency Tests
# =============================================================================

class TestLatency:
    """Latency regression tests."""
    
    def test_per_token_latency(self, model):
        """Test per-token latency."""
        num_tokens = 100
        latencies = []
        
        with torch.no_grad():
            ctx = model.embed(0)
            
            # Warmup
            for i in range(10):
                ctx = model.step(ctx, i % 10)
            torch.cuda.synchronize()
            
            # Measure each token
            for i in range(num_tokens):
                torch.cuda.synchronize()
                start = time.perf_counter()
                ctx = model.step(ctx, i % 10)
                torch.cuda.synchronize()
                latencies.append((time.perf_counter() - start) * 1000)  # ms
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
        threshold = BASELINE.latency_threshold()
        
        print(f"\nLatency - Avg: {avg_latency:.2f}ms, Max: {max_latency:.2f}ms, P99: {p99_latency:.2f}ms")
        print(f"Threshold: {threshold:.2f}ms")
        
        assert avg_latency <= threshold, (
            f"Latency regression: {avg_latency:.2f}ms > {threshold:.2f}ms "
            f"(baseline: {BASELINE.max_latency:.2f}ms)"
        )
    
    def test_jitter(self, model):
        """Test latency jitter (standard deviation).
        
        Note: High jitter is EXPECTED with lazy truncation (truncate_every=20).
        Latency spikes every 20 tokens are by design - they amortize SVD cost.
        We test that P95 latency is acceptable rather than standard deviation.
        """
        num_tokens = 100
        latencies = []
        
        with torch.no_grad():
            ctx = model.embed(0)
            
            # Warmup
            for i in range(50):
                ctx = model.step(ctx, i % 10)
            torch.cuda.synchronize()
            
            # Measure
            for i in range(num_tokens):
                torch.cuda.synchronize()
                start = time.perf_counter()
                ctx = model.step(ctx, i % 10)
                torch.cuda.synchronize()
                latencies.append((time.perf_counter() - start) * 1000)
        
        mean = sum(latencies) / len(latencies)
        variance = sum((x - mean) ** 2 for x in latencies) / len(latencies)
        std_dev = variance ** 0.5
        cv = std_dev / mean  # Coefficient of variation
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        
        print(f"\nLatency jitter - Mean: {mean:.2f}ms, StdDev: {std_dev:.2f}ms, CV: {cv:.2%}")
        print(f"P95: {p95:.2f}ms (high jitter expected due to lazy truncation)")
        
        # P95 latency should be under 2x max baseline (10ms)
        # This allows for periodic truncation spikes while ensuring
        # most tokens are processed quickly
        assert p95 < 10.0, f"P95 latency too high: {p95:.2f}ms > 10.0ms"
        
        # Mean should still be under threshold
        assert mean < BASELINE.latency_threshold(), (
            f"Mean latency too high: {mean:.2f}ms > {BASELINE.latency_threshold():.2f}ms"
        )


# =============================================================================
# Memory Tests
# =============================================================================

class TestMemory:
    """Memory usage regression tests."""
    
    def test_memory_bounded(self, model, memory_baseline):
        """Test that memory stays bounded regardless of sequence length."""
        memory_samples = []
        
        with torch.no_grad():
            ctx = model.embed(0)
            
            # Process 500 tokens, sampling memory every 100
            for i in range(500):
                ctx = model.step(ctx, i % 10)
                if i % 100 == 99:
                    torch.cuda.synchronize()
                    memory_samples.append(torch.cuda.memory_allocated() - memory_baseline)
        
        # Memory should stay roughly constant (within 2x of initial)
        initial = memory_samples[0]
        final = memory_samples[-1]
        max_memory = max(memory_samples)
        
        print(f"\nMemory samples (MB): {[m/1e6 for m in memory_samples]}")
        print(f"Initial: {initial/1e6:.2f}MB, Final: {final/1e6:.2f}MB, Max: {max_memory/1e6:.2f}MB")
        
        threshold = BASELINE.memory_threshold()
        
        assert max_memory < threshold, (
            f"Memory regression: {max_memory/1e6:.2f}MB > {threshold/1e6:.2f}MB"
        )
        
        # Memory should not grow unboundedly (final < 2x initial)
        assert final < initial * 2, (
            f"Memory growing unboundedly: {final/1e6:.2f}MB > 2× initial ({initial/1e6:.2f}MB)"
        )
    
    def test_no_memory_leak(self, model):
        """Test for memory leaks over repeated operations."""
        gc.collect()
        torch.cuda.empty_cache()
        
        initial_memory = torch.cuda.memory_allocated()
        
        # Run multiple forward passes
        for _ in range(10):
            with torch.no_grad():
                ctx = model.embed(0)
                for i in range(100):
                    ctx = model.step(ctx, i % 10)
            
            # Clean up between iterations
            del ctx
            gc.collect()
            torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated()
        leak = final_memory - initial_memory
        
        print(f"\nMemory leak check - Initial: {initial_memory/1e6:.2f}MB, Final: {final_memory/1e6:.2f}MB")
        print(f"Potential leak: {leak/1e6:.2f}MB")
        
        # Allow small variance (1MB) due to CUDA allocator behavior
        assert leak < 1 * 1024 * 1024, f"Memory leak detected: {leak/1e6:.2f}MB"


# =============================================================================
# Learning Tests
# =============================================================================

class TestLearning:
    """Learning capability regression tests."""
    
    @pytest.mark.slow
    def test_digit_prediction_accuracy(self):
        """Test that model can learn digit prediction task."""
        from fluidelite.llm.fluid_elite import FluidElite
        
        # Small model for faster testing
        model = FluidElite(
            num_sites=12,
            rank=64,
            mpo_rank=1,
            vocab_size=10,
            truncate_every=10,
            dtype=torch.float32
        )
        model.cuda()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Training data: 0,1,2,...,9,0,1,2,...
        train_seq = list(range(10)) * 10
        
        # Quick training (fewer epochs for test speed)
        for epoch in range(50):
            model.train()
            ctx = model.embed(train_seq[0])
            total_loss = 0
            
            for i in range(len(train_seq) - 1):
                logits = model.predict(ctx)
                target = torch.tensor([train_seq[i + 1]], device='cuda')
                loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), target)
                total_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                with torch.no_grad():
                    ctx = model.step(ctx.detach() if hasattr(ctx, 'detach') else ctx, train_seq[i + 1])
        
        # Evaluate
        model.eval()
        correct = 0
        total = 10
        
        with torch.no_grad():
            ctx = model.embed(0)
            for i in range(total):
                logits = model.predict(ctx)
                pred = logits.argmax().item()
                expected = (i + 1) % 10
                if pred == expected:
                    correct += 1
                ctx = model.step(ctx, expected)
        
        accuracy = correct / total
        threshold = BASELINE.accuracy_threshold()
        
        print(f"\nLearning accuracy: {accuracy:.1%} (threshold: {threshold:.1%})")
        
        # Cleanup
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
        
        assert accuracy >= threshold, (
            f"Learning regression: {accuracy:.1%} < {threshold:.1%} "
            f"(baseline: {BASELINE.min_accuracy:.1%})"
        )


# =============================================================================
# Fallback Tests
# =============================================================================

class TestFallback:
    """Tests for fallback mechanisms."""
    
    def test_fallback_detection(self):
        """Test that fallback capabilities are detected correctly."""
        from fluidelite.utils.fallback import get_capabilities, Backend
        
        caps = get_capabilities()
        
        print(f"\n{caps}")
        
        assert caps.has_cuda, "CUDA should be available for these tests"
        assert caps.recommended_backend != Backend.PYTORCH_CPU, (
            "Should recommend GPU backend when CUDA available"
        )
    
    def test_svd_fallback(self):
        """Test that batched SVD works with fallback."""
        from fluidelite.utils.fallback import batched_svd
        
        batch = torch.randn(16, 128, 64, device='cuda')
        
        # Should work without raising
        U, S, Vh = batched_svd(batch)
        
        assert U.shape == (16, 128, 64), f"U shape wrong: {U.shape}"
        assert S.shape == (16, 64), f"S shape wrong: {S.shape}"
        assert Vh.shape == (16, 64, 64), f"Vh shape wrong: {Vh.shape}"
    
    def test_mpo_contract_fallback(self):
        """Test that MPO contraction works with fallback."""
        from fluidelite.utils.fallback import mpo_contract
        
        L, chi, d, D = 16, 32, 2, 4
        mps = torch.randn(L, chi, d, chi, device='cuda')
        mpo = torch.randn(L, D, d, d, D, device='cuda')
        
        # Should work without raising
        result = mpo_contract(mps, mpo)
        
        expected_chi = chi * D
        assert result.shape == (L, expected_chi, d, expected_chi), (
            f"Result shape wrong: {result.shape}, expected (L, {expected_chi}, d, {expected_chi})"
        )


# =============================================================================
# Integration Test
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_pipeline(self, model):
        """Test full pipeline: embed → step → predict."""
        with torch.no_grad():
            # Embed
            ctx = model.embed(42)
            assert ctx.L == model.L, "Embedding site count mismatch"
            
            # Step multiple times
            for i in range(50):
                ctx = model.step(ctx, i % 10)
            
            # Predict
            logits = model.predict(ctx)
            assert logits.shape == (model.vocab_size,), f"Logits shape: {logits.shape}"
            
            # Should produce valid probabilities
            probs = torch.softmax(logits, dim=0)
            assert torch.isclose(probs.sum(), torch.tensor(1.0)), "Probabilities don't sum to 1"
            assert not torch.isnan(probs).any(), "NaN in probabilities"
    
    def test_error_handling(self):
        """Test that errors are handled gracefully."""
        from fluidelite.utils.cuda_utils import CUDAContext, cuda_error_context
        
        # Context manager should work without issues
        with CUDAContext() as ctx:
            assert ctx.has_cuda
            device = ctx.device
            assert device.type == "cuda"
        
        # Error context should wrap CUDA errors
        with cuda_error_context("test operation"):
            # Normal operation should pass
            x = torch.randn(100, 100, device='cuda')
            y = x @ x.T
            assert y.shape == (100, 100)


# =============================================================================
# Benchmark Summary
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def print_summary(request):
    """Print performance summary after all tests."""
    yield
    
    print("\n" + "=" * 60)
    print("PERFORMANCE REGRESSION TEST SUMMARY")
    print("=" * 60)
    print(f"Throughput baseline: {BASELINE.min_throughput} tok/s")
    print(f"Latency baseline: {BASELINE.max_latency} ms/token")
    print(f"Memory baseline: {BASELINE.max_memory_per_seq / 1e6:.1f} MB")
    print(f"Accuracy baseline: {BASELINE.min_accuracy:.0%}")
    print(f"Regression tolerance: {BASELINE.tolerance:.0%}")
    print("=" * 60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
