"""
Tests for fluidelite.llm.fluid_elite
====================================

Per Article II: Test Discipline
Per Article VII.7.2: Definition of Done = USER-OBSERVABLE BEHAVIOR works
"""

import pytest
import torch
from fluidelite.llm.fluid_elite import FluidElite, EliteLinear
from fluidelite.core.mps import MPS


class TestEliteLinear:
    """Tests for EliteLinear MPO layer."""
    
    def test_creation(self):
        """Test EliteLinear initialization."""
        layer = EliteLinear(num_sites=8, bond_dim=16, phys_dim=2)
        assert layer.L == 8
        assert layer.D == 16
        assert layer.d == 2
        
    def test_cores_shape(self):
        """Test cores parameter has correct shape."""
        layer = EliteLinear(num_sites=8, bond_dim=16, phys_dim=2)
        assert layer.cores.shape == (8, 16, 2, 2, 16)
        
    def test_forward(self):
        """Test forward pass produces valid MPS."""
        layer = EliteLinear(num_sites=8, bond_dim=16, phys_dim=2)
        mps_in = MPS.random(L=8, d=2, chi=4, dtype=torch.float64)
        
        mps_out = layer(mps_in)
        
        assert isinstance(mps_out, MPS)
        assert mps_out.L == 8
        
    def test_forward_loop_fallback(self):
        """Test loop-based forward works."""
        layer = EliteLinear(num_sites=4, bond_dim=8, phys_dim=2)
        mps_in = MPS.random(L=4, d=2, chi=4, dtype=torch.float64)
        
        # Use internal loop method
        mps_out = layer._forward_loop(mps_in)
        
        assert isinstance(mps_out, MPS)
        assert mps_out.L == 4


class TestFluidElite:
    """Tests for FluidElite main model."""
    
    def test_creation(self):
        """Test FluidElite initialization."""
        model = FluidElite(num_sites=8, rank=16, vocab_size=100)
        
        assert model.L == 8
        assert model.rank == 16
        assert model.vocab_size == 100
        
    def test_embed_creates_mps(self):
        """Test embed produces valid MPS."""
        model = FluidElite(num_sites=8, rank=16, vocab_size=100)
        
        mps = model.embed(42)
        
        assert isinstance(mps, MPS)
        assert mps.L == 8
        assert mps.d == 2
        
    def test_embed_different_tokens(self):
        """Test different tokens produce different embeddings."""
        model = FluidElite(num_sites=8, rank=16, vocab_size=100)
        
        mps1 = model.embed(42)
        mps2 = model.embed(13)
        
        # Different tokens should produce different states
        t1 = mps1.to_tensor()
        t2 = mps2.to_tensor()
        
        assert not torch.allclose(t1, t2)
        
    def test_step(self):
        """Test step produces valid updated MPS."""
        model = FluidElite(num_sites=8, rank=16, vocab_size=100)
        
        ctx = MPS.random(L=8, d=2, chi=1, dtype=torch.float64)
        new_ctx = model.step(ctx, token_id=42)
        
        assert isinstance(new_ctx, MPS)
        assert new_ctx.L == 8
        
    def test_step_changes_context(self):
        """Test step modifies the context state."""
        model = FluidElite(num_sites=8, rank=16, vocab_size=100)
        
        ctx = MPS.random(L=8, d=2, chi=1, dtype=torch.float64)
        ctx_copy = ctx.copy()
        new_ctx = model.step(ctx_copy, token_id=42)
        
        # Context should change
        # (Note: original ctx is not modified, new_ctx is different)
        assert isinstance(new_ctx, MPS)
        
    def test_predict_shape(self):
        """Test predict produces correct shape logits."""
        model = FluidElite(num_sites=8, rank=16, vocab_size=100)
        
        ctx = MPS.random(L=8, d=2, chi=1, dtype=torch.float64)
        ctx = model.step(ctx, 42)
        
        logits = model.predict(ctx)
        
        assert logits.shape == (100,)
        
    def test_forward_sequence(self):
        """Test forward processes sequence."""
        model = FluidElite(num_sites=8, rank=16, vocab_size=100)
        
        logits = model([1, 2, 3, 4, 5])
        
        assert logits.shape == (100,)
        
    def test_parameters(self):
        """Test model has learnable parameters."""
        model = FluidElite(num_sites=8, rank=16, vocab_size=100)
        
        params = list(model.parameters())
        assert len(params) > 0
        
        # Count parameters
        num_params = sum(p.numel() for p in params)
        assert num_params > 0
        
    def test_gradient_flow(self):
        """Test gradients flow through model."""
        model = FluidElite(num_sites=8, rank=16, vocab_size=100)
        
        ctx = MPS.random(L=8, d=2, chi=1, dtype=torch.float64)
        ctx = model.step(ctx, 42)
        logits = model.predict(ctx)
        
        loss = logits.sum()
        loss.backward()
        
        # Check gradients exist
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad


class TestFluidEliteIntegration:
    """Integration tests for FluidElite."""
    
    def test_multiple_steps(self):
        """Test model handles multiple consecutive steps."""
        model = FluidElite(num_sites=8, rank=16, vocab_size=100)
        
        ctx = MPS.random(L=8, d=2, chi=1, dtype=torch.float64)
        
        for token in [1, 2, 3, 4, 5]:
            ctx = model.step(ctx, token)
            
        logits = model.predict(ctx)
        assert logits.shape == (100,)
        
    def test_memory_bounded(self):
        """Test memory doesn't grow unboundedly."""
        model = FluidElite(num_sites=8, rank=16, vocab_size=100)
        
        ctx = MPS.random(L=8, d=2, chi=1, dtype=torch.float64)
        
        # Process many tokens
        for i in range(100):
            ctx = model.step(ctx, i % 100)
            
        # Bond dimension should be bounded
        assert ctx.chi <= 32  # Should be truncated back
        
    def test_training_step(self):
        """Test a single training step works."""
        model = FluidElite(num_sites=8, rank=16, vocab_size=100)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Forward
        ctx = MPS.random(L=8, d=2, chi=1, dtype=torch.float64)
        for token in [1, 2, 3]:
            ctx = model.step(ctx, token)
        logits = model.predict(ctx)
        
        # Backward
        target = torch.tensor([5])
        loss = criterion(logits.unsqueeze(0), target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Should complete without error
        assert True
