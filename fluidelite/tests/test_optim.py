"""
Tests for fluidelite.optim.riemannian

Constitutional Compliance:
    - Article II.2.2: 80% test coverage required
"""

import pytest
import torch
from fluidelite.optim.riemannian import RiemannianAdam
from fluidelite.llm.fluid_elite import FluidElite
from fluidelite.core.mps import MPS


class TestRiemannianAdamCreation:
    """Test RiemannianAdam initialization."""
    
    def test_create_optimizer(self):
        """Test basic optimizer creation."""
        model = FluidElite(num_sites=8, rank=16, vocab_size=64)
        optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
        assert optimizer is not None
    
    def test_default_parameters(self):
        """Test default parameter values."""
        model = FluidElite(num_sites=8, rank=16, vocab_size=64)
        optimizer = RiemannianAdam(model.parameters())
        # Check defaults
        assert optimizer.defaults['lr'] == 1e-3
        assert optimizer.defaults['betas'] == (0.9, 0.999)
        assert optimizer.defaults['eps'] == 1e-8
        assert optimizer.defaults['stabilize'] == True
    
    def test_custom_parameters(self):
        """Test custom parameter values."""
        model = FluidElite(num_sites=8, rank=16, vocab_size=64)
        optimizer = RiemannianAdam(
            model.parameters(),
            lr=1e-4,
            betas=(0.8, 0.99),
            eps=1e-6,
            stabilize=False
        )
        assert optimizer.defaults['lr'] == 1e-4
        assert optimizer.defaults['betas'] == (0.8, 0.99)
        assert optimizer.defaults['eps'] == 1e-6
        assert optimizer.defaults['stabilize'] == False


class TestRiemannianAdamStep:
    """Test RiemannianAdam optimization step."""
    
    def test_step_runs(self):
        """Test that step() executes without error."""
        model = FluidElite(num_sites=8, rank=16, vocab_size=64)
        optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
        
        # Forward pass
        ctx = MPS.random(L=8, d=2, chi=1, dtype=torch.float64)
        ctx = model.step(ctx, 42)
        logits = model.predict(ctx)
        
        # Backward pass
        loss = logits.sum()
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        # Clear gradients
        optimizer.zero_grad()
    
    def test_step_updates_params(self):
        """Test that step() actually changes parameters."""
        model = FluidElite(num_sites=8, rank=16, vocab_size=64)
        optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
        
        # Store original parameters
        orig_params = [p.clone() for p in model.parameters()]
        
        # Forward pass
        ctx = MPS.random(L=8, d=2, chi=1, dtype=torch.float64)
        ctx = model.step(ctx, 42)
        logits = model.predict(ctx)
        
        # Backward pass
        loss = logits.sum()
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        # Check parameters changed
        for p_orig, p_new in zip(orig_params, model.parameters()):
            assert not torch.allclose(p_orig, p_new, atol=1e-10)
    
    def test_multiple_steps(self):
        """Test multiple optimization steps."""
        model = FluidElite(num_sites=8, rank=16, vocab_size=64)
        optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
        
        for _ in range(3):
            optimizer.zero_grad()
            
            ctx = MPS.random(L=8, d=2, chi=1, dtype=torch.float64)
            ctx = model.step(ctx, 42)
            logits = model.predict(ctx)
            
            loss = logits.sum()
            loss.backward()
            
            optimizer.step()
    
    def test_no_stabilize(self):
        """Test optimization without stabilization."""
        model = FluidElite(num_sites=8, rank=16, vocab_size=64)
        optimizer = RiemannianAdam(model.parameters(), lr=1e-3, stabilize=False)
        
        ctx = MPS.random(L=8, d=2, chi=1, dtype=torch.float64)
        ctx = model.step(ctx, 42)
        logits = model.predict(ctx)
        
        loss = logits.sum()
        loss.backward()
        optimizer.step()


class TestRiemannianAdamState:
    """Test optimizer state management."""
    
    def test_state_dict(self):
        """Test state_dict() works."""
        model = FluidElite(num_sites=8, rank=16, vocab_size=64)
        optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
        
        # Run a step to initialize state
        ctx = MPS.random(L=8, d=2, chi=1, dtype=torch.float64)
        ctx = model.step(ctx, 42)
        logits = model.predict(ctx)
        loss = logits.sum()
        loss.backward()
        optimizer.step()
        
        state = optimizer.state_dict()
        assert 'state' in state
        assert 'param_groups' in state
    
    def test_load_state_dict(self):
        """Test loading optimizer state."""
        model = FluidElite(num_sites=8, rank=16, vocab_size=64)
        optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
        
        # Run a step
        ctx = MPS.random(L=8, d=2, chi=1, dtype=torch.float64)
        ctx = model.step(ctx, 42)
        logits = model.predict(ctx)
        loss = logits.sum()
        loss.backward()
        optimizer.step()
        
        # Save and reload state
        state = optimizer.state_dict()
        optimizer2 = RiemannianAdam(model.parameters(), lr=1e-3)
        optimizer2.load_state_dict(state)
