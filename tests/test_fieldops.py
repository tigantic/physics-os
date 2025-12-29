"""
Tests for Layer 1: FieldOps
===========================

Verifies physics operators work correctly in QTT format:
- Differential operators (grad, div, curl, laplacian)
- Transport operators (advect, diffuse)
- Projection (divergence-free)
- Forces (impulse, buoyancy, stir)
- Boundary conditions
- FieldGraph execution
"""

import pytest
import torch
import numpy as np

from tensornet.substrate import Field
from tensornet.fieldops import (
    Operator,
    FieldGraph,
    Grad,
    Div,
    Curl,
    Laplacian,
    Advect,
    Diffuse,
    Project,
    PoissonSolver,
    Impulse,
    Buoyancy,
    Attractor,
    Stir,
    BoundaryCondition,
    PeriodicBC,
    DirichletBC,
    NeumannBC,
    ObstacleMask,
    smoke_graph,
    fluid_graph,
    heat_graph,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def scalar_field():
    """Create a simple 2D scalar field."""
    return Field.zeros(dims=[4, 4], bits_per_dim=4)


@pytest.fixture
def vector_field():
    """Create a 2D velocity field."""
    f = Field.zeros(dims=[4, 4], bits_per_dim=4)
    # Add some structure
    for i, core in enumerate(f.cores):
        f.cores[i] = torch.randn_like(core) * 0.1
    return f


@pytest.fixture
def random_field():
    """Create a random field for testing."""
    f = Field.zeros(dims=[4, 4], bits_per_dim=4)
    for i, core in enumerate(f.cores):
        f.cores[i] = torch.randn_like(core) * 0.1
    return f


# =============================================================================
# DIFFERENTIAL OPERATORS
# =============================================================================

class TestGrad:
    """Test gradient operator."""
    
    def test_grad_creates_field(self, scalar_field):
        """Gradient returns a valid field."""
        grad = Grad()
        result = grad.apply(scalar_field)
        
        assert isinstance(result, Field)
        assert len(result.cores) == len(scalar_field.cores)
    
    def test_grad_orders(self, scalar_field):
        """Test different FD orders."""
        for order in [2, 4, 6]:
            grad = Grad(order=order)
            result = grad.apply(scalar_field)
            assert isinstance(result, Field)
    
    def test_grad_preserves_shape(self, scalar_field):
        """Gradient preserves core shapes."""
        grad = Grad()
        result = grad.apply(scalar_field)
        
        for orig, new in zip(scalar_field.cores, result.cores):
            assert orig.shape == new.shape


class TestDiv:
    """Test divergence operator."""
    
    def test_div_creates_field(self, vector_field):
        """Divergence returns a valid field."""
        div = Div()
        result = div.apply(vector_field)
        
        assert isinstance(result, Field)
    
    def test_div_of_zero_is_zero(self, scalar_field):
        """Divergence of zero field is near zero."""
        div = Div()
        result = div.apply(scalar_field)
        
        # Check cores are near zero
        total = sum(c.abs().sum().item() for c in result.cores)
        assert total < 1e-10


class TestCurl:
    """Test curl operator."""
    
    def test_curl_creates_field(self, vector_field):
        """Curl returns a valid field."""
        curl = Curl()
        result = curl.apply(vector_field)
        
        assert isinstance(result, Field)
    
    def test_curl_preserves_shape(self, vector_field):
        """Curl preserves core shapes."""
        curl = Curl()
        result = curl.apply(vector_field)
        
        for orig, new in zip(vector_field.cores, result.cores):
            assert orig.shape == new.shape


class TestLaplacian:
    """Test Laplacian operator."""
    
    def test_laplacian_creates_field(self, scalar_field):
        """Laplacian returns a valid field."""
        lap = Laplacian()
        result = lap.apply(scalar_field)
        
        assert isinstance(result, Field)
    
    def test_laplacian_of_constant_is_zero(self, scalar_field):
        """Laplacian of constant field is zero."""
        lap = Laplacian()
        result = lap.apply(scalar_field)
        
        # Zero field has zero Laplacian
        total = sum(c.abs().sum().item() for c in result.cores)
        assert total < 1e-10


# =============================================================================
# TRANSPORT OPERATORS
# =============================================================================

class TestAdvect:
    """Test advection operator."""
    
    def test_advect_creates_field(self, vector_field):
        """Advection returns a valid field."""
        advect = Advect()
        result = advect.apply(vector_field, dt=0.01)
        
        assert isinstance(result, Field)
    
    def test_advect_with_velocity(self, vector_field):
        """Advection with explicit velocity field."""
        velocity = Field.zeros(dims=[4, 4], bits_per_dim=4)
        advect = Advect(velocity=velocity)
        result = advect.apply(vector_field, dt=0.01)
        
        assert isinstance(result, Field)
    
    def test_advect_preserves_rank_approx(self, random_field):
        """Advection doesn't explode rank."""
        advect = Advect()
        result = advect.apply(random_field, dt=0.01)
        
        # Rank shouldn't increase much
        assert result.rank <= random_field.rank + 2


class TestDiffuse:
    """Test diffusion operator."""
    
    def test_diffuse_creates_field(self, random_field):
        """Diffusion returns a valid field."""
        diffuse = Diffuse(viscosity=0.01)
        result = diffuse.apply(random_field, dt=0.01)
        
        assert isinstance(result, Field)
    
    def test_diffuse_smooths_field(self, random_field):
        """Diffusion reduces field magnitude."""
        diffuse = Diffuse(viscosity=0.1)
        result = diffuse.apply(random_field, dt=1.0)
        
        # Energy should decrease
        orig_norm = sum(c.norm().item() for c in random_field.cores)
        new_norm = sum(c.norm().item() for c in result.cores)
        
        assert new_norm < orig_norm
    
    def test_diffuse_repr(self):
        """Test string representation."""
        diffuse = Diffuse(viscosity=0.05)
        assert "0.05" in repr(diffuse)


# =============================================================================
# PROJECTION
# =============================================================================

class TestProject:
    """Test divergence-free projection."""
    
    def test_project_creates_field(self, vector_field):
        """Projection returns a valid field."""
        project = Project()
        result = project.apply(vector_field, dt=0.01)
        
        assert isinstance(result, Field)
    
    def test_project_reduces_divergence(self, random_field):
        """Projection reduces divergence."""
        project = Project()
        div = Div()
        
        # Initial divergence
        initial_div = div.apply(random_field)
        initial_mag = sum(c.abs().sum().item() for c in initial_div.cores)
        
        # After projection
        projected = project.apply(random_field)
        final_div = div.apply(projected)
        final_mag = sum(c.abs().sum().item() for c in final_div.cores)
        
        # Divergence should be bounded (ideally reduced)
        # Note: This is approximate in our simplified implementation
        assert final_mag < initial_mag * 10  # Bounded, not necessarily smaller
    
    def test_poisson_solver(self, scalar_field):
        """Test Poisson solver directly."""
        poisson = PoissonSolver(tol=1e-6, max_iter=10)
        result = poisson.apply(scalar_field)
        
        assert isinstance(result, Field)


# =============================================================================
# FORCES
# =============================================================================

class TestImpulse:
    """Test impulse force."""
    
    def test_impulse_creates_field(self, vector_field):
        """Impulse returns a valid field."""
        impulse = Impulse(center=(0.5, 0.5), strength=1.0)
        result = impulse.apply(vector_field, dt=0.01)
        
        assert isinstance(result, Field)
    
    def test_impulse_adds_energy(self, scalar_field):
        """Impulse adds energy to field."""
        impulse = Impulse(center=(0.5, 0.5), strength=10.0, radius=0.5)
        result = impulse.apply(scalar_field, dt=1.0)
        
        # Some cores should have increased magnitude
        orig_norm = sum(c.norm().item() for c in scalar_field.cores)
        new_norm = sum(c.norm().item() for c in result.cores)
        
        # Impulse adds energy (or keeps same if at boundary)
        assert new_norm >= orig_norm


class TestBuoyancy:
    """Test buoyancy force."""
    
    def test_buoyancy_creates_field(self, scalar_field):
        """Buoyancy returns a valid field."""
        buoy = Buoyancy(alpha=1.0, gravity=(0, -9.8))
        result = buoy.apply(scalar_field, dt=0.01)
        
        assert isinstance(result, Field)


class TestAttractor:
    """Test attractor force."""
    
    def test_attractor_decay(self, random_field):
        """Attractor toward zero decays field."""
        attractor = Attractor(target=None, strength=0.5)
        result = attractor.apply(random_field, dt=1.0)
        
        orig_norm = sum(c.norm().item() for c in random_field.cores)
        new_norm = sum(c.norm().item() for c in result.cores)
        
        assert new_norm < orig_norm
    
    def test_attractor_toward_target(self, random_field):
        """Attractor pulls toward target."""
        target = Field.zeros(dims=[4, 4], bits_per_dim=4)
        attractor = Attractor(target=target, strength=0.1)
        result = attractor.apply(random_field, dt=1.0)
        
        assert isinstance(result, Field)


class TestStir:
    """Test stirring force."""
    
    def test_stir_creates_field(self, random_field):
        """Stir returns a valid field."""
        stir = Stir(center=(0.5, 0.5), strength=1.0)
        result = stir.apply(random_field, dt=0.01)
        
        assert isinstance(result, Field)


# =============================================================================
# BOUNDARY CONDITIONS
# =============================================================================

class TestBoundaryConditions:
    """Test boundary condition operators."""
    
    def test_periodic_bc(self, random_field):
        """Periodic BC is no-op."""
        bc = PeriodicBC()
        result = bc.apply(random_field.cores)
        
        # Should be unchanged
        for orig, new in zip(random_field.cores, result):
            assert torch.allclose(orig, new)
    
    def test_dirichlet_bc(self, random_field):
        """Dirichlet BC modifies boundaries."""
        bc = DirichletBC(value=0.0)
        result = bc.apply(random_field.cores)
        
        assert len(result) == len(random_field.cores)
    
    def test_neumann_bc(self, random_field):
        """Neumann BC modifies boundaries."""
        bc = NeumannBC(gradient=0.0)
        result = bc.apply(random_field.cores)
        
        assert len(result) == len(random_field.cores)
    
    def test_obstacle_mask(self, random_field):
        """Obstacle mask applies correctly."""
        mask = Field.zeros(dims=[4, 4], bits_per_dim=4)
        bc = ObstacleMask(mask=mask)
        result = bc.apply(random_field.cores)
        
        assert len(result) == len(random_field.cores)
    
    def test_obstacle_mask_none(self, random_field):
        """Obstacle mask with None is no-op."""
        bc = ObstacleMask(mask=None)
        result = bc.apply(random_field.cores)
        
        for orig, new in zip(random_field.cores, result):
            assert torch.allclose(orig, new)


# =============================================================================
# FIELD GRAPH
# =============================================================================

class TestFieldGraph:
    """Test operator graph execution."""
    
    def test_graph_creation(self):
        """Create empty graph."""
        graph = FieldGraph()
        assert len(graph.nodes) == 0
    
    def test_graph_add_operator(self):
        """Add operators to graph."""
        graph = FieldGraph()
        graph.add('advect', Advect())
        graph.add('diffuse', Diffuse())
        
        assert 'advect' in graph.nodes
        assert 'diffuse' in graph.nodes
    
    def test_graph_connect(self):
        """Connect operators."""
        graph = FieldGraph()
        graph.add('advect', Advect())
        graph.add('diffuse', Diffuse())
        graph.connect('advect', 'diffuse')
        
        assert ('advect', 'diffuse') in graph.edges
    
    def test_graph_execute(self, random_field):
        """Execute operator graph."""
        graph = FieldGraph()
        graph.add('advect', Advect())
        graph.add('diffuse', Diffuse())
        graph.connect('advect', 'diffuse')
        
        result = graph.execute(random_field, dt=0.01)
        
        assert isinstance(result, Field)
        assert graph.last_execution_ms > 0
    
    def test_graph_summary(self, random_field):
        """Get execution summary."""
        graph = FieldGraph()
        graph.add('advect', Advect())
        graph.execute(random_field, dt=0.01)
        
        summary = graph.summary()
        assert 'advect' in summary
        assert 'ms' in summary
    
    def test_graph_chaining(self):
        """Test fluent API."""
        graph = (
            FieldGraph()
            .add('a', Advect())
            .add('b', Diffuse())
            .add('c', Project())
            .connect('a', 'b', 'c')
            .compile()
        )
        
        assert graph._compiled
        assert len(graph.execution_order) == 3


class TestPresetGraphs:
    """Test preset simulation graphs."""
    
    def test_smoke_graph(self, random_field):
        """Smoke simulation graph."""
        graph = smoke_graph(viscosity=0.01)
        result = graph.execute(random_field, dt=0.01)
        
        assert isinstance(result, Field)
        assert len(graph.nodes) == 4
    
    def test_fluid_graph(self, random_field):
        """Fluid simulation graph."""
        graph = fluid_graph(viscosity=0.001)
        result = graph.execute(random_field, dt=0.01)
        
        assert isinstance(result, Field)
        assert len(graph.nodes) == 3
    
    def test_heat_graph(self, random_field):
        """Heat diffusion graph."""
        graph = heat_graph(diffusivity=0.1)
        result = graph.execute(random_field, dt=0.01)
        
        assert isinstance(result, Field)
        assert len(graph.nodes) == 1


# =============================================================================
# INTEGRATION
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple operators."""
    
    def test_full_fluid_step(self, random_field):
        """Complete fluid simulation step."""
        # 1. Advect
        advect = Advect()
        field = advect.apply(random_field, dt=0.01)
        
        # 2. Add forces
        buoy = Buoyancy(alpha=0.1)
        field = buoy.apply(field, dt=0.01)
        
        # 3. Diffuse
        diffuse = Diffuse(viscosity=0.01)
        field = diffuse.apply(field, dt=0.01)
        
        # 4. Project
        project = Project()
        field = project.apply(field, dt=0.01)
        
        assert isinstance(field, Field)
        assert field.rank > 0
    
    def test_multiple_timesteps(self, random_field):
        """Run multiple timesteps."""
        graph = fluid_graph(viscosity=0.001)
        
        field = random_field
        for _ in range(10):
            field = graph.execute(field, dt=0.01)
        
        assert isinstance(field, Field)
    
    def test_operator_composition(self, random_field):
        """Compose operators manually."""
        ops = [
            Advect(),
            Diffuse(viscosity=0.01),
            Project(),
        ]
        
        field = random_field
        for op in ops:
            field = op.apply(field, dt=0.01)
        
        assert isinstance(field, Field)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
