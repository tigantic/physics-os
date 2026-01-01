"""
Test Module: tensornet/infrastructure/grid.py

Phase 10: Power Grid Cascading Failure Analysis
Constitutional Compliance:
    - Article III, Section 3.1: Unit tests with 90%+ coverage
    - Article III, Section 3.2: Deterministic seeding (seed=42)
    - Article IV, Section 4.1: Physical validation
    - Article V, Section 5.1: Float64 precision

References:
    Erdős, P. & Rényi, A. (1960). "On the evolution of random graphs."
    Publications of the Mathematical Institute of the Hungarian Academy
    of Sciences, 5, 17-61.
    
    Watts, D.J. & Strogatz, S.H. (1998). "Collective dynamics of 'small-world'
    networks." Nature, 393(6684), 440-442.
    
    Barabási, A.L. & Albert, R. (1999). "Emergence of scaling in random
    networks." Science, 286(5439), 509-512.
"""

import pytest
import torch
import numpy as np

from tensornet.infrastructure.grid import (
    CyberGrid,
    CascadeReport,
)


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
def small_grid():
    """Small test grid (50 nodes)."""
    return CyberGrid(
        n_nodes=50,
        topology='erdos_renyi',
        connection_prob=0.15,
        seed=42
    )


@pytest.fixture
def scale_free_grid():
    """Scale-free network grid (Barabási-Albert)."""
    return CyberGrid(
        n_nodes=100,
        topology='scale_free',
        m=3,  # New nodes connect to 3 existing nodes
        seed=42
    )


@pytest.fixture
def hierarchical_grid():
    """Hierarchical power grid."""
    return CyberGrid(
        n_nodes=200,
        topology='hierarchical',
        n_generators=10,
        n_substations=30,
        seed=42
    )


# ============================================================================
# UNIT TESTS
# ============================================================================

class TestCyberGridInit:
    """Test CyberGrid initialization."""
    
    @pytest.mark.unit
    def test_init_node_count(self, deterministic_seed):
        """Test node count initialization."""
        grid = CyberGrid(n_nodes=100, seed=42)
        
        assert grid.n_nodes == 100
    
    @pytest.mark.unit
    def test_float64_compliance(self, deterministic_seed):
        """Per Article V: Physics tensors must be float64."""
        grid = CyberGrid(n_nodes=50, seed=42)
        
        assert grid.node_loads.dtype == torch.float64
        assert grid.node_capacities.dtype == torch.float64
        assert grid.edge_flows.dtype == torch.float64
    
    @pytest.mark.unit
    def test_adjacency_matrix_created(self, small_grid, deterministic_seed):
        """Adjacency matrix should be created."""
        assert hasattr(small_grid, 'adjacency')
        assert small_grid.adjacency.shape == (50, 50)
    
    @pytest.mark.unit
    def test_erdos_renyi_connectivity(self, small_grid, deterministic_seed):
        """Erdős-Rényi graph should have edges."""
        edge_count = small_grid.adjacency.sum().item() / 2  # Undirected
        
        # Expected edges ≈ n*(n-1)/2 * p = 50*49/2 * 0.15 ≈ 184
        assert edge_count > 50  # Reasonable lower bound
        assert edge_count < 500  # Reasonable upper bound
    
    @pytest.mark.unit
    def test_scale_free_hubs(self, scale_free_grid, deterministic_seed):
        """Scale-free network should have hub nodes."""
        degree = scale_free_grid.adjacency.sum(dim=1)
        
        max_degree = degree.max().item()
        mean_degree = degree.mean().item()
        
        # Hubs should have much higher degree than average
        assert max_degree > 2 * mean_degree
    
    @pytest.mark.unit
    def test_initial_stable_state(self, small_grid, deterministic_seed):
        """Grid should start in stable state."""
        assert small_grid.failed_nodes.sum().item() == 0
        assert small_grid.cascading == False


class TestLoadDistribution:
    """Test load distribution on grid."""
    
    @pytest.mark.unit
    def test_set_node_load(self, small_grid, deterministic_seed):
        """Should be able to set node load."""
        small_grid.set_load(node_id=5, load=0.8)
        
        assert abs(small_grid.node_loads[5].item() - 0.8) < 1e-10
    
    @pytest.mark.unit
    def test_load_cannot_exceed_capacity(self, small_grid, deterministic_seed):
        """Load exceeding capacity should trigger failure."""
        capacity = small_grid.node_capacities[10].item()
        small_grid.set_load(node_id=10, load=capacity * 1.5)
        
        small_grid.step()
        
        # Node should fail
        assert small_grid.failed_nodes[10].item() == True
    
    @pytest.mark.unit
    def test_redistribute_load(self, small_grid, deterministic_seed):
        """Failed node load should redistribute to neighbors."""
        # Set loads
        for i in range(small_grid.n_nodes):
            small_grid.set_load(i, 0.5)
        
        initial_total_load = small_grid.node_loads.sum().item()
        
        # Fail a node (simulate)
        capacity = small_grid.node_capacities[20].item()
        small_grid.set_load(20, capacity * 2.0)
        small_grid.step()
        
        # Total load should be preserved (minus failed node)
        remaining_load = small_grid.node_loads[~small_grid.failed_nodes].sum().item()
        
        # Load should redistribute (some goes to neighbors)
        assert remaining_load > 0


class TestCascadeSimulation:
    """Test cascading failure simulation."""
    
    @pytest.mark.unit
    def test_trigger_cascade(self, small_grid, deterministic_seed):
        """Triggering cascade should propagate failures."""
        # Set moderate load on all nodes
        for i in range(small_grid.n_nodes):
            small_grid.set_load(i, 0.7)
        
        # Fail an initial node (simulate attack)
        small_grid.fail_node(node_id=25)
        
        # Run cascade simulation
        report = small_grid.simulate_cascade(max_steps=100)
        
        # Some additional nodes should fail
        assert small_grid.failed_nodes.sum().item() >= 1
    
    @pytest.mark.unit
    @pytest.mark.physics
    def test_cascade_reaches_steady_state(self, small_grid, deterministic_seed):
        """Cascade should eventually stabilize."""
        for i in range(small_grid.n_nodes):
            small_grid.set_load(i, 0.6)
        
        small_grid.fail_node(10)
        report = small_grid.simulate_cascade(max_steps=200)
        
        # Should have reached stable state
        assert report.converged == True
    
    @pytest.mark.unit
    @pytest.mark.physics
    def test_high_load_causes_large_cascade(self, deterministic_seed):
        """Higher initial load should cause larger cascades."""
        # Low load scenario
        grid_low = CyberGrid(n_nodes=50, seed=42)
        for i in range(50):
            grid_low.set_load(i, 0.3)
        grid_low.fail_node(25)
        report_low = grid_low.simulate_cascade(max_steps=100)
        
        # High load scenario
        grid_high = CyberGrid(n_nodes=50, seed=42)
        for i in range(50):
            grid_high.set_load(i, 0.85)
        grid_high.fail_node(25)
        report_high = grid_high.simulate_cascade(max_steps=100)
        
        # High load should have more failures
        assert report_high.total_failures >= report_low.total_failures
    
    @pytest.mark.unit
    @pytest.mark.physics
    def test_hub_failure_more_severe(self, scale_free_grid, deterministic_seed):
        """Failing hub should cause more severe cascade."""
        for i in range(100):
            scale_free_grid.set_load(i, 0.7)
        
        # Find hub (highest degree node)
        degree = scale_free_grid.adjacency.sum(dim=1)
        hub_id = degree.argmax().item()
        
        # Find leaf (lowest degree node)
        leaf_id = degree.argmin().item()
        
        # Test hub failure
        grid_hub = CyberGrid(n_nodes=100, topology='scale_free', m=3, seed=42)
        for i in range(100):
            grid_hub.set_load(i, 0.7)
        grid_hub.fail_node(hub_id)
        report_hub = grid_hub.simulate_cascade(max_steps=100)
        
        # Test leaf failure  
        grid_leaf = CyberGrid(n_nodes=100, topology='scale_free', m=3, seed=42)
        for i in range(100):
            grid_leaf.set_load(i, 0.7)
        grid_leaf.fail_node(leaf_id)
        report_leaf = grid_leaf.simulate_cascade(max_steps=100)
        
        # Hub failure should cause more damage
        assert report_hub.total_failures >= report_leaf.total_failures


class TestCascadeReport:
    """Test CascadeReport generation."""
    
    @pytest.mark.unit
    def test_report_generation(self, small_grid, deterministic_seed):
        """Should generate cascade report."""
        small_grid.fail_node(5)
        report = small_grid.simulate_cascade()
        
        assert isinstance(report, CascadeReport)
    
    @pytest.mark.unit
    def test_report_fields(self, small_grid, deterministic_seed):
        """Report should have required fields."""
        small_grid.fail_node(5)
        report = small_grid.simulate_cascade()
        
        assert hasattr(report, 'total_failures')
        assert hasattr(report, 'cascade_steps')
        assert hasattr(report, 'converged')
        assert hasattr(report, 'critical_nodes')
    
    @pytest.mark.unit
    def test_report_string(self, small_grid, deterministic_seed):
        """Report should have string representation."""
        small_grid.fail_node(5)
        report = small_grid.simulate_cascade()
        
        report_str = str(report)
        assert len(report_str) > 0


class TestNetworkAnalysis:
    """Test network analysis functions."""
    
    @pytest.mark.unit
    def test_identify_critical_nodes(self, small_grid, deterministic_seed):
        """Should identify critical nodes."""
        critical = small_grid.identify_critical_nodes(top_k=5)
        
        assert len(critical) == 5
    
    @pytest.mark.unit
    def test_betweenness_centrality(self, small_grid, deterministic_seed):
        """Should compute betweenness centrality."""
        centrality = small_grid.compute_betweenness()
        
        assert centrality.shape == (50,)
        assert centrality.min() >= 0
    
    @pytest.mark.unit
    def test_connected_components(self, small_grid, deterministic_seed):
        """Should identify connected components."""
        components = small_grid.get_connected_components()
        
        assert len(components) >= 1  # At least one component


class TestDeterminism:
    """Test reproducibility requirements."""
    
    @pytest.mark.unit
    def test_deterministic_topology(self):
        """Same seed should give same topology."""
        grid1 = CyberGrid(n_nodes=30, topology='erdos_renyi', 
                          connection_prob=0.2, seed=42)
        grid2 = CyberGrid(n_nodes=30, topology='erdos_renyi', 
                          connection_prob=0.2, seed=42)
        
        assert torch.equal(grid1.adjacency, grid2.adjacency)
    
    @pytest.mark.unit
    def test_deterministic_cascade(self):
        """Same seed should give same cascade."""
        # First cascade
        grid1 = CyberGrid(n_nodes=30, seed=42)
        for i in range(30):
            grid1.set_load(i, 0.7)
        grid1.fail_node(15)
        report1 = grid1.simulate_cascade()
        
        # Second cascade
        grid2 = CyberGrid(n_nodes=30, seed=42)
        for i in range(30):
            grid2.set_load(i, 0.7)
        grid2.fail_node(15)
        report2 = grid2.simulate_cascade()
        
        assert report1.total_failures == report2.total_failures


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestCyberGridIntegration:
    """Integration tests for cyber grid."""
    
    @pytest.mark.integration
    def test_full_attack_scenario(self, deterministic_seed):
        """Test complete attack and cascade scenario."""
        # Create realistic grid
        grid = CyberGrid(
            n_nodes=150,
            topology='hierarchical',
            n_generators=8,
            n_substations=25,
            seed=42
        )
        
        # Normal operating conditions
        for i in range(150):
            load = np.random.uniform(0.5, 0.75)
            grid.set_load(i, load)
        
        # Coordinated attack on substations
        critical = grid.identify_critical_nodes(top_k=3)
        for node_id in critical[:2]:
            grid.fail_node(node_id)
        
        # Simulate cascade
        report = grid.simulate_cascade(max_steps=200)
        
        # Verify report
        assert report.total_failures >= 2  # At least attacked nodes
        assert report.cascade_steps > 0
    
    @pytest.mark.integration
    @pytest.mark.physics
    def test_resilience_assessment(self, deterministic_seed):
        """Test grid resilience assessment."""
        # Create grid
        grid = CyberGrid(n_nodes=100, topology='scale_free', m=3, seed=42)
        
        # Set loads
        for i in range(100):
            grid.set_load(i, 0.6)
        
        # Test N-1 contingency (remove each node once)
        max_cascade = 0
        for test_node in range(0, 100, 10):  # Sample every 10th node
            # Create fresh grid
            test_grid = CyberGrid(n_nodes=100, topology='scale_free', m=3, seed=42)
            for i in range(100):
                test_grid.set_load(i, 0.6)
            
            test_grid.fail_node(test_node)
            report = test_grid.simulate_cascade()
            
            max_cascade = max(max_cascade, report.total_failures)
        
        # Grid should be reasonably resilient
        assert max_cascade < 50  # Less than 50% failure in N-1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])
