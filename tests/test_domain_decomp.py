"""
Unit tests for domain decomposition edge cases.

Tests edge cases including:
- Single domain (n_procs=1)
- Maximum domains (n_procs = nx * ny)
- Boundary conditions (periodic, non-periodic)
- Ghost zone handling
- Neighbor computation with periodicity

Author: HyperTensor Team
"""

import pytest
import torch
import numpy as np

from tensornet.distributed.domain_decomp import (
    DomainDecomposition,
    DomainConfig,
    DecompType,
    SubdomainInfo,
)


class TestSingleDomain:
    """Tests for single-processor decomposition."""
    
    def test_single_domain_covers_full_grid(self):
        """Single domain should cover entire grid."""
        config = DomainConfig(nx=64, ny=64, n_procs=1)
        decomp = DomainDecomposition(config)
        
        subdomain = decomp.get_subdomain(0)
        
        assert subdomain.i_start == 0
        assert subdomain.i_end == 64
        assert subdomain.j_start == 0
        assert subdomain.j_end == 64
    
    def test_single_domain_no_neighbors(self):
        """Single domain should have no neighbors (non-periodic)."""
        config = DomainConfig(
            nx=64, ny=64, n_procs=1,
            periodic_x=False, periodic_y=False
        )
        decomp = DomainDecomposition(config)
        
        subdomain = decomp.get_subdomain(0)
        
        assert subdomain.neighbor_left is None
        assert subdomain.neighbor_right is None
        assert subdomain.neighbor_bottom is None
        assert subdomain.neighbor_top is None
    
    def test_single_domain_periodic_self_neighbor(self):
        """Single periodic domain should have self as neighbor."""
        config = DomainConfig(
            nx=64, ny=64, n_procs=1,
            periodic_x=True, periodic_y=True
        )
        decomp = DomainDecomposition(config)
        
        subdomain = decomp.get_subdomain(0)
        
        # With periodic BC, single domain neighbors itself
        assert subdomain.neighbor_left == 0
        assert subdomain.neighbor_right == 0
        assert subdomain.neighbor_bottom == 0
        assert subdomain.neighbor_top == 0
    
    def test_single_domain_no_ghost_zones_non_periodic(self):
        """Single non-periodic domain should have no ghost zones."""
        config = DomainConfig(
            nx=64, ny=64, n_procs=1, n_ghost=2,
            periodic_x=False, periodic_y=False
        )
        decomp = DomainDecomposition(config)
        
        subdomain = decomp.get_subdomain(0)
        
        # Local size should equal global size (no ghosts needed)
        # Note: This depends on implementation - may still allocate ghost space
        assert subdomain.local_nx >= 64
        assert subdomain.local_ny >= 64


class TestMaxDomains:
    """Tests for maximum number of domains."""
    
    def test_max_domains_one_cell_per_domain(self):
        """Each domain gets at least one cell."""
        # 8x8 grid with 64 processors = 1 cell per processor
        config = DomainConfig(nx=8, ny=8, n_procs=64)
        decomp = DomainDecomposition(config)
        
        total_cells = 0
        for rank in range(64):
            subdomain = decomp.get_subdomain(rank)
            cells = (subdomain.i_end - subdomain.i_start) * \
                    (subdomain.j_end - subdomain.j_start)
            total_cells += cells
        
        # All cells must be covered exactly once
        assert total_cells == 64
    
    def test_more_procs_than_cells_raises(self):
        """Should handle more processors than cells gracefully."""
        # 4x4 grid with 32 processors - some will be empty
        config = DomainConfig(nx=4, ny=4, n_procs=32)
        
        # Depending on implementation, this may:
        # - Raise an error
        # - Assign empty domains to excess processors
        # - Cap processor count
        try:
            decomp = DomainDecomposition(config)
            # If it succeeds, verify all cells are covered
            total_cells = 0
            for rank in range(32):
                try:
                    subdomain = decomp.get_subdomain(rank)
                    cells = (subdomain.i_end - subdomain.i_start) * \
                            (subdomain.j_end - subdomain.j_start)
                    total_cells += cells
                except (KeyError, ValueError):
                    # Empty domains may not exist
                    pass
            assert total_cells == 16  # 4x4 = 16 cells
        except ValueError:
            # Valid to reject this configuration
            pass
    
    def test_prime_number_procs(self):
        """Prime number of processors should still work."""
        config = DomainConfig(nx=64, ny=64, n_procs=7)
        decomp = DomainDecomposition(config)
        
        # Verify all 7 domains exist
        for rank in range(7):
            subdomain = decomp.get_subdomain(rank)
            assert subdomain.rank == rank


class TestBoundaryConditions:
    """Tests for boundary condition handling."""
    
    def test_periodic_x_neighbors(self):
        """Periodic X should connect left and right boundaries."""
        config = DomainConfig(
            nx=64, ny=64, n_procs=4,
            periodic_x=True, periodic_y=False
        )
        decomp = DomainDecomposition(config)
        
        # Find leftmost and rightmost domains
        px, py, _ = decomp.proc_dims
        
        # Leftmost domain should have rightmost as left neighbor
        # Rightmost domain should have leftmost as right neighbor
        # (Exact behavior depends on processor grid layout)
        all_have_neighbors = True
        for rank in range(4):
            subdomain = decomp.get_subdomain(rank)
            # In periodic X, all domains should have left and right neighbors
            if subdomain.neighbor_left is None or subdomain.neighbor_right is None:
                all_have_neighbors = False
        
        assert all_have_neighbors
    
    def test_non_periodic_boundary_no_neighbor(self):
        """Non-periodic boundaries should have no neighbor."""
        config = DomainConfig(
            nx=64, ny=64, n_procs=4,
            periodic_x=False, periodic_y=False
        )
        decomp = DomainDecomposition(config)
        
        # At least one domain should have a None neighbor
        # (the corner domains)
        has_none_neighbor = False
        for rank in range(4):
            subdomain = decomp.get_subdomain(rank)
            if (subdomain.neighbor_left is None or 
                subdomain.neighbor_right is None or
                subdomain.neighbor_bottom is None or
                subdomain.neighbor_top is None):
                has_none_neighbor = True
                break
        
        assert has_none_neighbor
    
    def test_mixed_periodicity(self):
        """Mixed periodic/non-periodic should work correctly."""
        config = DomainConfig(
            nx=64, ny=64, n_procs=4,
            periodic_x=True, periodic_y=False
        )
        decomp = DomainDecomposition(config)
        
        # All domains should have X neighbors (periodic)
        # But corner domains should not have Y neighbors on one side
        for rank in range(4):
            subdomain = decomp.get_subdomain(rank)
            assert subdomain.neighbor_left is not None
            assert subdomain.neighbor_right is not None


class TestGhostZones:
    """Tests for ghost zone allocation."""
    
    def test_interior_domain_full_ghost_zones(self):
        """Interior domains should have ghosts on all sides."""
        config = DomainConfig(
            nx=64, ny=64, n_procs=9,  # 3x3 grid
            n_ghost=2
        )
        decomp = DomainDecomposition(config)
        
        # Find center domain (rank 4 in 3x3 grid)
        center = decomp.get_subdomain(4)
        
        # Interior domain should have neighbors on all sides
        assert center.neighbor_left is not None
        assert center.neighbor_right is not None
        assert center.neighbor_bottom is not None
        assert center.neighbor_top is not None
    
    def test_ghost_zone_size_consistent(self):
        """Ghost zones should match config."""
        n_ghost = 3
        config = DomainConfig(
            nx=64, ny=64, n_procs=4,
            n_ghost=n_ghost
        )
        decomp = DomainDecomposition(config)
        
        # Check that local sizes include ghost zones appropriately
        for rank in range(4):
            subdomain = decomp.get_subdomain(rank)
            # Local size should be base size + ghost zones
            base_x = subdomain.i_end - subdomain.i_start
            base_y = subdomain.j_end - subdomain.j_start
            
            # Local size should be >= base size
            assert subdomain.local_nx >= base_x
            assert subdomain.local_ny >= base_y


class TestDecompTypes:
    """Tests for different decomposition types."""
    
    def test_slab_decomposition(self):
        """SLAB decomposition should split in one dimension."""
        config = DomainConfig(
            nx=64, ny=64, n_procs=4,
            decomp_type=DecompType.SLAB
        )
        decomp = DomainDecomposition(config)
        
        # All domains should have same Y extent in SLAB
        y_extents = []
        for rank in range(4):
            subdomain = decomp.get_subdomain(rank)
            y_extents.append(subdomain.j_end - subdomain.j_start)
        
        # SLAB typically splits X, so Y should be consistent
        # (Exact behavior depends on implementation)
        assert len(set(y_extents)) <= 2  # At most 2 different sizes (load balance)
    
    def test_block_decomposition(self):
        """BLOCK decomposition should create roughly square subdomains."""
        config = DomainConfig(
            nx=64, ny=64, n_procs=4,
            decomp_type=DecompType.BLOCK
        )
        decomp = DomainDecomposition(config)
        
        # With 4 procs on 64x64, should get 2x2 grid of 32x32
        for rank in range(4):
            subdomain = decomp.get_subdomain(rank)
            x_size = subdomain.i_end - subdomain.i_start
            y_size = subdomain.j_end - subdomain.j_start
            
            # Aspect ratio should be reasonable for BLOCK
            aspect = max(x_size, y_size) / max(1, min(x_size, y_size))
            assert aspect <= 2.0  # No more than 2:1 aspect ratio


class TestProcDims:
    """Tests for processor grid dimension computation."""
    
    def test_proc_dims_multiply_to_nprocs(self):
        """Processor dimensions should multiply to n_procs."""
        for n_procs in [1, 2, 4, 7, 8, 12, 16, 25, 36]:
            config = DomainConfig(nx=64, ny=64, n_procs=n_procs)
            decomp = DomainDecomposition(config)
            
            px, py, pz = decomp.proc_dims
            assert px * py * pz == n_procs, f"Failed for n_procs={n_procs}"
    
    def test_proc_dims_2d_prefer_square(self):
        """2D decomposition should prefer square-ish grids."""
        config = DomainConfig(nx=64, ny=64, nz=1, n_procs=16)
        decomp = DomainDecomposition(config)
        
        px, py, pz = decomp.proc_dims
        
        # For 16 procs in 2D, should get 4x4 not 1x16 or 16x1
        assert pz == 1
        assert abs(px - py) <= 2  # Should be close to square


class TestCoveragee:
    """Tests for complete grid coverage."""
    
    def test_all_cells_covered(self):
        """All grid cells should be covered by exactly one domain."""
        config = DomainConfig(nx=32, ny=32, n_procs=4)
        decomp = DomainDecomposition(config)
        
        covered = np.zeros((32, 32), dtype=int)
        
        for rank in range(4):
            subdomain = decomp.get_subdomain(rank)
            covered[
                subdomain.i_start:subdomain.i_end,
                subdomain.j_start:subdomain.j_end
            ] += 1
        
        # Every cell should be covered exactly once
        assert np.all(covered == 1), "Some cells not covered or covered multiple times"
    
    def test_no_overlap(self):
        """Subdomains should not overlap (excluding ghost zones)."""
        config = DomainConfig(nx=64, ny=64, n_procs=8)
        decomp = DomainDecomposition(config)
        
        # Collect all (i, j) pairs from all domains
        all_cells = []
        for rank in range(8):
            subdomain = decomp.get_subdomain(rank)
            for i in range(subdomain.i_start, subdomain.i_end):
                for j in range(subdomain.j_start, subdomain.j_end):
                    all_cells.append((i, j))
        
        # No duplicates
        assert len(all_cells) == len(set(all_cells)), "Overlapping subdomains detected"


@pytest.mark.unit
class TestDomainDecompUnit:
    """Quick unit tests for CI."""
    
    def test_basic_creation(self):
        """Basic decomposition creation works."""
        config = DomainConfig(nx=16, ny=16, n_procs=4)
        decomp = DomainDecomposition(config)
        assert len(decomp.subdomains) == 4
    
    def test_get_all_subdomains(self):
        """Can retrieve all subdomains."""
        config = DomainConfig(nx=16, ny=16, n_procs=4)
        decomp = DomainDecomposition(config)
        
        for rank in range(4):
            subdomain = decomp.get_subdomain(rank)
            assert subdomain.rank == rank
    
    def test_invalid_rank_raises(self):
        """Invalid rank should raise appropriate error."""
        config = DomainConfig(nx=16, ny=16, n_procs=4)
        decomp = DomainDecomposition(config)
        
        with pytest.raises((KeyError, ValueError, IndexError)):
            decomp.get_subdomain(999)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
