"""
Comprehensive test suite for HyperGrid.

Tests cover:
- Grid instantiation and properties
- Geometry primitives (box, cylinder, sphere)
- Boundary conditions (inlet, outlet, patches)
- Boolean operations (union, intersection, subtraction)
- Jump Flooding Algorithm SDF computation
- I/O and serialization with security validation
- Solver integration
- Visualization export

Run with: pytest tests/test_hypergrid.py -v
"""

import pytest
import torch
import tempfile
import os
from pathlib import Path

# Import HyperGrid components
from core.grid import (
    HyperGrid,
    BoundaryPatch,
    BooleanOp,
    jump_flooding_sdf_3d,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def device():
    """Return available device (GPU or CPU)."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def small_grid(device):
    """Create a small 32x32x32 grid for fast tests."""
    return HyperGrid(nx=32, ny=32, nz=32, lx=4.0, ly=4.0, lz=4.0, device=device)


@pytest.fixture
def medium_grid(device):
    """Create a medium 64x32x32 grid."""
    return HyperGrid(nx=64, ny=32, nz=32, lx=8.0, ly=4.0, lz=4.0, device=device)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for I/O tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# =============================================================================
# Test: Grid Instantiation
# =============================================================================

class TestGridInstantiation:
    """Tests for HyperGrid creation and basic properties."""

    def test_basic_instantiation(self, device):
        """Test that grid can be created with valid parameters."""
        grid = HyperGrid(nx=16, ny=16, nz=16, lx=2.0, ly=2.0, lz=2.0, device=device)
        assert grid.nx == 16
        assert grid.ny == 16
        assert grid.nz == 16
        assert grid.lx == 2.0
        assert grid.ly == 2.0
        assert grid.lz == 2.0

    def test_cell_spacing(self, device):
        """Test that cell spacing is computed correctly."""
        grid = HyperGrid(nx=10, ny=20, nz=40, lx=1.0, ly=2.0, lz=4.0, device=device)
        assert abs(grid.dx - 0.1) < 1e-6
        assert abs(grid.dy - 0.1) < 1e-6
        assert abs(grid.dz - 0.1) < 1e-6

    def test_geo_tensor_shape(self, small_grid):
        """Test geometry tensor has correct shape [5, Nx, Ny, Nz]."""
        assert small_grid.geo.shape == (5, 32, 32, 32)

    def test_initial_vol_frac_is_fluid(self, small_grid):
        """Test that initial grid is all fluid (vol_frac = 1)."""
        assert (small_grid.vol_frac == 1.0).all()

    def test_initial_sdf_is_positive(self, small_grid):
        """Test that initial SDF is positive (far from walls)."""
        assert (small_grid.sdf > 0).all()

    def test_cell_centers(self, device):
        """Test cell center coordinates are computed correctly."""
        grid = HyperGrid(nx=4, ny=4, nz=4, lx=4.0, ly=4.0, lz=4.0, device=device)
        X, Y, Z = grid.cell_centers
        
        # First cell center should be at dx/2
        assert abs(X[0, 0, 0].item() - 0.5) < 1e-5
        assert abs(Y[0, 0, 0].item() - 0.5) < 1e-5
        assert abs(Z[0, 0, 0].item() - 0.5) < 1e-5
        
        # Last cell center should be at lx - dx/2
        assert abs(X[-1, -1, -1].item() - 3.5) < 1e-5

    def test_device_placement(self, device):
        """Test that tensors are on the correct device."""
        grid = HyperGrid(nx=8, ny=8, nz=8, lx=1.0, ly=1.0, lz=1.0, device=device)
        assert grid.geo.device.type == device.split(':')[0]


# =============================================================================
# Test: Geometry Primitives
# =============================================================================

class TestGeometryPrimitives:
    """Tests for geometry primitive operations."""

    def test_add_box_creates_solid(self, small_grid):
        """Test that add_box creates solid region."""
        initial_solid = (small_grid.vol_frac < 0.5).sum()
        
        small_grid.add_box(1.0, 3.0, 1.0, 3.0, 1.0, 3.0)
        
        final_solid = (small_grid.vol_frac < 0.5).sum()
        assert final_solid > initial_solid

    def test_add_box_location(self, device):
        """Test that box is placed at correct location."""
        grid = HyperGrid(nx=32, ny=32, nz=32, lx=4.0, ly=4.0, lz=4.0, device=device)
        grid.add_box(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        
        # Check that solid is in expected region
        solid_mask = grid.vol_frac < 0.5
        
        # First 8 cells in each direction should be solid (roughly)
        corner_solid = solid_mask[:8, :8, :8].sum().item()
        far_solid = solid_mask[16:, 16:, 16:].sum().item()
        
        assert corner_solid > 0
        assert far_solid == 0

    def test_add_cylinder_z_axis(self, small_grid):
        """Test adding a cylinder along Z axis."""
        initial_solid = (small_grid.vol_frac < 0.5).sum()
        
        small_grid.add_cylinder(center=(2.0, 2.0), radius=0.5, z_min=0.0, z_max=4.0, axis='z')
        
        final_solid = (small_grid.vol_frac < 0.5).sum()
        assert final_solid > initial_solid

    def test_add_cylinder_x_axis(self, small_grid):
        """Test adding a cylinder along X axis."""
        initial_solid = (small_grid.vol_frac < 0.5).sum()
        
        small_grid.add_cylinder(center=(2.0, 2.0), radius=0.5, z_min=0.0, z_max=4.0, axis='x')
        
        final_solid = (small_grid.vol_frac < 0.5).sum()
        assert final_solid > initial_solid

    def test_add_sphere(self, small_grid):
        """Test adding a sphere."""
        initial_solid = (small_grid.vol_frac < 0.5).sum()
        
        small_grid.add_sphere(center=(2.0, 2.0, 2.0), radius=1.0)
        
        final_solid = (small_grid.vol_frac < 0.5).sum()
        assert final_solid > initial_solid

    def test_sphere_symmetry(self, device):
        """Test that sphere is symmetric."""
        grid = HyperGrid(nx=32, ny=32, nz=32, lx=4.0, ly=4.0, lz=4.0, device=device)
        grid.add_sphere(center=(2.0, 2.0, 2.0), radius=1.0)
        
        vf = grid.vol_frac
        
        # Check X-Y symmetry
        assert torch.allclose(vf, vf.flip(0), atol=0.1)
        assert torch.allclose(vf, vf.flip(1), atol=0.1)
        assert torch.allclose(vf, vf.flip(2), atol=0.1)

    def test_box_obstacle(self, small_grid):
        """Test add_box_obstacle creates solid region."""
        initial_solid = (small_grid.vol_frac < 0.5).sum()
        
        small_grid.add_box_obstacle(1.0, 3.0, 1.0, 3.0, 1.0, 3.0)
        
        final_solid = (small_grid.vol_frac < 0.5).sum()
        assert final_solid > initial_solid

    def test_multiple_obstacles(self, small_grid):
        """Test adding multiple obstacles."""
        small_grid.add_box(0.5, 1.5, 0.5, 1.5, 0.5, 1.5)
        n1 = (small_grid.vol_frac < 0.5).sum()
        
        small_grid.add_sphere(center=(3.0, 3.0, 3.0), radius=0.5)
        n2 = (small_grid.vol_frac < 0.5).sum()
        
        assert n2 > n1


# =============================================================================
# Test: Boundary Conditions
# =============================================================================

class TestBoundaryConditions:
    """Tests for boundary condition management."""

    def test_add_patch(self, medium_grid):
        """Test adding a generic patch."""
        patch = BoundaryPatch(
            name='test_patch',
            patch_type='wall',
            i_range=(0, 10),
            j_range=(0, 32),
            k_range=(0, 32),
            face='x-'
        )
        medium_grid.add_patch(patch)

        assert 'test_patch' in medium_grid.patches
        assert medium_grid.patches['test_patch'].patch_type == 'wall'

    def test_add_inlet(self, medium_grid):
        """Test adding an inlet boundary condition."""
        medium_grid.add_inlet(
            name='main_inlet',
            face='x-',
            range_1=(0.5, 3.5),  # y range
            range_2=(0.5, 3.5),  # z range
            velocity=(0.5, 0.0, 0.0)
        )

        assert 'main_inlet' in medium_grid.patches
        patch = medium_grid.patches['main_inlet']
        assert patch.patch_type == 'inlet'
        assert patch.velocity == (0.5, 0.0, 0.0)

    def test_add_outlet(self, medium_grid):
        """Test adding an outlet boundary condition."""
        medium_grid.add_outlet(
            name='main_outlet',
            face='x+',
            range_1=(0.5, 3.5),  # y range
            range_2=(0.5, 3.5)   # z range
        )

        assert 'main_outlet' in medium_grid.patches
        patch = medium_grid.patches['main_outlet']
        assert patch.patch_type == 'outlet'

    def test_boundary_patch_dataclass(self):
        """Test BoundaryPatch dataclass."""
        patch = BoundaryPatch(
            name='test',
            patch_type='inlet',
            i_range=(0, 5),
            j_range=(10, 20),
            k_range=(0, 32),
            face='x-',
            velocity=(1.0, 0.0, 0.0),
            temperature=300.0
        )
        
        assert patch.name == 'test'
        assert patch.velocity == (1.0, 0.0, 0.0)
        assert patch.temperature == 300.0


# =============================================================================
# Test: Boolean Operations
# =============================================================================

class TestBooleanOperations:
    """Tests for CSG-style boolean geometry operations."""

    def test_boolean_op_union(self, device):
        """Test BooleanOp.union (min of SDFs)."""
        sdf_a = torch.tensor([1.0, 2.0, -1.0], device=device)
        sdf_b = torch.tensor([2.0, 1.0, -2.0], device=device)
        
        result = BooleanOp.union(sdf_a, sdf_b)
        expected = torch.tensor([1.0, 1.0, -2.0], device=device)
        
        assert torch.allclose(result, expected)

    def test_boolean_op_intersection(self, device):
        """Test BooleanOp.intersection (max of SDFs)."""
        sdf_a = torch.tensor([1.0, 2.0, -1.0], device=device)
        sdf_b = torch.tensor([2.0, 1.0, -2.0], device=device)
        
        result = BooleanOp.intersection(sdf_a, sdf_b)
        expected = torch.tensor([2.0, 2.0, -1.0], device=device)
        
        assert torch.allclose(result, expected)

    def test_boolean_op_subtraction(self, device):
        """Test BooleanOp.subtraction (max(a, -b))."""
        sdf_a = torch.tensor([1.0, -2.0, -1.0], device=device)
        sdf_b = torch.tensor([-1.0, -1.0, 2.0], device=device)
        
        result = BooleanOp.subtraction(sdf_a, sdf_b)
        # max(a, -b) = max([1, -2, -1], [1, 1, -2]) = [1, 1, -1]
        expected = torch.tensor([1.0, 1.0, -1.0], device=device)
        
        assert torch.allclose(result, expected)

    def test_boolean_op_smooth_union(self, device):
        """Test smooth union with blend radius."""
        sdf_a = torch.tensor([0.1], device=device)
        sdf_b = torch.tensor([0.1], device=device)
        
        # Sharp union would give 0.1
        sharp = BooleanOp.union(sdf_a, sdf_b)
        
        # Smooth union should give less (more blending)
        smooth = BooleanOp.smooth_union(sdf_a, sdf_b, k=0.5)
        
        assert smooth < sharp

    def test_grid_boolean_union(self, device):
        """Test HyperGrid.boolean_union."""
        g1 = HyperGrid(nx=32, ny=32, nz=32, lx=4.0, ly=4.0, lz=4.0, device=device)
        g1.add_box(0.5, 2.0, 0.5, 2.0, 0.5, 2.0)
        g1.compute_sdf_from_geometry()
        n1 = (g1.vol_frac < 0.5).sum().item()
        
        g2 = HyperGrid(nx=32, ny=32, nz=32, lx=4.0, ly=4.0, lz=4.0, device=device)
        g2.add_box(2.0, 3.5, 2.0, 3.5, 2.0, 3.5)
        g2.compute_sdf_from_geometry()
        n2 = (g2.vol_frac < 0.5).sum().item()
        
        g_union = g1.copy()
        g_union.boolean_union(g2)
        n_union = (g_union.vol_frac < 0.5).sum().item()
        
        # Union should have at least as many cells as larger shape
        assert n_union >= max(n1, n2)

    def test_grid_boolean_intersection(self, device):
        """Test HyperGrid.boolean_intersection."""
        g1 = HyperGrid(nx=32, ny=32, nz=32, lx=4.0, ly=4.0, lz=4.0, device=device)
        g1.add_box(1.0, 3.0, 1.0, 3.0, 1.0, 3.0)
        g1.compute_sdf_from_geometry()
        n1 = (g1.vol_frac < 0.5).sum().item()
        
        g2 = HyperGrid(nx=32, ny=32, nz=32, lx=4.0, ly=4.0, lz=4.0, device=device)
        g2.add_sphere(center=(2.0, 2.0, 2.0), radius=1.0)
        g2.compute_sdf_from_geometry()
        n2 = (g2.vol_frac < 0.5).sum().item()
        
        g_inter = g1.copy()
        g_inter.boolean_intersection(g2)
        n_inter = (g_inter.vol_frac < 0.5).sum().item()
        
        # Intersection should have fewer cells than smaller shape
        assert n_inter <= min(n1, n2)

    def test_grid_boolean_subtract(self, device):
        """Test HyperGrid.boolean_subtract."""
        g1 = HyperGrid(nx=32, ny=32, nz=32, lx=4.0, ly=4.0, lz=4.0, device=device)
        g1.add_box(1.0, 3.0, 1.0, 3.0, 1.0, 3.0)
        g1.compute_sdf_from_geometry()
        n1 = (g1.vol_frac < 0.5).sum().item()
        
        g2 = HyperGrid(nx=32, ny=32, nz=32, lx=4.0, ly=4.0, lz=4.0, device=device)
        g2.add_sphere(center=(2.0, 2.0, 2.0), radius=0.5)
        g2.compute_sdf_from_geometry()
        
        g_sub = g1.copy()
        g_sub.boolean_subtract(g2)
        n_sub = (g_sub.vol_frac < 0.5).sum().item()
        
        # Subtraction should have fewer cells than original
        assert n_sub < n1

    def test_grid_copy(self, small_grid):
        """Test that copy creates independent grid."""
        small_grid.add_box(1.0, 2.0, 1.0, 2.0, 1.0, 2.0)
        n1 = (small_grid.vol_frac < 0.5).sum().item()
        
        copy = small_grid.copy()
        
        # Modify original
        small_grid.add_sphere(center=(3.0, 3.0, 3.0), radius=0.5)
        n2 = (small_grid.vol_frac < 0.5).sum().item()
        n_copy = (copy.vol_frac < 0.5).sum().item()
        
        # Copy should be unchanged
        assert n_copy == n1
        assert n2 > n1


# =============================================================================
# Test: Jump Flooding Algorithm SDF
# =============================================================================

class TestJumpFloodingSDF:
    """Tests for GPU-accelerated Jump Flooding SDF computation."""

    def test_jfa_basic(self, device):
        """Test basic JFA computation."""
        grid = HyperGrid(nx=32, ny=32, nz=32, lx=4.0, ly=4.0, lz=4.0, device=device)
        grid.add_sphere(center=(2.0, 2.0, 2.0), radius=1.0)
        
        solid = grid.vol_frac < 0.5
        sdf = jump_flooding_sdf_3d(solid, grid.dx, grid.dy, grid.dz)
        
        assert sdf.shape == (32, 32, 32)

    def test_jfa_sign_convention(self, device):
        """Test that JFA returns negative inside, positive outside."""
        grid = HyperGrid(nx=32, ny=32, nz=32, lx=4.0, ly=4.0, lz=4.0, device=device)
        grid.add_sphere(center=(2.0, 2.0, 2.0), radius=1.0)
        
        solid = grid.vol_frac < 0.5
        sdf = jump_flooding_sdf_3d(solid, grid.dx, grid.dy, grid.dz)
        
        # Interior (solid) should be negative
        assert sdf[solid].max() <= 0
        
        # Exterior (fluid) should be positive
        assert sdf[~solid].min() >= 0

    def test_jfa_boundary_near_zero(self, device):
        """Test that SDF is near zero at boundaries."""
        grid = HyperGrid(nx=64, ny=64, nz=64, lx=4.0, ly=4.0, lz=4.0, device=device)
        grid.add_sphere(center=(2.0, 2.0, 2.0), radius=1.0)
        
        solid = grid.vol_frac < 0.5
        sdf = jump_flooding_sdf_3d(solid, grid.dx, grid.dy, grid.dz)
        
        # Find boundary cells
        is_boundary = torch.zeros_like(solid)
        for dim in range(3):
            for shift in [-1, 1]:
                neighbor = torch.roll(solid.float(), shift, dim)
                is_boundary = is_boundary | (solid.float() != neighbor)
        
        # Boundary cells should have small |SDF|
        boundary_sdf = sdf[is_boundary].abs()
        assert boundary_sdf.max() < 2 * grid.dx

    def test_jfa_distance_scaling(self, device):
        """Test that SDF values scale with distance."""
        grid = HyperGrid(nx=64, ny=64, nz=64, lx=4.0, ly=4.0, lz=4.0, device=device)
        grid.add_sphere(center=(2.0, 2.0, 2.0), radius=1.0)
        
        solid = grid.vol_frac < 0.5
        sdf = jump_flooding_sdf_3d(solid, grid.dx, grid.dy, grid.dz)
        
        # At center of sphere (should be ~-1.0)
        center_idx = (32, 32, 32)
        center_sdf = sdf[center_idx].item()
        assert center_sdf < -0.5  # Should be roughly -1.0

    def test_compute_sdf_from_geometry(self, small_grid):
        """Test compute_sdf_from_geometry method."""
        small_grid.add_sphere(center=(2.0, 2.0, 2.0), radius=1.0)
        
        # Before: SDF is initialized but not computed from geometry
        small_grid.compute_sdf_from_geometry()
        
        # After: SDF should reflect sphere geometry
        solid = small_grid.vol_frac < 0.5
        assert small_grid.sdf[solid].max() <= 0


# =============================================================================
# Test: I/O and Serialization
# =============================================================================

class TestIO:
    """Tests for save/load functionality with security validation."""

    def test_save_and_load(self, small_grid, temp_dir, device):
        """Test basic save and load roundtrip."""
        # Add some geometry
        small_grid.add_box(1.0, 3.0, 1.0, 3.0, 1.0, 3.0)
        small_grid.add_inlet('inlet', 'x-', (0.5, 3.5), (0.5, 3.5), (0.5, 0, 0))
        
        # Save
        save_path = os.path.join(temp_dir, 'test_grid.pt')
        small_grid.save(save_path)
        
        assert os.path.exists(save_path)
        
        # Load with allowed_dirs
        loaded = HyperGrid.load(save_path, device=device, allowed_dirs=[temp_dir])
        
        assert loaded.nx == small_grid.nx
        assert loaded.ny == small_grid.ny
        assert loaded.nz == small_grid.nz
        assert torch.allclose(loaded.vol_frac.cpu(), small_grid.vol_frac.cpu())

    def test_load_preserves_patches(self, small_grid, temp_dir, device):
        """Test that patches are preserved after save/load."""
        small_grid.add_inlet('test_inlet', 'x-', (0.5, 3.5), (0.5, 3.5), (1.0, 0, 0))
        small_grid.add_outlet('test_outlet', 'x+', (0.5, 3.5), (0.5, 3.5))
        
        save_path = os.path.join(temp_dir, 'grid_with_patches.pt')
        small_grid.save(save_path)
        
        loaded = HyperGrid.load(save_path, device=device, allowed_dirs=[temp_dir])
        
        assert 'test_inlet' in loaded.patches
        assert 'test_outlet' in loaded.patches
        assert loaded.patches['test_inlet'].velocity == (1.0, 0, 0)

    def test_load_security_path_validation(self, small_grid, temp_dir, device):
        """Test that load rejects paths outside allowed directories."""
        save_path = os.path.join(temp_dir, 'test_grid.pt')
        small_grid.save(save_path)
        
        # Try to load with different allowed directory
        other_dir = os.path.join(temp_dir, 'subdir')
        os.makedirs(other_dir, exist_ok=True)
        
        with pytest.raises(ValueError, match="outside allowed directories"):
            HyperGrid.load(save_path, device=device, allowed_dirs=[other_dir])

    def test_load_file_not_found(self, device, temp_dir):
        """Test that load raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            HyperGrid.load('/nonexistent/path/grid.pt', device=device, allowed_dirs=[temp_dir])

    def test_load_without_security_validation(self, small_grid, temp_dir, device):
        """Test that load works without allowed_dirs (but with warning in docs)."""
        save_path = os.path.join(temp_dir, 'test_grid.pt')
        small_grid.save(save_path)
        
        # Load without allowed_dirs - should work but is not recommended
        loaded = HyperGrid.load(save_path, device=device, allowed_dirs=None)
        
        assert loaded.nx == small_grid.nx


# =============================================================================
# Test: Flux and Volume Computations
# =============================================================================

class TestFluxAndVolume:
    """Tests for flux area and volume fraction computations."""

    def test_get_flux_areas(self, small_grid):
        """Test flux area computation."""
        Ax, Ay, Az = small_grid.get_flux_areas()
        
        # All fluid grid should have full face areas
        assert Ax.shape == (32, 32, 32)
        assert torch.allclose(Ax, torch.full_like(Ax, small_grid.dy * small_grid.dz))

    def test_get_flux_areas_with_solid(self, small_grid):
        """Test that flux areas are reduced by solids."""
        small_grid.add_box(1.0, 3.0, 1.0, 3.0, 1.0, 3.0)
        
        Ax, Ay, Az = small_grid.get_flux_areas()
        
        # Should have some reduced areas
        assert Ax.min() < small_grid.dy * small_grid.dz

    def test_get_cell_volumes(self, small_grid):
        """Test cell volume computation."""
        volumes = small_grid.get_cell_volumes()
        
        assert volumes.shape == (32, 32, 32)
        expected = small_grid.dx * small_grid.dy * small_grid.dz
        assert torch.allclose(volumes, torch.full_like(volumes, expected))

    def test_get_cell_volumes_with_solid(self, small_grid):
        """Test that cell volumes are reduced by solids."""
        small_grid.add_box(1.0, 3.0, 1.0, 3.0, 1.0, 3.0)
        
        volumes = small_grid.get_cell_volumes()
        
        # Should have some reduced volumes
        assert volumes.min() < small_grid.dx * small_grid.dy * small_grid.dz

    def test_mask_solid(self, small_grid):
        """Test mask_solid zeros out solid cells."""
        small_grid.add_box(1.0, 3.0, 1.0, 3.0, 1.0, 3.0)
        
        field = torch.ones_like(small_grid.vol_frac)
        masked = small_grid.mask_solid(field, value=0.0)
        
        # Solid cells should be zero
        solid_mask = small_grid.vol_frac < 0.01
        assert (masked[solid_mask] == 0.0).all()
        
        # Fluid cells should be unchanged
        assert (masked[~solid_mask] == 1.0).all()


# =============================================================================
# Test: Visualization Export
# =============================================================================

class TestVisualization:
    """Tests for visualization export (requires pyvista)."""

    @pytest.mark.skipif(
        not pytest.importorskip('pyvista', reason='PyVista not installed'),
        reason='PyVista not installed'
    )
    def test_to_pyvista(self, small_grid):
        """Test PyVista export."""
        import pyvista as pv
        
        small_grid.add_box(1.0, 3.0, 1.0, 3.0, 1.0, 3.0)
        
        pv_grid = small_grid.to_pyvista()
        
        assert isinstance(pv_grid, pv.RectilinearGrid)
        assert 'vol_frac' in pv_grid.cell_data
        assert 'sdf' in pv_grid.cell_data


# =============================================================================
# Test: Integration with Solvers
# =============================================================================

class TestSolverIntegration:
    """Tests for integration with HyperFoam solvers."""

    def test_solver_can_use_grid(self, device):
        """Test that HyperFoamSolver can use HyperGrid."""
        pytest.importorskip('core.solver', reason='Solver module not available')
        from core.solver import HyperFoamSolver, ProjectionConfig
        
        grid = HyperGrid(nx=32, ny=16, nz=16, lx=4.0, ly=2.0, lz=2.0, device=device)
        grid.add_box(1.5, 2.5, 0.5, 1.5, 0.5, 1.5)
        grid.compute_sdf_from_geometry()
        
        config = ProjectionConfig(
            nx=32, ny=16, nz=16,
            Lx=4.0, Ly=2.0, Lz=2.0,
            nu=1.5e-5, inlet_velocity=0.5, dt=0.01
        )
        
        solver = HyperFoamSolver(grid, config)
        solver.init_uniform_flow()
        
        # Run a few steps
        for _ in range(5):
            solver.step()
        
        # Should not raise any errors
        assert solver.u.shape == (32, 16, 16)


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_grid(self, device):
        """Test that empty grid has all fluid cells."""
        grid = HyperGrid(nx=8, ny=8, nz=8, lx=1.0, ly=1.0, lz=1.0, device=device)
        assert (grid.vol_frac == 1.0).all()

    def test_fully_solid_box(self, device):
        """Test grid that is entirely solid."""
        grid = HyperGrid(nx=8, ny=8, nz=8, lx=1.0, ly=1.0, lz=1.0, device=device)
        grid.add_box(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        
        # Should have many solid cells
        solid_frac = (grid.vol_frac < 0.5).float().mean()
        assert solid_frac > 0.8

    def test_overlapping_obstacles(self, small_grid):
        """Test that overlapping obstacles work correctly."""
        small_grid.add_box(1.0, 2.5, 1.0, 2.5, 1.0, 2.5)
        n1 = (small_grid.vol_frac < 0.5).sum()
        
        # Add overlapping box
        small_grid.add_box(2.0, 3.5, 2.0, 3.5, 2.0, 3.5)
        n2 = (small_grid.vol_frac < 0.5).sum()
        
        # Should have more solid cells
        assert n2 > n1

    def test_very_small_obstacle(self, device):
        """Test obstacle smaller than grid cell."""
        grid = HyperGrid(nx=8, ny=8, nz=8, lx=8.0, ly=8.0, lz=8.0, device=device)
        # Cell size is 1.0, obstacle is 0.1
        grid.add_sphere(center=(4.0, 4.0, 4.0), radius=0.05)
        
        # Anti-aliasing should still create some partial cells
        # but may not create fully solid cells

    def test_obstacle_at_boundary(self, small_grid):
        """Test obstacle touching domain boundary."""
        small_grid.add_box(0.0, 1.0, 0.0, 4.0, 0.0, 4.0)
        
        # Should have solid cells at boundary
        assert (small_grid.vol_frac[:4, :, :] < 0.5).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
