"""
STL Voxelizer for HyperFOAM

Converts CAD geometry (STL files) into HyperGrid tensor format.
Uses ray tracing to determine inside/outside classification,
then updates volume and area fractions accordingly.

This is the bridge between the Architect's CAD model and our solver.

Dependencies:
    pip install trimesh rtree

Usage:
    vox = Voxelizer(grid)
    vox.load_stl("table.stl", translation=(4.5, 1.5, 0.0))
"""

import torch
import numpy as np
import time

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("WARNING: trimesh not installed. Run: pip install trimesh rtree")

from hyper_grid import HyperGrid


class Voxelizer:
    """
    Converts STL meshes to HyperGrid volume/area fractions.
    
    Uses Trimesh's ray-based inside/outside query which is
    accelerated by Embree/Rtree when available.
    """
    
    def __init__(self, grid: HyperGrid):
        self.grid = grid
        
    def load_stl(self, stl_path: str, scale: float = 1.0, 
                 translation: tuple = (0, 0, 0)) -> None:
        """
        Loads an STL, voxelizes it, and updates the HyperGrid.
        
        Args:
            stl_path: Path to STL file
            scale: Scale factor for the mesh
            translation: (x, y, z) offset to position mesh in room
        """
        if not TRIMESH_AVAILABLE:
            raise ImportError("trimesh is required. Run: pip install trimesh rtree")
        
        print(f"Loading mesh: {stl_path}")
        mesh = trimesh.load(stl_path)
        
        # 1. Apply Transform (Scale/Move mesh to fit in room)
        matrix = np.eye(4)
        matrix[:3, :3] *= scale
        matrix[:3, 3] = translation
        mesh.apply_transform(matrix)
        print(f"  Mesh Bounds: {mesh.bounds[0]} to {mesh.bounds[1]}")
        print(f"  Mesh Volume: {mesh.volume:.4f} m³")
        
        # 2. Create Query Points (Cell Centers)
        print("Generating query points...")
        
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        
        # Cell center coordinates
        x = np.linspace(self.grid.dx/2, self.grid.lx - self.grid.dx/2, nx)
        y = np.linspace(self.grid.dy/2, self.grid.ly - self.grid.dy/2, ny)
        z = np.linspace(self.grid.dz/2, self.grid.lz - self.grid.dz/2, nz)
        
        # Meshgrid (Index order 'ij' matches PyTorch tensor layout)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
        
        # 3. Ray Query (Inside/Outside check)
        print(f"Voxelizing {len(points):,} cells...")
        start = time.time()
        
        # Contains check: Returns bool array [N]
        # Trimesh uses Embree/Rtree for fast ray tracing
        is_inside = mesh.contains(points)
        
        elapsed = time.time() - start
        n_solid = is_inside.sum()
        print(f"  Voxelization complete in {elapsed:.2f}s")
        print(f"  Solid cells: {n_solid:,} ({100*n_solid/len(points):.1f}%)")
        
        # 4. Update HyperGrid Tensors
        # Reshape result back to [Nx, Ny, Nz]
        mask = torch.from_numpy(is_inside.reshape(nx, ny, nz)).to(self.grid.device)
        
        # Solid cells have vol_frac = 0.0
        # We perform a logical AND with existing geometry (allows multiple STLs)
        current_vol = self.grid.geo[0]
        self.grid.geo[0] = torch.where(
            mask, 
            torch.tensor(0.0, device=self.grid.device), 
            current_vol
        )
        
        # 5. Update Face Areas (Stair-stepping approximation)
        # If a cell is solid, its faces are blocked.
        # True fractional area would compute exact surface cuts,
        # but stair-stepping is sufficient for Phase 1.2.
        
        solid_mask_float = mask.float()
        
        # Block flow entering/leaving solid cells
        self.grid.geo[1] *= (1.0 - solid_mask_float)  # Area X
        self.grid.geo[2] *= (1.0 - solid_mask_float)  # Area Y
        self.grid.geo[3] *= (1.0 - solid_mask_float)  # Area Z
        
        print("✓ HyperGrid updated.")
    
    def load_primitive_box(self, x_min: float, x_max: float,
                           y_min: float, y_max: float,
                           z_min: float, z_max: float) -> None:
        """
        Faster path for box obstacles without STL.
        Directly updates grid without mesh operations.
        """
        # Convert physical coords to grid indices
        i_min = int(x_min / self.grid.dx)
        i_max = int(x_max / self.grid.dx)
        j_min = int(y_min / self.grid.dy)
        j_max = int(y_max / self.grid.dy)
        k_min = int(z_min / self.grid.dz)
        k_max = int(z_max / self.grid.dz)
        
        # Clamp to grid bounds
        i_min, i_max = max(0, i_min), min(self.grid.nx, i_max)
        j_min, j_max = max(0, j_min), min(self.grid.ny, j_max)
        k_min, k_max = max(0, k_min), min(self.grid.nz, k_max)
        
        # Block Volume
        self.grid.geo[0, i_min:i_max, j_min:j_max, k_min:k_max] = 0.0
        
        # Block Faces (extended by 1 to catch boundaries)
        self.grid.geo[1, i_min:i_max+1, j_min:j_max, k_min:k_max] = 0.0
        self.grid.geo[2, i_min:i_max, j_min:j_max+1, k_min:k_max] = 0.0
        self.grid.geo[3, i_min:i_max, j_min:j_max, k_min:k_max+1] = 0.0
        
        print(f"✓ Box obstacle: [{x_min}:{x_max}, {y_min}:{y_max}, {z_min}:{z_max}]")


def create_test_stl(filename: str = "test_sphere.stl") -> None:
    """Creates a simple sphere STL for testing."""
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh is required")
    
    mesh = trimesh.creation.icosphere(radius=0.5, subdivisions=2)
    mesh.export(filename)
    print(f"✓ Created test STL: {filename}")


# =============================================================================
# Test Harness
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("VOXELIZER TEST")
    print("=" * 70)
    
    if not TRIMESH_AVAILABLE:
        print("\nERROR: trimesh not installed!")
        print("Run: pip install trimesh rtree")
        exit(1)
    
    # Create a test STL (sphere)
    create_test_stl("test_sphere.stl")
    
    # Init Grid (Nielsen room)
    print("\nInitializing HyperGrid...")
    grid = HyperGrid(
        nx=128, ny=64, nz=64,
        lx=9.0, ly=3.0, lz=3.0,
        device='cuda'
    )
    
    print(f"Grid: {grid.nx}×{grid.ny}×{grid.nz}")
    print(f"Room: {grid.lx}×{grid.ly}×{grid.lz} m")
    
    # Run Voxelizer
    print("\nVoxelizing sphere at room center...")
    vox = Voxelizer(grid)
    vox.load_stl(
        "test_sphere.stl", 
        scale=1.0,
        translation=(4.5, 1.5, 1.5)  # Center of room
    )
    
    # Verification
    print("\n" + "=" * 50)
    print("VERIFICATION")
    print("=" * 50)
    
    # Check center (should be solid)
    cx = int(4.5 / grid.dx)
    cy = int(1.5 / grid.dy)
    cz = int(1.5 / grid.dz)
    center_vol = grid.geo[0, cx, cy, cz].item()
    print(f"Sphere center vol_frac: {center_vol:.2f} (expected: 0.0)")
    
    # Check corner (should be fluid)
    corner_vol = grid.geo[0, 0, 0, 0].item()
    print(f"Room corner vol_frac:   {corner_vol:.2f} (expected: 1.0)")
    
    # Count solid cells
    solid_cells = (grid.geo[0] < 0.5).sum().item()
    total_cells = grid.nx * grid.ny * grid.nz
    print(f"Solid cells: {solid_cells:,} / {total_cells:,} ({100*solid_cells/total_cells:.2f}%)")
    
    # Test passed?
    if center_vol < 0.5 and corner_vol > 0.5:
        print("\n✓ VOXELIZER TEST PASSED")
    else:
        print("\n✗ VOXELIZER TEST FAILED")
    
    # Cleanup
    import os
    if os.path.exists("test_sphere.stl"):
        os.remove("test_sphere.stl")
        print("\n(Cleaned up test_sphere.stl)")
