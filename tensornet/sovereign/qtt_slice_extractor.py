"""
QTT 3D → 2D Slice Extractor: GPU-Accelerated Phase 4 Pipeline

Extracts 2D slices from 3D QTT-compressed fields for real-time visualization.

Architecture:
    Input:  3D QTT field (512³, rank ≤ 32, ~10KB)
    Output: Dense 2D slice (1920×1080, ~8MB)
    
Pipeline:
    1. Generate 2D coordinate grid (GPU)
    2. Embed in 3D with fixed z-index
    3. Morton encode (x,y,z) → 1D indices (GPU)
    4. Evaluate QTT at these indices (GPU kernel)
    5. Reshape to [H, W]

Performance Target:
    512³ → 1920×1080 in <5ms on RTX 5070

Key Innovation:
    Morton-order aware indexing enables O(H×W×L×r²) complexity,
    avoiding full O(N³) decompression.

Author: HyperTensor Team  
Date: December 28, 2025
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
import time


@dataclass
class QTT3DState:
    """3D field in QTT format with Morton ordering."""
    cores: List[torch.Tensor]  # List of (r_left, 8, r_right) cores
    qubits_per_dim: int        # Bits per spatial dimension
    device: torch.device
    
    @property
    def n_cores(self) -> int:
        return len(self.cores)
    
    @property
    def grid_size(self) -> int:
        """Resolution per dimension: 2^qubits_per_dim"""
        return 2 ** self.qubits_per_dim
    
    @property
    def max_rank(self) -> int:
        return max(c.shape[0] for c in self.cores)


class QTTSliceExtractor:
    """
    Extract 2D slices from 3D QTT fields with GPU acceleration.
    
    Supports XY, XZ, and YZ plane extraction at arbitrary indices.
    Uses Morton encoding and GPU kernel for fast evaluation.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize slice extractor.
        
        Args:
            device: GPU device (defaults to CUDA if available)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not torch.cuda.is_available():
            raise RuntimeError("QTT slice extraction requires CUDA GPU")
    
    def morton_encode_3d_gpu(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        n_bits: int
    ) -> torch.Tensor:
        """
        Morton encode 3D coordinates on GPU.
        
        Interleaves bits as: x0,y0,z0, x1,y1,z1, ...
        Uses magic number bit-interleaving for O(1) complexity.
        
        Args:
            x, y, z: Integer coordinate tensors (same shape)
            n_bits: Number of bits per dimension (max 10 for 32-bit)
            
        Returns:
            Morton indices (same shape as inputs)
        """
        # L-002 FIX: Parallel bit-interleave using magic constants
        # This replaces O(n_bits) loop with O(1) bit operations
        def spread_bits_3d(v: torch.Tensor) -> torch.Tensor:
            """Spread bits for 3D Morton: insert 2 zeros between each bit."""
            v = v.long()
            # For 10-bit coordinates (up to 1024), we need these masks
            v = (v | (v << 16)) & 0x030000FF
            v = (v | (v << 8))  & 0x0300F00F
            v = (v | (v << 4))  & 0x030C30C3
            v = (v | (v << 2))  & 0x09249249
            return v
        
        return spread_bits_3d(x) | (spread_bits_3d(y) << 1) | (spread_bits_3d(z) << 2)
    
    def morton_decode_3d_gpu(
        self,
        morton_idx: torch.Tensor,
        n_bits: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode Morton indices to (x, y, z) coordinates.
        
        Inverse of morton_encode_3d_gpu using magic number extraction.
        
        Args:
            morton_idx: Morton indices
            n_bits: Number of bits per dimension (max 10 for 32-bit)
            
        Returns:
            (x, y, z) coordinate tensors
        """
        # L-003 FIX: Parallel bit-extraction using magic constants
        # This replaces O(n_bits) loop with O(1) bit operations
        def compact_bits_3d(v: torch.Tensor) -> torch.Tensor:
            """Extract every 3rd bit (inverse of spread_bits_3d)."""
            v = v.long()
            v = v & 0x09249249
            v = (v | (v >> 2))  & 0x030C30C3
            v = (v | (v >> 4))  & 0x0300F00F
            v = (v | (v >> 8))  & 0x030000FF
            v = (v | (v >> 16)) & 0x000003FF
            return v
        
        x = compact_bits_3d(morton_idx)
        y = compact_bits_3d(morton_idx >> 1)
        z = compact_bits_3d(morton_idx >> 2)
        
        return x, y, z
    
    def eval_qtt_at_indices_gpu(
        self,
        qtt: QTT3DState,
        morton_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate QTT at specific Morton indices using GPU.
        
        This is the performance-critical kernel. We contract QTT cores
        along the path defined by each Morton index's bit pattern.
        
        Args:
            qtt: 3D QTT state
            morton_indices: [N] Morton indices to evaluate
            
        Returns:
            [N] QTT values at specified indices
        """
        n_points = morton_indices.numel()
        n_cores = qtt.n_cores
        
        # Initialize with identity (rank 1)
        # result[i] starts as [1] vector
        result = torch.ones((n_points, 1), dtype=torch.float32, device=self.device)
        
        # L-020 NOTE: Sequential TT contraction inherent to algorithm - cannot parallelize
        # across cores (each core depends on prior result). Already batched over points.
        # Contract cores left-to-right
        for core_idx in range(n_cores):
            core = qtt.cores[core_idx]  # [r_left, 8, r_right]
            r_left, phys_dim, r_right = core.shape
            
            # Extract physical indices for this core from Morton bits
            # MSB-first: core 0 handles most significant bits
            bit_pos = n_cores - 1 - core_idx
            
            # Extract x, y, z bits at this position
            x_bit = (morton_indices >> (3 * bit_pos)) & 1
            y_bit = (morton_indices >> (3 * bit_pos + 1)) & 1
            z_bit = (morton_indices >> (3 * bit_pos + 2)) & 1
            
            # Physical index = 4*z + 2*y + x (for phys_dim=8)
            phys_idx = 4 * z_bit + 2 * y_bit + x_bit  # [n_points]
            
            # Select appropriate core slice for each point
            # core[phys_idx[i]] gives [r_left, r_right] matrix for point i
            core_slices = core[:, phys_idx, :]  # [r_left, n_points, r_right]
            
            # Matrix-vector product: result[i] @ core_slices[:, i, :]
            # Current result: [n_points, r_left]
            # Need: [n_points, r_right]
            result = torch.einsum('ni,inj->nj', result, core_slices)
        
        # Final contraction should yield scalar per point
        assert result.shape[1] == 1, f"Final rank should be 1, got {result.shape[1]}"
        
        return result.squeeze(1)  # [n_points]
    
    def extract_xy_slice(
        self,
        qtt: QTT3DState,
        z_index: Optional[int] = None,
        output_size: Tuple[int, int] = (1920, 1080),
        return_timings: bool = False
    ) -> Tuple[torch.Tensor, dict]:
        """
        Extract XY plane at fixed Z from 3D QTT.
        
        Args:
            qtt: 3D QTT state to slice
            z_index: Z coordinate (default: middle of domain)
            output_size: (width, height) of output slice
            return_timings: If True, return timing breakdown
            
        Returns:
            slice_tensor: [height, width] tensor on GPU
            timings: Dict with timing info (if return_timings=True)
        """
        if z_index is None:
            z_index = qtt.grid_size // 2  # Middle of domain
        
        assert 0 <= z_index < qtt.grid_size, \
            f"z_index {z_index} out of range [0, {qtt.grid_size})"
        
        width, height = output_size
        timings = {}
        
        # Step 1: Generate 2D coordinate grid
        start_grid = time.perf_counter()
        
        # Map [0, width) to [0, grid_size)
        x_coords = torch.linspace(0, qtt.grid_size - 1, width, device=self.device).long()
        y_coords = torch.linspace(0, qtt.grid_size - 1, height, device=self.device).long()
        
        # Create meshgrid
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='xy')
        z_grid = torch.full_like(x_grid, z_index)
        
        # Flatten for batch processing
        x_flat = x_grid.flatten()
        y_flat = y_grid.flatten()
        z_flat = z_grid.flatten()
        
        torch.cuda.synchronize()
        end_grid = time.perf_counter()
        timings['grid_gen_ms'] = (end_grid - start_grid) * 1000
        
        # Step 2: Morton encode
        start_morton = time.perf_counter()
        
        morton_indices = self.morton_encode_3d_gpu(
            x_flat, y_flat, z_flat, qtt.qubits_per_dim
        )
        
        torch.cuda.synchronize()
        end_morton = time.perf_counter()
        timings['morton_encode_ms'] = (end_morton - start_morton) * 1000
        
        # Step 3: Evaluate QTT
        start_eval = time.perf_counter()
        
        values = self.eval_qtt_at_indices_gpu(qtt, morton_indices)
        
        torch.cuda.synchronize()
        end_eval = time.perf_counter()
        timings['qtt_eval_ms'] = (end_eval - start_eval) * 1000
        
        # Step 4: Reshape to 2D
        slice_tensor = values.view(height, width)
        
        timings['total_ms'] = timings['grid_gen_ms'] + timings['morton_encode_ms'] + timings['qtt_eval_ms']
        
        if return_timings:
            return slice_tensor, timings
        else:
            return slice_tensor, {}
    
    def extract_xz_slice(
        self,
        qtt: QTT3DState,
        y_index: Optional[int] = None,
        output_size: Tuple[int, int] = (1920, 1080),
        return_timings: bool = False
    ) -> Tuple[torch.Tensor, dict]:
        """
        Extract XZ plane at fixed Y from 3D QTT.
        
        Similar to extract_xy_slice but fixes Y coordinate.
        """
        if y_index is None:
            y_index = qtt.grid_size // 2
        
        assert 0 <= y_index < qtt.grid_size
        
        width, height = output_size
        timings = {}
        
        start_grid = time.perf_counter()
        
        x_coords = torch.linspace(0, qtt.grid_size - 1, width, device=self.device).long()
        z_coords = torch.linspace(0, qtt.grid_size - 1, height, device=self.device).long()
        
        x_grid, z_grid = torch.meshgrid(x_coords, z_coords, indexing='xy')
        y_grid = torch.full_like(x_grid, y_index)
        
        x_flat = x_grid.flatten()
        y_flat = y_grid.flatten()
        z_flat = z_grid.flatten()
        
        torch.cuda.synchronize()
        end_grid = time.perf_counter()
        timings['grid_gen_ms'] = (end_grid - start_grid) * 1000
        
        start_morton = time.perf_counter()
        morton_indices = self.morton_encode_3d_gpu(x_flat, y_flat, z_flat, qtt.qubits_per_dim)
        torch.cuda.synchronize()
        end_morton = time.perf_counter()
        timings['morton_encode_ms'] = (end_morton - start_morton) * 1000
        
        start_eval = time.perf_counter()
        values = self.eval_qtt_at_indices_gpu(qtt, morton_indices)
        torch.cuda.synchronize()
        end_eval = time.perf_counter()
        timings['qtt_eval_ms'] = (end_eval - start_eval) * 1000
        
        slice_tensor = values.view(height, width)
        timings['total_ms'] = sum(timings.values())
        
        return slice_tensor, timings if return_timings else {}
    
    def extract_yz_slice(
        self,
        qtt: QTT3DState,
        x_index: Optional[int] = None,
        output_size: Tuple[int, int] = (1920, 1080),
        return_timings: bool = False
    ) -> Tuple[torch.Tensor, dict]:
        """
        Extract YZ plane at fixed X from 3D QTT.
        
        Similar to extract_xy_slice but fixes X coordinate.
        """
        if x_index is None:
            x_index = qtt.grid_size // 2
        
        assert 0 <= x_index < qtt.grid_size
        
        width, height = output_size
        timings = {}
        
        start_grid = time.perf_counter()
        
        y_coords = torch.linspace(0, qtt.grid_size - 1, width, device=self.device).long()
        z_coords = torch.linspace(0, qtt.grid_size - 1, height, device=self.device).long()
        
        y_grid, z_grid = torch.meshgrid(y_coords, z_coords, indexing='xy')
        x_grid = torch.full_like(y_grid, x_index)
        
        x_flat = x_grid.flatten()
        y_flat = y_grid.flatten()
        z_flat = z_grid.flatten()
        
        torch.cuda.synchronize()
        end_grid = time.perf_counter()
        timings['grid_gen_ms'] = (end_grid - start_grid) * 1000
        
        start_morton = time.perf_counter()
        morton_indices = self.morton_encode_3d_gpu(x_flat, y_flat, z_flat, qtt.qubits_per_dim)
        torch.cuda.synchronize()
        end_morton = time.perf_counter()
        timings['morton_encode_ms'] = (end_morton - start_morton) * 1000
        
        start_eval = time.perf_counter()
        values = self.eval_qtt_at_indices_gpu(qtt, morton_indices)
        torch.cuda.synchronize()
        end_eval = time.perf_counter()
        timings['qtt_eval_ms'] = (end_eval - start_eval) * 1000
        
        slice_tensor = values.view(height, width)
        timings['total_ms'] = sum(timings.values())
        
        return slice_tensor, timings if return_timings else {}


def test_slice_extractor():
    """Test QTT slice extraction with synthetic 3D field."""
    import torch
    from tensornet.cfd.pure_qtt_ops import dense_to_qtt
    
    print("Testing QTT Slice Extractor")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create synthetic 3D field
    qubits_per_dim = 6  # 64³ grid
    grid_size = 2 ** qubits_per_dim
    print(f"Grid size: {grid_size}³ = {grid_size**3:,} points")
    
    # Create 3D Gaussian bump
    x = torch.linspace(-1, 1, grid_size, device=device)
    y = torch.linspace(-1, 1, grid_size, device=device)
    z = torch.linspace(-1, 1, grid_size, device=device)
    
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    # 3D Gaussian: exp(-r²)
    R2 = X**2 + Y**2 + Z**2
    field_3d = torch.exp(-5 * R2)
    
    print(f"Field range: [{field_3d.min():.3f}, {field_3d.max():.3f}]")
    
    # Compress to QTT (using 2D function, need to adapt for 3D)
    # For now, just create fake QTT cores
    n_cores = 3 * qubits_per_dim  # Total qubits
    cores = []
    rank = 4
    
    # L-020 NOTE: Sequential TT core construction - inherent to TT format
    for i in range(n_cores):
        r_left = 1 if i == 0 else rank
        r_right = 1 if i == n_cores - 1 else rank
        core = torch.randn(r_left, 8, r_right, device=device)
        cores.append(core)
    
    qtt = QTT3DState(cores=cores, qubits_per_dim=qubits_per_dim, device=device)
    
    print(f"QTT cores: {len(cores)}, max rank: {qtt.max_rank}")
    
    # Test slice extraction
    extractor = QTTSliceExtractor(device=device)
    
    # Extract XY slice at middle Z
    z_idx = grid_size // 2
    print(f"\nExtracting XY slice at z={z_idx}")
    
    slice_tensor, timings = extractor.extract_xy_slice(
        qtt, z_index=z_idx, output_size=(512, 512), return_timings=True
    )
    
    print(f"Output shape: {slice_tensor.shape}")
    print(f"Output range: [{slice_tensor.min():.3f}, {slice_tensor.max():.3f}]")
    print(f"\nTiming breakdown:")
    for key, val in timings.items():
        print(f"  {key:20s}: {val:6.2f} ms")
    
    # Test all three orientations
    print("\n" + "=" * 60)
    print("Testing all slice orientations")
    print("=" * 60)
    
    for plane, method in [('xy', extractor.extract_xy_slice),
                          ('xz', extractor.extract_xz_slice),
                          ('yz', extractor.extract_yz_slice)]:
        slice_t, t = method(qtt, output_size=(1920, 1080), return_timings=True)
        print(f"\n{plane.upper()} plane @ 1920×1080:")
        print(f"  Total time: {t['total_ms']:.2f} ms")
        print(f"  Shape: {slice_t.shape}")
        print(f"  Range: [{slice_t.min():.3f}, {slice_t.max():.3f}]")
    
    print("\n" + "=" * 60)
    print("✓ QTT Slice Extractor test complete")


if __name__ == '__main__':
    test_slice_extractor()
