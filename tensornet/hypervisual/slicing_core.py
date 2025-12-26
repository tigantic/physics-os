"""
Morton-Aware Slicing Integration
================================

The high-performance slicing layer that uses true Morton projection
instead of point sampling.

Key Difference from SliceEngine:
    SliceEngine: O(N² × d × r²) - samples each point individually
    MortonSlicer: O(L × r²) + O(N²) - projects at QTT level, then contracts

This module provides:
    - MortonSlicer: Drop-in replacement for SliceEngine with Morton projection
    - Integration with Field class from tensornet.substrate
    - Compatibility with existing rendering pipeline

Author: HyperTensor Team
"""

from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any, Union
from enum import Enum
import time

from tensornet.substrate.morton_ops import (
    SlicedQTT2D,
    slice_morton_3d_z_plane,
    slice_morton_3d_y_plane,
    slice_morton_3d_x_plane,
    contract_2d_qtt,
    get_bit,
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class SlicePlane(Enum):
    """Standard slice planes for Morton slicing."""
    XY = 'xy'  # Fix Z
    XZ = 'xz'  # Fix Y
    YZ = 'yz'  # Fix X


@dataclass
class MortonSliceResult:
    """Result of Morton-aware slice operation."""
    data: torch.Tensor         # 2D slice data (N, N)
    plane: SlicePlane          # Which plane was extracted
    slice_index: int           # Index of fixed coordinate
    slice_depth: float         # Depth in normalized [0,1] coordinates
    resolution: int            # N = 2^bits_per_dim
    
    # Performance metrics
    projection_time_ms: float = 0.0   # Time for core projection
    contraction_time_ms: float = 0.0  # Time for 2D contraction
    total_time_ms: float = 0.0
    
    # Field statistics
    min_val: float = 0.0
    max_val: float = 1.0
    mean_val: float = 0.5
    
    def normalize(self) -> torch.Tensor:
        """Normalize data to [0, 1] range."""
        if self.max_val - self.min_val < 1e-10:
            return torch.zeros_like(self.data)
        return (self.data - self.min_val) / (self.max_val - self.min_val)
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return self.data.cpu().numpy()


# =============================================================================
# MORTON SLICER
# =============================================================================

class MortonSlicer:
    """
    True Morton-aware slicer for 3D QTT fields.
    
    Instead of sampling N² points individually (O(N² × d × r²)),
    this extracts a 2D QTT by projection (O(L × r²)) then contracts
    the 2D result (O(N²)).
    
    The key insight: fixing one coordinate at each core level
    reduces physical dimension from 8 → 4, giving a 2D QTT.
    
    Usage:
        slicer = MortonSlicer(field)
        result = slicer.slice_z(z_index=128)  # XY plane at Z=128
        result = slicer.slice_y(y_index=64)   # XZ plane at Y=64
        result = slicer.slice_x(x_index=256)  # YZ plane at X=256
        
        # Normalized depth (0 to 1)
        result = slicer.slice_z_normalized(z_depth=0.5)  # Middle Z plane
    """
    
    def __init__(self, field: 'Field'):
        """
        Initialize Morton slicer from a Field.
        
        Args:
            field: A 3D Field from tensornet.substrate
        """
        self.field = field
        self.cores = field.cores
        self.bits_per_dim = field.bits_per_dim
        self.dims = field.dims
        self.device = field.device
        
        if self.dims != 3:
            raise ValueError(f"MortonSlicer requires 3D field, got {self.dims}D")
        
        # Validate core structure
        self._validate_cores()
        
        # Cache for repeated slices
        self._slice_cache: Dict[Tuple[str, int], MortonSliceResult] = {}
        self.cache_enabled = True
        self.max_cache_size = 32
    
    def _validate_cores(self):
        """Ensure cores have correct Morton 3D structure."""
        for k, core in enumerate(self.cores):
            if core.dim() != 3:
                raise ValueError(f"Core {k} has {core.dim()} dims, expected 3")
            if core.shape[1] != 8:
                raise ValueError(
                    f"Core {k} has physical dim {core.shape[1]}, expected 8 for 3D Morton"
                )
    
    def slice_z(self, z_index: int) -> MortonSliceResult:
        """
        Extract XY plane at fixed Z index.
        
        Args:
            z_index: Z coordinate (0 to 2^bits_per_dim - 1)
            
        Returns:
            MortonSliceResult with (N, N) slice data
        """
        return self._slice('z', z_index)
    
    def slice_y(self, y_index: int) -> MortonSliceResult:
        """
        Extract XZ plane at fixed Y index.
        
        Args:
            y_index: Y coordinate (0 to 2^bits_per_dim - 1)
            
        Returns:
            MortonSliceResult with (N, N) slice data
        """
        return self._slice('y', y_index)
    
    def slice_x(self, x_index: int) -> MortonSliceResult:
        """
        Extract YZ plane at fixed X index.
        
        Args:
            x_index: X coordinate (0 to 2^bits_per_dim - 1)
            
        Returns:
            MortonSliceResult with (N, N) slice data
        """
        return self._slice('x', x_index)
    
    def slice_z_normalized(self, z_depth: float) -> MortonSliceResult:
        """
        Extract XY plane at normalized Z depth.
        
        Args:
            z_depth: Depth in [0, 1] range
            
        Returns:
            MortonSliceResult
        """
        N = 2 ** self.bits_per_dim
        z_index = int(z_depth * (N - 1))
        return self.slice_z(z_index)
    
    def slice_y_normalized(self, y_depth: float) -> MortonSliceResult:
        """Extract XZ plane at normalized Y depth."""
        N = 2 ** self.bits_per_dim
        y_index = int(y_depth * (N - 1))
        return self.slice_y(y_index)
    
    def slice_x_normalized(self, x_depth: float) -> MortonSliceResult:
        """Extract YZ plane at normalized X depth."""
        N = 2 ** self.bits_per_dim
        x_index = int(x_depth * (N - 1))
        return self.slice_x(x_index)
    
    def _slice(self, axis: str, index: int) -> MortonSliceResult:
        """
        Internal slice implementation with caching.
        """
        cache_key = (axis, index)
        
        # Check cache
        if self.cache_enabled and cache_key in self._slice_cache:
            return self._slice_cache[cache_key]
        
        N = 2 ** self.bits_per_dim
        
        # Validate index
        if not 0 <= index < N:
            raise ValueError(f"{axis.upper()} index {index} out of range [0, {N})")
        
        # Time projection
        t0 = time.perf_counter()
        
        if axis == 'z':
            sliced = slice_morton_3d_z_plane(self.cores, index, self.bits_per_dim)
            plane = SlicePlane.XY
        elif axis == 'y':
            sliced = slice_morton_3d_y_plane(self.cores, index, self.bits_per_dim)
            plane = SlicePlane.XZ
        elif axis == 'x':
            sliced = slice_morton_3d_x_plane(self.cores, index, self.bits_per_dim)
            plane = SlicePlane.YZ
        else:
            raise ValueError(f"Unknown axis: {axis}")
        
        t1 = time.perf_counter()
        projection_ms = (t1 - t0) * 1000
        
        # Contract to dense 2D
        dense_2d = sliced.to_dense()
        
        t2 = time.perf_counter()
        contraction_ms = (t2 - t1) * 1000
        
        # Compute statistics
        min_val = dense_2d.min().item()
        max_val = dense_2d.max().item()
        mean_val = dense_2d.mean().item()
        
        result = MortonSliceResult(
            data=dense_2d,
            plane=plane,
            slice_index=index,
            slice_depth=index / (N - 1) if N > 1 else 0.0,
            resolution=N,
            projection_time_ms=projection_ms,
            contraction_time_ms=contraction_ms,
            total_time_ms=projection_ms + contraction_ms,
            min_val=min_val,
            max_val=max_val,
            mean_val=mean_val,
        )
        
        # Update cache
        if self.cache_enabled:
            if len(self._slice_cache) >= self.max_cache_size:
                # Evict oldest
                oldest = next(iter(self._slice_cache))
                del self._slice_cache[oldest]
            self._slice_cache[cache_key] = result
        
        return result
    
    def clear_cache(self):
        """Clear slice cache."""
        self._slice_cache.clear()
    
    def profile(self) -> Dict[str, Any]:
        """
        Profile slicing performance.
        
        Returns timing breakdown for each operation phase.
        """
        N = 2 ** self.bits_per_dim
        mid = N // 2
        
        # Warm up
        _ = self.slice_z(mid)
        self.clear_cache()
        
        # Profile each axis
        results = {}
        
        for axis in ['z', 'y', 'x']:
            times = []
            for _ in range(5):
                self.clear_cache()
                t0 = time.perf_counter()
                _ = self._slice(axis, mid)
                times.append((time.perf_counter() - t0) * 1000)
            
            results[f'{axis}_slice_ms'] = sum(times) / len(times)
        
        results['resolution'] = N
        results['bits_per_dim'] = self.bits_per_dim
        results['n_cores'] = len(self.cores)
        results['max_rank'] = max(c.shape[0] for c in self.cores)
        
        return results


# =============================================================================
# COMPARISON UTILITIES
# =============================================================================

def compare_slicing_methods(
    field: 'Field',
    z_index: int = None
) -> Dict[str, Any]:
    """
    Compare Morton projection vs point sampling performance.
    
    Args:
        field: 3D Field to slice
        z_index: Z index for comparison (default: middle)
        
    Returns:
        Dict with timing and accuracy comparison
    """
    from tensornet.hypervisual.slicer import SliceEngine
    from tensornet.substrate import SliceSpec
    
    N = 2 ** field.bits_per_dim
    if z_index is None:
        z_index = N // 2
    
    z_depth = z_index / (N - 1)
    
    results = {}
    
    # Morton projection method
    morton_slicer = MortonSlicer(field)
    morton_slicer.cache_enabled = False
    
    t0 = time.perf_counter()
    morton_result = morton_slicer.slice_z(z_index)
    morton_time = (time.perf_counter() - t0) * 1000
    
    results['morton_time_ms'] = morton_time
    results['morton_projection_ms'] = morton_result.projection_time_ms
    results['morton_contraction_ms'] = morton_result.contraction_time_ms
    
    # Point sampling method (original SliceEngine)
    slice_engine = SliceEngine(field, device=field.device)
    spec = SliceSpec(
        plane='xy',
        z_coord=z_depth,
        resolution=(N, N)
    )
    
    t0 = time.perf_counter()
    sample_result = slice_engine.slice(spec)
    sample_time = (time.perf_counter() - t0) * 1000
    
    results['sample_time_ms'] = sample_time
    
    # Speedup
    results['speedup'] = sample_time / morton_time if morton_time > 0 else float('inf')
    
    # Accuracy comparison
    morton_np = morton_result.to_numpy()
    sample_np = sample_result.data
    
    diff = np.abs(morton_np - sample_np)
    results['max_diff'] = diff.max()
    results['mean_diff'] = diff.mean()
    results['match'] = diff.max() < 1e-5
    
    results['resolution'] = N
    results['z_index'] = z_index
    
    return results


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'SlicePlane',
    'MortonSliceResult',
    'MortonSlicer',
    'compare_slicing_methods',
]
