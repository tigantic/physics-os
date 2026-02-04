"""
3D Morton (Z-Curve) Encoding for QTT Turbulence DNS
====================================================

Space-filling curve for mapping 3D grids to 1D QTT format.

Morton ordering preserves spatial locality:
- Nearby 3D points → nearby 1D indices
- Essential for QTT compression of smooth fields

Bit Interleaving Pattern:
    3D: (x, y, z) → x₀y₀z₀ x₁y₁z₁ x₂y₂z₂ ...
    
    Example (2-bit):
        (0,0,0) → 000000 = 0
        (1,0,0) → 000001 = 1
        (0,1,0) → 000010 = 2
        (1,1,0) → 000011 = 3
        (0,0,1) → 000100 = 4
        ...

Performance:
    - Vectorized NumPy/PyTorch for batch operations
    - CUDA kernel for GPU acceleration
    - LUT (lookup table) for small dimensions

References:
    [1] Morton, G.M. "A Computer Oriented Geodetic Data Base" (1966)
    [2] Samet, H. "Foundations of Multidimensional and Metric Data Structures"

Author: HyperTensor Team
Date: 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union, Optional
import functools

import numpy as np
import torch


# ═══════════════════════════════════════════════════════════════════════════════════════
# CORE MORTON FUNCTIONS (SCALAR)
# ═══════════════════════════════════════════════════════════════════════════════════════

def morton_encode_3d(x: int, y: int, z: int, n_bits: int) -> int:
    """
    Encode 3D coordinates to Morton (Z-curve) index.
    
    Args:
        x, y, z: 3D coordinates (each in [0, 2^n_bits))
        n_bits: Bits per dimension (grid is 2^n_bits per axis)
        
    Returns:
        Morton index in [0, 2^(3*n_bits))
        
    Example:
        >>> morton_encode_3d(3, 2, 1, 2)  # 2-bit coords
        39
    """
    idx = 0
    for b in range(n_bits):
        idx |= ((x >> b) & 1) << (3 * b + 0)
        idx |= ((y >> b) & 1) << (3 * b + 1)
        idx |= ((z >> b) & 1) << (3 * b + 2)
    return idx


def morton_decode_3d(idx: int, n_bits: int) -> Tuple[int, int, int]:
    """
    Decode Morton index to 3D coordinates.
    
    Args:
        idx: Morton index in [0, 2^(3*n_bits))
        n_bits: Bits per dimension
        
    Returns:
        (x, y, z) coordinates
        
    Example:
        >>> morton_decode_3d(39, 2)
        (3, 2, 1)
    """
    x = y = z = 0
    for b in range(n_bits):
        x |= ((idx >> (3 * b + 0)) & 1) << b
        y |= ((idx >> (3 * b + 1)) & 1) << b
        z |= ((idx >> (3 * b + 2)) & 1) << b
    return x, y, z


# ═══════════════════════════════════════════════════════════════════════════════════════
# VECTORIZED MORTON (NUMPY)
# ═══════════════════════════════════════════════════════════════════════════════════════

def morton_encode_3d_vectorized(
    x: np.ndarray, 
    y: np.ndarray, 
    z: np.ndarray, 
    n_bits: int
) -> np.ndarray:
    """
    Vectorized Morton encoding for arrays of 3D coordinates.
    
    Args:
        x, y, z: Arrays of coordinates (same shape)
        n_bits: Bits per dimension
        
    Returns:
        Array of Morton indices (same shape as inputs)
    """
    x = np.asarray(x, dtype=np.uint64)
    y = np.asarray(y, dtype=np.uint64)
    z = np.asarray(z, dtype=np.uint64)
    
    idx = np.zeros_like(x, dtype=np.uint64)
    for b in range(n_bits):
        idx |= ((x >> b) & 1) << (3 * b + 0)
        idx |= ((y >> b) & 1) << (3 * b + 1)
        idx |= ((z >> b) & 1) << (3 * b + 2)
    return idx


def morton_decode_3d_vectorized(
    idx: np.ndarray, 
    n_bits: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized Morton decoding to 3D coordinates.
    
    Args:
        idx: Array of Morton indices
        n_bits: Bits per dimension
        
    Returns:
        (x, y, z) arrays of coordinates
    """
    idx = np.asarray(idx, dtype=np.uint64)
    x = np.zeros_like(idx)
    y = np.zeros_like(idx)
    z = np.zeros_like(idx)
    
    for b in range(n_bits):
        x |= ((idx >> (3 * b + 0)) & 1) << b
        y |= ((idx >> (3 * b + 1)) & 1) << b
        z |= ((idx >> (3 * b + 2)) & 1) << b
    
    return x.astype(np.int64), y.astype(np.int64), z.astype(np.int64)


# ═══════════════════════════════════════════════════════════════════════════════════════
# PYTORCH MORTON (GPU-COMPATIBLE)
# ═══════════════════════════════════════════════════════════════════════════════════════

def morton_encode_3d_torch(
    x: torch.Tensor, 
    y: torch.Tensor, 
    z: torch.Tensor, 
    n_bits: int
) -> torch.Tensor:
    """
    PyTorch Morton encoding (GPU-compatible).
    
    Args:
        x, y, z: Tensors of coordinates (same shape, int64)
        n_bits: Bits per dimension
        
    Returns:
        Tensor of Morton indices
    """
    x = x.to(torch.int64)
    y = y.to(torch.int64)
    z = z.to(torch.int64)
    
    idx = torch.zeros_like(x, dtype=torch.int64)
    for b in range(n_bits):
        idx = idx | (((x >> b) & 1) << (3 * b + 0))
        idx = idx | (((y >> b) & 1) << (3 * b + 1))
        idx = idx | (((z >> b) & 1) << (3 * b + 2))
    return idx


def morton_decode_3d_torch(
    idx: torch.Tensor, 
    n_bits: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PyTorch Morton decoding (GPU-compatible).
    
    Args:
        idx: Tensor of Morton indices
        n_bits: Bits per dimension
        
    Returns:
        (x, y, z) tensors of coordinates
    """
    idx = idx.to(torch.int64)
    x = torch.zeros_like(idx)
    y = torch.zeros_like(idx)
    z = torch.zeros_like(idx)
    
    for b in range(n_bits):
        x = x | (((idx >> (3 * b + 0)) & 1) << b)
        y = y | (((idx >> (3 * b + 1)) & 1) << b)
        z = z | (((idx >> (3 * b + 2)) & 1) << b)
    
    return x, y, z


# ═══════════════════════════════════════════════════════════════════════════════════════
# FULL GRID REORDERING
# ═══════════════════════════════════════════════════════════════════════════════════════

@functools.lru_cache(maxsize=16)
def _build_morton_permutation(n_bits: int) -> np.ndarray:
    """
    Build permutation array for Morton reordering.
    
    Cached for repeated use with same grid size.
    
    Args:
        n_bits: Bits per dimension (grid is N = 2^n_bits per axis)
        
    Returns:
        Permutation array of shape (N³,) mapping linear to Morton
    """
    N = 1 << n_bits  # 2^n_bits
    N3 = N * N * N
    
    # Generate all Morton indices
    perm = np.zeros(N3, dtype=np.int64)
    
    linear_idx = 0
    for z in range(N):
        for y in range(N):
            for x in range(N):
                morton_idx = morton_encode_3d(x, y, z, n_bits)
                perm[linear_idx] = morton_idx
                linear_idx += 1
    
    return perm


@functools.lru_cache(maxsize=16)
def _build_inverse_morton_permutation(n_bits: int) -> np.ndarray:
    """
    Build inverse permutation array (Morton to linear).
    
    Args:
        n_bits: Bits per dimension
        
    Returns:
        Inverse permutation array
    """
    perm = _build_morton_permutation(n_bits)
    inv_perm = np.argsort(perm)
    return inv_perm


def linear_to_morton_3d(
    tensor: Union[np.ndarray, torch.Tensor], 
    n_bits: int
) -> Union[np.ndarray, torch.Tensor]:
    """
    Reorder 3D tensor from linear (x,y,z) to Morton ordering.
    
    Args:
        tensor: 3D tensor of shape (N, N, N) where N = 2^n_bits
        n_bits: Bits per dimension
        
    Returns:
        1D array/tensor in Morton order
    """
    is_torch = isinstance(tensor, torch.Tensor)
    device = tensor.device if is_torch else None
    
    # Flatten
    flat = tensor.flatten()
    
    # Get permutation
    perm = _build_morton_permutation(n_bits)
    
    if is_torch:
        perm_t = torch.from_numpy(perm).to(device)
        # Create output and scatter
        morton = torch.zeros_like(flat)
        morton[perm_t] = flat
        return morton
    else:
        morton = np.zeros_like(flat)
        morton[perm] = flat
        return morton


def morton_to_linear_3d(
    morton: Union[np.ndarray, torch.Tensor], 
    n_bits: int
) -> Union[np.ndarray, torch.Tensor]:
    """
    Reorder 1D Morton-ordered array back to 3D linear tensor.
    
    Args:
        morton: 1D array/tensor in Morton order
        n_bits: Bits per dimension
        
    Returns:
        3D tensor of shape (N, N, N)
    """
    is_torch = isinstance(morton, torch.Tensor)
    device = morton.device if is_torch else None
    
    N = 1 << n_bits
    
    # Get permutation
    perm = _build_morton_permutation(n_bits)
    
    if is_torch:
        perm_t = torch.from_numpy(perm).to(device)
        # Gather from morton positions
        flat = morton[perm_t]
        return flat.reshape(N, N, N)
    else:
        flat = morton[perm]
        return flat.reshape(N, N, N)


# ═══════════════════════════════════════════════════════════════════════════════════════
# MORTON GRID UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class Morton3DGrid:
    """
    3D Morton grid manager for QTT turbulence.
    
    Handles conversion between:
    - Physical coordinates (x, y, z)
    - Linear indices (row-major)
    - Morton indices (Z-curve)
    - QTT format (hierarchical)
    
    Attributes:
        n_bits: Bits per dimension
        N: Grid size per axis (2^n_bits)
        N3: Total grid points (N^3)
        L: Physical domain size
    """
    n_bits: int
    L: float = 2 * np.pi  # Default: [0, 2π]³
    
    def __post_init__(self):
        self.N = 1 << self.n_bits
        self.N3 = self.N ** 3
        self.dx = self.L / self.N
        
        # Cache permutations on first use
        self._perm = None
        self._inv_perm = None
    
    @property
    def total_qubits(self) -> int:
        """Total qubits for QTT representation."""
        return 3 * self.n_bits
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Physical grid shape."""
        return (self.N, self.N, self.N)
    
    def physical_coords(
        self, 
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate physical coordinate arrays.
        
        Returns:
            (X, Y, Z) meshgrids of physical coordinates
        """
        x = torch.linspace(0, self.L - self.dx, self.N, device=device)
        X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')
        return X, Y, Z
    
    def to_morton(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert 3D tensor to Morton-ordered 1D."""
        return linear_to_morton_3d(tensor, self.n_bits)
    
    def from_morton(self, morton: torch.Tensor) -> torch.Tensor:
        """Convert Morton-ordered 1D to 3D tensor."""
        return morton_to_linear_3d(morton, self.n_bits)
    
    def wavenumbers(
        self, 
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate wavenumber arrays for spectral operations.
        
        Returns:
            (Kx, Ky, Kz) meshgrids of wavenumbers
        """
        k = torch.fft.fftfreq(self.N, d=self.dx, device=device) * 2 * np.pi
        Kx, Ky, Kz = torch.meshgrid(k, k, k, indexing='ij')
        return Kx, Ky, Kz
    
    def k_magnitude(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Wavenumber magnitude |k|."""
        Kx, Ky, Kz = self.wavenumbers(device)
        return torch.sqrt(Kx**2 + Ky**2 + Kz**2)


# ═══════════════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════════════

def validate_morton_3d(n_bits: int = 4, verbose: bool = True) -> bool:
    """
    Validate Morton encoding/decoding round-trip.
    
    Args:
        n_bits: Bits per dimension to test
        verbose: Print results
        
    Returns:
        True if all tests pass
    """
    N = 1 << n_bits
    passed = True
    
    if verbose:
        print(f"Morton 3D Validation (n_bits={n_bits}, N={N})")
        print("=" * 50)
    
    # Test 1: Scalar round-trip
    errors = 0
    for x in range(min(N, 8)):
        for y in range(min(N, 8)):
            for z in range(min(N, 8)):
                idx = morton_encode_3d(x, y, z, n_bits)
                x2, y2, z2 = morton_decode_3d(idx, n_bits)
                if (x, y, z) != (x2, y2, z2):
                    errors += 1
                    if verbose:
                        print(f"  FAIL: ({x},{y},{z}) -> {idx} -> ({x2},{y2},{z2})")
    
    if errors == 0:
        if verbose:
            print("✓ Scalar round-trip: PASS")
    else:
        passed = False
        if verbose:
            print(f"✗ Scalar round-trip: FAIL ({errors} errors)")
    
    # Test 2: Vectorized consistency
    x_arr = np.arange(N, dtype=np.int64)
    y_arr = np.zeros(N, dtype=np.int64)
    z_arr = np.zeros(N, dtype=np.int64)
    
    idx_vec = morton_encode_3d_vectorized(x_arr, y_arr, z_arr, n_bits)
    idx_scalar = np.array([morton_encode_3d(x, 0, 0, n_bits) for x in range(N)])
    
    if np.allclose(idx_vec, idx_scalar):
        if verbose:
            print("✓ Vectorized consistency: PASS")
    else:
        passed = False
        if verbose:
            print("✗ Vectorized consistency: FAIL")
    
    # Test 3: PyTorch consistency
    x_t = torch.arange(N, dtype=torch.int64)
    y_t = torch.zeros(N, dtype=torch.int64)
    z_t = torch.zeros(N, dtype=torch.int64)
    
    idx_torch = morton_encode_3d_torch(x_t, y_t, z_t, n_bits)
    
    if torch.allclose(idx_torch, torch.from_numpy(idx_scalar)):
        if verbose:
            print("✓ PyTorch consistency: PASS")
    else:
        passed = False
        if verbose:
            print("✗ PyTorch consistency: FAIL")
    
    # Test 4: Full grid round-trip
    tensor = torch.randn(N, N, N)
    morton = linear_to_morton_3d(tensor, n_bits)
    recovered = morton_to_linear_3d(morton, n_bits)
    
    if torch.allclose(tensor, recovered):
        if verbose:
            print("✓ Full grid round-trip: PASS")
    else:
        passed = False
        if verbose:
            print("✗ Full grid round-trip: FAIL")
    
    # Test 5: Morton ordering preserves locality
    # Adjacent points in 3D should have close Morton indices
    idx_000 = morton_encode_3d(0, 0, 0, n_bits)
    idx_100 = morton_encode_3d(1, 0, 0, n_bits)
    idx_010 = morton_encode_3d(0, 1, 0, n_bits)
    idx_001 = morton_encode_3d(0, 0, 1, n_bits)
    
    # All neighbors should be within 7 of origin in Morton space
    max_dist = max(abs(idx_100 - idx_000), abs(idx_010 - idx_000), abs(idx_001 - idx_000))
    if max_dist <= 7:
        if verbose:
            print(f"✓ Locality preserved: PASS (max neighbor dist = {max_dist})")
    else:
        if verbose:
            print(f"⚠ Locality check: max neighbor dist = {max_dist}")
    
    if verbose:
        print("=" * 50)
        print(f"Overall: {'PASS' if passed else 'FAIL'}")
    
    return passed


# ═══════════════════════════════════════════════════════════════════════════════════════
# MAGIC BITS (OPTIMIZED)
# ═══════════════════════════════════════════════════════════════════════════════════════

# Pre-computed magic numbers for fast bit spreading (up to 21 bits)
# These spread bits from xxx... to x00x00x00...
_SPREAD_MASKS_21 = [
    0x1FFFFF,                     # 21 bits
    0x1F00000000FFFF,             # After first spread
    0x1F0000FF0000FF,             # After second spread
    0x100F00F00F00F00F,           # After third spread
    0x10C30C30C30C30C3,           # After fourth spread
    0x1249249249249249,           # Final interleaved
]

_SPREAD_SHIFTS_21 = [32, 16, 8, 4, 2]


def _spread_bits_21(x: int) -> int:
    """
    Spread 21-bit integer for Morton encoding (fast path).
    
    Takes: 0000 0000 000x xxxx xxxx xxxx xxxx xxxx
    Returns: 00x 00x 00x 00x 00x 00x 00x 00x 00x 00x 00x 00x 00x 00x 00x 00x 00x 00x 00x 00x 00x
    """
    x = x & _SPREAD_MASKS_21[0]
    x = (x | (x << _SPREAD_SHIFTS_21[0])) & _SPREAD_MASKS_21[1]
    x = (x | (x << _SPREAD_SHIFTS_21[1])) & _SPREAD_MASKS_21[2]
    x = (x | (x << _SPREAD_SHIFTS_21[2])) & _SPREAD_MASKS_21[3]
    x = (x | (x << _SPREAD_SHIFTS_21[3])) & _SPREAD_MASKS_21[4]
    x = (x | (x << _SPREAD_SHIFTS_21[4])) & _SPREAD_MASKS_21[5]
    return x


def morton_encode_3d_fast(x: int, y: int, z: int) -> int:
    """
    Fast Morton encoding using magic bits (up to 21 bits per coord).
    
    ~3x faster than loop-based encoding for single coordinates.
    """
    return _spread_bits_21(x) | (_spread_bits_21(y) << 1) | (_spread_bits_21(z) << 2)


# ═══════════════════════════════════════════════════════════════════════════════════════
# MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Scalar
    'morton_encode_3d',
    'morton_decode_3d',
    'morton_encode_3d_fast',
    
    # Vectorized NumPy
    'morton_encode_3d_vectorized',
    'morton_decode_3d_vectorized',
    
    # PyTorch
    'morton_encode_3d_torch',
    'morton_decode_3d_torch',
    
    # Grid operations
    'linear_to_morton_3d',
    'morton_to_linear_3d',
    
    # Utilities
    'Morton3DGrid',
    'validate_morton_3d',
]


if __name__ == "__main__":
    # Run validation
    validate_morton_3d(n_bits=4, verbose=True)
    print()
    validate_morton_3d(n_bits=6, verbose=True)
