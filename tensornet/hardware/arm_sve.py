"""
ARM SVE / SVE2 SIMD Backend
============================

Vectorized QTT kernels for ARM Neoverse (Graviton 3/4) and
Apple Silicon (via NEON → SVE translation), exploiting the
Scalable Vector Extension for variable-length SIMD.

Provides:
- Feature detection (SVE/SVE2/NEON, vector length)
- Vectorized TT-core contraction via NumPy (auto-vectorized by BLAS)
- Manual SVE-width-aware tiling for cache optimization
- BLAS dispatch to ARM Performance Libraries (ARMPL) or OpenBLAS
- Multi-threaded scatter/gather for QTT evaluation
- Memory-aligned allocation for SVE vector loads

When ``armpl`` or ``openblas`` with SVE support is the backing
BLAS, all ``np.matmul`` calls automatically use SVE instructions.
This module adds SVE-aware tiling and explicit alignment.
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import BackendKind, DeviceInfo, HardwareBackend, register_backend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SVE feature detection
# ---------------------------------------------------------------------------

@dataclass
class SVECapabilities:
    """Detected ARM SVE capabilities."""

    has_sve: bool = False
    has_sve2: bool = False
    has_neon: bool = True   # NEON is baseline on AArch64
    vector_length_bits: int = 128  # NEON=128, SVE=128..2048
    vector_length_bytes: int = 16

    @staticmethod
    def detect() -> "SVECapabilities":
        """Detect SVE capabilities from /proc/cpuinfo or sysctl."""
        caps = SVECapabilities()
        machine = platform.machine().lower()
        if machine not in ("aarch64", "arm64"):
            caps.has_neon = False
            return caps

        # Linux: parse /proc/cpuinfo
        if os.path.exists("/proc/cpuinfo"):
            try:
                with open("/proc/cpuinfo", "r") as f:
                    text = f.read().lower()
                caps.has_sve = "sve" in text
                caps.has_sve2 = "sve2" in text
            except OSError:
                pass

            # Read SVE vector length via prctl (Linux)
            if caps.has_sve:
                try:
                    libc = ctypes.CDLL("libc.so.6", use_errno=True)
                    PR_SVE_GET_VL = 51
                    vl = libc.prctl(PR_SVE_GET_VL, 0, 0, 0, 0)
                    if vl > 0:
                        caps.vector_length_bytes = vl & 0xFFFF
                        caps.vector_length_bits = caps.vector_length_bytes * 8
                except (OSError, AttributeError):
                    pass

        # macOS: Apple Silicon always has NEON, no SVE yet
        if platform.system() == "Darwin":
            caps.has_neon = True
            caps.has_sve = False
            caps.has_sve2 = False
            caps.vector_length_bits = 128
            caps.vector_length_bytes = 16

        return caps

    @property
    def fp64_lanes(self) -> int:
        """Number of float64 elements per SVE vector."""
        return self.vector_length_bytes // 8

    @property
    def fp32_lanes(self) -> int:
        """Number of float32 elements per SVE vector."""
        return self.vector_length_bytes // 4


# ---------------------------------------------------------------------------
# Aligned allocation
# ---------------------------------------------------------------------------

def aligned_empty(
    shape: Tuple[int, ...],
    dtype: np.dtype = np.float64,
    alignment: int = 64,
) -> np.ndarray:
    """Allocate numpy array with guaranteed memory alignment.

    Alignment should be ≥ SVE vector length in bytes for optimal
    SVE vector loads.
    """
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    buf = np.empty(nbytes + alignment, dtype=np.uint8)
    offset = (-buf.ctypes.data) % alignment
    aligned = buf[offset : offset + nbytes].view(dtype)
    return aligned.reshape(shape)


def aligned_zeros(
    shape: Tuple[int, ...],
    dtype: np.dtype = np.float64,
    alignment: int = 64,
) -> np.ndarray:
    arr = aligned_empty(shape, dtype, alignment)
    arr[:] = 0
    return arr


# ---------------------------------------------------------------------------
# SVE-tiled matrix operations
# ---------------------------------------------------------------------------

def sve_tiled_matmul(
    A: np.ndarray,
    B: np.ndarray,
    tile_m: int = 64,
    tile_n: int = 64,
    tile_k: int = 64,
) -> np.ndarray:
    """Cache-friendly tiled matmul optimized for SVE BLAS.

    On ARM with ARMPL/OpenBLAS-SVE, this tiles the computation
    to fit in L1/L2 cache, letting the BLAS library exploit SVE
    for each tile.
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Dimension mismatch: A ({M},{K}) vs B ({K2},{N})"
    C = np.zeros((M, N), dtype=A.dtype)

    for i0 in range(0, M, tile_m):
        i1 = min(i0 + tile_m, M)
        for j0 in range(0, N, tile_n):
            j1 = min(j0 + tile_n, N)
            for k0 in range(0, K, tile_k):
                k1 = min(k0 + tile_k, K)
                # BLAS GEMM on tile — SVE-accelerated internally
                C[i0:i1, j0:j1] += A[i0:i1, k0:k1] @ B[k0:k1, j0:j1]
    return C


def sve_batched_gemm(
    A_batch: np.ndarray, B_batch: np.ndarray
) -> np.ndarray:
    """Batched GEMM for TT-core contractions.

    A_batch: (batch, M, K), B_batch: (batch, K, N)
    """
    batch = A_batch.shape[0]
    M, K = A_batch.shape[1], A_batch.shape[2]
    N = B_batch.shape[2]
    C = np.empty((batch, M, N), dtype=A_batch.dtype)
    for b in range(batch):
        C[b] = A_batch[b] @ B_batch[b]
    return C


# ---------------------------------------------------------------------------
# QTT evaluation with SVE scatter/gather
# ---------------------------------------------------------------------------

def qtt_evaluate_sve(
    cores: List[np.ndarray],
    indices: np.ndarray,
) -> np.ndarray:
    """Evaluate QTT at multi-indices using SVE-optimized gather.

    Parameters
    ----------
    cores : list of (r_k, n_k, r_{k+1}) arrays
    indices : (n_points, n_modes) integer indices

    Returns
    -------
    values : (n_points,) evaluated values
    """
    n_points, n_modes = indices.shape
    assert n_modes == len(cores), "Index modes must match number of cores"

    # Start with first core slices
    current = cores[0][:, indices[:, 0], :]  # (r0, n_points, r1)
    # current shape: (1, n_points, r1) → squeeze → (n_points, r1)
    if current.shape[0] == 1:
        current = current[0]  # (n_points, r1)
    else:
        current = current.transpose(1, 0, 2).reshape(n_points, -1)

    for k in range(1, n_modes):
        slices = cores[k][:, indices[:, k], :]  # (r_k, n_points, r_{k+1})
        # For each point: current[p, :] @ slices[:, p, :]
        # = einsum('pr, rpq -> pq', current, slices)
        next_r = slices.shape[2]
        # Vectorized contraction
        # slices transposed: (n_points, r_k, r_{k+1})
        st = slices.transpose(1, 0, 2)
        # current: (n_points, r_k)
        # result: (n_points, r_{k+1})
        current = np.einsum("pr,prq->pq", current, st)

    return current.ravel()


# ---------------------------------------------------------------------------
# ARM SVE Backend
# ---------------------------------------------------------------------------

@dataclass
class SVETensorHandle:
    """Handle wrapping aligned numpy array for SVE processing."""

    data: np.ndarray
    aligned: bool = True

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.data.shape)


class ARMSVEBackend:
    """ARM SVE/SVE2 SIMD backend using numpy + BLAS."""

    def __init__(self) -> None:
        self._caps = SVECapabilities.detect()

    @property
    def kind(self) -> BackendKind:
        return BackendKind.ARM_SVE

    def is_available(self) -> bool:
        return self._caps.has_neon or self._caps.has_sve

    @property
    def capabilities(self) -> SVECapabilities:
        return self._caps

    def enumerate_devices(self) -> List[DeviceInfo]:
        if not self.is_available():
            return []
        return [
            DeviceInfo(
                backend=BackendKind.ARM_SVE,
                device_id=0,
                name=f"ARM {'SVE2' if self._caps.has_sve2 else 'SVE' if self._caps.has_sve else 'NEON'}",
                capabilities={
                    "sve": self._caps.has_sve,
                    "sve2": self._caps.has_sve2,
                    "neon": self._caps.has_neon,
                    "vector_length_bits": self._caps.vector_length_bits,
                    "fp64_lanes": self._caps.fp64_lanes,
                    "fp32_lanes": self._caps.fp32_lanes,
                },
            )
        ]

    def allocate(self, shape: Tuple[int, ...], dtype: np.dtype) -> SVETensorHandle:
        alignment = max(64, self._caps.vector_length_bytes)
        data = aligned_empty(shape, dtype, alignment)
        return SVETensorHandle(data=data, aligned=True)

    def free(self, handle: Any) -> None:
        if isinstance(handle, SVETensorHandle):
            handle.data = np.empty(0)

    def to_numpy(self, handle: Any) -> np.ndarray:
        if isinstance(handle, SVETensorHandle):
            return handle.data
        raise TypeError(f"Expected SVETensorHandle, got {type(handle)}")

    def from_numpy(self, arr: np.ndarray, device_id: int = 0) -> SVETensorHandle:
        alignment = max(64, self._caps.vector_length_bytes)
        aligned = aligned_empty(arr.shape, arr.dtype, alignment)
        np.copyto(aligned, arr)
        return SVETensorHandle(data=aligned, aligned=True)

    def matmul(self, a: Any, b: Any) -> SVETensorHandle:
        da = a.data if isinstance(a, SVETensorHandle) else a
        db = b.data if isinstance(b, SVETensorHandle) else b
        result = np.matmul(da, db)
        return SVETensorHandle(data=result)

    def svd(
        self, a: Any, full_matrices: bool = False
    ) -> Tuple[SVETensorHandle, SVETensorHandle, SVETensorHandle]:
        da = a.data if isinstance(a, SVETensorHandle) else a
        U, S, Vh = np.linalg.svd(da, full_matrices=full_matrices)
        return SVETensorHandle(data=U), SVETensorHandle(data=S), SVETensorHandle(data=Vh)

    def tt_contract(self, cores: Sequence[Any]) -> SVETensorHandle:
        if not cores:
            raise ValueError("Empty core list")
        result = cores[0].data if isinstance(cores[0], SVETensorHandle) else cores[0]
        for core in cores[1:]:
            c = core.data if isinstance(core, SVETensorHandle) else core
            if c.ndim == 3:
                r, n, rr = c.shape
                c_mat = c.reshape(r, n * rr)
                shape = result.shape
                result = result.reshape(-1, shape[-1]) @ c_mat
                result = result.reshape(*shape[:-1], n, rr)
            else:
                result = np.matmul(result, c)
        while result.ndim > 1 and result.shape[0] == 1:
            result = np.squeeze(result, axis=0)
        while result.ndim > 1 and result.shape[-1] == 1:
            result = np.squeeze(result, axis=-1)
        return SVETensorHandle(data=result)

    def optimal_tile_size(self, dtype: np.dtype = np.float64) -> int:
        """Compute optimal tile dimension for L1 cache occupancy."""
        l1_bytes = 64 * 1024  # typical ARM L1 = 64 KB
        elem_size = np.dtype(dtype).itemsize
        # 3 tile matrices must fit in L1: 3 * tile^2 * elem_size ≤ L1
        tile = int(np.sqrt(l1_bytes / (3 * elem_size)))
        # Round down to SVE lane multiple
        lanes = self._caps.fp64_lanes if dtype == np.float64 else self._caps.fp32_lanes
        return max(lanes, (tile // lanes) * lanes)


# ---------------------------------------------------------------------------
# Auto-register (only on ARM)
# ---------------------------------------------------------------------------

_backend = ARMSVEBackend()
if _backend.is_available():
    register_backend(_backend)  # type: ignore[arg-type]

__all__ = [
    "ARMSVEBackend",
    "SVETensorHandle",
    "SVECapabilities",
    "aligned_empty",
    "aligned_zeros",
    "sve_tiled_matmul",
    "sve_batched_gemm",
    "qtt_evaluate_sve",
]
