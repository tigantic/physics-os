"""
Tensor Core Exploitation Module
================================

Direct mapping of tensor-network contractions to NVIDIA Tensor Core
WMMA (Warp Matrix Multiply Accumulate) and MMA (Matrix Multiply Accumulate)
instructions for maximum throughput.

Provides:
- WMMA tile shapes matched to hardware (16×16×16 for FP16, 8×4×8 for TF32)
- TC-aware operand layout (row-major A, col-major B, row-major C)
- Fragment-level simulation for CPU fallback
- Batched GEMM dispatch via TC-optimized paths
- Tensor-train contraction with TC-tiled inner products
- Accumulation policy (FP16→FP32, BF16→FP32, INT8→INT32)
- Occupancy-aware tile selection
- Kernel fusion hints for adjacent contractions

Works with CuPy, PyTorch, or falls back to NumPy emulation.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Precision & Tile definitions
# ---------------------------------------------------------------------------

class TCPrecision(Enum):
    """Supported Tensor Core precision modes."""

    FP16_FP16 = "fp16_fp16"      # FP16 input, FP16 accumulate
    FP16_FP32 = "fp16_fp32"      # FP16 input, FP32 accumulate
    BF16_FP32 = "bf16_fp32"      # BF16 input, FP32 accumulate
    TF32_FP32 = "tf32_fp32"      # TF32 input, FP32 accumulate
    FP64_FP64 = "fp64_fp64"      # DMMA (Ampere+)
    INT8_INT32 = "int8_int32"    # Integer quantized
    FP8_FP32 = "fp8_fp32"        # Hopper/Blackwell FP8


@dataclass(frozen=True)
class WMMATile:
    """WMMA tile configuration (M, N, K) per warp."""

    m: int
    n: int
    k: int
    input_dtype: np.dtype
    accum_dtype: np.dtype
    compute_capability: int  # minimum SM version

    @property
    def ops_per_tile(self) -> int:
        return 2 * self.m * self.n * self.k


# Canonical WMMA tile shapes by precision
WMMA_TILES: Dict[TCPrecision, WMMATile] = {
    TCPrecision.FP16_FP16: WMMATile(16, 16, 16, np.dtype("float16"), np.dtype("float16"), 70),
    TCPrecision.FP16_FP32: WMMATile(16, 16, 16, np.dtype("float16"), np.dtype("float32"), 70),
    TCPrecision.BF16_FP32: WMMATile(16, 16, 16, np.dtype("float16"), np.dtype("float32"), 80),
    TCPrecision.TF32_FP32: WMMATile(16, 16, 8, np.dtype("float32"), np.dtype("float32"), 80),
    TCPrecision.FP64_FP64: WMMATile(8, 8, 4, np.dtype("float64"), np.dtype("float64"), 80),
    TCPrecision.INT8_INT32: WMMATile(16, 16, 16, np.dtype("int8"), np.dtype("int32"), 72),
    TCPrecision.FP8_FP32: WMMATile(16, 16, 32, np.dtype("float16"), np.dtype("float32"), 89),
}


# ---------------------------------------------------------------------------
# Fragment (host emulation of WMMA fragments)
# ---------------------------------------------------------------------------

@dataclass
class Fragment:
    """CPU-side emulation of a WMMA fragment for testing."""

    data: np.ndarray
    role: str  # "a", "b", or "accumulator"
    tile: WMMATile

    @classmethod
    def load_a(cls, matrix: np.ndarray, tile: WMMATile, row: int, col: int) -> "Fragment":
        """Load an A-fragment (row-major) from matrix."""
        block = matrix[row:row + tile.m, col:col + tile.k].astype(tile.input_dtype)
        return cls(data=block, role="a", tile=tile)

    @classmethod
    def load_b(cls, matrix: np.ndarray, tile: WMMATile, row: int, col: int) -> "Fragment":
        """Load a B-fragment (col-major storage) from matrix."""
        block = matrix[row:row + tile.k, col:col + tile.n].astype(tile.input_dtype)
        return cls(data=block, role="b", tile=tile)

    @classmethod
    def zeros(cls, tile: WMMATile) -> "Fragment":
        """Create a zero-initialized accumulator fragment."""
        return cls(
            data=np.zeros((tile.m, tile.n), dtype=tile.accum_dtype),
            role="accumulator",
            tile=tile,
        )


def wmma_mma(frag_a: Fragment, frag_b: Fragment, frag_c: Fragment) -> Fragment:
    """Emulate WMMA MMA: C += A @ B.

    In hardware this is a single warp instruction.
    """
    assert frag_a.role == "a" and frag_b.role == "b" and frag_c.role == "accumulator"
    tile = frag_a.tile
    product = frag_a.data.astype(tile.accum_dtype) @ frag_b.data.astype(tile.accum_dtype)
    return Fragment(
        data=frag_c.data + product,
        role="accumulator",
        tile=tile,
    )


# ---------------------------------------------------------------------------
# TC-tiled GEMM
# ---------------------------------------------------------------------------

@dataclass
class TCGEMMConfig:
    """Configuration for Tensor-Core-tiled GEMM."""

    precision: TCPrecision = TCPrecision.FP16_FP32
    use_torch: bool = True
    use_cupy: bool = False
    split_k: int = 1  # K-dimension split for parallelism


def _pad_to_tile(mat: np.ndarray, m_tile: int, n_tile: int) -> np.ndarray:
    """Pad matrix to be divisible by tile dimensions."""
    rows, cols = mat.shape
    pad_rows = (m_tile - rows % m_tile) % m_tile
    pad_cols = (n_tile - cols % n_tile) % n_tile
    if pad_rows == 0 and pad_cols == 0:
        return mat
    return np.pad(mat, ((0, pad_rows), (0, pad_cols)), mode="constant")


def tc_gemm(
    a: np.ndarray,
    b: np.ndarray,
    config: Optional[TCGEMMConfig] = None,
) -> np.ndarray:
    """Tensor-Core-aware GEMM with tile-based execution.

    On GPU: dispatches to cuBLAS TC GEMM or PyTorch matmul with TC enabled.
    On CPU: emulates TC tile decomposition for correctness verification.

    Parameters
    ----------
    a : (M, K) matrix
    b : (K, N) matrix
    config : TC configuration

    Returns
    -------
    (M, N) result matrix
    """
    config = config or TCGEMMConfig()
    M_orig, K_a = a.shape
    K_b, N_orig = b.shape
    assert K_a == K_b, f"K-dimension mismatch: {K_a} vs {K_b}"

    tile = WMMA_TILES[config.precision]

    # Try GPU backends first
    if config.use_torch:
        result = _tc_gemm_torch(a, b, config.precision)
        if result is not None:
            return result[:M_orig, :N_orig]

    if config.use_cupy:
        result = _tc_gemm_cupy(a, b, config.precision)
        if result is not None:
            return result[:M_orig, :N_orig]

    # CPU tile emulation
    a_pad = _pad_to_tile(a, tile.m, tile.k)
    b_pad = _pad_to_tile(b, tile.k, tile.n)
    M, K = a_pad.shape
    _, N = b_pad.shape

    c = np.zeros((M, N), dtype=tile.accum_dtype)

    for i in range(0, M, tile.m):
        for j in range(0, N, tile.n):
            acc = Fragment.zeros(tile)
            for k in range(0, K, tile.k):
                fa = Fragment.load_a(a_pad, tile, i, k)
                fb = Fragment.load_b(b_pad, tile, k, j)
                acc = wmma_mma(fa, fb, acc)
            c[i:i + tile.m, j:j + tile.n] = acc.data

    return c[:M_orig, :N_orig]


def _tc_gemm_torch(
    a: np.ndarray,
    b: np.ndarray,
    precision: TCPrecision,
) -> Optional[np.ndarray]:
    """Dispatch GEMM to PyTorch with Tensor Core settings."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
    except ImportError:
        return None

    device = torch.device("cuda")

    # Map precision to torch dtype
    dtype_map = {
        TCPrecision.FP16_FP16: torch.float16,
        TCPrecision.FP16_FP32: torch.float16,
        TCPrecision.BF16_FP32: torch.bfloat16,
        TCPrecision.TF32_FP32: torch.float32,
        TCPrecision.FP64_FP64: torch.float64,
    }
    input_dtype = dtype_map.get(precision)
    if input_dtype is None:
        return None

    # Enable TF32 for FP32 inputs
    if precision == TCPrecision.TF32_FP32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    at = torch.from_numpy(a.astype(np.float32 if precision == TCPrecision.TF32_FP32 else a.dtype))
    bt = torch.from_numpy(b.astype(np.float32 if precision == TCPrecision.TF32_FP32 else b.dtype))
    at = at.to(device=device, dtype=input_dtype)
    bt = bt.to(device=device, dtype=input_dtype)

    ct = torch.matmul(at, bt)

    # Accumulate in higher precision
    accum_map = {
        TCPrecision.FP16_FP32: torch.float32,
        TCPrecision.BF16_FP32: torch.float32,
    }
    accum_dtype = accum_map.get(precision)
    if accum_dtype is not None:
        ct = ct.to(accum_dtype)

    return ct.cpu().numpy()


def _tc_gemm_cupy(
    a: np.ndarray,
    b: np.ndarray,
    precision: TCPrecision,
) -> Optional[np.ndarray]:
    """Dispatch GEMM to CuPy with Tensor Core enforcement."""
    try:
        import cupy as cp  # type: ignore[import-untyped]
    except ImportError:
        return None

    dtype_map = {
        TCPrecision.FP16_FP32: np.float16,
        TCPrecision.BF16_FP32: np.float16,
        TCPrecision.FP64_FP64: np.float64,
    }
    in_dtype = dtype_map.get(precision, np.float32)

    ac = cp.asarray(a.astype(in_dtype))
    bc = cp.asarray(b.astype(in_dtype))
    cc = cp.matmul(ac, bc)
    return cp.asnumpy(cc)


# ---------------------------------------------------------------------------
# TT-contraction via Tensor Cores
# ---------------------------------------------------------------------------

def tt_contract_tc(
    cores: Sequence[np.ndarray],
    precision: TCPrecision = TCPrecision.FP16_FP32,
) -> np.ndarray:
    """Contract a tensor-train using Tensor-Core-tiled GEMM.

    Each core is (r_k, n_k, r_{k+1}).  We reshape and multiply
    left-to-right using tc_gemm.

    Parameters
    ----------
    cores : sequence of 3-D arrays (r_k, n_k, r_{k+1})
    precision : TC precision mode

    Returns
    -------
    Fully contracted dense tensor
    """
    assert len(cores) >= 1
    config = TCGEMMConfig(precision=precision)

    # Start with first core reshaped to (r0 * n0, r1)
    result = cores[0].reshape(-1, cores[0].shape[-1])

    for core in cores[1:]:
        r, n, r_next = core.shape
        mat = core.reshape(r, n * r_next)
        result = tc_gemm(result, mat, config)
        # Reshape to (-1, r_next) for next contraction
        result = result.reshape(-1, r_next)

    return result


def batched_tc_gemm(
    a_batch: Sequence[np.ndarray],
    b_batch: Sequence[np.ndarray],
    precision: TCPrecision = TCPrecision.FP16_FP32,
) -> List[np.ndarray]:
    """Batched GEMM with Tensor Core dispatch.

    Parameters
    ----------
    a_batch : list of (M_i, K_i) matrices
    b_batch : list of (K_i, N_i) matrices
    precision : TC precision mode

    Returns
    -------
    List of result matrices
    """
    assert len(a_batch) == len(b_batch)
    config = TCGEMMConfig(precision=precision)
    return [tc_gemm(a, b, config) for a, b in zip(a_batch, b_batch)]


# ---------------------------------------------------------------------------
# Occupancy / throughput estimation
# ---------------------------------------------------------------------------

@dataclass
class TCThroughput:
    """Estimated TC throughput for a given configuration."""

    tiles_per_sm: int
    active_warps: int
    peak_tflops: float
    efficiency: float  # fraction of peak


def estimate_throughput(
    m: int,
    n: int,
    k: int,
    precision: TCPrecision,
    num_sms: int = 128,
    clock_ghz: float = 1.5,
) -> TCThroughput:
    """Estimate Tensor Core throughput for a GEMM problem.

    Parameters
    ----------
    m, n, k : problem dimensions
    precision : TC precision mode
    num_sms : number of SMs on target GPU
    clock_ghz : SM clock frequency

    Returns
    -------
    Throughput estimate
    """
    tile = WMMA_TILES[precision]

    # Number of tiles
    m_tiles = (m + tile.m - 1) // tile.m
    n_tiles = (n + tile.n - 1) // tile.n
    k_tiles = (k + tile.k - 1) // tile.k

    total_tiles = m_tiles * n_tiles * k_tiles
    ops_per_tile = tile.ops_per_tile

    # Each warp processes one tile per instruction
    # Assume 4 warps/SM active for TC workloads
    warps_per_sm = 4
    total_warps = num_sms * warps_per_sm
    tiles_per_sm = max(1, total_tiles // num_sms)

    # Peak: ops_per_tile * num_sms * warps_per_sm * clock
    peak_ops = ops_per_tile * total_warps * clock_ghz * 1e9
    peak_tflops = peak_ops / 1e12

    # Efficiency — depends on tile utilization
    total_ops = 2.0 * m * n * k
    time_tiles = total_tiles / (total_warps * clock_ghz * 1e9 / 1.0)
    achieved = total_ops / max(time_tiles, 1e-15) / 1e12
    efficiency = min(1.0, achieved / max(peak_tflops, 1e-15))

    return TCThroughput(
        tiles_per_sm=tiles_per_sm,
        active_warps=total_warps,
        peak_tflops=peak_tflops,
        efficiency=efficiency,
    )


# ---------------------------------------------------------------------------
# Kernel fusion hint
# ---------------------------------------------------------------------------

@dataclass
class FusionHint:
    """Hint for fusing adjacent TC operations."""

    op_names: List[str]
    shared_k_dim: int
    can_fuse: bool
    estimated_speedup: float


def analyze_fusion(
    shapes: Sequence[Tuple[int, int, int]],
) -> List[FusionHint]:
    """Analyze a sequence of GEMM shapes for fusion opportunities.

    Parameters
    ----------
    shapes : list of (M, K, N) for each GEMM

    Returns
    -------
    List of fusion hints for adjacent pairs
    """
    hints: List[FusionHint] = []
    for i in range(len(shapes) - 1):
        m1, k1, n1 = shapes[i]
        m2, k2, n2 = shapes[i + 1]

        # Can fuse if output of first (M1, N1) matches input of second
        can_fuse = (n1 == m2)
        shared_k = n1 if can_fuse else 0

        # Rough speedup: avoid writing/reading intermediate
        intermediate_bytes = n1 * m2 * 4  # FP32
        speedup = 1.3 if can_fuse and intermediate_bytes > 4096 else 1.0

        hints.append(FusionHint(
            op_names=[f"gemm_{i}", f"gemm_{i + 1}"],
            shared_k_dim=shared_k,
            can_fuse=can_fuse,
            estimated_speedup=speedup,
        ))

    return hints


__all__ = [
    "TCPrecision",
    "WMMATile",
    "WMMA_TILES",
    "Fragment",
    "wmma_mma",
    "TCGEMMConfig",
    "tc_gemm",
    "tt_contract_tc",
    "batched_tc_gemm",
    "TCThroughput",
    "estimate_throughput",
    "FusionHint",
    "analyze_fusion",
]
