#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║                    FLUIDELITE BLOCK-INDEPENDENT QTT INGEST ENGINE                    ║
║                                                                                      ║
║                  "RAID Array of Tensors" — Seekable, Parallel, Infinite              ║
║                                                                                      ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                      ║
║  ANTI-PATTERN (What We DON'T Do):                                                    ║
║  ─────────────────────────────────                                                   ║
║      ✗ Merge 1,000 chunks into one "Global Tensor Train"                             ║
║      ✗ Lose parallelism (can't decode middle without start)                          ║
║      ✗ Accumulate numerical error in long TT chains                                  ║
║                                                                                      ║
║  THE OPTIMIZATION: "Block-Independent QTT"                                           ║
║  ─────────────────────────────────────────                                           ║
║      Treat the file like a RAID array of tensors:                                    ║
║                                                                                      ║
║      Chunk 1 (0-64MB):    QTT_1 + Delta_1                                           ║
║      Chunk 2 (64-128MB):  QTT_2 + Delta_2                                           ║
║      ...                                                                             ║
║                                                                                      ║
║  CONTAINER FORMAT (.fluid):                                                          ║
║  ──────────────────────────                                                          ║
║      [Global Header] [Block Index] | [Core_1][Res_1] | [Core_2][Res_2] | ...        ║
║                                                                                      ║
║  RANDOM ACCESS:                                                                      ║
║  ──────────────                                                                      ║
║      Client wants "Minute 45 of the storm"?                                         ║
║      → Seek to Block_14. Load Core_14. Expand. Add Res_14.                          ║
║      → Instant, seekable, bit-perfect playback.                                     ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
"""

import io
import struct
import hashlib
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Iterator, BinaryIO, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import mmap
from datetime import datetime
import json
import lzma  # Fallback if zstandard not available

# Try zstandard, fall back to lzma
try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

# ═══════════════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════════════

BLOCK_SIZE = 64 * 1024 * 1024  # 64 MB - Optimal for L3 cache
MAGIC_BYTES = b'FLUD'          # File magic number
VERSION = 1                     # Container format version
DEFAULT_RANK = 32               # Default TT-SVD max rank
ZSTD_LEVEL = 3                  # Compression level (1-22)


# ═══════════════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class BlockMeta:
    """Metadata for a single compressed block."""
    block_id: int
    original_size: int           # Bytes in original chunk
    cores_offset: int            # Byte offset to cores in container
    cores_size: int              # Compressed size of cores
    residual_offset: int         # Byte offset to residual
    residual_size: int           # Compressed size of residual
    checksum: bytes              # SHA-256 of original block
    grid_shape: Tuple[int, ...]  # Shape used for TT decomposition
    n_cores: int                 # Number of TT cores
    max_rank: int                # Maximum bond dimension used
    residual_norm: float         # L2 norm of residual (quality metric)


@dataclass
class FluidHeader:
    """Global header for .fluid container."""
    magic: bytes = MAGIC_BYTES
    version: int = VERSION
    total_blocks: int = 0
    original_size: int = 0       # Total original size
    compressed_size: int = 0     # Total compressed size
    block_size: int = BLOCK_SIZE
    created_at: float = 0.0
    metadata_json: bytes = b''   # Optional JSON metadata


@dataclass
class CompressedBlock:
    """A fully compressed block ready for serialization."""
    meta: BlockMeta
    cores_data: bytes            # zstd-compressed TT cores
    residual_data: bytes         # zstd-compressed residual


# ═══════════════════════════════════════════════════════════════════════════════════════
# TT-SVD CORE: The Physics Engine
# ═══════════════════════════════════════════════════════════════════════════════════════

def _optimal_grid_shape(size: int) -> Tuple[int, ...]:
    """
    Find optimal reshaping of linear bytes to high-dimensional grid.
    
    Goal: Maximize tensorization dimensions while keeping each dim small.
    For 64MB = 67,108,864 bytes = 2^26
    
    We want dimensions like (4, 4, 4, ...) or (8, 8, 8, ...) to expose
    the maximum correlation structure for TT-SVD.
    """
    if size <= 0:
        return (1,)
    
    # Find power of 2 that fits
    log2_size = int(np.floor(np.log2(size)))
    padded_size = 2 ** log2_size
    
    # Create balanced dimensions: all 4s (2^2), with adjustment
    # 2^26 = 4^13 = (4,4,4,4,4,4,4,4,4,4,4,4,4)
    n_dims = log2_size // 2
    if n_dims < 1:
        n_dims = 1
    
    dim_size = 4  # Base dimension
    shape = [dim_size] * n_dims
    
    # Handle remainder
    remainder = log2_size - (n_dims * 2)
    if remainder > 0:
        shape[-1] *= (2 ** remainder)
    
    # Verify product
    product = np.prod(shape)
    if product != padded_size:
        # Fallback: simple binary tensorization
        shape = [2] * log2_size
    
    return tuple(shape)


def tt_svd_compress(
    data: np.ndarray,
    max_rank: int = DEFAULT_RANK,
    eps: float = 1e-6
) -> Tuple[List[np.ndarray], np.ndarray, Tuple[int, ...]]:
    """
    TT-SVD decomposition of data tensor.
    
    Algorithm:
    1. Reshape data to high-dimensional grid
    2. Sequential SVD from left to right
    3. Truncate singular values below eps or beyond max_rank
    4. Compute residual: original - approximation
    
    Returns:
        cores: List of TT cores [shape: (r_{k-1}, n_k, r_k)]
        residual: Original - Approximation (same shape as reshaped data)
        grid_shape: Shape used for tensorization
    """
    # Step 1: Reshape to high-dim grid
    original_size = data.size
    grid_shape = _optimal_grid_shape(original_size)
    target_size = np.prod(grid_shape)
    
    # Pad if necessary
    if original_size < target_size:
        padded = np.zeros(target_size, dtype=data.dtype)
        padded[:original_size] = data.ravel()
        tensor = padded.reshape(grid_shape)
    elif original_size > target_size:
        # Truncate to fit (shouldn't happen with proper sizing)
        tensor = data.ravel()[:target_size].reshape(grid_shape)
    else:
        tensor = data.reshape(grid_shape)
    
    # Step 2: TT-SVD decomposition
    cores = []
    n_dims = len(grid_shape)
    C = tensor.copy()
    
    r_prev = 1
    for k in range(n_dims - 1):
        # Reshape to matrix: (r_{k-1} * n_k) x (remaining dims)
        n_k = grid_shape[k]
        remaining = np.prod(grid_shape[k+1:])
        
        C_mat = C.reshape(r_prev * n_k, -1)
        
        # SVD
        U, S, Vt = np.linalg.svd(C_mat, full_matrices=False)
        
        # Truncate
        # Keep singular values above threshold
        s_sum = np.sum(S ** 2)
        s_cumsum = np.cumsum(S ** 2)
        
        # Find rank that captures (1 - eps^2) of the energy
        threshold = (1 - eps ** 2) * s_sum
        rank = np.searchsorted(s_cumsum, threshold) + 1
        rank = min(rank, max_rank, len(S))
        rank = max(rank, 1)  # At least rank 1
        
        # Truncated SVD
        U_trunc = U[:, :rank]
        S_trunc = S[:rank]
        Vt_trunc = Vt[:rank, :]
        
        # Core k: shape (r_{k-1}, n_k, r_k)
        core = U_trunc.reshape(r_prev, n_k, rank)
        cores.append(core)
        
        # Propagate: C = S @ Vt for next iteration
        C = np.diag(S_trunc) @ Vt_trunc
        r_prev = rank
    
    # Last core: shape (r_{n-2}, n_{n-1}, 1)
    n_last = grid_shape[-1]
    last_core = C.reshape(r_prev, n_last, 1)
    cores.append(last_core)
    
    # Step 3: Reconstruct approximation
    approx = tt_expand(cores)
    
    # Step 4: Compute residual
    residual = tensor - approx.reshape(grid_shape)
    
    return cores, residual, grid_shape


def tt_expand(cores: List[np.ndarray]) -> np.ndarray:
    """
    Expand TT cores back to full tensor.
    
    Contract: C_1 @ C_2 @ ... @ C_n
    """
    if not cores:
        return np.array([])
    
    # Start with first core: (1, n_1, r_1) -> (n_1, r_1)
    result = cores[0].squeeze(0)  # (n_1, r_1)
    
    for core in cores[1:]:
        # core shape: (r_{k-1}, n_k, r_k)
        # result shape: (n_1 * ... * n_{k-1}, r_{k-1})
        r_prev, n_k, r_k = core.shape
        
        # Contract: result @ core
        # Reshape core: (r_{k-1}, n_k * r_k)
        core_mat = core.reshape(r_prev, n_k * r_k)
        
        # result @ core_mat: (prod, n_k * r_k)
        result = result @ core_mat
        
        # Reshape: (prod * n_k, r_k)
        result = result.reshape(-1, r_k)
    
    # Final squeeze: (prod, 1) -> (prod,)
    return result.squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════════════
# COMPRESSION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════════════

class ZstdCompressor:
    """Thread-safe compressor pool (zstd or lzma fallback)."""
    
    _local = threading.local()
    
    @classmethod
    def get_compressor(cls):
        if HAS_ZSTD:
            if not hasattr(cls._local, 'compressor'):
                cls._local.compressor = zstd.ZstdCompressor(level=ZSTD_LEVEL)
            return cls._local.compressor
        return None
    
    @classmethod
    def get_decompressor(cls):
        if HAS_ZSTD:
            if not hasattr(cls._local, 'decompressor'):
                cls._local.decompressor = zstd.ZstdDecompressor()
            return cls._local.decompressor
        return None
    
    @classmethod
    def compress(cls, data: bytes) -> bytes:
        if HAS_ZSTD:
            return cls.get_compressor().compress(data)
        else:
            return lzma.compress(data, preset=3)
    
    @classmethod
    def decompress(cls, data: bytes) -> bytes:
        if HAS_ZSTD:
            return cls.get_decompressor().decompress(data)
        else:
            return lzma.decompress(data)


def _serialize_cores(cores: List[np.ndarray]) -> bytes:
    """Serialize TT cores to bytes."""
    buffer = io.BytesIO()
    
    # Number of cores
    buffer.write(struct.pack('<I', len(cores)))
    
    for core in cores:
        # Shape
        shape = core.shape
        buffer.write(struct.pack('<I', len(shape)))
        for dim in shape:
            buffer.write(struct.pack('<I', dim))
        
        # Dtype
        dtype_str = str(core.dtype).encode('utf-8')
        buffer.write(struct.pack('<I', len(dtype_str)))
        buffer.write(dtype_str)
        
        # Data
        data = core.tobytes()
        buffer.write(struct.pack('<Q', len(data)))
        buffer.write(data)
    
    return buffer.getvalue()


def _deserialize_cores(data: bytes) -> List[np.ndarray]:
    """Deserialize TT cores from bytes."""
    buffer = io.BytesIO(data)
    
    n_cores = struct.unpack('<I', buffer.read(4))[0]
    cores = []
    
    for _ in range(n_cores):
        # Shape
        n_dims = struct.unpack('<I', buffer.read(4))[0]
        shape = tuple(struct.unpack('<I', buffer.read(4))[0] for _ in range(n_dims))
        
        # Dtype
        dtype_len = struct.unpack('<I', buffer.read(4))[0]
        dtype_str = buffer.read(dtype_len).decode('utf-8')
        dtype = np.dtype(dtype_str)
        
        # Data
        data_len = struct.unpack('<Q', buffer.read(8))[0]
        core_data = buffer.read(data_len)
        
        core = np.frombuffer(core_data, dtype=dtype).reshape(shape)
        cores.append(core.copy())  # Copy to make writeable
    
    return cores


def compress_block(
    block_id: int,
    data: bytes,
    max_rank: int = DEFAULT_RANK,
    lossy: bool = False
) -> CompressedBlock:
    """
    Compress a single 64MB block using TT-SVD + entropy coding.
    
    This is the core "Physics Loop" - runs in parallel across blocks.
    
    1. Reshape: Map 64MB linear bytes → High-Dimensional Grid
    2. TT-SVD: Decompose to find Approximation (The Cores)
    3. Subtract: Original - Approx = Residual
    4. Compress: entropy-code(Residual)
    
    Args:
        block_id: Block index
        data: Raw bytes
        max_rank: Maximum TT bond dimension
        lossy: If True, skip residual for maximum compression (not bit-perfect)
    """
    original_size = len(data)
    checksum = hashlib.sha256(data).digest()
    
    # Convert to numpy array
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
    
    # TT-SVD decomposition
    cores, residual, grid_shape = tt_svd_compress(arr, max_rank=max_rank)
    
    # Compute quality metric
    residual_norm = np.linalg.norm(residual)
    
    # Convert to float16 for storage (2x compression of cores)
    cores_f16 = [c.astype(np.float16) for c in cores]
    cores_bytes = _serialize_cores(cores_f16)
    cores_compressed = ZstdCompressor.compress(cores_bytes)
    
    if lossy:
        # Lossy mode: no residual storage
        residual_compressed = b''
    else:
        # Lossless mode: compute residual using STORED precision (float16->float32)
        # This ensures bit-perfect reconstruction
        cores_restored = [c.astype(np.float32) for c in cores_f16]
        approx = tt_expand(cores_restored)
        
        # Handle size mismatch: grid may be padded or truncated
        grid_size = np.prod(grid_shape)
        approx_flat = approx.ravel()
        
        # Extend or truncate approximation to match original size
        if len(approx_flat) < original_size:
            # Pad with zeros
            approx_extended = np.zeros(original_size, dtype=np.float32)
            approx_extended[:len(approx_flat)] = approx_flat
            approx_flat = approx_extended
        else:
            approx_flat = approx_flat[:original_size]
        
        approx_rounded = np.clip(np.round(approx_flat), 0, 255).astype(np.uint8)
        original_uint8 = np.frombuffer(data, dtype=np.uint8)
        
        # Compute exact integer residual from the STORED approximation
        # This is in range [-255, 255], fits in int16
        residual_exact = original_uint8.astype(np.int16) - approx_rounded.astype(np.int16)
        
        # Pack and compress (int16 is exact)
        residual_packed = residual_exact.tobytes()
        residual_compressed = ZstdCompressor.compress(residual_packed)
    
    # Build metadata (offsets filled in during container write)
    meta = BlockMeta(
        block_id=block_id,
        original_size=original_size,
        cores_offset=0,  # Filled during serialization
        cores_size=len(cores_compressed),
        residual_offset=0,  # Filled during serialization
        residual_size=len(residual_compressed),
        checksum=checksum,
        grid_shape=grid_shape,
        n_cores=len(cores),
        max_rank=max(c.shape[-1] for c in cores),
        residual_norm=float(residual_norm)
    )
    
    return CompressedBlock(
        meta=meta,
        cores_data=cores_compressed,
        residual_data=residual_compressed
    )


def decompress_block(
    cores_data: bytes,
    residual_data: bytes,
    grid_shape: Tuple[int, ...],
    original_size: int
) -> bytes:
    """
    Decompress a single block back to original bytes.
    
    1. Decompress cores
    2. Expand TT to get approximation
    3. Decompress residual (if present)
    4. Add: round(Approx) + Residual = Original
    5. Convert back to bytes
    """
    # Decompress cores
    cores_bytes = ZstdCompressor.decompress(cores_data)
    cores = _deserialize_cores(cores_bytes)
    
    # Convert float16 back to float32 for computation
    cores = [c.astype(np.float32) for c in cores]
    
    # Expand TT
    approx = tt_expand(cores)
    approx_flat = approx.ravel()
    
    # Handle size mismatch: extend or truncate to original size
    if len(approx_flat) < original_size:
        approx_extended = np.zeros(original_size, dtype=np.float32)
        approx_extended[:len(approx_flat)] = approx_flat
        approx_flat = approx_extended
    else:
        approx_flat = approx_flat[:original_size]
    
    # Round to uint8
    approx_rounded = np.clip(np.round(approx_flat), 0, 255).astype(np.uint8)
    
    if len(residual_data) == 0:
        # Lossy mode: no residual
        return approx_rounded.tobytes()
    
    # Decompress exact residual
    residual_packed = ZstdCompressor.decompress(residual_data)
    residual_exact = np.frombuffer(residual_packed, dtype=np.int16)
    
    # Reconstruct: approx + residual
    # Add in int16 space to avoid overflow, then clip to uint8
    reconstructed = approx_rounded.astype(np.int16) + residual_exact[:original_size]
    reconstructed_uint8 = np.clip(reconstructed, 0, 255).astype(np.uint8)
    
    return reconstructed_uint8.tobytes()


# ═══════════════════════════════════════════════════════════════════════════════════════
# CONTAINER FORMAT (.fluid)
# ═══════════════════════════════════════════════════════════════════════════════════════

class FluidContainer:
    """
    .fluid container format for Block-Independent QTT storage.
    
    Layout:
    ┌─────────────────────────────────────────────────────────────────┐
    │ HEADER (64 bytes)                                               │
    │   - Magic (4B): "FLUD"                                          │
    │   - Version (4B)                                                │
    │   - Total Blocks (4B)                                           │
    │   - Original Size (8B)                                          │
    │   - Compressed Size (8B)                                        │
    │   - Block Size (4B)                                             │
    │   - Created At (8B)                                             │
    │   - Index Offset (8B)                                           │
    │   - Metadata Length (4B)                                        │
    │   - Reserved (12B)                                              │
    ├─────────────────────────────────────────────────────────────────┤
    │ METADATA (variable, JSON)                                       │
    ├─────────────────────────────────────────────────────────────────┤
    │ BLOCK DATA                                                      │
    │   [Core_1][Residual_1][Core_2][Residual_2]...                   │
    ├─────────────────────────────────────────────────────────────────┤
    │ BLOCK INDEX (at end for random access)                          │
    │   For each block:                                               │
    │   - Block ID (4B)                                               │
    │   - Original Size (4B)                                          │
    │   - Cores Offset (8B)                                           │
    │   - Cores Size (4B)                                             │
    │   - Residual Offset (8B)                                        │
    │   - Residual Size (4B)                                          │
    │   - Checksum (32B)                                              │
    │   - Grid Shape Length (4B) + dims                               │
    │   - N Cores (4B)                                                │
    │   - Max Rank (4B)                                               │
    │   - Residual Norm (4B)                                          │
    └─────────────────────────────────────────────────────────────────┘
    """
    
    HEADER_SIZE = 64
    
    @staticmethod
    def write(
        output: BinaryIO,
        blocks: List[CompressedBlock],
        metadata: Optional[dict] = None
    ) -> int:
        """Write blocks to .fluid container."""
        # Prepare metadata
        metadata_json = b''
        if metadata:
            metadata_json = json.dumps(metadata).encode('utf-8')
        
        # Calculate sizes
        original_size = sum(b.meta.original_size for b in blocks)
        
        # Write header placeholder
        header_pos = output.tell()
        output.write(b'\x00' * FluidContainer.HEADER_SIZE)
        
        # Write metadata
        output.write(metadata_json)
        
        # Write block data and update offsets
        data_start = output.tell()
        for block in blocks:
            block.meta.cores_offset = output.tell()
            output.write(block.cores_data)
            
            block.meta.residual_offset = output.tell()
            output.write(block.residual_data)
        
        # Write index
        index_offset = output.tell()
        for block in blocks:
            meta = block.meta
            output.write(struct.pack('<I', meta.block_id))
            output.write(struct.pack('<I', meta.original_size))
            output.write(struct.pack('<Q', meta.cores_offset))
            output.write(struct.pack('<I', meta.cores_size))
            output.write(struct.pack('<Q', meta.residual_offset))
            output.write(struct.pack('<I', meta.residual_size))
            output.write(meta.checksum)
            
            # Grid shape
            output.write(struct.pack('<I', len(meta.grid_shape)))
            for dim in meta.grid_shape:
                output.write(struct.pack('<I', dim))
            
            output.write(struct.pack('<I', meta.n_cores))
            output.write(struct.pack('<I', meta.max_rank))
            output.write(struct.pack('<f', meta.residual_norm))
        
        # Final size
        end_pos = output.tell()
        compressed_size = end_pos
        
        # Write header
        output.seek(header_pos)
        output.write(MAGIC_BYTES)
        output.write(struct.pack('<I', VERSION))
        output.write(struct.pack('<I', len(blocks)))
        output.write(struct.pack('<Q', original_size))
        output.write(struct.pack('<Q', compressed_size))
        output.write(struct.pack('<I', BLOCK_SIZE))
        output.write(struct.pack('<d', datetime.now().timestamp()))
        output.write(struct.pack('<Q', index_offset))
        output.write(struct.pack('<I', len(metadata_json)))
        output.write(b'\x00' * 12)  # Reserved
        
        output.seek(end_pos)
        return compressed_size
    
    @staticmethod
    def read_header(input: BinaryIO) -> FluidHeader:
        """Read container header."""
        magic = input.read(4)
        if magic != MAGIC_BYTES:
            raise ValueError(f"Invalid magic bytes: {magic}")
        
        version = struct.unpack('<I', input.read(4))[0]
        total_blocks = struct.unpack('<I', input.read(4))[0]
        original_size = struct.unpack('<Q', input.read(8))[0]
        compressed_size = struct.unpack('<Q', input.read(8))[0]
        block_size = struct.unpack('<I', input.read(4))[0]
        created_at = struct.unpack('<d', input.read(8))[0]
        index_offset = struct.unpack('<Q', input.read(8))[0]
        metadata_len = struct.unpack('<I', input.read(4))[0]
        input.read(12)  # Reserved
        
        # Read metadata
        metadata_json = input.read(metadata_len) if metadata_len > 0 else b''
        
        return FluidHeader(
            magic=magic,
            version=version,
            total_blocks=total_blocks,
            original_size=original_size,
            compressed_size=compressed_size,
            block_size=block_size,
            created_at=created_at,
            metadata_json=metadata_json
        )
    
    @staticmethod
    def read_index(input: BinaryIO, header: FluidHeader) -> List[BlockMeta]:
        """Read block index for random access."""
        # Seek to index
        input.seek(header.compressed_size)
        # Actually need to get index offset from header - let's re-read
        input.seek(FluidContainer.HEADER_SIZE - 12 - 4 - 8)
        index_offset = struct.unpack('<Q', input.read(8))[0]
        
        input.seek(index_offset)
        
        blocks = []
        for _ in range(header.total_blocks):
            block_id = struct.unpack('<I', input.read(4))[0]
            original_size = struct.unpack('<I', input.read(4))[0]
            cores_offset = struct.unpack('<Q', input.read(8))[0]
            cores_size = struct.unpack('<I', input.read(4))[0]
            residual_offset = struct.unpack('<Q', input.read(8))[0]
            residual_size = struct.unpack('<I', input.read(4))[0]
            checksum = input.read(32)
            
            n_dims = struct.unpack('<I', input.read(4))[0]
            grid_shape = tuple(struct.unpack('<I', input.read(4))[0] for _ in range(n_dims))
            
            n_cores = struct.unpack('<I', input.read(4))[0]
            max_rank = struct.unpack('<I', input.read(4))[0]
            residual_norm = struct.unpack('<f', input.read(4))[0]
            
            blocks.append(BlockMeta(
                block_id=block_id,
                original_size=original_size,
                cores_offset=cores_offset,
                cores_size=cores_size,
                residual_offset=residual_offset,
                residual_size=residual_size,
                checksum=checksum,
                grid_shape=grid_shape,
                n_cores=n_cores,
                max_rank=max_rank,
                residual_norm=residual_norm
            ))
        
        return blocks
    
    @staticmethod
    def read_block(
        input: BinaryIO,
        meta: BlockMeta
    ) -> bytes:
        """Read and decompress a single block by index."""
        # Read cores
        input.seek(meta.cores_offset)
        cores_data = input.read(meta.cores_size)
        
        # Read residual
        input.seek(meta.residual_offset)
        residual_data = input.read(meta.residual_size)
        
        # Decompress
        return decompress_block(
            cores_data,
            residual_data,
            meta.grid_shape,
            meta.original_size
        )


# ═══════════════════════════════════════════════════════════════════════════════════════
# STREAMING INGEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════════════════

class FluidIngestEngine:
    """
    Block-Independent QTT Ingest Engine.
    
    Features:
    - Stream & Chunk: Slice into 64MB blocks (optimal for L3 cache)
    - Parallel Processing: Each block is independent
    - Infinite Scale: Only ever hold 64MB in RAM
    - Random Access: Seek to any block instantly
    """
    
    def __init__(
        self,
        max_rank: int = DEFAULT_RANK,
        n_workers: int = 4,
        block_size: int = BLOCK_SIZE
    ):
        self.max_rank = max_rank
        self.n_workers = n_workers
        self.block_size = block_size
        self.stats = {
            'blocks_processed': 0,
            'original_bytes': 0,
            'compressed_bytes': 0,
            'processing_time': 0.0
        }
    
    def _chunk_stream(
        self,
        stream: BinaryIO,
        total_size: Optional[int] = None
    ) -> Iterator[Tuple[int, bytes]]:
        """
        The Slicer: Stream input and yield (block_id, data) tuples.
        
        Independent processing. No global state.
        """
        block_id = 0
        while True:
            chunk = stream.read(self.block_size)
            if not chunk:
                break
            yield block_id, chunk
            block_id += 1
    
    def _process_block(
        self,
        args: Tuple[int, bytes]
    ) -> CompressedBlock:
        """Process single block (for parallel execution)."""
        block_id, data = args
        return compress_block(block_id, data, self.max_rank)
    
    def ingest_file(
        self,
        input_path: Path,
        output_path: Path,
        metadata: Optional[dict] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> dict:
        """
        Ingest a file and produce .fluid output.
        
        Args:
            input_path: Source file path
            output_path: Output .fluid file path
            metadata: Optional metadata dict to embed
            progress_callback: Optional (blocks_done, total_blocks) callback
        
        Returns:
            Statistics dict
        """
        import time
        start_time = time.time()
        
        input_size = input_path.stat().st_size
        total_blocks = (input_size + self.block_size - 1) // self.block_size
        
        print(f"╔══════════════════════════════════════════════════════════════════╗")
        print(f"║  FLUIDELITE BLOCK-INDEPENDENT INGEST                             ║")
        print(f"╠══════════════════════════════════════════════════════════════════╣")
        print(f"║  Input:        {str(input_path)[:45]:<45} ║")
        print(f"║  Size:         {input_size:>15,} bytes                         ║")
        print(f"║  Blocks:       {total_blocks:>15,} × 64MB                       ║")
        print(f"║  Workers:      {self.n_workers:>15,}                                ║")
        print(f"║  Max Rank:     {self.max_rank:>15,}                                ║")
        print(f"╚══════════════════════════════════════════════════════════════════╝")
        
        compressed_blocks = []
        
        with open(input_path, 'rb') as infile:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                # Submit all blocks
                futures = {}
                for block_id, chunk in self._chunk_stream(infile):
                    future = executor.submit(self._process_block, (block_id, chunk))
                    futures[future] = block_id
                
                # Collect results in order
                results = {}
                for future in as_completed(futures):
                    block_id = futures[future]
                    results[block_id] = future.result()
                    
                    if progress_callback:
                        progress_callback(len(results), total_blocks)
                    
                    # Progress
                    pct = len(results) / total_blocks * 100
                    print(f"\r  Processing: {len(results)}/{total_blocks} blocks ({pct:.1f}%)", end='', flush=True)
        
        print()  # Newline after progress
        
        # Sort by block_id
        for block_id in sorted(results.keys()):
            compressed_blocks.append(results[block_id])
        
        # Write container
        print(f"  Writing container...")
        with open(output_path, 'wb') as outfile:
            compressed_size = FluidContainer.write(outfile, compressed_blocks, metadata)
        
        end_time = time.time()
        
        # Statistics
        total_cores_size = sum(b.meta.cores_size for b in compressed_blocks)
        total_residual_size = sum(b.meta.residual_size for b in compressed_blocks)
        
        stats = {
            'input_size': input_size,
            'output_size': compressed_size,
            'compression_ratio': input_size / compressed_size if compressed_size > 0 else 0,
            'total_blocks': total_blocks,
            'cores_size': total_cores_size,
            'residual_size': total_residual_size,
            'processing_time': end_time - start_time,
            'throughput_mbps': (input_size / (1024*1024)) / (end_time - start_time)
        }
        
        print(f"╔══════════════════════════════════════════════════════════════════╗")
        print(f"║  COMPLETE                                                        ║")
        print(f"╠══════════════════════════════════════════════════════════════════╣")
        print(f"║  Input:        {stats['input_size']:>15,} bytes                   ║")
        print(f"║  Output:       {stats['output_size']:>15,} bytes                  ║")
        print(f"║  Ratio:        {stats['compression_ratio']:>15.2f}x                      ║")
        print(f"║  Cores:        {stats['cores_size']:>15,} bytes                   ║")
        print(f"║  Residuals:    {stats['residual_size']:>15,} bytes                ║")
        print(f"║  Time:         {stats['processing_time']:>15.2f} seconds               ║")
        print(f"║  Throughput:   {stats['throughput_mbps']:>15.2f} MB/s                   ║")
        print(f"╚══════════════════════════════════════════════════════════════════╝")
        
        return stats
    
    def ingest_stream(
        self,
        stream: BinaryIO,
        output: BinaryIO,
        total_size: Optional[int] = None,
        metadata: Optional[dict] = None
    ) -> dict:
        """
        Ingest from arbitrary stream (S3, HTTP, pipe, etc.).
        
        This is the "infinite scale" path - never holds more than 64MB.
        """
        import time
        start_time = time.time()
        
        compressed_blocks = []
        bytes_read = 0
        
        for block_id, chunk in self._chunk_stream(stream, total_size):
            # Process immediately - no buffering
            compressed = compress_block(block_id, chunk, self.max_rank)
            compressed_blocks.append(compressed)
            bytes_read += len(chunk)
            
            if total_size:
                pct = bytes_read / total_size * 100
                print(f"\r  Streaming: {bytes_read:,} / {total_size:,} bytes ({pct:.1f}%)", end='', flush=True)
            else:
                print(f"\r  Streaming: {bytes_read:,} bytes, {len(compressed_blocks)} blocks", end='', flush=True)
        
        print()
        
        # Write container
        compressed_size = FluidContainer.write(output, compressed_blocks, metadata)
        
        end_time = time.time()
        
        return {
            'input_size': bytes_read,
            'output_size': compressed_size,
            'compression_ratio': bytes_read / compressed_size if compressed_size > 0 else 0,
            'total_blocks': len(compressed_blocks),
            'processing_time': end_time - start_time
        }


# ═══════════════════════════════════════════════════════════════════════════════════════
# RANDOM ACCESS READER
# ═══════════════════════════════════════════════════════════════════════════════════════

class FluidReader:
    """
    Random-access reader for .fluid files.
    
    The "Teleport" feature: grab any block instantly without decoding the whole file.
    """
    
    def __init__(self, path: Path):
        self.path = path
        self.file = open(path, 'rb')
        self.header = FluidContainer.read_header(self.file)
        self.file.seek(0)
        self.index = FluidContainer.read_index(self.file, self.header)
        
        # Build byte-range lookup
        self._build_offset_map()
    
    def _build_offset_map(self):
        """Map byte ranges to block indices for O(1) lookup."""
        self.block_starts = []
        offset = 0
        for meta in self.index:
            self.block_starts.append(offset)
            offset += meta.original_size
        self.total_size = offset
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def close(self):
        self.file.close()
    
    @property
    def metadata(self) -> Optional[dict]:
        """Get embedded metadata."""
        if self.header.metadata_json:
            return json.loads(self.header.metadata_json.decode('utf-8'))
        return None
    
    def read_block(self, block_id: int) -> bytes:
        """
        Read a single block by ID.
        
        This is O(1) seek + decompress. No scanning.
        """
        if block_id < 0 or block_id >= len(self.index):
            raise IndexError(f"Block {block_id} out of range [0, {len(self.index)})")
        
        meta = self.index[block_id]
        return FluidContainer.read_block(self.file, meta)
    
    def read_range(self, start: int, end: int) -> bytes:
        """
        Read a byte range from the original data.
        
        Automatically maps to the minimal set of blocks needed.
        """
        if start < 0:
            start = 0
        if end > self.total_size:
            end = self.total_size
        if start >= end:
            return b''
        
        # Find which blocks we need
        import bisect
        
        start_block = bisect.bisect_right(self.block_starts, start) - 1
        if start_block < 0:
            start_block = 0
        
        end_block = bisect.bisect_right(self.block_starts, end - 1)
        if end_block > len(self.index):
            end_block = len(self.index)
        
        # Read and concatenate
        result = b''
        for block_id in range(start_block, end_block):
            block_data = self.read_block(block_id)
            block_start = self.block_starts[block_id]
            block_end = block_start + len(block_data)
            
            # Slice within block
            local_start = max(0, start - block_start)
            local_end = min(len(block_data), end - block_start)
            
            result += block_data[local_start:local_end]
        
        return result
    
    def extract_all(self, output_path: Path) -> int:
        """Extract entire file to disk."""
        bytes_written = 0
        with open(output_path, 'wb') as out:
            for block_id in range(len(self.index)):
                data = self.read_block(block_id)
                out.write(data)
                bytes_written += len(data)
        return bytes_written
    
    def verify(self) -> Tuple[bool, List[str]]:
        """Verify all block checksums."""
        errors = []
        for block_id, meta in enumerate(self.index):
            data = self.read_block(block_id)
            actual_checksum = hashlib.sha256(data).digest()
            if actual_checksum != meta.checksum:
                errors.append(f"Block {block_id}: checksum mismatch")
        return len(errors) == 0, errors


# ═══════════════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='FluidElite Block-Independent QTT Ingest Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Compress a file:
    python fluidelite_ingest.py compress input.bin output.fluid
  
  Extract a file:
    python fluidelite_ingest.py extract archive.fluid output.bin
  
  Read specific block:
    python fluidelite_ingest.py read archive.fluid --block 14
  
  Read byte range:
    python fluidelite_ingest.py read archive.fluid --start 1000000 --end 2000000
  
  Verify integrity:
    python fluidelite_ingest.py verify archive.fluid
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Compress command
    compress_parser = subparsers.add_parser('compress', help='Compress a file')
    compress_parser.add_argument('input', type=Path, help='Input file')
    compress_parser.add_argument('output', type=Path, help='Output .fluid file')
    compress_parser.add_argument('--rank', type=int, default=DEFAULT_RANK, help='Max TT rank')
    compress_parser.add_argument('--workers', type=int, default=4, help='Parallel workers')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract a .fluid file')
    extract_parser.add_argument('input', type=Path, help='Input .fluid file')
    extract_parser.add_argument('output', type=Path, help='Output file')
    
    # Read command
    read_parser = subparsers.add_parser('read', help='Read from .fluid file')
    read_parser.add_argument('input', type=Path, help='Input .fluid file')
    read_parser.add_argument('--block', type=int, help='Block ID to read')
    read_parser.add_argument('--start', type=int, help='Start byte offset')
    read_parser.add_argument('--end', type=int, help='End byte offset')
    read_parser.add_argument('--output', type=Path, help='Output file (default: stdout)')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify .fluid file integrity')
    verify_parser.add_argument('input', type=Path, help='Input .fluid file')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show .fluid file info')
    info_parser.add_argument('input', type=Path, help='Input .fluid file')
    
    args = parser.parse_args()
    
    if args.command == 'compress':
        engine = FluidIngestEngine(max_rank=args.rank, n_workers=args.workers)
        engine.ingest_file(args.input, args.output)
    
    elif args.command == 'extract':
        with FluidReader(args.input) as reader:
            bytes_written = reader.extract_all(args.output)
            print(f"Extracted {bytes_written:,} bytes to {args.output}")
    
    elif args.command == 'read':
        with FluidReader(args.input) as reader:
            if args.block is not None:
                data = reader.read_block(args.block)
            elif args.start is not None and args.end is not None:
                data = reader.read_range(args.start, args.end)
            else:
                print("Error: specify --block or --start/--end")
                return 1
            
            if args.output:
                with open(args.output, 'wb') as f:
                    f.write(data)
                print(f"Wrote {len(data):,} bytes to {args.output}")
            else:
                import sys
                sys.stdout.buffer.write(data)
    
    elif args.command == 'verify':
        with FluidReader(args.input) as reader:
            ok, errors = reader.verify()
            if ok:
                print(f"✓ All {len(reader.index)} blocks verified successfully")
            else:
                print(f"✗ Verification failed:")
                for err in errors:
                    print(f"  - {err}")
                return 1
    
    elif args.command == 'info':
        with FluidReader(args.input) as reader:
            h = reader.header
            print(f"╔══════════════════════════════════════════════════════════════════╗")
            print(f"║  FLUID CONTAINER INFO                                            ║")
            print(f"╠══════════════════════════════════════════════════════════════════╣")
            print(f"║  Version:      {h.version:>15}                                ║")
            print(f"║  Blocks:       {h.total_blocks:>15,}                            ║")
            print(f"║  Original:     {h.original_size:>15,} bytes                   ║")
            print(f"║  Compressed:   {h.compressed_size:>15,} bytes                 ║")
            print(f"║  Ratio:        {h.original_size/h.compressed_size:>15.2f}x                      ║")
            print(f"║  Block Size:   {h.block_size:>15,} bytes                      ║")
            print(f"║  Created:      {datetime.fromtimestamp(h.created_at).isoformat()[:19]:>19}         ║")
            print(f"╚══════════════════════════════════════════════════════════════════╝")
            
            if reader.metadata:
                print(f"\nMetadata: {json.dumps(reader.metadata, indent=2)}")
    
    return 0


if __name__ == '__main__':
    exit(main())
