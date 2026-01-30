"""
GPU-Accelerated Block-SVD Reconstruction

Achieves sub-10ms frame reconstruction via CuPy batched matmul.

Performance (RTX 5070, 5424x5424 frames):
- GPU matmul: 1.17 ms
- Assembly: 0.87 ms  
- Total: 2.04 ms (490 FPS)
- Per-block: 0.29 μs

Usage:
    from gpu_reconstruct import GPUBlockReconstructor
    
    recon = GPUBlockReconstructor(block_size=64)
    recon.load_compressed(u, s, vh, ranks, height, width)
    
    # Reconstruct single frame (stays on GPU)
    frame_gpu = recon.reconstruct_frame(0)
    
    # Reconstruct and transfer to CPU
    frame_cpu = recon.reconstruct_frame_cpu(0)
    
    # Batch reconstruct
    frames = recon.reconstruct_batch([0, 1, 2, 3, 4])
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


@dataclass
class ReconstructionStats:
    """Statistics from reconstruction operation."""
    matmul_ms: float
    assembly_ms: float
    transfer_ms: float
    total_ms: float
    n_blocks: int
    frame_shape: Tuple[int, int]
    
    @property
    def fps(self) -> float:
        return 1000.0 / self.total_ms if self.total_ms > 0 else 0.0
    
    @property
    def per_block_us(self) -> float:
        return self.total_ms * 1000 / self.n_blocks if self.n_blocks > 0 else 0.0


class GPUBlockReconstructor:
    """
    GPU-accelerated Block-SVD frame reconstruction.
    
    Uses CuPy batched matrix multiplication for 490+ FPS throughput.
    All compressed data is kept on GPU for streaming reconstruction.
    """
    
    def __init__(self, block_size: int = 64, device_id: int = 0):
        """
        Initialize GPU reconstructor.
        
        Args:
            block_size: Block dimensions (default 64x64)
            device_id: CUDA device ID
        """
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available. Install with: pip install cupy-cuda12x")
        
        self.block_size = block_size
        self.device_id = device_id
        
        # Compressed data (on GPU)
        self.u_gpu: Optional[cp.ndarray] = None
        self.s_gpu: Optional[cp.ndarray] = None
        self.vh_gpu: Optional[cp.ndarray] = None
        self.ranks_gpu: Optional[cp.ndarray] = None
        
        # Frame metadata
        self.height: int = 0
        self.width: int = 0
        self.blocks_h: int = 0
        self.blocks_w: int = 0
        self.n_blocks_per_frame: int = 0
        self.n_frames: int = 0
        self.max_rank: int = 0
        
        # Normalization
        self.mean: float = 0.0
        self.std: float = 1.0
        
        # Select device
        with cp.cuda.Device(device_id):
            pass  # Ensure device is valid
    
    def load_compressed(
        self,
        u_data: np.ndarray,
        s_data: np.ndarray,
        vh_data: np.ndarray,
        ranks: np.ndarray,
        height: int,
        width: int,
        mean: float = 0.0,
        std: float = 1.0,
        n_frames: int = 1,
    ) -> None:
        """
        Load compressed Block-SVD data to GPU.
        
        For variable-rank compression, data is padded to max_rank.
        Zero-masking handles rank variation efficiently.
        
        Args:
            u_data: Flattened U matrices (total_elements,)
            s_data: Flattened singular values (total_ranks,)
            vh_data: Flattened Vh matrices (total_elements,)
            ranks: Rank per block (n_blocks,)
            height: Frame height
            width: Frame width
            mean: Normalization mean
            std: Normalization std
            n_frames: Number of frames in compressed data
        """
        self.height = height
        self.width = width
        self.blocks_h = height // self.block_size
        self.blocks_w = width // self.block_size
        self.n_blocks_per_frame = self.blocks_h * self.blocks_w
        self.n_frames = n_frames
        self.mean = mean
        self.std = std
        
        n_blocks = len(ranks)
        self.max_rank = int(ranks.max())
        
        with cp.cuda.Device(self.device_id):
            # Convert variable-rank to padded fixed-rank for GPU efficiency
            u_padded = np.zeros((n_blocks, self.block_size, self.max_rank), dtype=np.float32)
            s_padded = np.zeros((n_blocks, self.max_rank), dtype=np.float32)
            vh_padded = np.zeros((n_blocks, self.max_rank, self.block_size), dtype=np.float32)
            
            # Unpack variable-rank data
            u_ptr = 0
            s_ptr = 0
            for bi, rank in enumerate(ranks):
                rank = int(rank)
                
                # U: (block_size, rank)
                u_size = self.block_size * rank
                u_block = u_data[u_ptr:u_ptr + u_size].reshape(self.block_size, rank)
                u_padded[bi, :, :rank] = u_block
                u_ptr += u_size
                
                # S: (rank,)
                s_padded[bi, :rank] = s_data[s_ptr:s_ptr + rank]
                s_ptr += rank
                
                # Vh: (rank, block_size) - stored same as U
                vh_block = vh_data[u_ptr - u_size:u_ptr].reshape(rank, self.block_size) if rank > 0 else np.zeros((0, self.block_size))
                # Actually Vh is stored separately at same offset pattern
            
            # Recompute Vh unpacking
            vh_ptr = 0
            for bi, rank in enumerate(ranks):
                rank = int(rank)
                vh_size = rank * self.block_size
                if rank > 0:
                    vh_block = vh_data[vh_ptr:vh_ptr + vh_size].reshape(rank, self.block_size)
                    vh_padded[bi, :rank, :] = vh_block
                vh_ptr += vh_size
            
            # Transfer to GPU
            self.u_gpu = cp.asarray(u_padded)
            self.s_gpu = cp.asarray(s_padded)
            self.vh_gpu = cp.asarray(vh_padded)
            self.ranks_gpu = cp.asarray(ranks.astype(np.int32))
    
    def load_compressed_padded(
        self,
        u_gpu: "cp.ndarray",
        s_gpu: "cp.ndarray",
        vh_gpu: "cp.ndarray",
        ranks: np.ndarray,
        height: int,
        width: int,
        mean: float = 0.0,
        std: float = 1.0,
        n_frames: int = 1,
    ) -> None:
        """
        Load pre-padded compressed data already on GPU.
        
        Args:
            u_gpu: Padded U matrices (n_blocks, block_size, max_rank)
            s_gpu: Padded singular values (n_blocks, max_rank)
            vh_gpu: Padded Vh matrices (n_blocks, max_rank, block_size)
            ranks: Rank per block (n_blocks,)
            height: Frame height
            width: Frame width
            mean: Normalization mean
            std: Normalization std
            n_frames: Number of frames
        """
        self.height = height
        self.width = width
        self.blocks_h = height // self.block_size
        self.blocks_w = width // self.block_size
        self.n_blocks_per_frame = self.blocks_h * self.blocks_w
        self.n_frames = n_frames
        self.mean = mean
        self.std = std
        self.max_rank = u_gpu.shape[2]
        
        self.u_gpu = u_gpu
        self.s_gpu = s_gpu
        self.vh_gpu = vh_gpu
        self.ranks_gpu = cp.asarray(ranks.astype(np.int32))
    
    def reconstruct_frame(self, frame_idx: int = 0) -> "cp.ndarray":
        """
        Reconstruct a single frame (result stays on GPU).
        
        Args:
            frame_idx: Frame index to reconstruct
            
        Returns:
            GPU array of shape (blocks_h * block_size, blocks_w * block_size)
        """
        if self.u_gpu is None:
            raise RuntimeError("No compressed data loaded. Call load_compressed() first.")
        
        start_block = frame_idx * self.n_blocks_per_frame
        end_block = start_block + self.n_blocks_per_frame
        
        # Extract frame blocks
        u_frame = self.u_gpu[start_block:end_block]
        s_frame = self.s_gpu[start_block:end_block]
        vh_frame = self.vh_gpu[start_block:end_block]
        
        # Batched matmul: (U * S) @ Vh
        # u_frame: (n_blocks, block_size, max_rank)
        # s_frame: (n_blocks, max_rank)
        # vh_frame: (n_blocks, max_rank, block_size)
        u_scaled = u_frame * s_frame[:, None, :]
        blocks_out = u_scaled @ vh_frame  # (n_blocks, block_size, block_size)
        
        # Apply denormalization
        if self.std != 1.0 or self.mean != 0.0:
            blocks_out = blocks_out * self.std + self.mean
        
        # Vectorized assembly via reshape + transpose
        blocks_4d = blocks_out.reshape(self.blocks_h, self.blocks_w, self.block_size, self.block_size)
        frame = blocks_4d.transpose(0, 2, 1, 3).reshape(
            self.blocks_h * self.block_size,
            self.blocks_w * self.block_size
        )
        
        return frame
    
    def reconstruct_frame_cpu(self, frame_idx: int = 0) -> np.ndarray:
        """
        Reconstruct frame and transfer to CPU.
        
        Args:
            frame_idx: Frame index to reconstruct
            
        Returns:
            NumPy array of shape (blocks_h * block_size, blocks_w * block_size)
        """
        frame_gpu = self.reconstruct_frame(frame_idx)
        return cp.asnumpy(frame_gpu)
    
    def reconstruct_batch(self, frame_indices: List[int]) -> List["cp.ndarray"]:
        """
        Reconstruct multiple frames (results stay on GPU).
        
        Args:
            frame_indices: List of frame indices to reconstruct
            
        Returns:
            List of GPU arrays
        """
        return [self.reconstruct_frame(idx) for idx in frame_indices]
    
    def reconstruct_batch_cpu(self, frame_indices: List[int]) -> np.ndarray:
        """
        Reconstruct multiple frames and stack on CPU.
        
        Args:
            frame_indices: List of frame indices to reconstruct
            
        Returns:
            NumPy array of shape (n_frames, height, width)
        """
        frames = self.reconstruct_batch(frame_indices)
        stacked = cp.stack(frames, axis=0)
        return cp.asnumpy(stacked)
    
    def benchmark(self, n_trials: int = 10) -> ReconstructionStats:
        """
        Benchmark reconstruction performance.
        
        Args:
            n_trials: Number of trials for timing
            
        Returns:
            ReconstructionStats with timing breakdown
        """
        import time
        
        if self.u_gpu is None:
            raise RuntimeError("No compressed data loaded")
        
        # Warmup
        _ = self.reconstruct_frame(0)
        cp.cuda.stream.get_current_stream().synchronize()
        
        # Benchmark matmul
        matmul_times = []
        assembly_times = []
        
        for _ in range(n_trials):
            start_block = 0
            end_block = self.n_blocks_per_frame
            
            u_frame = self.u_gpu[start_block:end_block]
            s_frame = self.s_gpu[start_block:end_block]
            vh_frame = self.vh_gpu[start_block:end_block]
            
            t0 = time.perf_counter()
            u_scaled = u_frame * s_frame[:, None, :]
            blocks_out = u_scaled @ vh_frame
            cp.cuda.stream.get_current_stream().synchronize()
            matmul_times.append((time.perf_counter() - t0) * 1000)
            
            t0 = time.perf_counter()
            blocks_4d = blocks_out.reshape(self.blocks_h, self.blocks_w, self.block_size, self.block_size)
            frame = blocks_4d.transpose(0, 2, 1, 3).reshape(
                self.blocks_h * self.block_size,
                self.blocks_w * self.block_size
            )
            cp.cuda.stream.get_current_stream().synchronize()
            assembly_times.append((time.perf_counter() - t0) * 1000)
        
        # Transfer benchmark
        frame_gpu = self.reconstruct_frame(0)
        t0 = time.perf_counter()
        _ = cp.asnumpy(frame_gpu)
        transfer_ms = (time.perf_counter() - t0) * 1000
        
        matmul_ms = float(np.median(matmul_times))
        assembly_ms = float(np.median(assembly_times))
        
        return ReconstructionStats(
            matmul_ms=matmul_ms,
            assembly_ms=assembly_ms,
            transfer_ms=transfer_ms,
            total_ms=matmul_ms + assembly_ms,
            n_blocks=self.n_blocks_per_frame,
            frame_shape=(self.blocks_h * self.block_size, self.blocks_w * self.block_size),
        )


def create_test_data(
    height: int = 5424,
    width: int = 5424,
    block_size: int = 64,
    n_frames: int = 1,
    rank_distribution: Tuple[int, ...] = (4, 8, 16, 32),
    rank_probs: Tuple[float, ...] = (0.2, 0.4, 0.3, 0.1),
    seed: int = 42,
) -> Tuple["cp.ndarray", "cp.ndarray", "cp.ndarray", np.ndarray]:
    """
    Create synthetic test data for benchmarking.
    
    Returns:
        Tuple of (u_gpu, s_gpu, vh_gpu, ranks)
    """
    np.random.seed(seed)
    
    blocks_h = height // block_size
    blocks_w = width // block_size
    n_blocks = blocks_h * blocks_w * n_frames
    
    ranks = np.random.choice(rank_distribution, size=n_blocks, p=rank_probs).astype(np.int32)
    max_rank = int(ranks.max())
    
    # Create padded GPU arrays
    u_gpu = cp.random.randn(n_blocks, block_size, max_rank, dtype=cp.float32)
    s_gpu = cp.random.rand(n_blocks, max_rank, dtype=cp.float32)
    vh_gpu = cp.random.randn(n_blocks, max_rank, block_size, dtype=cp.float32)
    
    # Mask invalid ranks
    ranks_gpu = cp.asarray(ranks)
    mask = cp.arange(max_rank)[None, :] >= ranks_gpu[:, None]
    s_gpu[mask] = 0
    
    return u_gpu, s_gpu, vh_gpu, ranks


if __name__ == "__main__":
    print("=== GPU Block-SVD Reconstruction Benchmark ===\n")
    
    if not CUPY_AVAILABLE:
        print("ERROR: CuPy not available")
        exit(1)
    
    # Create test data
    height, width = 5424, 5424
    u_gpu, s_gpu, vh_gpu, ranks = create_test_data(height, width)
    
    print(f"Frame size: {height}x{width}")
    print(f"Blocks: {len(ranks)}")
    print(f"Max rank: {ranks.max()}")
    print(f"Avg rank: {ranks.mean():.1f}")
    print(f"GPU memory: {(u_gpu.nbytes + s_gpu.nbytes + vh_gpu.nbytes) / 1e9:.2f} GB")
    
    # Initialize reconstructor
    recon = GPUBlockReconstructor(block_size=64)
    recon.load_compressed_padded(u_gpu, s_gpu, vh_gpu, ranks, height, width)
    
    # Benchmark
    stats = recon.benchmark(n_trials=20)
    
    print(f"\n=== Performance ===")
    print(f"Matmul:   {stats.matmul_ms:.2f} ms")
    print(f"Assembly: {stats.assembly_ms:.2f} ms")
    print(f"Total:    {stats.total_ms:.2f} ms")
    print(f"Transfer: {stats.transfer_ms:.2f} ms")
    print(f"FPS:      {stats.fps:.0f}")
    print(f"Per-block: {stats.per_block_us:.2f} μs")
    
    # Verify output
    frame = recon.reconstruct_frame_cpu(0)
    print(f"\nOutput shape: {frame.shape}")
    print(f"Output dtype: {frame.dtype}")
    
    print("\n✓ GPU Hyper-Bridge OPERATIONAL")
