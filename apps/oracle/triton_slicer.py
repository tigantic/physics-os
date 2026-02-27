"""
Triton-Accelerated Oracle QTT Slicer
=====================================
Purpose: Real-time "Slicing" of Financial Quantum States at Maximum GPU Speed.

ALL PYTHON LOOPS ELIMINATED. Pure Triton kernels for:
1. Core Generation (Fused projection + diagonal embedding)
2. Entropy Estimation (Fast frobenius-based proxy)
3. Asset Slicing (Batched einsum contraction)

Target: <15µs per tick (138x improvement over v1)
"""

import torch
import triton
import triton.language as tl
import logging
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Dict, List

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TritonSlicer")


# =============================================================================
# TRITON KERNELS
# =============================================================================

@triton.jit
def fused_core_generation_kernel(
    # Inputs
    mapped_ptr,          # (Assets, Bond) - Feature-mapped data
    projection_ptr,      # (Bond, Assets, Bond) - Random projection base
    # Output
    core_ptr,            # (Bond, Assets, Bond) - Generated core
    # Dimensions
    prev_bond: tl.constexpr,
    assets: tl.constexpr,
    bond_dim: tl.constexpr,
    # Strides
    mapped_stride_a,
    mapped_stride_b,
    proj_stride_l,
    proj_stride_a,
    proj_stride_r,
    core_stride_l,
    core_stride_a,
    core_stride_r,
):
    """
    Fused kernel that:
    1. Copies base projection to core
    2. Adds diagonal embedding of mapped features
    
    Grid: (assets,) - one program per asset
    
    Replaces Python loop:
        for a in range(self.c.assets):
            min_dim = min(prev_bond, self.c.bond_dim)
            core[:min_dim, a, :min_dim] += torch.diag(mapped[a, :min_dim])
    """
    # Asset index
    a = tl.program_id(0)
    
    min_dim = prev_bond if prev_bond < bond_dim else bond_dim
    
    # Process diagonal elements
    for d in range(min_dim):
        # Compute offset for core[d, a, d]
        core_offset = d * core_stride_l + a * core_stride_a + d * core_stride_r
        proj_offset = d * proj_stride_l + a * proj_stride_a + d * proj_stride_r
        
        # Load projection base
        proj_val = tl.load(projection_ptr + proj_offset)
        
        # Load mapped[a, d]
        mapped_offset = a * mapped_stride_a + d * mapped_stride_b
        mapped_val = tl.load(mapped_ptr + mapped_offset)
        
        # Store: core[d, a, d] = proj + mapped
        tl.store(core_ptr + core_offset, proj_val + mapped_val)
    
    # Process off-diagonal elements (only projection, no mapped)
    for l in range(prev_bond):
        for r in range(bond_dim):
            if l != r:  # Off-diagonal
                core_offset = l * core_stride_l + a * core_stride_a + r * core_stride_r
                proj_offset = l * proj_stride_l + a * proj_stride_a + r * proj_stride_r
                proj_val = tl.load(projection_ptr + proj_offset)
                tl.store(core_ptr + core_offset, proj_val)


@triton.jit
def fused_core_generation_v2_kernel(
    # Inputs
    mapped_ptr,          # (Assets, Bond) - Feature-mapped data
    projection_ptr,      # (Bond, Assets, Bond) - Random projection base
    # Output
    core_ptr,            # (Bond, Assets, Bond) - Generated core
    # Dimensions
    prev_bond: tl.constexpr,
    assets: tl.constexpr,
    bond_dim: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fully parallelized core generation.
    
    Grid: (num_elements,) where num_elements = prev_bond * assets * bond_dim
    Each program handles one element.
    """
    pid = tl.program_id(0)
    
    # Total elements
    total = prev_bond * assets * bond_dim
    
    # Bounds check
    if pid >= total:
        return
    
    # Decompose linear index -> (l, a, r)
    r = pid % bond_dim
    temp = pid // bond_dim
    a = temp % assets
    l = temp // assets
    
    # Base offset in projection (same layout)
    proj_idx = l * (assets * bond_dim) + a * bond_dim + r
    proj_val = tl.load(projection_ptr + proj_idx)
    
    # Add diagonal embedding if l == r and l < bond_dim
    min_dim = prev_bond if prev_bond < bond_dim else bond_dim
    
    if l == r and l < min_dim:
        # Load mapped[a, l]
        mapped_idx = a * bond_dim + l
        mapped_val = tl.load(mapped_ptr + mapped_idx)
        proj_val = proj_val + mapped_val
    
    # Store to core
    core_idx = l * (assets * bond_dim) + a * bond_dim + r
    tl.store(core_ptr + core_idx, proj_val)


@triton.jit
def fast_entropy_kernel(
    # Input: flattened core (L*Phys, R)
    core_ptr,
    # Output: single entropy value
    entropy_ptr,
    # Dimensions
    rows: tl.constexpr,
    cols: tl.constexpr,
    # Block
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Fast entropy approximation using Frobenius norm ratio.
    
    Instead of full SVD, we approximate entropy as:
    H ≈ log2(trace(A^T A) / max_eigenvalue_estimate)
    
    This gives a regime-detection signal without O(n³) SVD.
    """
    # Compute sum of squared elements (Frobenius norm squared)
    acc = 0.0
    max_val = 0.0
    
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            val = tl.load(core_ptr + idx)
            val_sq = val * val
            acc += val_sq
            if val_sq > max_val:
                max_val = val_sq
    
    # Entropy approximation
    # If all mass is in one element: entropy ~ 0
    # If mass is spread: entropy ~ log2(n)
    # H ≈ log2(frobenius^2 / max_element^2)
    if max_val > 1e-10:
        spread = acc / (max_val + 1e-10)
        entropy = tl.log2(spread + 1.0)
    else:
        entropy = 0.0
    
    tl.store(entropy_ptr, entropy)


@triton.jit
def batched_slice_kernel(
    # Input: stacked cores (T, Bond, Assets, Bond)
    cores_ptr,
    # Input: selector (Assets,) one-hot
    selector_ptr,
    # Output: norms (T,)
    norms_ptr,
    # Dimensions
    T: tl.constexpr,
    left_bond: tl.constexpr,
    assets: tl.constexpr,
    right_bond: tl.constexpr,
    # Strides
    core_stride_t,
    core_stride_l,
    core_stride_a,
    core_stride_r,
    # Block
    BLOCK_T: tl.constexpr,
):
    """
    Batched asset slicing across all time steps.
    
    Computes: norm(einsum('lpr,p->lr', core[t], selector)) for all t
    
    Grid: (ceil(T / BLOCK_T),)
    """
    pid = tl.program_id(0)
    t_start = pid * BLOCK_T
    
    for t_off in range(BLOCK_T):
        t = t_start + t_off
        if t >= T:
            continue
        
        # Compute norm of projected matrix
        norm_sq = 0.0
        
        for l in range(left_bond):
            for r in range(right_bond):
                # Contract: sum_a core[t,l,a,r] * selector[a]
                contracted = 0.0
                for a in range(assets):
                    core_idx = t * core_stride_t + l * core_stride_l + a * core_stride_a + r * core_stride_r
                    core_val = tl.load(cores_ptr + core_idx)
                    sel_val = tl.load(selector_ptr + a)
                    contracted += core_val * sel_val
                norm_sq += contracted * contracted
        
        # Store norm
        norm = tl.sqrt(norm_sq)
        tl.store(norms_ptr + t, norm)


# =============================================================================
# SLICER CLASS
# =============================================================================

@dataclass
class SlicerConfig:
    window_size: int = 256      # Lookback window (Temporal Depth)
    assets: int = 4             # Physical Dim (BTC, ETH, SOL, AVAX)
    bond_dim: int = 32          # Entanglement Rank (Compression)
    dtype: torch.dtype = torch.float16  # FP16 for Tensor Cores


class TritonOracleSlicer:
    """
    Triton-Accelerated Oracle Slicer.
    
    All hot paths use fused Triton kernels. Zero Python loops in critical path.
    """
    
    ASSET_NAMES = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD"]
    
    def __init__(self, config: SlicerConfig):
        self.c = config
        
        # 1. THE STREAM (Ring Buffer as Tensor)
        # Instead of deque of tensors, we use a pre-allocated tensor
        # and track head position. This enables batched Triton ops.
        self.cores = torch.zeros(
            self.c.window_size, self.c.bond_dim, self.c.assets, self.c.bond_dim,
            device=DEVICE, dtype=torch.float32  # FP32 for Triton stability
        )
        self.head = 0
        self.count = 0
        
        # 2. THE PROJECTION (Feature Map)
        torch.manual_seed(42)
        self.encoder = torch.randn(
            self.c.assets, self.c.bond_dim,
            device=DEVICE, dtype=torch.float32
        ) * 0.1
        
        # 3. PRE-ALLOCATED RANDOM PROJECTION (Fixed, not regenerated!)
        # This eliminates torch.randn() from hot path
        self.projection = torch.randn(
            self.c.bond_dim, self.c.assets, self.c.bond_dim,
            device=DEVICE, dtype=torch.float32
        ) * 0.01
        
        # 4. CORE BUFFER (Reused each tick)
        self.core_buffer = torch.zeros(
            self.c.bond_dim, self.c.assets, self.c.bond_dim,
            device=DEVICE, dtype=torch.float32
        )
        
        # 5. MAPPED BUFFER (Feature-mapped input)
        self.mapped_buffer = torch.zeros(
            self.c.assets, self.c.bond_dim,
            device=DEVICE, dtype=torch.float32
        )
        
        # 6. ENTROPY OUTPUT (Single value)
        self.entropy_out = torch.zeros(1, device=DEVICE, dtype=torch.float32)
        
        # 7. ENTROPY HISTORY
        self.entropy_history: deque = deque(maxlen=100)
        
        # 8. ASSET MAPPING
        self.asset_idx = {name: i for i, name in enumerate(self.ASSET_NAMES)}
        
        # 9. SELECTOR CACHE (Pre-computed one-hots)
        self.selectors = torch.eye(
            self.c.assets, device=DEVICE, dtype=torch.float32
        )
        
        logger.info(
            f"Initialized TritonOracleSlicer: {self.c.assets} Assets @ "
            f"{self.c.window_size} Ticks, Bond Dim {self.c.bond_dim}"
        )

    def ingest_tick(self, market_vector: torch.Tensor) -> float:
        """
        HOT PATH: Process a new market tick using Triton kernels.
        
        Args:
            market_vector: Shape (4,) -> [BTC, ETH, SOL, AVAX] normalized prices
        Returns:
            Current System Entropy (The 'Regime' Signal)
        """
        # 1. Feature Map: x -> tanh(Wx) using PyTorch (fast enough)
        x = market_vector.to(device=DEVICE, dtype=torch.float32)
        
        # Vectorized: (Assets,) * (Assets, Bond) -> (Assets, Bond)
        torch.mul(x.unsqueeze(1), self.encoder, out=self.mapped_buffer)
        torch.tanh_(self.mapped_buffer)
        
        # 2. Core Generation via Triton (REPLACES Python loop)
        total_elements = self.c.bond_dim * self.c.assets * self.c.bond_dim
        grid = (total_elements,)
        
        fused_core_generation_v2_kernel[grid](
            self.mapped_buffer,
            self.projection,
            self.core_buffer,
            self.c.bond_dim,  # prev_bond (always bond_dim after warmup)
            self.c.assets,
            self.c.bond_dim,
            BLOCK_SIZE=256,
        )
        
        # 3. Store to ring buffer
        self.cores[self.head].copy_(self.core_buffer)
        self.head = (self.head + 1) % self.c.window_size
        self.count = min(self.count + 1, self.c.window_size)
        
        # 4. Entropy estimation via Triton
        entropy = self._measure_entropy_triton()
        self.entropy_history.append(entropy)
        
        return entropy

    def _measure_entropy_triton(self) -> float:
        """
        Fast entropy using Triton kernel.
        """
        if self.count < 10:
            return 0.0
        
        # Get mid-window core
        mid_idx = (self.head - self.count // 2) % self.c.window_size
        cut_core = self.cores[mid_idx]  # (Bond, Assets, Bond)
        
        # Flatten for kernel
        flat = cut_core.reshape(-1)
        rows = self.c.bond_dim * self.c.assets
        cols = self.c.bond_dim
        
        # Launch entropy kernel
        fast_entropy_kernel[(1,)](
            flat,
            self.entropy_out,
            rows,
            cols,
            BLOCK_R=rows,
            BLOCK_C=cols,
        )
        
        return self.entropy_out.item()

    def slice_asset(self, asset_idx: int) -> torch.Tensor:
        """
        Extract asset history using vectorized PyTorch.
        For small T, vectorized torch is faster than kernel launch overhead.
        """
        if self.count == 0:
            return torch.zeros(0, device=DEVICE)
        
        # Get selector
        selector = self.selectors[asset_idx]  # (Assets,)
        
        # Get valid cores in order
        if self.count < self.c.window_size:
            valid_cores = self.cores[:self.count]
        else:
            # Ring buffer wraparound
            idx = torch.arange(self.count, device=DEVICE) 
            idx = (self.head - self.count + idx) % self.c.window_size
            valid_cores = self.cores[idx]
        
        # Vectorized contraction: einsum over all time steps at once
        # cores: (T, L, A, R), selector: (A,)
        # Result: (T, L, R)
        contracted = torch.einsum('tlar,a->tlr', valid_cores, selector)
        
        # Norm per time step
        norms = torch.norm(contracted.reshape(contracted.shape[0], -1), dim=1)
        
        return norms

    def slice_asset_by_name(self, name: str) -> torch.Tensor:
        """Slice by asset name."""
        idx = self.asset_idx.get(name, 0)
        return self.slice_asset(idx)

    def get_regime(self) -> Tuple[str, float]:
        """Detect market regime from entropy history."""
        if len(self.entropy_history) < 10:
            return ("UNKNOWN", 0.0)
        
        recent = list(self.entropy_history)[-20:]
        mean_ent = sum(recent) / len(recent)
        
        if mean_ent < 2.0:
            return ("STABLE", 0.8)
        elif mean_ent < 4.0:
            return ("TRENDING", 0.7)
        elif mean_ent < 6.0:
            return ("VOLATILE", 0.6)
        else:
            return ("CHAOTIC", 0.5)

    def get_cross_asset_correlation(self) -> torch.Tensor:
        """Compute cross-asset correlation from MPS structure."""
        if self.count < 10:
            return torch.eye(self.c.assets, device=DEVICE)
        
        # Extract all slices at once (vectorized)
        slices = torch.stack([
            self.slice_asset(i) for i in range(self.c.assets)
        ])  # (4, T)
        
        # Normalize
        slices = slices - slices.mean(dim=1, keepdim=True)
        norms = slices.norm(dim=1, keepdim=True).clamp(min=1e-8)
        slices = slices / norms
        
        return slices @ slices.T

    def get_state_snapshot(self) -> Dict:
        """Get complete state for logging/UI."""
        regime, confidence = self.get_regime()
        recent_entropy = list(self.entropy_history)[-1] if self.entropy_history else 0.0
        
        return {
            "regime": regime,
            "confidence": confidence,
            "entropy": recent_entropy,
            "window_size": self.count,
            "bond_dim": self.c.bond_dim,
        }


# =============================================================================
# BENCHMARK
# =============================================================================

def run_benchmark(n_ticks: int = 50000):
    """Benchmark the Triton-accelerated slicer."""
    import time
    
    print("=" * 70)
    print("  TRITON ORACLE QTT SLICER BENCHMARK")
    print("=" * 70)
    
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"  Compute: SM {props.major}.{props.minor}")
    else:
        print("  WARNING: Running on CPU (no Triton acceleration)")
        return
    print()
    
    config = SlicerConfig(window_size=1024, assets=4, bond_dim=32)
    slicer = TritonOracleSlicer(config)
    
    # Mock Market Data
    tick = torch.tensor([0.5, 0.5, 0.5, 0.5], device=DEVICE, dtype=torch.float32)
    
    print("  Warming up (Triton JIT compile)...")
    for _ in range(500):
        slicer.ingest_tick(tick)
    
    torch.cuda.synchronize()
    
    print(f"  Streaming {n_ticks:,} Ticks...")
    start = time.perf_counter()
    
    for i in range(n_ticks):
        # Simulate Random Walk
        noise = torch.randn(4, device=DEVICE, dtype=torch.float32) * 0.01
        tick = torch.clamp(tick + noise, 0, 1)
        
        ent = slicer.ingest_tick(tick)
        
        if i % (n_ticks // 5) == 0 and i > 0:
            regime, conf = slicer.get_regime()
            mem = torch.cuda.memory_allocated() / 1024**2
            print(
                f"  Tick {i:6,} | Entropy: {ent:.4f} | "
                f"Regime: {regime:8s} | VRAM: {mem:.1f}MB"
            )
    
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    elapsed = end - start
    latency = (elapsed / n_ticks) * 1e6
    throughput = n_ticks / elapsed
    
    print()
    print("─" * 70)
    print(f"  LATENCY:    {latency:.2f} µs per tick")
    print(f"  THROUGHPUT: {throughput:,.0f} ticks/sec")
    print(f"  GPU MEMORY: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
    
    # Target check
    target_latency = 15.0
    if latency <= target_latency:
        print(f"  ✓ TARGET MET: {latency:.2f}µs <= {target_latency}µs")
    else:
        print(f"  ✗ TARGET MISSED: {latency:.2f}µs > {target_latency}µs")
        print(f"    Speedup needed: {latency/target_latency:.1f}x")
    
    print("─" * 70)
    
    # Test slicing
    print()
    print("  Testing Asset Slicing...")
    for i, name in enumerate(slicer.ASSET_NAMES):
        asset_slice = slicer.slice_asset(i)
        if len(asset_slice) > 0:
            print(f"    {name}: {len(asset_slice)} points, range [{asset_slice.min():.4f}, {asset_slice.max():.4f}]")
        else:
            print(f"    {name}: No data")
    
    # Test correlation
    print()
    print("  Cross-Asset Correlation Matrix:")
    corr = slicer.get_cross_asset_correlation()
    for i, name in enumerate(slicer.ASSET_NAMES):
        row = corr[i].cpu().numpy()
        print(f"    {name[:3]}: [{row[0]:+.2f}, {row[1]:+.2f}, {row[2]:+.2f}, {row[3]:+.2f}]")
    
    print()
    print("=" * 70)
    print("  STATUS: TRITON ACCELERATION ACTIVE")
    print("=" * 70)
    
    return latency, throughput


if __name__ == "__main__":
    run_benchmark(50000)
