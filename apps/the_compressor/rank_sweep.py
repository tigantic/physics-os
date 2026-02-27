#!/usr/bin/env python3
"""
Rank-Fidelity Sweep Engine
===========================
Phase 2 of the Bond Dimension Conquest

Executes a logarithmic compression sweep at ranks R = {8, 16, 32, 64, 128}
to find the 'Collapse Point' - the minimum rank where physical structures
remain mathematically indistinguishable from the original.

Memory Model:
- Uses mmap streaming from compress.py (zero RAM bloat)
- GPU SVD with immediate VRAM release
- Only test region loaded for fidelity check

Usage:
    python3 rank_sweep.py /path/to/raw/frames/ --ranks 8,16,32,64,128
"""

import numpy as np
import torch
import time
import argparse
import json
import gc
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

# Import the actual compression engine
from compress import compress_qtt_4d, morton_interleave_3d, rsvd_gpu_safe
from universal import QTTArchive

# Constants
L2_CACHE_BYTES = 2_500_000   # 2.5 MB
L3_CACHE_BYTES = 36_000_000  # 36 MB


@dataclass
class RankResult:
    """Result of compression at a specific rank."""
    rank: int
    compressed_bytes: int
    compression_ratio: float
    psnr_db: float
    l2_error: float
    max_error: float
    cache_alignment: str
    compress_time_s: float
    query_latency_us: float
    num_cores: int


def get_cache_alignment(size_bytes: int) -> str:
    """Determine cache alignment category."""
    if size_bytes <= L2_CACHE_BYTES:
        return "L2 RESIDENT"
    elif size_bytes <= L3_CACHE_BYTES:
        return "L3 RESIDENT"
    else:
        return "RAM SPILL"


def extract_test_region(data_dir: Path, 
                        region_frames: int = 4,
                        region_size: int = 512,
                        stride: int = 1) -> np.ndarray:
    """
    Extract a test region using mmap - only loads what we need.
    Returns shape (region_frames, region_size, region_size).
    """
    frames = sorted(data_dir.glob('frame_*.npy'))
    
    region = []
    for i in range(min(region_frames, len(frames))):
        # mmap load - doesn't load full frame into RAM
        frame = np.load(frames[i], mmap_mode='r')
        # Extract just the region we need
        sub = frame[:region_size * stride:stride, :region_size * stride:stride]
        region.append(np.array(sub[:region_size, :region_size], dtype=np.float32))
    
    return np.stack(region, axis=0)


def calculate_psnr(original: np.ndarray, reconstructed: np.ndarray) -> Tuple[float, float, float]:
    """Calculate PSNR and error metrics."""
    diff = original.flatten() - reconstructed.flatten()
    l2_error = np.sqrt(np.mean(diff ** 2))
    max_error = np.max(np.abs(diff))
    
    data_range = original.max() - original.min()
    if data_range > 0 and l2_error > 0:
        psnr = 20 * np.log10(data_range / l2_error)
    else:
        psnr = float('inf')
    
    return psnr, l2_error, max_error


def reconstruct_test_region(archive_path: Path, 
                            region_shape: Tuple[int, int, int]) -> np.ndarray:
    """Reconstruct test region via point queries."""
    archive = QTTArchive(archive_path)
    
    output = np.zeros(region_shape, dtype=np.float32)
    total = np.prod(region_shape)
    
    t0 = time.time()
    for t in range(region_shape[0]):
        for y in range(region_shape[1]):
            for x in range(region_shape[2]):
                output[t, y, x] = archive.query(t, y, x)
        
        elapsed = time.time() - t0
        rate = ((t + 1) * region_shape[1] * region_shape[2]) / elapsed
        print(f"      Frame {t+1}/{region_shape[0]} ({rate:.0f} pts/s)", end='\r')
    
    print()
    return output


def benchmark_query(archive_path: Path, n_queries: int = 1000) -> float:
    """Benchmark query latency. Returns microseconds per query."""
    archive = QTTArchive(archive_path)
    max_idx = 2 ** archive.total_bits
    indices = np.random.randint(0, max_idx, size=n_queries)
    
    # Warmup
    for idx in indices[:10]:
        _ = archive.query_morton(int(idx))
    
    t0 = time.time()
    for idx in indices:
        _ = archive.query_morton(int(idx))
    
    elapsed = time.time() - t0
    return (elapsed / n_queries) * 1e6


def run_sweep(data_dir: Path,
              ranks: List[int],
              output_dir: Path,
              n_t: int = 32,
              n_y: int = 2048,
              n_x: int = 2048,
              test_frames: int = 4,
              test_size: int = 512,
              device: str = 'cuda') -> List[RankResult]:
    """Execute the full rank-fidelity sweep using mmap streaming."""
    
    print("\n" + "=" * 70)
    print("🎯 RANK-FIDELITY SWEEP: Bond Dimension Conquest")
    print("=" * 70)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get original data size (without loading it)
    frames = sorted(data_dir.glob('frame_*.npy'))
    sample = np.load(frames[0], mmap_mode='r')
    H_orig, W_orig = sample.shape
    T_orig = len(frames)
    original_bytes = T_orig * H_orig * W_orig * 4
    stride = H_orig // n_y
    
    print(f"\n📂 DATA: {T_orig} frames x {H_orig} x {W_orig}")
    print(f"   Original size: {original_bytes / 1e9:.2f} GB")
    print(f"   Quantics target: {n_t} x {n_y} x {n_x}")
    print(f"   Stride: {stride}")
    
    # Extract test region ONCE (mmap-based, small memory)
    print(f"\n🔍 EXTRACTING TEST REGION: ({test_frames}, {test_size}, {test_size})")
    original_region = extract_test_region(data_dir, test_frames, test_size, stride)
    print(f"   Region size: {original_region.nbytes / 1e6:.2f} MB")
    
    results = []
    
    for rank in ranks:
        print(f"\n{'─' * 70}")
        print(f"🔧 RANK = {rank}")
        print(f"{'─' * 70}")
        
        archive_path = output_dir / f"rank_{rank}.npz"
        
        # Compress using the real engine (mmap streaming)
        print("   ⚡ Compressing (mmap stream + GPU SVD)...")
        t0 = time.time()
        stats = compress_qtt_4d(
            data_dir=data_dir,
            output_path=archive_path,
            n_t=n_t,
            n_y=n_y,
            n_x=n_x,
            max_rank=rank,
            device=device
        )
        compress_time = time.time() - t0
        
        file_size = archive_path.stat().st_size
        ratio = original_bytes / file_size
        cache = get_cache_alignment(file_size)
        
        print(f"\n   📦 Result: {file_size / 1024:.1f} KB | {ratio:,.0f}x | {cache}")
        
        # Reconstruct test region
        print(f"   🔄 Reconstructing test region...")
        reconstructed = reconstruct_test_region(
            archive_path, 
            (test_frames, test_size, test_size)
        )
        
        # Calculate fidelity
        psnr, l2_err, max_err = calculate_psnr(original_region, reconstructed)
        print(f"   📊 PSNR: {psnr:.2f} dB | L2: {l2_err:.6f} | Max: {max_err:.6f}")
        
        # Query benchmark
        latency = benchmark_query(archive_path, n_queries=1000)
        print(f"   ⏱️  Query: {latency:.1f} µs")
        
        result = RankResult(
            rank=rank,
            compressed_bytes=file_size,
            compression_ratio=ratio,
            psnr_db=psnr,
            l2_error=l2_err,
            max_error=max_err,
            cache_alignment=cache,
            compress_time_s=compress_time,
            query_latency_us=latency,
            num_cores=stats['num_cores'],
        )
        results.append(result)
        
        # Cleanup
        del reconstructed
        gc.collect()
        torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 SWEEP SUMMARY")
    print("=" * 70)
    print(f"\n{'Rank':<6} {'PSNR':<10} {'Ratio':<12} {'Size':<10} {'Cache':<15} {'µs/query':<10}")
    print("-" * 70)
    
    for r in results:
        size_str = f"{r.compressed_bytes / 1024:.0f} KB"
        print(f"{r.rank:<6} {r.psnr_db:<10.2f} {r.compression_ratio:<12,.0f} {size_str:<10} {r.cache_alignment:<15} {r.query_latency_us:<10.1f}")
    
    # Find collapse point
    print("\n🎯 COLLAPSE POINT ANALYSIS:")
    PSNR_EXCELLENT = 40.0
    PSNR_GOOD = 35.0
    PSNR_ACCEPTABLE = 30.0
    
    for r in results:
        if r.psnr_db >= PSNR_EXCELLENT:
            status = "✅ EXCELLENT - Visually indistinguishable"
        elif r.psnr_db >= PSNR_GOOD:
            status = "⚠️  GOOD - Minor artifacts"
        elif r.psnr_db >= PSNR_ACCEPTABLE:
            status = "⚠️  ACCEPTABLE - Noticeable"
        else:
            status = "❌ COLLAPSED - Unacceptable"
        print(f"   Rank {r.rank}: {r.psnr_db:.1f} dB {status}")
    
    # Optimal recommendation
    l2_results = [r for r in results if 'L2' in r.cache_alignment]
    if l2_results:
        best_l2 = max(l2_results, key=lambda r: r.psnr_db)
        print(f"\n🏆 OPTIMAL L2-RESIDENT: Rank {best_l2.rank}")
        print(f"   PSNR: {best_l2.psnr_db:.2f} dB | Ratio: {best_l2.compression_ratio:,.0f}x")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Rank-Fidelity Sweep Engine')
    parser.add_argument('data_dir', type=str, help='Directory with frame_*.npy files')
    parser.add_argument('--ranks', type=str, default='8,16,32,64,128',
                        help='Comma-separated ranks to test')
    parser.add_argument('--output', '-o', type=str, default='sweep_results',
                        help='Output directory')
    parser.add_argument('--n-t', type=int, default=32, help='Temporal quantics size')
    parser.add_argument('--n-y', type=int, default=2048, help='Height quantics size')
    parser.add_argument('--n-x', type=int, default=2048, help='Width quantics size')
    parser.add_argument('--test-frames', type=int, default=4, help='Test region frames')
    parser.add_argument('--test-size', type=int, default=512, help='Test region spatial size')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save-json', type=str, help='Save results to JSON')
    
    args = parser.parse_args()
    
    ranks = [int(r) for r in args.ranks.split(',')]
    
    results = run_sweep(
        data_dir=Path(args.data_dir),
        ranks=ranks,
        output_dir=Path(args.output),
        n_t=args.n_t,
        n_y=args.n_y,
        n_x=args.n_x,
        test_frames=args.test_frames,
        test_size=args.test_size,
        device=args.device,
    )
    
    if args.save_json:
        # Convert numpy types to native Python for JSON
        def to_json_safe(obj):
            if isinstance(obj, dict):
                return {k: to_json_safe(v) for k, v in obj.items()}
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        results_json = [to_json_safe(asdict(r)) for r in results]
        with open(args.save_json, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"\n💾 Saved to {args.save_json}")


if __name__ == '__main__':
    main()
