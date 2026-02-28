"""
Memory Profiling Benchmark
==========================

Measures VRAM usage of FluidElite vs context length.

Key claim to validate:
    FluidElite memory = O(L × χ²) and BOUNDED regardless of tokens processed.

This uses tensornet.gpu.memory.VRAMManager for profiling.

Usage:
    python -m fluidelite.benchmarks.memory_profile --max-length 16384

Author: FluidElite Team
Date: January 2026
"""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass

import torch

# Import VRAMManager from ontic if available
try:
    from ontic.engine.gpu.memory import VRAMManager, MemoryStats
    VRAM_MANAGER_AVAILABLE = True
except ImportError:
    VRAM_MANAGER_AVAILABLE = False


@dataclass
class MemoryMeasurement:
    """Memory measurement result."""
    context_length: int
    tokens_processed: int
    allocated_mb: float
    peak_mb: float
    chi: int
    time_sec: float


def get_memory_mb() -> tuple[float, float]:
    """
    Get current GPU memory usage.
    
    Returns:
        (allocated_mb, reserved_mb)
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0
    
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    return allocated, reserved


def clear_memory():
    """Force GPU memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def profile_memory(
    model,
    n_tokens: int,
    device: str = "cuda",
    verbose: bool = True,
) -> MemoryMeasurement:
    """
    Profile memory usage while processing tokens.
    
    Args:
        model: FluidElite model
        n_tokens: Number of tokens to process
        device: Device to run on
        verbose: Print progress
    
    Returns:
        MemoryMeasurement with stats
    """
    from fluidelite import MPS
    
    clear_memory()
    
    # Record baseline
    base_alloc, _ = get_memory_mb()
    
    # Initialize context
    ctx = MPS.random(
        L=model.L,
        d=2,
        chi=1,
        dtype=torch.float64,
        device=device,
    )
    
    start_time = time.perf_counter()
    peak_alloc = base_alloc
    
    # Process tokens
    for i in range(n_tokens):
        token = i % model.vocab_size
        ctx = model.step(ctx, token)
        
        # Measure memory
        alloc, _ = get_memory_mb()
        peak_alloc = max(peak_alloc, alloc)
        
        if verbose and (i + 1) % 1000 == 0:
            chi = max(t.shape[0] for t in ctx.tensors)
            print(f"  Token {i+1}/{n_tokens}: {alloc:.1f} MB allocated, χ={chi}")
    
    elapsed = time.perf_counter() - start_time
    final_alloc, _ = get_memory_mb()
    final_chi = max(t.shape[0] for t in ctx.tensors)
    
    result = MemoryMeasurement(
        context_length=model.L,
        tokens_processed=n_tokens,
        allocated_mb=final_alloc - base_alloc,
        peak_mb=peak_alloc - base_alloc,
        chi=final_chi,
        time_sec=elapsed,
    )
    
    if verbose:
        print(f"\nMemory Profile:")
        print(f"  Tokens:     {n_tokens:,}")
        print(f"  Final alloc: {result.allocated_mb:.1f} MB")
        print(f"  Peak alloc:  {result.peak_mb:.1f} MB")
        print(f"  Final χ:    {result.chi}")
    
    return result


def memory_vs_length(
    lengths: list[int],
    chi: int = 32,
    num_sites: int = 12,
    vocab_size: int = 256,
    tokens_per_test: int = 1000,
    device: str = "cuda",
) -> list[MemoryMeasurement]:
    """
    Measure memory usage vs number of tokens processed.
    
    This validates the BOUNDED memory claim.
    
    Args:
        lengths: List of token counts to test
        chi: Bond dimension
        num_sites: Number of MPS sites
        vocab_size: Vocabulary size
        tokens_per_test: Actually tokens to process at each length
        device: Device
    
    Returns:
        List of measurements
    """
    from fluidelite import FluidElite
    
    results = []
    
    print("="*60)
    print("MEMORY VS TOKENS PROCESSED")
    print("="*60)
    print(f"Model: L={num_sites}, χ={chi}")
    print(f"Device: {device}")
    print()
    
    model = FluidElite(
        num_sites=num_sites,
        rank=chi,
        vocab_size=vocab_size,
    ).to(device if device != "cpu" else "cpu")
    
    for n_tokens in lengths:
        print(f"\n--- {n_tokens:,} tokens ---")
        clear_memory()
        
        result = profile_memory(
            model=model,
            n_tokens=n_tokens,
            device=device,
            verbose=False,
        )
        results.append(result)
        
        print(f"Peak: {result.peak_mb:.1f} MB, χ={result.chi}")
    
    print("\n" + "="*60)
    print("SUMMARY: Memory should be CONSTANT regardless of tokens")
    print("="*60)
    print(f"{'Tokens':>10} | {'Peak MB':>10} | {'χ':>6} | {'Time':>8}")
    print("-"*45)
    for r in results:
        print(f"{r.tokens_processed:>10,} | {r.peak_mb:>10.1f} | {r.chi:>6} | {r.time_sec:>8.2f}s")
    
    # Check if memory is bounded
    peaks = [r.peak_mb for r in results]
    if max(peaks) < 1.5 * min(peaks):
        print("\n✓ MEMORY IS BOUNDED - Validates infinite context claim!")
    else:
        print(f"\n⚠ Memory varies: {min(peaks):.1f} - {max(peaks):.1f} MB")
    
    return results


def memory_vs_chi(
    chi_values: list[int],
    num_sites: int = 12,
    vocab_size: int = 256,
    n_tokens: int = 1000,
    device: str = "cuda",
) -> list[MemoryMeasurement]:
    """
    Measure memory usage vs bond dimension.
    
    Memory should scale as O(χ²).
    
    Args:
        chi_values: List of bond dimensions to test
        num_sites: Number of MPS sites
        vocab_size: Vocabulary size
        n_tokens: Tokens to process
        device: Device
    
    Returns:
        List of measurements
    """
    from fluidelite import FluidElite
    
    results = []
    
    print("="*60)
    print("MEMORY VS BOND DIMENSION χ")
    print("="*60)
    print(f"Expected scaling: O(χ²)")
    print()
    
    for chi in chi_values:
        print(f"\n--- χ = {chi} ---")
        clear_memory()
        
        model = FluidElite(
            num_sites=num_sites,
            rank=chi,
            vocab_size=vocab_size,
        ).to(device if device != "cpu" else "cpu")
        
        result = profile_memory(
            model=model,
            n_tokens=n_tokens,
            device=device,
            verbose=False,
        )
        result.chi = chi  # Use requested chi, not measured
        results.append(result)
        
        print(f"Peak: {result.peak_mb:.1f} MB")
    
    print("\n" + "="*60)
    print("SUMMARY: Memory scaling")
    print("="*60)
    print(f"{'χ':>6} | {'χ²':>10} | {'Peak MB':>10} | {'MB/χ²':>10}")
    print("-"*45)
    for r in results:
        chi_sq = r.chi ** 2
        ratio = r.peak_mb / chi_sq if chi_sq > 0 else 0
        print(f"{r.chi:>6} | {chi_sq:>10,} | {r.peak_mb:>10.1f} | {ratio:>10.4f}")
    
    return results


def theoretical_memory(num_sites: int, chi: int, d: int = 2, dtype_bytes: int = 8) -> float:
    """
    Compute theoretical memory for MPS state.
    
    MPS memory = L × d × χ² × dtype_bytes
    
    Args:
        num_sites: L
        chi: Bond dimension
        d: Physical dimension
        dtype_bytes: Bytes per element (8 for float64)
    
    Returns:
        Memory in MB
    """
    # MPS: L tensors of shape (chi, d, chi)
    mps_elements = num_sites * d * chi * chi
    mps_bytes = mps_elements * dtype_bytes
    
    return mps_bytes / 1024**2


def main():
    parser = argparse.ArgumentParser(description="Memory Profiling Benchmark")
    parser.add_argument("--sites", type=int, default=12, help="Number of MPS sites")
    parser.add_argument("--chi", type=int, default=32, help="Bond dimension")
    parser.add_argument("--vocab", type=int, default=256, help="Vocabulary size")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--mode", choices=["length", "chi"], default="length",
                        help="Sweep over token length or chi")
    args = parser.parse_args()
    
    if not torch.cuda.is_available() and args.device == "cuda":
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    if args.mode == "length":
        # Test with increasing token counts
        memory_vs_length(
            lengths=[100, 500, 1000, 2000, 5000, 10000],
            chi=args.chi,
            num_sites=args.sites,
            vocab_size=args.vocab,
            device=args.device,
        )
    else:
        # Test with increasing chi
        memory_vs_chi(
            chi_values=[8, 16, 32, 64, 128],
            num_sites=args.sites,
            vocab_size=args.vocab,
            n_tokens=500,
            device=args.device,
        )
    
    # Print theoretical memory
    print("\n" + "="*60)
    print("THEORETICAL MEMORY")
    print("="*60)
    for chi in [16, 32, 64, 128]:
        mem = theoretical_memory(args.sites, chi)
        print(f"L={args.sites}, χ={chi}: {mem:.2f} MB")


if __name__ == "__main__":
    main()
