"""
Throughput Benchmark
====================

Measures tokens/second for FluidElite vs context length.

Key claim to validate:
    FluidElite throughput = O(1) per token (independent of history).

Compare against Transformer where attention costs O(L) per token.

Usage:
    python -m fluidelite.benchmarks.throughput --chi 32 --max-length 8192

Author: FluidElite Team
Date: January 2026
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch


@dataclass
class ThroughputResult:
    """Throughput measurement result."""
    tokens: int
    time_sec: float
    tokens_per_sec: float
    ms_per_token: float
    chi: int
    num_sites: int


def benchmark_throughput(
    model,
    n_tokens: int,
    device: str = "cuda",
    warmup: int = 100,
    verbose: bool = True,
) -> ThroughputResult:
    """
    Benchmark throughput (tokens/second).
    
    Args:
        model: FluidElite model
        n_tokens: Number of tokens to process
        device: Device to run on
        warmup: Warmup tokens before timing
        verbose: Print progress
    
    Returns:
        ThroughputResult with timing stats
    """
    from fluidelite import MPS
    
    # Initialize context
    ctx = MPS.random(
        L=model.L,
        d=2,
        chi=1,
        dtype=torch.float64,
        device=device,
    )
    
    # Warmup
    if verbose:
        print(f"Warming up ({warmup} tokens)...")
    
    for i in range(warmup):
        token = i % model.vocab_size
        ctx = model.step(ctx, token)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    if verbose:
        print(f"Benchmarking ({n_tokens} tokens)...")
    
    start_time = time.perf_counter()
    
    for i in range(n_tokens):
        token = (warmup + i) % model.vocab_size
        ctx = model.step(ctx, token)
        _ = model.predict(ctx)  # Include prediction
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start_time
    
    final_chi = max(t.shape[0] for t in ctx.tensors)
    
    result = ThroughputResult(
        tokens=n_tokens,
        time_sec=elapsed,
        tokens_per_sec=n_tokens / elapsed,
        ms_per_token=1000 * elapsed / n_tokens,
        chi=final_chi,
        num_sites=model.L,
    )
    
    if verbose:
        print(f"\nThroughput Results:")
        print(f"  Tokens:     {n_tokens:,}")
        print(f"  Time:       {elapsed:.3f}s")
        print(f"  Throughput: {result.tokens_per_sec:.1f} tok/s")
        print(f"  Latency:    {result.ms_per_token:.2f} ms/tok")
        print(f"  Final χ:   {result.chi}")
    
    return result


def throughput_vs_length(
    lengths: list[int],
    chi: int = 32,
    num_sites: int = 12,
    vocab_size: int = 256,
    device: str = "cuda",
) -> list[ThroughputResult]:
    """
    Measure throughput at different context lengths.
    
    The key test: throughput should be CONSTANT regardless of
    how many tokens have been processed (unlike Transformers).
    
    Args:
        lengths: List of context lengths (tokens processed before measurement)
        chi: Bond dimension
        num_sites: Number of MPS sites
        vocab_size: Vocabulary size
        device: Device
    
    Returns:
        List of ThroughputResult
    """
    from fluidelite import FluidElite, MPS
    
    results = []
    
    print("="*60)
    print("THROUGHPUT VS CONTEXT LENGTH")
    print("="*60)
    print(f"Model: L={num_sites}, χ={chi}")
    print(f"Device: {device}")
    print(f"Key claim: Throughput should be CONSTANT")
    print()
    
    model = FluidElite(
        num_sites=num_sites,
        rank=chi,
        vocab_size=vocab_size,
    )
    
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    
    for context_len in lengths:
        print(f"\n--- After {context_len:,} tokens of context ---")
        
        # Initialize fresh context
        ctx = MPS.random(
            L=model.L,
            d=2,
            chi=1,
            dtype=torch.float64,
            device=device,
        )
        
        # Build up context
        for i in range(context_len):
            token = i % vocab_size
            ctx = model.step(ctx, token)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Now measure throughput for next 500 tokens
        n_measure = 500
        start = time.perf_counter()
        
        for i in range(n_measure):
            token = (context_len + i) % vocab_size
            ctx = model.step(ctx, token)
            _ = model.predict(ctx)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        
        result = ThroughputResult(
            tokens=n_measure,
            time_sec=elapsed,
            tokens_per_sec=n_measure / elapsed,
            ms_per_token=1000 * elapsed / n_measure,
            chi=max(t.shape[0] for t in ctx.tensors),
            num_sites=num_sites,
        )
        results.append(result)
        
        print(f"Throughput: {result.tokens_per_sec:.1f} tok/s, {result.ms_per_token:.2f} ms/tok")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Context':>10} | {'tok/s':>10} | {'ms/tok':>10} | {'χ':>6}")
    print("-"*45)
    for i, r in enumerate(results):
        print(f"{lengths[i]:>10,} | {r.tokens_per_sec:>10.1f} | {r.ms_per_token:>10.2f} | {r.chi:>6}")
    
    # Analyze variance
    throughputs = [r.tokens_per_sec for r in results]
    mean_tp = sum(throughputs) / len(throughputs)
    variance = sum((t - mean_tp)**2 for t in throughputs) / len(throughputs)
    cv = (variance ** 0.5) / mean_tp  # Coefficient of variation
    
    print(f"\nMean throughput: {mean_tp:.1f} tok/s")
    print(f"Std deviation:   {variance**0.5:.1f} tok/s")
    print(f"CV:              {cv:.2%}")
    
    if cv < 0.1:
        print("\n✓ THROUGHPUT IS CONSTANT - O(1) per token validated!")
    else:
        print(f"\n⚠ Throughput varies by {cv:.1%} - may need investigation")
    
    return results


def throughput_vs_chi(
    chi_values: list[int],
    num_sites: int = 12,
    vocab_size: int = 256,
    n_tokens: int = 1000,
    device: str = "cuda",
) -> list[ThroughputResult]:
    """
    Measure throughput vs bond dimension.
    
    Throughput should scale as O(1/χ³) due to contraction cost.
    
    Args:
        chi_values: List of bond dimensions to test
        num_sites: Number of MPS sites
        vocab_size: Vocabulary size
        n_tokens: Tokens to process
        device: Device
    
    Returns:
        List of results
    """
    from fluidelite import FluidElite
    
    results = []
    
    print("="*60)
    print("THROUGHPUT VS BOND DIMENSION χ")
    print("="*60)
    print(f"Expected scaling: O(1/χ³)")
    print()
    
    for chi in chi_values:
        print(f"\n--- χ = {chi} ---")
        
        model = FluidElite(
            num_sites=num_sites,
            rank=chi,
            vocab_size=vocab_size,
        )
        
        if device == "cuda" and torch.cuda.is_available():
            model = model.cuda()
        
        result = benchmark_throughput(
            model=model,
            n_tokens=n_tokens,
            device=device,
            warmup=100,
            verbose=False,
        )
        result.chi = chi
        results.append(result)
        
        print(f"Throughput: {result.tokens_per_sec:.1f} tok/s")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'χ':>6} | {'χ³':>12} | {'tok/s':>10} | {'tok/s × χ³':>12}")
    print("-"*50)
    for r in results:
        chi_cubed = r.chi ** 3
        scaled = r.tokens_per_sec * chi_cubed
        print(f"{r.chi:>6} | {chi_cubed:>12,} | {r.tokens_per_sec:>10.1f} | {scaled:>12,.0f}")
    
    return results


def compare_with_transformer_theoretical(
    lengths: list[int],
    chi: int = 32,
    d_model: int = 512,
    n_layers: int = 6,
) -> None:
    """
    Print theoretical comparison with Transformer.
    
    Transformer: O(L × d_model) per token for attention
    FluidElite:  O(χ³) per token, independent of L
    
    Args:
        lengths: Context lengths to compare
        chi: FluidElite bond dimension
        d_model: Transformer hidden dimension
        n_layers: Transformer layers
    """
    print("="*60)
    print("THEORETICAL COMPARISON: FluidElite vs Transformer")
    print("="*60)
    print(f"FluidElite: χ={chi}, cost = O(χ³) = O({chi**3:,})")
    print(f"Transformer: d_model={d_model}, n_layers={n_layers}")
    print(f"Transformer attention: O(L × d_model) per token")
    print()
    print(f"{'Context L':>12} | {'Transformer':>15} | {'FluidElite':>12} | {'Speedup':>10}")
    print("-"*60)
    
    fe_cost = chi ** 3
    
    for L in lengths:
        # Simplified Transformer cost: L × d_model × n_layers for attention
        tf_cost = L * d_model * n_layers
        speedup = tf_cost / fe_cost
        print(f"{L:>12,} | {tf_cost:>15,} | {fe_cost:>12,} | {speedup:>10.1f}×")
    
    print()
    print("Note: This is simplified analysis. Real costs depend on implementation.")


def main():
    parser = argparse.ArgumentParser(description="Throughput Benchmark")
    parser.add_argument("--sites", type=int, default=12, help="Number of MPS sites")
    parser.add_argument("--chi", type=int, default=32, help="Bond dimension")
    parser.add_argument("--vocab", type=int, default=256, help="Vocabulary size")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--mode", choices=["length", "chi", "compare"], default="length",
                        help="Sweep over length, chi, or theoretical comparison")
    args = parser.parse_args()
    
    if not torch.cuda.is_available() and args.device == "cuda":
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    if args.mode == "length":
        throughput_vs_length(
            lengths=[0, 100, 500, 1000, 2000, 5000, 10000],
            chi=args.chi,
            num_sites=args.sites,
            vocab_size=args.vocab,
            device=args.device,
        )
    elif args.mode == "chi":
        throughput_vs_chi(
            chi_values=[8, 16, 32, 64],
            num_sites=args.sites,
            vocab_size=args.vocab,
            device=args.device,
        )
    else:
        compare_with_transformer_theoretical(
            lengths=[512, 1024, 2048, 4096, 8192, 16384, 32768],
            chi=args.chi,
        )


if __name__ == "__main__":
    main()
