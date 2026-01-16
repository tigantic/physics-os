"""
WikiText Perplexity Benchmark
=============================

Evaluates FluidElite perplexity on WikiText-2 and WikiText-103.

Perplexity = exp(-1/N * Σ log p(x_t | x_<t))

This is the standard language modeling metric.

Usage:
    python -m fluidelite.benchmarks.wikitext --chi 32 --sites 12

Author: FluidElite Team
Date: January 2026
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F

# Try to import datasets (HuggingFace)
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: 'datasets' not installed. Run: pip install datasets")


def download_wikitext2(cache_dir: str = ".cache/wikitext") -> dict[str, str]:
    """
    Download WikiText-2 dataset.
    
    Returns:
        Dictionary with 'train', 'validation', 'test' text strings.
    """
    if not HF_AVAILABLE:
        raise ImportError("Install 'datasets': pip install datasets")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load from HuggingFace
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=cache_dir)
    
    result = {}
    for split in ["train", "validation", "test"]:
        # Concatenate all text
        texts = dataset[split]["text"]
        result[split] = "\n".join(texts)
    
    print(f"WikiText-2 loaded:")
    print(f"  Train: {len(result['train']):,} chars")
    print(f"  Valid: {len(result['validation']):,} chars")
    print(f"  Test:  {len(result['test']):,} chars")
    
    return result


def tokenize_text(text: str, vocab_size: int = 256) -> torch.Tensor:
    """
    Tokenize text to byte-level tokens.
    
    Args:
        text: Raw text string
        vocab_size: Vocabulary size (256 for byte-level)
    
    Returns:
        Tensor of token IDs
    """
    # Byte-level tokenization (UTF-8 bytes, clamped to vocab_size)
    tokens = [min(b, vocab_size - 1) for b in text.encode("utf-8", errors="replace")]
    return torch.tensor(tokens, dtype=torch.long)


def evaluate_perplexity(
    model,
    text: str,
    vocab_size: int = 256,
    batch_tokens: int = 1024,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """
    Evaluate perplexity of FluidElite model on text.
    
    Args:
        model: FluidElite model instance
        text: Text to evaluate on
        vocab_size: Vocabulary size
        batch_tokens: Number of tokens to process before reporting
        device: Device to run on
        verbose: Print progress
    
    Returns:
        Dictionary with 'perplexity', 'loss', 'tokens', 'time'
    """
    from fluidelite import MPS
    
    tokens = tokenize_text(text, vocab_size)
    n_tokens = len(tokens)
    
    if verbose:
        print(f"Evaluating on {n_tokens:,} tokens...")
    
    model.eval()
    total_loss = 0.0
    n_predictions = 0
    
    start_time = time.perf_counter()
    
    # Initialize context
    ctx = MPS.random(
        L=model.L,
        d=2,
        chi=1,
        dtype=torch.float64,
        device=device,
    )
    
    with torch.no_grad():
        for i in range(n_tokens - 1):
            token = tokens[i].item()
            target = tokens[i + 1].item()
            
            # Forward pass
            ctx = model.step(ctx, token)
            logits = model.predict(ctx)
            
            # Cross-entropy loss
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -log_probs[target].item()
            
            if not math.isnan(loss) and not math.isinf(loss):
                total_loss += loss
                n_predictions += 1
            
            # Progress reporting
            if verbose and (i + 1) % batch_tokens == 0:
                avg_loss = total_loss / max(n_predictions, 1)
                ppl = math.exp(min(avg_loss, 100))  # Cap to avoid overflow
                elapsed = time.perf_counter() - start_time
                tps = (i + 1) / elapsed
                print(f"  [{i+1:,}/{n_tokens:,}] Loss: {avg_loss:.4f}, PPL: {ppl:.2f}, {tps:.1f} tok/s")
    
    elapsed = time.perf_counter() - start_time
    avg_loss = total_loss / max(n_predictions, 1)
    perplexity = math.exp(min(avg_loss, 100))
    
    result = {
        "perplexity": perplexity,
        "loss": avg_loss,
        "tokens": n_predictions,
        "time_sec": elapsed,
        "tokens_per_sec": n_predictions / elapsed,
    }
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"RESULTS")
        print(f"{'='*50}")
        print(f"Perplexity: {perplexity:.2f}")
        print(f"Avg Loss:   {avg_loss:.4f}")
        print(f"Tokens:     {n_predictions:,}")
        print(f"Time:       {elapsed:.1f}s")
        print(f"Throughput: {result['tokens_per_sec']:.1f} tok/s")
    
    return result


def evaluate_perplexity_batched(
    model,
    text: str,
    vocab_size: int = 256,
    seq_len: int = 256,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """
    Evaluate perplexity with fresh context per sequence.
    
    This is more comparable to Transformer evaluation where
    each sequence is independent.
    
    Args:
        model: FluidElite model instance
        text: Text to evaluate on
        vocab_size: Vocabulary size
        seq_len: Sequence length for evaluation chunks
        device: Device to run on
        verbose: Print progress
    
    Returns:
        Dictionary with 'perplexity', 'loss', 'tokens', 'time'
    """
    from fluidelite import MPS
    
    tokens = tokenize_text(text, vocab_size)
    n_tokens = len(tokens)
    n_sequences = (n_tokens - 1) // seq_len
    
    if verbose:
        print(f"Evaluating {n_sequences} sequences of length {seq_len}...")
    
    model.eval()
    total_loss = 0.0
    n_predictions = 0
    
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for seq_idx in range(n_sequences):
            # Fresh context for each sequence
            ctx = MPS.random(
                L=model.L,
                d=2,
                chi=1,
                dtype=torch.float64,
                device=device,
            )
            
            start = seq_idx * seq_len
            end = start + seq_len + 1  # +1 for target
            
            for i in range(seq_len):
                token = tokens[start + i].item()
                target = tokens[start + i + 1].item()
                
                ctx = model.step(ctx, token)
                logits = model.predict(ctx)
                
                log_probs = F.log_softmax(logits, dim=-1)
                loss = -log_probs[target].item()
                
                if not math.isnan(loss) and not math.isinf(loss):
                    total_loss += loss
                    n_predictions += 1
            
            if verbose and (seq_idx + 1) % 10 == 0:
                avg_loss = total_loss / max(n_predictions, 1)
                ppl = math.exp(min(avg_loss, 100))
                print(f"  Seq {seq_idx+1}/{n_sequences}: PPL={ppl:.2f}")
    
    elapsed = time.perf_counter() - start_time
    avg_loss = total_loss / max(n_predictions, 1)
    perplexity = math.exp(min(avg_loss, 100))
    
    result = {
        "perplexity": perplexity,
        "loss": avg_loss,
        "tokens": n_predictions,
        "sequences": n_sequences,
        "time_sec": elapsed,
        "tokens_per_sec": n_predictions / elapsed,
    }
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"BATCHED RESULTS (seq_len={seq_len})")
        print(f"{'='*50}")
        print(f"Perplexity: {perplexity:.2f}")
        print(f"Avg Loss:   {avg_loss:.4f}")
        print(f"Sequences:  {n_sequences}")
        print(f"Tokens:     {n_predictions:,}")
        print(f"Time:       {elapsed:.1f}s")
        print(f"Throughput: {result['tokens_per_sec']:.1f} tok/s")
    
    return result


def run_wikitext_benchmark(
    num_sites: int = 12,
    rank: int = 32,
    vocab_size: int = 256,
    max_tokens: int = 10000,
    device: str = "cpu",
) -> dict:
    """
    Run complete WikiText-2 benchmark.
    
    Args:
        num_sites: Number of MPS sites
        rank: Bond dimension
        vocab_size: Vocabulary size
        max_tokens: Maximum tokens to evaluate (for speed)
        device: Device to run on
    
    Returns:
        Benchmark results dictionary
    """
    from fluidelite import FluidElite
    
    print("="*60)
    print("WIKITEXT-2 PERPLEXITY BENCHMARK")
    print("="*60)
    print(f"Model: L={num_sites}, χ={rank}, vocab={vocab_size}")
    print(f"Device: {device}")
    print()
    
    # Download dataset
    data = download_wikitext2()
    
    # Truncate for speed
    test_text = data["test"][:max_tokens]
    
    # Create untrained model (random weights)
    model = FluidElite(
        num_sites=num_sites,
        rank=rank,
        vocab_size=vocab_size,
    )
    model.to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Evaluate
    results = evaluate_perplexity(
        model=model,
        text=test_text,
        vocab_size=vocab_size,
        device=device,
        verbose=True,
    )
    
    results["model"] = {
        "num_sites": num_sites,
        "rank": rank,
        "vocab_size": vocab_size,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    
    return results


def chi_sweep(
    chi_values: list[int],
    text: str,
    num_sites: int = 12,
    vocab_size: int = 256,
    max_tokens: int = 5000,
    device: str = "cpu",
) -> list[dict]:
    """
    Sweep over bond dimensions to study perplexity vs χ.
    
    Args:
        chi_values: List of bond dimensions to test
        text: Evaluation text
        num_sites: Number of MPS sites
        vocab_size: Vocabulary size
        max_tokens: Max tokens per evaluation
        device: Device
    
    Returns:
        List of result dictionaries
    """
    from fluidelite import FluidElite
    
    results = []
    text = text[:max_tokens]
    
    print("="*60)
    print("CHI SWEEP: Perplexity vs Bond Dimension")
    print("="*60)
    
    for chi in chi_values:
        print(f"\n--- χ = {chi} ---")
        
        model = FluidElite(
            num_sites=num_sites,
            rank=chi,
            vocab_size=vocab_size,
        )
        model.to(device)
        
        result = evaluate_perplexity(
            model=model,
            text=text,
            vocab_size=vocab_size,
            device=device,
            verbose=False,
        )
        result["chi"] = chi
        result["parameters"] = sum(p.numel() for p in model.parameters())
        results.append(result)
        
        print(f"PPL: {result['perplexity']:.2f}, Params: {result['parameters']:,}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'χ':>6} | {'PPL':>10} | {'Params':>12} | {'tok/s':>10}")
    print("-"*50)
    for r in results:
        print(f"{r['chi']:>6} | {r['perplexity']:>10.2f} | {r['parameters']:>12,} | {r['tokens_per_sec']:>10.1f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="WikiText Perplexity Benchmark")
    parser.add_argument("--sites", type=int, default=12, help="Number of MPS sites")
    parser.add_argument("--chi", type=int, default=32, help="Bond dimension")
    parser.add_argument("--vocab", type=int, default=256, help="Vocabulary size")
    parser.add_argument("--max-tokens", type=int, default=10000, help="Max tokens to evaluate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--sweep", action="store_true", help="Run chi sweep")
    args = parser.parse_args()
    
    if args.sweep:
        data = download_wikitext2()
        chi_sweep(
            chi_values=[4, 8, 16, 32, 64],
            text=data["test"],
            num_sites=args.sites,
            vocab_size=args.vocab,
            max_tokens=args.max_tokens,
            device=args.device,
        )
    else:
        run_wikitext_benchmark(
            num_sites=args.sites,
            rank=args.chi,
            vocab_size=args.vocab,
            max_tokens=args.max_tokens,
            device=args.device,
        )


if __name__ == "__main__":
    main()
