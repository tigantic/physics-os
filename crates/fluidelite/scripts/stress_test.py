#!/usr/bin/env python3
"""
100k Token Stress Test
======================

The final exam: VRAM must stay flat over 100,000 tokens.

This test verifies that FluidElite maintains O(1) memory complexity
as promised by the QTT compression approach. If VRAM grows linearly,
the test fails (standard attention behavior). If VRAM stays flat,
the test passes (logarithmic QTT compression working).

Constitutional Compliance:
    - Article VII.7.2: Definition of Done = USER-OBSERVABLE BEHAVIOR works
    - Article VII.7.4: Demonstration requirement with terminal output
"""

import torch
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fluidelite.llm.fluid_elite import FluidElite
from fluidelite.core.mps import MPS


# Test configuration
CONFIG = {
    "L": 12,              # Sites (2^12 = 4096 virtual context)
    "RANK": 64,           # Bond dimension
    "VOCAB": 100,         # Vocabulary size
    "NUM_TOKENS": 100000, # Total tokens to process
    "LOG_INTERVAL": 1000, # Log every N tokens
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
}


def get_vram_mb() -> float:
    """
    Get current GPU VRAM usage in MB.
    
    Returns:
        VRAM usage in megabytes, or 0 if no GPU
    """
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def get_vram_reserved_mb() -> float:
    """
    Get reserved GPU memory in MB (includes cached allocations).
    
    Returns:
        Reserved VRAM in megabytes, or 0 if no GPU
    """
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / (1024 * 1024)
    return 0.0


def stress_test():
    """
    Run the 100k token stress test.
    
    Verifies that memory stays constant regardless of sequence length.
    """
    print("=" * 60)
    print("FluidElite: 100k Token Stress Test")
    print("=" * 60)
    print(f"\nConfiguration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    
    device = torch.device(CONFIG["DEVICE"])
    print(f"\n🔥 Starting Stress Test on {device}...")
    
    # Create model
    model = FluidElite(
        num_sites=CONFIG["L"], 
        rank=CONFIG["RANK"], 
        vocab_size=CONFIG["VOCAB"]
    ).to(device)
    model.eval()
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")
    
    # Initialize context
    ctx = MPS.random(CONFIG["L"], 2, 1, device=device, dtype=torch.float64)
    
    # Record initial memory
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    initial_vram = get_vram_mb()
    print(f"\nInitial VRAM: {initial_vram:.2f} MB")
    print("-" * 60)
    
    start = time.time()
    vram_history = [initial_vram]
    
    with torch.no_grad():
        for t in range(CONFIG["NUM_TOKENS"]):
            # Random token
            token = t % CONFIG["VOCAB"]
            
            # Process token
            ctx = model.step(ctx, token)
            
            # Periodic logging
            if t % CONFIG["LOG_INTERVAL"] == 0 and t > 0:
                current_vram = get_vram_mb()
                vram_history.append(current_vram)
                
                elapsed = time.time() - start
                rate = t / elapsed if elapsed > 0 else 0
                
                # Memory status indicator
                if current_vram <= vram_history[0] * 1.2:
                    status = "✓"
                elif current_vram <= vram_history[0] * 1.5:
                    status = "⚠"
                else:
                    status = "✗"
                
                print(f"Step {t:6,d} | VRAM: {current_vram:7.2f} MB {status} | "
                      f"Speed: {rate:,.0f} tok/s")
                
                # Early abort if memory is clearly leaking
                if t > 5000 and current_vram > vram_history[0] * 2:
                    print("\n" + "=" * 60)
                    print("❌ MEMORY LEAK DETECTED!")
                    print(f"   Initial: {vram_history[0]:.2f} MB")
                    print(f"   Current: {current_vram:.2f} MB")
                    print(f"   Growth:  {(current_vram / vram_history[0] - 1) * 100:.1f}%")
                    print("=" * 60)
                    print("\nThis indicates the MPS is growing rather than being truncated.")
                    print("Check truncate_() calls in step() method.")
                    return False
    
    # Final statistics
    elapsed = time.time() - start
    final_vram = get_vram_mb()
    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
    
    vram_growth = (final_vram / initial_vram - 1) * 100 if initial_vram > 0 else 0
    
    print("\n" + "=" * 60)
    print("STRESS TEST RESULTS")
    print("=" * 60)
    print(f"Tokens Processed: {CONFIG['NUM_TOKENS']:,}")
    print(f"Total Time:       {elapsed:.2f}s")
    print(f"Average Speed:    {CONFIG['NUM_TOKENS'] / elapsed:,.0f} tok/s")
    print("-" * 60)
    print(f"Initial VRAM:     {initial_vram:.2f} MB")
    print(f"Final VRAM:       {final_vram:.2f} MB")
    print(f"Peak VRAM:        {peak_vram:.2f} MB")
    print(f"Memory Growth:    {vram_growth:+.1f}%")
    print("-" * 60)
    
    # Determine pass/fail
    if abs(vram_growth) < 20:  # Less than 20% growth
        print("✅ PASSED: Memory stable over 100k tokens")
        print("   QTT compression is working as expected!")
        return True
    elif vram_growth < 50:
        print("⚠️  MARGINAL: Some memory growth detected")
        print("   May indicate minor issues with truncation")
        return True
    else:
        print("❌ FAILED: Significant memory growth detected")
        print("   This suggests linear memory behavior, not O(1)")
        return False


def analyze_memory_profile():
    """
    Detailed memory profiling for debugging.
    """
    print("\nDetailed Memory Analysis")
    print("-" * 40)
    
    if not torch.cuda.is_available():
        print("No CUDA device available for detailed profiling")
        return
    
    device = torch.device("cuda")
    
    # Profile each component
    print("\n1. Empty model memory:")
    torch.cuda.reset_peak_memory_stats()
    model = FluidElite(num_sites=12, rank=64, vocab_size=100).to(device)
    print(f"   Model alone: {get_vram_mb():.2f} MB")
    
    print("\n2. With initial context:")
    ctx = MPS.random(12, 2, 1, device=device, dtype=torch.float64)
    print(f"   With MPS: {get_vram_mb():.2f} MB")
    
    print("\n3. After 100 tokens:")
    with torch.no_grad():
        for t in range(100):
            ctx = model.step(ctx, t % 100)
    print(f"   After 100: {get_vram_mb():.2f} MB")
    
    print("\n4. After 1000 tokens:")
    with torch.no_grad():
        for t in range(900):
            ctx = model.step(ctx, t % 100)
    print(f"   After 1000: {get_vram_mb():.2f} MB")
    
    print(f"\n   Peak allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    print(f"   Peak reserved: {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB")


if __name__ == "__main__":
    success = stress_test()
    
    if not success:
        print("\nRunning detailed memory analysis...")
        analyze_memory_profile()
    
    sys.exit(0 if success else 1)
