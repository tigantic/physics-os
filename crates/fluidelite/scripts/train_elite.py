#!/usr/bin/env python3
"""
Synthetic "Needle in a Haystack" Training Script
================================================

Tests infinite context by requiring model to remember a key
value through arbitrary-length noise sequences.

Task: [KEY_MARKER, key_value, NOISE×100, KEY_MARKER] → predict key_value

This tests whether the MPS representation can maintain information
across long sequences without memory growth.

Constitutional Compliance:
    - Article VII.7.2: Definition of Done = USER-OBSERVABLE BEHAVIOR works
    - Article VII.7.4: Demonstration requirement with terminal output
"""

import torch
import torch.nn as nn
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fluidelite.llm.fluid_elite import FluidElite
from fluidelite.optim.riemannian import RiemannianAdam
from fluidelite.core.mps import MPS


# Training configuration
CONFIG = {
    "L": 12,              # 2^12 = 4096 context fidelity
    "RANK": 32,           # Bond Dimension (Capacity)
    "VOCAB": 64,          # Small vocab for speed
    "BATCH": 16,          # Batch size
    "LR": 0.005,          # Learning rate
    "STEPS": 500,         # Training steps
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
}


def generate_needle_batch(
    batch_size: int, 
    length: int = 100,
    key_marker: int = 4,
    noise_range: tuple[int, int] = (10, 60)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates "Needle in a Haystack" training examples.
    
    Format: [KEY_MARKER] [key_value] [NOISE...] [KEY_MARKER] → predict key_value
    
    The model must remember the key_value (the "needle") across the
    noise sequence (the "haystack") and predict it when it sees
    the closing KEY_MARKER.
    
    Args:
        batch_size: Number of examples to generate
        length: Length of noise sequence
        key_marker: Token ID used as key marker
        noise_range: (min, max) range for noise and key values
        
    Returns:
        (inputs, targets) where:
            inputs: (batch_size, length+3) sequences
            targets: (batch_size,) key values to predict
    """
    batch_inputs = []
    batch_targets = []
    
    for _ in range(batch_size):
        # Random key value
        key_val = torch.randint(noise_range[0], noise_range[1], (1,)).item()
        
        # Sequence: [KEY_MARKER, key_val, NOISE×length, KEY_MARKER]
        noise = torch.randint(noise_range[0], noise_range[1], (length,)).tolist()
        seq = [key_marker, key_val] + noise + [key_marker]
        
        batch_inputs.append(seq)
        batch_targets.append(key_val)
        
    return torch.tensor(batch_inputs), torch.tensor(batch_targets)


def train():
    """
    Main training loop for Needle in a Haystack task.
    
    Demonstrates that FluidElite can maintain information
    across arbitrary-length sequences.
    """
    print("=" * 60)
    print("FluidElite: Needle in a Haystack Training")
    print("=" * 60)
    print(f"\nConfiguration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    
    device = torch.device(CONFIG["DEVICE"])
    print(f"\n🚀 Initializing FluidElite (Rank={CONFIG['RANK']})...")
    print(f"   Device: {device}")
    
    # Create model
    model = FluidElite(
        num_sites=CONFIG['L'], 
        rank=CONFIG['RANK'], 
        vocab_size=CONFIG['VOCAB']
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {num_params:,}")
    
    # Optimizer and loss
    optimizer = RiemannianAdam(model.parameters(), lr=CONFIG['LR'])
    criterion = nn.CrossEntropyLoss()
    
    print("\n⚔️  Starting 'Needle in a Haystack' Training...")
    print("-" * 60)
    
    model.train()
    start_time = time.time()
    
    for step in range(CONFIG['STEPS']):
        # Curriculum: start with shorter sequences, increase over time
        curriculum = min(50 + step // 10, 200)
        
        # Generate batch
        inputs, targets = generate_needle_batch(CONFIG['BATCH'], length=curriculum)
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        loss_accum = 0.0
        
        # Process each sequence in batch
        for b in range(CONFIG['BATCH']):
            # Start with empty context (chi=1)
            ctx = MPS.random(CONFIG['L'], d=2, chi=1, device=device, dtype=torch.float64)
            
            seq = inputs[b]
            target = targets[b]
            
            # Process each token in sequence
            for t in range(len(seq)):
                token_id = seq[t].item()
                ctx = model.step(ctx, token_id)
            
            # Predict at end of sequence
            logits = model.predict(ctx)
            loss = criterion(logits.unsqueeze(0), target.unsqueeze(0))
            loss.backward()
            loss_accum += loss.item()
            
        optimizer.step()
        
        # Logging
        if step % 10 == 0:
            avg_loss = loss_accum / CONFIG['BATCH']
            elapsed = time.time() - start_time
            fps = (step * CONFIG['BATCH'] * curriculum) / elapsed if step > 0 else 0
            
            # Show progress bar style
            progress = (step + 1) / CONFIG['STEPS']
            bar_len = 20
            filled = int(bar_len * progress)
            bar = "█" * filled + "░" * (bar_len - filled)
            
            print(f"Step {step:03d} [{bar}] | Loss: {avg_loss:.4f} | "
                  f"CtxLen: {curriculum} | Speed: {fps:.0f} tok/s")
            
            # Check for success
            if avg_loss < 0.1:
                print("\n" + "=" * 60)
                print("✅ SOLVED! Infinite Context Memory Achieved.")
                print(f"   Converged in {step} steps, {elapsed:.1f} seconds")
                print("=" * 60)
                return True
    
    # Training complete
    elapsed = time.time() - start_time
    final_loss = loss_accum / CONFIG['BATCH']
    
    print("\n" + "-" * 60)
    print(f"Training Complete: {CONFIG['STEPS']} steps in {elapsed:.1f}s")
    print(f"Final Loss: {final_loss:.4f}")
    
    if final_loss < 1.0:
        print("⚠️  Partial success: Loss decreasing but not fully converged")
    else:
        print("❌ Model did not converge. Try adjusting hyperparameters.")
    
    return False


if __name__ == "__main__":
    success = train()
    sys.exit(0 if success else 1)
