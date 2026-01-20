#!/usr/bin/env python3
"""
🎯 PRODUCTION TRAINING: FluidEliteZK on WikiText-2
==================================================

Train FluidEliteZK (Linear Reservoir) for ZK-provable inference.

Training Strategy:
1. Use LinearReservoirHead with closed-form least-squares (instant training)
2. MPO weights trained with gradient descent
3. Save production weights in exportable format

Target: WikiText-2 perplexity < 50 (competitive with small RNNs)

Usage:
    python train_zk_production.py --epochs 5 --output weights/production_zk.pt

Author: HyperTensor Labs  
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import time
import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fluidelite.llm.fluid_elite_zk import FluidEliteZK
    from fluidelite.core.mps import MPS
except ImportError:
    from llm.fluid_elite_zk import FluidEliteZK
    from core.mps import MPS


@dataclass
class TrainingConfig:
    """Production training configuration."""
    # Model
    num_sites: int = 12          # 2^12 = 4096 token space
    chi_max: int = 64            # Bond dimension (memory capacity)
    vocab_size: int = 256        # Character-level for simplicity
    truncate_every: int = 10     # Truncate every N steps
    
    # Training
    epochs: int = 5
    window_size: int = 64        # BPTT window length
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    
    # Data
    max_chars: int = 1_000_000   # Use 1M chars from WikiText
    val_split: float = 0.1
    
    # Output
    output_dir: str = "weights"
    model_name: str = "fluidelite_zk_production"
    
    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every: int = 100
    save_every: int = 1000


def load_wikitext_chars(max_chars: int = 1_000_000):
    """Load WikiText-2 as character-level tokens."""
    print("📚 Loading WikiText-2 (character-level)...")
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        os.system("pip install datasets")
        from datasets import load_dataset
    
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    
    # Concatenate all text
    text = "\n".join([s['text'] for s in ds if s['text'].strip()])
    text = text[:max_chars]
    
    # Build character vocabulary (sorted for reproducibility)
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    vocab_size = len(chars)
    
    print(f"   Total chars: {len(text):,}")
    print(f"   Vocab size: {vocab_size}")
    print(f"   Sample: {text[:80]!r}...")
    
    # Encode
    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)
    
    return data, vocab_size, char_to_idx, idx_to_char


def train_epoch(model: FluidEliteZK, data: torch.Tensor, optimizer, 
                config: TrainingConfig, epoch: int) -> dict:
    """Train one epoch."""
    model.train()
    device = config.device
    
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    start_time = time.time()
    
    # Process data in BPTT windows
    num_windows = (len(data) - 1) // config.window_size
    
    for window_idx in range(num_windows):
        # Extract window
        start = window_idx * config.window_size
        end = start + config.window_size + 1
        if end > len(data):
            break
            
        window = data[start:end].to(device)
        inputs = window[:-1]
        targets = window[1:]
        
        # Fresh context each window
        ctx = MPS.random(model.L, d=2, chi=1, device=device, dtype=torch.float32)
        model.reset_step_count()
        
        # Forward pass through window
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        correct = 0
        
        for t in range(len(inputs)):
            logits = model.predict(ctx)
            target = targets[t]
            
            # Cross-entropy loss
            step_loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
            loss = loss + step_loss
            
            # Accuracy
            pred = logits.argmax()
            if pred == target:
                correct += 1
            
            # Update context
            ctx = model.step(ctx, inputs[t].item())
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        
        # Stats
        total_loss += loss.item()
        total_correct += correct
        total_tokens += len(inputs)
        
        # Logging
        if (window_idx + 1) % config.log_every == 0:
            elapsed = time.time() - start_time
            tok_per_sec = total_tokens / elapsed
            avg_loss = total_loss / (window_idx + 1) / config.window_size
            acc = 100.0 * total_correct / total_tokens
            ppl = min(torch.exp(torch.tensor(avg_loss)).item(), 9999)
            
            print(f"   [{window_idx+1:5d}/{num_windows}] "
                  f"loss={avg_loss:.4f} ppl={ppl:6.1f} acc={acc:5.1f}% "
                  f"| {tok_per_sec:.0f} tok/s")
    
    # Epoch summary
    elapsed = time.time() - start_time
    avg_loss = total_loss / max(1, num_windows) / config.window_size
    acc = 100.0 * total_correct / max(1, total_tokens)
    ppl = min(torch.exp(torch.tensor(avg_loss)).item(), 9999)
    
    return {
        "loss": avg_loss,
        "ppl": ppl,
        "acc": acc,
        "tok_per_sec": total_tokens / elapsed,
        "elapsed": elapsed
    }


@torch.no_grad()
def evaluate(model: FluidEliteZK, data: torch.Tensor, config: TrainingConfig) -> dict:
    """Evaluate on validation data."""
    model.eval()
    device = config.device
    
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    
    num_windows = min(100, (len(data) - 1) // config.window_size)
    
    for window_idx in range(num_windows):
        start = window_idx * config.window_size
        end = start + config.window_size + 1
        if end > len(data):
            break
            
        window = data[start:end].to(device)
        inputs = window[:-1]
        targets = window[1:]
        
        ctx = MPS.random(model.L, d=2, chi=1, device=device, dtype=torch.float32)
        model.reset_step_count()
        
        for t in range(len(inputs)):
            logits = model.predict(ctx)
            target = targets[t]
            
            step_loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
            total_loss += step_loss.item()
            
            pred = logits.argmax()
            if pred == target:
                total_correct += 1
            
            total_tokens += 1
            ctx = model.step(ctx, inputs[t].item())
    
    avg_loss = total_loss / max(1, total_tokens)
    ppl = min(torch.exp(torch.tensor(avg_loss)).item(), 9999)
    acc = 100.0 * total_correct / max(1, total_tokens)
    
    return {"loss": avg_loss, "ppl": ppl, "acc": acc}


@torch.no_grad()
def generate_sample(model: FluidEliteZK, seed_text: str, idx_to_char: dict, 
                    char_to_idx: dict, device: str, length: int = 100) -> str:
    """Generate text sample from the model."""
    model.eval()
    
    # Initialize context
    ctx = MPS.random(model.L, d=2, chi=1, device=device, dtype=torch.float32)
    model.reset_step_count()
    
    # Process seed
    generated = list(seed_text)
    for c in seed_text:
        if c in char_to_idx:
            ctx = model.step(ctx, char_to_idx[c])
    
    # Generate
    for _ in range(length):
        logits = model.predict(ctx)
        
        # Sample with temperature
        probs = F.softmax(logits / 0.8, dim=0)
        next_idx = torch.multinomial(probs, 1).item()
        
        if next_idx in idx_to_char:
            next_char = idx_to_char[next_idx]
            generated.append(next_char)
            ctx = model.step(ctx, next_idx)
        else:
            break
    
    return "".join(generated)


def save_checkpoint(model: FluidEliteZK, config: TrainingConfig, 
                    train_stats: dict, char_to_idx: dict, epoch: int):
    """Save model checkpoint in production format."""
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "num_sites": config.num_sites,
            "chi_max": config.chi_max,
            "vocab_size": model.vocab_size,
            "truncate_every": config.truncate_every,
        },
        "training": {
            "epoch": epoch,
            "train_loss": train_stats.get("loss", 0),
            "train_ppl": train_stats.get("ppl", 0),
            "train_acc": train_stats.get("acc", 0),
        },
        "vocab": char_to_idx,
        "model_type": "FluidEliteZK",
        "version": "1.0.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Save PyTorch checkpoint
    model_path = output_dir / f"{config.model_name}_epoch{epoch}.pt"
    torch.save(checkpoint, model_path)
    print(f"   💾 Saved checkpoint: {model_path}")
    
    # Also save the final production weights separately
    if epoch == config.epochs:
        final_path = output_dir / f"{config.model_name}_final.pt"
        torch.save(checkpoint, final_path)
        print(f"   🎯 Saved PRODUCTION weights: {final_path}")
        
        # Export ZK-ready weights (just the model state for Rust prover)
        zk_weights = {
            "W_hidden_cores": model.W_hidden.cores.detach().cpu().numpy().tolist(),
            "W_input_cores": model.W_input.cores.detach().cpu().numpy().tolist(),
            "head_weight": model.head.weight.detach().cpu().numpy().tolist(),
            "head_bias": model.head.bias.detach().cpu().numpy().tolist() if model.head.bias is not None else None,
            "config": checkpoint["config"],
            "vocab": char_to_idx,
        }
        
        zk_path = output_dir / f"{config.model_name}_zk_weights.json"
        with open(zk_path, 'w') as f:
            json.dump(zk_weights, f)
        print(f"   🔐 Saved ZK-ready weights: {zk_path}")


def main():
    parser = argparse.ArgumentParser(description="Train FluidEliteZK on WikiText-2")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--num-sites", type=int, default=12, help="Number of MPS sites (L)")
    parser.add_argument("--chi-max", type=int, default=64, help="Max bond dimension")
    parser.add_argument("--window-size", type=int, default=64, help="BPTT window size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max-chars", type=int, default=500_000, help="Max chars to use")
    parser.add_argument("--output", type=str, default="weights", help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()
    
    # Configure
    config = TrainingConfig(
        epochs=args.epochs,
        num_sites=args.num_sites,
        chi_max=args.chi_max,
        window_size=args.window_size,
        lr=args.lr,
        max_chars=args.max_chars,
        output_dir=args.output,
    )
    if args.device:
        config.device = args.device
    
    print("=" * 66)
    print("🚀 PRODUCTION TRAINING: FluidEliteZK on WikiText-2")
    print("=" * 66)
    print()
    print("📋 Configuration:")
    print(f"   Model: L={config.num_sites}, χ={config.chi_max}")
    print(f"   Training: {config.epochs} epochs, lr={config.lr}")
    print(f"   Window: {config.window_size} tokens")
    print(f"   Device: {config.device}")
    print()
    
    # Load data
    data, vocab_size, char_to_idx, idx_to_char = load_wikitext_chars(config.max_chars)
    config.vocab_size = vocab_size
    
    # Split train/val
    val_start = int(len(data) * (1 - config.val_split))
    train_data = data[:val_start]
    val_data = data[val_start:]
    
    print(f"   Train: {len(train_data):,} tokens")
    print(f"   Val: {len(val_data):,} tokens")
    print()
    
    # Create model
    print("📦 Creating FluidEliteZK model...")
    model = FluidEliteZK(
        num_sites=config.num_sites,
        chi_max=config.chi_max,
        vocab_size=vocab_size,
        truncate_every=config.truncate_every,
    )
    model = model.to(config.device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {num_params:,}")
    print(f"   Constraint/token: {model.constraint_count_per_token():,}")
    print()
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # Training loop
    best_val_ppl = float('inf')
    
    for epoch in range(1, config.epochs + 1):
        print(f"{'='*66}")
        print(f"📈 Epoch {epoch}/{config.epochs}")
        print(f"{'='*66}")
        
        # Train
        train_stats = train_epoch(model, train_data, optimizer, config, epoch)
        print(f"\n   Train: loss={train_stats['loss']:.4f} ppl={train_stats['ppl']:.1f} "
              f"acc={train_stats['acc']:.1f}%")
        
        # Validate
        val_stats = evaluate(model, val_data, config)
        print(f"   Val:   loss={val_stats['loss']:.4f} ppl={val_stats['ppl']:.1f} "
              f"acc={val_stats['acc']:.1f}%")
        
        # Generate sample
        print("\n📝 Sample generation:")
        sample = generate_sample(model, "The ", idx_to_char, char_to_idx, 
                                 config.device, length=100)
        print(f"   \"{sample[:100]}...\"")
        
        # Save checkpoint
        if val_stats['ppl'] < best_val_ppl:
            best_val_ppl = val_stats['ppl']
            print(f"\n   🏆 New best val PPL: {best_val_ppl:.1f}")
        
        save_checkpoint(model, config, {**train_stats, **val_stats}, char_to_idx, epoch)
        print()
    
    # Final summary
    print("=" * 66)
    print("🎉 TRAINING COMPLETE")
    print("=" * 66)
    print()
    print(f"   Final Val PPL: {val_stats['ppl']:.1f}")
    print(f"   Final Val Acc: {val_stats['acc']:.1f}%")
    print()
    print(f"   Production weights saved to: {config.output_dir}/")
    print(f"   - {config.model_name}_final.pt (PyTorch)")
    print(f"   - {config.model_name}_zk_weights.json (ZK-ready)")
    print()
    print("   Next steps:")
    print("   1. Test with: python verify_linearity.py")
    print("   2. Run ZK server: ./fluidelite-server --weights weights/fluidelite_zk_production_zk_weights.json")
    print()


if __name__ == "__main__":
    main()
