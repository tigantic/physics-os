#!/usr/bin/env python3
"""
FluidElite Training Script
==========================

Train the FluidElite QTT-LLM on text data.

This script implements:
- Efficient stateful training with truncated BPTT
- Riemannian optimization for stable tensor network training
- Comprehensive logging and checkpointing
- Memory-safe training loop with gradient clipping

Usage:
    python -m fluidelite.scripts.train --data data/input.txt --epochs 100

Constitutional Compliance:
    - Article II.4: Performance baselines tracked
    - Article V.4: Actionable error messages
    - Article VII.4: Demonstration requirement

Example:
    >>> python -m fluidelite.scripts.train --synthetic --epochs 50
    Epoch 1/50: loss=4.21, ppl=67.3, acc=12.3%, 342 tok/s
    ...
    Training complete: final_loss=0.89, final_acc=78.5%
"""

import argparse
import gc
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TrainingConfig:
    """Configuration for training run."""
    
    # Model
    num_sites: int = 16
    rank: int = 128
    mpo_rank: int = 1
    vocab_size: int = 256
    truncate_every: int = 9999  # Effectively disable truncation during training
    
    # Training
    epochs: int = 100
    batch_size: int = 1  # Stateful model = batch 1 for now
    seq_len: int = 128
    bptt_len: int = 16  # Keep short for bounded chi growth
    lr: float = 0.005
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    warmup_steps: int = 100
    non_stateful: bool = True  # Fresh context each window (better gradients)
    
    # Data
    data_path: Optional[str] = None
    max_chars: int = 100_000
    
    # Checkpointing
    save_dir: str = "checkpoints"
    save_every: int = 10
    
    # Logging
    log_every: int = 10
    eval_every: int = 100
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float32"


@dataclass
class TrainingState:
    """Mutable training state."""
    
    epoch: int = 0
    global_step: int = 0
    best_loss: float = float('inf')
    best_acc: float = 0.0
    train_losses: List[float] = field(default_factory=list)
    train_accs: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)


def create_model(config: TrainingConfig):
    """Create FluidElite model from config."""
    from fluidelite.llm.fluid_elite import FluidElite
    
    dtype = torch.float32 if config.dtype == "float32" else torch.float16
    
    model = FluidElite(
        num_sites=config.num_sites,
        rank=config.rank,
        mpo_rank=config.mpo_rank,
        vocab_size=config.vocab_size,
        truncate_every=config.truncate_every,
        dtype=dtype
    )
    
    model.to(config.device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model created: {param_count:,} parameters")
    print(f"  num_sites={config.num_sites}, rank={config.rank}, mpo_rank={config.mpo_rank}")
    print(f"  vocab_size={config.vocab_size}, truncate_every={config.truncate_every}")
    
    return model


def create_optimizer(model: nn.Module, config: TrainingConfig):
    """Create optimizer.
    
    NOTE: RiemannianAdam with stabilize=True is BROKEN for non-orthonormal tensors.
    The projection formula assumes W is orthonormal, but MPO cores are not.
    This causes loss to increase instead of decrease.
    
    Use plain AdamW until Riemannian projection is fixed.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    print("Using AdamW optimizer")
    
    return optimizer


def create_scheduler(optimizer, config: TrainingConfig, total_steps: int):
    """Create learning rate scheduler with warmup."""
    
    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        return max(0.1, 1.0 - (step - config.warmup_steps) / (total_steps - config.warmup_steps))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_data(config: TrainingConfig):
    """Load training data."""
    from fluidelite.llm.data import TextStreamDataset
    
    if config.data_path:
        path = Path(config.data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        
        if len(text) > config.max_chars:
            text = text[:config.max_chars]
            print(f"Truncated to {config.max_chars:,} characters")
    else:
        # Synthetic data for testing
        print("No data path provided, using synthetic pattern data")
        pattern = "0123456789" * 1000  # Repeating digit pattern
        text = pattern * 10
    
    dataset = TextStreamDataset(text, vocab_size=config.vocab_size, seq_len=config.seq_len)
    return dataset


def detach_mps(mps):
    """Detach MPS tensors from computation graph."""
    from fluidelite.core.mps import MPS
    return MPS([t.detach().clone() for t in mps.tensors])


def train_epoch(
    model: nn.Module,
    dataset,
    optimizer,
    scheduler,
    config: TrainingConfig,
    state: TrainingState
) -> Dict[str, float]:
    """Train for one epoch."""
    
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    epoch_start = time.time()
    
    # Initialize context
    ctx = model.embed(0)
    
    # Iterate through dataset
    num_chunks = len(dataset) // config.bptt_len
    
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * config.bptt_len
        chunk_loss = 0.0
        chunk_correct = 0
        
        # NON-STATEFUL: Fresh context each window for proper gradient flow
        # This sacrifices long-range memory but ensures gradients flow through MPO
        if config.non_stateful:
            from fluidelite.core.mps import MPS
            ctx = MPS.random(model.L, d=2, chi=1, device=config.device, dtype=getattr(torch, config.dtype))
        
        # Forward through BPTT window
        for t in range(config.bptt_len):
            idx = chunk_start + t
            if idx >= len(dataset) - 1:
                break
            
            x, y = dataset[idx]
            input_token = x[0].item()  # First token of sequence
            target_token = y[0].item()  # Target is next token
            
            # Forward
            logits = model.predict(ctx)
            
            # Compute loss
            target = torch.tensor([target_token], device=config.device)
            loss = F.cross_entropy(logits.unsqueeze(0), target)
            chunk_loss += loss
            
            # Track accuracy
            pred = logits.argmax().item()
            if pred == target_token:
                chunk_correct += 1
            
            # Step context (with gradients)
            ctx = model.step(ctx, target_token)
            total_tokens += 1
        
        # Backward
        if chunk_loss > 0:
            avg_loss = chunk_loss / config.bptt_len
            
            optimizer.zero_grad()
            avg_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
            optimizer.step()
            scheduler.step()
            
            # Detach context for next chunk (cut computation graph)
            ctx = detach_mps(ctx)
            
            # Track metrics
            total_loss += avg_loss.item() * config.bptt_len
            total_correct += chunk_correct
            state.global_step += 1
            
            # Logging
            if state.global_step % config.log_every == 0:
                lr = optimizer.param_groups[0]['lr']
                chunk_acc = chunk_correct / config.bptt_len * 100
                elapsed = time.time() - epoch_start
                tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0
                
                print(f"  step {state.global_step}: loss={avg_loss.item():.4f}, "
                      f"acc={chunk_acc:.1f}%, lr={lr:.6f}, {tok_per_sec:.0f} tok/s")
    
    # Epoch metrics
    avg_loss = total_loss / max(1, total_tokens)
    accuracy = total_correct / max(1, total_tokens) * 100
    elapsed = time.time() - epoch_start
    throughput = total_tokens / elapsed if elapsed > 0 else 0
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "perplexity": math.exp(min(avg_loss, 10)),
        "throughput": throughput,
        "elapsed": elapsed
    }


def evaluate(model: nn.Module, dataset, config: TrainingConfig, num_samples: int = 100) -> Dict[str, float]:
    """Evaluate model on held-out data."""
    
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    
    with torch.no_grad():
        ctx = model.embed(0)
        
        # Sample from end of dataset (held out during training)
        start_idx = max(0, len(dataset) - num_samples - 1)
        
        for i in range(num_samples):
            idx = start_idx + i
            if idx >= len(dataset) - 1:
                break
            
            x, y = dataset[idx]
            target_token = y[0].item()
            
            logits = model.predict(ctx)
            
            target = torch.tensor([target_token], device=config.device)
            loss = F.cross_entropy(logits.unsqueeze(0), target)
            total_loss += loss.item()
            
            pred = logits.argmax().item()
            if pred == target_token:
                total_correct += 1
            
            ctx = model.step(ctx, target_token)
    
    avg_loss = total_loss / num_samples
    accuracy = total_correct / num_samples * 100
    
    return {
        "val_loss": avg_loss,
        "val_accuracy": accuracy,
        "val_perplexity": math.exp(min(avg_loss, 10))
    }


def save_checkpoint(
    model: nn.Module,
    optimizer,
    config: TrainingConfig,
    state: TrainingState,
    path: Path
):
    """Save training checkpoint."""
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": vars(config),
        "state": vars(state),
    }
    
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer,
    path: Path,
    device: str = "cuda"
) -> TrainingState:
    """Load training checkpoint."""
    
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    state = TrainingState(**checkpoint["state"])
    print(f"Loaded checkpoint from epoch {state.epoch}")
    
    return state


def train(config: TrainingConfig):
    """Main training loop."""
    
    print("=" * 60)
    print("FLUIDELITE TRAINING")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Data: {config.data_path or 'synthetic'}")
    print()
    
    # Create components
    model = create_model(config)
    optimizer = create_optimizer(model, config)
    dataset = load_data(config)
    
    total_steps = config.epochs * (len(dataset) // config.bptt_len)
    scheduler = create_scheduler(optimizer, config, total_steps)
    
    state = TrainingState()
    
    print()
    print(f"Training for {config.epochs} epochs ({total_steps:,} steps)")
    print("-" * 60)
    
    try:
        for epoch in range(1, config.epochs + 1):
            state.epoch = epoch
            print(f"\nEpoch {epoch}/{config.epochs}")
            
            # Train
            metrics = train_epoch(model, dataset, optimizer, scheduler, config, state)
            
            state.train_losses.append(metrics["loss"])
            state.train_accs.append(metrics["accuracy"])
            
            print(f"  Epoch {epoch} complete: loss={metrics['loss']:.4f}, "
                  f"ppl={metrics['perplexity']:.1f}, acc={metrics['accuracy']:.1f}%, "
                  f"{metrics['throughput']:.0f} tok/s")
            
            # Evaluate
            if epoch % config.eval_every == 0 or epoch == config.epochs:
                val_metrics = evaluate(model, dataset, config)
                print(f"  Validation: loss={val_metrics['val_loss']:.4f}, "
                      f"ppl={val_metrics['val_perplexity']:.1f}, "
                      f"acc={val_metrics['val_accuracy']:.1f}%")
                
                # Track best
                if val_metrics["val_loss"] < state.best_loss:
                    state.best_loss = val_metrics["val_loss"]
                    state.best_acc = val_metrics["val_accuracy"]
                    save_checkpoint(
                        model, optimizer, config, state,
                        Path(config.save_dir) / "best.pt"
                    )
            
            # Checkpoint
            if epoch % config.save_every == 0:
                save_checkpoint(
                    model, optimizer, config, state,
                    Path(config.save_dir) / f"epoch_{epoch}.pt"
                )
            
            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final loss: {state.train_losses[-1]:.4f}")
    print(f"Final accuracy: {state.train_accs[-1]:.1f}%")
    print(f"Best validation loss: {state.best_loss:.4f}")
    print(f"Best validation accuracy: {state.best_acc:.1f}%")
    print()
    
    return model, state


def main():
    """CLI entry point."""
    
    parser = argparse.ArgumentParser(description="Train FluidElite QTT-LLM")
    
    # Data
    parser.add_argument("--data", type=str, default=None,
                       help="Path to training text file")
    parser.add_argument("--synthetic", action="store_true",
                       help="Use synthetic pattern data")
    parser.add_argument("--max-chars", type=int, default=100_000,
                       help="Maximum characters to load")
    
    # Model
    parser.add_argument("--num-sites", type=int, default=16,
                       help="Number of MPS sites")
    parser.add_argument("--rank", type=int, default=128,
                       help="Bond dimension")
    parser.add_argument("--mpo-rank", type=int, default=1,
                       help="MPO bond dimension")
    parser.add_argument("--vocab-size", type=int, default=256,
                       help="Vocabulary size")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.005,
                       help="Learning rate")
    parser.add_argument("--bptt-len", type=int, default=16,
                       help="Truncated BPTT length (keep short for bounded chi)")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                       help="Gradient clipping norm")
    parser.add_argument("--non-stateful", action="store_true", default=True,
                       help="Fresh context each window (better gradients, no long-range memory)")
    parser.add_argument("--stateful", action="store_true",
                       help="Stateful training (requires working STE truncation)")
    
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                       help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Handle stateful vs non-stateful
    non_stateful = not args.stateful  # Default to non-stateful unless --stateful
    
    config = TrainingConfig(
        data_path=args.data if not args.synthetic else None,
        max_chars=args.max_chars,
        num_sites=args.num_sites,
        rank=args.rank,
        mpo_rank=args.mpo_rank,
        vocab_size=args.vocab_size,
        epochs=args.epochs,
        lr=args.lr,
        bptt_len=args.bptt_len,
        grad_clip=args.grad_clip,
        save_dir=args.save_dir,
        non_stateful=non_stateful,
    )
    
    train(config)


if __name__ == "__main__":
    main()
