#!/usr/bin/env python3
"""
Train FluidElite on WikiText-2 - Real Text Validation

Uses the winning configuration from synthetic training:
- mpo_rank=1 (prevents chi explosion)
- truncate_every=9999 (disable truncation during training)
- non_stateful=True (fresh context each window)
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from datasets import load_dataset
import time
import argparse
from dataclasses import dataclass
from typing import Optional
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fluidelite.llm.fluid_elite import FluidElite
    from fluidelite.core.mps import MPS
except ImportError:
    from llm.fluid_elite import FluidElite
    from core.mps import MPS


@dataclass
class WikiTextConfig:
    # Model
    num_sites: int = 12
    rank: int = 64
    mpo_rank: int = 1  # Key: prevents chi explosion
    truncate_every: int = 9999  # Key: disable truncation during training
    
    # Training
    epochs: int = 3
    batch_size: int = 1  # FluidElite is stateful, process one sequence at a time
    bptt_len: int = 32  # Longer window for language
    lr: float = 1e-3
    
    # Data
    max_tokens: int = 100_000  # Limit for quick testing
    tokenizer_name: str = "gpt2"  # 50257 vocab, but we'll use character-level first
    use_char_level: bool = True  # Start simple: character-level
    
    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every: int = 100


def load_wikitext_chars(max_chars: int = 500_000) -> tuple[torch.Tensor, int]:
    """Load WikiText-2 as character-level tokens."""
    print("Loading WikiText-2 (character-level)...")
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    
    # Concatenate all text
    text = "\n".join([s['text'] for s in ds if s['text'].strip()])
    text = text[:max_chars]
    
    # Build character vocabulary
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)
    
    print(f"  Total chars: {len(text):,}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Sample: {text[:100]!r}")
    
    # Encode
    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)
    
    return data, vocab_size, chars


def load_wikitext_subword(max_tokens: int = 100_000) -> tuple[torch.Tensor, int]:
    """Load WikiText-2 with GPT-2 tokenizer."""
    print("Loading WikiText-2 (subword tokenizer)...")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    
    # Concatenate and tokenize
    text = "\n".join([s['text'] for s in ds if s['text'].strip()])
    tokens = tokenizer.encode(text)[:max_tokens]
    
    print(f"  Total tokens: {len(tokens):,}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    
    data = torch.tensor(tokens, dtype=torch.long)
    
    return data, tokenizer.vocab_size, tokenizer


def train_epoch(model: FluidElite, data: torch.Tensor, optimizer: torch.optim.Optimizer,
                config: WikiTextConfig, epoch: int) -> dict:
    """Train one epoch on WikiText data."""
    model.train()
    device = config.device
    
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    start_time = time.time()
    
    # Process data in BPTT windows
    num_windows = (len(data) - 1) // config.bptt_len
    
    for window_idx in range(num_windows):
        # Extract window
        start = window_idx * config.bptt_len
        end = start + config.bptt_len + 1  # +1 for target
        if end > len(data):
            break
            
        window = data[start:end].to(device)
        inputs = window[:-1]
        targets = window[1:]
        
        # Fresh context each window (non-stateful)
        ctx = MPS.random(model.L, d=2, chi=1, device=device, dtype=torch.float32)
        
        # Forward pass through window
        loss = torch.tensor(0.0, device=device)
        correct = 0
        
        for t in range(len(inputs)):
            logits = model.predict(ctx)  # [vocab_size]
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Stats
        total_loss += loss.item()
        total_correct += correct
        total_tokens += len(inputs)
        
        # Logging
        if (window_idx + 1) % config.log_every == 0:
            elapsed = time.time() - start_time
            tok_per_sec = total_tokens / elapsed
            avg_loss = total_loss / (window_idx + 1)
            acc = 100.0 * total_correct / total_tokens
            
            print(f"  window {window_idx+1}/{num_windows}: "
                  f"loss={avg_loss:.4f}, acc={acc:.1f}%, {tok_per_sec:.0f} tok/s")
    
    # Epoch summary
    elapsed = time.time() - start_time
    avg_loss = total_loss / max(1, num_windows)
    acc = 100.0 * total_correct / max(1, total_tokens)
    ppl = torch.exp(torch.tensor(avg_loss / config.bptt_len)).item()
    
    return {
        "loss": avg_loss,
        "ppl": ppl,
        "acc": acc,
        "tok_per_sec": total_tokens / elapsed,
        "elapsed": elapsed
    }


def evaluate(model: FluidElite, data: torch.Tensor, config: WikiTextConfig, 
             vocab_info, num_samples: int = 5) -> dict:
    """Evaluate on held-out data."""
    model.eval()
    device = config.device
    
    # Use last 10% of data for validation
    val_start = int(len(data) * 0.9)
    val_data = data[val_start:]
    
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        num_windows = min(100, (len(val_data) - 1) // config.bptt_len)
        
        for window_idx in range(num_windows):
            start = window_idx * config.bptt_len
            end = start + config.bptt_len + 1
            if end > len(val_data):
                break
                
            window = val_data[start:end].to(device)
            inputs = window[:-1]
            targets = window[1:]
            
            ctx = MPS.random(model.L, d=2, chi=1, device=device, dtype=torch.float32)
            
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
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    acc = 100.0 * total_correct / max(1, total_tokens)
    
    # Generate sample
    print("\n--- Sample Generation ---")
    with torch.no_grad():
        # Start with a seed
        seed_text = "The "
        if isinstance(vocab_info, list):  # char-level
            char_to_idx = {c: i for i, c in enumerate(vocab_info)}
            idx_to_char = vocab_info
            seed_tokens = [char_to_idx.get(c, 0) for c in seed_text]
        else:  # subword
            seed_tokens = vocab_info.encode(seed_text)
            idx_to_char = None
        
        ctx = MPS.random(model.L, d=2, chi=1, device=device, dtype=torch.float32)
        
        # Process seed
        for tok in seed_tokens:
            ctx = model.step(ctx, tok)
        
        # Generate
        generated = list(seed_tokens)
        for _ in range(50):
            logits = model.predict(ctx)
            # Sample with temperature
            probs = F.softmax(logits / 0.8, dim=-1)
            next_tok = torch.multinomial(probs, 1).item()
            generated.append(next_tok)
            ctx = model.step(ctx, next_tok)
        
        # Decode
        if idx_to_char:
            text = "".join(idx_to_char[t] for t in generated)
        else:
            text = vocab_info.decode(generated)
        
        print(f"Generated: {text!r}")
    
    return {"val_loss": avg_loss, "val_ppl": ppl, "val_acc": acc}


def main():
    parser = argparse.ArgumentParser(description="Train FluidElite on WikiText-2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--bptt-len", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--mpo-rank", type=int, default=1)
    parser.add_argument("--max-chars", type=int, default=500_000)
    parser.add_argument("--subword", action="store_true", help="Use GPT-2 tokenizer")
    args = parser.parse_args()
    
    config = WikiTextConfig(
        epochs=args.epochs,
        bptt_len=args.bptt_len,
        lr=args.lr,
        rank=args.rank,
        mpo_rank=args.mpo_rank,
        use_char_level=not args.subword,
    )
    
    print("=" * 70)
    print("FLUIDELITE WIKITEXT TRAINING")
    print("=" * 70)
    print(f"Config: rank={config.rank}, mpo_rank={config.mpo_rank}, bptt={config.bptt_len}")
    print(f"Device: {config.device}")
    print()
    
    # Load data
    if config.use_char_level:
        data, vocab_size, vocab_info = load_wikitext_chars(args.max_chars)
    else:
        data, vocab_size, vocab_info = load_wikitext_subword(config.max_tokens)
    
    print()
    
    # Create model
    model = FluidElite(
        num_sites=config.num_sites,
        rank=config.rank,
        mpo_rank=config.mpo_rank,
        vocab_size=vocab_size,
        truncate_every=config.truncate_every,
    ).to(config.device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params:,} parameters")
    print(f"  num_sites={model.L}, rank={config.rank}, mpo_rank={config.mpo_rank}")
    print(f"  vocab_size={vocab_size}, truncate_every={config.truncate_every}")
    print()
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config.lr)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    
    for epoch in range(config.epochs):
        print(f"Epoch {epoch + 1}/{config.epochs}")
        print("-" * 40)
        
        # Train
        train_stats = train_epoch(model, data, optimizer, config, epoch)
        train_losses.append(train_stats["loss"])
        
        print(f"\nTrain: loss={train_stats['loss']:.4f}, ppl={train_stats['ppl']:.2f}, "
              f"acc={train_stats['acc']:.1f}%, {train_stats['tok_per_sec']:.0f} tok/s")
        
        # Evaluate
        val_stats = evaluate(model, data, config, vocab_info)
        print(f"Val: loss={val_stats['val_loss']:.4f}, ppl={val_stats['val_ppl']:.2f}, "
              f"acc={val_stats['val_acc']:.1f}%")
        
        if val_stats["val_loss"] < best_val_loss:
            best_val_loss = val_stats["val_loss"]
            print("  *** New best! ***")
        
        print()
    
    # Summary
    print("=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Train losses: {' -> '.join(f'{l:.3f}' for l in train_losses)}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best val PPL: {torch.exp(torch.tensor(best_val_loss)).item():.2f}")
    
    loss_decreased = train_losses[-1] < train_losses[0]
    if loss_decreased:
        print("\n🎉 LOSS DECREASED ON REAL TEXT! Model is learning natural language.")
    else:
        print("\n⚠️  Loss did not decrease. May need more epochs or tuning.")
    
    return loss_decreased


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
