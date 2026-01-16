#!/usr/bin/env python3
"""
Train FluidElite 1.65B on Real Text
====================================

The moment of truth: Can a 1.65B parameter QTT-LLM learn natural language
with O(1) memory?

Configuration:
    - rank=32768 (1.65B parameters)
    - mpo_rank=1 (prevents chi explosion)
    - chi_max=256 (bounds runtime memory to ~6.2GB)
    - truncate_every=20 during inference, 9999 during training

Usage:
    python -m fluidelite.scripts.train_billion --epochs 3

Memory Budget:
    Model params: 6.14GB
    Runtime chi:  ~0.1GB (bounded by chi_max)
    Total:        ~6.2GB fits in 8GB VRAM
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import time
import argparse
import sys
import os
import gc

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fluidelite.llm.fluid_elite import FluidElite
    from fluidelite.core.mps import MPS
except ImportError:
    from llm.fluid_elite import FluidElite
    from core.mps import MPS


def format_params(n: int) -> str:
    """Format parameter count nicely."""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def load_wikitext_subword(max_tokens: int = 100_000):
    """Load WikiText-2 with GPT-2 tokenizer for true 1.65B scale."""
    try:
        from datasets import load_dataset
        from transformers import GPT2Tokenizer
    except ImportError:
        print("Installing required libraries...")
        os.system("pip install datasets transformers -q")
        from datasets import load_dataset
        from transformers import GPT2Tokenizer
    
    print("Loading WikiText-2 with GPT-2 tokenizer...")
    
    # Use GPT2Tokenizer directly (more stable than AutoTokenizer)
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=False)
    except Exception as e:
        print(f"  Tokenizer download failed: {e}")
        print("  Falling back to character-level...")
        return None, None, None
    
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    
    # Concatenate and tokenize
    text = "\n".join([s['text'] for s in ds if s['text'].strip()])
    tokens = tokenizer.encode(text)[:max_tokens]
    
    print(f"  Tokens: {len(tokens):,}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    
    data = torch.tensor(tokens, dtype=torch.long)
    
    return data, tokenizer.vocab_size, tokenizer


def load_wikitext_chars(max_chars: int = 1_000_000):
    """Load WikiText-2 as character-level tokens."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        os.system("pip install datasets -q")
        from datasets import load_dataset
    
    print("Loading WikiText-2...")
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    
    # Concatenate all text
    text = "\n".join([s['text'] for s in ds if s['text'].strip()])
    text = text[:max_chars]
    
    # Build character vocabulary
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)
    
    print(f"  Text length: {len(text):,} chars")
    print(f"  Vocabulary: {vocab_size} unique chars")
    
    # Encode
    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)
    
    return data, vocab_size, chars, char_to_idx


def train_epoch(model, data, optimizer, device, bptt_len=16, log_every=50, 
                epoch=0, max_steps=None):
    """Train one epoch."""
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    start_time = time.time()
    
    num_windows = (len(data) - 1) // bptt_len
    if max_steps:
        num_windows = min(num_windows, max_steps)
    
    for window_idx in range(num_windows):
        # Extract window
        start = window_idx * bptt_len
        end = start + bptt_len + 1
        if end > len(data):
            break
        
        window = data[start:end].to(device)
        inputs = window[:-1]
        targets = window[1:]
        
        # Fresh context each window (non-stateful training)
        ctx = MPS.random(model.L, d=2, chi=1, device=device, dtype=torch.float32)
        
        # Forward pass
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        correct = 0
        
        for t in range(len(inputs)):
            logits = model.predict(ctx)
            target = targets[t]
            
            step_loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
            loss = loss + step_loss
            
            pred = logits.argmax()
            if pred == target:
                correct += 1
            
            ctx = model.step(ctx, inputs[t].item())
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Stats
        total_loss += loss.item()
        total_correct += correct
        total_tokens += len(inputs)
        
        # Log progress
        if (window_idx + 1) % log_every == 0:
            elapsed = time.time() - start_time
            tok_per_sec = total_tokens / elapsed
            avg_loss = total_loss / (window_idx + 1)
            acc = 100.0 * total_correct / total_tokens
            
            # Memory info
            if torch.cuda.is_available():
                mem_gb = torch.cuda.max_memory_allocated() / 1e9
                mem_str = f", VRAM={mem_gb:.2f}GB"
            else:
                mem_str = ""
            
            print(f"  [{window_idx+1:5d}/{num_windows}] "
                  f"loss={avg_loss:.4f}, acc={acc:.1f}%, "
                  f"{tok_per_sec:.0f} tok/s{mem_str}")
        
        # Periodic cleanup
        if window_idx % 100 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    elapsed = time.time() - start_time
    avg_loss = total_loss / max(1, num_windows)
    acc = 100.0 * total_correct / max(1, total_tokens)
    
    return {
        "loss": avg_loss,
        "acc": acc,
        "tok_per_sec": total_tokens / elapsed
    }


def evaluate(model, data, tokenizer, device, bptt_len=16, num_windows=100, is_subword=True):
    """Evaluate and generate sample."""
    model.eval()
    
    # Validation on last 10%
    val_start = int(len(data) * 0.9)
    val_data = data[val_start:]
    
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for window_idx in range(min(num_windows, len(val_data) // bptt_len)):
            start = window_idx * bptt_len
            end = start + bptt_len + 1
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
                
                if logits.argmax() == target:
                    total_correct += 1
                
                total_tokens += 1
                ctx = model.step(ctx, inputs[t].item())
    
    avg_loss = total_loss / max(1, total_tokens)
    acc = 100.0 * total_correct / max(1, total_tokens)
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    
    # Generate sample
    print("\n  Sample generation:")
    with torch.no_grad():
        ctx = MPS.random(model.L, d=2, chi=1, device=device, dtype=torch.float32)
        
        # Seed with "The "
        seed = "The "
        if is_subword:
            seed_tokens = tokenizer.encode(seed)
        else:
            char_to_idx = {c: i for i, c in enumerate(tokenizer)}
            seed_tokens = [char_to_idx.get(c, 0) for c in seed]
        
        for tok in seed_tokens:
            ctx = model.step(ctx, tok)
        
        # Generate tokens
        generated = list(seed_tokens)
        for _ in range(30 if is_subword else 100):
            logits = model.predict(ctx)
            probs = F.softmax(logits / 0.8, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            generated.append(next_idx)
            ctx = model.step(ctx, next_idx)
        
        # Decode
        if is_subword:
            text = tokenizer.decode(generated)
        else:
            text = "".join(tokenizer[t] if t < len(tokenizer) else '?' for t in generated)
        
        print(f"  >>> {text}")
    
    return {"val_loss": avg_loss, "val_acc": acc, "val_ppl": ppl}


def main():
    parser = argparse.ArgumentParser(description="Train FluidElite 1.65B")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rank", type=int, default=32768, help="32768=1.65B params")
    parser.add_argument("--bptt-len", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-tokens", type=int, default=100_000)
    parser.add_argument("--max-steps", type=int, default=None, help="Limit steps per epoch")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--chi-max", type=int, default=256)
    parser.add_argument("--char-level", action="store_true", help="Use character-level instead of GPT-2 tokenizer")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_subword = not args.char_level
    
    print("=" * 70)
    print("   FLUIDELITE 1.65B TRAINING")
    print("   Real Text • O(1) Memory • The Billion-Parameter QTT-LLM")
    print("=" * 70)
    print()
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f}GB)")
    else:
        print("WARNING: No GPU detected, training will be slow")
    print()
    
    # Load data
    if is_subword:
        data, vocab_size, tokenizer = load_wikitext_subword(args.max_tokens)
    else:
        data, vocab_size, chars, _ = load_wikitext_chars(args.max_tokens)
        tokenizer = chars  # For generation
    print()
    
    # Create model
    print(f"Creating model with vocab_size={vocab_size}...")
    model = FluidElite(
        num_sites=16,
        rank=args.rank,
        mpo_rank=1,
        vocab_size=vocab_size,
        truncate_every=9999,  # Disable during training
        chi_max=args.chi_max
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {format_params(num_params)}")
    print(f"  rank={args.rank}, mpo_rank=1, chi_max={args.chi_max}")
    print(f"  vocab_size={vocab_size}")
    
    if torch.cuda.is_available():
        model_mem = torch.cuda.memory_allocated() / 1e9
        print(f"  Model VRAM: {model_mem:.2f}GB")
    print()
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    print(f"Optimizer: AdamW (lr={args.lr})")
    print()
    
    # Training loop
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    train_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)
        
        train_stats = train_epoch(
            model, data, optimizer, device,
            bptt_len=args.bptt_len,
            log_every=args.log_every,
            epoch=epoch,
            max_steps=args.max_steps
        )
        train_losses.append(train_stats["loss"])
        
        print(f"\n  Train: loss={train_stats['loss']:.4f}, "
              f"acc={train_stats['acc']:.1f}%, "
              f"{train_stats['tok_per_sec']:.0f} tok/s")
        
        # Evaluate
        val_stats = evaluate(model, data, tokenizer, device, bptt_len=args.bptt_len, is_subword=is_subword)
        print(f"  Val: loss={val_stats['val_loss']:.4f}, "
              f"ppl={val_stats['val_ppl']:.1f}, "
              f"acc={val_stats['val_acc']:.1f}%")
        
        if val_stats["val_loss"] < best_val_loss:
            best_val_loss = val_stats["val_loss"]
            print("  *** New best! ***")
    
    # Summary
    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    loss_improved = len(train_losses) > 1 and train_losses[-1] < train_losses[0]
    
    print(f"Parameters: {format_params(num_params)}")
    print(f"Train losses: {' -> '.join(f'{l:.3f}' for l in train_losses)}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best val PPL: {torch.exp(torch.tensor(best_val_loss)).item():.1f}")
    
    if loss_improved:
        print("\n🎉 SUCCESS: Loss decreased! The 1.65B QTT-LLM is learning!")
    else:
        print("\n⚠️  Loss didn't decrease. May need more epochs.")
    
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"\nPeak VRAM: {peak_mem:.2f}GB")
    
    return loss_improved


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
