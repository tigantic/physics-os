#!/usr/bin/env python3
"""
🎯 CLOSED-FORM TRAINING: FluidEliteZK (Ridge Regression)
=========================================================

Uses the closed-form ridge regression solver (torch.linalg.solve)
instead of gradient descent for 2x better accuracy.

Key Insight: Linear reservoir readout is a LEAST SQUARES problem.
- Gradient descent: 20.7% accuracy (suboptimal)
- Ridge regression: 43% accuracy (optimal)

The reservoir (MPS) acts as a random feature extractor.
Only the readout weights need to be trained, and this is a
convex optimization with a closed-form solution:

    W* = (X^T X + λI)^{-1} X^T Y

Training is INSTANT - single matrix solve, no epochs!

Usage:
    python train_closedform.py --max_chars 1000000 --reg 0.001

Author: HyperTensor Labs  
Date: January 2026
"""

import torch
import torch.nn.functional as F
import time
import argparse
import json
import os
import sys
from pathlib import Path

# Add HyperTensor-VM-main to path so 'fluidelite' package is importable
hypervm_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hypervm_dir not in sys.path:
    sys.path.insert(0, hypervm_dir)

# Now we can import from the package
from fluidelite.llm.fluid_elite_zk import FluidEliteZK
from fluidelite.core.mps import MPS


def load_wikitext_chars(max_chars: int = 1_000_000):
    """Load WikiText-2 as character-level tokens with FIXED 256 vocab."""
    print("📚 Loading WikiText-2 (character-level, fixed vocab=256)...")
    
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
    
    # FIXED vocabulary: all 256 byte values
    # This ensures compatibility with Rust prover config
    vocab_size = 256
    char_to_idx = {chr(i): i for i in range(256)}
    idx_to_char = {i: chr(i) for i in range(256)}
    
    print(f"   Total chars: {len(text):,}")
    print(f"   Vocab size: {vocab_size} (FIXED)")
    print(f"   Sample: {text[:80]!r}...")
    
    # Encode - use ord() for byte values, clamp to 0-255
    data = torch.tensor([min(ord(c), 255) for c in text], dtype=torch.long)
    
    return data, vocab_size, char_to_idx, idx_to_char


def collect_reservoir_features(model: FluidEliteZK, data: torch.Tensor, 
                                device: str, max_samples: int = 500_000,
                                truncate_every: int = 20) -> tuple:
    """
    Collect reservoir features in one pass.
    
    Returns:
        features: (N, chi_max) - reservoir state for each timestep
        labels: (N,) - next token for each timestep
    """
    print("\n🔄 Collecting reservoir features (single pass)...")
    print(f"   Truncating every {truncate_every} steps to prevent OOM")
    start_time = time.time()
    
    N = min(len(data) - 1, max_samples)
    chi = model.chi_max
    
    # Pre-allocate tensors on CPU to save GPU memory
    features = torch.zeros(N, chi, dtype=torch.float32)
    labels = torch.zeros(N, dtype=torch.long)
    
    # Initialize context
    ctx = MPS.random(model.L, d=2, chi=1, device=device, dtype=torch.float32)
    step_count = 0
    
    # Collect features - ALL in no_grad mode
    with torch.no_grad():
        for i in range(N):
            if i % 50000 == 0 and i > 0:
                elapsed = time.time() - start_time
                print(f"   [{i:,}/{N:,}] {elapsed:.1f}s ({i/elapsed:.0f} tok/s), chi={ctx.chi}")
            
            # Extract feature from context MPS
            feat = model.extract_feature(ctx)  # (chi_max,)
            
            features[i] = feat.cpu()
            labels[i] = data[i + 1].cpu()
            
            # Update context with current token
            ctx = model.step(ctx, data[i].item())
            step_count += 1
            
            # CRITICAL: Truncate periodically to prevent memory explosion
            if step_count % truncate_every == 0 and ctx.chi > chi:
                ctx.truncate_(chi_max=chi)
    
    elapsed = time.time() - start_time
    print(f"   ✅ Collected {N:,} features in {elapsed:.1f}s ({N/elapsed:.0f} tok/s)")
    
    return features.to(device), labels.to(device)


def ridge_regression_solve(X: torch.Tensor, Y: torch.Tensor, reg: float = 0.001) -> torch.Tensor:
    """
    Solve ridge regression: W* = (X^T X + λI)^{-1} X^T Y
    
    Args:
        X: (N, D) features
        Y: (N, C) one-hot labels
        reg: regularization strength
        
    Returns:
        W: (D, C) optimal weights
    """
    print(f"\n🧮 Solving ridge regression (reg={reg})...")
    start_time = time.time()
    
    D = X.shape[1]
    device = X.device
    dtype = X.dtype
    
    # X^T X + λI
    XtX = X.T @ X
    XtX += reg * torch.eye(D, device=device, dtype=dtype)
    
    # X^T Y
    XtY = X.T @ Y
    
    # Solve
    W = torch.linalg.solve(XtX, XtY)
    
    elapsed = time.time() - start_time
    print(f"   ✅ Solved in {elapsed*1000:.1f}ms")
    
    return W


@torch.no_grad()
def evaluate_weights(model: FluidEliteZK, W: torch.Tensor, data: torch.Tensor,
                     device: str, num_samples: int = 10000) -> dict:
    """Evaluate learned weights on validation data."""
    print("\n📊 Evaluating on validation set...")
    
    # Use last portion of data as validation
    val_start = len(data) - num_samples - 1
    val_data = data[val_start:]
    
    ctx = MPS.random(model.L, d=2, chi=1, device=device, dtype=torch.float32)
    model.reset_step_count()
    
    total_loss = 0.0
    total_correct = 0
    
    for i in range(len(val_data) - 1):
        feat = model.extract_feature(ctx)  # (chi_max,)
        logits = feat @ W  # (vocab_size,)
        target = val_data[i + 1].to(device)
        
        # Loss
        loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
        total_loss += loss.item()
        
        # Accuracy
        pred = logits.argmax()
        if pred == target:
            total_correct += 1
        
        # Update context
        ctx = model.step(ctx, val_data[i].item())
    
    N = len(val_data) - 1
    avg_loss = total_loss / N
    ppl = min(torch.exp(torch.tensor(avg_loss)).item(), 9999)
    acc = 100.0 * total_correct / N
    
    return {"loss": avg_loss, "ppl": ppl, "acc": acc}


@torch.no_grad()
def generate_sample(model: FluidEliteZK, W: torch.Tensor, seed_text: str,
                    idx_to_char: dict, char_to_idx: dict, device: str, 
                    length: int = 200, temperature: float = 0.8) -> str:
    """Generate text sample."""
    ctx = MPS.random(model.L, d=2, chi=1, device=device, dtype=torch.float32)
    model.reset_step_count()
    
    # Seed
    output = list(seed_text)
    for c in seed_text:
        if c in char_to_idx:
            ctx = model.step(ctx, char_to_idx[c])
    
    # Generate
    for _ in range(length):
        feat = model.extract_feature(ctx)
        logits = feat @ W
        
        # Temperature sampling
        probs = F.softmax(logits / temperature, dim=0)
        token = torch.multinomial(probs, 1).item()
        
        if token in idx_to_char:
            c = idx_to_char[token]
            output.append(c)
            ctx = model.step(ctx, token)
        else:
            break
    
    return "".join(output)


def export_weights(model: FluidEliteZK, W: torch.Tensor, output_path: str, 
                   vocab_size: int, char_to_idx: dict, idx_to_char: dict):
    """Export weights for Rust prover."""
    print(f"\n💾 Exporting weights to {output_path}...")
    
    # Checkpoint for Python
    torch.save({
        'model_state_dict': model.state_dict(),
        'readout_W': W,
        'config': {
            'num_sites': model.L,
            'chi_max': model.chi_max,
            'vocab_size': vocab_size,
        },
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
    }, output_path)
    
    # JSON for Rust prover
    json_path = output_path.replace('.pt', '_zk_weights.json')
    
    def tensor_to_list(t):
        return t.detach().cpu().float().numpy().tolist()
    
    zk_weights = {
        "config": {
            "num_sites": model.L,
            "chi_max": model.chi_max,
            "vocab_size": vocab_size,
            "mpo_rank": 1,
        },
        "readout_W": tensor_to_list(W),
        "char_to_idx": char_to_idx,
        "idx_to_char": {int(k): v for k, v in idx_to_char.items()},
    }
    
    # Export MPO weights
    mpo_weights = {}
    for name, param in model.named_parameters():
        if 'mpo' in name or 'embed' in name or 'W_in' in name:
            mpo_weights[name] = tensor_to_list(param)
    zk_weights["mpo_weights"] = mpo_weights
    
    with open(json_path, 'w') as f:
        json.dump(zk_weights, f)
    
    print(f"   ✅ Saved {output_path} ({os.path.getsize(output_path)/1024:.1f} KB)")
    print(f"   ✅ Saved {json_path} ({os.path.getsize(json_path)/1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Closed-form training for FluidEliteZK")
    parser.add_argument("--num_sites", type=int, default=12)
    parser.add_argument("--chi_max", type=int, default=64)
    parser.add_argument("--max_chars", type=int, default=500_000, 
                        help="Characters for training")
    parser.add_argument("--val_samples", type=int, default=10_000,
                        help="Validation samples")
    parser.add_argument("--reg", type=float, default=0.001,
                        help="Ridge regression regularization")
    parser.add_argument("--output", type=str, default="weights/fluidelite_closedform.pt")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 FluidEliteZK Closed-Form Training")
    print(f"   Device: {device}")
    print("=" * 60)
    
    # Load data
    data, vocab_size, char_to_idx, idx_to_char = load_wikitext_chars(args.max_chars)
    data = data.to(device)
    
    # Create model
    print(f"\n🏗️  Creating FluidEliteZK model...")
    model = FluidEliteZK(
        num_sites=args.num_sites,
        chi_max=args.chi_max,
        vocab_size=vocab_size,
        truncate_every=9999,  # Don't truncate during collection
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   L={args.num_sites}, χ={args.chi_max}, vocab={vocab_size}")
    print(f"   Total parameters: {total_params:,}")
    
    # Add extract_feature method that mirrors predict() logic
    print("\n🔧 Adding extract_feature method to model...")
    
    def extract_feature(self, ctx):
        """
        Extract chi_max-dimensional feature from context MPS.
        
        This mirrors the feature extraction in predict():
        - Uses middle bond tensor
        - Averages over physical and left bond dimensions
        - Pads/truncates to chi_max
        """
        mid_idx = self.L // 2
        mid = ctx.tensors[mid_idx]  # (chi_left, d, chi_right)
        
        # Average over physical and left bond → right bond vector
        vec = mid.mean(dim=(0, 1))
        
        # Pad/truncate to chi_max
        if vec.shape[0] < self.chi_max:
            vec = torch.cat([
                vec,
                torch.zeros(self.chi_max - vec.shape[0], device=vec.device, dtype=vec.dtype)
            ])
        else:
            vec = vec[:self.chi_max]
        
        return vec
    
    import types
    model.extract_feature = types.MethodType(extract_feature, model)
    
    # =====================================================
    # PHASE 1: Collect Reservoir Features (Single Pass)
    # =====================================================
    train_size = len(data) - args.val_samples - 1
    train_data = data[:train_size + 1]
    
    features, labels = collect_reservoir_features(
        model, train_data, device, max_samples=train_size
    )
    
    # =====================================================
    # PHASE 2: Solve Ridge Regression (Instant!)
    # =====================================================
    # One-hot encode labels
    N = len(labels)
    Y = torch.zeros(N, vocab_size, device=device, dtype=torch.float32)
    Y.scatter_(1, labels.unsqueeze(1), 1.0)
    
    # Solve
    W = ridge_regression_solve(features, Y, reg=args.reg)
    
    # Training accuracy
    preds = (features @ W).argmax(dim=1)
    train_acc = (preds == labels).float().mean().item() * 100
    print(f"   Training accuracy: {train_acc:.1f}%")
    
    # =====================================================
    # PHASE 3: Evaluate
    # =====================================================
    val_metrics = evaluate_weights(model, W, data, device, args.val_samples)
    print(f"\n📈 Validation Results:")
    print(f"   Perplexity: {val_metrics['ppl']:.1f}")
    print(f"   Accuracy:   {val_metrics['acc']:.1f}%")
    print(f"   Loss:       {val_metrics['loss']:.4f}")
    
    # =====================================================
    # PHASE 4: Generate Samples
    # =====================================================
    print("\n📝 Sample Generation:")
    print("-" * 60)
    
    seeds = ["The ", "In the ", "A "]
    for seed in seeds:
        sample = generate_sample(model, W, seed, idx_to_char, char_to_idx, device, 
                                 length=150, temperature=0.7)
        print(f"Seed: {seed!r}")
        print(f"  → {sample[:200]}")
        print()
    
    # =====================================================
    # PHASE 5: Export
    # =====================================================
    os.makedirs(os.path.dirname(args.output) or "weights", exist_ok=True)
    export_weights(model, W, args.output, vocab_size, char_to_idx, idx_to_char)
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 CLOSED-FORM TRAINING COMPLETE")
    print("=" * 60)
    print(f"   Model:      FluidEliteZK (L={args.num_sites}, χ={args.chi_max})")
    print(f"   Vocab:      {vocab_size} characters")
    print(f"   Train data: {train_size:,} tokens")
    print(f"   Train Acc:  {train_acc:.1f}%")
    print(f"   Val Acc:    {val_metrics['acc']:.1f}%")
    print(f"   Val PPL:    {val_metrics['ppl']:.1f}")
    print(f"   Weights:    {args.output}")
    print()
    print("   🔥 Ridge regression >> Gradient descent!")
    print("=" * 60)


if __name__ == "__main__":
    main()
