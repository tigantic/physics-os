#!/usr/bin/env python3
"""
QTT Scaling Experiment with Rank Sweeps at Each Scale

Following NS Millennium methodology: sweep rank at EACH scale point.
Rank 24 optimal at 4.2M may not be optimal at 67M or 256M.

Scale Points (dense equivalent):
  - 16K × 256 = 4.2M (baseline, optimal rank = 24)
  - 65K × 256 = 16.7M (4× scale)
  - 262K × 256 = 67M (16× scale)
  - 1M × 256 = 256M (64× scale)
  - 4M × 256 = 1B (target)

At each scale: sweep ranks [16, 24, 32, 48, 64, 96] to find optimal.
"""

import torch
import torch.nn.functional as F
import time
import math
import gc
from typing import List, Tuple, Dict

# Check CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================================
# QTT Core Operations
# ============================================================================

def init_qtt_cores(n_qubits: int, max_rank: int) -> List[torch.Tensor]:
    """Xavier-initialized QTT cores."""
    cores = []
    for i in range(n_qubits):
        r_left = 1 if i == 0 else min(max_rank, 2**i, 2**(n_qubits-i))
        r_right = 1 if i == n_qubits-1 else min(max_rank, 2**(i+1), 2**(n_qubits-i-1))
        r_left = min(r_left, max_rank)
        r_right = min(r_right, max_rank)
        
        std = math.sqrt(2.0 / (r_left + r_right))
        core = torch.randn(r_left, 2, r_right, device=device) * std
        cores.append(core)
    return cores


def qtt_to_dense(cores: List[torch.Tensor]) -> torch.Tensor:
    """Convert QTT cores to dense vector."""
    result = cores[0]  # (1, 2, r)
    for core in cores[1:]:
        r_left, _, r_mid = result.shape
        r_mid2, _, r_right = core.shape
        # Contract: result[i,j,k] * core[k,l,m] -> new[i,j*l,m]
        result = torch.einsum('ijk,klm->ijlm', result, core)
        result = result.reshape(r_left, -1, r_right)
    return result.squeeze(0).squeeze(-1)  # (2^n,)


def qtt_forward(feat_cores: List[torch.Tensor], vocab_cores: List[torch.Tensor], 
                X: torch.Tensor) -> torch.Tensor:
    """QTT forward pass: X @ W where W is QTT-compressed."""
    # Reconstruct W from cores
    feat_vec = qtt_to_dense(feat_cores)   # (n_features,)
    vocab_vec = qtt_to_dense(vocab_cores)  # (n_vocab,)
    
    # W = outer(feat_vec, vocab_vec) but we compute X @ W directly
    # X @ W = X @ (feat ⊗ vocab) = (X @ feat) * vocab
    projected = X @ feat_vec  # (batch,)
    logits = projected.unsqueeze(1) * vocab_vec.unsqueeze(0)  # (batch, vocab)
    return logits


def count_params(cores: List[torch.Tensor]) -> int:
    """Count parameters in QTT cores."""
    return sum(c.numel() for c in cores)


# ============================================================================
# Feature Extraction (Triton-style hashing)
# ============================================================================

def extract_features_chunked(data: torch.Tensor, n_features: int, 
                             context_len: int = 8, chunk_size: int = 50000) -> torch.Tensor:
    """Memory-efficient feature extraction."""
    n_samples = len(data) - context_len
    all_features = []
    
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk_features = torch.zeros(end - start, n_features, device=device)
        
        for i in range(end - start):
            ctx = data[start + i:start + i + context_len]
            for pos, token in enumerate(ctx):
                # Multiple hash functions for better coverage
                h1 = ((token.item() * 2654435761 + pos * 805306457) % n_features)
                h2 = ((token.item() * 1597334677 + pos * 789456123) % n_features)
                h3 = ((token.item() ^ (pos << 8)) * 2246822519 % n_features)
                chunk_features[i, h1] += 1.0
                chunk_features[i, h2] += 0.5
                chunk_features[i, h3] += 0.25
        
        all_features.append(chunk_features)
    
    return torch.cat(all_features, dim=0)


# ============================================================================
# Training Function
# ============================================================================

def train_qtt_scale(n_feat_qubits: int, n_vocab_qubits: int, max_rank: int,
                    n_samples: int = 100000, n_epochs: int = 5, 
                    batch_size: int = 512, lr: float = 0.03) -> Dict:
    """Train QTT at given scale and return metrics."""
    
    n_features = 2 ** n_feat_qubits
    n_vocab = 2 ** n_vocab_qubits
    dense_equiv = n_features * n_vocab
    
    print(f"\n{'='*60}")
    print(f"Scale: {n_features:,} features × {n_vocab} vocab = {dense_equiv:,} dense")
    print(f"Rank: {max_rank}")
    print(f"{'='*60}")
    
    # Initialize QTT cores
    feat_cores = init_qtt_cores(n_feat_qubits, max_rank)
    vocab_cores = init_qtt_cores(n_vocab_qubits, max_rank)
    
    # Make trainable
    for c in feat_cores + vocab_cores:
        c.requires_grad_(True)
    
    all_cores = feat_cores + vocab_cores
    qtt_params = count_params(feat_cores) + count_params(vocab_cores)
    compression = dense_equiv / qtt_params
    
    print(f"QTT params: {qtt_params:,}")
    print(f"Compression: {compression:.1f}×")
    
    # Load data
    try:
        with open('/tmp/wikitext_sample.txt', 'rb') as f:
            text = f.read()
        data = torch.tensor(list(text[:n_samples + 1000]), dtype=torch.long, device=device)
    except FileNotFoundError:
        print("Downloading WikiText sample...")
        import urllib.request
        url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
        urllib.request.urlretrieve(url, '/tmp/wikitext_sample.txt')
        with open('/tmp/wikitext_sample.txt', 'rb') as f:
            text = f.read()
        data = torch.tensor(list(text[:n_samples + 1000]), dtype=torch.long, device=device)
    
    # Extract features (this is the slow part for large n_features)
    print(f"Extracting {n_features:,}-dim features...")
    t0 = time.time()
    
    # For very large feature spaces, use chunked extraction
    if n_features > 100000:
        X = extract_features_chunked(data, n_features, chunk_size=20000)
    else:
        X = extract_features_chunked(data, n_features, chunk_size=50000)
    
    targets = data[8:8+len(X)] % n_vocab  # Byte-level targets
    
    print(f"Feature extraction: {time.time()-t0:.1f}s")
    print(f"X shape: {X.shape}, memory: {X.numel() * 4 / 1e6:.1f} MB")
    
    # Training
    optimizer = torch.optim.Adam(all_cores, lr=lr)
    
    t0 = time.time()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        perm = torch.randperm(len(X), device=device)
        
        for i in range(0, len(X), batch_size):
            idx = perm[i:i+batch_size]
            X_batch = X[idx]
            y_batch = targets[idx]
            
            optimizer.zero_grad()
            
            logits = qtt_forward(feat_cores, vocab_cores, X_batch)
            loss = F.cross_entropy(logits, y_batch)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        if epoch == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch+1}: loss={epoch_loss/n_batches:.4f}")
    
    train_time = time.time() - t0
    
    # Evaluate
    with torch.no_grad():
        # Sample evaluation (memory-efficient)
        eval_size = min(10000, len(X))
        eval_idx = torch.randperm(len(X))[:eval_size]
        X_eval = X[eval_idx]
        y_eval = targets[eval_idx]
        
        logits = qtt_forward(feat_cores, vocab_cores, X_eval)
        preds = logits.argmax(dim=1)
        accuracy = (preds == y_eval).float().mean().item() * 100
        
        probs = F.softmax(logits, dim=1)
        probs_correct = probs[torch.arange(len(y_eval)), y_eval]
        perplexity = torch.exp(-torch.log(probs_correct + 1e-10).mean()).item()
    
    # Cleanup
    del X, targets, X_eval, logits
    gc.collect()
    torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    result = {
        'n_features': n_features,
        'n_vocab': n_vocab,
        'dense_equiv': dense_equiv,
        'rank': max_rank,
        'qtt_params': qtt_params,
        'compression': compression,
        'accuracy': accuracy,
        'perplexity': perplexity,
        'train_time': train_time,
    }
    
    print(f"  Accuracy: {accuracy:.1f}%")
    print(f"  Perplexity: {perplexity:.2f}")
    print(f"  Time: {train_time:.1f}s")
    
    return result


# ============================================================================
# Main Experiment
# ============================================================================

def run_scale_sweep():
    """Run rank sweep at multiple scales."""
    
    # Scale points: (n_feat_qubits, n_vocab_qubits, dense_equivalent)
    # SKIP 16K - already swept, optimal rank = 24
    scales = [
        # (14, 8, "4.2M"),    # DONE: optimal rank = 24
        (16, 8, "16.7M"),   # 4× scale: 65K × 256
        (18, 8, "67M"),     # 16× scale: 262K × 256
        # (20, 8, "256M"),  # 64× scale: 1M × 256 (may OOM)
    ]
    
    # Ranks to sweep at each scale
    ranks = [16, 24, 32, 48, 64, 96]
    
    all_results = []
    
    print("\n" + "="*70)
    print("QTT SCALING EXPERIMENT WITH RANK SWEEPS")
    print("Following NS Millennium methodology: optimal rank varies with scale")
    print("="*70)
    
    for n_feat_qubits, n_vocab_qubits, scale_name in scales:
        print(f"\n\n{'#'*70}")
        print(f"# SCALE: {scale_name} dense equivalent")
        print(f"# Features: 2^{n_feat_qubits} = {2**n_feat_qubits:,}")
        print(f"# Vocab: 2^{n_vocab_qubits} = {2**n_vocab_qubits}")
        print(f"{'#'*70}")
        
        scale_results = []
        
        for rank in ranks:
            try:
                result = train_qtt_scale(
                    n_feat_qubits=n_feat_qubits,
                    n_vocab_qubits=n_vocab_qubits,
                    max_rank=rank,
                    n_samples=100000,
                    n_epochs=5,
                    batch_size=512,
                )
                result['scale_name'] = scale_name
                scale_results.append(result)
                all_results.append(result)
                
            except torch.cuda.OutOfMemoryError:
                print(f"  ⚠️ OOM at rank {rank}, skipping")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"  ⚠️ Error at rank {rank}: {e}")
                continue
        
        # Print scale summary
        if scale_results:
            print(f"\n📊 SCALE {scale_name} SUMMARY:")
            print(f"{'Rank':<8} {'Accuracy':<12} {'Perplexity':<12} {'Params':<12} {'Compression':<12}")
            print("-" * 60)
            
            best_acc = max(scale_results, key=lambda x: x['accuracy'])
            best_ppl = min(scale_results, key=lambda x: x['perplexity'])
            
            for r in scale_results:
                acc_mark = " ⭐" if r == best_acc else ""
                ppl_mark = " 🎯" if r == best_ppl else ""
                print(f"{r['rank']:<8} {r['accuracy']:.1f}%{acc_mark:<6} "
                      f"{r['perplexity']:.2f}{ppl_mark:<6} "
                      f"{r['qtt_params']:,}  {r['compression']:.0f}×")
            
            print(f"\n🏆 Optimal rank at {scale_name}: {best_acc['rank']} (accuracy), {best_ppl['rank']} (perplexity)")
    
    # Final summary
    print("\n\n" + "="*70)
    print("FINAL RESULTS: OPTIMAL RANK BY SCALE")
    print("="*70)
    print(f"{'Scale':<12} {'Best Rank':<12} {'Accuracy':<12} {'Perplexity':<12} {'Compression':<12}")
    print("-" * 60)
    
    # Group by scale
    from itertools import groupby
    for scale_name, group in groupby(sorted(all_results, key=lambda x: x['dense_equiv']), 
                                      key=lambda x: x['scale_name']):
        group_list = list(group)
        best = max(group_list, key=lambda x: x['accuracy'])
        print(f"{scale_name:<12} {best['rank']:<12} {best['accuracy']:.1f}%       "
              f"{best['perplexity']:.2f}        {best['compression']:.0f}×")
    
    return all_results


if __name__ == '__main__':
    results = run_scale_sweep()
