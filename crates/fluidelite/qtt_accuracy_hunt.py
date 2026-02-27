#!/usr/bin/env python3
"""
QTT Accuracy Hunt - NS Millennium Methodology

"Track rank evolution - don't fix rank, watch it grow and see where it stabilizes."

Input: target_accuracy
Output: minimal rank needed, compression falls out

Let the physics tell you the rank.
"""

import torch
import torch.nn.functional as F
import time
import math
import gc
from typing import List, Dict, Optional

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
    result = cores[0]
    for core in cores[1:]:
        result = torch.einsum('ijk,klm->ijlm', result, core)
        r_left = result.shape[0]
        r_right = result.shape[-1]
        result = result.reshape(r_left, -1, r_right)
    return result.squeeze(0).squeeze(-1)


def qtt_forward(feat_cores: List[torch.Tensor], vocab_cores: List[torch.Tensor], 
                X: torch.Tensor) -> torch.Tensor:
    """QTT forward pass."""
    feat_vec = qtt_to_dense(feat_cores)
    vocab_vec = qtt_to_dense(vocab_cores)
    projected = X @ feat_vec
    logits = projected.unsqueeze(1) * vocab_vec.unsqueeze(0)
    return logits


def count_params(cores: List[torch.Tensor]) -> int:
    return sum(c.numel() for c in cores)


# ============================================================================
# Feature Extraction
# ============================================================================

def extract_features(data: torch.Tensor, n_features: int, context_len: int = 8) -> torch.Tensor:
    """Extract features with hashing."""
    n_samples = len(data) - context_len
    X = torch.zeros(n_samples, n_features, device=device)
    
    for i in range(n_samples):
        ctx = data[i:i + context_len]
        for pos, token in enumerate(ctx):
            h1 = ((token.item() * 2654435761 + pos * 805306457) % n_features)
            h2 = ((token.item() * 1597334677 + pos * 789456123) % n_features)
            h3 = ((token.item() ^ (pos << 8)) * 2246822519 % n_features)
            X[i, h1] += 1.0
            X[i, h2] += 0.5
            X[i, h3] += 0.25
    
    return X


# ============================================================================
# Training Function
# ============================================================================

def train_and_eval(n_feat_qubits: int, n_vocab_qubits: int, max_rank: int,
                   X: torch.Tensor, targets: torch.Tensor,
                   n_epochs: int = 5, batch_size: int = 512, lr: float = 0.03) -> Dict:
    """Train at given rank, return accuracy."""
    
    feat_cores = init_qtt_cores(n_feat_qubits, max_rank)
    vocab_cores = init_qtt_cores(n_vocab_qubits, max_rank)
    
    for c in feat_cores + vocab_cores:
        c.requires_grad_(True)
    
    all_cores = feat_cores + vocab_cores
    qtt_params = count_params(feat_cores) + count_params(vocab_cores)
    
    optimizer = torch.optim.Adam(all_cores, lr=lr)
    
    for epoch in range(n_epochs):
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
    
    # Evaluate
    with torch.no_grad():
        eval_size = min(10000, len(X))
        eval_idx = torch.randperm(len(X))[:eval_size]
        logits = qtt_forward(feat_cores, vocab_cores, X[eval_idx])
        preds = logits.argmax(dim=1)
        accuracy = (preds == targets[eval_idx]).float().mean().item() * 100
        
        probs = F.softmax(logits, dim=1)
        probs_correct = probs[torch.arange(len(eval_idx)), targets[eval_idx]]
        perplexity = torch.exp(-torch.log(probs_correct + 1e-10).mean()).item()
    
    return {
        'rank': max_rank,
        'params': qtt_params,
        'accuracy': accuracy,
        'perplexity': perplexity,
    }


# ============================================================================
# NS Millennium: Find Minimal Rank for Target Accuracy
# ============================================================================

def find_minimal_rank(target_accuracy: float, n_feat_qubits: int, n_vocab_qubits: int,
                      X: torch.Tensor, targets: torch.Tensor,
                      max_attempts: int = 20) -> Dict:
    """
    NS Millennium methodology: Find the MINIMAL rank that achieves target accuracy.
    
    Start at rank=2, grow until target is met.
    Let the physics tell you the rank.
    """
    
    n_features = 2 ** n_feat_qubits
    n_vocab = 2 ** n_vocab_qubits
    dense_equiv = n_features * n_vocab
    
    print(f"\n{'='*60}")
    print(f"TARGET: {target_accuracy:.1f}% accuracy")
    print(f"Scale: {n_features:,} × {n_vocab} = {dense_equiv:,} dense equivalent")
    print(f"{'='*60}")
    
    # Start low, grow rank until target met
    rank = 2
    best_result = None
    history = []
    
    while rank <= 512 and len(history) < max_attempts:
        print(f"  Testing rank {rank}...", end=" ", flush=True)
        
        try:
            result = train_and_eval(n_feat_qubits, n_vocab_qubits, rank, X, targets)
            result['compression'] = dense_equiv / result['params']
            history.append(result)
            
            print(f"accuracy={result['accuracy']:.1f}%, params={result['params']:,}")
            
            if result['accuracy'] >= target_accuracy:
                print(f"\n  ✅ TARGET MET at rank {rank}!")
                best_result = result
                break
            
            # Adaptive rank growth
            if result['accuracy'] < target_accuracy * 0.5:
                rank = int(rank * 2)  # Far from target, jump
            elif result['accuracy'] < target_accuracy * 0.8:
                rank = int(rank * 1.5)  # Getting closer
            else:
                rank = rank + 4  # Fine-tune near target
                
        except torch.cuda.OutOfMemoryError:
            print(f"OOM at rank {rank}")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"Error: {e}")
            break
    
    if best_result is None and history:
        # Target not met - report best achieved
        best_result = max(history, key=lambda x: x['accuracy'])
        print(f"\n  ⚠️ TARGET NOT MET. Best: rank {best_result['rank']} → {best_result['accuracy']:.1f}%")
    
    return {
        'target': target_accuracy,
        'achieved': best_result,
        'history': history,
    }


# ============================================================================
# Main: Sweep Target Accuracies
# ============================================================================

def run_accuracy_hunt():
    """Find minimal rank for each target accuracy level."""
    
    print("\n" + "="*70)
    print("QTT ACCURACY HUNT - NS MILLENNIUM METHODOLOGY")
    print("Input: target accuracy")
    print("Output: minimal rank needed, compression falls out")
    print("="*70)
    
    # Scale to test (start with baseline, then scale up)
    scales = [
        (14, 8, "4.2M"),    # 16K × 256
        (16, 8, "16.7M"),   # 65K × 256
    ]
    
    # Target accuracies to hunt for
    targets = [90.0, 80.0, 70.0, 60.0, 50.0, 40.0]
    
    # Load data once
    print("\nLoading data...")
    try:
        with open('/tmp/wikitext_sample.txt', 'rb') as f:
            text = f.read()
    except FileNotFoundError:
        print("Downloading WikiText sample...")
        import urllib.request
        url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
        urllib.request.urlretrieve(url, '/tmp/wikitext_sample.txt')
        with open('/tmp/wikitext_sample.txt', 'rb') as f:
            text = f.read()
    
    n_samples = 100000
    data = torch.tensor(list(text[:n_samples + 1000]), dtype=torch.long, device=device)
    
    all_results = {}
    
    for n_feat_qubits, n_vocab_qubits, scale_name in scales:
        n_features = 2 ** n_feat_qubits
        n_vocab = 2 ** n_vocab_qubits
        
        print(f"\n\n{'#'*70}")
        print(f"# SCALE: {scale_name} ({n_features:,} features)")
        print(f"{'#'*70}")
        
        # Extract features for this scale
        print(f"Extracting {n_features:,}-dim features...")
        t0 = time.time()
        X = extract_features(data, n_features)
        targets_tensor = data[8:8+len(X)] % n_vocab
        print(f"  Done in {time.time()-t0:.1f}s")
        
        scale_results = []
        
        for target_acc in targets:
            result = find_minimal_rank(
                target_accuracy=target_acc,
                n_feat_qubits=n_feat_qubits,
                n_vocab_qubits=n_vocab_qubits,
                X=X,
                targets=targets_tensor,
            )
            scale_results.append(result)
            
            # Early exit if we can't even hit 50%
            if result['achieved'] and result['achieved']['accuracy'] < 35:
                print(f"\n  Accuracy plateau detected. Stopping hunt.")
                break
        
        all_results[scale_name] = scale_results
        
        # Print scale summary
        print(f"\n\n{'='*70}")
        print(f"SCALE {scale_name} SUMMARY")
        print(f"{'='*70}")
        print(f"{'Target':<12} {'Achieved':<12} {'Rank':<8} {'Params':<12} {'Compression':<12}")
        print("-" * 60)
        
        for r in scale_results:
            if r['achieved']:
                a = r['achieved']
                print(f"{r['target']:.0f}%        {a['accuracy']:.1f}%       "
                      f"{a['rank']:<8} {a['params']:,}    {a['compression']:.0f}×")
        
        # Cleanup
        del X, targets_tensor
        gc.collect()
        torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    # Final summary
    print("\n\n" + "="*70)
    print("FINAL RESULTS: NS MILLENNIUM ACCURACY HUNT")
    print("="*70)
    print("\nKey insight: What rank does the PHYSICS demand for each accuracy level?")
    print("Not: What accuracy do we get for each rank?\n")
    
    for scale_name, results in all_results.items():
        print(f"\n{scale_name}:")
        for r in results:
            if r['achieved']:
                a = r['achieved']
                status = "✅" if a['accuracy'] >= r['target'] else "⚠️"
                print(f"  {status} {r['target']:.0f}% target → rank {a['rank']} → "
                      f"{a['accuracy']:.1f}% actual → {a['compression']:.0f}× compression")
    
    return all_results


if __name__ == '__main__':
    results = run_accuracy_hunt()
