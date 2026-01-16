#!/usr/bin/env python3
"""
QTT via TCI - Let Physics Determine Rank

NO GRADIENTS. Build oracle, SVD, read the spectrum.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import time
import math
from datasets import load_dataset

device = torch.device('cuda')

N_FEATURES = 16384   # 2^14
N_VOCAB = 256        # 2^8
CTX_LEN = 16

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================================
# Triton Feature Extraction (same as before)
# ============================================================================

@triton.jit
def extract_features_kernel(data_ptr, positions_ptr, features_ptr, stride_feat):
    pid = tl.program_id(0)
    pos = tl.load(positions_ptr + pid)
    out_base = features_ptr + pid * stride_feat
    
    for i in tl.static_range(4):
        byte_val = tl.load(data_ptr + pos + 16 - 4 + i).to(tl.int32)
        idx = (i * 256 + byte_val) % 1024
        tl.atomic_add(out_base + idx, 1.0)
    
    for i in tl.static_range(15):
        b1 = tl.load(data_ptr + pos + i).to(tl.int32)
        b2 = tl.load(data_ptr + pos + i + 1).to(tl.int32)
        idx = 1024 + ((i * 65537 + b1 * 257 + b2) % 4096)
        tl.atomic_add(out_base + idx, 1.0)
    
    for i in tl.static_range(14):
        b1 = tl.load(data_ptr + pos + i).to(tl.int32)
        b2 = tl.load(data_ptr + pos + i + 1).to(tl.int32)
        b3 = tl.load(data_ptr + pos + i + 2).to(tl.int32)
        idx = 5120 + ((b1 * 65537 + b2 * 257 + b3) % 4096)
        tl.atomic_add(out_base + idx, 1.0)
    
    for i in tl.static_range(13):
        b1 = tl.load(data_ptr + pos + i).to(tl.int32)
        b2 = tl.load(data_ptr + pos + i + 1).to(tl.int32)
        b3 = tl.load(data_ptr + pos + i + 2).to(tl.int32)
        b4 = tl.load(data_ptr + pos + i + 3).to(tl.int32)
        idx = 9216 + ((b1 * 16777259 + b2 * 65537 + b3 * 257 + b4) % 4096)
        tl.atomic_add(out_base + idx, 1.0)
    
    for i in tl.static_range(14):
        b1 = tl.load(data_ptr + pos + i).to(tl.int32)
        b3 = tl.load(data_ptr + pos + i + 2).to(tl.int32)
        idx = 13312 + ((b1 * 257 + b3) % 3072)
        tl.atomic_add(out_base + idx, 1.0)


def extract_features(data: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    B = positions.shape[0]
    features = torch.zeros((B, N_FEATURES), dtype=torch.float32, device=device)
    extract_features_kernel[(B,)](data, positions, features, N_FEATURES)
    return features


# ============================================================================
# TCI: Build Oracle, SVD, Compress
# ============================================================================

def build_oracle_matrix(train_data: torch.Tensor, n_samples: int, batch_size: int = 10000):
    """
    Build the oracle by solving least squares: find W that minimizes ||XW - Y||^2
    where X is features and Y is one-hot targets.
    
    This is the closed-form solution to what SGD is approximating!
    W* = (X^T X)^{-1} X^T Y
    """
    print(f"\n  Building oracle via least squares from {n_samples:,} samples...")
    
    # Accumulate X^T X and X^T Y
    XtX = torch.zeros(N_FEATURES, N_FEATURES, dtype=torch.float32, device=device)
    XtY = torch.zeros(N_FEATURES, N_VOCAB, dtype=torch.float32, device=device)
    
    n_train = len(train_data) - CTX_LEN - 1
    step = max(1, n_train // n_samples)
    
    start = time.time()
    processed = 0
    
    for b_start in range(0, n_samples, batch_size):
        b_end = min(b_start + batch_size, n_samples)
        batch_n = b_end - b_start
        
        # Get positions for this batch
        positions = torch.arange(b_start * step, b_end * step, step, dtype=torch.int64, device=device)[:batch_n]
        targets = train_data[positions + CTX_LEN]
        
        # Extract features
        X = extract_features(train_data, positions)  # [batch, N_FEATURES]
        
        # One-hot targets
        Y = torch.zeros(batch_n, N_VOCAB, device=device)
        Y.scatter_(1, targets.long().unsqueeze(1), 1.0)
        
        # Accumulate
        XtX += X.T @ X
        XtY += X.T @ Y
        
        processed += batch_n
        if processed % 100000 == 0:
            elapsed = time.time() - start
            print(f"    {processed:,}/{n_samples:,} ({100*processed/n_samples:.1f}%) - {elapsed:.1f}s", flush=True)
    
    elapsed = time.time() - start
    print(f"  XtX and XtY built in {elapsed:.1f}s")
    
    # Solve normal equations: W = (X^T X + λI)^{-1} X^T Y
    # Add regularization for numerical stability
    print(f"  Solving normal equations...")
    start = time.time()
    
    lambda_reg = 1e-6 * torch.trace(XtX).item() / N_FEATURES
    XtX_reg = XtX + lambda_reg * torch.eye(N_FEATURES, device=device)
    
    W = torch.linalg.solve(XtX_reg, XtY)
    
    elapsed = time.time() - start
    print(f"  Solved in {elapsed:.1f}s")
    print(f"  W shape: {W.shape}")
    print(f"  W range: [{W.min().item():.4f}, {W.max().item():.4f}]")
    
    return W


def analyze_spectrum(W: torch.Tensor):
    """
    SVD the oracle matrix. Let the singular values speak.
    """
    print(f"\n  Computing SVD of {W.shape[0]}×{W.shape[1]} matrix...")
    
    start = time.time()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    elapsed = time.time() - start
    
    print(f"  SVD done in {elapsed:.1f}s")
    print(f"  Singular values shape: {S.shape}")
    
    # Analyze spectrum
    S_np = S.cpu().numpy()
    total_var = (S**2).sum().item()
    cumvar = torch.cumsum(S**2, dim=0) / total_var
    
    print(f"\n  === SPECTRUM ANALYSIS ===")
    print(f"  Top 10 singular values: {S_np[:10]}")
    print(f"  S[0]/S[-1] ratio: {S_np[0]/S_np[-1]:.2e}")
    
    # Find ranks for various thresholds
    for thresh in [0.90, 0.95, 0.99, 0.999]:
        rank = (cumvar >= thresh).nonzero()[0][0].item() + 1
        print(f"  Rank for {thresh*100:.1f}% variance: {rank}")
    
    return U, S, Vh, cumvar


def compress_to_rank(U, S, Vh, rank: int):
    """
    Compress to given rank: W_approx = U[:,:r] @ diag(S[:r]) @ Vh[:r,:]
    """
    W_compressed = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]
    return W_compressed


def evaluate_compression(W_original, W_compressed, test_data, test_pos, test_targets):
    """
    Evaluate both original and compressed matrices.
    """
    results = {}
    
    for name, W in [("Original", W_original), ("Compressed", W_compressed)]:
        with torch.no_grad():
            X = extract_features(test_data, test_pos)
            
            # Normalize W to get log-probabilities
            # W is counts, convert to probabilities per feature
            # Actually, just use as logits
            logits = X @ W
            probs = F.softmax(logits, dim=1)
            
            preds = probs.argmax(dim=1)
            accuracy = (preds == test_targets).float().mean().item() * 100
            
            p_correct = probs[torch.arange(len(test_targets), device=device), test_targets.long()]
            perplexity = torch.exp(-torch.log(p_correct.clamp(min=1e-10)).mean()).item()
        
        results[name] = {'accuracy': accuracy, 'perplexity': perplexity}
        print(f"  {name}: accuracy={accuracy:.1f}%, perplexity={perplexity:.2f}")
    
    return results


def main():
    print("\n" + "="*70)
    print("QTT via TCI - PHYSICS DETERMINES RANK")
    print("="*70)
    print("\nNO GRADIENTS. Build oracle → SVD → Read spectrum → Compress.")
    
    # Load data
    print("\nLoading WikiText-103...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    train_text = "\n".join(dataset["train"]["text"]).encode('utf-8')
    test_text = "\n".join(dataset["test"]["text"]).encode('utf-8')
    
    train_data = torch.tensor(list(train_text), dtype=torch.int32, device=device)
    test_data = torch.tensor(list(test_text), dtype=torch.int32, device=device)
    print(f"Train: {len(train_data)/1e6:.1f}M bytes, Test: {len(test_data)/1e6:.1f}M bytes")
    
    # Test positions
    n_test = len(test_data) - CTX_LEN - 1
    TEST_N = 20000
    test_step = n_test // TEST_N
    test_pos = torch.arange(0, n_test, test_step, dtype=torch.int64, device=device)[:TEST_N]
    test_targets = test_data[test_pos + CTX_LEN]
    
    # =========================================================================
    # STEP 1: Build Oracle Matrix
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 1: BUILD ORACLE MATRIX")
    print(f"{'='*70}")
    
    N_SAMPLES = 2_000_000  # 2M samples to build oracle
    W = build_oracle_matrix(train_data, N_SAMPLES)
    
    # =========================================================================
    # STEP 2: SVD - Let Physics Speak
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 2: SVD - PHYSICS SPEAKS")
    print(f"{'='*70}")
    
    U, S, Vh, cumvar = analyze_spectrum(W)
    
    # =========================================================================
    # STEP 3: Evaluate at Different Ranks
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 3: EVALUATE COMPRESSIONS")
    print(f"{'='*70}")
    
    dense_params = N_FEATURES * N_VOCAB
    
    print(f"\n  Original (dense): {dense_params:,} params")
    
    for rank in [10, 24, 50, 100, 200, 256]:
        print(f"\n  --- Rank {rank} ---")
        W_comp = compress_to_rank(U, S, Vh, rank)
        
        # Params for rank-r: U is 16K×r, S is r, Vh is r×256
        # But we store as W_comp which is 16K×256
        # For actual compression: store U[:,:r], S[:r], Vh[:r,:]
        stored_params = N_FEATURES * rank + rank + rank * N_VOCAB
        compression = dense_params / stored_params
        
        print(f"  Stored params: {stored_params:,} ({compression:.1f}× compression)")
        evaluate_compression(W, W_comp, test_data, test_pos, test_targets)
    
    # =========================================================================
    # FINAL: Report Natural Rank
    # =========================================================================
    print(f"\n{'='*70}")
    print("FINAL: PHYSICS-DETERMINED RANK")
    print(f"{'='*70}")
    
    # Find rank for 99% variance
    rank_99 = (cumvar >= 0.99).nonzero()[0][0].item() + 1
    print(f"\n  Natural rank (99% variance): {rank_99}")
    
    W_natural = compress_to_rank(U, S, Vh, rank_99)
    stored_params = N_FEATURES * rank_99 + rank_99 + rank_99 * N_VOCAB
    compression = dense_params / stored_params
    
    print(f"  Stored params: {stored_params:,} ({compression:.1f}× compression)")
    print(f"\n  Performance at natural rank:")
    evaluate_compression(W, W_natural, test_data, test_pos, test_targets)


if __name__ == '__main__':
    main()
