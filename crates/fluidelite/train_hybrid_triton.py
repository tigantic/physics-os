#!/usr/bin/env python3
"""
Train FluidEliteHybrid with TRITON kernels for maximum speed.

Architecture (from FINDINGS.md):
- Lookup table: 100% on seen contexts (O(1) hash)
- Least squares fallback: 46% on unseen (rank-24 compressed W)

No MPS. No MPO. No runtime truncation.
"""
import torch
import triton
import triton.language as tl
import time
import math
import os
import struct
import hashlib
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === CONFIG (Production Spec) ===
L = 12           # Context length
VOCAB = 256      # Byte-level
CHI = 24         # Rank for W compression

# Feature dimensions (adapted for L=12)
N_UNI = 1024     # 4 positions × 256 = 1024
N_BI = 8192      # Bigram hashes
N_TRI = 8192     # Trigram hashes
N_SKIP = 4096    # Skipgram hashes
TOTAL_F = N_UNI + N_BI + N_TRI + N_SKIP  # 21504


@triton.jit
def extract_features_kernel_L12(
    data_ptr,        # [N] bytes as int32
    positions_ptr,   # [B] start positions
    features_ptr,    # [B, TOTAL_F] output
    stride_feat,     # stride for features
):
    """Triton kernel for L=12 context feature extraction."""
    pid = tl.program_id(0)
    
    pos = tl.load(positions_ptr + pid)
    out_base = features_ptr + pid * stride_feat
    
    # === UNIGRAMS (last 4 bytes) ===
    for i in tl.static_range(4):
        byte_val = tl.load(data_ptr + pos + 12 - 4 + i)  # L=12
        idx = (i * 256 + byte_val) % 1024
        tl.atomic_add(out_base + idx, 1.0)
    
    # === BIGRAMS ===
    for i in tl.static_range(11):  # L - 1
        b1 = tl.load(data_ptr + pos + i)
        b2 = tl.load(data_ptr + pos + i + 1)
        idx = 1024 + ((i * 65537 + b1 * 257 + b2) % 8192)
        tl.atomic_add(out_base + idx, 1.0)
    
    # === TRIGRAMS ===
    for i in tl.static_range(10):  # L - 2
        b1 = tl.load(data_ptr + pos + i)
        b2 = tl.load(data_ptr + pos + i + 1)
        b3 = tl.load(data_ptr + pos + i + 2)
        idx = 1024 + 8192 + ((b1 * 65537 + b2 * 257 + b3) % 8192)
        tl.atomic_add(out_base + idx, 1.0)
    
    # === SKIPGRAMS ===
    for i in tl.static_range(10):  # L - 2
        b1 = tl.load(data_ptr + pos + i)
        b3 = tl.load(data_ptr + pos + i + 2)
        idx = 1024 + 8192 + 8192 + ((b1 * 257 + b3) % 4096)
        tl.atomic_add(out_base + idx, 1.0)


def extract_features_triton(data: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Extract features using Triton kernel."""
    B = positions.shape[0]
    features = torch.zeros((B, TOTAL_F), dtype=torch.float32, device=device)
    
    extract_features_kernel_L12[(B,)](
        data, positions, features,
        TOTAL_F,
    )
    return features


def context_hash(context: bytes) -> int:
    """Hash context for lookup table."""
    return int(hashlib.sha256(context).hexdigest()[:16], 16)


def build_lookup_table(data_bytes: bytes) -> dict:
    """Build lookup table: hash(context) -> next_byte."""
    print("Building lookup table...")
    start = time.perf_counter()
    
    lookup = {}
    n = len(data_bytes) - L - 1
    
    for i in range(n):
        ctx = data_bytes[i:i+L]
        target = data_bytes[i+L]
        h = context_hash(ctx)
        lookup[h] = target  # Last occurrence wins (fine for our purposes)
        
        if i % 1000000 == 0 and i > 0:
            print(f"  {i/n*100:.1f}% ({len(lookup):,} unique contexts)")
    
    elapsed = time.perf_counter() - start
    print(f"Lookup table: {len(lookup):,} entries in {elapsed:.1f}s")
    return lookup


def train_streaming_covariance(data_gpu: torch.Tensor, n_samples: int = 500000, batch_size: int = 50000):
    """Train W using streaming covariance (GPU-only)."""
    n = len(data_gpu) - L - 1
    step = max(1, n // n_samples)
    
    positions = torch.arange(0, n, step, dtype=torch.int64, device=device)[:n_samples]
    targets = data_gpu[positions + L]
    
    print(f"\nTraining with {len(positions):,} samples")
    
    XtX = torch.zeros((TOTAL_F, TOTAL_F), dtype=torch.float32, device=device)
    XtY = torch.zeros((TOTAL_F, VOCAB), dtype=torch.float32, device=device)
    
    start = time.perf_counter()
    
    for b in range(0, len(positions), batch_size):
        batch_pos = positions[b:b+batch_size]
        batch_tgt = targets[b:b+batch_size]
        
        X = extract_features_triton(data_gpu, batch_pos)
        
        Y = torch.zeros((len(batch_pos), VOCAB), dtype=torch.float32, device=device)
        Y.scatter_(1, batch_tgt.unsqueeze(1).long(), 1.0)
        
        XtX += X.T @ X
        XtY += X.T @ Y
        
        del X, Y
        torch.cuda.empty_cache()
        
        elapsed = time.perf_counter() - start
        samples_done = min(b + batch_size, len(positions))
        rate = samples_done / elapsed
        print(f"  {samples_done:,}/{len(positions):,} ({rate:.0f} samples/sec)")
    
    total_time = time.perf_counter() - start
    print(f"Covariance accumulation: {total_time:.1f}s ({len(positions)/total_time:.0f} samples/sec)")
    
    # Solve
    print("\nSolving least squares...")
    XtX += 1e-4 * torch.eye(TOTAL_F, device=device)
    W = torch.linalg.solve(XtX, XtY)
    
    del XtX, XtY
    torch.cuda.empty_cache()
    
    return W


def compress_W(W: torch.Tensor, rank: int = CHI) -> tuple:
    """Compress W to low rank via SVD."""
    print(f"\nCompressing W ({W.shape}) to rank {rank}...")
    
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vt_r = Vt[:rank, :]
    
    W_compressed = U_r @ torch.diag(S_r) @ Vt_r
    
    # Measure compression error
    error = torch.norm(W - W_compressed) / torch.norm(W)
    print(f"Compression error: {error.item()*100:.2f}%")
    print(f"Original params: {W.numel():,}")
    print(f"Compressed: ({TOTAL_F} × {rank}) + ({rank} × {VOCAB}) = {TOTAL_F*rank + rank*VOCAB:,} params")
    
    return U_r, S_r, Vt_r


def evaluate(data_gpu: torch.Tensor, W: torch.Tensor, lookup: dict = None, n_test: int = 30000):
    """Evaluate accuracy and perplexity."""
    n = len(data_gpu) - L - 1
    step = max(1, n // n_test)
    
    test_pos = torch.arange(0, n, step, dtype=torch.int64, device=device)[:n_test]
    test_targets = data_gpu[test_pos + L]
    
    X = extract_features_triton(data_gpu, test_pos)
    logits = X @ W
    probs = torch.softmax(logits, dim=1)
    
    preds = probs.argmax(dim=1)
    ls_correct = (preds == test_targets).sum().item()
    
    # Lookup accuracy (if available)
    lookup_hits = 0
    if lookup:
        data_cpu = data_gpu.cpu().numpy().astype('uint8').tobytes()
        for i, pos in enumerate(test_pos.cpu().numpy()):
            ctx = data_cpu[pos:pos+L]
            h = context_hash(ctx)
            if h in lookup:
                if lookup[h] == test_targets[i].item():
                    lookup_hits += 1
    
    # Perplexity
    p_actual = probs[torch.arange(n_test, device=device), test_targets.long()]
    log_prob = torch.log(p_actual.clamp(min=1e-10)).sum().item()
    ppl = math.exp(-log_prob / n_test)
    
    return {
        'ls_accuracy': 100 * ls_correct / n_test,
        'lookup_hits': 100 * lookup_hits / n_test if lookup else 0,
        'perplexity': ppl,
        'n_test': n_test
    }


def save_binary(lookup: dict, U_r: torch.Tensor, S_r: torch.Tensor, Vt_r: torch.Tensor, 
                filepath: str):
    """Export model to binary format for Rust prover."""
    print(f"\nSaving to {filepath}...")
    
    with open(filepath, 'wb') as f:
        # Header
        f.write(struct.pack('<4s', b'FLHB'))  # Magic: FluideLite HyBrid
        f.write(struct.pack('<I', 1))          # Version
        f.write(struct.pack('<I', L))          # Context length
        f.write(struct.pack('<I', VOCAB))      # Vocab size
        f.write(struct.pack('<I', CHI))        # Rank
        f.write(struct.pack('<I', TOTAL_F))    # Feature dim
        f.write(struct.pack('<Q', len(lookup)))  # Lookup table size
        
        # Lookup table
        for h, v in lookup.items():
            f.write(struct.pack('<QB', h, v))
        
        # Compressed W components
        U_np = U_r.cpu().numpy().astype('float32')
        S_np = S_r.cpu().numpy().astype('float32')
        Vt_np = Vt_r.cpu().numpy().astype('float32')
        
        f.write(U_np.tobytes())
        f.write(S_np.tobytes())
        f.write(Vt_np.tobytes())
    
    size = os.path.getsize(filepath)
    print(f"Saved: {size:,} bytes ({size/1e6:.2f} MB)")


def main():
    print("=" * 60)
    print("FluidEliteHybrid Training with TRITON")
    print("=" * 60)
    print(f"Config: L={L}, vocab={VOCAB}, chi={CHI}, features={TOTAL_F}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Triton: {triton.__version__}")
    
    # Load data
    data_dir = Path("/home/brad/TiganticLabz/Main_Projects/HyperTensor-VM-main/fluidelite/data")
    train_path = data_dir / "wikitext2_train.txt"
    test_path = data_dir / "wikitext2_test.txt"
    
    if not train_path.exists():
        print("\nDownloading WikiText-2...")
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        data_dir.mkdir(exist_ok=True)
        
        train_text = "\n".join(ds["train"]["text"])
        test_text = "\n".join(ds["test"]["text"])
        
        train_path.write_text(train_text)
        test_path.write_text(test_text)
    
    print("\nLoading data...")
    train_bytes = train_path.read_bytes()
    test_bytes = test_path.read_bytes()
    print(f"Train: {len(train_bytes):,} bytes")
    print(f"Test: {len(test_bytes):,} bytes")
    
    # GPU tensors
    print("\nTransferring to GPU...")
    train_gpu = torch.tensor(list(train_bytes), dtype=torch.int32, device=device)
    test_gpu = torch.tensor(list(test_bytes), dtype=torch.int32, device=device)
    print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # 1. Build lookup table (CPU)
    lookup = build_lookup_table(train_bytes)
    
    # 2. Train W with streaming covariance (GPU)
    W = train_streaming_covariance(train_gpu, n_samples=500000, batch_size=50000)
    
    # 3. Compress W to rank-24
    U_r, S_r, Vt_r = compress_W(W, rank=CHI)
    
    # Reconstruct compressed W for evaluation
    W_compressed = U_r @ torch.diag(S_r) @ Vt_r
    
    # 4. Evaluate on test set
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    print("\n--- Full W ---")
    metrics_full = evaluate(test_gpu, W, lookup)
    print(f"LS Accuracy: {metrics_full['ls_accuracy']:.1f}%")
    print(f"Lookup hits: {metrics_full['lookup_hits']:.1f}%")
    print(f"Perplexity: {metrics_full['perplexity']:.2f}")
    
    print("\n--- Compressed W (rank={}) ---".format(CHI))
    metrics_compressed = evaluate(test_gpu, W_compressed, lookup)
    print(f"LS Accuracy: {metrics_compressed['ls_accuracy']:.1f}%")
    print(f"Lookup hits: {metrics_compressed['lookup_hits']:.1f}%")
    print(f"Perplexity: {metrics_compressed['perplexity']:.2f}")
    
    # 5. Save binary
    output_path = data_dir / "fluidelite_hybrid.bin"
    save_binary(lookup, U_r, S_r, Vt_r, str(output_path))
    
    # Summary
    print("\n" + "=" * 60)
    print("FLUIDELITE HYBRID TRAINED")
    print("=" * 60)
    print(f"Lookup table: {len(lookup):,} entries")
    print(f"Compressed W: {TOTAL_F} × {CHI} + {CHI} × {VOCAB} = {TOTAL_F*CHI + CHI*VOCAB:,} params")
    print(f"Output: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
