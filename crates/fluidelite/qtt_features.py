"""
WikiText-103 with QTT-Compressed Covariance Matrix

Key insight: XtX = Σ x_i x_i^T is a sum of rank-1 outer products.
Instead of dense [21504 × 21504], we can:
1. Store XtX in QTT format with low rank
2. Solve (XtX + λI)W = XtY using iterative methods with QTT

Even more radical: Skip XtX entirely - use QTT to represent W directly!
"""
import torch
import triton
import triton.language as tl
import time
import math
import gc
from typing import List

device = torch.device('cuda')

# Feature dims (power of 2 for QTT!)
N_FEATURES = 16384  # 2^14 - perfect for QTT
CTX_LEN = 16
N_VOCAB = 256      # 2^8

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Feature dims: {N_FEATURES:,} (2^{int(math.log2(N_FEATURES))})")
print(f"Dense XtX would be: {N_FEATURES**2 * 4 / 1e9:.2f} GB")


# === Triton kernel for feature extraction (same as before) ===
@triton.jit
def extract_features_kernel(
    data_ptr, positions_ptr, features_ptr, stride_feat,
):
    pid = tl.program_id(0)
    pos = tl.load(positions_ptr + pid)
    out_base = features_ptr + pid * stride_feat
    
    # Unigrams (hash to first 1024)
    for i in tl.static_range(4):
        byte_val = tl.load(data_ptr + pos + 16 - 4 + i)
        idx = (i * 256 + byte_val) % 1024
        tl.atomic_add(out_base + idx, 1.0)
    
    # Bigrams (hash to 1024:5120)
    for i in tl.static_range(15):
        b1 = tl.load(data_ptr + pos + i)
        b2 = tl.load(data_ptr + pos + i + 1)
        idx = 1024 + ((i * 65537 + b1 * 257 + b2) % 4096)
        tl.atomic_add(out_base + idx, 1.0)
    
    # Trigrams (hash to 5120:9216)
    for i in tl.static_range(14):
        b1 = tl.load(data_ptr + pos + i)
        b2 = tl.load(data_ptr + pos + i + 1)
        b3 = tl.load(data_ptr + pos + i + 2)
        idx = 5120 + ((b1 * 65537 + b2 * 257 + b3) % 4096)
        tl.atomic_add(out_base + idx, 1.0)
    
    # 4-grams (hash to 9216:13312)
    for i in tl.static_range(13):
        b1 = tl.load(data_ptr + pos + i)
        b2 = tl.load(data_ptr + pos + i + 1)
        b3 = tl.load(data_ptr + pos + i + 2)
        b4 = tl.load(data_ptr + pos + i + 3)
        idx = 9216 + ((b1 * 16777259 + b2 * 65537 + b3 * 257 + b4) % 4096)
        tl.atomic_add(out_base + idx, 1.0)
    
    # Skipgrams (hash to 13312:16384)
    for i in tl.static_range(14):
        b1 = tl.load(data_ptr + pos + i)
        b3 = tl.load(data_ptr + pos + i + 2)
        idx = 13312 + ((b1 * 257 + b3) % 3072)
        tl.atomic_add(out_base + idx, 1.0)


def extract_features_triton(data: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    B = positions.shape[0]
    features = torch.zeros((B, N_FEATURES), dtype=torch.float32, device=device)
    extract_features_kernel[(B,)](data, positions, features, N_FEATURES)
    return features


# === QTT Core Operations ===
def qtt_cores_from_matrix(mat: torch.Tensor, max_rank: int = 64) -> List[torch.Tensor]:
    """Convert matrix to QTT format via TT-SVD.
    
    For [N, M] matrix where N=2^n, M=2^m:
    Returns cores of shape (r_left, 2, 2, r_right) for each qubit pair
    """
    N, M = mat.shape
    n_qubits = int(math.log2(N))
    m_qubits = int(math.log2(M))
    
    assert 2**n_qubits == N and 2**m_qubits == M, "Dimensions must be powers of 2"
    
    # Reshape to tensor: [2, 2, ..., 2] with n+m indices
    total_qubits = n_qubits + m_qubits
    shape = [2] * total_qubits
    tensor = mat.reshape(shape)
    
    # TT-SVD from left to right
    cores = []
    remaining = tensor.reshape(2, -1)
    r_left = 1
    
    for i in range(total_qubits - 1):
        m_dim, n_dim = remaining.shape
        r = min(max_rank, m_dim, n_dim)
        
        # Randomized SVD
        if min(m_dim, n_dim) > 4 * r:
            U, S, Vh = torch.svd_lowrank(remaining, q=r + 10, niter=2)
            U, S = U[:, :r], S[:r]
            Vh = Vh[:, :r].T
        else:
            U, S, Vh = torch.linalg.svd(remaining, full_matrices=False)
            U, S, Vh = U[:, :r], S[:r], Vh[:r, :]
        
        core = U.reshape(r_left, 2, r)
        cores.append(core)
        
        remaining = (torch.diag(S) @ Vh).reshape(r * 2, -1) if i < total_qubits - 2 else (torch.diag(S) @ Vh)
        r_left = r
    
    # Last core
    cores.append(remaining.reshape(r_left, 2, 1))
    
    return cores


def qtt_memory_size(cores: List[torch.Tensor]) -> int:
    """Compute memory size of QTT cores in bytes."""
    return sum(c.numel() * c.element_size() for c in cores)


# === Conjugate Gradient Solver (avoids materializing XtX) ===
def cg_solve_streaming(
    data: torch.Tensor,
    positions: torch.Tensor,
    targets: torch.Tensor,  # [N] indices
    lambda_reg: float = 1e-4,
    max_iter: int = 100,
    tol: float = 1e-6,
    batch_size: int = 50000,
) -> torch.Tensor:
    """
    Solve (X^T X + λI) W = X^T Y using Conjugate Gradient.
    
    Key: We never form X^T X explicitly!
    Instead, compute (X^T X) @ v by:
      1. v' = X @ v       (forward pass)
      2. result = X^T @ v' (backward pass)
    
    This streams through data, using O(batch × features) memory instead of O(features²).
    """
    N = len(positions)
    
    # Compute X^T Y by streaming (this we need once)
    print("  Computing X^T Y...")
    XtY = torch.zeros((N_FEATURES, N_VOCAB), dtype=torch.float32, device=device)
    
    for b in range(0, N, batch_size):
        batch_pos = positions[b:b+batch_size]
        batch_tgt = targets[b:b+batch_size]
        
        X = extract_features_triton(data, batch_pos)
        Y = torch.zeros((len(batch_pos), N_VOCAB), dtype=torch.float32, device=device)
        Y.scatter_(1, batch_tgt.unsqueeze(1).long(), 1.0)
        
        XtY += X.T @ Y
        del X, Y
    
    torch.cuda.empty_cache()
    
    def matvec_XtX(v: torch.Tensor) -> torch.Tensor:
        """Compute (X^T X + λI) @ v without forming X^T X."""
        result = lambda_reg * v.clone()
        
        for b in range(0, N, batch_size):
            batch_pos = positions[b:b+batch_size]
            X = extract_features_triton(data, batch_pos)
            
            # v' = X @ v
            Xv = X @ v  # [batch, vocab]
            # X^T @ v'
            result += X.T @ Xv
            
            del X, Xv
        
        return result
    
    # Conjugate Gradient
    print("  Running Conjugate Gradient...")
    W = torch.zeros((N_FEATURES, N_VOCAB), dtype=torch.float32, device=device)
    r = XtY.clone()  # residual = b - Ax, initially x=0 so r=b
    p = r.clone()    # search direction
    rs_old = (r * r).sum()
    
    for i in range(max_iter):
        Ap = matvec_XtX(p)
        alpha = rs_old / ((p * Ap).sum() + 1e-10)
        W += alpha * p
        r -= alpha * Ap
        rs_new = (r * r).sum()
        
        rel_res = (rs_new / (rs_old + 1e-10)).sqrt().item()
        if i % 10 == 0:
            print(f"    CG iter {i}: residual = {rel_res:.2e}")
        
        if rel_res < tol:
            print(f"    Converged at iter {i}")
            break
        
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    
    return W


def main():
    # Load data
    from datasets import load_dataset
    print("\nLoading WikiText-103...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    train_text = "\n".join(dataset["train"]["text"]).encode('utf-8')
    test_text = "\n".join(dataset["test"]["text"]).encode('utf-8')
    print(f"Train: {len(train_text)/1e6:.1f}MB, Test: {len(test_text)/1e6:.1f}MB")
    
    # To GPU
    print("Transferring to GPU...")
    train_data = torch.tensor(list(train_text), dtype=torch.int32, device=device)
    test_data = torch.tensor(list(test_text), dtype=torch.int32, device=device)
    print(f"VRAM after data: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Training samples
    n_train = len(train_data) - CTX_LEN - 1
    MAX_SAMPLES = 2_000_000  # 2M samples
    step = n_train // MAX_SAMPLES
    positions = torch.arange(0, n_train, step, dtype=torch.int64, device=device)[:MAX_SAMPLES]
    targets = train_data[positions + CTX_LEN]
    print(f"Training samples: {len(positions):,}")
    
    # === METHOD 1: Conjugate Gradient (no XtX materialization) ===
    print("\n=== Method 1: Conjugate Gradient (Matrix-Free) ===")
    print(f"Memory saving: {N_FEATURES**2 * 4 / 1e9:.2f} GB (no XtX)")
    
    start = time.perf_counter()
    W = cg_solve_streaming(train_data, positions, targets, max_iter=50, batch_size=50000)
    cg_time = time.perf_counter() - start
    
    print(f"CG solve time: {cg_time:.1f}s")
    print(f"VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    
    # === Evaluate ===
    print("\n=== Evaluating ===")
    n_test = len(test_data) - CTX_LEN - 1
    TEST_N = 30000
    test_step = n_test // TEST_N
    test_pos = torch.arange(0, n_test, test_step, dtype=torch.int64, device=device)[:TEST_N]
    test_actual = test_data[test_pos + CTX_LEN]
    
    X_test = extract_features_triton(test_data, test_pos)
    logits = X_test @ W
    probs = torch.softmax(logits, dim=1)
    
    preds = probs.argmax(dim=1)
    correct = (preds == test_actual).sum().item()
    
    p_actual = probs[torch.arange(TEST_N, device=device), test_actual.long()]
    log_prob = torch.log(p_actual.clamp(min=1e-10)).sum().item()
    
    acc = 100 * correct / TEST_N
    ppl = math.exp(-log_prob / TEST_N)
    
    # === QTT compress W ===
    print("\n=== QTT Compressing W ===")
    print(f"Dense W: {W.numel() * 4 / 1e6:.1f} MB")
    
    # W is [16384, 256] = [2^14, 2^8]
    W_qtt = qtt_cores_from_matrix(W, max_rank=64)
    qtt_size = qtt_memory_size(W_qtt)
    compression = W.numel() * 4 / qtt_size
    
    print(f"QTT W: {qtt_size / 1e6:.2f} MB ({len(W_qtt)} cores)")
    print(f"Compression: {compression:.1f}×")
    
    # Print core shapes
    print("Core shapes:", [tuple(c.shape) for c in W_qtt])
    
    print(f"\n{'='*55}")
    print(f"WIKITEXT-103 WITH MATRIX-FREE CG + QTT")
    print(f"{'='*55}")
    print(f"Training time: {cg_time:.1f}s (CG, 50 iters)")
    print(f"Accuracy: {acc:.1f}% ({correct/TEST_N*256:.0f}× random)")
    print(f"Perplexity: {ppl:.2f}")
    print(f"Dense params: {W.numel():,}")
    print(f"QTT params: {sum(c.numel() for c in W_qtt):,}")
    print(f"QTT compression: {compression:.1f}×")
    print(f"VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"XtX memory saved: {N_FEATURES**2 * 4 / 1e9:.2f} GB")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
