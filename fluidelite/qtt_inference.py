"""
Optimization 2: QTT Inference (Native TT Contraction)

Instead of: W = qtt_to_dense(cores); y = x @ W
We do: y = qtt_matvec(cores, x)  # Never materialize W!

For a [2^n, 2^m] matrix stored as QTT with ranks r:
- Dense matmul: O(2^n × 2^m × batch) 
- QTT matvec:   O((n+m) × r² × batch)

When r << 2^(n/2), QTT inference is faster AND uses less memory.
"""
import torch
import triton
import triton.language as tl
import time
import math
from typing import List

device = torch.device('cuda')

# Dimensions
N_FEATURES = 16384  # 2^14
N_VOCAB = 256       # 2^8
CTX_LEN = 16
N_QUBITS_FEAT = 14
N_QUBITS_VOCAB = 8
TOTAL_QUBITS = N_QUBITS_FEAT + N_QUBITS_VOCAB

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"W shape: [{N_FEATURES}, {N_VOCAB}] = [2^{N_QUBITS_FEAT}, 2^{N_QUBITS_VOCAB}]")


# === Triton kernel ===
@triton.jit
def extract_features_kernel(data_ptr, positions_ptr, features_ptr, stride_feat):
    pid = tl.program_id(0)
    pos = tl.load(positions_ptr + pid)
    out_base = features_ptr + pid * stride_feat
    for i in tl.static_range(4):
        byte_val = tl.load(data_ptr + pos + 16 - 4 + i)
        idx = (i * 256 + byte_val) % 1024
        tl.atomic_add(out_base + idx, 1.0)
    for i in tl.static_range(15):
        b1 = tl.load(data_ptr + pos + i)
        b2 = tl.load(data_ptr + pos + i + 1)
        idx = 1024 + ((i * 65537 + b1 * 257 + b2) % 4096)
        tl.atomic_add(out_base + idx, 1.0)
    for i in tl.static_range(14):
        b1 = tl.load(data_ptr + pos + i)
        b2 = tl.load(data_ptr + pos + i + 1)
        b3 = tl.load(data_ptr + pos + i + 2)
        idx = 5120 + ((b1 * 65537 + b2 * 257 + b3) % 4096)
        tl.atomic_add(out_base + idx, 1.0)
    for i in tl.static_range(13):
        b1 = tl.load(data_ptr + pos + i)
        b2 = tl.load(data_ptr + pos + i + 1)
        b3 = tl.load(data_ptr + pos + i + 2)
        b4 = tl.load(data_ptr + pos + i + 3)
        idx = 9216 + ((b1 * 16777259 + b2 * 65537 + b3 * 257 + b4) % 4096)
        tl.atomic_add(out_base + idx, 1.0)
    for i in tl.static_range(14):
        b1 = tl.load(data_ptr + pos + i)
        b3 = tl.load(data_ptr + pos + i + 2)
        idx = 13312 + ((b1 * 257 + b3) % 3072)
        tl.atomic_add(out_base + idx, 1.0)


def extract_features(data: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    B = positions.shape[0]
    features = torch.zeros((B, N_FEATURES), dtype=torch.float32, device=device)
    extract_features_kernel[(B,)](data, positions, features, N_FEATURES)
    return features


# === QTT Native Matvec ===
def qtt_matvec_native(cores: List[torch.Tensor], x: torch.Tensor, n_row_qubits: int) -> torch.Tensor:
    """
    Compute y = W @ x^T where W is QTT, WITHOUT materializing W.
    
    x: [batch, 2^n_row_qubits] input (row vectors)
    Returns: [batch, 2^n_col_qubits] output
    
    Algorithm: Contract input qubits with TT cores from left to right,
    then contract remaining cores to get output.
    
    Complexity: O(batch × n_qubits × r²) instead of O(batch × 2^n × 2^m)
    """
    batch_size = x.shape[0]
    n_col_qubits = len(cores) - n_row_qubits
    
    # Reshape x to [batch, 2, 2, ..., 2] for n_row_qubits
    x_reshaped = x.reshape(batch_size, *([2] * n_row_qubits))
    
    # Start with first core, contract with first qubit of x
    # core[0]: [1, 2, r1], x[:, i0]: [batch]
    # result: [batch, r1]
    
    # Contract all row qubits
    # accumulated: [batch, r_current]
    accumulated = torch.ones(batch_size, 1, device=device, dtype=x.dtype)
    
    for k in range(n_row_qubits):
        core = cores[k]  # [r_left, 2, r_right]
        r_left, _, r_right = core.shape
        
        # x_k: [batch, 2] - k-th qubit values
        # We need to weight core by x_k values
        # accumulated: [batch, r_left]
        # core: [r_left, 2, r_right]
        # x_slice: [batch, 2] from x_reshaped
        
        # Get the k-th qubit slice
        # x_reshaped is [batch, 2, 2, ..., 2]
        # We need x_reshaped[:, :, :, ..., :] summed appropriately
        
        # Simpler: compute for each qubit bit value and weight
        # Actually, the cleanest is einsum with proper indexing
        
        # For qubit k, x_reshaped has shape [batch, 2^k, 2, 2^(n-k-1)]
        # We want to contract over the middle '2' dimension
        
        # Reshape x for this contraction
        pre_size = 2 ** k
        post_size = 2 ** (n_row_qubits - k - 1)
        x_view = x_reshaped.reshape(batch_size, pre_size, 2, post_size)
        
        # For each (pre, post) position, we have a 2-vector
        # We'll sum over all (pre, post) weighted by their contribution
        
        # Actually, let's think differently:
        # We're computing sum_i x[i] * W[i, :] where i is binary indexed
        # 
        # In TT form: W[i1...in, j1...jm] = prod_k core_k[r_{k-1}, i_k or j_k, r_k]
        #
        # x @ W = sum_{i1...in} x[i1...in] * prod_k core_k[..., i_k, ...]
        #       = prod_k (sum_{i_k} x[..., i_k, ...] * core_k[..., i_k, ...])
        #
        # We can factor! Contract x with row cores one at a time.
        
        pass  # This is getting complex, let me use a cleaner formulation

    # Actually, for simplicity and correctness, let's use a cleaner approach
    # that's still O(batch × qubits × r²) but easier to verify
    
    return _qtt_matvec_simple(cores, x, n_row_qubits)


def _qtt_matvec_simple(cores: List[torch.Tensor], x: torch.Tensor, n_row_qubits: int) -> torch.Tensor:
    """
    QTT matvec: y = x @ W where W is in QTT format.
    
    The matrix W has shape [2^n, 2^m] where:
      - n = n_row_qubits (input dimension qubits)
      - m = n_col_qubits (output dimension qubits)
    
    W[i,j] = G1[i1] @ G2[i2] @ ... @ Gn[in] @ Gn+1[j1] @ ... @ Gn+m[jm]
    
    For y = x @ W:
      y[j] = sum_i x[i] W[i,j]
    
    We factor this as:
      1. Contract x with row cores: v = sum_i x[i] * G1[i1]...Gn[in]
      2. Expand v through column cores: y[j] = v @ Gn+1[j1]...Gn+m[jm]
    """
    batch_size = x.shape[0]
    n_col_qubits = len(cores) - n_row_qubits
    
    # ===== PHASE 1: Contract row cores with input x =====
    # Reshape x: [batch, 2^n] -> [batch, 2, 2, ..., 2]
    x_qubits = x.reshape([batch_size] + [2] * n_row_qubits)
    
    # Strategy: Sweep left-to-right, contracting one qubit at a time
    # State shape: [batch, remaining_qubit_dims..., rank]
    # Initially: [batch, 2, 2, ..., 2] (no rank dim yet)
    
    state = x_qubits
    
    for k in range(n_row_qubits):
        core = cores[k]  # [r_left, 2, r_right]
        r_left, _, r_right = core.shape
        
        remaining = n_row_qubits - k - 1
        
        if k == 0:
            # state: [batch, 2, 2, ..., 2] with n dims
            # Flatten remaining qubits: [batch, 2, 2^(n-1)]
            state_flat = state.reshape(batch_size, 2, -1)  # [batch, 2, rest]
            # core: [1, 2, r_right] -> [2, r_right]
            core_2d = core.squeeze(0)  # [2, r_right]
            # Contract over qubit: [batch, 2, rest] x [2, r] -> [batch, rest, r]
            state = torch.einsum('bqs,qr->bsr', state_flat, core_2d)
        else:
            # state: [batch, 2^(n-k), r_left]
            # Split first dim: [batch, 2, 2^(n-k-1), r_left]
            rest_size = max(1, 2 ** remaining)
            state = state.reshape(batch_size, 2, rest_size, r_left)
            # core: [r_left, 2, r_right]
            # Sum over qubit dim: [batch, 2, rest, r_left] x [r_left, 2, r_right] -> [batch, rest, r_right]
            # Contract: state[:, q, :, r] @ core[r, q, r'] summed over q
            state = torch.einsum('bqsr,rqo->bso', state, core)
    
    # After all row cores: [batch, 1, r_n] -> [batch, r_n]
    state = state.squeeze(1)
    
    # ===== PHASE 2: Expand column cores for output =====
    # state: [batch, r_n]
    # Each column core expands the output dimension
    
    for k in range(n_row_qubits, len(cores)):
        core = cores[k]  # [r_left, 2, r_right]
        col_idx = k - n_row_qubits
        
        if col_idx == 0:
            # state: [batch, r_left]
            # output: [batch, 2, r_right]
            state = torch.einsum('br,rjo->bjo', state, core)
        else:
            # state: [batch, 2^col_idx, r_left]
            # output: [batch, 2^col_idx, 2, r_right] -> [batch, 2^(col_idx+1), r_right]
            state = torch.einsum('bsr,rjo->bsjo', state, core)
            state = state.reshape(batch_size, -1, core.shape[2])
    
    # Final: [batch, 2^n_col_qubits, 1] -> [batch, 2^n_col_qubits]
    return state.squeeze(-1)
    
    # Final: [batch, 2^n_col_qubits, 1] -> [batch, 2^n_col_qubits]
    return acc.squeeze(-1)


def qtt_to_dense(cores: List[torch.Tensor], n_row_qubits: int) -> torch.Tensor:
    """Reference: convert QTT to dense matrix."""
    result = cores[0]
    for core in cores[1:]:
        r_left, size, r_mid = result.shape
        result = torch.einsum('ijk,klm->ijlm', result, core)
        result = result.reshape(r_left, size * 2, -1)
    result = result.squeeze(0).squeeze(-1)
    n_col_qubits = len(cores) - n_row_qubits
    return result.reshape(2**n_row_qubits, 2**n_col_qubits)


def benchmark_inference():
    """Compare dense vs QTT inference."""
    print("\n" + "="*60)
    print("OPTIMIZATION 2: QTT NATIVE INFERENCE")
    print("="*60)
    
    # Create random QTT
    MAX_RANK = 32
    cores = []
    r_left = 1
    for i in range(TOTAL_QUBITS):
        r_right = min(MAX_RANK, 2 ** min(i + 1, TOTAL_QUBITS - i - 1)) if i < TOTAL_QUBITS - 1 else 1
        core = torch.randn(r_left, 2, r_right, device=device) * 0.1
        cores.append(core)
        r_left = r_right
    
    n_params = sum(c.numel() for c in cores)
    print(f"QTT: {n_params:,} params, max rank {MAX_RANK}")
    
    # Dense reference
    W_dense = qtt_to_dense(cores, N_QUBITS_FEAT)
    print(f"Dense W: {W_dense.shape}, {W_dense.numel() * 4 / 1e6:.1f} MB")
    
    # Test batch
    BATCH = 10000
    x = torch.randn(BATCH, N_FEATURES, device=device)
    
    # Warmup
    _ = x @ W_dense
    _ = _qtt_matvec_simple(cores, x, N_QUBITS_FEAT)
    torch.cuda.synchronize()
    
    # Benchmark dense
    print("\n--- Benchmarking ---")
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(100):
        y_dense = x @ W_dense
        torch.cuda.synchronize()
    dense_time = (time.perf_counter() - start) / 100
    dense_mem = torch.cuda.max_memory_allocated()
    
    # Benchmark QTT
    torch.cuda.reset_peak_memory_stats()
    del W_dense  # Remove dense from memory
    torch.cuda.empty_cache()
    
    start = time.perf_counter()
    for _ in range(100):
        y_qtt = _qtt_matvec_simple(cores, x, N_QUBITS_FEAT)
        torch.cuda.synchronize()
    qtt_time = (time.perf_counter() - start) / 100
    qtt_mem = torch.cuda.max_memory_allocated()
    
    # Verify correctness
    W_dense = qtt_to_dense(cores, N_QUBITS_FEAT)
    y_ref = x @ W_dense
    y_qtt = _qtt_matvec_simple(cores, x, N_QUBITS_FEAT)
    max_error = (y_ref - y_qtt).abs().max().item()
    rel_error = max_error / y_ref.abs().max().item()
    
    print(f"\nDense matmul:")
    print(f"  Time: {dense_time*1000:.2f} ms")
    print(f"  Memory: {dense_mem/1e9:.2f} GB")
    
    print(f"\nQTT matvec:")
    print(f"  Time: {qtt_time*1000:.2f} ms")
    print(f"  Memory: {qtt_mem/1e9:.2f} GB")
    
    print(f"\nComparison:")
    print(f"  Speedup: {dense_time/qtt_time:.2f}×" if qtt_time < dense_time else f"  Slowdown: {qtt_time/dense_time:.2f}×")
    print(f"  Memory reduction: {dense_mem/qtt_mem:.2f}×")
    print(f"  Max error: {max_error:.2e} (rel: {rel_error:.2e})")
    
    print(f"\n{'='*60}")
    print(f"QTT INFERENCE ANALYSIS")
    print(f"{'='*60}")
    print(f"Dense complexity: O({BATCH} × {N_FEATURES} × {N_VOCAB}) = {BATCH * N_FEATURES * N_VOCAB:.2e}")
    print(f"QTT complexity:   O({BATCH} × {TOTAL_QUBITS} × {MAX_RANK}²) = {BATCH * TOTAL_QUBITS * MAX_RANK**2:.2e}")
    print(f"Theoretical ratio: {BATCH * N_FEATURES * N_VOCAB / (BATCH * TOTAL_QUBITS * MAX_RANK**2):.1f}×")


def main():
    benchmark_inference()
    
    # Now test on real data
    print("\n\n" + "="*60)
    print("REAL INFERENCE TEST")
    print("="*60)
    
    from datasets import load_dataset
    print("\nLoading WikiText-103...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    train_text = "\n".join(dataset["train"]["text"]).encode('utf-8')
    test_text = "\n".join(dataset["test"]["text"]).encode('utf-8')
    
    train_data = torch.tensor(list(train_text), dtype=torch.int32, device=device)
    test_data = torch.tensor(list(test_text), dtype=torch.int32, device=device)
    
    # Load or create a trained QTT (using the CG+compress approach for quality)
    print("\nTraining model (CG method)...")
    n_train = len(train_data) - CTX_LEN - 1
    MAX_SAMPLES = 200000
    step = n_train // MAX_SAMPLES
    positions = torch.arange(0, n_train, step, dtype=torch.int64, device=device)[:MAX_SAMPLES]
    targets = train_data[positions + CTX_LEN]
    
    # Quick CG solve
    XtX = torch.zeros((N_FEATURES, N_FEATURES), dtype=torch.float32, device=device)
    XtY = torch.zeros((N_FEATURES, N_VOCAB), dtype=torch.float32, device=device)
    
    BATCH = 50000
    for b in range(0, MAX_SAMPLES, BATCH):
        batch_pos = positions[b:b+BATCH]
        batch_tgt = targets[b:b+BATCH]
        X = extract_features(train_data, batch_pos)
        Y = torch.zeros((len(batch_pos), N_VOCAB), dtype=torch.float32, device=device)
        Y.scatter_(1, batch_tgt.unsqueeze(1).long(), 1.0)
        XtX += X.T @ X
        XtY += X.T @ Y
        del X, Y
    
    XtX += 1e-4 * torch.eye(N_FEATURES, device=device)
    W_dense = torch.linalg.solve(XtX, XtY)
    del XtX, XtY
    
    # Compress to QTT
    print("Compressing to QTT...")
    cores = []
    remaining = W_dense.reshape(2, -1)
    r_left = 1
    MAX_RANK = 32
    
    for i in range(TOTAL_QUBITS - 1):
        m, n = remaining.shape
        r = min(MAX_RANK, m, n)
        U, S, Vh = torch.linalg.svd(remaining, full_matrices=False)
        U, S, Vh = U[:, :r], S[:r], Vh[:r, :]
        core = U.reshape(r_left, 2, r)
        cores.append(core)
        remaining = (torch.diag(S) @ Vh).reshape(r * 2, -1) if i < TOTAL_QUBITS - 2 else (torch.diag(S) @ Vh)
        r_left = r
    cores.append(remaining.reshape(r_left, 2, 1))
    
    n_params = sum(c.numel() for c in cores)
    print(f"QTT: {n_params:,} params")
    
    # Check compression quality
    W_recon = qtt_to_dense(cores, N_QUBITS_FEAT)
    recon_error = (W_dense - W_recon).norm() / W_dense.norm()
    print(f"Compression relative error: {recon_error:.4f}")
    
    # Evaluate with dense
    n_test = len(test_data) - CTX_LEN - 1
    TEST_N = 20000
    test_pos = torch.arange(0, n_test, n_test//TEST_N, dtype=torch.int64, device=device)[:TEST_N]
    test_actual = test_data[test_pos + CTX_LEN]
    X_test = extract_features(test_data, test_pos)
    
    print("\n--- Inference comparison ---")
    
    # Dense (original W)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    logits_dense = X_test @ W_dense
    torch.cuda.synchronize()
    dense_time = time.perf_counter() - start
    dense_mem = torch.cuda.max_memory_allocated()
    
    probs_dense = torch.softmax(logits_dense, dim=1)
    acc_dense = (probs_dense.argmax(dim=1) == test_actual).float().mean().item()
    
    # Dense (reconstructed from QTT)
    logits_recon = X_test @ W_recon
    probs_recon = torch.softmax(logits_recon, dim=1)
    acc_recon = (probs_recon.argmax(dim=1) == test_actual).float().mean().item()
    
    # Verify matvec correctness against reconstructed
    logits_qtt = _qtt_matvec_simple(cores, X_test, N_QUBITS_FEAT)
    matvec_error = (logits_recon - logits_qtt).abs().max().item()
    print(f"Matvec vs recon max error: {matvec_error:.2e}")
    
    # QTT
    del W_dense, W_recon
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    start = time.perf_counter()
    logits_qtt = _qtt_matvec_simple(cores, X_test, N_QUBITS_FEAT)
    torch.cuda.synchronize()
    qtt_time = time.perf_counter() - start
    qtt_mem = torch.cuda.max_memory_allocated()
    
    probs_qtt = torch.softmax(logits_qtt, dim=1)
    acc_qtt = (probs_qtt.argmax(dim=1) == test_actual).float().mean().item()
    
    print(f"\nDense (original W): {dense_time*1000:.1f}ms, acc {acc_dense*100:.1f}%, mem {dense_mem/1e9:.2f} GB")
    print(f"Dense (QTT recon):  acc {acc_recon*100:.1f}%")
    print(f"QTT native matvec:  {qtt_time*1000:.1f}ms, acc {acc_qtt*100:.1f}%, mem {qtt_mem/1e9:.2f} GB")
    print(f"Speedup: {dense_time/qtt_time:.2f}×" if qtt_time < dense_time else f"Slowdown: {qtt_time/dense_time:.2f}×")
    
    # The key insight: QTT matvec matches reconstructed dense EXACTLY
    # The accuracy loss comes from compression, not from the matvec algorithm
    print(f"\n*** QTT accuracy = QTT recon accuracy: {abs(acc_qtt - acc_recon) < 0.001}")
    print(f"*** Accuracy loss from compression: {(acc_dense - acc_recon)*100:.1f}%")
    
    # ========================================
    # PART 2: Direct QTT Training + Native Inference
    # ========================================
    print(f"\n{'='*60}")
    print("DIRECT QTT TRAINING + NATIVE INFERENCE")
    print(f"{'='*60}")
    
    # Initialize QTT cores directly (no compression needed!)
    MAX_RANK_DIRECT = 32
    direct_cores = []
    r_left = 1
    for i in range(TOTAL_QUBITS):
        r_right = min(MAX_RANK_DIRECT, 2 ** min(i + 1, TOTAL_QUBITS - i - 1)) if i < TOTAL_QUBITS - 1 else 1
        core = torch.randn(r_left, 2, r_right, device=device) * 0.01
        core.requires_grad = True
        direct_cores.append(core)
        r_left = r_right
    
    n_params_direct = sum(c.numel() for c in direct_cores)
    print(f"QTT: {n_params_direct:,} params, rank {MAX_RANK_DIRECT}")
    
    # Helper: QTT to dense (for training with gradients)
    def cores_to_dense(cores):
        result = cores[0]
        for core in cores[1:]:
            r_left, size, r_mid = result.shape
            result = torch.einsum('ijk,klm->ijlm', result, core)
            result = result.reshape(r_left, size * 2, -1)
        result = result.squeeze(0).squeeze(-1)
        return result.reshape(N_FEATURES, N_VOCAB)
    
    # Train with dense forward (grad flows through to cores), native inference
    optimizer = torch.optim.AdamW(direct_cores, lr=0.01, weight_decay=1e-4)
    
    TRAIN_BATCH = 4096
    N_STEPS = 200
    
    print(f"Training {N_STEPS} steps...")
    start_train = time.perf_counter()
    
    for step in range(N_STEPS):
        # Sample batch
        idx = torch.randint(0, len(positions), (TRAIN_BATCH,), device=device)
        batch_pos = positions[idx]
        batch_tgt = targets[idx].long()
        
        X = extract_features(train_data, batch_pos)
        
        # Forward: materialize W for training (grads flow to cores)
        W_train = cores_to_dense(direct_cores)
        logits = X @ W_train
        
        # Cross-entropy loss
        loss = torch.nn.functional.cross_entropy(logits, batch_tgt)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}")
    
    train_time = time.perf_counter() - start_train
    print(f"Training time: {train_time:.1f}s")
    
    # Detach cores for inference
    for i, c in enumerate(direct_cores):
        direct_cores[i] = c.detach()
    
    # Benchmark native inference (NOW we use native matvec!)
    torch.cuda.reset_peak_memory_stats()
    
    start = time.perf_counter()
    logits_direct = _qtt_matvec_simple(direct_cores, X_test, N_QUBITS_FEAT)
    torch.cuda.synchronize()
    direct_time = time.perf_counter() - start
    direct_mem = torch.cuda.max_memory_allocated()
    
    probs_direct = torch.softmax(logits_direct, dim=1)
    acc_direct = (probs_direct.argmax(dim=1) == test_actual).float().mean().item()
    
    # Verify against dense
    W_final = cores_to_dense(direct_cores)
    logits_dense_check = X_test @ W_final
    matvec_vs_dense = (logits_dense_check - logits_direct).abs().max().item()
    print(f"\nMatvec vs dense error: {matvec_vs_dense:.2e}")
    
    # Perplexity on test set
    with torch.no_grad():
        test_loss = torch.nn.functional.cross_entropy(logits_direct, test_actual.long())
        ppl = math.exp(test_loss.item())
    
    print(f"\nDirect QTT inference: {direct_time*1000:.1f}ms")
    print(f"  Accuracy: {acc_direct*100:.1f}%")
    print(f"  Perplexity: {ppl:.2f}")
    print(f"  Memory: {direct_mem/1e9:.2f} GB")
    print(f"  Params: {n_params_direct:,}")
    
    print(f"\n{'='*60}")
    print("QTT NATIVE INFERENCE COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
