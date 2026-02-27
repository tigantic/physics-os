"""
Optimization 2: QTT Native Inference (Revised)

Key insight: Use DIRECTLY TRAINED QTT cores, not post-hoc compression.
Post-hoc compression loses 70%+ of information.
Direct training learns weights that live on the TT manifold.

Then: Use native TT contraction for inference WITHOUT materializing W.
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


# === QTT Operations ===
def qtt_to_dense(cores: List[torch.Tensor], n_row_qubits: int) -> torch.Tensor:
    """Convert QTT cores to dense matrix."""
    result = cores[0]
    for core in cores[1:]:
        r_left, size, _ = result.shape
        result = torch.einsum('ijk,klm->ijlm', result, core)
        result = result.reshape(r_left, size * 2, -1)
    result = result.squeeze(0).squeeze(-1)
    n_col_qubits = len(cores) - n_row_qubits
    return result.reshape(2**n_row_qubits, 2**n_col_qubits)


def qtt_matvec_native(cores: List[torch.Tensor], x: torch.Tensor, n_row_qubits: int) -> torch.Tensor:
    """
    Native QTT matvec: y = x @ W without materializing W.
    
    x: [batch, 2^n_row_qubits]
    returns: [batch, 2^n_col_qubits]
    """
    batch_size = x.shape[0]
    n_col_qubits = len(cores) - n_row_qubits
    
    # Phase 1: Contract row cores with input x
    x_qubits = x.reshape([batch_size] + [2] * n_row_qubits)
    state = x_qubits
    
    for k in range(n_row_qubits):
        core = cores[k]
        r_left, _, r_right = core.shape
        remaining = n_row_qubits - k - 1
        
        if k == 0:
            state_flat = state.reshape(batch_size, 2, -1)
            core_2d = core.squeeze(0)
            state = torch.einsum('bqs,qr->bsr', state_flat, core_2d)
        else:
            rest_size = max(1, 2 ** remaining)
            state = state.reshape(batch_size, 2, rest_size, r_left)
            state = torch.einsum('bqsr,rqo->bso', state, core)
    
    state = state.squeeze(1)
    
    # Phase 2: Expand column cores for output
    for k in range(n_row_qubits, len(cores)):
        core = cores[k]
        col_idx = k - n_row_qubits
        
        if col_idx == 0:
            state = torch.einsum('br,rjo->bjo', state, core)
        else:
            state = torch.einsum('bsr,rjo->bsjo', state, core)
            state = state.reshape(batch_size, -1, core.shape[2])
    
    return state.squeeze(-1)


def qtt_matvec_fused(cores: List[torch.Tensor], x: torch.Tensor, n_row_qubits: int) -> torch.Tensor:
    """
    Optimized QTT matvec - process samples in smaller chunks to reduce memory.
    """
    batch_size = x.shape[0]
    n_col_qubits = len(cores) - n_row_qubits
    n_out = 2 ** n_col_qubits
    
    # Contract all cores into a single tensor for row/col split
    # W[i,j] where i indexes rows (input), j indexes columns (output)
    # We'll compute this as a sequence of contractions
    
    # Method: materialize the TT-chain in blocks
    # For small dimensions, just do full contraction
    
    # Phase 1: Build row contractor (contracts with x[b, i])
    # Phase 2: Build column expander (outputs y[b, j])
    
    # Simpler: use torch.compile
    return qtt_matvec_native(cores, x, n_row_qubits)


def init_qtt_xavier(n_qubits: int, max_rank: int) -> List[torch.Tensor]:
    """Initialize QTT cores with Xavier initialization."""
    cores = []
    r_left = 1
    for i in range(n_qubits):
        r_right = min(max_rank, 2 ** min(i + 1, n_qubits - i - 1)) if i < n_qubits - 1 else 1
        std = math.sqrt(2.0 / (r_left + r_right))
        core = torch.randn(r_left, 2, r_right, device=device) * std
        core.requires_grad = True
        cores.append(core)
        r_left = r_right
    return cores


def main():
    from datasets import load_dataset
    
    print("="*60)
    print("OPTIMIZATION 2: QTT NATIVE INFERENCE (REVISED)")
    print("="*60)
    print("\nKey: Train QTT directly, then use native inference")
    
    # Load data
    print("\nLoading WikiText-103...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    train_text = "\n".join(dataset["train"]["text"]).encode('utf-8')
    test_text = "\n".join(dataset["test"]["text"]).encode('utf-8')
    
    train_data = torch.tensor(list(train_text), dtype=torch.int32, device=device)
    test_data = torch.tensor(list(test_text), dtype=torch.int32, device=device)
    
    # Training samples
    n_train = len(train_data) - CTX_LEN - 1
    MAX_SAMPLES = 200000
    step = n_train // MAX_SAMPLES
    positions = torch.arange(0, n_train, step, dtype=torch.int64, device=device)[:MAX_SAMPLES]
    targets = train_data[positions + CTX_LEN]
    
    # === PHASE 1: Train QTT directly ===
    print("\n--- Phase 1: Direct QTT Training ---")
    MAX_RANK = 64
    cores = init_qtt_xavier(TOTAL_QUBITS, MAX_RANK)
    n_params = sum(c.numel() for c in cores)
    print(f"QTT: {n_params:,} params, rank {MAX_RANK}")
    
    optimizer = torch.optim.Adam(cores, lr=0.03)
    
    BATCH = 4096
    N_EPOCHS = 10
    
    print(f"Training {N_EPOCHS} epochs...")
    start = time.perf_counter()
    
    for epoch in range(N_EPOCHS):
        epoch_loss = 0.0
        n_batches = 0
        perm = torch.randperm(len(positions), device=device)
        
        for b in range(0, len(positions), BATCH):
            idx = perm[b:b+BATCH]
            batch_pos = positions[idx]
            batch_tgt = targets[idx].long()
            
            X = extract_features(train_data, batch_pos)
            
            # Use dense forward for training (gradients flow to cores)
            W = qtt_to_dense(cores, N_QUBITS_FEAT)
            logits = X @ W
            
            loss = torch.nn.functional.cross_entropy(logits, batch_tgt)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        print(f"  Epoch {epoch+1}/{N_EPOCHS}: loss = {epoch_loss/n_batches:.4f}")
    
    train_time = time.perf_counter() - start
    print(f"Training time: {train_time:.1f}s")
    
    # Detach cores
    cores = [c.detach() for c in cores]
    
    # === PHASE 2: Compare Dense vs Native Inference ===
    print("\n--- Phase 2: Inference Comparison ---")
    
    # Test set
    n_test = len(test_data) - CTX_LEN - 1
    TEST_N = 20000
    test_pos = torch.arange(0, n_test, n_test//TEST_N, dtype=torch.int64, device=device)[:TEST_N]
    test_actual = test_data[test_pos + CTX_LEN]
    X_test = extract_features(test_data, test_pos)
    
    # Verify correctness first
    W_dense = qtt_to_dense(cores, N_QUBITS_FEAT)
    logits_dense = X_test @ W_dense
    logits_native = qtt_matvec_native(cores, X_test, N_QUBITS_FEAT)
    
    error = (logits_dense - logits_native).abs().max().item()
    print(f"Native vs Dense max error: {error:.2e}")
    
    # Benchmark dense inference
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(10):
        logits = X_test @ W_dense
        torch.cuda.synchronize()
    dense_time = (time.perf_counter() - start) / 10
    dense_mem = torch.cuda.max_memory_allocated()
    
    probs = torch.softmax(logits, dim=1)
    acc_dense = (probs.argmax(dim=1) == test_actual).float().mean().item()
    ppl_dense = math.exp(torch.nn.functional.cross_entropy(logits, test_actual.long()).item())
    
    print(f"\nDense inference:")
    print(f"  Time: {dense_time*1000:.1f} ms")
    print(f"  Memory: {dense_mem/1e9:.2f} GB")
    print(f"  Accuracy: {acc_dense*100:.1f}%")
    print(f"  Perplexity: {ppl_dense:.2f}")
    
    # Benchmark native inference with torch.compile
    del W_dense
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    # Compile the native function
    qtt_matvec_compiled = torch.compile(qtt_matvec_native, mode='reduce-overhead')
    
    # Warmup
    _ = qtt_matvec_compiled(cores, X_test[:100], N_QUBITS_FEAT)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(10):
        logits = qtt_matvec_compiled(cores, X_test, N_QUBITS_FEAT)
        torch.cuda.synchronize()
    native_time = (time.perf_counter() - start) / 10
    native_mem = torch.cuda.max_memory_allocated()
    
    probs = torch.softmax(logits, dim=1)
    acc_native = (probs.argmax(dim=1) == test_actual).float().mean().item()
    ppl_native = math.exp(torch.nn.functional.cross_entropy(logits, test_actual.long()).item())
    
    print(f"\nNative QTT inference:")
    print(f"  Time: {native_time*1000:.1f} ms")
    print(f"  Memory: {native_mem/1e9:.2f} GB")
    print(f"  Accuracy: {acc_native*100:.1f}%")
    print(f"  Perplexity: {ppl_native:.2f}")
    
    # Summary
    print(f"\n{'='*60}")
    print("NATIVE INFERENCE RESULTS")
    print(f"{'='*60}")
    print(f"Speed: {dense_time/native_time:.2f}x {'faster' if native_time < dense_time else 'slower'}")
    print(f"Memory: {dense_mem/native_mem:.2f}x {'less' if native_mem < dense_mem else 'more'}")
    print(f"Accuracy match: {abs(acc_dense - acc_native) < 0.001}")
    print(f"\nQTT params: {n_params:,}")
    print(f"Dense W size: {N_FEATURES * N_VOCAB * 4 / 1e6:.1f} MB")
    print(f"QTT cores size: {n_params * 4 / 1e6:.2f} MB")
    print(f"Compression: {N_FEATURES * N_VOCAB / n_params:.1f}x")


if __name__ == "__main__":
    main()
