"""
WikiText-103 with TRITON KERNELS - eliminate Python loops entirely.
"""
import torch
import triton
import triton.language as tl
import time
import math

device = torch.device('cuda')

# Feature dims
N_UNI = 1024
N_BI = 8192
N_TRI = 8192
N_SKIP = 4096
TOTAL_F = N_UNI + N_BI + N_TRI + N_SKIP
CTX_LEN = 16


@triton.jit
def extract_features_kernel(
    data_ptr,        # [N] bytes
    positions_ptr,   # [B] start positions
    features_ptr,    # [B, TOTAL_F] output
    stride_feat,     # stride for features (TOTAL_F)
):
    """Each program handles one sample."""
    pid = tl.program_id(0)
    
    # Load start position for this sample
    pos = tl.load(positions_ptr + pid)
    
    # Output pointer for this sample
    out_base = features_ptr + pid * stride_feat
    
    # === UNIGRAMS (last 4 bytes) ===
    for i in tl.static_range(4):
        byte_val = tl.load(data_ptr + pos + 16 - 4 + i)  # CTX_LEN=16
        idx = (i * 256 + byte_val) % 1024  # N_UNI=1024
        tl.atomic_add(out_base + idx, 1.0)
    
    # === BIGRAMS ===
    for i in tl.static_range(15):  # CTX_LEN - 1
        b1 = tl.load(data_ptr + pos + i)
        b2 = tl.load(data_ptr + pos + i + 1)
        idx = 1024 + ((i * 65537 + b1 * 257 + b2) % 8192)  # N_UNI + hash % N_BI
        tl.atomic_add(out_base + idx, 1.0)
    
    # === TRIGRAMS ===
    for i in tl.static_range(14):  # CTX_LEN - 2
        b1 = tl.load(data_ptr + pos + i)
        b2 = tl.load(data_ptr + pos + i + 1)
        b3 = tl.load(data_ptr + pos + i + 2)
        idx = 1024 + 8192 + ((b1 * 65537 + b2 * 257 + b3) % 8192)  # N_UNI + N_BI + hash % N_TRI
        tl.atomic_add(out_base + idx, 1.0)
    
    # === SKIPGRAMS ===
    for i in tl.static_range(14):  # CTX_LEN - 2
        b1 = tl.load(data_ptr + pos + i)
        b3 = tl.load(data_ptr + pos + i + 2)
        idx = 1024 + 8192 + 8192 + ((b1 * 257 + b3) % 4096)  # N_UNI + N_BI + N_TRI + hash % N_SKIP
        tl.atomic_add(out_base + idx, 1.0)


def extract_features_triton(data: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Triton-accelerated feature extraction."""
    B = positions.shape[0]
    features = torch.zeros((B, TOTAL_F), dtype=torch.float32, device=device)
    
    # Launch kernel - one program per sample
    extract_features_kernel[(B,)](
        data, positions, features,
        TOTAL_F,  # stride
    )
    return features


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Triton: {triton.__version__}")
    
    # Load data
    from datasets import load_dataset
    print("\nLoading WikiText-103...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    train_text = "\n".join(dataset["train"]["text"]).encode('utf-8')
    test_text = "\n".join(dataset["test"]["text"]).encode('utf-8')
    print(f"Train: {len(train_text)/1e6:.1f}MB, Test: {len(test_text)/1e6:.1f}MB")
    
    # Convert to GPU tensor ONCE
    print("Transferring to GPU...")
    train_data = torch.tensor(list(train_text), dtype=torch.int32, device=device)
    test_data = torch.tensor(list(test_text), dtype=torch.int32, device=device)
    print(f"VRAM after data: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # === Benchmark ===
    print("\n=== Benchmarking Triton ===")
    n_train = len(train_data) - CTX_LEN - 1
    test_positions = torch.arange(0, 100000, 10, dtype=torch.int64, device=device)
    
    # Warmup
    _ = extract_features_triton(train_data, test_positions[:100])
    torch.cuda.synchronize()
    
    # Time Triton
    start = time.perf_counter()
    for _ in range(10):
        X = extract_features_triton(train_data, test_positions)
        torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / 10
    print(f"Triton: {len(test_positions)} samples in {triton_time*1000:.1f}ms ({len(test_positions)/triton_time:.0f} samples/sec)")
    
    # === Full training ===
    print("\n=== Training with Triton ===")
    MAX_SAMPLES = 500000
    step = n_train // MAX_SAMPLES
    positions_all = torch.arange(0, n_train, step, dtype=torch.int64, device=device)[:MAX_SAMPLES]
    targets_all = train_data[positions_all + CTX_LEN]
    
    print(f"Training samples: {len(positions_all):,}")
    
    # Streaming covariance
    XtX = torch.zeros((TOTAL_F, TOTAL_F), dtype=torch.float32, device=device)
    XtY = torch.zeros((TOTAL_F, 256), dtype=torch.float32, device=device)
    
    BATCH = 50000
    start_time = time.perf_counter()
    
    for b in range(0, len(positions_all), BATCH):
        batch_pos = positions_all[b:b+BATCH]
        batch_tgt = targets_all[b:b+BATCH]
        
        # TRITON feature extraction!
        X = extract_features_triton(train_data, batch_pos)
        
        # One-hot targets
        Y = torch.zeros((len(batch_pos), 256), dtype=torch.float32, device=device)
        Y.scatter_(1, batch_tgt.unsqueeze(1).long(), 1.0)
        
        # Accumulate
        XtX += X.T @ X
        XtY += X.T @ Y
        
        del X, Y
        torch.cuda.empty_cache()
        
        if b % 100000 == 0:
            elapsed = time.perf_counter() - start_time
            print(f"  Batch {b//BATCH + 1}/{len(positions_all)//BATCH}: {elapsed:.1f}s, VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    train_time = time.perf_counter() - start_time
    print(f"Training complete in {train_time:.1f}s")
    
    # Solve
    print("\n=== Solving ===")
    XtX += 1e-4 * torch.eye(TOTAL_F, device=device)
    W = torch.linalg.solve(XtX, XtY)
    del XtX, XtY
    torch.cuda.empty_cache()
    print(f"W: {W.shape}, Params: {W.numel():,}")
    
    # === Evaluate ===
    print("\n=== Evaluating ===")
    n_test = len(test_data) - CTX_LEN - 1
    TEST_N = 30000
    test_step = n_test // TEST_N
    test_pos = torch.arange(0, n_test, test_step, dtype=torch.int64, device=device)[:TEST_N]
    test_actual = test_data[test_pos + CTX_LEN]
    
    # Triton feature extraction for test
    X_test = extract_features_triton(test_data, test_pos)
    logits = X_test @ W
    probs = torch.softmax(logits, dim=1)
    
    preds = probs.argmax(dim=1)
    correct = (preds == test_actual).sum().item()
    
    p_actual = probs[torch.arange(TEST_N, device=device), test_actual.long()]
    log_prob = torch.log(p_actual.clamp(min=1e-10)).sum().item()
    
    acc = 100 * correct / TEST_N
    ppl = math.exp(-log_prob / TEST_N)
    
    print(f"\n{'='*55}")
    print(f"WIKITEXT-103 WITH TRITON KERNELS")
    print(f"{'='*55}")
    print(f"Training time: {train_time:.1f}s")
    print(f"Throughput: {len(positions_all)/train_time:.0f} samples/sec")
    print(f"Accuracy: {acc:.1f}% ({correct/TEST_N*256:.0f}× random)")
    print(f"Perplexity: {ppl:.2f}")
    print(f"Parameters: {W.numel():,}")
    print(f"VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"{'='*55}")
    print("ZERO GRADIENTS. TRITON KERNELS. PURE GPU.")


if __name__ == "__main__":
    main()
