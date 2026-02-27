"""
Optimization 3: Scale to Larger Feature Spaces

With QTT, we can scale to MUCH larger feature dimensions:
- Dense [2^14, 256] = 16.8 MB
- Dense [2^16, 256] = 67.1 MB  
- Dense [2^18, 256] = 268 MB
- Dense [2^20, 256] = 1.07 GB

But QTT params grow only linearly with qubits:
- QTT 22 qubits, rank 64 = 92K params
- QTT 24 qubits, rank 64 = 108K params (+16K)
- QTT 26 qubits, rank 64 = 124K params (+16K)
- QTT 28 qubits, rank 64 = 140K params (+16K)

So we can have 1M+ features with <200K params!
"""
import torch
import triton
import triton.language as tl
import time
import math
from typing import List

device = torch.device('cuda')

print(f"GPU: {torch.cuda.get_device_name(0)}")

# Base feature sizes (powers of 2 for QTT)
# We'll scale by increasing hash bucket sizes

CTX_LEN = 16
N_VOCAB = 256
N_QUBITS_VOCAB = 8


# === Feature extraction kernels for different scales ===
@triton.jit
def extract_features_16k_kernel(data_ptr, positions_ptr, features_ptr, stride_feat):
    """16,384 features = 2^14 (baseline)"""
    pid = tl.program_id(0)
    pos = tl.load(positions_ptr + pid)
    out_base = features_ptr + pid * stride_feat
    
    # 1024 unigrams + 4096 bigrams + 4096 trigrams + 4096 4grams + 3072 skipgrams = 16384
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


@triton.jit
def extract_features_64k_kernel(data_ptr, positions_ptr, features_ptr, stride_feat):
    """65,536 features = 2^16 (4× more hash buckets)"""
    pid = tl.program_id(0)
    pos = tl.load(positions_ptr + pid)
    out_base = features_ptr + pid * stride_feat
    
    # 4096 unigrams + 16384 bigrams + 16384 trigrams + 16384 4grams + 12288 skipgrams = 65536
    for i in tl.static_range(4):
        byte_val = tl.load(data_ptr + pos + 16 - 4 + i)
        idx = (i * 256 + byte_val) % 4096
        tl.atomic_add(out_base + idx, 1.0)
    
    for i in tl.static_range(15):
        b1 = tl.load(data_ptr + pos + i)
        b2 = tl.load(data_ptr + pos + i + 1)
        idx = 4096 + ((i * 65537 + b1 * 257 + b2) % 16384)
        tl.atomic_add(out_base + idx, 1.0)
    
    for i in tl.static_range(14):
        b1 = tl.load(data_ptr + pos + i)
        b2 = tl.load(data_ptr + pos + i + 1)
        b3 = tl.load(data_ptr + pos + i + 2)
        idx = 20480 + ((b1 * 65537 + b2 * 257 + b3) % 16384)
        tl.atomic_add(out_base + idx, 1.0)
    
    for i in tl.static_range(13):
        b1 = tl.load(data_ptr + pos + i)
        b2 = tl.load(data_ptr + pos + i + 1)
        b3 = tl.load(data_ptr + pos + i + 2)
        b4 = tl.load(data_ptr + pos + i + 3)
        idx = 36864 + ((b1 * 16777259 + b2 * 65537 + b3 * 257 + b4) % 16384)
        tl.atomic_add(out_base + idx, 1.0)
    
    for i in tl.static_range(14):
        b1 = tl.load(data_ptr + pos + i)
        b3 = tl.load(data_ptr + pos + i + 2)
        idx = 53248 + ((b1 * 257 + b3) % 12288)
        tl.atomic_add(out_base + idx, 1.0)


@triton.jit
def extract_features_256k_kernel(data_ptr, positions_ptr, features_ptr, stride_feat):
    """262,144 features = 2^18 (16× more hash buckets)"""
    pid = tl.program_id(0)
    pos = tl.load(positions_ptr + pid)
    out_base = features_ptr + pid * stride_feat
    
    # 16384 unigrams + 65536 bigrams + 65536 trigrams + 65536 4grams + 49152 skipgrams = 262144
    for i in tl.static_range(4):
        byte_val = tl.load(data_ptr + pos + 16 - 4 + i)
        idx = (i * 256 + byte_val) % 16384
        tl.atomic_add(out_base + idx, 1.0)
    
    for i in tl.static_range(15):
        b1 = tl.load(data_ptr + pos + i)
        b2 = tl.load(data_ptr + pos + i + 1)
        idx = 16384 + ((i * 65537 + b1 * 257 + b2) % 65536)
        tl.atomic_add(out_base + idx, 1.0)
    
    for i in tl.static_range(14):
        b1 = tl.load(data_ptr + pos + i)
        b2 = tl.load(data_ptr + pos + i + 1)
        b3 = tl.load(data_ptr + pos + i + 2)
        idx = 81920 + ((b1 * 65537 + b2 * 257 + b3) % 65536)
        tl.atomic_add(out_base + idx, 1.0)
    
    for i in tl.static_range(13):
        b1 = tl.load(data_ptr + pos + i)
        b2 = tl.load(data_ptr + pos + i + 1)
        b3 = tl.load(data_ptr + pos + i + 2)
        b4 = tl.load(data_ptr + pos + i + 3)
        idx = 147456 + ((b1 * 16777259 + b2 * 65537 + b3 * 257 + b4) % 65536)
        tl.atomic_add(out_base + idx, 1.0)
    
    for i in tl.static_range(14):
        b1 = tl.load(data_ptr + pos + i)
        b3 = tl.load(data_ptr + pos + i + 2)
        idx = 212992 + ((b1 * 257 + b3) % 49152)
        tl.atomic_add(out_base + idx, 1.0)


def extract_features(data: torch.Tensor, positions: torch.Tensor, n_features: int) -> torch.Tensor:
    """Extract features at given scale."""
    B = positions.shape[0]
    features = torch.zeros((B, n_features), dtype=torch.float32, device=device)
    
    if n_features == 16384:
        extract_features_16k_kernel[(B,)](data, positions, features, n_features)
    elif n_features == 65536:
        extract_features_64k_kernel[(B,)](data, positions, features, n_features)
    elif n_features == 262144:
        extract_features_256k_kernel[(B,)](data, positions, features, n_features)
    else:
        raise ValueError(f"Unsupported feature size: {n_features}")
    
    return features


class ScaledQTT:
    """QTT model with configurable feature dimensions."""
    
    def __init__(self, n_qubits_feat: int, n_qubits_vocab: int = 8, max_rank: int = 64):
        self.n_qubits_feat = n_qubits_feat
        self.n_qubits_vocab = n_qubits_vocab
        self.n_qubits = n_qubits_feat + n_qubits_vocab
        self.max_rank = max_rank
        
        self.n_features = 2 ** n_qubits_feat
        self.n_vocab = 2 ** n_qubits_vocab
        
        # Initialize cores with Xavier initialization
        self.cores = []
        r_left = 1
        for i in range(self.n_qubits):
            r_right = min(max_rank, 2 ** min(i + 1, self.n_qubits - i - 1)) if i < self.n_qubits - 1 else 1
            # Xavier init: scale by sqrt(2/(fan_in + fan_out))
            std = math.sqrt(2.0 / (r_left + r_right))
            core = torch.randn(r_left, 2, r_right, device=device) * std
            core.requires_grad = True
            self.cores.append(core)
            r_left = r_right
        
        self.n_params = sum(c.numel() for c in self.cores)
    
    def to_dense(self) -> torch.Tensor:
        """Convert QTT to dense matrix."""
        result = self.cores[0]
        for core in self.cores[1:]:
            r_left, size, _ = result.shape
            result = torch.einsum('ijk,klm->ijlm', result, core)
            result = result.reshape(r_left, size * 2, -1)
        result = result.squeeze(0).squeeze(-1)
        return result.reshape(self.n_features, self.n_vocab)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x @ W."""
        W = self.to_dense()
        return x @ W
    
    def parameters(self):
        return self.cores


def train_and_evaluate(n_qubits_feat: int, max_rank: int = 64):
    """Train and evaluate QTT at given scale."""
    
    from datasets import load_dataset
    
    n_features = 2 ** n_qubits_feat
    n_qubits_total = n_qubits_feat + 8
    
    print(f"\n{'='*60}")
    print(f"SCALE TEST: {n_features:,} features ({n_qubits_feat} qubits)")
    print(f"{'='*60}")
    
    # Model
    model = ScaledQTT(n_qubits_feat, n_qubits_vocab=8, max_rank=max_rank)
    
    print(f"QTT: {model.n_params:,} params, {n_qubits_total} qubits, rank {max_rank}")
    print(f"Dense equivalent: {n_features * 256 * 4 / 1e6:.1f} MB")
    print(f"QTT memory: {model.n_params * 4 / 1e6:.2f} MB")
    print(f"Compression: {n_features * 256 / model.n_params:.1f}×")
    
    # Load data
    print("\nLoading data...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    train_text = "\n".join(dataset["train"]["text"]).encode('utf-8')
    test_text = "\n".join(dataset["test"]["text"]).encode('utf-8')
    
    train_data = torch.tensor(list(train_text), dtype=torch.int32, device=device)
    test_data = torch.tensor(list(test_text), dtype=torch.int32, device=device)
    
    n_train = len(train_data) - CTX_LEN - 1
    MAX_SAMPLES = 200000
    step = n_train // MAX_SAMPLES
    positions = torch.arange(0, n_train, step, dtype=torch.int64, device=device)[:MAX_SAMPLES]
    targets = train_data[positions + CTX_LEN]
    
    # Training
    print(f"\nTraining...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    BATCH = 4096
    N_EPOCHS = 5  # Fewer epochs for faster testing
    
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    
    for epoch in range(N_EPOCHS):
        epoch_loss = 0.0
        n_batches = 0
        
        perm = torch.randperm(len(positions), device=device)
        
        for b in range(0, len(positions), BATCH):
            idx = perm[b:b+BATCH]
            batch_pos = positions[idx]
            batch_tgt = targets[idx].long()
            
            X = extract_features(train_data, batch_pos, n_features)
            logits = model.forward(X)
            
            loss = torch.nn.functional.cross_entropy(logits, batch_tgt)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        print(f"  Epoch {epoch+1}/{N_EPOCHS}: loss = {epoch_loss/n_batches:.4f}")
    
    train_time = time.perf_counter() - start
    vram_peak = torch.cuda.max_memory_allocated()
    
    # Evaluation
    print(f"\nEvaluating...")
    n_test = len(test_data) - CTX_LEN - 1
    TEST_N = 20000
    test_pos = torch.arange(0, n_test, n_test//TEST_N, dtype=torch.int64, device=device)[:TEST_N]
    test_actual = test_data[test_pos + CTX_LEN]
    
    with torch.no_grad():
        X_test = extract_features(test_data, test_pos, n_features)
        logits = model.forward(X_test)
        
        probs = torch.softmax(logits, dim=1)
        acc = (probs.argmax(dim=1) == test_actual).float().mean().item()
        
        test_loss = torch.nn.functional.cross_entropy(logits, test_actual.long())
        ppl = math.exp(test_loss.item())
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {n_features:,} features")
    print(f"{'='*60}")
    print(f"Accuracy: {acc*100:.1f}%")
    print(f"Perplexity: {ppl:.2f}")
    print(f"Train time: {train_time:.1f}s")
    print(f"Parameters: {model.n_params:,}")
    print(f"VRAM: {vram_peak/1e9:.2f} GB")
    print(f"Compression: {n_features * 256 / model.n_params:.1f}×")
    
    return {
        'n_features': n_features,
        'n_qubits': n_qubits_total,
        'params': model.n_params,
        'accuracy': acc,
        'perplexity': ppl,
        'train_time': train_time,
        'vram': vram_peak,
        'compression': n_features * 256 / model.n_params
    }


def main():
    print("="*60)
    print("OPTIMIZATION 3: SCALING TO LARGER FEATURE SPACES")
    print("="*60)
    
    results = []
    
    # Test multiple scales with OPTIMAL RANK (from NS methodology sweep)
    # 14 qubits = 16,384 features (baseline)
    # 16 qubits = 65,536 features (4× more)
    # 18 qubits = 262,144 features (16× more)
    
    OPTIMAL_RANK = 24  # From rank sweep (was 64)
    
    for n_qubits in [14, 16, 18]:
        try:
            result = train_and_evaluate(n_qubits, max_rank=OPTIMAL_RANK)
            results.append(result)
        except torch.cuda.OutOfMemoryError:
            print(f"\nOOM at {2**n_qubits:,} features - stopping")
            break
        
        torch.cuda.empty_cache()
    
    # Summary
    print(f"\n{'='*60}")
    print("SCALING SUMMARY")
    print(f"{'='*60}")
    print(f"{'Features':<12} {'Params':<10} {'Acc':<8} {'PPL':<8} {'Time':<8} {'Compress'}")
    print("-"*60)
    for r in results:
        print(f"{r['n_features']:<12,} {r['params']:<10,} {r['accuracy']*100:<8.1f} {r['perplexity']:<8.2f} {r['train_time']:<8.1f} {r['compression']:.0f}×")


if __name__ == "__main__":
    main()
