#!/usr/bin/env python3
"""
QTT Training - 100M samples (~18% corpus)

Output every step for visibility.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import time
import math
from typing import List, Dict
from datasets import load_dataset

device = torch.device('cuda')

# Dimensions - 16K scale (4.2M dense equivalent)
N_FEATURES = 16384   # 2^14
N_VOCAB = 256        # 2^8
CTX_LEN = 16
N_QUBITS_FEAT = 14
N_QUBITS_VOCAB = 8
TOTAL_QUBITS = N_QUBITS_FEAT + N_QUBITS_VOCAB
MAX_RANK = 24        # Same rank that got 36% before

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================================
# Triton Feature Extraction
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
# QTT Operations
# ============================================================================

class QTTMatrix:
    def __init__(self, cores: List[torch.Tensor]):
        self.cores = cores
        self.n_qubits = len(cores)
    
    @classmethod
    def random_init(cls, n_qubits: int, max_rank: int) -> 'QTTMatrix':
        cores = []
        r_left = 1
        for i in range(n_qubits):
            if i < n_qubits - 1:
                r_right = min(max_rank, 2 ** min(i + 1, n_qubits - i - 1))
            else:
                r_right = 1
            
            std = math.sqrt(2.0 / (r_left + r_right))
            core = torch.randn(r_left, 2, r_right, device=device) * std
            cores.append(core)
            r_left = r_right
        return cls(cores)
    
    def n_params(self) -> int:
        return sum(c.numel() for c in self.cores)
    
    def to_dense(self, n_row_qubits: int) -> torch.Tensor:
        result = self.cores[0]
        for core in self.cores[1:]:
            r_left, size, r_mid = result.shape
            result = torch.einsum('ijk,klm->ijlm', result, core)
            result = result.reshape(r_left, size * 2, result.shape[-1])
        result = result.squeeze(0).squeeze(-1)
        return result.reshape(2**n_row_qubits, 2**(self.n_qubits - n_row_qubits))


def qtt_forward(qtt: QTTMatrix, x: torch.Tensor) -> torch.Tensor:
    W = qtt.to_dense(N_QUBITS_FEAT)
    return x @ W


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_at_rank(train_data: torch.Tensor, positions: torch.Tensor, targets: torch.Tensor,
                  max_rank: int, n_epochs: int = 10, batch_size: int = 4000, lr: float = 0.01) -> 'QTTMatrix':
    """Train QTT at given rank - streaming to handle large datasets."""
    
    qtt = QTTMatrix.random_init(TOTAL_QUBITS, max_rank=max_rank)
    
    for core in qtt.cores:
        core.requires_grad_(True)
    
    optimizer = torch.optim.Adam(qtt.cores, lr=lr)
    n_samples = len(positions)
    total_batches = (n_samples + batch_size - 1) // batch_size
    
    print(f"    Total batches per epoch: {total_batches:,}")
    
    global_step = 0
    start_time = time.time()
    
    for epoch in range(n_epochs):
        # TRUE SHUFFLE: generate random permutation and stream through it
        # Can't materialize 541M permutation, so shuffle in chunks
        chunk_size = 1_000_000  # 1M positions per chunk (8MB)
        n_chunks = (n_samples + chunk_size - 1) // chunk_size
        chunk_order = torch.randperm(n_chunks)  # Shuffle chunk order
        
        epoch_loss = 0
        n_batches = 0
        
        for chunk_idx in chunk_order:
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, n_samples)
            
            # Shuffle within this chunk
            chunk_perm = torch.randperm(chunk_end - chunk_start)
            chunk_positions = positions[chunk_start:chunk_end][chunk_perm]
            chunk_targets = targets[chunk_start:chunk_end][chunk_perm]
            
            # Process batches from shuffled chunk
            for b in range(0, len(chunk_positions), batch_size):
                end = min(b + batch_size, len(chunk_positions))
                batch_pos = chunk_positions[b:end].to(device)
                batch_tgt = chunk_targets[b:end].to(device)
            
                x = extract_features(train_data, batch_pos)
            
                optimizer.zero_grad()
                logits = qtt_forward(qtt, x)
                loss = F.cross_entropy(logits, batch_tgt.long())
                loss.backward()
            
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(qtt.cores, max_norm=1.0)
            
                optimizer.step()
            
                epoch_loss += loss.item()
                n_batches += 1
                global_step += 1
                
                # Print EVERY step
                elapsed = time.time() - start_time
                steps_per_sec = global_step / elapsed
                print(f"    step {global_step:,} | epoch {epoch+1} | loss={loss.item():.4f} | {steps_per_sec:.1f} steps/s", flush=True)
        
        print(f"  === EPOCH {epoch+1}/{n_epochs} DONE: avg_loss={epoch_loss/n_batches:.4f} ===", flush=True)
    
    for core in qtt.cores:
        core.requires_grad_(False)
    
    return qtt


def evaluate(qtt: QTTMatrix, test_data: torch.Tensor, test_pos: torch.Tensor, 
             test_targets: torch.Tensor) -> Dict:
    """Evaluate QTT model."""
    with torch.no_grad():
        X = extract_features(test_data, test_pos)
        logits = qtt_forward(qtt, X)
        probs = F.softmax(logits, dim=1)
        
        preds = probs.argmax(dim=1)
        accuracy = (preds == test_targets).float().mean().item() * 100
        
        p_correct = probs[torch.arange(len(test_targets), device=device), test_targets.long()]
        perplexity = torch.exp(-torch.log(p_correct.clamp(min=1e-10)).mean()).item()
    
    return {'accuracy': accuracy, 'perplexity': perplexity}


# ============================================================================
# Main - Just train and report
# ============================================================================

def main():
    print("\n" + "="*70)
    print("QTT TRAINING - 100M SAMPLES (~18% corpus)")
    print("="*70)
    print(f"\nRank: {MAX_RANK}")
    
    # Load data
    print("\nLoading WikiText-103...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    train_text = "\n".join(dataset["train"]["text"]).encode('utf-8')
    test_text = "\n".join(dataset["test"]["text"]).encode('utf-8')
    
    train_data = torch.tensor(list(train_text), dtype=torch.int32, device=device)
    test_data = torch.tensor(list(test_text), dtype=torch.int32, device=device)
    print(f"Train: {len(train_data)/1e6:.1f}M bytes, Test: {len(test_data)/1e6:.1f}M bytes")
    
    # Training positions - 100M samples
    n_train = len(train_data) - CTX_LEN - 1
    MAX_SAMPLES = 100_000_000  # 100M samples
    step = max(1, n_train // MAX_SAMPLES)
    positions = torch.arange(0, n_train, step, dtype=torch.int64)[:MAX_SAMPLES]  # CPU
    targets = train_data.cpu()[positions + CTX_LEN]  # CPU
    print(f"Training samples: {len(positions):,} ({100*len(positions)/n_train:.1f}% of corpus)")
    
    # Test positions
    n_test = len(test_data) - CTX_LEN - 1
    TEST_N = 20000
    test_step = n_test // TEST_N
    test_pos = torch.arange(0, n_test, test_step, dtype=torch.int64, device=device)[:TEST_N]
    test_targets = test_data[test_pos + CTX_LEN]
    print(f"Test samples: {len(test_pos):,}")
    
    # Dense baseline
    dense_params = N_FEATURES * N_VOCAB
    print(f"\nDense equivalent: {dense_params:,} params ({dense_params*4/1e6:.1f} MB)")
    
    # JUST TRAIN
    print(f"\n{'='*70}")
    print(f"TRAINING at rank {MAX_RANK}...")
    print(f"{'='*70}")
    
    qtt = train_at_rank(train_data, positions, targets, max_rank=MAX_RANK, n_epochs=10)
    
    # EVALUATE
    result = evaluate(qtt, test_data, test_pos, test_targets)
    
    # REPORT
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  Accuracy: {result['accuracy']:.1f}%")
    print(f"  Perplexity: {result['perplexity']:.2f}")
    print(f"  Params: {qtt.n_params():,}")
    print(f"  Compression: {dense_params / qtt.n_params():.0f}×")
    print(f"\n  Training: 100M samples ({len(positions):,})")
    
    return result


if __name__ == '__main__':
    result = main()
