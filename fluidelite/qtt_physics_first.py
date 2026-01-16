#!/usr/bin/env python3
"""
NS Millennium - Physics First

1. Validate data is real
2. Train with rank as CAP, not control
3. Let TT-SVD find natural rank via retraction
4. Let compression fall out naturally
5. Target: 90%+ accuracy

Quick test at 16K first, then 134M.
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

# ============================================================================
# STEP 1: VALIDATE DATA
# ============================================================================

def validate_data():
    """Check WikiText is real text, not garbage."""
    print("\n" + "="*70)
    print("STEP 1: DATA VALIDATION")
    print("="*70)
    
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    train_text = "\n".join(dataset["train"]["text"])
    test_text = "\n".join(dataset["test"]["text"])
    
    # Show samples
    print(f"\nTrain size: {len(train_text):,} chars")
    print(f"Test size: {len(test_text):,} chars")
    
    print("\n--- Train sample (first 500 chars) ---")
    print(train_text[:500])
    
    print("\n--- Train sample (random 500 chars) ---")
    import random
    start = random.randint(1000000, len(train_text) - 1000)
    print(train_text[start:start+500])
    
    # Character distribution
    from collections import Counter
    char_counts = Counter(train_text[:100000])
    print("\n--- Top 20 characters ---")
    for char, count in char_counts.most_common(20):
        display = repr(char) if char in '\n\t ' else char
        print(f"  {display}: {count}")
    
    # Check it's English-ish
    alpha_ratio = sum(1 for c in train_text[:10000] if c.isalpha()) / 10000
    space_ratio = sum(1 for c in train_text[:10000] if c == ' ') / 10000
    print(f"\nAlpha ratio: {alpha_ratio:.1%}")
    print(f"Space ratio: {space_ratio:.1%}")
    
    if alpha_ratio < 0.5:
        print("⚠️ WARNING: Data may not be proper English text!")
    else:
        print("✅ Data looks like valid English text")
    
    return train_text.encode('utf-8'), test_text.encode('utf-8')


# ============================================================================
# STEP 2-4: TRAIN WITH RANK AS CAP
# ============================================================================

# Dimensions for 16K test
N_FEATURES = 16384  # 2^14
N_VOCAB = 256       # 2^8
CTX_LEN = 16
N_QUBITS_FEAT = 14
N_QUBITS_VOCAB = 8
TOTAL_QUBITS = N_QUBITS_FEAT + N_QUBITS_VOCAB

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


def extract_features(data: torch.Tensor, positions: torch.Tensor, n_features: int) -> torch.Tensor:
    B = positions.shape[0]
    features = torch.zeros((B, n_features), dtype=torch.float32, device=device)
    extract_features_kernel[(B,)](data, positions, features, n_features)
    return features


class QTTMatrix:
    def __init__(self, cores: List[torch.Tensor]):
        self.cores = cores
        self.n_qubits = len(cores)
    
    @classmethod
    def random_init(cls, n_qubits: int, max_rank: int) -> 'QTTMatrix':
        """Initialize with max_rank as CAP - actual ranks determined by position."""
        cores = []
        r_left = 1
        for i in range(n_qubits):
            if i < n_qubits - 1:
                # Natural rank growth capped by max_rank
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
    
    def get_ranks(self) -> List[int]:
        """Return actual ranks at each bond."""
        return [c.shape[2] for c in self.cores[:-1]]
    
    def to_dense(self, n_row_qubits: int) -> torch.Tensor:
        result = self.cores[0]
        for core in self.cores[1:]:
            result = torch.einsum('ijk,klm->ijlm', result, core)
            result = result.reshape(result.shape[0], -1, result.shape[-1])
        result = result.squeeze(0).squeeze(-1)
        return result.reshape(2**n_row_qubits, 2**(self.n_qubits - n_row_qubits))


def train_to_target(train_data, positions, targets, test_data, test_pos, test_targets,
                    n_feat_qubits, n_vocab_qubits, n_features,
                    max_rank_cap: int = 256, target_acc: float = 90.0,
                    max_epochs: int = 100, batch_size: int = 2000, lr: float = 0.03):
    """
    Train until target accuracy or convergence.
    
    max_rank_cap: Upper limit on rank (not forced, just capped)
    Let physics determine actual compression.
    """
    
    total_qubits = n_feat_qubits + n_vocab_qubits
    dense_equiv = n_features * N_VOCAB
    
    print(f"\n" + "="*70)
    print(f"TRAINING TO TARGET: {target_acc}%")
    print(f"="*70)
    print(f"  Dense equivalent: {dense_equiv:,} params")
    print(f"  Max rank cap: {max_rank_cap}")
    print(f"  Max epochs: {max_epochs}")
    
    # Initialize QTT with high rank cap
    qtt = QTTMatrix.random_init(total_qubits, max_rank=max_rank_cap)
    
    print(f"  Initial QTT params: {qtt.n_params():,}")
    print(f"  Initial ranks: {qtt.get_ranks()}")
    
    for core in qtt.cores:
        core.requires_grad_(True)
    
    optimizer = torch.optim.Adam(qtt.cores, lr=lr)
    n_samples = len(positions)
    
    best_acc = 0.0
    best_epoch = 0
    plateau_count = 0
    
    for epoch in range(max_epochs):
        t0 = time.time()
        perm = torch.randperm(n_samples, device=device)
        epoch_loss = 0.0
        n_batches = 0
        
        for b in range(0, n_samples, batch_size):
            batch_idx = perm[b:b+batch_size]
            batch_pos = positions[batch_idx]
            batch_tgt = targets[batch_idx]
            
            x = extract_features(train_data, batch_pos, n_features)
            
            optimizer.zero_grad()
            W = qtt.to_dense(n_feat_qubits)
            logits = x @ W
            loss = F.cross_entropy(logits, batch_tgt.long())
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        # Evaluate
        with torch.no_grad():
            X_test = extract_features(test_data, test_pos, n_features)
            W = qtt.to_dense(n_feat_qubits)
            logits = X_test @ W
            preds = logits.argmax(dim=1)
            accuracy = (preds == test_targets).float().mean().item() * 100
            
            probs = F.softmax(logits, dim=1)
            p_correct = probs[torch.arange(len(test_targets), device=device), test_targets.long()]
            perplexity = torch.exp(-torch.log(p_correct.clamp(min=1e-10)).mean()).item()
        
        elapsed = time.time() - t0
        compression = dense_equiv / qtt.n_params()
        
        print(f"  Epoch {epoch+1:3d}: loss={epoch_loss/n_batches:.4f}, "
              f"acc={accuracy:.1f}%, ppl={perplexity:.2f}, "
              f"params={qtt.n_params():,}, compress={compression:.0f}×, "
              f"time={elapsed:.1f}s")
        
        # Check if target met
        if accuracy >= target_acc:
            print(f"\n  ✅ TARGET {target_acc}% ACHIEVED at epoch {epoch+1}!")
            return {
                'met': True,
                'accuracy': accuracy,
                'perplexity': perplexity,
                'params': qtt.n_params(),
                'compression': compression,
                'epochs': epoch + 1,
                'ranks': qtt.get_ranks(),
            }
        
        # Track best and plateau
        if accuracy > best_acc + 0.5:
            best_acc = accuracy
            best_epoch = epoch
            plateau_count = 0
        else:
            plateau_count += 1
        
        # Early stopping if plateaued for 5 epochs
        if plateau_count >= 5:
            print(f"\n  ⚠️ PLATEAU detected - no improvement for 5 epochs")
            break
    
    compression = dense_equiv / qtt.n_params()
    return {
        'met': False,
        'accuracy': best_acc,
        'perplexity': perplexity,
        'params': qtt.n_params(),
        'compression': compression,
        'epochs': best_epoch + 1,
        'ranks': qtt.get_ranks(),
    }


def main():
    print("\n" + "="*70)
    print("NS MILLENNIUM - PHYSICS FIRST")
    print("="*70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Step 1: Validate data
    train_bytes, test_bytes = validate_data()
    
    train_data = torch.tensor(list(train_bytes), dtype=torch.int32, device=device)
    test_data = torch.tensor(list(test_bytes), dtype=torch.int32, device=device)
    
    # Step 2-4: Quick test on 16K scale
    print("\n" + "="*70)
    print("STEP 2-4: QUICK TEST ON 16K SCALE")
    print("="*70)
    
    n_train = len(train_data) - CTX_LEN - 1
    MAX_SAMPLES = 100000  # Smaller for quick test
    step = n_train // MAX_SAMPLES
    positions = torch.arange(0, n_train, step, dtype=torch.int64, device=device)[:MAX_SAMPLES]
    targets = train_data[positions + CTX_LEN]
    
    n_test = len(test_data) - CTX_LEN - 1
    TEST_N = 10000
    test_step = n_test // TEST_N
    test_pos = torch.arange(0, n_test, test_step, dtype=torch.int64, device=device)[:TEST_N]
    test_targets = test_data[test_pos + CTX_LEN]
    
    print(f"\nTraining samples: {len(positions):,}")
    print(f"Test samples: {len(test_pos):,}")
    
    # Train with rank cap of 128, see what physics chooses
    result = train_to_target(
        train_data, positions, targets,
        test_data, test_pos, test_targets,
        n_feat_qubits=N_QUBITS_FEAT,
        n_vocab_qubits=N_QUBITS_VOCAB,
        n_features=N_FEATURES,
        max_rank_cap=128,  # CAP, not control
        target_acc=90.0,
        max_epochs=20,  # Quick test
        batch_size=2000,
        lr=0.03,
    )
    
    print("\n" + "="*70)
    print("16K SCALE RESULTS")
    print("="*70)
    print(f"Target: 90%")
    print(f"Achieved: {result['accuracy']:.1f}%")
    print(f"Params: {result['params']:,}")
    print(f"Compression: {result['compression']:.0f}×")
    print(f"Epochs: {result['epochs']}")
    print(f"Final ranks: {result['ranks']}")
    
    if result['met']:
        print("\n✅ 16K scale can hit 90% - proceed to 134M")
    else:
        print(f"\n❌ 16K scale plateaus at {result['accuracy']:.1f}%")
        print("   Architecture may be the bottleneck, not training.")
    
    return result


if __name__ == '__main__':
    result = main()
