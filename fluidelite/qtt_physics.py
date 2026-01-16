#!/usr/bin/env python3
"""
NS Millennium - Let Physics Determine Rank

Two approaches:
1. SVD spectrum analysis - what rank captures 90/95/99% energy?
2. Train high, retract - what rank does the model actually need?

Start with 16K (validated) to sanity check, then scale.
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

# 16K scale (validated)
N_FEATURES = 16384  # 2^14
N_VOCAB = 256       # 2^8
CTX_LEN = 16
N_QUBITS_FEAT = 14
N_QUBITS_VOCAB = 8
TOTAL_QUBITS = N_QUBITS_FEAT + N_QUBITS_VOCAB
MAX_RANK_CAP = 256  # High cap - let physics decide

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Dense equivalent: {N_FEATURES * N_VOCAB:,} params")

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
# Step 1: Validate Data Isn't Bullshit
# ============================================================================

def validate_data(train_data, positions, targets):
    """Show actual samples to verify data makes sense."""
    print("\n" + "="*70)
    print("STEP 1: DATA VALIDATION")
    print("="*70)
    
    print(f"\nData shape: {train_data.shape}")
    print(f"Positions: {len(positions):,}")
    print(f"Targets: {len(targets):,}")
    
    # Show 5 samples
    print("\nSample context → target:")
    for i in range(5):
        pos = positions[i].item()
        ctx = train_data[pos:pos+CTX_LEN].cpu().numpy()
        tgt = targets[i].item()
        
        # Decode bytes to text
        try:
            ctx_text = bytes(ctx).decode('utf-8', errors='replace')
        except:
            ctx_text = str(ctx)
        
        try:
            tgt_char = chr(tgt) if 32 <= tgt < 127 else f"[{tgt}]"
        except:
            tgt_char = f"[{tgt}]"
        
        print(f"  {i}: '{ctx_text}' → '{tgt_char}'")
    
    # Check target distribution
    unique, counts = torch.unique(targets, return_counts=True)
    top_k = 10
    top_idx = counts.argsort(descending=True)[:top_k]
    
    print(f"\nTop {top_k} target tokens:")
    for idx in top_idx:
        token = unique[idx].item()
        count = counts[idx].item()
        pct = 100 * count / len(targets)
        try:
            char = chr(token) if 32 <= token < 127 else f"[{token}]"
        except:
            char = f"[{token}]"
        print(f"  '{char}' (byte {token}): {count:,} ({pct:.1f}%)")
    
    # Random baseline
    random_acc = 100.0 / N_VOCAB
    top1_baseline = 100 * counts.max().item() / len(targets)
    print(f"\nBaselines:")
    print(f"  Random guess: {random_acc:.2f}%")
    print(f"  Always predict most common: {top1_baseline:.1f}%")
    
    return True


# ============================================================================
# Step 2: Build Oracle Matrix via Sampling (for SVD analysis)
# ============================================================================

def build_sampled_oracle(train_data, positions, targets, n_samples=50000):
    """Build W matrix from samples: W[i,j] = P(target=j | feature_i active)"""
    print("\n" + "="*70)
    print("STEP 2: BUILD ORACLE MATRIX (sampled)")
    print("="*70)
    
    # Sample subset
    idx = torch.randperm(len(positions))[:n_samples]
    pos_sample = positions[idx]
    tgt_sample = targets[idx]
    
    # Extract features
    print(f"Extracting features for {n_samples:,} samples...")
    X = extract_features(train_data, pos_sample)
    
    # Build empirical W: accumulate feature → target co-occurrences
    print("Building W matrix...")
    W = torch.zeros(N_FEATURES, N_VOCAB, device=device)
    
    # One-hot targets
    Y = F.one_hot(tgt_sample.long(), N_VOCAB).float()
    
    # W = X^T @ Y (unnormalized)
    W = X.T @ Y
    
    # Normalize each row (feature) to get conditional probabilities
    row_sums = W.sum(dim=1, keepdim=True).clamp(min=1)
    W = W / row_sums
    
    print(f"W shape: {W.shape}")
    print(f"W non-zero: {(W > 0).sum().item():,} / {W.numel():,}")
    
    return W


# ============================================================================
# Step 3: SVD Spectrum Analysis
# ============================================================================

def analyze_spectrum(W):
    """SVD W and see what rank captures 90/95/99% energy."""
    print("\n" + "="*70)
    print("STEP 3: SVD SPECTRUM ANALYSIS")
    print("="*70)
    
    print("Computing SVD...")
    t0 = time.time()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    print(f"SVD done in {time.time()-t0:.1f}s")
    
    # Energy spectrum
    energy = S ** 2
    total_energy = energy.sum()
    cumulative = torch.cumsum(energy, dim=0) / total_energy
    
    print(f"\nSingular value spectrum:")
    print(f"  Top 10 singular values: {S[:10].cpu().numpy()}")
    print(f"  Total energy: {total_energy.item():.2f}")
    
    # Find rank for energy thresholds
    thresholds = [0.50, 0.80, 0.90, 0.95, 0.99, 0.999]
    print(f"\nRANK REQUIRED FOR ENERGY CAPTURE:")
    print(f"{'Threshold':<12} {'Rank':<10} {'Compression':<12}")
    print("-" * 40)
    
    for thresh in thresholds:
        mask = cumulative >= thresh
        if mask.any():
            rank = mask.nonzero()[0].item() + 1
        else:
            rank = len(S)
        compression = (N_FEATURES * N_VOCAB) / (rank * (N_FEATURES + N_VOCAB))
        print(f"{thresh*100:.1f}%         {rank:<10} {compression:.1f}×")
    
    return S, cumulative


# ============================================================================
# Step 4: Train with High Cap, See What Rank Emerges
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
    
    def get_ranks(self) -> List[int]:
        return [c.shape[2] for c in self.cores[:-1]] + [1]
    
    def to_dense(self, n_row_qubits: int) -> torch.Tensor:
        result = self.cores[0]
        for core in self.cores[1:]:
            result = torch.einsum('ijk,klm->ijlm', result, core)
            result = result.reshape(result.shape[0], -1, result.shape[-1])
        result = result.squeeze(0).squeeze(-1)
        return result.reshape(2**n_row_qubits, 2**(self.n_qubits - n_row_qubits))


def tt_svd_retract(W: torch.Tensor, n_qubits: int, n_row_qubits: int, threshold: float = 0.01):
    """TT-SVD decomposition - see what ranks physics chooses."""
    # Reshape to QTT form
    shape = [2] * n_qubits
    T = W.reshape(shape)
    
    cores = []
    remaining = T.reshape(2, -1)
    
    ranks_chosen = []
    
    for i in range(n_qubits - 1):
        m, n = remaining.shape
        
        # SVD
        U, S, Vh = torch.linalg.svd(remaining, full_matrices=False)
        
        # Find rank that captures (1-threshold) energy
        energy = S ** 2
        cumulative = torch.cumsum(energy, dim=0) / energy.sum()
        
        mask = cumulative >= (1 - threshold)
        if mask.any():
            r = mask.nonzero()[0].item() + 1
        else:
            r = len(S)
        
        r = max(1, min(r, len(S)))  # Clamp
        
        ranks_chosen.append(r)
        
        U, S, Vh = U[:, :r], S[:r], Vh[:r, :]
        
        core = U.reshape(-1, 2, r)
        cores.append(core)
        
        remaining = torch.diag(S) @ Vh
        if i < n_qubits - 2:
            remaining = remaining.reshape(r * 2, -1)
    
    # Final core
    cores.append(remaining.reshape(-1, 2, 1))
    ranks_chosen.append(1)
    
    return cores, ranks_chosen


def train_and_retract(train_data, positions, targets, n_epochs=30):
    """Train with high rank cap, then retract to see natural rank."""
    print("\n" + "="*70)
    print("STEP 4: TRAIN WITH HIGH CAP, RETRACT")
    print("="*70)
    
    # Initialize with HIGH rank cap
    print(f"Initializing QTT with max_rank={MAX_RANK_CAP} (high cap)")
    qtt = QTTMatrix.random_init(TOTAL_QUBITS, max_rank=MAX_RANK_CAP)
    
    for core in qtt.cores:
        core.requires_grad_(True)
    
    optimizer = torch.optim.Adam(qtt.cores, lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    n_samples = len(positions)
    batch_size = 2000
    
    print(f"Training for {n_epochs} epochs (until convergence)...")
    best_acc = 0
    patience = 5
    no_improve = 0
    
    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples, device=device)
        epoch_loss = 0
        n_batches = 0
        
        for b in range(0, n_samples, batch_size):
            batch_idx = perm[b:b+batch_size]
            batch_pos = positions[batch_idx]
            batch_tgt = targets[batch_idx]
            
            x = extract_features(train_data, batch_pos)
            
            optimizer.zero_grad()
            W = qtt.to_dense(N_QUBITS_FEAT)
            logits = x @ W
            loss = F.cross_entropy(logits, batch_tgt.long())
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / n_batches
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                n_eval = min(10000, len(positions))
                eval_idx = torch.randperm(len(positions))[:n_eval]
                X_eval = extract_features(train_data, positions[eval_idx])
                W = qtt.to_dense(N_QUBITS_FEAT)
                logits = X_eval @ W
                acc = (logits.argmax(1) == targets[eval_idx]).float().mean().item() * 100
                print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, acc={acc:.1f}%")
                
                if acc > best_acc + 0.5:
                    best_acc = acc
                    no_improve = 0
                else:
                    no_improve += 1
                
                if no_improve >= patience:
                    print(f"  Early stop - no improvement for {patience} checks")
                    break
        else:
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}")
    
    # Get trained W
    for core in qtt.cores:
        core.requires_grad_(False)
    
    W = qtt.to_dense(N_QUBITS_FEAT)
    
    # Retract with TT-SVD at different thresholds
    print(f"\nRetracting to find natural ranks...")
    
    for threshold in [0.01, 0.05, 0.10]:
        cores, ranks = tt_svd_retract(W, TOTAL_QUBITS, N_QUBITS_FEAT, threshold)
        max_rank = max(ranks)
        total_params = sum(c.numel() for c in cores)
        compression = (N_FEATURES * N_VOCAB) / total_params
        
        print(f"\n  Threshold {threshold*100:.0f}% error allowed:")
        print(f"    Max rank physics chose: {max_rank}")
        print(f"    Rank profile: {ranks[:5]}...{ranks[-5:]}")
        print(f"    Params: {total_params:,}")
        print(f"    Compression: {compression:.0f}×")
    
    # Evaluate accuracy
    print(f"\nEvaluating trained model...")
    with torch.no_grad():
        n_test = min(20000, len(positions))
        test_idx = torch.randperm(len(positions))[:n_test]
        X_test = extract_features(train_data, positions[test_idx])
        logits = X_test @ W
        preds = logits.argmax(dim=1)
        accuracy = (preds == targets[test_idx]).float().mean().item() * 100
        print(f"  Accuracy: {accuracy:.1f}%")
    
    return W, accuracy


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "="*70)
    print("NS MILLENNIUM - LET PHYSICS DETERMINE RANK")
    print("="*70)
    print(f"Scale: 16K features × 256 vocab = {N_FEATURES * N_VOCAB:,} dense")
    
    # Load data
    print("\nLoading WikiText-103...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    train_text = "\n".join(dataset["train"]["text"]).encode('utf-8')
    
    train_data = torch.tensor(list(train_text), dtype=torch.int32, device=device)
    print(f"Train: {len(train_data)/1e6:.1f}M bytes")
    
    # Sample positions
    n_train = len(train_data) - CTX_LEN - 1
    n_samples = 200000
    step = n_train // n_samples
    positions = torch.arange(0, n_train, step, dtype=torch.int64, device=device)[:n_samples]
    targets = train_data[positions + CTX_LEN]
    
    # Step 1: Validate data
    validate_data(train_data, positions, targets)
    
    # Step 2: Build oracle matrix
    W = build_sampled_oracle(train_data, positions, targets, n_samples=50000)
    
    # Step 3: SVD spectrum analysis
    S, cumulative = analyze_spectrum(W)
    
    # Step 4: Train and retract
    W_trained, accuracy = train_and_retract(train_data, positions, targets, n_epochs=50)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Physics analysis complete at 16K scale.")
    print(f"Final accuracy: {accuracy:.1f}%")


if __name__ == '__main__':
    main()
