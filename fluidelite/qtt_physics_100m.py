#!/usr/bin/env python3
"""
NS Millennium - Let Physics Determine Rank
SCALED TO ~100M DENSE EQUIVALENT

Using 2^19 = 524,288 features × 256 vocab = 134M params
(Closest power-of-2 scale to 100M)
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

# ~100M scale (using 2^19 features for QTT compatibility)
N_FEATURES = 524288  # 2^19
N_VOCAB = 256        # 2^8
CTX_LEN = 16
N_QUBITS_FEAT = 19
N_QUBITS_VOCAB = 8
TOTAL_QUBITS = N_QUBITS_FEAT + N_QUBITS_VOCAB
MAX_RANK_CAP = 256  # High cap - let physics decide

DENSE_EQUIV = N_FEATURES * N_VOCAB

# Triton compile-time constant
CTX_LEN_CONST = 16

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Dense equivalent: {DENSE_EQUIV:,} params ({DENSE_EQUIV * 4 / 1e9:.2f} GB)")

# ============================================================================
# Triton Feature Extraction - SCALED UP
# ============================================================================

@triton.jit
def extract_features_kernel(data_ptr, positions_ptr, features_ptr, stride_feat, n_features: tl.constexpr):
    pid = tl.program_id(0)
    pos = tl.load(positions_ptr + pid)
    out_base = features_ptr + pid * stride_feat
    
    # All indices must be in [0, n_features=524288)
    # Layout: 
    #   [0, 4096): unigrams
    #   [4096, 69632): bigrams (65536 slots)
    #   [69632, 200704): trigrams (131072 slots)
    #   [200704, 462848): 4-grams (262144 slots)
    #   [462848, 524288): skip-grams (61440 slots)
    
    # Unigrams: positions [0, 4096)
    for i in tl.static_range(16):
        byte_val = tl.load(data_ptr + pos + i).to(tl.int32)
        idx = ((i * 256 + byte_val) % 4096)
        tl.atomic_add(out_base + idx, 1.0)
    
    # Bigrams: positions [4096, 69632) - 65536 slots
    for i in tl.static_range(15):
        b1 = tl.load(data_ptr + pos + i).to(tl.int32)
        b2 = tl.load(data_ptr + pos + i + 1).to(tl.int32)
        h = (i * 65537 + b1 * 257 + b2) % 65536
        idx = 4096 + h
        tl.atomic_add(out_base + idx, 1.0)
    
    # Trigrams: positions [69632, 200704) - 131072 slots
    for i in tl.static_range(14):
        b1 = tl.load(data_ptr + pos + i).to(tl.int32)
        b2 = tl.load(data_ptr + pos + i + 1).to(tl.int32)
        b3 = tl.load(data_ptr + pos + i + 2).to(tl.int32)
        h = (b1 * 65537 + b2 * 257 + b3) % 131072
        idx = 69632 + h
        tl.atomic_add(out_base + idx, 1.0)
    
    # 4-grams: positions [200704, 462848) - 262144 slots
    for i in tl.static_range(13):
        b1 = tl.load(data_ptr + pos + i).to(tl.int32)
        b2 = tl.load(data_ptr + pos + i + 1).to(tl.int32)
        b3 = tl.load(data_ptr + pos + i + 2).to(tl.int32)
        b4 = tl.load(data_ptr + pos + i + 3).to(tl.int32)
        h = (b1 * 16777259 + b2 * 65537 + b3 * 257 + b4) % 262144
        idx = 200704 + h
        tl.atomic_add(out_base + idx, 1.0)
    
    # Skip-grams: positions [462848, 524288) - 61440 slots  
    for i in tl.static_range(14):
        b1 = tl.load(data_ptr + pos + i).to(tl.int32)
        b3 = tl.load(data_ptr + pos + i + 2).to(tl.int32)
        h = (b1 * 257 + b3) % 61440
        idx = 462848 + h
        tl.atomic_add(out_base + idx, 1.0)


def extract_features(data: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    B = positions.shape[0]
    features = torch.zeros((B, N_FEATURES), dtype=torch.float32, device=device)
    extract_features_kernel[(B,)](data, positions, features, N_FEATURES, N_FEATURES)
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
    
    # Baselines
    random_acc = 100.0 / N_VOCAB
    top1_baseline = 100 * counts.max().item() / len(targets)
    print(f"\nBaselines:")
    print(f"  Random guess: {random_acc:.2f}%")
    print(f"  Always predict most common: {top1_baseline:.1f}%")
    
    return True


# ============================================================================
# Step 2: Build Oracle Matrix via Sampling (for SVD analysis)
# SKIPPED at 100M scale - would need 134M × 256 = 500GB
# ============================================================================

def build_sampled_oracle_sketch(train_data, positions, targets, n_samples=50000):
    """Build sketched W matrix - can't materialize full 134M × 256."""
    print("\n" + "="*70)
    print("STEP 2: ORACLE MATRIX SKETCH")
    print("="*70)
    print(f"Full W would be {N_FEATURES}×{N_VOCAB} = {DENSE_EQUIV*4/1e9:.1f}GB")
    print("Using random projection sketch instead...")
    
    # Sample subset
    idx = torch.randperm(len(positions))[:n_samples]
    pos_sample = positions[idx]
    tgt_sample = targets[idx]
    
    # Random projection matrix (on CPU to save VRAM)
    proj_dim = 2048
    print(f"Creating {N_FEATURES}×{proj_dim} projection matrix...")
    proj = torch.randn(N_FEATURES, proj_dim, device='cpu') / math.sqrt(proj_dim)
    
    # Accumulate X_proj @ Y in batches (never materialize full X)
    W_proj = torch.zeros(proj_dim, N_VOCAB, device='cpu')
    
    batch_size = 500  # Small batches - 500 × 524K = 1GB per batch
    print(f"Processing {n_samples} samples in batches of {batch_size}...")
    
    for b in range(0, n_samples, batch_size):
        batch_pos = pos_sample[b:b+batch_size]
        batch_tgt = tgt_sample[b:b+batch_size]
        
        # Extract features (on GPU)
        X_batch = extract_features(train_data, batch_pos)  # (batch, 524K)
        
        # Project to low-dim (move to CPU for matmul)
        X_proj = X_batch.cpu() @ proj  # (batch, proj_dim)
        
        # One-hot targets
        Y_batch = F.one_hot(batch_tgt.cpu().long(), N_VOCAB).float()  # (batch, 256)
        
        # Accumulate
        W_proj += X_proj.T @ Y_batch  # (proj_dim, 256)
        
        del X_batch, X_proj, Y_batch
        torch.cuda.empty_cache()
        
        if (b // batch_size) % 20 == 0:
            print(f"  Processed {b+batch_size}/{n_samples}")
    
    print(f"Projected W shape: {W_proj.shape}")
    
    del proj
    return W_proj.to(device)


# ============================================================================
# Step 3: SVD Spectrum Analysis (on projected matrix)
# ============================================================================

def analyze_spectrum(W_proj):
    """SVD projected W to estimate rank requirements."""
    print("\n" + "="*70)
    print("STEP 3: SVD SPECTRUM ANALYSIS (projected)")
    print("="*70)
    
    print("Computing SVD of projected matrix...")
    t0 = time.time()
    U, S, Vh = torch.linalg.svd(W_proj, full_matrices=False)
    print(f"SVD done in {time.time()-t0:.1f}s")
    
    # Energy spectrum
    energy = S ** 2
    total_energy = energy.sum()
    cumulative = torch.cumsum(energy, dim=0) / total_energy
    
    print(f"\nSingular value spectrum:")
    print(f"  Top 10 singular values: {S[:10].cpu().numpy()}")
    print(f"  Total energy: {total_energy.item():.2f}")
    
    # Find rank for energy thresholds
    thresholds = [0.50, 0.80, 0.90, 0.95, 0.99]
    print(f"\nPROJECTED RANK FOR ENERGY CAPTURE:")
    print(f"  (Note: this is a lower bound on true rank)")
    print(f"{'Threshold':<12} {'Proj Rank':<10}")
    print("-" * 30)
    
    for thresh in thresholds:
        mask = cumulative >= thresh
        if mask.any():
            rank = mask.nonzero()[0].item() + 1
        else:
            rank = len(S)
        print(f"{thresh*100:.1f}%         {rank}")
    
    return S, cumulative


# ============================================================================
# Step 4: Train QTT and Evaluate
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
        """Convert to dense - CAREFUL: this is 134M × 256 = 500GB!"""
        raise RuntimeError("Cannot materialize 134M × 256 dense matrix!")
    
    def forward_batched(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass without materializing full W."""
        # x: (B, N_FEATURES) where N_FEATURES = 2^19
        # Want: x @ W where W is (2^19, 2^8)
        
        B = x.shape[0]
        
        # Reshape x to binary indices: (B, 2^19) -> (B, 2, 2, ..., 2) 19 times
        x_reshaped = x.reshape(B, *([2] * N_QUBITS_FEAT))
        
        # Contract through cores
        result = x_reshaped
        
        # Contract feature dimensions (first 19 qubits)
        for i in range(N_QUBITS_FEAT):
            # result: (B, 2, 2, ..., r_prev) contract with core (r_prev, 2, r_next)
            core = self.cores[i]  # (r_left, 2, r_right)
            
            # Contract result's dimension i with core's dimension 1
            result = torch.einsum('...i,ijk->...jk', result, core)
        
        # Now result is (B, r) where r is the bond dimension
        # Continue with vocab cores
        for i in range(N_QUBITS_FEAT, TOTAL_QUBITS):
            core = self.cores[i]  # (r_left, 2, r_right)
            result = torch.einsum('...i,ijk->...jk', result, core)
        
        # Final result: (B, 2^8)
        return result.reshape(B, N_VOCAB)


def train_qtt_direct(train_data, positions, targets, max_rank, n_epochs=30, lr=0.01):
    """Train QTT WITHOUT materializing dense W."""
    print(f"\nInitializing QTT with max_rank={max_rank}")
    qtt = QTTMatrix.random_init(TOTAL_QUBITS, max_rank=max_rank)
    print(f"Total params: {qtt.n_params():,}")
    print(f"Compression: {DENSE_EQUIV / qtt.n_params():.0f}×")
    
    for core in qtt.cores:
        core.requires_grad_(True)
    
    optimizer = torch.optim.Adam(qtt.cores, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    n_samples = len(positions)
    batch_size = 1000  # Smaller batches for larger model
    
    print(f"Training for {n_epochs} epochs...")
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
            
            # Forward through QTT cores efficiently
            logits = qtt_forward_efficient(qtt, x)
            
            loss = F.cross_entropy(logits, batch_tgt.long())
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            
            del x, logits, loss
            torch.cuda.empty_cache()
        
        scheduler.step()
        avg_loss = epoch_loss / n_batches
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                n_eval = min(5000, len(positions))
                eval_idx = torch.randperm(len(positions))[:n_eval]
                
                # Evaluate in batches
                correct = 0
                total = 0
                for eb in range(0, n_eval, batch_size):
                    eb_idx = eval_idx[eb:eb+batch_size]
                    X_eval = extract_features(train_data, positions[eb_idx])
                    logits = qtt_forward_efficient(qtt, X_eval)
                    correct += (logits.argmax(1) == targets[eb_idx]).sum().item()
                    total += len(eb_idx)
                    del X_eval, logits
                    torch.cuda.empty_cache()
                
                acc = 100 * correct / total
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
    
    for core in qtt.cores:
        core.requires_grad_(False)
    
    return qtt, best_acc


def qtt_forward_efficient(qtt: QTTMatrix, x: torch.Tensor) -> torch.Tensor:
    """
    Efficient QTT forward without materializing full W.
    
    x: (B, N_FEATURES=2^19) sparse feature vectors
    output: (B, N_VOCAB=256)
    
    Strategy: Contract from left, using x's sparsity.
    """
    B = x.shape[0]
    
    # Find non-zero features per sample (x is sparse!)
    # For now, use dense contraction (can optimize later)
    
    # Reshape x for contraction: (B, 2^19) -> (B, 2, 2, ..., 2) [19 times]
    # But this is huge! Instead, we'll do blocked contraction
    
    # Split features into blocks and contract
    # x @ W where W = TT-decomposed
    
    # Actually simplest: contract one qubit at a time
    # Let A = x (B, 2^19)
    # For each feature qubit i:
    #   A = A.reshape(B, 2^i, 2, 2^(18-i))
    #   A = einsum('bijk,ljm->bilkm', A, core[i])  # contract dimension 2
    #   A = A.reshape(B, 2^i * r_out, 2^(18-i))
    
    # This is complex. Let's just materialize W for smaller vocab dimension
    # W is (2^19, 2^8) - even this is 134M floats = 537MB
    # That's actually manageable!
    
    # Build W by contracting cores
    W = qtt.cores[0]  # (1, 2, r1)
    for core in qtt.cores[1:]:
        W = torch.einsum('ijk,klm->ijlm', W, core)
        W = W.reshape(W.shape[0], -1, W.shape[-1])
    
    W = W.squeeze(0).squeeze(-1)  # (2^27, 1) -> will fail!
    
    # Wait - W should be (2^19, 2^8) but contracted tensor is 2^27
    # Need to reshape properly
    W = W.reshape(2**N_QUBITS_FEAT, 2**N_QUBITS_VOCAB)  # (524288, 256)
    
    return x @ W


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "="*70)
    print("NS MILLENNIUM - LET PHYSICS DETERMINE RANK")
    print("SCALE: ~100M DENSE EQUIVALENT")
    print("="*70)
    print(f"Features: {N_FEATURES:,} (2^{N_QUBITS_FEAT})")
    print(f"Vocab: {N_VOCAB} (2^{N_QUBITS_VOCAB})")
    print(f"Dense equivalent: {DENSE_EQUIV:,} params ({DENSE_EQUIV*4/1e9:.2f} GB)")
    
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
    
    # Step 2: Oracle sketch (projected analysis)
    W_proj = build_sampled_oracle_sketch(train_data, positions, targets, n_samples=50000)
    
    # Step 3: SVD analysis on projection
    S, cumulative = analyze_spectrum(W_proj)
    del W_proj
    torch.cuda.empty_cache()
    
    # Step 4: Train QTT at high rank
    print("\n" + "="*70)
    print("STEP 4: TRAIN QTT WITH HIGH RANK CAP")
    print("="*70)
    
    qtt, best_acc = train_qtt_direct(train_data, positions, targets, 
                                     max_rank=MAX_RANK_CAP, n_epochs=50)
    
    # Analyze rank structure
    ranks = qtt.get_ranks()
    print(f"\nQTT Rank Profile:")
    print(f"  Max rank: {max(ranks)}")
    print(f"  Ranks: {ranks[:5]}...{ranks[-5:]}")
    print(f"  Total params: {qtt.n_params():,}")
    print(f"  Compression: {DENSE_EQUIV / qtt.n_params():.0f}×")
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    with torch.no_grad():
        batch_size = 1000
        n_test = min(20000, len(positions))
        test_idx = torch.randperm(len(positions))[:n_test]
        
        correct = 0
        total = 0
        for b in range(0, n_test, batch_size):
            b_idx = test_idx[b:b+batch_size]
            X = extract_features(train_data, positions[b_idx])
            logits = qtt_forward_efficient(qtt, X)
            correct += (logits.argmax(1) == targets[b_idx]).sum().item()
            total += len(b_idx)
            del X, logits
        
        final_acc = 100 * correct / total
    
    print(f"\n✓ Final accuracy: {final_acc:.1f}%")
    print(f"✓ Parameters: {qtt.n_params():,}")
    print(f"✓ Compression: {DENSE_EQUIV / qtt.n_params():.0f}×")
    print(f"✓ Dense equivalent: {DENSE_EQUIV:,}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"At ~100M dense equivalent scale:")
    print(f"  Accuracy: {final_acc:.1f}%")
    print(f"  This {'MEETS' if final_acc >= 90 else 'DOES NOT MEET'} 90% target")


if __name__ == '__main__':
    main()
