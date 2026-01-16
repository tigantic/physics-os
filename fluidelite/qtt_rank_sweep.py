"""
QTT Rank Sweep: Proper experimental design following NS Millennium methodology.

Key insight from NS framework: Don't fix rank, study how it scales!
- Vary ONE parameter at a time
- Sweep ranks for each feature dimension
- Track rank evolution during training

See: docs/architecture/NS_MILLENNIUM_FRAMEWORK.md
"""
import torch
import triton
import triton.language as tl
import time
import math
import csv
from typing import List, Tuple
from dataclasses import dataclass

device = torch.device('cuda')

# Constants
CTX_LEN = 16
N_VOCAB = 256

print(f"GPU: {torch.cuda.get_device_name(0)}")


# === Triton kernel for 16K features ===
@triton.jit
def extract_features_16k_kernel(data_ptr, positions_ptr, features_ptr, stride_feat):
    """16K = 1024 unigram + 4096 bigram + 4096 trigram + 4096 skip + 3072 gap."""
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


def extract_features_16k(data: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    B = positions.shape[0]
    features = torch.zeros((B, 16384), dtype=torch.float32, device=device)
    extract_features_16k_kernel[(B,)](data, positions, features, 16384)
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


def init_qtt_uniform(n_qubits: int, max_rank: int) -> List[torch.Tensor]:
    """Initialize QTT cores with uniform max rank and Xavier init."""
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


def init_qtt_hourglass(n_feat_qubits: int, n_vocab_qubits: int, peak_rank: int) -> List[torch.Tensor]:
    """
    Initialize QTT cores with hourglass rank structure.
    Rank grows to peak_rank in the middle, then decreases.
    """
    n_qubits = n_feat_qubits + n_vocab_qubits
    cores = []
    r_left = 1
    
    for i in range(n_qubits):
        # Hourglass: rank peaks at boundary between features and vocab
        dist_from_boundary = abs(i - n_feat_qubits)
        max_at_position = peak_rank // (2 ** max(0, dist_from_boundary - 3))
        max_at_position = max(2, max_at_position)
        
        # Also respect QTT dimension constraints
        r_right = min(max_at_position, 2 ** min(i + 1, n_qubits - i - 1)) if i < n_qubits - 1 else 1
        
        std = math.sqrt(2.0 / (r_left + r_right))
        core = torch.randn(r_left, 2, r_right, device=device) * std
        core.requires_grad = True
        cores.append(core)
        r_left = r_right
    
    return cores


def get_rank_profile(cores: List[torch.Tensor]) -> List[int]:
    """Get the rank at each bond."""
    return [c.shape[2] for c in cores]


@dataclass
class ExperimentResult:
    n_features: int
    max_rank: int
    rank_type: str
    accuracy: float
    perplexity: float
    n_params: int
    compression: float
    train_time: float
    final_loss: float
    rank_profile: List[int]


def run_experiment(
    train_data: torch.Tensor,
    test_data: torch.Tensor,
    n_feat_qubits: int,
    max_rank: int,
    rank_type: str = "uniform"
) -> ExperimentResult:
    """Run a single QTT training experiment."""
    
    n_vocab_qubits = 8  # 256 tokens
    n_qubits = n_feat_qubits + n_vocab_qubits
    n_features = 2 ** n_feat_qubits
    
    print(f"\n--- Experiment: {n_features} features, rank={max_rank}, type={rank_type} ---")
    
    # Initialize cores
    if rank_type == "uniform":
        cores = init_qtt_uniform(n_qubits, max_rank)
    elif rank_type == "hourglass":
        cores = init_qtt_hourglass(n_feat_qubits, n_vocab_qubits, max_rank)
    else:
        cores = init_qtt_uniform(n_qubits, max_rank)
    
    rank_profile = get_rank_profile(cores)
    n_params = sum(c.numel() for c in cores)
    compression = (n_features * N_VOCAB) / n_params
    
    print(f"  Params: {n_params:,}, Compression: {compression:.1f}x")
    print(f"  Rank profile: {rank_profile[:5]}...{rank_profile[-5:]}")
    
    # Training data
    n_train = len(train_data) - CTX_LEN - 1
    MAX_SAMPLES = 200000
    step = max(1, n_train // MAX_SAMPLES)
    positions = torch.arange(0, n_train, step, dtype=torch.int64, device=device)[:MAX_SAMPLES]
    targets = train_data[positions + CTX_LEN]
    
    # Optimizer
    optimizer = torch.optim.Adam(cores, lr=0.03)
    
    BATCH = 4096
    N_EPOCHS = 10
    
    start = time.perf_counter()
    final_loss = 0.0
    
    for epoch in range(N_EPOCHS):
        epoch_loss = 0.0
        n_batches = 0
        perm = torch.randperm(len(positions), device=device)
        
        for b in range(0, len(positions), BATCH):
            idx = perm[b:b+BATCH]
            batch_pos = positions[idx]
            batch_tgt = targets[idx].long()
            
            X = extract_features_16k(train_data, batch_pos)
            W = qtt_to_dense(cores, n_feat_qubits)
            logits = X @ W
            
            loss = torch.nn.functional.cross_entropy(logits, batch_tgt)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        final_loss = epoch_loss / n_batches
        print(f"  Epoch {epoch+1}/{N_EPOCHS}: loss = {final_loss:.4f}")
    
    train_time = time.perf_counter() - start
    
    # Evaluation
    cores = [c.detach() for c in cores]
    
    n_test = len(test_data) - CTX_LEN - 1
    TEST_N = 20000
    test_pos = torch.arange(0, n_test, n_test//TEST_N, dtype=torch.int64, device=device)[:TEST_N]
    test_actual = test_data[test_pos + CTX_LEN]
    X_test = extract_features_16k(test_data, test_pos)
    
    W = qtt_to_dense(cores, n_feat_qubits)
    logits = X_test @ W
    
    probs = torch.softmax(logits, dim=1)
    accuracy = (probs.argmax(dim=1) == test_actual).float().mean().item()
    perplexity = math.exp(torch.nn.functional.cross_entropy(logits, test_actual.long()).item())
    
    print(f"  Results: acc={accuracy*100:.1f}%, ppl={perplexity:.2f}")
    
    return ExperimentResult(
        n_features=n_features,
        max_rank=max_rank,
        rank_type=rank_type,
        accuracy=accuracy,
        perplexity=perplexity,
        n_params=n_params,
        compression=compression,
        train_time=train_time,
        final_loss=final_loss,
        rank_profile=rank_profile
    )


def main():
    from datasets import load_dataset
    
    print("="*70)
    print("QTT RANK SWEEP: Following NS Millennium Methodology")
    print("="*70)
    print("\nPrinciple: Vary ONE parameter at a time, but SWEEP ranks for each.")
    
    # Load data
    print("\nLoading WikiText-103...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    train_text = "\n".join(dataset["train"]["text"]).encode('utf-8')
    test_text = "\n".join(dataset["test"]["text"]).encode('utf-8')
    
    train_data = torch.tensor(list(train_text), dtype=torch.int32, device=device)
    test_data = torch.tensor(list(test_text), dtype=torch.int32, device=device)
    
    # Experiment matrix
    # Fix features at 16K (14 qubits), sweep ranks
    experiments = [
        # (n_feat_qubits, max_rank, rank_type)
        (14, 16, "uniform"),
        (14, 32, "uniform"),
        (14, 64, "uniform"),
        (14, 128, "uniform"),
        # Hourglass comparison at best uniform rank
        (14, 64, "hourglass"),
    ]
    
    results = []
    
    for n_feat_qubits, max_rank, rank_type in experiments:
        try:
            result = run_experiment(train_data, test_data, n_feat_qubits, max_rank, rank_type)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Summary
    print("\n" + "="*70)
    print("RANK SWEEP RESULTS")
    print("="*70)
    print(f"{'Features':<10} {'Rank':<8} {'Type':<12} {'Acc':<8} {'PPL':<8} {'Params':<10} {'Comp':<8}")
    print("-"*70)
    
    for r in results:
        print(f"{r.n_features:<10} {r.max_rank:<8} {r.rank_type:<12} {r.accuracy*100:>5.1f}% {r.perplexity:>7.2f} {r.n_params:>10,} {r.compression:>6.1f}x")
    
    # Save to CSV
    with open("rank_sweep_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["features", "max_rank", "rank_type", "accuracy", "perplexity", "params", "compression", "train_time", "final_loss"])
        for r in results:
            writer.writerow([r.n_features, r.max_rank, r.rank_type, r.accuracy, r.perplexity, r.n_params, r.compression, r.train_time, r.final_loss])
    
    print(f"\nResults saved to rank_sweep_results.csv")
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS: Does rank saturate?")
    print("="*70)
    
    uniform_results = [r for r in results if r.rank_type == "uniform"]
    if len(uniform_results) >= 2:
        # Check if accuracy plateaus
        ranks = [r.max_rank for r in uniform_results]
        accs = [r.accuracy for r in uniform_results]
        
        # Find saturation point
        best_idx = accs.index(max(accs))
        best_rank = ranks[best_idx]
        
        print(f"Best accuracy at rank={best_rank}: {max(accs)*100:.1f}%")
        
        if best_idx < len(ranks) - 1:
            print(f"Rank {ranks[best_idx+1]} shows diminishing returns")
            print("→ Rank is SATURATING (NS Millennium pattern confirmed)")
        else:
            print("→ Higher ranks may help (need more sweeps)")


if __name__ == "__main__":
    main()
