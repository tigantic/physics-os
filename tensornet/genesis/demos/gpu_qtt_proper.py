#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                          ║
║                G P U   Q T T   -   P R O P E R   I M P L E M E N T A T I O N            ║
║                                                                                          ║
║      For wide matrices (m << n): compute SVD via A @ A.T eigendecomposition             ║
║      This avoids cusolver limitations on extremely wide matrices                        ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import time
import math
import gc
import sys
from typing import List

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n✓ Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def format_bytes(b: float) -> str:
    if b >= 1e12: return f"{b/1e12:.2f} TB"
    elif b >= 1e9: return f"{b/1e9:.2f} GB"
    elif b >= 1e6: return f"{b/1e6:.2f} MB"
    elif b >= 1e3: return f"{b/1e3:.2f} KB"
    return f"{b:.0f} B"


def svd_wide(A: torch.Tensor, k: int):
    """
    SVD for wide matrices (m << n) where cusolver fails.
    
    For A of shape (m, n) with m << n:
    - Compute Gram matrix G = A @ A.T  (m × m, small!)
    - Eigendecompose G = U @ Λ @ U.T
    - Singular values: S = sqrt(Λ)
    - Right singular vectors: V = A.T @ U @ diag(1/S)
    
    Memory: O(m²) instead of O(mn)
    """
    m, n = A.shape
    
    if m >= n:
        # Tall or square - use standard SVD
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        return U[:, :k], S[:k], Vh[:k, :]
    
    # Wide matrix: use Gram matrix
    # G = A @ A.T is only (m × m)
    G = A @ A.T
    
    # Eigendecomposition (symmetric positive semi-definite)
    eigenvalues, U = torch.linalg.eigh(G)
    
    # Sort descending (eigh returns ascending)
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    U = U[:, idx]
    
    # Singular values = sqrt of eigenvalues
    # Clamp to avoid sqrt of tiny negatives
    S = torch.sqrt(torch.clamp(eigenvalues, min=0))
    
    # Truncate to k
    k = min(k, m, (S > 1e-14).sum().item())
    k = max(1, k)
    
    U = U[:, :k]
    S = S[:k]
    
    # Compute V = A.T @ U @ diag(1/S)
    # V.T = diag(1/S) @ U.T @ A
    inv_S = 1.0 / torch.clamp(S, min=1e-14)
    Vh = (inv_S.unsqueeze(1)) * (U.T @ A)
    
    return U, S, Vh


def tt_svd_gpu(tensor: torch.Tensor, max_rank: int = 64, tol: float = 1e-10) -> List[torch.Tensor]:
    """
    TT-SVD on GPU using wide-matrix-safe SVD.
    """
    n = tensor.numel()
    d = int(math.log2(n))
    assert 2**d == n, f"Length must be power of 2"
    
    cores = []
    C = tensor.reshape(1, -1)  # (1, 2^d)
    
    for k in range(d - 1):
        m = C.shape[0] * 2
        n_cols = C.shape[1] // 2
        C = C.reshape(m, n_cols)
        
        # Use wide-safe SVD
        U, S, Vh = svd_wide(C, k=max_rank)
        
        # Truncate by tolerance
        rank = min(max_rank, len(S))
        if len(S) > 1 and S[0] > 0:
            cutoff = (S > tol * S[0]).sum().item()
            rank = min(rank, max(1, cutoff))
        
        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]
        
        # Store core
        core = U.reshape(-1, 2, rank)
        cores.append(core)
        
        # Continue
        C = torch.diag(S) @ Vh
    
    # Last core
    core = C.reshape(-1, 2, 1)
    cores.append(core)
    
    return cores


def compress_test(n_bits: int, device=DEVICE):
    """Test compression at scale."""
    n = 2 ** n_bits
    print(f"\n  2^{n_bits} = {n:,} elements ({format_bytes(n * 4)})")
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    # Generate structured data on device
    t0 = time.perf_counter()
    x = torch.linspace(0, 2 * math.pi, n, device=device, dtype=torch.float32)
    signal = 20 * torch.cos(x) + 5 * torch.sin(3 * x) + 2 * torch.sin(7 * x)
    del x
    t_gen = time.perf_counter() - t0
    
    # Compress
    t0 = time.perf_counter()
    cores = tt_svd_gpu(signal, max_rank=64)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_compress = time.perf_counter() - t0
    
    # Stats
    compressed = sum(c.numel() * c.element_size() for c in cores)
    original = n * 4
    ratio = original / compressed
    max_rank = max(c.shape[-1] for c in cores)
    
    if torch.cuda.is_available():
        vram = torch.cuda.max_memory_allocated() / 1e9
        print(f"    → {format_bytes(compressed)} | {ratio:,.0f}x | rank={max_rank} | {t_compress:.2f}s | {vram:.2f}GB VRAM")
    else:
        print(f"    → {format_bytes(compressed)} | {ratio:,.0f}x | rank={max_rank} | {t_compress:.2f}s")
    
    del signal, cores
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return ratio, max_rank


def main():
    print()
    print("═══════════════════════════════════════════════════════════════════════════════")
    print("          GPU QTT with Wide-Matrix-Safe SVD (Gram Matrix Method)")
    print("═══════════════════════════════════════════════════════════════════════════════")
    print()
    print("  For (m × n) with m << n:")
    print("    - Compute G = A @ A.T  → O(m²) memory")
    print("    - Eigendecompose G     → O(m³) time")
    print("    - Recover V from U, S  → O(mk) memory")
    print()
    
    max_bits = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    
    for n_bits in range(20, max_bits + 1, 2):
        try:
            ratio, rank = compress_test(n_bits)
        except torch.cuda.OutOfMemoryError:
            print(f"\n  OOM at 2^{n_bits}")
            break
        except Exception as e:
            print(f"\n  Error at 2^{n_bits}: {e}")
            break
    
    print()
    print("═══════════════════════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
