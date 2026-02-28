"""
Minimal QTT Core Structures for Self-Contained Workflow

Extracted from ontic.cfd for standalone execution.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch import Tensor


# ═══════════════════════════════════════════════════════════════════════════════════════
# RANDOMIZED SVD
# ═══════════════════════════════════════════════════════════════════════════════════════

def rsvd(
    A: Tensor,
    rank: int,
    oversampling: int = 10,
    n_iter: int = 2,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Randomized SVD for O(mnk) complexity."""
    m, n = A.shape
    k = min(rank, min(m, n))
    l = min(k + oversampling, min(m, n))
    
    device = A.device
    dtype = A.dtype
    
    Omega = torch.randn(n, l, device=device, dtype=dtype)
    Y = A @ Omega
    
    for _ in range(n_iter):
        Q_temp, _ = torch.linalg.qr(Y)
        Y = A @ (A.T @ Q_temp)
    
    Q, _ = torch.linalg.qr(Y)
    B = Q.T @ A
    
    try:
        U_small, S, Vh = torch.linalg.svd(B, full_matrices=False)
    except RuntimeError:
        k = min(k, B.shape[0], B.shape[1])
        U_small = torch.eye(B.shape[0], k, device=device, dtype=dtype)
        S = torch.ones(k, device=device, dtype=dtype) * torch.norm(B).item() / k
        Vh = torch.eye(k, B.shape[1], device=device, dtype=dtype)
    
    U = Q @ U_small
    return U[:, :k], S[:k], Vh[:k, :]


def rsvd_truncate(
    A: Tensor,
    max_rank: int,
    tol: float = 1e-10,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Adaptive SVD with tolerance-based truncation."""
    m, n = A.shape
    
    if m * n > 512 * 512 or min(m, n) > 512:
        U, S, Vh = rsvd(A, max_rank)
    else:
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        k = min(len(S), max_rank)
        U, S, Vh = U[:, :k], S[:k], Vh[:k, :]
    
    if tol > 0 and len(S) > 0 and S[0] > 0:
        rel_s = S / S[0]
        valid = rel_s > tol
        k = valid.sum().item()
        k = max(1, min(k, max_rank, len(S)))
        return U[:, :k], S[:k], Vh[:k, :]
    
    return U, S, Vh


# ═══════════════════════════════════════════════════════════════════════════════════════
# QTT CORES
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class QTTCores:
    """
    Native QTT representation.
    cores[k] has shape (r_{k-1}, 2, r_k)
    """
    cores: List[Tensor]
    
    @property
    def num_sites(self) -> int:
        return len(self.cores)
    
    @property
    def ranks(self) -> List[int]:
        ranks = [1]
        for c in self.cores:
            ranks.append(c.shape[2])
        return ranks
    
    @property
    def max_rank(self) -> int:
        return max(self.ranks)
    
    @property
    def mean_rank(self) -> float:
        return np.mean(self.ranks[1:-1]) if len(self.ranks) > 2 else 1.0
    
    @property
    def total_params(self) -> int:
        return sum(c.numel() for c in self.cores)
    
    @property
    def device(self) -> torch.device:
        return self.cores[0].device if self.cores else torch.device('cpu')
    
    @property
    def dtype(self) -> torch.dtype:
        return self.cores[0].dtype if self.cores else torch.float32
    
    def clone(self) -> 'QTTCores':
        return QTTCores([c.clone() for c in self.cores])
    
    def to(self, device: torch.device) -> 'QTTCores':
        return QTTCores([c.to(device) for c in self.cores])


# ═══════════════════════════════════════════════════════════════════════════════════════
# QTT TRUNCATION SWEEP
# ═══════════════════════════════════════════════════════════════════════════════════════

def qtt_truncate_sweep(
    cores: List[Tensor],
    max_rank: int,
    tol: float = 1e-10,
    use_rsvd: bool = True,
    use_canonical: bool = True,
) -> List[Tensor]:
    """Canonical TT-round: QR sweep + SVD sweep."""
    L = len(cores)
    if L == 0:
        return cores
    
    new_cores = [c.clone() for c in cores]
    
    if use_canonical:
        for k in range(L - 1):
            core = new_cores[k]
            r_left, d, r_right = core.shape
            mat = core.reshape(r_left * d, r_right)
            Q, R = torch.linalg.qr(mat)
            r_new = Q.shape[1]
            new_cores[k] = Q.reshape(r_left, d, r_new)
            next_core = new_cores[k + 1]
            new_cores[k + 1] = torch.einsum('ij,jkl->ikl', R, next_core)
    
    for k in range(L - 1, 0, -1):
        core = new_cores[k]
        r_left, d, r_right = core.shape
        mat = core.reshape(r_left, d * r_right)
        
        U, S, Vh = rsvd_truncate(mat, max_rank, tol)
        r_new = len(S)
        
        new_cores[k] = Vh.reshape(r_new, d, r_right)
        
        prev_core = new_cores[k - 1]
        r_prev_l, d_prev, r_prev_r = prev_core.shape
        prev_mat = prev_core.reshape(r_prev_l * d_prev, r_prev_r)
        prev_mat = prev_mat @ (U * S)
        new_cores[k - 1] = prev_mat.reshape(r_prev_l, d_prev, r_new)
    
    return new_cores


# ═══════════════════════════════════════════════════════════════════════════════════════
# 3D QTT STATE
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class QTT3DNative:
    """3D QTT field - Morton interleaved over 3n qubits."""
    cores: QTTCores
    n_bits: int
    
    @property
    def N(self) -> int:
        return 1 << self.n_bits
    
    @property
    def total_qubits(self) -> int:
        return 3 * self.n_bits
    
    @property
    def max_rank(self) -> int:
        return self.cores.max_rank
    
    @property
    def mean_rank(self) -> float:
        return self.cores.mean_rank
    
    @property
    def device(self) -> torch.device:
        return self.cores.device
    
    @property
    def dtype(self) -> torch.dtype:
        return self.cores.dtype
    
    @property
    def compression_ratio(self) -> float:
        dense_params = self.N ** 3
        qtt_params = self.cores.total_params
        return dense_params / qtt_params if qtt_params > 0 else float('inf')
    
    def clone(self) -> 'QTT3DNative':
        return QTT3DNative(self.cores.clone(), self.n_bits)
    
    def to(self, device: torch.device) -> 'QTT3DNative':
        return QTT3DNative(self.cores.to(device), self.n_bits)


@dataclass
class QTT3DVectorNative:
    """3D vector field - three QTT3DNative components."""
    x: QTT3DNative
    y: QTT3DNative
    z: QTT3DNative
    
    @property
    def n_bits(self) -> int:
        return self.x.n_bits
    
    @property
    def max_rank(self) -> int:
        return max(self.x.max_rank, self.y.max_rank, self.z.max_rank)
    
    @property
    def mean_rank(self) -> float:
        return (self.x.mean_rank + self.y.mean_rank + self.z.mean_rank) / 3
    
    @property
    def compression_ratio(self) -> float:
        total_dense = 3 * self.x.N ** 3
        total_qtt = (self.x.cores.total_params + 
                     self.y.cores.total_params + 
                     self.z.cores.total_params)
        return total_dense / total_qtt if total_qtt > 0 else float('inf')
    
    def clone(self) -> 'QTT3DVectorNative':
        return QTT3DVectorNative(self.x.clone(), self.y.clone(), self.z.clone())
