"""
QTT-NTT: Tensor Train Accelerated Number Theoretic Transform
============================================================

Plugs into The Ontic Engine's existing QTT infrastructure for ZK proving acceleration.

Key insight: NTT butterfly is an MPO with special structure.
We use existing qtt_native_ops.apply_mpo_cuda for the heavy lifting.

Usage:
    from ontic.cuda.qtt_ntt import QTTNTT, qtt_ntt_forward, qtt_ntt_inverse
    
    # For complex DFT
    ntt = QTTNTT(n_bits=10)  # N = 1024
    X = ntt.forward(x)
    x_recovered = ntt.inverse(X)
    
    # For finite field NTT (ZK proving)
    ntt = QTTNTT(n_bits=20, field='goldilocks')  # Plonky2
    ntt = QTTNTT(n_bits=20, field='bn254')       # Plonk

Author: Brad / Tigantic Holdings
Date: 2026-01-19
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import List, Tuple, Optional, Literal
from dataclasses import dataclass

# Import existing QTT infrastructure
try:
    from ontic.cuda.qtt_native_ops import (
        apply_mpo_cuda,
        qtt_hadamard_cuda,
        _flatten_cores,
        _unflatten_cores,
        is_cuda_available,
    )
    from ontic.core.decompositions import svd_truncated
    HAS_TENSORNET = True
except ImportError:
    HAS_TENSORNET = False
    print("[QTT-NTT] Warning: ontic not found, using standalone mode")


# =============================================================================
# FINITE FIELD CONSTANTS
# =============================================================================

# Goldilocks field (used by Plonky2, Polygon zkEVM)
# p = 2^64 - 2^32 + 1
GOLDILOCKS_PRIME = (1 << 64) - (1 << 32) + 1
GOLDILOCKS_GENERATOR = 7  # Primitive element

# BN254 scalar field (used by Plonk, Groth16)
# p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
BN254_PRIME = 21888242871839275222246405745257275088548364400416034343698204186575808495617
BN254_GENERATOR = 5

# Baby Bear field (used by RISC Zero)
BABY_BEAR_PRIME = (1 << 31) - (1 << 27) + 1
BABY_BEAR_GENERATOR = 31


@dataclass
class FieldParams:
    """Parameters for a finite field."""
    prime: int
    generator: int
    name: str
    bits: int  # Bits needed to represent elements
    

FIELD_PARAMS = {
    'goldilocks': FieldParams(GOLDILOCKS_PRIME, GOLDILOCKS_GENERATOR, 'Goldilocks', 64),
    'bn254': FieldParams(BN254_PRIME, BN254_GENERATOR, 'BN254', 254),
    'babybear': FieldParams(BABY_BEAR_PRIME, BABY_BEAR_GENERATOR, 'Baby Bear', 31),
    'complex': FieldParams(0, 0, 'Complex', 0),  # Standard complex FFT
}


# =============================================================================
# FINITE FIELD ARITHMETIC
# =============================================================================

def find_primitive_root(p: int, n: int) -> int:
    """
    Find primitive n-th root of unity in F_p.
    
    Requires: n | (p - 1)
    """
    assert (p - 1) % n == 0, f"n={n} does not divide p-1={p-1}"
    
    # g^((p-1)/n) is an n-th root of unity
    # Need to find g such that this root is primitive (has order exactly n)
    
    g = 2
    while g < p:
        omega = pow(g, (p - 1) // n, p)
        # Check if omega is primitive: omega^(n/q) != 1 for all prime divisors q of n
        is_primitive = True
        
        # For n = 2^k, only need to check omega^(n/2) != 1
        if pow(omega, n // 2, p) == 1:
            is_primitive = False
        
        if is_primitive and pow(omega, n, p) == 1:
            return omega
        g += 1
    
    raise ValueError(f"No primitive {n}-th root found in F_{p}")


def montgomery_reduce(x: int, p: int, r: int, p_inv: int) -> int:
    """
    Montgomery reduction: compute x * R^{-1} mod p.
    
    Args:
        x: Value to reduce (must be < p * R)
        p: Prime modulus
        r: Montgomery R = 2^k where k = bits(p)
        p_inv: -p^{-1} mod R
    """
    # m = (x mod R) * p_inv mod R
    m = ((x & (r - 1)) * p_inv) & (r - 1)
    # t = (x + m * p) / R
    t = (x + m * p) >> r.bit_length() - 1
    # Conditional subtraction
    return t if t < p else t - p


# =============================================================================
# BUTTERFLY OPERATIONS
# =============================================================================

def build_butterfly_mpo(
    n_bits: int,
    stage: int,
    omega: complex | int,
    field: str = 'complex',
    device: torch.device = None,
    dtype: torch.dtype = torch.complex128,
) -> List[Tensor]:
    """
    Build butterfly operator as MPO for stage k of FFT.
    
    The butterfly at stage k operates on bit position k:
    - Pairs elements 2^k apart
    - Applies [[1, 1], [1, -omega^j]] where j depends on position
    
    In MPO form, this is a tensor product of identities and one non-trivial 2x2.
    
    Args:
        n_bits: Total bits (N = 2^n_bits)
        stage: Which butterfly stage (0 to n_bits-1)
        omega: Primitive N-th root of unity
        field: 'complex' or finite field name
        device: torch device
        dtype: Data type
    
    Returns:
        List of MPO cores, each shape (r_left, d_out, d_in, r_right)
    """
    N = 2 ** n_bits
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    cores = []
    
    for k in range(n_bits):
        if k == stage:
            # Non-trivial butterfly core
            # Shape: (1, 2, 2, 2) or (2, 2, 2, 1) depending on position
            
            # The butterfly matrix for this stage
            # B[out_bit, in_bit, twiddle_idx] where twiddle_idx encodes position
            
            # For now, simplified: identity with twiddle on the edge
            # Full implementation needs position-dependent twiddles
            
            if k == 0:
                # First core: (1, 2, 2, 2)
                core = torch.zeros(1, 2, 2, 2, dtype=dtype, device=device)
                # out=0: [1, 1] (sum)
                core[0, 0, 0, 0] = 1.0
                core[0, 0, 1, 1] = 1.0
                # out=1: [1, -1] (difference) - twiddle applied later
                core[0, 1, 0, 0] = 1.0
                core[0, 1, 1, 1] = -1.0
            elif k == n_bits - 1:
                # Last core: (2, 2, 2, 1)
                core = torch.zeros(2, 2, 2, 1, dtype=dtype, device=device)
                # Similar structure
                core[0, 0, 0, 0] = 1.0
                core[1, 0, 1, 0] = 1.0
                core[0, 1, 0, 0] = 1.0
                core[1, 1, 1, 0] = -1.0
            else:
                # Middle core: (2, 2, 2, 2)
                core = torch.zeros(2, 2, 2, 2, dtype=dtype, device=device)
                # Pass through with butterfly structure
                core[0, 0, 0, 0] = 1.0
                core[1, 0, 1, 1] = 1.0
                core[0, 1, 0, 0] = 1.0
                core[1, 1, 1, 1] = -1.0
        else:
            # Identity core at position k
            if k == 0:
                # First: (1, 2, 2, 1)
                core = torch.zeros(1, 2, 2, 1, dtype=dtype, device=device)
                core[0, 0, 0, 0] = 1.0
                core[0, 1, 1, 0] = 1.0
            elif k == n_bits - 1:
                # Last: (1, 2, 2, 1)
                core = torch.zeros(1, 2, 2, 1, dtype=dtype, device=device)
                core[0, 0, 0, 0] = 1.0
                core[0, 1, 1, 0] = 1.0
            else:
                # Middle: (1, 2, 2, 1)
                core = torch.zeros(1, 2, 2, 1, dtype=dtype, device=device)
                core[0, 0, 0, 0] = 1.0
                core[0, 1, 1, 0] = 1.0
        
        cores.append(core)
    
    return cores


def apply_twiddle_factors(
    qtt_cores: List[Tensor],
    stage: int,
    omega: complex | int,
    field: str = 'complex',
) -> List[Tensor]:
    """
    Apply twiddle factors for butterfly stage.
    
    Twiddle factors are diagonal, so this is element-wise multiplication.
    In QTT form, diagonal operators have rank 1.
    """
    n_bits = len(qtt_cores)
    N = 2 ** n_bits
    
    # Build twiddle QTT
    # Twiddle[j] = omega^(j * stride) where stride = N / (2^(stage+1))
    stride = N >> (stage + 1)
    
    twiddle_cores = []
    for k in range(n_bits):
        r_left = 1 if k == 0 else 2
        r_right = 1 if k == n_bits - 1 else 2
        
        core = torch.zeros(r_left, 2, r_right, dtype=qtt_cores[0].dtype, device=qtt_cores[0].device)
        
        # Twiddle depends on bits above stage
        if k < stage:
            # Identity
            core[0, 0, 0] = 1.0
            core[0, 1, 0] = 1.0 if r_right == 1 else 0.0
            if r_right > 1:
                core[0, 0, 1] = 0.0
                core[0, 1, 1] = 1.0
        else:
            # Twiddle factor
            # This is simplified - full implementation tracks bit position
            core[0, 0, 0] = 1.0
            if k == stage:
                core[0, 1, 0] = complex(omega) if field == 'complex' else float(omega)
            else:
                core[0, 1, 0] = 1.0
        
        twiddle_cores.append(core)
    
    # Element-wise multiply (Hadamard product)
    if HAS_TENSORNET:
        return qtt_hadamard_cuda(qtt_cores, twiddle_cores)
    else:
        # Fallback
        result = []
        for c1, c2 in zip(qtt_cores, twiddle_cores):
            r1L, d, r1R = c1.shape
            r2L, _, r2R = c2.shape
            kron = torch.einsum("adb,cde->acdbe", c1, c2)
            result.append(kron.reshape(r1L * r2L, d, r1R * r2R))
        return result


# =============================================================================
# MAIN QTT-NTT CLASS
# =============================================================================

class QTTNTT:
    """
    QTT-accelerated NTT for ZK proving.
    
    Uses The Ontic Engine's existing QTT infrastructure with NTT-specific optimizations.
    
    Example:
        # Complex FFT
        ntt = QTTNTT(10)  # N = 1024
        X = ntt.forward(x)
        
        # Goldilocks field (Plonky2)
        ntt = QTTNTT(20, field='goldilocks')
        X = ntt.forward_ff(x)  # x is list of field elements
    """
    
    def __init__(
        self,
        n_bits: int,
        field: Literal['complex', 'goldilocks', 'bn254', 'babybear'] = 'complex',
        device: torch.device = None,
    ):
        self.n_bits = n_bits
        self.N = 2 ** n_bits
        self.field = field
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get field parameters
        self.field_params = FIELD_PARAMS[field]
        
        # Compute root of unity
        if field == 'complex':
            self.omega = torch.exp(torch.tensor(-2j * torch.pi / self.N))
            self.omega_inv = torch.exp(torch.tensor(2j * torch.pi / self.N))
            self.dtype = torch.complex128
        else:
            p = self.field_params.prime
            self.omega = find_primitive_root(p, self.N)
            self.omega_inv = pow(self.omega, p - 2, p)  # Modular inverse
            self.dtype = torch.int64  # For finite fields
        
        # Pre-build butterfly MPOs
        self._butterfly_mpos = None  # Lazy init
    
    def _ensure_butterflies(self):
        """Lazy initialization of butterfly MPOs."""
        if self._butterfly_mpos is None:
            self._butterfly_mpos = []
            for stage in range(self.n_bits):
                mpo = build_butterfly_mpo(
                    self.n_bits, stage, self.omega,
                    self.field, self.device, self.dtype
                )
                self._butterfly_mpos.append(mpo)
    
    def _vector_to_qtt(self, x: Tensor, max_rank: int = 64) -> List[Tensor]:
        """
        Convert vector to QTT format.
        
        Uses SVD-based decomposition from ontic.
        """
        assert len(x) == self.N, f"Expected length {self.N}, got {len(x)}"
        
        # Reshape to tensor
        tensor = x.reshape([2] * self.n_bits)
        
        # TT-SVD decomposition
        cores = []
        remainder = tensor.flatten()
        left_rank = 1
        
        for k in range(self.n_bits - 1):
            remainder = remainder.reshape(left_rank * 2, -1)
            
            if HAS_TENSORNET:
                U, S, Vh, info = svd_truncated(remainder, chi_max=max_rank, return_info=True)
            else:
                U, S, Vh = torch.linalg.svd(remainder, full_matrices=False)
                rank = min(max_rank, len(S))
                U, S, Vh = U[:, :rank], S[:rank], Vh[:rank, :]
            
            rank = len(S)
            cores.append(U.reshape(left_rank, 2, rank))
            remainder = torch.diag(S) @ Vh
            left_rank = rank
        
        cores.append(remainder.reshape(left_rank, 2, 1))
        return cores
    
    def _qtt_to_vector(self, cores: List[Tensor]) -> Tensor:
        """Reconstruct vector from QTT cores."""
        result = cores[0]
        for core in cores[1:]:
            result = torch.tensordot(result, core, dims=([-1], [0]))
        return result.flatten()
    
    def forward(self, x: Tensor, max_rank: int = 64) -> Tensor:
        """
        Forward NTT using QTT representation.
        
        Args:
            x: Input vector of length N
            max_rank: Maximum QTT rank (controls accuracy/speed tradeoff)
        
        Returns:
            X: NTT of x
        """
        x = x.to(self.device)
        
        # Convert to QTT
        qtt = self._vector_to_qtt(x, max_rank)
        
        # Apply butterfly stages
        self._ensure_butterflies()
        
        for stage in range(self.n_bits):
            # Apply butterfly MPO
            if HAS_TENSORNET:
                qtt = apply_mpo_cuda(self._butterfly_mpos[stage], qtt)
            else:
                qtt = self._apply_mpo_cpu(self._butterfly_mpos[stage], qtt)
            
            # Apply twiddle factors
            qtt = apply_twiddle_factors(qtt, stage, self.omega, self.field)
            
            # Truncate to control rank growth
            qtt = self._truncate_qtt(qtt, max_rank)
        
        # Bit reversal permutation (reorder cores)
        qtt = qtt[::-1]
        
        # Convert back to vector
        return self._qtt_to_vector(qtt)
    
    def inverse(self, X: Tensor, max_rank: int = 64) -> Tensor:
        """Inverse NTT."""
        # INTT = conj(NTT(conj(X))) / N for complex
        if self.field == 'complex':
            x = torch.conj(self.forward(torch.conj(X), max_rank)) / self.N
            return x
        else:
            # Finite field: use omega_inv
            # TODO: Implement finite field inverse
            raise NotImplementedError("Finite field INTT coming soon")
    
    def _truncate_qtt(self, cores: List[Tensor], max_rank: int) -> List[Tensor]:
        """Truncate QTT to maximum rank using SVD."""
        new_cores = []
        
        for i, core in enumerate(cores):
            r_left, d, r_right = core.shape
            
            if r_right > max_rank and i < len(cores) - 1:
                # SVD truncation
                mat = core.reshape(r_left * d, r_right)
                
                if HAS_TENSORNET:
                    U, S, Vh, _ = svd_truncated(mat, chi_max=max_rank, return_info=True)
                else:
                    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
                    rank = min(max_rank, len(S))
                    U, S, Vh = U[:, :rank], S[:rank], Vh[:rank, :]
                
                new_cores.append(U.reshape(r_left, d, len(S)))
                
                # Absorb S @ Vh into next core
                if i + 1 < len(cores):
                    cores[i + 1] = torch.einsum('ij,jdk->idk', torch.diag(S) @ Vh, cores[i + 1])
            else:
                new_cores.append(core)
        
        return new_cores
    
    def _apply_mpo_cpu(self, mpo: List[Tensor], qtt: List[Tensor]) -> List[Tensor]:
        """CPU fallback for MPO application."""
        new_cores = []
        for O, P in zip(mpo, qtt):
            rLo, d_out, d_in, rRo = O.shape
            rLp, d_in_p, rRp = P.shape
            result = torch.einsum("oabr,pbq->oparq", O, P)
            result = result.reshape(rLo * rLp, d_out, rRo * rRp)
            new_cores.append(result)
        return new_cores


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def qtt_ntt_forward(x: Tensor, n_bits: int = None, max_rank: int = 64) -> Tensor:
    """
    Quick forward NTT using QTT.
    
    Args:
        x: Input vector (length must be power of 2)
        n_bits: log2(len(x)), computed automatically if None
        max_rank: Maximum QTT rank
    """
    if n_bits is None:
        n_bits = int(torch.log2(torch.tensor(len(x))).item())
    
    ntt = QTTNTT(n_bits)
    return ntt.forward(x, max_rank)


def qtt_ntt_inverse(X: Tensor, n_bits: int = None, max_rank: int = 64) -> Tensor:
    """Quick inverse NTT using QTT."""
    if n_bits is None:
        n_bits = int(torch.log2(torch.tensor(len(X))).item())
    
    ntt = QTTNTT(n_bits)
    return ntt.inverse(X, max_rank)


def qtt_poly_multiply(a: Tensor, b: Tensor, max_rank: int = 64) -> Tensor:
    """
    Polynomial multiplication using QTT-NTT.
    
    This is THE core operation in ZK proving.
    """
    # Pad to power of 2
    n = len(a) + len(b) - 1
    n_bits = int(torch.ceil(torch.log2(torch.tensor(n))).item())
    N = 2 ** n_bits
    
    a_pad = torch.zeros(N, dtype=a.dtype, device=a.device)
    b_pad = torch.zeros(N, dtype=b.dtype, device=b.device)
    a_pad[:len(a)] = a
    b_pad[:len(b)] = b
    
    # NTT-based convolution
    ntt = QTTNTT(n_bits)
    A = ntt.forward(a_pad, max_rank)
    B = ntt.forward(b_pad, max_rank)
    C = A * B  # Pointwise
    c = ntt.inverse(C, max_rank)
    
    return c[:n].real if c.is_complex() else c[:n]


# =============================================================================
# BENCHMARKS
# =============================================================================

def benchmark_qtt_ntt(n_bits_list: List[int] = None, max_rank: int = 64):
    """Benchmark QTT-NTT against standard FFT."""
    import time
    
    if n_bits_list is None:
        n_bits_list = [8, 10, 12, 14, 16]
    
    print("\n" + "="*70)
    print("QTT-NTT BENCHMARK")
    print("="*70)
    print(f"\n{'N':>10} {'FFT (ms)':>12} {'QTT-NTT (ms)':>14} {'Ratio':>10} {'Error':>12}")
    print("-"*60)
    
    for n_bits in n_bits_list:
        N = 2 ** n_bits
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        x = torch.randn(N, dtype=torch.complex128, device=device)
        
        # Standard FFT
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t0 = time.perf_counter()
        for _ in range(5):
            y_fft = torch.fft.fft(x)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        fft_time = (time.perf_counter() - t0) / 5 * 1000
        
        # QTT-NTT
        ntt = QTTNTT(n_bits, device=device)
        
        # Warmup
        _ = ntt.forward(x, max_rank)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        
        t0 = time.perf_counter()
        for _ in range(5):
            y_qtt = ntt.forward(x, max_rank)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        qtt_time = (time.perf_counter() - t0) / 5 * 1000
        
        # Error
        error = torch.max(torch.abs(y_qtt - y_fft)).item() / torch.max(torch.abs(y_fft)).item()
        
        ratio = qtt_time / fft_time
        
        print(f"{N:>10} {fft_time:>12.3f} {qtt_time:>14.3f} {ratio:>10.1f}x {error:>12.2e}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Quick test
    print("QTT-NTT Module Loaded")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  TensorNet available: {HAS_TENSORNET}")
    
    # Run benchmark
    benchmark_qtt_ntt([8, 10, 12])
