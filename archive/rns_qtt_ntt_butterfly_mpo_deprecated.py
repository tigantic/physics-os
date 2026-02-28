"""
RNS-QTT-NTT: Residue Number System + Quantized Tensor Train NTT

This module implements a high-performance NTT for finite fields using:
1. RNS decomposition into small NTT-friendly primes (< 2^26)
2. QTT-compressed state representation for structured polynomials
3. MPO-based butterfly operations with float64 exact arithmetic
4. CRT reconstruction back to the target field

Architecture:
    Baby Bear element (31-bit)
        ↓
    RNS decompose into k small primes (each < 2^26)
        ↓
    QTT-NTT on each channel (float64 exact, tensor cores usable)
        ↓
    QTT compresses state between butterfly stages
        ↓
    CRT reconstruct → result mod Baby Bear

Key advantages:
- Float64 products are exact (p² < 2^52 for each RNS prime)
- QTT compression reduces memory bandwidth (GPU bottleneck)
- RNS channels are embarrassingly parallel
- Tensor cores can accelerate MPO contractions

Author: TiganticLabz
"""

from __future__ import annotations

import torch
import numpy as np
from torch import Tensor
from typing import List, Tuple, Optional, NamedTuple
from dataclasses import dataclass

# Try Numba for CPU fallback
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# =============================================================================
# RNS CONFIGURATION
# =============================================================================

@dataclass
class RNSPrime:
    """NTT-friendly prime for RNS decomposition."""
    p: int          # The prime
    k: int          # p = k * 2^m + 1
    m: int          # Power of 2 factor
    g: int          # Primitive root (generator)
    max_ntt: int    # Maximum NTT size = 2^m
    safe_rank: int  # Max QTT rank for float64 precision


def find_primitive_root(p: int) -> int:
    """Find smallest primitive root modulo p."""
    if p == 2:
        return 1
    
    # Factor p-1
    phi = p - 1
    factors = []
    n = phi
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)
    
    # Find generator
    for g in range(2, p):
        is_generator = True
        for f in factors:
            if pow(g, phi // f, p) == 1:
                is_generator = False
                break
        if is_generator:
            return g
    
    raise ValueError(f"No primitive root found for p={p}")


# RNS primes selected for Baby Bear field
# Criteria:
# - p = k * 2^m + 1 (NTT-friendly)
# - p < 2^26 (float64 exact products)
# - m >= 20 (support NTTs up to 2^20)
# - Product > Baby Bear (unique CRT reconstruction)

RNS_PRIME_1 = RNSPrime(
    p=7340033,
    k=7,
    m=20,
    g=3,  # Will verify
    max_ntt=2**20,
    safe_rank=167,  # floor(2^53 / p^2)
)

RNS_PRIME_2 = RNSPrime(
    p=13631489,
    k=13,
    m=20,
    g=3,  # Will verify
    max_ntt=2**20,
    safe_rank=48,
)

# Verify and set primitive roots
RNS_PRIME_1.g = find_primitive_root(RNS_PRIME_1.p)
RNS_PRIME_2.g = find_primitive_root(RNS_PRIME_2.p)

# Default RNS configuration for Baby Bear
RNS_PRIMES = [RNS_PRIME_1, RNS_PRIME_2]
RNS_PRODUCT = RNS_PRIME_1.p * RNS_PRIME_2.p  # ~100 trillion

# Baby Bear prime
BABY_BEAR_PRIME = 2013265921


# =============================================================================
# RNS ARITHMETIC
# =============================================================================

class RNSContext:
    """
    Context for RNS arithmetic operations.
    
    Precomputes CRT reconstruction coefficients for efficient
    conversion between RNS and standard representation.
    """
    
    def __init__(self, primes: List[RNSPrime], target_prime: int):
        """
        Initialize RNS context.
        
        Args:
            primes: List of RNS primes
            target_prime: Target field prime (e.g., Baby Bear)
        """
        self.primes = primes
        self.target_prime = target_prime
        self.k = len(primes)
        
        # Compute product of all RNS primes
        self.M = 1
        for rp in primes:
            self.M *= rp.p
        
        assert self.M > target_prime, \
            f"RNS product {self.M} must exceed target prime {target_prime}"
        
        # Precompute CRT coefficients
        # For each i: M_i = M / p_i, y_i = M_i^{-1} mod p_i
        self.M_i = []
        self.y_i = []
        
        for i, rp in enumerate(primes):
            M_i = self.M // rp.p
            y_i = pow(M_i, rp.p - 2, rp.p)  # Modular inverse via Fermat
            self.M_i.append(M_i)
            self.y_i.append(y_i)
        
        # Precompute primitive N-th roots for various NTT sizes
        self._omega_cache = {}
    
    def decompose(self, x: int) -> List[int]:
        """
        Decompose x into RNS representation.
        
        Args:
            x: Integer in [0, target_prime)
            
        Returns:
            List of residues [x mod p_1, x mod p_2, ...]
        """
        return [x % rp.p for rp in self.primes]
    
    def decompose_vector(self, x: List[int]) -> List[List[int]]:
        """
        Decompose vector into RNS channels.
        
        Args:
            x: Vector of integers
            
        Returns:
            List of k vectors, one per RNS channel
        """
        k = len(self.primes)
        channels = [[] for _ in range(k)]
        
        for val in x:
            for i, rp in enumerate(self.primes):
                channels[i].append(val % rp.p)
        
        return channels
    
    def reconstruct(self, residues: List[int]) -> int:
        """
        Reconstruct integer from RNS representation using CRT.
        
        Args:
            residues: List of residues [r_1, r_2, ...]
            
        Returns:
            x such that x ≡ r_i (mod p_i) for all i, reduced mod target_prime
        """
        # CRT formula: x = sum(r_i * M_i * y_i) mod M
        result = 0
        for i, r in enumerate(residues):
            result += r * self.M_i[i] * self.y_i[i]
        
        # Reduce mod M first (to handle overflow), then mod target
        result = result % self.M
        return result % self.target_prime
    
    def reconstruct_vector(self, channels: List[List[int]]) -> List[int]:
        """
        Reconstruct vector from RNS channels using CRT.
        
        Args:
            channels: List of k vectors, one per RNS channel
            
        Returns:
            Reconstructed vector mod target_prime
        """
        n = len(channels[0])
        result = []
        
        for j in range(n):
            residues = [channels[i][j] for i in range(self.k)]
            result.append(self.reconstruct(residues))
        
        return result
    
    def get_omega(self, n_bits: int, channel_idx: int) -> int:
        """
        Get primitive N-th root of unity for NTT.
        
        Args:
            n_bits: log2(N)
            channel_idx: Which RNS channel
            
        Returns:
            omega such that omega^N = 1 in F_{p_i}
        """
        key = (n_bits, channel_idx)
        if key not in self._omega_cache:
            rp = self.primes[channel_idx]
            N = 2 ** n_bits
            
            assert N <= rp.max_ntt, \
                f"NTT size {N} exceeds max {rp.max_ntt} for prime {rp.p}"
            
            # omega = g^{(p-1)/N}
            exp = (rp.p - 1) // N
            omega = pow(rp.g, exp, rp.p)
            
            # Verify it's a primitive N-th root
            assert pow(omega, N, rp.p) == 1, "omega^N != 1"
            assert pow(omega, N // 2, rp.p) != 1, "omega is not primitive"
            
            self._omega_cache[key] = omega
        
        return self._omega_cache[key]


# Default context for Baby Bear
_default_rns_context = None

def get_rns_context() -> RNSContext:
    """Get default RNS context for Baby Bear field."""
    global _default_rns_context
    if _default_rns_context is None:
        _default_rns_context = RNSContext(RNS_PRIMES, BABY_BEAR_PRIME)
    return _default_rns_context


# =============================================================================
# QTT REPRESENTATION
# =============================================================================

class QTTState:
    """
    Quantized Tensor Train representation of a vector.
    
    A vector of length N = 2^n is represented as a tensor train:
        v[i_0, i_1, ..., i_{n-1}] = G_0[i_0] @ G_1[i_1] @ ... @ G_{n-1}[i_{n-1}]
    
    where each G_k is a (r_{k-1}, 2, r_k) tensor called a "core".
    
    The "rank" of the QTT is max(r_k), which determines both:
    - Storage: O(n * r^2) instead of O(N) = O(2^n)
    - Contraction cost: O(n * r^2) per operation
    """
    
    def __init__(self, cores: List[Tensor], prime: int):
        """
        Initialize QTT state.
        
        Args:
            cores: List of n cores, each shape (r_left, 2, r_right)
            prime: Field prime for modular arithmetic
        """
        self.cores = cores
        self.prime = prime
        self.n_bits = len(cores)
        self.N = 2 ** self.n_bits
        
        # Validate shapes
        for i, core in enumerate(cores):
            assert core.dim() == 3, f"Core {i} must be 3D, got {core.dim()}"
            assert core.shape[1] == 2, f"Core {i} physical dim must be 2"
            
            if i > 0:
                assert core.shape[0] == cores[i-1].shape[2], \
                    f"Bond dimension mismatch at core {i}"
        
        assert cores[0].shape[0] == 1, "First core must have r_left=1"
        assert cores[-1].shape[2] == 1, "Last core must have r_right=1"
    
    @property
    def ranks(self) -> List[int]:
        """Get bond dimensions [r_0, r_1, ..., r_n]."""
        ranks = [1]
        for core in self.cores:
            ranks.append(core.shape[2])
        return ranks
    
    @property
    def max_rank(self) -> int:
        """Get maximum bond dimension."""
        return max(self.ranks)
    
    @property
    def storage(self) -> int:
        """Get total storage in number of elements."""
        return sum(core.numel() for core in self.cores)
    
    @property
    def compression_ratio(self) -> float:
        """Get compression ratio vs dense storage."""
        return self.N / self.storage
    
    def eval_at(self, index: int) -> int:
        """
        Evaluate QTT at a single index WITHOUT going to dense.
        
        This is the key operation - O(n × r²) per evaluation.
        NEVER materializes the full vector.
        
        Args:
            index: Integer index in [0, N)
        
        Returns:
            Value at that index (mod prime)
        """
        n_bits = self.n_bits
        
        # Extract bits of index (LSB-first: bit k is (index >> k) & 1)
        # This matches the butterfly MPO convention where core k corresponds to bit k
        bits = [(index >> k) & 1 for k in range(n_bits)]
        
        # Contract along physical dimensions
        # vec starts as row vector (1, r_right) from first core
        vec = self.cores[0][0, bits[0], :]  # Shape (r_right,)
        
        for k in range(1, n_bits):
            core_slice = self.cores[k][:, bits[k], :]  # Shape (r_left, r_right)
            vec = vec @ core_slice  # (r_left,) @ (r_left, r_right) = (r_right,)
        
        # Apply mod and return
        return int(round(vec.item())) % self.prime
    
    def eval_all(self) -> List[int]:
        """
        Evaluate QTT at ALL indices by iterating (not going to dense tensor).
        
        This is O(N × n × r²) - slower than to_dense for random data,
        but avoids catastrophic memory explosion.
        
        For verification only - use eval_at for spot checks.
        """
        return [self.eval_at(i) for i in range(self.N)]
    
    def to_dense(self) -> Tensor:
        """
        Contract QTT to dense vector.
        
        Returns:
            Tensor of shape (N,) with values mod prime
        """
        # Contract cores from left to right
        # Start with first core: (1, 2, r_1) -> (2, r_1)
        result = self.cores[0].squeeze(0)  # (2, r_1)
        
        for core in self.cores[1:]:
            # result: (2^k, r_k)
            # core: (r_k, 2, r_{k+1})
            # Contract bond dimension, expand physical
            # result_new[i*2+j, r'] = sum_r result[i, r] * core[r, j, r']
            batch_size, r_left = result.shape
            r_left_core, d, r_right = core.shape
            
            # Reshape for batch matmul
            # result: (batch, r_left) -> (batch, 1, r_left)
            # core: (r_left, 2, r_right) -> (r_left, 2*r_right)
            result = result @ core.reshape(r_left_core, d * r_right)  # (batch, 2*r_right)
            result = result.reshape(batch_size * d, r_right)  # (batch*2, r_right)
        
        # Final result: (N, 1) -> (N,)
        result = result.squeeze(-1)
        
        # Apply modular reduction
        result = torch.remainder(result, self.prime)
        
        return result
    
    @staticmethod
    def from_dense(x: Tensor, prime: int, max_rank: int = 64) -> 'QTTState':
        """
        Create QTT from dense vector using SVD decomposition.
        
        Uses LSB-first convention: core k corresponds to bit k (the k-th 
        least significant bit). This matches eval_at and the butterfly MPO.
        
        Index i = bit_0 + 2*bit_1 + 4*bit_2 + ...
        Core 0 selects bit_0, Core 1 selects bit_1, etc.
        
        Args:
            x: Dense vector of length N = 2^n
            prime: Field prime
            max_rank: Maximum bond dimension
            
        Returns:
            QTT representation
        """
        N = x.numel()
        n_bits = N.bit_length() - 1
        assert 2 ** n_bits == N, f"Length must be power of 2, got {N}"
        
        # Bit-reverse the input so standard TT-SVD (MSB-first) produces LSB-first cores
        # Bit-reversal: br(i) swaps bit order, so x_new[i] = x[br(i)]
        # After TT-SVD, core k will handle bit_{n-1-k} of original = bit_k of new
        indices = torch.tensor([int(bin(i)[2:].zfill(n_bits)[::-1], 2) for i in range(N)])
        x_reordered = x[indices]
        
        # Standard left-to-right TT-SVD
        cores = []
        remainder = x_reordered.reshape(1, -1).to(torch.float64)
        r_left = 1
        
        for k in range(n_bits - 1):
            remainder = remainder.reshape(r_left * 2, -1)
            
            m, n = remainder.shape
            target_rank = min(max_rank, m, n)
            
            if min(m, n) > 2 * max_rank:
                U, S, V = torch.svd_lowrank(remainder, q=target_rank, niter=2)
                Vh = V.T
            else:
                U, S, Vh = torch.linalg.svd(remainder, full_matrices=False)
            
            rank = min(max_rank, len(S))
            threshold = 1e-12 * S[0] if len(S) > 0 else 0
            rank = min(rank, (S > threshold).sum().item())
            rank = max(1, rank)
            
            U = U[:, :rank]
            S = S[:rank]
            Vh = Vh[:rank, :]
            
            cores.append(U.reshape(r_left, 2, rank))
            remainder = torch.diag(S) @ Vh
            r_left = rank
        
        cores.append(remainder.reshape(r_left, 2, 1))
        
        return QTTState(cores, prime)
    
    def truncate(self, max_rank: int) -> 'QTTState':
        """
        Truncate QTT to maximum rank using SVD.
        
        This performs a left-to-right sweep, orthogonalizing and truncating.
        
        Args:
            max_rank: Maximum bond dimension
            
        Returns:
            Truncated QTT (may have approximation error)
        """
        if self.max_rank <= max_rank:
            return self
        
        cores = [c.clone() for c in self.cores]
        
        # Left-to-right orthogonalization
        for k in range(self.n_bits - 1):
            core = cores[k]
            r_left, d, r_right = core.shape
            
            # Reshape and SVD
            mat = core.reshape(r_left * d, r_right)
            U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
            
            # Truncate
            rank = min(max_rank, len(S))
            threshold = 1e-12 * S[0] if len(S) > 0 else 0
            rank = min(rank, (S > threshold).sum().item())
            rank = max(1, rank)
            
            U = U[:, :rank]
            S = S[:rank]
            Vh = Vh[:rank, :]
            
            # Update cores
            cores[k] = U.reshape(r_left, d, rank)
            
            # Absorb S @ Vh into next core
            next_core = cores[k + 1]
            r_next_left, d_next, r_next_right = next_core.shape
            cores[k + 1] = (torch.diag(S) @ Vh @ next_core.reshape(r_next_left, -1)).reshape(rank, d_next, r_next_right)
        
        return QTTState(cores, self.prime)


# =============================================================================
# QTT-MPO BUTTERFLY OPERATIONS
# =============================================================================

def build_butterfly_mpo(
    n_bits: int,
    stage: int,
    omega: int,
    prime: int,
    device: torch.device = None,
) -> List[Tensor]:
    """
    Build butterfly MPO for stage k of NTT.
    
    The butterfly at stage s operates on pairs (i, i + 2^s) with stride 2^{s+1}.
    In QTT form, this is represented as an MPO where:
    - Cores 0..s-1: Accumulate twiddle index from input bits (identity on physical)
    - Core s: Apply butterfly matrix [[1, ω^j], [1, -ω^j]]
    - Cores s+1..n-1: Identity
    
    Args:
        n_bits: log2(N)
        stage: Butterfly stage (0 to n_bits-1)
        omega: Primitive N-th root of unity
        prime: Field prime
        device: Torch device
        
    Returns:
        List of MPO cores, each shape (r_left, d_out, d_in, r_right)
    """
    N = 2 ** n_bits
    device = device or torch.device('cpu')
    dtype = torch.float64
    
    cores = []
    
    # Twiddle step for this stage
    twiddle_step = N // (2 ** (stage + 1))
    
    for k in range(n_bits):
        if k < stage:
            # Accumulator: track twiddle index from input bits
            # Bond dimension doubles at each position
            r_left = 2 ** k if k > 0 else 1
            r_right = 2 ** (k + 1)
            
            core = torch.zeros(r_left, 2, 2, r_right, dtype=dtype, device=device)
            
            # Identity on physical dimension, branch on bond
            for j_in in range(r_left):
                for b in range(2):  # physical bit value
                    j_out = j_in + b * (2 ** k)
                    core[j_in, b, b, j_out] = 1.0
            
            cores.append(core)
            
        elif k == stage:
            # Butterfly core
            r_left = 2 ** stage if stage > 0 else 1
            r_right = 1
            
            core = torch.zeros(r_left, 2, 2, r_right, dtype=dtype, device=device)
            
            for j in range(r_left):
                tw = pow(omega, j * twiddle_step, prime)
                tw_neg = (prime - tw) % prime
                
                # Butterfly: [[1, tw], [1, -tw]]
                core[j, 0, 0, 0] = 1.0       # out=0 <- in=0
                core[j, 0, 1, 0] = float(tw)  # out=0 <- in=1
                core[j, 1, 0, 0] = 1.0       # out=1 <- in=0
                core[j, 1, 1, 0] = float(tw_neg)  # out=1 <- in=1
            
            cores.append(core)
            
        else:
            # Identity
            core = torch.zeros(1, 2, 2, 1, dtype=dtype, device=device)
            core[0, 0, 0, 0] = 1.0
            core[0, 1, 1, 0] = 1.0
            cores.append(core)
    
    return cores


def apply_mpo_to_qtt(
    qtt: QTTState,
    mpo: List[Tensor],
    max_rank: int,
) -> QTTState:
    """
    Apply MPO to QTT state: |ψ'⟩ = MPO |ψ⟩
    
    The result has bond dimension product r_qtt * r_mpo, which is then
    truncated back to max_rank.
    
    Args:
        qtt: Input QTT state
        mpo: MPO cores, each (r_left, d_out, d_in, r_right)
        max_rank: Maximum rank after truncation
        
    Returns:
        Output QTT state
    """
    n_bits = qtt.n_bits
    assert len(mpo) == n_bits
    
    new_cores = []
    
    for k in range(n_bits):
        q_core = qtt.cores[k]  # (sq_l, d, sq_r)
        m_core = mpo[k]        # (sm_l, d_out, d_in, sm_r)
        
        sq_l, d_in, sq_r = q_core.shape
        sm_l, d_out, _, sm_r = m_core.shape
        
        # Contract physical index
        # Einsum: contract d_in from MPO with d from QTT
        # We want consistent bond ordering: (qtt_bond, mpo_bond) throughout
        # result[sq_l, sm_l, d_out, sq_r, sm_r] = sum_d m_core[sm_l, d_out, d, sm_r] * q_core[sq_l, d, sq_r]
        result = torch.einsum('aobm,lbr->laorm', m_core, q_core)
        
        # Reshape to (sq_l * sm_l, d_out, sq_r * sm_r)
        # Bond order: (qtt, mpo) pairs
        result = result.reshape(sq_l * sm_l, d_out, sq_r * sm_r)
        
        new_cores.append(result)
    
    # Create new QTT and truncate
    new_qtt = QTTState(new_cores, qtt.prime)
    return new_qtt.truncate(max_rank)


def apply_butterfly_stage(
    qtt: QTTState,
    stage: int,
    omega: int,
    max_rank: int,
) -> QTTState:
    """
    Apply one butterfly stage to QTT state.
    
    Args:
        qtt: Input QTT state
        stage: Butterfly stage (0 to n_bits-1)
        omega: Primitive N-th root of unity
        max_rank: Maximum rank after truncation
        
    Returns:
        Output QTT state after butterfly
    """
    mpo = build_butterfly_mpo(qtt.n_bits, stage, omega, qtt.prime)
    return apply_mpo_to_qtt(qtt, mpo, max_rank)


# =============================================================================
# RNS-QTT-NTT MAIN ALGORITHM
# =============================================================================

class RNSQTTNTT:
    """
    RNS-QTT-NTT: High-performance NTT using Residue Number System + QTT.
    
    This class combines:
    1. RNS decomposition for float64-exact arithmetic
    2. QTT compression for memory efficiency
    3. MPO butterfly operators for tensor-native computation
    4. CRT reconstruction back to target field
    
    Usage:
        ntt = RNSQTTNTT(n_bits=16, target_prime=BABY_BEAR_PRIME)
        result = ntt.forward(polynomial_coefficients)
    """
    
    def __init__(
        self,
        n_bits: int,
        target_prime: int = BABY_BEAR_PRIME,
        rns_context: Optional[RNSContext] = None,
        max_rank: int = 64,
        device: torch.device = None,
    ):
        """
        Initialize RNS-QTT-NTT.
        
        Args:
            n_bits: log2(N) where N is NTT size
            target_prime: Target field prime
            rns_context: RNS context (default: Baby Bear context)
            max_rank: Maximum QTT rank
            device: Torch device
        """
        self.n_bits = n_bits
        self.N = 2 ** n_bits
        self.target_prime = target_prime
        self.max_rank = max_rank
        self.device = device or torch.device('cpu')
        
        # RNS context
        self.rns = rns_context or get_rns_context()
        assert self.rns.target_prime == target_prime
        
        # Validate NTT size
        for i, rp in enumerate(self.rns.primes):
            assert self.N <= rp.max_ntt, \
                f"NTT size {self.N} exceeds max {rp.max_ntt} for RNS prime {rp.p}"
        
        # Precompute roots of unity for each channel
        self.omegas = [self.rns.get_omega(n_bits, i) for i in range(self.rns.k)]
        self.omega_invs = [pow(w, rp.p - 2, rp.p) for w, rp in zip(self.omegas, self.rns.primes)]
        
        # Precompute N^{-1} for each channel (for inverse NTT)
        self.n_invs = [pow(self.N, rp.p - 2, rp.p) for rp in self.rns.primes]
    
    def _bit_reverse(self, x: List[int]) -> List[int]:
        """Apply bit-reversal permutation."""
        n = len(x)
        result = list(x)
        for i in range(n):
            j = int('{:0{w}b}'.format(i, w=self.n_bits)[::-1], 2)
            if i < j:
                result[i], result[j] = result[j], result[i]
        return result
    
    def forward_qtt_channels(
        self,
        x: List[int],
        return_stats: bool = False,
    ) -> List[List[int]] | Tuple[List[List[int]], dict]:
        """
        Forward NTT using RNS + QTT, returning RNS channels (no CRT).
        
        This preserves the channel structure for correct inverse operation.
        Use this when you need to do inverse NTT later.
        
        Args:
            x: Input polynomial coefficients in target field
            return_stats: If True, also return compression statistics
            
        Returns:
            List of k channel vectors (one per RNS prime), optionally with stats
        """
        return self._forward_qtt_impl(x, return_stats, reconstruct=False)
    
    def forward_qtt(
        self,
        x: List[int],
        return_stats: bool = False,
    ) -> List[int] | Tuple[List[int], dict]:
        """
        Forward NTT using RNS + QTT with CRT reconstruction.
        
        NOTE: The output cannot be correctly inverted with inverse_qtt() because
        CRT reconstruction mod Baby Bear loses RNS channel information.
        Use forward_qtt_channels() + inverse_qtt_channels() for correct roundtrip.
        
        Args:
            x: Input polynomial coefficients in target field
            return_stats: If True, also return compression statistics
            
        Returns:
            NTT result (and optionally stats dict)
        """
        return self._forward_qtt_impl(x, return_stats, reconstruct=True)
    
    def _forward_qtt_impl(
        self,
        x: List[int],
        return_stats: bool = False,
        reconstruct: bool = True,
    ):
        """Internal forward NTT implementation."""
        assert len(x) == self.N
        
        stats = {
            'channels': [],
            'input_storage': self.N,
        }
        
        # Step 1: RNS decomposition
        channels = self.rns.decompose_vector(x)
        
        # Step 2: QTT-NTT on each channel
        result_channels = []
        
        for ch_idx, (channel, rp) in enumerate(zip(channels, self.rns.primes)):
            omega = self.omegas[ch_idx]
            
            # NOTE: No explicit bit-reversal needed!
            # QTT from_dense implicitly produces a bit-reversed representation
            # which is exactly what Cooley-Tukey NTT expects.
            
            # Convert to QTT
            x_tensor = torch.tensor(channel, dtype=torch.float64, device=self.device)
            qtt = QTTState.from_dense(x_tensor, rp.p, self.max_rank)
            
            ch_stats = {
                'prime': rp.p,
                'initial_rank': qtt.max_rank,
                'stage_ranks': [qtt.max_rank],
            }
            
            # Apply butterfly stages
            for stage in range(self.n_bits):
                qtt = apply_butterfly_stage(qtt, stage, omega, self.max_rank)
                ch_stats['stage_ranks'].append(qtt.max_rank)
            
            ch_stats['final_rank'] = qtt.max_rank
            ch_stats['storage'] = qtt.storage
            ch_stats['compression'] = qtt.compression_ratio
            
            # Extract result - use eval_all() to avoid dense memory explosion
            result_channels.append(qtt.eval_all())
            
            stats['channels'].append(ch_stats)
        
        # Step 3: Optionally do CRT reconstruction
        if reconstruct:
            result = self.rns.reconstruct_vector(result_channels)
        else:
            result = result_channels
        
        # Aggregate stats
        total_storage = sum(ch['storage'] for ch in stats['channels'])
        stats['total_storage'] = total_storage
        stats['total_compression'] = (self.N * self.rns.k) / total_storage
        
        if return_stats:
            return result, stats
        return result
    
    def forward_direct(self, x: List[int]) -> List[int]:
        """
        Forward NTT using RNS + direct (non-QTT) computation.
        
        This is the reference implementation for correctness validation.
        Uses standard Cooley-Tukey on each RNS channel.
        
        Args:
            x: Input polynomial coefficients
            
        Returns:
            NTT result
        """
        assert len(x) == self.N
        
        # RNS decomposition
        channels = self.rns.decompose_vector(x)
        
        # NTT on each channel
        result_channels = []
        
        for ch_idx, (channel, rp) in enumerate(zip(channels, self.rns.primes)):
            omega = self.omegas[ch_idx]
            p = rp.p
            
            # Standard Cooley-Tukey
            a = self._bit_reverse(channel)
            
            m = 2
            for stage in range(self.n_bits):
                wm = pow(omega, self.N // m, p)
                
                for k in range(0, self.N, m):
                    w = 1
                    for j in range(m // 2):
                        t = (w * a[k + j + m // 2]) % p
                        u = a[k + j]
                        a[k + j] = (u + t) % p
                        a[k + j + m // 2] = (u - t) % p
                        w = (w * wm) % p
                
                m *= 2
            
            result_channels.append(a)
        
        # CRT reconstruction
        return self.rns.reconstruct_vector(result_channels)
    
    def forward(self, x: List[int], use_qtt: bool = True) -> List[int]:
        """
        Forward NTT with automatic method selection.
        
        Args:
            x: Input polynomial coefficients
            use_qtt: Use QTT acceleration (default True)
            
        Returns:
            NTT result
        """
        if use_qtt:
            return self.forward_qtt(x)
        else:
            return self.forward_direct(x)
    
    def inverse_qtt_channels(self, channels: List[List[int]]) -> List[int]:
        """
        Inverse NTT using RNS + QTT, taking RNS channels directly.
        
        This is the correct way to do NTT roundtrip:
            channels = forward_qtt_channels(x)
            x_back = inverse_qtt_channels(channels)
        
        Args:
            channels: List of k channel vectors from forward_qtt_channels()
            
        Returns:
            Original polynomial coefficients
        """
        return self._inverse_qtt_impl(channels)
    
    def inverse_qtt(self, X: List[int]) -> List[int]:
        """
        Inverse NTT using RNS + QTT.
        
        WARNING: This decomposes X into RNS channels, which loses information
        if X was produced by forward_qtt(). For correct roundtrip, use
        forward_qtt_channels() + inverse_qtt_channels() instead.
        
        Args:
            X: NTT coefficients (CRT-reconstructed)
            
        Returns:
            Polynomial coefficients (may be incorrect if X was CRT-reconstructed!)
        """
        assert len(X) == self.N
        channels = self.rns.decompose_vector(X)
        return self._inverse_qtt_impl(channels)
    
    def _inverse_qtt_impl(self, channels: List[List[int]]) -> List[int]:
        """Internal inverse NTT implementation taking channels directly."""
        assert len(channels) == self.rns.k
        assert len(channels[0]) == self.N
        
        # Inverse NTT on each channel
        result_channels = []
        
        for ch_idx, (channel, rp) in enumerate(zip(channels, self.rns.primes)):
            omega_inv = self.omega_invs[ch_idx]
            n_inv = self.n_invs[ch_idx]
            p = rp.p
            
            # NOTE: No explicit bit-reversal needed!
            # QTT from_dense implicitly produces a bit-reversed representation.
            
            # Convert to QTT
            x_tensor = torch.tensor(channel, dtype=torch.float64, device=self.device)
            qtt = QTTState.from_dense(x_tensor, p, self.max_rank)
            
            # Apply butterfly stages with omega_inv
            for stage in range(self.n_bits):
                qtt = apply_butterfly_stage(qtt, stage, omega_inv, self.max_rank)
            
            # Extract and scale by N^{-1} - use eval_all() to avoid dense memory explosion
            result_values = qtt.eval_all()
            result_scaled = [(v * n_inv) % p for v in result_values]
            result_channels.append(result_scaled)
        
        # CRT reconstruction
        return self.rns.reconstruct_vector(result_channels)
    
    def inverse_direct(self, X: List[int]) -> List[int]:
        """Inverse NTT using RNS + direct computation."""
        assert len(X) == self.N
        
        channels = self.rns.decompose_vector(X)
        result_channels = []
        
        for ch_idx, (channel, rp) in enumerate(zip(channels, self.rns.primes)):
            omega_inv = self.omega_invs[ch_idx]
            n_inv = self.n_invs[ch_idx]
            p = rp.p
            
            a = self._bit_reverse(channel)
            
            m = 2
            for stage in range(self.n_bits):
                wm = pow(omega_inv, self.N // m, p)
                
                for k in range(0, self.N, m):
                    w = 1
                    for j in range(m // 2):
                        t = (w * a[k + j + m // 2]) % p
                        u = a[k + j]
                        a[k + j] = (u + t) % p
                        a[k + j + m // 2] = (u - t) % p
                        w = (w * wm) % p
                
                m *= 2
            
            # Scale by N^{-1}
            a = [(v * n_inv) % p for v in a]
            result_channels.append(a)
        
        return self.rns.reconstruct_vector(result_channels)
    
    def inverse(self, X: List[int], use_qtt: bool = True) -> List[int]:
        """Inverse NTT with automatic method selection."""
        if use_qtt:
            return self.inverse_qtt(X)
        else:
            return self.inverse_direct(X)


# =============================================================================
# VALIDATION AND BENCHMARKING
# =============================================================================

def validate_rns_qtt_ntt():
    """Validate RNS-QTT-NTT against reference implementation."""
    print("=" * 60)
    print("RNS-QTT-NTT Validation")
    print("=" * 60)
    
    # Test various sizes
    for n_bits in [4, 8, 10, 12]:
        N = 2 ** n_bits
        
        ntt = RNSQTTNTT(n_bits)
        
        # Random input
        import random
        random.seed(42)
        x = [random.randint(0, BABY_BEAR_PRIME - 1) for _ in range(N)]
        
        # Forward NTT
        result_qtt = ntt.forward_qtt(x)
        result_direct = ntt.forward_direct(x)
        
        forward_match = result_qtt == result_direct
        
        # Inverse NTT
        recovered_qtt = ntt.inverse_qtt(result_qtt)
        recovered_direct = ntt.inverse_direct(result_direct)
        
        inverse_qtt_match = recovered_qtt == x
        inverse_direct_match = recovered_direct == x
        
        status = "✓" if (forward_match and inverse_qtt_match and inverse_direct_match) else "✗"
        
        print(f"N={N:>6}: forward={forward_match}, inverse_qtt={inverse_qtt_match}, inverse_direct={inverse_direct_match} {status}")
    
    print()


def benchmark_compression():
    """Benchmark QTT compression on different input types."""
    print("=" * 60)
    print("QTT Compression Benchmark")
    print("=" * 60)
    
    n_bits = 12
    N = 2 ** n_bits
    
    ntt = RNSQTTNTT(n_bits, max_rank=64)
    
    import random
    
    test_cases = [
        ("Random", [random.randint(0, BABY_BEAR_PRIME - 1) for _ in range(N)]),
        ("Sparse (1%)", [random.randint(1, BABY_BEAR_PRIME - 1) if random.random() < 0.01 else 0 for _ in range(N)]),
        ("Low-degree (N/16)", [random.randint(0, BABY_BEAR_PRIME - 1) if i < N // 16 else 0 for i in range(N)]),
        ("Constant", [42] * N),
        ("Linear", [i % BABY_BEAR_PRIME for i in range(N)]),
    ]
    
    print(f"\n{'Input Type':<20} {'Max Rank':<12} {'Compression':<12} {'Correct':<10}")
    print("-" * 60)
    
    for name, x in test_cases:
        result, stats = ntt.forward_qtt(x, return_stats=True)
        
        # Verify correctness
        result_ref = ntt.forward_direct(x)
        correct = result == result_ref
        
        max_rank = max(max(ch['stage_ranks']) for ch in stats['channels'])
        compression = stats['total_compression']
        
        print(f"{name:<20} {max_rank:<12} {compression:<12.2f}x {'✓' if correct else '✗'}")
    
    print()


if __name__ == "__main__":
    validate_rns_qtt_ntt()
    benchmark_compression()
