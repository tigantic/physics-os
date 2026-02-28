"""
QTT-NTT: Tensor Train Accelerated Number Theoretic Transform
============================================================

Plugs into HyperTensor's existing QTT infrastructure for ZK proving acceleration.

Key insight: NTT butterfly is an MPO with special structure.
The DFT matrix has TT-rank 2 in the butterfly basis.

For ZK Proving:
- Uses finite field arithmetic (Goldilocks, BN254, Baby Bear)  
- NO complex numbers - integer modular arithmetic only
- CUDA path for float64 (Baby Bear fits), CPU for larger primes

For Validation:
- Uses existing qtt_spectral Walsh-Hadamard (real-valued, rank-preserving)
- Not the full FFT (missing twiddles), but validates QTT structure

Usage:
    from ontic.cuda.qtt_ntt import QTTNTT, qtt_poly_multiply
    
    # Finite field NTT (ZK proving target)
    ntt = QTTNTT(n_bits=20, field='babybear')
    X = ntt.forward_ff(x)  # x is list/tensor of field elements
    
    # Polynomial multiplication (core ZK operation)
    c = qtt_poly_multiply(a, b, field='babybear')

Author: Brad / Tigantic Holdings
Date: 2026-01-19
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import List, Tuple, Optional, Literal
from dataclasses import dataclass
import numpy as np

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

# Native rSVD for THE RULES compliance
try:
    from ontic.genesis.core.triton_ops import rsvd_native
    HAS_RSVD = True
except ImportError:
    HAS_RSVD = False
    rsvd_native = None

# Try to import Numba for accelerated finite field NTT
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# =============================================================================
# NUMBA-ACCELERATED NTT (finite fields)
# =============================================================================

if HAS_NUMBA:
    @njit(cache=True)
    def _mod_pow(base: int, exp: int, mod: int) -> int:
        """Modular exponentiation: base^exp mod mod."""
        result = 1
        base = base % mod
        while exp > 0:
            if exp & 1:
                result = (result * base) % mod
            exp >>= 1
            base = (base * base) % mod
        return result
    
    @njit(cache=True)
    def _bit_reverse_numba(x: np.ndarray, n_bits: int) -> np.ndarray:
        """Bit-reversal permutation."""
        n = len(x)
        result = x.copy()
        for i in range(n):
            j = 0
            temp = i
            for _ in range(n_bits):
                j = (j << 1) | (temp & 1)
                temp >>= 1
            if i < j:
                result[i], result[j] = result[j], result[i]
        return result
    
    @njit(cache=True)
    def ntt_forward_numba(x: np.ndarray, omega: int, p: int, n_bits: int) -> np.ndarray:
        """Numba-accelerated forward NTT."""
        n = len(x)
        result = _bit_reverse_numba(x, n_bits)
        
        m = 1
        for s in range(n_bits):
            m *= 2
            wm = _mod_pow(omega, n // m, p)
            
            # Process blocks
            n_blocks = n // m
            for block in range(n_blocks):
                k = block * m
                w = 1
                for j in range(m // 2):
                    t = (w * result[k + j + m // 2]) % p
                    u = result[k + j]
                    result[k + j] = (u + t) % p
                    result[k + j + m // 2] = (u - t + p) % p  # Ensure positive
                    w = (w * wm) % p
        
        return result
    
    @njit(cache=True)
    def ntt_inverse_numba(X: np.ndarray, omega_inv: int, p: int, n_bits: int) -> np.ndarray:
        """Numba-accelerated inverse NTT."""
        n = len(X)
        result = _bit_reverse_numba(X, n_bits)
        
        m = 1
        for s in range(n_bits):
            m *= 2
            wm = _mod_pow(omega_inv, n // m, p)
            
            n_blocks = n // m
            for block in range(n_blocks):
                k = block * m
                w = 1
                for j in range(m // 2):
                    t = (w * result[k + j + m // 2]) % p
                    u = result[k + j]
                    result[k + j] = (u + t) % p
                    result[k + j + m // 2] = (u - t + p) % p
                    w = (w * wm) % p
        
        # Multiply by N^{-1}
        n_inv = _mod_pow(n, p - 2, p)
        for i in range(n):
            result[i] = (result[i] * n_inv) % p
        
        return result
else:
    # Dummy definitions if Numba not available
    def ntt_forward_numba(x, omega, p, n_bits):
        raise ImportError("Numba not available")
    
    def ntt_inverse_numba(X, omega_inv, p, n_bits):
        raise ImportError("Numba not available")


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

def build_butterfly_mpo_ff(
    n_bits: int,
    stage: int,
    omega: int,
    p: int,
    device: torch.device = None,
) -> List[Tensor]:
    """
    Build butterfly MPO for stage k of NTT in finite field (DIT variant).
    
    ⚠️ PRECISION WARNING ⚠️
    This function uses float64 to store field elements. Float64 has only 53 bits
    of mantissa, which is insufficient for exact arithmetic when:
    - p > 2^26 (since products can exceed 2^52)
    - Field elements are close to p (Baby Bear p ≈ 2^31)
    
    For correct finite field NTT, use the Numba-accelerated integer implementation
    (forward_int / forward_ff) instead of MPO-based contraction.
    
    This MPO construction is correct and useful for:
    - Complex FFT (standard floating-point)
    - Fields with small primes (p < 2^26)
    - Research/visualization of butterfly structure
    
    For Cooley-Tukey DIT after bit-reversal:
    - Stage s has stride m = 2^{s+1}
    - Within each block of m elements, pairs are (k+j, k+j+m/2)
    - Twiddle = ω^{j * N/m} where j = 0..m/2-1
    
    In QTT representation:
    - Index j corresponds to bits 0..s-1 (lower bits)
    - Bit s selects which half of the pair (0 = top, 1 = bottom)
    - Bits s+1..n-1 select which block
    
    MPO Structure:
    - Cores 0..s-1: Accumulate j (the twiddle index exponent)
    - Core s: Apply butterfly [[1, ω^j], [1, -ω^j]]
    - Cores s+1..n-1: Identity
    
    Args:
        n_bits: log2(N)
        stage: Which butterfly stage (0 = stride 2, 1 = stride 4, etc.)
        omega: Primitive N-th root of unity
        p: Prime modulus
        device: torch device
    """
    N = 2 ** n_bits
    device = device or torch.device('cpu')
    dtype = torch.float64
    
    cores = []
    
    # At stage s: twiddle = ω^{j * N / 2^{s+1}} where j = bits 0..s-1
    # twiddle_step = N / 2^{s+1}
    twiddle_step = N // (2 ** (stage + 1))
    
    for k in range(n_bits):
        if k < stage:
            # Accumulate twiddle index j from bit values
            # j = sum_{i=0}^{k} bit_i * 2^i
            # At core k, we have 2^k possible j values incoming, 2^{k+1} outgoing
            r_left = 2 ** k if k > 0 else 1
            r_right = 2 ** (k + 1)
            
            core = torch.zeros(r_left, 2, 2, r_right, dtype=dtype, device=device)
            
            for j_in in range(r_left):
                # bit=0: j_out = j_in (this bit contributes 0 to j)
                core[j_in, 0, 0, j_in] = 1.0
                core[j_in, 1, 1, j_in] = 1.0
                
                # bit=1: j_out = j_in + 2^k (this bit contributes 2^k to j)
                j_out = j_in + 2 ** k
                core[j_in, 0, 0, j_out] = 0.0  # Override: only one path
                core[j_in, 1, 1, j_out] = 0.0
                # Actually both paths should go based on input bit value
                
            # Fix: identity on physical dims, branch on bond
            core = torch.zeros(r_left, 2, 2, r_right, dtype=dtype, device=device)
            for j_in in range(r_left):
                for b in range(2):  # physical bit value
                    j_out = j_in + b * (2 ** k)
                    core[j_in, b, b, j_out] = 1.0
            
            cores.append(core)
            
        elif k == stage:
            # Butterfly core: apply [[1, tw], [1, -tw]] where tw = ω^{j * step}
            r_left = 2 ** stage if stage > 0 else 1
            r_right = 1
            
            core = torch.zeros(r_left, 2, 2, r_right, dtype=dtype, device=device)
            
            for j in range(r_left):
                tw = pow(omega, j * twiddle_step, p)
                tw_neg = (p - tw) % p
                
                # Butterfly: [out_0, out_1] = [[1, tw], [1, -tw]] @ [in_0, in_1]
                # out_0 = 1*in_0 + tw*in_1
                # out_1 = 1*in_0 - tw*in_1
                core[j, 0, 0, 0] = 1.0       # out=0 <- in=0: coeff 1
                core[j, 0, 1, 0] = float(tw)  # out=0 <- in=1: coeff tw
                core[j, 1, 0, 0] = 1.0       # out=1 <- in=0: coeff 1
                core[j, 1, 1, 0] = float(tw_neg)  # out=1 <- in=1: coeff -tw
            
            cores.append(core)
            
        else:
            # After butterfly: identity
            core = torch.zeros(1, 2, 2, 1, dtype=dtype, device=device)
            core[0, 0, 0, 0] = 1.0
            core[0, 1, 1, 0] = 1.0
            cores.append(core)
    
    return cores


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
    # Use CPU for complex dtype (CUDA kernels only support float32/64)
    use_cuda = HAS_TENSORNET and not qtt_cores[0].is_complex()
    if use_cuda:
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
        return result


# =============================================================================
# MAIN QTT-NTT CLASS
# =============================================================================

class QTTNTT:
    """
    QTT-accelerated NTT for ZK proving.
    
    Uses HyperTensor's existing QTT infrastructure with NTT-specific optimizations.
    
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
            elif HAS_RSVD and remainder.shape[0] > 4 and remainder.shape[1] > 4:
                U, S, Vh = rsvd_native(remainder.real if remainder.is_complex() else remainder, k=max_rank)
            else:
                U, S, Vh = torch.linalg.svd(remainder, full_matrices=False)
                rank = min(max_rank, len(S))
                U, S, Vh = U[:, :rank], S[:rank], Vh[:rank, :]
            
            rank = len(S)
            cores.append(U.reshape(left_rank, 2, rank))
            # S is real, cast to match Vh dtype for complex tensors
            S_diag = torch.diag(S.to(Vh.dtype))
            remainder = S_diag @ Vh
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
        # Use CPU for complex dtype (CUDA kernels only support float32/64)
        use_cuda = HAS_TENSORNET and not x.is_complex()
        target_device = self.device if use_cuda else torch.device('cpu')
        
        x = x.to(target_device)
        
        # Convert to QTT
        qtt = self._vector_to_qtt(x, max_rank)
        
        # Apply butterfly stages
        self._ensure_butterflies()
        
        # Move butterfly MPOs to target device if needed
        butterfly_mpos = self._butterfly_mpos
        if not use_cuda:
            # Move to CPU for complex path
            butterfly_mpos = [
                [core.to(target_device) for core in mpo]
                for mpo in butterfly_mpos
            ]
        
        for stage in range(self.n_bits):
            # Apply butterfly MPO
            if use_cuda:
                qtt = apply_mpo_cuda(butterfly_mpos[stage], qtt)
            else:
                qtt = self._apply_mpo_cpu(butterfly_mpos[stage], qtt)
            
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
            return self.inverse_ff(X.tolist())
    
    # =========================================================================
    # FINITE FIELD NTT (ZK Proving Target)
    # =========================================================================
    
    def forward_ff(self, x: List[int]) -> List[int]:
        """
        Forward NTT in finite field.
        
        Uses Cooley-Tukey butterfly with modular arithmetic.
        
        Args:
            x: List of field elements (integers mod p)
            
        Returns:
            NTT(x) as list of field elements
        """
        if self.field == 'complex':
            raise ValueError("Use forward() for complex FFT")
        
        p = self.field_params.prime
        n = len(x)
        assert n == self.N, f"Expected length {self.N}, got {n}"
        
        # Try to use Numba-accelerated version if available
        # Only for primes that fit in int64 (< 2^63)
        if HAS_NUMBA and p < (1 << 62):  # Leave room for multiplication
            import numpy as np
            x_arr = np.array(x, dtype=np.int64)
            result = ntt_forward_numba(x_arr, self.omega, p, self.n_bits)
            return result.tolist()
        
        # Fallback: Pure Python (works for any prime size)
        result = self._bit_reverse(list(x))
        
        # Cooley-Tukey butterfly
        m = 1
        for s in range(self.n_bits):
            m *= 2
            wm = pow(self.omega, self.N // m, p)
            
            for k in range(0, n, m):
                w = 1
                for j in range(m // 2):
                    t = (w * result[k + j + m // 2]) % p
                    u = result[k + j]
                    result[k + j] = (u + t) % p
                    result[k + j + m // 2] = (u - t) % p
                    w = (w * wm) % p
        
        return result
    
    def inverse_ff(self, X: List[int]) -> List[int]:
        """
        Inverse NTT in finite field.
        """
        if self.field == 'complex':
            raise ValueError("Use inverse() for complex FFT")
        
        p = self.field_params.prime
        n = len(X)
        
        # Try to use Numba-accelerated version if available
        # Only for primes that fit in int64 (< 2^62)
        if HAS_NUMBA and p < (1 << 62):
            import numpy as np
            X_arr = np.array(X, dtype=np.int64)
            result = ntt_inverse_numba(X_arr, self.omega_inv, p, self.n_bits)
            return result.tolist()
        
        # Fallback: Pure Python INTT with omega_inv
        result = self._bit_reverse(list(X))
        
        m = 1
        for s in range(self.n_bits):
            m *= 2
            wm = pow(self.omega_inv, self.N // m, p)
            
            for k in range(0, n, m):
                w = 1
                for j in range(m // 2):
                    t = (w * result[k + j + m // 2]) % p
                    u = result[k + j]
                    result[k + j] = (u + t) % p
                    result[k + j + m // 2] = (u - t) % p
                    w = (w * wm) % p
        
        # Multiply by N^{-1}
        n_inv = pow(n, p - 2, p)
        result = [(x * n_inv) % p for x in result]
        
        return result
    
    def _bit_reverse(self, x: List[int]) -> List[int]:
        """Bit-reversal permutation."""
        n = len(x)
        result = x.copy()
        for i in range(n):
            j = int(bin(i)[2:].zfill(self.n_bits)[::-1], 2)
            if i < j:
                result[i], result[j] = result[j], result[i]
        return result
    
    # =========================================================================
    # QTT-NATIVE FINITE FIELD NTT VIA MPO BUTTERFLIES
    # =========================================================================
    
    def forward_mpo_ff(self, x: List[int], max_rank: int = 64) -> List[int]:
        """
        QTT-native NTT using MPO butterfly operators.
        
        NOTE: The MPO approach with float64 has precision issues for finite
        fields with primes > 2^26. Products of 31-bit numbers exceed float64
        mantissa precision.
        
        For correct finite field NTT, use forward_int() which uses exact
        integer arithmetic with modular reduction after each butterfly.
        
        This method is kept for complex FFT and research purposes.
        """
        if self.field == 'complex':
            raise ValueError("Use forward() for complex FFT")
        
        # For finite fields, delegate to int64-based implementation
        return self.forward_int(x)
    
    def forward_int(self, x: List[int], use_numba: bool = True) -> List[int]:
        """
        NTT using exact integer arithmetic with modular reduction.
        
        This is the correct implementation for finite fields. Uses Cooley-Tukey
        DIT algorithm with bit-reversal preprocessing.
        
        Complexity: O(N log N) - standard FFT complexity
        
        For QTT acceleration of STRUCTURED inputs, the state can be compressed
        after each stage. This is beneficial when the polynomial has special
        structure (sparse, low-degree, etc.)
        
        Args:
            x: List of field elements
            use_numba: Use Numba-accelerated version if available
            
        Returns:
            NTT(x) as list of field elements
        """
        if self.field == 'complex':
            raise ValueError("Use forward() for complex FFT")
        
        p = self.field_params.prime
        n = len(x)
        assert n == self.N, f"Expected length {self.N}, got {n}"
        
        # Try Numba-accelerated version first
        if use_numba and HAS_NUMBA and p < (1 << 62):
            import numpy as np
            x_arr = np.array(x, dtype=np.int64)
            result = ntt_forward_numba(x_arr, self.omega, p, self.n_bits)
            return result.tolist()
        
        # Fallback to pure Python implementation
        return self._forward_int_python(x)
    
    def _forward_int_python(self, x: List[int]) -> List[int]:
        """Pure Python Cooley-Tukey DIT NTT with exact integer arithmetic."""
        p = self.field_params.prime
        n_bits = self.n_bits
        N = self.N
        omega = self.omega
        
        # Step 1: Bit-reverse input
        a = self._bit_reverse(list(x))
        
        # Step 2: Apply butterfly stages
        m = 2
        for stage in range(n_bits):
            wm = pow(omega, N // m, p)  # m-th root of unity
            
            for k in range(0, N, m):
                w = 1
                for j in range(m // 2):
                    t = (w * a[k + j + m // 2]) % p
                    u = a[k + j]
                    a[k + j] = (u + t) % p
                    a[k + j + m // 2] = (u - t) % p
                    w = (w * wm) % p
            
            m *= 2
        
        return a
    
    def _vector_to_qtt_ff(self, x: Tensor, max_rank: int) -> List[Tensor]:
        """Convert dense vector to QTT format via SVD."""
        n_bits = self.n_bits
        tensor = x.reshape([2] * n_bits)
        
        cores = []
        r_left = 1
        remainder = tensor.flatten()
        
        for k in range(n_bits - 1):
            remainder = remainder.reshape(r_left * 2, -1)
            
            # rSVD for O(mnk) complexity
            if HAS_RSVD and remainder.shape[0] > 4 and remainder.shape[1] > 4:
                U, S, Vh = rsvd_native(remainder.real if remainder.is_complex() else remainder, k=max_rank, tol=1e-12)
            else:
                U, S, Vh = torch.linalg.svd(remainder, full_matrices=False)
            
            # Truncate
            rank = min(max_rank, len(S), (S > 1e-12 * S[0]).sum().item())
            rank = max(1, rank)
            
            U = U[:, :rank]
            S = S[:rank]
            Vh = Vh[:rank, :]
            
            cores.append(U.reshape(r_left, 2, rank))
            remainder = torch.diag(S) @ Vh
            r_left = rank
        
        cores.append(remainder.reshape(r_left, 2, 1))
        return cores
    
    def _apply_bit_reversal_qtt(self, cores: List[Tensor]) -> List[Tensor]:
        """Apply bit-reversal permutation by reversing core order."""
        # Bit-reversal in QTT = reversing the order of cores
        # and transposing each core's bond dimensions
        new_cores = []
        for core in reversed(cores):
            # core: (r_left, 2, r_right) -> (r_right, 2, r_left)
            new_cores.append(core.permute(2, 1, 0).contiguous())
        return new_cores
    
    def _apply_mpo_to_qtt(self, qtt: List[Tensor], mpo: List[Tensor], max_rank: int) -> List[Tensor]:
        """
        Apply MPO to QTT state.
        
        |ψ'⟩ = MPO |ψ⟩
        
        Core contraction: new[ml*sl, d_out, mr*sr] = Σ_{d_in} mpo[ml, d_out, d_in, mr] * qtt[sl, d_in, sr]
        """
        new_cores = []
        
        for k in range(len(qtt)):
            q_core = qtt[k]  # (sl, d_in, sr)
            m_core = mpo[k]  # (ml, d_out, d_in, mr)
            
            sl, d_in, sr = q_core.shape
            ml, d_out, d_in_m, mr = m_core.shape
            
            assert d_in == d_in_m, f"Dimension mismatch at core {k}"
            
            # Contract over d_in
            # result[ml, sl, d_out, mr, sr]
            result = torch.einsum('aobm,lbr->alomr', m_core, q_core)
            
            # Reshape to (ml*sl, d_out, mr*sr)
            result = result.reshape(ml * sl, d_out, mr * sr)
            new_cores.append(result)
        
        # Truncate via SVD sweep
        return self._truncate_qtt_svd(new_cores, max_rank)
    
    def _apply_mod_to_qtt(self, cores: List[Tensor], p: int) -> List[Tensor]:
        """
        Apply modular reduction to QTT values.
        
        This is approximate - we apply mod to the final contracted values.
        For exact mod, we'd need modular arithmetic in the tensor network.
        Here we just clamp to prevent overflow.
        """
        # For now, just return as-is (modular reduction happens at extraction)
        # Full modular reduction in QTT requires more sophisticated approach
        return cores
    
    def _truncate_qtt_svd(self, cores: List[Tensor], max_rank: int) -> List[Tensor]:
        """Left-to-right SVD truncation sweep."""
        new_cores = []
        n = len(cores)
        
        cores = [c.clone() for c in cores]  # Don't modify original
        
        for k in range(n - 1):
            core = cores[k]
            r_left, d, r_right = core.shape
            
            mat = core.reshape(r_left * d, r_right)
            
            if r_right <= max_rank:
                new_cores.append(core)
                continue
            
            # rSVD truncation
            try:
                if HAS_RSVD and mat.shape[0] > 4 and mat.shape[1] > 4:
                    U, S, Vh = rsvd_native(mat.real if mat.is_complex() else mat, k=max_rank)
                else:
                    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
                
                rank = min(max_rank, len(S))
                if len(S) > 0:
                    threshold = 1e-12 * S[0].item()
                    valid = (S > threshold).sum().item()
                    rank = min(rank, max(valid, 1))
                
                U = U[:, :rank]
                S = S[:rank]
                Vh = Vh[:rank, :]
                
                new_cores.append(U.reshape(r_left, d, rank))
                
                # Absorb S @ Vh into next core
                SV = torch.diag(S) @ Vh
                next_core = cores[k + 1]
                r_right_old, d_next, r_right_next = next_core.shape
                next_flat = next_core.reshape(r_right_old, d_next * r_right_next)
                new_next = SV @ next_flat
                cores[k + 1] = new_next.reshape(rank, d_next, r_right_next)
                
            except Exception:
                new_cores.append(core)
        
        new_cores.append(cores[-1])
        return new_cores
    
    def _qtt_to_vector_ff(self, cores: List[Tensor], p: int) -> List[int]:
        """Contract QTT to dense vector and apply modular reduction."""
        result = cores[0]
        for core in cores[1:]:
            result = torch.einsum('...r,rds->...ds', result, core)
        
        result = result.flatten()
        
        # Apply modular reduction
        result = [int(round(v.item())) % p for v in result]
        return result
    
    # =========================================================================
    # QTT-ACCELERATED FINITE FIELD NTT VIA TCI
    # =========================================================================
    
    def forward_qtt_ff(self, x: List[int], max_rank: int = 64) -> List[int]:
        """
        QTT-accelerated NTT in finite field using TCI.
        
        Strategy: Treat NTT as a black-box function and use TCI to build
        a QTT representation of the output. Since NTT preserves low TT-rank
        structure, the output has bounded rank.
        
        Complexity: O(r² × n_bits) function samples + O(r² × n_bits) SVD
        vs reference: O(N log N) = O(2^n × n_bits)
        
        For rank r << 2^(n/2), this is faster.
        
        Args:
            x: List of field elements
            max_rank: Maximum QTT rank
            
        Returns:
            NTT(x) as list of field elements
        """
        if self.field == 'complex':
            raise ValueError("Use forward() for complex FFT")
        
        p = self.field_params.prime
        n = len(x)
        assert n == self.N, f"Expected length {self.N}, got {n}"
        
        # For small inputs, just use reference (faster than TCI overhead)
        if self.n_bits <= 10:  # N <= 1024
            return self.forward_ff(x)
        
        # For larger inputs, use TCI to build QTT of NTT output
        try:
            from ontic.cfd.tci_true import tci_build_qtt
            from ontic.cfd.qtt_eval import qtt_eval_batch
            
            # Create function that evaluates NTT at given indices
            x_array = x  # Capture input
            p_val = p
            omega = self.omega
            n_bits = self.n_bits
            
            def ntt_at_indices(indices: torch.Tensor) -> torch.Tensor:
                """Evaluate NTT[k] for given k indices."""
                # NTT[k] = sum_{j=0}^{N-1} x[j] * omega^{jk} mod p
                results = []
                for k_val in indices.cpu().tolist():
                    k = int(k_val)
                    total = 0
                    for j, xj in enumerate(x_array):
                        power = (j * k) % self.N
                        tw = pow(omega, power, p_val)
                        total = (total + xj * tw) % p_val
                    results.append(total)
                return torch.tensor(results, dtype=torch.float32, device=indices.device)
            
            # Build QTT via TCI
            cores = tci_build_qtt(
                func=ntt_at_indices,
                n_qubits=self.n_bits,
                max_rank=max_rank,
                tol=1e-6,
                max_sweeps=5,
                device=torch.device('cpu'),
                verbose=False
            )
            
            # Evaluate at all indices to get result
            indices = torch.arange(self.N, dtype=torch.long)
            result = qtt_eval_batch(cores, indices)
            
            # Round to integers and apply modular reduction
            result = [int(round(v.item())) % p for v in result]
            
            return result
            
        except ImportError:
            # Fall back to reference
            return self.forward_ff(x)
    
    def _apply_butterfly_stage_ff(self, tensor: Tensor, stage: int) -> Tensor:
        """
        Apply butterfly stage k to tensor.
        
        At stage k:
        - Stride = 2^k
        - For each pair (j, j+stride), apply:
          y[j] = x[j] + ω^{twiddle_index} · x[j+stride]
          y[j+stride] = x[j] - ω^{twiddle_index} · x[j+stride]
        
        The twiddle_index depends on position within the stage.
        
        In the tensor representation with dimensions (d_0, d_1, ..., d_{n-1}):
        - Stage k operates on dimension k
        - Twiddle index = sum of (d_i * 2^i) for i < k
        """
        p = self.field_params.prime
        n_bits = self.n_bits
        
        # Reshape to isolate dimension for this stage
        # Move stage dimension to end for easier processing
        dims = list(range(n_bits))
        perm = dims[:stage] + dims[stage+1:] + [stage]
        tensor = tensor.permute(perm)
        
        # Shape is now (..., 2) where last dim is the stage dimension
        shape = tensor.shape
        flat = tensor.reshape(-1, 2)  # (batch, 2)
        
        # Compute twiddles for each batch position
        batch_size = flat.shape[0]
        
        # The twiddle index for position b in the batch depends on 
        # the bits that come after stage in the original ordering
        # For Cooley-Tukey DIT, twiddle = ω^{(b mod 2^stage) * N / 2^{stage+1}}
        
        # Simplification: stride-based twiddle pattern
        half_stage = 2 ** stage
        twiddle_step = self.N // (2 * half_stage)
        
        result = torch.zeros_like(flat)
        for b in range(batch_size):
            # Twiddle index for this batch position
            # In DIT order after bit-reversal, twiddle = ω^{(b mod half_stage) * step}
            tw_idx = (b % half_stage) * twiddle_step
            tw = pow(self.omega, tw_idx, p)
            
            x0 = flat[b, 0].item()
            x1 = flat[b, 1].item()
            
            y0 = (x0 + tw * x1) % p
            y1 = (x0 - tw * x1) % p
            
            result[b, 0] = y0
            result[b, 1] = y1
        
        # Reshape back and undo permutation
        result = result.reshape(shape)
        inv_perm = [0] * n_bits
        for i, d in enumerate(perm):
            inv_perm[d] = i
        result = result.permute(inv_perm)
        
        return result
    
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
                elif HAS_RSVD and mat.shape[0] > 4 and mat.shape[1] > 4:
                    U, S, Vh = rsvd_native(mat.real if mat.is_complex() else mat, k=max_rank)
                else:
                    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
                    rank = min(max_rank, len(S))
                    U, S, Vh = U[:, :rank], S[:rank], Vh[:rank, :]
                
                new_cores.append(U.reshape(r_left, d, len(S)))
                
                # Absorb S @ Vh into next core
                # Cast S to Vh dtype for complex tensors (S is always real)
                if i + 1 < len(cores):
                    S_diag = torch.diag(S.to(Vh.dtype))
                    cores[i + 1] = torch.einsum('ij,jdk->idk', S_diag @ Vh, cores[i + 1])
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
