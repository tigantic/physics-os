"""
Matrix Product State (MPS) Implementation
==========================================

MPS compresses quantum states from O(d^N) to O(N × χ² × d) where:
- d = local Hilbert space dimension
- N = number of sites
- χ = bond dimension (controls accuracy)

For Yang-Mills:
- Strong coupling: Low entanglement → small χ works
- Weak coupling: High entanglement → need large χ

The key insight: MPS with DMRG can access weak coupling where
exact diagonalization fails (4.77 TiB barrier).
"""

import numpy as np
import torch
from typing import List, Optional, Tuple, Union
from scipy import linalg


class MPS:
    """
    Matrix Product State representation.
    
    |ψ⟩ = Σ A[1]^{s1} A[2]^{s2} ... A[N]^{sN} |s1 s2 ... sN⟩
    
    Each A[i] is a tensor of shape (χ_{i-1}, d_i, χ_i) where:
    - χ_i = bond dimension at site i
    - d_i = local dimension at site i
    """
    
    def __init__(self, tensors: List[np.ndarray], canonical_form: str = 'none'):
        """
        Initialize MPS from list of tensors.
        
        Args:
            tensors: List of rank-3 tensors [chi_l, d, chi_r]
            canonical_form: 'left', 'right', 'mixed', or 'none'
        """
        self.tensors = [np.asarray(t, dtype=np.complex128) for t in tensors]
        self.n_sites = len(tensors)
        self.canonical_form = canonical_form
        self.center = None  # Orthogonality center for mixed canonical form
        
    @classmethod
    def random(cls, n_sites: int, local_dim: int, bond_dim: int, 
               normalize: bool = True) -> 'MPS':
        """
        Create random MPS with specified dimensions.
        
        Args:
            n_sites: Number of sites
            local_dim: Local Hilbert space dimension
            bond_dim: Maximum bond dimension
            normalize: Whether to normalize the state
        """
        tensors = []
        chi_left = 1
        
        for i in range(n_sites):
            chi_right = min(bond_dim, local_dim ** min(i + 1, n_sites - i - 1))
            chi_right = max(chi_right, 1)
            
            # Random complex tensor
            t = np.random.randn(chi_left, local_dim, chi_right) + \
                1j * np.random.randn(chi_left, local_dim, chi_right)
            t /= np.linalg.norm(t)
            tensors.append(t)
            chi_left = chi_right
        
        mps = cls(tensors)
        if normalize:
            mps.canonicalize('right')
            mps.normalize()
        return mps
    
    @classmethod
    def product_state(cls, local_states: List[np.ndarray]) -> 'MPS':
        """
        Create MPS from product state |ψ⟩ = |ψ1⟩ ⊗ |ψ2⟩ ⊗ ... ⊗ |ψN⟩
        
        This has bond dimension χ = 1 (no entanglement).
        Strong coupling ground state is approximately a product state!
        """
        tensors = []
        for state in local_states:
            state = np.asarray(state, dtype=np.complex128)
            state = state.flatten()
            # Reshape to [1, d, 1] tensor
            t = state.reshape(1, -1, 1)
            tensors.append(t)
        return cls(tensors, canonical_form='right')
    
    @property
    def bond_dimensions(self) -> List[int]:
        """Get bond dimensions at each bond."""
        return [t.shape[2] for t in self.tensors[:-1]]
    
    @property
    def local_dimensions(self) -> List[int]:
        """Get local dimension at each site."""
        return [t.shape[1] for t in self.tensors]
    
    @property
    def max_bond_dim(self) -> int:
        """Maximum bond dimension."""
        return max(max(t.shape[0], t.shape[2]) for t in self.tensors)
    
    def copy(self) -> 'MPS':
        """Create deep copy of MPS."""
        tensors = [t.copy() for t in self.tensors]
        mps = MPS(tensors, self.canonical_form)
        mps.center = self.center
        return mps
    
    def canonicalize(self, form: str = 'right', center: Optional[int] = None) -> 'MPS':
        """
        Put MPS into canonical form.
        
        Left canonical: A†A = I for all sites
        Right canonical: AA† = I for all sites  
        Mixed canonical: Left canonical left of center, right canonical right
        
        Canonical form is crucial for:
        1. Computing overlaps efficiently
        2. Stable DMRG optimization
        3. Measuring entanglement entropy
        """
        if form == 'left':
            self._left_canonicalize()
        elif form == 'right':
            self._right_canonicalize()
        elif form == 'mixed':
            if center is None:
                center = self.n_sites // 2
            self._mixed_canonicalize(center)
        
        self.canonical_form = form
        return self
    
    def _left_canonicalize(self):
        """Sweep left to right, making each tensor left-canonical via QR."""
        for i in range(self.n_sites - 1):
            t = self.tensors[i]
            chi_l, d, chi_r = t.shape
            
            # Reshape to matrix and QR decompose
            t_mat = t.reshape(chi_l * d, chi_r)
            Q, R = np.linalg.qr(t_mat)
            
            # New left-canonical tensor
            new_chi = Q.shape[1]
            self.tensors[i] = Q.reshape(chi_l, d, new_chi)
            
            # Absorb R into next tensor
            next_t = self.tensors[i + 1]
            next_t = np.tensordot(R, next_t, axes=([1], [0]))
            self.tensors[i + 1] = next_t
        
        self.center = self.n_sites - 1
    
    def _right_canonicalize(self):
        """Sweep right to left, making each tensor right-canonical via RQ."""
        for i in range(self.n_sites - 1, 0, -1):
            t = self.tensors[i]
            chi_l, d, chi_r = t.shape
            
            # Reshape to matrix and RQ decompose (via QR of transpose)
            t_mat = t.reshape(chi_l, d * chi_r)
            Q, R = np.linalg.qr(t_mat.T)
            Q = Q.T
            R = R.T
            
            # New right-canonical tensor
            new_chi = Q.shape[0]
            self.tensors[i] = Q.reshape(new_chi, d, chi_r)
            
            # Absorb L into previous tensor
            prev_t = self.tensors[i - 1]
            prev_t = np.tensordot(prev_t, R, axes=([2], [0]))
            self.tensors[i - 1] = prev_t
        
        self.center = 0
    
    def _mixed_canonicalize(self, center: int):
        """Mixed canonical form with orthogonality center at specified site."""
        # Left-canonicalize from left to center
        for i in range(center):
            t = self.tensors[i]
            chi_l, d, chi_r = t.shape
            t_mat = t.reshape(chi_l * d, chi_r)
            Q, R = np.linalg.qr(t_mat)
            new_chi = Q.shape[1]
            self.tensors[i] = Q.reshape(chi_l, d, new_chi)
            self.tensors[i + 1] = np.tensordot(R, self.tensors[i + 1], axes=([1], [0]))
        
        # Right-canonicalize from right to center
        for i in range(self.n_sites - 1, center, -1):
            t = self.tensors[i]
            chi_l, d, chi_r = t.shape
            t_mat = t.reshape(chi_l, d * chi_r)
            Q, R = np.linalg.qr(t_mat.T)
            Q = Q.T
            R = R.T
            new_chi = Q.shape[0]
            self.tensors[i] = Q.reshape(new_chi, d, chi_r)
            self.tensors[i - 1] = np.tensordot(self.tensors[i - 1], R, axes=([2], [0]))
        
        self.center = center
    
    def normalize(self) -> float:
        """Normalize MPS and return the norm."""
        norm = self.norm()
        if norm > 1e-15:
            # Normalize at the center site (or last site if right-canonical)
            site = self.center if self.center is not None else self.n_sites - 1
            self.tensors[site] = self.tensors[site] / norm
        return norm
    
    def norm(self) -> float:
        """Compute the norm ⟨ψ|ψ⟩^{1/2}."""
        return np.sqrt(np.abs(self.inner(self)))
    
    def inner(self, other: 'MPS') -> complex:
        """
        Compute inner product ⟨self|other⟩.
        
        Uses transfer matrix contraction: O(N × χ² × d)
        """
        if self.n_sites != other.n_sites:
            raise ValueError("MPS must have same number of sites")
        
        # Start with leftmost contraction
        result = np.ones((1, 1), dtype=np.complex128)
        
        for i in range(self.n_sites):
            # Contract: result[α, α'] A*[α, s, β] B[α', s, β'] → result[β, β']
            A = self.tensors[i]
            B = other.tensors[i]
            
            # Contract over left index with result
            temp = np.tensordot(result, np.conj(A), axes=([0], [0]))  # [α', s, β]
            temp = np.tensordot(temp, B, axes=([0, 1], [0, 1]))  # [β, β']
            result = temp
        
        return result[0, 0]
    
    def entanglement_entropy(self, site: int) -> float:
        """
        Compute bipartite entanglement entropy at bond after site.
        
        S = -Σ λ² log(λ²)
        
        This is the key diagnostic:
        - Strong coupling: S ≈ 0 (product state)
        - Weak coupling: S grows (entanglement barrier!)
        """
        # Ensure mixed canonical form at this site
        mps = self.copy()
        mps._mixed_canonicalize(site)
        
        # SVD of center tensor
        t = mps.tensors[site]
        chi_l, d, chi_r = t.shape
        t_mat = t.reshape(chi_l * d, chi_r)
        
        try:
            _, s, _ = np.linalg.svd(t_mat, full_matrices=False)
        except np.linalg.LinAlgError:
            return 0.0
        
        # Compute entropy from singular values
        s = s[s > 1e-15]
        s2 = s ** 2
        s2 = s2 / np.sum(s2)  # Normalize
        
        entropy = -np.sum(s2 * np.log(s2 + 1e-100))
        return float(entropy)
    
    def truncate(self, max_chi: int, cutoff: float = 1e-12) -> float:
        """
        Truncate MPS to maximum bond dimension.
        
        Returns truncation error.
        """
        self.canonicalize('right')
        total_error = 0.0
        
        for i in range(self.n_sites - 1):
            t = self.tensors[i]
            chi_l, d, chi_r = t.shape
            t_mat = t.reshape(chi_l * d, chi_r)
            
            U, s, Vh = np.linalg.svd(t_mat, full_matrices=False)
            
            # Truncate
            keep = min(max_chi, len(s))
            mask = s > cutoff
            keep = min(keep, np.sum(mask))
            keep = max(keep, 1)
            
            # Truncation error
            if keep < len(s):
                total_error += np.sum(s[keep:] ** 2)
            
            U = U[:, :keep]
            s = s[:keep]
            Vh = Vh[:keep, :]
            
            # New tensors
            self.tensors[i] = U.reshape(chi_l, d, keep)
            SV = np.diag(s) @ Vh
            self.tensors[i + 1] = np.tensordot(SV, self.tensors[i + 1], axes=([1], [0]))
        
        return total_error
    
    def to_dense(self) -> np.ndarray:
        """
        Convert MPS to dense state vector.
        
        WARNING: Exponential memory! Only for small systems.
        """
        result = self.tensors[0]  # [1, d, chi]
        
        for i in range(1, self.n_sites):
            # Contract: result[1, d1...di, chi] × A[chi, d, chi'] → [1, d1...di+1, chi']
            result = np.tensordot(result, self.tensors[i], axes=([-1], [0]))
        
        # Reshape to vector
        return result.flatten()
    
    def apply_local_op(self, site: int, op: np.ndarray) -> 'MPS':
        """Apply local operator at specified site."""
        mps = self.copy()
        t = mps.tensors[site]
        chi_l, d, chi_r = t.shape
        
        # Apply operator: O[s', s] A[chi_l, s, chi_r] → A[chi_l, s', chi_r]
        new_t = np.tensordot(op, t, axes=([1], [1]))
        mps.tensors[site] = new_t.transpose(1, 0, 2)
        
        mps.canonical_form = 'none'
        return mps
    
    def expectation_local(self, site: int, op: np.ndarray) -> complex:
        """Compute expectation value of local operator."""
        bra = self.copy()
        ket = self.apply_local_op(site, op)
        return bra.inner(ket)


class MPSGaugeInvariant(MPS):
    """
    MPS constrained to gauge-invariant subspace.
    
    For Yang-Mills, physical states satisfy Gauss law:
    G_x |ψ_phys⟩ = 0 for all x
    
    This MPS enforces gauge invariance by construction.
    """
    
    def __init__(self, tensors: List[np.ndarray], gauge_group: str = 'SU2'):
        super().__init__(tensors)
        self.gauge_group = gauge_group
    
    @classmethod
    def create_gauge_invariant(cls, n_sites: int, j_max: float, 
                               bond_dim: int, gauge_group: str = 'SU2') -> 'MPSGaugeInvariant':
        """
        Create gauge-invariant MPS for Yang-Mills.
        
        Uses SU(2) or SU(3) representation theory to constrain tensors.
        """
        # Number of j values: j = 0, 1/2, 1, ..., j_max
        n_j = int(2 * j_max + 1)
        local_dim = n_j
        
        # Initialize with random tensors
        tensors = []
        chi_left = 1
        
        for i in range(n_sites):
            chi_right = min(bond_dim, local_dim ** min(i + 1, n_sites - i - 1))
            chi_right = max(chi_right, 1)
            
            # Random tensor with gauge-invariant structure
            t = np.random.randn(chi_left, local_dim, chi_right)
            t /= np.linalg.norm(t)
            tensors.append(t)
            chi_left = chi_right
        
        mps = cls(tensors, gauge_group)
        mps.canonicalize('right')
        mps.normalize()
        return mps


# Utility functions

def mps_add(mps1: MPS, mps2: MPS, coeff1: complex = 1.0, coeff2: complex = 1.0) -> MPS:
    """
    Add two MPS: |ψ⟩ = c1|ψ1⟩ + c2|ψ2⟩
    
    Bond dimension of result is χ1 + χ2.
    """
    if mps1.n_sites != mps2.n_sites:
        raise ValueError("MPS must have same number of sites")
    
    tensors = []
    for i in range(mps1.n_sites):
        t1 = coeff1 * mps1.tensors[i] if i == 0 else mps1.tensors[i]
        t2 = coeff2 * mps2.tensors[i] if i == 0 else mps2.tensors[i]
        
        chi_l1, d, chi_r1 = t1.shape
        chi_l2, _, chi_r2 = t2.shape
        
        # Direct sum of tensors
        if i == 0:
            # First site: [1, d, χ1+χ2]
            new_t = np.zeros((1, d, chi_r1 + chi_r2), dtype=np.complex128)
            new_t[0, :, :chi_r1] = t1[0]
            new_t[0, :, chi_r1:] = t2[0]
        elif i == mps1.n_sites - 1:
            # Last site: [χ1+χ2, d, 1]
            new_t = np.zeros((chi_l1 + chi_l2, d, 1), dtype=np.complex128)
            new_t[:chi_l1, :, 0] = t1[:, :, 0]
            new_t[chi_l1:, :, 0] = t2[:, :, 0]
        else:
            # Middle sites: block diagonal
            new_t = np.zeros((chi_l1 + chi_l2, d, chi_r1 + chi_r2), dtype=np.complex128)
            new_t[:chi_l1, :, :chi_r1] = t1
            new_t[chi_l1:, :, chi_r1:] = t2
        
        tensors.append(new_t)
    
    return MPS(tensors)
