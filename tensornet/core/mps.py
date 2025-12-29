"""
Matrix Product State (MPS)
==========================

Core MPS class for representing 1D quantum states and classical fields.

Tensor convention:
    A[i] : (χ_left, d, χ_right)
    
    For a chain of L sites with physical dimension d and bond dimension χ:
    |ψ⟩ = Σ A[0]_{1,σ₀,α₀} A[1]_{α₀,σ₁,α₁} ... A[L-1]_{α_{L-2},σ_{L-1},1} |σ₀σ₁...σ_{L-1}⟩
"""

from __future__ import annotations
from typing import Optional, List, Sequence
import torch
from torch import Tensor
import math

from tensornet.core.decompositions import svd_truncated, qr_positive


class MPS:
    """
    Matrix Product State representation.
    
    Attributes:
        tensors: List of tensors A[i] with shape (χ_left, d, χ_right)
        L: Number of sites
        d: Physical dimension (assumed uniform)
        
    Example:
        >>> mps = MPS.random(L=10, d=2, chi=32)
        >>> print(f"Norm: {mps.norm():.6f}")
        >>> mps.canonicalize_left_()
        >>> entropy = mps.entropy(bond=4)
    """
    
    def __init__(self, tensors: List[Tensor]):
        """
        Initialize MPS from list of tensors.
        
        Args:
            tensors: List of tensors with shape (χ_left, d, χ_right)
        """
        self.tensors = tensors
        self._canonical_center: Optional[int] = None
    
    @property
    def L(self) -> int:
        """Number of sites."""
        return len(self.tensors)
    
    @property
    def d(self) -> int:
        """Physical dimension (from first site)."""
        return self.tensors[0].shape[1]
    
    @property
    def chi(self) -> int:
        """Maximum bond dimension."""
        return max(
            max(t.shape[0], t.shape[2]) for t in self.tensors
        )
    
    @property
    def dtype(self) -> torch.dtype:
        """Data type of tensors."""
        return self.tensors[0].dtype
    
    @property
    def device(self) -> torch.device:
        """Device of tensors."""
        return self.tensors[0].device
    
    def bond_dims(self) -> List[int]:
        """Return list of bond dimensions [χ₀, χ₁, ..., χ_{L-1}]."""
        return [self.tensors[i].shape[2] for i in range(self.L - 1)]
    
    @classmethod
    def random(
        cls,
        L: int,
        d: int,
        chi: int,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
        normalize: bool = True,
    ) -> MPS:
        """
        Create random MPS with given dimensions.
        
        Args:
            L: Number of sites
            d: Physical dimension
            chi: Bond dimension
            dtype: Data type
            device: Device
            normalize: If True, normalize the state
            
        Returns:
            Random MPS
        """
        if device is None:
            device = torch.device('cpu')
        
        tensors = []
        for i in range(L):
            chi_left = 1 if i == 0 else min(chi, d**i, d**(L-i))
            chi_right = 1 if i == L-1 else min(chi, d**(i+1), d**(L-i-1))
            
            A = torch.randn(chi_left, d, chi_right, dtype=dtype, device=device)
            tensors.append(A)
        
        mps = cls(tensors)
        if normalize:
            mps.normalize_()
        return mps
    
    @classmethod
    def from_tensor(
        cls,
        tensor: Tensor,
        chi_max: Optional[int] = None,
        cutoff: float = 1e-14,
    ) -> MPS:
        """
        Convert dense tensor to MPS via successive SVD.
        
        Args:
            tensor: Dense tensor of shape (d, d, ..., d)
            chi_max: Maximum bond dimension
            cutoff: SVD cutoff
            
        Returns:
            MPS approximation
        """
        L = tensor.ndim
        d = tensor.shape[0]
        dtype = tensor.dtype
        device = tensor.device
        
        tensors = []
        current = tensor.reshape(d, -1)
        
        for i in range(L - 1):
            U, S, Vh = svd_truncated(current, chi_max=chi_max, cutoff=cutoff)
            
            chi = len(S)
            chi_left = 1 if i == 0 else tensors[-1].shape[2]
            
            # Reshape U to (chi_left, d, chi)
            A = U.reshape(chi_left, d, chi)
            tensors.append(A)
            
            # Prepare next iteration
            current = torch.diag(S) @ Vh
            remaining = L - i - 1
            if remaining > 1:
                current = current.reshape(chi * d, -1)
            else:
                current = current.reshape(chi, d, 1)
        
        # Last tensor
        tensors.append(current)
        
        return cls(tensors)
    
    def to_tensor(self) -> Tensor:
        """
        Contract MPS to dense tensor.
        
        Warning: Exponential in L. Only use for small systems.
        
        Returns:
            Dense tensor of shape (d, d, ..., d)
        """
        result = self.tensors[0]  # (1, d, χ)
        
        for i in range(1, self.L):
            # result: (..., χ_prev), A[i]: (χ_prev, d, χ_next)
            result = torch.einsum('...i,idj->...dj', result, self.tensors[i])
        
        # Remove boundary bonds
        result = result.squeeze(0).squeeze(-1)
        return result
    
    def copy(self) -> MPS:
        """Return deep copy of MPS."""
        return MPS([t.clone() for t in self.tensors])
    
    def norm(self) -> Tensor:
        """
        Compute norm ⟨ψ|ψ⟩^{1/2}.
        
        Returns:
            Scalar tensor with norm
        """
        # Contract from left
        env = torch.ones(1, 1, dtype=self.dtype, device=self.device)
        
        for A in self.tensors:
            # env: (χ, χ'), A: (χ, d, χ_next), A*: (χ', d, χ'_next)
            env = torch.einsum('ij,idk,jdl->kl', env, A, A.conj())
        
        return torch.sqrt(env.squeeze())
    
    def normalize_(self) -> MPS:
        """Normalize MPS in-place. Returns self."""
        n = self.norm()
        if n > 0:
            # Distribute normalization across all tensors
            factor = n ** (1.0 / self.L)
            for i in range(self.L):
                self.tensors[i] = self.tensors[i] / factor
        return self
    
    def canonicalize_left_(self) -> MPS:
        """
        Left-canonicalize MPS in-place.
        
        After this, A[i]^† @ A[i] = I for all i < L-1.
        The norm is absorbed into the last tensor.
        
        Returns:
            self
        """
        for i in range(self.L - 1):
            A = self.tensors[i]
            chi_l, d, chi_r = A.shape
            
            # Reshape to matrix and QR
            A_mat = A.reshape(chi_l * d, chi_r)
            Q, R = qr_positive(A_mat)
            
            # Update tensors
            self.tensors[i] = Q.reshape(chi_l, d, -1)
            self.tensors[i + 1] = torch.einsum('ij,jdk->idk', R, self.tensors[i + 1])
        
        self._canonical_center = self.L - 1
        return self
    
    def canonicalize_right_(self) -> MPS:
        """
        Right-canonicalize MPS in-place.
        
        After this, A[i] @ A[i]^† = I for all i > 0.
        The norm is absorbed into the first tensor.
        
        Returns:
            self
        """
        for i in range(self.L - 1, 0, -1):
            A = self.tensors[i]
            chi_l, d, chi_r = A.shape
            
            # Reshape and QR from right
            A_mat = A.reshape(chi_l, d * chi_r).T
            Q, R = qr_positive(A_mat)
            
            # Update tensors
            self.tensors[i] = Q.T.reshape(-1, d, chi_r)
            self.tensors[i - 1] = torch.einsum('idk,kj->idj', self.tensors[i - 1], R.T)
        
        self._canonical_center = 0
        return self
    
    def canonicalize_to_(self, site: int) -> MPS:
        """
        Mixed-canonical form with orthogonality center at site.
        
        Args:
            site: Orthogonality center (0 to L-1)
            
        Returns:
            self
        """
        # Left-canonicalize up to site
        for i in range(site):
            A = self.tensors[i]
            chi_l, d, chi_r = A.shape
            A_mat = A.reshape(chi_l * d, chi_r)
            Q, R = qr_positive(A_mat)
            self.tensors[i] = Q.reshape(chi_l, d, -1)
            self.tensors[i + 1] = torch.einsum('ij,jdk->idk', R, self.tensors[i + 1])
        
        # Right-canonicalize from site+1 to end
        for i in range(self.L - 1, site, -1):
            A = self.tensors[i]
            chi_l, d, chi_r = A.shape
            A_mat = A.reshape(chi_l, d * chi_r).T
            Q, R = qr_positive(A_mat)
            self.tensors[i] = Q.T.reshape(-1, d, chi_r)
            self.tensors[i - 1] = torch.einsum('idk,kj->idj', self.tensors[i - 1], R.T)
        
        self._canonical_center = site
        return self
    
    def entropy(self, bond: int) -> Tensor:
        """
        Compute von Neumann entanglement entropy at bond.
        
        S = -Tr(ρ log ρ) where ρ is the reduced density matrix.
        
        Args:
            bond: Bond index (0 to L-2)
            
        Returns:
            Entanglement entropy
        """
        # Canonicalize to bond+1
        mps = self.copy()
        mps.canonicalize_to_(bond + 1)
        
        # Get tensor at bond+1 and compute singular values
        A = mps.tensors[bond + 1]
        chi_l, d, chi_r = A.shape
        A_mat = A.reshape(chi_l, d * chi_r)
        
        # rSVD - faster above 100x100
        m, n = A_mat.shape
        if min(m, n) > 100:
            _, S, _ = torch.svd_lowrank(A_mat, q=min(100, min(m, n)))
        else:
            _, S, _ = torch.linalg.svd(A_mat, full_matrices=False)
        
        # Normalize singular values
        S = S / S.norm()
        
        # Compute entropy: S = -Σ p log(p) where p = s²
        p = S ** 2
        p = p[p > 1e-14]  # Remove zeros
        entropy = -torch.sum(p * torch.log(p))
        
        return entropy
    
    def expectation_local(self, op: Tensor, site: int) -> Tensor:
        """
        Compute ⟨ψ|O_site|ψ⟩ for local operator O.
        
        Args:
            op: Local operator of shape (d, d)
            site: Site index
            
        Returns:
            Expectation value
        """
        # Contract from left up to site
        env_L = torch.ones(1, 1, dtype=self.dtype, device=self.device)
        for i in range(site):
            A = self.tensors[i]
            env_L = torch.einsum('ij,idk,jdl->kl', env_L, A, A.conj())
        
        # Contract from right down to site
        env_R = torch.ones(1, 1, dtype=self.dtype, device=self.device)
        for i in range(self.L - 1, site, -1):
            A = self.tensors[i]
            env_R = torch.einsum('idk,jdl,kl->ij', A, A.conj(), env_R)
        
        # Apply operator at site
        A = self.tensors[site]
        result = torch.einsum(
            'ij,idk,de,jel,kl->',
            env_L, A, op, A.conj(), env_R
        )
        
        # Normalize
        norm_sq = self.norm() ** 2
        return result / norm_sq
    
    def truncate_(self, chi_max: int, cutoff: float = 1e-14) -> MPS:
        """
        Truncate bond dimension via SVD.
        
        Args:
            chi_max: Maximum bond dimension
            cutoff: SVD cutoff
            
        Returns:
            self
        """
        # Left-canonicalize then sweep right with truncation
        self.canonicalize_left_()
        
        for i in range(self.L - 1, 0, -1):
            A = self.tensors[i]
            chi_l, d, chi_r = A.shape
            
            A_mat = A.reshape(chi_l, d * chi_r)
            U, S, Vh = svd_truncated(A_mat, chi_max=chi_max, cutoff=cutoff)
            
            self.tensors[i] = Vh.reshape(-1, d, chi_r)
            self.tensors[i - 1] = torch.einsum(
                'idk,kj,j->idj',
                self.tensors[i - 1], U, S
            )
        
        return self
    
    def __repr__(self) -> str:
        return f"MPS(L={self.L}, d={self.d}, χ={self.chi}, dtype={self.dtype})"
