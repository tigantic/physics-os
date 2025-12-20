"""
Distributed MPS Operations Module
=================================

Distributed operations for Matrix Product States
supporting cross-partition contractions and merging.

Features:
- Cross-node contractions
- Partition merging
- Distributed compression
- Global operations
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from concurrent.futures import ThreadPoolExecutor, Future, as_completed

import numpy as np
import torch


class CompressionStrategy(Enum):
    """Strategy for distributed compression."""
    
    SVD = auto()        # Standard SVD compression
    VARIATIONAL = auto()  # Variational optimization
    DENSITY_MATRIX = auto()  # Density matrix truncation


@dataclass
class MPSPartition:
    """A partition of an MPS.
    
    Attributes:
        partition_id: Unique identifier
        start_site: Global start site
        end_site: Global end site  
        tensors: Local MPS tensors
        left_bond: Left boundary bond dimension
        right_bond: Right boundary bond dimension
        local_norm: Local contribution to norm
    """
    
    partition_id: int
    start_site: int
    end_site: int
    tensors: List[torch.Tensor]
    left_bond: int = 1
    right_bond: int = 1
    local_norm: float = 1.0
    
    @property
    def num_sites(self) -> int:
        """Number of sites."""
        return self.end_site - self.start_site
    
    def get_tensor(self, local_idx: int) -> torch.Tensor:
        """Get tensor at local index."""
        return self.tensors[local_idx]
    
    def set_tensor(self, local_idx: int, tensor: torch.Tensor) -> None:
        """Set tensor at local index."""
        self.tensors[local_idx] = tensor


@dataclass
class CrossNodeContraction:
    """Handles cross-node tensor contractions.
    
    Attributes:
        left_partition_id: ID of left partition
        right_partition_id: ID of right partition
        bond_site: Global site at boundary
        contracted_tensor: Result of contraction
        contraction_time: Time taken
    """
    
    left_partition_id: int
    right_partition_id: int
    bond_site: int
    contracted_tensor: Optional[torch.Tensor] = None
    contraction_time: float = 0.0
    
    def contract_boundary(
        self,
        left_tensor: torch.Tensor,
        right_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Contract tensors at boundary.
        
        Args:
            left_tensor: Right-most tensor of left partition
            right_tensor: Left-most tensor of right partition
            
        Returns:
            Contracted tensor
        """
        start = time.perf_counter()
        
        # left_tensor[i,s,j] right_tensor[j,t,k] -> [i,s,t,k]
        result = torch.einsum('isj,jtk->istk', left_tensor, right_tensor)
        
        self.contracted_tensor = result
        self.contraction_time = time.perf_counter() - start
        
        return result


@dataclass
class DistributedMPSConfig:
    """Configuration for distributed MPS.
    
    Attributes:
        num_partitions: Number of partitions
        chi_max: Maximum bond dimension
        cutoff: Truncation cutoff
        compression_strategy: How to compress
        num_workers: Number of worker threads
    """
    
    num_partitions: int = 4
    chi_max: int = 64
    cutoff: float = 1e-10
    compression_strategy: CompressionStrategy = CompressionStrategy.SVD
    num_workers: int = 4


class DistributedMPS:
    """Distributed MPS with partition support.
    
    Manages MPS across multiple partitions with
    support for cross-partition operations.
    """
    
    def __init__(
        self,
        config: Optional[DistributedMPSConfig] = None,
    ) -> None:
        """Initialize distributed MPS.
        
        Args:
            config: Configuration
        """
        self.config = config or DistributedMPSConfig()
        self.partitions: List[MPSPartition] = []
        self.total_sites: int = 0
        self.executor = ThreadPoolExecutor(max_workers=self.config.num_workers)
    
    def from_tensors(
        self,
        tensors: List[torch.Tensor],
    ) -> None:
        """Initialize from list of tensors.
        
        Args:
            tensors: List of MPS tensors
        """
        self.total_sites = len(tensors)
        sites_per_partition = self.total_sites // self.config.num_partitions
        remainder = self.total_sites % self.config.num_partitions
        
        self.partitions = []
        start = 0
        
        for i in range(self.config.num_partitions):
            extra = 1 if i < remainder else 0
            end = start + sites_per_partition + extra
            
            partition = MPSPartition(
                partition_id=i,
                start_site=start,
                end_site=end,
                tensors=[t.clone() for t in tensors[start:end]],
                left_bond=tensors[start].shape[0],
                right_bond=tensors[end - 1].shape[2],
            )
            self.partitions.append(partition)
            start = end
    
    def to_tensors(self) -> List[torch.Tensor]:
        """Convert back to tensor list.
        
        Returns:
            List of MPS tensors
        """
        tensors = []
        for partition in self.partitions:
            tensors.extend(partition.tensors)
        return tensors
    
    def local_compression(
        self,
        partition_id: int,
    ) -> float:
        """Compress a single partition.
        
        Args:
            partition_id: ID of partition
            
        Returns:
            Total truncation error
        """
        partition = self.partitions[partition_id]
        total_error = 0.0
        
        # Right to left sweep
        for i in range(len(partition.tensors) - 1, 0, -1):
            B = partition.tensors[i]
            chi_left, d, chi_right = B.shape
            
            # Reshape and QR
            B_mat = B.reshape(chi_left, d * chi_right)
            Q, R = torch.linalg.qr(B_mat.T)
            
            # Q is now (d*chi_right, chi_new), R is (chi_new, chi_left)
            chi_new = Q.shape[1]
            partition.tensors[i] = Q.T.reshape(chi_new, d, chi_right)
            
            # Absorb R into left tensor
            A = partition.tensors[i - 1]
            chi_ll, d_l, _ = A.shape
            partition.tensors[i - 1] = torch.einsum('ijk,kl->ijl', A, R.T)
        
        # Left to right sweep with truncation
        for i in range(len(partition.tensors) - 1):
            A = partition.tensors[i]
            chi_left, d, chi_right = A.shape
            
            # Reshape and SVD
            A_mat = A.reshape(chi_left * d, chi_right)
            U, S, Vh = torch.linalg.svd(A_mat, full_matrices=False)
            
            # Truncate
            chi_new = min(self.config.chi_max, len(S))
            mask = S > S[0] * self.config.cutoff
            chi_new = min(chi_new, mask.sum().item())
            chi_new = max(1, chi_new)
            
            # Error
            if chi_new < len(S):
                total_error += float(torch.sum(S[chi_new:] ** 2))
            
            U = U[:, :chi_new]
            S = S[:chi_new]
            Vh = Vh[:chi_new, :]
            
            partition.tensors[i] = U.reshape(chi_left, d, chi_new)
            
            # Absorb S*Vh into right tensor
            SV = torch.diag(S) @ Vh
            B = partition.tensors[i + 1]
            partition.tensors[i + 1] = torch.einsum('ij,jkl->ikl', SV, B)
        
        return total_error
    
    def global_compression(self) -> float:
        """Compress all partitions in parallel.
        
        Returns:
            Total truncation error
        """
        futures = []
        for i in range(len(self.partitions)):
            future = self.executor.submit(self.local_compression, i)
            futures.append(future)
        
        total_error = 0.0
        for future in as_completed(futures):
            total_error += future.result()
        
        return total_error
    
    def cross_partition_contraction(
        self,
        left_id: int,
        right_id: int,
        operator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Contract across partition boundary.
        
        Args:
            left_id: Left partition ID
            right_id: Right partition ID
            operator: Optional operator to apply
            
        Returns:
            Result of contraction
        """
        left_partition = self.partitions[left_id]
        right_partition = self.partitions[right_id]
        
        left_tensor = left_partition.tensors[-1]
        right_tensor = right_partition.tensors[0]
        
        # Contract boundary
        # [chi_l, d, chi_m] [chi_m, d, chi_r] -> [chi_l, d, d, chi_r]
        result = torch.einsum('isj,jtk->istk', left_tensor, right_tensor)
        
        if operator is not None:
            # Apply two-site operator
            result = torch.einsum('abcd,istk->abcd', operator, result)
        
        return result
    
    def compute_local_norm(
        self,
        partition_id: int,
    ) -> float:
        """Compute norm contribution of partition.
        
        Args:
            partition_id: Partition ID
            
        Returns:
            Local norm squared
        """
        partition = self.partitions[partition_id]
        
        # For normalized MPS, compute <psi|psi> locally
        # Using left-to-right contraction
        env = torch.eye(partition.left_bond, dtype=partition.tensors[0].dtype)
        
        for tensor in partition.tensors:
            # env[i,j] A[i,s,k] A*[j,s,l] -> new_env[k,l]
            env = torch.einsum('ij,isk,jsl->kl', env, tensor, tensor.conj())
        
        norm_sq = float(torch.trace(env).real)
        partition.local_norm = math.sqrt(norm_sq)
        
        return norm_sq
    
    def compute_global_norm(self) -> float:
        """Compute total norm of MPS.
        
        Returns:
            Total norm
        """
        # Start with identity
        env = torch.eye(1, dtype=self.partitions[0].tensors[0].dtype)
        
        for partition in self.partitions:
            for tensor in partition.tensors:
                # Contract through
                env = torch.einsum('ij,isk,jsl->kl', env, tensor, tensor.conj())
        
        return float(torch.sqrt(torch.trace(env).real))
    
    def apply_mpo(
        self,
        mpo_tensors: List[torch.Tensor],
        compress: bool = True,
    ) -> float:
        """Apply MPO to the MPS.
        
        Args:
            mpo_tensors: MPO tensors W[i,s',s,j]
            compress: Whether to compress after
            
        Returns:
            Truncation error if compressed
        """
        mpo_idx = 0
        
        for partition in self.partitions:
            for i in range(len(partition.tensors)):
                A = partition.tensors[i]  # [chi_l, d, chi_r]
                W = mpo_tensors[mpo_idx]  # [w_l, d', d, w_r]
                
                # Contract: A[chi_l, s, chi_r] W[w_l, s', s, w_r] 
                # -> [chi_l, w_l, s', chi_r, w_r] -> [(chi_l*w_l), s', (chi_r*w_r)]
                result = torch.einsum('iaj,wbak->iwbkj', A, W)
                chi_l, w_l, d_new, chi_r, w_r = result.shape
                
                partition.tensors[i] = result.reshape(chi_l * w_l, d_new, chi_r * w_r)
                mpo_idx += 1
        
        if compress:
            return self.global_compression()
        
        return 0.0
    
    def measure_local(
        self,
        partition_id: int,
        local_site: int,
        operator: torch.Tensor,
    ) -> complex:
        """Measure expectation of local operator.
        
        Args:
            partition_id: Partition ID
            local_site: Local site index
            operator: Local operator O[s',s]
            
        Returns:
            Expectation value <O>
        """
        partition = self.partitions[partition_id]
        
        # Build left environment
        env_left = torch.eye(partition.left_bond, dtype=partition.tensors[0].dtype)
        for i in range(local_site):
            A = partition.tensors[i]
            env_left = torch.einsum('ij,isk,jsl->kl', env_left, A, A.conj())
        
        # Apply operator
        A = partition.tensors[local_site]
        # env[i,j] A[i,s,k] O[t,s] A*[j,t,l] -> new_env[k,l]
        env_op = torch.einsum('ij,isk,ts,jtl->kl', env_left, A, operator, A.conj())
        
        # Build right environment
        env_right = torch.eye(partition.right_bond, dtype=partition.tensors[0].dtype)
        for i in range(len(partition.tensors) - 1, local_site, -1):
            A = partition.tensors[i]
            env_right = torch.einsum('isk,jsl,kl->ij', A, A.conj(), env_right)
        
        # Contract
        result = torch.einsum('ij,ij->', env_op, env_right)
        
        return complex(result)
    
    def shutdown(self) -> None:
        """Shutdown executor."""
        self.executor.shutdown(wait=True)


def merge_partitions(
    partitions: List[MPSPartition],
) -> List[torch.Tensor]:
    """Merge partitions into single MPS.
    
    Args:
        partitions: List of partitions
        
    Returns:
        List of MPS tensors
    """
    tensors = []
    for partition in partitions:
        tensors.extend([t.clone() for t in partition.tensors])
    return tensors


def split_into_partitions(
    tensors: List[torch.Tensor],
    num_partitions: int,
) -> List[MPSPartition]:
    """Split MPS into partitions.
    
    Args:
        tensors: MPS tensors
        num_partitions: Number of partitions
        
    Returns:
        List of partitions
    """
    num_sites = len(tensors)
    sites_per_partition = num_sites // num_partitions
    remainder = num_sites % num_partitions
    
    partitions = []
    start = 0
    
    for i in range(num_partitions):
        extra = 1 if i < remainder else 0
        end = start + sites_per_partition + extra
        
        partition = MPSPartition(
            partition_id=i,
            start_site=start,
            end_site=end,
            tensors=[t.clone() for t in tensors[start:end]],
            left_bond=tensors[start].shape[0],
            right_bond=tensors[end - 1].shape[2],
        )
        partitions.append(partition)
        start = end
    
    return partitions
