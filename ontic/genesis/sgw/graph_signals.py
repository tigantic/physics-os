"""
QTT Graph Signals — Layer 21 Component

Signals on graphs represented in QTT (Quantized Tensor Train) format.

A signal f on a graph with N = 2^d nodes is represented as a tensor train:
    f[i] = A₁[i₁] A₂[i₂] ... A_d[i_d]
    
where i = i₁i₂...i_d in binary.

RULES ENFORCED:
1. QTT should be Native — NEVER decompress to dense unless debugging
2. SVD = rSVD (always randomized, never full torch.linalg.svd)
3. Native dot/norm/add operations without dense materialization
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import torch
import math

from ontic.genesis.core.rsvd import rsvd_gpu
from ontic.genesis.core.triton_ops import (
    qtt_dot_native,
    qtt_norm_native,
    qtt_add_native,
    qtt_sub_native,
    qtt_round_native,
    rsvd_native,
)


@dataclass
class QTTSignal:
    """
    Signal on a graph in QTT format.
    
    A graph signal f: V → ℝ is stored as a tensor train, enabling
    O(r² log N) storage and O(r³ log N) operations.
    
    Attributes:
        cores: List of TT cores, shape (r_left, 2, r_right)
        num_nodes: Number of nodes (must be 2^d)
    """
    cores: List[torch.Tensor]
    num_nodes: int
    
    def __post_init__(self):
        """Validate cores."""
        if not self.cores:
            raise ValueError("Cores list cannot be empty")
        
        # Check dimensions match
        d = len(self.cores)
        expected_nodes = 2 ** d
        if self.num_nodes != expected_nodes:
            pass  # Allow mismatch for now (2D/3D grids)
    
    @property
    def num_qubits(self) -> int:
        """Number of tensor train sites."""
        return len(self.cores)
    
    @property
    def max_rank(self) -> int:
        """Maximum TT rank."""
        ranks = [c.shape[2] for c in self.cores[:-1]]
        return max(ranks) if ranks else 1
    
    @property
    def dtype(self) -> torch.dtype:
        """Data type of cores."""
        return self.cores[0].dtype
    
    @classmethod
    def from_cores(cls, cores: List[torch.Tensor]) -> 'QTTSignal':
        """
        Create QTTSignal directly from TT cores.
        
        This is the preferred way to convert between QTT representations
        (e.g., from QTTDistribution) WITHOUT going through dense format.
        
        Args:
            cores: List of TT cores, each shape (r_{k-1}, 2, r_k)
            
        Returns:
            QTTSignal wrapping the cores
        """
        if not cores:
            raise ValueError("Cores list cannot be empty")
        
        d = len(cores)
        num_nodes = 2 ** d
        
        # Clone cores to avoid aliasing issues
        cloned = [c.clone() for c in cores]
        
        return cls(cores=cloned, num_nodes=num_nodes)
    
    @classmethod
    def zeros(cls, num_nodes: int, dtype: torch.dtype = torch.float64) -> 'QTTSignal':
        """Create zero signal."""
        d = int(math.log2(num_nodes))
        if 2**d != num_nodes:
            raise ValueError(f"num_nodes must be power of 2, got {num_nodes}")
        
        cores = []
        for k in range(d):
            core = torch.zeros(1, 2, 1, dtype=dtype)
            cores.append(core)
        
        return cls(cores=cores, num_nodes=num_nodes)
    
    @classmethod
    def constant(cls, num_nodes: int, value: float = 1.0, 
                 dtype: torch.dtype = torch.float64) -> 'QTTSignal':
        """Create constant signal f[i] = value for all i."""
        d = int(math.log2(num_nodes))
        if 2**d != num_nodes:
            raise ValueError(f"num_nodes must be power of 2, got {num_nodes}")
        
        # For constant signal, each core is [1, 1] (row vector form)
        # When contracted: [[1,1]] @ [[1,1]]^T @ ... = 2^d * scale_factor
        # We need the sum to equal value, so each entry = value/2^d is wrong
        # Actually, TT format: result[i] = product of cores[k][0, bits[k], 0]
        # For constant signal, we want result[i] = value for all i
        # If each core = [1, 1], then product = 1 for any index
        # So we set first core to [value, value] and rest to [1, 1]
        cores = []
        for k in range(d):
            core = torch.ones(1, 2, 1, dtype=dtype)
            if k == 0:
                core = core * value  # First core carries the value
            cores.append(core)
        
        return cls(cores=cores, num_nodes=num_nodes)
    
    @classmethod
    def delta(cls, num_nodes: int, node_index: int = 0,
              dtype: torch.dtype = torch.float64) -> 'QTTSignal':
        """
        Create delta (impulse) signal at a specific node.
        
        f[i] = 1 if i == node_index, else 0
        """
        d = int(math.log2(num_nodes))
        if 2**d != num_nodes:
            raise ValueError(f"num_nodes must be power of 2, got {num_nodes}")
        
        if node_index < 0 or node_index >= num_nodes:
            raise ValueError(f"node_index {node_index} out of range [0, {num_nodes})")
        
        # Binary representation of node_index
        bits = [(node_index >> (d - 1 - k)) & 1 for k in range(d)]
        
        cores = []
        for k in range(d):
            core = torch.zeros(1, 2, 1, dtype=dtype)
            core[0, bits[k], 0] = 1.0
            cores.append(core)
        
        return cls(cores=cores, num_nodes=num_nodes)
    
    @classmethod
    def random(cls, num_nodes: int, rank: int = 5,
               dtype: torch.dtype = torch.float64,
               seed: Optional[int] = None) -> 'QTTSignal':
        """
        Create random signal with specified TT rank.
        
        Args:
            num_nodes: Number of nodes (power of 2)
            rank: TT rank (controls complexity)
            dtype: Data type
            seed: Random seed for reproducibility
        """
        d = int(math.log2(num_nodes))
        if 2**d != num_nodes:
            raise ValueError(f"num_nodes must be power of 2, got {num_nodes}")
        
        if seed is not None:
            torch.manual_seed(seed)
        
        cores = []
        for k in range(d):
            r_left = 1 if k == 0 else rank
            r_right = 1 if k == d - 1 else rank
            
            core = torch.randn(r_left, 2, r_right, dtype=dtype)
            # Normalize to prevent explosion
            core = core / (core.norm() + 1e-10)
            cores.append(core)
        
        signal = cls(cores=cores, num_nodes=num_nodes)
        return signal.normalize()
    
    @classmethod
    def from_function(cls, num_nodes: int, func, 
                      grid_bounds: tuple = (-1, 1),
                      max_rank: int = 50,
                      tol: float = 1e-10,
                      dtype: torch.dtype = torch.float64) -> 'QTTSignal':
        """
        Create signal from a function via TT-SVD.
        
        Args:
            num_nodes: Number of nodes (power of 2)
            func: Function f(x) to sample
            grid_bounds: (xmin, xmax) for sampling
            max_rank: Maximum TT rank
            tol: SVD truncation tolerance
        """
        d = int(math.log2(num_nodes))
        if 2**d != num_nodes:
            raise ValueError(f"num_nodes must be power of 2, got {num_nodes}")
        
        # Sample function
        x = torch.linspace(grid_bounds[0], grid_bounds[1], num_nodes, dtype=dtype)
        values = torch.tensor([func(xi.item()) for xi in x], dtype=dtype)
        
        # TT-SVD decomposition
        return cls.from_dense(values, max_rank=max_rank, tol=tol)
    
    @classmethod
    def from_dense(cls, values: torch.Tensor, max_rank: int = 50,
                   tol: float = 1e-10) -> 'QTTSignal':
        """
        Create QTT signal from dense vector via TT-SVD.
        
        Args:
            values: Dense vector of length N = 2^d
            max_rank: Maximum TT rank
            tol: SVD truncation tolerance
        """
        N = len(values)
        d = int(math.log2(N))
        if 2**d != N:
            raise ValueError(f"Vector length must be power of 2, got {N}")
        
        dtype = values.dtype
        
        # Reshape to tensor
        tensor = values.reshape([2] * d)
        
        # TT-SVD (left-to-right sweep)
        cores = []
        C = tensor.reshape(1, -1)  # (1, 2^d)
        
        for k in range(d - 1):
            # Reshape for SVD
            m = C.shape[0] * 2
            n = C.shape[1] // 2
            C = C.reshape(m, n)
            
            # GPU-native rSVD
            U, S, Vh = rsvd_gpu(C, k=max_rank, tol=tol)
            
            # Truncate
            rank = max(1, len(S))
            rank = min(rank, max_rank)
            
            U = U[:, :rank]
            S = S[:rank]
            Vh = Vh[:rank, :]
            
            # Store core
            core = U.reshape(-1, 2, rank)
            cores.append(core)
            
            # Continue with S @ Vh
            C = torch.diag(S) @ Vh
        
        # Last core
        core = C.reshape(-1, 2, 1)
        cores.append(core)
        
        return cls(cores=cores, num_nodes=N)
    
    def norm(self) -> float:
        """
        Compute L2 norm of signal WITHOUT dense materialization.
        
        Uses native QTT transfer matrix contraction.
        Complexity: O(d r^4) vs O(2^d) for dense.
        """
        return qtt_norm_native(self.cores)
    
    def dot(self, other: 'QTTSignal') -> float:
        """
        Inner product WITHOUT dense materialization.
        
        Uses native QTT transfer matrix contraction.
        Complexity: O(d r^4) vs O(2^d) for dense.
        """
        if self.num_nodes != other.num_nodes:
            raise ValueError("Signals must have same size")
        return qtt_dot_native(self.cores, other.cores)
    
    def normalize(self) -> 'QTTSignal':
        """Return signal normalized to unit L2 norm."""
        n = self.norm()
        if n < 1e-15:
            return self
        return self.scale(1.0 / n)
    
    def scale(self, alpha: float) -> 'QTTSignal':
        """Scale signal by scalar."""
        new_cores = [c.clone() for c in self.cores]
        new_cores[0] = new_cores[0] * alpha
        return QTTSignal(cores=new_cores, num_nodes=self.num_nodes)
    
    def add(self, other: 'QTTSignal', max_rank: Optional[int] = None) -> 'QTTSignal':
        """
        Add two signals WITHOUT dense materialization.
        
        Uses native QTT addition with optional rank truncation.
        Complexity: O(d (r1+r2)^3) vs O(2^d) for dense.
        """
        if self.num_nodes != other.num_nodes:
            raise ValueError("Signals must have same size")
        
        result_cores = qtt_add_native(self.cores, other.cores, max_rank=max_rank)
        return QTTSignal(cores=result_cores, num_nodes=self.num_nodes)
    
    def sub(self, other: 'QTTSignal', max_rank: Optional[int] = None) -> 'QTTSignal':
        """
        Subtract two signals WITHOUT dense materialization.
        
        Uses native QTT subtraction (negate + add) with optional rank truncation.
        Complexity: O(d (r1+r2)^3) vs O(2^d) for dense.
        """
        if self.num_nodes != other.num_nodes:
            raise ValueError("Signals must have same size")
        
        result_cores = qtt_sub_native(self.cores, other.cores, max_rank=max_rank)
        return QTTSignal(cores=result_cores, num_nodes=self.num_nodes)
    
    def hadamard(self, other: 'QTTSignal') -> 'QTTSignal':
        """Element-wise product (rank-multiplicative)."""
        if self.num_nodes != other.num_nodes:
            raise ValueError("Signals must have same size")
        
        # Determine device
        device = self.cores[0].device if len(self.cores) > 0 else torch.device('cpu')
        
        result_cores = []
        for c1, c2 in zip(self.cores, other.cores):
            c1 = c1.to(device)
            c2 = c2.to(device)
            
            r1_l, d1, r1_r = c1.shape
            r2_l, d2, r2_r = c2.shape
            
            # Kronecker product of cores
            new_core = torch.einsum('ijk,lmn->iljkmn', c1, c2)
            new_core = new_core.reshape(r1_l * r2_l, d1, r1_r * r2_r)
            
            result_cores.append(new_core)
        
        return QTTSignal(cores=result_cores, num_nodes=self.num_nodes)
    
    def round(self, tol: float = 1e-10, max_rank: Optional[int] = None) -> 'QTTSignal':
        """
        Reduce rank via TT-rounding WITHOUT dense materialization.
        
        Uses native QTT rounding with rSVD.
        Complexity: O(d r^3) vs O(2^d) for dense.
        
        Args:
            tol: Relative tolerance for truncation
            max_rank: Maximum rank (optional, default 50)
        """
        if max_rank is None:
            max_rank = 50
        
        result_cores = qtt_round_native(self.cores, max_rank=max_rank, tol=tol)
        return QTTSignal(cores=result_cores, num_nodes=self.num_nodes)
    
    def to_dense(self) -> torch.Tensor:
        """
        Materialize as dense vector.
        
        WARNING: FOR DEBUGGING ONLY! Violates QTT principles.
        Use sparingly and only for small grids (N ≤ 2^20).
        
        Returns:
            Dense vector of length num_nodes
        """
        if self.num_nodes > 2**20:
            raise ValueError(
                f"Signal too large to materialize: {self.num_nodes}. "
                f"Use native QTT operations instead of to_dense()."
            )
        
        # Contract all cores
        result = self.cores[0]  # (1, 2, r)
        
        for core in self.cores[1:]:
            # result: (1, 2^k, r_left)
            # core: (r_left, 2, r_right)
            result = torch.einsum('ijk,klm->ijlm', result, core)
            result = result.reshape(1, -1, core.shape[2])
        
        # Final: (1, N, 1)
        return result.squeeze(0).squeeze(-1)
    
    def __repr__(self) -> str:
        return (f"QTTSignal(num_nodes={self.num_nodes}, "
                f"num_qubits={self.num_qubits}, max_rank={self.max_rank})")
