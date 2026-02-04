"""
QTT 3D State and Operations for Turbulence DNS
===============================================

Native 3D QTT representation and operations for incompressible
Navier-Stokes equations.

Key Design Principles:
1. No dense arrays in hot path (O(log N) memory)
2. Morton ordering for spatial locality
3. Scale-adaptive compression for turbulence
4. CUDA acceleration where beneficial

Complexity Analysis:
    Dense 3D:     O(N³) memory, O(N³ log N) FFT
    QTT 3D:       O(r² × 3 log₂ N) memory, O(r³ × 3 log₂ N) ops
    
    For N=1024, r=64:  Dense=4GB, QTT=~1MB → 4000× compression

Author: HyperTensor Team
Date: 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union, Callable
from enum import Enum, auto

import numpy as np
import torch
from torch import Tensor

from tensornet.cfd.morton_3d import (
    Morton3DGrid,
    linear_to_morton_3d,
    morton_to_linear_3d,
)
from tensornet.cfd.pure_qtt_ops import (
    QTTState,
    dense_to_qtt,
    qtt_to_dense,
    qtt_add,
)
from tensornet.cfd.nd_shift_mpo import (
    make_nd_shift_mpo,
    apply_nd_shift_mpo,
    truncate_cores,
    truncate_cores_adaptive,
)


# ═══════════════════════════════════════════════════════════════════════════════════════
# QTT 3D STATE
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class QTT3DState:
    """
    3D field stored in QTT format with Morton ordering.
    
    The field is stored as a 1D QTT over 3n qubits where n = log₂(N).
    Morton interleaving: qubit k belongs to dimension (k mod 3).
    
    Attributes:
        cores: List of QTT cores, each of shape (r_left, 2, r_right)
        n_bits: Bits per spatial dimension (N = 2^n_bits per axis)
        device: Torch device
        dtype: Data type
        
    Properties:
        N: Grid size per axis
        shape: (N, N, N)
        total_qubits: 3 * n_bits
        max_rank: Maximum bond dimension
        
    Example:
        >>> state = QTT3DState.from_dense(velocity_field, max_rank=64)
        >>> print(f"Compression: {state.compression_ratio:.1f}×")
    """
    cores: List[Tensor]
    n_bits: int
    device: torch.device = field(default_factory=lambda: torch.device('cpu'))
    dtype: torch.dtype = torch.float32
    
    @property
    def N(self) -> int:
        """Grid size per axis."""
        return 1 << self.n_bits
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Physical grid shape."""
        return (self.N, self.N, self.N)
    
    @property
    def total_qubits(self) -> int:
        """Total QTT sites."""
        return len(self.cores)
    
    @property
    def max_rank(self) -> int:
        """Maximum bond dimension."""
        return max(c.shape[0] for c in self.cores) if self.cores else 0
    
    @property
    def mean_rank(self) -> float:
        """Mean bond dimension."""
        if not self.cores:
            return 0.0
        return np.mean([c.shape[0] for c in self.cores])
    
    @property
    def qtt_parameters(self) -> int:
        """Total parameters in QTT representation."""
        return sum(c.numel() for c in self.cores)
    
    @property
    def dense_parameters(self) -> int:
        """Parameters in equivalent dense representation."""
        return self.N ** 3
    
    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs dense."""
        qtt_params = self.qtt_parameters
        if qtt_params == 0:
            return float('inf')
        return self.dense_parameters / qtt_params
    
    def clone(self) -> 'QTT3DState':
        """Deep copy."""
        return QTT3DState(
            cores=[c.clone() for c in self.cores],
            n_bits=self.n_bits,
            device=self.device,
            dtype=self.dtype,
        )
    
    def to(self, device: torch.device) -> 'QTT3DState':
        """Move to device."""
        return QTT3DState(
            cores=[c.to(device) for c in self.cores],
            n_bits=self.n_bits,
            device=device,
            dtype=self.dtype,
        )
    
    @classmethod
    def from_dense(
        cls,
        tensor: Tensor,
        max_rank: int = 64,
        tol: float = 1e-6,
    ) -> 'QTT3DState':
        """
        Compress 3D tensor to QTT format with tolerance-controlled rank.
        
        Uses TT-SVD with relative tolerance truncation. The rank is determined
        by physics (singular value decay), not the max_rank cap.
        
        Args:
            tensor: 3D tensor of shape (N, N, N)
            max_rank: Maximum bond dimension (hard ceiling)
            tol: Truncation tolerance (relative to largest singular value)
            
        Returns:
            QTT3DState with physics-determined rank
        """
        N = tensor.shape[0]
        n_bits = int(np.log2(N))
        assert 2**n_bits == N, f"N={N} must be a power of 2"
        assert tensor.shape == (N, N, N), f"Expected ({N}, {N}, {N}), got {tensor.shape}"
        
        device = tensor.device
        dtype = tensor.dtype
        
        # Convert to Morton order
        morton = linear_to_morton_3d(tensor, n_bits)
        
        # Tolerance-aware TT-SVD
        n = 3 * n_bits
        reshaped = morton.reshape([2] * n)
        cores = []
        current = reshaped.reshape(1, -1)
        
        for i in range(n):
            r_left = current.shape[0]
            remaining = current.numel() // (r_left * 2)
            mat = current.reshape(r_left * 2, remaining) if remaining > 0 else current.reshape(r_left * 2, 1)
            
            if i < n - 1:
                # Full SVD for proper tolerance truncation
                try:
                    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
                except torch.linalg.LinAlgError:
                    # Fallback for numerical issues
                    cores.append(mat.reshape(r_left, 2, mat.shape[1]))
                    current = torch.eye(mat.shape[1], device=device, dtype=dtype).reshape(mat.shape[1], -1)
                    continue
                
                # Tolerance-based rank determination
                if tol > 0 and len(S) > 0:
                    threshold = tol * S[0]
                    rank = int(torch.sum(S > threshold).item())
                    rank = max(rank, 1)
                else:
                    rank = len(S)
                
                # Also enforce max_rank ceiling
                rank = min(rank, max_rank, len(S))
                
                U = U[:, :rank]
                S = S[:rank]
                Vh = Vh[:rank, :]
                
                cores.append(U.reshape(r_left, 2, rank))
                current = torch.diag(S) @ Vh
            else:
                cores.append(mat.reshape(r_left, 2, 1))
        
        return cls(
            cores=cores,
            n_bits=n_bits,
            device=device,
            dtype=dtype,
        )
    
    def to_dense(self) -> Tensor:
        """
        Decompress to full 3D tensor.
        
        Returns:
            Tensor of shape (N, N, N)
        """
        # QTT to 1D Morton
        qtt = QTTState(cores=self.cores, num_qubits=self.total_qubits)
        morton = qtt_to_dense(qtt)
        
        # Morton to 3D
        tensor = morton_to_linear_3d(morton, self.n_bits)
        
        return tensor
    
    @classmethod
    def zeros(
        cls,
        n_bits: int,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ) -> 'QTT3DState':
        """Create zero field (rank-1 QTT)."""
        if device is None:
            device = torch.device('cpu')
        
        total_qubits = 3 * n_bits
        cores = []
        for i in range(total_qubits):
            r_left = 1
            r_right = 1
            core = torch.zeros(r_left, 2, r_right, device=device, dtype=dtype)
            cores.append(core)
        
        return cls(cores=cores, n_bits=n_bits, device=device, dtype=dtype)
    
    @classmethod
    def ones(
        cls,
        n_bits: int,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ) -> 'QTT3DState':
        """Create constant-one field (rank-1 QTT)."""
        if device is None:
            device = torch.device('cpu')
        
        total_qubits = 3 * n_bits
        cores = []
        for i in range(total_qubits):
            r_left = 1
            r_right = 1
            core = torch.ones(r_left, 2, r_right, device=device, dtype=dtype)
            cores.append(core)
        
        return cls(cores=cores, n_bits=n_bits, device=device, dtype=dtype)


# ═══════════════════════════════════════════════════════════════════════════════════════
# QTT 3D VECTOR FIELD (VELOCITY / VORTICITY)
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class QTT3DVectorField:
    """
    3D vector field in QTT format.
    
    Stores three QTT3DState components (x, y, z).
    
    Used for:
    - Velocity field u = (u_x, u_y, u_z)
    - Vorticity field ω = (ω_x, ω_y, ω_z)
    - Gradient fields, etc.
    """
    x: QTT3DState
    y: QTT3DState
    z: QTT3DState
    
    @property
    def n_bits(self) -> int:
        return self.x.n_bits
    
    @property
    def N(self) -> int:
        return self.x.N
    
    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """(3, N, N, N)"""
        return (3, self.x.N, self.x.N, self.x.N)
    
    @property
    def max_rank(self) -> int:
        return max(self.x.max_rank, self.y.max_rank, self.z.max_rank)
    
    @property
    def mean_rank(self) -> float:
        return (self.x.mean_rank + self.y.mean_rank + self.z.mean_rank) / 3
    
    @property
    def qtt_parameters(self) -> int:
        return self.x.qtt_parameters + self.y.qtt_parameters + self.z.qtt_parameters
    
    @property
    def dense_parameters(self) -> int:
        return 3 * self.x.dense_parameters
    
    @property
    def compression_ratio(self) -> float:
        qtt_params = self.qtt_parameters
        if qtt_params == 0:
            return float('inf')
        return self.dense_parameters / qtt_params
    
    def clone(self) -> 'QTT3DVectorField':
        return QTT3DVectorField(
            x=self.x.clone(),
            y=self.y.clone(),
            z=self.z.clone(),
        )
    
    def to(self, device: torch.device) -> 'QTT3DVectorField':
        return QTT3DVectorField(
            x=self.x.to(device),
            y=self.y.to(device),
            z=self.z.to(device),
        )
    
    @classmethod
    def from_dense(
        cls,
        ux: Tensor,
        uy: Tensor,
        uz: Tensor,
        max_rank: int = 64,
        tol: float = 1e-6,
    ) -> 'QTT3DVectorField':
        """Compress 3D vector field to QTT with tolerance-controlled rank."""
        return cls(
            x=QTT3DState.from_dense(ux, max_rank=max_rank, tol=tol),
            y=QTT3DState.from_dense(uy, max_rank=max_rank, tol=tol),
            z=QTT3DState.from_dense(uz, max_rank=max_rank, tol=tol),
        )
    
    def to_dense(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Decompress to (ux, uy, uz)."""
        return self.x.to_dense(), self.y.to_dense(), self.z.to_dense()
    
    @classmethod
    def zeros(
        cls,
        n_bits: int,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ) -> 'QTT3DVectorField':
        return cls(
            x=QTT3DState.zeros(n_bits, device, dtype),
            y=QTT3DState.zeros(n_bits, device, dtype),
            z=QTT3DState.zeros(n_bits, device, dtype),
        )


# ═══════════════════════════════════════════════════════════════════════════════════════
# QTT 3D ARITHMETIC OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════════════

def qtt3d_add(
    a: QTT3DState,
    b: QTT3DState,
    max_rank: int = 64,
) -> QTT3DState:
    """
    Add two QTT3D states: c = a + b.
    
    Complexity: O(r³ × 3n) where r = max rank, n = bits per dim
    """
    assert a.n_bits == b.n_bits, "Mismatched n_bits"
    
    qtt_a = QTTState(cores=a.cores, num_qubits=a.total_qubits)
    qtt_b = QTTState(cores=b.cores, num_qubits=b.total_qubits)
    
    qtt_c = qtt_add(qtt_a, qtt_b, max_bond=max_rank)
    
    return QTT3DState(
        cores=qtt_c.cores,
        n_bits=a.n_bits,
        device=a.device,
        dtype=a.dtype,
    )


def qtt3d_scale(
    a: QTT3DState,
    scalar: float,
) -> QTT3DState:
    """
    Scale QTT3D state: c = scalar × a.
    
    Complexity: O(1) - just scales first core
    """
    cores = [c.clone() for c in a.cores]
    cores[0] = cores[0] * scalar
    
    return QTT3DState(
        cores=cores,
        n_bits=a.n_bits,
        device=a.device,
        dtype=a.dtype,
    )


def qtt3d_sub(
    a: QTT3DState,
    b: QTT3DState,
    max_rank: int = 64,
) -> QTT3DState:
    """
    Subtract QTT3D states: c = a - b.
    """
    neg_b = qtt3d_scale(b, -1.0)
    return qtt3d_add(a, neg_b, max_rank=max_rank)


def qtt3d_truncate(
    a: QTT3DState,
    max_rank: int = 64,
    tol: float = 0.0,
    rank_hint: int | None = None,
) -> tuple[QTT3DState, int]:
    """
    Truncate QTT3D to lower rank with adaptive SVD sizing.
    
    Uses SVD sweep for optimal truncation. Returns both the truncated
    state and the max observed rank for feeding into subsequent calls.
    
    Args:
        a: Input QTT3D state
        max_rank: Maximum bond dimension (hard ceiling)
        tol: Truncation tolerance (relative to max singular value)
        rank_hint: Estimated rank from previous operation
        
    Returns:
        (truncated_state, max_observed_rank)
    """
    cores, max_observed = truncate_cores_adaptive(
        a.cores, max_rank=max_rank, tol=tol, rank_hint=rank_hint
    )
    
    state = QTT3DState(
        cores=cores,
        n_bits=a.n_bits,
        device=a.device,
        dtype=a.dtype,
    )
    return state, max_observed


def qtt3d_truncate_simple(
    a: QTT3DState,
    max_rank: int = 64,
    tol: float = 0.0,
) -> QTT3DState:
    """Legacy wrapper - truncate without rank tracking."""
    state, _ = qtt3d_truncate(a, max_rank, tol)
    return state


# ═══════════════════════════════════════════════════════════════════════════════════════
# QTT 3D DERIVATIVE OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════════════

class QTT3DDerivatives:
    """
    Derivative operators for 3D QTT fields.
    
    Uses shift MPOs for O(log N) derivatives.
    
    Periodic boundary conditions via carry propagation.
    
    Example:
        >>> deriv = QTT3DDerivatives(n_bits=6, max_rank=64)
        >>> dfdx = deriv.ddx(f)
        >>> laplacian = deriv.laplacian(f)
    """
    
    def __init__(
        self,
        n_bits: int,
        max_rank: int = 64,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        L: float = 2 * np.pi,
    ):
        """
        Initialize derivative operators.
        
        Args:
            n_bits: Bits per dimension (N = 2^n_bits per axis)
            max_rank: Maximum rank after derivative
            device: Torch device
            dtype: Data type
            L: Physical domain size
        """
        self.n_bits = n_bits
        self.max_rank = max_rank
        self.device = device or torch.device('cpu')
        self.dtype = dtype
        self.L = L
        self.N = 1 << n_bits
        self.dx = L / self.N
        
        total_qubits = 3 * n_bits
        
        # Pre-build shift MPOs
        self._shift_plus = {}
        self._shift_minus = {}
        
        for axis in range(3):
            self._shift_plus[axis] = make_nd_shift_mpo(
                total_qubits, num_dims=3, axis_idx=axis, direction=+1,
                device=self.device, dtype=self.dtype,
            )
            self._shift_minus[axis] = make_nd_shift_mpo(
                total_qubits, num_dims=3, axis_idx=axis, direction=-1,
                device=self.device, dtype=self.dtype,
            )
    
    def _shift(self, f: QTT3DState, axis: int, direction: int) -> QTT3DState:
        """Apply shift operator."""
        mpo = self._shift_plus[axis] if direction > 0 else self._shift_minus[axis]
        cores = apply_nd_shift_mpo(f.cores, mpo, max_rank=self.max_rank)
        return QTT3DState(
            cores=cores,
            n_bits=f.n_bits,
            device=f.device,
            dtype=f.dtype,
        )
    
    def ddx(self, f: QTT3DState) -> QTT3DState:
        """
        First derivative ∂f/∂x (central difference).
        
        ∂f/∂x ≈ (f[i+1] - f[i-1]) / (2Δx)
        """
        f_plus = self._shift(f, axis=0, direction=+1)
        f_minus = self._shift(f, axis=0, direction=-1)
        
        diff = qtt3d_sub(f_plus, f_minus, max_rank=self.max_rank)
        return qtt3d_scale(diff, 1.0 / (2 * self.dx))
    
    def ddy(self, f: QTT3DState) -> QTT3DState:
        """First derivative ∂f/∂y (central difference)."""
        f_plus = self._shift(f, axis=1, direction=+1)
        f_minus = self._shift(f, axis=1, direction=-1)
        
        diff = qtt3d_sub(f_plus, f_minus, max_rank=self.max_rank)
        return qtt3d_scale(diff, 1.0 / (2 * self.dx))
    
    def ddz(self, f: QTT3DState) -> QTT3DState:
        """First derivative ∂f/∂z (central difference)."""
        f_plus = self._shift(f, axis=2, direction=+1)
        f_minus = self._shift(f, axis=2, direction=-1)
        
        diff = qtt3d_sub(f_plus, f_minus, max_rank=self.max_rank)
        return qtt3d_scale(diff, 1.0 / (2 * self.dx))
    
    def d2dx2(self, f: QTT3DState) -> QTT3DState:
        """
        Second derivative ∂²f/∂x² (central difference).
        
        ∂²f/∂x² ≈ (f[i+1] - 2f[i] + f[i-1]) / Δx²
        """
        f_plus = self._shift(f, axis=0, direction=+1)
        f_minus = self._shift(f, axis=0, direction=-1)
        
        # f+ + f- - 2f
        sum_shifted = qtt3d_add(f_plus, f_minus, max_rank=self.max_rank)
        neg_2f = qtt3d_scale(f, -2.0)
        result = qtt3d_add(sum_shifted, neg_2f, max_rank=self.max_rank)
        
        return qtt3d_scale(result, 1.0 / (self.dx ** 2))
    
    def d2dy2(self, f: QTT3DState) -> QTT3DState:
        """Second derivative ∂²f/∂y²."""
        f_plus = self._shift(f, axis=1, direction=+1)
        f_minus = self._shift(f, axis=1, direction=-1)
        
        sum_shifted = qtt3d_add(f_plus, f_minus, max_rank=self.max_rank)
        neg_2f = qtt3d_scale(f, -2.0)
        result = qtt3d_add(sum_shifted, neg_2f, max_rank=self.max_rank)
        
        return qtt3d_scale(result, 1.0 / (self.dx ** 2))
    
    def d2dz2(self, f: QTT3DState) -> QTT3DState:
        """Second derivative ∂²f/∂z²."""
        f_plus = self._shift(f, axis=2, direction=+1)
        f_minus = self._shift(f, axis=2, direction=-1)
        
        sum_shifted = qtt3d_add(f_plus, f_minus, max_rank=self.max_rank)
        neg_2f = qtt3d_scale(f, -2.0)
        result = qtt3d_add(sum_shifted, neg_2f, max_rank=self.max_rank)
        
        return qtt3d_scale(result, 1.0 / (self.dx ** 2))
    
    def laplacian(self, f: QTT3DState) -> QTT3DState:
        """
        Laplacian ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z².
        """
        d2x = self.d2dx2(f)
        d2y = self.d2dy2(f)
        d2z = self.d2dz2(f)
        
        result = qtt3d_add(d2x, d2y, max_rank=self.max_rank)
        result = qtt3d_add(result, d2z, max_rank=self.max_rank)
        
        return result
    
    def gradient(self, f: QTT3DState) -> QTT3DVectorField:
        """
        Gradient ∇f = (∂f/∂x, ∂f/∂y, ∂f/∂z).
        """
        return QTT3DVectorField(
            x=self.ddx(f),
            y=self.ddy(f),
            z=self.ddz(f),
        )
    
    def divergence(self, v: QTT3DVectorField) -> QTT3DState:
        """
        Divergence ∇·v = ∂vx/∂x + ∂vy/∂y + ∂vz/∂z.
        """
        dvx_dx = self.ddx(v.x)
        dvy_dy = self.ddy(v.y)
        dvz_dz = self.ddz(v.z)
        
        result = qtt3d_add(dvx_dx, dvy_dy, max_rank=self.max_rank)
        result = qtt3d_add(result, dvz_dz, max_rank=self.max_rank)
        
        return result
    
    def curl(self, v: QTT3DVectorField) -> QTT3DVectorField:
        """
        Curl ∇×v = (∂vz/∂y - ∂vy/∂z, ∂vx/∂z - ∂vz/∂x, ∂vy/∂x - ∂vx/∂y).
        """
        # ωx = ∂vz/∂y - ∂vy/∂z
        dvz_dy = self.ddy(v.z)
        dvy_dz = self.ddz(v.y)
        omega_x = qtt3d_sub(dvz_dy, dvy_dz, max_rank=self.max_rank)
        
        # ωy = ∂vx/∂z - ∂vz/∂x
        dvx_dz = self.ddz(v.x)
        dvz_dx = self.ddx(v.z)
        omega_y = qtt3d_sub(dvx_dz, dvz_dx, max_rank=self.max_rank)
        
        # ωz = ∂vy/∂x - ∂vx/∂y
        dvy_dx = self.ddx(v.y)
        dvx_dy = self.ddy(v.x)
        omega_z = qtt3d_sub(dvy_dx, dvx_dy, max_rank=self.max_rank)
        
        return QTT3DVectorField(x=omega_x, y=omega_y, z=omega_z)
    
    def laplacian_vector(self, v: QTT3DVectorField) -> QTT3DVectorField:
        """
        Vector Laplacian ∇²v = (∇²vx, ∇²vy, ∇²vz).
        """
        return QTT3DVectorField(
            x=self.laplacian(v.x),
            y=self.laplacian(v.y),
            z=self.laplacian(v.z),
        )


# ═══════════════════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class QTT3DDiagnostics:
    """Diagnostics for 3D QTT turbulence simulation."""
    time: float
    kinetic_energy: float      # E = ½∫|u|² dV
    enstrophy: float           # Z = ½∫|ω|² dV
    max_vorticity: float       # max|ω|
    max_velocity: float        # max|u|
    divergence_max: float      # max|∇·u| (should be ~0)
    max_rank: int              # Max QTT rank
    mean_rank: float           # Mean QTT rank
    compression_ratio: float   # Dense/QTT parameters


def compute_diagnostics(
    u: QTT3DVectorField,
    omega: QTT3DVectorField,
    deriv: QTT3DDerivatives,
    t: float,
) -> QTT3DDiagnostics:
    """
    Compute comprehensive diagnostics.
    
    Note: This decompresses fields for accurate diagnostics.
    In production, use approximate QTT-native diagnostics.
    """
    # Decompress for accurate diagnostics
    ux, uy, uz = u.to_dense()
    ox, oy, oz = omega.to_dense()
    
    dx = deriv.dx
    dV = dx ** 3
    
    # Kinetic energy
    u_sq = ux**2 + uy**2 + uz**2
    kinetic_energy = 0.5 * u_sq.sum().item() * dV
    
    # Enstrophy
    omega_sq = ox**2 + oy**2 + oz**2
    enstrophy = 0.5 * omega_sq.sum().item() * dV
    
    # Max values
    max_vorticity = torch.sqrt(omega_sq).max().item()
    max_velocity = torch.sqrt(u_sq).max().item()
    
    # Divergence
    div = deriv.divergence(u)
    div_dense = div.to_dense()
    divergence_max = torch.abs(div_dense).max().item()
    
    return QTT3DDiagnostics(
        time=t,
        kinetic_energy=kinetic_energy,
        enstrophy=enstrophy,
        max_vorticity=max_vorticity,
        max_velocity=max_velocity,
        divergence_max=divergence_max,
        max_rank=u.max_rank,
        mean_rank=u.mean_rank,
        compression_ratio=u.compression_ratio,
    )


# ═══════════════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════════════

__all__ = [
    'QTT3DState',
    'QTT3DVectorField',
    'qtt3d_add',
    'qtt3d_sub',
    'qtt3d_scale',
    'qtt3d_truncate',
    'QTT3DDerivatives',
    'QTT3DDiagnostics',
    'compute_diagnostics',
]
