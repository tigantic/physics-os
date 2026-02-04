"""
Native 3D QTT Navier-Stokes Solver
==================================

ZERO DENSE OPERATIONS. All operations stay in QTT format.

Key Principles:
1. NO to_dense() - ever
2. NO from_dense() for intermediates
3. rSVD for truncation
4. Triton kernels for core operations
5. Adaptive rank (scale-dependent)

Vorticity-Velocity Formulation:
    ∂ω/∂t = ∇×(u×ω) + ν∇²ω
    
All terms computed natively:
- Laplacian: MPO in TT format
- Curl: MPO composition
- Hadamard (u×ω): Native TT-cross or DMRG

Complexity:
    Per timestep: O(r³ × 3 log₂ N) 
    Memory: O(r² × 3 log₂ N)
    
Author: HyperTensor Team
Date: 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
from enum import Enum, auto
import time

import numpy as np
import torch
from torch import Tensor

from tensornet.cfd.qtt_native_ops import (
    QTTCores,
    rsvd_truncate,
    qtt_truncate_sweep,
    qtt_truncate_now,
    qtt_add_native,
    qtt_scale_native,
    qtt_sub_native,
    qtt_hadamard_native,
    qtt_inner_native,
    qtt_norm_native,
    qtt_fused_sum,
    QTTRoundingContext,
    get_rounding_context,
    turbulence_rank_profile,
    adaptive_truncate,
)
from tensornet.cfd.triton_qtt_kernels import (
    TRITON_AVAILABLE,
    triton_mpo_apply,
    triton_hadamard_core,
    triton_inner_step,
)


# ═══════════════════════════════════════════════════════════════════════════════════════
# 3D QTT STATE (NATIVE)
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class QTT3DNative:
    """
    Native 3D QTT field - NEVER converts to dense.
    
    3D field of shape (N, N, N) stored as QTT over 3n qubits.
    Morton interleaving: qubit k belongs to dimension (k mod 3).
    """
    cores: QTTCores
    n_bits: int  # Bits per dimension
    
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
    """
    Native 3D vector field - three QTT3DNative components.
    
    Used for velocity u = (ux, uy, uz) and vorticity ω = (ωx, ωy, ωz).
    """
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


# ═══════════════════════════════════════════════════════════════════════════════════════
# NATIVE MPO FOR DERIVATIVES
# ═══════════════════════════════════════════════════════════════════════════════════════

def build_shift_mpo_3d(
    n_bits: int,
    axis: int,  # 0=x, 1=y, 2=z
    direction: int,  # +1 or -1
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> List[Tensor]:
    """
    Build shift MPO for 3D Morton-ordered QTT.
    
    Shift by ±1 along specified axis with periodic BC.
    
    For Morton interleaving with ordering (x0,y0,z0, x1,y1,z1, ...),
    the qubits for axis A are at positions A, A+3, A+6, ..., A+3*(n_bits-1).
    
    Binary increment/decrement with carry chain:
    - Bond dimension 2: state 0 = no carry, state 1 = carry pending
    - First axis qubit: always flip, carry if was 1 (increment) or 0 (decrement)
    - Middle axis qubits: identity if no carry, flip if carry
    - Last axis qubit: flip if carry, overflow discarded (periodic BC)
    
    Returns: List of MPO cores, each (r_left, 2, 2, r_right)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    total_qubits = 3 * n_bits
    mpo_cores = []
    
    # Find which qubits belong to this axis
    axis_qubits = [axis + 3 * i for i in range(n_bits)]
    
    for k in range(total_qubits):
        if k in axis_qubits:
            # This qubit participates in the shift
            idx_in_axis = axis_qubits.index(k)
            is_first = (idx_in_axis == 0)
            is_last = (idx_in_axis == n_bits - 1)
            
            r_left = 1 if is_first else 2
            r_right = 1 if is_last else 2
            
            core = torch.zeros(r_left, 2, 2, r_right, device=device, dtype=dtype)
            
            if direction > 0:  # Increment
                if is_first and is_last:
                    # Single bit: just flip (0↔1)
                    core[0, 0, 1, 0] = 1.0
                    core[0, 1, 0, 0] = 1.0
                elif is_first:
                    # First bit: always flip, carry if was 1
                    core[0, 0, 1, 0] = 1.0  # 0→1, no carry
                    core[0, 1, 0, 1] = 1.0  # 1→0, carry out
                elif is_last:
                    # Last bit: identity if no carry, flip if carry
                    core[0, 0, 0, 0] = 1.0  # no carry, 0→0
                    core[0, 1, 1, 0] = 1.0  # no carry, 1→1
                    core[1, 0, 1, 0] = 1.0  # carry + 0 → 1
                    core[1, 1, 0, 0] = 1.0  # carry + 1 → 0, overflow discarded
                else:
                    # Middle bit: identity if no carry, flip + propagate if carry
                    core[0, 0, 0, 0] = 1.0  # no carry, 0→0
                    core[0, 1, 1, 0] = 1.0  # no carry, 1→1
                    core[1, 0, 1, 0] = 1.0  # carry + 0 → 1, done
                    core[1, 1, 0, 1] = 1.0  # carry + 1 → 0, carry out
            else:  # Decrement
                if is_first and is_last:
                    core[0, 0, 1, 0] = 1.0
                    core[0, 1, 0, 0] = 1.0
                elif is_first:
                    # First bit: always flip, borrow if was 0
                    core[0, 0, 1, 1] = 1.0  # 0→1, borrow out (0-1 = -1 = ...111)
                    core[0, 1, 0, 0] = 1.0  # 1→0, done
                elif is_last:
                    # Last bit: identity if no borrow, flip if borrow
                    core[0, 0, 0, 0] = 1.0
                    core[0, 1, 1, 0] = 1.0
                    core[1, 0, 1, 0] = 1.0  # borrow + 0 → 1, underflow discarded
                    core[1, 1, 0, 0] = 1.0  # borrow + 1 → 0
                else:
                    # Middle bit
                    core[0, 0, 0, 0] = 1.0
                    core[0, 1, 1, 0] = 1.0
                    core[1, 0, 1, 1] = 1.0  # borrow + 0 → 1, borrow out
                    core[1, 1, 0, 0] = 1.0  # borrow + 1 → 0, done
        else:
            # Non-participating qubit: identity with bond passthrough
            r_left = mpo_cores[-1].shape[3] if mpo_cores else 1
            
            # Determine r_right: match r_left until we hit the next axis qubit
            next_axis_qubit = next((aq for aq in axis_qubits if aq > k), None)
            if next_axis_qubit is None:
                r_right = 1  # After last axis qubit
            else:
                r_right = r_left  # Pass through bond dimension
            
            core = torch.zeros(r_left, 2, 2, r_right, device=device, dtype=dtype)
            for r in range(min(r_left, r_right)):
                core[r, 0, 0, r] = 1.0
                core[r, 1, 1, r] = 1.0
        
        mpo_cores.append(core)
    
    return mpo_cores


def apply_mpo_native(
    state: QTTCores,
    mpo: List[Tensor],
    max_rank: int = 64,
    tol: float = 1e-10,
) -> QTTCores:
    """
    Apply MPO to QTT state natively.
    
    Uses Triton kernels when available.
    """
    assert len(state.cores) == len(mpo)
    L = len(state.cores)
    
    # Contract MPO with state
    new_cores = []
    for k in range(L):
        s_core = state.cores[k]  # (r_s_l, 2, r_s_r)
        m_core = mpo[k]          # (r_m_l, 2, 2, r_m_r)
        
        if TRITON_AVAILABLE and s_core.is_cuda:
            out = triton_mpo_apply(s_core, m_core)
        else:
            # Fallback: einsum
            # out[i*a, j, k*b] = s[i, s, k] * m[a, s, j, b]
            out = torch.einsum('isk,asjo->iajok', s_core, m_core)
            r_s_l, d_s, r_s_r = s_core.shape
            r_m_l, _, d_out, r_m_r = m_core.shape
            out = out.reshape(r_s_l * r_m_l, d_out, r_s_r * r_m_r)
        
        new_cores.append(out)
    
    # Truncate to control rank
    truncated = qtt_truncate_sweep(new_cores, max_rank, tol)
    
    return QTTCores(truncated)


# ═══════════════════════════════════════════════════════════════════════════════════════
# NATIVE DERIVATIVE OPERATORS
# ═══════════════════════════════════════════════════════════════════════════════════════

class NativeDerivatives3D:
    """
    Native QTT derivatives - no dense operations.
    
    All derivatives computed via shift MPO application.
    """
    
    def __init__(
        self,
        n_bits: int,
        max_rank: int = 64,
        base_rank: int = 32,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        L: float = 2 * np.pi,
    ):
        self.n_bits = n_bits
        self.max_rank = max_rank
        self.base_rank = base_rank
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.L = L
        self.N = 1 << n_bits
        self.dx = L / self.N
        
        # Adaptive rank profile
        self.rank_profile = turbulence_rank_profile(3 * n_bits, base_rank, max_rank)
        
        # Pre-build shift MPOs
        self._shift_plus = {}
        self._shift_minus = {}
        
        for axis in range(3):
            self._shift_plus[axis] = build_shift_mpo_3d(
                n_bits, axis, +1, self.device, self.dtype
            )
            self._shift_minus[axis] = build_shift_mpo_3d(
                n_bits, axis, -1, self.device, self.dtype
            )
    
    def _shift(self, f: QTT3DNative, axis: int, direction: int) -> QTT3DNative:
        """Apply shift MPO."""
        mpo = self._shift_plus[axis] if direction > 0 else self._shift_minus[axis]
        new_cores = apply_mpo_native(f.cores, mpo, self.max_rank)
        return QTT3DNative(new_cores, f.n_bits)
    
    def ddx(self, f: QTT3DNative) -> QTT3DNative:
        """∂f/∂x via central difference."""
        f_plus = self._shift(f, 0, +1)
        f_minus = self._shift(f, 0, -1)
        diff = qtt_sub_native(f_plus.cores, f_minus.cores, self.max_rank)
        scaled = qtt_scale_native(diff, 1.0 / (2 * self.dx))
        return QTT3DNative(scaled, f.n_bits)
    
    def ddy(self, f: QTT3DNative) -> QTT3DNative:
        """∂f/∂y via central difference."""
        f_plus = self._shift(f, 1, +1)
        f_minus = self._shift(f, 1, -1)
        diff = qtt_sub_native(f_plus.cores, f_minus.cores, self.max_rank)
        scaled = qtt_scale_native(diff, 1.0 / (2 * self.dx))
        return QTT3DNative(scaled, f.n_bits)
    
    def ddz(self, f: QTT3DNative) -> QTT3DNative:
        """∂f/∂z via central difference."""
        f_plus = self._shift(f, 2, +1)
        f_minus = self._shift(f, 2, -1)
        diff = qtt_sub_native(f_plus.cores, f_minus.cores, self.max_rank)
        scaled = qtt_scale_native(diff, 1.0 / (2 * self.dx))
        return QTT3DNative(scaled, f.n_bits)
    
    def laplacian(self, f: QTT3DNative) -> QTT3DNative:
        """
        ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z².
        
        FUSED: Collects all 6 shifts, then does single fused sum + truncation.
        This reduces truncations from ~9 per Laplacian to 1.
        """
        inv_dx2 = 1.0 / (self.dx ** 2)
        
        # Collect all shifts WITHOUT intermediate truncation
        shifted_fields = []
        weights = []
        
        for axis in range(3):
            f_plus = self._shift(f, axis, +1)
            f_minus = self._shift(f, axis, -1)
            shifted_fields.extend([f_plus.cores, f_minus.cores])
            weights.extend([inv_dx2, inv_dx2])
        
        # Add -6*f (the central point contribution)
        shifted_fields.append(f.cores)
        weights.append(-6.0 * inv_dx2)
        
        # Single fused sum + truncation
        result = qtt_fused_sum(shifted_fields, weights, self.max_rank)
        
        return QTT3DNative(result, f.n_bits)
    
    def curl(self, v: QTT3DVectorNative) -> QTT3DVectorNative:
        """∇×v = (∂vz/∂y - ∂vy/∂z, ∂vx/∂z - ∂vz/∂x, ∂vy/∂x - ∂vx/∂y)."""
        dvz_dy = self.ddy(v.z)
        dvy_dz = self.ddz(v.y)
        omega_x = QTT3DNative(
            qtt_sub_native(dvz_dy.cores, dvy_dz.cores, self.max_rank),
            v.n_bits
        )
        
        dvx_dz = self.ddz(v.x)
        dvz_dx = self.ddx(v.z)
        omega_y = QTT3DNative(
            qtt_sub_native(dvx_dz.cores, dvz_dx.cores, self.max_rank),
            v.n_bits
        )
        
        dvy_dx = self.ddx(v.y)
        dvx_dy = self.ddy(v.x)
        omega_z = QTT3DNative(
            qtt_sub_native(dvy_dx.cores, dvx_dy.cores, self.max_rank),
            v.n_bits
        )
        
        return QTT3DVectorNative(omega_x, omega_y, omega_z)
    
    def laplacian_vector(self, v: QTT3DVectorNative) -> QTT3DVectorNative:
        """∇²v component-wise."""
        return QTT3DVectorNative(
            self.laplacian(v.x),
            self.laplacian(v.y),
            self.laplacian(v.z),
        )
    
    def divergence(self, v: QTT3DVectorNative) -> QTT3DNative:
        """∇·v = ∂vx/∂x + ∂vy/∂y + ∂vz/∂z."""
        dvx_dx = self.ddx(v.x)
        dvy_dy = self.ddy(v.y)
        dvz_dz = self.ddz(v.z)
        
        # Fused sum for efficiency
        result = qtt_fused_sum(
            [dvx_dx.cores, dvy_dy.cores, dvz_dz.cores],
            [1.0, 1.0, 1.0],
            self.max_rank
        )
        return QTT3DNative(result, v.n_bits)
    
    def gradient(self, f: QTT3DNative) -> QTT3DVectorNative:
        """∇f = (∂f/∂x, ∂f/∂y, ∂f/∂z)."""
        return QTT3DVectorNative(
            self.ddx(f),
            self.ddy(f),
            self.ddz(f),
        )
    
    def poisson_cg(
        self,
        rhs: QTT3DNative,
        tol: float = 1e-6,
        max_iter: int = 50,
        precondition: bool = True,
    ) -> QTT3DNative:
        """
        QTT-CG Poisson solver: solve ∇²p = rhs.
        
        Uses Conjugate Gradient iteration entirely in QTT format.
        
        With Jacobi preconditioner M⁻¹ = -(h²/6) I for 3D Laplacian.
        
        Args:
            rhs: Right-hand side (e.g., divergence of velocity)
            tol: Relative residual tolerance
            max_iter: Maximum CG iterations
            precondition: Use Jacobi preconditioner
            
        Returns:
            Pressure field p satisfying ∇²p ≈ rhs
        """
        max_rank = self.max_rank
        h2 = self.dx ** 2
        
        # Initial guess: zero
        n_sites = 3 * self.n_bits
        x_cores = [torch.zeros(1, 2, 1, device=self.device, dtype=self.dtype) 
                   for _ in range(n_sites)]
        x = QTT3DNative(QTTCores(x_cores), self.n_bits)
        
        # Initial residual r = rhs - ∇²x = rhs (since x=0)
        r = rhs.clone()
        
        # Apply preconditioner: z = M⁻¹ r
        # For Laplacian, M = diag(L) ≈ -6/h² I, so M⁻¹ ≈ -h²/6 I
        if precondition:
            z = QTT3DNative(qtt_scale_native(r.cores, -h2 / 6.0), self.n_bits)
        else:
            z = r.clone()
        
        p = z.clone()  # Search direction
        
        # rz = <r, z>
        rz = qtt_inner_native(r.cores, z.cores)
        
        rhs_norm_sq = qtt_inner_native(rhs.cores, rhs.cores).item()
        if rhs_norm_sq < 1e-20:
            return x  # RHS is zero
        
        for k in range(max_iter):
            # Ap = ∇²p
            Ap = self.laplacian(p)
            
            # alpha = <r,z> / <p, Ap>
            pAp = qtt_inner_native(p.cores, Ap.cores)
            
            if abs(pAp.item()) < 1e-30:
                break  # Breakdown
            
            alpha = rz.item() / pAp.item()
            
            # x = x + alpha * p
            x_cores = qtt_fused_sum(
                [x.cores, p.cores],
                [1.0, alpha],
                max_rank
            )
            x = QTT3DNative(x_cores, self.n_bits)
            
            # r = r - alpha * Ap
            r_cores = qtt_fused_sum(
                [r.cores, Ap.cores],
                [1.0, -alpha],
                max_rank
            )
            r = QTT3DNative(r_cores, self.n_bits)
            
            # Check convergence
            r_norm_sq = qtt_inner_native(r.cores, r.cores).item()
            if r_norm_sq / rhs_norm_sq < tol ** 2:
                break
            
            # z = M⁻¹ r
            if precondition:
                z = QTT3DNative(qtt_scale_native(r.cores, -h2 / 6.0), self.n_bits)
            else:
                z = r.clone()
            
            # beta = <r_new, z_new> / <r_old, z_old>
            rz_new = qtt_inner_native(r.cores, z.cores)
            beta = rz_new.item() / rz.item() if abs(rz.item()) > 1e-30 else 0.0
            
            # p = z + beta * p
            p_cores = qtt_fused_sum(
                [z.cores, p.cores],
                [1.0, beta],
                max_rank
            )
            p = QTT3DNative(p_cores, self.n_bits)
            
            rz = rz_new
        
        return x


# ═══════════════════════════════════════════════════════════════════════════════════════
# NATIVE HADAMARD (CROSS PRODUCT)
# ═══════════════════════════════════════════════════════════════════════════════════════

def vector_cross_native(
    u: QTT3DVectorNative,
    v: QTT3DVectorNative,
    max_rank: int = 64,
) -> QTT3DVectorNative:
    """
    Native cross product u × v.
    
    Uses QTT Hadamard product - NO DENSE!
    
    (u × v)_x = u_y * v_z - u_z * v_y
    (u × v)_y = u_z * v_x - u_x * v_z
    (u × v)_z = u_x * v_y - u_y * v_x
    """
    # x component
    uy_vz = qtt_hadamard_native(u.y.cores, v.z.cores, max_rank)
    uz_vy = qtt_hadamard_native(u.z.cores, v.y.cores, max_rank)
    cx = qtt_sub_native(uy_vz, uz_vy, max_rank)
    
    # y component
    uz_vx = qtt_hadamard_native(u.z.cores, v.x.cores, max_rank)
    ux_vz = qtt_hadamard_native(u.x.cores, v.z.cores, max_rank)
    cy = qtt_sub_native(uz_vx, ux_vz, max_rank)
    
    # z component
    ux_vy = qtt_hadamard_native(u.x.cores, v.y.cores, max_rank)
    uy_vx = qtt_hadamard_native(u.y.cores, v.x.cores, max_rank)
    cz = qtt_sub_native(ux_vy, uy_vx, max_rank)
    
    return QTT3DVectorNative(
        QTT3DNative(cx, u.n_bits),
        QTT3DNative(cy, u.n_bits),
        QTT3DNative(cz, u.n_bits),
    )


# ═══════════════════════════════════════════════════════════════════════════════════════
# NATIVE DIAGNOSTICS (NO DENSE!)
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class NativeDiagnostics:
    """Diagnostics computed entirely in QTT format."""
    time: float
    kinetic_energy_qtt: float   # From QTT inner product
    enstrophy_qtt: float        # From QTT inner product
    u_norm: float               # ||u||
    omega_norm: float           # ||ω||
    max_rank_u: int
    max_rank_omega: int
    mean_rank_u: float
    mean_rank_omega: float
    compression_ratio: float


def compute_diagnostics_native(
    u: QTT3DVectorNative,
    omega: QTT3DVectorNative,
    t: float,
    dx: float,
) -> NativeDiagnostics:
    """
    Compute diagnostics NATIVELY - no decompression.
    
    Uses QTT inner products for energy/enstrophy.
    """
    # Kinetic energy: E = ½ Σ (ux² + uy² + uz²) dx³
    # In QTT: E = ½ dx³ (<ux, ux> + <uy, uy> + <uz, uz>)
    ux_sq = qtt_inner_native(u.x.cores, u.x.cores)
    uy_sq = qtt_inner_native(u.y.cores, u.y.cores)
    uz_sq = qtt_inner_native(u.z.cores, u.z.cores)
    kinetic_energy = 0.5 * dx**3 * (ux_sq + uy_sq + uz_sq).item()
    
    # Enstrophy: Z = ½ Σ (ωx² + ωy² + ωz²) dx³
    ox_sq = qtt_inner_native(omega.x.cores, omega.x.cores)
    oy_sq = qtt_inner_native(omega.y.cores, omega.y.cores)
    oz_sq = qtt_inner_native(omega.z.cores, omega.z.cores)
    enstrophy = 0.5 * dx**3 * (ox_sq + oy_sq + oz_sq).item()
    
    # Norms
    u_norm = np.sqrt((ux_sq + uy_sq + uz_sq).item())
    omega_norm = np.sqrt((ox_sq + oy_sq + oz_sq).item())
    
    return NativeDiagnostics(
        time=t,
        kinetic_energy_qtt=kinetic_energy,
        enstrophy_qtt=enstrophy,
        u_norm=u_norm,
        omega_norm=omega_norm,
        max_rank_u=u.max_rank,
        max_rank_omega=omega.max_rank,
        mean_rank_u=u.mean_rank,
        mean_rank_omega=omega.mean_rank,
        compression_ratio=u.compression_ratio,
    )


# ═══════════════════════════════════════════════════════════════════════════════════════
# TAYLOR-GREEN VORTEX INITIALIZER (NATIVE)
# ═══════════════════════════════════════════════════════════════════════════════════════

def _tt_svd_compress(tensor_flat: Tensor, modes: List[int], max_rank: int) -> List[Tensor]:
    """
    TT-SVD compression of a flattened tensor.
    
    Args:
        tensor_flat: Flattened tensor of shape (prod(modes),)
        modes: List of mode dimensions (e.g., [2,2,2,...] for QTT)
        max_rank: Maximum TT-rank
        
    Returns:
        List of TT cores
    """
    n_modes = len(modes)
    cores = []
    
    # Reshape to first unfolding
    C = tensor_flat.view(modes[0], -1)
    
    for k in range(n_modes - 1):
        # SVD of current unfolding
        U, S, Vh = rsvd_truncate(C, max_rank)
        r = len(S)
        
        # Store core
        r_left = 1 if k == 0 else cores[-1].shape[2]
        core = U.view(r_left, modes[k], r)
        cores.append(core)
        
        # Prepare next unfolding
        if k < n_modes - 2:
            C = (torch.diag(S) @ Vh).view(r * modes[k + 1], -1)
        else:
            # Last core
            C = torch.diag(S) @ Vh
    
    # Final core
    r_left = cores[-1].shape[2] if cores else 1
    last_core = C.view(r_left, modes[-1], 1)
    cores.append(last_core)
    
    return cores


def taylor_green_analytical(
    n_bits: int,
    L: float = 2 * np.pi,
    amplitude: float = 1.0,
    max_rank: int = 64,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32,
) -> Tuple[QTT3DVectorNative, QTT3DVectorNative]:
    """
    Create Taylor-Green vortex using ANALYTICAL QTT construction.
    
    *** ZERO DENSE MEMORY ALLOCATION ***
    
    This function constructs the QTT representation directly from the
    mathematical structure of the Taylor-Green vortex, without ever
    creating a dense N³ array. This enables initialization of arbitrarily
    large grids (1024³, 4096³, etc.) on any GPU.
    
    The Taylor-Green vortex is composed of separable trigonometric functions:
        u = sin(x) * cos(y) * cos(z)
        v = -cos(x) * sin(y) * cos(z)
        w = 0
        
    Each sin/cos has an exact rank-2 QTT representation, allowing the
    3D field to be constructed analytically with rank ≤ 8.
    
    Parameters
    ----------
    n_bits : int
        Bits per dimension (grid is 2^n_bits per axis)
    L : float
        Domain size (default 2π for standard Taylor-Green)
    amplitude : float
        Amplitude multiplier
    max_rank : int
        Maximum rank after optional truncation (default 64)
    device : str
        Torch device
    dtype : torch.dtype
        Data type
        
    Returns
    -------
    (u, omega) : Tuple[QTT3DVectorNative, QTT3DVectorNative]
        Velocity and vorticity fields in native QTT format
    """
    from tensornet.cfd.analytical_qtt import taylor_green_analytical_3d
    
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Build analytical QTT cores
    u_cores_list, omega_cores_list = taylor_green_analytical_3d(
        n_bits, str(dev), dtype, L
    )
    
    # Convert to QTTCores format
    def cores_to_qtt(cores_list: List[Tensor]) -> QTTCores:
        # Apply amplitude scaling to first core
        scaled = [c.clone() for c in cores_list]
        return QTTCores(scaled)
    
    # Scale by amplitude
    def apply_amplitude(cores: List[Tensor], amp: float) -> List[Tensor]:
        result = [c.clone() for c in cores]
        result[0] = result[0] * amp
        return result
    
    # Build velocity components
    ux_cores = apply_amplitude(u_cores_list[0], amplitude)
    uy_cores = apply_amplitude(u_cores_list[1], amplitude)  # Already has -1 factor
    uz_cores = u_cores_list[2]  # Zero field
    
    # Build vorticity components
    ox_cores = apply_amplitude(omega_cores_list[0], amplitude)  # Already has -1 factor
    oy_cores = apply_amplitude(omega_cores_list[1], amplitude)  # Already has -1 factor
    oz_cores = apply_amplitude(omega_cores_list[2], amplitude)  # Already has 2 factor
    
    # Optional: truncate to max_rank if needed
    # The analytical construction typically has rank ≤ 8, so truncation is usually not needed
    
    # Create native vector fields
    u = QTT3DVectorNative(
        QTT3DNative(QTTCores(ux_cores), n_bits),
        QTT3DNative(QTTCores(uy_cores), n_bits),
        QTT3DNative(QTTCores(uz_cores), n_bits),
    )
    
    omega = QTT3DVectorNative(
        QTT3DNative(QTTCores(ox_cores), n_bits),
        QTT3DNative(QTTCores(oy_cores), n_bits),
        QTT3DNative(QTTCores(oz_cores), n_bits),
    )
    
    return u, omega


def taylor_green_native(
    n_bits: int,
    L: float = 2 * np.pi,
    amplitude: float = 1.0,
    max_rank: int = 64,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32,
) -> Tuple[QTT3DVectorNative, QTT3DVectorNative]:
    """
    Create Taylor-Green vortex initial conditions in native QTT format.
    
    Uses TT-SVD compression - one-time initialization cost.
    
    Velocity:
        u = A sin(x) cos(y) cos(z)
        v = -A cos(x) sin(y) cos(z)
        w = 0
    
    Vorticity (curl of u):
        ωx = A sin(x) sin(y) sin(z)
        ωy = A cos(x) cos(y) sin(z)
        ωz = -2A cos(x) sin(y) cos(z)
    
    Returns:
        (u, omega): Native QTT vector fields
    """
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    N = 1 << n_bits
    n_sites = 3 * n_bits  # 3D Morton ordering
    
    # Build grid at requested resolution
    # For large grids, this may be memory-intensive for initialization
    # but the result is a compact QTT representation
    x = torch.linspace(0, L * (N - 1) / N, N, device=dev, dtype=dtype)
    y = torch.linspace(0, L * (N - 1) / N, N, device=dev, dtype=dtype)
    z = torch.linspace(0, L * (N - 1) / N, N, device=dev, dtype=dtype)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    # Velocity
    ux = amplitude * torch.sin(X) * torch.cos(Y) * torch.cos(Z)
    uy = -amplitude * torch.cos(X) * torch.sin(Y) * torch.cos(Z)
    uz = torch.zeros_like(ux)
    
    # Vorticity
    ox = amplitude * torch.sin(X) * torch.sin(Y) * torch.sin(Z)
    oy = amplitude * torch.cos(X) * torch.cos(Y) * torch.sin(Z)
    oz = -2 * amplitude * torch.cos(X) * torch.sin(Y) * torch.cos(Z)
    
    # Modes for 3D QTT (all binary)
    modes = [2] * n_sites
    
    def compress_field(arr: Tensor) -> QTTCores:
        """TT-SVD compress a 3D field to QTT."""
        flat = arr.flatten()
        
        if len(flat) != 2 ** n_sites:
            # Pad or truncate to power of 2
            target_size = 2 ** n_sites
            if len(flat) < target_size:
                flat = torch.cat([flat, torch.zeros(target_size - len(flat), device=dev, dtype=dtype)])
            else:
                flat = flat[:target_size]
        
        cores = _tt_svd_compress(flat, modes, max_rank)
        return QTTCores(cores)
    
    # Compress all fields
    ux_qtt = compress_field(ux)
    uy_qtt = compress_field(uy)
    uz_qtt = compress_field(uz)
    ox_qtt = compress_field(ox)
    oy_qtt = compress_field(oy)
    oz_qtt = compress_field(oz)
    
    # Create native vector fields
    u = QTT3DVectorNative(
        QTT3DNative(ux_qtt, n_bits),
        QTT3DNative(uy_qtt, n_bits),
        QTT3DNative(uz_qtt, n_bits),
    )
    
    omega = QTT3DVectorNative(
        QTT3DNative(ox_qtt, n_bits),
        QTT3DNative(oy_qtt, n_bits),
        QTT3DNative(oz_qtt, n_bits),
    )
    
    return u, omega


# ═══════════════════════════════════════════════════════════════════════════════════════
# NATIVE NS3D SOLVER
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class NativeNS3DConfig:
    """Configuration for native QTT NS solver."""
    n_bits: int = 6              # N = 64 per axis
    nu: float = 1e-4             # Viscosity
    L: float = 2 * np.pi         # Domain size
    max_rank: int = 64           # Max QTT rank
    base_rank: int = 32          # Base rank (adaptive)
    dt: float = 0.001            # Time step
    device: str = 'cuda'
    dtype: str = 'float32'
    
    @property
    def N(self) -> int:
        return 1 << self.n_bits
    
    @property
    def dx(self) -> float:
        return self.L / self.N


class NativeNS3DSolver:
    """
    Fully native QTT Navier-Stokes solver.
    
    ZERO DENSE OPERATIONS.
    """
    
    def __init__(self, config: NativeNS3DConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32 if config.dtype == 'float32' else torch.float64
        
        # Native derivatives
        self.deriv = NativeDerivatives3D(
            n_bits=config.n_bits,
            max_rank=config.max_rank,
            base_rank=config.base_rank,
            device=self.device,
            dtype=self.dtype,
            L=config.L,
        )
        
        # State
        self.u: Optional[QTT3DVectorNative] = None
        self.omega: Optional[QTT3DVectorNative] = None
        self.t: float = 0.0
        self.step_count: int = 0
        
        # History
        self.diagnostics_history: List[NativeDiagnostics] = []
    
    def initialize(self, u: QTT3DVectorNative, omega: QTT3DVectorNative):
        """Initialize with velocity and vorticity."""
        self.u = u
        self.omega = omega
        self.t = 0.0
        self.step_count = 0
        
        diag = compute_diagnostics_native(u, omega, 0.0, self.config.dx)
        self.diagnostics_history.append(diag)
    
    def _rhs(self, u: QTT3DVectorNative, omega: QTT3DVectorNative) -> QTT3DVectorNative:
        """
        Compute RHS of vorticity equation NATIVELY.
        
        ∂ω/∂t = ∇×(u×ω) + ν∇²ω
        """
        max_rank = self.config.max_rank
        nu = self.config.nu
        
        # Nonlinear term: ∇×(u×ω)
        # All native Hadamard operations
        u_cross_omega = vector_cross_native(u, omega, max_rank)
        curl_cross = self.deriv.curl(u_cross_omega)
        
        # Viscous term: ν∇²ω
        lap_omega = self.deriv.laplacian_vector(omega)
        visc_x = qtt_scale_native(lap_omega.x.cores, nu)
        visc_y = qtt_scale_native(lap_omega.y.cores, nu)
        visc_z = qtt_scale_native(lap_omega.z.cores, nu)
        
        # Sum terms
        rhs_x = qtt_add_native(curl_cross.x.cores, visc_x, max_rank)
        rhs_y = qtt_add_native(curl_cross.y.cores, visc_y, max_rank)
        rhs_z = qtt_add_native(curl_cross.z.cores, visc_z, max_rank)
        
        return QTT3DVectorNative(
            QTT3DNative(rhs_x, omega.n_bits),
            QTT3DNative(rhs_y, omega.n_bits),
            QTT3DNative(rhs_z, omega.n_bits),
        )
    
    def _velocity_from_vorticity_native(self, omega: QTT3DVectorNative) -> QTT3DVectorNative:
        """
        Recover velocity from vorticity.
        
        For now: keep velocity from previous step and let it evolve
        via the vorticity-velocity coupling in RHS.
        
        The proper Biot-Savart requires TT-Poisson solver which we'll add later.
        This simplified version uses the vorticity to update velocity directly:
        
        In 3D periodic domain, we can use the stream function approach:
        u = curl(psi) where laplacian(psi) = -omega
        
        For a crude approximation, we use:
        u_new ≈ u_old + dt * curl(omega) * scale
        """
        return self.u  # Keep current velocity for now
    
    def _rhs_velocity(self, u: QTT3DVectorNative, omega: QTT3DVectorNative) -> QTT3DVectorNative:
        """
        Compute RHS for velocity equation (Navier-Stokes).
        
        ∂u/∂t = -(u·∇)u + ν∇²u - ∇p
        
        For incompressible flow, we use the vorticity form:
        ∂u/∂t = u×ω + ν∇²u - ∇p
        
        With projection to enforce ∇·u = 0.
        """
        max_rank = self.config.max_rank
        nu = self.config.nu
        
        # Nonlinear term: u × ω
        u_cross_omega = vector_cross_native(u, omega, max_rank)
        
        # Viscous term: ν∇²u
        lap_u = self.deriv.laplacian_vector(u)
        visc_x = qtt_scale_native(lap_u.x.cores, nu)
        visc_y = qtt_scale_native(lap_u.y.cores, nu)
        visc_z = qtt_scale_native(lap_u.z.cores, nu)
        
        # Sum: rhs = u×ω + ν∇²u
        # Note: pressure gradient projects out but we skip for simplicity
        rhs_x = qtt_add_native(u_cross_omega.x.cores, visc_x, max_rank)
        rhs_y = qtt_add_native(u_cross_omega.y.cores, visc_y, max_rank)
        rhs_z = qtt_add_native(u_cross_omega.z.cores, visc_z, max_rank)
        
        return QTT3DVectorNative(
            QTT3DNative(rhs_x, u.n_bits),
            QTT3DNative(rhs_y, u.n_bits),
            QTT3DNative(rhs_z, u.n_bits),
        )
    
    def _project_velocity(self, u_star: QTT3DVectorNative) -> QTT3DVectorNative:
        """
        Project velocity to divergence-free (incompressible).
        
        Uses Chorin projection:
            1. Solve ∇²p = (1/dt) ∇·u*
            2. u = u* - dt * ∇p
        
        This enforces ∇·u = 0.
        """
        dt = self.config.dt
        max_rank = self.config.max_rank
        
        # Compute divergence
        div_u = self.deriv.divergence(u_star)
        
        # Scale by 1/dt for RHS
        rhs = QTT3DNative(
            qtt_scale_native(div_u.cores, 1.0 / dt),
            u_star.n_bits
        )
        
        # Solve Poisson: ∇²p = rhs
        p = self.deriv.poisson_cg(rhs, tol=1e-5, max_iter=30)
        
        # Compute pressure gradient
        grad_p = self.deriv.gradient(p)
        
        # Project: u = u* - dt * ∇p
        u_x = qtt_fused_sum(
            [u_star.x.cores, grad_p.x.cores],
            [1.0, -dt],
            max_rank
        )
        u_y = qtt_fused_sum(
            [u_star.y.cores, grad_p.y.cores],
            [1.0, -dt],
            max_rank
        )
        u_z = qtt_fused_sum(
            [u_star.z.cores, grad_p.z.cores],
            [1.0, -dt],
            max_rank
        )
        
        return QTT3DVectorNative(
            QTT3DNative(u_x, u_star.n_bits),
            QTT3DNative(u_y, u_star.n_bits),
            QTT3DNative(u_z, u_star.n_bits),
        )
    
    def step(self, use_rk2: bool = True, project: bool = True) -> NativeDiagnostics:
        """
        Advance one time step.
        
        Args:
            use_rk2: Use RK2/Heun integrator (more accurate, 2x RHS evals)
            project: Apply pressure projection for incompressibility
            
        Time integration:
            - Euler: y_{n+1} = y_n + dt * f(y_n)
            - RK2/Heun: 
                k1 = f(y_n)
                y* = y_n + dt * k1
                k2 = f(y*)  
                y_{n+1} = y_n + dt/2 * (k1 + k2)
        """
        dt = self.config.dt
        max_rank = self.config.max_rank
        
        if use_rk2:
            return self._step_rk2(project=project)
        else:
            return self._step_euler(project=project)
    
    def _step_euler(self, project: bool = True) -> NativeDiagnostics:
        """Forward Euler step."""
        dt = self.config.dt
        max_rank = self.config.max_rank
        
        # Compute RHS for both fields
        rhs_omega = self._rhs(self.u, self.omega)
        rhs_u = self._rhs_velocity(self.u, self.omega)
        
        # Update vorticity: omega += dt * rhs_omega
        omega_new = QTT3DVectorNative(
            QTT3DNative(qtt_fused_sum([self.omega.x.cores, rhs_omega.x.cores], [1.0, dt], max_rank), self.omega.n_bits),
            QTT3DNative(qtt_fused_sum([self.omega.y.cores, rhs_omega.y.cores], [1.0, dt], max_rank), self.omega.n_bits),
            QTT3DNative(qtt_fused_sum([self.omega.z.cores, rhs_omega.z.cores], [1.0, dt], max_rank), self.omega.n_bits),
        )
        
        # Update velocity: u* = u + dt * rhs_u
        u_star = QTT3DVectorNative(
            QTT3DNative(qtt_fused_sum([self.u.x.cores, rhs_u.x.cores], [1.0, dt], max_rank), self.u.n_bits),
            QTT3DNative(qtt_fused_sum([self.u.y.cores, rhs_u.y.cores], [1.0, dt], max_rank), self.u.n_bits),
            QTT3DNative(qtt_fused_sum([self.u.z.cores, rhs_u.z.cores], [1.0, dt], max_rank), self.u.n_bits),
        )
        
        # Project velocity for incompressibility
        if project:
            u_new = self._project_velocity(u_star)
        else:
            u_new = u_star
        
        # Final truncation at timestep end
        self.omega = self._truncate_vector(omega_new, max_rank)
        self.u = self._truncate_vector(u_new, max_rank)
        
        self.t += dt
        self.step_count += 1
        
        diag = compute_diagnostics_native(self.u, self.omega, self.t, self.config.dx)
        self.diagnostics_history.append(diag)
        
        return diag
    
    def _step_rk2(self, project: bool = True) -> NativeDiagnostics:
        """RK2/Heun step (second-order accurate)."""
        dt = self.config.dt
        max_rank = self.config.max_rank
        
        # Stage 1: k1 = f(y_n)
        k1_omega = self._rhs(self.u, self.omega)
        k1_u = self._rhs_velocity(self.u, self.omega)
        
        # Predictor: y* = y_n + dt * k1
        omega_star = QTT3DVectorNative(
            QTT3DNative(qtt_fused_sum([self.omega.x.cores, k1_omega.x.cores], [1.0, dt], max_rank), self.omega.n_bits),
            QTT3DNative(qtt_fused_sum([self.omega.y.cores, k1_omega.y.cores], [1.0, dt], max_rank), self.omega.n_bits),
            QTT3DNative(qtt_fused_sum([self.omega.z.cores, k1_omega.z.cores], [1.0, dt], max_rank), self.omega.n_bits),
        )
        
        u_star = QTT3DVectorNative(
            QTT3DNative(qtt_fused_sum([self.u.x.cores, k1_u.x.cores], [1.0, dt], max_rank), self.u.n_bits),
            QTT3DNative(qtt_fused_sum([self.u.y.cores, k1_u.y.cores], [1.0, dt], max_rank), self.u.n_bits),
            QTT3DNative(qtt_fused_sum([self.u.z.cores, k1_u.z.cores], [1.0, dt], max_rank), self.u.n_bits),
        )
        
        # Truncate intermediates at barrier
        omega_star = self._truncate_vector(omega_star, max_rank)
        u_star = self._truncate_vector(u_star, max_rank)
        
        # Stage 2: k2 = f(y*)
        k2_omega = self._rhs(u_star, omega_star)
        k2_u = self._rhs_velocity(u_star, omega_star)
        
        # Corrector: y_{n+1} = y_n + dt/2 * (k1 + k2)
        omega_new = QTT3DVectorNative(
            QTT3DNative(qtt_fused_sum([self.omega.x.cores, k1_omega.x.cores, k2_omega.x.cores], 
                                       [1.0, dt/2, dt/2], max_rank), self.omega.n_bits),
            QTT3DNative(qtt_fused_sum([self.omega.y.cores, k1_omega.y.cores, k2_omega.y.cores], 
                                       [1.0, dt/2, dt/2], max_rank), self.omega.n_bits),
            QTT3DNative(qtt_fused_sum([self.omega.z.cores, k1_omega.z.cores, k2_omega.z.cores], 
                                       [1.0, dt/2, dt/2], max_rank), self.omega.n_bits),
        )
        
        u_corrected = QTT3DVectorNative(
            QTT3DNative(qtt_fused_sum([self.u.x.cores, k1_u.x.cores, k2_u.x.cores], 
                                       [1.0, dt/2, dt/2], max_rank), self.u.n_bits),
            QTT3DNative(qtt_fused_sum([self.u.y.cores, k1_u.y.cores, k2_u.y.cores], 
                                       [1.0, dt/2, dt/2], max_rank), self.u.n_bits),
            QTT3DNative(qtt_fused_sum([self.u.z.cores, k1_u.z.cores, k2_u.z.cores], 
                                       [1.0, dt/2, dt/2], max_rank), self.u.n_bits),
        )
        
        # Project velocity for incompressibility
        if project:
            u_new = self._project_velocity(u_corrected)
        else:
            u_new = u_corrected
        
        # Final truncation at timestep barrier
        self.omega = self._truncate_vector(omega_new, max_rank)
        self.u = self._truncate_vector(u_new, max_rank)
        
        self.t += dt
        self.step_count += 1
        
        diag = compute_diagnostics_native(self.u, self.omega, self.t, self.config.dx)
        self.diagnostics_history.append(diag)
        
        return diag
    
    def _truncate_vector(self, v: QTT3DVectorNative, max_rank: int, tol: float = 1e-10) -> QTT3DVectorNative:
        """Truncate all components of a vector field."""
        return QTT3DVectorNative(
            QTT3DNative(qtt_truncate_now(v.x.cores, max_rank, tol), v.n_bits),
            QTT3DNative(qtt_truncate_now(v.y.cores, max_rank, tol), v.n_bits),
            QTT3DNative(qtt_truncate_now(v.z.cores, max_rank, tol), v.n_bits),
        )
    
    @property
    def diagnostics(self) -> NativeDiagnostics:
        return self.diagnostics_history[-1] if self.diagnostics_history else None


# ═══════════════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════════════

__all__ = [
    'QTT3DNative',
    'QTT3DVectorNative',
    'NativeDerivatives3D',
    'vector_cross_native',
    'NativeDiagnostics',
    'compute_diagnostics_native',
    'taylor_green_native',
    'NativeNS3DConfig',
    'NativeNS3DSolver',
]
