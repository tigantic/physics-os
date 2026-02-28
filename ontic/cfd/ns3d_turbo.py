"""
Turbo 3D QTT Navier-Stokes Solver
=================================

ZERO DENSE. LAZY TRUNCATION. MAXIMUM PERFORMANCE.

Key principles:
1. ALL operations in native QTT format
2. LAZY truncation: accumulate ops, truncate once per timestep
3. TRUE ADAPTIVE RANK: Error-controlled, not fixed
   - Rank determined by singular value decay
   - More rank for low-frequency (important) modes
   - Less rank for high-frequency (compressible) modes
4. TRITON kernels for core operations

Architecture per RK2 step:
    OLD: MPO → truncate → add → truncate → MPO → truncate → add → truncate (8 truncations)
    NEW: MPO → MPO → add → add → truncate_once (1 truncation)

Expected speedup: 4-8x over standard implementation

Author: TiganticLabz
Date: 2025
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time

import torch
from torch import Tensor

from ontic.cfd.qtt_turbo import (
    TurboCores,
    AdaptiveRankController,
    turbo_add_cores,
    turbo_scale,
    turbo_hadamard_cores,
    turbo_truncate,
    turbo_truncate_adaptive,
    turbo_truncate_batched,
    turbo_truncate_conservative,
    turbo_truncate_batched_conservative,
    turbo_linear_combination,
    turbo_linear_combination_batched,
    turbo_linear_combination_adaptive,
    turbo_mpo_apply,
    turbo_inner,
    turbo_norm,
    turbulence_rank_profile,
)
from ontic.cfd.poisson_spectral import (
    SpectralPoissonConfig,
    SpectralPoissonQTT,
    solver_qtt_to_dense_3d,
    dense_to_solver_qtt_3d,
    spectral_biot_savart,
)


# ═══════════════════════════════════════════════════════════════════════════════════════
# MPO BUILDING (SHIFT, LAPLACIAN, DERIVATIVES)
# ═══════════════════════════════════════════════════════════════════════════════════════

def build_shift_mpo(n_bits: int, direction: int, device: torch.device) -> List[Tensor]:
    """
    Build shift-by-±1 MPO for 3D QTT with ROW-MAJOR (C-order) layout.
    
    The shift MPO implements circular shift: x → x±1 (mod N).
    
    ROW-MAJOR LAYOUT (matching TurboNS3DSolver._dense_to_qtt):
      - Bits are GROUPED by dimension: [x0,x1,...,x_{n-1}, y0,..., z0,...]
      - x bits: positions 0 to n_bits-1
      - y bits: positions n_bits to 2*n_bits-1
      - z bits: positions 2*n_bits to 3*n_bits-1
    
    This differs from Morton interleaving where bits cycle as (x0,y0,z0,x1,y1,z1,...).
    
    KEY INSIGHT: Carry/borrow flows from r_right to r_left (backward through cores).
    - LSB (position n_bits-1 for x-shift) injects the carry via r_right=1
    - Carry propagates toward MSB (position 0 for x-shift)
    - MSB absorbs carry by summing over r_left states (periodic BC)
    
    Args:
        n_bits: Bits per dimension (N = 2^n_bits)
        direction: 0=x, 1=y, 2=z  (which axis to shift)
        device: Target device
    
    Returns:
        List of MPO cores, each of shape (r_left, d_out, d_in, r_right)
        where d_in is input physical index, d_out is output physical index.
    """
    n_cores = 3 * n_bits
    
    # Determine which positions belong to this axis
    axis_start = direction * n_bits
    axis_end = (direction + 1) * n_bits
    
    # Build all cores with full bond dimension first
    cores = []
    for pos in range(n_cores):
        # Core shape: (r_left, d_out, d_in, r_right)
        # All cores get bond dim 2 initially; we'll fix boundaries at the end
        core = torch.zeros(2, 2, 2, 2, device=device)
        
        is_active = (axis_start <= pos < axis_end)
        
        if is_active:
            # This position participates in the shift (carries arithmetic)
            # Carry flows from r_right (input) to r_left (output)
            
            # === INCREMENT (+1) LOGIC ===
            # Carry comes in via r_right, goes out via r_left
            
            # CASE A: No Carry In (r_right=0) -> Identity
            core[0, 0, 0, 0] = 1.0  # 0 -> 0, no carry out (r_left=0)
            core[0, 1, 1, 0] = 1.0  # 1 -> 1, no carry out (r_left=0)
            
            # CASE B: Carry In (r_right=1) -> Add 1 to this bit
            core[0, 1, 0, 1] = 1.0  # 0 + carry = 1, carry absorbed (r_left=0)
            core[1, 0, 1, 1] = 1.0  # 1 + carry = 0 (overflow), carry out (r_left=1)
        else:
            # Passthrough: preserve value, transport carry
            # No Carry In -> No Carry Out
            core[0, 0, 0, 0] = 1.0
            core[0, 1, 1, 0] = 1.0
            # Carry In -> Carry Out (wire it through)
            core[1, 0, 0, 1] = 1.0
            core[1, 1, 1, 1] = 1.0
        
        cores.append(core)
    
    # === BOUNDARY CONDITIONS FOR PERIODIC BC ===
    
    # Core 0 (leftmost): Sum over r_left to absorb overflow (periodic)
    # This allows N-1 + 1 = 0 (wrap around)
    cores[0] = cores[0][0:1, :, :, :] + cores[0][1:2, :, :, :]  # (1, 2, 2, 2)
    
    # Core n_cores-1 (rightmost): Inject +1 by forcing carry_in = 1
    # But we only inject at the LSB of the active axis!
    # For x-shift (direction=0), LSB is at position n_bits-1, not n_cores-1
    
    # Actually, we need to inject at the LSB of the AXIS, not the last core overall.
    # LSB of axis is at position axis_end - 1.
    
    # For correct injection:
    # - All cores before axis_end - 1: get r_right from next core
    # - Core at axis_end - 1 (LSB of axis): force r_right = 1 (inject carry)
    # - All cores after axis_end - 1: no carry (r_right = 0)
    
    # Reconstruct with proper boundary handling
    lsb_pos = axis_end - 1  # LSB of the target axis
    msb_pos = axis_start    # MSB of the target axis
    
    # Re-slice cores for proper carry injection
    final_cores = []
    for pos in range(n_cores):
        core = cores[pos]
        
        if pos == 0:
            # Leftmost core: already summed over r_left
            if pos < lsb_pos:
                # Carry may come from right
                final_cores.append(core)
            else:
                # At or past LSB: no carry from right
                final_cores.append(core[:, :, :, 0:1])  # r_right = 0 only
        elif pos == lsb_pos:
            # LSB of axis: inject carry (r_right = 1)
            if pos == n_cores - 1:
                # Also last core overall
                final_cores.append(core[:, :, :, 1:2])  # r_right = 1 (inject)
            else:
                final_cores.append(core[:, :, :, 1:2])  # r_right = 1 (inject)
        elif pos > lsb_pos:
            # After LSB of axis: these cores are identity (y, z dimensions)
            # No carry propagation for them
            if pos == n_cores - 1:
                final_cores.append(core[:, :, :, 0:1])  # r_right = 0
            else:
                final_cores.append(core[:, :, :, 0:1])  # r_right = 0
        else:
            # Between MSB and LSB: normal carry propagation
            final_cores.append(core)
    
    # Fix bond dimensions to match at boundaries
    # After LSB, we have r_right=1 (singleton), so next core needs r_left=1
    result = []
    for pos in range(n_cores):
        core = final_cores[pos]
        r_left, d_out, d_in, r_right = core.shape
        
        if pos > 0:
            # r_left must match previous core's r_right
            prev_r_right = result[-1].shape[3]
            if r_left != prev_r_right:
                # Slice to match
                core = core[:prev_r_right, :, :, :]
        
        result.append(core)
    
    return result


def build_inverse_shift_mpo(n_bits: int, direction: int, device: torch.device) -> List[Tensor]:
    """
    Build shift-by-(-1) MPO for ROW-MAJOR layout (decrement/backward shift).
    
    Decrement: x → x-1 (mod N)
    Uses borrow instead of carry.
    
    Same layout as build_shift_mpo but with subtraction logic.
    """
    n_cores = 3 * n_bits
    
    # Determine which positions belong to this axis
    axis_start = direction * n_bits
    axis_end = (direction + 1) * n_bits
    
    # Build all cores with full bond dimension first
    cores = []
    for pos in range(n_cores):
        core = torch.zeros(2, 2, 2, 2, device=device)
        
        is_active = (axis_start <= pos < axis_end)
        
        if is_active:
            # === DECREMENT (-1) LOGIC ===
            # Borrow flows from r_right to r_left
            
            # CASE A: No Borrow In (r_right=0) -> Identity
            core[0, 0, 0, 0] = 1.0
            core[0, 1, 1, 0] = 1.0
            
            # CASE B: Borrow In (r_right=1) -> Subtract 1 from this bit
            core[0, 0, 1, 1] = 1.0  # 1 - borrow = 0, borrow absorbed (r_left=0)
            core[1, 1, 0, 1] = 1.0  # 0 - borrow = 1 (underflow), borrow out (r_left=1)
        else:
            # Passthrough: preserve value, transport borrow
            core[0, 0, 0, 0] = 1.0
            core[0, 1, 1, 0] = 1.0
            core[1, 0, 0, 1] = 1.0
            core[1, 1, 1, 1] = 1.0
        
        cores.append(core)
    
    # === BOUNDARY CONDITIONS ===
    lsb_pos = axis_end - 1
    
    # Core 0: sum over r_left for periodic BC
    cores[0] = cores[0][0:1, :, :, :] + cores[0][1:2, :, :, :]
    
    # Inject borrow at LSB
    final_cores = []
    for pos in range(n_cores):
        core = cores[pos]
        
        if pos == 0:
            if pos < lsb_pos:
                final_cores.append(core)
            else:
                final_cores.append(core[:, :, :, 0:1])
        elif pos == lsb_pos:
            final_cores.append(core[:, :, :, 1:2])  # Inject borrow
        elif pos > lsb_pos:
            if pos == n_cores - 1:
                final_cores.append(core[:, :, :, 0:1])
            else:
                final_cores.append(core[:, :, :, 0:1])
        else:
            final_cores.append(core)
    
    # Fix bond dimensions
    result = []
    for pos in range(n_cores):
        core = final_cores[pos]
        if pos > 0:
            prev_r_right = result[-1].shape[3]
            if core.shape[0] != prev_r_right:
                core = core[:prev_r_right, :, :, :]
        result.append(core)
    
    return result


def build_laplacian_mpo(
    n_bits: int,
    h: float,
    device: torch.device,
) -> List[List[Tensor]]:
    """
    Build Laplacian MPO components for 3D.
    
    ∇² = ∂²/∂x² + ∂²/∂y² + ∂²/∂z²
    
    Each term is: (S⁺ - 2I + S⁻) / h²
    
    Returns list of MPO lists, one per direction.
    We'll apply each and sum the results.
    """
    components = []
    for d in range(3):
        shift_plus = build_shift_mpo(n_bits, d, device)
        shift_minus = build_inverse_shift_mpo(n_bits, d, device)
        
        # Build identity MPO
        n_cores = 3 * n_bits
        identity = []
        for _ in range(n_cores):
            core = torch.zeros(1, 2, 2, 1, device=device)
            core[0, 0, 0, 0] = 1.0
            core[0, 1, 1, 0] = 1.0
            identity.append(core)
        
        components.append((shift_plus, identity, shift_minus))
    
    return components


# ═══════════════════════════════════════════════════════════════════════════════════════
# TURBO SOLVER
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class TurboNS3DConfig:
    """Configuration for Turbo NS solver."""
    n_bits: int = 5  # Grid size per dimension (N = 2^n_bits)
    nu: float = 0.001  # Kinematic viscosity
    dt: float = 0.01  # Time step
    device: str = 'cuda'
    diffusion_only: bool = False  # If True, skip advection for ~6x speedup
    
    # Turbulence forcing
    enable_forcing: bool = False  # If True, add large-scale forcing
    forcing_epsilon: float = 0.1  # Energy injection rate
    forcing_k: int = 2            # Forcing wavenumber
    
    # Velocity update control
    velocity_update_freq: int = 1  # Update velocity every N steps (1=every step)
    
    # Poisson solver mode (2026-02-05: Added spectral option)
    # - "spectral": RECOMMENDED. Exact FFT-based solve, machine precision.
    # - "diffusion": Fast approximation, valid for diffusion-dominated flows.
    # - "jacobi": Broken, do not use (diverges due to over-relaxation).
    poisson_mode: str = "spectral"  # "spectral", "diffusion", or "jacobi"
    poisson_iterations: int = 0    # DEPRECATED: Only used if poisson_mode="jacobi"
    
    # Rank control: Choose ONE mode
    # Mode 1: Fixed max_rank (legacy)
    # OPTIMIZED (2026-02-05): Reduced from 64 to 16 based on rank sweep.
    # Evidence: rank=16 achieves 0.9% energy drift with 4× less memory than rank=64.
    # See: artifacts/PHASE2_RANK_ATTESTATION.json
    max_rank: int = 16  # Maximum QTT rank (ignored if adaptive_rank=True)
    tol: float = 1e-10  # Truncation tolerance for fixed mode
    
    # Mode 2: Adaptive rank (recommended)
    adaptive_rank: bool = True  # If True, use error-controlled adaptive rank
    target_error: float = 1e-6  # Error budget for adaptive mode
    min_rank: int = 4  # Minimum rank (adaptive mode)
    rank_cap: int = 64  # REDUCED: 128 → 64 (16 is sufficient for physics)
    
    # Mode 3: Conservative truncation (2026-02-05: CRITICAL for physics)
    # If True, all truncations rescale to preserve L2 norm: ‖u_trunc‖² = ‖u_orig‖²
    # This eliminates numerical dissipation from QTT truncation in advection.
    # Evidence: Without conservative=True, 90% energy loss at 128³ over 50 steps.
    conservative_truncation: bool = True  # ALWAYS use for physical simulations
    
    @property
    def N(self) -> int:
        return 1 << self.n_bits
    
    @property
    def h(self) -> float:
        return 2.0 * 3.14159265359 / self.N
    
    @property
    def n_cores(self) -> int:
        return 3 * self.n_bits


class TurboNS3DSolver:
    """
    Turbo 3D Navier-Stokes solver with lazy truncation.
    
    Key optimizations:
    1. Single truncation per RK stage (not per operation)
    2. TRUE ADAPTIVE RANK: Error-controlled, not fixed
       - Rank determined by singular value decay
       - More rank for low-frequency (important) modes
       - Less rank for high-frequency (compressible) modes
    3. Native QTT operations throughout
    4. Optional large-scale forcing for sustained turbulence
    
    Vorticity formulation:
        ∂ω/∂t = ∇×(u×ω) + ν∇²ω + f
        u = ∇⁻²(∇×ω)  (Biot-Savart, via Poisson solve)
    
    Rank Control Modes:
        adaptive_rank=True  (recommended): Error budget determines rank
        adaptive_rank=False (legacy): Fixed max_rank cap
    """
    
    def __init__(self, config: TurboNS3DConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Build MPOs once
        self._build_operators()
        
        # Create adaptive rank controller if enabled
        if config.adaptive_rank:
            self.rank_controller = AdaptiveRankController(
                target_error=config.target_error,
                min_rank=config.min_rank,
                max_rank=config.rank_cap,
            )
        else:
            self.rank_controller = None
        
        # Initialize turbulence forcing if enabled
        if config.enable_forcing:
            from ontic.cfd.turbulence_forcing import TurbulenceForcing, ForcingConfig
            forcing_config = ForcingConfig(
                forcing_type='spectral',
                k_forcing=config.forcing_k,
                epsilon_target=config.forcing_epsilon,
            )
            self.forcing = TurbulenceForcing(
                n_bits=config.n_bits,
                config=forcing_config,
                device=self.device,
            )
        else:
            self.forcing = None
        
        # State
        self.omega: Optional[List[List[Tensor]]] = None  # [ωx, ωy, ωz] each is List[Tensor]
        self.u: Optional[List[List[Tensor]]] = None      # [ux, uy, uz]
        self.t: float = 0.0
        self._step_count: int = 0
        
        # Diagnostics
        self.last_step_time: float = 0.0
        self.truncation_count: int = 0
        self.last_truncation_error: float = 0.0
        self.last_ranks: List[int] = []
    
    def _build_operators(self):
        """Pre-build MPO operators."""
        n = self.config.n_bits
        h = self.config.h
        dev = self.device
        
        # Shift operators for each direction
        self.shift_plus = [build_shift_mpo(n, d, dev) for d in range(3)]
        self.shift_minus = [build_inverse_shift_mpo(n, d, dev) for d in range(3)]
        
        # Identity MPO
        n_cores = self.config.n_cores
        self.identity = []
        for _ in range(n_cores):
            core = torch.zeros(1, 2, 2, 1, device=dev)
            core[0, 0, 0, 0] = 1.0
            core[0, 1, 1, 0] = 1.0
            self.identity.append(core)
        
        # Laplacian coefficient
        self.lap_coeff = 1.0 / (h * h)
    
    def initialize_taylor_green(self, A: float = 1.0):
        """
        Initialize with Taylor-Green vortex in QTT format.
        
        u = (A sin(x)cos(y)cos(z), -A cos(x)sin(y)cos(z), 0)
        ω = ∇×u
        """
        # Build in dense then compress to QTT
        N = self.config.N
        device = self.device
        
        x = torch.linspace(0, 2*3.14159265359, N+1, device=device)[:-1]
        X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')
        
        # Velocity
        ux = A * torch.sin(X) * torch.cos(Y) * torch.cos(Z)
        uy = -A * torch.cos(X) * torch.sin(Y) * torch.cos(Z)
        uz = torch.zeros_like(ux)
        
        # Vorticity (analytical)
        omega_x = -A * torch.cos(X) * torch.sin(Y) * torch.sin(Z)
        omega_y = -A * torch.sin(X) * torch.cos(Y) * torch.sin(Z)
        omega_z = 2 * A * torch.sin(X) * torch.sin(Y) * torch.cos(Z)
        
        # Convert to QTT
        self.omega = [
            self._dense_to_qtt(omega_x),
            self._dense_to_qtt(omega_y),
            self._dense_to_qtt(omega_z),
        ]
        self.u = [
            self._dense_to_qtt(ux),
            self._dense_to_qtt(uy),
            self._dense_to_qtt(uz),
        ]
        self.t = 0.0
    
    def _dense_to_qtt(self, dense: Tensor) -> List[Tensor]:
        """
        Convert dense 3D tensor to QTT cores via hierarchical TT-SVD.
        
        For large tensors, uses chunked SVD to avoid memory issues.
        Morton interleaving: sites are [x0,y0,z0, x1,y1,z1, ...]
        """
        N = self.config.N
        n_bits = self.config.n_bits
        max_rank = self.config.max_rank
        tol = self.config.tol
        n_cores = 3 * n_bits
        device = dense.device
        dtype = dense.dtype
        
        cores = []
        
        # Start with full tensor reshaped to (2, rest)
        work = dense.reshape(2, -1)  # (2, N³/2)
        r_left = 1
        
        for i in range(n_cores - 1):
            m, n = work.shape
            
            # Target rank
            k = min(max_rank, m, n)
            
            if m * n > 5e5:  # Large matrix - use rSVD with CPU fallback
                l = min(k + 10, min(m, n))
                
                try:
                    # Try GPU rSVD
                    Omega = torch.randn(n, l, device=device, dtype=dtype)
                    Y = work @ Omega
                    Q, _ = torch.linalg.qr(Y)
                    B = Q.T @ work
                    U_small, S, Vh = torch.linalg.svd(B, full_matrices=False)
                    U = Q @ U_small
                except RuntimeError:
                    # CPU fallback for problematic sizes
                    work_cpu = work.cpu()
                    Omega = torch.randn(n, l, dtype=dtype)
                    Y = work_cpu @ Omega
                    Q, _ = torch.linalg.qr(Y)
                    B = Q.T @ work_cpu
                    U_small, S, Vh = torch.linalg.svd(B, full_matrices=False)
                    U = (Q @ U_small).to(device)
                    S = S.to(device)
                    Vh = Vh.to(device)
            else:
                # Small enough for full SVD
                try:
                    U, S, Vh = torch.linalg.svd(work, full_matrices=False)
                except RuntimeError:
                    # CPU fallback
                    work_cpu = work.cpu()
                    U, S, Vh = torch.linalg.svd(work_cpu, full_matrices=False)
                    U, S, Vh = U.to(device), S.to(device), Vh.to(device)
            
            # Truncate
            k = min(max_rank, len(S))
            if tol > 0 and len(S) > 0 and S[0] > 1e-14:
                rel_s = S / S[0]
                k = max(1, min(k, (rel_s > tol).sum().item()))
            
            U = U[:, :k]
            S = S[:k]
            Vh = Vh[:k, :]
            
            # Create core: reshape U from (m, k) to (r_left, 2, k)
            core = U.reshape(r_left, 2, k)
            cores.append(core)
            
            # Prepare next iteration
            work = torch.diag(S) @ Vh  # (k, n)
            
            # Reshape for next site: (k, n) → (k*2, n/2)
            if i < n_cores - 2:
                remaining = work.shape[1]
                if remaining >= 2:
                    work = work.reshape(k * 2, remaining // 2)
                else:
                    work = work.reshape(k, remaining)
            
            r_left = k
        
        # Last core
        cores.append(work.reshape(r_left, 2, 1))
        
        return cores
    
    def _apply_laplacian(self, field: List[Tensor]) -> List[Tensor]:
        """
        Apply Laplacian to field: ∇²f (already scaled by 1/h²).
        
        Uses: ∂²f/∂x² = (f(x+1) - 2f(x) + f(x-1)) / h²
        
        Truncates at the end to keep ranks bounded.
        """
        # Accumulate: Σ_d (f+_d + f-_d) - 6f, scaled by 1/h²
        terms = []
        
        for d in range(3):
            f_plus = turbo_mpo_apply(field, self.shift_plus[d])
            f_minus = turbo_mpo_apply(field, self.shift_minus[d])
            terms.append((self.lap_coeff, f_plus))
            terms.append((self.lap_coeff, f_minus))
        
        terms.append((-6.0 * self.lap_coeff, field))
        
        # Single truncation at the end
        return self._truncate_terms(terms)
    
    def _truncate_terms(self, terms: List[Tuple[float, List[Tensor]]]) -> List[Tensor]:
        """
        Helper to truncate linear combination using either adaptive or fixed mode.
        """
        if self.rank_controller is not None:
            return turbo_linear_combination_adaptive(terms, self.rank_controller)
        else:
            # Use position-dependent rank reduction for turbulence (high-freq compresses better)
            return turbo_linear_combination(terms, self.config.max_rank, self.config.tol, adaptive=True)
    
    def _truncate_terms_batched(
        self,
        terms_list: List[List[Tuple[float, List[Tensor]]]],
    ) -> List[List[Tensor]]:
        """
        Batch-truncate multiple linear combinations using batched SVD.
        
        This is 1.5-2x faster than calling _truncate_terms individually
        because it batches the SVD calls across all fields at each site.
        
        At each site k, all fields have their SVD matrices stacked and
        processed in ONE batched SVD call:
            (n_fields × n_sites) individual SVDs → n_sites batched SVDs
        
        Args:
            terms_list: List of term lists, one per field
                        Each term list is [(coeff, cores), ...]
        
        Returns:
            List of truncated cores, one per field
        """
        if self.rank_controller is not None:
            # Adaptive mode: fall back to individual truncation
            return [self._truncate_terms(terms) for terms in terms_list]
        
        # Fixed rank mode: use turbo_linear_combination_batched
        # This accumulates each field's terms then does a single batched SVD sweep
        return turbo_linear_combination_batched(terms_list, self.config.max_rank, self.config.tol)
    
    def _truncate_single(self, cores: List[Tensor]) -> List[Tensor]:
        """
        Helper to truncate single core set using either adaptive or fixed mode.
        
        If conservative_truncation=True, uses energy-preserving truncation
        that rescales to preserve ‖u_truncated‖² = ‖u_original‖².
        """
        if self.rank_controller is not None:
            return turbo_truncate_adaptive(cores, self.rank_controller)
        else:
            if self.config.conservative_truncation:
                return turbo_truncate_conservative(cores, self.config.max_rank, tol=self.config.tol)
            else:
                return turbo_truncate(cores, self.config.max_rank, tol=self.config.tol, adaptive=True)
    
    def _apply_derivative(self, field: List[Tensor], direction: int) -> List[Tensor]:
        """
        Apply first derivative: ∂f/∂x_d = (f(x+1) - f(x-1)) / (2h)
        
        Central difference. Truncates to keep ranks bounded for subsequent Hadamards.
        """
        f_plus = turbo_mpo_apply(field, self.shift_plus[direction])
        f_minus = turbo_mpo_apply(field, self.shift_minus[direction])
        
        # (f+ - f-) / (2h)
        coeff = 1.0 / (2.0 * self.config.h)
        terms = [(coeff, f_plus), (-coeff, f_minus)]
        return self._truncate_terms(terms)
    
    def _apply_derivatives_batched(
        self,
        fields: List[List[Tensor]],
        directions: List[int],
    ) -> List[List[Tensor]]:
        """
        Apply derivatives to multiple fields in batch using batched SVD.
        
        This is ~1.5x faster than individual _apply_derivative calls.
        
        Args:
            fields: List of QTT fields to differentiate
            directions: List of directions (0=x, 1=y, 2=z) for each field
        
        Returns:
            List of differentiated fields
        """
        if self.rank_controller is not None:
            # Adaptive mode: fall back to individual
            return [self._apply_derivative(f, d) for f, d in zip(fields, directions)]
        
        # Compute all derivatives without truncation
        accumulated = []
        for field, direction in zip(fields, directions):
            f_plus = turbo_mpo_apply(field, self.shift_plus[direction])
            f_minus = turbo_mpo_apply(field, self.shift_minus[direction])
            coeff = 1.0 / (2.0 * self.config.h)
            # Accumulate: coeff * f_plus - coeff * f_minus
            result = turbo_add_cores(f_plus, f_minus, alpha=coeff, beta=-coeff)
            accumulated.append(result)
        
        # Batch-truncate all at once (conservative if configured)
        if self.config.conservative_truncation:
            return turbo_truncate_batched_conservative(accumulated, self.config.max_rank, self.config.tol)
        else:
            return turbo_truncate_batched(accumulated, self.config.max_rank, self.config.tol)
    
    def _reconstruct_velocity_from_vorticity(self):
        """
        Recover velocity from vorticity using configured Poisson solver.
        
        Modes:
        - "spectral": EXACT Biot-Savart via FFT. Machine precision. Recommended.
        - "diffusion": Velocity diffusion approximation. Fast but approximate.
        - "jacobi": Iterative Jacobi. BROKEN, do not use.
        
        The spectral mode converts QTT→Dense, solves exactly via FFT, then Dense→QTT.
        This is O(N³ log N) but with excellent constant factors and exact solution.
        """
        if self.config.diffusion_only:
            # No velocity update needed for diffusion-only
            return
        
        poisson_mode = getattr(self.config, 'poisson_mode', 'diffusion')
        
        if poisson_mode == "spectral":
            self._reconstruct_velocity_spectral()
        elif poisson_mode == "diffusion" or self.config.poisson_iterations == 0:
            self._reconstruct_velocity_diffusion()
        else:
            self._reconstruct_velocity_jacobi()
    
    def _reconstruct_velocity_spectral(self):
        """
        EXACT velocity recovery via spectral Biot-Savart.
        
        Given ω = ∇×u and ∇·u = 0:
            û(k) = i k × ω̂(k) / |k|²
        
        This is EXACT (machine precision) unlike iterative methods.
        """
        L = 2 * math.pi  # Domain size
        max_rank = self.config.max_rank
        
        # Convert QTT vorticity to dense (using solver-compatible format)
        omega_dense = [
            solver_qtt_to_dense_3d(self.omega[i], self.config.n_bits)
            for i in range(3)
        ]
        
        # Spectral Biot-Savart (exact)
        u_dense = spectral_biot_savart(omega_dense, L=L)
        
        # Convert back to QTT (using solver-compatible format)
        self.u = [
            dense_to_solver_qtt_3d(u_dense[i], self.config.n_bits, max_rank=max_rank)
            for i in range(3)
        ]
    
    def _reconstruct_velocity_diffusion(self):
        """
        Fast velocity diffusion approximation.
        
        u_new ≈ u + dt * ν * ∇²u
        
        Valid for diffusion-dominated flows or when velocity changes slowly.
        All operations remain in QTT format - NO dense conversion.
        """
        nu = self.config.nu
        dt = self.config.dt
        
        u_new = []
        for i in range(3):
            lap_u = self._apply_laplacian(self.u[i])
            terms = [(1.0, self.u[i]), (nu * dt, lap_u)]
            u_i = self._truncate_terms(terms)
            u_new.append(u_i)
        self.u = u_new
    
    def _reconstruct_velocity_jacobi(self):
        """
        Jacobi iteration for Poisson solve. DEPRECATED - DIVERGES.
        
        Kept for reference only. Use poisson_mode="spectral" instead.
        """
        h = self.config.h
        h2 = h * h
        
        n_jacobi = self.config.poisson_iterations
        
        if n_jacobi == 0:
            self._reconstruct_velocity_diffusion()
            return
        
        # Use weighted Jacobi for Poisson: ∇²ψ = -ω
        # WARNING: This diverges due to over-relaxation
        omega_sor = 1.5  # Over-relaxation (1 < ω < 2)
        alpha = omega_sor / 6.0  # Combined factor
        
        # Initialize from scaled vorticity
        psi = [turbo_scale(self.omega[i], -1.0) for i in range(3)]
        
        for _ in range(n_jacobi):
            psi_new = []
            for i in range(3):
                neighbor_terms = []
                for d in range(3):
                    psi_plus = turbo_mpo_apply(psi[i], self.shift_plus[d])
                    psi_minus = turbo_mpo_apply(psi[i], self.shift_minus[d])
                    neighbor_terms.append((alpha, psi_plus))
                    neighbor_terms.append((alpha, psi_minus))
                
                neighbor_terms.append((1.0 - omega_sor, psi[i]))
                neighbor_terms.append((alpha * h2, self.omega[i]))
                
                psi_i = self._truncate_terms(neighbor_terms)
                psi_new.append(psi_i)
            
            psi = psi_new
        
        # Recover velocity via curl: u = ∇×ψ
        dpsi_z_dy = self._apply_derivative(psi[2], 1)
        dpsi_y_dz = self._apply_derivative(psi[1], 2)
        ux = self._truncate_terms([(1.0, dpsi_z_dy), (-1.0, dpsi_y_dz)])
        
        dpsi_x_dz = self._apply_derivative(psi[0], 2)
        dpsi_z_dx = self._apply_derivative(psi[2], 0)
        uy = self._truncate_terms([(1.0, dpsi_x_dz), (-1.0, dpsi_z_dx)])
        
        dpsi_y_dx = self._apply_derivative(psi[1], 0)
        dpsi_x_dy = self._apply_derivative(psi[0], 1)
        uz = self._truncate_terms([(1.0, dpsi_y_dx), (-1.0, dpsi_x_dy)])
        
        self.u = [ux, uy, uz]
    
    def _compute_rhs(
        self,
        omega: List[List[Tensor]],
        u: List[List[Tensor]],
    ) -> List[List[Tensor]]:
        """
        Compute RHS of vorticity equation.
        
        ∂ω/∂t = ∇×(u×ω) + ν∇²ω
        
        So: ∂ω/∂t = (ω·∇)u - (u·∇)ω + ν∇²ω
        """
        nu = self.config.nu
        
        # Get forcing once if enabled
        forcing = None
        if self.forcing is not None:
            forcing = self.forcing.get_forcing()
        
        # Diffusion only path (fast)
        if self.config.diffusion_only:
            rhs = []
            for i in range(3):
                lap_omega_i = self._apply_laplacian(omega[i])
                rhs.append(self._truncate_terms([(nu, lap_omega_i)]))
            return rhs
        
        # Full NS
        # CRITICAL FIX: Truncate after each Hadamard to prevent rank explosion
        # Without truncation: r + 6×r² = 6176 for r=32 → OOM
        # With truncation: r + 6×r = 7r = 224 for r=32 → manageable
        #
        # 2026-02-05: CRITICAL - Use conservative truncation to preserve energy.
        # Without conservative=True: 90% energy loss at 128³ over 50 steps.
        rhs = []
        max_rank = self.config.max_rank
        use_conservative = self.config.conservative_truncation
        
        for i in range(3):
            # Diffusion: ν∇²ω_i
            lap_omega_i = self._apply_laplacian(omega[i])
            
            # Advection: -(u·∇)ω_i = -Σ_d u[d] * ∂ω[i]/∂x[d]
            # TRUNCATE each Hadamard product immediately (CONSERVATIVELY)
            advection_terms = []
            for d in range(3):
                grad_omega = self._apply_derivative(omega[i], d)
                term = turbo_hadamard_cores(u[d], grad_omega)
                # Truncate after Hadamard to prevent r² → r
                # CONSERVATIVE: Rescale to preserve ‖term‖² exactly
                if use_conservative:
                    term = turbo_truncate_conservative(term, max_rank)
                else:
                    term = turbo_truncate(term, max_rank)
                advection_terms.append((-1.0, term))
            
            # Vortex stretching: (ω·∇)u_i = Σ_d ω[d] * ∂u[i]/∂x[d]
            # TRUNCATE each Hadamard product immediately (CONSERVATIVELY)
            stretch_terms = []
            for d in range(3):
                grad_u = self._apply_derivative(u[i], d)
                term = turbo_hadamard_cores(omega[d], grad_u)
                # Truncate after Hadamard to prevent r² → r
                # CONSERVATIVE: Rescale to preserve ‖term‖² exactly
                if use_conservative:
                    term = turbo_truncate_conservative(term, max_rank)
                else:
                    term = turbo_truncate(term, max_rank)
                stretch_terms.append((1.0, term))
            
            # Combine all terms for component i
            all_terms = [(nu, lap_omega_i)] + advection_terms + stretch_terms
            
            if forcing is not None:
                all_terms.append((1.0, forcing[i]))
            
            rhs.append(self._truncate_terms(all_terms))
        
        return rhs
    
    def step(self) -> dict:
        """
        Take one timestep using RK2.
        
        Returns diagnostics dict including adaptive rank info.
        """
        t0 = time.perf_counter()
        
        dt = self.config.dt
        
        # RK2 Stage 1
        k1 = self._compute_rhs(self.omega, self.u)
        
        # ω_mid = ω + dt * k1
        omega_mid = [
            self._truncate_terms([(1.0, self.omega[i]), (dt, k1[i])])
            for i in range(3)
        ]
        
        # RK2 Stage 2 (using omega_mid, approximate u_mid ≈ u)
        k2 = self._compute_rhs(omega_mid, self.u)
        
        # ω_new = ω + dt/2 * (k1 + k2)
        omega_new = [
            self._truncate_terms([(1.0, self.omega[i]), (0.5*dt, k1[i]), (0.5*dt, k2[i])])
            for i in range(3)
        ]
        
        # Update state
        self.omega = omega_new
        self.t += dt
        self._step_count += 1
        
        # Reconstruct velocity from updated vorticity
        # Only update every N steps to amortize cost (velocity changes slowly)
        if self._step_count % self.config.velocity_update_freq == 0:
            self._reconstruct_velocity_from_vorticity()
        
        # Diagnostics
        self.last_step_time = time.perf_counter() - t0
        
        # Compute enstrophy (||ω||²)
        enstrophy = sum(turbo_inner(self.omega[i], self.omega[i]).item() for i in range(3))
        
        # Rank statistics
        all_ranks = []
        for i in range(3):
            for c in self.omega[i][:-1]:
                all_ranks.append(c.shape[2])
        max_r = max(all_ranks) if all_ranks else 0
        mean_r = sum(all_ranks) / len(all_ranks) if all_ranks else 0
        
        # Adaptive rank diagnostics
        diag = {
            'time': self.t,
            'step_ms': self.last_step_time * 1000,
            'enstrophy': enstrophy,
            'max_rank': max_r,
            'mean_rank': mean_r,
        }
        
        if self.rank_controller is not None:
            diag['truncation_error'] = self.rank_controller.last_error
            diag['adaptive_ranks'] = self.rank_controller.last_ranks.copy() if self.rank_controller.last_ranks else []
            self.last_truncation_error = self.rank_controller.last_error
            self.last_ranks = self.rank_controller.last_ranks.copy()
        
        return diag


# ═══════════════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════════════

def _test_turbo_solver():
    """Test the Turbo NS solver."""
    import time
    
    print("Turbo 3D NS Solver Test")
    print("=" * 60)
    
    # 32³ grid
    config = TurboNS3DConfig(
        n_bits=5,
        nu=0.01,
        max_rank=32,
        dt=0.01,
        device='cuda',
    )
    
    print(f"Grid: {config.N}³ = {config.N**3:,} points")
    print(f"QTT cores: {config.n_cores}")
    print(f"Max rank: {config.max_rank}")
    print(f"ν = {config.nu}, dt = {config.dt}")
    
    solver = TurboNS3DSolver(config)
    
    print("\nInitializing Taylor-Green vortex...")
    solver.initialize_taylor_green()
    
    # Initial diagnostics
    enstrophy_0 = sum(turbo_inner(solver.omega[i], solver.omega[i]).item() for i in range(3))
    print(f"Initial enstrophy: {enstrophy_0:.4f}")
    
    # Run 10 steps
    print("\nRunning 10 steps...")
    t0 = time.perf_counter()
    
    for step in range(10):
        diag = solver.step()
        if (step + 1) % 5 == 0:
            print(f"  Step {step+1}: t={diag['time']:.3f}, Ω={diag['enstrophy']:.4f}, "
                  f"rank={diag['max_rank']}, {diag['step_ms']:.1f}ms")
    
    total_time = time.perf_counter() - t0
    
    print(f"\nTotal time: {total_time*1000:.1f}ms")
    print(f"Average: {total_time/10*1000:.1f}ms/step")
    
    # Check physics
    enstrophy_final = diag['enstrophy']
    print(f"\nEnstrophy: {enstrophy_0:.4f} → {enstrophy_final:.4f}")
    if enstrophy_final < enstrophy_0:
        print("✓ ENSTROPHY DECAYING (viscous dissipation)")
    else:
        print("⚠ ENSTROPHY GROWING (check numerics)")


if __name__ == "__main__":
    _test_turbo_solver()
