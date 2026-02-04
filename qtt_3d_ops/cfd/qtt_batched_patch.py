"""
Integration Patch: Batched QTT Operations for ns3d_turbo.py
============================================================

This file shows the exact changes needed to integrate batched operations
into the existing TurboNS3DSolver. Copy these methods into ns3d_turbo.py
or import and monkey-patch.

CHANGES SUMMARY:
    1. _truncate_terms(): Route to fixed-rank path when adaptive_rank=False (BUG FIX)
    2. _compute_rhs(): Replace per-operation truncation with phase-level batched
    3. step(): Use batched RK2

EXPECTED PERFORMANCE:
    Before: 840 individual SVDs/step → 1800ms (32³, rank 32)
    After:  ~120 batched SVD calls → ~400ms

USAGE:
    # Option A: Monkey-patch (quickest integration)
    from tensornet.cfd.qtt_batched_patch import patch_solver
    solver = TurboNS3DSolver(config)
    patch_solver(solver)
    
    # Option B: Copy methods into TurboNS3DSolver class
    
    # Option C: Subclass
    from tensornet.cfd.qtt_batched_patch import BatchedTurboNS3DSolver
    solver = BatchedTurboNS3DSolver(config)
"""

import torch
import time
from typing import List, Tuple, Optional

from .qtt_batched_ops import (
    batched_truncation_sweep,
    single_truncation_sweep,
    add_cores_raw,
    hadamard_cores_raw,
    scale_cores,
    mpo_apply_raw,
    compute_rhs_batched,
    rk2_step_batched,
    batched_diagnostics,
    qtt_inner,
    qtt_norm,
    batched_linear_combination,
    batched_cross_product,
    batched_curl,
    batched_laplacian_vector,
)


# ===========================================================================
# Bug fix: _truncate_terms dispatch
# ===========================================================================

def _truncate_terms_fixed(self, coeffs, terms, max_rank=None):
    """
    Fixed version of _truncate_terms that respects adaptive_rank flag.
    
    BUG: Original always dispatches to turbo_linear_combination_adaptive,
    which calls torch.linalg.svd directly (bypassing rSVD) for rank estimation.
    When adaptive_rank=False, this wastes time on 1008 full SVDs vs 840 rSVDs.
    
    FIX: When adaptive_rank=False, use batched_linear_combination (fixed rank).
    When adaptive_rank=True, keep original adaptive path.
    """
    if max_rank is None:
        max_rank = self.config.max_rank
    
    if not self.config.adaptive_rank:
        # FIXED PATH: batched linear combination with fixed rank
        return batched_linear_combination(coeffs, terms, max_rank)
    else:
        # Original adaptive path (kept for compatibility)
        from .qtt_turbo import turbo_linear_combination_adaptive
        return turbo_linear_combination_adaptive(coeffs, terms, max_rank=max_rank)


# ===========================================================================
# Replacement: _compute_rhs with batched operations
# ===========================================================================

def _compute_rhs_batched(self, u, omega):
    """
    Replacement for _compute_rhs that uses phase-level batched truncation.
    
    SVD calls per invocation:
        Original: ~765 individual SVDs
        Batched:  ~60 batched SVDs (each processing 3 fields)
    """
    return compute_rhs_batched(
        u=u,
        omega=omega,
        nu=self.config.nu,
        dx=self.dx,
        shift_plus=self.shift_plus_mpos,
        shift_minus=self.shift_minus_mpos,
        max_rank=self.config.max_rank,
    )


# ===========================================================================
# Replacement: step() with batched RK2
# ===========================================================================

def _step_batched(self):
    """
    Replacement for step() that uses fully batched operations.
    """
    dt = self.config.dt
    max_rank = self.config.max_rank
    
    # Forcing (if enabled)
    if self.config.enable_forcing:
        self._apply_forcing()
    
    # RK2 stage 1
    k1 = self._compute_rhs_batched(self.u, self.omega)
    
    # Euler predictor
    omega_star = []
    for comp in range(3):
        raw = add_cores_raw(self.omega[comp], k1[comp], alpha=1.0, beta=dt)
        omega_star.append(raw)
    omega_star = batched_truncation_sweep(omega_star, max_rank)
    
    # Velocity update (periodic based on velocity_update_freq)
    if hasattr(self.config, 'velocity_update_freq'):
        freq = self.config.velocity_update_freq
        if self._step_count % freq == 0:
            self._reconstruct_velocity_from_vorticity()
    
    # RK2 stage 2
    k2 = self._compute_rhs_batched(self.u, omega_star)
    
    # Heun combine: omega_new = omega + dt/2 * (k1 + k2)
    omega_new = []
    for comp in range(3):
        k_avg = add_cores_raw(k1[comp], k2[comp], alpha=1.0, beta=1.0)
        raw = add_cores_raw(self.omega[comp], k_avg, alpha=1.0, beta=dt / 2.0)
        omega_new.append(raw)
    
    self.omega = batched_truncation_sweep(omega_new, max_rank)
    
    self.t += dt
    self._step_count += 1
    
    return batched_diagnostics(self.u, self.omega)


# ===========================================================================
# Monkey-patch function
# ===========================================================================

def patch_solver(solver):
    """
    Monkey-patch an existing TurboNS3DSolver to use batched operations.
    
    Usage:
        solver = TurboNS3DSolver(config)
        solver.initialize_taylor_green()
        patch_solver(solver)
        solver.step()  # Now uses batched ops
    """
    import types
    
    # Store references to MPO data
    # The solver should have shift_plus and shift_minus MPOs stored somewhere.
    # Typical names: self.shift_mpos_plus, self.shift_mpos_minus, 
    #                self._shift_plus, self._shift_minus,
    #                self.deriv._shift_plus, etc.
    # We need to find them. Check common attribute names:
    
    mpo_attrs_plus = [
        'shift_plus_mpos', '_shift_plus', 'shift_mpos_plus',
        'deriv._shift_plus', 'mpo_shift_plus',
    ]
    mpo_attrs_minus = [
        'shift_minus_mpos', '_shift_minus', 'shift_mpos_minus', 
        'deriv._shift_minus', 'mpo_shift_minus',
    ]
    
    def _find_attr(obj, candidates):
        for attr in candidates:
            if '.' in attr:
                parts = attr.split('.')
                cur = obj
                try:
                    for p in parts:
                        cur = getattr(cur, p)
                    return cur
                except AttributeError:
                    continue
            elif hasattr(obj, attr):
                return getattr(obj, attr)
        return None
    
    plus = _find_attr(solver, mpo_attrs_plus)
    minus = _find_attr(solver, mpo_attrs_minus)
    
    if plus is None or minus is None:
        raise AttributeError(
            "Could not find shift MPO attributes on solver. "
            "Expected one of: shift_plus_mpos, _shift_plus, etc. "
            f"Available attrs: {[a for a in dir(solver) if 'shift' in a.lower() or 'mpo' in a.lower()]}"
        )
    
    solver.shift_plus_mpos = plus
    solver.shift_minus_mpos = minus
    
    # Compute dx
    import numpy as np
    N = 2 ** solver.config.n_bits
    L = getattr(solver.config, 'L', 2 * np.pi)
    solver.dx = L / N
    
    # Track step count
    if not hasattr(solver, '_step_count'):
        solver._step_count = 0
    
    # Patch methods
    solver._truncate_terms = types.MethodType(_truncate_terms_fixed, solver)
    solver._compute_rhs_batched = types.MethodType(_compute_rhs_batched, solver)
    solver.step_original = solver.step
    solver.step = types.MethodType(_step_batched, solver)
    
    return solver


# ===========================================================================
# Subclass approach (cleaner for new code)
# ===========================================================================

class BatchedTurboMixin:
    """
    Mixin class that overrides TurboNS3DSolver hot-path methods.
    
    Usage:
        class MyBatchedSolver(BatchedTurboMixin, TurboNS3DSolver):
            pass
        
        solver = MyBatchedSolver(config)
    """
    
    def _truncate_terms(self, coeffs, terms, max_rank=None):
        return _truncate_terms_fixed(self, coeffs, terms, max_rank)
    
    def _compute_rhs(self, u, omega):
        return _compute_rhs_batched(self, u, omega)
    
    def step(self):
        return _step_batched(self)


# ===========================================================================
# Public API
# ===========================================================================

__all__ = [
    'patch_solver',
    'BatchedTurboMixin',
    '_truncate_terms_fixed',
    '_compute_rhs_batched',
    '_step_batched',
]
