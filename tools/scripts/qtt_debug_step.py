#!/usr/bin/env python3
"""
Focused diagnostic: trace through a single QTT solver step and identify
where the blowup originates by reconstructing dense fields at each stage.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import time

from ontic.cfd.qtt_3d_state import QTT3DState
from ontic.cfd.nd_shift_mpo import make_nd_shift_mpo, apply_nd_shift_mpo
from ontic.cfd.pure_qtt_ops import truncate_qtt, qtt_hadamard


# ── QTT 3D helper ops (same as civic_aero_qtt.py) ──────────────────────

def qtt3d_add(a: QTT3DState, b: QTT3DState, mr: int) -> QTT3DState:
    ab_cores = []
    n = len(a.cores)
    for i in range(n):
        ca, cb = a.cores[i], b.cores[i]
        ra_l, d, ra_r = ca.shape
        rb_l, _, rb_r = cb.shape
        c = torch.zeros(ra_l + rb_l, d, ra_r + rb_r, device=ca.device, dtype=ca.dtype)
        c[:ra_l, :, :ra_r] = ca
        c[ra_l:, :, ra_r:] = cb
        ab_cores.append(c)
    # fix boundary
    ab_cores[0] = torch.cat([ab_cores[0][:1], ab_cores[0][ab_cores[0].shape[0]//2:ab_cores[0].shape[0]//2+1]], dim=2)
    # Actually that's wrong. Let me just use the same pattern from the solver:
    # For first core: (1, d, r_a + r_b) — just stack along last dim
    # For last core:  (r_a + r_b, d, 1) — sum left bond
    # Actually, let me just implement it properly:
    pass

# That's getting complicated. Let me just use the solver's own operations directly.

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    n_bits = 7  # 128³
    N = 1 << n_bits
    max_rank = 16
    work_rank = 32
    
    Lx, Ly, Lz = 16.0, 4.0, 4.0
    dx = Lx / N
    dy = Ly / N
    dz = Lz / N
    
    u_inf = 30.0
    nu_eff = max(u_inf * dx / 2.0, u_inf * dy / 2.0, u_inf * dz / 2.0) * 1.1
    
    inv_dx2_sum = 1/dx**2 + 1/dy**2 + 1/dz**2
    dt = 0.4 / (nu_eff * inv_dx2_sum)
    
    print(f"Device: {device}")
    print(f"Grid: {N}³, dx={dx:.5f}, dy={dy:.5f}, dz={dz:.5f}")
    print(f"dt = {dt:.6e}, nu_eff = {nu_eff:.4f}")
    print(f"max_rank = {max_rank}, work_rank = {work_rank}")
    print()
    
    # ── Test 1: Create a simple uniform field and check derivatives ──────
    print("=== Test 1: Uniform field derivatives ===")
    u_val = 1.0  # simple test value
    u = QTT3DState.from_dense(
        torch.full((N, N, N), u_val, device=device, dtype=dtype),
        max_rank=max_rank,
    )
    print(f"  Uniform field: rank = {u.max_rank}")
    
    # Build shift MPOs
    total_qubits = 3 * n_bits
    shift_plus = {}
    shift_minus = {}
    for axis in range(3):
        shift_plus[axis] = make_nd_shift_mpo(
            total_qubits, num_dims=3, axis_idx=axis, direction=+1,
            device=device, dtype=dtype,
        )
        shift_minus[axis] = make_nd_shift_mpo(
            total_qubits, num_dims=3, axis_idx=axis, direction=-1,
            device=device, dtype=dtype,
        )
    print("  Shift MPOs built")
    
    # Apply shift
    def shift(f, axis, direction):
        mpo = shift_plus[axis] if direction > 0 else shift_minus[axis]
        cores = apply_nd_shift_mpo(f.cores, mpo, max_rank=work_rank)
        return QTT3DState(cores=cores, n_bits=f.n_bits, device=f.device, dtype=f.dtype)
    
    # Derivative: (f+ - f-) / (2dx)
    def ddx(f, dx_val):
        fp = shift(f, axis=0, direction=+1)
        fm = shift(f, axis=0, direction=-1)
        # Subtract
        sub_cores = []
        for i in range(len(fp.cores)):
            ca, cb = fp.cores[i], fm.cores[i]
            ra_l, d, ra_r = ca.shape
            rb_l, _, rb_r = cb.shape
            c = torch.zeros(ra_l + rb_l, d, ra_r + rb_r, device=ca.device, dtype=ca.dtype)
            c[:ra_l, :, :ra_r] = ca
            c[ra_l:, :, ra_r:] = -cb  # minus sign for subtraction
            sub_cores.append(c)
        # Fix boundary cores for proper addition/subtraction
        # First core: concat along right bond
        c0 = sub_cores[0]
        if c0.shape[0] == 2:  # both parts have left-bond = 1
            sub_cores[0] = c0[:1, :, :]  # Take first row, but this is wrong...
        # Actually the standard way is:
        # First core: shape (1, d, r_a + r_b) → just concatenate along dim 2
        c0_a = fp.cores[0]  # (1, d, r_a)
        c0_b = fm.cores[0]  # (1, d, r_b)
        sub_cores[0] = torch.cat([c0_a, -c0_b], dim=2)  # (1, d, r_a + r_b)
        
        # Last core: shape (r_a + r_b, d, 1) → concat along dim 0
        cn_a = fp.cores[-1]  # (r_a, d, 1)
        cn_b = fm.cores[-1]  # (r_b, d, 1)
        sub_cores[-1] = torch.cat([cn_a, -cn_b], dim=0)  # (r_a + r_b, d, 1)
        
        # Wait, the sign should only be on one part, not both.
        # For subtraction a - b: first core concat [a, -b] along right,
        # last core concat [a, b] along left (no extra minus)
        # Middle cores: block diagonal [a, 0; 0, b]
        # Actually it's: first core [a_0, -b_0] right-concat,
        # middle cores block-diag [a_i, b_i], last core [a_n, b_n] left-concat
        
        # Let me just reconstruct to dense and do it there for diagnostics
        pass
    
    # ── Simpler: reconstruct to dense and do operations there ────────────
    print("\n=== Test 2: Round-trip dense → QTT → dense ===")
    
    # Create a test field with body-like discontinuity
    x = torch.linspace(0, Lx, N, device=device, dtype=dtype)
    y = torch.linspace(0, Ly, N, device=device, dtype=dtype)
    z = torch.linspace(0, Lz, N, device=device, dtype=dtype)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    
    # Simple uniform body mask (box)
    nose_x = 4.0
    length = 4.649
    width = 1.799 / 2
    height = 1.416
    ground_clearance = 0.132
    
    body = ((xx >= nose_x) & (xx <= nose_x + length) &
            (yy <= width) &
            (zz >= ground_clearance) & (zz <= ground_clearance + height)).float()
    
    n_body = body.sum().item()
    print(f"  Body cells: {n_body:.0f}")
    
    # Velocity field: freestream outside body, zero inside
    u_dense = u_inf * (1.0 - body)
    print(f"  u_dense: min={u_dense.min():.2f}, max={u_dense.max():.2f}, "
          f"mean={u_dense.mean():.2f}")
    
    # QTT representation
    t0 = time.perf_counter()
    u_qtt = QTT3DState.from_dense(u_dense, max_rank=max_rank)
    t1 = time.perf_counter()
    print(f"  QTT rank: {u_qtt.max_rank}, from_dense time: {t1-t0:.2f}s")
    
    # Reconstruct
    u_recon = u_qtt.to_dense()
    err = (u_recon - u_dense).abs()
    print(f"  Reconstruction error: max={err.max():.4f}, mean={err.mean():.6f}, "
          f"L2={err.norm():.4f}")
    print(f"  u_recon: min={u_recon.min():.2f}, max={u_recon.max():.2f}")
    
    # ── Test 3: Shift and derivative on the discontinuous field ──────────
    print("\n=== Test 3: Shift + derivative ===")
    
    # x-derivative of step function
    du_dx_dense = torch.zeros_like(u_dense)
    du_dx_dense[1:-1] = (u_dense[2:] - u_dense[:-2]) / (2 * dx)
    du_dx_dense[0] = (u_dense[1] - u_dense[-1]) / (2 * dx)
    du_dx_dense[-1] = (u_dense[0] - u_dense[-2]) / (2 * dx)
    print(f"  du/dx dense: min={du_dx_dense.min():.2f}, max={du_dx_dense.max():.2f}")
    
    # QTT shift
    up = shift(u_qtt, axis=0, direction=+1)
    um = shift(u_qtt, axis=0, direction=-1)
    
    up_dense = up.to_dense()
    um_dense = um.to_dense()
    
    # Check shift correctness
    up_expected = torch.roll(u_dense, -1, dims=0)  # shift +1 = neighbor at i+1
    um_expected = torch.roll(u_dense, +1, dims=0)  # shift -1 = neighbor at i-1
    
    err_up = (up_dense - up_expected).abs()
    err_um = (um_dense - um_expected).abs()
    print(f"  shift+1 error: max={err_up.max():.4f}, mean={err_up.mean():.6f}")
    print(f"  shift-1 error: max={err_um.max():.4f}, mean={err_um.mean():.6f}")
    print(f"  shift+1: min={up_dense.min():.2f}, max={up_dense.max():.2f}, rank={up.max_rank}")
    print(f"  shift-1: min={um_dense.min():.2f}, max={um_dense.max():.2f}, rank={um.max_rank}")
    
    # ── Test 4: Laplacian ────────────────────────────────────────────────
    print("\n=== Test 4: Laplacian ===")
    
    # Dense Laplacian (x-direction only for simplicity)
    lap_x_dense = (torch.roll(u_dense, -1, 0) + torch.roll(u_dense, 1, 0) - 2*u_dense) / dx**2
    lap_y_dense = (torch.roll(u_dense, -1, 1) + torch.roll(u_dense, 1, 1) - 2*u_dense) / dy**2
    lap_z_dense = (torch.roll(u_dense, -1, 2) + torch.roll(u_dense, 1, 2) - 2*u_dense) / dz**2
    lap_dense = lap_x_dense + lap_y_dense + lap_z_dense
    
    print(f"  Dense Laplacian: min={lap_dense.min():.2e}, max={lap_dense.max():.2e}")
    print(f"  nu * dt * |Lap|_max = {nu_eff * dt * lap_dense.abs().max():.4f}")
    
    # After one Euler step of pure diffusion
    u_diff = u_dense + dt * nu_eff * lap_dense
    print(f"  After diffusion step: min={u_diff.min():.2f}, max={u_diff.max():.2f}")
    
    # ── Test 5: Full predictor step in dense ─────────────────────────────
    print("\n=== Test 5: Dense predictor step ===")
    
    # Advection: u * du/dx (just x-component for diagnostic)
    adv_x_dense = u_dense * du_dx_dense
    print(f"  Advection u*du/dx: min={adv_x_dense.min():.2e}, max={adv_x_dense.max():.2e}")
    print(f"  dt * |adv|_max = {dt * adv_x_dense.abs().max():.4f}")
    
    # Predictor: u* = u + dt * (-adv + nu*lap)
    rhs_dense = -adv_x_dense + nu_eff * lap_dense
    u_star_dense = u_dense + dt * rhs_dense
    print(f"  u* = u + dt*rhs: min={u_star_dense.min():.2f}, max={u_star_dense.max():.2f}")
    delta_dense = u_star_dense - u_dense
    print(f"  |delta|_dense = {delta_dense.norm():.4f}")
    
    # ── Test 6: QTT Hadamard (advection core operation) ──────────────────
    print("\n=== Test 6: QTT Hadamard product ===")
    
    # u ⊙ du/dx in QTT
    du_dx_qtt_dense = (up_dense - um_dense) / (2 * dx)
    du_dx_qtt = QTT3DState.from_dense(du_dx_qtt_dense, max_rank=max_rank)
    print(f"  du/dx QTT rank: {du_dx_qtt.max_rank}")
    
    from ontic.cfd.pure_qtt_ops import qtt_hadamard
    
    # Hadamard: u * du/dx
    had_cores = qtt_hadamard(u_qtt.cores, du_dx_qtt.cores)
    had_qtt = QTT3DState(cores=had_cores, n_bits=n_bits, device=device, dtype=dtype)
    print(f"  Hadamard rank (pre-trunc): {had_qtt.max_rank}")
    
    # Truncate
    trunc_cores, info = truncate_qtt(had_cores, max_bond=max_rank)
    had_trunc = QTT3DState(cores=trunc_cores, n_bits=n_bits, device=device, dtype=dtype)
    print(f"  Hadamard rank (post-trunc): {had_trunc.max_rank}")
    
    had_dense = had_trunc.to_dense()
    had_expected = u_dense * du_dx_dense  # approximate expected
    err_had = (had_dense - had_expected).abs()
    print(f"  Hadamard error: max={err_had.max():.4e}, mean={err_had.mean():.6e}")
    print(f"  Hadamard result: min={had_dense.min():.2e}, max={had_dense.max():.2e}")
    print(f"  Expected result: min={had_expected.min():.2e}, max={had_expected.max():.2e}")
    
    # ── Summary ──────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"  Velocity scale: {u_inf:.0f} m/s")
    print(f"  Gradient scale: {u_inf/dx:.0f} m/s/m (x), {u_inf/dy:.0f} m/s/m (y,z)")
    print(f"  Advection u*du/dx scale: {u_inf**2/dx:.0f} m/s² (x)")
    print(f"  dt * advection: {dt * u_inf**2/dx:.4f} m/s")
    print(f"  Laplacian scale: {u_inf/dy**2:.0f} m/s/m²")
    print(f"  dt * nu * laplacian: {dt * nu_eff * u_inf/dy**2:.4f} m/s")


if __name__ == "__main__":
    main()
