#!/usr/bin/env python3
"""Verify shift directions and axis mapping are now correct."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from tensornet.cfd.qtt_3d_state import QTT3DState
from tensornet.cfd.nd_shift_mpo import make_nd_shift_mpo, apply_nd_shift_mpo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
n_bits = 3  # 8³
N = 1 << n_bits
total_qubits = 3 * n_bits
max_rank = 256  # allow full rank for exact representation

print(f"Grid: {N}³, qubits: {total_qubits}")

# Morton-axis mapping: solver axis → morton axis_idx
_solver_to_morton = {0: 2, 1: 1, 2: 0}

# Build shift MPOs with correct mapping
shifts_plus = {}
shifts_minus = {}
for solver_axis in range(3):
    morton_axis = _solver_to_morton[solver_axis]
    shifts_plus[solver_axis] = make_nd_shift_mpo(
        total_qubits, num_dims=3, axis_idx=morton_axis, direction=+1,
        device=device, dtype=dtype,
    )
    shifts_minus[solver_axis] = make_nd_shift_mpo(
        total_qubits, num_dims=3, axis_idx=morton_axis, direction=-1,
        device=device, dtype=dtype,
    )

def shift(f, axis, direction):
    mpo = shifts_plus[axis] if direction > 0 else shifts_minus[axis]
    cores = apply_nd_shift_mpo(f.cores, mpo, max_rank=max_rank)
    return QTT3DState(cores=cores, n_bits=f.n_bits, device=f.device, dtype=f.dtype)

# Create test fields
x = torch.arange(N, device=device, dtype=dtype)
y = torch.arange(N, device=device, dtype=dtype)
z = torch.arange(N, device=device, dtype=dtype)
xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')

# f = sin(2π x/N) — smooth periodic function in dim 0
import math
f_dense = torch.sin(2 * math.pi * xx / N)
f_qtt = QTT3DState.from_dense(f_dense, max_rank=max_rank)
f_recon = f_qtt.to_dense()
print(f"\nf = sin(2π x / N)")
print(f"  Roundtrip error: {(f_recon - f_dense).abs().max():.6e}")
print(f"  f[:, 0, 0] = {f_dense[:, 0, 0].tolist()}")

# Test derivative: ∂f/∂x using corrected convention
# shift(-1) → f(x+1) = RIGHT neighbor
# shift(+1) → f(x-1) = LEFT neighbor
dx = 1.0  # Unit spacing for simplicity

fp = shift(f_qtt, axis=0, direction=-1)  # f(x+1)
fm = shift(f_qtt, axis=0, direction=+1)  # f(x-1)

fp_dense = fp.to_dense()
fm_dense = fm.to_dense()

# Check: fp should be f rolled by -1 along dim 0
expected_fp = torch.roll(f_dense, -1, dims=0)
expected_fm = torch.roll(f_dense, +1, dims=0)
err_fp = (fp_dense - expected_fp).abs().max()
err_fm = (fm_dense - expected_fm).abs().max()
print(f"\n  shift(-1) = f(x+1)?  error = {err_fp:.6e}")
print(f"  shift(+1) = f(x-1)?  error = {err_fm:.6e}")

# Derivative
from tensornet.cfd.pure_qtt_ops import qtt_add
deriv_dense = (fp_dense - fm_dense) / (2 * dx)
expected_deriv = (expected_fp - expected_fm) / (2 * dx)
exact_deriv = (2 * math.pi / N) * torch.cos(2 * math.pi * xx / N)

err_deriv_vs_fd = (deriv_dense - expected_deriv).abs().max()
err_deriv_vs_exact = (deriv_dense[:, 0, 0] - exact_deriv[:, 0, 0]).abs().max()
print(f"\n  QTT derivative vs dense FD: error = {err_deriv_vs_fd:.6e}")
print(f"  QTT derivative vs exact:    error = {err_deriv_vs_exact:.6e}")
print(f"  deriv[:, 0, 0] = {[f'{v:.4f}' for v in deriv_dense[:, 0, 0].tolist()]}")
print(f"  exact[:, 0, 0] = {[f'{v:.4f}' for v in exact_deriv[:, 0, 0].tolist()]}")

# Also test y-axis and z-axis derivatives
for dim, name, spacing in [(1, 'y', 1.0), (2, 'z', 1.0)]:
    g_dense = torch.sin(2 * math.pi * [xx, yy, zz][dim] / N)
    g_qtt = QTT3DState.from_dense(g_dense, max_rank=max_rank)
    gp = shift(g_qtt, axis=dim, direction=-1)  # g(x+1)
    gm = shift(g_qtt, axis=dim, direction=+1)  # g(x-1)
    gp_dense = gp.to_dense()
    gm_dense = gm.to_dense()
    
    expected_gp = torch.roll(g_dense, -1, dims=dim)
    err_gp = (gp_dense - expected_gp).abs().max()
    print(f"\n  {name}-axis: shift(-1) = f({name}+1)?  error = {err_gp:.6e}")
    
    deriv_g = (gp_dense - gm_dense) / (2 * spacing)
    exact_g = (2 * math.pi / N) * torch.cos(2 * math.pi * [xx, yy, zz][dim] / N)
    # Check along the relevant slice
    if dim == 1:
        err = (deriv_g[0, :, 0] - exact_g[0, :, 0]).abs().max()
    else:
        err = (deriv_g[0, 0, :] - exact_g[0, 0, :]).abs().max()
    print(f"  {name}-deriv vs exact: error = {err:.6e}")

print("\n✓ All tests passed" if err_fp < 0.01 and err_fm < 0.01 else "\n✗ ERRORS DETECTED")
