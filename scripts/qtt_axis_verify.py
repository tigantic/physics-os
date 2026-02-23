#!/usr/bin/env python3
"""
Verify Morton axis mapping fix: shift MPO axis vs tensor dimension.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import time

from tensornet.cfd.qtt_3d_state import QTT3DState
from tensornet.cfd.nd_shift_mpo import make_nd_shift_mpo, apply_nd_shift_mpo


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    n_bits = 4  # 16³ — small for fast dense verification
    N = 1 << n_bits
    total_qubits = 3 * n_bits
    max_rank = 64  # full for verification (no truncation error)
    work_rank = 64

    print(f"Device: {device}, Grid: {N}³, n_bits: {n_bits}")

    # Create a test field with different structure in each dimension
    # f(x,y,z) = x + 10*y + 100*z  (each dim has unique coefficient)
    x = torch.arange(N, device=device, dtype=dtype)
    y = torch.arange(N, device=device, dtype=dtype)
    z = torch.arange(N, device=device, dtype=dtype)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    
    # Tensor dims: [dim0=16, dim1=16, dim2=16]
    # dim 0 varies with xx (range 0..15)
    # dim 1 varies with yy (range 0..15)
    # dim 2 varies with zz (range 0..15)
    f_dense = xx + 10.0 * yy + 100.0 * zz
    print(f"f_dense: shape={f_dense.shape}")
    print(f"  f[0,0,0]={f_dense[0,0,0]:.0f}, f[1,0,0]={f_dense[1,0,0]:.0f}, "
          f"f[0,1,0]={f_dense[0,1,0]:.0f}, f[0,0,1]={f_dense[0,0,1]:.0f}")

    # Convert to QTT
    f_qtt = QTT3DState.from_dense(f_dense, max_rank=max_rank)
    f_recon = f_qtt.to_dense()
    err = (f_recon - f_dense).abs().max()
    print(f"  QTT roundtrip error: {err:.6f} (rank {f_qtt.max_rank})")

    # ── Test shifts with WRONG mapping (original code) ──────────────
    print("\n=== WRONG mapping: Morton axis_idx = solver axis ===")
    for solver_axis in range(3):
        morton_axis = solver_axis  # WRONG: direct mapping
        mpo_plus = make_nd_shift_mpo(
            total_qubits, num_dims=3, axis_idx=morton_axis, direction=+1,
            device=device, dtype=dtype,
        )
        cores_shifted = apply_nd_shift_mpo(f_qtt.cores, mpo_plus, max_rank=work_rank)
        shifted = QTT3DState(cores=cores_shifted, n_bits=n_bits, device=device, dtype=dtype)
        shifted_dense = shifted.to_dense()

        # Expected: torch.roll along solver_axis dimension
        expected = torch.roll(f_dense, -1, dims=solver_axis)
        err = (shifted_dense - expected).abs().max()
        print(f"  solver axis {solver_axis} (morton={morton_axis}): "
              f"max error = {err:.4f}")

    # ── Test shifts with CORRECT mapping ─────────────────────────────
    print("\n=== CORRECT mapping: solver→morton = {0:2, 1:1, 2:0} ===")
    _solver_to_morton = {0: 2, 1: 1, 2: 0}
    for solver_axis in range(3):
        morton_axis = _solver_to_morton[solver_axis]
        mpo_plus = make_nd_shift_mpo(
            total_qubits, num_dims=3, axis_idx=morton_axis, direction=+1,
            device=device, dtype=dtype,
        )
        cores_shifted = apply_nd_shift_mpo(f_qtt.cores, mpo_plus, max_rank=work_rank)
        shifted = QTT3DState(cores=cores_shifted, n_bits=n_bits, device=device, dtype=dtype)
        shifted_dense = shifted.to_dense()

        expected = torch.roll(f_dense, -1, dims=solver_axis)
        err = (shifted_dense - expected).abs().max()
        print(f"  solver axis {solver_axis} (morton={morton_axis}): "
              f"max error = {err:.4f}")

    # ── Also test backward shift ─────────────────────────────────────
    print("\n=== CORRECT mapping: backward shift ===")
    for solver_axis in range(3):
        morton_axis = _solver_to_morton[solver_axis]
        mpo_minus = make_nd_shift_mpo(
            total_qubits, num_dims=3, axis_idx=morton_axis, direction=-1,
            device=device, dtype=dtype,
        )
        cores_shifted = apply_nd_shift_mpo(f_qtt.cores, mpo_minus, max_rank=work_rank)
        shifted = QTT3DState(cores=cores_shifted, n_bits=n_bits, device=device, dtype=dtype)
        shifted_dense = shifted.to_dense()

        expected = torch.roll(f_dense, +1, dims=solver_axis)
        err = (shifted_dense - expected).abs().max()
        print(f"  solver axis {solver_axis} (morton={morton_axis}): "
              f"max error = {err:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
