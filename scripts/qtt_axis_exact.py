#!/usr/bin/env python3
"""Exact verification of Morton axis mapping at N=4."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from tensornet.cfd.qtt_3d_state import QTT3DState
from tensornet.cfd.nd_shift_mpo import make_nd_shift_mpo, apply_nd_shift_mpo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
n_bits = 2  # 4³
N = 1 << n_bits
total_qubits = 3 * n_bits  # 6
max_rank = 64

print(f"Grid: {N}³, qubits: {total_qubits}")

# Simple test: f[i,j,k] = i (varies only in tensor dim 0)
x = torch.arange(N, device=device, dtype=dtype)
y = torch.zeros(N, device=device, dtype=dtype)
z = torch.zeros(N, device=device, dtype=dtype)
xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
f_dense = xx.clone()  # f = 0,1,2,3 along dim 0
print(f"\nf = xx (varies in dim 0 only)")
print(f"f[:, 0, 0] = {f_dense[:, 0, 0].tolist()}")

f_qtt = QTT3DState.from_dense(f_dense, max_rank=max_rank)
f_recon = f_qtt.to_dense()
roundtrip_err = (f_recon - f_dense).abs().max()
print(f"QTT roundtrip error: {roundtrip_err:.6f}, rank: {f_qtt.max_rank}")

# Test all 6 Morton axis_idx values and compare to torch.roll
print("\nShift +1 results:")
_solver_to_morton = {0: 2, 1: 1, 2: 0}

for morton_axis in range(3):
    mpo = make_nd_shift_mpo(total_qubits, num_dims=3, axis_idx=morton_axis, direction=+1,
                            device=device, dtype=dtype)
    cores = apply_nd_shift_mpo(f_qtt.cores, mpo, max_rank=max_rank)
    shifted = QTT3DState(cores=cores, n_bits=n_bits, device=device, dtype=dtype)
    s_dense = shifted.to_dense()
    print(f"\n  Morton axis_idx={morton_axis}:")
    print(f"    shifted[:, 0, 0] = {s_dense[:, 0, 0].tolist()}")
    print(f"    shifted[0, :, 0] = {s_dense[0, :, 0].tolist()}")
    print(f"    shifted[0, 0, :] = {s_dense[0, 0, :].tolist()}")

print("\nExpected for shift +1 (roll -1):")
for dim in range(3):
    rolled = torch.roll(f_dense, -1, dims=dim)
    print(f"  torch.roll(-1, dim={dim})[:, 0, 0] = {rolled[:, 0, 0].tolist()}")
    print(f"  torch.roll(-1, dim={dim})[0, :, 0] = {rolled[0, :, 0].tolist()}")
    print(f"  torch.roll(-1, dim={dim})[0, 0, :] = {rolled[0, 0, :].tolist()}")

# Also test with f = 100 * zz (varies in dim 2 only)
print(f"\n\nf = 100*zz (varies in dim 2 only)")
f2_dense = 100.0 * zz
f2_qtt = QTT3DState.from_dense(f2_dense, max_rank=max_rank)
print(f"f[0, 0, :] = {f2_dense[0, 0, :].tolist()}")

for morton_axis in range(3):
    mpo = make_nd_shift_mpo(total_qubits, num_dims=3, axis_idx=morton_axis, direction=+1,
                            device=device, dtype=dtype)
    cores = apply_nd_shift_mpo(f2_qtt.cores, mpo, max_rank=max_rank)
    shifted = QTT3DState(cores=cores, n_bits=n_bits, device=device, dtype=dtype)
    s_dense = shifted.to_dense()
    print(f"  Morton axis_idx={morton_axis}: shifted[0,0,:] = {s_dense[0, 0, :].tolist()}")

print("\nExpected for shift +1 (roll -1):")
for dim in range(3):
    rolled = torch.roll(f2_dense, -1, dims=dim)
    print(f"  torch.roll(-1, dim={dim})[0,0,:] = {rolled[0, 0, :].tolist()}")
