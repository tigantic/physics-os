#!/usr/bin/env python3
"""
Minimal single-step diagnostics: test each operator in isolation.
Compare QTT results against dense ground truth at each stage.
"""

import sys, os
_tools_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_repo_root = os.path.join(_tools_dir, "..")
sys.path.insert(0, _tools_dir)
sys.path.insert(0, _repo_root)

import torch
import math
import time

from tensornet.cfd.qtt_3d_state import QTT3DState
from tensornet.cfd.nd_shift_mpo import make_nd_shift_mpo, apply_nd_shift_mpo
from tensornet.cfd.pure_qtt_ops import truncate_qtt, qtt_hadamard

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
n_bits = 7  # 128³
N = 1 << n_bits
total_qubits = 3 * n_bits
max_rank = 16
work_rank = 32

# Morton axis mapping
_s2m = {0: 2, 1: 1, 2: 0}

Lx, Ly, Lz = 16.0, 4.0, 4.0
dx, dy, dz = Lx/N, Ly/N, Lz/N
nu_eff = max(30.0 * dx / 2, 30.0 * dy / 2, 30.0 * dz / 2) * 1.1
dt = 0.4 / (nu_eff * (1/dx**2 + 1/dy**2 + 1/dz**2))

print(f"Grid: {N}³, rank={max_rank}, dt={dt:.2e}, nu={nu_eff:.4f}")
print(f"dx={dx:.5f}, dy={dy:.5f}, dz={dz:.5f}")

# Build shifts
print("Building shifts...")
shifts_p = {}
shifts_m = {}
for sa in range(3):
    ma = _s2m[sa]
    shifts_p[sa] = make_nd_shift_mpo(total_qubits, 3, ma, +1, device, dtype)
    shifts_m[sa] = make_nd_shift_mpo(total_qubits, 3, ma, -1, device, dtype)

def shift(f, axis, direction):
    mpo = shifts_p[axis] if direction > 0 else shifts_m[axis]
    cores = apply_nd_shift_mpo(f.cores, mpo, max_rank=work_rank)
    return QTT3DState(cores=cores, n_bits=f.n_bits, device=f.device, dtype=f.dtype)

# ── Create initial fields ──
print("Building body mask...")
x = torch.linspace(0, Lx, N, device=device, dtype=dtype)
y = torch.linspace(0, Ly, N, device=device, dtype=dtype)
z = torch.linspace(0, Lz, N, device=device, dtype=dtype)
xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')

# Body (simplified box)
nose_x, length = 4.0, 4.649
width, height = 1.799/2, 1.416
gc = 0.132
body_dense = torch.sigmoid(10.0 * (
    torch.min(torch.min(xx - nose_x, nose_x + length - xx),
              torch.min(torch.min(width - yy, yy - 0.0),
                        torch.min(zz - gc, gc + height - zz)))
)).to(device=device, dtype=dtype)
body_qtt = QTT3DState.from_dense(body_dense, max_rank=max_rank)
print(f"  body rank: {body_qtt.max_rank}")

# Initial velocity = 0.3 m/s everywhere (1% ramp)
u_val = 0.3
u_dense = torch.full((N, N, N), u_val, device=device, dtype=dtype)
v_dense = torch.zeros_like(u_dense)
w_dense = torch.zeros_like(u_dense)

u_qtt = QTT3DState.from_dense(u_dense, max_rank=max_rank)
v_qtt = QTT3DState.from_dense(v_dense, max_rank=max_rank)
w_qtt = QTT3DState.from_dense(w_dense, max_rank=max_rank)

print(f"  u rank: {u_qtt.max_rank}, v rank: {v_qtt.max_rank}")
u_recon = u_qtt.to_dense()
print(f"  u roundtrip error: {(u_recon - u_dense).abs().max():.6e}")

# ── Test 1: Pure diffusion step (no advection, no body) ──
print("\n=== Test 1: Pure diffusion (no advection, no body) ===")

# Dense Laplacian
lap_dense = (
    (torch.roll(u_dense, -1, 0) + torch.roll(u_dense, 1, 0) - 2*u_dense) / dx**2 +
    (torch.roll(u_dense, -1, 1) + torch.roll(u_dense, 1, 1) - 2*u_dense) / dy**2 +
    (torch.roll(u_dense, -1, 2) + torch.roll(u_dense, 1, 2) - 2*u_dense) / dz**2
)
u_diff_dense = u_dense + dt * nu_eff * lap_dense
print(f"  Dense: lap max={lap_dense.abs().max():.2e}, "
      f"u_diff range=[{u_diff_dense.min():.4f}, {u_diff_dense.max():.4f}]")

# QTT Laplacian  
idx2, idy2, idz2 = 1/dx**2, 1/dy**2, 1/dz**2
diag_coeff = 2*(idx2 + idy2 + idz2)

t0 = time.perf_counter()
fxp = shift(u_qtt, 0, +1)
fxm = shift(u_qtt, 0, -1)
fyp = shift(u_qtt, 1, +1)
fym = shift(u_qtt, 1, -1)
fzp = shift(u_qtt, 2, +1)
fzm = shift(u_qtt, 2, -1)
t1 = time.perf_counter()
print(f"  Shifts: {t1-t0:.2f}s, fxp rank={fxp.max_rank}")

# Check shift of uniform field
fxp_dense = fxp.to_dense()
fxp_expected = torch.roll(u_dense, +1, dims=0)  # shift(+1) = roll(+1) = f(x-1)
err = (fxp_dense - fxp_expected).abs().max()
print(f"  shift(+1,axis=0) error: {err:.6e}")

# For uniform field u=0.3, all shifts should give 0.3
# And Laplacian should be ~0
fxp_val = fxp_dense.mean()
fxm_val = fxm.to_dense().mean()
print(f"  fxp mean={fxp_val:.6f}, fxm mean={fxm_val:.6f} (expected {u_val})")

# QTT Laplacian assembly
from scripts.civic_aero_qtt import qtt3d_add, qtt3d_sub, qtt3d_scale, qtt3d_truncate

diag = qtt3d_scale(u_qtt, diag_coeff)
sx = qtt3d_scale(qtt3d_add(fxp, fxm, work_rank), idx2)
sy = qtt3d_scale(qtt3d_add(fyp, fym, work_rank), idy2)
sz = qtt3d_scale(qtt3d_add(fzp, fzm, work_rank), idz2)
off = qtt3d_add(sx, sy, work_rank)
off = qtt3d_add(off, sz, work_rank)
neg_lap = qtt3d_sub(diag, off, work_rank)
neg_lap, _ = qtt3d_truncate(neg_lap, max_rank, tol=1e-8)
lap_qtt = qtt3d_scale(neg_lap, -1.0)

lap_qtt_dense = lap_qtt.to_dense()
lap_err = (lap_qtt_dense - lap_dense).abs()
print(f"  QTT Laplacian error: max={lap_err.max():.4e}, mean={lap_err.mean():.6e}")
print(f"  QTT Laplacian: max abs val = {lap_qtt_dense.abs().max():.4e}")

# Diffusion step in QTT
u_diff_qtt = qtt3d_add(u_qtt, qtt3d_scale(lap_qtt, nu_eff * dt), work_rank)
u_diff_qtt, _ = qtt3d_truncate(u_diff_qtt, max_rank, tol=1e-8)
u_diff_recon = u_diff_qtt.to_dense()
err_diff = (u_diff_recon - u_diff_dense).abs()
print(f"  Diffusion step error: max={err_diff.max():.4e}, "
      f"u_diff range=[{u_diff_recon.min():.4f}, {u_diff_recon.max():.4f}]")

# ── Test 2: Brinkman on body ──
print("\n=== Test 2: Brinkman penalty ===")
penalty_factor = 0.5  # λ*dt = 5445 * 9.18e-5

# Dense
pen_dense = body_dense * u_dense
u_brk_dense = u_dense - penalty_factor * pen_dense
print(f"  Dense: u_brk range=[{u_brk_dense.min():.4f}, {u_brk_dense.max():.4f}]")

# QTT Hadamard
from tensornet.cfd.pure_qtt_ops import qtt_hadamard as qtt_had_raw

def qtt3d_had(a, b, mr):
    cores = qtt_had_raw(
        type('Q', (), {'cores': a.cores, 'num_qubits': len(a.cores)})(),
        type('Q', (), {'cores': b.cores, 'num_qubits': len(b.cores)})(),
    )
    trunc, _ = truncate_qtt(cores, max_bond=mr)
    return QTT3DState(cores=trunc, n_bits=a.n_bits, device=a.device, dtype=a.dtype)

pen_qtt = qtt3d_had(body_qtt, u_qtt, max_rank)
pen_recon = pen_qtt.to_dense()
err_pen = (pen_recon - pen_dense).abs()
print(f"  Hadamard (body*u) error: max={err_pen.max():.4e}, "
      f"mean={err_pen.mean():.6e}, rank={pen_qtt.max_rank}")

u_brk_qtt = qtt3d_sub(u_qtt, qtt3d_scale(pen_qtt, penalty_factor), work_rank)
u_brk_qtt, _ = qtt3d_truncate(u_brk_qtt, max_rank, tol=1e-8)
u_brk_recon = u_brk_qtt.to_dense()
err_brk = (u_brk_recon - u_brk_dense).abs()
print(f"  Brinkman step error: max={err_brk.max():.4e}, "
      f"u_brk range=[{u_brk_recon.min():.4f}, {u_brk_recon.max():.4f}]")

# ── Test 3: Full step accumulation ──
print("\n=== Test 3: Dense reference step ===")
# Advection
du_dx = (torch.roll(u_dense, -1, 0) - torch.roll(u_dense, 1, 0)) / (2*dx)
adv_dense = u_dense * du_dx  # v=w=0, so only u*du/dx
print(f"  Advection max: {adv_dense.abs().max():.4e}")

# Full predictor
rhs_dense = -adv_dense + nu_eff * lap_dense
u_star_dense = u_dense + dt * rhs_dense
print(f"  Predictor: u* range=[{u_star_dense.min():.4f}, {u_star_dense.max():.4f}]")

# After Brinkman
u_star_brk = u_star_dense - penalty_factor * body_dense * u_star_dense
print(f"  After Brinkman: u* range=[{u_star_brk.min():.4f}, {u_star_brk.max():.4f}]")

# Divergence
div_dense = (
    (torch.roll(u_star_brk, -1, 0) - torch.roll(u_star_brk, 1, 0)) / (2*dx) +
    0.0 + 0.0  # v=w=0
)
print(f"  Divergence: max={div_dense.abs().max():.4e}")

# AC pressure
beta = 0.5 * min(dx,dy,dz)**2 / dt
p_dense = torch.zeros_like(u_dense)
p_new = p_dense - beta * dt * div_dense
print(f"  Pressure: max={p_new.abs().max():.4e}, β={beta:.2f}")

# Velocity correction
dp_dx = (torch.roll(p_new, -1, 0) - torch.roll(p_new, 1, 0)) / (2*dx)
u_final = u_star_brk - dt * dp_dx
print(f"  After correction: u range=[{u_final.min():.4f}, {u_final.max():.4f}]")

delta = u_final - u_dense
res = delta.norm()
print(f"  Residual (dense): {res:.4e}")
print(f"  Max |delta|: {delta.abs().max():.4e}")

print("\nDone.")
