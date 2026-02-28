#!/usr/bin/env python3
"""Gradient diagnostic for Chu optimizer — traces signal at each stage."""
import sys, math, torch
sys.path.insert(0, '.')
from ontic.em.chu_limit_gpu import *
from ontic.em.qtt_helmholtz_gpu import *

config = make_chu_gpu_config('4096')
device = torch.device('cuda')
dtype = torch.complex64
N = config.N
n_bits = config.n_bits
k0_norm = config.k0_normalised
h = 1.0 / N
dv = h ** 3

print(f'N={N}, n_bits={n_bits}, k0_norm={k0_norm:.4f}, dv={dv:.6e}')

# Build infrastructure
pml_cfg = config.pml_config()
sigma_pml_tt = build_pml_sigma_tt_gpu(n_bits, k0_norm, pml_cfg, device, dtype, max_rank=config.max_rank)
print("PML done")
design_mask_tt = build_sphere_mask_tt_gpu(n_bits, (0.5, 0.5, 0.5), config.sphere_radius_normalised, device, dtype, config.max_rank)
print("Sphere mask done")
design_flat_idx = spherical_mask_flat_indices_gpu(n_bits, (0.5, 0.5, 0.5), config.sphere_radius_normalised, device)
n_design = design_flat_idx.shape[0]
print(f'n_design={n_design}')

# Init density
density = config.vol_target + 0.1*(2.0*torch.rand(n_design, device=device, dtype=torch.float64)-1.0)
density = density.clamp(0.01, 0.99)
ix_flat = design_flat_idx // (N*N)
iy_flat = (design_flat_idx // N) % N
wire_mask = ((ix_flat - N//2).abs() <= 2) & ((iy_flat - N//2).abs() <= 2)
density[wire_mask] = 0.95
print(f'density: mean={density.mean().item():.4f}')

# Vol-preserving projection
rho_filt = density_filter_gpu(density, config.filter_radius)
rho_proj, eta_vol = volume_preserving_projection_gpu(rho_filt, 1.0, config.vol_target)
print(f'eta_vol={eta_vol:.4f}, proj_mean={rho_proj.mean().item():.4f}')

# Forward solve
sigma_max_0 = config.sigma_max_at_iter(0)
simp_p_0 = config.simp_p_at_iter(0)
print("Starting forward solve...")
H_cores, E_cores, source_cores, res_fwd, _ = solve_forward_gpu(
    density, design_mask_tt, n_bits, k0_norm, config.domain_size,
    config.pml_cells, config.pml_sigma_max, config.sigma_min, sigma_max_0,
    simp_p_0, 1.0, eta_vol, config.filter_radius, config.damping,
    config.max_rank, config.n_sweeps, config.solver_tol, config.source_width,
    device, dtype, False,
)
print(f'Forward residual: {res_fwd:.4e}')

# Field stats
E_norm_sq = tt_inner_gpu(E_cores, E_cores).real.item()
print(f'||E||^2 = {E_norm_sq:.6e}')

E_design = tt_evaluate_at_indices_gpu(E_cores, design_flat_idx, n_bits)
E_abs = E_design.abs()
print(f'|E_design|: mean={E_abs.mean().item():.6e}, max={E_abs.max().item():.6e}, min={E_abs.min().item():.6e}')

# Power
P_pml = compute_pml_power_tt_gpu(E_cores, sigma_pml_tt, dv, config.max_rank)
W_near = 0.5 * dv * E_norm_sq
Q_proxy = k0_norm * W_near / (abs(P_pml) + 1e-30)
print(f'P_pml={P_pml:.6e}  W_near={W_near:.6e}  Q={Q_proxy:.4f}')

# Weights
w_pml = -1.0 / (P_pml + 1e-12)
w_wnear = 1.0 / (W_near + 1e-12)
print(f'w_pml={w_pml:.4e}  w_wnear={w_wnear:.4e}')
print(f'w_wnear*0.5*dv={w_wnear*0.5*dv:.4e}  w_pml*0.5*dv={w_pml*0.5*dv:.4e}')

# Adjoint RHS
sigma_vals = simp_sigma_gpu(rho_proj, config.sigma_min, sigma_max_0, simp_p_0)
sigma_mean = sigma_vals.mean().item()
sigma_design_tt = tt_scale_gpu([c.clone() for c in design_mask_tt], sigma_mean)

# PML-only RHS (no W_near)
g_pml_only = build_adjoint_rhs_tt_gpu(
    E_cores, sigma_pml_tt, sigma_design_tt, w_pml, 0.0, k0_norm, dv, config.max_rank)
g_pml_norm = tt_norm_gpu(g_pml_only)
print(f'||g_pml_only|| = {g_pml_norm:.6e}')

# W_near RHS alone
wnear_rhs = tt_scale_gpu([c.clone() for c in E_cores], w_wnear * 0.5 * dv)
g_wnear_norm = tt_norm_gpu(wnear_rhs)
print(f'||g_wnear||    = {g_wnear_norm:.6e}')

# Combined
g_cores = tt_add_gpu(g_pml_only, wnear_rhs)
g_cores = tt_round_gpu(g_cores, max_rank=config.max_rank)
g_norm = tt_norm_gpu(g_cores)
print(f'||g_combined|| = {g_norm:.6e}')

# Check if PML and W_near RHS cancel
print(f'Cancellation ratio ||combined||/(||pml||+||wnear||) = {g_norm/(g_pml_norm + g_wnear_norm):.4f}')

# Adjoint solve
print("Starting adjoint solve...")
H_H = mpo_hermitian_conjugate_gpu(H_cores)
adj_result = tt_amen_solve_gpu(
    H_H, g_cores, max_rank=config.max_rank,
    n_sweeps=config.n_sweeps, tol=config.solver_tol, verbose=False)
print(f'Adjoint residual: {adj_result.final_residual:.4e}')

lam_design = tt_evaluate_at_indices_gpu(adj_result.x, design_flat_idx, n_bits)
lam_abs = lam_design.abs()
print(f'|lam_design|: mean={lam_abs.mean().item():.6e}, max={lam_abs.max().item():.6e}')

# Gradient terms
dsigma = simp_dsigma_drho_gpu(rho_proj, config.sigma_min, sigma_max_0, simp_p_0).to(torch.float64)
E_f = E_design.to(torch.complex128)
lam_f = lam_design.to(torch.complex128)
k2_damp = k0_norm**2 * (1.0 + 1j * config.damping)
dH_drho_E = k2_damp * (-1j) * dsigma.to(torch.complex128) * E_f
adjoint_term = -(lam_f.conj() * dH_drho_E).real

print(f'|dH_drho_E|: max={dH_drho_E.abs().max().item():.6e}')
print(f'|adjoint_term|: mean={adjoint_term.abs().mean().item():.6e}, max={adjoint_term.abs().max().item():.6e}')

# Check for cancellation in Re[lam* * dH_drho_E]
product = lam_f.conj() * dH_drho_E
print(f'|Re(lam*dH*E)|/|lam*dH*E|: mean={product.real.abs().mean().item()/(product.abs().mean().item()+1e-30):.4f}')

h_grad = heaviside_gradient_gpu(rho_filt, 1.0, eta_vol).to(torch.float64)
print(f'h_grad: mean={h_grad.mean().item():.4e}')
final_grad = density_filter_gradient_gpu(adjoint_term * h_grad, config.filter_radius)
print(f'FINAL GRAD: max={final_grad.abs().max().item():.6e}, norm={final_grad.norm().item():.6e}')

print("\nDone.")
