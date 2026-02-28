#!/usr/bin/env python3
"""Ultra-minimal per-voxel σ test — just construction + one forward solve.

Tests that the per-voxel σ path produces a valid eps_tt and the
forward solve works. Avoids the adjoint solve to isolate issues.
"""
import sys, os, signal, traceback
signal.signal(signal.SIGINT, signal.SIG_IGN)
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def log(msg):
    print(msg, flush=True)

try:
    import torch
    log(f"CUDA: {torch.cuda.is_available()}")

    from ontic.em.chu_limit_gpu import (
        make_chu_gpu_config, build_sphere_mask_tt_gpu,
        spherical_mask_flat_indices_gpu, build_pml_sigma_tt_gpu,
        build_conductivity_eps_tt_gpu, solve_forward_gpu,
        compute_adjoint_gradient_gpu, density_filter_gpu,
        volume_preserving_projection_gpu, compute_pml_power_tt_gpu,
        tt_evaluate_at_indices_gpu, simp_sigma_gpu,
        heaviside_projection_gpu, simp_dsigma_drho_gpu,
    )
    from ontic.em.qtt_helmholtz_gpu import (
        tt_inner_gpu, tt_norm_gpu, helmholtz_mpo_3d_gpu,
        tt_amen_solve_gpu, tt_scale_gpu, mpo_add_gpu,
        diag_mpo_from_tt_gpu, tt_matvec_gpu, tt_add_gpu, tt_round_gpu,
        gaussian_source_tt_gpu,
    )
    from ontic.em.boundaries import PMLConfig

    log("Imports OK")

    device = torch.device("cuda")
    dtype = torch.complex64
    n_bits = 7  # 128³
    config = make_chu_gpu_config("128")
    C0 = 299_792_458.0
    wavelength = C0 / config.frequency_hz
    k0_norm = 2.0 * 3.14159265 / wavelength * config.domain_wavelengths * wavelength
    r_norm = config.ka / k0_norm
    h = 1.0 / (2**n_bits)
    dv = h**3

    log(f"k0_norm={k0_norm:.4f}, n_bits={n_bits}")

    # Build infrastructure
    design_flat_idx = spherical_mask_flat_indices_gpu(n_bits, radius=r_norm, device=device)
    design_mask_tt = build_sphere_mask_tt_gpu(n_bits, radius=r_norm, device=device, dtype=dtype)
    n_design = design_flat_idx.shape[0]
    log(f"Design voxels: {n_design}")

    pml = PMLConfig(n_cells=config.pml_cells, sigma_max=config.pml_sigma_max)
    sigma_pml_tt = build_pml_sigma_tt_gpu(n_bits, k0_norm, pml, device=device, dtype=dtype)
    log("Infrastructure built")

    # Density
    density = 0.3 * torch.ones(n_design, device=device, dtype=torch.float64)
    noise = 0.10 * (2.0 * torch.rand(n_design, device=device, dtype=torch.float64) - 1.0)
    density = (density + noise).clamp(0.01, 0.99)
    log(f"Density: mean={density.mean():.3f}, std={density.std():.3f}")

    # === TEST 1: Per-voxel eps ===
    log("\n=== TEST 1: Per-voxel ε construction ===")
    eps_tt, sigma_tt = build_conductivity_eps_tt_gpu(
        density, design_mask_tt, config.sigma_min, 30.0, 1.0,
        1.0, 0.5, 0, n_bits, device, dtype, config.max_rank,
        design_flat_indices=design_flat_idx,
    )
    log(f"eps_tt max_rank={max(c.shape[-1] for c in eps_tt)}")
    log(f"sigma_tt max_rank={max(c.shape[-1] for c in sigma_tt)}")
    log(f"||sigma_tt||={tt_norm_gpu(sigma_tt):.4f}")

    # === TEST 2: Forward solve ===
    log("\n=== TEST 2: Forward solve with per-voxel ε ===")
    H_cores, E_cores, source_cores, res, sigma_fwd = solve_forward_gpu(
        density, design_mask_tt, n_bits, k0_norm,
        config.domain_size, config.pml_cells, config.pml_sigma_max,
        config.sigma_min, 30.0, 1.0,
        1.0, 0.5, 0, 0.02, config.max_rank, config.n_sweeps,
        config.solver_tol, config.source_width,
        device, dtype, verbose=False,
        design_flat_indices=design_flat_idx,
    )
    log(f"Residual: {res:.4e}")
    log(f"||E||={tt_norm_gpu(E_cores):.4e}")

    # === TEST 3: Extract E at design voxels ===
    log("\n=== TEST 3: Field extraction ===")
    E_design = tt_evaluate_at_indices_gpu(E_cores, design_flat_idx, n_bits)
    log(f"|E_design| range: [{E_design.abs().min():.4e}, {E_design.abs().max():.4e}]")
    log(f"|E_design| mean: {E_design.abs().mean():.4e}")

    # === TEST 4: Manual gradient (skip adjoint, just explicit term) ===
    log("\n=== TEST 4: Manual gradient components ===")
    rho_filt = density_filter_gpu(density, 0)
    rho_proj = heaviside_projection_gpu(rho_filt, 1.0, 0.5)
    sigma_vals = simp_sigma_gpu(rho_proj, config.sigma_min, 30.0, 1.0)
    dsigma = simp_dsigma_drho_gpu(rho_proj, config.sigma_min, 30.0, 1.0)
    log(f"σ range: [{sigma_vals.min():.2f}, {sigma_vals.max():.2f}]")
    log(f"dσ/dρ range: [{dsigma.min():.2f}, {dsigma.max():.2f}]")

    # P_cond using per-voxel sigma_tt
    P_cond = 0.5 * k0_norm**2 * dv * tt_inner_gpu(
        E_cores,
        tt_matvec_gpu(diag_mpo_from_tt_gpu(sigma_fwd), E_cores, max_rank=config.max_rank),
    ).real.item()
    log(f"P_cond (per-voxel σ): {P_cond:.4e}")

    # P_input
    P_input = abs(0.5 * dv * tt_inner_gpu(source_cores, E_cores).real.item())
    log(f"P_input: {P_input:.4e}")

    # P_pml
    P_pml = compute_pml_power_tt_gpu(E_cores, sigma_pml_tt, dv, config.max_rank)
    log(f"P_pml: {P_pml:.4e}")

    # W_near
    W_near = 0.5 * dv * tt_inner_gpu(E_cores, E_cores).real.item()
    Q_proxy = k0_norm * W_near / (P_input + 1e-30)
    log(f"W_near: {W_near:.4e}, Q_proxy: {Q_proxy:.2f}")

    # Explicit gradient term (no adjoint needed): dP_cond/dρ
    E_design_sq = E_design.abs().pow(2).real.to(torch.float64)
    dJ_explicit = 0.5 * k0_norm**2 * dsigma.to(torch.float64) * E_design_sq * dv
    log(f"|dJ_explicit| range: [{dJ_explicit.abs().min():.4e}, {dJ_explicit.abs().max():.4e}]")
    log(f"|dJ_explicit| variation (std/mean): {dJ_explicit.std()/dJ_explicit.abs().mean():.2f}")

    # === TEST 5: Full adjoint gradient ===
    log("\n=== TEST 5: Full adjoint gradient ===")
    _, eta_vol = volume_preserving_projection_gpu(rho_filt, 1.0, config.vol_target)
    log(f"eta_vol={eta_vol:.4f}")

    J_val, grad, metrics, residual = compute_adjoint_gradient_gpu(
        density=density, design_mask_tt=design_mask_tt,
        sigma_pml_tt=sigma_pml_tt, design_flat_indices=design_flat_idx,
        n_bits=n_bits, k0_norm=k0_norm, domain_size=config.domain_size,
        pml_cells=config.pml_cells, pml_sigma_max=config.pml_sigma_max,
        sigma_min=config.sigma_min, sigma_max=30.0, simp_p=1.0,
        beta=1.0, eta=eta_vol, filter_radius=0, damping=0.02,
        max_rank=config.max_rank, n_sweeps=config.n_sweeps,
        solver_tol=config.solver_tol, source_width=config.source_width,
        alpha_loss=0.0, use_log=True, device=device, dtype=dtype,
        verbose=False,
    )
    log(f"\nJ={J_val:.6f}")
    log(f"gmax={grad.abs().max():.4e}")
    log(f"|grad|={grad.norm():.4e}")
    log(f"grad std={grad.std():.4e}")

    gsorted = grad.abs().sort()[0]
    g10 = gsorted[int(0.1*len(gsorted))].item()
    g50 = gsorted[int(0.5*len(gsorted))].item()
    g90 = gsorted[int(0.9*len(gsorted))].item()
    log(f"Grad: p10={g10:.4e}, p50={g50:.4e}, p90={g90:.4e}")
    log(f"Spatial ratio p90/p10: {g90/(g10+1e-30):.1f}")

    gpu_mb = torch.cuda.memory_allocated() / 1024**2
    log(f"GPU: {gpu_mb:.0f} MB")
    log("\n=== ALL TESTS PASSED ===")

except Exception as e:
    log(f"\n!!! ERROR: {e}")
    traceback.print_exc(file=sys.stdout)
    sys.exit(1)
