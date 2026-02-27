#!/usr/bin/env python3
"""Minimal test: just one forward solve + gradient with per-voxel σ at 128³.

Run with: PYTHONUNBUFFERED=1 python3 test_pervoxel_mini.py
"""
import sys
import os
import traceback

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)  # line-buffered

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    print(f"CUDA: {torch.cuda.is_available()}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    from tensornet.em.chu_limit_gpu import (
        make_chu_gpu_config,
        build_sphere_mask_tt_gpu,
        spherical_mask_flat_indices_gpu,
        build_pml_sigma_tt_gpu,
        build_conductivity_eps_tt_gpu,
        solve_forward_gpu,
        compute_adjoint_gradient_gpu,
        density_filter_gpu,
        volume_preserving_projection_gpu,
        heaviside_projection_gpu,
        simp_sigma_gpu,
        compute_pml_power_tt_gpu,
    )
    from tensornet.em.qtt_helmholtz_gpu import tt_inner_gpu, tt_norm_gpu
    from tensornet.em.boundaries import PMLConfig

    print("Imports OK", flush=True)

    device = torch.device("cuda")
    dtype = torch.complex64
    n_bits = 7  # 128³
    total_sites = 3 * n_bits
    config = make_chu_gpu_config("128")

    # Physical
    C0 = 299_792_458.0
    wavelength = C0 / config.frequency_hz
    k0 = 2.0 * 3.14159265 / wavelength
    domain_m = config.domain_wavelengths * wavelength
    k0_norm = k0 * domain_m
    h = 1.0 / (2 ** n_bits)
    dv = h ** 3

    print(f"k0_norm={k0_norm:.4f}, domain={domain_m*1e3:.1f}mm", flush=True)
    print(f"total_sites={total_sites}, per-voxel threshold=24", flush=True)
    print(f"Per-voxel enabled: {total_sites <= 24}", flush=True)

    # Sphere mask
    r_norm = config.ka / k0_norm
    print(f"r_norm={r_norm:.4f}", flush=True)
    design_flat_idx = spherical_mask_flat_indices_gpu(n_bits, radius=r_norm, device=device)
    n_design = design_flat_idx.shape[0]
    print(f"Design voxels: {n_design}", flush=True)

    design_mask_tt = build_sphere_mask_tt_gpu(n_bits, radius=r_norm, device=device, dtype=dtype)
    print("Mask TT built", flush=True)

    # PML sigma
    pml = PMLConfig(n_cells=config.pml_cells, sigma_max=config.pml_sigma_max)
    sigma_pml_tt = build_pml_sigma_tt_gpu(n_bits, k0_norm, pml, device=device, dtype=dtype)
    print("PML sigma TT built", flush=True)

    # Density initialization
    density = 0.3 * torch.ones(n_design, device=device, dtype=torch.float64)
    noise = 0.10 * (2.0 * torch.rand(n_design, device=device, dtype=torch.float64) - 1.0)
    density = (density + noise).clamp(0.01, 0.99)
    print(f"Density: mean={density.mean().item():.3f}, std={density.std().item():.3f}", flush=True)

    # Test build_conductivity_eps_tt_gpu with per-voxel
    print("\n--- Testing per-voxel ε construction ---", flush=True)
    eps_tt, sigma_tt = build_conductivity_eps_tt_gpu(
        density, design_mask_tt, config.sigma_min, config.sigma_max_init,
        config.simp_p_init, 1.0, 0.5, 0, n_bits, device, dtype,
        config.max_rank, design_flat_indices=design_flat_idx,
    )
    eps_norm = tt_norm_gpu(eps_tt)
    sigma_norm = tt_norm_gpu(sigma_tt)
    print(f"||eps_tt||={eps_norm:.4f}", flush=True)
    print(f"||sigma_tt||={sigma_norm:.4f}", flush=True)
    print(f"eps_tt ranks: {[c.shape for c in eps_tt[:3]]}...{[c.shape for c in eps_tt[-2:]]}", flush=True)
    print(f"sigma_tt ranks: {[c.shape for c in sigma_tt[:3]]}...{[c.shape for c in sigma_tt[-2:]]}", flush=True)

    # Test without per-voxel (mean-field) for comparison
    print("\n--- Testing mean-field ε construction ---", flush=True)
    eps_mean, sigma_mean_tt = build_conductivity_eps_tt_gpu(
        density, design_mask_tt, config.sigma_min, config.sigma_max_init,
        config.simp_p_init, 1.0, 0.5, 0, n_bits, device, dtype,
        config.max_rank, design_flat_indices=None,  # force mean-field
    )
    eps_mean_norm = tt_norm_gpu(eps_mean)
    sigma_mean_norm = tt_norm_gpu(sigma_mean_tt)
    print(f"||eps_mean||={eps_mean_norm:.4f}", flush=True)
    print(f"||sigma_mean_tt||={sigma_mean_norm:.4f}", flush=True)

    # Forward solve with per-voxel
    print("\n--- Forward solve (per-voxel) ---", flush=True)
    H_cores, E_cores, source_cores, res, sigma_fwd_tt = solve_forward_gpu(
        density, design_mask_tt, n_bits, k0_norm,
        config.domain_size, config.pml_cells, config.pml_sigma_max,
        config.sigma_min, config.sigma_max_init, config.simp_p_init,
        1.0, 0.5, 0, config.damping, config.max_rank,
        config.n_sweeps, config.solver_tol, config.source_width,
        device, dtype, verbose=False,
        design_flat_indices=design_flat_idx,
    )
    print(f"Residual: {res:.4e}", flush=True)
    E_norm = tt_norm_gpu(E_cores)
    print(f"||E||={E_norm:.4e}", flush=True)

    # Full gradient computation
    print("\n--- Full gradient (per-voxel) ---", flush=True)
    rho_filt = density_filter_gpu(density, 0)
    _, eta_vol = volume_preserving_projection_gpu(rho_filt, 1.0, config.vol_target)

    J_val, grad, metrics, residual = compute_adjoint_gradient_gpu(
        density=density,
        design_mask_tt=design_mask_tt,
        sigma_pml_tt=sigma_pml_tt,
        design_flat_indices=design_flat_idx,
        n_bits=n_bits,
        k0_norm=k0_norm,
        domain_size=config.domain_size,
        pml_cells=config.pml_cells,
        pml_sigma_max=config.pml_sigma_max,
        sigma_min=config.sigma_min,
        sigma_max=config.sigma_max_init,
        simp_p=config.simp_p_init,
        beta=1.0,
        eta=eta_vol,
        filter_radius=0,
        damping=config.damping,
        max_rank=config.max_rank,
        n_sweeps=config.n_sweeps,
        solver_tol=config.solver_tol,
        source_width=config.source_width,
        alpha_loss=0.0,
        use_log=True,
        device=device,
        dtype=dtype,
        verbose=True,
    )
    print(f"\nJ={J_val:.6f}", flush=True)
    print(f"P_input={metrics.P_input:.4e}, P_pml={metrics.P_pml:.4e}, P_cond={metrics.P_cond:.4e}", flush=True)
    print(f"W_near={metrics.W_near:.4e}, Q_proxy={metrics.Q_proxy:.2f}, Q_rad={metrics.Q_rad:.2f}", flush=True)
    print(f"Volume={metrics.vol:.3f}", flush=True)
    print(f"|grad|={grad.norm().item():.4e}, gmax={grad.abs().max().item():.4e}", flush=True)
    print(f"grad mean={grad.mean().item():.4e}, std={grad.std().item():.4e}", flush=True)

    # Key metric: gradient should have SPATIAL variation (different voxels different grads)
    gsorted = grad.abs().sort()[0]
    g10 = gsorted[int(0.1*len(gsorted))].item()
    g50 = gsorted[int(0.5*len(gsorted))].item()
    g90 = gsorted[int(0.9*len(gsorted))].item()
    print(f"Grad percentiles: p10={g10:.4e}, p50={g50:.4e}, p90={g90:.4e}", flush=True)
    print(f"Grad spatial ratio (p90/p10): {g90/(g10+1e-30):.1f}", flush=True)

    gpu_mem = torch.cuda.memory_allocated() / 1024**2
    print(f"\nGPU memory: {gpu_mem:.0f} MB", flush=True)
    print("\n=== TEST PASSED ===", flush=True)

except Exception as e:
    print(f"\n!!! ERROR: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
