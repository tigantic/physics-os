#!/usr/bin/env python3
"""Test per-voxel σ forward solve + adjoint gradient at 256³."""
import sys, os, signal, traceback, time
signal.signal(signal.SIGINT, signal.SIG_IGN)
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def log(msg):
    print(msg, flush=True)

try:
    log("Starting per-voxel 256³ test...")
    import torch
    vram = torch.cuda.mem_get_info()[0] / 1024**2
    log(f"CUDA: {torch.cuda.is_available()}, VRAM free: {vram:.0f} MB")

    from tensornet.em.chu_limit_gpu import (
        make_chu_gpu_config, build_sphere_mask_tt_gpu,
        spherical_mask_flat_indices_gpu, solve_forward_gpu,
        compute_adjoint_gradient_gpu, build_pml_sigma_tt_gpu,
        density_filter_gpu, heaviside_projection_gpu,
        volume_preserving_projection_gpu,
    )
    log("Imports OK")

    device = torch.device("cuda")
    dtype = torch.complex64
    config = make_chu_gpu_config("256")
    n_bits = config.n_bits
    C0 = 299_792_458.0
    wavelength = C0 / config.frequency_hz
    k0_norm = 2.0 * 3.14159265 / wavelength * config.domain_wavelengths * wavelength
    r_norm = config.ka / k0_norm
    log(f"n_bits={n_bits}, k0_norm={k0_norm:.4f}, r_norm={r_norm:.4f}")

    design_flat_idx = spherical_mask_flat_indices_gpu(n_bits, radius=r_norm, device=device)
    n_design = design_flat_idx.shape[0]
    log(f"Design voxels: {n_design}")

    design_mask_tt = build_sphere_mask_tt_gpu(n_bits, radius=r_norm, device=device, dtype=dtype)
    log(f"Mask TT rank={max(c.shape[-1] for c in design_mask_tt)}")

    sigma_pml_tt = build_pml_sigma_tt_gpu(
        n_bits, config.pml_cells, config.pml_sigma_max, device, dtype
    )
    log(f"PML TT rank={max(c.shape[-1] for c in sigma_pml_tt)}")

    # Non-zero density — uniform 0.5 to test per-voxel path
    density = torch.full((n_design,), 0.5, device=device, dtype=torch.float64)

    # Bisect eta to enforce volume = 0.3
    rho_filt = density_filter_gpu(density, config.filter_radius)
    rho_proj, eta_val = volume_preserving_projection_gpu(rho_filt, beta=1.0, vol_target=0.3)
    log(f"eta={eta_val:.4f}, vol={rho_proj.mean().item():.4f}")

    # === Test 1: Per-voxel forward solve ===
    log("\n=== TEST 1: Per-voxel forward solve ===")
    t0 = time.time()
    H_cores, E_cores, source_cores, residual, sigma_tt = solve_forward_gpu(
        density, design_mask_tt, n_bits, k0_norm,
        config.domain_size, config.pml_cells, config.pml_sigma_max,
        config.sigma_min, config.sigma_max_init, config.simp_p_init,
        1.0, eta_val, config.filter_radius, config.damping, config.max_rank,
        config.n_sweeps, config.solver_tol, config.source_width,
        device, dtype, verbose=False,
        design_flat_indices=design_flat_idx,
    )
    dt = time.time() - t0
    log(f"Forward solve: res={residual:.4e}, time={dt:.1f}s")
    log(f"E_cores rank={max(c.shape[-1] for c in E_cores)}")
    log(f"sigma_tt rank={max(c.shape[-1] for c in sigma_tt)}")
    vram_after = torch.cuda.mem_get_info()[0] / 1024**2
    log(f"VRAM free after forward: {vram_after:.0f} MB")

    # === Test 2: Full adjoint gradient ===
    log("\n=== TEST 2: Adjoint gradient ===")
    torch.cuda.empty_cache()
    t0 = time.time()
    J_val, grad, metrics, res_fwd = compute_adjoint_gradient_gpu(
        density, design_mask_tt, sigma_pml_tt, design_flat_idx,
        n_bits, k0_norm, config.domain_size, config.pml_cells,
        config.pml_sigma_max, config.sigma_min, config.sigma_max_init,
        config.simp_p_init, 1.0, eta_val, config.filter_radius,
        config.damping, config.max_rank, config.n_sweeps,
        config.solver_tol, config.source_width,
        alpha_loss=0.01, use_log=True,
        device=device, dtype=dtype, verbose=True,
    )
    dt = time.time() - t0
    log(f"Gradient computed: J={J_val:.4f}, res={res_fwd:.4e}, time={dt:.1f}s")
    log(f"P_input={metrics.P_input:.4e}, P_pml={metrics.P_pml:.4e}, P_cond={metrics.P_cond:.4e}")
    log(f"Q_proxy={metrics.Q_proxy:.4f}, W_near={metrics.W_near:.4e}")
    log(f"grad shape={grad.shape}, |grad|_max={grad.abs().max().item():.4e}")
    log(f"grad mean={grad.mean().item():.4e}, std={grad.std().item():.4e}")

    # Spatial variation check
    g_abs = grad.abs()
    p90 = torch.quantile(g_abs.float(), 0.9).item()
    p10 = torch.quantile(g_abs.float(), 0.1).item()
    ratio = p90 / (p10 + 1e-30)
    log(f"p90/p10 ratio={ratio:.2f} (>2 means spatial variation)")
    log(f"grad nonzero: {(g_abs > 1e-15).sum().item()}/{n_design}")

    vram_final = torch.cuda.mem_get_info()[0] / 1024**2
    log(f"VRAM free final: {vram_final:.0f} MB")
    log("\nPER-VOXEL 256³ TEST PASSED")

except Exception as e:
    log(f"\n!!! EXCEPTION: {type(e).__name__}: {e}")
    traceback.print_exc(file=sys.stdout)
    sys.exit(1)
