#!/usr/bin/env python3
"""Test baseline air solve at 256³ — minimum viable forward solve."""
import sys, os, signal, traceback
signal.signal(signal.SIGINT, signal.SIG_IGN)
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def log(msg):
    print(msg, flush=True)

try:
    log("Starting...")
    import torch
    log(f"CUDA: {torch.cuda.is_available()}, VRAM free: {torch.cuda.mem_get_info()[0]/1024**2:.0f} MB")

    from ontic.em.chu_limit_gpu import (
        make_chu_gpu_config, build_sphere_mask_tt_gpu,
        spherical_mask_flat_indices_gpu, solve_forward_gpu,
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
    log(f"Mask TT built, rank={max(c.shape[-1] for c in design_mask_tt)}")

    # Baseline: density = 0
    density_air = torch.zeros(n_design, device=device, dtype=torch.float64)
    log("Starting baseline air solve...")

    result = solve_forward_gpu(
        density_air, design_mask_tt, n_bits, k0_norm,
        config.domain_size, config.pml_cells, config.pml_sigma_max,
        config.sigma_min, config.sigma_max_init, config.simp_p_init,
        1.0, 0.5, 0, config.damping, config.max_rank,
        config.n_sweeps, config.solver_tol, config.source_width,
        device, dtype, verbose=False,
    )
    log(f"solve_forward_gpu returned {len(result)} values")
    H_cores, E_cores, source_cores, residual, sigma_tt = result
    log(f"Baseline residual: {residual:.4e}")
    log(f"GPU after solve: {torch.cuda.memory_allocated()/1024**2:.0f} MB")
    log("BASELINE SOLVE OK")

except Exception as e:
    log(f"\n!!! EXCEPTION: {type(e).__name__}: {e}")
    traceback.print_exc(file=sys.stdout)
    sys.exit(1)
