#!/usr/bin/env python3
"""Isolate crash: run forward solve with per-voxel σ and CUDA error checking."""
import sys, os, signal, traceback
signal.signal(signal.SIGINT, signal.SIG_IGN)
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def log(msg):
    print(msg, flush=True)

try:
    import torch
    torch.cuda.set_device(0)
    log(f"CUDA: {torch.cuda.is_available()}, blocking=1")

    from tensornet.em.chu_limit_gpu import (
        make_chu_gpu_config, build_sphere_mask_tt_gpu,
        spherical_mask_flat_indices_gpu, build_pml_sigma_tt_gpu,
        build_conductivity_eps_tt_gpu, solve_forward_gpu,
    )
    from tensornet.em.qtt_helmholtz_gpu import tt_norm_gpu
    from tensornet.em.boundaries import PMLConfig

    log("Imports OK")

    device = torch.device("cuda")
    dtype = torch.complex64
    n_bits = 7
    config = make_chu_gpu_config("128")
    C0 = 299_792_458.0
    wavelength = C0 / config.frequency_hz
    k0_norm = 2.0 * 3.14159265 / wavelength * config.domain_wavelengths * wavelength
    r_norm = config.ka / k0_norm

    design_flat_idx = spherical_mask_flat_indices_gpu(n_bits, radius=r_norm, device=device)
    design_mask_tt = build_sphere_mask_tt_gpu(n_bits, radius=r_norm, device=device, dtype=dtype)
    n_design = design_flat_idx.shape[0]
    log(f"Design voxels: {n_design}")

    pml = PMLConfig(n_cells=config.pml_cells, sigma_max=config.pml_sigma_max)
    sigma_pml_tt = build_pml_sigma_tt_gpu(n_bits, k0_norm, pml, device=device, dtype=dtype)

    density = 0.3 * torch.ones(n_design, device=device, dtype=torch.float64)
    noise = 0.10 * (2.0 * torch.rand(n_design, device=device, dtype=torch.float64) - 1.0)
    density = (density + noise).clamp(0.01, 0.99)
    log(f"Density: mean={density.mean():.3f}")

    # Build per-voxel eps
    log("Building per-voxel eps...")
    torch.cuda.synchronize()
    eps_tt, sigma_tt = build_conductivity_eps_tt_gpu(
        density, design_mask_tt, config.sigma_min, 30.0, 1.0,
        1.0, 0.5, 0, n_bits, device, dtype, config.max_rank,
        design_flat_indices=design_flat_idx,
    )
    torch.cuda.synchronize()
    log(f"eps_tt rank={max(c.shape[-1] for c in eps_tt)}, sigma_tt rank={max(c.shape[-1] for c in sigma_tt)}")
    log(f"GPU after eps: {torch.cuda.memory_allocated()/1024**2:.0f} MB")

    # Forward solve
    log("Forward solve...")
    torch.cuda.synchronize()
    H_cores, E_cores, source_cores, res, sigma_fwd = solve_forward_gpu(
        density, design_mask_tt, n_bits, k0_norm,
        config.domain_size, config.pml_cells, config.pml_sigma_max,
        config.sigma_min, 30.0, 1.0,
        1.0, 0.5, 0, 0.02, config.max_rank, config.n_sweeps,
        config.solver_tol, config.source_width,
        device, dtype, verbose=False,
        design_flat_indices=design_flat_idx,
    )
    torch.cuda.synchronize()
    log(f"Residual: {res:.4e}")
    log(f"GPU after solve: {torch.cuda.memory_allocated()/1024**2:.0f} MB")
    log("FORWARD SOLVE OK")

except Exception as e:
    log(f"\n!!! EXCEPTION: {type(e).__name__}: {e}")
    traceback.print_exc(file=sys.stdout)
    sys.exit(1)
except SystemExit:
    pass
finally:
    log("Script exiting")
