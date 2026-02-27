#!/usr/bin/env python3
"""Test per-voxel σ: Run 1 optimization iteration at 256³ and check J moves."""
import sys, os, signal, traceback, time
signal.signal(signal.SIGINT, signal.SIG_IGN)
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def log(msg):
    print(msg, flush=True)

try:
    log("Starting 1-iteration 256³ optimization test...")
    import torch
    vram = torch.cuda.mem_get_info()[0] / 1024**2
    log(f"CUDA: {torch.cuda.is_available()}, VRAM free: {vram:.0f} MB")

    from tensornet.em.chu_limit_gpu import optimize_chu_antenna_gpu, make_chu_gpu_config
    log("Imports OK")

    device = torch.device("cuda")
    dtype = torch.complex64
    config = make_chu_gpu_config("256")
    config.max_iterations = 2  # Only 2 iterations for testing
    log(f"Config: n_bits={config.n_bits}, max_rank={config.max_rank}, domain={config.domain_wavelengths}λ")

    # Run just 2 iterations with verbose=True to see everything
    t0 = time.time()
    result = optimize_chu_antenna_gpu(
        config=config,
        verbose=True,
    )
    dt = time.time() - t0
    log(f"\nCompleted in {dt:.1f}s")
    log(f"Final J values: {[h['J'] for h in result.history]}")
    log(f"Final Q_proxy: {[h.get('Q_proxy', 'N/A') for h in result.history]}")
    log(f"Final vol: {[h.get('vol', 'N/A') for h in result.history]}")
    
    if len(result.history) >= 2:
        dJ = abs(result.history[-1]['J'] - result.history[0]['J'])
        log(f"ΔJ between iter 0 and 1: {dJ:.6f}")
        if dJ > 1e-6:
            log("J IS MOVING — per-voxel σ fix works!")
        else:
            log("WARNING: J still flat — further investigation needed")
    
    log("\n1-ITER TEST COMPLETE")

except Exception as e:
    log(f"\n!!! EXCEPTION: {type(e).__name__}: {e}")
    traceback.print_exc(file=sys.stdout)
    sys.exit(1)
