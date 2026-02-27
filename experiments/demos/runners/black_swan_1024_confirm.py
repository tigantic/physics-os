#!/usr/bin/env python3
"""
🦢 BLACK SWAN 1024³ CONFIRMATION
================================

Reproducing the Dec 28, 2025 Black Swan at 1024³ to confirm:
- Blowup time convergence (should be ~1.55 if physical)
- Resolution independence of the singularity

Original 512³ result:
- Steps: 945
- Blowup time: t = 1.5463
- Initial rank: 81
- Final rank: 403
- Seed: 42
"""

import time
import torch
import numpy as np
import sys
import os
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from millennium_hunter import (
    MillenniumSolver, EulerState3D, QTT3DState,
    build_rank1_3d_qtt
)


def create_random_turb_ic(solver: MillenniumSolver, seed: int = 42) -> EulerState3D:
    """
    Create random high-k turbulence IC matching original Black Swan.
    
    Uses sum of random Fourier modes with Kolmogorov spectrum.
    """
    print(f"Creating Random High-k Turbulence IC ({solver.N}³ = {solver.N**3:,} points)...")
    print(f"  Random seed: {seed}")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    t0 = time.perf_counter()
    
    x = torch.linspace(0, solver.L, solver.N, dtype=solver.dtype, device=solver.device)
    
    # Parameters matching original initial rank ~81
    # At 1024³, fewer modes needed to achieve same rank
    n_modes_per_component = 3  # Reduced for 1024³ to get initial rank ~80
    k_max = 4  # Lower wavenumber range
    
    def build_turbulent_field():
        """Build one velocity component as sum of random modes."""
        result = None
        
        for _ in range(n_modes_per_component):
            # Random wavenumbers
            kx = np.random.randint(1, k_max + 1)
            ky = np.random.randint(1, k_max + 1)
            kz = np.random.randint(1, k_max + 1)
            
            # Random phases
            phi_x = np.random.uniform(0, 2 * np.pi)
            phi_y = np.random.uniform(0, 2 * np.pi)
            phi_z = np.random.uniform(0, 2 * np.pi)
            
            # Kolmogorov amplitude scaling: E(k) ~ k^(-5/3)
            k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
            amplitude = 1.0 / (k_mag ** (5.0/6.0))  # sqrt of energy scaling
            
            # Build separable mode
            fx = torch.cos(kx * x + phi_x) * amplitude
            fy = torch.cos(ky * x + phi_y)
            fz = torch.cos(kz * x + phi_z)
            
            mode = build_rank1_3d_qtt(fx, fy, fz, solver.n_qubits, solver.max_rank)
            
            if result is None:
                result = mode
            else:
                result = solver._add(result, mode)
        
        # Truncate to control rank
        return solver._truncate(result, tol=1e-8)
    
    # Build each velocity component
    u = build_turbulent_field()
    v = build_turbulent_field()
    w = build_turbulent_field()
    
    t1 = time.perf_counter()
    print(f"  IC created in {t1-t0:.3f}s")
    print(f"  Initial ranks: u={u.max_rank}, v={v.max_rank}, w={w.max_rank}")
    
    return EulerState3D(u, v, w)


def run_1024_confirmation(max_time: float = 2.0, threshold: int = 400):
    """
    Run 1024³ confirmation of Black Swan.
    """
    print("=" * 70)
    print("🦢 BLACK SWAN 1024³ CONFIRMATION RUN")
    print("=" * 70)
    print()
    print("Original 512³ result:")
    print("  • Blowup time: t = 1.5463")
    print("  • Steps: 945")
    print("  • Initial rank: 81 → Final rank: 403")
    print()
    print("If blowup time CONVERGES at 1024³, singularity is PHYSICAL.")
    print("If blowup time SHIFTS later, it was numerical artifact.")
    print()
    
    # 1024³ = 10 qubits per dimension
    n_qubits = 10
    max_rank = 1024  # Higher cap for 1024³
    
    print("CONFIGURATION:")
    print(f"  • Grid: 1024³ = {1024**3:,} points")
    print(f"  • Max Rank: {max_rank}")
    print(f"  • Blowup Threshold: {threshold}")
    print()
    
    # Create solver
    solver = MillenniumSolver(
        qubits_per_dim=n_qubits,
        max_rank=max_rank,
        cfl=0.15  # Slightly lower CFL for stability at higher resolution
    )
    
    # Create random_turb IC with same seed
    state = create_random_turb_ic(solver, seed=42)
    initial_rank = state.max_rank()
    
    # Compute timestep
    dt = solver.compute_dt(state)
    print(f"  • Timestep: dt = {dt:.6f}")
    print()
    
    # Run simulation
    print("🎯 BEGINNING THE HUNT...")
    print("-" * 70)
    
    t = 0.0
    step = 0
    rank_history = []
    start_time = time.time()
    last_print = time.time()
    
    os.makedirs("logs", exist_ok=True)
    
    while t < max_time:
        rank = state.max_rank()
        rank_history.append({"t": t, "step": step, "rank": rank})
        
        # Print every 10 steps or every 60 seconds
        if step % 10 == 0 or (time.time() - last_print) > 60:
            elapsed = time.time() - start_time
            print(f"   Step {step:5d} | t={t:.4f} | Rank={rank:4d} | {elapsed:.1f}s")
            last_print = time.time()
        
        # Check for Black Swan
        if rank > threshold:
            elapsed = time.time() - start_time
            
            print()
            print("=" * 70)
            print("🚨 BLACK SWAN DETECTED AT 1024³! 🚨")
            print("=" * 70)
            print(f"   Condition Met: Rank {rank} > {threshold}")
            print(f"   Simulation Time: t = {t:.6f}")
            print(f"   Wall Time: {elapsed:.2f}s")
            print(f"   Steps Completed: {step}")
            print()
            
            # Compare to original
            print("COMPARISON TO 512³ ORIGINAL:")
            print(f"   512³ blowup time: t = 1.5463")
            print(f"   1024³ blowup time: t = {t:.4f}")
            print(f"   Difference: {abs(t - 1.5463):.4f} ({abs(t - 1.5463)/1.5463*100:.1f}%)")
            print()
            
            if abs(t - 1.5463) < 0.1:  # Within 6% 
                print("   ✅ BLOWUP TIME CONVERGED!")
                print("   This strongly suggests a PHYSICAL singularity!")
            else:
                print("   ⚠️ Blowup time shifted significantly")
                print("   Needs further investigation")
            
            print()
            
            # Save evidence
            evidence = {
                "event": "BLACK_SWAN_1024_CONFIRMATION",
                "grid": "1024³",
                "points": 1024**3,
                "t": t,
                "step": step,
                "rank": rank,
                "threshold": threshold,
                "initial_rank": initial_rank,
                "u_rank": state.u.max_rank,
                "v_rank": state.v.max_rank,
                "w_rank": state.w.max_rank,
                "original_512_t": 1.5463,
                "time_difference": abs(t - 1.5463),
                "time_difference_percent": abs(t - 1.5463) / 1.5463 * 100,
                "converged": abs(t - 1.5463) < 0.1,
                "wall_time_seconds": elapsed,
                "rank_history_last_20": rank_history[-20:],
                "timestamp_utc": datetime.now(timezone.utc).isoformat()
            }
            
            filename = f"logs/BLACK_SWAN_1024_confirmation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, "w") as f:
                json.dump(evidence, f, indent=2, default=str)
            print(f"   Evidence saved: {filename}")
            print("=" * 70)
            
            return evidence
        
        # Step forward
        state = solver.step(state, dt)
        t += dt
        step += 1
    
    # No blowup
    elapsed = time.time() - start_time
    print()
    print("=" * 70)
    print("HUNT COMPLETE - NO BLOWUP DETECTED")
    print("=" * 70)
    print(f"   Final time: t = {t:.4f}")
    print(f"   Final rank: {state.max_rank()}")
    print(f"   Wall time: {elapsed:.2f}s")
    print()
    print("   This could mean:")
    print("   1. Singularity was numerical artifact (resolved away at 1024³)")
    print("   2. Blowup time shifted beyond our simulation window")
    print("   3. Need to run longer")
    print("=" * 70)
    
    return {
        "event": "NO_BLOWUP",
        "grid": "1024³",
        "final_t": t,
        "final_rank": state.max_rank()
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="1024³ Black Swan Confirmation")
    parser.add_argument("--max-time", type=float, default=2.0)
    parser.add_argument("--threshold", type=int, default=400)
    args = parser.parse_args()
    
    result = run_1024_confirmation(max_time=args.max_time, threshold=args.threshold)
