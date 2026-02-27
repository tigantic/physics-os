#!/usr/bin/env python3
"""
Figure 1: QTT Rank vs Reynolds Number
=====================================

THESIS: QTT operational rank saturates at O(log N) regardless of Reynolds number.
If true → O(N log N) turbulence simulation, not O(N^3).

METHODOLOGY:
- max_rank = 2048 (HIGH - never the binding constraint)
- tol_svd = 1e-6 (THIS controls rank via relative SVD truncation)
- Measure: physics-determined rank = what SVD truncation settles to

The rank we observe is the rank the physics NEEDS, not the cap.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from math import pi
from pathlib import Path
from typing import Any

import torch

# Import the ACTUAL working API
from tensornet.cfd.ns3d_qtt_native import (
    NS3DConfig,
    NS3DQTTSolver,
    taylor_green_3d,
)


@dataclass
class RankMeasurement:
    """Single measurement of rank at a given Re and time."""
    reynolds: float
    time: float
    step: int
    # Velocity field rank
    u_max_rank: int
    u_mean_rank: float
    u_ranks_per_component: list[int]  # [u_x, u_y, u_z]
    # Vorticity field rank
    omega_max_rank: int
    omega_mean_rank: float
    omega_ranks_per_component: list[int]  # [omega_x, omega_y, omega_z]
    # Physics
    kinetic_energy: float


def get_bond_dimensions(qtt_vector_field) -> tuple[int, float, list[int]]:
    """
    Extract bond dimensions from a QTT3DVectorField.
    
    Returns:
        (max_rank, mean_rank, per_component_max_ranks)
    """
    max_rank = qtt_vector_field.max_rank
    mean_rank = qtt_vector_field.mean_rank
    
    # Get per-component ranks
    per_component = []
    for field in [qtt_vector_field.x, qtt_vector_field.y, qtt_vector_field.z]:
        # QTT3DState.cores is list of cores with shape (r_left, d, r_right)
        component_max = max(c.shape[0] for c in field.cores)
        per_component.append(component_max)
    
    return max_rank, mean_rank, per_component


def run_single_re(
    reynolds: float,
    n_bits: int = 6,  # 64³ grid
    n_steps: int = 200,
    sample_interval: int = 20,
    max_rank: int = 2048,  # HIGH - not the constraint
    tol_svd: float = 1e-6,  # THIS controls rank
    device: str = "cuda",
) -> list[RankMeasurement]:
    """
    Run Taylor-Green vortex at given Reynolds number, measure rank evolution.
    """
    print(f"\n{'='*60}")
    print(f"Re = {reynolds:.0f}")
    print(f"{'='*60}")
    
    # Viscosity from Reynolds: nu = U*L / Re
    # For TG vortex with U=1, L=2π: nu = 2π/Re
    nu = 2 * pi / reynolds
    
    # Time step scales with viscosity for stability
    # CFL-like condition: dt ~ dx² / nu for diffusion
    # With 64³ grid (dx ~ 2π/64), use conservative timestep
    dt = min(0.01, 0.5 * (2 * pi / (2 ** n_bits)) ** 2 / nu)
    dt = max(dt, 1e-4)  # But not too small
    
    # For high Re, viscosity is small, can use larger dt
    # CFL for convection: dt < dx / U_max ~ dx (since U_max ~ 1)
    dx = 2 * pi / (2 ** n_bits)
    dt = min(dt, 0.1 * dx)  # CFL safety
    
    print(f"  nu = {nu:.6f}, dt = {dt:.6f}")
    print(f"  max_rank = {max_rank}, tol_svd = {tol_svd:.1e}")
    
    config = NS3DConfig(
        n_bits=n_bits,
        nu=nu,
        dt=dt,
        L=2 * pi,           # Domain size [0, 2π]³
        max_rank=max_rank,  # HIGH - won't be the constraint
        tol_svd=tol_svd,    # THIS determines actual rank
        device=device,
    )
    
    solver = NS3DQTTSolver(config)
    
    # Create IC - tolerance controls rank during compression
    u_init, omega_init = taylor_green_3d(config)
    
    solver.initialize(u_init, omega_init)
    
    measurements = []
    
    # Initial measurement (step 0)
    u_max, u_mean, u_per_comp = get_bond_dimensions(solver.u)
    omega_max, omega_mean, omega_per_comp = get_bond_dimensions(solver.omega)
    
    # Get initial KE from diagnostics history
    ke = solver.diagnostics_history[-1].kinetic_energy if solver.diagnostics_history else 0.0
    
    meas = RankMeasurement(
        reynolds=reynolds,
        time=solver.t,
        step=0,
        u_max_rank=u_max,
        u_mean_rank=u_mean,
        u_ranks_per_component=u_per_comp,
        omega_max_rank=omega_max,
        omega_mean_rank=omega_mean,
        omega_ranks_per_component=omega_per_comp,
        kinetic_energy=ke,
    )
    measurements.append(meas)
    
    print(f"  step {0:4d}, t={solver.t:.4f}: "
          f"u_rank={u_max:3d} (mean {u_mean:.1f}), "
          f"ω_rank={omega_max:3d}, "
          f"KE={ke:.4f}")
    
    for step in range(1, n_steps + 1):
        # step() returns diagnostics
        diag = solver.step()
        
        if step % sample_interval == 0:
            # Measure current state
            u_max, u_mean, u_per_comp = get_bond_dimensions(solver.u)
            omega_max, omega_mean, omega_per_comp = get_bond_dimensions(solver.omega)
            
            # Get KE from returned diagnostics
            ke = diag.kinetic_energy
            
            meas = RankMeasurement(
                reynolds=reynolds,
                time=solver.t,
                step=step,
                u_max_rank=u_max,
                u_mean_rank=u_mean,
                u_ranks_per_component=u_per_comp,
                omega_max_rank=omega_max,
                omega_mean_rank=omega_mean,
                omega_ranks_per_component=omega_per_comp,
                kinetic_energy=ke,
            )
            measurements.append(meas)
            
            print(f"  step {step:4d}, t={solver.t:.4f}: "
                  f"u_rank={u_max:3d} (mean {u_mean:.1f}), "
                  f"ω_rank={omega_max:3d}, "
                  f"KE={ke:.4f}")
    
    return measurements


def main():
    """Run Figure 1 experiment: Rank vs Reynolds number."""
    
    # Check CUDA
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = "cpu"
        print("WARNING: Running on CPU (slow)")
    
    # Experiment parameters
    reynolds_numbers = [200, 400, 800, 1600]  # Span ~order of magnitude
    n_bits = 6  # 64³ grid
    n_steps = 200
    sample_interval = 20
    
    # CORRECT: High ceiling (never constrains), tolerance controls rank
    # Adaptive SVD sizing gives O(actual_rank²) cost
    max_rank = 2048
    tol_svd = 1e-6
    
    print("\n" + "=" * 70)
    print("FIGURE 1: QTT RANK VS REYNOLDS NUMBER")
    print("=" * 70)
    print(f"Grid: {2**n_bits}³ = {(2**n_bits)**3:,} points")
    print(f"Reynolds numbers: {reynolds_numbers}")
    print(f"max_rank = {max_rank} (high ceiling, never constrains)")
    print(f"tol_svd = {tol_svd:.1e} (controls physics-determined rank)")
    print("Adaptive SVD sizing: O(actual_rank²) cost, not O(max_rank²)")
    print("=" * 70)
    
    all_results: dict[float, list[dict[str, Any]]] = {}
    
    for Re in reynolds_numbers:
        try:
            measurements = run_single_re(
                reynolds=Re,
                n_bits=n_bits,
                n_steps=n_steps,
                sample_interval=sample_interval,
                max_rank=max_rank,
                tol_svd=tol_svd,
                device=device,
            )
            
            # Convert to dicts for JSON
            all_results[Re] = [
                {
                    "reynolds": m.reynolds,
                    "time": m.time,
                    "step": m.step,
                    "u_max_rank": m.u_max_rank,
                    "u_mean_rank": m.u_mean_rank,
                    "u_ranks_per_component": m.u_ranks_per_component,
                    "omega_max_rank": m.omega_max_rank,
                    "omega_mean_rank": m.omega_mean_rank,
                    "omega_ranks_per_component": m.omega_ranks_per_component,
                    "kinetic_energy": m.kinetic_energy,
                }
                for m in measurements
            ]
            
        except Exception as e:
            print(f"  ERROR at Re={Re}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: FINAL STATE RANKS")
    print("=" * 70)
    print(f"{'Re':>6} | {'u_max':>6} | {'u_mean':>7} | {'ω_max':>6} | {'ω_mean':>7} | {'KE':>8}")
    print("-" * 70)
    
    final_ranks = []
    for Re in reynolds_numbers:
        if Re in all_results and all_results[Re]:
            final = all_results[Re][-1]
            print(f"{Re:6.0f} | {final['u_max_rank']:6d} | "
                  f"{final['u_mean_rank']:7.1f} | "
                  f"{final['omega_max_rank']:6d} | "
                  f"{final['omega_mean_rank']:7.1f} | "
                  f"{final['kinetic_energy']:8.4f}")
            final_ranks.append({
                "Re": Re,
                "u_max_rank": final["u_max_rank"],
                "omega_max_rank": final["omega_max_rank"],
            })
    
    print("-" * 70)
    
    # Check if cap was saturated
    any_saturated = any(
        r["u_max_rank"] >= max_rank * 0.95 or r["omega_max_rank"] >= max_rank * 0.95
        for r in final_ranks
    )
    if any_saturated:
        print(f"\n⚠ WARNING: Rank saturated at cap ({max_rank})")
        print("  → Need to re-run with higher max_rank to get true physics rank")
    else:
        print(f"\n✓ Rank stayed below cap ({max_rank}) - physics-determined")
    
    # Check thesis: ranks should NOT scale with Re
    if len(final_ranks) >= 2:
        re_ratio = reynolds_numbers[-1] / reynolds_numbers[0]
        rank_ratio = final_ranks[-1]["u_max_rank"] / max(final_ranks[0]["u_max_rank"], 1)
        
        print(f"\nRe increased by {re_ratio:.1f}×")
        print(f"Rank increased by {rank_ratio:.1f}×")
        
        if rank_ratio < 2.0:
            print("\n✓ THESIS SUPPORTED: Rank does NOT scale with Re")
            print("  → QTT complexity is O(N log N), not O(N³)")
        else:
            print("\n✗ THESIS NOT CLEARLY SUPPORTED")
            print("  → More investigation needed")
    
    # Save results
    output_file = Path("rank_vs_re_results.json")
    results_data = {
        "experiment": "Figure 1: QTT Rank vs Reynolds Number",
        "parameters": {
            "n_bits": n_bits,
            "grid_size": 2 ** n_bits,
            "n_steps": n_steps,
            "max_rank": max_rank,
            "tol_svd": tol_svd,
            "reynolds_numbers": reynolds_numbers,
        },
        "results": {str(k): v for k, v in all_results.items()},
        "summary": final_ranks,
    }
    
    with open(output_file, "w") as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
