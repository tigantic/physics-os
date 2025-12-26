"""
Kelvin-Helmholtz Instability Validation Demo

This script validates the 2D Euler solver with Strang splitting
using the Kelvin-Helmholtz instability test case.

Expected behavior:
1. Initial state: Two shear layers with sinusoidal perturbation
2. Evolution: Perturbation grows, vortices roll up
3. Late time: Complex vortex sheet structures

Key metrics to watch:
- Rank dynamics: Starts low (~10), spikes during roll-up (~64-128)
- Conservation: Mass, momentum, energy should be preserved
- Vortex formation: Visual confirmation of roll-up

Author: HyperTensor Team
Date: December 2025
"""

import torch
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tensornet.cfd.euler2d_strang import (
    Euler2D_Strang, Euler2DConfig, Euler2DState, 
    create_kelvin_helmholtz_ic
)
from tensornet.cfd.kelvin_helmholtz import build_kh_via_dense, KHConfig, analyze_kh_ranks
from tensornet.cfd.qtt_2d import qtt_2d_to_dense


def run_kh_validation(n_bits: int = 6, max_steps: int = 100, t_final: float = 2.0):
    """
    Run Kelvin-Helmholtz instability validation.
    
    Args:
        n_bits: Bits per dimension (grid is 2^n x 2^n)
        max_steps: Maximum number of time steps
        t_final: Final simulation time
    """
    N = 2 ** n_bits
    print("=" * 70)
    print(f"Kelvin-Helmholtz Instability Validation")
    print(f"Grid: {N}x{N}, T_final={t_final}, max_steps={max_steps}")
    print("=" * 70)
    
    # Configuration
    config = Euler2DConfig(
        gamma=1.4,
        cfl=0.3,
        max_rank=64,
        dtype=torch.float64,
        device=torch.device('cpu')
    )
    
    # Create solver (nx = ny = n_bits for square grid)
    solver = Euler2D_Strang(n_bits, n_bits, config)
    
    # Create initial conditions
    print("\nGenerating initial conditions...")
    t0 = time.time()
    state = create_kelvin_helmholtz_ic(n_bits, n_bits, config)
    t_ic = time.time() - t0
    print(f"  IC generation time: {t_ic:.3f}s")
    
    # Initial diagnostics
    print("\nInitial state diagnostics:")
    print_diagnostics(state, solver)
    
    # Storage for history
    history = {
        't': [0.0],
        'mass': [],
        'E_total': [],
        'max_rank': []
    }
    
    # Record initial values
    mass0 = compute_mass(state)
    E0 = compute_total_energy(state)
    history['mass'].append(mass0)
    history['E_total'].append(E0)
    history['max_rank'].append(max_rank(state))
    
    print(f"\n  Initial mass: {mass0:.6f}")
    print(f"  Initial energy: {E0:.6f}")
    print(f"  Initial max rank: {max_rank(state)}")
    
    # Time evolution
    print("\n" + "-" * 70)
    print("Time Evolution")
    print("-" * 70)
    print(f"{'Step':>6}  {'Time':>10}  {'dt':>10}  {'MaxRank':>8}  {'dMass':>12}  {'dE':>12}")
    print("-" * 70)
    
    t = 0.0
    step = 0
    t_start = time.time()
    
    while t < t_final and step < max_steps:
        # Compute adaptive dt
        dt = solver.compute_dt(state)
        
        # Don't overshoot t_final
        if t + dt > t_final:
            dt = t_final - t
        
        # Take a step
        state = solver.step(state, dt)
        t += dt
        step += 1
        
        # Record history
        history['t'].append(t)
        mass = compute_mass(state)
        E = compute_total_energy(state)
        rank = max_rank(state)
        
        history['mass'].append(mass)
        history['E_total'].append(E)
        history['max_rank'].append(rank)
        
        # Print progress every 10 steps
        if step % 10 == 0 or t >= t_final:
            dMass = (mass - mass0) / mass0
            dE = (E - E0) / E0
            print(f"{step:6d}  {t:10.4f}  {dt:10.6f}  {rank:8d}  {dMass:12.2e}  {dE:12.2e}")
    
    t_elapsed = time.time() - t_start
    
    # Final diagnostics
    print("-" * 70)
    print("\nFinal state diagnostics:")
    print_diagnostics(state, solver)
    
    # Conservation check
    print("\n" + "=" * 70)
    print("Conservation Check")
    print("=" * 70)
    
    mass_final = history['mass'][-1]
    E_final = history['E_total'][-1]
    
    dMass = abs(mass_final - mass0) / mass0
    dE = abs(E_final - E0) / E0
    
    print(f"  Mass: {mass0:.6f} -> {mass_final:.6f} (relative error: {dMass:.2e})")
    print(f"  Energy: {E0:.6f} -> {E_final:.6f} (relative error: {dE:.2e})")
    
    # Check conservation (should be < 1% for a good solver)
    mass_ok = dMass < 0.01
    energy_ok = dE < 0.01
    
    print(f"\n  Mass conservation: {'PASS' if mass_ok else 'FAIL'} (threshold: 1%)")
    print(f"  Energy conservation: {'PASS' if energy_ok else 'FAIL'} (threshold: 1%)")
    
    # Performance summary
    print("\n" + "=" * 70)
    print("Performance Summary")
    print("=" * 70)
    print(f"  Grid size: {N}x{N} = {N*N} cells")
    print(f"  Total steps: {step}")
    print(f"  Wall time: {t_elapsed:.2f}s")
    print(f"  Time per step: {1000*t_elapsed/step:.2f}ms")
    print(f"  Initial max rank: {history['max_rank'][0]}")
    print(f"  Peak max rank: {max(history['max_rank'])}")
    print(f"  Final max rank: {history['max_rank'][-1]}")
    
    # Rank dynamics interpretation
    print("\n" + "=" * 70)
    print("Rank Dynamics Analysis")
    print("=" * 70)
    print(f"  Rank started at {history['max_rank'][0]} (smooth IC)")
    print(f"  Rank peaked at {max(history['max_rank'])} (vortex formation)")
    print(f"  Rank ended at {history['max_rank'][-1]} (complex flow)")
    
    if max(history['max_rank']) > history['max_rank'][0]:
        print("\n  ✓ Rank increased during evolution (expected for vortex formation)")
    else:
        print("\n  ⚠ Rank did not increase significantly (may need longer evolution)")
    
    # Overall verdict
    print("\n" + "=" * 70)
    all_pass = mass_ok and energy_ok
    if all_pass:
        print("VALIDATION: PASS")
    else:
        print("VALIDATION: FAIL")
    print("=" * 70)
    
    return state, history


def compute_mass(state: Euler2DState) -> float:
    """Compute total mass (integral of rho)."""
    rho = qtt_2d_to_dense(state.rho)
    return float(rho.sum())


def compute_total_energy(state: Euler2DState) -> float:
    """Compute total energy (integral of E)."""
    E = qtt_2d_to_dense(state.E)
    return float(E.sum())


def max_rank(state: Euler2DState) -> int:
    """Get maximum rank across all fields."""
    ranks = []
    for field in [state.rho, state.rhou, state.rhov, state.E]:
        for c in field.cores:
            ranks.append(c.shape[0])  # left bond
            ranks.append(c.shape[-1])  # right bond
    return max(ranks)


def print_diagnostics(state: Euler2DState, solver: Euler2D_Strang):
    """Print state diagnostics."""
    def field_info(qtt, name):
        ranks = [c.shape[0] for c in qtt.cores]
        dense = qtt_2d_to_dense(qtt)
        return {
            'name': name,
            'max_rank': max(ranks),
            'min': float(dense.min()),
            'max': float(dense.max()),
            'mean': float(dense.mean())
        }
    
    print(f"  {'Field':>8}  {'MaxRank':>8}  {'Min':>12}  {'Max':>12}  {'Mean':>12}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}")
    
    for field, name in [(state.rho, 'rho'), 
                        (state.rhou, 'rho*u'), 
                        (state.rhov, 'rho*v'), 
                        (state.E, 'E')]:
        info = field_info(field, name)
        print(f"  {info['name']:>8}  {info['max_rank']:>8}  {info['min']:>12.4f}  "
              f"{info['max']:>12.4f}  {info['mean']:>12.4f}")


def run_scaling_test():
    """
    Test how the solver scales with grid size.
    """
    print("=" * 70)
    print("Scaling Test: Time per step vs Grid Size")
    print("=" * 70)
    
    results = []
    
    for n_bits in [5, 6, 7]:
        N = 2 ** n_bits
        print(f"\nGrid {N}x{N}...")
        
        config = Euler2DConfig(
            gamma=1.4,
            cfl=0.3,
            max_rank=32,
            dtype=torch.float64
        )
        
        solver = Euler2D_Strang(n_bits, n_bits, config)
        state = create_kelvin_helmholtz_ic(n_bits, n_bits, config)
        
        # Time 10 steps
        n_steps = 10
        t0 = time.time()
        for _ in range(n_steps):
            dt = solver.compute_dt(state)
            state = solver.step(state, dt)
        t_elapsed = time.time() - t0
        
        ms_per_step = 1000 * t_elapsed / n_steps
        results.append((N, ms_per_step))
        print(f"  {ms_per_step:.1f} ms/step")
    
    print("\n" + "=" * 70)
    print("Scaling Summary")
    print("=" * 70)
    print(f"{'Grid':>10}  {'ms/step':>10}  {'Scaling':>10}")
    print("-" * 35)
    
    for i, (N, ms) in enumerate(results):
        if i > 0:
            prev_N, prev_ms = results[i-1]
            ratio = (N / prev_N) ** 2  # Expected for O(N^2) dense ops
            actual = ms / prev_ms
            scaling = f"×{actual:.1f} (expect ×{ratio:.0f})"
        else:
            scaling = "baseline"
        print(f"{N:>10}  {ms:>10.1f}  {scaling:>10}")
    
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Kelvin-Helmholtz Validation")
    parser.add_argument('--n', type=int, default=6, help='Bits per dimension (grid=2^n)')
    parser.add_argument('--steps', type=int, default=50, help='Max time steps')
    parser.add_argument('--time', type=float, default=1.0, help='Final time')
    parser.add_argument('--scaling', action='store_true', help='Run scaling test')
    
    args = parser.parse_args()
    
    if args.scaling:
        run_scaling_test()
    else:
        run_kh_validation(args.n, args.steps, args.time)
