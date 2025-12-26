"""
Sod Shock Tube Benchmark
========================

Validates the 1D Euler solver against the exact Riemann solution.

The Sod shock tube is a standard benchmark for compressible flow codes:
- Initial discontinuity at x = 0.5
- Left state: ρ = 1, u = 0, p = 1
- Right state: ρ = 0.125, u = 0, p = 0.1
- γ = 1.4

The exact solution at t = 0.2 contains:
1. Left-going rarefaction wave
2. Contact discontinuity (entropy wave)
3. Right-going shock wave

Reference: Sod, G.A. (1978) "A survey of several finite difference methods 
for systems of nonlinear hyperbolic conservation laws"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt

from tensornet.cfd import Euler1D, sod_shock_tube_ic, exact_riemann


def run_sod_benchmark(
    N: int = 200,
    t_final: float = 0.2,
    cfl: float = 0.5,
    plot: bool = True,
    save_path: str = None,
):
    """
    Run Sod shock tube and compare to exact solution.
    
    Args:
        N: Number of grid cells
        t_final: Final simulation time
        cfl: CFL number
        plot: Whether to generate plots
        save_path: Path to save figure (None = display)
        
    Returns:
        Dictionary with error metrics
    """
    print("=" * 60)
    print("Sod Shock Tube Benchmark")
    print("=" * 60)
    
    # Initialize solver
    solver = Euler1D(N=N, x_min=0.0, x_max=1.0, gamma=1.4, cfl=cfl)
    
    # Set initial condition
    ic = sod_shock_tube_ic(N, x_min=0.0, x_max=1.0)
    solver.set_initial_condition(ic)
    
    print(f"Grid: N = {N}, dx = {solver.dx:.4e}")
    print(f"CFL = {cfl}, t_final = {t_final}")
    
    # Solve
    print("\nRunning simulation...")
    snapshots = solver.solve(t_final)
    print(f"Completed in {len(snapshots)} time steps")
    
    # Get final state
    final_state = solver.state
    
    # Compute exact solution
    print("\nComputing exact solution...")
    x = solver.x_cell
    rho_exact, u_exact, p_exact = exact_riemann(
        rho_L=1.0, u_L=0.0, p_L=1.0,
        rho_R=0.125, u_R=0.0, p_R=0.1,
        gamma=1.4,
        x=x,
        t=t_final,
        x0=0.5,
    )
    
    # Compute errors
    rho_err = torch.abs(final_state.rho - rho_exact)
    u_err = torch.abs(final_state.u - u_exact)
    p_err = torch.abs(final_state.p - p_exact)
    
    L1_rho = rho_err.mean().item()
    L1_u = u_err.mean().item()
    L1_p = p_err.mean().item()
    
    Linf_rho = rho_err.max().item()
    Linf_u = u_err.max().item()
    Linf_p = p_err.max().item()
    
    print("\n" + "=" * 60)
    print("Error Analysis")
    print("=" * 60)
    print(f"{'Variable':<10} {'L1 Error':<15} {'Linf Error':<15}")
    print("-" * 40)
    print(f"{'Density':<10} {L1_rho:<15.4e} {Linf_rho:<15.4e}")
    print(f"{'Velocity':<10} {L1_u:<15.4e} {Linf_u:<15.4e}")
    print(f"{'Pressure':<10} {L1_p:<15.4e} {Linf_p:<15.4e}")
    
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        x_np = x.numpy()
        
        # Density
        ax = axes[0, 0]
        ax.plot(x_np, final_state.rho.numpy(), 'b-', linewidth=1.5, label='Numerical')
        ax.plot(x_np, rho_exact.numpy(), 'k--', linewidth=1.5, label='Exact')
        ax.set_xlabel('x')
        ax.set_ylabel('Density ρ')
        ax.set_title('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Velocity
        ax = axes[0, 1]
        ax.plot(x_np, final_state.u.numpy(), 'b-', linewidth=1.5, label='Numerical')
        ax.plot(x_np, u_exact.numpy(), 'k--', linewidth=1.5, label='Exact')
        ax.set_xlabel('x')
        ax.set_ylabel('Velocity u')
        ax.set_title('Velocity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Pressure
        ax = axes[1, 0]
        ax.plot(x_np, final_state.p.numpy(), 'b-', linewidth=1.5, label='Numerical')
        ax.plot(x_np, p_exact.numpy(), 'k--', linewidth=1.5, label='Exact')
        ax.set_xlabel('x')
        ax.set_ylabel('Pressure p')
        ax.set_title('Pressure')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Internal Energy
        ax = axes[1, 1]
        e_numerical = final_state.p / ((solver.gamma - 1) * final_state.rho)
        e_exact = p_exact / ((solver.gamma - 1) * rho_exact)
        ax.plot(x_np, e_numerical.numpy(), 'b-', linewidth=1.5, label='Numerical')
        ax.plot(x_np, e_exact.numpy(), 'k--', linewidth=1.5, label='Exact')
        ax.set_xlabel('x')
        ax.set_ylabel('Internal Energy e')
        ax.set_title('Internal Energy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Sod Shock Tube (N={N}, t={t_final})', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nFigure saved to: {save_path}")
        else:
            plt.show()
    
    return {
        'L1_rho': L1_rho,
        'L1_u': L1_u,
        'L1_p': L1_p,
        'Linf_rho': Linf_rho,
        'Linf_u': Linf_u,
        'Linf_p': Linf_p,
        'num_steps': len(snapshots),
    }


def convergence_study(
    N_values: list = [50, 100, 200, 400, 800],
    t_final: float = 0.2,
):
    """
    Perform grid convergence study.
    
    For first-order schemes, expect O(h^0.5) near discontinuities
    due to smearing, O(h) in smooth regions.
    """
    print("\n" + "=" * 60)
    print("Grid Convergence Study")
    print("=" * 60)
    
    results = []
    for N in N_values:
        print(f"\nN = {N}...")
        res = run_sod_benchmark(N=N, t_final=t_final, plot=False)
        res['N'] = N
        res['dx'] = 1.0 / N
        results.append(res)
    
    # Print convergence table
    print("\n" + "=" * 60)
    print("Convergence Results")
    print("=" * 60)
    print(f"{'N':<8} {'dx':<12} {'L1(ρ)':<12} {'L1(u)':<12} {'L1(p)':<12}")
    print("-" * 56)
    
    for r in results:
        print(f"{r['N']:<8} {r['dx']:<12.4e} {r['L1_rho']:<12.4e} {r['L1_u']:<12.4e} {r['L1_p']:<12.4e}")
    
    # Compute convergence rates
    print("\nConvergence Rates (log(error_{coarse}/error_{fine})/log(2)):")
    for i in range(1, len(results)):
        rate_rho = (torch.log(torch.tensor(results[i-1]['L1_rho'] / results[i]['L1_rho'])) / 
                   torch.log(torch.tensor(2.0))).item()
        rate_u = (torch.log(torch.tensor(results[i-1]['L1_u'] / (results[i]['L1_u'] + 1e-15))) / 
                 torch.log(torch.tensor(2.0))).item()
        rate_p = (torch.log(torch.tensor(results[i-1]['L1_p'] / results[i]['L1_p'])) / 
                 torch.log(torch.tensor(2.0))).item()
        
        print(f"  N: {results[i-1]['N']} -> {results[i]['N']}: "
              f"ρ: {rate_rho:.2f}, u: {rate_u:.2f}, p: {rate_p:.2f}")
    
    return results


if __name__ == '__main__':
    # Run single benchmark
    results = run_sod_benchmark(N=200, t_final=0.2, plot=True)
    
    # Optionally run convergence study
    # convergence_results = convergence_study()
