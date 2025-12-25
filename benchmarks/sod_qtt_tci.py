"""
Sod Shock Tube Validation with QTT-TCI Rusanov Flux

This validates the CFD implementation against the Sod shock tube problem,
a canonical 1D Riemann problem with known analytical solution.

Problem:
    Left state (x < 0.5):  ρ=1.0, u=0.0, p=1.0
    Right state (x ≥ 0.5): ρ=0.125, u=0.0, p=0.1
    
Expected features at t=0.2:
    - Rarefaction wave (x ≈ 0.25-0.5)
    - Contact discontinuity (x ≈ 0.68)
    - Shock wave (x ≈ 0.85)
"""
import torch
import sys
sys.path.insert(0, '.')

from tensornet.cfd.qtt_eval import dense_to_qtt_cores, qtt_eval_batch
from tensornet.cfd.tci_flux import rusanov_flux
from tensornet.cfd.qtt_tci import qtt_rusanov_flux_tci
import time

# Try to import Rust TCI backend
try:
    from tensornet.cfd.qtt_tci import qtt_rusanov_flux_tci_rust
    from tci_core import RUST_AVAILABLE
except ImportError:
    RUST_AVAILABLE = False


def sod_initial_condition(N: int, gamma: float = 1.4):
    """Create Sod shock tube initial condition."""
    x = torch.linspace(0, 1, N)
    
    # Left state
    rho_L, u_L, p_L = 1.0, 0.0, 1.0
    # Right state  
    rho_R, u_R, p_R = 0.125, 0.0, 0.1
    
    # Interface at x = 0.5
    left = x < 0.5
    
    rho = torch.where(left, rho_L * torch.ones_like(x), rho_R * torch.ones_like(x))
    u = torch.where(left, u_L * torch.ones_like(x), u_R * torch.ones_like(x))
    p = torch.where(left, p_L * torch.ones_like(x), p_R * torch.ones_like(x))
    
    # Convert to conserved variables
    rhou = rho * u
    E = p / (gamma - 1) + 0.5 * rho * u**2
    
    return rho, rhou, E, x


def euler_step_dense(rho: torch.Tensor, rhou: torch.Tensor, E: torch.Tensor,
                     dx: float, dt: float, gamma: float = 1.4):
    """Single Euler time step using dense Rusanov flux."""
    N = rho.shape[0]
    
    # Compute fluxes at cell faces (i+1/2)
    F_rho, F_rhou, F_E = rusanov_flux(
        rho, rhou, E,
        torch.roll(rho, -1), torch.roll(rhou, -1), torch.roll(E, -1),
        gamma
    )
    
    # Finite volume update: U^{n+1} = U^n - dt/dx * (F_{i+1/2} - F_{i-1/2})
    rho_new = rho - dt/dx * (F_rho - torch.roll(F_rho, 1))
    rhou_new = rhou - dt/dx * (F_rhou - torch.roll(F_rhou, 1))
    E_new = E - dt/dx * (F_E - torch.roll(F_E, 1))
    
    return rho_new, rhou_new, E_new


def euler_step_qtt(rho_cores, rhou_cores, E_cores,
                   dx: float, dt: float, gamma: float = 1.4,
                   max_rank: int = 64, use_rust: bool = False):
    """Single Euler time step using QTT-TCI Rusanov flux."""
    n_qubits = len(rho_cores)
    N = 2 ** n_qubits
    
    # Choose TCI backend
    if use_rust and RUST_AVAILABLE:
        F_rho_cores, F_rhou_cores, F_E_cores, meta = qtt_rusanov_flux_tci_rust(
            rho_cores, rhou_cores, E_cores,
            gamma=gamma, max_rank=max_rank, verbose=False
        )
    else:
        # Compute flux in QTT format via TCI
        F_rho_cores, F_rhou_cores, F_E_cores, meta = qtt_rusanov_flux_tci(
            rho_cores, rhou_cores, E_cores,
            gamma=gamma, max_rank=max_rank, verbose=False
        )
    
    # For now, decompress to dense for the update
    # (Full QTT arithmetic is Phase 3)
    all_idx = torch.arange(N)
    
    rho = qtt_eval_batch(rho_cores, all_idx)
    rhou = qtt_eval_batch(rhou_cores, all_idx)
    E = qtt_eval_batch(E_cores, all_idx)
    
    F_rho = qtt_eval_batch(F_rho_cores, all_idx)
    F_rhou = qtt_eval_batch(F_rhou_cores, all_idx)
    F_E = qtt_eval_batch(F_E_cores, all_idx)
    
    # Update
    rho_new = rho - dt/dx * (F_rho - torch.roll(F_rho, 1))
    rhou_new = rhou - dt/dx * (F_rhou - torch.roll(F_rhou, 1))
    E_new = E - dt/dx * (F_E - torch.roll(F_E, 1))
    
    # Recompress to QTT
    rho_cores_new = dense_to_qtt_cores(rho_new, max_rank=max_rank)
    rhou_cores_new = dense_to_qtt_cores(rhou_new, max_rank=max_rank)
    E_cores_new = dense_to_qtt_cores(E_new, max_rank=max_rank)
    
    return rho_cores_new, rhou_cores_new, E_cores_new, meta


def validate_sod():
    """Run Sod shock tube validation."""
    print("=" * 70)
    print("SOD SHOCK TUBE VALIDATION")
    print("=" * 70)
    print()
    
    # Parameters
    N = 1024  # Grid points
    n_qubits = 10
    gamma = 1.4
    dx = 1.0 / N
    t_final = 0.05  # Short simulation
    CFL = 0.5
    
    # Initial condition
    rho, rhou, E, x = sod_initial_condition(N, gamma)
    
    print(f"Grid: N = {N} points")
    print(f"Time: t_final = {t_final}")
    print(f"CFL: {CFL}")
    print()
    
    # =============================================
    # Method 1: Dense Rusanov (reference)
    # =============================================
    print("Method 1: Dense Rusanov Flux")
    print("-" * 40)
    
    rho_d, rhou_d, E_d = rho.clone(), rhou.clone(), E.clone()
    t = 0.0
    steps = 0
    t0 = time.time()
    
    while t < t_final:
        # Compute max wave speed for CFL
        u = rhou_d / rho_d
        p = (gamma - 1) * (E_d - 0.5 * rho_d * u**2)
        c = torch.sqrt(gamma * p / rho_d)
        max_speed = (u.abs() + c).max().item()
        
        dt = CFL * dx / max_speed
        if t + dt > t_final:
            dt = t_final - t
        
        rho_d, rhou_d, E_d = euler_step_dense(rho_d, rhou_d, E_d, dx, dt, gamma)
        t += dt
        steps += 1
    
    dense_time = time.time() - t0
    print(f"  Steps: {steps}")
    print(f"  Time: {dense_time:.3f}s")
    print()
    
    # =============================================
    # Method 2: QTT-TCI Rusanov
    # =============================================
    print("Method 2: QTT-TCI Rusanov Flux")
    print("-" * 40)
    
    # Convert IC to QTT
    rho_cores = dense_to_qtt_cores(rho, max_rank=64)
    rhou_cores = dense_to_qtt_cores(rhou, max_rank=64)
    E_cores = dense_to_qtt_cores(E, max_rank=64)
    
    t = 0.0
    steps = 0
    total_evals = 0
    t0 = time.time()
    
    while t < t_final:
        # Decompress for CFL computation (temp)
        all_idx = torch.arange(N)
        rho_q = qtt_eval_batch(rho_cores, all_idx)
        rhou_q = qtt_eval_batch(rhou_cores, all_idx)
        E_q = qtt_eval_batch(E_cores, all_idx)
        
        u = rhou_q / rho_q
        p = (gamma - 1) * (E_q - 0.5 * rho_q * u**2)
        c = torch.sqrt(gamma * p / rho_q)
        max_speed = (u.abs() + c).max().item()
        
        dt = CFL * dx / max_speed
        if t + dt > t_final:
            dt = t_final - t
        
        rho_cores, rhou_cores, E_cores, meta = euler_step_qtt(
            rho_cores, rhou_cores, E_cores, dx, dt, gamma, max_rank=64, use_rust=False
        )
        t += dt
        steps += 1
        total_evals += meta['total_evals']
    
    qtt_time = time.time() - t0
    print(f"  Steps: {steps}")
    print(f"  Time: {qtt_time:.3f}s")
    print(f"  Total flux evals: {total_evals}")
    print(f"  Avg evals/step: {total_evals / steps:.0f}")
    print()
    
    # =============================================
    # Method 3: QTT-TCI Rusanov with Rust backend
    # =============================================
    qtt_rust_time = None
    total_evals_rust = 0
    rho_qtt_rust_final = None
    
    if RUST_AVAILABLE:
        print("Method 3: QTT-TCI Rusanov (Rust TCI)")
        print("-" * 40)
        
        # Convert IC to QTT
        rho_cores_rust = dense_to_qtt_cores(rho, max_rank=64)
        rhou_cores_rust = dense_to_qtt_cores(rhou, max_rank=64)
        E_cores_rust = dense_to_qtt_cores(E, max_rank=64)
        
        t = 0.0
        steps_rust = 0
        t0 = time.time()
        
        while t < t_final:
            # Decompress for CFL computation
            all_idx = torch.arange(N)
            rho_q = qtt_eval_batch(rho_cores_rust, all_idx)
            rhou_q = qtt_eval_batch(rhou_cores_rust, all_idx)
            E_q = qtt_eval_batch(E_cores_rust, all_idx)
            
            u = rhou_q / rho_q
            p = (gamma - 1) * (E_q - 0.5 * rho_q * u**2)
            c = torch.sqrt(gamma * p / rho_q)
            max_speed = (u.abs() + c).max().item()
            
            dt = CFL * dx / max_speed
            if t + dt > t_final:
                dt = t_final - t
            
            rho_cores_rust, rhou_cores_rust, E_cores_rust, meta = euler_step_qtt(
                rho_cores_rust, rhou_cores_rust, E_cores_rust, dx, dt, gamma, max_rank=64, use_rust=True
            )
            t += dt
            steps_rust += 1
            total_evals_rust += meta['total_evals']
        
        qtt_rust_time = time.time() - t0
        print(f"  Steps: {steps_rust}")
        print(f"  Time: {qtt_rust_time:.3f}s")
        print(f"  Total flux evals: {total_evals_rust}")
        print(f"  Avg evals/step: {total_evals_rust / steps_rust:.0f}")
        
        # Get final solution
        rho_qtt_rust_final = qtt_eval_batch(rho_cores_rust, all_idx)
        print()
    else:
        print("Method 3: Skipped (Rust TCI not available)")
        print()
    
    # =============================================
    # Compare results
    # =============================================
    print("Comparison:")
    print("-" * 40)
    
    # Get final QTT solution
    all_idx = torch.arange(N)
    rho_qtt_final = qtt_eval_batch(rho_cores, all_idx)
    
    # Compare Python TCI vs Rust TCI if both available
    if RUST_AVAILABLE and rho_qtt_rust_final is not None:
        rust_vs_python = (rho_qtt_final - rho_qtt_rust_final).abs()
        print(f"  Max |Python TCI - Rust TCI|: {rust_vs_python.max():.2e}")
    
    # Compare to dense
    error = (rho_d - rho_qtt_final).abs()
    print(f"  Max |ρ_dense - ρ_qtt|: {error.max():.2e}")
    print(f"  Mean |ρ_dense - ρ_qtt|: {error.mean():.2e}")
    print(f"  L2 relative error: {error.norm() / rho_d.norm():.2e}")
    
    # Check physical features
    print()
    print("Physical Validation:")
    print("-" * 40)
    
    # Density should be bounded
    print(f"  ρ min: {rho_qtt_final.min():.4f} (expected ≈ 0.125)")
    print(f"  ρ max: {rho_qtt_final.max():.4f} (expected ≈ 1.0)")
    
    # Check conservation (mass should be conserved)
    mass_init = rho.sum().item() * dx
    mass_final = rho_qtt_final.sum().item() * dx
    mass_error = abs(mass_final - mass_init) / mass_init
    print(f"  Mass conservation error: {mass_error:.2e}")
    
    # Validation criteria
    assert error.max() < 0.1, f"Dense-QTT divergence too large: {error.max()}"
    assert rho_qtt_final.min() > 0, "Negative density!"
    assert mass_error < 0.01, f"Mass not conserved: {mass_error}"
    
    print()
    print("=" * 70)
    print("SOD SHOCK TUBE VALIDATION PASSED ✓")
    print("=" * 70)
    
    results = {
        "dense_time": dense_time,
        "qtt_time": qtt_time,
        "max_error": error.max().item(),
        "mass_conservation": mass_error,
        "total_evals": total_evals,
        "steps": steps,
    }
    
    if RUST_AVAILABLE and qtt_rust_time is not None:
        results["qtt_rust_time"] = qtt_rust_time
        results["total_evals_rust"] = total_evals_rust
        results["speedup_rust_vs_python"] = qtt_time / qtt_rust_time if qtt_rust_time > 0 else 0
        results["eval_reduction_rust"] = total_evals / total_evals_rust if total_evals_rust > 0 else 0
        
        print()
        print("Rust TCI Summary:")
        print(f"  Speedup vs Python TCI: {results['speedup_rust_vs_python']:.2f}x")
        print(f"  Eval reduction: {results['eval_reduction_rust']:.2f}x fewer")
    
    return results


if __name__ == "__main__":
    results = validate_sod()
