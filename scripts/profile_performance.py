#!/usr/bin/env python
"""
Performance profiling script for HyperTensor.

Uses PyTorch profiler to identify bottlenecks in:
1. MPS algorithms (DMRG, TEBD)
2. CFD solvers (Euler1D, Euler2D)

Usage:
    python scripts/profile_performance.py [--dmrg|--tebd|--cfd1d|--cfd2d] [--gpu]

Output:
    Profiling traces in ./profiling_output/
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from torch.profiler import profile, record_function, ProfilerActivity

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def profile_dmrg(device: str = "cpu", chi: int = 64, L: int = 20):
    """Profile DMRG ground state search."""
    from tensornet import dmrg, heisenberg_mpo
    from tensornet.core import random_mps
    
    print(f"Profiling DMRG: L={L}, chi={chi}, device={device}")
    
    # Create MPS and Hamiltonian
    mps = random_mps(L=L, d=2, chi=chi, device=device)
    H = heisenberg_mpo(L, J=1.0, Jz=1.0, device=device)
    
    # Warmup
    print("Warmup run...")
    _ = dmrg(H, chi_max=chi, num_sweeps=1)
    
    # Profile
    print("Profiling...")
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)
    
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("dmrg_full"):
            result = dmrg(H, chi_max=chi, num_sweeps=2)
    
    print(f"DMRG converged to E = {result.energy:.8f}")
    
    return prof


def profile_tebd(device: str = "cpu", chi: int = 64, L: int = 20):
    """Profile TEBD time evolution."""
    from tensornet.algorithms.tebd import tebd_step
    from tensornet.core import MPS
    
    print(f"Profiling TEBD: L={L}, chi={chi}, device={device}")
    
    # Create initial product state
    psi = MPS.random(L, d=2, chi=chi, dtype=torch.float64, device=device, normalize=True)
    
    # Create simple Heisenberg two-site gate (imaginary time evolution)
    dt = 0.01
    J = 1.0
    Sx = torch.tensor([[0, 0.5], [0.5, 0]], dtype=torch.float64, device=device)
    Sz = torch.tensor([[0.5, 0], [0, -0.5]], dtype=torch.float64, device=device)
    
    # H_bond = J * (Sx⊗Sx + Sz⊗Sz)
    H_bond = J * (torch.kron(Sx, Sx) + torch.kron(Sz, Sz))
    
    # exp(-dt * H_bond) - gate shape should be (s1', s2', s1, s2)
    U = torch.linalg.matrix_exp(-dt * H_bond)  # (d*d, d*d)
    gate = U.reshape(2, 2, 2, 2)  # (s1, s2, s1', s2') -> transpose to (s1', s2', s1, s2)
    gate = gate.permute(2, 3, 0, 1)  # Now (s1', s2', s1, s2)
    
    # Create gate lists for odd and even bonds
    # Odd bonds: 0-1, 2-3, 4-5, ... 
    n_odd = L // 2
    # Even bonds: 1-2, 3-4, 5-6, ...
    n_even = (L - 1) // 2
    
    gates_odd = [gate for _ in range(n_odd)]
    gates_even = [gate for _ in range(n_even)]
    
    # Warmup
    print("Warmup run...")
    for _ in range(5):
        tebd_step(psi, gates_odd, gates_even, chi_max=chi, order=2)
    
    # Profile
    print("Profiling...")
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)
    
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("tebd_full"):
            for _ in range(20):
                tebd_step(psi, gates_odd, gates_even, chi_max=chi, order=2)
    
    print(f"TEBD completed 20 steps")
    
    return prof


def profile_cfd_1d(Nx: int = 500):
    """Profile 1D Euler solver."""
    from tensornet.cfd.euler_1d import Euler1D, EulerState
    
    print(f"Profiling Euler1D: Nx={Nx}")
    
    # Sod shock tube
    solver = Euler1D(N=Nx, x_min=0.0, x_max=1.0, gamma=1.4, cfl=0.4)
    
    # Set Sod IC
    x = solver.x_cell
    rho = torch.where(x < 0.5, torch.ones_like(x), 0.125 * torch.ones_like(x))
    u = torch.zeros_like(x)
    p = torch.where(x < 0.5, torch.ones_like(x), 0.1 * torch.ones_like(x))
    
    state = EulerState.from_primitive(rho, u, p, gamma=1.4)
    solver.set_initial_condition(state)
    
    # Warmup
    print("Warmup run...")
    for _ in range(10):
        solver.step()
    
    # Profile
    print("Profiling...")
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("euler1d_solve"):
            for _ in range(100):
                solver.step()
    
    print(f"Euler1D: t = {solver.t:.4f}")
    
    return prof


def profile_cfd_2d(Nx: int = 100, Ny: int = 100):
    """Profile 2D Euler solver."""
    from tensornet.cfd.euler_2d import Euler2D, Euler2DState, BCType
    
    print(f"Profiling Euler2D: Nx={Nx}, Ny={Ny}")
    
    # Uniform supersonic flow
    gamma = 1.4
    M_inf = 2.0
    
    rho = torch.ones(Ny, Nx, dtype=torch.float64)
    u = M_inf * torch.ones(Ny, Nx, dtype=torch.float64)
    v = torch.zeros(Ny, Nx, dtype=torch.float64)
    p = (1.0 / gamma) * torch.ones(Ny, Nx, dtype=torch.float64)
    
    state = Euler2DState(rho, u, v, p)
    
    solver = Euler2D(
        state,
        dx=1.0/Nx,
        dy=1.0/Ny,
        bc_left=BCType.SUPERSONIC_INFLOW,
        bc_right=BCType.OUTFLOW,
        bc_bottom=BCType.REFLECTIVE,
        bc_top=BCType.OUTFLOW
    )
    solver.inflow_state = state
    
    # Warmup
    print("Warmup run...")
    for _ in range(5):
        solver.step(cfl=0.3)
    
    # Profile
    print("Profiling...")
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("euler2d_solve"):
            for _ in range(20):
                solver.step(cfl=0.3)
    
    print(f"Euler2D: t = {solver.time:.6f}, {solver.step_count} steps")
    
    return prof


def print_profiler_summary(prof, top_n: int = 20):
    """Print profiler summary table."""
    print("\n" + "="*80)
    print("TOP CPU TIME:")
    print("="*80)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=top_n))
    
    print("\n" + "="*80)
    print("TOP SELF CPU TIME:")
    print("="*80)
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=top_n))


def save_trace(prof, name: str, output_dir: Path):
    """Save profiler trace to file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_file = output_dir / f"{name}_trace.json"
    prof.export_chrome_trace(str(trace_file))
    print(f"\nTrace saved to: {trace_file}")
    print("Open in Chrome at: chrome://tracing/")


def main():
    parser = argparse.ArgumentParser(description="Profile HyperTensor performance")
    parser.add_argument("--dmrg", action="store_true", help="Profile DMRG")
    parser.add_argument("--tebd", action="store_true", help="Profile TEBD")
    parser.add_argument("--cfd1d", action="store_true", help="Profile Euler1D")
    parser.add_argument("--cfd2d", action="store_true", help="Profile Euler2D")
    parser.add_argument("--all", action="store_true", help="Profile all")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--chi", type=int, default=64, help="Bond dimension for MPS")
    parser.add_argument("--L", type=int, default=20, help="Chain length for MPS")
    parser.add_argument("--Nx", type=int, default=200, help="Grid size for CFD")
    parser.add_argument("--save", action="store_true", help="Save trace files")
    
    args = parser.parse_args()
    
    # Default: run all
    if not any([args.dmrg, args.tebd, args.cfd1d, args.cfd2d, args.all]):
        args.all = True
    
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    if args.gpu and not torch.cuda.is_available():
        print("Warning: GPU requested but CUDA not available, using CPU")
    
    output_dir = Path(__file__).parent.parent / "profiling_output"
    
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    if args.dmrg or args.all:
        print("\n" + "="*80)
        print("DMRG PROFILING")
        print("="*80)
        prof = profile_dmrg(device=device, chi=args.chi, L=args.L)
        print_profiler_summary(prof)
        if args.save:
            save_trace(prof, "dmrg", output_dir)
    
    if args.tebd or args.all:
        print("\n" + "="*80)
        print("TEBD PROFILING")
        print("="*80)
        prof = profile_tebd(device=device, chi=args.chi, L=args.L)
        print_profiler_summary(prof)
        if args.save:
            save_trace(prof, "tebd", output_dir)
    
    if args.cfd1d or args.all:
        print("\n" + "="*80)
        print("EULER 1D PROFILING")
        print("="*80)
        prof = profile_cfd_1d(Nx=args.Nx)
        print_profiler_summary(prof)
        if args.save:
            save_trace(prof, "euler1d", output_dir)
    
    if args.cfd2d or args.all:
        print("\n" + "="*80)
        print("EULER 2D PROFILING")
        print("="*80)
        prof = profile_cfd_2d(Nx=args.Nx, Ny=args.Nx)
        print_profiler_summary(prof)
        if args.save:
            save_trace(prof, "euler2d", output_dir)
    
    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
