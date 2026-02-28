#!/usr/bin/env python3
"""
Kelvin-Helmholtz Instability — Full QTT Simulation

Runs the KH instability on Morton Z-curve QTT format using
Strang splitting with native MPO shift operators (Rusanov flux).

What this demonstrates:
  - 2D compressible Euler equations in pure QTT
  - Morton Z-curve spatial ordering for data locality
  - Strang splitting: U^{n+1} = Lx(dt/2) Ly(dt) Lx(dt/2) U^n
  - Rusanov (local Lax-Friedrichs) flux
  - Conservation verification (mass, momentum, energy)
  - QTT rank dynamics through vortex roll-up

Physics:
  Top half (y > 0.5): ρ=2.0, u=+0.5 (heavy, right)
  Bottom half (y<0.5): ρ=1.0, u=−0.5 (light, left)
  Perturbation: v = A sin(4πx) exp(−(y−0.5)²/σ²)
  Pressure: P = 2.5 (uniform)
  γ = 1.4

Author: HyperTensor Team
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch

from ontic.cfd.euler2d_strang import (
    Euler2D_Strang,
    Euler2DConfig,
    Euler2DState,
    create_kelvin_helmholtz_ic,
)
from ontic.cfd.kelvin_helmholtz import KHConfig, analyze_kh_ranks
from ontic.cfd.qtt_2d import qtt_2d_to_dense


# ─────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────

def compute_conserved(state: Euler2DState, gamma: float = 1.4) -> dict:
    """Compute global conserved quantities from QTT state."""
    rho = qtt_2d_to_dense(state.rho)
    rhou = qtt_2d_to_dense(state.rhou)
    rhov = qtt_2d_to_dense(state.rhov)
    E = qtt_2d_to_dense(state.E)

    return {
        "mass": float(rho.sum()),
        "mom_x": float(rhou.sum()),
        "mom_y": float(rhov.sum()),
        "energy": float(E.sum()),
        "rho_min": float(rho.min()),
        "rho_max": float(rho.max()),
    }


def compute_max_rank(state: Euler2DState) -> int:
    """Maximum bond dimension across all 4 conserved fields."""
    ranks = []
    for field in [state.rho, state.rhou, state.rhov, state.E]:
        for c in field.cores:
            ranks.append(max(c.shape[0], c.shape[-1]))
    return max(ranks) if ranks else 1


def compute_enstrophy_proxy(state: Euler2DState, gamma: float = 1.4) -> float:
    """
    Enstrophy proxy: integral of |omega|^2 where omega = dv/dx - du/dy.

    Uses finite differences for the curl.  This is only a diagnostic
    (not conserved by Euler) but tracks vortex activity.
    """
    rho_d, u_d, v_d, P_d = state.get_primitives(gamma)
    Nx, Ny = rho_d.shape
    dx = 1.0 / Nx
    dy = 1.0 / Ny

    dvdx = (torch.roll(v_d, -1, dims=0) - torch.roll(v_d, 1, dims=0)) / (2 * dx)
    dudy = (torch.roll(u_d, -1, dims=1) - torch.roll(u_d, 1, dims=1)) / (2 * dy)
    omega = dvdx - dudy

    return float((omega**2).sum()) * dx * dy


def qtt_memory_bytes(state: Euler2DState) -> int:
    """Total QTT storage in bytes."""
    total = 0
    for field in [state.rho, state.rhou, state.rhov, state.E]:
        for c in field.cores:
            total += c.numel() * c.element_size()
    return total


# ─────────────────────────────────────────────────────────────────────
# Main simulation loop
# ─────────────────────────────────────────────────────────────────────

def run_kh_simulation(
    n_bits: int = 7,
    t_final: float = 2.0,
    max_steps: int = 5000,
    max_rank: int = 64,
    cfl: float = 0.3,
    print_every: int = 20,
) -> dict:
    """
    Run a Kelvin-Helmholtz instability simulation.

    Args:
        n_bits: Bits per spatial dimension (grid = 2^n × 2^n)
        t_final: Physical end time
        max_steps: Hard cap on number of time steps
        max_rank: Maximum QTT bond dimension
        cfl: CFL number for time-step selection
        print_every: Print diagnostics every N steps

    Returns:
        Dictionary of simulation metadata and results
    """
    N = 2**n_bits
    gamma = 1.4

    print("=" * 76)
    print(f"  Kelvin-Helmholtz Instability — {N}×{N} QTT Simulation")
    print("=" * 76)
    print(f"  Grid:        {N}×{N} = {N*N:,} cells")
    print(f"  QTT qubits:  {2*n_bits} (Morton Z-curve)")
    print(f"  Max rank:    {max_rank}")
    print(f"  CFL:         {cfl}")
    print(f"  T_final:     {t_final}")
    print(f"  Dense mem:   {N*N*4*8/1024:.1f} KB (4 fields × float64)")

    # ── Create solver ────────────────────────────────────────────────
    config = Euler2DConfig(
        gamma=gamma,
        cfl=cfl,
        max_rank=max_rank,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    solver = Euler2D_Strang(n_bits, n_bits, config)

    # ── Create IC ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    state = create_kelvin_helmholtz_ic(n_bits, n_bits, config)
    t_ic = time.perf_counter() - t0

    qtt_bytes = qtt_memory_bytes(state)
    dense_bytes = N * N * 4 * 8  # 4 fields × float64
    compress = dense_bytes / qtt_bytes if qtt_bytes > 0 else 0

    print(f"  QTT mem:     {qtt_bytes/1024:.1f} KB ({compress:.1f}× compression)")
    print(f"  IC time:     {t_ic:.3f}s")

    # ── Record initial conserved quantities ──────────────────────────
    c0 = compute_conserved(state, gamma)
    rank0 = compute_max_rank(state)
    enst0 = compute_enstrophy_proxy(state, gamma)

    print(f"\n  Initial state:")
    print(f"    ρ  = [{c0['rho_min']:.4f}, {c0['rho_max']:.4f}]")
    print(f"    Mass     = {c0['mass']:.6f}")
    print(f"    Mom_x    = {c0['mom_x']:.6f}")
    print(f"    Energy   = {c0['energy']:.6f}")
    print(f"    MaxRank  = {rank0}")
    print(f"    Enstrophy= {enst0:.4f}")

    # ── Time-stepping ────────────────────────────────────────────────
    print(f"\n{'─'*76}")
    print(f"  {'Step':>6}  {'Time':>8}  {'dt':>10}  {'Rank':>5}  "
          f"{'ΔMass':>10}  {'ΔMom_x':>10}  {'ΔE':>10}  {'Enst':>10}  {'ms':>6}")
    print(f"{'─'*76}")

    history = {
        "t": [0.0],
        "mass": [c0["mass"]],
        "mom_x": [c0["mom_x"]],
        "energy": [c0["energy"]],
        "max_rank": [rank0],
        "enstrophy": [enst0],
        "step_ms": [],
    }

    sim_t = 0.0
    step = 0
    wall_start = time.perf_counter()

    while sim_t < t_final and step < max_steps:
        dt = solver.compute_dt(state)
        if sim_t + dt > t_final:
            dt = t_final - sim_t

        t_step_start = time.perf_counter()
        state = solver.step(state, dt)
        step_ms = (time.perf_counter() - t_step_start) * 1000.0

        sim_t += dt
        step += 1

        # Diagnostics at intervals
        if step % print_every == 0 or sim_t >= t_final:
            c = compute_conserved(state, gamma)
            rank = compute_max_rank(state)
            enst = compute_enstrophy_proxy(state, gamma)

            dm = (c["mass"] - c0["mass"]) / c0["mass"]
            dp = (c["mom_x"] - c0["mom_x"]) / (abs(c0["mom_x"]) + 1e-30)
            de = (c["energy"] - c0["energy"]) / c0["energy"]

            history["t"].append(sim_t)
            history["mass"].append(c["mass"])
            history["mom_x"].append(c["mom_x"])
            history["energy"].append(c["energy"])
            history["max_rank"].append(rank)
            history["enstrophy"].append(enst)
            history["step_ms"].append(step_ms)

            print(
                f"  {step:6d}  {sim_t:8.4f}  {dt:10.2e}  {rank:5d}  "
                f"{dm:10.2e}  {dp:10.2e}  {de:10.2e}  {enst:10.4f}  "
                f"{step_ms:6.0f}"
            )

    wall_total = time.perf_counter() - wall_start

    # ── Final diagnostics ────────────────────────────────────────────
    c_final = compute_conserved(state, gamma)
    rank_final = compute_max_rank(state)
    enst_final = compute_enstrophy_proxy(state, gamma)

    qtt_bytes_final = qtt_memory_bytes(state)
    compress_final = dense_bytes / qtt_bytes_final if qtt_bytes_final > 0 else 0

    dm_abs = abs(c_final["mass"] - c0["mass"]) / c0["mass"]
    de_abs = abs(c_final["energy"] - c0["energy"]) / c0["energy"]

    print(f"\n{'='*76}")
    print(f"  SIMULATION COMPLETE")
    print(f"{'='*76}")
    print(f"  Total steps:  {step}")
    print(f"  Simulated t:  {sim_t:.4f}")
    print(f"  Wall time:    {wall_total:.1f}s ({wall_total/step*1000:.1f} ms/step)")
    print(f"\n  Conservation:")
    print(f"    |ΔMass/Mass₀|   = {dm_abs:.2e}  {'✓' if dm_abs < 1e-6 else '✗'}")
    print(f"    |ΔEnergy/E₀|    = {de_abs:.2e}  {'✓' if de_abs < 1e-3 else '✗'}")
    print(f"\n  Rank dynamics:")
    print(f"    Initial rank:    {rank0}")
    print(f"    Peak rank:       {max(history['max_rank'])}")
    print(f"    Final rank:      {rank_final}")
    print(f"\n  Compression:")
    print(f"    Initial:  {dense_bytes/qtt_bytes:.1f}×")
    print(f"    Final:    {compress_final:.1f}×")
    print(f"    QTT mem:  {qtt_bytes_final/1024:.1f} KB vs dense {dense_bytes/1024:.1f} KB")
    print(f"\n  Vortex activity:")
    print(f"    Enstrophy:  {enst0:.4f} → {enst_final:.4f}  (×{enst_final/(enst0+1e-30):.1f})")
    print(f"\n  Physics:")
    rho_d, u_d, v_d, P_d = state.get_primitives(gamma)
    print(f"    ρ = [{float(rho_d.min()):.4f}, {float(rho_d.max()):.4f}]")
    print(f"    u = [{float(u_d.min()):.4f}, {float(u_d.max()):.4f}]")
    print(f"    v = [{float(v_d.min()):.4f}, {float(v_d.max()):.4f}]")
    print(f"    P = [{float(P_d.min()):.4f}, {float(P_d.max()):.4f}]")

    valid = float(rho_d.min()) > 0 and float(P_d.min()) > 0
    conserved = dm_abs < 1e-6 and de_abs < 1e-3
    vortex = enst_final > enst0

    print(f"\n{'='*76}")
    if valid and conserved:
        print(f"  RESULT: ✓ PASS — Physics valid, conservation verified")
        if vortex:
            print(f"          ✓ Vortex roll-up detected (enstrophy {enst_final/enst0:.1f}× growth)")
    else:
        reasons = []
        if not valid:
            reasons.append("negative ρ or P")
        if not conserved:
            reasons.append(f"conservation exceeded threshold (dM={dm_abs:.2e}, dE={de_abs:.2e})")
        print(f"  RESULT: ✗ FAIL — {', '.join(reasons)}")
    print(f"{'='*76}")

    return {
        "n_bits": n_bits,
        "N": N,
        "steps": step,
        "wall_time": wall_total,
        "mass_error": dm_abs,
        "energy_error": de_abs,
        "peak_rank": max(history["max_rank"]),
        "final_rank": rank_final,
        "enstrophy_growth": enst_final / (enst0 + 1e-30),
        "valid": valid,
        "conserved": conserved,
        "history": history,
    }


# ─────────────────────────────────────────────────────────────────────
# Scaling study
# ─────────────────────────────────────────────────────────────────────

def run_scaling_study(n_bits_list: list[int], max_rank: int = 64, n_steps: int = 20):
    """Run KH at multiple grid sizes and report scaling."""
    print("=" * 76)
    print("  Kelvin-Helmholtz Scaling Study")
    print("=" * 76)

    results = []
    for n_bits in n_bits_list:
        N = 2**n_bits
        config = Euler2DConfig(
            gamma=1.4, cfl=0.3, max_rank=max_rank, dtype=torch.float64
        )
        solver = Euler2D_Strang(n_bits, n_bits, config)
        state = create_kelvin_helmholtz_ic(n_bits, n_bits, config)

        rho0_d = qtt_2d_to_dense(state.rho)
        mass0 = float(rho0_d.sum())
        qtt_bytes = qtt_memory_bytes(state)

        t0 = time.perf_counter()
        t_phys = 0.0
        for _ in range(n_steps):
            dt = solver.compute_dt(state)
            state = solver.step(state, dt)
            t_phys += dt
        wall = time.perf_counter() - t0

        rho_f = qtt_2d_to_dense(state.rho)
        mass_f = float(rho_f.sum())
        dm = abs(mass_f - mass0) / mass0
        rank = compute_max_rank(state)
        qtt_final = qtt_memory_bytes(state)
        dense_bytes = N * N * 4 * 8

        results.append({
            "N": N,
            "n_bits": n_bits,
            "ms_per_step": wall / n_steps * 1000,
            "rank": rank,
            "dM": dm,
            "compress": dense_bytes / qtt_final,
        })

    print(f"\n  {'Grid':>10}  {'ms/step':>10}  {'Rank':>6}  {'ΔMass':>10}  {'Compress':>10}")
    print(f"  {'─'*10}  {'─'*10}  {'─'*6}  {'─'*10}  {'─'*10}")
    for r in results:
        print(
            f"  {r['N']}×{r['N']:>4}  {r['ms_per_step']:>10.1f}  "
            f"{r['rank']:>6}  {r['dM']:>10.2e}  {r['compress']:>9.1f}×"
        )

    if len(results) >= 2:
        r0, r1 = results[-2], results[-1]
        grid_ratio = (r1["N"] / r0["N"]) ** 2
        time_ratio = r1["ms_per_step"] / r0["ms_per_step"]
        print(f"\n  Grid ×{grid_ratio:.0f} → Time ×{time_ratio:.1f} "
              f"(O(N²) would be ×{grid_ratio:.0f})")

    print("=" * 76)


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kelvin-Helmholtz Instability — QTT Simulation"
    )
    parser.add_argument(
        "-n", "--n-bits", type=int, default=7,
        help="Bits per dimension (grid = 2^n × 2^n). Default: 7 (128×128)"
    )
    parser.add_argument(
        "-t", "--time", type=float, default=1.0,
        help="Final simulation time. Default: 1.0"
    )
    parser.add_argument(
        "--max-steps", type=int, default=5000,
        help="Maximum number of time steps"
    )
    parser.add_argument(
        "-r", "--max-rank", type=int, default=64,
        help="Maximum QTT bond dimension. Default: 64"
    )
    parser.add_argument(
        "--cfl", type=float, default=0.3,
        help="CFL number. Default: 0.3"
    )
    parser.add_argument(
        "--print-every", type=int, default=20,
        help="Print diagnostics every N steps"
    )
    parser.add_argument(
        "--scaling", action="store_true",
        help="Run scaling study instead of single simulation"
    )
    parser.add_argument(
        "--rank-analysis", action="store_true",
        help="Run IC rank analysis only"
    )

    args = parser.parse_args()

    if args.rank_analysis:
        analyze_kh_ranks(args.n_bits)
    elif args.scaling:
        run_scaling_study([5, 6, 7, 8], max_rank=args.max_rank, n_steps=20)
    else:
        run_kh_simulation(
            n_bits=args.n_bits,
            t_final=args.time,
            max_steps=args.max_steps,
            max_rank=args.max_rank,
            cfl=args.cfl,
            print_every=args.print_every,
        )
