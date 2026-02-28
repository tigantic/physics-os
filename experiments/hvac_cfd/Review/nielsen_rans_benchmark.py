#!/usr/bin/env python3
"""
Nielsen Benchmark - 2D RANS with Mixing-Length Model
=====================================================

The 2D laminar solver gives ~26% RMS error. This is because at Re=5000,
the ceiling jet is turbulent and needs enhanced mixing.

This script adds a simple mixing-length turbulence model:
    ν_t = l_m² * |∂u/∂y|
    
Where l_m = κ * y * (1 - exp(-y⁺/A⁺)) (van Driest damping)

Target: <10% RMS error against Aalborg experimental data.
"""

import json
import sys
import time
sys.path.insert(0, '/home/brad/TiganticLabz/Main_Projects/Project HyperTensor')

import numpy as np
import torch
from torch import Tensor

# Import the base solver
from ontic.hvac.projection_solver import ProjectionConfig, ProjectionSolver, SolverState


# Aalborg experimental data
AALBORG_DATA = {
    "x_H_1.0": {
        "y_H": np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        "u_Uinlet": np.array([-0.05, -0.08, -0.10, -0.08, -0.02, 0.05, 0.12, 0.22, 0.38, 0.58, 0.85]),
    },
    "x_H_2.0": {
        "y_H": np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        "u_Uinlet": np.array([-0.03, -0.05, -0.06, -0.05, -0.03, 0.00, 0.05, 0.12, 0.22, 0.35, 0.52]),
    },
}


class MixingLengthSolver(ProjectionSolver):
    """
    Projection solver with mixing-length turbulence model.
    
    Adds eddy viscosity for enhanced turbulent mixing.
    """
    
    def __init__(self, config: ProjectionConfig, C_ml: float = 0.1):
        super().__init__(config)
        self.C_ml = C_ml  # Mixing length coefficient
        
        # Van Driest damping constant
        self.A_plus = 26.0
        self.kappa = 0.41
        
        # Precompute wall distances
        self._compute_wall_distances()
    
    def _compute_wall_distances(self):
        """Compute distance from nearest wall for each cell."""
        cfg = self.config
        
        # y-coordinate (distance from floor)
        y = torch.linspace(0, cfg.height, cfg.ny, dtype=self.dtype, device=self.device)
        
        # Distance to nearest wall (floor or ceiling)
        y_wall = torch.minimum(y, cfg.height - y)
        
        # Store as 2D field
        self.y_wall = y_wall.unsqueeze(0).expand(cfg.nx, -1)
    
    def compute_eddy_viscosity(self, u: Tensor, v: Tensor) -> Tensor:
        """
        Compute eddy viscosity using mixing-length model.
        
        ν_t = l_m² * |S|
        
        where l_m = κ * y_wall * damping
        """
        cfg = self.config
        nu = cfg.nu
        dx, dy = self.dx, self.dy
        
        # Strain rate: |S| = sqrt(2 * S_ij * S_ij)
        # Simplified: mainly |∂u/∂y| for ceiling jet
        dudy = torch.zeros_like(u)
        dudy[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dy)
        
        dvdx = torch.zeros_like(v)
        dvdx[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2 * dx)
        
        dudx = torch.zeros_like(u)
        dudx[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dx)
        
        dvdy = torch.zeros_like(v)
        dvdy[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dy)
        
        # Strain rate magnitude
        S_mag = torch.sqrt(2 * (dudx**2 + dvdy**2) + (dudy + dvdx)**2 + 1e-10)
        
        # Mixing length with van Driest damping
        # Estimate y+ from laminar estimate
        u_tau = torch.sqrt(nu * S_mag.max() / dy)  # Rough estimate
        y_plus = self.y_wall * u_tau / nu
        
        # Van Driest damping: 1 - exp(-y+/A+)
        damping = 1.0 - torch.exp(-y_plus / self.A_plus)
        
        # Mixing length
        l_m = self.C_ml * self.y_wall * damping
        
        # Eddy viscosity
        nu_t = l_m ** 2 * S_mag
        
        # Limit to prevent instability
        nu_t_max = 100 * nu
        nu_t = torch.clamp(nu_t, max=nu_t_max)
        
        return nu_t
    
    def step(self, state: SolverState) -> SolverState:
        """
        Advance one timestep with turbulent viscosity.
        """
        cfg = self.config
        u, v, p = state.u, state.v, state.p
        
        # Compute effective viscosity
        nu_t = self.compute_eddy_viscosity(u, v)
        nu_eff = cfg.nu + nu_t
        
        # Store mean eddy viscosity for diagnostics
        self.mean_nu_t = nu_t.mean().item()
        
        # Compute stable timestep
        dt = self.compute_timestep(state)
        
        # --- PREDICTOR: advection + diffusion ---
        
        # Advection (use parent class TVD method)
        adv_u = self.compute_advection(u, u, v)
        adv_v = self.compute_advection(v, u, v)
        
        # Diffusion with variable viscosity
        diff_u = self.compute_diffusion_variable(u, nu_eff)
        diff_v = self.compute_diffusion_variable(v, nu_eff)
        
        # Predict velocity
        u_star = u + dt * (adv_u + diff_u)
        v_star = v + dt * (adv_v + diff_v)
        
        # Apply velocity BCs
        u_star, v_star = self.apply_velocity_bc(u_star, v_star)
        
        # --- CORRECTOR: pressure projection ---
        
        div = self.compute_divergence(u_star, v_star)
        rhs = div / dt
        
        # Solve pressure Poisson
        p = self.solve_pressure(rhs, p)
        
        # Correct velocity
        dpdx, dpdy = self.compute_pressure_gradient(p)
        u_new = u_star - dt * dpdx
        v_new = v_star - dt * dpdy
        
        # Final BC application
        u_new, v_new = self.apply_velocity_bc(u_new, v_new)
        
        # Compute residual
        residual = max(
            torch.abs(u_new - u).max().item(),
            torch.abs(v_new - v).max().item(),
        )
        
        return SolverState(
            u=u_new,
            v=v_new,
            p=p,
            iteration=state.iteration + 1,
            converged=residual < cfg.convergence_tol,
            residual_history=state.residual_history + [residual],
        )
    
    def compute_diffusion_variable(self, phi: Tensor, nu: Tensor) -> Tensor:
        """
        Diffusion with spatially-varying viscosity: ∇·(ν∇φ)
        """
        dx, dy = self.dx, self.dy
        dx2, dy2 = dx**2, dy**2
        
        diff = torch.zeros_like(phi)
        
        # Face-averaged viscosity for conservation
        nu_ip = 0.5 * (nu[1:-1, 1:-1] + nu[2:, 1:-1])
        nu_im = 0.5 * (nu[1:-1, 1:-1] + nu[:-2, 1:-1])
        nu_jp = 0.5 * (nu[1:-1, 1:-1] + nu[1:-1, 2:])
        nu_jm = 0.5 * (nu[1:-1, 1:-1] + nu[1:-1, :-2])
        
        diff[1:-1, 1:-1] = (
            (nu_ip * (phi[2:, 1:-1] - phi[1:-1, 1:-1]) -
             nu_im * (phi[1:-1, 1:-1] - phi[:-2, 1:-1])) / dx2 +
            (nu_jp * (phi[1:-1, 2:] - phi[1:-1, 1:-1]) -
             nu_jm * (phi[1:-1, 1:-1] - phi[1:-1, :-2])) / dy2
        )
        
        return diff


def compute_rms_error(y_sim, u_sim, y_exp, u_exp):
    """Compute normalized RMS error."""
    u_sim_interp = np.interp(y_exp, y_sim, u_sim)
    u_scale = max(abs(u_exp.max()), abs(u_exp.min()), 1.0)
    rms = np.sqrt(np.mean((u_sim_interp - u_exp) ** 2)) / u_scale
    return rms


def run_rans_benchmark(nx=128, ny=64, max_iter=1500, C_ml=0.1):
    """Run RANS benchmark with mixing-length model."""
    
    config = ProjectionConfig(
        nx=nx, ny=ny,
        Re=5000,
        max_iterations=max_iter,
        convergence_tol=1e-5,
        dt_safety=0.25,
        advection_scheme='tvd',
        tvd_limiter='van_leer',
        alpha_u=0.8,
        alpha_p=0.4,
        pressure_iterations=100,
        verbose=True,
        diag_interval=100,
    )
    
    print(f"\n{'='*60}")
    print(f"2D RANS Nielsen Benchmark: {nx}×{ny}, C_ml={C_ml}")
    print(f"{'='*60}")
    print(f"Grid: {nx}×{ny} = {nx*ny:,} cells")
    print(f"Re = 5000")
    
    solver = MixingLengthSolver(config, C_ml=C_ml)
    
    start_time = time.time()
    state = solver.solve()
    solve_time = time.time() - start_time
    
    print(f"\nSolve time: {solve_time:.1f}s")
    print(f"Iterations: {state.iteration}")
    print(f"Converged: {state.converged}")
    print(f"Mean ν_t/ν: {solver.mean_nu_t / config.nu:.1f}")
    
    # Extract profiles
    H = config.height
    U_in = config.inlet_velocity
    
    y1, u1, _ = solver.extract_profile(state, x_position=3.0)
    y2, u2, _ = solver.extract_profile(state, x_position=6.0)
    
    y1_H = y1.numpy() / H
    u1_U = u1.numpy() / U_in
    y2_H = y2.numpy() / H
    u2_U = u2.numpy() / U_in
    
    rms_1 = compute_rms_error(y1_H, u1_U, AALBORG_DATA["x_H_1.0"]["y_H"], 
                              AALBORG_DATA["x_H_1.0"]["u_Uinlet"])
    rms_2 = compute_rms_error(y2_H, u2_U, AALBORG_DATA["x_H_2.0"]["y_H"], 
                              AALBORG_DATA["x_H_2.0"]["u_Uinlet"])
    rms_avg = (rms_1 + rms_2) / 2.0
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"RMS error @ x/H=1.0: {rms_1*100:.1f}%")
    print(f"RMS error @ x/H=2.0: {rms_2*100:.1f}%")
    print(f"RMS average:         {rms_avg*100:.1f}%")
    print(f"Target: <10%")
    print(f"Status: {'PASS' if rms_avg < 0.10 else 'FAIL'}")
    
    return {
        "solver": f"2D-RANS-ML(C={C_ml})",
        "grid": f"{nx}×{ny}",
        "cells": nx * ny,
        "iterations": state.iteration,
        "converged": state.converged,
        "solve_time_s": round(solve_time, 1),
        "nu_t_ratio": round(solver.mean_nu_t / config.nu, 1),
        "rms_x1": round(rms_1 * 100, 1),
        "rms_x2": round(rms_2 * 100, 1),
        "rms_avg": round(rms_avg * 100, 1),
        "passed": bool(rms_avg < 0.10),
    }


if __name__ == "__main__":
    results = []
    
    # Test different mixing length coefficients
    for C_ml in [0.05, 0.10, 0.15]:
        results.append(run_rans_benchmark(nx=256, ny=128, max_iter=2000, C_ml=C_ml))
    
    # Save results
    output_path = '/home/brad/TiganticLabz/Main_Projects/Project HyperTensor/HVAC_CFD/rans_result.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  {r['solver']:>25s}: {r['rms_avg']:5.1f}% (ν_t/ν={r['nu_t_ratio']:.0f}) [{status}]")
    print(f"\nResults saved to {output_path}")
