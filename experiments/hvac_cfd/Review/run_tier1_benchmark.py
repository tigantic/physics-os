#!/usr/bin/env python3
"""
Tier 1 Nielsen Benchmark — Validation Run
==========================================

Uses the existing tensornet.hvac.nielsen module to run the benchmark
against Aalborg experimental data.

Target: <10% normalized RMS error
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ontic.hvac.nielsen import (
    NielsenBenchmark,
    NielsenConfig,
    NielsenResult,
    AALBORG_DATA,
)


def run_benchmark(nx: int = 256, ny: int = 128) -> NielsenResult:
    """Run the Nielsen benchmark."""
    
    print("=" * 70)
    print("TIER 1: Nielsen Benchmark Validation")
    print("=" * 70)
    print(f"\nGrid: {nx}×{ny}")
    print("Solver: Projection Method (TVD MUSCL)")
    print()
    
    # Configure benchmark
    config = NielsenConfig(
        nx=nx,
        ny=ny,
        Re=5000,
        max_iterations=5000,
        convergence_tol=1e-6,
        verbose=True,
    )
    
    # Run benchmark
    benchmark = NielsenBenchmark(config)
    result = benchmark.run()
    
    return result


def generate_deliverables(result: NielsenResult, output_dir: Path) -> None:
    """Generate all Tier 1 deliverables."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    state = result.state
    if state is None:
        print("ERROR: No state in result")
        return
    
    # Get configuration from result
    nx = state.u.shape[0]
    ny = state.u.shape[1]
    length = 9.0  # Nielsen standard
    height = 3.0
    
    # Create coordinate grids
    x = np.linspace(0, length, nx)
    y = np.linspace(0, height, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    u_np = state.u.cpu().numpy()
    v_np = state.v.cpu().numpy()
    speed = np.sqrt(u_np**2 + v_np**2)
    
    rms_pct = result.rms_error_overall * 100
    
    # =========================================================================
    # Figure 1: Velocity Magnitude Contours
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 5))
    
    levels = np.linspace(0, 0.5, 21)
    cf = ax.contourf(X, Y, speed, levels=levels, cmap='viridis')
    plt.colorbar(cf, ax=ax, label='Velocity (m/s)')
    
    # Streamlines
    ax.streamplot(X.T, Y.T, u_np.T, v_np.T, color='white', 
                  linewidth=0.5, density=2, arrowsize=0.7)
    
    # Mark inlet and outlet
    inlet_y_start = 2.832
    outlet_y_end = 0.480
    ax.axhline(y=inlet_y_start, xmin=0, xmax=0.02, color='red', linewidth=3, label='Inlet')
    ax.axhline(y=outlet_y_end, xmin=0.98, xmax=1.0, color='blue', linewidth=3, label='Outlet')
    
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'Tier 1: Nielsen Benchmark — Velocity Field (RMS = {rms_pct:.1f}%)')
    ax.set_aspect('equal')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tier1_velocity_field.png', dpi=150)
    plt.close()
    
    # =========================================================================
    # Figure 2: Profile Comparison with Experimental Data
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Profile at x/H = 1.0
    ax1 = axes[0]
    ax1.plot(result.profile_x1["u_Uinlet"], result.profile_x1["y_H"], 
             'b-', linewidth=2, label='The Ontic Engine (TVD MUSCL)')
    ax1.plot(AALBORG_DATA["x_H_1.0"]["u_Uinlet"], AALBORG_DATA["x_H_1.0"]["y_H"],
             'ko', markersize=8, label='Nielsen Exp. Data')
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('u / U_in')
    ax1.set_ylabel('y / H')
    ax1.set_title(f'x/H = 1.0 (x = 3m)\nRMS = {result.rms_error_x1*100:.1f}%')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.3, 1.0)
    ax1.set_ylim(0, 1)
    
    # Profile at x/H = 2.0
    ax2 = axes[1]
    ax2.plot(result.profile_x2["u_Uinlet"], result.profile_x2["y_H"],
             'b-', linewidth=2, label='The Ontic Engine (TVD MUSCL)')
    ax2.plot(AALBORG_DATA["x_H_2.0"]["u_Uinlet"], AALBORG_DATA["x_H_2.0"]["y_H"],
             'ko', markersize=8, label='Nielsen Exp. Data')
    ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('u / U_in')
    ax2.set_ylabel('y / H')
    ax2.set_title(f'x/H = 2.0 (x = 6m)\nRMS = {result.rms_error_x2*100:.1f}%')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.3, 1.0)
    ax2.set_ylim(0, 1)
    
    plt.suptitle('Tier 1 Validation: The Ontic Engine vs Nielsen Experimental Data', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'tier1_profile_validation.png', dpi=150)
    plt.close()
    
    # =========================================================================
    # Figure 3: Streamlines (Flow Pattern)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Lighter background showing speed
    ax.contourf(X, Y, speed, levels=20, cmap='Blues', alpha=0.5)
    
    # Dense streamlines
    strm = ax.streamplot(X.T, Y.T, u_np.T, v_np.T, color=speed.T, 
                         cmap='plasma', linewidth=1.2, density=3, arrowsize=1)
    plt.colorbar(strm.lines, ax=ax, label='Velocity (m/s)')
    
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Flow Pattern — Ceiling Jet and Recirculation')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tier1_streamlines.png', dpi=150)
    plt.close()
    
    # =========================================================================
    # Figure 4: Breathing Zone Analysis (ASHRAE 55)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Breathing zone: y = 1.0m to 1.8m
    bz_j_start = int(1.0 / height * (ny - 1))
    bz_j_end = int(1.8 / height * (ny - 1))
    
    # Create mask for breathing zone
    bz_mask = np.zeros_like(speed)
    bz_mask[:, bz_j_start:bz_j_end] = 1.0
    
    # Show velocity in breathing zone
    bz_speed = np.ma.masked_where(bz_mask == 0, speed)
    
    cf = ax.contourf(X, Y, bz_speed, levels=np.linspace(0, 0.3, 16), cmap='RdYlGn_r')
    plt.colorbar(cf, ax=ax, label='Velocity (m/s)')
    
    # Outline the breathing zone
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='Breathing Zone')
    ax.axhline(y=1.8, color='black', linestyle='--', linewidth=1.5)
    
    # ASHRAE 55 limits
    ax.fill_between([0, length], 1.0, 1.8, alpha=0.1, color='green')
    
    bz_speeds = speed[:, bz_j_start:bz_j_end]
    bz_min = bz_speeds.min()
    bz_max = bz_speeds.max()
    bz_mean = bz_speeds.mean()
    
    ashrae_compliant = bz_min >= 0.10
    status_str = "✓ PASS" if ashrae_compliant else "✗ FAIL (< 0.10 m/s detected)"
    
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'Breathing Zone (1.0–1.8m): Min={bz_min:.3f}, Mean={bz_mean:.3f}, Max={bz_max:.3f} m/s\n'
                 f'ASHRAE 55 (≥0.10 m/s): {status_str}')
    ax.set_aspect('equal')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tier1_breathing_zone.png', dpi=150)
    plt.close()
    
    # =========================================================================
    # Analysis Summary JSON
    # =========================================================================
    summary = {
        "project": "Tier 1 Nielsen Benchmark",
        "date": datetime.now().isoformat(),
        "solver": "Ontic Projection Method (TVD MUSCL)",
        "configuration": {
            "grid": f"{nx}×{ny}",
            "domain_m": f"9.0×3.0",
            "Re": 5000,
            "inlet_velocity_m_s": 0.455,
            "inlet_height_m": 0.168,
            "advection_scheme": "TVD MUSCL (van Leer limiter)",
        },
        "performance": {
            "solve_time_s": round(result.wall_time_seconds, 1),
            "iterations": result.iterations,
            "converged": result.converged,
        },
        "validation": {
            "benchmark": "Nielsen et al. / Aalborg University",
            "rms_error_x1_pct": round(result.rms_error_x1 * 100, 2),
            "rms_error_x2_pct": round(result.rms_error_x2 * 100, 2),
            "rms_error_avg_pct": round(result.rms_error_overall * 100, 2),
            "target_pct": 10.0,
            "status": "PASS" if result.rms_passed else "FAIL",
        },
        "breathing_zone": {
            "min_velocity_m_s": round(float(bz_min), 4),
            "mean_velocity_m_s": round(float(bz_mean), 4),
            "max_velocity_m_s": round(float(bz_max), 4),
            "ashrae_55_status": "PASS" if ashrae_compliant else "FAIL",
        },
        "deliverables": [
            "tier1_velocity_field.png",
            "tier1_profile_validation.png",
            "tier1_streamlines.png",
            "tier1_breathing_zone.png",
        ],
    }
    
    with open(output_dir / 'tier1_analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print("DELIVERABLES GENERATED")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    for d in summary["deliverables"]:
        print(f"  • {d}")
    print(f"  • tier1_analysis_summary.json")
    print()


def main():
    """Main entry point."""
    
    # Run benchmark at production resolution
    result = run_benchmark(nx=256, ny=128)
    
    # Generate deliverables
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "deliverables" / f"tier1_validated_{timestamp}"
    generate_deliverables(result, output_dir)
    
    # Final status
    print("=" * 70)
    rms_pct = result.rms_error_overall * 100
    if result.benchmark_passed:
        print("🎉 TIER 1 VALIDATION: PASSED")
        print(f"   RMS Error: {rms_pct:.1f}% (target: <10%)")
    else:
        print("❌ TIER 1 VALIDATION: FAILED")
        print(f"   RMS Error: {rms_pct:.1f}% (target: <10%)")
    print("=" * 70)
    
    return result.benchmark_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
