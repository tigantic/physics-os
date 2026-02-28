#!/usr/bin/env python3
"""
Ahmed Body IB Solver — Energy Spectrum Analysis
=================================================

Runs the QTT Brinkman IB solver, reconstructs the converged velocity
field to dense, computes the 3D radially-averaged energy spectrum E(k),
and generates publication-quality diagnostic plots:

  1. Energy time series  (E vs step)
  2. E(k) spectrum with k^{-5/3} reference
  3. Compensated spectrum E(k)·k^{5/3}

Author: Brad Adams / Tigantic Holdings LLC
Date: February 2026
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import torch

# ── The Ontic Engine imports ───────────────────────────────────────────
_tools_dir = Path(__file__).resolve().parent.parent   # tools/
_repo_root = _tools_dir.parent                        # repo root
sys.path.insert(0, str(_tools_dir))
sys.path.insert(0, str(_repo_root))
from scripts.ahmed_body_ib_solver import (
    AhmedBodyParams,
    AhmedBodyConfig,
    AhmedBodyIBSolver,
    ahmed_body_sdf,
    create_body_mask,
)
from ontic.cfd.kolmogorov_spectrum import (
    compute_energy_spectrum_3d,
    fit_power_law,
    find_inertial_range,
    analyze_spectrum,
    SpectrumResult,
)


# ═══════════════════════════════════════════════════════════════════
# QTT → DENSE RECONSTRUCTION
# ═══════════════════════════════════════════════════════════════════

def qtt_field_to_dense_3d(qtt_3d, n_bits: int) -> np.ndarray:
    """
    Reconstruct a QTT3DNative scalar field to a dense (N,N,N) array.

    Performs full TT contraction then un-interleaves Morton ordering.
    """
    cores = qtt_3d.cores.cores
    n_cores = len(cores)

    # Full TT contraction: left-to-right
    result = cores[0].squeeze(0)  # (d0, r1)
    for i in range(1, n_cores):
        r_left, d, r_right = cores[i].shape
        result = result @ cores[i].reshape(r_left, d * r_right)
        result = result.reshape(-1, r_right)
    result = result.squeeze(-1)  # (N^3,)
    dense = result.detach().cpu().to(torch.float64).numpy()

    # Morton decode → Cartesian
    N = 1 << n_bits
    field_3d = np.empty((N, N, N), dtype=np.float64)
    for idx in range(N ** 3):
        x, y, z = 0, 0, 0
        for bit in range(n_bits):
            gb = bit * 3
            x |= ((idx >> gb) & 1) << bit
            y |= ((idx >> (gb + 1)) & 1) << bit
            z |= ((idx >> (gb + 2)) & 1) << bit
        field_3d[x, y, z] = dense[idx]
    return field_3d


def velocity_to_dense(solver: AhmedBodyIBSolver) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract dense (N,N,N) velocity components from solver state."""
    nb = solver.config.n_bits
    ux = qtt_field_to_dense_3d(solver.u.x, nb)
    uy = qtt_field_to_dense_3d(solver.u.y, nb)
    uz = qtt_field_to_dense_3d(solver.u.z, nb)
    return ux, uy, uz


# ═══════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════

def plot_ahmed_spectrum(
    diagnostics: List[Dict[str, Any]],
    spectrum: SpectrumResult,
    config: AhmedBodyConfig,
    save_dir: str,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
) -> str:
    """
    Generate a three-panel diagnostic plot:
      (a) Energy vs step
      (b) E(k) spectrum with k^{-5/3} reference
      (c) Compensated spectrum

    Returns the path to the saved figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(
        f"Ahmed Body IB — QTT Volumetric Synthesis  "
        f"({config.N}³, rank {config.max_rank}, "
        f"Re_eff {config.Re_eff:.0f})",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # ── (a) Energy time series ──────────────────────────────────
    ax = axes[0]
    steps = [d["step"] for d in diagnostics]
    energies = [d["energy"] for d in diagnostics]
    clamped = [d.get("clamped", False) for d in diagnostics]
    E0 = energies[0]

    ax.plot(steps, energies, "b-", linewidth=1.5, label="$E_k$")
    clamp_steps = [s for s, c in zip(steps, clamped) if c]
    clamp_E = [e for e, c in zip(energies, clamped) if c]
    if clamp_steps:
        ax.scatter(clamp_steps, clamp_E, color="red", s=18, zorder=5,
                   label=f"E-clamp ({len(clamp_steps)})")
    ax.axhline(E0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
               label=f"$E_0$ = {E0:.3e}")
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Kinetic Energy", fontsize=11)
    ax.set_title("(a) Energy Evolution", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── (b) Energy spectrum ─────────────────────────────────────
    ax = axes[1]
    k = spectrum.wavenumbers
    Ek = spectrum.spectrum
    valid = (k > 0) & (Ek > 0)
    kv, Ev = k[valid], Ek[valid]

    ax.loglog(kv, Ev, "b-", linewidth=2, label="$E(k)$")

    # k^{-5/3} reference line
    k_ref = kv
    E_ref_scale = spectrum.fitted_prefactor if spectrum.fitted_prefactor > 0 else np.max(Ev)
    if spectrum.fitted_prefactor > 0:
        E_ref = spectrum.fitted_prefactor * k_ref ** (-5 / 3)
    else:
        E_ref = Ev[len(Ev) // 4] * (k_ref / kv[len(Ev) // 4]) ** (-5 / 3)
    ax.loglog(k_ref, E_ref, "r--", linewidth=1.5,
              label=f"$k^{{-5/3}}$  (fit: $k^{{{spectrum.fitted_exponent:.2f}}}$)")

    # Inertial range shading
    k_lo = k[spectrum.inertial_range[0]]
    k_hi = k[spectrum.inertial_range[1]]
    ax.axvspan(k_lo, k_hi, alpha=0.15, color="green", label="Inertial range")

    ax.set_xlabel("Wavenumber $k$", fontsize=11)
    ax.set_ylabel("$E(k)$", fontsize=11)
    ax.set_title("(b) Energy Spectrum", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    ax.yaxis.set_minor_locator(LogLocator(subs="all", numticks=20))
    ax.yaxis.set_minor_formatter(NullFormatter())

    # ── (c) Compensated spectrum ────────────────────────────────
    ax = axes[2]
    comp = spectrum.compensated
    comp_valid = comp[valid]
    ax.semilogx(kv, comp_valid, "b-", linewidth=2)

    # Mean in inertial range
    ir0, ir1 = spectrum.inertial_range
    ir_vals = comp[ir0:ir1 + 1]
    ir_vals = ir_vals[ir_vals > 0]
    if len(ir_vals) > 0:
        mean_comp = np.mean(ir_vals)
        ax.axhline(mean_comp, color="red", linestyle="--", linewidth=1.2,
                   label=f"Mean = {mean_comp:.2e}")
    ax.axvspan(k_lo, k_hi, alpha=0.15, color="green")

    ax.set_xlabel("Wavenumber $k$", fontsize=11)
    ax.set_ylabel("$E(k) \\cdot k^{5/3}$", fontsize=11)
    ax.set_title("(c) Compensated Spectrum", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / "ahmed_body_spectrum.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_velocity_slices(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    mask_dense: np.ndarray,
    config: AhmedBodyConfig,
    save_dir: str,
) -> str:
    """
    Mid-plane velocity magnitude and streamwise velocity with body outline.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    N = config.N
    L = config.L
    dx = config.dx
    umag = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)
    U_inf = config.body_params.velocity

    # Mid-z slice
    iz_mid = N // 2
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        f"Ahmed Body IB — Velocity Slices  "
        f"({N}³, rank {config.max_rank}, z = L/2)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    extent = [0, L, 0, L]

    # (a) |u| magnitude
    ax = axes[0]
    im = ax.imshow(
        umag[:, :, iz_mid].T, origin="lower", extent=extent,
        cmap="inferno", norm=Normalize(0, U_inf * 1.5),
    )
    # Body contour
    ax.contour(
        mask_dense[:, :, iz_mid].T, levels=[0.5], colors="white",
        linewidths=1.5, extent=extent,
    )
    fig.colorbar(im, ax=ax, label="$|u|$ [m/s]", shrink=0.85)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("(a) Velocity Magnitude $|u|$")
    ax.set_aspect("equal")

    # (b) u_x streamwise
    ax = axes[1]
    im = ax.imshow(
        ux[:, :, iz_mid].T, origin="lower", extent=extent,
        cmap="RdBu_r", norm=Normalize(-U_inf * 0.3, U_inf * 1.3),
    )
    ax.contour(
        mask_dense[:, :, iz_mid].T, levels=[0.5], colors="black",
        linewidths=1.5, extent=extent,
    )
    fig.colorbar(im, ax=ax, label="$u_x$ [m/s]", shrink=0.85)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("(b) Streamwise Velocity $u_x$")
    ax.set_aspect("equal")

    plt.tight_layout()

    out_dir = Path(save_dir)
    out_path = str(out_dir / "ahmed_body_velocity_slices.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser("Ahmed Body Spectrum Analysis")
    ap.add_argument("--n-bits", type=int, default=7)
    ap.add_argument("--max-rank", type=int, default=48)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfl", type=float, default=0.08)
    ap.add_argument("--domain-size", type=float, default=4.0)
    ap.add_argument("--eta", type=float, default=1e-3)
    ap.add_argument("--results-dir", type=str, default="./ahmed_ib_results")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    bp = AhmedBodyParams()
    cfg = AhmedBodyConfig(
        n_bits=args.n_bits, max_rank=args.max_rank, L=args.domain_size,
        body_params=bp, eta_brinkman=args.eta, cfl=args.cfl,
        n_steps=args.steps, results_dir=args.results_dir, device=args.device,
    )

    print("=" * 72)
    print("  AHMED BODY IB — SPECTRUM ANALYSIS")
    print("=" * 72)
    print(f"  Grid: {cfg.N}³   rank: {cfg.max_rank}   CFL: {cfg.cfl}")
    print(f"  Re_phys: {bp.Re:.0f}   Re_eff: {cfg.Re_eff:.0f}")
    print()

    # ── Step 1: Run solver ──────────────────────────────────────
    print("─" * 72)
    print("STEP 1 — SOLVER RUN")
    print("─" * 72)
    t0 = time.perf_counter()
    solver = AhmedBodyIBSolver(cfg)
    diagnostics = solver.run(verbose=True)
    run_time = time.perf_counter() - t0
    print(f"  Wall time: {run_time:.1f} s")

    # ── Step 2: Reconstruct dense velocity ──────────────────────
    print()
    print("─" * 72)
    print("STEP 2 — DENSE RECONSTRUCTION")
    print("─" * 72)
    t0 = time.perf_counter()
    ux, uy, uz = velocity_to_dense(solver)
    recon_time = time.perf_counter() - t0
    N = cfg.N
    print(f"  Reconstructed {N}³ velocity → ({N},{N},{N}) × 3 components")
    print(f"  ux: [{ux.min():.3f}, {ux.max():.3f}]  "
          f"uy: [{uy.min():.3f}, {uy.max():.3f}]  "
          f"uz: [{uz.min():.3f}, {uz.max():.3f}]")
    print(f"  Reconstruction time: {recon_time:.1f} s")

    # ── Step 3: Spectrum analysis ───────────────────────────────
    print()
    print("─" * 72)
    print("STEP 3 — ENERGY SPECTRUM")
    print("─" * 72)
    t0 = time.perf_counter()

    # Subtract freestream for fluctuation spectrum
    ux_fluct = ux - bp.velocity
    uy_fluct = uy
    uz_fluct = uz

    k, Ek = compute_energy_spectrum_3d(ux_fluct, uy_fluct, uz_fluct, L=cfg.L)
    spectrum = analyze_spectrum(k, Ek, nu=cfg.nu_eff)
    spec_time = time.perf_counter() - t0

    print(f"  k_max = {k[-1]:.1f}   dk = {k[1]-k[0]:.3f}")
    print(f"  Inertial range: k in [{k[spectrum.inertial_range[0]]:.2f}, "
          f"{k[spectrum.inertial_range[1]]:.2f}]")
    print(f"  Fitted exponent: alpha = {spectrum.fitted_exponent:.4f}"
          f"  (Kolmogorov: {-5/3:.4f})")
    print(f"  R-squared:       {spectrum.r_squared:.4f}")
    print(f"  Kolmogorov eta:  {spectrum.kolmogorov_length:.6f}")
    print(f"  Integral length: {spectrum.integral_length:.4f}")
    print(f"  Spectrum time:   {spec_time:.1f} s")

    # ── Step 4: Plots ───────────────────────────────────────────
    print()
    print("─" * 72)
    print("STEP 4 — PLOTTING")
    print("─" * 72)

    p1 = plot_ahmed_spectrum(
        diagnostics, spectrum, cfg, args.results_dir,
        ux_fluct, uy_fluct, uz_fluct,
    )
    print(f"  Spectrum plot: {p1}")

    # Reconstruct body mask from SDF (no dense arrays stored on solver).
    cfg = solver.config
    bp = cfg.body_params
    x1d = np.linspace(0, cfg.L * (cfg.N - 1) / cfg.N, cfg.N)
    X, Y, Z = np.meshgrid(x1d, x1d, x1d, indexing="ij")
    sdf = ahmed_body_sdf(X, Y, Z, bp, solver.body_center)
    mask_dense = create_body_mask(sdf, cfg.dx, 2.0)
    p2 = plot_velocity_slices(ux, uy, uz, mask_dense, cfg, args.results_dir)
    print(f"  Velocity plot: {p2}")

    # ── Step 5: Save spectrum data ──────────────────────────────
    rd = Path(args.results_dir)
    spec_data = {
        "k": k.tolist(),
        "E_k": Ek.tolist(),
        "compensated": spectrum.compensated.tolist(),
        "fitted_exponent": float(spectrum.fitted_exponent),
        "fitted_prefactor": float(spectrum.fitted_prefactor),
        "r_squared": float(spectrum.r_squared),
        "inertial_range_k": [float(k[spectrum.inertial_range[0]]),
                              float(k[spectrum.inertial_range[1]])],
        "kolmogorov_length": float(spectrum.kolmogorov_length),
        "integral_length": float(spectrum.integral_length),
    }
    spec_path = rd / "spectrum_data.json"
    with open(spec_path, "w") as f:
        json.dump(spec_data, f, indent=2)
    print(f"  Spectrum data:  {spec_path}")

    print()
    print("=" * 72)
    print("  COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
