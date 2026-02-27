#!/usr/bin/env python3
"""1024³ Chu Limit — QTT-Native Physics-Valid Run.

First grid level where PML clearance (0.69λ) is adequate for
a clean far-field proxy.  ~145k design voxels, 32.6 cells
per antenna radius.

ALL operations stay in compressed QTT format:
  - Forward solve via QTT ε + MPO Helmholtz
  - Power via QTT inner products (no reconstruct)
  - Adjoint RHS via QTT Hadamard products
  - Adjoint solve in QTT
  - Gradient extraction via tt_evaluate_at_indices

The ONLY dense objects are 1D arrays of length n_design (~145k).
Peak memory: O(N · r²) ≈ O(1024 · 128²) ≈ 16 MB per core chain.

Phase 1: Single forward solve + gradient diagnostic.
Phase 2: 10-iteration smoke test if gradient is viable.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from tensornet.em.chu_limit import (
    ChuOptConfig,
    AntennaGeometry3D,
    make_chu_antenna_schedule,
    optimize_chu_antenna,
    compute_pml_power_tt,
    compute_adjoint_gradient_power_tt,
    solve_forward_conductivity_tt,
    PowerObjectiveConfig,
)
from tensornet.em.qtt_3d import (
    build_pml_sigma_tt,
    build_sphere_mask_tt,
)


def run_1024_smoke() -> None:
    """Phase 1: Single forward+gradient, Phase 2: 10-iter optimization."""
    print("=" * 70)
    print("  1024³ CHU LIMIT — QTT-NATIVE SMOKE TEST")
    print("=" * 70)
    sys.stdout.flush()

    cfg, opt = make_chu_antenna_schedule("1024")

    # Smoke-test overrides: gentle ramps, 10 iterations
    opt_smoke = ChuOptConfig(
        max_iterations=10,
        learning_rate=opt.learning_rate,
        optimizer=opt.optimizer,
        adam_beta1=opt.adam_beta1,
        adam_beta2=opt.adam_beta2,
        adam_eps=opt.adam_eps,
        beta_init=opt.beta_init,
        beta_max=8.0,
        beta_increase_every=opt.beta_increase_every,
        use_power_adjoint=True,
        alpha_loss=opt.alpha_loss,
        alpha_intro_mode="auto",
        alpha_stable_window=5,
        vol_target=opt.vol_target,
        al_mu_init=opt.al_mu_init,
        feed_seed_clamp_iters=5,
        feed_seed_clamp_radius=opt.feed_seed_clamp_radius,
        use_coupling_constraint=True,
        coupling_density_threshold=opt.coupling_density_threshold,
        coupling_radius=opt.coupling_radius,
        sigma_max_init=opt.sigma_max_init,
        sigma_max_final=200.0,
        sigma_ramp_iters=10,
        simp_p_init=opt.simp_p_init,
        simp_p_final=2.0,
        simp_p_ramp_iters=10,
        filter_radius=opt.filter_radius,
    )

    print(cfg.summary())
    print(f"\n  Smoke test: {opt_smoke.max_iterations} iterations")
    print(f"  Optimizer:   {opt_smoke.optimizer}")
    print(f"  lr:          {opt_smoke.learning_rate}")
    print(f"  Max rank:    {cfg.max_rank}")
    print(f"  n_sweeps:    {cfg.n_sweeps}")
    print(f"  Solver tol:  {cfg.solver_tol}")
    print(f"  Filter rad:  {opt_smoke.filter_radius}")
    sys.stdout.flush()

    # ----------------------------------------------------------------
    # Phase 1: Single forward solve + gradient diagnostic
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PHASE 1: QTT-native forward solve + gradient diagnostic")
    print("=" * 70)
    sys.stdout.flush()

    geometry = AntennaGeometry3D.with_monopole_seed(cfg)
    print(f"  Design voxels: {geometry.n_design:,}")
    print(f"  Feed position: {geometry.feed_position}")
    sys.stdout.flush()

    k0_norm = cfg.k0_normalised
    pml = cfg.pml_config()

    # Build QTT infrastructure (one-time, no dense N³)
    print("\n  Building QTT infrastructure...")
    sys.stdout.flush()
    t0 = time.perf_counter()

    sigma_pml_tt = build_pml_sigma_tt(
        cfg.n_bits, k0_norm, pml, max_rank=cfg.max_rank,
    )
    r_pml = max(c.shape[2] for c in sigma_pml_tt)
    print(f"    σ_pml QTT rank: {r_pml}")
    sys.stdout.flush()

    design_mask_tt = build_sphere_mask_tt(
        cfg.n_bits,
        centre=(0.5, 0.5, 0.5),
        radius=cfg.sphere_radius_normalised,
        max_rank=min(cfg.max_rank, 64),
    )
    r_mask = max(c.shape[2] for c in design_mask_tt)
    print(f"    Design mask QTT rank: {r_mask}")

    design_flat_idx = geometry.design_flat_indices
    t_infra = time.perf_counter() - t0
    print(f"    Built in {t_infra:.1f}s")
    sys.stdout.flush()

    # Forward solve
    print("\n  Forward solve (QTT-native)...")
    sys.stdout.flush()
    t0 = time.perf_counter()

    H_cores, E_cores, residual, eps_tt = solve_forward_conductivity_tt(
        geometry=geometry, k0_norm=k0_norm,
        design_mask_tt=design_mask_tt,
        beta=1.0, eta=0.5, filter_radius=opt_smoke.filter_radius,
        max_rank=cfg.max_rank, n_sweeps=cfg.n_sweeps,
        solver_tol=cfg.solver_tol, damping=cfg.damping,
        verbose=True,
    )
    t_solve = time.perf_counter() - t0
    r_E = max(c.shape[2] for c in E_cores)
    print(f"  Solve: {t_solve:.1f}s, residual={residual:.4e}, E rank={r_E}")
    sys.stdout.flush()

    # P_pml via QTT inner product
    h = 1.0 / cfg.N
    dv = h ** 3
    P_pml = compute_pml_power_tt(E_cores, sigma_pml_tt, dv, cfg.max_rank)
    print(f"  P_pml (QTT): {P_pml:.4e}")
    sys.stdout.flush()

    # Adjoint gradient
    print("\n  Adjoint gradient (QTT-native)...")
    sys.stdout.flush()
    t0 = time.perf_counter()

    obj_cfg = PowerObjectiveConfig(alpha_loss=0.0)
    J_val, grad, metrics, adj_res = compute_adjoint_gradient_power_tt(
        geometry=geometry, k0_norm=k0_norm, pml=pml,
        obj_cfg=obj_cfg,
        design_mask_tt=design_mask_tt,
        sigma_pml_tt=sigma_pml_tt,
        design_flat_indices=design_flat_idx,
        beta=1.0, eta_h=0.5,
        filter_radius=opt_smoke.filter_radius,
        max_rank=cfg.max_rank,
        n_sweeps=cfg.n_sweeps,
        solver_tol=cfg.solver_tol,
        damping=cfg.damping,
        verbose=True,
    )
    t_grad = time.perf_counter() - t0
    print(f"  Gradient: {t_grad:.1f}s")
    print(f"  J={J_val:.6f}, P_pml={metrics.P_pml:.4e}, "
          f"P_cond={metrics.P_cond:.4e}")
    sys.stdout.flush()

    # Spatial analysis
    rho = geometry.density
    wire = rho > 0.3
    air = rho < 0.1

    print(f"\n  Gradient analysis:")
    print(f"    |grad|:     {np.linalg.norm(grad):.4e}")
    print(f"    range:      [{grad.min():.4e}, {grad.max():.4e}]")
    print(f"    std:        {np.std(grad):.4e}")
    if np.sum(wire) > 0:
        print(f"    Wire ({np.sum(wire)}): "
              f"[{grad[wire].min():.4e}, {grad[wire].max():.4e}]")
    if np.sum(air) > 0:
        print(f"    Air ({np.sum(air)}):  "
              f"std={np.std(grad[air]):.4e}, "
              f"range=[{grad[air].min():.4e}, {grad[air].max():.4e}]")
    sys.stdout.flush()

    air_std = np.std(grad[air]) if np.sum(air) > 0 else 0.0
    if air_std < 1e-15:
        print("\n  *** FAIL: No gradient spatial signal ***")
        sys.stdout.flush()
        return

    print("\n  PASS: Gradient has spatial variation")
    sys.stdout.flush()

    # ----------------------------------------------------------------
    # Phase 2: 10-iteration QTT-native optimization
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PHASE 2: 10-iteration QTT-native optimization")
    print("=" * 70)
    sys.stdout.flush()

    result = optimize_chu_antenna(
        config=cfg,
        opt_config=opt_smoke,
        verbose=True,
        extract_q=False,
        qtt_native=True,
    )

    print(result.summary())

    if result.power_metrics_history:
        print(f"\n  ~P trajectory:")
        for i, m in enumerate(result.power_metrics_history):
            print(f"    iter {i+1:3d}: ~P={m.P_pml_norm:.4f}")

        P0 = result.power_metrics_history[0].P_pml_norm
        Pf = result.power_metrics_history[-1].P_pml_norm
        ratio = Pf / max(P0, 1e-30)
        print(f"    {P0:.4f} → {Pf:.4f} ({ratio:.2f}×)")

        if ratio > 1.5:
            print("  PASS: ~P reversal — physics-valid scale WORKS")
        elif ratio > 1.1:
            print("  MODERATE: ~P increasing — needs more iterations")
        else:
            print("  STALLED: ~P flat")

    print("\n" + "=" * 70)
    print("  1024³ SMOKE TEST COMPLETE")
    print("=" * 70)
    sys.stdout.flush()


if __name__ == "__main__":
    run_1024_smoke()
