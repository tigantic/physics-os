#!/usr/bin/env python3
"""PWA Engine — Full Experiment Suite & Publication Figures.

Single command that runs every deliverable:
    1.  Convention reduction test (general → simplified)
    2.  One-bin end-to-end parameter recovery
    3.  Baseline vs Gram-accelerated normalization benchmark
    4.  Wave-set scan with robustness atlas
    5.  QTT compression of Gram matrix at scale
    6.  Angular moment validation (data vs model)
    7.  Beam asymmetry sensitivity (polarized vs unpolarized)
    8.  Bootstrap uncertainty quantification
    9.  Coupled-channel joint fit
    10. Mass-dependent Breit–Wigner extraction
    11. Publication-quality figures (600 DPI, JCP style)

Usage:
    python3 experiments/run_pwa_engine.py

Authority: Adams (2026), physics-os Platform V3.0.0
Hardware:  NVIDIA GeForce RTX 5070 Laptop GPU
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

# ─── matplotlib setup ────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.7",
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 5,
    "axes.grid": False,
})

C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_RED = "#D55E00"
C_PURPLE = "#CC79A7"
C_CYAN = "#56B4E9"
C_BLACK = "#000000"
C_GREY = "#999999"

SINGLE_COL = (3.5, 2.8)
DOUBLE_COL = (7.0, 3.5)

# ─── engine imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.pwa_engine.core import (
    BasisAmplitudes,
    BreitWigner,
    ExtendedLikelihood,
    GramMatrix,
    IntensityModel,
    LBFGSFitter,
    SyntheticDataGenerator,
    WaveSet,
    benchmark_normalization,
    build_wave_set,
    compress_gram_qtt,
    convention_reduction_test,
    coupled_channel_test,
    mass_dependent_fit,
    moment_comparison,
    beam_asymmetry_sensitivity_test,
    bootstrap_uncertainty,
    wave_set_scan,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"


# ════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: CONVENTION REDUCTION TEST
# ════════════════════════════════════════════════════════════════════════════════

def run_convention_test(device: torch.device) -> Dict[str, Any]:
    """Deliverable 1: prove the general model reduces to the simplified form."""
    print("=" * 70)
    print("EXPERIMENT 1: Convention Reduction Test")
    print("-" * 70)

    result = convention_reduction_test(device=device, n_events=1000, seed=42)

    print(f"  Test 1 (full with ε=-1 killed → single ε):   err = {result['test_1_full_vs_single_eps']:.2e}")
    print(f"  Test 2 (IntensityModel → manual |ΣVψ|²):     err = {result['test_2_model_vs_manual']:.2e}")
    print(f"  Test 3 (full ρ matrix → diagonal ρ):          err = {result['test_3_full_rho_vs_diagonal']:.2e}")
    print(f"  All pass (< 1e-12): {'YES' if result['all_pass'] else 'NO'}")
    print(f"  Amplitudes: full={result['n_amp_full']}, single={result['n_amp_single']}")

    return result


# ════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: ONE-BIN END-TO-END PARAMETER RECOVERY
# ════════════════════════════════════════════════════════════════════════════════

def run_parameter_recovery(device: torch.device) -> Dict[str, Any]:
    """Deliverable 2: generate synthetic data, fit, recover known parameters."""
    print()
    print("=" * 70)
    print("EXPERIMENT 2: One-Bin End-to-End Parameter Recovery")
    print("-" * 70)

    # True model: J up to 5/2, single reflectivity, 12 amplitudes
    true_j_max = 2.5
    ws_true = build_wave_set(true_j_max, reflectivities=(+1,))
    n_amp = ws_true.n_amplitudes

    # True parameters: realistic amplitude pattern
    rng = np.random.default_rng(42)
    magnitudes = rng.exponential(scale=1.0, size=n_amp)
    phases = rng.uniform(-np.pi, np.pi, size=n_amp)
    V_true_np = magnitudes * np.exp(1j * phases)
    V_true_np /= np.linalg.norm(V_true_np)  # normalize

    print(f"  True model: J_max={true_j_max}, n_amp={n_amp}")
    print(f"  True yields: {(np.abs(V_true_np)**2).round(4)}")

    # Generate synthetic data
    n_data = 10000
    n_generated = 500000
    gen = SyntheticDataGenerator(ws_true, V_true_np, seed=42)
    data = gen.generate(n_data, n_generated, device=device)

    print(f"  Data events:    {data['n_data']}")
    print(f"  MC accepted:    {data['n_mc_accepted']}")
    print(f"  MC generated:   {data['n_generated']}")
    print(f"  Acceptance rate: {data['acceptance_rate']:.3f}")

    # Build engine components
    basis_data = BasisAmplitudes(ws_true, data["theta_data"], data["phi_data"], device=device)
    basis_mc = BasisAmplitudes(ws_true, data["theta_mc"], data["phi_mc"], device=device)

    model_data = IntensityModel(basis_data)
    model_mc = IntensityModel(basis_mc)
    gram = GramMatrix(basis_mc, data["n_generated"])

    # Likelihood with Gram acceleration
    nll_gram = ExtendedLikelihood(
        model_data, None, gram, data["n_data"], data["n_generated"], use_gram=True
    )
    # Likelihood without Gram (baseline)
    nll_baseline = ExtendedLikelihood(
        model_data, model_mc, None, data["n_data"], data["n_generated"], use_gram=False
    )

    # Multi-start fit with Gram
    print("\n  Fitting (Gram-accelerated, 40 starts)...")
    fitter_gram = LBFGSFitter(nll_gram, max_iter=500)
    ms_gram = fitter_gram.multi_start_fit(n_starts=40, seed_base=100)

    V_best = ms_gram["best_fit"]["V_best"]
    V_best_np = V_best.detach().cpu().numpy()

    # The fit rescales V to satisfy N̄(V*) = n_data (property of extended likelihood).
    # Physical observables are RELATIVE yields and phase differences.
    # Normalize both to unit total yield, then phase-align.
    yield_true_raw = np.abs(V_true_np) ** 2
    yield_fit_raw = np.abs(V_best_np) ** 2
    yield_true = yield_true_raw / yield_true_raw.sum()   # relative fractions
    yield_fit = yield_fit_raw / yield_fit_raw.sum()

    # Phase-align: fix phase of the dominant amplitude
    idx_max = np.argmax(yield_true)
    phase_shift = np.exp(1j * (np.angle(V_true_np[idx_max]) - np.angle(V_best_np[idx_max])))
    V_aligned = V_best_np * phase_shift

    yield_rmse = np.sqrt(np.mean((yield_true - yield_fit) ** 2))
    phase_diff = np.angle(V_true_np * V_aligned.conj())
    phase_rmse = np.sqrt(np.mean(phase_diff**2))

    # Normalization diagnostic
    V_torch = torch.tensor(V_true_np, dtype=torch.complex128, device=device)
    N_true = gram.normalization(V_torch, data["n_data"]).item()

    V_best_tensor = ms_gram["best_fit"]["V_best"]
    N_fit = gram.normalization(V_best_tensor, data["n_data"]).item()

    # Hessian and covariance at minimum
    print("  Computing covariance (Hessian at minimum)...")
    try:
        cov = fitter_gram.covariance(V_best)
        param_errors = torch.sqrt(torch.diag(cov).clamp(min=0)).cpu().numpy()
        cov_ok = True
    except Exception as e:
        param_errors = np.zeros(2 * n_amp)
        cov_ok = False
        print(f"  Warning: Hessian computation failed — {e}")

    print(f"\n  --- RESULTS ---")
    print(f"  Best NLL:        {ms_gram['best_nll']:.2f}")
    print(f"  Basin fraction:  {ms_gram['basin_fraction']:.0%} ({ms_gram['n_near_best']}/{ms_gram['n_starts']})")
    print(f"  Yield RMSE:      {yield_rmse:.6f}")
    print(f"  Phase RMSE:      {phase_rmse:.4f} rad")
    print(f"  Normalization:   N_true={N_true:.2f}, N_fit={N_fit:.2f}")
    print(f"  Fit time:        {ms_gram['best_fit']['time_s']:.2f}s")
    print(f"  Covariance:      {'OK' if cov_ok else 'FAILED'}")

    # Goodness: simplified χ² from relative yield comparison
    if cov_ok:
        # Scale covariance for relative yields
        cov_diag = param_errors[:n_amp]
        chi2 = float(np.sum(((yield_true - yield_fit) / (np.maximum(yield_true, 1e-6))) ** 2))
    else:
        chi2 = float(np.sum(((yield_true - yield_fit) / (np.maximum(yield_true, 1e-6))) ** 2))

    print(f"  Yield χ²/ndf:    {chi2:.2f} / {n_amp}")

    return {
        "V_true": V_true_np,
        "V_fit": V_aligned,
        "yield_true": yield_true,
        "yield_fit": yield_fit,
        "yield_rmse": yield_rmse,
        "phase_diff": phase_diff,
        "phase_rmse": phase_rmse,
        "best_nll": ms_gram["best_nll"],
        "basin_fraction": ms_gram["basin_fraction"],
        "N_true": N_true,
        "N_fit": N_fit,
        "chi2": chi2,
        "n_amp": n_amp,
        "n_data": data["n_data"],
        "n_generated": data["n_generated"],
        "n_mc_accepted": data["n_mc_accepted"],
        "fit_time_s": ms_gram["best_fit"]["time_s"],
        "n_starts": ms_gram["n_starts"],
        "all_fits": ms_gram["all_fits"],
        "cov_ok": cov_ok,
        "param_errors": param_errors if cov_ok else None,
        "wave_labels": [w.label for w in ws_true.waves],
    }


# ════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: BASELINE vs GRAM ACCELERATION
# ════════════════════════════════════════════════════════════════════════════════

def run_acceleration_benchmark(device: torch.device) -> Dict[str, Any]:
    """Deliverable 3: benchmark baseline vs Gram normalization speed."""
    print()
    print("=" * 70)
    print("EXPERIMENT 3: Normalization Acceleration Benchmark")
    print("-" * 70)

    results: List[Dict[str, Any]] = []

    configs = [
        (1000, 12),
        (5000, 12),
        (10000, 12),
        (50000, 12),
        (100000, 12),
        (10000, 6),
        (10000, 20),
        (10000, 42),
    ]

    for n_mc, n_amp_target in configs:
        # Find j_max that gives approximately n_amp_target amplitudes
        j_max = 0.5
        while True:
            ws = build_wave_set(j_max, reflectivities=(+1,))
            if ws.n_amplitudes >= n_amp_target:
                break
            j_max += 1.0
        ws = build_wave_set(j_max, reflectivities=(+1,))

        bench = benchmark_normalization(
            ws, n_mc, n_generated=n_mc * 10, n_evals=100, device=device
        )
        results.append(bench)

        print(
            f"  n_MC={n_mc:>7,}  n_amp={bench['n_amplitudes']:>3}  "
            f"baseline={bench['baseline_per_eval_ms']:>8.3f}ms  "
            f"gram={bench['gram_per_eval_ms']:>8.3f}ms  "
            f"speedup={bench['speedup']:>7.1f}×  "
            f"agree={bench['relative_agreement']:.1e}"
        )

    return {"benchmarks": results}


# ════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: WAVE-SET SCAN & ROBUSTNESS ATLAS
# ════════════════════════════════════════════════════════════════════════════════

def run_wave_scan(device: torch.device) -> Dict[str, Any]:
    """Deliverable 4: systematic wave-set scan with stability diagnostics."""
    print()
    print("=" * 70)
    print("EXPERIMENT 4: Wave-Set Scan & Robustness Atlas")
    print("-" * 70)

    # True model: J_max = 2.5 (12 amplitudes)
    true_j_max = 2.5
    ws_true = build_wave_set(true_j_max, reflectivities=(+1,))
    rng = np.random.default_rng(42)
    n_true = ws_true.n_amplitudes
    V_true = rng.standard_normal(n_true) + 1j * rng.standard_normal(n_true)
    V_true /= np.linalg.norm(V_true)

    j_max_values = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]

    result = wave_set_scan(
        j_max_values=j_max_values,
        V_true=V_true,
        true_j_max=true_j_max,
        n_data=5000,
        n_generated=200000,
        n_starts=10,
        device=device,
        seed=42,
    )

    print(f"\n  {'J_max':>5}  {'n_amp':>5}  {'NLL':>10}  {'Basin%':>7}  {'σ_param':>8}  {'G_rank':>6}")
    print(f"  {'-'*5}  {'-'*5}  {'-'*10}  {'-'*7}  {'-'*8}  {'-'*6}")
    for r in result["scan_results"]:
        print(
            f"  {r['j_max']:>5.1f}  {r['n_amplitudes']:>5}  {r['best_nll']:>10.1f}  "
            f"{r['basin_fraction']:>6.0%}  {r['param_std']:>8.4f}  {r['gram_rank_1e8']:>6}"
        )

    return result


# ════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5: QTT COMPRESSION OF GRAM MATRIX
# ════════════════════════════════════════════════════════════════════════════════

def run_gram_qtt(device: torch.device) -> Dict[str, Any]:
    """Deliverable 3-bonus: QTT compression of Gram matrix at scale."""
    print()
    print("=" * 70)
    print("EXPERIMENT 5: QTT Compression of Gram Matrix")
    print("-" * 70)

    results: List[Dict[str, Any]] = []
    rng = np.random.default_rng(42)

    for j_max in [1.5, 2.5, 3.5, 4.5, 5.5]:
        ws = build_wave_set(j_max, reflectivities=(+1,))
        n_amp = ws.n_amplitudes

        # Generate MC events for Gram
        n_mc = 50000
        theta = np.pi * rng.uniform(size=n_mc)
        phi = 2.0 * np.pi * rng.uniform(size=n_mc)
        basis = BasisAmplitudes(ws, theta, phi, device=device)
        gram = GramMatrix(basis, n_mc * 10)

        # QTT compress
        qtt_result = compress_gram_qtt(gram, max_rank=64, tol=1e-10)
        results.append(qtt_result)

        print(
            f"  J_max={j_max:>4.1f}  n_amp={n_amp:>3}  "
            f"χ_max={qtt_result['chi_max']:>3}  "
            f"CR={qtt_result['compression_ratio']:>6.1f}×  "
            f"G_rank={int(np.sum(qtt_result['gram_svd'] > qtt_result['gram_svd'][0]*1e-8)):>3}"
        )

    return {"gram_qtt_results": results}


# ════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 6: ANGULAR MOMENT VALIDATION
# ════════════════════════════════════════════════════════════════════════════════

def run_moment_validation(
    recov_result: Dict[str, Any], device: torch.device
) -> Dict[str, Any]:
    """Deliverable 6: moment-based goodness-of-fit diagnostic."""
    print()
    print("=" * 70)
    print("EXPERIMENT 6: Angular Moment Validation")
    print("-" * 70)

    # Re-use the fitted amplitudes and data from experiment 2
    ws = build_wave_set(2.5, reflectivities=(+1,))
    V_fit_np = recov_result["V_fit"]
    V_fit = torch.tensor(V_fit_np, dtype=torch.complex128, device=device)

    # We need data and MC kinematics; regenerate with same seed
    rng = np.random.default_rng(42)
    n_amp = ws.n_amplitudes
    magnitudes = rng.exponential(scale=1.0, size=n_amp)
    phases = rng.uniform(-np.pi, np.pi, size=n_amp)
    V_true_np = magnitudes * np.exp(1j * phases)
    V_true_np /= np.linalg.norm(V_true_np)

    from experiments.pwa_engine.core import SyntheticDataGenerator as _SDG
    gen = _SDG(ws, V_true_np, seed=42)
    data = gen.generate(10000, 500000, device=device)

    result = moment_comparison(
        data["theta_data"], data["phi_data"],
        data["theta_mc"], data["phi_mc"],
        V_fit, ws, data["n_generated"],
        L_max=6, device=device,
    )

    print(f"  L_max = {result['L_max']}")
    print(f"  Moments computed: {result['ndf']}")
    print(f"  χ²/ndf = {result['chi2_per_ndf']:.2f}")
    print(f"  σ (stat) = {result['sigma']:.4f}")
    print()

    # Print top-10 pulls
    pulls_sorted = sorted(result["pulls"].items(), key=lambda x: x[1], reverse=True)
    print(f"  {'(L,M)':>8}  {'|pull|':>8}  {'Re(data)':>10}  {'Re(model)':>10}")
    for (L, M), pull in pulls_sorted[:10]:
        yd = result["moments_data"][(L, M)]
        ym = result["moments_model"][(L, M)]
        print(f"  ({L},{M:+d})    {pull:>8.2f}  {yd.real:>10.5f}  {ym.real:>10.5f}")

    return result


# ════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 7: BEAM ASYMMETRY SENSITIVITY
# ════════════════════════════════════════════════════════════════════════════════

def run_beam_asymmetry(device: torch.device) -> Dict[str, Any]:
    """Deliverable 7: polarization observable breaks phase ambiguities."""
    print()
    print("=" * 70)
    print("EXPERIMENT 7: Beam Asymmetry Sensitivity Test")
    print("-" * 70)

    result = beam_asymmetry_sensitivity_test(
        n_events=5000,
        n_generated=200_000,
        n_starts=20,
        device=device,
        seed=42,
    )

    print(f"  Model: J_max=1.5, ε=±1, n_amp={result['n_amp']}")
    print(f"  Data:  {result['n_data']} events, {result['n_generated']} gen")
    print()
    print(f"  {'Metric':<24}  {'Unpolarized':>12}  {'Polarized':>12}  {'Ratio':>8}")
    print(f"  {'-'*24}  {'-'*12}  {'-'*12}  {'-'*8}")
    print(f"  {'Phase RMSE (rad)':<24}  {result['phase_rmse_unpol']:>12.4f}  "
          f"{result['phase_rmse_pol']:>12.4f}  {result['phase_improvement']:>7.1f}×")
    print(f"  {'Yield RMSE':<24}  {result['yield_rmse_unpol']:>12.4f}  "
          f"{result['yield_rmse_pol']:>12.4f}")
    print(f"  {'Σ RMSE':<24}  {result['sigma_rmse_unpol']:>12.4f}  "
          f"{result['sigma_rmse_pol']:>12.4f}  {result['sigma_improvement']:>7.1f}×")
    print(f"  {'Basin fraction':<24}  {result['basin_frac_unpol']:>11.0%}  "
          f"{result['basin_frac_pol']:>11.0%}")

    return result


# ════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 8: BOOTSTRAP UNCERTAINTY
# ════════════════════════════════════════════════════════════════════════════════

def run_bootstrap(recov_result: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Deliverable 8: bootstrap uncertainty estimation (200 resamples)."""
    print()
    print("=" * 70)
    print("EXPERIMENT 8: Bootstrap Uncertainty (200 resamples)")
    print("-" * 70)

    ws = build_wave_set(2.5, reflectivities=(+1,))

    # Regenerate data with same seed to get kinematics
    rng = np.random.default_rng(42)
    n_amp = ws.n_amplitudes
    magnitudes = rng.exponential(scale=1.0, size=n_amp)
    phases = rng.uniform(-np.pi, np.pi, size=n_amp)
    V_true_np = magnitudes * np.exp(1j * phases)
    V_true_np /= np.linalg.norm(V_true_np)

    from experiments.pwa_engine.core import SyntheticDataGenerator as _SDG
    gen = _SDG(ws, V_true_np, seed=42)
    data = gen.generate(10000, 500000, device=device)

    V_seed = torch.tensor(recov_result["V_fit"], dtype=torch.complex128, device=device)

    result = bootstrap_uncertainty(
        wave_set=ws,
        theta_data=data["theta_data"],
        phi_data=data["phi_data"],
        theta_mc=data["theta_mc"],
        phi_mc=data["phi_mc"],
        n_generated=data["n_generated"],
        V_seed=V_seed,
        n_bootstrap=200,
        max_iter=300,
        device=device,
        seed=42,
    )

    print(f"  Resamples:   {result['n_bootstrap']}")
    print(f"  Converged:   {result['converged_fraction']:.0%}")
    print(f"  Wall time:   {result['time_s']:.1f}s")
    print()
    print(f"  {'Amp':>4}  {'Yield±σ':>16}  {'Phase±σ (deg)':>16}")
    print(f"  {'-'*4}  {'-'*16}  {'-'*16}")
    for i in range(result["n_amp"]):
        ym = result["yield_mean"][i]
        ys = result["yield_std"][i]
        pm = np.degrees(result["phase_mean"][i])
        ps = np.degrees(result["phase_std"][i])
        print(f"  {i:>4}  {ym:>8.4f}±{ys:<6.4f}  {pm:>+8.1f}±{ps:<5.1f}")

    return result


# ════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 9: COUPLED-CHANNEL PWA
# ════════════════════════════════════════════════════════════════════════════════

def run_coupled_channel(device: torch.device) -> Dict[str, Any]:
    """Deliverable 9: coupled-channel fitting vs independent channels."""
    print()
    print("=" * 70)
    print("EXPERIMENT 9: Coupled-Channel PWA")
    print("-" * 70)

    result = coupled_channel_test(
        n_data_ch1=5000,
        n_data_ch2=3000,
        n_generated=200_000,
        n_starts=20,
        device=device,
        seed=42,
    )

    print(f"  Channels: ηπ  ({result['n_amp_ch1']} amp, {result['n_data_ch1']} events)")
    print(f"            η'π ({result['n_amp_ch2']} amp, {result['n_data_ch2']} events)")
    print(f"  Global:   {result['n_amp_global']} shared amplitudes")
    print()
    print(f"  {'Metric':<30}  {'Joint':>10}  {'Ch1 alone':>10}  {'Ch2 alone':>10}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*10}")
    print(f"  {'Yield RMSE (all)':<30}  {result['joint_yield_rmse']:>10.4f}  "
          f"{result['ch1_yield_rmse']:>10.4f}  {'—':>10}")
    print(f"  {'Yield RMSE (shared)':<30}  {result['joint_yield_rmse_shared']:>10.4f}  "
          f"{'—':>10}  {result['ch2_yield_rmse']:>10.4f}")
    print(f"  {'Phase RMSE (rad, all)':<30}  {result['joint_phase_rmse']:>10.4f}  "
          f"{result['ch1_phase_rmse']:>10.4f}  {'—':>10}")
    print(f"  {'Basin fraction':<30}  {result['joint_basin_frac']:>9.0%}  "
          f"{result['ch1_basin_frac']:>9.0%}  {result['ch2_basin_frac']:>9.0%}")
    print()
    print(f"  Yield improvement vs ch1: {result['yield_improvement_vs_ch1']:.2f}×")
    print(f"  Yield improvement vs ch2: {result['yield_improvement_vs_ch2']:.2f}×")

    return result


# ════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 10: MASS-DEPENDENT BREIT-WIGNER FIT
# ════════════════════════════════════════════════════════════════════════════════

def run_mass_dependent(device: torch.device) -> Dict[str, Any]:
    """Deliverable 10: mass-dependent BW fit across mass bins."""
    print()
    print("=" * 70)
    print("EXPERIMENT 10: Mass-Dependent Breit-Wigner Fit")
    print("-" * 70)

    result = mass_dependent_fit(
        n_mass_bins=15,
        m_range=(0.8, 2.0),
        n_events_per_bin=2000,
        n_generated_per_bin=100_000,
        n_starts=10,
        device=device,
        seed=42,
    )

    print(f"  Mass bins: {result['n_mass_bins']} in "
          f"[{result['m_range'][0]:.1f}, {result['m_range'][1]:.1f}] GeV")
    print(f"  Events/bin: {result['n_events_per_bin']}")
    print(f"  Waves: {result['n_amp']} (S-wave J=0.5 + D-wave J=1.5)")
    print()
    print(f"  {'Resonance':<12}  {'m₀ true':>8}  {'m₀ fit':>8}  "
          f"{'Γ₀ true':>8}  {'Γ₀ fit':>8}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    print(f"  {'R₁ (S-wave)':<12}  {result['m0_true_s']:>8.3f}  "
          f"{result['m0_fit_s']:>8.3f}  {result['gamma_true_s']:>8.3f}  "
          f"{result['gamma_fit_s']:>8.3f}")
    print(f"  {'R₂ (D-wave)':<12}  {result['m0_true_d']:>8.3f}  "
          f"{result['m0_fit_d']:>8.3f}  {result['gamma_true_d']:>8.3f}  "
          f"{result['gamma_fit_d']:>8.3f}")

    m_err_r1 = abs(result["m0_fit_s"] - result["m0_true_s"]) * 1000
    m_err_r2 = abs(result["m0_fit_d"] - result["m0_true_d"]) * 1000
    g_err_r1 = abs(result["gamma_fit_s"] - result["gamma_true_s"]) * 1000
    g_err_r2 = abs(result["gamma_fit_d"] - result["gamma_true_d"]) * 1000
    print(f"\n  Mass recovery: R₁ Δm₀={m_err_r1:.0f} MeV, R₂ Δm₀={m_err_r2:.0f} MeV")
    print(f"  Width recovery: R₁ ΔΓ₀={g_err_r1:.0f} MeV, R₂ ΔΓ₀={g_err_r2:.0f} MeV")

    return result


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE GENERATION
# ════════════════════════════════════════════════════════════════════════════════

def fig_convention_test(conv_result: Dict) -> None:
    """Convention reduction — bar chart of errors per test."""
    fig, ax = plt.subplots(figsize=SINGLE_COL)
    tests = [
        ("Full→Single ε", conv_result["test_1_full_vs_single_eps"]),
        ("Model→Manual", conv_result["test_2_model_vs_manual"]),
        ("Full ρ→Diag ρ", conv_result["test_3_full_rho_vs_diagonal"]),
    ]
    labels = [t[0] for t in tests]
    errs = [max(t[1], 1e-17) for t in tests]  # floor for log scale

    bars = ax.bar(labels, errs, color=[C_BLUE, C_GREEN, C_ORANGE], edgecolor=C_BLACK, linewidth=0.3)
    ax.set_yscale("log")
    ax.axhline(1e-12, color=C_RED, linestyle="--", linewidth=0.8, label="Machine ε threshold")
    ax.set_ylabel("Max pointwise error")
    ax.set_title("Convention reduction test (Eq. 5.48)")
    ax.legend(fontsize=7)
    ax.tick_params(direction="in")
    ax.set_ylim(1e-18, 1e-8)

    fig.tight_layout()
    for fmt in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"pwa_convention_test.{fmt}")
    plt.close(fig)
    print("  Fig 1 saved: pwa_convention_test.pdf")


def fig_parameter_recovery(recov: Dict) -> None:
    """Parameter recovery — true vs fitted yields and phases."""
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COL)

    n_amp = recov["n_amp"]
    idx = np.arange(n_amp)

    # Left: yields
    ax = axes[0]
    width = 0.35
    ax.bar(idx - width / 2, recov["yield_true"], width, color=C_BLUE, label="True", edgecolor=C_BLACK, linewidth=0.3)
    ax.bar(idx + width / 2, recov["yield_fit"], width, color=C_ORANGE, label="Fitted", edgecolor=C_BLACK, linewidth=0.3)
    ax.set_xlabel("Amplitude index")
    ax.set_ylabel(r"Yield $|V_\alpha|^2$")
    ax.set_title(f"Yield recovery (RMSE = {recov['yield_rmse']:.4f})")
    ax.legend(fontsize=7)
    ax.tick_params(direction="in")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Right: phase differences
    ax = axes[1]
    ax.bar(idx, np.degrees(recov["phase_diff"]), color=C_GREEN, edgecolor=C_BLACK, linewidth=0.3)
    ax.axhline(0, color=C_BLACK, linewidth=0.5)
    ax.set_xlabel("Amplitude index")
    ax.set_ylabel("Phase error (deg)")
    ax.set_title(f"Phase recovery (RMSE = {np.degrees(recov['phase_rmse']):.1f}°)")
    ax.tick_params(direction="in")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout(w_pad=2.0)
    for fmt in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"pwa_parameter_recovery.{fmt}")
    plt.close(fig)
    print("  Fig 2 saved: pwa_parameter_recovery.pdf")


def fig_nll_landscape(recov: Dict) -> None:
    """NLL values across multi-start fits — shows basin structure."""
    fig, ax = plt.subplots(figsize=SINGLE_COL)

    nlls = sorted([f["nll"] for f in recov["all_fits"]])
    ax.bar(range(len(nlls)), nlls, color=C_BLUE, edgecolor=C_BLACK, linewidth=0.3)
    ax.axhline(recov["best_nll"], color=C_RED, linestyle="--", linewidth=0.8,
               label=f"Best NLL = {recov['best_nll']:.1f}")
    ax.set_xlabel("Fit index (sorted)")
    ax.set_ylabel("NLL")
    ax.set_title(f"Multi-start landscape ({recov['n_starts']} starts)")
    ax.legend(fontsize=7)
    ax.tick_params(direction="in")

    fig.tight_layout()
    for fmt in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"pwa_nll_landscape.{fmt}")
    plt.close(fig)
    print("  Fig 3 saved: pwa_nll_landscape.pdf")


def fig_speedup(accel: Dict) -> None:
    """Normalization speedup: baseline vs Gram across MC sizes."""
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COL)

    # Left: time vs n_MC (fixed n_amp=12)
    bench12 = [b for b in accel["benchmarks"] if b["n_amplitudes"] <= 13 and b["n_amplitudes"] >= 11]
    if bench12:
        ax = axes[0]
        n_mcs = [b["n_mc_events"] for b in bench12]
        t_base = [b["baseline_per_eval_ms"] for b in bench12]
        t_gram = [b["gram_per_eval_ms"] for b in bench12]
        ax.loglog(n_mcs, t_base, "o-", color=C_RED, label="Baseline", markersize=5)
        ax.loglog(n_mcs, t_gram, "s-", color=C_BLUE, label="Gram", markersize=5)
        ax.set_xlabel("Number of MC events")
        ax.set_ylabel("Time per evaluation (ms)")
        ax.set_title(f"Normalization cost (n_amp≈12)")
        ax.legend(fontsize=7)
        ax.tick_params(direction="in", which="both")

    # Right: speedup vs n_MC
    ax = axes[1]
    if bench12:
        speedups = [b["speedup"] for b in bench12]
        ax.semilogx(n_mcs, speedups, "o-", color=C_GREEN, markersize=6, linewidth=1.5)
        ax.set_xlabel("Number of MC events")
        ax.set_ylabel("Speedup (baseline / Gram)")
        ax.set_title("Gram acceleration factor")
        ax.axhline(1.0, color=C_GREY, linestyle=":", linewidth=0.5)
        ax.tick_params(direction="in", which="both")

    fig.tight_layout(w_pad=2.0)
    for fmt in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"pwa_speedup.{fmt}")
    plt.close(fig)
    print("  Fig 4 saved: pwa_speedup.pdf")


def fig_wave_scan_heatmap(scan: Dict) -> None:
    """Wave-set scan: NLL and stability vs J_max."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))

    res = scan["scan_results"]
    j_vals = [r["j_max"] for r in res]

    # Left: NLL vs J_max
    ax = axes[0]
    nlls = [r["best_nll"] for r in res]
    ax.plot(j_vals, nlls, "o-", color=C_BLUE, markersize=6)
    ax.axvline(scan["true_j_max"], color=C_RED, linestyle="--", linewidth=0.8,
               label=f"True J_max={scan['true_j_max']}")
    ax.set_xlabel(r"$J_{\max}$")
    ax.set_ylabel("Best NLL")
    ax.set_title("Fit quality vs wave-set size")
    ax.legend(fontsize=7)
    ax.tick_params(direction="in")

    # Center: basin fraction (stability)
    ax = axes[1]
    basins = [r["basin_fraction"] for r in res]
    ax.bar(j_vals, basins, width=0.6, color=C_GREEN, edgecolor=C_BLACK, linewidth=0.3)
    ax.set_xlabel(r"$J_{\max}$")
    ax.set_ylabel("Basin fraction")
    ax.set_title("Fit stability")
    ax.set_ylim(0, 1.05)
    ax.tick_params(direction="in")

    # Right: Gram matrix rank
    ax = axes[2]
    ranks = [r["gram_rank_1e8"] for r in res]
    n_amps = [r["n_amplitudes"] for r in res]
    ax.bar(j_vals, ranks, width=0.6, color=C_ORANGE, edgecolor=C_BLACK, linewidth=0.3, label="Gram rank")
    ax.plot(j_vals, n_amps, "s-", color=C_RED, markersize=5, linewidth=1.0, label="n_amp")
    ax.set_xlabel(r"$J_{\max}$")
    ax.set_ylabel("Count")
    ax.set_title("Gram rank vs wave-set size")
    ax.legend(fontsize=7)
    ax.tick_params(direction="in")

    fig.tight_layout(w_pad=2.0)
    for fmt in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"pwa_wave_scan.{fmt}")
    plt.close(fig)
    print("  Fig 5 saved: pwa_wave_scan.pdf")


def fig_gram_qtt(gram_results: Dict) -> None:
    """QTT compression of Gram matrix across wave-set sizes."""
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COL)

    res = gram_results["gram_qtt_results"]

    # Left: compression ratio and chi_max
    ax = axes[0]
    n_amps = [r["n_original"] for r in res]
    crs = [r["compression_ratio"] for r in res]
    chis = [r["chi_max"] for r in res]
    ax.bar(range(len(n_amps)), crs, color=C_BLUE, edgecolor=C_BLACK, linewidth=0.3)
    ax.set_xticks(range(len(n_amps)))
    ax.set_xticklabels([str(n) for n in n_amps])
    ax.set_xlabel("Gram matrix dimension")
    ax.set_ylabel("Compression ratio")
    ax.set_title("QTT compression of Gram matrix")
    for i, (cr, chi) in enumerate(zip(crs, chis)):
        ax.annotate(f"χ={chi}", (i, cr), textcoords="offset points",
                    xytext=(0, 5), fontsize=6, ha="center", color=C_GREY)
    ax.tick_params(direction="in")

    # Right: SVD spectrum of Gram matrix (largest case)
    ax = axes[1]
    largest = res[-1]
    S = largest["gram_svd"]
    S_norm = S / S[0] if S[0] > 0 else S
    ax.semilogy(np.arange(1, len(S_norm) + 1), S_norm, "o-", color=C_BLUE, markersize=3)
    ax.set_xlabel("Singular value index")
    ax.set_ylabel(r"$\sigma_i / \sigma_1$")
    ax.set_title(f"Gram SVD spectrum (n={largest['n_original']})")
    ax.tick_params(direction="in", which="both")

    fig.tight_layout(w_pad=2.0)
    for fmt in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"pwa_gram_qtt.{fmt}")
    plt.close(fig)
    print("  Fig 6 saved: pwa_gram_qtt.pdf")


def fig_moment_pulls(moment_result: Dict) -> None:
    """Angular moment pulls — scatter of |pull| by (L,M)."""
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COL)

    pulls = moment_result["pulls"]
    L_max = moment_result["L_max"]

    # Left: pulls as bar chart ordered by L
    ax = axes[0]
    labels = []
    vals = []
    for L in range(L_max + 1):
        for M in range(-L, L + 1):
            key = (L, M)
            if key in pulls:
                labels.append(f"({L},{M:+d})")
                vals.append(pulls[key])

    colors = [C_RED if v > 2.0 else C_BLUE for v in vals]
    ax.bar(range(len(vals)), vals, color=colors, edgecolor=C_BLACK, linewidth=0.2, width=0.8)
    ax.axhline(2.0, color=C_RED, linestyle="--", linewidth=0.8, label="2σ threshold")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_ylabel("|Pull|")
    ax.set_title(f"Moment pulls (χ²/ndf = {moment_result['chi2_per_ndf']:.1f})")
    ax.legend(fontsize=7)
    ax.tick_params(direction="in")

    # Right: real part comparison — data vs model
    ax = axes[1]
    data_re = []
    model_re = []
    for L in range(L_max + 1):
        for M in range(-L, L + 1):
            key = (L, M)
            if key in moment_result["moments_data"]:
                data_re.append(moment_result["moments_data"][key].real)
                model_re.append(moment_result["moments_model"][key].real)

    data_re = np.array(data_re)
    model_re = np.array(model_re)
    lim = max(abs(data_re).max(), abs(model_re).max()) * 1.2
    ax.scatter(data_re, model_re, s=20, c=C_BLUE, edgecolors=C_BLACK, linewidth=0.3, zorder=3)
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.5, zorder=1)
    ax.set_xlabel(r"Re $\langle Y_L^M \rangle_{\mathrm{data}}$")
    ax.set_ylabel(r"Re $\langle Y_L^M \rangle_{\mathrm{model}}$")
    ax.set_title("Moment comparison")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.tick_params(direction="in")

    fig.tight_layout(w_pad=2.0)
    for fmt in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"pwa_moment_pulls.{fmt}")
    plt.close(fig)
    print("  Fig 7 saved: pwa_moment_pulls.pdf")


def fig_beam_asymmetry(pol_result: Dict) -> None:
    """Beam asymmetry — Σ(θ) comparison and phase recovery."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))

    theta = pol_result["theta_data"]
    sigma_true = pol_result["sigma_true"]
    sigma_unpol = pol_result["sigma_unpol"]
    sigma_pol = pol_result["sigma_pol"]

    # Sort by theta for clean plot
    order = np.argsort(theta)

    # Left: Σ(θ) comparison
    ax = axes[0]
    # Bin into cos(θ) bins for clarity
    n_bins = 25
    cos_edges = np.linspace(-1, 1, n_bins + 1)
    cos_theta = np.cos(theta)
    sigma_true_binned = np.zeros(n_bins)
    sigma_unpol_binned = np.zeros(n_bins)
    sigma_pol_binned = np.zeros(n_bins)
    cos_centers = 0.5 * (cos_edges[:-1] + cos_edges[1:])

    for i in range(n_bins):
        mask = (cos_theta >= cos_edges[i]) & (cos_theta < cos_edges[i + 1])
        if mask.sum() > 0:
            sigma_true_binned[i] = sigma_true[mask].mean()
            sigma_unpol_binned[i] = sigma_unpol[mask].mean()
            sigma_pol_binned[i] = sigma_pol[mask].mean()

    ax.plot(cos_centers, sigma_true_binned, "o-", color=C_BLACK, markersize=4, label="True Σ", linewidth=1.0)
    ax.plot(cos_centers, sigma_unpol_binned, "s--", color=C_RED, markersize=3, label="Unpol. fit", linewidth=0.8)
    ax.plot(cos_centers, sigma_pol_binned, "^-", color=C_BLUE, markersize=3, label="Pol. fit", linewidth=0.8)
    ax.set_xlabel(r"$\cos\theta$")
    ax.set_ylabel(r"$\Sigma$")
    ax.set_title(r"Beam asymmetry $\Sigma(\theta)$")
    ax.legend(fontsize=6)
    ax.tick_params(direction="in")

    # Center: phase recovery comparison
    ax = axes[1]
    n_amp = pol_result["n_amp"]
    V_true = pol_result["V_true"]
    V_unpol = pol_result["V_unpol"]
    V_pol = pol_result["V_pol"]

    # Phase-align each
    idx_ref = int(np.argmax(np.abs(V_true)**2))
    phase_unpol = np.angle(V_true * (V_unpol * np.exp(1j * (np.angle(V_true[idx_ref]) - np.angle(V_unpol[idx_ref])))).conj())
    phase_pol = np.angle(V_true * (V_pol * np.exp(1j * (np.angle(V_true[idx_ref]) - np.angle(V_pol[idx_ref])))).conj())

    width = 0.35
    idx = np.arange(n_amp)
    ax.bar(idx - width/2, np.degrees(phase_unpol), width, color=C_RED, label="Unpol.", edgecolor=C_BLACK, linewidth=0.3)
    ax.bar(idx + width/2, np.degrees(phase_pol), width, color=C_BLUE, label="Pol.", edgecolor=C_BLACK, linewidth=0.3)
    ax.axhline(0, color=C_BLACK, linewidth=0.5)
    ax.set_xlabel("Amplitude index")
    ax.set_ylabel("Phase error (deg)")
    ax.set_title("Phase recovery")
    ax.legend(fontsize=7)
    ax.tick_params(direction="in")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Right: summary metrics
    ax = axes[2]
    metrics = ["Phase\nRMSE", "Yield\nRMSE", "Σ RMSE"]
    unpol_vals = [np.degrees(pol_result["phase_rmse_unpol"]),
                  pol_result["yield_rmse_unpol"],
                  pol_result["sigma_rmse_unpol"]]
    pol_vals = [np.degrees(pol_result["phase_rmse_pol"]),
                pol_result["yield_rmse_pol"],
                pol_result["sigma_rmse_pol"]]

    x = np.arange(len(metrics))
    ax.bar(x - 0.2, unpol_vals, 0.35, color=C_RED, label="Unpol.", edgecolor=C_BLACK, linewidth=0.3)
    ax.bar(x + 0.2, pol_vals, 0.35, color=C_BLUE, label="Pol.", edgecolor=C_BLACK, linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=7)
    ax.set_ylabel("RMSE")
    ax.set_title("Sensitivity summary")
    ax.legend(fontsize=7)
    ax.tick_params(direction="in")

    fig.tight_layout(w_pad=2.0)
    for fmt in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"pwa_beam_asymmetry.{fmt}")
    plt.close(fig)
    print("  Fig 8 saved: pwa_beam_asymmetry.pdf")


def fig_bootstrap(boot_result: Dict) -> None:
    """Bootstrap uncertainty — yield distributions and correlation."""
    n_amp = boot_result["n_amp"]
    n_show = min(n_amp, 6)  # Show first 6 amplitudes in detail

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.5))

    # Top row: yield distributions for first 3 amplitudes
    for i in range(min(3, n_show)):
        ax = axes[0, i]
        yields_i = boot_result["yields_all"][:, i]
        ax.hist(yields_i, bins=30, color=C_BLUE, edgecolor=C_BLACK, linewidth=0.3, alpha=0.8)
        ax.axvline(boot_result["yield_mean"][i], color=C_RED, linewidth=1.0, linestyle="--",
                   label=f"μ={boot_result['yield_mean'][i]:.4f}")
        ax.set_xlabel(f"Relative yield $|V_{i}|^2$")
        ax.set_ylabel("Count")
        ax.set_title(f"Amp {i}: σ={boot_result['yield_std'][i]:.4f}")
        ax.legend(fontsize=6)
        ax.tick_params(direction="in")

    # Bottom-left: yield error bars for all amplitudes
    ax = axes[1, 0]
    idx = np.arange(n_amp)
    ax.errorbar(idx, boot_result["yield_mean"], yerr=boot_result["yield_std"],
                fmt="o", color=C_BLUE, markersize=4, capsize=3, linewidth=0.8)
    ax.set_xlabel("Amplitude index")
    ax.set_ylabel("Relative yield ± σ")
    ax.set_title(f"Bootstrap yields ({boot_result['n_bootstrap']} resamp.)")
    ax.tick_params(direction="in")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Bottom-center: phase error bars
    ax = axes[1, 1]
    ax.errorbar(idx, np.degrees(boot_result["phase_mean"]),
                yerr=np.degrees(boot_result["phase_std"]),
                fmt="s", color=C_GREEN, markersize=4, capsize=3, linewidth=0.8)
    ax.set_xlabel("Amplitude index")
    ax.set_ylabel("Phase ± σ (deg)")
    ax.set_title("Bootstrap phases")
    ax.tick_params(direction="in")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Bottom-right: yield correlation matrix
    ax = axes[1, 2]
    corr = boot_result["yield_corr"]
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Amplitude")
    ax.set_title("Yield correlation")
    ax.tick_params(direction="in")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout(w_pad=1.5, h_pad=2.0)
    for fmt in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"pwa_bootstrap.{fmt}")
    plt.close(fig)
    print("  Fig 9 saved: pwa_bootstrap.pdf")


def fig_coupled_channel(cc_result: Dict) -> None:
    """Coupled-channel: joint vs independent yield recovery comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))

    n_global = cc_result["n_amp_global"]
    n_shared = cc_result["n_amp_ch2"]
    V_true = cc_result["V_true"]
    V_joint = cc_result["V_joint"]
    V_ch1 = cc_result["V_indep_ch1"]
    V_ch2 = cc_result["V_indep_ch2"]

    # Normalize yields
    y_true = np.abs(V_true) ** 2
    y_true = y_true / max(y_true.sum(), 1e-30)
    y_joint = np.abs(V_joint) ** 2
    y_joint = y_joint / max(y_joint.sum(), 1e-30)
    y_ch1 = np.abs(V_ch1) ** 2
    y_ch1 = y_ch1 / max(y_ch1.sum(), 1e-30)

    # Left: all amplitudes — true vs joint vs ch1-alone
    ax = axes[0]
    idx = np.arange(n_global)
    w = 0.25
    ax.bar(idx - w, y_true, w, color=C_BLACK, label="True",
           edgecolor=C_BLACK, linewidth=0.3)
    ax.bar(idx, y_joint, w, color=C_BLUE, label="Joint",
           edgecolor=C_BLACK, linewidth=0.3)
    ax.bar(idx + w, y_ch1, w, color=C_RED, label="Ch1 alone",
           edgecolor=C_BLACK, linewidth=0.3, alpha=0.7)
    ax.set_xlabel("Amplitude index")
    ax.set_ylabel("Relative yield")
    ax.set_title(f"Full wave set ({n_global} amp)")
    ax.legend(fontsize=6)
    ax.tick_params(direction="in")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Center: shared waves — true vs joint vs ch2-alone
    ax = axes[1]
    y_ch2 = np.abs(V_ch2) ** 2
    y_ch2 = y_ch2 / max(y_ch2.sum(), 1e-30)
    y_true_s = y_true[:n_shared]
    y_true_s = y_true_s / max(y_true_s.sum(), 1e-30)
    y_joint_s = y_joint[:n_shared]
    y_joint_s = y_joint_s / max(y_joint_s.sum(), 1e-30)
    idx_s = np.arange(n_shared)
    ax.bar(idx_s - w, y_true_s, w, color=C_BLACK, label="True",
           edgecolor=C_BLACK, linewidth=0.3)
    ax.bar(idx_s, y_joint_s, w, color=C_BLUE, label="Joint",
           edgecolor=C_BLACK, linewidth=0.3)
    ax.bar(idx_s + w, y_ch2, w, color=C_GREEN, label="Ch2 alone",
           edgecolor=C_BLACK, linewidth=0.3, alpha=0.7)
    ax.set_xlabel("Amplitude index")
    ax.set_ylabel("Relative yield (shared)")
    ax.set_title(f"Shared waves ({n_shared} amp)")
    ax.legend(fontsize=6)
    ax.tick_params(direction="in")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Right: RMSE comparison bar chart
    ax = axes[2]
    metrics = ["Yield\n(all)", "Yield\n(shared)", "Phase\n(all)"]
    joint_vals = [
        cc_result["joint_yield_rmse"],
        cc_result["joint_yield_rmse_shared"],
        cc_result["joint_phase_rmse"],
    ]
    indep_vals = [
        cc_result["ch1_yield_rmse"],
        cc_result["ch2_yield_rmse"],
        cc_result["ch1_phase_rmse"],
    ]
    x = np.arange(len(metrics))
    ax.bar(x - 0.18, joint_vals, 0.32, color=C_BLUE, label="Joint",
           edgecolor=C_BLACK, linewidth=0.3)
    ax.bar(x + 0.18, indep_vals, 0.32, color=C_RED, label="Independent",
           edgecolor=C_BLACK, linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=7)
    ax.set_ylabel("RMSE")
    ax.set_title("Joint vs independent")
    ax.legend(fontsize=7)
    ax.tick_params(direction="in")

    fig.tight_layout(w_pad=2.0)
    for fmt in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"pwa_coupled_channel.{fmt}")
    plt.close(fig)
    print("  Fig 10 saved: pwa_coupled_channel.pdf")


def fig_mass_dependent(md_result: Dict) -> None:
    """Mass-dependent fit: wave fractions and BW resonance extraction."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))

    m = md_result["m_vals"]
    m_fine = np.linspace(m[0], m[-1], 200)

    # Left: S-wave fraction vs mass (primary observable)
    ax = axes[0]
    ax.plot(m, md_result["frac_s_true"], "ko", markersize=5,
            label="True", zorder=3)
    ax.plot(m, md_result["frac_s_fit"], "s", color=C_BLUE, markersize=4,
            label="Fitted", zorder=3)
    # Overlay BW fraction model
    if not np.isnan(md_result["m0_fit_s"]):
        bw1 = BreitWigner(md_result["m0_fit_s"], md_result["gamma_fit_s"])
        bw2 = BreitWigner(md_result["m0_fit_d"], md_result["gamma_fit_d"])
        I1 = bw1.intensity(m_fine)
        I2 = bw2.intensity(m_fine)
        r = md_result["r_fit"]
        frac_model = I1 / np.maximum(I1 + r * I2, 1e-30)
        ax.plot(m_fine, frac_model, "-", color=C_RED, linewidth=1.0,
                label="BW fit")
    ax.set_xlabel("Mass (GeV)")
    ax.set_ylabel(r"$f_S = I_S / (I_S + I_D)$")
    ax.set_title("S-wave fraction vs mass")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=6)
    ax.tick_params(direction="in")

    # Center: per-wave yields across mass bins
    ax = axes[1]
    ax.plot(m, md_result["yield_s_true"], "o-", color=C_BLUE,
            markersize=4, linewidth=0.8, label="S true")
    ax.plot(m, md_result["yield_s_fit"], "s--", color=C_CYAN,
            markersize=3, linewidth=0.8, label="S fit")
    ax.plot(m, md_result["yield_d_true"], "o-", color=C_RED,
            markersize=4, linewidth=0.8, label="D true")
    ax.plot(m, md_result["yield_d_fit"], "^--", color=C_ORANGE,
            markersize=3, linewidth=0.8, label="D fit")
    ax.set_xlabel("Mass (GeV)")
    ax.set_ylabel("Per-wave yield")
    ax.set_title("S/D-wave yields")
    ax.legend(fontsize=6, ncol=2)
    ax.tick_params(direction="in")

    # Right: mass/width recovery summary
    ax = axes[2]
    params = ["m₀(R₁)", "Γ₀(R₁)", "m₀(R₂)", "Γ₀(R₂)"]
    true_vals = [md_result["m0_true_s"], md_result["gamma_true_s"],
                 md_result["m0_true_d"], md_result["gamma_true_d"]]
    fit_vals = [md_result["m0_fit_s"], md_result["gamma_fit_s"],
                md_result["m0_fit_d"], md_result["gamma_fit_d"]]
    fit_errs = [md_result["m0_err_s"], md_result["gamma_err_s"],
                md_result["m0_err_d"], md_result["gamma_err_d"]]
    # Replace nan errors with 0 for plotting
    fit_errs = [0.0 if np.isnan(e) else e for e in fit_errs]
    x = np.arange(len(params))
    ax.bar(x - 0.18, true_vals, 0.32, color=C_BLACK, label="True",
           edgecolor=C_BLACK, linewidth=0.3)
    ax.bar(x + 0.18, fit_vals, 0.32, color=C_BLUE, label="Fitted",
           edgecolor=C_BLACK, linewidth=0.3, yerr=fit_errs, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(params, fontsize=7)
    ax.set_ylabel("Value (GeV)")
    ax.set_title("Resonance parameter recovery")
    ax.legend(fontsize=7)
    ax.tick_params(direction="in")

    fig.tight_layout(w_pad=2.0)
    for fmt in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"pwa_mass_dependent.{fmt}")
    plt.close(fig)
    print("  Fig 11 saved: pwa_mass_dependent.pdf")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now(timezone.utc).isoformat()

    print()
    print("=" * 70)
    print("PWA COMPUTE ENGINE — Adams (2026)")
    print("  Full Eq. 5.48 Implementation with Gram-Accelerated Likelihood")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Device: {device}")
    print(f"  Time:   {timestamp}")
    print("=" * 70)

    t_total = time.perf_counter()

    # Run all experiments
    conv_result = run_convention_test(device)
    recov_result = run_parameter_recovery(device)
    accel_result = run_acceleration_benchmark(device)
    scan_result = run_wave_scan(device)
    gram_qtt_result = run_gram_qtt(device)
    moment_result = run_moment_validation(recov_result, device)
    pol_result = run_beam_asymmetry(device)
    boot_result = run_bootstrap(recov_result, device)
    cc_result = run_coupled_channel(device)
    md_result = run_mass_dependent(device)

    # Generate figures
    print()
    print("=" * 70)
    print("FIGURE GENERATION (600 DPI vector PDF)")
    print("-" * 70)

    fig_convention_test(conv_result)
    fig_parameter_recovery(recov_result)
    fig_nll_landscape(recov_result)
    fig_speedup(accel_result)
    fig_wave_scan_heatmap(scan_result)
    fig_gram_qtt(gram_qtt_result)
    fig_moment_pulls(moment_result)
    fig_beam_asymmetry(pol_result)
    fig_bootstrap(boot_result)
    fig_coupled_channel(cc_result)
    fig_mass_dependent(md_result)

    t_total = time.perf_counter() - t_total

    # Write metadata
    metadata = {
        "experiment": "pwa_engine_full_suite",
        "timestamp": timestamp,
        "device": str(device),
        "convention_test": {
            "test_1_err": conv_result["test_1_full_vs_single_eps"],
            "test_2_err": conv_result["test_2_model_vs_manual"],
            "test_3_err": conv_result["test_3_full_rho_vs_diagonal"],
            "all_pass": conv_result["all_pass"],
        },
        "parameter_recovery": {
            "n_data": recov_result["n_data"],
            "n_generated": recov_result["n_generated"],
            "n_amp": recov_result["n_amp"],
            "yield_rmse": recov_result["yield_rmse"],
            "phase_rmse_rad": recov_result["phase_rmse"],
            "best_nll": recov_result["best_nll"],
            "basin_fraction": recov_result["basin_fraction"],
            "chi2_per_ndf": recov_result["chi2"] / recov_result["n_amp"],
        },
        "acceleration": {
            "benchmarks": [
                {
                    "n_mc": b["n_mc_events"],
                    "n_amp": b["n_amplitudes"],
                    "speedup": b["speedup"],
                    "agreement": b["relative_agreement"],
                }
                for b in accel_result["benchmarks"]
            ]
        },
        "wave_scan": {
            "j_max_values": scan_result["j_max_values"],
            "true_j_max": scan_result["true_j_max"],
            "results": [
                {
                    "j_max": r["j_max"],
                    "n_amp": r["n_amplitudes"],
                    "nll": r["best_nll"],
                    "basin_fraction": r["basin_fraction"],
                    "gram_rank": r["gram_rank_1e8"],
                }
                for r in scan_result["scan_results"]
            ],
        },
        "gram_qtt": [
            {
                "n_original": r["n_original"],
                "chi_max": r["chi_max"],
                "compression": r["compression_ratio"],
            }
            for r in gram_qtt_result["gram_qtt_results"]
        ],
        "moment_validation": {
            "L_max": moment_result["L_max"],
            "chi2_per_ndf": moment_result["chi2_per_ndf"],
            "n_moments": moment_result["ndf"],
        },
        "beam_asymmetry": {
            "n_amp": pol_result["n_amp"],
            "phase_rmse_unpol": pol_result["phase_rmse_unpol"],
            "phase_rmse_pol": pol_result["phase_rmse_pol"],
            "phase_improvement": pol_result["phase_improvement"],
            "sigma_rmse_unpol": pol_result["sigma_rmse_unpol"],
            "sigma_rmse_pol": pol_result["sigma_rmse_pol"],
        },
        "bootstrap": {
            "n_resamples": boot_result["n_bootstrap"],
            "converged_fraction": boot_result["converged_fraction"],
            "mean_yield_sigma": float(boot_result["yield_std"].mean()),
            "mean_phase_sigma_deg": float(np.degrees(boot_result["phase_std"].mean())),
            "time_s": boot_result["time_s"],
        },
        "coupled_channel": {
            "n_channels": 2,
            "n_amp_global": cc_result["n_amp_global"],
            "joint_yield_rmse": cc_result["joint_yield_rmse"],
            "ch1_yield_rmse": cc_result["ch1_yield_rmse"],
            "yield_improvement_vs_ch1": cc_result["yield_improvement_vs_ch1"],
            "joint_basin_frac": cc_result["joint_basin_frac"],
        },
        "mass_dependent": {
            "n_mass_bins": md_result["n_mass_bins"],
            "m0_true_s": md_result["m0_true_s"],
            "m0_fit_s": md_result["m0_fit_s"],
            "gamma_true_s": md_result["gamma_true_s"],
            "gamma_fit_s": md_result["gamma_fit_s"],
            "m0_true_d": md_result["m0_true_d"],
            "m0_fit_d": md_result["m0_fit_d"],
            "gamma_true_d": md_result["gamma_true_d"],
            "gamma_fit_d": md_result["gamma_fit_d"],
        },
        "total_time_s": t_total,
    }

    meta_path = OUTPUT_DIR / "pwa_engine_metadata.json"
    meta_json = json.dumps(metadata, indent=2)
    meta_path.write_text(meta_json)
    sha = hashlib.sha256(meta_json.encode()).hexdigest()[:16]

    print(f"\n{'=' * 70}")
    print("COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Figures:  pwa_*.{{pdf,png}} in {OUTPUT_DIR}")
    print(f"  Metadata: pwa_engine_metadata.json (SHA-256: {sha}...)")
    print(f"  Wall:     {t_total:.1f}s")
    print()
    print("  DELIVERABLES:")
    print(f"    1. Convention test:       {'PASS' if conv_result['all_pass'] else 'FAIL'} (all < 1e-12)")
    print(f"    2. Parameter recovery:    RMSE(yield)={recov_result['yield_rmse']:.4f}, "
          f"RMSE(phase)={np.degrees(recov_result['phase_rmse']):.1f}°")
    print(f"    3. Gram acceleration:     up to "
          f"{max(b['speedup'] for b in accel_result['benchmarks']):.0f}× speedup")
    print(f"    4. Wave-set scan:         {len(scan_result['j_max_values'])} J_max values, "
          f"robustness atlas complete")
    print(f"    5. QTT Gram compression:  up to "
          f"{max(r['compression_ratio'] for r in gram_qtt_result['gram_qtt_results']):.1f}× "
          f"compression")
    print(f"    6. Moment validation:     χ²/ndf = {moment_result['chi2_per_ndf']:.1f} "
          f"({moment_result['ndf']} moments)")
    print(f"    7. Beam asymmetry:        phase improvement "
          f"{pol_result['phase_improvement']:.1f}× with polarization")
    print(f"    8. Bootstrap:             {boot_result['n_bootstrap']} resamples, "
          f"σ(yield)={boot_result['yield_std'].mean():.4f}, "
          f"σ(phase)={np.degrees(boot_result['phase_std'].mean()):.1f}°")
    print(f"    9. Coupled-channel:       yield improvement "
          f"{cc_result['yield_improvement_vs_ch1']:.1f}× vs ch1 alone")
    m_err_r1 = abs(md_result["m0_fit_s"] - md_result["m0_true_s"]) * 1000
    m_err_r2 = abs(md_result["m0_fit_d"] - md_result["m0_true_d"]) * 1000
    print(f"   10. Mass-dependent BW:     "
          f"Δm₀(R₁)={m_err_r1:.0f} MeV, Δm₀(R₂)={m_err_r2:.0f} MeV")


if __name__ == "__main__":
    main()
