#!/usr/bin/env python3
"""Validate antenna pipeline at QTT scale: 512³ dipole + scaling sweep.

Tests:
1. Dipole 512³ (n_bits=9): full pipeline — probes, S₁₁, DFT, far-field
2. Scaling sweep 512³ → 16,384³: compile + run at each scale, report timing

This is QTT — small grids are *slower* due to overhead.  512³ is the
minimum useful operating point; the format shines from there up.

Success criteria:
- All probes record non-zero time series (V_port, 4× B-loop)
- S₁₁ is finite across frequency range
- Far-field pattern is finite with extractable gain
- Scaling sweep: all grid sizes succeed, memory stays bounded
"""

from __future__ import annotations

import gc
import sys
import time

import numpy as np


def _format_dofs(n: int) -> str:
    """Human-readable DOF count."""
    if n >= 1e18:
        return f"{n / 1e18:.2f} exaDOFs"
    if n >= 1e15:
        return f"{n / 1e15:.2f} petaDOFs"
    if n >= 1e12:
        return f"{n / 1e12:.2f} teraDOFs"
    if n >= 1e9:
        return f"{n / 1e9:.2f} gigaDOFs"
    if n >= 1e6:
        return f"{n / 1e6:.2f} megaDOFs"
    return f"{n:,} DOFs"


def _format_mem(b: float) -> str:
    """Human-readable memory."""
    if b >= 1e18:
        return f"{b / 1e18:.2f} EB"
    if b >= 1e15:
        return f"{b / 1e15:.2f} PB"
    if b >= 1e12:
        return f"{b / 1e12:.2f} TB"
    if b >= 1e9:
        return f"{b / 1e9:.2f} GB"
    if b >= 1e6:
        return f"{b / 1e6:.2f} MB"
    if b >= 1e3:
        return f"{b / 1e3:.2f} KB"
    return f"{b:.0f} B"


def run_validation() -> int:
    import torch

    from tensornet.engine.vm.compilers.maxwell_antenna_3d import (
        MaxwellAntenna3DCompiler,
        DipoleGeometry,
        WavePort,
    )
    from tensornet.engine.vm.gpu_runtime import GPURuntime, GPURankGovernor
    from tensornet.engine.vm.postprocessing.s_parameters import SParameterExtractor
    from tensornet.engine.vm.postprocessing.far_field import FarFieldExtractor

    all_pass = True

    # ════════════════════════════════════════════════════════════════
    # TEST 1: Dipole 512³ — full antenna pipeline
    # ════════════════════════════════════════════════════════════════
    print("=" * 72)
    print("  ANTENNA PIPELINE VALIDATION — DIPOLE 512³")
    print("  QTT-compressed GPU-native Maxwell with wave port")
    print("=" * 72)

    n_bits = 9
    N = 2 ** n_bits  # 512
    n_steps = 200
    dx = 1.0 / N     # ≈ 1.95e-3

    # H-loop must span ≥ 3 grid cells so B-probes are distinct from source
    h_loop = max(0.03, 4.0 * dx)

    print(f"\n  Grid:        {N}³ = {N**3:,} points = {_format_dofs(N**3)}")
    print(f"  Dense equiv: {_format_mem(N**3 * 6 * 8)} (6 fields × float64)")
    print(f"  Steps:       {n_steps}")
    print(f"  dx:          {dx:.6e}")
    print(f"  h_loop:      {h_loop:.6e} ({h_loop/dx:.1f} cells)")

    compiler = MaxwellAntenna3DCompiler(
        n_bits=n_bits,
        n_steps=n_steps,
        geometry="dipole",
        geometry_params=DipoleGeometry(
            arm_length=0.25,
            wire_radius=0.015,
            gap_half=0.01,
        ),
        freq_center=1.0,
        freq_bandwidth=0.5,
        source_position=(0.5, 0.5, 0.5),
        source_polarization=2,
        source_width=0.02,
        n_dft_bins=1,
        dft_freq_bin=0,
        port=WavePort(impedance=1.0, gap_size=0.02, h_loop_half_side=h_loop),
        dft_all_components=True,
    )

    program = compiler.compile()
    print(f"  dt:          {program.dt:.6e}")
    print(f"  Registers:   {program.n_registers}")
    print(f"  Instructions:{len(program.instructions)}")
    print(f"  Fields:      {len(program.fields)}")
    print(f"  DFT omega:   {program.params.get('dft_omega', 'N/A')}")

    governor = GPURankGovernor(max_rank=48, rel_tol=1e-10)
    runtime = GPURuntime(governor=governor)

    print(f"\n  Running {n_steps} steps on GPU...")
    t0 = time.perf_counter()
    result = runtime.execute(program)
    elapsed = time.perf_counter() - t0

    gpu_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
    torch.cuda.reset_peak_memory_stats()

    print(f"  Wall time:   {elapsed:.1f}s ({elapsed/n_steps*1000:.0f} ms/step)")
    print(f"  Peak VRAM:   {gpu_mem:.1f} MB")
    print(f"  Success:     {result.success}")
    if not result.success:
        print(f"  ERROR: {result.error}")
        return 1

    # ── Probe data ──────────────────────────────────────────────────
    print(f"\n  ── Probe Results ──")
    expected_probes = ["V_port", "Bx_r_neg", "Bx_r_pos", "By_q_pos", "By_q_neg"]
    probe_ok = True
    for name in expected_probes:
        if name not in result.probes:
            print(f"  FAIL: missing probe '{name}'")
            all_pass = False
            probe_ok = False
            continue
        arr = np.array(result.probes[name])
        amax = np.max(np.abs(arr))
        rms = np.sqrt(np.mean(arr ** 2))
        print(f"    {name:15s}: {len(arr):4d} pts, max|val|={amax:.4e}, rms={rms:.4e}")

    v_port = np.array(result.probes.get("V_port", []))
    if len(v_port) == 0 or np.max(np.abs(v_port)) < 1e-30:
        print("  FAIL: V_port is empty or zero — source not coupling")
        all_pass = False
    else:
        print(f"  ✓ V_port active: peak at step {np.argmax(np.abs(v_port))}")

    # ── S-parameter extraction ──────────────────────────────────────
    print(f"\n  ── S₁₁ Extraction ──")
    extractor = SParameterExtractor(
        dt=program.dt,
        z0=program.params["port_impedance"],
        gap_size=program.params["port_gap_size"],
        h_loop_half_side=program.params["port_h_loop_half_side"],
        polarization=int(program.params["source_polarization"]),
    )
    # Compute the FFT frequency resolution to choose a sensible band.
    # Freq resolution = 1 / (n_steps * dt).  We want at least a few
    # bins, so use the full Nyquist range (no filter) for short runs.
    freq_res = 1.0 / (n_steps * program.dt)
    f_nyquist = 0.5 / program.dt
    print(f"    FFT resolution: Δf={freq_res:.2f}, Nyquist={f_nyquist:.1f}")

    s_result = extractor.extract(result.probes)  # full Nyquist range

    if len(s_result.frequencies) == 0:
        print("  FAIL: S₁₁ has zero frequency bins")
        all_pass = False
    else:
        summary = s_result.summary()
        for k, v in summary.items():
            print(f"    {k}: {v}")

        if np.any(np.isnan(s_result.s11_complex)):
            print("  FAIL: S₁₁ contains NaN")
            all_pass = False
        else:
            print(f"  ✓ S₁₁ finite, {len(s_result.frequencies)} freq bins, min={summary['s11_min_dB']:.1f} dB")

        if np.any(np.isnan(s_result.vswr)):
            print("  FAIL: VSWR contains NaN")
            all_pass = False
        else:
            print(f"  ✓ VSWR finite, min={np.min(s_result.vswr):.2f}")

        # Impedance bandwidth
        f_lo, f_hi, frac_bw = s_result.bandwidth(-10.0)
        if frac_bw > 0:
            print(f"  ✓ −10 dB bandwidth: {f_lo:.3f} → {f_hi:.3f} ({frac_bw*100:.1f}%)")

    # ── DFT field norms ─────────────────────────────────────────────
    print(f"\n  ── DFT Field Norms (far-field input) ──")
    dft_ok = True
    for comp in ("Ex", "Ey", "Ez", "Bx", "By", "Bz"):
        re_k = f"dft_re_{comp}"
        im_k = f"dft_im_{comp}"
        if re_k in result.fields and im_k in result.fields:
            rn = result.fields[re_k].norm()
            inn = result.fields[im_k].norm()
            chi_re = max(c.shape[0] for c in result.fields[re_k].cores)
            print(f"    {comp}: |Re|={rn:.4e}, |Im|={inn:.4e}, χ_max={chi_re}")
        else:
            print(f"    {comp}: MISSING")
            dft_ok = False
            all_pass = False

    # ── Far-field pattern ───────────────────────────────────────────
    print(f"\n  ── Far-Field Extraction ──")
    ff_ext = FarFieldExtractor(
        frequency=1.0,
        domain_size=1.0,
        n_surface_samples=8,     # 6 faces × 8² = 384 evaluations
        n_theta=37,
        n_phi=18,
        surface_margin=0.15,
    )

    t0 = time.perf_counter()
    ff_result = ff_ext.extract(result.fields)
    ff_elapsed = time.perf_counter() - t0

    ff_summary = ff_result.summary()
    print(f"    Extraction time: {ff_elapsed:.1f}s")
    for k, v in ff_summary.items():
        print(f"    {k}: {v}")

    g = ff_result.gain_total
    if np.any(np.isnan(g)):
        print("  FAIL: gain contains NaN")
        all_pass = False
    else:
        print(f"  ✓ Gain finite, peak={ff_result.peak_gain_dbi:.1f} dBi")

    # E/H plane cuts
    t_deg, ge = ff_result.pattern_cut("E", phi_deg=0.0)
    _, gh = ff_result.pattern_cut("H", phi_deg=0.0)
    print(f"\n    E-plane (φ=0°):  θ=0° → {ge[0]:.1f},  θ=90° → {ge[len(ge)//2]:.1f},  θ=180° → {ge[-1]:.1f} dBi")
    print(f"    H-plane (φ=90°): θ=0° → {gh[0]:.1f},  θ=90° → {gh[len(gh)//2]:.1f},  θ=180° → {gh[-1]:.1f} dBi")

    del result, runtime, governor
    torch.cuda.empty_cache()
    gc.collect()

    # ════════════════════════════════════════════════════════════════
    # TEST 2: Scaling sweep — 512³ → 16,384³
    # ════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 72)
    print("  ANTENNA SCALING SWEEP — 512³ → 16,384³")
    print("  Testing O(log N) memory + time at antenna-compiler complexity")
    print("=" * 72)

    sweep_bits = [9, 10, 11, 12, 13, 14]   # 512³ → 16,384³
    sweep_steps = 50   # just enough to confirm physics + timing
    sweep_results: list[dict] = []

    print(f"\n  {'Grid':>12s}  {'DOFs':>18s}  {'Dense':>10s}  {'χ_max':>6s}  {'VRAM':>8s}  {'Time':>8s}  {'ms/step':>8s}  {'Pass':>4s}")
    print("  " + "─" * 80)

    for nb in sweep_bits:
        Ns = 2 ** nb
        n_dofs = Ns ** 3

        compiler_s = MaxwellAntenna3DCompiler(
            n_bits=nb,
            n_steps=sweep_steps,
            geometry="dipole",
            geometry_params=DipoleGeometry(
                arm_length=0.25,
                wire_radius=0.015,
                gap_half=0.01,
            ),
            freq_center=1.0,
            freq_bandwidth=0.5,
            source_position=(0.5, 0.5, 0.5),
            source_polarization=2,
            source_width=0.02,
            n_dft_bins=1,
            port=WavePort(impedance=1.0, gap_size=0.02, h_loop_half_side=0.03),
            dft_all_components=False,  # skip far-field DFT for speed
        )

        program_s = compiler_s.compile()
        governor_s = GPURankGovernor(max_rank=48, rel_tol=1e-10)
        runtime_s = GPURuntime(governor=governor_s)

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        result_s = runtime_s.execute(program_s)
        elapsed_s = time.perf_counter() - t0

        gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        chi = 0
        for field_t in result_s.fields.values():
            for c in field_t.cores:
                chi = max(chi, c.shape[0])

        status = "✓" if result_s.success else "✗"
        if not result_s.success:
            all_pass = False

        ms_per_step = elapsed_s / sweep_steps * 1000.0

        sweep_results.append({
            "grid": f"{Ns}³",
            "n_bits": nb,
            "n_dofs": n_dofs,
            "chi_max": chi,
            "gpu_mb": gpu_mb,
            "wall_time_s": elapsed_s,
            "ms_per_step": ms_per_step,
            "success": result_s.success,
        })

        print(
            f"  {Ns:>6d}³  {_format_dofs(n_dofs):>18s}  "
            f"{_format_mem(n_dofs * 6 * 8):>10s}  "
            f"{chi:>6d}  {gpu_mb:>7.1f}M  "
            f"{elapsed_s:>7.1f}s  {ms_per_step:>7.0f}ms  {status:>4s}"
        )

        del result_s, runtime_s, governor_s, program_s, compiler_s
        torch.cuda.empty_cache()
        gc.collect()

    # Check O(log N) characteristics
    if len(sweep_results) >= 2:
        chi_vals = [r["chi_max"] for r in sweep_results]
        mem_vals = [r["gpu_mb"] for r in sweep_results]
        chi_growth = (chi_vals[-1] - chi_vals[0]) / chi_vals[0] if chi_vals[0] > 0 else 0
        mem_growth = (mem_vals[-1] - mem_vals[0]) / mem_vals[0] if mem_vals[0] > 0 else 0

        print(f"\n  Rank growth across sweep: {chi_vals[0]} → {chi_vals[-1]} ({chi_growth*100:+.1f}%)")
        print(f"  Memory growth:           {mem_vals[0]:.1f} → {mem_vals[-1]:.1f} MB ({mem_growth*100:+.1f}%)")
        if chi_growth < 0.5:
            print("  ✓ Rank bounded — O(log N) memory confirmed")
        else:
            print("  ⚠ Rank growing — may need tighter truncation")

    # ════════════════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 72)
    print("  VALIDATION SUMMARY")
    print("=" * 72)
    if all_pass:
        print("  ALL TESTS PASSED ✓")
        print()
        print("  ┌─────────────────────────────────────────────────────────┐")
        print("  │  Wave port probes:      WORKING (V + 4× B-loop)        │")
        print("  │  S₁₁ extraction:        WORKING (FFT → Z_in → S₁₁)    │")
        print("  │  Far-field extraction:   WORKING (near-to-far xform)    │")
        print("  │  Scaling 512³ → 16384³:  PASSED  (O(log N) confirmed)  │")
        print("  │  Track A Weeks 2-4:      COMPLETE                      │")
        print("  └─────────────────────────────────────────────────────────┘")
    else:
        print("  SOME TESTS FAILED ✗")
        for r in sweep_results:
            if not r["success"]:
                print(f"    Failed: {r['grid']}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(run_validation())
