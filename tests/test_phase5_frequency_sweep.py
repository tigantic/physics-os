"""Phase 5 Validation: QTT Frequency Sweep.

Tests:
  1. Result containers (FrequencyPoint, FrequencySweepResult)
  2. Fresnel analytical sweep (compare with exact slab formulas)
  3. Resonance utilities (quarter-wave, half-wave k₀)
  4. Rational interpolation (AAA approximant accuracy)
  5. Geometry summary helper
  6. Uniform sweep — free-space S₁₁ vs frequency (flat, near 0)
  7. Uniform sweep — slab S₁₁ vs Fresnel (frequency-dependent R)
  8. Adaptive sweep — slab with resonance detection
"""

import math
import sys
import numpy as np

sys.path.insert(0, "/home/brad/TiganticLabz/FRONT_VAULT/03_SOURCE/Main_Projects/HyperTensor-VM-main")

from ontic.em.frequency_sweep import (
    FrequencyPoint,
    FrequencySweepResult,
    frequency_sweep_uniform,
    frequency_sweep_adaptive,
    s11_sweep,
    rational_interpolation,
    fresnel_slab_sweep,
    quarter_wave_resonance_k0,
    half_wave_resonance_k0,
    compare_sweep_to_analytical,
    _geometry_summary,
    _solve_at_frequency,
)
from ontic.em.boundaries import (
    Geometry1D,
    PMLConfig,
    MaterialRegion,
    dielectric_slab_geometry,
    free_space_geometry,
)
from ontic.em.s_parameters import (
    Port,
    fresnel_slab_reflection,
    fresnel_slab_transmission,
    s_to_db,
)


def run_tests():
    passed = 0
    failed = 0
    total = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed, total
        total += 1
        if condition:
            passed += 1
            print(f"  ✓ {name}")
        else:
            failed += 1
            print(f"  ✗ {name} — {detail}")

    # =================================================================
    # Test 1: Result container — FrequencyPoint
    # =================================================================
    print("\n=== Test 1: FrequencyPoint container ===")

    S_test = np.array([[0.1 + 0.2j]], dtype=np.complex128)
    Z_test = np.array([50.0 + 0j], dtype=np.complex128)
    fp = FrequencyPoint(
        k0=12.566, frequency_hz=6e8, S=S_test, Z_in=Z_test,
        solver_residual=1e-5, solve_time_s=0.5,
    )
    check("k0", abs(fp.k0 - 12.566) < 1e-10)
    check("frequency_hz", abs(fp.frequency_hz - 6e8) < 1)
    check("S shape", fp.S.shape == (1, 1))
    check("S value", abs(fp.S[0, 0] - (0.1 + 0.2j)) < 1e-15)
    check("Z_in", abs(fp.Z_in[0] - 50.0) < 1e-10)
    check("solver_residual", abs(fp.solver_residual - 1e-5) < 1e-15)
    check("E_solutions default None", fp.E_solutions is None)

    # =================================================================
    # Test 2: Result container — FrequencySweepResult
    # =================================================================
    print("\n=== Test 2: FrequencySweepResult container ===")

    port1 = Port(position=0.3, ref_position=0.35, direction=1)
    points = []
    for i, k0 in enumerate([4.0, 8.0, 12.0]):
        s11 = 0.1 * (i + 1) + 0j
        S_i = np.array([[s11]], dtype=np.complex128)
        Z_i = np.array([50.0], dtype=np.complex128)
        points.append(FrequencyPoint(
            k0=k0, frequency_hz=k0 * 299792458.0 / (2 * math.pi),
            S=S_i, Z_in=Z_i, solver_residual=1e-4 * (i + 1),
            solve_time_s=1.0,
        ))

    sweep = FrequencySweepResult(
        points=points, ports=[port1],
        geometry_description="test", sweep_type="uniform",
        total_time_s=3.0,
    )

    check("n_points", sweep.n_points == 3)
    check("n_ports", sweep.n_ports == 1)
    check("k0_array", len(sweep.k0_array) == 3 and
          abs(sweep.k0_array[0] - 4.0) < 1e-10)
    check("s_parameter shape", len(sweep.s_parameter(0, 0)) == 3)
    check("s_parameter value", abs(sweep.s_parameter(0, 0)[1] - 0.2) < 1e-10)
    check("s_parameter_db", all(np.isfinite(sweep.s_parameter_db(0, 0))))
    check("impedance shape", len(sweep.impedance(0)) == 3)
    check("residuals", len(sweep.residuals()) == 3)
    check("max_residual", abs(sweep.max_residual() - 3e-4) < 1e-10)

    # =================================================================
    # Test 3: Fresnel analytical sweep
    # =================================================================
    print("\n=== Test 3: Fresnel analytical sweep ===")

    eps_slab = 4.0 + 0j
    thickness = 0.1
    k0_arr, R_arr, T_arr = fresnel_slab_sweep(
        k0_min=2.0, k0_max=30.0, n_freq=100,
        eps_slab=eps_slab, thickness=thickness,
    )
    check("sweep length", len(k0_arr) == 100)
    check("R is complex", R_arr.dtype == np.complex128)
    check("T is complex", T_arr.dtype == np.complex128)

    # Verify a few points against direct Fresnel
    for idx in [0, 50, 99]:
        k = k0_arr[idx]
        R_direct = fresnel_slab_reflection(k, eps_slab, thickness)
        T_direct = fresnel_slab_transmission(k, eps_slab, thickness)
        check(f"R[{idx}] matches",
              abs(R_arr[idx] - R_direct) < 1e-14,
              f"|err|={abs(R_arr[idx] - R_direct):.2e}")
        check(f"T[{idx}] matches",
              abs(T_arr[idx] - T_direct) < 1e-14,
              f"|err|={abs(T_arr[idx] - T_direct):.2e}")

    # Energy conservation: |R|² + |T|² = 1 for lossless slab
    for idx in [0, 50, 99]:
        energy = abs(R_arr[idx])**2 + abs(T_arr[idx])**2
        check(f"energy conservation [{idx}]",
              abs(energy - 1.0) < 1e-12,
              f"|R|²+|T|²={energy:.15f}")

    # =================================================================
    # Test 4: Resonance frequency formulas
    # =================================================================
    print("\n=== Test 4: Resonance formulas ===")

    eps_s = 4.0 + 0j
    d = 0.2
    n_slab = math.sqrt(4.0)

    # Quarter-wave: k₀·n·d = π/2 → k₀ = π/(2·n·d)
    k_qw1 = quarter_wave_resonance_k0(eps_s, d, n_harmonic=1)
    expected_qw1 = math.pi / (2.0 * n_slab * d)
    check("quarter-wave n=1",
          abs(k_qw1 - expected_qw1) < 1e-12,
          f"k={k_qw1:.6f} vs expected={expected_qw1:.6f}")

    k_qw2 = quarter_wave_resonance_k0(eps_s, d, n_harmonic=2)
    expected_qw2 = 3.0 * math.pi / (2.0 * n_slab * d)
    check("quarter-wave n=2",
          abs(k_qw2 - expected_qw2) < 1e-12,
          f"k={k_qw2:.6f} vs expected={expected_qw2:.6f}")

    # Half-wave: k₀·n·d = nπ → k₀ = nπ/(n·d)
    k_hw1 = half_wave_resonance_k0(eps_s, d, n_harmonic=1)
    expected_hw1 = math.pi / (n_slab * d)
    check("half-wave n=1",
          abs(k_hw1 - expected_hw1) < 1e-12,
          f"k={k_hw1:.6f} vs expected={expected_hw1:.6f}")

    # Verify Fresnel R = 0 at half-wave resonance
    R_at_hw = fresnel_slab_reflection(k_hw1, eps_s, d)
    check("R=0 at half-wave",
          abs(R_at_hw) < 1e-12,
          f"|R|={abs(R_at_hw):.2e}")

    # =================================================================
    # Test 5: Rational interpolation (AAA)
    # =================================================================
    print("\n=== Test 5: Rational interpolation ===")

    # Test with a simple rational function: f(z) = 1/(z - pole)
    pole = 5.0 + 0.2j
    z_samples = np.linspace(1.0, 10.0, 20)
    f_samples = 1.0 / (z_samples - pole)

    z_eval = np.linspace(1.0, 10.0, 200)
    f_exact = 1.0 / (z_eval - pole)
    f_interp = rational_interpolation(z_samples, f_samples, z_eval)

    err = np.max(np.abs(f_interp - f_exact))
    check("rational interp accuracy",
          err < 1e-6,
          f"max |err| = {err:.2e}")

    # Test with Fresnel slab response (oscillatory rational function)
    k_samples = np.linspace(3.0, 25.0, 15)
    R_samples = np.array([
        fresnel_slab_reflection(k, 4.0+0j, 0.15) for k in k_samples
    ])
    k_dense = np.linspace(3.0, 25.0, 200)
    R_exact = np.array([
        fresnel_slab_reflection(k, 4.0+0j, 0.15) for k in k_dense
    ])
    R_interp = rational_interpolation(k_samples, R_samples, k_dense)
    err_slab = np.max(np.abs(R_interp - R_exact))
    check("Fresnel rational interp",
          err_slab < 1.0,
          f"max |err| = {err_slab:.4f}")

    # =================================================================
    # Test 6: Geometry summary
    # =================================================================
    print("\n=== Test 6: Geometry summary ===")

    geo = Geometry1D(
        n_bits=12,
        background_eps=1.0,
        pml=PMLConfig(n_cells=40, sigma_max=100.0),
    )
    geo.add_dielectric_slab(4.0, 0.3, 0.7, "slab")
    summary = _geometry_summary(geo)
    check("summary contains n_bits", "12-bit" in summary)
    check("summary contains PML", "PML" in summary)
    check("summary contains slab", "slab" in summary)

    # =================================================================
    # Test 7: Single frequency solve (free space, S₁₁ ≈ 0)
    # =================================================================
    print("\n=== Test 7: Single-frequency free-space S₁₁ ===")

    n_bits = 12
    k0 = 4.0 * math.pi
    pml_cfg = PMLConfig.for_problem(n_bits=n_bits, k=k0, target_R_dB=-60.0)
    geo_fs = Geometry1D(
        n_bits=n_bits,
        background_eps=1.0,
        pml=pml_cfg,
    )
    port_fs = Port(
        position=0.3, ref_position=0.35, direction=1,
        eps_r=1.0, width=0.02, label="Port 1",
    )

    pt = _solve_at_frequency(
        geometry=geo_fs, k0=k0, ports=[port_fs],
        max_rank=128, solver_tol=1e-4, n_sweeps=40,
        damping=pml_cfg.damping, n_probes=8,
    )

    s11_mag = abs(pt.S[0, 0])
    s11_db = s_to_db(pt.S[0, 0])
    check("free-space S₁₁ < 0.05",
          s11_mag < 0.05,
          f"|S₁₁| = {s11_mag:.6f} ({s11_db:.1f} dB)")
    check("solver converged",
          pt.solver_residual < 1e-2,
          f"residual = {pt.solver_residual:.2e}")
    check("solve time finite",
          pt.solve_time_s > 0 and pt.solve_time_s < 120,
          f"t = {pt.solve_time_s:.1f}s")

    # =================================================================
    # Test 8: Uniform sweep — free-space flat response
    # =================================================================
    print("\n=== Test 8: Uniform sweep — free-space 3 points ===")

    sweep_fs = frequency_sweep_uniform(
        geometry=geo_fs,
        ports=[port_fs],
        k0_min=3.0 * math.pi,
        k0_max=5.0 * math.pi,
        n_freq=3,
        max_rank=128,
        solver_tol=1e-4,
        n_sweeps=40,
        damping=pml_cfg.damping,
        n_probes=8,
        verbose=True,
    )

    check("sweep type", sweep_fs.sweep_type == "uniform")
    check("n_points = 3", sweep_fs.n_points == 3)
    s11_all = np.abs(sweep_fs.s_parameter(0, 0))
    check("all |S₁₁| < 0.1",
          np.all(s11_all < 0.1),
          f"max |S₁₁| = {np.max(s11_all):.4f}")
    check("max residual < 0.01",
          sweep_fs.max_residual() < 0.01,
          f"max res = {sweep_fs.max_residual():.2e}")
    check("total_time > 0",
          sweep_fs.total_time_s > 0,
          f"t = {sweep_fs.total_time_s:.1f}s")

    # =================================================================
    # Test 9: Uniform sweep — slab vs Fresnel
    # =================================================================
    print("\n=== Test 9: Slab sweep vs Fresnel (3 points) ===")

    eps_test = 4.0
    slab_start = 0.40
    slab_end = 0.60
    slab_d = slab_end - slab_start

    geo_slab = Geometry1D(
        n_bits=n_bits,
        background_eps=1.0,
        pml=pml_cfg,
    )
    geo_slab.add_dielectric_slab(eps_test, slab_start, slab_end, "slab")

    port_slab = Port(
        position=0.20, ref_position=0.25, direction=1,
        eps_r=1.0, width=0.02, label="Port 1",
    )

    # Pick 3 k₀ values that give varied Fresnel response
    k0_list = [3.0 * math.pi, 4.0 * math.pi, 5.0 * math.pi]
    sweep_slab = frequency_sweep_uniform(
        geometry=geo_slab,
        ports=[port_slab],
        k0_min=k0_list[0],
        k0_max=k0_list[-1],
        n_freq=3,
        max_rank=128,
        solver_tol=1e-4,
        n_sweeps=40,
        damping=pml_cfg.damping,
        n_probes=8,
        verbose=True,
    )

    comparison = compare_sweep_to_analytical(
        sweep_slab, eps_test + 0j, slab_d, port_idx=0,
    )
    check("slab max |error| < 0.15",
          comparison["max_error_mag"] < 0.15,
          f"max err = {comparison['max_error_mag']:.4f}")
    check("slab RMS error < 0.1",
          comparison["rms_error_mag"] < 0.1,
          f"RMS err = {comparison['rms_error_mag']:.4f}")

    # Individual point check: |S₁₁| magnitudes should roughly track Fresnel
    s11_num = np.abs(sweep_slab.s_parameter(0, 0))
    k0_used = sweep_slab.k0_array
    for i in range(len(k0_used)):
        R_exact = abs(fresnel_slab_reflection(k0_used[i], eps_test+0j, slab_d))
        err_i = abs(s11_num[i] - R_exact)
        check(f"  k₀={k0_used[i]:.3f} |err| < 0.15",
              err_i < 0.15,
              f"|S₁₁|={s11_num[i]:.4f}, |R|={R_exact:.4f}, err={err_i:.4f}")

    # =================================================================
    # Test 10: Resonance detection
    # =================================================================
    print("\n=== Test 10: Resonance detection & bandwidth ===")

    # Build a sweep result with a known resonance pattern
    # Simulated S₁₁ dB: flat at -5 dB with a dip to -20 dB at k₀ = 10
    pts_sim = []
    for k0_val in np.linspace(5.0, 15.0, 21):
        # Gaussian dip centered at k₀ = 10
        s11_mag = 10.0 ** ((-5.0 + (-15.0) *
                            np.exp(-((k0_val - 10.0) / 1.5) ** 2)) / 20.0)
        S_sim = np.array([[s11_mag + 0j]], dtype=np.complex128)
        Z_sim = np.array([50.0], dtype=np.complex128)
        pts_sim.append(FrequencyPoint(
            k0=k0_val,
            frequency_hz=k0_val * 299792458.0 / (2 * math.pi),
            S=S_sim, Z_in=Z_sim,
            solver_residual=1e-5, solve_time_s=0.1,
        ))

    sweep_sim = FrequencySweepResult(
        points=pts_sim, ports=[port1],
        geometry_description="simulated resonance",
        sweep_type="uniform", total_time_s=2.1,
    )

    resonances = sweep_sim.find_resonances(port_idx=0, threshold_db=-10.0)
    check("found 1 resonance",
          len(resonances) == 1,
          f"found {len(resonances)}")
    if resonances:
        check("resonance near k₀=10",
              abs(resonances[0]["k0"] - 10.0) < 0.6,
              f"k₀={resonances[0]['k0']:.2f}")
        check("resonance depth < -15 dB",
              resonances[0]["s11_db"] < -15.0,
              f"depth={resonances[0]['s11_db']:.1f} dB")

    bw = sweep_sim.bandwidth_3db(port_idx=0)
    check("bandwidth found", bw is not None)
    if bw is not None:
        check("bandwidth center near 10",
              abs(bw["center_k0"] - 10.0) < 0.6,
              f"center={bw['center_k0']:.2f}")
        check("bandwidth > 0",
              bw["bandwidth_k0"] > 0,
              f"BW={bw['bandwidth_k0']:.3f}")
        check("Q > 1",
              bw["Q_factor"] > 1.0,
              f"Q={bw['Q_factor']:.1f}")

    # =================================================================
    # Test 11: Passivity validation across sweep
    # =================================================================
    print("\n=== Test 11: Passivity across frequency ===")

    passive_flags = sweep_sim.validate_passivity(tol=0.01)
    check("passivity array correct length",
          len(passive_flags) == 21)
    check("all points passive",
          np.all(passive_flags),
          f"{np.sum(~passive_flags)} non-passive points")

    # =================================================================
    # Test 12: s11_sweep convenience wrapper
    # =================================================================
    print("\n=== Test 12: s11_sweep convenience wrapper ===")

    sweep_conv = s11_sweep(
        geometry=geo_fs,
        port=port_fs,
        k0_min=3.5 * math.pi,
        k0_max=4.5 * math.pi,
        n_freq=2,
        adaptive=False,
        max_rank=128,
        solver_tol=1e-4,
        n_sweeps=40,
        damping=pml_cfg.damping,
        n_probes=8,
        verbose=True,
    )
    check("convenience wrapper n_points",
          sweep_conv.n_points == 2)
    check("convenience wrapper type",
          sweep_conv.sweep_type == "uniform")
    s11_conv = np.abs(sweep_conv.s_parameter(0, 0))
    check("convenience wrapper |S₁₁| < 0.1",
          np.all(s11_conv < 0.1),
          f"max |S₁₁| = {np.max(s11_conv):.4f}")

    # =================================================================
    # Summary
    # =================================================================
    print(f"\n{'='*60}")
    print(f"Phase 5 Validation: {passed}/{total} PASSED, {failed} FAILED")
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
