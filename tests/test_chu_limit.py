"""Tests for 3D QTT Operator Construction and Chu Limit Challenge.

Tests:
  1. Analytical Q limits (Chu, McLean, Thal, Gustafsson)
  2. 3D array ↔ QTT round-trip
  3. 3D spherical mask construction
  4. 3D Kronecker MPO embedding
  5. 3D identity MPO acts correctly
  6. 3D Laplacian MPO structure (bond dimensions)
  7. 3D Helmholtz MPO with PML (assembles without error)
  8. 3D point source in QTT (nonzero, correct shape)
  9. 3D gap source in QTT (localised)
  10. S₁₁ extraction infrastructure
  11. PEC penalty operator construction
  12. AntennaGeometry3D container
  13. 3D forward solve (free-space, small grid)
  14. ChuProblemConfig derived quantities
  15. ChuOptConfig defaults
  16. Objective functions (minimize, target)
  17. Heaviside binarisation on 3D density (smoke)
  18. 3D conductor mask generation
  19. Integration: solve_and_extract_s11 (small grid)
  20. Integration: adjoint gradient (smoke, small grid)
"""

import math
import sys
import numpy as np

sys.path.insert(
    0,
    "/home/brad/TiganticLabz/FRONT_VAULT/03_SOURCE/Main_Projects/HyperTensor-VM-main",
)

from tensornet.em.qtt_3d import (
    array_3d_to_tt,
    reconstruct_3d,
    stretched_laplacian_mpo_3d,
    _embed_1d_mpo_complex,
    helmholtz_mpo_3d_pml,
    point_source_3d,
    gap_source_3d,
    extract_s11_3d,
    compute_impedance_3d,
    build_pec_penalty_3d,
    solve_helmholtz_3d,
    spherical_mask,
    spherical_shell_mask,
)
from tensornet.em.chu_limit import (
    C0,
    chu_limit_q,
    thal_limit_q,
    mclean_limit_q,
    gustafsson_limit_q,
    print_q_limits,
    ChuProblemConfig,
    ChuOptConfig,
    AntennaGeometry3D,
    solve_and_extract_s11,
    compute_adjoint_gradient_3d,
    objective_minimize_s11,
    objective_target_s11_db,
)
from tensornet.em.boundaries import PMLConfig
from tensornet.engine.vm.operators import identity_mpo


def run_tests():
    passed = 0
    failed = 0
    total = 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal passed, failed, total
        total += 1
        if condition:
            passed += 1
            print(f"  PASS [{total:2d}] {name}")
        else:
            failed += 1
            msg = f"  FAIL [{total:2d}] {name}"
            if detail:
                msg += f" — {detail}"
            print(msg)

    # ==================================================================
    # 1. Analytical Q limits
    # ==================================================================
    print("\n--- Test 1: Analytical Q limits ---")

    ka = 0.3
    Q_chu = chu_limit_q(ka)
    Q_expected = 1.0 / (ka ** 3) + 1.0 / ka
    check("Chu Q formula",
          abs(Q_chu - Q_expected) < 1e-10,
          f"got {Q_chu:.4f}, expected {Q_expected:.4f}")

    Q_mclean = mclean_limit_q(ka)
    check("McLean Q = 0.5 * Chu Q",
          abs(Q_mclean - 0.5 * Q_chu) < 1e-10)

    Q_thal = thal_limit_q(ka)
    check("Thal Q = 1.5 * Chu Q",
          abs(Q_thal - 1.5 * Q_chu) < 1e-10)

    Q_gust = gustafsson_limit_q(ka)
    check("Gustafsson Q = McLean Q (sphere)",
          abs(Q_gust - Q_mclean) < 1e-10)

    # Q monotonically decreases with ka
    check("Q decreases with ka",
          chu_limit_q(0.2) > chu_limit_q(0.5) > chu_limit_q(1.0))

    # ka=0.3: Q_Chu ≈ 40.37
    check("Q_Chu(0.3) ≈ 40.4",
          abs(Q_chu - 40.37) < 0.1,
          f"got {Q_chu:.2f}")

    # ==================================================================
    # 2. 3D array ↔ QTT round-trip
    # ==================================================================
    print("\n--- Test 2: 3D array ↔ QTT round-trip ---")

    n_bits = 3  # 8³ = 512 points
    N = 2 ** n_bits

    # Random 3D array
    rng = np.random.default_rng(42)
    arr_3d = rng.standard_normal((N, N, N)) + 1j * rng.standard_normal((N, N, N))

    tt_cores = array_3d_to_tt(arr_3d, n_bits, max_rank=64)
    check("QTT has 3n sites",
          len(tt_cores) == 3 * n_bits,
          f"got {len(tt_cores)}, expected {3 * n_bits}")

    check("First core left bond = 1",
          tt_cores[0].shape[0] == 1)

    check("Last core right bond = 1",
          tt_cores[-1].shape[2] == 1)

    # Round-trip
    arr_recon = reconstruct_3d(tt_cores, n_bits)
    check("Round-trip shape",
          arr_recon.shape == (N, N, N))

    rel_err = np.linalg.norm(arr_recon - arr_3d) / np.linalg.norm(arr_3d)
    check("Round-trip relative error < 1e-10",
          rel_err < 1e-10,
          f"err = {rel_err:.2e}")

    # Smooth function (should compress well)
    coords = np.linspace(0, 1, N, endpoint=False)
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing='ij')
    smooth = np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy) * np.exp(-zz)

    tt_smooth = array_3d_to_tt(smooth.astype(np.complex128), n_bits, max_rank=16)
    recon_smooth = reconstruct_3d(tt_smooth, n_bits)
    err_smooth = np.linalg.norm(recon_smooth - smooth) / np.linalg.norm(smooth)
    check("Smooth function compresses (err < 0.01)",
          err_smooth < 0.01,
          f"err = {err_smooth:.2e}")

    # ==================================================================
    # 3. Spherical mask
    # ==================================================================
    print("\n--- Test 3: Spherical mask ---")

    mask = spherical_mask(n_bits, centre=(0.5, 0.5, 0.5), radius=0.3)
    check("Mask shape",
          mask.shape == (N, N, N))
    check("Mask dtype bool",
          mask.dtype == bool)
    check("Mask has True entries",
          np.sum(mask) > 0)
    check("Mask centre is True",
          mask[N // 2, N // 2, N // 2])

    # Corners should be outside
    check("Corner (0,0,0) is False",
          not mask[0, 0, 0])

    # Shell mask
    shell = spherical_shell_mask(
        n_bits, centre=(0.5, 0.5, 0.5),
        inner_radius=0.2, outer_radius=0.3,
    )
    check("Shell has True entries",
          np.sum(shell) > 0)
    check("Shell ⊂ outer sphere",
          np.all(shell <= spherical_mask(n_bits, (0.5, 0.5, 0.5), 0.3)))

    # ==================================================================
    # 4. 3D Kronecker MPO embedding
    # ==================================================================
    print("\n--- Test 4: Kronecker MPO embedding ---")

    I_1d = [c.astype(np.complex128) for c in identity_mpo(n_bits)]

    # Embed identity along dim=0 → should give 3n identity-like cores
    I_3d = _embed_1d_mpo_complex(I_1d, dim=0, n_bits=n_bits)
    check("Embedded MPO has 3n cores",
          len(I_3d) == 3 * n_bits)

    # Each core should have physical dimension 2
    all_phys_2 = all(c.shape[1] == 2 and c.shape[2] == 2 for c in I_3d)
    check("All cores have physical dim 2×2",
          all_phys_2)

    # ==================================================================
    # 5. 3D identity MPO acts correctly
    # ==================================================================
    print("\n--- Test 5: 3D identity MPO ---")

    total_sites = 3 * n_bits
    I_full = [c.astype(np.complex128) for c in identity_mpo(total_sites)]
    check("3D identity MPO has 3n cores",
          len(I_full) == total_sites)

    # Apply to a test vector
    from tensornet.em.qtt_helmholtz import array_to_tt, reconstruct_1d
    from tensornet.qtt.sparse_direct import tt_matvec

    N3 = N ** 3
    test_vec = rng.standard_normal(N3) + 1j * rng.standard_normal(N3)
    test_tt = array_to_tt(test_vec, max_rank=64)
    result_tt = tt_matvec(I_full, test_tt)
    result_dense = reconstruct_1d(result_tt)
    id_err = np.linalg.norm(result_dense - test_vec) / np.linalg.norm(test_vec)
    check("Identity MPO preserves vector (err < 1e-10)",
          id_err < 1e-10,
          f"err = {id_err:.2e}")

    # ==================================================================
    # 6. 3D Laplacian MPO structure
    # ==================================================================
    print("\n--- Test 6: 3D Laplacian MPO structure ---")

    # Build Helmholtz MPO (small grid, free space)
    pml = PMLConfig.for_problem(n_bits=n_bits, k=6.0, target_R_dB=-20.0)
    H = helmholtz_mpo_3d_pml(
        n_bits=n_bits, k=6.0, pml=pml, max_rank=32,
    )
    check("Helmholtz MPO has 3n cores",
          len(H) == 3 * n_bits)

    # Bond dimensions bounded
    max_bond = max(
        max(c.shape[0], c.shape[3]) if c.ndim == 4 else
        max(c.shape[0], c.shape[2])
        for c in H
    )
    check("Max bond dim < 100",
          max_bond < 100,
          f"max_bond = {max_bond}")

    # ==================================================================
    # 7. Helmholtz MPO with PML (assembles)
    # ==================================================================
    print("\n--- Test 7: Helmholtz MPO with PML ---")

    # Build with explicit ε
    eps_3d = np.ones((N, N, N), dtype=np.complex128)
    eps_3d[2:6, 2:6, 2:6] = 4.0  # Dielectric block

    H_eps = helmholtz_mpo_3d_pml(
        n_bits=n_bits, k=6.0, pml=pml,
        eps_3d=eps_3d, max_rank=32,
    )
    check("Helmholtz with ε has 3n cores",
          len(H_eps) == 3 * n_bits)

    # ==================================================================
    # 8. Point source in QTT
    # ==================================================================
    print("\n--- Test 8: Point source ---")

    src = point_source_3d(
        n_bits=n_bits,
        position=(0.5, 0.5, 0.5),
        width=0.05,
        max_rank=32,
    )
    check("Source has 3n cores",
          len(src) == 3 * n_bits)

    # Reconstruct and check nonzero
    src_3d = reconstruct_3d(src, n_bits)
    check("Source is nonzero",
          np.linalg.norm(src_3d) > 0)

    # Source is localised near centre
    centre_val = abs(src_3d[N // 2, N // 2, N // 2])
    corner_val = abs(src_3d[0, 0, 0])
    check("Source localised (centre > corner)",
          centre_val > corner_val,
          f"centre={centre_val:.2e}, corner={corner_val:.2e}")

    # ==================================================================
    # 9. Gap source in QTT
    # ==================================================================
    print("\n--- Test 9: Gap source ---")

    gap_src = gap_source_3d(
        n_bits=n_bits,
        feed_position=(0.5, 0.5, 0.3),
        gap_height=0.1,
        gap_radius=0.1,
        max_rank=32,
    )
    check("Gap source has 3n cores",
          len(gap_src) == 3 * n_bits)

    gap_3d = reconstruct_3d(gap_src, n_bits)
    check("Gap source is nonzero",
          np.linalg.norm(gap_3d) > 0)

    # ==================================================================
    # 10. S₁₁ extraction infrastructure
    # ==================================================================
    print("\n--- Test 10: S₁₁ extraction ---")

    # Use a synthetic field (plane wave)
    k0 = 6.0
    E_plane = np.zeros((N, N, N), dtype=np.complex128)
    coords_1d = np.linspace(1.0 / (2 * N), 1.0 - 1.0 / (2 * N), N)
    for iz in range(N):
        E_plane[:, :, iz] = np.exp(1j * k0 * coords_1d[iz])

    E_tt = array_3d_to_tt(E_plane, n_bits, max_rank=32)
    s11 = extract_s11_3d(
        E_cores=E_tt,
        n_bits=n_bits,
        k0=k0,
        feed_position=(0.5, 0.5, 0.3),
        ref_distance=0.1,
        damping=0.01,
    )
    check("S₁₁ is complex",
          isinstance(s11, complex))

    # Impedance computation
    Z = compute_impedance_3d(0.5 + 0.1j)
    check("Z_in is finite",
          np.isfinite(Z))

    Z_matched = compute_impedance_3d(0.0)
    check("Z_in = Z₀ for S₁₁=0",
          abs(Z_matched - 50.0) < 1e-10)

    # ==================================================================
    # 11. PEC penalty operator
    # ==================================================================
    print("\n--- Test 11: PEC penalty ---")

    cond = np.zeros((N, N, N), dtype=bool)
    cond[3:5, 3:5, 3:5] = True

    P = build_pec_penalty_3d(n_bits, cond, penalty=1e6, max_rank=32)
    check("PEC penalty has 3n cores",
          len(P) == 3 * n_bits)

    # ==================================================================
    # 12. AntennaGeometry3D container
    # ==================================================================
    print("\n--- Test 12: AntennaGeometry3D ---")

    cfg = ChuProblemConfig(
        frequency_hz=1e9,
        ka=0.3,
        n_bits=3,
        domain_wavelengths=0.15,
        max_rank=32,
        n_sweeps=10,
        solver_tol=1e-2,
    )

    geom = AntennaGeometry3D(config=cfg)
    check("Geometry has design voxels",
          geom.n_design > 0,
          f"n_design = {geom.n_design}")

    check("Density shape matches n_design",
          len(geom.density) == geom.n_design)

    check("Design mask shape",
          geom.design_mask.shape == (N, N, N))

    check("Feed position has 3 coords",
          len(geom.feed_position) == 3)

    # Build conductor mask
    cmask = geom.build_conductor_mask(beta=8.0)
    check("Conductor mask shape",
          cmask.shape == (N, N, N))

    # Build source
    src_geom = geom.build_source(max_rank=32)
    check("Geometry source has 3n cores",
          len(src_geom) == 3 * n_bits)

    # ==================================================================
    # 13. 3D forward solve (free-space, small grid)
    # ==================================================================
    print("\n--- Test 13: 3D forward solve ---")

    pml_small = PMLConfig.for_problem(
        n_bits=n_bits, k=cfg.k0_normalised, target_R_dB=-20.0,
    )
    rhs_small = point_source_3d(
        n_bits=n_bits,
        position=(0.5, 0.5, 0.5),
        width=0.05,
        max_rank=32,
    )
    result = solve_helmholtz_3d(
        n_bits=n_bits,
        k=cfg.k0_normalised,
        pml=pml_small,
        rhs_cores=rhs_small,
        max_rank=32,
        n_sweeps=10,
        tol=0.1,
        damping=0.05,
        verbose=False,
    )
    check("Solver converges or reaches max sweeps",
          result.final_residual < 10.0,
          f"residual = {result.final_residual:.2e}")

    E_sol = reconstruct_3d(result.x, n_bits)
    check("Solution is nonzero",
          np.linalg.norm(E_sol) > 0)

    check("Solution is finite",
          np.all(np.isfinite(E_sol)))

    # ==================================================================
    # 14. ChuProblemConfig derived quantities
    # ==================================================================
    print("\n--- Test 14: ChuProblemConfig ---")

    cfg2 = ChuProblemConfig(frequency_hz=1e9, ka=0.3, n_bits=6)

    check("Wavelength ≈ 300 mm",
          abs(cfg2.wavelength - 0.3) < 0.01,
          f"λ = {cfg2.wavelength:.4f} m")

    check("Sphere radius ≈ 14.3 mm",
          abs(cfg2.sphere_radius * 1e3 - 14.3) < 0.5,
          f"a = {cfg2.sphere_radius*1e3:.2f} mm")

    check("k₀ ≈ 20.9 rad/m",
          abs(cfg2.k0 - 2 * math.pi / 0.3) < 0.5,
          f"k₀ = {cfg2.k0:.2f}")

    check("Q_Chu ≈ 40.4",
          abs(cfg2.q_chu - 40.37) < 0.1)

    check("N = 64",
          cfg2.N == 64)

    check("Summary is string",
          isinstance(cfg2.summary(), str))

    # ==================================================================
    # 15. ChuOptConfig defaults
    # ==================================================================
    print("\n--- Test 15: ChuOptConfig ---")

    opt = ChuOptConfig()
    check("Default max_iterations > 0",
          opt.max_iterations > 0)
    check("Default beta_init < beta_max",
          opt.beta_init < opt.beta_max)
    check("Default learning_rate > 0",
          opt.learning_rate > 0)
    check("Default eta = 0.5",
          opt.eta == 0.5)

    # ==================================================================
    # 16. Objective functions
    # ==================================================================
    print("\n--- Test 16: Objective functions ---")

    s11_test = 0.3 - 0.2j
    val, grad = objective_minimize_s11(s11_test)
    check("minimize_s11 value = |S₁₁|²",
          abs(val - abs(s11_test) ** 2) < 1e-12)
    check("minimize_s11 grad = conj(S₁₁)",
          abs(grad - np.conj(s11_test)) < 1e-12)

    target_fn = objective_target_s11_db(-15.0)
    val2, grad2 = target_fn(0.1 + 0.05j)
    check("target_s11_db returns float value",
          isinstance(val2, float))
    check("target_s11_db returns complex gradient",
          isinstance(grad2, complex))

    # ==================================================================
    # 17. Heaviside binarisation on 3D density
    # ==================================================================
    print("\n--- Test 17: Heaviside on 3D density ---")

    from tensornet.em.topology_opt import heaviside_projection, binarisation_metric

    rho_test = rng.uniform(0, 1, 100)
    rho_proj = heaviside_projection(rho_test, beta=16.0)
    check("Projected density ∈ [0,1]",
          np.all(rho_proj >= 0) and np.all(rho_proj <= 1))
    # binarisation_metric returns 0 when binary, 1 when uniform 0.5
    check("High β → low binarisation (near binary)",
          binarisation_metric(rho_proj) < 0.3,
          f"binarisation = {binarisation_metric(rho_proj):.4f}")

    # ==================================================================
    # 18. 3D conductor mask generation
    # ==================================================================
    print("\n--- Test 18: 3D conductor mask ---")

    cfg3 = ChuProblemConfig(
        frequency_hz=1e9, ka=0.3, n_bits=3,
        domain_wavelengths=1.0, max_rank=16,
    )
    geom3 = AntennaGeometry3D(
        config=cfg3,
        density=np.ones(
            int(np.sum(spherical_mask(3, (0.5, 0.5, 0.5),
                                      cfg3.sphere_radius_normalised))),
        ),
    )
    full_mask = geom3.build_conductor_mask(beta=32.0)
    check("Full density → all design voxels are conductor",
          np.sum(full_mask) == geom3.n_design,
          f"conductor={np.sum(full_mask)}, design={geom3.n_design}")

    # ==================================================================
    # 19. Integration: solve_and_extract_s11
    # ==================================================================
    print("\n--- Test 19: solve_and_extract_s11 ---")

    cfg_int = ChuProblemConfig(
        frequency_hz=1e9, ka=0.3, n_bits=3,
        domain_wavelengths=1.0, max_rank=16,
        n_sweeps=5, solver_tol=0.5, damping=0.05,
    )
    geom_int = AntennaGeometry3D(config=cfg_int)

    s11_int, res_int, E_int = solve_and_extract_s11(
        geometry=geom_int,
        k0_norm=cfg_int.k0_normalised,
        beta=1.0,
        max_rank=16,
        n_sweeps=5,
        solver_tol=0.5,
        damping=0.05,
        verbose=False,
    )
    check("S₁₁ is complex",
          isinstance(s11_int, complex))
    check("Residual is finite",
          np.isfinite(res_int))
    check("E_cores returned",
          len(E_int) == 3 * cfg_int.n_bits)

    # ==================================================================
    # 20. Integration: adjoint gradient (smoke)
    # ==================================================================
    print("\n--- Test 20: Adjoint gradient (smoke) ---")

    J_val, grad_adj, s11_adj, res_adj = compute_adjoint_gradient_3d(
        geometry=geom_int,
        k0_norm=cfg_int.k0_normalised,
        objective_fn=objective_minimize_s11,
        beta=1.0,
        max_rank=16,
        n_sweeps=5,
        solver_tol=0.5,
        damping=0.05,
        verbose=False,
    )
    check("Objective value is finite",
          np.isfinite(J_val),
          f"J = {J_val}")
    check("Gradient has correct length",
          len(grad_adj) == geom_int.n_design,
          f"len={len(grad_adj)}, n_design={geom_int.n_design}")
    check("Gradient is finite",
          np.all(np.isfinite(grad_adj)))
    check("S₁₁ from adjoint is complex",
          isinstance(s11_adj, complex))

    # ==================================================================
    # 21. Monopole seed geometry
    # ==================================================================
    print("\n--- Test 21: Monopole seed geometry ---")

    from tensornet.em.chu_limit import PowerSweepResult

    cfg_seed = ChuProblemConfig(
        frequency_hz=1e9, ka=0.3, n_bits=3,
        domain_wavelengths=0.15, max_rank=16,
        n_sweeps=5, solver_tol=0.5,
    )
    geom_seed = AntennaGeometry3D.with_monopole_seed(cfg_seed)
    check("Monopole seed has correct length",
          len(geom_seed.density) == geom_seed.n_design)
    check("Monopole seed density in [0, 1]",
          np.all(geom_seed.density >= 0) and np.all(geom_seed.density <= 1))
    # Should have some high-density voxels (the wire)
    check("Monopole seed has wire (max > 0.8)",
          float(np.max(geom_seed.density)) > 0.8,
          f"max density = {np.max(geom_seed.density):.4f}")
    # Should have some low-density voxels (background)
    check("Monopole seed has background (min < 0.2)",
          float(np.min(geom_seed.density)) < 0.2,
          f"min density = {np.min(geom_seed.density):.4f}")

    # ==================================================================
    # 22. ChuOptConfig P_rad fields and damping schedule
    # ==================================================================
    print("\n--- Test 22: ChuOptConfig P_rad fields ---")

    opt_prad = ChuOptConfig()
    check("use_p_rad defaults False (power_adjoint is primary)",
          opt_prad.use_p_rad is False)
    check("use_monopole_seed defaults True",
          opt_prad.use_monopole_seed is True)
    check("damping_init > damping_final",
          opt_prad.damping_init > opt_prad.damping_final)

    # Damping schedule
    d0 = opt_prad.damping_at_iter(0)
    check("damping_at_iter(0) = damping_init",
          abs(d0 - opt_prad.damping_init) < 1e-12,
          f"got {d0}")
    d_end = opt_prad.damping_at_iter(opt_prad.damping_schedule_iters)
    check("damping_at_iter(end) = damping_final",
          abs(d_end - opt_prad.damping_final) < 1e-12,
          f"got {d_end}")
    d_mid = opt_prad.damping_at_iter(opt_prad.damping_schedule_iters // 2)
    expected_mid = 0.5 * (opt_prad.damping_init + opt_prad.damping_final)
    check("damping_at_iter(mid) ~ midpoint",
          abs(d_mid - expected_mid) < 0.01,
          f"got {d_mid}, expected ~{expected_mid}")
    # Beyond schedule: should clamp to final
    d_beyond = opt_prad.damping_at_iter(1000)
    check("damping_at_iter(1000) = damping_final",
          abs(d_beyond - opt_prad.damping_final) < 1e-12)

    # ==================================================================
    # 23. qtt_3d compute_input_power
    # ==================================================================
    print("\n--- Test 23: compute_input_power ---")

    from tensornet.em.qtt_3d import (
        compute_complex_power,
        compute_input_power,
        compute_reactive_power,
        compute_radiation_resistance,
        monopole_seed_density,
    )

    # Create simple known TT vectors and verify power computation
    n_test = 3
    N_test = 2 ** n_test
    # Uniform E = 1+0j, uniform J = 1+0j on all N^3 voxels
    E_uniform = np.ones((N_test, N_test, N_test), dtype=np.complex128)
    J_uniform = np.ones((N_test, N_test, N_test), dtype=np.complex128)
    E_tt = array_3d_to_tt(E_uniform, n_test, max_rank=4)
    J_tt = array_3d_to_tt(J_uniform, n_test, max_rank=4)
    vol = 0.001  # arbitrary
    S = compute_complex_power(E_tt, J_tt, vol)
    # S = -0.5 * <J, E> * vol = -0.5 * N^3 * vol (since <1, 1> = N^3)
    expected_S = -0.5 * (N_test ** 3) * vol
    check("Complex power S for uniform fields",
          abs(S - expected_S) / abs(expected_S) < 1e-6,
          f"S={S:.6e}, expected={expected_S:.6e}")

    p_in = compute_input_power(E_tt, J_tt, vol)
    check("P_in = Re(S)",
          abs(p_in - expected_S.real) < 1e-10)

    q_r = compute_reactive_power(E_tt, J_tt, vol)
    check("Q_reactive = Im(S) (should be ~0 for real fields)",
          abs(q_r) < 1e-10)

    # ==================================================================
    # 24. monopole_seed_density function
    # ==================================================================
    print("\n--- Test 24: monopole_seed_density ---")

    mask_24 = spherical_mask(n_test, (0.5, 0.5, 0.5), 0.3)
    seed_24 = monopole_seed_density(
        n_bits=n_test,
        centre=(0.5, 0.5, 0.5),
        sphere_radius=0.3,
        design_mask=mask_24,
        wire_radius_cells=1,
        base_density=0.05,
        wire_density=0.95,
        top_hat=False,
    )
    check("Seed length matches mask",
          len(seed_24) == int(np.sum(mask_24)))
    check("Seed in [0, 1]",
          np.all(seed_24 >= 0) and np.all(seed_24 <= 1))
    check("Seed has wire voxels",
          np.any(seed_24 > 0.9))
    check("Seed has background voxels",
          np.any(seed_24 < 0.1))

    # ==================================================================
    # 25. Spherical Hankel functions
    # ==================================================================
    print("\n--- Test 25: Spherical Hankel functions ---")

    from tensornet.em.qtt_3d import _spherical_hankel_h1, _spherical_hankel_h2

    z_test = 1.5 + 0.1j
    h1_0 = _spherical_hankel_h1(0, z_test)
    # h1_0(z) = -i * exp(iz) / z
    expected_h1_0 = -1j * np.exp(1j * z_test) / z_test
    check("h1(0, z) matches formula",
          abs(h1_0 - expected_h1_0) < 1e-10,
          f"got {h1_0}, expected {expected_h1_0}")

    h2_0 = _spherical_hankel_h2(0, z_test)
    # h2_0(z) = i * exp(-iz) / z
    expected_h2_0 = 1j * np.exp(-1j * z_test) / z_test
    check("h2(0, z) matches formula",
          abs(h2_0 - expected_h2_0) < 1e-10,
          f"got {h2_0}, expected {expected_h2_0}")

    # h1 and h2 are conjugates for real z
    z_real = 2.0
    h1_r = _spherical_hankel_h1(1, z_real)
    h2_r = _spherical_hankel_h2(1, z_real)
    check("h1(n,x) = conj(h2(n,x)) for real x",
          abs(h1_r - np.conj(h2_r)) < 1e-10)

    # ==================================================================
    # 26. Legendre polynomials
    # ==================================================================
    print("\n--- Test 26: Legendre polynomials ---")

    from tensornet.em.qtt_3d import _legendre_p

    check("P_0(x) = 1",
          abs(_legendre_p(0, 0.5) - 1.0) < 1e-12)
    check("P_1(x) = x",
          abs(_legendre_p(1, 0.7) - 0.7) < 1e-12)
    check("P_2(x) = (3x^2 - 1)/2",
          abs(_legendre_p(2, 0.6) - (3 * 0.36 - 1) / 2) < 1e-12)

    # ==================================================================
    # 27. solve_and_extract_p_rad (smoke)
    # ==================================================================
    print("\n--- Test 27: solve_and_extract_p_rad ---")

    from tensornet.em.chu_limit import solve_and_extract_p_rad

    cfg_prad = ChuProblemConfig(
        frequency_hz=1e9, ka=0.3, n_bits=3,
        domain_wavelengths=1.0, max_rank=16,
        n_sweeps=5, solver_tol=0.5, damping=0.05,
    )
    geom_prad = AntennaGeometry3D(config=cfg_prad)

    p_in_27, q_react_27, res_27, E_27, J_27 = solve_and_extract_p_rad(
        geometry=geom_prad,
        k0_norm=cfg_prad.k0_normalised,
        beta=1.0,
        max_rank=16,
        n_sweeps=5,
        solver_tol=0.5,
        damping=0.05,
        verbose=False,
    )
    check("P_in is finite",
          np.isfinite(p_in_27),
          f"P_in = {p_in_27}")
    check("Q_reactive is finite",
          np.isfinite(q_react_27))
    check("E_cores returned",
          len(E_27) == 3 * cfg_prad.n_bits)
    check("J_cores returned",
          len(J_27) == 3 * cfg_prad.n_bits)

    # ==================================================================
    # 28. compute_adjoint_gradient_p_rad (smoke)
    # ==================================================================
    print("\n--- Test 28: P_rad adjoint gradient ---")

    from tensornet.em.chu_limit import compute_adjoint_gradient_p_rad

    obj_28, grad_28, pin_28, qr_28, res_28 = compute_adjoint_gradient_p_rad(
        geometry=geom_prad,
        k0_norm=cfg_prad.k0_normalised,
        beta=1.0,
        max_rank=16,
        n_sweeps=5,
        solver_tol=0.5,
        damping=0.05,
        verbose=False,
    )
    check("P_rad objective is finite",
          np.isfinite(obj_28),
          f"obj = {obj_28}")
    check("P_rad gradient has correct length",
          len(grad_28) == geom_prad.n_design)
    check("P_rad gradient is finite",
          np.all(np.isfinite(grad_28)))
    check("obj = -P_in",
          abs(obj_28 - (-pin_28)) < 1e-12)

    # ==================================================================
    # 29. ChuOptResult dataclass with defaults
    # ==================================================================
    print("\n--- Test 29: ChuOptResult defaults ---")

    from tensornet.em.chu_limit import ChuOptResult

    dummy_result = ChuOptResult(
        density_final=np.array([0.5]),
        conductor_mask=np.array([True]),
    )
    check("ChuOptResult creates with defaults",
          dummy_result.n_iterations == 0)
    check("ChuOptResult summary is string",
          isinstance(dummy_result.summary(), str))
    check("ChuOptResult s11_db is finite",
          np.isfinite(dummy_result.s11_db))


    # ==================================================================
    # 30. Conductivity SIMP model
    # ==================================================================
    print("\n--- Test 30: Conductivity SIMP model ---")

    from tensornet.em.chu_limit import simp_sigma, simp_dsigma_drho

    rho_test = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    sig = simp_sigma(rho_test, sigma_min=0.0, sigma_max=100.0, p=3.0)
    check("simp_sigma(0) = sigma_min",
          abs(sig[0] - 0.0) < 1e-12)
    check("simp_sigma(1) = sigma_max",
          abs(sig[-1] - 100.0) < 1e-12)
    check("simp_sigma monotone",
          all(sig[i] <= sig[i+1] for i in range(len(sig)-1)))
    check("simp_sigma(0.5, p=3) = 100*0.125 = 12.5",
          abs(sig[2] - 12.5) < 1e-10)

    dsig = simp_dsigma_drho(rho_test, sigma_min=0.0, sigma_max=100.0, p=3.0)
    check("dsigma/drho(1) = p * sigma_max",
          abs(dsig[-1] - 300.0) < 1e-8)
    check("dsigma/drho is non-negative",
          np.all(dsig >= 0.0))

    # Finite difference check for dsigma
    delta_rho = 1e-5
    rho_mid = np.array([0.5])
    sig_p = simp_sigma(rho_mid + delta_rho, 0.0, 100.0, 3.0)
    sig_m = simp_sigma(rho_mid - delta_rho, 0.0, 100.0, 3.0)
    dsig_fd = (sig_p - sig_m) / (2 * delta_rho)
    dsig_an = simp_dsigma_drho(rho_mid, 0.0, 100.0, 3.0)
    check("simp_dsigma FD matches analytic",
          abs(dsig_fd[0] - dsig_an[0]) / abs(dsig_an[0]) < 1e-6,
          f"FD={dsig_fd[0]:.6f}, analytic={dsig_an[0]:.6f}")

    # ==================================================================
    # 31. PML sigma 3D construction
    # ==================================================================
    print("\n--- Test 31: PML sigma 3D ---")

    from tensornet.em.chu_limit import build_pml_sigma_3d
    from tensornet.em.boundaries import PMLConfig as _PMLConfig_31

    nb_31 = 5
    N_31 = 2 ** nb_31
    k_31 = 2.0 * np.pi
    pml_31 = _PMLConfig_31(n_cells=4, sigma_max=5.0, poly_order=3,
                             kappa_max=1.0, damping=0.01)
    sigma_pml_3d = build_pml_sigma_3d(nb_31, k_31, pml_31)

    check("PML sigma shape correct",
          sigma_pml_3d.shape == (N_31, N_31, N_31))
    check("PML sigma non-negative",
          np.all(sigma_pml_3d >= 0.0))
    check("PML sigma zero at center",
          sigma_pml_3d[N_31//2, N_31//2, N_31//2] < 1e-10)
    check("PML sigma nonzero at boundary",
          sigma_pml_3d.max() > 0.0)

    # ==================================================================
    # 32. MPO Hermitian conjugate
    # ==================================================================
    print("\n--- Test 32: MPO Hermitian conjugate ---")

    from tensornet.em.chu_limit import mpo_hermitian_conjugate

    # Build a simple complex MPO core and check swap + conjugate
    core_a = np.random.randn(2, 4, 4, 3) + 1j * np.random.randn(2, 4, 4, 3)
    core_b = np.random.randn(3, 4, 4, 1) + 1j * np.random.randn(3, 4, 4, 1)
    cores_orig = [core_a, core_b]
    cores_hc = mpo_hermitian_conjugate(cores_orig)

    check("H^H core count preserved",
          len(cores_hc) == 2)
    check("H^H core shape transposed",
          cores_hc[0].shape == (2, 4, 4, 3) and
          cores_hc[1].shape == (3, 4, 4, 1))
    check("H^H conjugate + swap axes",
          np.allclose(cores_hc[0], np.conj(core_a.swapaxes(1, 2))))

    # ==================================================================
    # 33. AntennaGeometry3D conductivity eps
    # ==================================================================
    print("\n--- Test 33: Conductivity eps_3d ---")

    cfg_33 = ChuProblemConfig(n_bits=4, sigma_max=100.0, simp_p=2.0)
    geom_33 = AntennaGeometry3D(config=cfg_33)
    eps_cond = geom_33.build_conductivity_eps_3d(beta=1.0)

    check("Conductivity eps shape",
          eps_cond.shape == (cfg_33.N,) * 3)
    check("Conductivity eps complex",
          np.iscomplexobj(eps_cond))
    check("Background eps = 1",
          np.allclose(eps_cond[~geom_33.design_mask], 1.0))
    # In design region, eps should have negative imaginary part (lossy)
    design_eps = eps_cond[geom_33.design_mask]
    check("Design region has real part = 1",
          np.allclose(np.real(design_eps), 1.0))
    check("Design region has imag part <= 0 (lossy)",
          np.all(np.imag(design_eps) <= 1e-12))

    # ==================================================================
    # 34. Augmented Lagrangian volume constraint
    # ==================================================================
    print("\n--- Test 34: Augmented Lagrangian volume ---")

    from tensornet.em.chu_limit import augmented_lagrangian_volume

    rho_v = np.full(100, 0.4)
    Jv, dJv, V, g = augmented_lagrangian_volume(rho_v, vol_target=0.3,
                                                  al_lambda=0.0, al_mu=10.0)
    check("Volume fraction correct",
          abs(V - 0.4) < 1e-12)
    check("Constraint violation correct",
          abs(g - 0.1) < 1e-12)
    check("J_vol > 0 when over target",
          Jv > 0)
    check("dJ_vol/drho positive when over target",
          np.all(dJv > 0))

    # Test with lambda (dual variable)
    Jv2, dJv2, V2, g2 = augmented_lagrangian_volume(
        rho_v, vol_target=0.3, al_lambda=5.0, al_mu=10.0)
    check("Lambda shifts J_vol upward",
          Jv2 > Jv)

    # ==================================================================
    # 35. PowerObjectiveConfig defaults
    # ==================================================================
    print("\n--- Test 35: PowerObjectiveConfig ---")

    from tensornet.em.chu_limit import PowerObjectiveConfig

    poc = PowerObjectiveConfig()
    check("PowerObjectiveConfig alpha_loss default",
          poc.alpha_loss == 0.1)
    check("PowerObjectiveConfig vol_target default",
          poc.vol_target == 0.3)
    check("PowerObjectiveConfig al_lambda default",
          poc.al_lambda == 0.0)
    check("PowerObjectiveConfig al_mu default",
          poc.al_mu == 10.0)

    # ==================================================================
    # 36. ChuOptConfig continuation schedules
    # ==================================================================
    print("\n--- Test 36: ChuOptConfig continuation ---")

    oc36 = ChuOptConfig()
    check("use_power_adjoint defaults True",
          oc36.use_power_adjoint is True)
    check("sigma_max_at_iter(0) = sigma_max_init",
          abs(oc36.sigma_max_at_iter(0) - oc36.sigma_max_init) < 1e-12)
    check("sigma_max ramps to final",
          abs(oc36.sigma_max_at_iter(oc36.sigma_ramp_iters) -
              oc36.sigma_max_final) < 1e-12)
    check("simp_p_at_iter(0) = simp_p_init",
          abs(oc36.simp_p_at_iter(0) - oc36.simp_p_init) < 1e-12)
    check("simp_p ramps to final",
          abs(oc36.simp_p_at_iter(oc36.simp_p_ramp_iters) -
              oc36.simp_p_final) < 1e-12)

    # ==================================================================
    # 37. Forward solve with conductivity SIMP
    # ==================================================================
    print("\n--- Test 37: Forward solve conductivity ---")

    from tensornet.em.chu_limit import solve_forward_conductivity

    cfg_37 = ChuProblemConfig(n_bits=4, max_rank=16, n_sweeps=5,
                               solver_tol=0.5, sigma_max=50.0, simp_p=2.0)
    geom_37 = AntennaGeometry3D(config=cfg_37)
    k_37 = cfg_37.k0_normalised

    H_cores_37, E_cores_37, res_37 = solve_forward_conductivity(
        geometry=geom_37, k0_norm=k_37, beta=1.0, eta=0.5,
        max_rank=16, n_sweeps=5, solver_tol=0.5, damping=0.01,
    )

    check("H_cores returned",
          isinstance(H_cores_37, list) and len(H_cores_37) > 0)
    check("E_cores returned",
          isinstance(E_cores_37, list) and len(E_cores_37) > 0)
    check("Residual is finite",
          np.isfinite(res_37))

    # ==================================================================
    # 38. Power metrics computation
    # ==================================================================
    print("\n--- Test 38: Power metrics ---")

    from tensornet.em.chu_limit import compute_power_metrics, PowerMetrics

    E_3d_38 = reconstruct_3d(E_cores_37, cfg_37.n_bits)
    pml_38 = cfg_37.pml_config()

    pm = compute_power_metrics(
        E_3d_38, geom_37, k_37, pml_38,
        beta=1.0, eta=0.5, filter_radius=0, damping=0.01,
    )

    check("PowerMetrics P_pml finite",
          np.isfinite(pm.P_pml))
    check("PowerMetrics P_cond finite",
          np.isfinite(pm.P_cond))
    check("PowerMetrics P_input finite",
          np.isfinite(pm.P_input))
    check("PowerMetrics P_pml >= 0",
          pm.P_pml >= 0)
    check("PowerMetrics P_cond >= 0",
          pm.P_cond >= 0)
    check("PowerMetrics eta_rad in [0,1]",
          0.0 <= pm.eta_rad <= 1.0 + 1e-12)

    # ==================================================================
    # 39. Exact adjoint gradient (smoke test)
    # ==================================================================
    print("\n--- Test 39: Exact adjoint gradient ---")

    from tensornet.em.chu_limit import compute_adjoint_gradient_power

    cfg_39 = ChuProblemConfig(n_bits=4, max_rank=16, n_sweeps=5,
                               solver_tol=0.5, sigma_max=50.0, simp_p=2.0)
    geom_39 = AntennaGeometry3D(config=cfg_39)
    pml_39 = cfg_39.pml_config()
    obj_cfg_39 = PowerObjectiveConfig(alpha_loss=0.1, vol_target=0.3)

    J_39, grad_39, metrics_39, res_39 = compute_adjoint_gradient_power(
        geometry=geom_39, k0_norm=cfg_39.k0_normalised,
        pml=pml_39, obj_cfg=obj_cfg_39, beta=1.0, eta_h=0.5,
        filter_radius=0, max_rank=16, n_sweeps=5,
        solver_tol=0.5, damping=0.01,
    )

    check("Power adjoint J is finite",
          np.isfinite(J_39))
    check("Power adjoint grad length correct",
          len(grad_39) == geom_39.n_design)
    check("Power adjoint grad is finite",
          np.all(np.isfinite(grad_39)))
    check("Power adjoint grad nonzero",
          np.any(np.abs(grad_39) > 0))
    check("Power adjoint metrics returned",
          isinstance(metrics_39, PowerMetrics))

    # ==================================================================
    # 40. ChuOptResult with power_metrics_history
    # ==================================================================
    print("\n--- Test 40: ChuOptResult power fields ---")

    from tensornet.em.chu_limit import ChuOptResult as _CR40

    result_40 = _CR40(
        density_final=np.array([0.5]),
        conductor_mask=np.array([True]),
        power_metrics_history=[pm],
    )
    summary_40 = result_40.summary()
    check("ChuOptResult with power_metrics_history",
          "P_pml" in summary_40)
    check("ChuOptResult volume fraction",
          0.0 <= result_40.volume_fraction <= 1.0)

    # ==================================================================
    # 41. Staged log objective: J = -log(P_pml+eps) + AL
    # ==================================================================
    print("\n--- Test 41: Staged log objective ---")

    from tensornet.em.chu_limit import PowerObjectiveConfig as _POC41

    poc41 = _POC41(alpha_loss=0.0, vol_target=0.3, use_log_objective=True)
    check("POC use_log_objective default True",
          poc41.use_log_objective is True)
    check("POC log_eps default 1e-12",
          abs(poc41.log_eps - 1e-12) < 1e-20)

    # Gradient with alpha=0 (pure radiation stage)
    cfg_41 = ChuProblemConfig(n_bits=4, max_rank=16, n_sweeps=5,
                               solver_tol=0.5, sigma_max=50.0, simp_p=2.0)
    geom_41 = AntennaGeometry3D(config=cfg_41)
    pml_41 = cfg_41.pml_config()
    J_41, grad_41, met_41, _ = compute_adjoint_gradient_power(
        geometry=geom_41, k0_norm=cfg_41.k0_normalised,
        pml=pml_41, obj_cfg=poc41, beta=1.0, eta_h=0.5,
        filter_radius=0, max_rank=16, n_sweeps=5,
        solver_tol=0.5, damping=0.01,
    )
    check("Log objective J finite (alpha=0)",
          np.isfinite(J_41))
    check("Log grad finite (alpha=0)",
          np.all(np.isfinite(grad_41)))
    check("Log grad nonzero (alpha=0)",
          np.any(np.abs(grad_41) > 0))

    # With alpha > 0 (stage 1)
    poc41b = _POC41(alpha_loss=0.5, vol_target=0.3, use_log_objective=True)
    J_41b, grad_41b, _, _ = compute_adjoint_gradient_power(
        geometry=geom_41, k0_norm=cfg_41.k0_normalised,
        pml=pml_41, obj_cfg=poc41b, beta=1.0, eta_h=0.5,
        filter_radius=0, max_rank=16, n_sweeps=5,
        solver_tol=0.5, damping=0.01,
    )
    check("Log objective J finite (alpha=0.5)",
          np.isfinite(J_41b))
    check("Log grad finite (alpha=0.5)",
          np.all(np.isfinite(grad_41b)))
    # alpha > 0 should shift the gradient
    check("Alpha changes gradient",
          not np.allclose(grad_41, grad_41b))

    # Linear fallback
    poc41c = _POC41(alpha_loss=0.0, vol_target=0.3, use_log_objective=False)
    J_41c, grad_41c, _, _ = compute_adjoint_gradient_power(
        geometry=geom_41, k0_norm=cfg_41.k0_normalised,
        pml=pml_41, obj_cfg=poc41c, beta=1.0, eta_h=0.5,
        filter_radius=0, max_rank=16, n_sweeps=5,
        solver_tol=0.5, damping=0.01,
    )
    check("Linear fallback J finite",
          np.isfinite(J_41c))

    # ==================================================================
    # 42. PowerMetrics extended fields
    # ==================================================================
    print("\n--- Test 42: PowerMetrics extended fields ---")

    check("PowerMetrics has E2_metal_avg",
          hasattr(met_41, 'E2_metal_avg'))
    check("PowerMetrics E2_metal_avg finite",
          np.isfinite(met_41.E2_metal_avg))
    check("PowerMetrics has P_pml_norm",
          hasattr(met_41, 'P_pml_norm'))

    # ==================================================================
    # 43. ChuOptConfig staged fields
    # ==================================================================
    print("\n--- Test 43: ChuOptConfig staged fields ---")

    oc43 = ChuOptConfig()
    check("alpha_intro_iter default 20",
          oc43.alpha_intro_iter == 20)
    check("feed_seed_clamp_iters default 15",
          oc43.feed_seed_clamp_iters == 15)
    check("feed_seed_clamp_radius default 2",
          oc43.feed_seed_clamp_radius == 2)

    # ==================================================================
    # 44. M_dead metric and dynamic alpha config
    # ==================================================================
    print("\n--- Test 44: M_dead metric and dynamic alpha config ---")

    check("PowerMetrics has M_dead",
          hasattr(met_41, 'M_dead'))
    check("PowerMetrics M_dead finite",
          np.isfinite(met_41.M_dead))
    check("PowerMetrics M_dead default 0",
          met_41.M_dead == 0.0)  # not set by compute_power_metrics alone

    oc44 = ChuOptConfig()
    check("alpha_intro_mode default auto",
          oc44.alpha_intro_mode == "auto")
    check("alpha_stable_window default 20",
          oc44.alpha_stable_window == 20)
    check("m_dead_percentile default 10",
          oc44.m_dead_percentile == 10.0)

    # Verify fixed mode still works
    oc44f = ChuOptConfig(alpha_intro_mode="fixed", alpha_intro_iter=5)
    check("alpha_intro_mode fixed accepted",
          oc44f.alpha_intro_mode == "fixed")
    check("alpha_intro_iter respected in fixed mode",
          oc44f.alpha_intro_iter == 5)

    # ==================================================================
    # 45. Coupling constraint (augmented Lagrangian)
    # ==================================================================
    print("\n--- Test 45: Soft coupling constraint ---")

    from tensornet.em.chu_limit import augmented_lagrangian_coupling

    # 10 design voxels, mask = first 3, threshold = 0.5
    rho_45 = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5])
    mask_45 = np.array([True, True, True, False, False,
                        False, False, False, False, False])
    Jc_45, dJc_45, rn_45 = augmented_lagrangian_coupling(
        density=rho_45, coupling_mask=mask_45,
        threshold=0.5, al_lambda_c=0.0, al_mu_c=10.0,
    )
    check("Coupling rho_near correct",
          abs(rn_45 - 0.2) < 1e-10)
    check("Coupling J > 0 when violated",
          Jc_45 > 0.0)
    check("Coupling grad nonzero on mask",
          np.any(dJc_45[:3] != 0.0))
    check("Coupling grad zero off mask",
          np.all(dJc_45[3:] == 0.0))

    # Not violated: rho_near > threshold
    rho_45b = np.array([0.8, 0.9, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    Jc_45b, dJc_45b, rn_45b = augmented_lagrangian_coupling(
        density=rho_45b, coupling_mask=mask_45,
        threshold=0.5, al_lambda_c=0.0, al_mu_c=10.0,
    )
    check("Coupling no penalty when satisfied (lambda=0)",
          Jc_45b == 0.0)

    # ==================================================================
    # 46. ChuOptConfig coupling constraint fields
    # ==================================================================
    print("\n--- Test 46: ChuOptConfig coupling fields ---")

    oc46 = ChuOptConfig()
    check("use_coupling_constraint default True",
          oc46.use_coupling_constraint is True)
    check("coupling_density_threshold default 0.3",
          oc46.coupling_density_threshold == 0.3)
    check("coupling_al_mu_init default 10.0",
          oc46.coupling_al_mu_init == 10.0)
    check("coupling_radius default 3",
          oc46.coupling_radius == 3)

    # ==================================================================
    # 47. make_chu_antenna_schedule factory
    # ==================================================================
    print("\n--- Test 47: make_chu_antenna_schedule ---")

    from tensornet.em.chu_limit import make_chu_antenna_schedule

    for level in ("16", "32", "64", "128"):
        cfg_s, opt_s = make_chu_antenna_schedule(level)
        check(f"Schedule {level} returns ChuProblemConfig",
              isinstance(cfg_s, ChuProblemConfig))
        check(f"Schedule {level} returns ChuOptConfig",
              isinstance(opt_s, ChuOptConfig))
        check(f"Schedule {level} use_power_adjoint",
              opt_s.use_power_adjoint is True)
        check(f"Schedule {level} auto alpha",
              opt_s.alpha_intro_mode == "auto")

    # 64 schedule uses auto domain for adequate cells/radius
    cfg64, opt64 = make_chu_antenna_schedule("64")
    check("64 has meaningful design region",
          cfg64.N >= 64)
    check("64 filter_radius >= 1",
          opt64.filter_radius >= 1)
    # Verify auto-domain gives enough cells per radius
    cells_per_r = cfg64.sphere_radius / cfg64.h_physical
    check("64 cells/radius >= 6",
          cells_per_r >= 6)

    # 128 schedule has explicit domain for PML clearance
    cfg128, opt128 = make_chu_antenna_schedule("128")
    check("128 sigma_max_final >= 500",
          opt128.sigma_max_final >= 500.0)
    check("128 filter_radius >= 2",
          opt128.filter_radius >= 2)
    check("128 domain_wavelengths >= 0.5",
          cfg128.domain_wavelengths >= 0.5)

    # Invalid level
    try:
        make_chu_antenna_schedule("999")
        check("Invalid level raises", False)
    except ValueError:
        check("Invalid level raises", True)

    # ==================================================================
    # 48. Physics-valid schedule levels (1024, 4096)
    # ==================================================================
    print("\n--- Test 48: Physics-valid schedule levels ---")

    for level in ("1024", "4096"):
        cfg_pv, opt_pv = make_chu_antenna_schedule(level)
        check(f"Schedule {level} returns ChuProblemConfig",
              isinstance(cfg_pv, ChuProblemConfig))
        check(f"Schedule {level} returns ChuOptConfig",
              isinstance(opt_pv, ChuOptConfig))
        check(f"Schedule {level} use_power_adjoint",
              opt_pv.use_power_adjoint is True)
        check(f"Schedule {level} optimizer is adam",
              opt_pv.optimizer == "adam")

    # 1024: domain locked at 1.5λ, ~32 cells/radius
    cfg1024, opt1024 = make_chu_antenna_schedule("1024")
    check("1024 domain_wavelengths == 1.5",
          cfg1024.domain_wavelengths == 1.5)
    check("1024 N == 1024",
          cfg1024.N == 1024)
    cells_1024 = cfg1024.sphere_radius / cfg1024.h_physical
    check("1024 cells/radius >= 30",
          cells_1024 >= 30)
    check("1024 max_rank <= 128 (higher compress)",
          cfg1024.max_rank <= 128)
    check("1024 solver_tol <= 0.01",
          cfg1024.solver_tol <= 0.01)
    check("1024 filter_radius >= 3",
          opt1024.filter_radius >= 3)
    check("1024 learning_rate < 0.01",
          opt1024.learning_rate < 0.01)

    # 4096: domain locked at 1.5λ, ~130 cells/radius
    cfg4096, opt4096 = make_chu_antenna_schedule("4096")
    check("4096 domain_wavelengths == 1.5",
          cfg4096.domain_wavelengths == 1.5)
    check("4096 N == 4096",
          cfg4096.N == 4096)
    cells_4096 = cfg4096.sphere_radius / cfg4096.h_physical
    check("4096 cells/radius >= 100",
          cells_4096 >= 100)
    check("4096 max_rank <= 64 (maximum compress)",
          cfg4096.max_rank <= 64)
    check("4096 solver_tol <= 0.01",
          cfg4096.solver_tol <= 0.01)
    check("4096 sigma_max_final >= 2000",
          opt4096.sigma_max_final >= 2000.0)
    check("4096 filter_radius >= 5",
          opt4096.filter_radius >= 5)
    check("4096 learning_rate <= 0.001",
          opt4096.learning_rate <= 0.001)
    check("4096 max_iterations >= 300",
          opt4096.max_iterations >= 300)

    # PML clearance for 1024 and 4096
    for label, cfg_cl in [("1024", cfg1024), ("4096", cfg4096)]:
        pml_n = cfg_cl.pml_config().n_cells
        pml_frac = pml_n / cfg_cl.N
        sphere_frac = cfg_cl.sphere_radius_normalised
        clearance = (0.5 - sphere_frac - pml_frac) * cfg_cl.domain_size / cfg_cl.wavelength
        check(f"{label} PML clearance >= 0.3λ",
              clearance >= 0.3)

    # ==================================================================
    # 49. ChuValidation scorecard
    # ==================================================================
    print("\n--- Test 49: ChuValidation scorecard ---")

    from tensornet.em.chu_limit import ChuValidation

    v49 = ChuValidation()
    check("ChuValidation has Q_halfpower",
          hasattr(v49, "Q_halfpower"))
    check("ChuValidation has Q_impedance",
          hasattr(v49, "Q_impedance"))
    check("ChuValidation has P_pml",
          hasattr(v49, "P_pml"))
    check("ChuValidation has P_poynting",
          hasattr(v49, "P_poynting"))
    check("ChuValidation has Q_dual_pass",
          hasattr(v49, "Q_dual_pass"))
    check("ChuValidation has power_pass",
          hasattr(v49, "power_pass"))
    check("ChuValidation has resonance_pass",
          hasattr(v49, "resonance_pass"))
    check("ChuValidation has summary()",
          callable(getattr(v49, "summary", None)))

    # Test summary output
    summ49 = v49.summary()
    check("ChuValidation summary is string",
          isinstance(summ49, str))
    check("ChuValidation summary contains DUAL-Q",
          "DUAL-Q" in summ49)
    check("ChuValidation summary contains RESONANCE",
          "RESONANCE" in summ49)
    check("ChuValidation summary contains POWER VALIDATION",
          "POWER VALIDATION" in summ49)
    check("ChuValidation summary contains BOUND COMPARISON",
          "BOUND COMPARISON" in summ49)

    # ==================================================================
    # 50. compute_poynting_flux_shell smoke test
    # ==================================================================
    print("\n--- Test 50: compute_poynting_flux_shell ---")

    from tensornet.em.chu_limit import compute_poynting_flux_shell

    # Use a 16³ grid with a synthetic field to verify the function runs
    cfg_pf = ChuProblemConfig(
        frequency_hz=1e9, ka=0.3, n_bits=4,
        domain_wavelengths=0.0,
        max_rank=8, n_sweeps=2, solver_tol=1.0,
    )
    geom_pf = AntennaGeometry3D.with_monopole_seed(cfg_pf)
    # Create a simple outgoing spherical wave: E ~ exp(ikr)/r
    N_pf = cfg_pf.N
    h_pf = 1.0 / N_pf
    coords_pf = np.linspace(h_pf / 2, 1.0 - h_pf / 2, N_pf)
    E_pf = np.zeros((N_pf, N_pf, N_pf), dtype=complex)
    k0n_pf = cfg_pf.k0_normalised
    for ix in range(N_pf):
        for iy in range(N_pf):
            for iz in range(N_pf):
                dx = coords_pf[ix] - 0.5
                dy = coords_pf[iy] - 0.5
                dz = coords_pf[iz] - 0.5
                r = math.sqrt(dx**2 + dy**2 + dz**2)
                if r > 0.01:
                    E_pf[ix, iy, iz] = np.exp(1j * k0n_pf * r) / r

    P_pf = compute_poynting_flux_shell(
        E_pf, geom_pf, k0n_pf,
        shell_radius_normalised=0.2,
        shell_thickness_cells=2,
    )
    check("Poynting flux returns float",
          isinstance(P_pf, float))
    # Outgoing wave should give positive flux
    check("Poynting flux positive for outgoing wave",
          P_pf > 0)

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"  Chu Limit Tests: {passed}/{total} passed, {failed} failed")
    print(f"{'='*60}")

    return failed


if __name__ == "__main__":
    sys.exit(run_tests())
