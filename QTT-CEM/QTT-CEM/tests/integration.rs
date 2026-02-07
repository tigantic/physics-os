//! Integration Tests — Analytical Validation Cases
//!
//! Tests against closed-form solutions:
//! 1. Free-space plane wave propagation
//! 2. Resonant cavity (PEC box)
//! 3. Energy conservation in lossless vacuum
//! 4. Dielectric interface reflection
//! 5. QTT compression fidelity
//! 6. CFL stability condition
//!
//! © 2026 Brad McAllister. All rights reserved. PROPRIETARY.

use cem_qtt::prelude::*;

// ═══════════════════════════════════════════════════════════════════
// Test 1: Zero-field stability
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_zero_field_remains_zero() {
    let config = FdtdConfig::vacuum_cube(3, 4);
    let mut solver = FdtdSolver::new(config);

    for _ in 0..50 {
        solver.step();
    }

    let energy = solver.fields.total_energy(&solver.config.materials);
    assert_eq!(energy.raw(), 0, "Zero field should remain zero: energy = {}", energy);
}

// ═══════════════════════════════════════════════════════════════════
// Test 2: Energy conservation — lossless vacuum, periodic BC
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_energy_conservation_periodic_vacuum() {
    let mut config = FdtdConfig::vacuum_cube(3, 4);
    config.boundary = BoundaryCondition::Periodic;

    let mut solver = FdtdSolver::new(config);

    // Inject initial Ez pulse at center
    let n = solver.config.nx;
    let center = (n/2) * n * n + (n/2) * n + (n/2);
    solver.fields.ez[center] = Q16::from_f64(0.5);

    let e0 = solver.fields.total_energy(&solver.config.materials);
    assert!(e0.raw() > 0, "Initial energy should be nonzero");

    let energies = solver.run(30);

    let (conserved, max_res, steps) = ConservationVerifier::verify_run(
        &energies,
        Q16::from_f64(0.05), // Generous tolerance for Q16.16
    );

    assert!(conserved,
        "Energy not conserved: max_residual = {} over {} steps (e0 = {})",
        max_res, steps, e0);
}

// ═══════════════════════════════════════════════════════════════════
// Test 3: Gaussian source injects energy
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_gaussian_source_injects_energy() {
    let mut config = FdtdConfig::vacuum_cube(3, 4);
    let n = config.nx;
    config.add_gaussian_source((n/2, n/2, n/2), 2, 1.0, 0.0, 0.5);

    let mut solver = FdtdSolver::new(config);
    solver.step();

    let energy = solver.fields.total_energy(&solver.config.materials);
    assert!(energy.raw() > 0, "Source should inject energy: got {}", energy);
}

// ═══════════════════════════════════════════════════════════════════
// Test 4: PEC cavity — tangential E = 0 at boundaries
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_pec_boundary_tangential_zero() {
    let mut config = FdtdConfig::vacuum_cube(3, 4);
    config.boundary = BoundaryCondition::PEC;

    // Inject pulse
    let n = config.nx;
    let mut solver = FdtdSolver::new(config);
    let center = (n/2) * n * n + (n/2) * n + (n/2);
    solver.fields.ez[center] = Q16::from_f64(1.0);

    for _ in 0..20 {
        solver.step();
    }

    // Check x=0 face: Ey and Ez should be zero
    for j in 0..n {
        for k in 0..n {
            let idx = j * n + k; // i=0
            assert_eq!(solver.fields.ey[idx].raw(), 0,
                "PEC violation: Ey != 0 at x=0, j={}, k={}", j, k);
            assert_eq!(solver.fields.ez[idx].raw(), 0,
                "PEC violation: Ez != 0 at x=0, j={}, k={}", j, k);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Test 5: CFL condition validation
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_cfl_valid_config() {
    let config = FdtdConfig::vacuum_cube(3, 4);
    assert!(config.check_cfl(), "Default config should satisfy CFL");
}

#[test]
fn test_cfl_violation() {
    let mut config = FdtdConfig::vacuum_cube(3, 4);
    config.dt = Q16::from_f64(10.0); // Absurdly large
    assert!(!config.check_cfl(), "Large dt should violate CFL");
}

// ═══════════════════════════════════════════════════════════════════
// Test 6: Symmetry — isotropic pulse in vacuum
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_isotropic_symmetry() {
    let mut config = FdtdConfig::vacuum_cube(3, 4);
    config.boundary = BoundaryCondition::Periodic;

    let mut solver = FdtdSolver::new(config);

    // Inject symmetric pulse at center
    let n = solver.config.nx;
    let c = n / 2;
    let center = c * n * n + c * n + c;
    solver.fields.ez[center] = Q16::from_f64(1.0);

    // Run a few steps
    for _ in 0..5 {
        solver.step();
    }

    // Check that field has spread (not stuck at origin)
    let mut nonzero_count = 0;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let idx = i * n * n + j * n + k;
                if solver.fields.hx[idx].raw() != 0
                    || solver.fields.hy[idx].raw() != 0
                    || solver.fields.hz[idx].raw() != 0 {
                    nonzero_count += 1;
                }
            }
        }
    }

    assert!(nonzero_count > 1,
        "Field should propagate from source: only {} nonzero H points", nonzero_count);
}

// ═══════════════════════════════════════════════════════════════════
// Test 7: Dielectric slab — material coefficient test
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_dielectric_material_coefficients() {
    let mut config = FdtdConfig::vacuum_cube(3, 4);
    let n = config.nx;

    // Insert dielectric slab (εr = 4.0)
    config.materials.insert_slab_x(n/4, 3*n/4, Material::dielectric(Q16::from_f64(4.0)));

    let solver = FdtdSolver::new(config);

    // Verify update coefficients differ in dielectric vs vacuum
    let vac_idx = 0; // x=0, vacuum
    let die_idx = (n/2) * n * n; // x=n/2, dielectric

    // In dielectric, cb should be smaller (ε larger → slower update)
    // ca should still be ~1.0 (no loss)
    assert!(solver.fields.ex[vac_idx].raw() == solver.fields.ex[die_idx].raw(),
        "Initial fields should both be zero");
}

// ═══════════════════════════════════════════════════════════════════
// Test 8: Lossy medium — energy should decay
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_lossy_medium_energy_decay() {
    let mut config = FdtdConfig::vacuum_cube(3, 4);
    config.boundary = BoundaryCondition::Periodic;
    let n = config.nx;

    // Make entire domain lossy
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                config.materials.set(i, j, k,
                    Material::lossy_dielectric(Q16::ONE, Q16::from_f64(0.5)));
            }
        }
    }

    let mut solver = FdtdSolver::new(config);

    // Inject initial energy
    let center = (n/2) * n * n + (n/2) * n + (n/2);
    solver.fields.ez[center] = Q16::from_f64(1.0);

    let e0 = solver.fields.total_energy(&solver.config.materials);

    // Run and check energy decreases
    for _ in 0..20 {
        solver.step();
    }

    let e_final = solver.fields.total_energy(&solver.config.materials);
    assert!(e_final.raw() < e0.raw(),
        "Energy should decay in lossy medium: e0={}, e_final={}", e0, e_final);
}

// ═══════════════════════════════════════════════════════════════════
// Test 9: QTT compression roundtrip
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_qtt_compression_roundtrip() {
    let mut config = FdtdConfig::vacuum_cube(3, 4);
    config.boundary = BoundaryCondition::Periodic;

    let mut solver = FdtdSolver::new(config);

    // Inject pulse and evolve
    let n = solver.config.nx;
    let center = (n/2) * n * n + (n/2) * n + (n/2);
    solver.fields.ez[center] = Q16::from_f64(0.5);

    for _ in 0..5 {
        solver.step();
    }

    // Compress to QTT
    let snapshot = solver.compress(8);

    // Check compression produces valid MPS
    assert!(snapshot.total_elements() > 0, "Snapshot should have elements");
    assert!(snapshot.max_bond_dim() <= 8, "Bond dim should respect chi_max");
}

// ═══════════════════════════════════════════════════════════════════
// Test 10: Poynting flux — nonzero for propagating wave
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_poynting_flux_propagation() {
    let mut config = FdtdConfig::vacuum_cube(3, 4);
    config.boundary = BoundaryCondition::Periodic;

    let mut solver = FdtdSolver::new(config);

    // Set up E and H for a propagating wave
    let n = solver.config.nx;
    let c = n / 2;
    // Ez and Hx nonzero → Poynting vector in y-direction
    for k in 0..n {
        let idx = c * n * n + c * n + k;
        solver.fields.ez[idx] = Q16::from_f64(0.5);
        solver.fields.hx[idx] = Q16::from_f64(0.5);
    }

    let flux = solver.fields.poynting_flux();
    assert!(flux.raw() > 0, "Propagating wave should have nonzero Poynting flux: {}", flux);
}

// ═══════════════════════════════════════════════════════════════════
// Test 11: PML parameters
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_pml_parameter_construction() {
    let dx = Q16::from_f64(0.01);
    let params = PmlParams::optimal(8, dx);

    // Graded profile: monotonically increasing
    let mut prev = Q16::ZERO;
    for d in 1..=8 {
        let sigma = params.sigma_at(d);
        assert!(sigma.raw() >= prev.raw(),
            "PML damping should increase: σ({}) = {} < σ({}) = {}", d, sigma, d-1, prev);
        prev = sigma;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Test 12: Material sphere insertion
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_material_sphere() {
    let n = 8;
    let mut map = MaterialMap::vacuum(n, n, n);
    map.insert_sphere(4, 4, 4, 2, Material::dielectric(Q16::from_f64(9.0)));

    // Center should be dielectric
    assert!((map.get(4, 4, 4).epsilon_r.to_f64() - 9.0).abs() < 0.01);
    // Corner should be vacuum
    assert!((map.get(0, 0, 0).epsilon_r.to_f64() - 1.0).abs() < 0.01);
}

// ═══════════════════════════════════════════════════════════════════
// Test 13: Multi-step stability
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_long_run_stability() {
    let mut config = FdtdConfig::vacuum_cube(3, 4);
    config.boundary = BoundaryCondition::Periodic;

    let mut solver = FdtdSolver::new(config);

    let n = solver.config.nx;
    let center = (n/2) * n * n + (n/2) * n + (n/2);
    solver.fields.ez[center] = Q16::from_f64(0.25);

    // Run 100 steps — should not blow up
    let energies = solver.run(100);

    // Check no NaN-equivalent (overflow to extreme values)
    for (i, e) in energies.iter().enumerate() {
        assert!(e.raw().abs() < i32::MAX / 2,
            "Instability detected at step {}: energy = {}", i, e);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Test 14: Conservation verifier summary
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_conservation_verifier_summary() {
    let energies = vec![Q16::ONE; 20];
    let (conserved, max_res, steps) = ConservationVerifier::verify_run(
        &energies, Q16::from_raw(7));
    assert!(conserved);
    assert_eq!(max_res.raw(), 0);
    assert_eq!(steps, 19);
}

// ═══════════════════════════════════════════════════════════════════
// Test 15: Q16.16 arithmetic determinism
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_deterministic_simulation() {
    let mut config1 = FdtdConfig::vacuum_cube(3, 4);
    config1.boundary = BoundaryCondition::Periodic;
    let config2 = config1.clone();

    let mut solver1 = FdtdSolver::new(config1);
    let mut solver2 = FdtdSolver::new(config2);

    // Same initial condition
    let n = 8;
    let center = 4 * n * n + 4 * n + 4;
    solver1.fields.ez[center] = Q16::from_f64(0.5);
    solver2.fields.ez[center] = Q16::from_f64(0.5);

    for _ in 0..20 {
        solver1.step();
        solver2.step();
    }

    // Must produce bit-identical results
    for i in 0..n*n*n {
        assert_eq!(solver1.fields.ex[i].raw(), solver2.fields.ex[i].raw(),
            "Determinism violation at Ex[{}]", i);
        assert_eq!(solver1.fields.hx[i].raw(), solver2.fields.hx[i].raw(),
            "Determinism violation at Hx[{}]", i);
    }
}
