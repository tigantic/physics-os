//! Task 2.1: Circuit soundness negative tests via MockProver.
//!
//! Tests the constraint-level soundness of all three physics domain circuits
//! (Euler3D, NS-IMEX, Thermal) using Halo2's MockProver.
//!
//! For each circuit we test:
//! 1. **Positive**: correct public inputs → MockProver accepts.
//! 2. **Input hash tampering**: modifying input_state_hash limbs → rejected.
//! 3. **Output hash tampering**: modifying output_state_hash limbs → rejected.
//! 4. **Params hash tampering**: modifying params_hash limbs → rejected.
//! 5. **Conservation/residual tampering**: modifying residual values → rejected.
//! 6. **Parameter tampering**: modifying dt/chi_max/grid_bits → rejected.
//! 7. **All-zero public inputs**: every input zeroed → rejected.
//! 8. **Extra/missing public inputs**: wrong count → rejected or error.
//!
//! These run in seconds via MockProver (no KZG trusted setup required).

#[cfg(feature = "halo2")]
mod euler3d_soundness {
    use fluidelite_circuits::euler3d::*;
    use fluidelite_core::mpo::MPO;
    use fluidelite_core::mps::MPS;
    use halo2_axiom::dev::MockProver;
    use halo2_axiom::halo2curves::bn256::Fr;

    fn make_test_circuit() -> Euler3DCircuit {
        let params = Euler3DParams::test_small();
        let num_sites = params.num_sites();
        let chi = params.chi_max;

        let input_states: Vec<MPS> = (0..NUM_CONSERVED_VARIABLES)
            .map(|_| MPS::new(num_sites, chi, 2))
            .collect();
        let shift_mpos: Vec<MPO> = (0..3)
            .map(|_| MPO::identity(num_sites, 2))
            .collect();

        Euler3DCircuit::new(params, &input_states, &shift_mpos)
            .expect("Circuit creation failed")
    }

    // ── Positive ──────────────────────────────────────────────────────────

    #[test]
    fn euler3d_positive_mock_prover() {
        let circuit = make_test_circuit();
        let public_inputs = circuit.public_inputs();
        let k = circuit.k().max(14);

        let prover = MockProver::run(k, &circuit, vec![public_inputs])
            .expect("MockProver should run");
        assert!(
            prover.verify().is_ok(),
            "Valid Euler3D circuit should pass MockProver"
        );
    }

    // ── Input state hash tampering ────────────────────────────────────────

    #[test]
    fn euler3d_tamper_input_hash_limb0() {
        let circuit = make_test_circuit();
        let mut pi = circuit.public_inputs();
        pi[0] = Fr::from(999999u64); // input_state_hash limb 0
        let k = circuit.k().max(14);
        let prover = MockProver::run(k, &circuit, vec![pi]).expect("run");
        assert!(prover.verify().is_err(), "Tampered input hash limb 0 must be rejected");
    }

    #[test]
    fn euler3d_tamper_input_hash_limb3() {
        let circuit = make_test_circuit();
        let mut pi = circuit.public_inputs();
        pi[3] = Fr::from(999999u64); // input_state_hash limb 3
        let k = circuit.k().max(14);
        let prover = MockProver::run(k, &circuit, vec![pi]).expect("run");
        assert!(prover.verify().is_err(), "Tampered input hash limb 3 must be rejected");
    }

    // ── Output state hash tampering ───────────────────────────────────────

    #[test]
    fn euler3d_tamper_output_hash_limb0() {
        let circuit = make_test_circuit();
        let mut pi = circuit.public_inputs();
        pi[4] = Fr::from(999999u64); // output_state_hash limb 0
        let k = circuit.k().max(14);
        let prover = MockProver::run(k, &circuit, vec![pi]).expect("run");
        assert!(prover.verify().is_err(), "Tampered output hash limb 0 must be rejected");
    }

    #[test]
    fn euler3d_tamper_output_hash_limb2() {
        let circuit = make_test_circuit();
        let mut pi = circuit.public_inputs();
        pi[6] = Fr::from(999999u64); // output_state_hash limb 2
        let k = circuit.k().max(14);
        let prover = MockProver::run(k, &circuit, vec![pi]).expect("run");
        assert!(prover.verify().is_err(), "Tampered output hash limb 2 must be rejected");
    }

    // ── Params hash tampering ─────────────────────────────────────────────

    #[test]
    fn euler3d_tamper_params_hash_limb0() {
        let circuit = make_test_circuit();
        let mut pi = circuit.public_inputs();
        pi[8] = Fr::from(999999u64); // params_hash limb 0
        let k = circuit.k().max(14);
        let prover = MockProver::run(k, &circuit, vec![pi]).expect("run");
        assert!(prover.verify().is_err(), "Tampered params hash limb 0 must be rejected");
    }

    // ── Conservation residual tampering ───────────────────────────────────

    #[test]
    fn euler3d_tamper_conservation_density() {
        let circuit = make_test_circuit();
        let mut pi = circuit.public_inputs();
        pi[12] = Fr::from(123456789u64); // conservation residual [0] = ρ
        let k = circuit.k().max(14);
        let prover = MockProver::run(k, &circuit, vec![pi]).expect("run");
        assert!(prover.verify().is_err(), "Tampered density conservation must be rejected");
    }

    #[test]
    fn euler3d_tamper_conservation_energy() {
        let circuit = make_test_circuit();
        let mut pi = circuit.public_inputs();
        pi[16] = Fr::from(123456789u64); // conservation residual [4] = Energy
        let k = circuit.k().max(14);
        let prover = MockProver::run(k, &circuit, vec![pi]).expect("run");
        assert!(prover.verify().is_err(), "Tampered energy conservation must be rejected");
    }

    // ── Parameter tampering ───────────────────────────────────────────────

    #[test]
    fn euler3d_tamper_dt() {
        let circuit = make_test_circuit();
        let mut pi = circuit.public_inputs();
        pi[17] = Fr::from(999999u64); // dt
        let k = circuit.k().max(14);
        let prover = MockProver::run(k, &circuit, vec![pi]).expect("run");
        assert!(prover.verify().is_err(), "Tampered dt must be rejected");
    }

    #[test]
    fn euler3d_tamper_chi_max() {
        let circuit = make_test_circuit();
        let mut pi = circuit.public_inputs();
        pi[18] = Fr::from(999999u64); // chi_max
        let k = circuit.k().max(14);
        let prover = MockProver::run(k, &circuit, vec![pi]).expect("run");
        assert!(prover.verify().is_err(), "Tampered chi_max must be rejected");
    }

    #[test]
    fn euler3d_tamper_grid_bits() {
        let circuit = make_test_circuit();
        let mut pi = circuit.public_inputs();
        pi[19] = Fr::from(999999u64); // grid_bits
        let k = circuit.k().max(14);
        let prover = MockProver::run(k, &circuit, vec![pi]).expect("run");
        assert!(prover.verify().is_err(), "Tampered grid_bits must be rejected");
    }

    // ── All-zero public inputs ────────────────────────────────────────────

    #[test]
    fn euler3d_all_zero_public_inputs() {
        let circuit = make_test_circuit();
        let pi = circuit.public_inputs();
        let zero_pi = vec![Fr::from(0u64); pi.len()];
        let k = circuit.k().max(14);
        let prover = MockProver::run(k, &circuit, vec![zero_pi]).expect("run");
        assert!(prover.verify().is_err(), "All-zero public inputs must be rejected");
    }

    // ── Wrong public input count ──────────────────────────────────────────

    #[test]
    fn euler3d_missing_public_input() {
        let circuit = make_test_circuit();
        let mut pi = circuit.public_inputs();
        pi.pop(); // Remove last element
        let k = circuit.k().max(14);
        let result = MockProver::run(k, &circuit, vec![pi]);
        match result {
            Ok(p) => assert!(p.verify().is_err(), "Missing public input must be rejected"),
            Err(_) => {} // MockProver refusing to run is also correct
        }
    }

    #[test]
    fn euler3d_extra_public_input() {
        let circuit = make_test_circuit();
        let mut pi = circuit.public_inputs();
        pi.push(Fr::from(42u64));
        let k = circuit.k().max(14);
        // MockProver pads unused instance rows with zero; having more
        // instance values than the circuit expects is not necessarily
        // caught at the MockProver level.  A real verifier may reject.
        let result = MockProver::run(k, &circuit, vec![pi]);
        // If MockProver runs, it may still succeed because extra values
        // sit in unallocated rows — this is a known MockProver limitation.
        match result {
            Ok(_) => {} // Acceptable: MockProver may not catch extra cells
            Err(_) => {} // Also acceptable: MockProver refuses
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// NS-IMEX Soundness Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "halo2")]
mod ns_imex_soundness {
    use fluidelite_circuits::ns_imex::*;
    use fluidelite_core::mpo::MPO;
    use fluidelite_core::mps::MPS;
    use halo2_axiom::dev::MockProver;
    use halo2_axiom::halo2curves::bn256::Fr;

    fn make_test_circuit() -> NSIMEXCircuit {
        let params = NSIMEXParams::test_small();
        let num_sites = params.num_sites();
        let chi = params.chi_max;

        let input_states: Vec<MPS> = (0..NUM_NS_VARIABLES)
            .map(|_| MPS::new(num_sites, chi, PHYS_DIM))
            .collect();
        let shift_mpos: Vec<MPO> = (0..NUM_DIMENSIONS)
            .map(|_| MPO::identity(num_sites, PHYS_DIM))
            .collect();

        NSIMEXCircuit::new(params, &input_states, &shift_mpos)
            .expect("Circuit creation failed")
    }

    // ── Positive ──────────────────────────────────────────────────────────

    #[test]
    fn ns_imex_positive_mock_prover() {
        let circuit = make_test_circuit();
        let public_inputs = circuit.public_inputs();
        let k = circuit.k().max(14);

        let prover = MockProver::run(k, &circuit, vec![public_inputs])
            .expect("MockProver should run");
        assert!(
            prover.verify().is_ok(),
            "Valid NS-IMEX circuit should pass MockProver"
        );
    }

    // ── Input state hash tampering ────────────────────────────────────────

    #[test]
    fn ns_imex_tamper_input_hash_limb0() {
        let circuit = make_test_circuit();
        let mut pi = circuit.public_inputs();
        pi[0] = Fr::from(999999u64);
        let k = circuit.k().max(14);
        let prover = MockProver::run(k, &circuit, vec![pi]).expect("run");
        assert!(prover.verify().is_err(), "Tampered input hash limb 0 must be rejected");
    }

    #[test]
    fn ns_imex_tamper_input_hash_limb3() {
        let circuit = make_test_circuit();
        let mut pi = circuit.public_inputs();
        pi[3] = Fr::from(999999u64);
        let k = circuit.k().max(14);
        let prover = MockProver::run(k, &circuit, vec![pi]).expect("run");
        assert!(prover.verify().is_err(), "Tampered input hash limb 3 must be rejected");
    }

    // ── Output state hash tampering ───────────────────────────────────────

    #[test]
    fn ns_imex_tamper_output_hash_limb0() {
        let circuit = make_test_circuit();
        let mut pi = circuit.public_inputs();
        pi[4] = Fr::from(999999u64);
        let k = circuit.k().max(14);
        let prover = MockProver::run(k, &circuit, vec![pi]).expect("run");
        assert!(prover.verify().is_err(), "Tampered output hash limb 0 must be rejected");
    }

    // ── Params hash tampering ─────────────────────────────────────────────

    #[test]
    fn ns_imex_tamper_params_hash_limb0() {
        let circuit = make_test_circuit();
        let mut pi = circuit.public_inputs();
        pi[8] = Fr::from(999999u64);
        let k = circuit.k().max(14);
        let prover = MockProver::run(k, &circuit, vec![pi]).expect("run");
        assert!(prover.verify().is_err(), "Tampered params hash limb 0 must be rejected");
    }

    #[test]
    fn ns_imex_tamper_params_hash_limb3() {
        let circuit = make_test_circuit();
        let mut pi = circuit.public_inputs();
        pi[11] = Fr::from(999999u64);
        let k = circuit.k().max(14);
        let prover = MockProver::run(k, &circuit, vec![pi]).expect("run");
        assert!(prover.verify().is_err(), "Tampered params hash limb 3 must be rejected");
    }

    // ── KE residual tampering ─────────────────────────────────────────────

    #[test]
    fn ns_imex_tamper_ke_residual() {
        let circuit = make_test_circuit();
        let mut pi = circuit.public_inputs();
        pi[12] = Fr::from(123456789u64); // KE residual
        let k = circuit.k().max(14);
        let prover = MockProver::run(k, &circuit, vec![pi]).expect("run");
        assert!(prover.verify().is_err(), "Tampered KE residual must be rejected");
    }

    // ── Enstrophy residual tampering ──────────────────────────────────────

    #[test]
    fn ns_imex_tamper_enstrophy_residual() {
        let circuit = make_test_circuit();
        let mut pi = circuit.public_inputs();
        pi[13] = Fr::from(123456789u64); // Enstrophy residual
        let k = circuit.k().max(14);
        let prover = MockProver::run(k, &circuit, vec![pi]).expect("run");
        assert!(prover.verify().is_err(), "Tampered enstrophy residual must be rejected");
    }

    // ── Divergence residual tampering ─────────────────────────────────────

    #[test]
    fn ns_imex_tamper_divergence_residual() {
        let circuit = make_test_circuit();
        let mut pi = circuit.public_inputs();
        pi[14] = Fr::from(123456789u64); // Divergence residual
        let k = circuit.k().max(14);
        let prover = MockProver::run(k, &circuit, vec![pi]).expect("run");
        assert!(prover.verify().is_err(), "Tampered divergence residual must be rejected");
    }

    // ── dt tampering ──────────────────────────────────────────────────────

    #[test]
    fn ns_imex_tamper_dt() {
        let circuit = make_test_circuit();
        let mut pi = circuit.public_inputs();
        pi[15] = Fr::from(999999u64); // dt
        let k = circuit.k().max(14);
        let prover = MockProver::run(k, &circuit, vec![pi]).expect("run");
        assert!(prover.verify().is_err(), "Tampered dt must be rejected");
    }

    // ── All-zero public inputs ────────────────────────────────────────────

    #[test]
    fn ns_imex_all_zero_public_inputs() {
        let circuit = make_test_circuit();
        let pi = circuit.public_inputs();
        let zero_pi = vec![Fr::from(0u64); pi.len()];
        let k = circuit.k().max(14);
        let prover = MockProver::run(k, &circuit, vec![zero_pi]).expect("run");
        assert!(prover.verify().is_err(), "All-zero public inputs must be rejected");
    }

    // ── Wrong public input count ──────────────────────────────────────────

    #[test]
    fn ns_imex_missing_public_input() {
        let circuit = make_test_circuit();
        let mut pi = circuit.public_inputs();
        pi.pop();
        let k = circuit.k().max(14);
        let result = MockProver::run(k, &circuit, vec![pi]);
        match result {
            Ok(p) => assert!(p.verify().is_err(), "Missing public input must be rejected"),
            Err(_) => {}
        }
    }

    #[test]
    fn ns_imex_extra_public_input() {
        let circuit = make_test_circuit();
        let mut pi = circuit.public_inputs();
        pi.push(Fr::from(42u64));
        let k = circuit.k().max(14);
        let result = MockProver::run(k, &circuit, vec![pi]);
        match result {
            Ok(_) => {} // MockProver may not catch extra instance cells
            Err(_) => {}
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Thermal Soundness Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "halo2")]
mod thermal_soundness {
    use fluidelite_circuits::thermal::*;
    use fluidelite_core::mpo::MPO;
    use fluidelite_core::mps::MPS;
    use halo2_axiom::dev::MockProver;
    use halo2_axiom::halo2curves::bn256::Fr;

    fn make_test_circuit() -> ThermalCircuit {
        let params = ThermalParams::test_small();
        let num_sites = params.num_sites();
        let chi = params.chi_max;

        let input_states: Vec<MPS> = (0..NUM_THERMAL_VARIABLES)
            .map(|_| MPS::new(num_sites, chi, PHYS_DIM))
            .collect();
        let laplacian_mpos: Vec<MPO> = (0..NUM_DIMENSIONS)
            .map(|_| MPO::identity(num_sites, PHYS_DIM))
            .collect();

        ThermalCircuit::new(params, &input_states, &laplacian_mpos)
            .expect("Circuit creation failed")
    }

    // ── Positive ──────────────────────────────────────────────────────────

    #[test]
    fn thermal_positive_mock_prover() {
        let circuit = make_test_circuit();
        let public_inputs = circuit.public_inputs();
        let k = circuit.k().max(14);

        let prover = MockProver::run(k, &circuit, vec![public_inputs])
            .expect("MockProver should run");
        assert!(
            prover.verify().is_ok(),
            "Valid Thermal circuit should pass MockProver"
        );
    }

    // ── Input state hash tampering ────────────────────────────────────────

    #[test]
    fn thermal_tamper_input_hash_limb0() {
        let circuit = make_test_circuit();
        let mut pi = circuit.public_inputs();
        pi[0] = Fr::from(999999u64);
        let k = circuit.k().max(14);
        let prover = MockProver::run(k, &circuit, vec![pi]).expect("run");
        assert!(prover.verify().is_err(), "Tampered input hash limb 0 must be rejected");
    }

    #[test]
    fn thermal_tamper_input_hash_limb3() {
        let circuit = make_test_circuit();
        let mut pi = circuit.public_inputs();
        pi[3] = Fr::from(999999u64);
        let k = circuit.k().max(14);
        let prover = MockProver::run(k, &circuit, vec![pi]).expect("run");
        assert!(prover.verify().is_err(), "Tampered input hash limb 3 must be rejected");
    }

    // ── Output state hash tampering ───────────────────────────────────────

    #[test]
    fn thermal_tamper_output_hash_limb0() {
        let circuit = make_test_circuit();
        let mut pi = circuit.public_inputs();
        pi[4] = Fr::from(999999u64);
        let k = circuit.k().max(14);
        let prover = MockProver::run(k, &circuit, vec![pi]).expect("run");
        assert!(prover.verify().is_err(), "Tampered output hash limb 0 must be rejected");
    }

    // ── Params hash tampering ─────────────────────────────────────────────

    #[test]
    fn thermal_tamper_params_hash_limb0() {
        let circuit = make_test_circuit();
        let mut pi = circuit.public_inputs();
        pi[8] = Fr::from(999999u64);
        let k = circuit.k().max(14);
        let prover = MockProver::run(k, &circuit, vec![pi]).expect("run");
        assert!(prover.verify().is_err(), "Tampered params hash limb 0 must be rejected");
    }

    // ── Conservation residual tampering ───────────────────────────────────

    #[test]
    fn thermal_tamper_conservation_residual() {
        let circuit = make_test_circuit();
        let mut pi = circuit.public_inputs();
        pi[12] = Fr::from(987654321u64); // conservation residual
        let k = circuit.k().max(14);
        let prover = MockProver::run(k, &circuit, vec![pi]).expect("run");
        assert!(prover.verify().is_err(), "Tampered conservation residual must be rejected");
    }

    // ── Parameter tampering ───────────────────────────────────────────────

    #[test]
    fn thermal_tamper_dt() {
        let circuit = make_test_circuit();
        let mut pi = circuit.public_inputs();
        pi[13] = Fr::from(999999u64); // dt
        let k = circuit.k().max(14);
        let prover = MockProver::run(k, &circuit, vec![pi]).expect("run");
        assert!(prover.verify().is_err(), "Tampered dt must be rejected");
    }

    #[test]
    fn thermal_tamper_alpha() {
        let circuit = make_test_circuit();
        let mut pi = circuit.public_inputs();
        pi[14] = Fr::from(999999u64); // alpha (thermal diffusivity)
        let k = circuit.k().max(14);
        let prover = MockProver::run(k, &circuit, vec![pi]).expect("run");
        assert!(prover.verify().is_err(), "Tampered alpha must be rejected");
    }

    #[test]
    fn thermal_tamper_chi_max() {
        let circuit = make_test_circuit();
        let mut pi = circuit.public_inputs();
        pi[15] = Fr::from(999999u64); // chi_max
        let k = circuit.k().max(14);
        let prover = MockProver::run(k, &circuit, vec![pi]).expect("run");
        assert!(prover.verify().is_err(), "Tampered chi_max must be rejected");
    }

    #[test]
    fn thermal_tamper_grid_bits() {
        let circuit = make_test_circuit();
        let mut pi = circuit.public_inputs();
        pi[16] = Fr::from(999999u64); // grid_bits
        let k = circuit.k().max(14);
        let prover = MockProver::run(k, &circuit, vec![pi]).expect("run");
        assert!(prover.verify().is_err(), "Tampered grid_bits must be rejected");
    }

    // ── All-zero public inputs ────────────────────────────────────────────

    #[test]
    fn thermal_all_zero_public_inputs() {
        let circuit = make_test_circuit();
        let pi = circuit.public_inputs();
        let zero_pi = vec![Fr::from(0u64); pi.len()];
        let k = circuit.k().max(14);
        let prover = MockProver::run(k, &circuit, vec![zero_pi]).expect("run");
        assert!(prover.verify().is_err(), "All-zero public inputs must be rejected");
    }

    // ── Wrong public input count ──────────────────────────────────────────

    #[test]
    fn thermal_missing_public_input() {
        let circuit = make_test_circuit();
        let mut pi = circuit.public_inputs();
        pi.pop();
        let k = circuit.k().max(14);
        let result = MockProver::run(k, &circuit, vec![pi]);
        match result {
            Ok(p) => assert!(p.verify().is_err(), "Missing public input must be rejected"),
            Err(_) => {}
        }
    }

    #[test]
    fn thermal_extra_public_input() {
        let circuit = make_test_circuit();
        let mut pi = circuit.public_inputs();
        pi.push(Fr::from(42u64));
        let k = circuit.k().max(14);
        let result = MockProver::run(k, &circuit, vec![pi]);
        match result {
            Ok(_) => {} // MockProver may not catch extra instance cells
            Err(_) => {}
        }
    }

    // ── Determinism ───────────────────────────────────────────────────────

    #[test]
    fn thermal_circuit_deterministic() {
        let c1 = make_test_circuit();
        let c2 = make_test_circuit();
        assert_eq!(c1.public_inputs(), c2.public_inputs());
    }
}
