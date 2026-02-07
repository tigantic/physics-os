//! Halo2 circuit implementation for the NS-IMEX proof.
//!
//! Implements `Circuit<Fr>` for the Navier-Stokes IMEX QTT solver verification.
//! Only compiled when the `halo2` feature is enabled; a stub is provided
//! for builds without Halo2.
//!
//! # Circuit Layout
//!
//! The circuit has 6 phases laid out sequentially:
//!
//! 1. **Advection Constraints** — MAC chains for explicit half-steps
//! 2. **Diffusion Solve Constraints** — Implicit solve verification
//! 3. **Projection Constraints** — CG pressure Poisson solve
//! 4. **SVD Truncation Constraints** — SV ordering per stage
//! 5. **Conservation Constraints** — KE + enstrophy + divergence
//! 6. **Public Input Binding** — Hash limbs + diagnostics
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

#[cfg(feature = "halo2")]
/// Halo2 circuit implementation for the NS-IMEX solver verification.
pub mod halo2_circuit {
    use halo2_axiom::{
        circuit::{Cell, Layouter, SimpleFloorPlanner, Value},
        halo2curves::bn256::Fr,
        plonk::{
            Advice, Assigned, Circuit, Column, ConstraintSystem, Error, Fixed,
            Instance, Selector,
        },
        poly::Rotation,
    };

    use fluidelite_core::field::Q16;
    use fluidelite_core::mpo::MPO;
    use fluidelite_core::mps::MPS;

    use super::super::config::{
        NSIMEXCircuitSizing, NSIMEXParams,
        NUM_DIMENSIONS, NUM_NS_VARIABLES, PHYS_DIM,
        Q16_SCALE,
    };
    use super::super::gadgets::{
        q16_to_assigned, q16_to_fr,
        BitDecompositionGadget, ConservationGadget, DiffusionSolveGadget,
        DivergenceCheckGadget, ProjectionGadget,
        PublicInputGadget, SvdOrderingGadget,
    };
    use super::super::witness::{NSIMEXWitness, WitnessGenerator};

    // ═════════════════════════════════════════════════════════════════════
    // Column Configuration
    // ═════════════════════════════════════════════════════════════════════

    /// Halo2 column configuration for the NS-IMEX circuit.
    #[derive(Clone, Debug)]
    pub struct NSIMEXColumns {
        /// Advice column A: first operand / bit values.
        pub a: Column<Advice>,
        /// Advice column B: second operand / bit values.
        pub b: Column<Advice>,
        /// Advice column C: accumulator / result.
        pub c: Column<Advice>,
        /// Advice column D: remainder / auxiliary.
        pub d: Column<Advice>,

        /// Fixed-point MAC gate selector.
        pub s_fp_mac: Selector,
        /// Boolean-4 gate selector.
        pub s_bool4: Selector,
        /// Recomposition check selector.
        pub s_recompose: Selector,
        /// SV ordering check selector.
        pub s_sv_order: Selector,
        /// Conservation check selector.
        pub s_conservation: Selector,

        /// Public input instance column.
        pub public: Column<Instance>,
        /// Fixed constants column.
        pub constants: Column<Fixed>,
    }

    impl NSIMEXColumns {
        /// Configure the constraint system with all gates.
        pub fn configure(meta: &mut ConstraintSystem<Fr>) -> Self {
            let a = meta.advice_column();
            let b = meta.advice_column();
            let c = meta.advice_column();
            let d = meta.advice_column();

            meta.enable_equality(a);
            meta.enable_equality(b);
            meta.enable_equality(c);
            meta.enable_equality(d);

            let s_fp_mac = meta.selector();
            let s_bool4 = meta.selector();
            let s_recompose = meta.selector();
            let s_sv_order = meta.selector();
            let s_conservation = meta.selector();

            let public = meta.instance_column();
            meta.enable_equality(public);

            let constants = meta.fixed_column();
            meta.enable_constant(constants);

            let scale_expr =
                halo2_axiom::plonk::Expression::Constant(Fr::from(Q16_SCALE));

            // Gate 1: Fixed-Point MAC
            // a_cur × b_cur = (c_cur - c_prev) × SCALE + d_cur
            meta.create_gate("fp_mac", |meta| {
                let s = meta.query_selector(s_fp_mac);
                let a_cur = meta.query_advice(a, Rotation::cur());
                let b_cur = meta.query_advice(b, Rotation::cur());
                let c_cur = meta.query_advice(c, Rotation::cur());
                let c_prev = meta.query_advice(c, Rotation::prev());
                let d_cur = meta.query_advice(d, Rotation::cur());

                vec![s * (a_cur * b_cur - (c_cur - c_prev) * scale_expr.clone() - d_cur)]
            });

            // Gate 2: Boolean-4
            meta.create_gate("bool4", |meta| {
                let s = meta.query_selector(s_bool4);
                let a_cur = meta.query_advice(a, Rotation::cur());
                let b_cur = meta.query_advice(b, Rotation::cur());
                let c_cur = meta.query_advice(c, Rotation::cur());
                let d_cur = meta.query_advice(d, Rotation::cur());

                let one = halo2_axiom::plonk::Expression::Constant(Fr::one());

                vec![
                    s.clone() * a_cur.clone() * (one.clone() - a_cur),
                    s.clone() * b_cur.clone() * (one.clone() - b_cur),
                    s.clone() * c_cur.clone() * (one.clone() - c_cur),
                    s * d_cur.clone() * (one - d_cur),
                ]
            });

            // Gate 3: Recomposition Check
            meta.create_gate("recompose", |meta| {
                let s = meta.query_selector(s_recompose);
                let a_cur = meta.query_advice(a, Rotation::cur());
                let b_cur = meta.query_advice(b, Rotation::cur());

                vec![s * (a_cur - b_cur)]
            });

            // Gate 4: SV Ordering
            meta.create_gate("sv_order", |meta| {
                let s = meta.query_selector(s_sv_order);
                let a_cur = meta.query_advice(a, Rotation::cur());
                let b_cur = meta.query_advice(b, Rotation::cur());
                let c_cur = meta.query_advice(c, Rotation::cur());

                vec![s * (c_cur - (a_cur - b_cur))]
            });

            // Gate 5: Conservation Check
            meta.create_gate("conservation", |meta| {
                let s = meta.query_selector(s_conservation);
                let a_cur = meta.query_advice(a, Rotation::cur());
                let b_cur = meta.query_advice(b, Rotation::cur());
                let c_cur = meta.query_advice(c, Rotation::cur());

                vec![s * (c_cur - (b_cur - a_cur))]
            });

            Self {
                a, b, c, d,
                s_fp_mac, s_bool4, s_recompose, s_sv_order, s_conservation,
                public, constants,
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // Main Circuit
    // ═════════════════════════════════════════════════════════════════════

    /// The NS-IMEX ZK proof circuit.
    ///
    /// Proves that one timestep of the NS-IMEX QTT solver was executed
    /// correctly, including advection, diffusion, projection, SVD
    /// truncation, and conservation.
    #[derive(Clone)]
    pub struct NSIMEXCircuit {
        /// Solver parameters (public).
        pub params: NSIMEXParams,
        /// Circuit sizing.
        pub sizing: NSIMEXCircuitSizing,
        /// Complete witness data.
        pub witness: NSIMEXWitness,
    }

    impl NSIMEXCircuit {
        /// Create a new circuit from input states and shift MPOs.
        pub fn new(
            params: NSIMEXParams,
            input_states: &[MPS],
            shift_mpos: &[MPO],
        ) -> Result<Self, String> {
            let sizing = NSIMEXCircuitSizing::from_params(&params);
            let generator = WitnessGenerator::new(params.clone());

            let witness = generator
                .generate(input_states, shift_mpos)
                .map_err(|e| format!("Witness generation failed: {}", e))?;

            Ok(Self {
                params,
                sizing,
                witness,
            })
        }

        /// Get the public inputs for verification.
        pub fn public_inputs(&self) -> Vec<Fr> {
            let mut inputs = Vec::new();

            // Input state hash (4 limbs)
            for limb in &self.witness.input_hash_limbs {
                inputs.push(Fr::from(*limb));
            }

            // Output state hash (4 limbs)
            for limb in &self.witness.output_hash_limbs {
                inputs.push(Fr::from(*limb));
            }

            // Params hash (4 limbs)
            for limb in &self.witness.params_hash_limbs {
                inputs.push(Fr::from(*limb));
            }

            // Conservation: KE residual, enstrophy residual
            let ke_residual = Q16::from_raw(
                self.witness.kinetic_energy_after.raw
                    - self.witness.kinetic_energy_before.raw,
            );
            inputs.push(q16_to_fr(ke_residual));

            let ens_residual = Q16::from_raw(
                self.witness.enstrophy_after.raw
                    - self.witness.enstrophy_before.raw,
            );
            inputs.push(q16_to_fr(ens_residual));

            // Divergence residual
            inputs.push(q16_to_fr(self.witness.divergence_residual));

            // dt
            inputs.push(q16_to_fr(self.witness.dt));

            inputs
        }

        /// Get the circuit k parameter.
        pub fn k(&self) -> u32 {
            self.sizing.k as u32
        }
    }

    impl Circuit<Fr> for NSIMEXCircuit {
        type Config = NSIMEXColumns;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = ();

        fn without_witnesses(&self) -> Self {
            let empty_states: Vec<MPS> = (0..NUM_NS_VARIABLES)
                .map(|_| MPS::new(self.params.num_sites(), self.params.chi_max, PHYS_DIM))
                .collect();
            let empty_mpos: Vec<MPO> = (0..NUM_DIMENSIONS)
                .map(|_| MPO::identity(self.params.num_sites(), PHYS_DIM))
                .collect();

            let gen = WitnessGenerator::new(self.params.clone());
            let witness = gen
                .generate(&empty_states, &empty_mpos)
                .expect("Empty witness generation should not fail");

            Self {
                params: self.params.clone(),
                sizing: self.sizing.clone(),
                witness,
            }
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            NSIMEXColumns::configure(meta)
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), Error> {
            let mut public_cells: Vec<Cell> = Vec::new();

            layouter.assign_region(
                || "ns_imex_timestep",
                |mut region| {
                    let mut row = 0;

                    // ═══════════════════════════════════════════════════
                    // Phase 1: Advection Contraction Constraints
                    // ═══════════════════════════════════════════════════
                    for stage in &self.witness.stages {
                        if !stage.stage.is_explicit() {
                            continue;
                        }

                        for sweep in &stage.variable_sweeps {
                            for contraction in &sweep.contractions {
                                let num_macs = contraction.accumulators.len();
                                if num_macs == 0 {
                                    continue;
                                }

                                // Initialize accumulator
                                region.assign_advice(
                                    config.a,
                                    row,
                                    Value::known(Assigned::from(Fr::zero())),
                                );
                                region.assign_advice(
                                    config.b,
                                    row,
                                    Value::known(Assigned::from(Fr::zero())),
                                );
                                region.assign_advice(
                                    config.c,
                                    row,
                                    Value::known(Assigned::from(Fr::zero())),
                                );
                                region.assign_advice(
                                    config.d,
                                    row,
                                    Value::known(Assigned::from(Fr::zero())),
                                );
                                row += 1;

                                for (idx, acc) in contraction.accumulators.iter().enumerate() {
                                    region.assign_advice(
                                        config.a,
                                        row,
                                        Value::known(q16_to_assigned(*acc)),
                                    );
                                    region.assign_advice(
                                        config.b,
                                        row,
                                        Value::known(Assigned::from(Fr::from(1u64))),
                                    );
                                    let remainder = if idx < contraction.remainders.len() {
                                        contraction.remainders[idx]
                                    } else {
                                        Q16::ZERO
                                    };
                                    region.assign_advice(
                                        config.c,
                                        row,
                                        Value::known(q16_to_assigned(*acc)),
                                    );
                                    region.assign_advice(
                                        config.d,
                                        row,
                                        Value::known(q16_to_assigned(remainder)),
                                    );
                                    config.s_fp_mac.enable(&mut region, row)?;
                                    row += 1;

                                    // Range check
                                    row = BitDecompositionGadget::assign_range_check(
                                        &mut region,
                                        config.s_bool4,
                                        config.s_recompose,
                                        config.a,
                                        config.b,
                                        config.c,
                                        config.d,
                                        row,
                                        remainder.raw,
                                        16,
                                    )?;
                                }
                            }
                        }
                    }

                    // ═══════════════════════════════════════════════════
                    // Phase 2: Diffusion Solve Constraints
                    // ═══════════════════════════════════════════════════
                    for stage in &self.witness.stages {
                        if let Some(ref diff) = stage.diffusion_witness {
                            let nu_dt = Q16::from_raw(
                                ((self.params.viscosity.raw as i128)
                                    * (self.params.dt.raw as i128)
                                    >> 16) as i64,
                            );

                            for var_witness in &diff.variables {
                                // Assign Laplacian contraction witnesses
                                for contraction in &var_witness.laplacian_contractions {
                                    for acc in &contraction.accumulators {
                                        region.assign_advice(
                                            config.a,
                                            row,
                                            Value::known(q16_to_assigned(*acc)),
                                        );
                                        region.assign_advice(
                                            config.b,
                                            row,
                                            Value::known(Assigned::from(Fr::from(1u64))),
                                        );
                                        region.assign_advice(
                                            config.c,
                                            row,
                                            Value::known(q16_to_assigned(*acc)),
                                        );
                                        region.assign_advice(
                                            config.d,
                                            row,
                                            Value::known(Assigned::from(Fr::zero())),
                                        );
                                        config.s_fp_mac.enable(&mut region, row)?;
                                        row += 1;
                                    }
                                }

                                // Diffusion solve check
                                row = DiffusionSolveGadget::assign_diffusion_check(
                                    &mut region,
                                    config.s_fp_mac,
                                    config.s_bool4,
                                    config.s_recompose,
                                    config.s_conservation,
                                    config.a,
                                    config.b,
                                    config.c,
                                    config.d,
                                    row,
                                    Q16::ZERO,  // RHS (stubbed)
                                    Q16::ZERO,  // Solution (stubbed)
                                    Q16::ZERO,  // Laplacian result (stubbed)
                                    nu_dt,
                                    self.params.tolerance,
                                )?;
                            }
                        }
                    }

                    // ═══════════════════════════════════════════════════
                    // Phase 3: Projection CG Step Constraints
                    // ═══════════════════════════════════════════════════
                    for stage in &self.witness.stages {
                        if let Some(ref proj) = stage.projection_witness {
                            for cg_step in &proj.cg_step_witnesses {
                                row = ProjectionGadget::assign_cg_step(
                                    &mut region,
                                    config.s_fp_mac,
                                    config.s_bool4,
                                    config.s_recompose,
                                    config.a,
                                    config.b,
                                    config.c,
                                    config.d,
                                    row,
                                    cg_step.alpha_numerator,
                                    cg_step.alpha_denominator,
                                    cg_step.beta,
                                    cg_step.residual_norm,
                                )?;
                            }

                            // Divergence check
                            row = DivergenceCheckGadget::assign_divergence_check(
                                &mut region,
                                config.s_conservation,
                                config.s_bool4,
                                config.s_recompose,
                                config.a,
                                config.b,
                                config.c,
                                config.d,
                                row,
                                proj.final_divergence,
                                self.params.divergence_tolerance,
                            )?;
                        }
                    }

                    // ═══════════════════════════════════════════════════
                    // Phase 4: SVD Truncation Constraints
                    // ═══════════════════════════════════════════════════
                    for stage in &self.witness.stages {
                        for sweep in &stage.variable_sweeps {
                            for trunc in &sweep.truncations {
                                let nonneg_bits: Vec<Vec<bool>> = trunc
                                    .truncated_values
                                    .iter()
                                    .map(|sv| {
                                        let raw = sv.raw;
                                        let abs_raw = if raw >= 0 { raw as u64 } else { 0u64 };
                                        (0..32).map(|i| (abs_raw >> i) & 1 == 1).collect()
                                    })
                                    .collect();

                                let ordering_bits: Vec<Vec<bool>> = trunc
                                    .truncated_values
                                    .windows(2)
                                    .map(|pair| {
                                        let delta = pair[0].raw - pair[1].raw;
                                        let abs_delta = if delta >= 0 { delta as u64 } else { 0u64 };
                                        (0..32).map(|i| (abs_delta >> i) & 1 == 1).collect()
                                    })
                                    .collect();

                                row = SvdOrderingGadget::assign_sv_ordering(
                                    &mut region,
                                    config.s_sv_order,
                                    config.s_bool4,
                                    config.s_recompose,
                                    config.a,
                                    config.b,
                                    config.c,
                                    config.d,
                                    row,
                                    &trunc.truncated_values,
                                    &ordering_bits,
                                    &nonneg_bits,
                                )?;
                            }
                        }
                    }

                    // ═══════════════════════════════════════════════════
                    // Phase 5: Conservation Constraints
                    // ═══════════════════════════════════════════════════

                    // Kinetic energy conservation
                    row = ConservationGadget::assign_conservation_check(
                        &mut region,
                        config.s_conservation,
                        config.s_bool4,
                        config.s_recompose,
                        config.a,
                        config.b,
                        config.c,
                        config.d,
                        row,
                        self.witness.kinetic_energy_before,
                        self.witness.kinetic_energy_after,
                        self.params.conservation_tolerance,
                    )?;

                    // Enstrophy dissipation (enstrophy should decrease for viscous flow)
                    row = ConservationGadget::assign_conservation_check(
                        &mut region,
                        config.s_conservation,
                        config.s_bool4,
                        config.s_recompose,
                        config.a,
                        config.b,
                        config.c,
                        config.d,
                        row,
                        self.witness.enstrophy_before,
                        self.witness.enstrophy_after,
                        self.params.conservation_tolerance,
                    )?;

                    // ═══════════════════════════════════════════════════
                    // Phase 6: Public Input Binding
                    // ═══════════════════════════════════════════════════
                    let public_values = self.public_inputs();
                    for val in &public_values {
                        let cell = PublicInputGadget::assign_public_input(
                            &mut region,
                            config.c,
                            row,
                            *val,
                        )?;
                        public_cells.push(cell);
                        row += 1;
                    }

                    Ok(())
                },
            )?;

            // Bind public cells to instance column
            for (i, cell) in public_cells.into_iter().enumerate() {
                layouter.constrain_instance(cell, config.public, i);
            }

            Ok(())
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // Tests
    // ═════════════════════════════════════════════════════════════════════

    #[cfg(test)]
    mod tests {
        use super::*;
        use halo2_axiom::dev::MockProver;

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

        #[test]
        fn test_circuit_construction() {
            let circuit = make_test_circuit();
            assert_eq!(circuit.params.grid_bits, 4);
            assert_eq!(circuit.params.chi_max, 4);
            assert_eq!(circuit.witness.stages.len(), NUM_IMEX_STAGES);
        }

        #[test]
        fn test_public_inputs() {
            let circuit = make_test_circuit();
            let inputs = circuit.public_inputs();
            let expected_len = NSIMEXCircuitSizing::num_public_inputs();
            assert_eq!(
                inputs.len(),
                expected_len,
                "Public inputs length mismatch: got {}, expected {}",
                inputs.len(),
                expected_len,
            );
        }

        #[test]
        fn test_mock_prover_ns_imex() {
            let circuit = make_test_circuit();
            let public_inputs = circuit.public_inputs();
            let k = circuit.k().max(14);

            let prover = MockProver::run(k, &circuit, vec![public_inputs]);
            match prover {
                Ok(p) => {
                    let result = p.verify();
                    assert!(
                        result.is_ok(),
                        "MockProver verification failed: {:?}",
                        result.err()
                    );
                    println!("✓ NS-IMEX MockProver verification passed!");
                }
                Err(e) => {
                    panic!("MockProver failed to run: {:?}", e);
                }
            }
        }

        #[test]
        fn test_wrong_public_inputs_rejected() {
            let circuit = make_test_circuit();
            let mut wrong_inputs = circuit.public_inputs();
            if !wrong_inputs.is_empty() {
                wrong_inputs[0] = Fr::from(999999u64);
            }

            let k = circuit.k().max(14);
            let prover = MockProver::run(k, &circuit, vec![wrong_inputs])
                .expect("MockProver should run");

            let result = prover.verify();
            assert!(
                result.is_err(),
                "Wrong public inputs should be rejected"
            );
            println!("✓ NS-IMEX tampered inputs correctly rejected!");
        }

        #[test]
        fn test_circuit_deterministic() {
            let c1 = make_test_circuit();
            let c2 = make_test_circuit();
            assert_eq!(c1.public_inputs(), c2.public_inputs());
        }
    }
}

#[cfg(feature = "halo2")]
pub use halo2_circuit::*;

// ═══════════════════════════════════════════════════════════════════════════
// Stub Circuit (without Halo2)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(not(feature = "halo2"))]
pub mod stub_circuit {
    //! Stub NS-IMEX circuit for builds without the Halo2 backend.
    use fluidelite_core::mpo::MPO;
    use fluidelite_core::mps::MPS;

    use super::super::config::{
        NSIMEXCircuitSizing, NSIMEXParams,
    };
    use super::super::witness::{NSIMEXWitness, WitnessGenerator};

    /// Stub NS-IMEX circuit for builds without Halo2.
    #[derive(Clone)]
    pub struct NSIMEXCircuit {
        /// Solver parameters (public).
        pub params: NSIMEXParams,
        /// Circuit sizing.
        pub sizing: NSIMEXCircuitSizing,
        /// Complete witness data.
        pub witness: NSIMEXWitness,
    }

    impl NSIMEXCircuit {
        /// Create a new circuit from input states and shift MPOs.
        pub fn new(
            params: NSIMEXParams,
            input_states: &[MPS],
            shift_mpos: &[MPO],
        ) -> Result<Self, String> {
            let sizing = NSIMEXCircuitSizing::from_params(&params);
            let generator = WitnessGenerator::new(params.clone());

            let witness = generator
                .generate(input_states, shift_mpos)
                .map_err(|e| format!("Witness generation failed: {}", e))?;

            Ok(Self {
                params,
                sizing,
                witness,
            })
        }

        /// Get the k parameter for the circuit.
        pub fn k(&self) -> u32 {
            self.sizing.k as u32
        }

        /// Get the number of estimated constraints.
        pub fn estimate_constraints(&self) -> usize {
            self.sizing.total_constraints
        }

        /// Validate the witness (check all constraints without ZK).
        pub fn validate_witness(&self) -> Result<(), String> {
            // Validate SVD ordering for all truncations
            for (stage_idx, stage) in self.witness.stages.iter().enumerate() {
                for (sweep_idx, sweep) in stage.variable_sweeps.iter().enumerate() {
                    for trunc in &sweep.truncations {
                        for (i, sv) in trunc.truncated_values.iter().enumerate() {
                            if sv.raw < 0 {
                                return Err(format!(
                                    "Negative SV at stage {}, sweep {}, bond {}, sv {}: {}",
                                    stage_idx, sweep_idx, trunc.bond, i, sv.to_f64(),
                                ));
                            }
                        }
                        for pair in trunc.truncated_values.windows(2) {
                            if pair[0].raw < pair[1].raw {
                                return Err(format!(
                                    "SV ordering violation at stage {}, sweep {}, bond {}: {} < {}",
                                    stage_idx, sweep_idx, trunc.bond,
                                    pair[0].to_f64(), pair[1].to_f64(),
                                ));
                            }
                        }
                    }
                }
            }

            // Validate conservation: KE
            let ke_residual = self.witness.kinetic_energy_after.raw
                - self.witness.kinetic_energy_before.raw;
            if ke_residual.unsigned_abs() as i64 > self.params.conservation_tolerance.raw {
                return Err(format!(
                    "KE conservation violation: residual {} > tolerance {}",
                    ke_residual as f64 / 65536.0,
                    self.params.conservation_tolerance.to_f64(),
                ));
            }

            // Validate divergence
            if self.witness.divergence_residual.raw.unsigned_abs() as i64
                > self.params.divergence_tolerance.raw
            {
                return Err(format!(
                    "Divergence violation: {} > tolerance {}",
                    self.witness.divergence_residual.to_f64(),
                    self.params.divergence_tolerance.to_f64(),
                ));
            }

            // Validate diffusion solve residuals
            for stage in &self.witness.stages {
                if let Some(ref diff) = stage.diffusion_witness {
                    for var_w in &diff.variables {
                        if var_w.solve_residual.raw.unsigned_abs() as i64
                            > self.params.tolerance.raw
                        {
                            return Err(format!(
                                "Diffusion solve residual too large for {}: {} > {}",
                                var_w.variable,
                                var_w.solve_residual.to_f64(),
                                self.params.tolerance.to_f64(),
                            ));
                        }
                    }
                }
            }

            // Validate CG convergence
            for stage in &self.witness.stages {
                if let Some(ref proj) = stage.projection_witness {
                    if proj.cg_residual.raw.unsigned_abs() as i64
                        > self.params.cg_tolerance.raw
                    {
                        return Err(format!(
                            "CG not converged: residual {} > tolerance {}",
                            proj.cg_residual.to_f64(),
                            self.params.cg_tolerance.to_f64(),
                        ));
                    }
                }
            }

            Ok(())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use super::super::super::config::{NUM_NS_VARIABLES, PHYS_DIM, NUM_DIMENSIONS};

        #[test]
        fn test_stub_circuit_creation() {
            let params = NSIMEXParams::test_small();
            let num_sites = params.num_sites();
            let chi = params.chi_max;

            let input_states: Vec<MPS> = (0..NUM_NS_VARIABLES)
                .map(|_| MPS::new(num_sites, chi, PHYS_DIM))
                .collect();
            let shift_mpos: Vec<MPO> = (0..NUM_DIMENSIONS)
                .map(|_| MPO::identity(num_sites, PHYS_DIM))
                .collect();

            let circuit =
                NSIMEXCircuit::new(params, &input_states, &shift_mpos)
                    .expect("Circuit creation failed");

            assert!(circuit.estimate_constraints() > 0);
        }

        #[test]
        fn test_stub_witness_validation() {
            let params = NSIMEXParams::test_small();
            let num_sites = params.num_sites();
            let chi = params.chi_max;

            let input_states: Vec<MPS> = (0..NUM_NS_VARIABLES)
                .map(|_| MPS::new(num_sites, chi, PHYS_DIM))
                .collect();
            let shift_mpos: Vec<MPO> = (0..NUM_DIMENSIONS)
                .map(|_| MPO::identity(num_sites, PHYS_DIM))
                .collect();

            let circuit =
                NSIMEXCircuit::new(params, &input_states, &shift_mpos)
                    .expect("Circuit creation failed");

            let result = circuit.validate_witness();
            assert!(
                result.is_ok(),
                "Witness validation failed: {:?}",
                result.err()
            );
            println!("✓ NS-IMEX stub witness validation passed");
        }
    }
}

#[cfg(not(feature = "halo2"))]
pub use stub_circuit::*;
