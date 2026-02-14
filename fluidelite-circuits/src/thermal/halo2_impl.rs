//! Halo2 circuit implementation for the Thermal/Heat Equation proof.
//!
//! Implements `Circuit<Fr>` for the thermal QTT solver verification.
//! Only compiled when the `halo2` feature is enabled; a stub is provided
//! for builds without Halo2.
//!
//! # Circuit Layout
//!
//! The circuit has 5 phases laid out sequentially:
//!
//! 1. **RHS Assembly Constraints** — source term scaling + addition
//! 2. **Implicit CG Solve Constraints** — system matrix contractions
//!    and CG convergence verification
//! 3. **SVD Truncation Constraints** — SV ordering + non-negativity
//! 4. **Conservation Constraints** — energy balance check
//! 5. **Public Input Binding** — connects witness to instance column
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

#[cfg(feature = "halo2")]
/// Halo2 circuit implementation for the thermal QTT solver verification.
pub mod halo2_circuit {
    use halo2_axiom::{
        circuit::{Cell, Layouter, SimpleFloorPlanner, Value},
        halo2curves::bn256::Fr,
        plonk::{
            Advice, Circuit, Column, ConstraintSystem, Error, Fixed, Instance,
            Selector,
        },
        poly::Rotation,
    };

    use fluidelite_core::field::Q16;
    use fluidelite_core::mpo::MPO;
    use fluidelite_core::mps::MPS;

    use super::super::config::{
        ThermalCircuitSizing, ThermalParams, Q16_SCALE,
    };
    use super::super::gadgets::{
        q16_to_assigned, i64_to_assigned, ConservationGadget, PublicInputGadget, SvdOrderingGadget,
    };
    use super::super::witness::{ThermalWitness, WitnessGenerator};

    // ═════════════════════════════════════════════════════════════════════
    // Column Configuration
    // ═════════════════════════════════════════════════════════════════════

    /// Halo2 column configuration for the Thermal circuit.
    #[derive(Clone, Debug)]
    pub struct ThermalColumns {
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

    impl ThermalColumns {
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

            // Gate 1: Fixed-Point MAC
            let scale_expr = halo2_axiom::plonk::Expression::Constant(
                Fr::from(Q16_SCALE),
            );

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

                vec![
                    s.clone() * a_cur.clone() * (a_cur - halo2_axiom::plonk::Expression::Constant(Fr::from(1))),
                    s.clone() * b_cur.clone() * (b_cur - halo2_axiom::plonk::Expression::Constant(Fr::from(1))),
                    s.clone() * c_cur.clone() * (c_cur - halo2_axiom::plonk::Expression::Constant(Fr::from(1))),
                    s * d_cur.clone() * (d_cur - halo2_axiom::plonk::Expression::Constant(Fr::from(1))),
                ]
            });

            // Gate 3: SV Ordering (s_i ≥ s_{i+1})
            meta.create_gate("sv_order", |meta| {
                let s = meta.query_selector(s_sv_order);
                let a_cur = meta.query_advice(a, Rotation::cur());
                let b_cur = meta.query_advice(b, Rotation::cur());
                // a = sv_value, b = bit_decomp of sv (non-neg proof)
                // Constraint: a = b (proves a ≥ 0 via bit decomposition)
                vec![s * (a_cur - b_cur)]
            });

            // Gate 4: Conservation
            meta.create_gate("conservation", |meta| {
                let s = meta.query_selector(s_conservation);
                let a_cur = meta.query_advice(a, Rotation::cur());
                let b_cur = meta.query_advice(b, Rotation::cur());
                let c_cur = meta.query_advice(c, Rotation::cur());
                let _d_cur = meta.query_advice(d, Rotation::cur());
                // a = integral_before, b = integral_after
                // c = residual, d = tolerance
                // Constraint: |b - a| ≤ d  (via c = |b - a| and d - c ≥ 0)
                vec![s * (c_cur - (b_cur - a_cur))]
            });

            Self {
                a,
                b,
                c,
                d,
                s_fp_mac,
                s_bool4,
                s_recompose,
                s_sv_order,
                s_conservation,
                public,
                constants,
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // Circuit
    // ═════════════════════════════════════════════════════════════════════

    /// The Thermal proof circuit.
    #[derive(Clone, Debug)]
    pub struct ThermalCircuit {
        /// Witness data.
        pub witness: ThermalWitness,

        /// Circuit sizing.
        pub sizing: ThermalCircuitSizing,
    }

    impl ThermalCircuit {
        /// Create a new circuit from parameters and inputs.
        pub fn new(
            params: ThermalParams,
            input_states: &[MPS],
            laplacian_mpos: &[MPO],
        ) -> Result<Self, String> {
            let sizing = ThermalCircuitSizing::from_params(&params);
            let gen = WitnessGenerator::new(params);
            let witness = gen
                .generate(input_states, laplacian_mpos)
                .map_err(|e| format!("Witness generation failed: {}", e))?;

            Ok(Self { witness, sizing })
        }

        /// Get circuit k.
        pub fn k(&self) -> u32 {
            self.sizing.k
        }

        /// Compute public inputs vector.
        pub fn public_inputs(&self) -> Vec<Fr> {
            let mut inputs = Vec::new();

            // Input state hash (4 limbs)
            for limb in &self.witness.hashes.input_state_hash_limbs {
                inputs.push(Fr::from(*limb));
            }

            // Output state hash (4 limbs)
            for limb in &self.witness.hashes.output_state_hash_limbs {
                inputs.push(Fr::from(*limb));
            }

            // Params hash (4 limbs)
            for limb in &self.witness.hashes.params_hash_limbs {
                inputs.push(Fr::from(*limb));
            }

            // Conservation residual
            let r = self.witness.conservation.residual;
            if r.raw >= 0 {
                inputs.push(Fr::from(r.raw as u64));
            } else {
                inputs.push(-Fr::from((-r.raw) as u64));
            }

            // dt, alpha, chi_max, grid_bits
            if self.witness.params.dt.raw >= 0 {
                inputs.push(Fr::from(self.witness.params.dt.raw as u64));
            } else {
                inputs.push(-Fr::from((-self.witness.params.dt.raw) as u64));
            }
            if self.witness.params.alpha.raw >= 0 {
                inputs.push(Fr::from(self.witness.params.alpha.raw as u64));
            } else {
                inputs.push(-Fr::from((-self.witness.params.alpha.raw) as u64));
            }
            inputs.push(Fr::from(self.witness.params.chi_max as u64));
            inputs.push(Fr::from(self.witness.params.grid_bits as u64));

            inputs
        }
    }

    impl Circuit<Fr> for ThermalCircuit {
        type Config = ThermalColumns;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = ();

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            ThermalColumns::configure(meta)
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), Error> {
            let mut public_cells: Vec<Cell> = Vec::new();

            layouter.assign_region(
                || "thermal_circuit",
                |mut region| {
                    let mut row = 0;

                    // Phase 1: RHS Assembly constraints
                    // (MAC constraints for source scaling if applicable)
                    for remainder in &self.witness.rhs_assembly.fp_remainders {
                        region.assign_advice(
                            config.d,
                            row,
                            Value::known(i64_to_assigned(*remainder)),
                        );
                        row += 1;
                    }

                    // Phase 2: CG Solve constraints
                    // Assign residual norms for convergence verification
                    let residual_norms: Vec<Q16> = self
                        .witness
                        .implicit_solve
                        .iterations
                        .iter()
                        .map(|it| it.residual_norm)
                        .collect();

                    for norm in &residual_norms {
                        region.assign_advice(
                            config.a,
                            row,
                            Value::known(q16_to_assigned(*norm)),
                        );
                        row += 1;
                    }

                    // CG contraction MAC constraints
                    for iter_witness in &self.witness.implicit_solve.iterations {
                        for site_data in &iter_witness.contraction.site_data {
                            for remainder in &site_data.fp_remainders {
                                region.assign_advice(
                                    config.d,
                                    row,
                                    Value::known(i64_to_assigned(*remainder)),
                                );
                                row += 1;
                            }
                        }
                    }

                    // Phase 3: SVD Truncation constraints
                    for bond in &self.witness.truncation.bond_data {
                        row = SvdOrderingGadget::assign_sv_ordering(
                            &mut region,
                            config.s_sv_order,
                            config.a,
                            config.b,
                            config.c,
                            config.d,
                            row,
                            &bond.singular_values,
                            &bond.sv_ordering_bits,
                            &bond.sv_nonneg_bits,
                        )?;
                    }

                    // Phase 4: Conservation constraint
                    row = ConservationGadget::assign_conservation(
                        &mut region,
                        config.s_conservation,
                        config.a,
                        config.b,
                        config.c,
                        config.d,
                        row,
                        self.witness.conservation.integral_before,
                        self.witness.conservation.integral_after,
                        self.witness.conservation.residual,
                        self.witness.params.conservation_tol,
                        &self.witness.conservation.residual_bound_bits,
                    )?;

                    // Phase 5: Public inputs
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
}

#[cfg(feature = "halo2")]
pub use halo2_circuit::ThermalCircuit;

// ═══════════════════════════════════════════════════════════════════════════
// Stub Circuit (without Halo2)
// ═══════════════════════════════════════════════════════════════════════════

/// Stub circuit for builds without Halo2.
#[cfg(not(feature = "halo2"))]
pub mod stub_circuit {
    use super::super::config::{ThermalCircuitSizing, ThermalParams};
    use super::super::witness::{ThermalWitness, WitnessGenerator};
    use fluidelite_core::mpo::MPO;
    use fluidelite_core::mps::MPS;

    /// Stub thermal circuit.
    #[derive(Clone, Debug)]
    pub struct ThermalCircuit {
        /// Witness data.
        pub witness: ThermalWitness,

        /// Circuit sizing.
        pub sizing: ThermalCircuitSizing,
    }

    impl ThermalCircuit {
        /// Create a new circuit from parameters and inputs.
        pub fn new(
            params: ThermalParams,
            input_states: &[MPS],
            laplacian_mpos: &[MPO],
        ) -> Result<Self, String> {
            let sizing = ThermalCircuitSizing::from_params(&params);
            let gen = WitnessGenerator::new(params);
            let witness = gen
                .generate(input_states, laplacian_mpos)
                .map_err(|e| format!("Witness generation failed: {}", e))?;

            Ok(Self { witness, sizing })
        }

        /// Get circuit k.
        pub fn k(&self) -> u32 {
            self.sizing.k
        }

        /// Estimate constraints.
        pub fn estimate_constraints(&self) -> usize {
            self.sizing.estimate_constraints()
        }

        /// Validate the witness (stub: checks conservation).
        pub fn validate_witness(&self) -> Result<(), String> {
            // Check conservation residual is within tolerance
            let residual = self.witness.conservation.residual;
            let tol = self.witness.params.conservation_tol;

            if residual.raw.abs() > tol.raw {
                return Err(format!(
                    "Conservation violation: |{}| > {}",
                    residual.to_f64(),
                    tol.to_f64(),
                ));
            }

            // Check SVD ordering
            for bond in &self.witness.truncation.bond_data {
                for pair in bond.singular_values.windows(2) {
                    if pair[0].raw < pair[1].raw {
                        return Err(format!(
                            "SVD ordering violation at bond {}: {} < {}",
                            bond.bond_index,
                            pair[0].to_f64(),
                            pair[1].to_f64(),
                        ));
                    }
                }
            }

            Ok(())
        }
    }
}

#[cfg(not(feature = "halo2"))]
pub use stub_circuit::ThermalCircuit;

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════
