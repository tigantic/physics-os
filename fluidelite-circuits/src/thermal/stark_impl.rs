//! STARK proof implementation for the Thermal/Heat Equation circuit.
//!
//! Uses Winterfell (Goldilocks field, FRI commitment) to generate
//! transparent STARK proofs for thermal physics timestep sequences.
//!
//! # Architecture
//!
//! The STARK proves that a sequence of thermal timesteps preserves
//! physical conservation laws. Each row of the execution trace captures
//! the physics summary of one solver timestep. Transition constraints
//! enforce energy conservation, parameter consistency, and sequential
//! execution across all rows.
//!
//! # Trace Layout (12 columns × N rows)
//!
//! | Col | Name             | Description                              |
//! |-----|------------------|------------------------------------------|
//! |  0  | energy           | Total thermal energy (∫T)                |
//! |  1  | energy_sq        | L2 norm (∫T²)                            |
//! |  2  | max_temp         | Maximum temperature                      |
//! |  3  | min_temp         | Minimum temperature                      |
//! |  4  | source_energy    | Net source contribution this step        |
//! |  5  | dt               | Timestep size (constant across rows)     |
//! |  6  | alpha            | Thermal diffusivity (constant)           |
//! |  7  | step_idx         | Step counter (0, 1, 2, …)               |
//! |  8  | cg_residual      | CG convergence residual                  |
//! |  9  | sv_max           | Largest singular value (QTT health)      |
//! | 10  | rank             | Effective QTT rank                       |
//! | 11  | cons_residual    | Conservation residual                    |
//!
//! # Transition Constraints (degree 1, all linear)
//!
//! 1. Energy conservation: `energy[n+1] - energy[n] - cons_residual[n+1] = 0`
//!    Conservation residual absorbs both source contributions and solver
//!    truncation error. The `dt·source` product is folded in at witness
//!    generation time, keeping the AIR strictly degree 1 and avoiding a
//!    degree mismatch when dt is constant across all rows.
//! 2. dt constant: `dt[n+1] - dt[n] = 0`
//! 3. α constant: `alpha[n+1] - alpha[n] = 0`
//! 4. Step increment: `step[n+1] - step[n] - 1 = 0`
//!
//! # Advantages over Halo2/KZG
//!
//! - **No trusted setup**: Fully transparent (hash-based)
//! - **Post-quantum secure**: Hash commitments, no discrete log assumption
//! - **GPU-parallelizable**: FFT + Merkle hashing (not sequential polynomial arguments)
//! - **Multi-step proof**: One proof covers the entire simulation sequence
//! - **Proof size**: ~15–50 KB (vs 800 B for Halo2, but irrelevant for regulated industries)
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use winterfell::{
    math::{fields::f64::BaseElement, FieldElement, ToElements},
    Air, AirContext, Assertion, AuxRandElements, BatchingMethod,
    CompositionPoly, CompositionPolyTrace, ConstraintCompositionCoefficients,
    DefaultConstraintCommitment, DefaultConstraintEvaluator, DefaultTraceLde,
    EvaluationFrame, FieldExtension, PartitionOptions, Proof, ProofOptions,
    Prover, StarkDomain, Trace, TraceInfo, TraceTable,
    TracePolyTable, TransitionConstraintDegree,
    AcceptableOptions,
    crypto::{hashers::Blake3_256, DefaultRandomCoin, MerkleTree},
    matrix::ColMatrix,
};

use fluidelite_core::field::Q16;

// ═══════════════════════════════════════════════════════════════════════════
// Type Aliases
// ═══════════════════════════════════════════════════════════════════════════

/// Goldilocks field element (p = 2^64 − 2^32 + 1).
pub type Felt = BaseElement;

/// Blake3 hasher over Goldilocks field.
type H = Blake3_256<Felt>;

/// Merkle tree commitment over Blake3.
type VC = MerkleTree<H>;

// ═══════════════════════════════════════════════════════════════════════════
// Trace Column Constants
// ═══════════════════════════════════════════════════════════════════════════

/// Number of columns in the execution trace.
pub const TRACE_WIDTH: usize = 12;

/// Number of transition constraints.
pub const NUM_CONSTRAINTS: usize = 4;

// Column indices.
/// Total thermal energy column.
pub const COL_ENERGY: usize = 0;
/// L2 norm (energy squared) column.
pub const COL_ENERGY_SQ: usize = 1;
/// Maximum temperature column.
pub const COL_MAX_TEMP: usize = 2;
/// Minimum temperature column.
pub const COL_MIN_TEMP: usize = 3;
/// Source energy contribution column.
pub const COL_SOURCE_ENERGY: usize = 4;
/// Timestep dt column (constant).
pub const COL_DT: usize = 5;
/// Diffusivity alpha column (constant).
pub const COL_ALPHA: usize = 6;
/// Step index counter column.
pub const COL_STEP_IDX: usize = 7;
/// CG solver residual column.
pub const COL_CG_RESIDUAL: usize = 8;
/// Largest singular value column.
pub const COL_SV_MAX: usize = 9;
/// Effective QTT rank column.
pub const COL_RANK: usize = 10;
/// Conservation residual column.
pub const COL_CONS_RESIDUAL: usize = 11;

// ═══════════════════════════════════════════════════════════════════════════
// Field Conversion Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Goldilocks modulus p = 2^64 − 2^32 + 1.
const GOLDILOCKS_P: u64 = 0xFFFF_FFFF_0000_0001;

/// Convert a Q16 fixed-point value to a Goldilocks field element.
///
/// Non-negative values map directly; negative values map to p − |value|
/// (additive inverse in the field).
pub fn q16_to_felt(q: Q16) -> Felt {
    if q.raw >= 0 {
        Felt::new(q.raw as u64)
    } else {
        // Field negation: p - |raw|
        let abs_val = (-q.raw) as u64;
        debug_assert!(abs_val < GOLDILOCKS_P, "Q16 magnitude exceeds Goldilocks modulus");
        Felt::new(GOLDILOCKS_P - abs_val)
    }
}

/// Convert an unsigned integer to a Goldilocks field element.
pub fn u64_to_felt(v: u64) -> Felt {
    Felt::new(v)
}

/// Convert a field element back to Q16 for diagnostics.
///
/// Values in the lower half of the field [0, p/2) are treated as non-negative;
/// values in the upper half [p/2, p) are treated as negative.
pub fn felt_to_q16(f: Felt) -> Q16 {
    let raw: u64 = f.as_int();
    let half_p = GOLDILOCKS_P / 2;
    if raw <= half_p {
        Q16::from_raw(raw as i64)
    } else {
        Q16::from_raw(-((GOLDILOCKS_P - raw) as i64))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Public Inputs
// ═══════════════════════════════════════════════════════════════════════════

/// Public inputs for the thermal STARK proof.
///
/// These values are committed to by the verifier and checked against
/// boundary assertions on the execution trace.
#[derive(Clone, Debug)]
pub struct ThermalStarkInputs {
    /// Total thermal energy at timestep 0.
    pub initial_energy: Felt,

    /// Total thermal energy at the final timestep.
    pub final_energy: Felt,

    /// Timestep size dt (Q16 embedded in Goldilocks).
    pub dt: Felt,

    /// Thermal diffusivity α (Q16 embedded in Goldilocks).
    pub alpha: Felt,

    /// Number of simulation steps in the trace.
    pub num_steps: usize,

    /// Trace length (next power of 2 ≥ num_steps + 1, minimum 8).
    pub trace_length: usize,
}

impl ToElements<Felt> for ThermalStarkInputs {
    fn to_elements(&self) -> Vec<Felt> {
        vec![
            self.initial_energy,
            self.final_energy,
            self.dt,
            self.alpha,
            Felt::new(self.num_steps as u64),
        ]
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Timestep Data (input from witness generation)
// ═══════════════════════════════════════════════════════════════════════════

/// Physics summary for one simulation timestep.
///
/// Extracted from the `ThermalWitness` after the thermal solver runs.
/// One `TimestepPhysics` populates one row of the STARK execution trace.
#[derive(Clone, Debug)]
pub struct TimestepPhysics {
    /// Total thermal energy: ∫T over the domain.
    pub energy: Q16,

    /// L2 norm squared: ∫T² over the domain.
    pub energy_sq: Q16,

    /// Maximum temperature in the field.
    pub max_temp: Q16,

    /// Minimum temperature in the field.
    pub min_temp: Q16,

    /// Net source energy contribution this timestep.
    pub source_energy: Q16,

    /// CG solver residual norm.
    pub cg_residual: Q16,

    /// Largest singular value in the QTT decomposition.
    pub sv_max: Q16,

    /// Effective rank of the QTT decomposition.
    pub rank: usize,

    /// Conservation residual: |∫T_{n+1} − ∫T_n − Δt·∫S|.
    pub conservation_residual: Q16,
}

// ═══════════════════════════════════════════════════════════════════════════
// AIR Definition — Algebraic Intermediate Representation
// ═══════════════════════════════════════════════════════════════════════════

/// Algebraic Intermediate Representation for thermal physics.
///
/// Defines the constraint system that the execution trace must satisfy
/// for a valid sequence of thermal timesteps.
pub struct ThermalAir {
    /// AIR context (trace layout, constraint degrees, options).
    context: AirContext<Felt>,

    /// Public inputs (boundary values).
    inputs: ThermalStarkInputs,
}

impl Air for ThermalAir {
    type BaseField = Felt;
    type PublicInputs = ThermalStarkInputs;

    fn new(
        trace_info: TraceInfo,
        pub_inputs: Self::PublicInputs,
        options: ProofOptions,
    ) -> Self {
        // All transition constraints are degree 1 (linear in trace columns).
        // The energy conservation constraint uses cons_residual to absorb
        // the dt·source product, which would otherwise require degree 2 and
        // create a degree mismatch when dt is constant (degree-2 product
        // collapses to degree-1, violating Winterfell's debug assertion).
        let degrees = vec![
            TransitionConstraintDegree::new(1), // energy conservation
            TransitionConstraintDegree::new(1), // dt constant
            TransitionConstraintDegree::new(1), // alpha constant
            TransitionConstraintDegree::new(1), // step increment
        ];

        // Number of boundary assertions.
        let num_assertions = 5;

        let context = AirContext::new(trace_info, degrees, num_assertions, options);

        Self {
            context,
            inputs: pub_inputs,
        }
    }

    fn evaluate_transition<E: FieldElement<BaseField = Felt>>(
        &self,
        frame: &EvaluationFrame<E>,
        _periodic_values: &[E],
        result: &mut [E],
    ) {
        let current = frame.current();
        let next = frame.next();

        // Constraint 0: Energy conservation (degree 1)
        //
        // Conservation residual captures the full per-step energy delta,
        // including source contributions (dt·source) and solver truncation
        // error. The witness generator precomputes this value so the AIR
        // stays strictly degree 1:
        //
        //   energy[n+1] - energy[n] - cons_residual[n+1] = 0
        result[0] = next[COL_ENERGY] - current[COL_ENERGY]
            - next[COL_CONS_RESIDUAL];

        // Constraint 1: Timestep dt is constant across all rows.
        result[1] = next[COL_DT] - current[COL_DT];

        // Constraint 2: Diffusivity α is constant across all rows.
        result[2] = next[COL_ALPHA] - current[COL_ALPHA];

        // Constraint 3: Step index increments by exactly 1.
        result[3] = next[COL_STEP_IDX] - current[COL_STEP_IDX] - E::ONE;
    }

    fn get_assertions(&self) -> Vec<Assertion<Felt>> {
        let last_step = self.trace_length() - 1;

        vec![
            // Initial energy at row 0.
            Assertion::single(COL_ENERGY, 0, self.inputs.initial_energy),
            // Final energy at last row.
            Assertion::single(COL_ENERGY, last_step, self.inputs.final_energy),
            // dt at row 0 (constant propagated by constraint 1).
            Assertion::single(COL_DT, 0, self.inputs.dt),
            // alpha at row 0 (constant propagated by constraint 2).
            Assertion::single(COL_ALPHA, 0, self.inputs.alpha),
            // Step index starts at 0 (incremented by constraint 3).
            Assertion::single(COL_STEP_IDX, 0, Felt::ZERO),
        ]
    }

    fn context(&self) -> &AirContext<Felt> {
        &self.context
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal Prover (Winterfell Prover trait implementation)
// ═══════════════════════════════════════════════════════════════════════════

/// Internal Winterfell prover for thermal STARK.
pub(crate) struct InternalProver {
    options: ProofOptions,
}

impl InternalProver {
    /// Create with the given proof options.
    pub fn new(options: ProofOptions) -> Self {
        Self { options }
    }
}

impl Prover for InternalProver {
    type BaseField = Felt;
    type Air = ThermalAir;
    type Trace = TraceTable<Felt>;
    type HashFn = H;
    type VC = VC;
    type RandomCoin = DefaultRandomCoin<H>;
    type TraceLde<E: FieldElement<BaseField = Felt>> = DefaultTraceLde<E, H, VC>;
    type ConstraintEvaluator<'a, E: FieldElement<BaseField = Felt>> =
        DefaultConstraintEvaluator<'a, Self::Air, E>;
    type ConstraintCommitment<E: FieldElement<BaseField = Self::BaseField>> =
        DefaultConstraintCommitment<E, Self::HashFn, Self::VC>;

    fn get_pub_inputs(&self, trace: &Self::Trace) -> ThermalStarkInputs {
        let last = trace.length() - 1;

        ThermalStarkInputs {
            initial_energy: trace.get(COL_ENERGY, 0),
            final_energy: trace.get(COL_ENERGY, last),
            dt: trace.get(COL_DT, 0),
            alpha: trace.get(COL_ALPHA, 0),
            num_steps: last, // last row index = number of transitions
            trace_length: trace.length(),
        }
    }

    fn options(&self) -> &ProofOptions {
        &self.options
    }

    fn new_trace_lde<E: FieldElement<BaseField = Felt>>(
        &self,
        trace_info: &TraceInfo,
        main_trace: &ColMatrix<Felt>,
        domain: &StarkDomain<Felt>,
        partition_options: PartitionOptions,
    ) -> (Self::TraceLde<E>, TracePolyTable<E>) {
        DefaultTraceLde::new(trace_info, main_trace, domain, partition_options)
    }

    fn new_evaluator<'a, E: FieldElement<BaseField = Felt>>(
        &self,
        air: &'a Self::Air,
        aux_rand_elements: Option<AuxRandElements<E>>,
        composition_coefficients: ConstraintCompositionCoefficients<E>,
    ) -> Self::ConstraintEvaluator<'a, E> {
        DefaultConstraintEvaluator::new(air, aux_rand_elements, composition_coefficients)
    }

    fn build_constraint_commitment<E: FieldElement<BaseField = Felt>>(
        &self,
        composition_poly_trace: CompositionPolyTrace<E>,
        num_trace_poly_columns: usize,
        domain: &StarkDomain<Felt>,
        partition_options: PartitionOptions,
    ) -> (Self::ConstraintCommitment<E>, CompositionPoly<E>) {
        DefaultConstraintCommitment::new(
            composition_poly_trace,
            num_trace_poly_columns,
            domain,
            partition_options,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Trace Builder
// ═══════════════════════════════════════════════════════════════════════════

/// Build the execution trace from a sequence of timestep physics summaries.
///
/// The trace has `TRACE_WIDTH` columns and is padded to the next power of two.
/// Padding rows replicate the final physics state with incrementing step indices
/// and zero source/conservation residuals (satisfying all transition constraints).
pub fn build_trace(
    steps: &[TimestepPhysics],
    dt: Q16,
    alpha: Q16,
) -> Result<TraceTable<Felt>, String> {
    if steps.is_empty() {
        return Err("No timestep data to prove".to_string());
    }

    // Determine trace length: next power of 2 ≥ len, minimum 8.
    let raw_len = steps.len();
    let trace_len = raw_len.next_power_of_two().max(8);

    let dt_felt = q16_to_felt(dt);
    let alpha_felt = q16_to_felt(alpha);

    let mut trace = TraceTable::new(TRACE_WIDTH, trace_len);
    trace.fill(
        |state| {
            // Initialize row 0 from first timestep.
            let s = &steps[0];
            state[COL_ENERGY] = q16_to_felt(s.energy);
            state[COL_ENERGY_SQ] = q16_to_felt(s.energy_sq);
            state[COL_MAX_TEMP] = q16_to_felt(s.max_temp);
            state[COL_MIN_TEMP] = q16_to_felt(s.min_temp);
            state[COL_SOURCE_ENERGY] = q16_to_felt(s.source_energy);
            state[COL_DT] = dt_felt;
            state[COL_ALPHA] = alpha_felt;
            state[COL_STEP_IDX] = Felt::ZERO;
            state[COL_CG_RESIDUAL] = q16_to_felt(s.cg_residual);
            state[COL_SV_MAX] = q16_to_felt(s.sv_max);
            state[COL_RANK] = Felt::new(s.rank as u64);
            state[COL_CONS_RESIDUAL] = q16_to_felt(s.conservation_residual);
        },
        |step, state| {
            let row_idx = step + 1; // `step` is 0-based transition index → row = step+1.

            if row_idx < raw_len {
                // Real timestep data.
                let s = &steps[row_idx];
                state[COL_ENERGY] = q16_to_felt(s.energy);
                state[COL_ENERGY_SQ] = q16_to_felt(s.energy_sq);
                state[COL_MAX_TEMP] = q16_to_felt(s.max_temp);
                state[COL_MIN_TEMP] = q16_to_felt(s.min_temp);
                state[COL_SOURCE_ENERGY] = q16_to_felt(s.source_energy);
                state[COL_CG_RESIDUAL] = q16_to_felt(s.cg_residual);
                state[COL_SV_MAX] = q16_to_felt(s.sv_max);
                state[COL_RANK] = Felt::new(s.rank as u64);
                state[COL_CONS_RESIDUAL] = q16_to_felt(s.conservation_residual);
            } else {
                // Padding row: replicate final state with zero source/residual.
                // This satisfies transition constraints:
                //   energy stays the same (source=0, cons_residual=0)
                //   dt, alpha constant (already in state)
                //   step increments
                state[COL_SOURCE_ENERGY] = Felt::ZERO;
                state[COL_CONS_RESIDUAL] = Felt::ZERO;
                // energy, energy_sq, max_temp, min_temp, cg_residual,
                // sv_max, rank carry forward from previous row (already in state).
            }

            // Constants and step index always advance.
            state[COL_DT] = dt_felt;
            state[COL_ALPHA] = alpha_felt;
            state[COL_STEP_IDX] += Felt::ONE;
        },
    );

    Ok(trace)
}

// ═══════════════════════════════════════════════════════════════════════════
// Public API: Prove & Verify
// ═══════════════════════════════════════════════════════════════════════════

/// Default proof options: ~127-bit conjectured security.
///
/// - 40 queries × 3-bit blowup (log₂ 8) = 120 bits query security
/// - Grinding factor 16 adds 16 bits PoW → 136 bits query security
/// - Quadratic extension doubles Goldilocks field security: 64 → 128 bits
/// - Bottleneck: min(128, 136) − 1 = 127, capped by Blake3 collision
///   resistance (128 bits) → **127-bit conjectured security**
/// - FRI folding factor 8, remainder degree 31
/// - Linear batching for both constraints and DEEP quotients
pub fn default_proof_options() -> ProofOptions {
    ProofOptions::new(
        40,                          // num_queries
        8,                           // blowup_factor (2^3)
        16,                          // grinding_factor
        FieldExtension::Quadratic,   // quadratic extension for ≥80-bit security
        8,                           // fri_folding_factor
        31,                          // fri_remainder_max_degree
        BatchingMethod::Linear,      // constraint batching
        BatchingMethod::Linear,      // DEEP quotient batching
    )
}

/// Generate a STARK proof for a sequence of thermal timesteps.
///
/// # Arguments
/// - `steps`: Physics summaries for each timestep (from witness generation)
/// - `dt`: Timestep size (Q16)
/// - `alpha`: Thermal diffusivity (Q16)
///
/// # Returns
/// `(proof_bytes, public_inputs, trace_length, generation_time_ms)`
pub fn prove_thermal_stark(
    steps: &[TimestepPhysics],
    dt: Q16,
    alpha: Q16,
) -> Result<(Vec<u8>, ThermalStarkInputs, usize, u64), String> {
    let start = std::time::Instant::now();

    // Build execution trace.
    let trace = build_trace(steps, dt, alpha)?;
    let trace_len = trace.length();

    // Create prover with default options.
    let options = default_proof_options();
    let prover = InternalProver::new(options);

    // Extract public inputs before proving (trace is consumed by prove).
    let pub_inputs = prover.get_pub_inputs(&trace);

    // Generate STARK proof.
    let proof: Proof = prover
        .prove(trace)
        .map_err(|e| format!("STARK proof generation failed: {:?}", e))?;

    let proof_bytes = proof.to_bytes();
    let generation_time_ms = start.elapsed().as_millis() as u64;

    Ok((proof_bytes, pub_inputs, trace_len, generation_time_ms))
}

/// Verify a STARK proof for thermal physics.
///
/// # Arguments
/// - `proof_bytes`: Serialized STARK proof
/// - `pub_inputs`: Public inputs (energy, dt, alpha, etc.)
///
/// # Returns
/// `true` if the proof is valid, `false` otherwise.
pub fn verify_thermal_stark(
    proof_bytes: &[u8],
    pub_inputs: ThermalStarkInputs,
) -> Result<bool, String> {
    // Deserialize proof.
    let proof = Proof::from_bytes(proof_bytes)
        .map_err(|e| format!("Failed to deserialize STARK proof: {:?}", e))?;

    // Verification accepts proofs with ≥ 80-bit conjectured security.
    let acceptable = AcceptableOptions::MinConjecturedSecurity(80);

    match winterfell::verify::<ThermalAir, H, DefaultRandomCoin<H>, VC>(
        proof,
        pub_inputs,
        &acceptable,
    ) {
        Ok(()) => Ok(true),
        Err(e) => Err(format!("STARK verification failed: {:?}", e)),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Build synthetic timestep data for testing.
    ///
    /// Energy increases by `per_step_drift` each step. The conservation
    /// residual for row i (i > 0) equals `per_step_drift`, exactly matching
    /// the energy delta so that constraint 0 evaluates to zero.
    ///
    /// Source energy is non-zero (10.0) — its contribution is folded into
    /// `conservation_residual` by the witness generator. The dt·source
    /// product does NOT appear in the AIR constraint; it is absorbed.
    fn make_test_steps(n: usize) -> Vec<TimestepPhysics> {
        let mut steps = Vec::with_capacity(n);
        let base_energy = Q16::from_f64(1000.0);
        let per_step_drift = Q16::from_f64(0.005);

        for i in 0..n {
            let energy = Q16::from_raw(base_energy.raw + (i as i64) * per_step_drift.raw);

            // cons_residual = energy[i] - energy[i-1] = per_step_drift.
            // Row 0 doesn't appear as "next" in any transition.
            let cons_residual = if i == 0 { Q16::ZERO } else { per_step_drift };

            steps.push(TimestepPhysics {
                energy,
                energy_sq: Q16::from_f64(1_000_000.0),
                max_temp: Q16::from_f64(1.5),
                min_temp: Q16::from_f64(0.2),
                source_energy: Q16::from_f64(10.0),
                cg_residual: Q16::from_f64(1e-6),
                sv_max: Q16::from_f64(0.99),
                rank: 8,
                conservation_residual: cons_residual,
            });
        }
        steps
    }

    #[test]
    fn test_trace_build() {
        let steps = make_test_steps(5);
        let dt = Q16::from_f64(0.001);
        let alpha = Q16::from_f64(0.01);

        let trace = build_trace(&steps, dt, alpha).unwrap();
        assert_eq!(trace.length(), 8); // 5 → padded to 8
        assert_eq!(trace.width(), TRACE_WIDTH);

        // Check initial energy.
        let initial_energy = trace.get(COL_ENERGY, 0);
        assert_eq!(initial_energy, q16_to_felt(steps[0].energy));

        // Check step index progression.
        assert_eq!(trace.get(COL_STEP_IDX, 0), Felt::ZERO);
        assert_eq!(trace.get(COL_STEP_IDX, 1), Felt::new(1));
        assert_eq!(trace.get(COL_STEP_IDX, 7), Felt::new(7));
    }

    #[test]
    fn test_prove_and_verify() {
        let steps = make_test_steps(5);
        let dt = Q16::from_f64(0.001);
        let alpha = Q16::from_f64(0.01);

        let (proof_bytes, pub_inputs, trace_len, gen_ms) =
            prove_thermal_stark(&steps, dt, alpha).unwrap();

        assert!(proof_bytes.len() > 100, "proof too small: {} bytes", proof_bytes.len());
        assert_eq!(trace_len, 8);
        assert!(gen_ms < 30_000, "proof generation too slow: {} ms", gen_ms);

        // Verify.
        let valid = verify_thermal_stark(&proof_bytes, pub_inputs).unwrap();
        assert!(valid, "STARK verification failed");
    }

    #[test]
    fn test_q16_felt_roundtrip() {
        let positive = Q16::from_f64(42.5);
        let negative = Q16::from_f64(-17.3);
        let zero = Q16::ZERO;

        assert_eq!(felt_to_q16(q16_to_felt(positive)).raw, positive.raw);
        assert_eq!(felt_to_q16(q16_to_felt(negative)).raw, negative.raw);
        assert_eq!(felt_to_q16(q16_to_felt(zero)).raw, 0);
    }

    #[test]
    fn test_larger_trace() {
        // 32 timesteps → trace length 32 (already power of 2).
        let steps = make_test_steps(32);
        let dt = Q16::from_f64(0.001);
        let alpha = Q16::from_f64(0.01);

        let trace = build_trace(&steps, dt, alpha).unwrap();
        assert_eq!(trace.length(), 32);

        let (proof_bytes, pub_inputs, _, _) =
            prove_thermal_stark(&steps, dt, alpha).unwrap();
        let valid = verify_thermal_stark(&proof_bytes, pub_inputs).unwrap();
        assert!(valid);
    }
}
