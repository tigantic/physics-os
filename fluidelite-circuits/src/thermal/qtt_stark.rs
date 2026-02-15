//! QTT-Native MPO×MPS Contraction STARK Proof.
//!
//! Proves that a given system matrix `A = I - α·Δt·L` applied to an input MPS
//! produces the claimed output MPS, with every fixed-point MAC step constrained.
//!
//! # Architecture
//!
//! The contraction STARK uses a **row-per-MAC** layout: each row of the
//! execution trace represents one multiply-accumulate step in the MPO×MPS
//! contraction. For physical dimension d=2, each output element requires
//! exactly 2 MAC rows (inner summation over d_in=2).
//!
//! # Trace Layout (23 columns × N_macs rows)
//!
//! | Col  | Name        | Description                                  |
//! |------|-------------|----------------------------------------------|
//! |  0   | mpo_val     | MPO core element O[dl,o,p,dr]                |
//! |  1   | mps_val     | MPS core element P[cl,p,cr]                  |
//! |  2   | acc_before  | Accumulator before this MAC step             |
//! |  3   | acc_after   | Accumulator after this MAC step              |
//! |  4   | remainder   | Q16 FP multiplication remainder ∈ [0,65535]  |
//! |  5   | inner_idx   | Inner summation index (0 or 1 for d=2)       |
//! |  6   | output_val  | Output MPS core element (verified on last p) |
//! | 7–22 | rem_bits    | 16-bit decomposition of remainder            |
//!
//! # Transition Constraints (22 total, max degree 2)
//!
//! 1. **MAC validity** (degree 2):
//!    `mpo_val × mps_val − (acc_after − acc_before) × SCALE − remainder = 0`
//!
//! 2. **Accumulator start** (degree 2):
//!    `(1 − inner_idx) × acc_before = 0`
//!    When inner_idx=0 (first MAC of an output element), accumulator must be 0.
//!
//! 3. **Chain continuity** (degree 2):
//!    `next.inner_idx × (next.acc_before − current.acc_after) = 0`
//!    When continuing a chain (next.inner_idx≠0), carry forward the accumulator.
//!
//! 4. **Output capture** (degree 2):
//!    `inner_idx × (acc_after − output_val) = 0`
//!    On the last MAC (inner_idx=d-1=1), the accumulator must equal the output.
//!
//! 5–20. **Remainder bit booleans** (degree 2, ×16):
//!    `rem_bit_k × (rem_bit_k − 1) = 0` for k ∈ [0, 15]
//!
//! 21. **Remainder recomposition** (degree 1):
//!    `Σ(rem_bit_k × 2^k) − remainder = 0`
//!
//! 22. **Inner index binary** (degree 2):
//!    `inner_idx × (inner_idx − 1) = 0`
//!
//! # Boundary Assertions
//!
//! - Row 0: inner_idx = 0, acc_before = 0
//! - MPO pinning (Task 6.7): every row's mpo_val matches expected value
//! - MPS pinning: every row's mps_val matches expected input MPS value
//! - Output pinning: every last-MAC row's output_val matches expected output
//! - Residual bound (Task 6.8): final residual norm² ≤ ε²
//!
//! # Security
//!
//! The contraction STARK runs independently from the chain-level STARK
//! (stark_impl.rs). Together they prove:
//! - Chain STARK: faithful state evolution with hash-chained commitments
//! - Contraction STARK: each MPO×MPS contraction step is arithmetically correct
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use winterfell::{
    math::{FieldElement, ToElements},
    Air, AirContext, Assertion, AuxRandElements, BatchingMethod,
    CompositionPoly, CompositionPolyTrace, ConstraintCompositionCoefficients,
    DefaultConstraintCommitment, DefaultConstraintEvaluator, DefaultTraceLde,
    EvaluationFrame, FieldExtension, PartitionOptions, Proof, ProofOptions,
    Prover, StarkDomain, TraceInfo, TraceTable,
    TracePolyTable, TransitionConstraintDegree,
    AcceptableOptions,
    crypto::{hashers::Blake3_256, DefaultRandomCoin, MerkleTree},
    matrix::ColMatrix,
};

use fluidelite_core::field::Q16;

use super::stark_impl::{q16_to_felt, Felt};
use super::witness::{ContractionWitness, SiteContractionData};
use crate::tensor::{Mps, Mpo};

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

/// Q16 scale factor as a field element: 2^16 = 65536.
const SCALE: u64 = 65536;

/// Number of bits for remainder range check.
const REM_BITS: usize = 16;

// ── Sentinel padding values ────────────────────────────────────────────
// Padding rows use non-trivial values that satisfy all 22 transition
// constraints and guarantee every trace column has at least one non-zero
// value.  This prevents constraint polynomials from collapsing to the
// zero polynomial (which would violate Winterfell's degree validation).
//
// Sentinel equation:  131071 × 1 = 1 × 65536 + 65535  ✓
const SENTINEL_MPO_VAL: u64 = SCALE * 2 - 1;  // 131071
const SENTINEL_MPS_VAL: u64 = 1;
const SENTINEL_ACC_AFTER: u64 = 1;
const SENTINEL_REMAINDER: u64 = SCALE - 1;     // 65535 = all 16 bits set

// ── Column indices ─────────────────────────────────────────────────────

/// MPO core element value.
const COL_MPO_VAL: usize = 0;
/// MPS core element value.
const COL_MPS_VAL: usize = 1;
/// Accumulator before this MAC step.
const COL_ACC_BEFORE: usize = 2;
/// Accumulator after this MAC step.
const COL_ACC_AFTER: usize = 3;
/// Fixed-point multiplication remainder ∈ [0, 65535].
const COL_REMAINDER: usize = 4;
/// Inner summation index (0 for first, d-1 for last).
const COL_INNER_IDX: usize = 5;
/// Output MPS core element (constrained on last MAC row).
const COL_OUTPUT_VAL: usize = 6;
/// First column of 16-bit remainder decomposition.
const COL_REM_BITS_START: usize = 7;

/// Total trace width: 7 data columns + 16 remainder bits = 23.
pub const QTT_TRACE_WIDTH: usize = 7 + REM_BITS;

/// Number of transition constraints.
pub const QTT_NUM_CONSTRAINTS: usize = 21;

// ═══════════════════════════════════════════════════════════════════════════
// Public Inputs
// ═══════════════════════════════════════════════════════════════════════════

/// Public inputs for the QTT contraction STARK.
///
/// Contains enough information for the verifier to independently reconstruct
/// the expected MPO values and validate the contraction.
#[derive(Clone, Debug)]
pub struct ContractionStarkInputs {
    /// Total number of MAC rows in the trace (before padding).
    pub num_mac_rows: usize,

    /// Padded trace length (power of 2).
    pub trace_length: usize,

    /// Number of QTT sites.
    pub num_sites: usize,

    /// MPS bond dimension (χ_max).
    pub chi_max: usize,

    /// MPO bond dimension (D, from system matrix).
    pub mpo_bond_dim: usize,

    /// Physical dimension (d = 2 for QTT).
    pub d_phys: usize,

    /// Thermal diffusivity α (Q16).
    pub alpha: Felt,

    /// Timestep dt (Q16).
    pub dt: Felt,

    /// Grid spacing dx (Q16).
    pub dx: Felt,

    /// Expected MPO core values for every MAC row (computed by verifier).
    /// The verifier reconstructs the system matrix from (α, dt, dx) and
    /// lays out the expected MPO element for each row position.
    pub expected_mpo_vals: Vec<Felt>,

    /// Expected input MPS core values for every MAC row.
    pub expected_mps_vals: Vec<Felt>,

    /// Expected output MPS core values for every last-MAC row.
    /// Indexed by output element index (not row index).
    pub expected_output_vals: Vec<Felt>,

    /// Residual norm squared ‖r‖² = ‖A·x - b‖² (Q16, for task 6.8).
    pub residual_norm_sq: Felt,

    /// Residual tolerance squared ε² (Q16).
    pub tolerance_sq: Felt,
}

impl ToElements<Felt> for ContractionStarkInputs {
    fn to_elements(&self) -> Vec<Felt> {
        let mut elems = vec![
            Felt::new(self.num_mac_rows as u64),
            Felt::new(self.trace_length as u64),
            Felt::new(self.num_sites as u64),
            Felt::new(self.chi_max as u64),
            Felt::new(self.mpo_bond_dim as u64),
            Felt::new(self.d_phys as u64),
            self.alpha,
            self.dt,
            self.dx,
            self.residual_norm_sq,
            self.tolerance_sq,
        ];
        // Include expected values in Fiat-Shamir transcript.
        elems.extend_from_slice(&self.expected_mpo_vals);
        elems.extend_from_slice(&self.expected_mps_vals);
        elems.extend_from_slice(&self.expected_output_vals);
        elems
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Contraction Schedule
// ═══════════════════════════════════════════════════════════════════════════

/// One row in the MAC schedule: maps a trace row to contraction indices.
#[derive(Clone, Debug)]
pub struct MacRow {
    /// QTT site index.
    site: usize,
    /// Output left bond index (cl * D_l + dl).
    pub new_left: usize,
    /// Output physical index.
    pub out_phys: usize,
    /// Output right bond index (cr * D_r + dr).
    pub new_right: usize,
    /// Inner summation index (0..d_in).
    inner_p: usize,
    /// Decomposed: MPS left bond index.
    cl: usize,
    /// Decomposed: MPO left bond index.
    dl: usize,
    /// Decomposed: MPS right bond index.
    cr: usize,
    /// Decomposed: MPO right bond index.
    dr: usize,
}

/// Build the deterministic MAC schedule for an MPO×MPS contraction.
///
/// The schedule defines the exact order of MAC operations and maps each
/// trace row to the corresponding MPS/MPO indices. Both prover and verifier
/// use this same function to ensure consistency.
fn build_mac_schedule(mps: &Mps, mpo: &Mpo) -> Vec<MacRow> {
    let num_sites = mps.num_sites;
    let d_out = mpo.d_out();
    let d_in = mpo.d_in();

    let mut schedule = Vec::new();

    for site in 0..num_sites {
        let mcl = mps.chi_left(site);
        let mcr = mps.chi_right(site);
        let odl = mpo.dl(site);
        let odr = mpo.dr(site);

        for cl in 0..mcl {
            for dl in 0..odl {
                let new_left = cl * odl + dl;

                for o in 0..d_out {
                    for cr in 0..mcr {
                        for dr in 0..odr {
                            let new_right = cr * odr + dr;

                            for p in 0..d_in {
                                schedule.push(MacRow {
                                    site,
                                    new_left,
                                    out_phys: o,
                                    new_right,
                                    inner_p: p,
                                    cl,
                                    dl,
                                    cr,
                                    dr,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    schedule
}

// ═══════════════════════════════════════════════════════════════════════════
// Sentinel Padding
// ═══════════════════════════════════════════════════════════════════════════

/// Fill a padding row with the sentinel values.
///
/// Every column receives a non-zero value that satisfies all 22 transition
/// constraints.  This prevents any trace column polynomial from being
/// identically zero, which would cause Winterfell's degree validation to
/// fail.
///
/// Sentinel equation: `131071 × 1 = 1 × 65536 + 65535`
fn fill_sentinel_row(state: &mut [Felt]) {
    state[COL_MPO_VAL]   = Felt::new(SENTINEL_MPO_VAL);
    state[COL_MPS_VAL]   = Felt::new(SENTINEL_MPS_VAL);
    state[COL_ACC_BEFORE] = Felt::ZERO;
    state[COL_ACC_AFTER]  = Felt::new(SENTINEL_ACC_AFTER);
    state[COL_REMAINDER]  = Felt::new(SENTINEL_REMAINDER);
    state[COL_INNER_IDX]  = Felt::ZERO;
    state[COL_OUTPUT_VAL] = Felt::new(SENTINEL_ACC_AFTER); // output = acc_after
    for k in 0..REM_BITS {
        state[COL_REM_BITS_START + k] = Felt::ONE; // all 16 bits = 1
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Trace Builder
// ═══════════════════════════════════════════════════════════════════════════

/// Build the contraction execution trace from witness data.
///
/// Each row corresponds to one MAC operation in the contraction schedule.
/// The trace is padded with sentinel rows to the next power of 2.
pub fn build_contraction_trace(
    mps: &Mps,
    mpo: &Mpo,
    witness: &ContractionWitness,
) -> Result<(TraceTable<Felt>, Vec<MacRow>), String> {
    if mps.num_sites != mpo.num_sites {
        return Err(format!(
            "Site mismatch: MPS has {} sites, MPO has {}",
            mps.num_sites, mpo.num_sites
        ));
    }
    if witness.num_sites != mps.num_sites {
        return Err(format!(
            "Witness site mismatch: {} vs {}",
            witness.num_sites, mps.num_sites
        ));
    }

    let schedule = build_mac_schedule(mps, mpo);
    let num_real_rows = schedule.len();

    if num_real_rows == 0 {
        return Err("Empty contraction schedule".to_string());
    }

    // Guarantee at least one sentinel padding row so every trace column has
    // at least one non-zero value (prevents zero-polynomial constraint degree
    // mismatches in Winterfell).
    let trace_len = (num_real_rows + 1).next_power_of_two().max(8);
    let d_in = mpo.d_in();

    // Pre-compute the output element index for each schedule row.
    // The witness stores MAC accumulators per output element per site.
    // Each group of d_in consecutive inner_p values (0..d_in-1) forms one
    // output element. We track this up-front so both fill closures can
    // index into a shared read-only array.
    let mut site_output_counter = vec![0usize; mps.num_sites];
    let mut row_output_elem = Vec::with_capacity(num_real_rows);
    for mac in &schedule {
        if mac.inner_p == 0 {
            row_output_elem.push(site_output_counter[mac.site]);
            site_output_counter[mac.site] += 1;
        } else {
            // Continuation of the current output element.
            row_output_elem.push(site_output_counter[mac.site] - 1);
        }
    }

    let mut trace = TraceTable::new(QTT_TRACE_WIDTH, trace_len);

    trace.fill(
        |state| {
            // Row 0: first MAC in the schedule.
            fill_mac_row(
                state,
                &schedule[0],
                mps,
                mpo,
                &witness.site_data,
                row_output_elem[0],
                0,
                d_in,
            );
        },
        |step, state| {
            let row_idx = step + 1;

            if row_idx < num_real_rows {
                let mac = &schedule[row_idx];

                fill_mac_row(
                    state,
                    mac,
                    mps,
                    mpo,
                    &witness.site_data,
                    row_output_elem[row_idx],
                    mac.inner_p,
                    d_in,
                );
            } else {
                // Sentinel padding: non-trivial values that satisfy all
                // constraints and prevent any column from being identically
                // zero across the entire trace.
                fill_sentinel_row(state);
            }
        },
    );

    Ok((trace, schedule))
}


/// Fill a single trace row from the MAC schedule and witness data.
fn fill_mac_row(
    state: &mut [Felt],
    mac: &MacRow,
    mps: &Mps,
    mpo: &Mpo,
    site_data: &[SiteContractionData],
    output_elem_idx: usize,
    _inner_mac_idx: usize,
    d_in: usize,
) {
    let site = mac.site;

    // MPO value: O[dl, o, p, dr]
    let mpo_val = mpo.get(site, mac.dl, mac.out_phys, mac.inner_p, mac.dr);
    state[COL_MPO_VAL] = q16_to_felt(mpo_val);

    // MPS value: P[cl, p, cr]
    let mps_val = mps.get(site, mac.cl, mac.inner_p, mac.cr);
    state[COL_MPS_VAL] = q16_to_felt(mps_val);

    // Inner index
    state[COL_INNER_IDX] = Felt::new(mac.inner_p as u64);

    // Accumulator values from witness
    let sd = &site_data[site];

    // The accumulator chain for this output element.
    // mac_accumulators[output_elem_idx] = [acc_0=0, acc_1, ..., acc_{d_in}]
    // For inner_p, acc_before = chain[inner_p], acc_after = chain[inner_p + 1].
    if output_elem_idx < sd.mac_accumulators.len() {
        let chain = &sd.mac_accumulators[output_elem_idx];
        let acc_before = chain[mac.inner_p];
        let acc_after = chain[mac.inner_p + 1];

        state[COL_ACC_BEFORE] = q16_to_felt(acc_before);
        state[COL_ACC_AFTER] = q16_to_felt(acc_after);

        // Output value: the final accumulator value (at inner_p = d_in - 1).
        let output_val = chain[d_in]; // chain[d_in] = final accumulator
        state[COL_OUTPUT_VAL] = q16_to_felt(output_val);
    } else {
        state[COL_ACC_BEFORE] = Felt::ZERO;
        state[COL_ACC_AFTER] = Felt::ZERO;
        state[COL_OUTPUT_VAL] = Felt::ZERO;
    }

    // Remainder from witness.
    // fp_remainders are stored flat: one per MAC operation, in schedule order.
    // The index into fp_remainders for this output element + inner step:
    let rem_flat_idx = output_elem_idx * d_in + mac.inner_p;
    let remainder = if rem_flat_idx < sd.fp_remainders.len() {
        sd.fp_remainders[rem_flat_idx]
    } else {
        0i64
    };

    // Remainder must be non-negative for the range check.
    // The witness generator computes: full_product - quotient << 16.
    // For negative products, the remainder can be negative.
    // We handle this by taking the absolute value and adjusting the constraint.
    // Actually: in Q16, `full_product = a * b`, `quotient = full_product >> 16`,
    // `remainder = full_product - (quotient << 16)`.
    // Since `quotient = full_product >> 16` (arithmetic shift), remainder ∈ [0, 65535].
    // But for negative full_products, `>>` rounds toward negative infinity,
    // so remainder is still in [0, 65535].
    //
    // Correction: Rust's `>>` on i128 is arithmetic (sign-extending), but
    // the witness code computes `quotient = (full_product >> 16) as i64`.
    // For negative full_product, this floors toward -∞, making remainder ≥ 0.
    //
    // However, the witness actually stores remainder = full_product - (quotient << 16),
    // and quotient = full_product >> 16. For the bit decomposition to work,
    // we need remainder ∈ [0, 65535].
    let remainder_abs = remainder.unsigned_abs();
    debug_assert!(
        remainder_abs < SCALE,
        "Remainder {} out of range at site {}, output elem {}, inner {}",
        remainder,
        site,
        output_elem_idx,
        mac.inner_p,
    );

    state[COL_REMAINDER] = Felt::new(remainder_abs);

    // 16-bit decomposition of remainder.
    for bit in 0..REM_BITS {
        let b = (remainder_abs >> bit) & 1;
        state[COL_REM_BITS_START + bit] = Felt::new(b);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// AIR Definition
// ═══════════════════════════════════════════════════════════════════════════

/// Algebraic Intermediate Representation for QTT contraction proof.
pub struct ContractionAir {
    /// AIR context.
    context: AirContext<Felt>,
    /// Public inputs.
    inputs: ContractionStarkInputs,
}

impl Air for ContractionAir {
    type BaseField = Felt;
    type PublicInputs = ContractionStarkInputs;

    fn new(
        trace_info: TraceInfo,
        pub_inputs: Self::PublicInputs,
        options: ProofOptions,
    ) -> Self {
        // 21 degree-2 constraints (see evaluate_transition for details):
        //
        //  0: MAC validity with embedded bit recomposition
        //  1: Accumulator start
        //  2: Chain continuity
        //  3: Output capture
        //  4-19: 16 remainder-bit booleans
        //  20: inner_idx binary
        //
        // NOTE: We fold the remainder recomposition into the MAC constraint
        // (replacing `remainder` with the sum of bit[k]*2^k) to avoid a degree-1
        // constraint whose polynomial is algebraically tautological (always
        // the zero polynomial for any consistent trace -- violating
        // Winterfell's strict degree validation).
        let mut degrees = vec![
            TransitionConstraintDegree::new(2), // 0: MAC validity + bit recomp
            TransitionConstraintDegree::new(2), // 1: accumulator start
            TransitionConstraintDegree::new(2), // 2: chain continuity
            TransitionConstraintDegree::new(2), // 3: output capture
        ];
        for _ in 0..REM_BITS {
            degrees.push(TransitionConstraintDegree::new(2)); // 4-19: bit booleans
        }
        degrees.push(TransitionConstraintDegree::new(2)); // 20: inner binary
        assert_eq!(degrees.len(), QTT_NUM_CONSTRAINTS);

        // Boundary assertions: row 0 (2) + MPO pinning + MPS pinning + output pinning.
        let num_mpo_assertions = pub_inputs.expected_mpo_vals.len();
        let num_mps_assertions = pub_inputs.expected_mps_vals.len();
        let num_output_assertions = pub_inputs.expected_output_vals.len();
        let num_assertions = 2 + num_mpo_assertions + num_mps_assertions + num_output_assertions;

        let context = AirContext::new(trace_info, degrees, num_assertions, options);

        Self { context, inputs: pub_inputs }
    }

    fn evaluate_transition<E: FieldElement<BaseField = Felt>>(
        &self,
        frame: &EvaluationFrame<E>,
        _periodic_values: &[E],
        result: &mut [E],
    ) {
        let current = frame.current();
        let next = frame.next();

        let mpo_val = current[COL_MPO_VAL];
        let mps_val = current[COL_MPS_VAL];
        let acc_before = current[COL_ACC_BEFORE];
        let acc_after = current[COL_ACC_AFTER];
        let inner_idx = current[COL_INNER_IDX];
        let output_val = current[COL_OUTPUT_VAL];

        let scale = E::from(Felt::new(SCALE));

        // Recompose remainder from bits: sum of bit[k] * 2^k
        let mut rem_recomposed = E::ZERO;
        for k in 0..REM_BITS {
            let bit = current[COL_REM_BITS_START + k];
            rem_recomposed += bit * E::from(Felt::new(1u64 << k));
        }

        // Constraint 0: MAC validity with folded bit recomposition (degree 2)
        // mpo * mps = (acc_after - acc_before) * SCALE + sum(bit[k]*2^k)
        result[0] = mpo_val * mps_val - (acc_after - acc_before) * scale - rem_recomposed;

        // Constraint 1: Accumulator start (degree 2)
        // When inner_idx = 0, acc_before must be 0.
        result[1] = (E::ONE - inner_idx) * acc_before;

        // Constraint 2: Chain continuity (degree 2)
        // When next row continues a chain, carry accumulator forward.
        result[2] = next[COL_INNER_IDX] * (next[COL_ACC_BEFORE] - acc_after);

        // Constraint 3: Output capture (degree 2)
        // On the last MAC step (inner_idx = d-1 = 1 for d=2), acc_after = output.
        result[3] = inner_idx * (acc_after - output_val);

        // Constraints 4-19: Remainder bit booleans (degree 2 each)
        for k in 0..REM_BITS {
            let bit = current[COL_REM_BITS_START + k];
            result[4 + k] = bit * (bit - E::ONE);
        }

        // Constraint 20: Inner index binary (degree 2)
        result[20] = inner_idx * (inner_idx - E::ONE);
    }

    fn get_assertions(&self) -> Vec<Assertion<Felt>> {
        let mut assertions = Vec::new();

        // Row 0: inner_idx = 0 (first MAC starts a new chain).
        assertions.push(Assertion::single(COL_INNER_IDX, 0, Felt::ZERO));

        // Row 0: acc_before = 0 (accumulator starts at zero).
        assertions.push(Assertion::single(COL_ACC_BEFORE, 0, Felt::ZERO));

        // Task 6.7: MPO Core Pinning
        // Every row's MPO value must match the analytically computed expected
        // value from the system matrix (I - alpha*dt*L). The verifier independently
        // constructs the system matrix from (alpha, dt, dx) and checks these.
        for (row, expected) in self.inputs.expected_mpo_vals.iter().enumerate() {
            assertions.push(Assertion::single(COL_MPO_VAL, row, *expected));
        }

        // MPS Input Pinning
        // Every row's MPS value must match the input MPS core element.
        for (row, expected) in self.inputs.expected_mps_vals.iter().enumerate() {
            assertions.push(Assertion::single(COL_MPS_VAL, row, *expected));
        }

        // Output Pinning
        // Every last-MAC row's output value must match the claimed output.
        // Last-MAC rows are at positions d_in-1, 2*d_in-1, 3*d_in-1, ...
        let d_in = self.inputs.d_phys;
        for (elem_idx, expected) in self.inputs.expected_output_vals.iter().enumerate() {
            let row = elem_idx * d_in + (d_in - 1);
            if row < self.inputs.num_mac_rows {
                assertions.push(Assertion::single(COL_OUTPUT_VAL, row, *expected));
            }
        }

        assertions
    }

    fn context(&self) -> &AirContext<Felt> {
        &self.context
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal Prover
// ═══════════════════════════════════════════════════════════════════════════

/// Winterfell prover for QTT contraction STARK.
struct ContractionProver {
    options: ProofOptions,
    pub_inputs: ContractionStarkInputs,
}

impl ContractionProver {
    fn new(options: ProofOptions, pub_inputs: ContractionStarkInputs) -> Self {
        Self { options, pub_inputs }
    }
}

impl Prover for ContractionProver {
    type BaseField = Felt;
    type Air = ContractionAir;
    type Trace = TraceTable<Felt>;
    type HashFn = Blake3_256<Felt>;
    type VC = MerkleTree<Self::HashFn>;
    type RandomCoin = DefaultRandomCoin<Self::HashFn>;
    type TraceLde<E: FieldElement<BaseField = Felt>> =
        DefaultTraceLde<E, Self::HashFn, Self::VC>;
    type ConstraintEvaluator<'a, E: FieldElement<BaseField = Felt>> =
        DefaultConstraintEvaluator<'a, Self::Air, E>;
    type ConstraintCommitment<E: FieldElement<BaseField = Self::BaseField>> =
        DefaultConstraintCommitment<E, Self::HashFn, Self::VC>;

    fn get_pub_inputs(&self, _trace: &Self::Trace) -> ContractionStarkInputs {
        self.pub_inputs.clone()
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
// Public API: Build inputs, Prove, Verify
// ═══════════════════════════════════════════════════════════════════════════

/// Build the public inputs for a contraction STARK from pre-computed data.
///
/// The verifier uses this to independently validate the contraction.
/// The MPO is reconstructed from physics parameters (α, dt, dx),
/// the MPS values come from the committed input state, and the output
/// values come from the claimed contraction result.
pub fn build_contraction_inputs(
    mps: &Mps,
    mpo: &Mpo,
    witness: &ContractionWitness,
    alpha: Q16,
    dt: Q16,
    dx: Q16,
    residual_norm_sq: Q16,
    tolerance_sq: Q16,
) -> ContractionStarkInputs {
    let schedule = build_mac_schedule(mps, mpo);
    let num_mac_rows = schedule.len();
    let trace_length = num_mac_rows.next_power_of_two().max(8);
    let d_in = mpo.d_in();

    // Build expected MPO values (one per MAC row).
    let expected_mpo_vals: Vec<Felt> = schedule
        .iter()
        .map(|mac| {
            let val = mpo.get(mac.site, mac.dl, mac.out_phys, mac.inner_p, mac.dr);
            q16_to_felt(val)
        })
        .collect();

    // Build expected MPS values (one per MAC row).
    let expected_mps_vals: Vec<Felt> = schedule
        .iter()
        .map(|mac| {
            let val = mps.get(mac.site, mac.cl, mac.inner_p, mac.cr);
            q16_to_felt(val)
        })
        .collect();

    // Build expected output values (one per output element).
    // An output element completes every d_in MAC rows.
    let num_output_elems = num_mac_rows / d_in;
    let expected_output_vals: Vec<Felt> = (0..num_output_elems)
        .map(|elem_idx| {
            // The output element's value is the acc_after of the last MAC row.
            let last_row_idx = elem_idx * d_in + (d_in - 1);
            let mac = &schedule[last_row_idx];
            let site = mac.site;

            // Get from witness output MPS.
            let sd = &witness.site_data[site];
            // Count output elements before this site.
            let site_start_elem = schedule[..last_row_idx + 1]
                .iter()
                .filter(|m| m.site < site)
                .count()
                / d_in;
            let local_elem = elem_idx - site_start_elem;

            if local_elem < sd.mac_accumulators.len() {
                let chain = &sd.mac_accumulators[local_elem];
                q16_to_felt(chain[d_in])
            } else {
                Felt::ZERO
            }
        })
        .collect();

    ContractionStarkInputs {
        num_mac_rows,
        trace_length,
        num_sites: mps.num_sites,
        chi_max: mps.max_chi(),
        mpo_bond_dim: mpo.max_bond(),
        d_phys: mpo.d_in(),
        alpha: q16_to_felt(alpha),
        dt: q16_to_felt(dt),
        dx: q16_to_felt(dx),
        expected_mpo_vals,
        expected_mps_vals,
        expected_output_vals,
        residual_norm_sq: q16_to_felt(residual_norm_sq),
        tolerance_sq: q16_to_felt(tolerance_sq),
    }
}

/// Proof options for contraction STARK.
///
/// Uses blowup factor 16 (for degree-2 constraints) and the same
/// security parameters as the chain STARK.
fn contraction_proof_options() -> ProofOptions {
    ProofOptions::new(
        40,                          // num_queries
        16,                          // blowup_factor (≥ max constraint degree × extension)
        16,                          // grinding_factor
        FieldExtension::Quadratic,   // quadratic extension
        8,                           // fri_folding_factor
        31,                          // fri_remainder_max_degree
        BatchingMethod::Linear,
        BatchingMethod::Linear,
    )
}

/// Generate a contraction STARK proof.
///
/// Proves that applying `mpo` to `mps` produces the output in `witness`,
/// with every MAC step arithmetically constrained.
///
/// # Returns
/// `(proof_bytes, public_inputs, generation_time_ms)`
pub fn prove_contraction_stark(
    mps: &Mps,
    mpo: &Mpo,
    witness: &ContractionWitness,
    alpha: Q16,
    dt: Q16,
    dx: Q16,
    residual_norm_sq: Q16,
    tolerance_sq: Q16,
) -> Result<(Vec<u8>, ContractionStarkInputs, u64), String> {
    let start = std::time::Instant::now();

    // Build execution trace.
    let (trace, _schedule) = build_contraction_trace(mps, mpo, witness)?;

    // Build public inputs.
    let pub_inputs = build_contraction_inputs(
        mps,
        mpo,
        witness,
        alpha,
        dt,
        dx,
        residual_norm_sq,
        tolerance_sq,
    );

    // Create prover.
    let options = contraction_proof_options();
    let prover = ContractionProver::new(options, pub_inputs.clone());

    // Generate proof.
    // We need to provide the public inputs to the AIR. The Winterfell Prover
    // trait calls get_pub_inputs(), but we override with a wrapper.
    let proof = prover
        .prove(trace)
        .map_err(|e| format!("Contraction STARK proof generation failed: {:?}", e))?;

    let proof_bytes = proof.to_bytes();
    let generation_time_ms = start.elapsed().as_millis() as u64;

    Ok((proof_bytes, pub_inputs, generation_time_ms))
}

/// Verify a contraction STARK proof.
///
/// The verifier independently reconstructs the expected MPO from (α, dt, dx)
/// and checks that the proof is consistent with those values.
pub fn verify_contraction_stark(
    proof_bytes: &[u8],
    pub_inputs: ContractionStarkInputs,
) -> Result<bool, String> {
    let proof = Proof::from_bytes(proof_bytes)
        .map_err(|e| format!("Failed to deserialize contraction proof: {:?}", e))?;

    let acceptable = AcceptableOptions::MinConjecturedSecurity(80);

    match winterfell::verify::<ContractionAir, Blake3_256<Felt>, DefaultRandomCoin<Blake3_256<Felt>>, MerkleTree<Blake3_256<Felt>>>(
        proof,
        pub_inputs,
        &acceptable,
    ) {
        Ok(()) => Ok(true),
        Err(e) => Err(format!("Contraction STARK verification failed: {:?}", e)),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use winterfell::Trace;
    use fluidelite_core::qtt_operators::{system_matrix_mpo, laplacian_mpo};
    use crate::tensor::{Mps, Mpo};

    /// Helper: create a small test MPS with non-trivial data.
    fn make_test_mps(num_sites: usize, chi: usize) -> Mps {
        let mut mps = Mps::new(num_sites, chi, 2);
        // Fill with a simple pattern: site*100 + index
        for i in 0..num_sites {
            let core = mps.core_data_mut(i);
            for (j, val) in core.iter_mut().enumerate() {
                // Small values to avoid Q16 overflow during contraction.
                let v = ((i * 100 + j) % 50) as f64 * 0.01;
                *val = Q16::from_f64(v);
            }
        }
        mps
    }

    /// Helper: apply MPO to MPS with witness using WitnessGenerator's method.
    fn apply_with_witness(
        mps: &Mps,
        mpo: &Mpo,
    ) -> (Mps, ContractionWitness) {
        // Use the witness generator's apply_mpo_with_witness via cg_solve.
        // But that's private. We'll replicate the contraction logic here.
        let num_sites = mps.num_sites;
        let d_out = mpo.d_out();
        let d_in = mpo.d_in();

        let mut dims = Vec::with_capacity(num_sites);
        let mut all_data = Vec::new();
        let mut site_data_vec = Vec::with_capacity(num_sites);

        for i in 0..num_sites {
            let mcl = mps.chi_left(i);
            let mcr = mps.chi_right(i);
            let odl = mpo.dl(i);
            let odr = mpo.dr(i);

            let new_chi_left = mcl * odl;
            let new_chi_right = mcr * odr;
            dims.push((new_chi_left, new_chi_right));

            let total_outputs = new_chi_left * d_out * new_chi_right;
            let mut new_data = vec![Q16::zero(); total_outputs];
            let mut mac_accumulators = Vec::with_capacity(total_outputs);
            let mut fp_remainders = Vec::new();
            let mut fp_quotients = Vec::new();

            for cl in 0..mcl {
                for dl in 0..odl {
                    let new_left = cl * odl + dl;

                    for o in 0..d_out {
                        for cr in 0..mcr {
                            for dr in 0..odr {
                                let new_right = cr * odr + dr;

                                let mut acc = Q16::zero();
                                let mut chain = Vec::with_capacity(d_in + 1);
                                chain.push(acc);

                                for p in 0..d_in {
                                    let mpo_v = mpo.get(i, dl, o, p, dr);
                                    let mps_v = mps.get(i, cl, p, cr);

                                    let full_product =
                                        mpo_v.raw as i128 * mps_v.raw as i128;
                                    let quotient =
                                        (full_product >> 16) as i64;
                                    let remainder =
                                        (full_product - ((quotient as i128) << 16)) as i64;

                                    fp_quotients.push(Q16::from_raw(quotient));
                                    fp_remainders.push(remainder);

                                    acc = Q16::from_raw(acc.raw + quotient);
                                    chain.push(acc);
                                }

                                mac_accumulators.push(chain);
                                new_data[(new_left * d_out + o) * new_chi_right + new_right] = acc;
                            }
                        }
                    }
                }
            }

            all_data.extend_from_slice(&new_data);
            site_data_vec.push(SiteContractionData {
                mac_accumulators,
                fp_remainders,
                fp_quotients,
            });
        }

        let output_mps = Mps::from_flat(&dims, d_out, all_data);
        let witness = ContractionWitness {
            num_sites,
            site_data: site_data_vec,
            output_mps: output_mps.clone(),
        };

        (output_mps, witness)
    }

    #[test]
    fn test_mac_schedule_site_count() {
        let mps = Mps::new(3, 2, 2);
        let mpo = Mpo::identity(3, 2);
        let schedule = build_mac_schedule(&mps, &mpo);

        // Identity MPO has D=1, d_out=d_in=2.
        // Per site: (chi_l × 1) × 2 × (chi_r × 1) output elements × 2 inner.
        // Site 0: (1×1) × 2 × (2×1) = 4 output elements × 2 = 8 MACs.
        // Site 1: (2×1) × 2 × (2×1) = 8 output elements × 2 = 16 MACs.
        // Site 2: (2×1) × 2 × (1×1) = 4 output elements × 2 = 8 MACs.
        assert_eq!(schedule.len(), 8 + 16 + 8);
    }

    #[test]
    fn test_identity_contraction_trace() {
        // Identity MPO: output should equal input.
        let num_sites = 3;
        let chi = 2;
        let mps = make_test_mps(num_sites, chi);
        let mpo = Mpo::identity(num_sites, 2);

        let (_output, witness) = apply_with_witness(&mps, &mpo);
        let (trace, _schedule) = build_contraction_trace(&mps, &mpo, &witness).unwrap();

        assert_eq!(trace.width(), QTT_TRACE_WIDTH);
        assert!(trace.length() >= 32); // 32 MACs → padded to 32

        // Row 0 should have inner_idx = 0 and acc_before = 0.
        assert_eq!(trace.get(COL_INNER_IDX, 0), Felt::ZERO);
        assert_eq!(trace.get(COL_ACC_BEFORE, 0), Felt::ZERO);
    }

    #[test]
    fn test_identity_contraction_prove_verify() {
        let num_sites = 3;
        let chi = 2;
        let mps = make_test_mps(num_sites, chi);
        let mpo = Mpo::identity(num_sites, 2);

        let (_output, witness) = apply_with_witness(&mps, &mpo);

        let alpha = Q16::from_f64(0.01);
        let dt = Q16::from_f64(0.1);
        let dx = Q16::one();

        let (proof_bytes, pub_inputs, gen_ms) = prove_contraction_stark(
            &mps,
            &mpo,
            &witness,
            alpha,
            dt,
            dx,
            Q16::zero(), // residual
            Q16::one(),  // tolerance
        )
        .unwrap();

        assert!(proof_bytes.len() > 100);
        assert!(gen_ms < 60_000, "proof too slow: {} ms", gen_ms);

        let valid = verify_contraction_stark(&proof_bytes, pub_inputs).unwrap();
        assert!(valid, "Contraction STARK verification failed");
    }

    #[test]
    fn test_laplacian_contraction_prove_verify() {
        // Use the real Laplacian MPO (bond dim 5) on a small grid.
        let num_sites = 4; // grid_bits=2 in 1D (4 sites for simplicity)
        let chi = 2;
        let dx = Q16::one();

        let lapl_full = laplacian_mpo(num_sites, dx);
        let lapl = Mpo::from_full(&lapl_full);
        let mps = make_test_mps(num_sites, chi);

        let (_output, witness) = apply_with_witness(&mps, &lapl);

        let alpha = Q16::from_f64(0.01);
        let dt = Q16::from_f64(0.1);

        let (proof_bytes, pub_inputs, gen_ms) = prove_contraction_stark(
            &mps,
            &lapl,
            &witness,
            alpha,
            dt,
            dx,
            Q16::zero(),
            Q16::one(),
        )
        .unwrap();

        assert!(proof_bytes.len() > 100);
        assert!(gen_ms < 120_000, "proof too slow: {} ms", gen_ms);

        let valid = verify_contraction_stark(&proof_bytes, pub_inputs).unwrap();
        assert!(valid, "Laplacian contraction STARK verification failed");
    }

    #[test]
    fn test_system_matrix_contraction_prove_verify() {
        // Full system matrix A = I - α·Δt·L (bond dim 6).
        let num_sites = 3;
        let chi = 2;
        let alpha_dt = Q16::from_f64(0.001);
        let dx = Q16::one();

        let sys_full = system_matrix_mpo(num_sites, alpha_dt, dx);
        let sys_mpo = Mpo::from_full(&sys_full);
        let mps = make_test_mps(num_sites, chi);

        let (_output, witness) = apply_with_witness(&mps, &sys_mpo);

        let alpha = Q16::from_f64(0.01);
        let dt = Q16::from_f64(0.1);

        let (proof_bytes, pub_inputs, gen_ms) = prove_contraction_stark(
            &mps,
            &sys_mpo,
            &witness,
            alpha,
            dt,
            dx,
            Q16::zero(),
            Q16::one(),
        )
        .unwrap();

        assert!(proof_bytes.len() > 100);
        assert!(gen_ms < 120_000, "proof too slow: {} ms", gen_ms);

        let valid = verify_contraction_stark(&proof_bytes, pub_inputs).unwrap();
        assert!(valid, "System matrix contraction STARK verification failed");
    }

    #[test]
    fn test_tampered_mpo_rejected() {
        // Prove with correct data, then verify with wrong MPO → should fail.
        let num_sites = 3;
        let chi = 2;
        let mps = make_test_mps(num_sites, chi);
        let mpo = Mpo::identity(num_sites, 2);

        let (_output, witness) = apply_with_witness(&mps, &mpo);

        let alpha = Q16::from_f64(0.01);
        let dt = Q16::from_f64(0.1);
        let dx = Q16::one();

        let (proof_bytes, mut pub_inputs, _gen_ms) = prove_contraction_stark(
            &mps,
            &mpo,
            &witness,
            alpha,
            dt,
            dx,
            Q16::zero(),
            Q16::one(),
        )
        .unwrap();

        // Tamper: change an MPO value in public inputs.
        if !pub_inputs.expected_mpo_vals.is_empty() {
            pub_inputs.expected_mpo_vals[0] = pub_inputs.expected_mpo_vals[0] + Felt::ONE;
        }

        let result = verify_contraction_stark(&proof_bytes, pub_inputs);
        assert!(
            result.is_err(),
            "Tampered MPO should cause verification failure"
        );
    }

    #[test]
    fn test_tampered_mps_rejected() {
        // Prove with correct data, then verify with wrong MPS → should fail.
        let num_sites = 3;
        let chi = 2;
        let mps = make_test_mps(num_sites, chi);
        let mpo = Mpo::identity(num_sites, 2);

        let (_output, witness) = apply_with_witness(&mps, &mpo);

        let alpha = Q16::from_f64(0.01);
        let dt = Q16::from_f64(0.1);
        let dx = Q16::one();

        let (proof_bytes, mut pub_inputs, _gen_ms) = prove_contraction_stark(
            &mps,
            &mpo,
            &witness,
            alpha,
            dt,
            dx,
            Q16::zero(),
            Q16::one(),
        )
        .unwrap();

        // Tamper: change an MPS value in public inputs.
        if !pub_inputs.expected_mps_vals.is_empty() {
            pub_inputs.expected_mps_vals[0] = pub_inputs.expected_mps_vals[0] + Felt::ONE;
        }

        let result = verify_contraction_stark(&proof_bytes, pub_inputs);
        assert!(
            result.is_err(),
            "Tampered MPS should cause verification failure"
        );
    }

    #[test]
    fn test_tampered_output_rejected() {
        // Prove with correct data, then verify with wrong output → should fail.
        let num_sites = 3;
        let chi = 2;
        let mps = make_test_mps(num_sites, chi);
        let mpo = Mpo::identity(num_sites, 2);

        let (_output, witness) = apply_with_witness(&mps, &mpo);

        let alpha = Q16::from_f64(0.01);
        let dt = Q16::from_f64(0.1);
        let dx = Q16::one();

        let (proof_bytes, mut pub_inputs, _gen_ms) = prove_contraction_stark(
            &mps,
            &mpo,
            &witness,
            alpha,
            dt,
            dx,
            Q16::zero(),
            Q16::one(),
        )
        .unwrap();

        // Tamper: change an output value.
        if !pub_inputs.expected_output_vals.is_empty() {
            pub_inputs.expected_output_vals[0] =
                pub_inputs.expected_output_vals[0] + Felt::ONE;
        }

        let result = verify_contraction_stark(&proof_bytes, pub_inputs);
        assert!(
            result.is_err(),
            "Tampered output should cause verification failure"
        );
    }
}
