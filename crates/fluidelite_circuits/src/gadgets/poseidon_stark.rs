//! Poseidon Algebraic Hash over Goldilocks Field (STARK AIR).
//!
//! Implements the Poseidon hash function as an algebraically constrained
//! STARK execution trace, enabling **in-circuit** hash verification where
//! the digest is a computed value rather than a free witness.
//!
//! # Parameters
//!
//! | Parameter  | Value | Description                           |
//! |------------|-------|---------------------------------------|
//! | Field      | Goldilocks | p = 2^64 - 2^32 + 1             |
//! | Width (t)  | 12    | Sponge state width                    |
//! | Rate       | 8     | Elements absorbed per permutation      |
//! | Capacity   | 4     | Security elements (state\[0..4\])     |
//! | Digest     | 4     | Output elements from rate\[0..4\]     |
//! | S-box (a)  | 7     | x^7 (gcd(7, p-1) = 1)                |
//! | R_F        | 8     | Full rounds (4 before + 4 after)      |
//! | R_P        | 22    | Partial rounds (S-box on element 0)   |
//! | MDS        | 12x12 | Circulant \[7,23,8,26,13,10,9,7,6,22,21,8\] |
//!
//! # Trace Layout
//!
//! | Row  | Content                          |
//! |------|----------------------------------|
//! | 0    | Input state (pre-round 0)        |
//! | 1-30 | State after rounds 0-29          |
//! | 31   | Padding (copy of row 30)         |
//!
//! 12 columns (one per state element), 32 rows (power of 2).
//!
//! # Periodic Columns (13 total)
//!
//! | Index | Name      | Description                              |
//! |-------|-----------|------------------------------------------|
//! | 0     | is_full   | 1 for full rounds, 0 for partial rounds  |
//! | 1-12  | RC\[0-11\]| Round constants per state element        |
//!
//! # Transition Constraints (12, all degree 7)
//!
//! For each output element i:
//! ```text
//! next[i] = SUM_j( MDS[i][j] * eff_sbox[j] )
//! ```
//! where:
//! - `eff_sbox[0] = (state[0] + RC[0])^7` (always S-boxed)
//! - `eff_sbox[j>0] = inp[j] + is_full * (inp[j]^7 - inp[j])` (conditional)
//! - `inp[j] = state[j] + RC[j]`
//!
//! # Boundary Assertions
//!
//! - Row 0: 12 input state assertions
//! - Row 30: 12 output state assertions
//!
//! # Security
//!
//! - Constraint degree 7 with one periodic cycle (is_full, period 32)
//! - Minimum blowup factor: 8 (using 16 for margin)
//! - 40-query FRI, quadratic extension, grinding 16 => 127-bit security
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use std::sync::OnceLock;

use winterfell::{
    math::{fields::f64::BaseElement, FieldElement, ToElements},
    Air, AirContext, Assertion, AuxRandElements, BatchingMethod,
    CompositionPoly, CompositionPolyTrace, ConstraintCompositionCoefficients,
    DefaultConstraintCommitment, DefaultConstraintEvaluator, DefaultTraceLde,
    EvaluationFrame, FieldExtension, PartitionOptions, Proof, ProofOptions,
    Prover, StarkDomain, TraceInfo, TraceTable,
    TracePolyTable, TransitionConstraintDegree, AcceptableOptions,
    crypto::{hashers::Blake3_256, DefaultRandomCoin, MerkleTree},
    matrix::ColMatrix,
};

use sha2::{Sha256, Digest as Sha2Digest};

// ═══════════════════════════════════════════════════════════════════════════
// Type Aliases
// ═══════════════════════════════════════════════════════════════════════════

/// Goldilocks field element (p = 2^64 - 2^32 + 1).
pub type Felt = BaseElement;

/// Blake3 hasher over Goldilocks.
type H = Blake3_256<Felt>;

/// Merkle tree commitment.
type VC = MerkleTree<H>;

/// Goldilocks modulus.
const GOLDILOCKS_P: u64 = 0xFFFF_FFFF_0000_0001;

// ═══════════════════════════════════════════════════════════════════════════
// Poseidon Parameters
// ═══════════════════════════════════════════════════════════════════════════

/// Sponge state width (field elements).
pub const POSEIDON_WIDTH: usize = 12;

/// Rate: number of elements absorbed per permutation.
pub const POSEIDON_RATE: usize = 8;

/// Capacity: security portion of the state (indices 0..4).
pub const POSEIDON_CAPACITY: usize = 4;

/// Digest size: 4 field elements (256 bits over Goldilocks).
pub const POSEIDON_DIGEST_SIZE: usize = 4;

/// Number of full rounds (4 before + 4 after partial rounds).
pub const POSEIDON_R_F: usize = 8;

/// Number of partial rounds (S-box applied only to element 0).
pub const POSEIDON_R_P: usize = 22;

/// Total number of rounds.
pub const POSEIDON_NUM_ROUNDS: usize = POSEIDON_R_F + POSEIDON_R_P;

/// S-box exponent (x -> x^7).
pub const POSEIDON_ALPHA: u64 = 7;

// ═══════════════════════════════════════════════════════════════════════════
// Trace Parameters
// ═══════════════════════════════════════════════════════════════════════════

/// Trace width: one column per state element.
const POSEIDON_TRACE_WIDTH: usize = POSEIDON_WIDTH;

/// Trace length: next power of 2 above NUM_ROUNDS + 1 = 31.
const POSEIDON_TRACE_LEN: usize = 32;

/// Number of AIR transition constraints (one per output element).
const POSEIDON_NUM_CONSTRAINTS: usize = POSEIDON_WIDTH;

/// Number of boundary assertions: 12 input + 12 output.
const POSEIDON_NUM_ASSERTIONS: usize = 2 * POSEIDON_WIDTH;

// ═══════════════════════════════════════════════════════════════════════════
// MDS Matrix
// ═══════════════════════════════════════════════════════════════════════════

/// First row of the 12x12 circulant MDS matrix.
///
/// This is the same matrix used by Winterfell (Polygon Zero/Meta) and is
/// proven MDS over the Goldilocks field. The circulant structure gives:
/// `M[i][j] = FIRST_ROW[(j - i) mod 12]`.
const MDS_FIRST_ROW: [u64; POSEIDON_WIDTH] = [7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8];

/// Compute the full 12x12 MDS matrix from the circulant first row.
fn compute_mds_matrix() -> [[Felt; POSEIDON_WIDTH]; POSEIDON_WIDTH] {
    let mut m = [[Felt::ZERO; POSEIDON_WIDTH]; POSEIDON_WIDTH];
    for i in 0..POSEIDON_WIDTH {
        for j in 0..POSEIDON_WIDTH {
            let idx = (j + POSEIDON_WIDTH - i) % POSEIDON_WIDTH;
            m[i][j] = Felt::new(MDS_FIRST_ROW[idx]);
        }
    }
    m
}

/// Cached MDS matrix (computed once on first access).
fn mds_matrix() -> &'static [[Felt; POSEIDON_WIDTH]; POSEIDON_WIDTH] {
    static MDS: OnceLock<[[Felt; POSEIDON_WIDTH]; POSEIDON_WIDTH]> = OnceLock::new();
    MDS.get_or_init(compute_mds_matrix)
}

// ═══════════════════════════════════════════════════════════════════════════
// Round Constants
// ═══════════════════════════════════════════════════════════════════════════

/// Generate Poseidon round constants deterministically.
///
/// Uses SHA-256 with domain separator
/// `"Ontic_Poseidon_Goldilocks_t12_RF8_RP22_alpha7"` followed by
/// the round index and element index (both as little-endian u64).
/// The first 8 bytes of each hash are reduced mod p.
///
/// This is a Nothing-Up-My-Sleeve (NUMS) construction: the constants are
/// fully determined by the parameter string and can be independently
/// verified by any party.
fn generate_round_constants() -> Vec<[Felt; POSEIDON_WIDTH]> {
    let domain = b"Ontic_Poseidon_Goldilocks_t12_RF8_RP22_alpha7";
    let mut constants = Vec::with_capacity(POSEIDON_NUM_ROUNDS);
    for round in 0..POSEIDON_NUM_ROUNDS {
        let mut row = [Felt::ZERO; POSEIDON_WIDTH];
        for elem in 0..POSEIDON_WIDTH {
            let mut hasher = Sha256::new();
            hasher.update(domain);
            hasher.update((round as u64).to_le_bytes());
            hasher.update((elem as u64).to_le_bytes());
            let hash = hasher.finalize();
            let raw = u64::from_le_bytes(hash[0..8].try_into().unwrap());
            row[elem] = Felt::new(raw % GOLDILOCKS_P);
        }
        constants.push(row);
    }
    constants
}

/// Cached round constants (computed once on first access).
fn round_constants() -> &'static Vec<[Felt; POSEIDON_WIDTH]> {
    static RC: OnceLock<Vec<[Felt; POSEIDON_WIDTH]>> = OnceLock::new();
    RC.get_or_init(generate_round_constants)
}

// ═══════════════════════════════════════════════════════════════════════════
// S-Box and Permutation
// ═══════════════════════════════════════════════════════════════════════════

/// Compute x^7 over Felt via addition chain: x^2 -> x^4 -> x^3 -> x^7.
///
/// Uses 4 multiplications (optimal for exponent 7).
#[inline]
fn sbox_felt(x: Felt) -> Felt {
    let x2 = x * x;
    let x4 = x2 * x2;
    let x3 = x2 * x;
    x4 * x3
}

/// Generic S-box for constraint evaluation over extension fields.
#[inline]
fn sbox_generic<E: FieldElement>(x: E) -> E {
    let x2 = x * x;
    let x4 = x2 * x2;
    let x3 = x2 * x;
    x4 * x3
}

/// Determine if a round index corresponds to a full round.
///
/// Full rounds: 0..R_F/2 and (R_F/2 + R_P)..NUM_ROUNDS.
/// Partial rounds: R_F/2..(R_F/2 + R_P).
#[inline]
fn is_full_round(round: usize) -> bool {
    round < POSEIDON_R_F / 2 || round >= POSEIDON_R_F / 2 + POSEIDON_R_P
}

/// Apply one Poseidon permutation in-place (out-of-circuit reference).
///
/// This is the canonical reference implementation used for:
/// - Computing hash digests out-of-circuit
/// - Generating execution trace data
/// - Cross-validating the AIR constraints
pub fn poseidon_permutation(state: &mut [Felt; POSEIDON_WIDTH]) {
    let rcs = round_constants();
    let mds = mds_matrix();

    for round in 0..POSEIDON_NUM_ROUNDS {
        // Step 1: Add round constants (ARK)
        for i in 0..POSEIDON_WIDTH {
            state[i] += rcs[round][i];
        }

        // Step 2: S-box (x -> x^7)
        if is_full_round(round) {
            for i in 0..POSEIDON_WIDTH {
                state[i] = sbox_felt(state[i]);
            }
        } else {
            // Partial round: S-box only on element 0
            state[0] = sbox_felt(state[0]);
        }

        // Step 3: MDS matrix multiply
        let input = *state;
        for i in 0..POSEIDON_WIDTH {
            state[i] = Felt::ZERO;
            for j in 0..POSEIDON_WIDTH {
                state[i] += mds[i][j] * input[j];
            }
        }
    }
}

/// Poseidon sponge hash: absorb field elements, squeeze 4-element digest.
///
/// # Sponge construction
///
/// - **Capacity** (state\[0..4\]): state\[0\] initialized to input length
/// - **Rate** (state\[4..12\]): receives input elements via addition
/// - **Digest**: state\[4..8\] after final permutation
///
/// Absorption proceeds in chunks of RATE (= 8) elements. Each chunk is
/// added to the rate portion, then a full permutation is applied.
pub fn poseidon_hash(elements: &[Felt]) -> [Felt; POSEIDON_DIGEST_SIZE] {
    let mut state = [Felt::ZERO; POSEIDON_WIDTH];

    // Domain separation: capacity[0] = number of elements
    state[0] = Felt::new(elements.len() as u64);

    if elements.is_empty() {
        poseidon_permutation(&mut state);
    } else {
        for chunk in elements.chunks(POSEIDON_RATE) {
            for (j, &elem) in chunk.iter().enumerate() {
                state[POSEIDON_CAPACITY + j] += elem;
            }
            poseidon_permutation(&mut state);
        }
    }

    // Squeeze: read digest from rate[0..4] = state[4..8]
    let mut digest = [Felt::ZERO; POSEIDON_DIGEST_SIZE];
    digest.copy_from_slice(&state[POSEIDON_CAPACITY..POSEIDON_CAPACITY + POSEIDON_DIGEST_SIZE]);
    digest
}

// ═══════════════════════════════════════════════════════════════════════════
// Public Inputs
// ═══════════════════════════════════════════════════════════════════════════

/// Public inputs for a single Poseidon permutation STARK.
///
/// The verifier checks that applying the Poseidon permutation to
/// `input_state` produces `output_state`.
#[derive(Clone, Debug)]
pub struct PoseidonPublicInputs {
    /// State before the permutation (12 field elements).
    pub input_state: [Felt; POSEIDON_WIDTH],
    /// State after the permutation (12 field elements).
    pub output_state: [Felt; POSEIDON_WIDTH],
}

impl ToElements<Felt> for PoseidonPublicInputs {
    fn to_elements(&self) -> Vec<Felt> {
        let mut v = Vec::with_capacity(2 * POSEIDON_WIDTH);
        v.extend_from_slice(&self.input_state);
        v.extend_from_slice(&self.output_state);
        v
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// AIR Definition
// ═══════════════════════════════════════════════════════════════════════════

/// Algebraic Intermediate Representation for Poseidon permutation.
///
/// Encodes 30 rounds of the Poseidon permutation as 12 degree-7 transition
/// constraints evaluated over a 32-row execution trace. Round constants and
/// round type (full/partial) are injected as periodic columns.
pub struct PoseidonAir {
    context: AirContext<Felt>,
    pub_inputs: PoseidonPublicInputs,
    mds: [[Felt; POSEIDON_WIDTH]; POSEIDON_WIDTH],
}

impl Air for PoseidonAir {
    type BaseField = Felt;
    type PublicInputs = PoseidonPublicInputs;

    fn new(
        trace_info: TraceInfo,
        pub_inputs: Self::PublicInputs,
        options: ProofOptions,
    ) -> Self {
        // All 12 constraints are degree 7 with one periodic cycle (is_full).
        let degrees = vec![
            TransitionConstraintDegree::with_cycles(7, vec![POSEIDON_TRACE_LEN]);
            POSEIDON_NUM_CONSTRAINTS
        ];

        // Exempt 2 transitions: the minimum 1 (to avoid OOB) plus 1 for the
        // padding row. This means steps 0..29 are checked (all 30 real
        // Poseidon rounds), and step 30 (row 30 → padding row 31) is exempt.
        // Output correctness at row 30 is guaranteed by assertions.
        let context = AirContext::new(
            trace_info,
            degrees,
            POSEIDON_NUM_ASSERTIONS,
            options,
        )
        .set_num_transition_exemptions(2);

        let mds = *mds_matrix();

        Self { context, pub_inputs, mds }
    }

    fn context(&self) -> &AirContext<Self::BaseField> {
        &self.context
    }

    fn evaluate_transition<E: FieldElement<BaseField = Felt>>(
        &self,
        frame: &EvaluationFrame<E>,
        periodic_values: &[E],
        result: &mut [E],
    ) {
        let current = frame.current();
        let next = frame.next();

        // Periodic values layout: [is_full, RC[0], RC[1], ..., RC[11]]
        let is_full = periodic_values[0];

        // Step 1: Compute S-box inputs (state + round constants)
        let mut sbox_in = [E::ZERO; POSEIDON_WIDTH];
        for j in 0..POSEIDON_WIDTH {
            sbox_in[j] = current[j] + periodic_values[1 + j];
        }

        // Step 2: Compute effective S-box outputs
        //   Element 0: always S-boxed (both full and partial rounds)
        //   Elements 1-11: S-boxed in full rounds, identity in partial rounds
        let mut sbox_out = [E::ZERO; POSEIDON_WIDTH];
        sbox_out[0] = sbox_generic(sbox_in[0]);
        for j in 1..POSEIDON_WIDTH {
            let full_sbox = sbox_generic(sbox_in[j]);
            // eff = input + is_full * (sbox(input) - input)
            //     = identity when is_full=0, sbox when is_full=1
            sbox_out[j] = sbox_in[j] + is_full * (full_sbox - sbox_in[j]);
        }

        // Step 3: MDS multiply and constrain
        for i in 0..POSEIDON_WIDTH {
            let mut expected = E::ZERO;
            for j in 0..POSEIDON_WIDTH {
                expected += E::from(self.mds[i][j]) * sbox_out[j];
            }
            result[i] = next[i] - expected;
        }
    }

    fn get_assertions(&self) -> Vec<Assertion<Felt>> {
        let mut assertions = Vec::with_capacity(POSEIDON_NUM_ASSERTIONS);

        // Input state at row 0
        for col in 0..POSEIDON_WIDTH {
            assertions.push(Assertion::single(col, 0, self.pub_inputs.input_state[col]));
        }

        // Output state at row POSEIDON_NUM_ROUNDS (= 30)
        for col in 0..POSEIDON_WIDTH {
            assertions.push(Assertion::single(
                col,
                POSEIDON_NUM_ROUNDS,
                self.pub_inputs.output_state[col],
            ));
        }

        assertions
    }

    fn get_periodic_column_values(&self) -> Vec<Vec<Felt>> {
        let rcs = round_constants();
        let mut columns = Vec::with_capacity(1 + POSEIDON_WIDTH);

        // Column 0: is_full (1 for full rounds, 0 for partial rounds)
        let mut is_full_col = vec![Felt::ZERO; POSEIDON_TRACE_LEN];
        for k in 0..POSEIDON_TRACE_LEN {
            if k < POSEIDON_NUM_ROUNDS && is_full_round(k) {
                is_full_col[k] = Felt::ONE;
            }
        }
        columns.push(is_full_col);

        // Columns 1-12: round constants (one column per state element)
        for elem in 0..POSEIDON_WIDTH {
            let mut col = vec![Felt::ZERO; POSEIDON_TRACE_LEN];
            for round in 0..POSEIDON_NUM_ROUNDS {
                col[round] = rcs[round][elem];
            }
            // Rows 30-31: zero (padding, exempt from constraints)
            columns.push(col);
        }

        columns
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Trace Builder
// ═══════════════════════════════════════════════════════════════════════════

/// Build the execution trace for a single Poseidon permutation.
///
/// Returns `(trace, output_state)` where `output_state` is the permutation
/// result (state at row 30).
///
/// The trace has 12 columns and 32 rows:
/// - Row 0: input state
/// - Rows 1-30: state after each of the 30 rounds
/// - Row 31: padding (copy of row 30)
pub fn build_poseidon_trace(
    input_state: [Felt; POSEIDON_WIDTH],
) -> (TraceTable<Felt>, [Felt; POSEIDON_WIDTH]) {
    let rcs = round_constants();
    let mds = mds_matrix();

    // Pre-compute all 32 row states
    let mut rows = [[Felt::ZERO; POSEIDON_WIDTH]; POSEIDON_TRACE_LEN];
    rows[0] = input_state;

    for round in 0..POSEIDON_NUM_ROUNDS {
        let mut s = rows[round];

        // ARK: add round constants
        for i in 0..POSEIDON_WIDTH {
            s[i] += rcs[round][i];
        }

        // S-box
        if is_full_round(round) {
            for i in 0..POSEIDON_WIDTH {
                s[i] = sbox_felt(s[i]);
            }
        } else {
            s[0] = sbox_felt(s[0]);
        }

        // MDS matrix multiply
        let pre_mds = s;
        for i in 0..POSEIDON_WIDTH {
            s[i] = Felt::ZERO;
            for j in 0..POSEIDON_WIDTH {
                s[i] += mds[i][j] * pre_mds[j];
            }
        }

        rows[round + 1] = s;
    }

    // Padding: row 31 = copy of row 30 (sentinel for exempt transition)
    rows[POSEIDON_NUM_ROUNDS + 1] = rows[POSEIDON_NUM_ROUNDS];

    let output_state = rows[POSEIDON_NUM_ROUNDS];

    // Build TraceTable
    let mut trace = TraceTable::new(POSEIDON_TRACE_WIDTH, POSEIDON_TRACE_LEN);
    trace.fill(
        |state| {
            for c in 0..POSEIDON_TRACE_WIDTH {
                state[c] = rows[0][c];
            }
        },
        |step, next| {
            let row_idx = step + 1;
            for c in 0..POSEIDON_TRACE_WIDTH {
                next[c] = rows[row_idx][c];
            }
        },
    );

    (trace, output_state)
}

// ═══════════════════════════════════════════════════════════════════════════
// Prover
// ═══════════════════════════════════════════════════════════════════════════

/// Internal Winterfell prover for Poseidon permutation.
struct PoseidonProver {
    options: ProofOptions,
    pub_inputs: PoseidonPublicInputs,
}

impl PoseidonProver {
    fn new(options: ProofOptions, pub_inputs: PoseidonPublicInputs) -> Self {
        Self { options, pub_inputs }
    }
}

impl Prover for PoseidonProver {
    type BaseField = Felt;
    type Air = PoseidonAir;
    type Trace = TraceTable<Felt>;
    type HashFn = H;
    type VC = VC;
    type RandomCoin = DefaultRandomCoin<H>;
    type TraceLde<E: FieldElement<BaseField = Felt>> = DefaultTraceLde<E, H, VC>;
    type ConstraintEvaluator<'a, E: FieldElement<BaseField = Felt>> =
        DefaultConstraintEvaluator<'a, PoseidonAir, E>;
    type ConstraintCommitment<E: FieldElement<BaseField = Felt>> =
        DefaultConstraintCommitment<E, H, VC>;

    fn get_pub_inputs(&self, _trace: &Self::Trace) -> PoseidonPublicInputs {
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
// Proof Options
// ═══════════════════════════════════════════════════════════════════════════

/// Proof options for Poseidon permutation STARK.
///
/// Blowup factor 16 (minimum 8 for degree-7 constraints + 1 periodic cycle).
/// 40 queries, quadratic extension, grinding 16 => 127-bit security.
fn poseidon_proof_options() -> ProofOptions {
    ProofOptions::new(
        40,                        // num_queries
        16,                        // blowup_factor (>= 8 required for degree 7)
        16,                        // grinding_factor
        FieldExtension::Quadratic, // quadratic extension
        8,                         // FRI folding factor
        31,                        // FRI remainder max degree
        BatchingMethod::Linear,    // constraint batching
        BatchingMethod::Linear,    // DEEP poly batching
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════════

/// Prove a single Poseidon permutation.
///
/// Generates a STARK proof that applying the Poseidon permutation to
/// `input_state` produces the claimed output. The output is computed
/// internally and returned as part of the public inputs.
///
/// # Returns
/// `(proof, public_inputs)` on success.
pub fn prove_poseidon(
    input_state: [Felt; POSEIDON_WIDTH],
) -> Result<(Proof, PoseidonPublicInputs), String> {
    let (trace, output_state) = build_poseidon_trace(input_state);
    let pub_inputs = PoseidonPublicInputs { input_state, output_state };
    let prover = PoseidonProver::new(poseidon_proof_options(), pub_inputs.clone());
    let proof = prover
        .prove(trace)
        .map_err(|e| format!("Poseidon proof generation failed: {:?}", e))?;
    Ok((proof, pub_inputs))
}

/// Verify a Poseidon permutation STARK proof.
///
/// Checks that the proof is valid for the given public inputs (input state
/// and claimed output state).
pub fn verify_poseidon(
    proof: &Proof,
    pub_inputs: &PoseidonPublicInputs,
) -> Result<(), String> {
    let acceptable = AcceptableOptions::OptionSet(vec![poseidon_proof_options()]);
    winterfell::verify::<PoseidonAir, H, DefaultRandomCoin<H>, VC>(
        proof.clone(),
        pub_inputs.clone(),
        &acceptable,
    )
    .map_err(|e| format!("Poseidon verification failed: {:?}", e))
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use winterfell::Trace;

    // ─── Reference Implementation Tests ──────────────────────────────────

    #[test]
    fn test_poseidon_permutation_deterministic() {
        let mut s1 = [Felt::ZERO; POSEIDON_WIDTH];
        let mut s2 = [Felt::ZERO; POSEIDON_WIDTH];
        poseidon_permutation(&mut s1);
        poseidon_permutation(&mut s2);
        assert_eq!(s1, s2, "Permutation must be deterministic");
    }

    #[test]
    fn test_poseidon_permutation_nonzero_output() {
        let mut state = [Felt::ZERO; POSEIDON_WIDTH];
        poseidon_permutation(&mut state);
        let any_nonzero = state.iter().any(|&x| x != Felt::ZERO);
        assert!(any_nonzero, "Output of zero input must not be all-zero");
    }

    #[test]
    fn test_poseidon_permutation_input_sensitivity() {
        let mut s1 = [Felt::ONE; POSEIDON_WIDTH];
        let mut s2 = [Felt::ONE; POSEIDON_WIDTH];
        s2[0] = Felt::new(2);
        poseidon_permutation(&mut s1);
        poseidon_permutation(&mut s2);
        assert_ne!(s1, s2, "Different inputs must produce different outputs");
    }

    #[test]
    fn test_sbox_correctness() {
        // 0^7 = 0
        assert_eq!(sbox_felt(Felt::ZERO), Felt::ZERO);
        // 1^7 = 1
        assert_eq!(sbox_felt(Felt::ONE), Felt::ONE);
        // 2^7 = 128
        assert_eq!(sbox_felt(Felt::new(2)), Felt::new(128));
        // 3^7 = 2187
        assert_eq!(sbox_felt(Felt::new(3)), Felt::new(2187));
        // 10^7 = 10_000_000
        assert_eq!(sbox_felt(Felt::new(10)), Felt::new(10_000_000));
    }

    #[test]
    fn test_mds_is_circulant() {
        let mds = mds_matrix();
        for i in 1..POSEIDON_WIDTH {
            for j in 0..POSEIDON_WIDTH {
                let expected_idx = (j + POSEIDON_WIDTH - i) % POSEIDON_WIDTH;
                assert_eq!(
                    mds[i][j],
                    Felt::new(MDS_FIRST_ROW[expected_idx]),
                    "MDS[{i}][{j}] mismatch"
                );
            }
        }
    }

    #[test]
    fn test_mds_nonsingular() {
        // A simple sanity check: applying MDS to an all-ones vector
        // should produce a non-zero result different from the input.
        let mds = mds_matrix();
        let input = [Felt::ONE; POSEIDON_WIDTH];
        let mut output = [Felt::ZERO; POSEIDON_WIDTH];
        for i in 0..POSEIDON_WIDTH {
            for j in 0..POSEIDON_WIDTH {
                output[i] += mds[i][j] * input[j];
            }
        }
        // Sum of first row = 7+23+8+26+13+10+9+7+6+22+21+8 = 160
        // Every output element should equal 160 (MDS * ones = row_sum * ones for circulant)
        let expected_sum: u64 = MDS_FIRST_ROW.iter().sum();
        for i in 0..POSEIDON_WIDTH {
            assert_eq!(output[i], Felt::new(expected_sum), "MDS row {i} sum mismatch");
        }
    }

    #[test]
    fn test_round_constants_deterministic() {
        let rc1 = round_constants();
        let rc2 = generate_round_constants();
        assert_eq!(rc1.len(), POSEIDON_NUM_ROUNDS);
        assert_eq!(rc2.len(), POSEIDON_NUM_ROUNDS);
        for r in 0..POSEIDON_NUM_ROUNDS {
            for e in 0..POSEIDON_WIDTH {
                assert_eq!(rc1[r][e], rc2[r][e], "RC[{r}][{e}] mismatch");
            }
        }
    }

    #[test]
    fn test_round_constants_nonzero() {
        let rcs = round_constants();
        for r in 0..POSEIDON_NUM_ROUNDS {
            let any_nonzero = rcs[r].iter().any(|&x| x != Felt::ZERO);
            assert!(any_nonzero, "Round {r} constants must not be all-zero");
        }
    }

    // ─── Sponge Hash Tests ───────────────────────────────────────────────

    #[test]
    fn test_poseidon_hash_empty() {
        let d = poseidon_hash(&[]);
        let any_nonzero = d.iter().any(|&x| x != Felt::ZERO);
        assert!(any_nonzero, "Hash of empty must be non-trivial");
    }

    #[test]
    fn test_poseidon_hash_different_inputs() {
        let d1 = poseidon_hash(&[Felt::new(1)]);
        let d2 = poseidon_hash(&[Felt::new(2)]);
        assert_ne!(d1, d2, "Different inputs must produce different digests");
    }

    #[test]
    fn test_poseidon_hash_length_dependent() {
        // [0] vs [0, 0]: different lengths, different domain separation
        let d1 = poseidon_hash(&[Felt::ZERO]);
        let d2 = poseidon_hash(&[Felt::ZERO, Felt::ZERO]);
        assert_ne!(d1, d2, "Different lengths must produce different digests");
    }

    #[test]
    fn test_poseidon_hash_multi_block() {
        // Hash more than RATE elements (forces multiple permutations)
        let elements: Vec<Felt> = (0..20u64).map(|i| Felt::new(i + 1)).collect();
        let d = poseidon_hash(&elements);
        let any_nonzero = d.iter().any(|&x| x != Felt::ZERO);
        assert!(any_nonzero, "Multi-block hash must be non-trivial");

        // Must differ from single-block hash of first 8 elements
        let d_short = poseidon_hash(&elements[..8]);
        assert_ne!(d, d_short, "Multi-block must differ from single-block");
    }

    #[test]
    fn test_poseidon_hash_exact_rate_boundary() {
        // Exactly RATE elements: single permutation
        let elems: Vec<Felt> = (0..8u64).map(|i| Felt::new(i)).collect();
        let d = poseidon_hash(&elems);
        // RATE+1 elements: two permutations
        let mut elems2 = elems.clone();
        elems2.push(Felt::new(8));
        let d2 = poseidon_hash(&elems2);
        assert_ne!(d, d2, "Rate boundary must change digest");
    }

    // ─── Trace Construction Tests ────────────────────────────────────────

    #[test]
    fn test_trace_matches_reference() {
        let input = [Felt::new(42); POSEIDON_WIDTH];
        let (trace, output) = build_poseidon_trace(input);

        // Reference computation
        let mut ref_state = input;
        poseidon_permutation(&mut ref_state);

        assert_eq!(output, ref_state, "Trace output must match reference permutation");

        // Verify row 0 = input
        for col in 0..POSEIDON_WIDTH {
            assert_eq!(trace.get(col, 0), input[col], "Row 0, col {col} mismatch");
        }

        // Verify row 30 = output
        for col in 0..POSEIDON_WIDTH {
            assert_eq!(
                trace.get(col, POSEIDON_NUM_ROUNDS),
                output[col],
                "Row 30, col {col} mismatch"
            );
        }

        // Verify padding: row 31 = row 30
        for col in 0..POSEIDON_WIDTH {
            assert_eq!(
                trace.get(col, POSEIDON_NUM_ROUNDS + 1),
                trace.get(col, POSEIDON_NUM_ROUNDS),
                "Padding row 31, col {col} must equal row 30"
            );
        }
    }

    #[test]
    fn test_trace_dimensions() {
        let input = [Felt::ZERO; POSEIDON_WIDTH];
        let (trace, _) = build_poseidon_trace(input);
        assert_eq!(trace.width(), POSEIDON_TRACE_WIDTH, "Trace width mismatch");
        assert_eq!(trace.length(), POSEIDON_TRACE_LEN, "Trace length mismatch");
    }

    // ─── STARK Prove/Verify Tests ────────────────────────────────────────

    #[test]
    fn test_poseidon_prove_verify_zero_input() {
        let input = [Felt::ZERO; POSEIDON_WIDTH];
        let (proof, pub_inputs) = prove_poseidon(input).expect("Proving failed");
        verify_poseidon(&proof, &pub_inputs).expect("Verification failed");
    }

    #[test]
    fn test_poseidon_prove_verify_nonzero_input() {
        let input: [Felt; POSEIDON_WIDTH] = core::array::from_fn(|i| Felt::new((i + 1) as u64));
        let (proof, pub_inputs) = prove_poseidon(input).expect("Proving failed");
        verify_poseidon(&proof, &pub_inputs).expect("Verification failed");
    }

    #[test]
    fn test_poseidon_tampered_output_rejected() {
        let input = [Felt::new(7); POSEIDON_WIDTH];
        let (proof, mut pub_inputs) = prove_poseidon(input).expect("Proving failed");

        // Tamper: modify one output element
        pub_inputs.output_state[0] += Felt::ONE;

        assert!(
            verify_poseidon(&proof, &pub_inputs).is_err(),
            "Tampered output must be rejected"
        );
    }

    #[test]
    fn test_poseidon_tampered_input_rejected() {
        let input = [Felt::new(7); POSEIDON_WIDTH];
        let (proof, mut pub_inputs) = prove_poseidon(input).expect("Proving failed");

        // Tamper: modify the claimed input
        pub_inputs.input_state[0] += Felt::ONE;

        assert!(
            verify_poseidon(&proof, &pub_inputs).is_err(),
            "Tampered input must be rejected"
        );
    }

    // ─── Round Structure Tests ───────────────────────────────────────────

    #[test]
    fn test_round_type_layout() {
        // Rounds 0-3: full, 4-25: partial, 26-29: full
        for r in 0..POSEIDON_NUM_ROUNDS {
            let expected = r < 4 || r >= 26;
            assert_eq!(
                is_full_round(r), expected,
                "Round {r}: expected is_full={expected}"
            );
        }
    }

    #[test]
    fn test_periodic_column_dimensions() {
        let input = [Felt::ZERO; POSEIDON_WIDTH];
        let trace_info = TraceInfo::new(POSEIDON_TRACE_WIDTH, POSEIDON_TRACE_LEN);
        let pub_inputs = PoseidonPublicInputs {
            input_state: input,
            output_state: input,
        };
        let air = PoseidonAir::new(trace_info, pub_inputs, poseidon_proof_options());
        let periodic = air.get_periodic_column_values();

        // 1 (is_full) + 12 (RCs) = 13
        assert_eq!(periodic.len(), 1 + POSEIDON_WIDTH, "Expected 13 periodic columns");

        // Each column has POSEIDON_TRACE_LEN values
        for (i, col) in periodic.iter().enumerate() {
            assert_eq!(
                col.len(), POSEIDON_TRACE_LEN,
                "Periodic column {i} length mismatch"
            );
        }

        // Verify is_full column matches round type
        let is_full_col = &periodic[0];
        for k in 0..POSEIDON_NUM_ROUNDS {
            let expected = if is_full_round(k) { Felt::ONE } else { Felt::ZERO };
            assert_eq!(is_full_col[k], expected, "is_full[{k}] mismatch");
        }
    }
}
