//! GPU-accelerated Trace Low-Degree Extension for STARK proving.
//!
//! Replaces Winterfell's `DefaultTraceLde` with an ICICLE-backed implementation
//! that offloads NTT/iNTT to GPU via the Goldilocks field backend.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────┐
//! │              GpuTraceLde::new()                      │
//! ├──────────────────────────────────────────────────────┤
//! │  1. Extract columns from ColMatrix<Felt>             │
//! │  2. Convert Felt → ICICLE ScalarField (u64 → limbs) │
//! │  3. GPU iNTT: evaluations → polynomial coefficients  │
//! │  4. GPU NTT (coset): coefficients → LDE evaluations  │
//! │  5. Convert back ScalarField → Felt                  │
//! │  6. Build ColMatrix<Felt> for polys + LDE            │
//! │  7. CPU Merkle: hash LDE rows → MerkleTree           │
//! └──────────────────────────────────────────────────────┘
//! ```
//!
//! # Findings-Informed Design
//!
//! - `docs/QTT_NTT_FINDINGS.md`: Butterfly MPO abandoned. Standard GPU NTT is optimal.
//!   NTT is memory-bound — ICICLE's kernel handles this natively.
//! - `fluidelite-zk/FluidEliteZK_FINDINGS.md`: ICICLE v4 proven on RTX 5070 (103.8 TPS).
//! - `fluidelite/FINDINGS.md`: GPU slower for small matrices due to kernel launch overhead.
//!   We batch all columns into a single NTT call to amortize launch cost.
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use core::marker::PhantomData;

use winterfell::{
    math::{FieldElement, fields::f64::BaseElement},
    EvaluationFrame, StarkDomain, TraceLde, TraceInfo, TracePolyTable,
    matrix::ColMatrix,
};

// winter-air for proof::Queries (not re-exported by winterfell umbrella crate)
use winter_air::proof::Queries;

// Re-import crypto traits through winterfell's re-export
use winterfell::crypto::{ElementHasher, VectorCommitment};

use icicle_core::bignum::BigNum;
use icicle_core::ntt::{
    self as icicle_ntt, NTTConfig, NTTDir, NTTInitDomainConfig,
    Ordering as NttOrdering,
};
use icicle_goldilocks::field::ScalarField as IcicleGoldilocks;
use icicle_runtime::{
    device::Device,
    memory::HostSlice,
    runtime,
};

/// Goldilocks field element (same as stark_impl::Felt, but standalone for this module).
type Felt = BaseElement;

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

/// Minimum trace length to justify GPU transfer overhead.
/// Below this, CPU FFT is faster due to kernel launch + memcpy latency.
/// Derived from fluidelite/FINDINGS.md §11: GPU slower for small matrices.
const GPU_MIN_TRACE_LENGTH: usize = 256;

// ═══════════════════════════════════════════════════════════════════════════
// Felt ↔ ICICLE Conversion
// ═══════════════════════════════════════════════════════════════════════════

/// Convert a Winterfell Goldilocks `Felt` to an ICICLE `ScalarField`.
///
/// Both represent elements of GF(2^64 - 2^32 + 1) as u64 values.
/// ICICLE stores the u64 as `[u32; 2]` limbs (little-endian).
#[inline(always)]
fn felt_to_icicle(f: Felt) -> IcicleGoldilocks {
    let val = f.as_int();
    IcicleGoldilocks::from([val as u32, (val >> 32) as u32])
}

/// Convert an ICICLE `ScalarField` back to a Winterfell `Felt`.
#[inline(always)]
fn icicle_to_felt(s: &IcicleGoldilocks) -> Felt {
    let limbs: &[u32; 2] = s.limbs();
    let val = limbs[0] as u64 | ((limbs[1] as u64) << 32);
    Felt::new(val)
}

/// Batch convert a column of `Felt` values to ICICLE scalars.
fn felt_column_to_icicle(column: &[Felt]) -> Vec<IcicleGoldilocks> {
    column.iter().map(|&f| felt_to_icicle(f)).collect()
}

/// Batch convert ICICLE scalars back to `Felt` values.
fn icicle_to_felt_column(scalars: &[IcicleGoldilocks]) -> Vec<Felt> {
    scalars.iter().map(icicle_to_felt).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// GPU NTT Engine
// ═══════════════════════════════════════════════════════════════════════════

/// Manages ICICLE GPU device initialization and NTT domain setup.
pub(crate) struct GpuNttEngine {
    /// Whether the GPU backend was successfully initialized.
    initialized: bool,
}

impl GpuNttEngine {
    /// Initialize the ICICLE runtime with CUDA backend and set up the NTT domain.
    ///
    /// `max_size` must be a power of two and ≥ the largest NTT size needed
    /// (typically `trace_length * blowup_factor`).
    pub fn init(max_size: u64) -> Result<Self, String> {
        // Load ICICLE backend (CUDA or CPU fallback)
        runtime::load_backend_from_env_or_default()
            .map_err(|e| format!("ICICLE backend load failed: {:?}", e))?;

        // Select CUDA device 0
        let device = Device::new("CUDA", 0);
        icicle_runtime::set_device(&device)
            .map_err(|e| format!("ICICLE set_device(CUDA:0) failed: {:?}", e))?;

        // Get the primitive root of unity for the Goldilocks field at the required order
        let rou = icicle_ntt::get_root_of_unity::<IcicleGoldilocks>(max_size)
            .map_err(|e| format!("ICICLE get_root_of_unity({max_size}) failed: {:?}", e))?;

        // Initialize the NTT domain with this root
        let init_cfg = NTTInitDomainConfig::default();
        icicle_ntt::initialize_domain(rou, &init_cfg)
            .map_err(|e| format!("ICICLE initialize_domain failed: {:?}", e))?;

        Ok(Self { initialized: true })
    }

    /// Perform batched iNTT (inverse NTT) on multiple columns simultaneously.
    ///
    /// Input: `columns` — each inner Vec is a column of trace evaluations.
    /// Output: polynomial coefficients in the same column layout.
    pub fn batch_intt(
        &self,
        columns: &[Vec<IcicleGoldilocks>],
    ) -> Result<Vec<Vec<IcicleGoldilocks>>, String> {
        if columns.is_empty() {
            return Ok(Vec::new());
        }

        let n = columns[0].len();
        let num_cols = columns.len();

        // Flatten columns into a contiguous buffer.
        // Layout: [col0[0..n], col1[0..n], ..., colK[0..n]]
        let mut flat_input: Vec<IcicleGoldilocks> = Vec::with_capacity(n * num_cols);
        for col in columns {
            assert_eq!(col.len(), n, "all columns must have equal length");
            flat_input.extend_from_slice(col);
        }

        let mut flat_output = vec![IcicleGoldilocks::zero(); n * num_cols];

        let mut cfg = NTTConfig::<IcicleGoldilocks>::default();
        cfg.batch_size = num_cols as i32;
        cfg.columns_batch = false; // each batch element is a contiguous run of n elements
        cfg.ordering = NttOrdering::kNN;

        let input_slice = HostSlice::from_slice(&flat_input);
        let output_slice = HostSlice::from_mut_slice(&mut flat_output);

        icicle_ntt::ntt(input_slice, NTTDir::kInverse, &cfg, output_slice)
            .map_err(|e| format!("ICICLE iNTT failed: {:?}", e))?;

        // Unflatten back into columns
        let mut result = Vec::with_capacity(num_cols);
        for i in 0..num_cols {
            let start = i * n;
            result.push(flat_output[start..start + n].to_vec());
        }

        Ok(result)
    }

    /// Perform batched coset NTT (forward NTT with domain offset) for LDE evaluation.
    ///
    /// Input: `poly_columns` — polynomial coefficients (one column per polynomial),
    ///         each of length `n`.
    /// Output: LDE evaluations over the coset `{offset * ω^i}` of size `n * blowup`,
    ///         one column per polynomial.
    pub fn batch_coset_ntt(
        &self,
        poly_columns: &[Vec<IcicleGoldilocks>],
        coset_gen: IcicleGoldilocks,
        blowup: usize,
    ) -> Result<Vec<Vec<IcicleGoldilocks>>, String> {
        if poly_columns.is_empty() {
            return Ok(Vec::new());
        }

        let n = poly_columns[0].len();
        let lde_size = n * blowup;
        let num_cols = poly_columns.len();

        // Zero-pad each polynomial column from n to lde_size, then flatten
        let mut flat_input: Vec<IcicleGoldilocks> = Vec::with_capacity(lde_size * num_cols);
        for col in poly_columns {
            assert_eq!(col.len(), n, "all polynomial columns must have equal length");
            flat_input.extend_from_slice(col);
            // Pad with zeros to reach lde_size
            flat_input.resize(flat_input.len() + (lde_size - n), IcicleGoldilocks::zero());
        }

        let mut flat_output = vec![IcicleGoldilocks::zero(); lde_size * num_cols];

        let mut cfg = NTTConfig::<IcicleGoldilocks>::default();
        cfg.batch_size = num_cols as i32;
        cfg.columns_batch = false;
        cfg.ordering = NttOrdering::kNN;
        cfg.coset_gen = coset_gen;

        let input_slice = HostSlice::from_slice(&flat_input);
        let output_slice = HostSlice::from_mut_slice(&mut flat_output);

        icicle_ntt::ntt(input_slice, NTTDir::kForward, &cfg, output_slice)
            .map_err(|e| format!("ICICLE coset NTT failed: {:?}", e))?;

        // Unflatten back into columns
        let mut result = Vec::with_capacity(num_cols);
        for i in 0..num_cols {
            let start = i * lde_size;
            result.push(flat_output[start..start + lde_size].to_vec());
        }

        Ok(result)
    }
}

impl Drop for GpuNttEngine {
    fn drop(&mut self) {
        if self.initialized {
            let _ = icicle_ntt::release_domain::<IcicleGoldilocks>();
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GpuTraceLde — Custom TraceLde Implementation
// ═══════════════════════════════════════════════════════════════════════════

/// GPU-accelerated Trace LDE that uses ICICLE's Goldilocks NTT for polynomial
/// interpolation and coset evaluation, replacing Winterfell's CPU-based FFT.
///
/// Storage is column-major via `ColMatrix<Felt>`, with a `VectorCommitment`
/// (Merkle tree) built from row hashes for query proofs.
pub struct GpuTraceLde<
    E: FieldElement,
    H: ElementHasher<BaseField = E::BaseField>,
    V: VectorCommitment<H>,
> {
    /// Column-major LDE of the main trace segment.
    main_segment_lde: ColMatrix<E::BaseField>,
    /// Merkle tree commitment over the main trace LDE rows.
    main_segment_commitment: V,
    /// Column-major LDE of the auxiliary trace segment (if present).
    aux_segment_lde: Option<ColMatrix<E>>,
    /// Merkle tree commitment over the auxiliary trace LDE rows.
    aux_segment_commitment: Option<V>,
    /// LDE blowup factor.
    blowup: usize,
    /// Trace metadata.
    trace_info: TraceInfo,

    _h: PhantomData<H>,
}

impl<E, H, V> GpuTraceLde<E, H, V>
where
    E: FieldElement<BaseField = Felt>,
    H: ElementHasher<BaseField = Felt>,
    V: VectorCommitment<H>,
{
    /// Construct a GPU-accelerated trace LDE.
    ///
    /// 1. Converts trace columns to ICICLE format
    /// 2. Runs GPU iNTT → polynomial coefficients
    /// 3. Runs GPU coset NTT → LDE evaluations
    /// 4. Builds CPU Merkle tree from LDE rows
    pub fn new(
        trace_info: &TraceInfo,
        main_trace: &ColMatrix<Felt>,
        domain: &StarkDomain<Felt>,
    ) -> Result<(Self, TracePolyTable<E>), String> {
        let num_cols = main_trace.num_cols();
        let trace_len = main_trace.num_rows();
        let blowup = domain.trace_to_lde_blowup();
        let lde_size = trace_len * blowup;

        // ── Step 0: Initialize GPU NTT engine ──
        let engine = GpuNttEngine::init(lde_size as u64)?;

        // ── Step 1: Convert Winterfell columns → ICICLE format ──
        let icicle_columns: Vec<Vec<IcicleGoldilocks>> = (0..num_cols)
            .map(|c| felt_column_to_icicle(main_trace.get_column(c)))
            .collect();

        // ── Step 2: GPU iNTT → polynomial coefficients ──
        let icicle_polys = engine.batch_intt(&icicle_columns)?;

        // Convert polynomial coefficients back to Winterfell Felt for TracePolyTable
        let poly_columns: Vec<Vec<E::BaseField>> = icicle_polys
            .iter()
            .map(|col| icicle_to_felt_column(col))
            .collect();
        let trace_polys = ColMatrix::new(poly_columns);

        // ── Step 3: GPU coset NTT → LDE evaluations ──
        let offset = domain.offset();
        let coset_gen = felt_to_icicle(offset);

        let icicle_lde = engine.batch_coset_ntt(&icicle_polys, coset_gen, blowup)?;

        // Convert LDE evaluations back to Winterfell Felt
        let lde_columns: Vec<Vec<E::BaseField>> = icicle_lde
            .iter()
            .map(|col| icicle_to_felt_column(col))
            .collect();
        let main_segment_lde = ColMatrix::new(lde_columns);

        assert_eq!(main_segment_lde.num_cols(), num_cols);
        assert_eq!(main_segment_lde.num_rows(), lde_size);

        // ── Step 4: Build Merkle tree commitment from LDE rows ──
        // ColMatrix::commit_to_rows hashes each row and builds a VectorCommitment.
        // Compatible with non-partitioned ProofOptions (partition_size == num_cols).
        let main_segment_commitment = main_segment_lde.commit_to_rows::<H, V>();

        // ── Step 5: Package ──
        let trace_poly_table = TracePolyTable::new(trace_polys);
        let trace_lde = Self {
            main_segment_lde,
            main_segment_commitment,
            aux_segment_lde: None,
            aux_segment_commitment: None,
            blowup,
            trace_info: trace_info.clone(),
            _h: PhantomData,
        };

        Ok((trace_lde, trace_poly_table))
    }

    /// CPU fallback constructor using Winterfell's ColMatrix methods.
    ///
    /// Called when GPU initialization fails at runtime. Produces an identical
    /// `GpuTraceLde` struct (same type, compatible with the Prover trait) but
    /// performs all NTT work on CPU via Winterfell's built-in FFT.
    pub fn new_cpu_fallback(
        trace_info: &TraceInfo,
        main_trace: &ColMatrix<Felt>,
        domain: &StarkDomain<Felt>,
    ) -> (Self, TracePolyTable<E>) {
        let blowup = domain.trace_to_lde_blowup();

        // CPU iFFT: evaluations → polynomial coefficients
        let trace_polys = main_trace.interpolate_columns();
        // CPU FFT (coset): coefficients → LDE evaluations
        let main_segment_lde = trace_polys.evaluate_columns_over(domain);
        // CPU Merkle tree
        let main_segment_commitment = main_segment_lde.commit_to_rows::<H, V>();

        let trace_poly_table = TracePolyTable::new(trace_polys);
        let trace_lde = Self {
            main_segment_lde,
            main_segment_commitment,
            aux_segment_lde: None,
            aux_segment_commitment: None,
            blowup,
            trace_info: trace_info.clone(),
            _h: PhantomData,
        };

        (trace_lde, trace_poly_table)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TraceLde Trait Implementation
// ═══════════════════════════════════════════════════════════════════════════

impl<E, H, V> TraceLde<E> for GpuTraceLde<E, H, V>
where
    E: FieldElement<BaseField = Felt>,
    H: ElementHasher<BaseField = Felt> + core::marker::Sync,
    V: VectorCommitment<H> + core::marker::Sync,
{
    type HashFn = H;
    type VC = V;

    fn get_main_trace_commitment(&self) -> H::Digest {
        self.main_segment_commitment.commitment()
    }

    fn set_aux_trace(
        &mut self,
        aux_trace: &ColMatrix<E>,
        domain: &StarkDomain<E::BaseField>,
    ) -> (ColMatrix<E>, H::Digest) {
        // For auxiliary traces, fall back to CPU interpolation + evaluation.
        // Our thermal and contraction STARKs don't use auxiliary trace segments,
        // but we implement this for trait completeness.
        let aux_polys = aux_trace.interpolate_columns();
        let aux_lde = aux_polys.evaluate_columns_over(domain);

        let aux_commitment: V = aux_lde.commit_to_rows::<H, V>();
        let digest = aux_commitment.commitment();

        assert!(
            self.aux_segment_lde.is_none(),
            "auxiliary trace has already been set"
        );
        assert_eq!(
            self.main_segment_lde.num_rows(),
            aux_lde.num_rows(),
            "auxiliary segment must have the same number of rows as main segment"
        );

        self.aux_segment_lde = Some(aux_lde);
        self.aux_segment_commitment = Some(aux_commitment);

        (aux_polys, digest)
    }

    fn read_main_trace_frame_into(
        &self,
        lde_step: usize,
        frame: &mut EvaluationFrame<E::BaseField>,
    ) {
        let next_lde_step = (lde_step + self.blowup) % self.main_segment_lde.num_rows();
        let width = frame.current().len();

        for col_idx in 0..width {
            frame.current_mut()[col_idx] = self.main_segment_lde.get(col_idx, lde_step);
        }
        for col_idx in 0..width {
            frame.next_mut()[col_idx] = self.main_segment_lde.get(col_idx, next_lde_step);
        }
    }

    fn read_aux_trace_frame_into(&self, lde_step: usize, frame: &mut EvaluationFrame<E>) {
        let next_lde_step = (lde_step + self.blowup) % self.main_segment_lde.num_rows();
        let segment = self
            .aux_segment_lde
            .as_ref()
            .expect("auxiliary trace segment not set");
        let width = frame.current().len();

        for col_idx in 0..width {
            frame.current_mut()[col_idx] = segment.get(col_idx, lde_step);
        }
        for col_idx in 0..width {
            frame.next_mut()[col_idx] = segment.get(col_idx, next_lde_step);
        }
    }

    fn query(&self, positions: &[usize]) -> Vec<Queries> {
        let main_states: Vec<Vec<E::BaseField>> = positions
            .iter()
            .map(|&pos| {
                let mut row = vec![E::BaseField::ZERO; self.main_segment_lde.num_cols()];
                self.main_segment_lde.read_row_into(pos, &mut row);
                row
            })
            .collect();

        let main_proof = self
            .main_segment_commitment
            .open_many(positions)
            .expect("failed to generate batch opening proof for main trace");

        let mut result = vec![Queries::new::<H, E::BaseField, V>(
            main_proof.1,
            main_states,
        )];

        if let (Some(ref aux_lde), Some(ref aux_commitment)) =
            (&self.aux_segment_lde, &self.aux_segment_commitment)
        {
            let aux_states: Vec<Vec<E>> = positions
                .iter()
                .map(|&pos| {
                    let mut row = vec![E::ZERO; aux_lde.num_cols()];
                    aux_lde.read_row_into(pos, &mut row);
                    row
                })
                .collect();

            let aux_proof = aux_commitment
                .open_many(positions)
                .expect("failed to generate batch opening proof for auxiliary trace");

            result.push(Queries::new::<H, E, V>(aux_proof.1, aux_states));
        }

        result
    }

    fn trace_len(&self) -> usize {
        self.main_segment_lde.num_rows()
    }

    fn blowup(&self) -> usize {
        self.blowup
    }

    fn trace_info(&self) -> &TraceInfo {
        &self.trace_info
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════════

/// Attempt to build a GPU-accelerated trace LDE.
///
/// Returns `Ok(...)` with the GPU trace LDE and polynomial table on success,
/// or `Err(reason)` if the GPU path fails (missing CUDA, ICICLE not loaded, etc.).
///
/// Callers should fall back to `DefaultTraceLde::new()` on error.
pub fn try_gpu_trace_lde<E, H, V>(
    trace_info: &TraceInfo,
    main_trace: &ColMatrix<Felt>,
    domain: &StarkDomain<Felt>,
) -> Result<(GpuTraceLde<E, H, V>, TracePolyTable<E>), String>
where
    E: FieldElement<BaseField = Felt>,
    H: ElementHasher<BaseField = Felt> + core::marker::Sync,
    V: VectorCommitment<H> + core::marker::Sync,
{
    let trace_len = main_trace.num_rows();

    if trace_len < GPU_MIN_TRACE_LENGTH {
        return Err(format!(
            "trace length {} below GPU threshold {}; use CPU",
            trace_len, GPU_MIN_TRACE_LENGTH
        ));
    }

    GpuTraceLde::new(trace_info, main_trace, domain)
}
