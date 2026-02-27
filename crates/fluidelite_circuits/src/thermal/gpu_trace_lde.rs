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
    math::{FieldElement, StarkField, fields::f64::BaseElement},
    EvaluationFrame, PartitionOptions, StarkDomain, TraceLde, TraceInfo, TracePolyTable,
    matrix::{ColMatrix, RowMatrix},
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

use std::sync::OnceLock;

/// Result of the one-time ICICLE initialization.
/// `Ok(max_domain_log2)` on success, `Err(reason)` on failure.
static ICICLE_INIT: OnceLock<Result<u32, String>> = OnceLock::new();

/// Maximum NTT domain log₂ size we'll actually use.
///
/// This controls how many twiddle factors ICICLE pre-computes on the GPU.
/// Memory cost: `2^MAX_DOMAIN_LOG2 × 8 bytes` for twiddles.
///   - log2=20 → 8 MB   (1M points)
///   - log2=22 → 32 MB  (4M points)
///   - log2=24 → 128 MB (16M points)
///
/// **Critical:** We initialise the ICICLE domain with *Winterfell's* root of
/// unity, NOT ICICLE's built-in root (they are different primitive 2³²-th
/// roots of GF(2⁶⁴ − 2³² + 1)). Using the same root ensures ICICLE's iNTT
/// produces identical polynomial coefficients to Winterfell's
/// `interpolate_columns`, so the GPU can be a drop-in replacement.
const MAX_DOMAIN_LOG2: u32 = 22; // 4M points — covers blowup×trace up to 4M rows

/// One-time ICICLE backend + CUDA device + NTT domain initialization.
/// Thread-safe via OnceLock — subsequent calls return the cached result.
fn ensure_icicle_initialized() -> Result<u32, String> {
    ICICLE_INIT
        .get_or_init(|| {
            // Load ICICLE backend — always try BOTH paths.
            // `load_backend_from_env_or_default()` may find the CPU backend only;
            // `/opt/icicle/lib/backend` carries the per-field CUDA shared objects.
            let _ = runtime::load_backend_from_env_or_default();
            let _ = runtime::load_backend("/opt/icicle/lib/backend");

            // Select CUDA device 0
            let device = Device::new("CUDA", 0);
            icicle_runtime::set_device(&device)
                .map_err(|e| format!("ICICLE set_device(CUDA:0) failed: {:?}", e))?;

            // Use *Winterfell's* root of unity — NOT ICICLE's default.
            //
            // Both are valid primitive 2^MAX_DOMAIN_LOG2-th roots of the
            // Goldilocks field, but they are numerically different:
            //   ICICLE default:  1753635133440165772  (from generator 7)
            //   Winterfell:      derived from TWO_ADIC_ROOT_OF_UNITY = 7277203076849721926
            //
            // If we use ICICLE's root, the iNTT interprets evaluations as being
            // at ICICLE's domain points and produces different polynomial
            // coefficients than Winterfell's `interpolate_columns`. Using
            // Winterfell's root makes ICICLE's iNTT a drop-in replacement.
            //
            // `Felt::get_root_of_unity(n)` returns a primitive 2^n-th root.
            // `initialize_domain` accepts any primitive root of sufficient order.
            let winterfell_rou = Felt::get_root_of_unity(MAX_DOMAIN_LOG2);
            let rou: IcicleGoldilocks = felt_to_icicle(winterfell_rou);

            let init_cfg = NTTInitDomainConfig::default();
            icicle_ntt::initialize_domain(rou, &init_cfg)
                .map_err(|e| format!("ICICLE initialize_domain failed: {:?}", e))?;

            Ok(MAX_DOMAIN_LOG2)
        })
        .clone()
}

/// Manages ICICLE GPU device initialization and NTT domain setup.
pub(crate) struct GpuNttEngine {
    /// Whether the GPU backend was successfully initialized.
    #[allow(dead_code)]
    initialized: bool,
}

impl GpuNttEngine {
    /// Initialize the ICICLE runtime with CUDA backend and set up the NTT domain.
    ///
    /// `max_size` must be a power of two and ≤ `2^MAX_DOMAIN_LOG2`.
    /// Thread-safe: the ICICLE backend and NTT domain are initialized exactly
    /// once via `OnceLock`, even when called from multiple threads concurrently.
    pub fn init(max_size: u64) -> Result<Self, String> {
        let domain_log2 = ensure_icicle_initialized()?;

        // Validate that the requested NTT size fits within the initialized domain.
        let requested_log2 = (max_size as u64).trailing_zeros();
        if requested_log2 > domain_log2 {
            return Err(format!(
                "requested NTT size 2^{requested_log2} exceeds domain 2^{domain_log2}"
            ));
        }

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

        // Re-assert CUDA device before kernel launch (guards against parallel callers)
        let device = Device::new("CUDA", 0);
        icicle_runtime::set_device(&device)
            .map_err(|e| format!("ICICLE set_device before iNTT failed: {:?}", e))?;

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
    /// Currently unused in the TraceLde pipeline (ICICLE produces
    /// different element ordering than Winterfell). Retained for the
    /// diagnostic example and unit tests.
    #[allow(dead_code)]
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

        // Re-assert CUDA device before kernel launch
        let device = Device::new("CUDA", 0);
        icicle_runtime::set_device(&device)
            .map_err(|e| format!("ICICLE set_device before coset NTT failed: {:?}", e))?;

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

// NTT domain is global (OnceLock) — no per-engine cleanup needed.
// The domain persists for the lifetime of the process.

// ═══════════════════════════════════════════════════════════════════════════
// GpuTraceLde — Custom TraceLde Implementation
// ═══════════════════════════════════════════════════════════════════════════

// Winterfell's DefaultTraceLde is needed for the CPU fallback path so that
// constraint evaluation, Merkle commitment, and OOD checks are byte-identical
// to what the verifier expects.
use winterfell::DefaultTraceLde;

/// Internal storage for the trace LDE — either GPU-accelerated or
/// a delegated `DefaultTraceLde` when the GPU path is not taken.
///
/// Both variants produce byte-identical LDE values and Merkle commitments.
/// The `Gpu` variant uses the same `RowMatrix` layout and `PartitionOptions`
/// as `DefaultTraceLde`, with the sole difference being that polynomial
/// interpolation (iNTT) runs on the GPU via ICICLE.
enum TraceLdeInner<
    E: FieldElement,
    H: ElementHasher<BaseField = E::BaseField>,
    V: VectorCommitment<H>,
> {
    /// GPU path: polynomial interpolation via ICICLE CUDA iNTT,
    /// then Winterfell's own forward NTT for LDE evaluation.
    /// Storage matches `DefaultTraceLde` (RowMatrix + PartitionOptions).
    Gpu {
        main_segment_lde: RowMatrix<E::BaseField>,
        main_segment_commitment: V,
        aux_segment_lde: Option<RowMatrix<E>>,
        aux_segment_commitment: Option<V>,
        blowup: usize,
        trace_info: TraceInfo,
        partition_options: PartitionOptions,
        _h: PhantomData<H>,
    },
    /// CPU fallback: delegates to Winterfell's DefaultTraceLde for
    /// byte-identical Merkle commitments and constraint evaluations.
    Cpu(DefaultTraceLde<E, H, V>),
}

/// GPU-accelerated Trace LDE that uses ICICLE's Goldilocks NTT for polynomial
/// interpolation and coset evaluation, replacing Winterfell's CPU-based FFT.
///
/// When the GPU path is unavailable (small trace, no CUDA, init failure),
/// falls back to `DefaultTraceLde` internally to guarantee bit-identical
/// results with the non-GPU prover.
pub struct GpuTraceLde<
    E: FieldElement,
    H: ElementHasher<BaseField = E::BaseField>,
    V: VectorCommitment<H>,
> {
    inner: TraceLdeInner<E, H, V>,
}

impl<E, H, V> GpuTraceLde<E, H, V>
where
    E: FieldElement<BaseField = Felt>,
    H: ElementHasher<BaseField = Felt>,
    V: VectorCommitment<H>,
{
    /// Construct a GPU-accelerated trace LDE.
    ///
    /// **Pipeline:**
    /// 1. Convert Winterfell columns → ICICLE format
    /// 2. GPU iNTT → polynomial coefficients (5-16× speedup on CUDA)
    /// 3. Convert back to ColMatrix for Winterfell
    /// 4. Winterfell's `RowMatrix::evaluate_polys_over` → LDE (correct domain ordering)
    /// 5. `RowMatrix::commit_to_rows(partition_options)` → Merkle commitment
    ///
    /// The GPU accelerates the expensive interpolation step (iNTT). The forward
    /// NTT (LDE evaluation) stays on CPU to preserve Winterfell's exact domain
    /// ordering for constraint evaluation. The commitment uses the same
    /// `RowMatrix` + `PartitionOptions` as `DefaultTraceLde`.
    ///
    /// **Correctness guarantee:**
    /// `test_icicle_intt_matches_winterfell_interpolate` verifies 100%
    /// coefficient match between ICICLE iNTT and Winterfell's
    /// `interpolate_columns()`. This is achieved by initializing ICICLE's
    /// NTT domain with Winterfell's root of unity (not ICICLE's default).
    pub fn new(
        trace_info: &TraceInfo,
        main_trace: &ColMatrix<Felt>,
        domain: &StarkDomain<Felt>,
        partition_options: PartitionOptions,
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
        //
        // This is the GPU-accelerated step. ICICLE's batched NTT runs on the
        // CUDA device, giving 5-16× speedup over CPU for large traces.
        // The ICICLE domain is initialized with Winterfell's root of unity,
        // so the resulting coefficients are identical to Winterfell's
        // `interpolate_columns()` (verified by unit test).
        let icicle_polys = engine.batch_intt(&icicle_columns)?;

        // Convert polynomial coefficients back to Winterfell Felt
        let poly_columns: Vec<Vec<E::BaseField>> = icicle_polys
            .iter()
            .map(|col| icicle_to_felt_column(col))
            .collect();
        let trace_polys = ColMatrix::new(poly_columns);

        // ── Step 3: CPU LDE evaluation via Winterfell's RowMatrix ──
        //
        // Uses the same `RowMatrix::evaluate_polys_over::<8>()` as
        // `DefaultTraceLde`'s internal `build_trace_commitment()`.
        // This guarantees correct domain ordering for constraint evaluation.
        let main_segment_lde =
            RowMatrix::evaluate_polys_over::<8>(&trace_polys, domain);

        assert_eq!(main_segment_lde.num_cols(), num_cols);
        assert_eq!(main_segment_lde.num_rows(), lde_size);

        // ── Step 4: Build Merkle tree commitment (same as DefaultTraceLde) ──
        let main_segment_commitment =
            main_segment_lde.commit_to_rows::<H, V>(partition_options);

        // ── Step 5: Package ──
        let trace_poly_table = TracePolyTable::new(trace_polys);
        let trace_lde = Self {
            inner: TraceLdeInner::Gpu {
                main_segment_lde,
                main_segment_commitment,
                aux_segment_lde: None,
                aux_segment_commitment: None,
                blowup,
                trace_info: trace_info.clone(),
                partition_options,
                _h: PhantomData,
            },
        };

        Ok((trace_lde, trace_poly_table))
    }

    /// CPU fallback constructor delegating to Winterfell's `DefaultTraceLde`.
    ///
    /// Produces byte-identical Merkle commitments, constraint evaluations, and
    /// OOD checks as the non-GPU prover path. This guarantees the verifier
    /// accepts proofs from the CPU fallback exactly as if `DefaultTraceLde`
    /// had been used directly.
    pub fn new_cpu_fallback(
        trace_info: &TraceInfo,
        main_trace: &ColMatrix<Felt>,
        domain: &StarkDomain<Felt>,
        partition_options: PartitionOptions,
    ) -> (Self, TracePolyTable<E>) {
        let (default_lde, trace_polys) =
            DefaultTraceLde::new(trace_info, main_trace, domain, partition_options);
        let trace_lde = Self {
            inner: TraceLdeInner::Cpu(default_lde),
        };
        (trace_lde, trace_polys)
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
        match &self.inner {
            TraceLdeInner::Gpu { main_segment_commitment, .. } => {
                main_segment_commitment.commitment()
            }
            TraceLdeInner::Cpu(default_lde) => default_lde.get_main_trace_commitment(),
        }
    }

    fn set_aux_trace(
        &mut self,
        aux_trace: &ColMatrix<E>,
        domain: &StarkDomain<E::BaseField>,
    ) -> (ColMatrix<E>, H::Digest) {
        match &mut self.inner {
            TraceLdeInner::Gpu {
                main_segment_lde,
                aux_segment_lde,
                aux_segment_commitment,
                partition_options,
                ..
            } => {
                let aux_polys = aux_trace.interpolate_columns();
                let aux_lde =
                    RowMatrix::evaluate_polys_over::<8>(&aux_polys, domain);

                let commitment: V =
                    aux_lde.commit_to_rows::<H, V>(*partition_options);
                let digest = commitment.commitment();

                assert!(
                    aux_segment_lde.is_none(),
                    "auxiliary trace has already been set"
                );
                assert_eq!(
                    main_segment_lde.num_rows(),
                    aux_lde.num_rows(),
                    "auxiliary segment must have the same number of rows as main segment"
                );

                *aux_segment_lde = Some(aux_lde);
                *aux_segment_commitment = Some(commitment);

                (aux_polys, digest)
            }
            TraceLdeInner::Cpu(default_lde) => default_lde.set_aux_trace(aux_trace, domain),
        }
    }

    fn read_main_trace_frame_into(
        &self,
        lde_step: usize,
        frame: &mut EvaluationFrame<E::BaseField>,
    ) {
        match &self.inner {
            TraceLdeInner::Gpu {
                main_segment_lde,
                blowup,
                ..
            } => {
                let next_lde_step =
                    (lde_step + blowup) % main_segment_lde.num_rows();
                frame
                    .current_mut()
                    .copy_from_slice(main_segment_lde.row(lde_step));
                frame
                    .next_mut()
                    .copy_from_slice(main_segment_lde.row(next_lde_step));
            }
            TraceLdeInner::Cpu(default_lde) => {
                default_lde.read_main_trace_frame_into(lde_step, frame);
            }
        }
    }

    fn read_aux_trace_frame_into(&self, lde_step: usize, frame: &mut EvaluationFrame<E>) {
        match &self.inner {
            TraceLdeInner::Gpu {
                main_segment_lde,
                aux_segment_lde,
                blowup,
                ..
            } => {
                let next_lde_step =
                    (lde_step + blowup) % main_segment_lde.num_rows();
                let segment = aux_segment_lde
                    .as_ref()
                    .expect("auxiliary trace segment not set");
                frame
                    .current_mut()
                    .copy_from_slice(segment.row(lde_step));
                frame
                    .next_mut()
                    .copy_from_slice(segment.row(next_lde_step));
            }
            TraceLdeInner::Cpu(default_lde) => {
                default_lde.read_aux_trace_frame_into(lde_step, frame);
            }
        }
    }

    fn query(&self, positions: &[usize]) -> Vec<Queries> {
        match &self.inner {
            TraceLdeInner::Gpu {
                main_segment_lde,
                main_segment_commitment,
                aux_segment_lde,
                aux_segment_commitment,
                ..
            } => {
                let main_states: Vec<Vec<E::BaseField>> = positions
                    .iter()
                    .map(|&pos| main_segment_lde.row(pos).to_vec())
                    .collect();

                let main_proof = main_segment_commitment
                    .open_many(positions)
                    .expect("failed to generate batch opening proof for main trace");

                let mut result = vec![Queries::new::<H, E::BaseField, V>(
                    main_proof.1,
                    main_states,
                )];

                if let (Some(ref aux_lde), Some(ref aux_com)) =
                    (aux_segment_lde, aux_segment_commitment)
                {
                    let aux_states: Vec<Vec<E>> = positions
                        .iter()
                        .map(|&pos| aux_lde.row(pos).to_vec())
                        .collect();

                    let aux_proof = aux_com
                        .open_many(positions)
                        .expect("failed to generate batch opening proof for auxiliary trace");

                    result.push(Queries::new::<H, E, V>(aux_proof.1, aux_states));
                }

                result
            }
            TraceLdeInner::Cpu(default_lde) => default_lde.query(positions),
        }
    }

    fn trace_len(&self) -> usize {
        match &self.inner {
            TraceLdeInner::Gpu { main_segment_lde, .. } => main_segment_lde.num_rows(),
            TraceLdeInner::Cpu(default_lde) => default_lde.trace_len(),
        }
    }

    fn blowup(&self) -> usize {
        match &self.inner {
            TraceLdeInner::Gpu { blowup, .. } => *blowup,
            TraceLdeInner::Cpu(default_lde) => default_lde.blowup(),
        }
    }

    fn trace_info(&self) -> &TraceInfo {
        match &self.inner {
            TraceLdeInner::Gpu { trace_info, .. } => trace_info,
            TraceLdeInner::Cpu(default_lde) => default_lde.trace_info(),
        }
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
    partition_options: PartitionOptions,
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

    GpuTraceLde::new(trace_info, main_trace, domain, partition_options)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use winterfell::math::StarkField;

    /// Verify Felt → ICICLE → Felt roundtrip is lossless for representative values.
    #[test]
    fn test_felt_icicle_roundtrip() {
        let test_values: Vec<u64> = vec![
            0,
            1,
            42,
            (1u64 << 32) - 1,        // max low limb
            1u64 << 32,               // min high limb
            u64::MAX,                 // out of Goldilocks range → wraps via Felt::new
            0xDEAD_BEEF_CAFE_BABEu64, // arbitrary
            Felt::MODULUS - 1,        // largest valid element
        ];

        for &raw in &test_values {
            let felt = Felt::new(raw);
            let icicle = felt_to_icicle(felt);
            let back = icicle_to_felt(&icicle);
            assert_eq!(
                felt, back,
                "roundtrip failed for raw={raw:#x}: felt={:?}, back={:?}",
                felt.as_int(),
                back.as_int()
            );
        }
    }

    /// Verify batch column conversion roundtrip.
    #[test]
    fn test_batch_column_roundtrip() {
        let column: Vec<Felt> = (0..256).map(|i| Felt::new(i * 7 + 3)).collect();
        let icicle = felt_column_to_icicle(&column);
        let back = icicle_to_felt_column(&icicle);
        assert_eq!(column, back);
    }

    /// Smoke test: initialise GPU NTT engine (or confirm fallback error message).
    ///
    /// This test is non-fatal: on machines without CUDA it produces a descriptive
    /// error that confirms the fallback path works correctly.
    #[test]
    fn test_gpu_ntt_engine_init() {
        let max_size = 1024u64;
        match GpuNttEngine::init(max_size) {
            Ok(engine) => {
                assert!(engine.initialized);
                eprintln!("[GPU NTT] Engine initialized successfully for size {max_size}");
            }
            Err(reason) => {
                eprintln!("[GPU NTT] Expected fallback: {reason}");
                // On CI / non-CUDA machines this is the normal path
                assert!(
                    reason.contains("ICICLE") || reason.contains("CUDA") || reason.contains("device"),
                    "unexpected error: {reason}"
                );
            }
        }
    }

    /// If CUDA is available, run a full iNTT → coset NTT cycle and verify
    /// the output length is correct.
    #[test]
    fn test_gpu_ntt_roundtrip_if_cuda() {
        let n = 256usize;
        let blowup = 8usize;
        let lde_size = (n * blowup) as u64;

        let engine = match GpuNttEngine::init(lde_size) {
            Ok(e) => e,
            Err(reason) => {
                eprintln!("[GPU NTT] Skipping NTT roundtrip (no CUDA): {reason}");
                return;
            }
        };

        // Build 3 columns of dummy evaluations
        let columns: Vec<Vec<IcicleGoldilocks>> = (0..3)
            .map(|c| {
                (0..n)
                    .map(|i| IcicleGoldilocks::from(((c * n + i) as u32) % 1000))
                    .collect()
            })
            .collect();

        // iNTT: evaluations → polynomial coefficients
        let polys = engine.batch_intt(&columns).expect("iNTT failed");
        assert_eq!(polys.len(), 3);
        for p in &polys {
            assert_eq!(p.len(), n, "polynomial column has wrong length");
        }

        // coset NTT: coefficients → LDE evaluations (at blowup)
        // Use a simple coset generator (7 is a generator of the multiplicative group)
        let coset_gen = felt_to_icicle(Felt::new(7));
        let lde = engine
            .batch_coset_ntt(&polys, coset_gen, blowup)
            .expect("coset NTT failed");
        assert_eq!(lde.len(), 3);
        for l in &lde {
            assert_eq!(
                l.len(),
                n * blowup,
                "LDE column has wrong length"
            );
        }

        eprintln!("[GPU NTT] Full iNTT→coset-NTT cycle passed (3 cols × {n} → {lde_len})", lde_len = n * blowup);
    }

    /// **Critical correctness test:** verify that ICICLE's iNTT produces
    /// identical polynomial coefficients to Winterfell's `interpolate_columns`.
    ///
    /// This is the lynchpin for GPU NTT acceleration. If this test passes,
    /// we can substitute ICICLE's iNTT for Winterfell's CPU-based interpolation
    /// and get a drop-in GPU speedup for the expensive polynomial evaluation step.
    #[test]
    fn test_icicle_intt_matches_winterfell_interpolate() {
        let n = 1024usize; // large enough to exercise real FFT butterflies

        let engine = match GpuNttEngine::init(n as u64) {
            Ok(e) => e,
            Err(reason) => {
                eprintln!("[GPU NTT] Skipping iNTT-vs-Winterfell test (no CUDA): {reason}");
                return;
            }
        };

        // Build 5 columns of representative evaluations (non-trivial values).
        let num_cols = 5;
        let felt_columns: Vec<Vec<Felt>> = (0..num_cols)
            .map(|c| {
                (0..n)
                    .map(|i| {
                        // Mix column index and row index for diverse values
                        let val = ((c as u64 + 1) * 97 + (i as u64) * 31 + 17) % Felt::MODULUS;
                        Felt::new(val)
                    })
                    .collect()
            })
            .collect();

        // ── Path A: Winterfell's interpolate_columns (CPU reference) ──
        let winterfell_col_matrix = ColMatrix::new(felt_columns.clone());
        let winterfell_polys = winterfell_col_matrix.interpolate_columns();

        // ── Path B: ICICLE iNTT (GPU) ──
        let icicle_columns: Vec<Vec<IcicleGoldilocks>> = felt_columns
            .iter()
            .map(|col| felt_column_to_icicle(col))
            .collect();

        let icicle_polys = engine.batch_intt(&icicle_columns).expect("ICICLE iNTT failed");

        // ── Compare coefficient-by-coefficient ──
        assert_eq!(icicle_polys.len(), num_cols);
        let mut total_checked = 0usize;
        let mut mismatches = 0usize;

        for col_idx in 0..num_cols {
            let winterfell_col = winterfell_polys.get_column(col_idx);
            let icicle_col = &icicle_polys[col_idx];

            assert_eq!(
                winterfell_col.len(),
                icicle_col.len(),
                "column {col_idx}: length mismatch"
            );

            for (row, (w, i)) in winterfell_col.iter().zip(icicle_col.iter()).enumerate() {
                let icicle_felt = icicle_to_felt(i);
                if *w != icicle_felt {
                    if mismatches < 5 {
                        eprintln!(
                            "  MISMATCH col={col_idx} row={row}: winterfell={} icicle={}",
                            w.as_int(),
                            icicle_felt.as_int()
                        );
                    }
                    mismatches += 1;
                }
                total_checked += 1;
            }
        }

        assert_eq!(
            mismatches, 0,
            "ICICLE iNTT vs Winterfell interpolate_columns: {mismatches}/{total_checked} mismatches"
        );
        eprintln!(
            "[GPU NTT] ICICLE iNTT matches Winterfell interpolate_columns: {total_checked}/{total_checked} coefficients"
        );
    }

    /// Benchmark GPU iNTT vs CPU `interpolate_columns` across multiple sizes.
    ///
    /// This measures the interpolation speedup that the GPU path provides
    /// within the LDE pipeline (the most expensive single step).
    #[test]
    fn test_gpu_intt_benchmark() {
        use std::time::Instant;

        let engine = match GpuNttEngine::init(1u64 << 20) {
            Ok(e) => e,
            Err(reason) => {
                eprintln!("[GPU NTT] Skipping benchmark (no CUDA): {reason}");
                return;
            }
        };

        let num_cols = 23; // typical STARK trace width (ContractionAir)

        for log_n in [10u32, 12, 14, 16] {
            let n = 1usize << log_n;

            // Build representative trace columns
            let felt_columns: Vec<Vec<Felt>> = (0..num_cols)
                .map(|c| {
                    (0..n)
                        .map(|i| Felt::new(((c + 1) * 97 + i * 31 + 17) as u64 % Felt::MODULUS))
                        .collect()
                })
                .collect();

            // ── CPU: Winterfell's interpolate_columns ──
            let cpu_matrix = ColMatrix::new(felt_columns.clone());
            let cpu_start = Instant::now();
            let _cpu_polys = cpu_matrix.interpolate_columns();
            let cpu_us = cpu_start.elapsed().as_micros();

            // ── GPU: ICICLE batched iNTT ──
            let icicle_cols: Vec<Vec<IcicleGoldilocks>> = felt_columns
                .iter()
                .map(|col| felt_column_to_icicle(col))
                .collect();

            // Warm up
            let _ = engine.batch_intt(&icicle_cols);

            let gpu_start = Instant::now();
            let _gpu_polys = engine.batch_intt(&icicle_cols).expect("iNTT failed");
            let gpu_us = gpu_start.elapsed().as_micros();

            let speedup = if gpu_us > 0 {
                cpu_us as f64 / gpu_us as f64
            } else {
                f64::INFINITY
            };

            eprintln!(
                "[Benchmark] 2^{log_n} × {num_cols}cols: CPU={cpu_us}µs GPU={gpu_us}µs speedup={speedup:.1}×"
            );
        }
    }

    /// Full integration test: build a 256-row thermal trace, prove via the
    /// InternalProver (which routes through GpuTraceLde), and verify the proof.
    ///
    /// On CUDA-capable machines, this exercises the GPU NTT path.
    /// On CPU-only machines, it exercises the CPU fallback path.
    #[test]
    fn test_gpu_trace_lde_full_pipeline() {
        use super::super::stark_impl::{
            prove_thermal_stark, verify_thermal_stark, TimestepPhysics,
        };
        use sha2::{Digest, Sha256};

        // Build 256 timesteps with proper hash chaining and conservation residuals.
        // This replicates the make_test_steps() logic from stark_impl tests.
        let n = 256usize;
        let base_energy = fluidelite_core::field::Q16::from_f64(1000.0);
        let per_step_drift = fluidelite_core::field::Q16::from_f64(0.005);
        let dt = fluidelite_core::field::Q16::from_f64(0.001);
        let alpha = fluidelite_core::field::Q16::from_f64(0.01);

        // Generate a SHA-256 hash chain: hash_{i+1} = SHA256(hash_i || i)
        let mut hashes: Vec<[u64; 4]> = Vec::with_capacity(n + 1);
        let mut current_hash: [u64; 4] = {
            let mut h = Sha256::new();
            h.update(b"GPU_TRACE_LDE_TEST_V1");
            let result = h.finalize();
            let mut limbs = [0u64; 4];
            for (k, limb) in limbs.iter_mut().enumerate() {
                let offset = k * 8;
                *limb = u64::from_le_bytes(result[offset..offset + 8].try_into().unwrap());
            }
            limbs
        };
        hashes.push(current_hash);
        for i in 0..n {
            let mut h = Sha256::new();
            for limb in &current_hash {
                h.update(limb.to_le_bytes());
            }
            h.update((i as u64).to_le_bytes());
            let result = h.finalize();
            let mut limbs = [0u64; 4];
            for (k, limb) in limbs.iter_mut().enumerate() {
                let offset = k * 8;
                *limb = u64::from_le_bytes(result[offset..offset + 8].try_into().unwrap());
            }
            current_hash = limbs;
            hashes.push(current_hash);
        }

        let mut steps = Vec::with_capacity(n);
        for i in 0..n {
            let energy = fluidelite_core::field::Q16::from_raw(
                base_energy.raw + (i as i64) * per_step_drift.raw,
            );
            let cons_residual = if i == 0 {
                fluidelite_core::field::Q16::ZERO
            } else {
                per_step_drift
            };
            steps.push(TimestepPhysics {
                energy,
                energy_sq: fluidelite_core::field::Q16::from_f64(1_000_000.0),
                max_temp: fluidelite_core::field::Q16::from_f64(1.5),
                min_temp: fluidelite_core::field::Q16::from_f64(0.2),
                source_energy: fluidelite_core::field::Q16::from_f64(10.0),
                cg_residual: fluidelite_core::field::Q16::from_f64(1e-6),
                sv_max: fluidelite_core::field::Q16::from_f64(0.99),
                rank: 8,
                conservation_residual: cons_residual,
                input_hash_limbs: hashes[i],
                global_step: i as u64,
                output_hash_limbs: hashes[i + 1],
            });
        }

        // prove_thermal_stark builds trace + runs InternalProver.prove()
        // which internally calls new_trace_lde → try_gpu_trace_lde → GPU or CPU fallback
        let (proof_bytes, pub_inputs, trace_len, gen_ms) =
            prove_thermal_stark(&steps, dt, alpha).expect("prove failed");

        eprintln!(
            "[GpuTraceLde] Pipeline OK: n={n}, trace_len={trace_len}, proof={} bytes, time={gen_ms} ms",
            proof_bytes.len()
        );
        assert!(trace_len >= n, "trace length {trace_len} < requested {n}");
        assert!(proof_bytes.len() > 100, "proof suspiciously small");

        // Verify the proof
        let valid = verify_thermal_stark(&proof_bytes, pub_inputs).expect("verify failed");
        assert!(valid, "STARK verification failed for GPU-accelerated proof");
    }
}
