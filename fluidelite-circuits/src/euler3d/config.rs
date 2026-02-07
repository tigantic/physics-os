//! Configuration for the Euler 3D Proof Circuit
//!
//! Defines physics parameters, circuit sizing, and constraint estimation
//! for the compressible Euler equations ZK proof.
//!
//! # Design Rationale
//!
//! The Euler 3D solver operates on 5 conserved variables (ρ, ρu, ρv, ρw, E)
//! using Strang splitting with 5 directional stages per timestep. Each stage
//! applies shift MPOs and truncates via SVD. The circuit proves every operation.
//!
//! Because QTT operations are O(r³ log N) instead of O(N³), the circuit is
//! exponentially smaller than a dense CFD proof would be.
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use std::fmt;

use fluidelite_core::field::Q16;

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

/// Q16.16 scale factor as u64 for circuit constants.
pub const Q16_SCALE: u64 = 1u64 << 16;

/// Number of bits in Q16 fractional part (for range check decomposition).
pub const Q16_FRAC_BITS: usize = 16;

/// Number of conserved variables in 3D Euler equations.
pub const NUM_CONSERVED_VARIABLES: usize = 5;

/// Number of Strang splitting stages per timestep.
pub const NUM_STRANG_STAGES: usize = 5;

/// Number of spatial dimensions.
pub const NUM_DIMENSIONS: usize = 3;

/// Physical dimension for binary QTT embedding.
pub const PHYS_DIM: usize = 2;

/// Minimum circuit k parameter for valid euler3d circuits.
pub const MIN_EULER3D_K: u32 = 10;

/// Maximum circuit k parameter to prevent OOM.
pub const MAX_EULER3D_K: u32 = 26;

/// Default gamma for ideal gas (1.4).
pub const DEFAULT_GAMMA_RAW: i64 = 91750; // Q16 representation of 1.4

/// Default CFL number (0.5).
pub const DEFAULT_CFL_RAW: i64 = 32768; // Q16 representation of 0.5

/// Default SVD truncation tolerance (1e-6 ≈ 0 in Q16 due to limited precision).
pub const DEFAULT_TOLERANCE_RAW: i64 = 1; // Smallest non-zero Q16 value

/// Default conservation tolerance in Q16.
pub const DEFAULT_CONSERVATION_TOL_RAW: i64 = 7; // ~1e-4 in Q16

/// Rows consumed per fixed-point MAC operation in the circuit.
/// 1 MAC row + 16 boolean rows (range check) + 1 recompose = 18.
/// With parallel decomposition (2 per row using 4 advice cols): 1 + 8 + 1 = 10.
pub const ROWS_PER_FP_MAC: usize = 10;

/// Rows per SVD ordering constraint (per singular value pair).
/// 1 subtraction + 8 rows range check = 9.
pub const ROWS_PER_SV_ORDER: usize = 9;

/// Rows per conservation check (sum + comparison).
pub const ROWS_PER_CONSERVATION: usize = 20;

/// Rows per public input binding.
pub const ROWS_PER_PUBLIC_INPUT: usize = 1;

// ═══════════════════════════════════════════════════════════════════════════
// Enumerations
// ═══════════════════════════════════════════════════════════════════════════

/// Conserved variable in the 3D Euler equations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConservedVariable {
    /// Mass density ρ
    Density,
    /// X-momentum ρu
    MomentumX,
    /// Y-momentum ρv
    MomentumY,
    /// Z-momentum ρw
    MomentumZ,
    /// Total energy E = ρe + 0.5ρ(u²+v²+w²)
    Energy,
}

impl ConservedVariable {
    /// All conserved variables in canonical order.
    pub const ALL: [ConservedVariable; NUM_CONSERVED_VARIABLES] = [
        ConservedVariable::Density,
        ConservedVariable::MomentumX,
        ConservedVariable::MomentumY,
        ConservedVariable::MomentumZ,
        ConservedVariable::Energy,
    ];

    /// Index of this variable (0-4).
    pub fn index(self) -> usize {
        match self {
            ConservedVariable::Density => 0,
            ConservedVariable::MomentumX => 1,
            ConservedVariable::MomentumY => 2,
            ConservedVariable::MomentumZ => 3,
            ConservedVariable::Energy => 4,
        }
    }
}

impl fmt::Display for ConservedVariable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConservedVariable::Density => write!(f, "ρ"),
            ConservedVariable::MomentumX => write!(f, "ρu"),
            ConservedVariable::MomentumY => write!(f, "ρv"),
            ConservedVariable::MomentumZ => write!(f, "ρw"),
            ConservedVariable::Energy => write!(f, "E"),
        }
    }
}

/// Strang splitting stage.
///
/// The Strang splitting for 3D is:
///   L_x(Δt/2) ∘ L_y(Δt/2) ∘ L_z(Δt) ∘ L_y(Δt/2) ∘ L_x(Δt/2)
/// This ensures second-order temporal accuracy via symmetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StrangStage {
    /// X half-step (first)
    XHalf1,
    /// Y half-step (first)
    YHalf1,
    /// Z full step
    ZFull,
    /// Y half-step (second)
    YHalf2,
    /// X half-step (second)
    XHalf2,
}

impl StrangStage {
    /// All stages in execution order.
    pub const ALL: [StrangStage; NUM_STRANG_STAGES] = [
        StrangStage::XHalf1,
        StrangStage::YHalf1,
        StrangStage::ZFull,
        StrangStage::YHalf2,
        StrangStage::XHalf2,
    ];

    /// The spatial axis this stage operates on (0=X, 1=Y, 2=Z).
    pub fn axis(self) -> usize {
        match self {
            StrangStage::XHalf1 | StrangStage::XHalf2 => 0,
            StrangStage::YHalf1 | StrangStage::YHalf2 => 1,
            StrangStage::ZFull => 2,
        }
    }

    /// Whether this stage uses a half timestep (true) or full timestep (false).
    pub fn is_half_step(self) -> bool {
        !matches!(self, StrangStage::ZFull)
    }

    /// The stage index in the Strang sequence (0-4).
    pub fn index(self) -> usize {
        match self {
            StrangStage::XHalf1 => 0,
            StrangStage::YHalf1 => 1,
            StrangStage::ZFull => 2,
            StrangStage::YHalf2 => 3,
            StrangStage::XHalf2 => 4,
        }
    }
}

impl fmt::Display for StrangStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StrangStage::XHalf1 => write!(f, "L_x(Δt/2) [1st]"),
            StrangStage::YHalf1 => write!(f, "L_y(Δt/2) [1st]"),
            StrangStage::ZFull => write!(f, "L_z(Δt)"),
            StrangStage::YHalf2 => write!(f, "L_y(Δt/2) [2nd]"),
            StrangStage::XHalf2 => write!(f, "L_x(Δt/2) [2nd]"),
        }
    }
}

/// QTT operation type within a single directional sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QttOperation {
    /// Apply shift MPO to compute flux differences.
    ShiftMpoApply,
    /// Subtract original state from shifted state.
    QttSubtract,
    /// Scale flux difference by -Δt/Δx coefficient.
    QttScale,
    /// Add scaled flux to original state.
    QttAdd,
    /// SVD-based truncation to chi_max.
    SvdTruncate,
}

impl QttOperation {
    /// All operations in a single directional sweep.
    pub const SWEEP_OPS: [QttOperation; 5] = [
        QttOperation::ShiftMpoApply,
        QttOperation::QttSubtract,
        QttOperation::QttScale,
        QttOperation::QttAdd,
        QttOperation::SvdTruncate,
    ];
}

// ═══════════════════════════════════════════════════════════════════════════
// Physics Parameters
// ═══════════════════════════════════════════════════════════════════════════

/// Physics parameters for the Euler 3D solver.
///
/// These are public inputs to the circuit — the verifier knows the physical
/// setup being simulated.
#[derive(Debug, Clone)]
pub struct Euler3DParams {
    /// Heat capacity ratio γ (typically 1.4 for air).
    pub gamma: Q16,

    /// CFL number for timestep control (typically 0.3–0.8).
    pub cfl: Q16,

    /// Grid bits per dimension: grid size = 2^grid_bits per axis.
    pub grid_bits: usize,

    /// Maximum QTT bond dimension χ_max.
    pub chi_max: usize,

    /// SVD truncation tolerance ε.
    pub tolerance: Q16,

    /// Conservation error tolerance ε_cons.
    pub conservation_tolerance: Q16,

    /// MPO bond dimension for shift operators (typically 1).
    pub mpo_bond_dim: usize,

    /// Timestep Δt in Q16.16 format.
    pub dt: Q16,

    /// Grid spacing Δx in Q16.16 format.
    pub dx: Q16,
}

impl Euler3DParams {
    /// Create default physics parameters for testing.
    pub fn test_small() -> Self {
        Self {
            gamma: Q16::from_raw(DEFAULT_GAMMA_RAW),
            cfl: Q16::from_raw(DEFAULT_CFL_RAW),
            grid_bits: 4,
            chi_max: 4,
            tolerance: Q16::from_f64(0.001), // Relaxed for test (vs 1 ULP for production)
            conservation_tolerance: Q16::from_f64(0.01), // Relaxed for test
            mpo_bond_dim: 1,
            dt: Q16::from_f64(0.001),
            dx: Q16::from_f64(1.0 / 16.0), // 2^4 = 16 grid points
        }
    }

    /// Create production physics parameters.
    pub fn production() -> Self {
        Self {
            gamma: Q16::from_raw(DEFAULT_GAMMA_RAW),
            cfl: Q16::from_raw(DEFAULT_CFL_RAW),
            grid_bits: 16,
            chi_max: 32,
            tolerance: Q16::from_raw(DEFAULT_TOLERANCE_RAW),
            conservation_tolerance: Q16::from_raw(DEFAULT_CONSERVATION_TOL_RAW),
            mpo_bond_dim: 1,
            dt: Q16::from_f64(0.0001),
            dx: Q16::from_f64(1.0 / 65536.0), // 2^16 = 65536 grid points
        }
    }

    /// Number of QTT sites (= grid_bits).
    pub fn num_sites(&self) -> usize {
        self.grid_bits
    }

    /// Compute the CFL-limited timestep coefficient: dt / dx.
    pub fn dt_over_dx(&self) -> Q16 {
        if self.dx.raw == 0 {
            Q16::zero()
        } else {
            // Fixed-point division: (dt.raw << FRAC_BITS) / dx.raw
            let numerator = (self.dt.raw as i128) << Q16_FRAC_BITS;
            let result = (numerator / self.dx.raw as i128) as i64;
            Q16::from_raw(result)
        }
    }

    /// SHA-256 hash of the parameter set (used as public input).
    pub fn hash(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(b"EULER3D_PARAMS_V1");
        hasher.update(self.gamma.raw.to_le_bytes());
        hasher.update(self.cfl.raw.to_le_bytes());
        hasher.update((self.grid_bits as u64).to_le_bytes());
        hasher.update((self.chi_max as u64).to_le_bytes());
        hasher.update(self.tolerance.raw.to_le_bytes());
        hasher.update(self.conservation_tolerance.raw.to_le_bytes());
        hasher.update((self.mpo_bond_dim as u64).to_le_bytes());
        hasher.update(self.dt.raw.to_le_bytes());
        hasher.update(self.dx.raw.to_le_bytes());
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }
}

impl Default for Euler3DParams {
    fn default() -> Self {
        Self::test_small()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Circuit Sizing
// ═══════════════════════════════════════════════════════════════════════════

/// Circuit sizing and row estimation for the Euler 3D proof.
#[derive(Debug, Clone)]
pub struct Euler3DCircuitSizing {
    /// Number of QTT sites (= grid_bits).
    pub num_sites: usize,

    /// Maximum bond dimension χ_max.
    pub chi_max: usize,

    /// Physical dimension d (2 for binary QTT).
    pub phys_dim: usize,

    /// MPO bond dimension D (typically 1).
    pub mpo_bond_dim: usize,

    /// Number of conserved variables (5).
    pub num_variables: usize,

    /// Number of Strang stages (5).
    pub num_strang_stages: usize,

    /// Circuit k parameter: total rows = 2^k.
    pub k: u32,
}

impl Euler3DCircuitSizing {
    /// Create sizing from physics parameters.
    pub fn from_params(params: &Euler3DParams) -> Self {
        let num_sites = params.num_sites();
        let chi_max = params.chi_max;
        let phys_dim = PHYS_DIM;
        let mpo_bond_dim = params.mpo_bond_dim;
        let num_variables = NUM_CONSERVED_VARIABLES;
        let num_strang_stages = NUM_STRANG_STAGES;

        let total_rows = Self::estimate_total_rows(
            num_sites,
            chi_max,
            phys_dim,
            mpo_bond_dim,
            num_variables,
            num_strang_stages,
        );

        // k must satisfy: 2^k >= total_rows
        let k = Self::compute_k(total_rows);

        Self {
            num_sites,
            chi_max,
            phys_dim,
            mpo_bond_dim,
            num_variables,
            num_strang_stages,
            k,
        }
    }

    /// Compute minimum k for given row count.
    fn compute_k(total_rows: usize) -> u32 {
        if total_rows == 0 {
            return MIN_EULER3D_K;
        }
        let k = ((total_rows as f64).log2().ceil() as u32) + 1;
        k.clamp(MIN_EULER3D_K, MAX_EULER3D_K)
    }

    /// Estimate total rows needed for the circuit.
    fn estimate_total_rows(
        num_sites: usize,
        chi_max: usize,
        phys_dim: usize,
        mpo_bond_dim: usize,
        num_variables: usize,
        num_strang_stages: usize,
    ) -> usize {
        let macs_per_contraction =
            Self::macs_per_contraction(num_sites, chi_max, phys_dim, mpo_bond_dim);
        let contractions_per_timestep = num_strang_stages * num_variables;
        let total_macs = contractions_per_timestep * macs_per_contraction;

        // QTT add/subtract/scale: roughly same as contraction (same core structure)
        let total_add_sub_scale = contractions_per_timestep * 3 * num_sites * chi_max * phys_dim;

        // SVD truncation: ordering checks per truncation
        let truncations = contractions_per_timestep;
        let sv_ordering_rows = truncations * chi_max * ROWS_PER_SV_ORDER;

        // Conservation checks: one per variable per timestep
        let conservation_rows = num_variables * ROWS_PER_CONSERVATION;

        // Public inputs
        let public_input_rows = Self::num_public_inputs(num_variables) * ROWS_PER_PUBLIC_INPUT;

        // Range table loading (for bit decomposition)
        let range_table_rows = 0; // Using in-circuit bit decomposition, no separate table

        // Total
        let mac_rows = total_macs * ROWS_PER_FP_MAC;
        let add_rows = total_add_sub_scale; // Additions are 1 row each (no range check needed)

        mac_rows + add_rows + sv_ordering_rows + conservation_rows
            + public_input_rows + range_table_rows
    }

    /// Number of MAC operations per single MPO×MPS contraction.
    ///
    /// For each site: output has (χ_l × D_l) × d_out × (χ_r × D_r) elements.
    /// Each output element requires d_in MAC operations.
    /// With D=1, d=2: 4χ² MACs per site, total = L × 4χ².
    pub fn macs_per_contraction(
        num_sites: usize,
        chi_max: usize,
        phys_dim: usize,
        mpo_bond_dim: usize,
    ) -> usize {
        let chi_d = chi_max * mpo_bond_dim;
        num_sites * chi_d * chi_d * phys_dim * phys_dim
    }

    /// Estimate total arithmetic constraints.
    pub fn estimate_constraints(&self) -> usize {
        let macs_per_contraction = Self::macs_per_contraction(
            self.num_sites,
            self.chi_max,
            self.phys_dim,
            self.mpo_bond_dim,
        );
        let contractions_per_timestep = self.num_strang_stages * self.num_variables;

        // Each MAC contributes ~3 constraints (mul + range check decomposition + accumulate)
        let mac_constraints = contractions_per_timestep * macs_per_contraction * 3;

        // SVD ordering: 1 constraint per adjacent pair
        let sv_constraints = contractions_per_timestep * self.chi_max;

        // Conservation: ~5 constraints per variable
        let conservation_constraints = self.num_variables * 5;

        mac_constraints + sv_constraints + conservation_constraints
    }

    /// Number of public inputs for the circuit.
    pub fn num_public_inputs(num_variables: usize) -> usize {
        // input_state_hash (4 field elements for 256-bit hash)
        // + output_state_hash (4 field elements)
        // + params_hash (4 field elements)
        // + conservation_residuals (num_variables field elements)
        // + dt (1 field element)
        // + chi_max (1 field element)
        // + grid_bits (1 field element)
        4 + 4 + 4 + num_variables + 3
    }

    /// Estimated proof generation time in milliseconds on CPU.
    pub fn estimate_proof_time_ms(&self) -> f64 {
        let constraints = self.estimate_constraints() as f64;
        // Halo2 CPU: ~100ns per constraint for MSM (conservative estimate)
        let msm_ns = constraints * 100.0;
        // Polynomial overhead: +50%
        let total_ns = msm_ns * 1.5;
        total_ns / 1_000_000.0
    }

    /// Estimated proof size in bytes.
    pub fn estimate_proof_size_bytes(&self) -> usize {
        // Halo2/KZG proof size is roughly constant: ~800-1200 bytes
        // Independent of circuit size (logarithmic in constraints)
        let num_commitments = 30; // Approximate for this circuit shape
        let commitment_size = 32; // BN254 G1 point compressed
        let opening_proof_size = 128;
        num_commitments * commitment_size + opening_proof_size
    }

    /// Create test sizing with small parameters.
    pub fn test_small() -> Self {
        Self::from_params(&Euler3DParams::test_small())
    }

    /// Create production sizing.
    pub fn production() -> Self {
        Self::from_params(&Euler3DParams::production())
    }
}

impl Default for Euler3DCircuitSizing {
    fn default() -> Self {
        Self::test_small()
    }
}

impl fmt::Display for Euler3DCircuitSizing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Euler3D Circuit: L={}, χ={}, d={}, D={}, k={} (2^{} = {} rows)\n\
             Variables: {}, Strang stages: {}\n\
             Estimated constraints: {}\n\
             Estimated proof time: {:.1} ms\n\
             Estimated proof size: {} bytes",
            self.num_sites,
            self.chi_max,
            self.phys_dim,
            self.mpo_bond_dim,
            self.k,
            self.k,
            1usize << self.k,
            self.num_variables,
            self.num_strang_stages,
            self.estimate_constraints(),
            self.estimate_proof_time_ms(),
            self.estimate_proof_size_bytes(),
        )
    }
}
