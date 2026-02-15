//! Configuration for the Thermal/Heat Equation Proof Circuit
//!
//! Defines physics parameters, circuit sizing, and constraint estimation
//! for the heat equation ZK proof. The heat equation:
//!
//!   ∂T/∂t = α∇²T + S(x,t)
//!
//! where T is temperature, α is thermal diffusivity, and S is a source term.
//!
//! # QTT Heat Equation Solver
//!
//! The solver uses QTT decomposition with implicit time stepping:
//!   (I - α·Δt·L) T^{n+1} = T^n + Δt·S^n
//!
//! where L is the discrete Laplacian MPO. The implicit solve is done via
//! conjugate gradient (CG) in QTT format, identical to the NS-IMEX diffusion.
//!
//! # Boundary Conditions
//!
//! Supported: Dirichlet (fixed temperature), Neumann (fixed flux), periodic.
//! Boundary conditions are encoded in the Laplacian MPO construction.
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use fluidelite_core::field::Q16;
use std::fmt;

// ═════════════════════════════════════════════════════════════════════════════
// Constants
// ═════════════════════════════════════════════════════════════════════════════

/// Q16.16 fixed-point scale factor.
pub const Q16_SCALE: u64 = 65536;

/// Fractional bits in Q16.16.
pub const Q16_FRAC_BITS: u32 = 16;

/// Number of thermal variables (T only; extensions: T, qx, qy, qz for flux).
pub const NUM_THERMAL_VARIABLES: usize = 1;

/// Number of spatial dimensions.
pub const NUM_DIMENSIONS: usize = 3;

/// Physical dimension for binary QTT embedding.
pub const PHYS_DIM: usize = 2;

/// Number of implicit solve stages per timestep.
/// 1. Build RHS: r = T^n + Δt·S^n
/// 2. Implicit solve: (I - α·Δt·L) T^{n+1} = r
/// 3. SVD truncation of result
/// 4. Conservation check
pub const NUM_THERMAL_STAGES: usize = 4;

/// Minimum circuit k for thermal circuits.
pub const MIN_THERMAL_K: usize = 10;

/// Maximum circuit k for thermal circuits.
pub const MAX_THERMAL_K: usize = 25;

/// Rows per fixed-point MAC operation (same as Euler/NS).
pub const ROWS_PER_FP_MAC: usize = 10;

/// Rows per SVD singular value ordering check.
pub const ROWS_PER_SV_ORDER: usize = 9;

/// Rows per conservation check (energy balance).
pub const ROWS_PER_CONSERVATION: usize = 20;

/// Rows per public input binding.
pub const ROWS_PER_PUBLIC_INPUT: usize = 1;

/// Rows per CG solve verification (implicit diffusion step).
pub const ROWS_PER_CG_SOLVE: usize = 15;

/// Rows per boundary condition verification.
pub const ROWS_PER_BC_CHECK: usize = 8;

/// Default thermal diffusivity α in Q16.16: 0.01 → 655.
pub const DEFAULT_DIFFUSIVITY_RAW: i64 = 655;

/// Default CFL number in Q16.16: 0.25 → 16384 (more restrictive for diffusion).
pub const DEFAULT_CFL_RAW: i64 = 16384;

/// Default truncation tolerance in Q16.16: ~1.5e-5 → 1.
pub const DEFAULT_TOLERANCE_RAW: i64 = 1;

/// Default conservation tolerance in Q16.16: ~1.1e-4 → 7.
pub const DEFAULT_CONSERVATION_TOL_RAW: i64 = 7;

/// Default source term magnitude in Q16.16: 0.0 → 0 (no source).
pub const DEFAULT_SOURCE_RAW: i64 = 0;

// ═════════════════════════════════════════════════════════════════════════════
// Enums
// ═════════════════════════════════════════════════════════════════════════════

/// Thermal variables tracked by the solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ThermalVariable {
    /// Temperature field T.
    Temperature,
}

impl ThermalVariable {
    /// All thermal variables in canonical order.
    pub const ALL: [ThermalVariable; NUM_THERMAL_VARIABLES] = [
        ThermalVariable::Temperature,
    ];

    /// Index in the variable array.
    pub fn index(self) -> usize {
        match self {
            ThermalVariable::Temperature => 0,
        }
    }

    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            ThermalVariable::Temperature => "T",
        }
    }
}

impl fmt::Display for ThermalVariable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Boundary condition types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BoundaryCondition {
    /// Fixed temperature: T = T_bc on boundary.
    Dirichlet,
    /// Fixed flux: ∂T/∂n = q_bc on boundary.
    Neumann,
    /// Periodic boundaries.
    Periodic,
}

impl BoundaryCondition {
    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            BoundaryCondition::Dirichlet => "Dirichlet",
            BoundaryCondition::Neumann => "Neumann",
            BoundaryCondition::Periodic => "Periodic",
        }
    }
}

impl fmt::Display for BoundaryCondition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Stages within one thermal timestep.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ThermalStage {
    /// Build RHS: r = T^n + Δt·S^n.
    BuildRhs,
    /// Implicit diffusion solve: (I - α·Δt·L) T^{n+1} = r.
    ImplicitSolve,
    /// SVD truncation of the result.
    SvdTruncation,
    /// Conservation check: energy balance verification.
    ConservationCheck,
}

impl ThermalStage {
    /// All stages in execution order.
    pub const ALL: [ThermalStage; NUM_THERMAL_STAGES] = [
        ThermalStage::BuildRhs,
        ThermalStage::ImplicitSolve,
        ThermalStage::SvdTruncation,
        ThermalStage::ConservationCheck,
    ];

    /// Index in the stage array.
    pub fn index(self) -> usize {
        match self {
            ThermalStage::BuildRhs => 0,
            ThermalStage::ImplicitSolve => 1,
            ThermalStage::SvdTruncation => 2,
            ThermalStage::ConservationCheck => 3,
        }
    }
}

impl fmt::Display for ThermalStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ThermalStage::BuildRhs => write!(f, "build_rhs"),
            ThermalStage::ImplicitSolve => write!(f, "implicit_solve"),
            ThermalStage::SvdTruncation => write!(f, "svd_truncation"),
            ThermalStage::ConservationCheck => write!(f, "conservation_check"),
        }
    }
}

/// QTT operations performed during a thermal timestep.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ThermalQttOperation {
    /// MPO × MPS contraction (Laplacian application).
    MpoContraction,
    /// MPS + MPS addition (RHS assembly).
    MpsAddition,
    /// SVD truncation to bond dimension χ.
    SvdTruncation,
    /// CG solve iteration (one step of conjugate gradient).
    CgIteration,
    /// Boundary condition enforcement.
    BcEnforcement,
}

// ═════════════════════════════════════════════════════════════════════════════
// Parameters
// ═════════════════════════════════════════════════════════════════════════════

/// Full parameter set for the thermal proof circuit.
#[derive(Debug, Clone)]
pub struct ThermalParams {
    /// Log₂ of grid points per dimension: N = 2^grid_bits.
    pub grid_bits: usize,

    /// Maximum QTT bond dimension χ_max.
    pub chi_max: usize,

    /// Thermal diffusivity α in Q16.16.
    pub alpha: Q16,

    /// Time step Δt in Q16.16.
    pub dt: Q16,

    /// CFL number in Q16.16.
    pub cfl: Q16,

    /// SVD truncation tolerance in Q16.16.
    pub tolerance: Q16,

    /// Conservation tolerance in Q16.16.
    pub conservation_tol: Q16,

    /// Boundary condition type.
    pub boundary_condition: BoundaryCondition,

    /// Maximum CG iterations for implicit solve.
    pub max_cg_iterations: usize,

    /// CG convergence tolerance in Q16.16.
    pub cg_tolerance: Q16,

    /// Source term magnitude (uniform) in Q16.16.
    pub source_magnitude: Q16,
}

impl ThermalParams {
    /// Create a small test configuration.
    pub fn test_small() -> Self {
        Self {
            grid_bits: 4,
            chi_max: 4,
            alpha: Q16::from_raw(DEFAULT_DIFFUSIVITY_RAW),
            dt: Q16::from_raw(6554), // ~0.1
            cfl: Q16::from_raw(DEFAULT_CFL_RAW),
            tolerance: Q16::from_raw(DEFAULT_TOLERANCE_RAW),
            conservation_tol: Q16::from_raw(DEFAULT_CONSERVATION_TOL_RAW),
            boundary_condition: BoundaryCondition::Periodic,
            max_cg_iterations: 50,
            cg_tolerance: Q16::from_raw(1),
            source_magnitude: Q16::from_raw(DEFAULT_SOURCE_RAW),
        }
    }

    /// Create a medium configuration for benchmarking.
    pub fn test_medium() -> Self {
        Self {
            grid_bits: 8,
            chi_max: 8,
            alpha: Q16::from_raw(DEFAULT_DIFFUSIVITY_RAW),
            dt: Q16::from_raw(3277), // ~0.05
            cfl: Q16::from_raw(DEFAULT_CFL_RAW),
            tolerance: Q16::from_raw(DEFAULT_TOLERANCE_RAW),
            conservation_tol: Q16::from_raw(DEFAULT_CONSERVATION_TOL_RAW),
            boundary_condition: BoundaryCondition::Periodic,
            max_cg_iterations: 100,
            cg_tolerance: Q16::from_raw(1),
            source_magnitude: Q16::from_raw(DEFAULT_SOURCE_RAW),
        }
    }

    /// Create a production configuration.
    pub fn production() -> Self {
        Self {
            grid_bits: 16,
            chi_max: 32,
            alpha: Q16::from_raw(DEFAULT_DIFFUSIVITY_RAW),
            dt: Q16::from_raw(655), // ~0.01
            cfl: Q16::from_raw(DEFAULT_CFL_RAW),
            tolerance: Q16::from_raw(DEFAULT_TOLERANCE_RAW),
            conservation_tol: Q16::from_raw(DEFAULT_CONSERVATION_TOL_RAW),
            boundary_condition: BoundaryCondition::Dirichlet,
            max_cg_iterations: 200,
            cg_tolerance: Q16::from_raw(1),
            source_magnitude: Q16::from_raw(DEFAULT_SOURCE_RAW),
        }
    }

    /// Number of QTT sites: grid_bits × NUM_DIMENSIONS.
    pub fn num_sites(&self) -> usize {
        self.grid_bits * NUM_DIMENSIONS
    }

    /// Total grid points: 2^grid_bits per dimension, 3D.
    pub fn total_grid_points(&self) -> usize {
        1usize << (self.grid_bits * NUM_DIMENSIONS)
    }

    /// SHA-256 hash of the parameters (used as public input commitment).
    pub fn hash(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(b"THERMAL_PARAMS_V1");
        hasher.update(self.alpha.raw.to_le_bytes());
        hasher.update(self.dt.raw.to_le_bytes());
        hasher.update(self.cfl.raw.to_le_bytes());
        hasher.update((self.grid_bits as u64).to_le_bytes());
        hasher.update((self.chi_max as u64).to_le_bytes());
        hasher.update(self.tolerance.raw.to_le_bytes());
        hasher.update(self.conservation_tol.raw.to_le_bytes());
        hasher.update(self.source_magnitude.raw.to_le_bytes());
        hasher.update((self.max_cg_iterations as u64).to_le_bytes());
        hasher.update(self.cg_tolerance.raw.to_le_bytes());
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }

    /// Validate parameters for circuit construction.
    pub fn validate(&self) -> Result<(), String> {
        if self.grid_bits == 0 || self.grid_bits > 20 {
            return Err(format!(
                "grid_bits must be in [1, 20], got {}",
                self.grid_bits
            ));
        }
        if self.chi_max == 0 || self.chi_max > 256 {
            return Err(format!(
                "chi_max must be in [1, 256], got {}",
                self.chi_max
            ));
        }
        if self.alpha.raw <= 0 {
            return Err(format!(
                "thermal diffusivity α must be > 0, got {}",
                self.alpha.to_f64()
            ));
        }
        if self.dt.raw <= 0 {
            return Err(format!("timestep dt must be > 0, got {}", self.dt.to_f64()));
        }
        if self.max_cg_iterations == 0 {
            return Err("max_cg_iterations must be > 0".to_string());
        }
        Ok(())
    }
}

impl fmt::Display for ThermalParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ThermalParams(grid=2^{}, χ={}, α={:.4}, Δt={:.4}, BC={}, CG_max={})",
            self.grid_bits,
            self.chi_max,
            self.alpha.to_f64(),
            self.dt.to_f64(),
            self.boundary_condition,
            self.max_cg_iterations,
        )
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Circuit Sizing
// ═════════════════════════════════════════════════════════════════════════════

/// Circuit sizing for the thermal proof.
#[derive(Debug, Clone)]
pub struct ThermalCircuitSizing {
    /// Circuit k parameter: 2^k rows in the circuit table.
    pub k: u32,

    /// Number of rows consumed by the thermal circuit.
    pub total_rows: usize,

    /// Estimated constraint count.
    pub estimated_constraints: usize,

    /// Breakdown by operation type.
    pub breakdown: ThermalCircuitBreakdown,
}

/// Constraint breakdown by operation.
#[derive(Debug, Clone)]
pub struct ThermalCircuitBreakdown {
    /// RHS assembly: MPO × MPS + source addition.
    pub rhs_assembly: usize,
    /// Implicit solve: CG iteration verification.
    pub implicit_solve: usize,
    /// SVD truncation ordering.
    pub svd_truncation: usize,
    /// Truncation error bound (Σσᵢ² ≤ ε²).
    pub truncation_error_bound: usize,
    /// Conservation checks.
    pub conservation: usize,
    /// Boundary condition enforcement.
    pub boundary_conditions: usize,
    /// Public input binding.
    pub public_inputs: usize,
}

impl ThermalCircuitSizing {
    /// Compute circuit sizing from parameters.
    pub fn from_params(params: &ThermalParams) -> Self {
        let n_sites = params.num_sites();
        let chi = params.chi_max;
        let d = PHYS_DIM;

        // RHS assembly: Laplacian MPO × T^n + source term
        // MPO contraction: n_sites × chi² × d² MACs
        let rhs_macs = n_sites * chi * chi * d * d;
        let rhs_rows = rhs_macs * ROWS_PER_FP_MAC;

        // Implicit CG solve: max_cg_iterations × (residual + direction update)
        // Each CG step: 1 MPO×MPS contraction + 2 dot products + 1 vector update
        let cg_macs_per_iter = n_sites * chi * chi * d * d + 2 * n_sites * chi * d;
        let cg_rows = params.max_cg_iterations * cg_macs_per_iter * ROWS_PER_FP_MAC / 10;
        // /10: CG verification is cheaper (we verify residual bound, not every step)

        // SVD truncation ordering: chi singular values per site
        let svd_rows = n_sites * chi * ROWS_PER_SV_ORDER;

        // Truncation error bound (Task 6.13):
        // MAC chain init (1 row) + per-truncated-SV (MAC + range check = 6 rows)
        // + nonneg check for bound (9 rows).
        // Worst case: chi_out = chi × Laplacian_bond_dim ≈ 4χ, truncated count ≈ 3χ per bond.
        // Conservative estimate: n_bonds × 3χ truncated SVs + overhead.
        let n_bonds = if n_sites > 0 { n_sites - 1 } else { 0 };
        let max_trunc_svs = n_bonds * chi * 3; // upper bound on truncated SVs
        let trunc_error_rows = 1 + max_trunc_svs * (1 + 5) + 9; // init + MACs + nonneg

        // Conservation: energy balance |∫T^{n+1} - ∫T^n - Δt·∫S| ≤ ε
        let conservation_rows = NUM_THERMAL_VARIABLES * ROWS_PER_CONSERVATION;

        // Boundary condition checks (per boundary site)
        let bc_rows = n_sites * ROWS_PER_BC_CHECK / 4; // Only boundary sites

        // Public inputs: 4 hash limbs × 3 hashes + dt + α + chi + grid_bits + residual
        let public_input_rows = (4 * 3 + 5) * ROWS_PER_PUBLIC_INPUT;

        let total_rows = rhs_rows + cg_rows + svd_rows + trunc_error_rows
            + conservation_rows + bc_rows + public_input_rows;

        // k: smallest power of 2 that fits all rows + headroom
        let k = ((total_rows as f64).log2().ceil() as u32 + 1)
            .max(MIN_THERMAL_K as u32)
            .min(MAX_THERMAL_K as u32);

        Self {
            k,
            total_rows,
            estimated_constraints: total_rows,
            breakdown: ThermalCircuitBreakdown {
                rhs_assembly: rhs_rows,
                implicit_solve: cg_rows,
                svd_truncation: svd_rows,
                truncation_error_bound: trunc_error_rows,
                conservation: conservation_rows,
                boundary_conditions: bc_rows,
                public_inputs: public_input_rows,
            },
        }
    }

    /// Estimate total constraints for this sizing.
    pub fn estimate_constraints(&self) -> usize {
        self.estimated_constraints
    }
}

impl fmt::Display for ThermalCircuitSizing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ThermalCircuitSizing(k={}, rows={}, constraints={})",
            self.k, self.total_rows, self.estimated_constraints
        )
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════════════
