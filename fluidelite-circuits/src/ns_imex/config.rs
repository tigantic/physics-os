//! NS-IMEX Configuration — Parameters, Sizing, and Constants
//!
//! Configuration for the Navier-Stokes IMEX (Implicit-Explicit) ZK proof
//! circuit. The IMEX scheme splits the NS equations into:
//!   - Explicit part: advection (nonlinear convective terms)
//!   - Implicit part: diffusion (linear viscous terms)
//!   - Projection: pressure Poisson solve + velocity correction
//!
//! This mirrors the Python `NativeNS3DSolver` which uses:
//!   ∂ω/∂t + (u·∇)ω = ν∇²ω + (ω·∇)u   (vorticity equation)
//!   ∇²ψ = -ω                              (stream function Poisson)
//!   u = ∇×ψ                                (velocity recovery)
//!
//! In the incompressible formulation we track 3 velocity components (u,v,w)
//! or equivalently 3 vorticity components (ωx,ωy,ωz). The pressure is
//! recovered via projection (Chorin splitting).
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

/// Number of velocity/vorticity components (u, v, w) or (ωx, ωy, ωz).
pub const NUM_NS_VARIABLES: usize = 3;

/// Number of IMEX stages per timestep:
///   1. Explicit advection half-step
///   2. Implicit diffusion full-step
///   3. Explicit advection half-step
///   4. Projection (pressure solve + velocity correction)
pub const NUM_IMEX_STAGES: usize = 4;

/// Number of spatial dimensions.
pub const NUM_DIMENSIONS: usize = 3;

/// Physical dimension of QTT site tensors.
pub const PHYS_DIM: usize = 2;

/// Minimum Halo2 circuit degree k for NS-IMEX.
pub const MIN_NS_IMEX_K: usize = 11;

/// Maximum Halo2 circuit degree k for NS-IMEX.
pub const MAX_NS_IMEX_K: usize = 27;

/// Rows per fixed-point MAC operation (same as Euler 3D).
pub const ROWS_PER_FP_MAC: usize = 10;

/// Rows per singular value ordering check.
pub const ROWS_PER_SV_ORDER: usize = 9;

/// Rows per conservation check.
pub const ROWS_PER_CONSERVATION: usize = 20;

/// Rows per public input binding.
pub const ROWS_PER_PUBLIC_INPUT: usize = 1;

/// Rows per diffusion solve verification (implicit step).
/// Verifies: (I - ν·Δt·L) u^{n+1} = u^* where L is the Laplacian MPO.
pub const ROWS_PER_DIFFUSION_SOLVE: usize = 15;

/// Rows per projection step verification.
/// Verifies: ∇·u^{n+1} ≈ 0 (divergence-free to tolerance).
pub const ROWS_PER_PROJECTION: usize = 12;

/// Default kinematic viscosity (ν) in Q16.16: 0.01 → 655.
pub const DEFAULT_VISCOSITY_RAW: i64 = 655;

/// Default Reynolds number: Re = U*L/ν.
pub const DEFAULT_REYNOLDS_NUMBER: u32 = 100;

/// Default CFL number in Q16.16: 0.5 → 32768.
pub const DEFAULT_CFL_RAW: i64 = 32768;

/// Default tolerance in Q16.16: ~1.5e-5 → 1.
pub const DEFAULT_TOLERANCE_RAW: i64 = 1;

/// Default conservation tolerance in Q16.16: ~1.1e-4 → 7.
pub const DEFAULT_CONSERVATION_TOL_RAW: i64 = 7;

/// Default divergence tolerance in Q16.16: ~1.5e-5 → 1.
pub const DEFAULT_DIVERGENCE_TOL_RAW: i64 = 1;

// ═════════════════════════════════════════════════════════════════════════════
// Enums
// ═════════════════════════════════════════════════════════════════════════════

/// Velocity/vorticity components tracked by the NS solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NSVariable {
    /// X-component of velocity (u) or vorticity (ωx).
    VelocityX = 0,
    /// Y-component of velocity (v) or vorticity (ωy).
    VelocityY = 1,
    /// Z-component of velocity (w) or vorticity (ωz).
    VelocityZ = 2,
}

impl NSVariable {
    /// Return all variable variants in order.
    pub fn all() -> [Self; NUM_NS_VARIABLES] {
        [Self::VelocityX, Self::VelocityY, Self::VelocityZ]
    }

    /// Variable index (0..2).
    pub fn index(self) -> usize {
        self as usize
    }
}

impl fmt::Display for NSVariable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VelocityX => write!(f, "u"),
            Self::VelocityY => write!(f, "v"),
            Self::VelocityZ => write!(f, "w"),
        }
    }
}

/// IMEX time integration stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IMEXStage {
    /// Explicit advection half-step: u* = u^n + (Δt/2)·A(u^n)
    AdvectionHalf1 = 0,
    /// Implicit diffusion full-step: (I - ν·Δt·L) u** = u*
    DiffusionFull = 1,
    /// Explicit advection half-step: u*** = u** + (Δt/2)·A(u**)
    AdvectionHalf2 = 2,
    /// Projection: u^{n+1} = u*** - Δt·∇p, where ∇²p = (1/Δt)·∇·u***
    Projection = 3,
}

impl IMEXStage {
    /// Return all stages in execution order.
    pub fn all() -> [Self; NUM_IMEX_STAGES] {
        [
            Self::AdvectionHalf1,
            Self::DiffusionFull,
            Self::AdvectionHalf2,
            Self::Projection,
        ]
    }

    /// Stage index (0..3).
    pub fn index(self) -> usize {
        self as usize
    }

    /// Whether this stage is explicit (advection) or implicit (diffusion/projection).
    pub fn is_explicit(self) -> bool {
        matches!(self, Self::AdvectionHalf1 | Self::AdvectionHalf2)
    }

    /// Whether this stage is the implicit diffusion solve.
    pub fn is_implicit(self) -> bool {
        matches!(self, Self::DiffusionFull)
    }

    /// Whether this stage is the pressure projection.
    pub fn is_projection(self) -> bool {
        matches!(self, Self::Projection)
    }
}

impl fmt::Display for IMEXStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AdvectionHalf1 => write!(f, "AdvHalf1"),
            Self::DiffusionFull => write!(f, "DiffFull"),
            Self::AdvectionHalf2 => write!(f, "AdvHalf2"),
            Self::Projection => write!(f, "Project"),
        }
    }
}

/// QTT operations performed in the NS-IMEX solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NsQttOperation {
    /// Shift MPO application (for finite differences: ∂/∂x, ∂/∂y, ∂/∂z).
    ShiftMpoApply,
    /// SVD truncation after contraction.
    SvdTruncation,
    /// QTT rounding (global rank reduction).
    QttRounding,
    /// Cross product for vorticity: ω = ∇ × u.
    CrossProduct,
    /// Laplacian MPO application: ∇²u (for diffusion).
    LaplacianMpoApply,
    /// Conjugate gradient iteration (for pressure Poisson solve).
    ConjugateGradientStep,
    /// Divergence computation: ∇·u.
    DivergenceCompute,
    /// Gradient computation: ∇p (for projection).
    GradientCompute,
}

impl NsQttOperation {
    /// Estimated constraint rows per operation instance.
    pub fn rows_per_instance(self) -> usize {
        match self {
            Self::ShiftMpoApply => ROWS_PER_FP_MAC,
            Self::SvdTruncation => ROWS_PER_SV_ORDER,
            Self::QttRounding => ROWS_PER_SV_ORDER,
            Self::CrossProduct => ROWS_PER_FP_MAC * 2,  // Two MAC chains
            Self::LaplacianMpoApply => ROWS_PER_FP_MAC * 3, // 3 second-derivatives
            Self::ConjugateGradientStep => ROWS_PER_FP_MAC * 2, // Inner products
            Self::DivergenceCompute => ROWS_PER_FP_MAC,
            Self::GradientCompute => ROWS_PER_FP_MAC,
        }
    }
}

impl fmt::Display for NsQttOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShiftMpoApply => write!(f, "ShiftMPO"),
            Self::SvdTruncation => write!(f, "SVDTrunc"),
            Self::QttRounding => write!(f, "QTTRound"),
            Self::CrossProduct => write!(f, "Cross×"),
            Self::LaplacianMpoApply => write!(f, "Laplacian"),
            Self::ConjugateGradientStep => write!(f, "CGStep"),
            Self::DivergenceCompute => write!(f, "Div"),
            Self::GradientCompute => write!(f, "Grad"),
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Parameters
// ═════════════════════════════════════════════════════════════════════════════

/// NS-IMEX solver parameters committed to the ZK proof.
#[derive(Debug, Clone)]
pub struct NSIMEXParams {
    /// Grid bits per dimension (L). Grid size N = 2^L.
    pub grid_bits: usize,
    /// Maximum MPS bond dimension (χ_max).
    pub chi_max: usize,
    /// SVD truncation tolerance in Q16.16.
    pub tolerance: Q16,
    /// Conservation tolerance in Q16.16.
    pub conservation_tolerance: Q16,
    /// Divergence-free tolerance in Q16.16.
    pub divergence_tolerance: Q16,
    /// MPO bond dimension for shift/Laplacian operators.
    pub mpo_bond_dim: usize,
    /// Kinematic viscosity ν in Q16.16.
    pub viscosity: Q16,
    /// CFL number in Q16.16.
    pub cfl: Q16,
    /// Timestep Δt in Q16.16 (computed from CFL if zero).
    pub dt: Q16,
    /// Grid spacing Δx in Q16.16 (= L_domain / N, uniform).
    pub dx: Q16,
    /// Maximum conjugate gradient iterations for pressure Poisson solve.
    pub max_cg_iterations: usize,
    /// CG convergence tolerance in Q16.16.
    pub cg_tolerance: Q16,
}

impl NSIMEXParams {
    /// Number of QTT sites = grid_bits × NUM_DIMENSIONS (3D).
    pub fn num_sites(&self) -> usize {
        self.grid_bits * NUM_DIMENSIONS
    }

    /// Total grid points N = 2^grid_bits.
    pub fn grid_size(&self) -> usize {
        1 << self.grid_bits
    }

    /// Reynolds number estimate Re = (1/ν) (assuming U~1, L~1).
    pub fn reynolds_number(&self) -> f64 {
        let nu = self.viscosity.to_f64();
        if nu.abs() < 1e-15 { f64::INFINITY } else { 1.0 / nu }
    }

    /// Create minimal test parameters.
    pub fn test_small() -> Self {
        Self {
            grid_bits: 4,
            chi_max: 4,
            tolerance: Q16::from_f64(0.001),
            conservation_tolerance: Q16::from_f64(0.01),
            divergence_tolerance: Q16::from_f64(0.001),
            mpo_bond_dim: 2,
            viscosity: Q16::from_f64(0.01),
            cfl: Q16::from_f64(0.5),
            dt: Q16::from_f64(0.001),
            dx: Q16::from_f64(0.0625), // 1.0 / 16
            max_cg_iterations: 10,
            cg_tolerance: Q16::from_f64(0.001),
        }
    }

    /// Create production parameters.
    pub fn production() -> Self {
        Self {
            grid_bits: 16,
            chi_max: 32,
            tolerance: Q16::from_f64(1e-5),
            conservation_tolerance: Q16::from_f64(1e-4),
            divergence_tolerance: Q16::from_f64(1e-5),
            mpo_bond_dim: 4,
            viscosity: Q16::from_f64(0.001),
            cfl: Q16::from_f64(0.5),
            dt: Q16::from_f64(0.0),  // Auto-compute from CFL
            dx: Q16::from_f64(0.0),  // Auto-compute from grid_bits
            max_cg_iterations: 200,
            cg_tolerance: Q16::from_f64(1e-8),
        }
    }

    /// Compute SHA-256 hash of parameters for public input commitment.
    pub fn hash(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(b"NSIMEX_PARAMS_V1");
        hasher.update(self.grid_bits.to_le_bytes());
        hasher.update(self.chi_max.to_le_bytes());
        hasher.update(self.tolerance.raw.to_le_bytes());
        hasher.update(self.conservation_tolerance.raw.to_le_bytes());
        hasher.update(self.divergence_tolerance.raw.to_le_bytes());
        hasher.update(self.mpo_bond_dim.to_le_bytes());
        hasher.update(self.viscosity.raw.to_le_bytes());
        hasher.update(self.cfl.raw.to_le_bytes());
        hasher.update(self.dt.raw.to_le_bytes());
        hasher.update(self.dx.raw.to_le_bytes());
        hasher.update(self.max_cg_iterations.to_le_bytes());
        hasher.update(self.cg_tolerance.raw.to_le_bytes());
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }
}

impl PartialEq for NSIMEXParams {
    fn eq(&self, other: &Self) -> bool {
        self.grid_bits == other.grid_bits
            && self.chi_max == other.chi_max
            && self.tolerance == other.tolerance
            && self.conservation_tolerance == other.conservation_tolerance
            && self.divergence_tolerance == other.divergence_tolerance
            && self.mpo_bond_dim == other.mpo_bond_dim
            && self.viscosity == other.viscosity
            && self.cfl == other.cfl
            && self.dt == other.dt
            && self.dx == other.dx
            && self.max_cg_iterations == other.max_cg_iterations
            && self.cg_tolerance == other.cg_tolerance
    }
}

impl Eq for NSIMEXParams {}

// ═════════════════════════════════════════════════════════════════════════════
// Circuit Sizing
// ═════════════════════════════════════════════════════════════════════════════

/// Estimated circuit sizing for the NS-IMEX proof.
#[derive(Debug, Clone)]
pub struct NSIMEXCircuitSizing {
    /// Source parameters.
    pub params: NSIMEXParams,
    /// Number of QTT sites.
    pub num_sites: usize,
    /// Total variables × sites (for wire layout).
    pub total_variable_sites: usize,
    /// Estimated total constraints.
    pub total_constraints: usize,
    /// Halo2 circuit degree k (rows = 2^k).
    pub k: usize,
    /// Number of public input cells.
    pub num_public_inputs: usize,
}

impl NSIMEXCircuitSizing {
    /// Compute circuit sizing from parameters.
    pub fn from_params(params: &NSIMEXParams) -> Self {
        let num_sites = params.num_sites();
        let total_variable_sites = NUM_NS_VARIABLES * num_sites;

        let total_constraints = Self::estimate_constraints(params);
        let k = Self::compute_k(total_constraints);
        let num_public_inputs = Self::num_public_inputs();

        Self {
            params: params.clone(),
            num_sites,
            total_variable_sites,
            total_constraints,
            k,
            num_public_inputs,
        }
    }

    /// Estimate total constraint rows.
    ///
    /// NS-IMEX has more constraints than Euler 3D due to:
    ///   - Implicit diffusion solve (matrix-vector product verification)
    ///   - Pressure Poisson solve (CG iteration verification)
    ///   - Divergence-free constraint
    pub fn estimate_constraints(params: &NSIMEXParams) -> usize {
        let l = params.num_sites();
        let chi = params.chi_max;
        let d = PHYS_DIM;

        // Per-variable, per-site MPO×MPS contraction (advection)
        let advection_macs = NUM_NS_VARIABLES * l * chi * chi * d;
        // Two half-steps for Strang-like IMEX splitting
        let advection_total = advection_macs * 2;

        // Diffusion solve: (I - ν·Δt·L) u = u*
        // Each component: Laplacian MPO apply (3 second-derivatives)
        // + linear combination + implicit solve check
        let diffusion_per_var = l * chi * chi * d * 3 // Laplacian (3D)
            + l * ROWS_PER_DIFFUSION_SOLVE; // Solve verification
        let diffusion_total = NUM_NS_VARIABLES * diffusion_per_var;

        // Projection: pressure Poisson + velocity correction
        // CG iterations: each has inner product + MPO apply + update
        let cg_per_iter = l * chi * chi * d * 2; // Two MPO applies
        let projection_total = params.max_cg_iterations * cg_per_iter
            + NUM_NS_VARIABLES * l * ROWS_PER_PROJECTION; // Divergence check

        // SVD truncations after each stage
        let svd_per_stage = NUM_NS_VARIABLES * l * ROWS_PER_SV_ORDER;
        let svd_total = NUM_IMEX_STAGES * svd_per_stage;

        // Conservation checks (kinetic energy, enstrophy)
        let conservation_checks = 2 * ROWS_PER_CONSERVATION;

        // Public input binding
        let public_inputs = Self::num_public_inputs() * ROWS_PER_PUBLIC_INPUT;

        advection_total
            + diffusion_total
            + projection_total
            + svd_total
            + conservation_checks
            + public_inputs
    }

    /// Compute minimum Halo2 k such that 2^k ≥ constraints + margin.
    pub fn compute_k(total_constraints: usize) -> usize {
        let with_margin = (total_constraints as f64 * 1.1).ceil() as usize + 256;
        let mut k = MIN_NS_IMEX_K;
        while (1usize << k) < with_margin && k < MAX_NS_IMEX_K {
            k += 1;
        }
        k
    }

    /// Number of public inputs for the NS-IMEX circuit.
    ///
    /// Public inputs:
    ///   - input_state_hash (4 × u64 limbs)
    ///   - output_state_hash (4 × u64 limbs)
    ///   - params_hash (4 × u64 limbs)
    ///   - conservation_residuals: kinetic energy + enstrophy (2)
    ///   - divergence_residual (1)
    ///   - timestep_dt (1)
    pub fn num_public_inputs() -> usize {
        4 + 4 + 4  // hashes
        + 2        // conservation (KE + enstrophy)
        + 1        // divergence residual
        + 1        // dt
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════════════
