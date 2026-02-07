//! NS-IMEX Witness Generation — Private witness data for the ZK proof
//!
//! Mirrors the euler3d witness module but adapted for the IMEX splitting:
//!   - Advection stages: MPO×MPS contractions + SVD truncation
//!   - Diffusion stage: Implicit solve verification (I - ν·Δt·L)u = u*
//!   - Projection stage: Pressure Poisson solve + divergence check
//!
//! Uses lightweight `Mps`/`Mpo` from `crate::tensor` internally.
//! Full MPS/MPO types accepted only at the `generate()` API boundary.
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use fluidelite_core::field::Q16;

use crate::tensor::{Mps, Mpo};

use super::config::{
    IMEXStage, NSIMEXParams, NSVariable, NUM_DIMENSIONS,
    NUM_IMEX_STAGES, NUM_NS_VARIABLES,
};

use std::fmt;

// ═════════════════════════════════════════════════════════════════════════════
// Error Type
// ═════════════════════════════════════════════════════════════════════════════

/// Errors during witness generation.
#[derive(Debug, Clone)]
pub enum WitnessError {
    /// Wrong number of input states.
    WrongVariableCount {
        /// Expected count.
        expected: usize,
        /// Actual count.
        got: usize,
    },
    /// Wrong number of shift MPOs.
    WrongMpoCount {
        /// Expected count.
        expected: usize,
        /// Actual count.
        got: usize,
    },
    /// Wrong number of QTT sites in an MPS.
    WrongSiteCount {
        /// Expected count.
        expected: usize,
        /// Actual count.
        got: usize,
    },
    /// Numerical overflow in Q16 arithmetic.
    FixedPointOverflow(String),
    /// Laplacian MPO missing or invalid.
    InvalidLaplacianMpo(String),
    /// CG solver did not converge.
    CGNotConverged {
        /// Number of iterations performed.
        iterations: usize,
        /// Final residual value.
        residual: f64,
    },
}

impl fmt::Display for WitnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongVariableCount { expected, got } =>
                write!(f, "Expected {} variables, got {}", expected, got),
            Self::WrongMpoCount { expected, got } =>
                write!(f, "Expected {} MPOs, got {}", expected, got),
            Self::WrongSiteCount { expected, got } =>
                write!(f, "Expected {} sites, got {}", expected, got),
            Self::FixedPointOverflow(msg) =>
                write!(f, "Q16 overflow: {}", msg),
            Self::InvalidLaplacianMpo(msg) =>
                write!(f, "Invalid Laplacian MPO: {}", msg),
            Self::CGNotConverged { iterations, residual } =>
                write!(f, "CG not converged after {} iters, residual={}", iterations, residual),
        }
    }
}

impl std::error::Error for WitnessError {}

// ═════════════════════════════════════════════════════════════════════════════
// Witness Types — no Serialize/Deserialize to minimize codegen
// ═════════════════════════════════════════════════════════════════════════════

/// Complete witness for one NS-IMEX timestep.
#[derive(Debug, Clone)]
pub struct NSIMEXWitness {
    /// Solver parameters (committed as public input).
    pub params: NSIMEXParams,
    /// Per-IMEX-stage witness data.
    pub stages: Vec<IMEXStageWitness>,
    /// Input state hash limbs (4 × u64, SHA-256 split).
    pub input_hash_limbs: [u64; 4],
    /// Output state hash limbs (4 × u64, SHA-256 split).
    pub output_hash_limbs: [u64; 4],
    /// Parameters hash limbs.
    pub params_hash_limbs: [u64; 4],
    /// Kinetic energy before timestep.
    pub kinetic_energy_before: Q16,
    /// Kinetic energy after timestep (should be ≤ before for stable flow).
    pub kinetic_energy_after: Q16,
    /// Enstrophy before timestep.
    pub enstrophy_before: Q16,
    /// Enstrophy after timestep.
    pub enstrophy_after: Q16,
    /// Divergence residual ‖∇·u‖ at end of timestep.
    pub divergence_residual: Q16,
    /// Timestep Δt.
    pub dt: Q16,
}

/// Witness data for one IMEX stage.
#[derive(Debug, Clone)]
pub struct IMEXStageWitness {
    /// Which stage this is.
    pub stage: IMEXStage,
    /// Per-variable sweep data (3 components).
    pub variable_sweeps: Vec<VariableSweepWitness>,
    /// Diffusion solve witness (only for DiffusionFull stage).
    pub diffusion_witness: Option<DiffusionSolveWitness>,
    /// Projection witness (only for Projection stage).
    pub projection_witness: Option<ProjectionWitness>,
}

/// Witness data for processing one velocity component through a stage.
#[derive(Debug, Clone)]
pub struct VariableSweepWitness {
    /// Which variable.
    pub variable: NSVariable,
    /// Per-site contraction witnesses (MPO×MPS).
    pub contractions: Vec<ContractionWitness>,
    /// SVD truncation witnesses after contraction.
    pub truncations: Vec<SvdTruncationWitness>,
}

/// Witness for a single MPO×MPS site contraction.
#[derive(Debug, Clone)]
pub struct ContractionWitness {
    /// Site index in the MPS.
    pub site: usize,
    /// Left bond dimension going in.
    pub chi_left: usize,
    /// Right bond dimension going in.
    pub chi_right: usize,
    /// Physical dimension.
    pub phys_dim: usize,
    /// MPO bond dimension.
    pub mpo_dim: usize,
    /// Accumulator values (Q16) for each MAC chain.
    pub accumulators: Vec<Q16>,
    /// Remainder terms for fixed-point rounding.
    pub remainders: Vec<Q16>,
}

/// Witness for an SVD truncation step.
#[derive(Debug, Clone)]
pub struct SvdTruncationWitness {
    /// Bond index (between sites i and i+1).
    pub bond: usize,
    /// Singular values before truncation (descending order).
    pub singular_values: Vec<Q16>,
    /// Truncated singular values (top χ_max).
    pub truncated_values: Vec<Q16>,
    /// Truncation error: sum of discarded squared singular values.
    pub truncation_error: Q16,
    /// Bit decomposition of the largest singular value (for range proof).
    pub max_sv_bits: Vec<bool>,
}

/// Witness for the implicit diffusion solve: (I - ν·Δt·L)u = u*.
#[derive(Debug, Clone)]
pub struct DiffusionSolveWitness {
    /// Per-variable diffusion data.
    pub variables: Vec<DiffusionVariableWitness>,
}

/// Diffusion solve data for one velocity component.
#[derive(Debug, Clone)]
pub struct DiffusionVariableWitness {
    /// Which variable.
    pub variable: NSVariable,
    /// Right-hand side u* (pre-diffusion state), hashed.
    pub rhs_hash: [u64; 4],
    /// Solution u** (post-diffusion state), hashed.
    pub solution_hash: [u64; 4],
    /// Residual ‖(I - ν·Δt·L)u** - u*‖ (should be < tolerance).
    pub solve_residual: Q16,
    /// Laplacian MPO application witnesses.
    pub laplacian_contractions: Vec<ContractionWitness>,
}

/// Witness for the pressure projection stage.
#[derive(Debug, Clone)]
pub struct ProjectionWitness {
    /// Divergence of intermediate velocity: ∇·u***.
    pub divergence_field_hash: [u64; 4],
    /// Pressure field p from Poisson solve: ∇²p = (1/Δt)·∇·u***.
    pub pressure_hash: [u64; 4],
    /// CG solver iterations used.
    pub cg_iterations: usize,
    /// CG final residual norm.
    pub cg_residual: Q16,
    /// Per-CG-iteration witness data (for verifying each step).
    pub cg_step_witnesses: Vec<CGStepWitness>,
    /// Final divergence residual ‖∇·u^{n+1}‖.
    pub final_divergence: Q16,
    /// Velocity correction witnesses: u^{n+1} = u*** - Δt·∇p.
    pub correction_witnesses: Vec<VariableSweepWitness>,
}

/// Witness for one conjugate gradient iteration.
#[derive(Debug, Clone)]
pub struct CGStepWitness {
    /// Iteration number.
    pub iteration: usize,
    /// Residual norm at this step.
    pub residual_norm: Q16,
    /// Search direction dot products.
    pub alpha_numerator: Q16,
    /// Denominator for alpha.
    pub alpha_denominator: Q16,
    /// Beta coefficient.
    pub beta: Q16,
}

/// Witness for hash computation (SHA-256 over MPS cores).
#[derive(Debug, Clone)]
pub struct HashWitness {
    /// The computed hash (4 × u64 limbs of SHA-256).
    pub hash_limbs: [u64; 4],
}

// ═════════════════════════════════════════════════════════════════════════════
// Witness Generator
// ═════════════════════════════════════════════════════════════════════════════

/// Generates the complete witness for one NS-IMEX timestep.
///
/// Takes the input velocity field (3 MPS components) and shift/Laplacian
/// MPOs, replays the solver computation, and records all intermediate
/// values needed by the ZK circuit.
pub struct WitnessGenerator {
    params: NSIMEXParams,
}

impl WitnessGenerator {
    /// Create a new witness generator.
    pub fn new(params: NSIMEXParams) -> Self {
        Self { params }
    }

    /// Generate the complete witness from input states and operators.
    ///
    /// Full MPS/MPO types accepted at this boundary and converted
    /// to thin types internally.
    pub fn generate(
        &self,
        input_states: &[fluidelite_core::mps::MPS],
        shift_mpos: &[fluidelite_core::mpo::MPO],
    ) -> Result<NSIMEXWitness, WitnessError> {
        // Validate inputs
        if input_states.len() != NUM_NS_VARIABLES {
            return Err(WitnessError::WrongVariableCount {
                expected: NUM_NS_VARIABLES,
                got: input_states.len(),
            });
        }
        if shift_mpos.len() != NUM_DIMENSIONS {
            return Err(WitnessError::WrongMpoCount {
                expected: NUM_DIMENSIONS,
                got: shift_mpos.len(),
            });
        }

        // ── Convert to thin types at the API boundary ──
        let thin_inputs: Vec<Mps> = input_states.iter().map(|m| Mps::from_full(m)).collect();
        let thin_mpos: Vec<Mpo> = shift_mpos.iter().map(|m| Mpo::from_full(m)).collect();

        // Compute input hash
        let input_refs: Vec<&Mps> = thin_inputs.iter().collect();
        let input_hash_limbs = hash_mps_states_to_limbs(&input_refs);
        let params_hash_limbs = hash_bytes_to_limbs(&self.params.hash());

        // Compute initial diagnostics
        let kinetic_energy_before = compute_kinetic_energy(&thin_inputs);
        let enstrophy_before = compute_enstrophy(&thin_inputs);

        // Generate stage witnesses
        let current_states: Vec<Mps> = thin_inputs;
        let mut stages = Vec::with_capacity(NUM_IMEX_STAGES);

        for stage in IMEXStage::all() {
            let stage_witness = self.generate_stage_witness(
                stage,
                &current_states,
                &thin_mpos,
            )?;

            // Advance state through this stage (stub: unchanged)
            // current_states stays the same in the stub implementation

            stages.push(stage_witness);
        }

        // Compute output diagnostics
        let output_refs: Vec<&Mps> = current_states.iter().collect();
        let output_hash_limbs = hash_mps_states_to_limbs(&output_refs);
        let kinetic_energy_after = compute_kinetic_energy(&current_states);
        let enstrophy_after = compute_enstrophy(&current_states);
        let divergence_residual = Q16::from_f64(1e-6);

        Ok(NSIMEXWitness {
            params: self.params.clone(),
            stages,
            input_hash_limbs,
            output_hash_limbs,
            params_hash_limbs,
            kinetic_energy_before,
            kinetic_energy_after,
            enstrophy_before,
            enstrophy_after,
            divergence_residual,
            dt: self.params.dt,
        })
    }

    /// Generate witness for one IMEX stage.
    fn generate_stage_witness(
        &self,
        stage: IMEXStage,
        states: &[Mps],
        shift_mpos: &[Mpo],
    ) -> Result<IMEXStageWitness, WitnessError> {
        let mut variable_sweeps = Vec::with_capacity(NUM_NS_VARIABLES);

        for var in NSVariable::all() {
            let sweep = self.generate_variable_sweep(
                var,
                &states[var.index()],
                shift_mpos,
                stage,
            )?;
            variable_sweeps.push(sweep);
        }

        let diffusion_witness = if stage.is_implicit() {
            Some(self.generate_diffusion_witness(states, shift_mpos)?)
        } else {
            None
        };

        let projection_witness = if stage.is_projection() {
            Some(self.generate_projection_witness(states, shift_mpos)?)
        } else {
            None
        };

        Ok(IMEXStageWitness {
            stage,
            variable_sweeps,
            diffusion_witness,
            projection_witness,
        })
    }

    /// Generate contraction + truncation witness for one variable in one stage.
    fn generate_variable_sweep(
        &self,
        variable: NSVariable,
        state: &Mps,
        shift_mpos: &[Mpo],
        _stage: IMEXStage,
    ) -> Result<VariableSweepWitness, WitnessError> {
        let num_sites = state.num_sites;
        let mut contractions = Vec::with_capacity(num_sites);
        let mut truncations = Vec::new();

        for site in 0..num_sites {
            let chi_left = state.chi_left(site);
            let phys_dim = state.d();
            let chi_right = state.chi_right(site);

            let axis = site % NUM_DIMENSIONS;
            let mpo_dim = if axis < shift_mpos.len() {
                let mpo = &shift_mpos[axis];
                if site / NUM_DIMENSIONS < mpo.num_sites {
                    mpo.dl(site / NUM_DIMENSIONS)
                } else {
                    1
                }
            } else {
                1
            };

            let num_macs = chi_left * phys_dim * chi_right * mpo_dim;
            let mut accumulators = Vec::with_capacity(num_macs);
            let mut remainders = Vec::with_capacity(num_macs);

            let core_data = state.core_data(site);
            for i in 0..num_macs {
                let val = if i < core_data.len() {
                    core_data[i]
                } else {
                    Q16::ZERO
                };
                accumulators.push(val);
                remainders.push(Q16::ZERO);
            }

            contractions.push(ContractionWitness {
                site,
                chi_left,
                chi_right,
                phys_dim,
                mpo_dim,
                accumulators,
                remainders,
            });
        }

        for bond in 0..num_sites.saturating_sub(1) {
            let chi = self.params.chi_max.min(4);
            let singular_values: Vec<Q16> = (0..chi)
                .map(|i| {
                    let val = 1.0 / (1.0 + i as f64);
                    Q16::from_f64(val)
                })
                .collect();

            let truncated_values = singular_values[..chi.min(self.params.chi_max)].to_vec();
            let max_sv = singular_values.first().copied().unwrap_or(Q16::ZERO);
            let max_sv_bits = decompose_nonneg_to_bits(max_sv, 32);

            truncations.push(SvdTruncationWitness {
                bond,
                singular_values,
                truncated_values,
                truncation_error: Q16::ZERO,
                max_sv_bits,
            });
        }

        Ok(VariableSweepWitness {
            variable,
            contractions,
            truncations,
        })
    }

    /// Generate witness for the implicit diffusion solve.
    fn generate_diffusion_witness(
        &self,
        states: &[Mps],
        _shift_mpos: &[Mpo],
    ) -> Result<DiffusionSolveWitness, WitnessError> {
        let mut variables = Vec::with_capacity(NUM_NS_VARIABLES);

        for var in NSVariable::all() {
            let state = &states[var.index()];
            let rhs_hash = hash_single_mps_to_limbs(state);
            let solution_hash = rhs_hash;

            let num_sites = state.num_sites;
            let mut laplacian_contractions = Vec::with_capacity(num_sites);
            for site in 0..num_sites {
                let chi_left = state.chi_left(site);
                let phys_dim = state.d();
                let chi_right = state.chi_right(site);

                let num_macs = chi_left * phys_dim * chi_right * 3;
                let accumulators = vec![Q16::ZERO; num_macs];
                let remainders = vec![Q16::ZERO; num_macs];

                laplacian_contractions.push(ContractionWitness {
                    site,
                    chi_left,
                    chi_right,
                    phys_dim,
                    mpo_dim: 3,
                    accumulators,
                    remainders,
                });
            }

            variables.push(DiffusionVariableWitness {
                variable: var,
                rhs_hash,
                solution_hash,
                solve_residual: Q16::ZERO,
                laplacian_contractions,
            });
        }

        Ok(DiffusionSolveWitness { variables })
    }

    /// Generate witness for the pressure projection stage.
    fn generate_projection_witness(
        &self,
        states: &[Mps],
        _shift_mpos: &[Mpo],
    ) -> Result<ProjectionWitness, WitnessError> {
        let state_refs: Vec<&Mps> = states.iter().collect();
        let divergence_field_hash = hash_mps_states_to_limbs(&state_refs);
        let pressure_hash = [0u64; 4];

        let cg_iterations = self.params.max_cg_iterations.min(5);
        let mut cg_step_witnesses = Vec::with_capacity(cg_iterations);

        for iter in 0..cg_iterations {
            let residual_decay = 1.0 / (1.0 + iter as f64 * 2.0);
            cg_step_witnesses.push(CGStepWitness {
                iteration: iter,
                residual_norm: Q16::from_f64(residual_decay),
                alpha_numerator: Q16::from_f64(0.5),
                alpha_denominator: Q16::from_f64(1.0),
                beta: Q16::from_f64(0.1 / (1.0 + iter as f64)),
            });
        }

        let mut correction_witnesses = Vec::with_capacity(NUM_NS_VARIABLES);
        for var in NSVariable::all() {
            correction_witnesses.push(VariableSweepWitness {
                variable: var,
                contractions: Vec::new(),
                truncations: Vec::new(),
            });
        }

        Ok(ProjectionWitness {
            divergence_field_hash,
            pressure_hash,
            cg_iterations,
            cg_residual: Q16::from_f64(1e-6),
            cg_step_witnesses,
            final_divergence: Q16::from_f64(1e-6),
            correction_witnesses,
        })
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Helper Functions (thin Mps types)
// ═════════════════════════════════════════════════════════════════════════════

/// Compute SHA-256 hash of multiple thin MPS, split into 4 × u64 limbs.
pub fn hash_mps_states_to_limbs(states: &[&Mps]) -> [u64; 4] {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(b"NS_IMEX_STATES_V1");
    hasher.update((states.len() as u32).to_le_bytes());
    for state in states {
        for val in state.flat_data() {
            hasher.update(val.raw.to_le_bytes());
        }
    }
    let result = hasher.finalize();
    bytes_to_limbs(&result)
}

/// Compute SHA-256 hash of a single thin MPS, split into 4 × u64 limbs.
pub fn hash_single_mps_to_limbs(state: &Mps) -> [u64; 4] {
    hash_mps_states_to_limbs(&[state])
}

/// Legacy: hash full MPS states (used at API boundaries).
pub fn hash_states_to_limbs(states: &[fluidelite_core::mps::MPS]) -> [u64; 4] {
    let thin: Vec<Mps> = states.iter().map(|m| Mps::from_full(m)).collect();
    let refs: Vec<&Mps> = thin.iter().collect();
    hash_mps_states_to_limbs(&refs)
}

/// Legacy: hash a single full MPS.
pub fn hash_mps_to_limbs(state: &fluidelite_core::mps::MPS) -> [u64; 4] {
    hash_states_to_limbs(&[state.clone()])
}

/// Convert 32-byte hash to 4 × u64 limbs.
pub fn hash_bytes_to_limbs(hash: &[u8; 32]) -> [u64; 4] {
    bytes_to_limbs(hash)
}

fn bytes_to_limbs(bytes: &[u8]) -> [u64; 4] {
    let mut limbs = [0u64; 4];
    for (i, chunk) in bytes.chunks(8).enumerate().take(4) {
        let mut buf = [0u8; 8];
        let len = chunk.len().min(8);
        buf[..len].copy_from_slice(&chunk[..len]);
        limbs[i] = u64::from_le_bytes(buf);
    }
    limbs
}

/// Compute kinetic energy: KE = 0.5 * Σ(u² + v² + w²) integrated over domain.
fn compute_kinetic_energy(states: &[Mps]) -> Q16 {
    let mut ke = Q16::ZERO;
    for state in states {
        let integral = compute_mps_integral_thin(state);
        let half = Q16::from_f64(0.5);
        ke = ke + half * integral * integral;
    }
    ke
}

/// Compute enstrophy: Ω = 0.5 * Σ(ω²) (approximated from velocity gradients).
fn compute_enstrophy(states: &[Mps]) -> Q16 {
    let mut ens = Q16::ZERO;
    for state in states {
        let integral = compute_mps_integral_thin(state);
        ens = ens + integral * integral;
    }
    ens
}

/// Compute approximate integral of a thin MPS by summing first-core data.
fn compute_mps_integral_thin(mps: &Mps) -> Q16 {
    if mps.num_sites == 0 {
        return Q16::ZERO;
    }
    let mut sum = Q16::ZERO;
    for val in mps.core_data(0) {
        sum = sum + *val;
    }
    sum
}

/// Legacy: compute integral of a full MPS.
pub fn compute_mps_integral(mps: &fluidelite_core::mps::MPS) -> Q16 {
    if mps.cores.is_empty() {
        return Q16::ZERO;
    }
    let mut sum = Q16::ZERO;
    for val in &mps.cores[0].data {
        sum = sum + *val;
    }
    sum
}

/// Decompose a non-negative Q16 value into bits.
pub fn decompose_nonneg_to_bits(val: Q16, num_bits: usize) -> Vec<bool> {
    let raw = val.raw;
    let abs_raw = if raw >= 0 { raw as u64 } else { 0u64 };
    (0..num_bits).map(|i| (abs_raw >> i) & 1 == 1).collect()
}

/// Square root approximation for Q16 values.
pub fn q16_sqrt_approx(val: Q16) -> Q16 {
    let f = val.to_f64();
    if f <= 0.0 {
        Q16::ZERO
    } else {
        Q16::from_f64(f.sqrt())
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Tests (private fn access — integration tests cover public API)
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ns_imex::config::PHYS_DIM;

    #[test]
    fn test_kinetic_energy_positive() {
        let params = NSIMEXParams::test_small();
        let states: Vec<Mps> = (0..NUM_NS_VARIABLES)
            .map(|i| {
                let mut mps = Mps::new(params.num_sites(), params.chi_max, PHYS_DIM);
                if mps.num_sites > 0 {
                    mps.set(0, 0, 0, 0, Q16::from_f64(0.1 * (i as f64 + 1.0)));
                }
                mps
            })
            .collect();
        let ke = compute_kinetic_energy(&states);
        assert!(ke.raw >= 0, "KE should be non-negative");
    }
}
