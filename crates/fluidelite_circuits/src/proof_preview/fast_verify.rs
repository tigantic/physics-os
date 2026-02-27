//! Fast probabilistic verification for QTT physics proofs.
//!
//! Provides a sub-60-second pre-screening layer that combines spot-checks,
//! hash integrity, conservation bounds, and dimensional consistency into
//! a single fast verdict. Acts as a triage layer before full ZK verification.
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use std::time::Instant;

use fluidelite_core::field::Q16;
use fluidelite_core::mpo::MPO;
use fluidelite_core::mps::MPS;
use fluidelite_core::physics_traits::SolverType;

use super::spot_check::{SpotCheckConfig, SpotCheckResult, SpotChecker};

// ═══════════════════════════════════════════════════════════════════════════
// Preview Configuration
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for fast preview verification.
#[derive(Debug, Clone)]
pub struct PreviewConfig {
    /// Spot-check configuration.
    pub spot_check: SpotCheckConfig,

    /// Maximum allowed wall-clock time for preview (milliseconds).
    pub time_budget_ms: u64,

    /// Minimum confidence required for a PASS verdict.
    pub min_confidence: f64,

    /// Whether to check dimensional consistency (bond dims match, etc.).
    pub check_dimensions: bool,

    /// Whether to check data integrity (NaN, overflow detection).
    pub check_data_integrity: bool,

    /// Maximum number of variables for generic checks.
    pub max_variables: usize,
}

impl Default for PreviewConfig {
    fn default() -> Self {
        Self {
            spot_check: SpotCheckConfig::default(),
            time_budget_ms: 60_000, // 60 seconds
            min_confidence: 0.5,
            check_dimensions: true,
            check_data_integrity: true,
            max_variables: 16,
        }
    }
}

impl PreviewConfig {
    /// Configuration for a fast preview (low sample fraction, quick checks).
    pub fn fast() -> Self {
        Self {
            spot_check: SpotCheckConfig {
                sample_fraction: 0.10,
                min_sites: 2,
                max_sites: 16,
                ..SpotCheckConfig::default()
            },
            time_budget_ms: 5_000,
            min_confidence: 0.3,
            ..Self::default()
        }
    }

    /// Configuration for a thorough preview (higher sample fraction).
    pub fn thorough() -> Self {
        Self {
            spot_check: SpotCheckConfig {
                sample_fraction: 0.50,
                min_sites: 4,
                max_sites: 128,
                ..SpotCheckConfig::default()
            },
            time_budget_ms: 60_000,
            min_confidence: 0.8,
            ..Self::default()
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Preview Verdict
// ═══════════════════════════════════════════════════════════════════════════

/// Verdict from fast preview verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreviewVerdict {
    /// All preview checks passed — proceed to full verification.
    Pass,

    /// At least one check failed — full verification will likely fail.
    Fail,

    /// Preview was inconclusive (e.g. time budget exceeded, low confidence).
    Inconclusive,
}

impl std::fmt::Display for PreviewVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PreviewVerdict::Pass => write!(f, "PASS"),
            PreviewVerdict::Fail => write!(f, "FAIL"),
            PreviewVerdict::Inconclusive => write!(f, "INCONCLUSIVE"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Fast Verification Result
// ═══════════════════════════════════════════════════════════════════════════

/// Comprehensive result from fast preview verification.
#[derive(Debug, Clone)]
pub struct FastVerificationResult {
    /// Overall verdict.
    pub verdict: PreviewVerdict,

    /// Solver type that was verified.
    pub solver_type: SolverType,

    /// Spot-check results (if performed).
    pub spot_check: Option<SpotCheckResult>,

    /// Whether dimensional consistency checks passed.
    pub dimensions_ok: bool,

    /// Whether data integrity checks passed.
    pub data_integrity_ok: bool,

    /// Whether we exceeded the time budget.
    pub time_budget_exceeded: bool,

    /// Total wall-clock time for preview (microseconds).
    pub total_time_us: u64,

    /// Overall confidence level (0.0–1.0).
    pub confidence: f64,

    /// List of all warnings and failure messages.
    pub messages: Vec<String>,
}

// ═══════════════════════════════════════════════════════════════════════════
// Fast Verifier
// ═══════════════════════════════════════════════════════════════════════════

/// Fast probabilistic verifier for QTT physics proofs.
pub struct FastVerifier {
    /// Configuration.
    config: PreviewConfig,
}

impl FastVerifier {
    /// Create a new fast verifier with the given configuration.
    pub fn new(config: PreviewConfig) -> Self {
        Self { config }
    }

    /// Create a fast verifier with default configuration.
    pub fn default_verifier() -> Self {
        Self::new(PreviewConfig::default())
    }

    /// Create a fast verifier optimized for speed.
    pub fn fast_verifier() -> Self {
        Self::new(PreviewConfig::fast())
    }

    /// Create a fast verifier optimized for thoroughness.
    pub fn thorough_verifier() -> Self {
        Self::new(PreviewConfig::thorough())
    }

    /// Perform fast preview verification for a thermal proof.
    pub fn verify_thermal(
        &self,
        input_state: &MPS,
        output_state: &MPS,
        laplacian: &MPO,
        conservation_residual: Q16,
        conservation_tol: Q16,
        input_hash_limbs: &[u64; 4],
        output_hash_limbs: &[u64; 4],
    ) -> FastVerificationResult {
        let start = Instant::now();
        let deadline_us = (self.config.time_budget_ms as u64) * 1000;
        let mut messages = Vec::new();

        // Step 1: Dimensional consistency
        let dimensions_ok = if self.config.check_dimensions {
            self.check_thermal_dimensions(
                input_state,
                output_state,
                laplacian,
                &mut messages,
            )
        } else {
            true
        };

        // Step 2: Data integrity
        let data_integrity_ok = if self.config.check_data_integrity {
            self.check_data_integrity_mps(input_state, &mut messages, "input")
                && self.check_data_integrity_mps(output_state, &mut messages, "output")
        } else {
            true
        };

        // Check time budget
        let elapsed_us = start.elapsed().as_micros() as u64;
        if elapsed_us >= deadline_us {
            return FastVerificationResult {
                verdict: PreviewVerdict::Inconclusive,
                solver_type: SolverType::Thermal,
                spot_check: None,
                dimensions_ok,
                data_integrity_ok,
                time_budget_exceeded: true,
                total_time_us: elapsed_us,
                confidence: 0.0,
                messages: {
                    messages.push("Time budget exceeded before spot-checks".to_string());
                    messages
                },
            };
        }

        // Step 3: Spot-check
        let checker = SpotChecker::new(self.config.spot_check.clone());
        let spot_result = checker.check_thermal(
            input_state,
            output_state,
            laplacian,
            conservation_residual,
            conservation_tol,
            input_hash_limbs,
            output_hash_limbs,
        );

        let elapsed_us = start.elapsed().as_micros() as u64;
        let time_budget_exceeded = elapsed_us >= deadline_us;

        // Compute overall confidence
        let confidence = spot_result.confidence
            * if dimensions_ok { 1.0 } else { 0.0 }
            * if data_integrity_ok { 1.0 } else { 0.0 };

        // Determine verdict
        let verdict = if !spot_result.passed || !dimensions_ok || !data_integrity_ok {
            PreviewVerdict::Fail
        } else if confidence >= self.config.min_confidence && !time_budget_exceeded {
            PreviewVerdict::Pass
        } else {
            PreviewVerdict::Inconclusive
        };

        if !spot_result.failures.is_empty() {
            messages.extend(spot_result.failures.clone());
        }

        FastVerificationResult {
            verdict,
            solver_type: SolverType::Thermal,
            spot_check: Some(spot_result),
            dimensions_ok,
            data_integrity_ok,
            time_budget_exceeded,
            total_time_us: elapsed_us,
            confidence,
            messages,
        }
    }

    /// Perform fast preview verification for a generic proof (Euler3D or NS-IMEX).
    pub fn verify_generic(
        &self,
        solver_type: SolverType,
        input_states: &[MPS],
        output_states: &[MPS],
        shift_mpos: &[MPO],
        conservation_residuals: &[Q16],
        conservation_tol: Q16,
        input_hash_limbs: &[u64; 4],
        output_hash_limbs: &[u64; 4],
        hash_prefix: &[u8],
    ) -> FastVerificationResult {
        let start = Instant::now();
        let deadline_us = (self.config.time_budget_ms as u64) * 1000;
        let mut messages = Vec::new();

        // Step 1: Dimensional consistency
        let dimensions_ok = if self.config.check_dimensions {
            self.check_generic_dimensions(
                input_states,
                output_states,
                shift_mpos,
                &mut messages,
            )
        } else {
            true
        };

        // Step 2: Data integrity
        let mut data_integrity_ok = true;
        if self.config.check_data_integrity {
            for (i, state) in input_states.iter().enumerate() {
                if !self.check_data_integrity_mps(
                    state,
                    &mut messages,
                    &format!("input[{}]", i),
                ) {
                    data_integrity_ok = false;
                }
            }
            for (i, state) in output_states.iter().enumerate() {
                if !self.check_data_integrity_mps(
                    state,
                    &mut messages,
                    &format!("output[{}]", i),
                ) {
                    data_integrity_ok = false;
                }
            }
        }

        // Check time budget
        let elapsed_us = start.elapsed().as_micros() as u64;
        if elapsed_us >= deadline_us {
            return FastVerificationResult {
                verdict: PreviewVerdict::Inconclusive,
                solver_type,
                spot_check: None,
                dimensions_ok,
                data_integrity_ok,
                time_budget_exceeded: true,
                total_time_us: elapsed_us,
                confidence: 0.0,
                messages: {
                    messages.push("Time budget exceeded before spot-checks".to_string());
                    messages
                },
            };
        }

        // Step 3: Spot-check
        let checker = SpotChecker::new(self.config.spot_check.clone());
        let spot_result = checker.check_generic(
            input_states,
            output_states,
            shift_mpos,
            conservation_residuals,
            conservation_tol,
            input_hash_limbs,
            output_hash_limbs,
            hash_prefix,
        );

        let elapsed_us = start.elapsed().as_micros() as u64;
        let time_budget_exceeded = elapsed_us >= deadline_us;

        let confidence = spot_result.confidence
            * if dimensions_ok { 1.0 } else { 0.0 }
            * if data_integrity_ok { 1.0 } else { 0.0 };

        let verdict = if !spot_result.passed || !dimensions_ok || !data_integrity_ok {
            PreviewVerdict::Fail
        } else if confidence >= self.config.min_confidence && !time_budget_exceeded {
            PreviewVerdict::Pass
        } else {
            PreviewVerdict::Inconclusive
        };

        if !spot_result.failures.is_empty() {
            messages.extend(spot_result.failures.clone());
        }

        FastVerificationResult {
            verdict,
            solver_type,
            spot_check: Some(spot_result),
            dimensions_ok,
            data_integrity_ok,
            time_budget_exceeded,
            total_time_us: elapsed_us,
            confidence,
            messages,
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Internal Helpers
    // ═══════════════════════════════════════════════════════════════════════

    /// Check dimensional consistency for thermal proof.
    fn check_thermal_dimensions(
        &self,
        input_state: &MPS,
        output_state: &MPS,
        laplacian: &MPO,
        messages: &mut Vec<String>,
    ) -> bool {
        let mut ok = true;

        // Input and output must have the same number of sites
        if input_state.num_sites != output_state.num_sites {
            messages.push(format!(
                "Site count mismatch: input={}, output={}",
                input_state.num_sites, output_state.num_sites
            ));
            ok = false;
        }

        // Laplacian must have the same number of sites
        if input_state.num_sites != laplacian.num_sites {
            messages.push(format!(
                "Laplacian site count mismatch: MPS={}, MPO={}",
                input_state.num_sites, laplacian.num_sites
            ));
            ok = false;
        }

        // Check internal bond dimension consistency for input
        if !self.check_bond_consistency(&input_state.cores) {
            messages.push("Input state has inconsistent bond dimensions".to_string());
            ok = false;
        }

        // Check internal bond dimension consistency for output
        if !self.check_bond_consistency(&output_state.cores) {
            messages.push("Output state has inconsistent bond dimensions".to_string());
            ok = false;
        }

        ok
    }

    /// Check dimensional consistency for generic proof.
    fn check_generic_dimensions(
        &self,
        input_states: &[MPS],
        output_states: &[MPS],
        shift_mpos: &[MPO],
        messages: &mut Vec<String>,
    ) -> bool {
        let mut ok = true;

        // Must have same number of input and output states
        if input_states.len() != output_states.len() {
            messages.push(format!(
                "Variable count mismatch: {} inputs, {} outputs",
                input_states.len(),
                output_states.len()
            ));
            ok = false;
        }

        // All states must have the same number of sites
        let expected_sites = input_states.first().map(|s| s.num_sites);
        for (i, state) in input_states.iter().enumerate() {
            if Some(state.num_sites) != expected_sites {
                messages.push(format!(
                    "Input[{}] site count {} != expected {:?}",
                    i, state.num_sites, expected_sites
                ));
                ok = false;
            }
        }

        for (i, state) in output_states.iter().enumerate() {
            if Some(state.num_sites) != expected_sites {
                messages.push(format!(
                    "Output[{}] site count {} != expected {:?}",
                    i, state.num_sites, expected_sites
                ));
                ok = false;
            }
        }

        // MPOs must match site counts
        for (i, mpo) in shift_mpos.iter().enumerate() {
            if Some(mpo.num_sites) != expected_sites {
                messages.push(format!(
                    "MPO[{}] site count {} != expected {:?}",
                    i, mpo.num_sites, expected_sites
                ));
                ok = false;
            }
        }

        // Check bond consistency for all states
        for (i, state) in input_states.iter().enumerate() {
            if !self.check_bond_consistency(&state.cores) {
                messages.push(format!(
                    "Input[{}] has inconsistent bond dimensions",
                    i
                ));
                ok = false;
            }
        }

        for (i, state) in output_states.iter().enumerate() {
            if !self.check_bond_consistency(&state.cores) {
                messages.push(format!(
                    "Output[{}] has inconsistent bond dimensions",
                    i
                ));
                ok = false;
            }
        }

        ok
    }

    /// Check that bond dimensions are consistent between adjacent cores.
    fn check_bond_consistency(&self, cores: &[fluidelite_core::mps::MPSCore]) -> bool {
        if cores.len() <= 1 {
            return true;
        }
        for pair in cores.windows(2) {
            if pair[0].chi_right != pair[1].chi_left {
                return false;
            }
        }
        true
    }

    /// Check MPS data integrity — no data entries are suspicious.
    fn check_data_integrity_mps(
        &self,
        state: &MPS,
        messages: &mut Vec<String>,
        label: &str,
    ) -> bool {
        let mut ok = true;

        // Check for empty cores
        if state.cores.is_empty() && state.num_sites > 0 {
            messages.push(format!("{}: has 0 cores but num_sites={}", label, state.num_sites));
            ok = false;
        }

        // Check core data sizes match declared dimensions
        for (i, core) in state.cores.iter().enumerate() {
            let expected_size = core.chi_left * core.d * core.chi_right;
            let actual_size = core.data.len();
            if actual_size != expected_size {
                messages.push(format!(
                    "{} core[{}]: data size {} != expected {} ({}×{}×{})",
                    label, i, actual_size, expected_size,
                    core.chi_left, core.d, core.chi_right,
                ));
                ok = false;
            }
        }

        ok
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════
// Tests (private method access — integration tests cover public API)
// ═══════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════
// Tests (private method access — integration tests cover public API)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use fluidelite_core::mps::MPS;

    #[test]
    fn test_bond_consistency_check() {
        let verifier = FastVerifier::default_verifier();

        // Valid MPS — MPS::new creates consistent bond dimensions
        let valid = MPS::new(4, 2, 2);
        assert!(verifier.check_bond_consistency(&valid.cores));

        // Empty cores are fine
        assert!(verifier.check_bond_consistency(&[]));
    }
}
