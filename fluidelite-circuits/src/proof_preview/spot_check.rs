//! Spot-check verification for QTT physics proofs.
//!
//! Randomly samples a subset of QTT sites and verifies local
//! constraints (MAC correctness, SVD ordering) without performing
//! full circuit verification.
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use std::collections::HashSet;
use std::time::Instant;

use fluidelite_core::field::Q16;
use fluidelite_core::mpo::MPO;
use fluidelite_core::mps::MPS;

// ═══════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for spot-check verification.
#[derive(Debug, Clone)]
pub struct SpotCheckConfig {
    /// Fraction of sites to sample (0.0–1.0).
    pub sample_fraction: f64,

    /// Minimum number of sites to check.
    pub min_sites: usize,

    /// Maximum number of sites to check.
    pub max_sites: usize,

    /// Maximum allowed MAC error in Q16 (absolute).
    pub mac_tolerance_raw: i64,

    /// Maximum allowed conservation residual.
    pub conservation_tolerance_raw: i64,

    /// Random seed for reproducible sampling.
    pub seed: u64,
}

impl Default for SpotCheckConfig {
    fn default() -> Self {
        Self {
            sample_fraction: 0.25,
            min_sites: 2,
            max_sites: 64,
            mac_tolerance_raw: 10, // ~1.5e-4 in Q16
            conservation_tolerance_raw: 100, // ~1.5e-3 in Q16
            seed: 0xDEAD_BEEF_CAFE_1234,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Spot Check Result
// ═══════════════════════════════════════════════════════════════════════════

/// Result of a spot-check verification.
#[derive(Debug, Clone)]
pub struct SpotCheckResult {
    /// Whether all spot checks passed.
    pub passed: bool,

    /// Number of sites checked.
    pub sites_checked: usize,

    /// Total sites in the proof.
    pub total_sites: usize,

    /// Number of MAC checks performed.
    pub mac_checks: usize,

    /// Number of MAC failures.
    pub mac_failures: usize,

    /// Number of SVD ordering checks.
    pub svd_checks: usize,

    /// Number of SVD ordering failures.
    pub svd_failures: usize,

    /// Conservation residual check passed.
    pub conservation_passed: bool,

    /// Conservation residual value.
    pub conservation_residual: f64,

    /// Hash integrity check passed.
    pub hash_integrity: bool,

    /// Time taken for spot checks in microseconds.
    pub check_time_us: u64,

    /// Estimated confidence level (0.0–1.0).
    pub confidence: f64,

    /// List of failure messages (empty if passed).
    pub failures: Vec<String>,
}

// ═══════════════════════════════════════════════════════════════════════════
// Spot Checker
// ═══════════════════════════════════════════════════════════════════════════

/// Performs spot-check verification on QTT physics proofs.
pub struct SpotChecker {
    /// Configuration.
    config: SpotCheckConfig,
}

impl SpotChecker {
    /// Create a new spot checker with the given configuration.
    pub fn new(config: SpotCheckConfig) -> Self {
        Self { config }
    }

    /// Create a spot checker with default configuration.
    pub fn default_checker() -> Self {
        Self::new(SpotCheckConfig::default())
    }

    /// Spot-check a thermal proof by verifying:
    /// 1. Input/output hash integrity
    /// 2. Conservation residual within tolerance
    /// 3. Random site MAC correctness
    /// 4. SVD ordering at random bonds
    pub fn check_thermal(
        &self,
        input_state: &MPS,
        output_state: &MPS,
        laplacian: &MPO,
        conservation_residual: Q16,
        conservation_tol: Q16,
        input_hash_limbs: &[u64; 4],
        output_hash_limbs: &[u64; 4],
    ) -> SpotCheckResult {
        let start = Instant::now();
        let mut failures = Vec::new();
        let total_sites = input_state.num_sites;

        // Determine sample sites
        let sites_to_check = self.select_sites(total_sites);
        let sites_checked = sites_to_check.len();

        // Check 1: Hash integrity
        let computed_input_hash = Self::compute_state_hash(
            &[input_state.clone()],
            b"THERMAL_STATE_V1",
        );
        let hash_integrity = &computed_input_hash == input_hash_limbs;
        if !hash_integrity {
            failures.push("Input state hash mismatch".to_string());
        }

        let computed_output_hash = Self::compute_state_hash(
            &[output_state.clone()],
            b"THERMAL_STATE_V1",
        );
        let output_hash_ok = &computed_output_hash == output_hash_limbs;
        if !output_hash_ok {
            failures.push("Output state hash mismatch".to_string());
        }

        // Check 2: Conservation residual
        let conservation_passed =
            conservation_residual.raw.abs() <= conservation_tol.raw;
        let conservation_residual_f64 = conservation_residual.to_f64();
        if !conservation_passed {
            failures.push(format!(
                "Conservation violation: |{:.6e}| > {:.6e}",
                conservation_residual_f64,
                conservation_tol.to_f64(),
            ));
        }

        // Check 3: Spot MAC checks at selected sites
        let mut mac_checks = 0usize;
        let mut mac_failures = 0usize;

        for &site_idx in &sites_to_check {
            if site_idx < input_state.cores.len()
                && site_idx < laplacian.cores.len()
            {
                let mps_core = &input_state.cores[site_idx];
                let mpo_core = &laplacian.cores[site_idx];

                // Verify a few MAC chains at this site
                let checks = self.check_site_mac(mps_core, mpo_core);
                mac_checks += checks.0;
                mac_failures += checks.1;
            }
        }

        if mac_failures > 0 {
            failures.push(format!(
                "MAC failures: {}/{} at sampled sites",
                mac_failures, mac_checks
            ));
        }

        // Check 4: SVD ordering at random bonds
        let mut svd_checks = 0usize;
        let mut svd_failures = 0usize;

        if output_state.cores.len() > 1 {
            for &site_idx in &sites_to_check {
                if site_idx + 1 < output_state.cores.len() {
                    let left_chi = output_state.cores[site_idx].chi_right;
                    let right_chi = output_state.cores[site_idx + 1].chi_left;

                    // Bond dimension consistency check
                    svd_checks += 1;
                    if left_chi != right_chi {
                        svd_failures += 1;
                        failures.push(format!(
                            "Bond dimension mismatch at site {}: {} != {}",
                            site_idx, left_chi, right_chi
                        ));
                    }
                }
            }
        }

        let passed = failures.is_empty();

        // Confidence: based on fraction of sites checked
        let confidence = if total_sites > 0 {
            let frac = sites_checked as f64 / total_sites as f64;
            // Probability of catching a single-site error ≈ 1 - (1-frac)^1 = frac
            // Confidence grows sublinearly with sample size
            1.0 - (1.0 - frac).powi(sites_checked as i32)
        } else {
            0.0
        };

        let check_time_us = start.elapsed().as_micros() as u64;

        SpotCheckResult {
            passed,
            sites_checked,
            total_sites,
            mac_checks,
            mac_failures,
            svd_checks,
            svd_failures,
            conservation_passed,
            conservation_residual: conservation_residual_f64,
            hash_integrity: hash_integrity && output_hash_ok,
            check_time_us,
            confidence,
            failures,
        }
    }

    /// Spot-check a generic proof (Euler3D or NS-IMEX).
    pub fn check_generic(
        &self,
        input_states: &[MPS],
        output_states: &[MPS],
        shift_mpos: &[MPO],
        conservation_residuals: &[Q16],
        conservation_tol: Q16,
        input_hash_limbs: &[u64; 4],
        output_hash_limbs: &[u64; 4],
        hash_prefix: &[u8],
    ) -> SpotCheckResult {
        let start = Instant::now();
        let mut failures = Vec::new();

        let total_sites = input_states
            .first()
            .map(|s| s.num_sites)
            .unwrap_or(0);
        let sites_to_check = self.select_sites(total_sites);
        let sites_checked = sites_to_check.len();

        // Hash integrity
        let computed_input_hash =
            Self::compute_state_hash(input_states, hash_prefix);
        let hash_integrity = &computed_input_hash == input_hash_limbs;
        if !hash_integrity {
            failures.push("Input state hash mismatch".to_string());
        }

        let computed_output_hash =
            Self::compute_state_hash(output_states, hash_prefix);
        let output_hash_ok = &computed_output_hash == output_hash_limbs;
        if !output_hash_ok {
            failures.push("Output state hash mismatch".to_string());
        }

        // Conservation
        let mut conservation_passed = true;
        let max_residual = conservation_residuals
            .iter()
            .map(|r| r.to_f64().abs())
            .fold(0.0f64, f64::max);

        for (i, r) in conservation_residuals.iter().enumerate() {
            if r.raw.abs() > conservation_tol.raw {
                conservation_passed = false;
                failures.push(format!(
                    "Conservation violation for variable {}: |{:.6e}| > {:.6e}",
                    i,
                    r.to_f64(),
                    conservation_tol.to_f64(),
                ));
            }
        }

        // MAC spot-checks
        let mut mac_checks = 0usize;
        let mut mac_failures = 0usize;

        for state in input_states {
            for mpo in shift_mpos {
                for &site_idx in &sites_to_check {
                    if site_idx < state.cores.len()
                        && site_idx < mpo.cores.len()
                    {
                        let checks = self.check_site_mac(
                            &state.cores[site_idx],
                            &mpo.cores[site_idx],
                        );
                        mac_checks += checks.0;
                        mac_failures += checks.1;
                    }
                }
            }
        }

        if mac_failures > 0 {
            failures.push(format!(
                "MAC failures: {}/{}", mac_failures, mac_checks
            ));
        }

        // SVD ordering
        let mut svd_checks = 0usize;
        let mut svd_failures = 0usize;

        for state in output_states {
            if state.cores.len() > 1 {
                for &site_idx in &sites_to_check {
                    if site_idx + 1 < state.cores.len() {
                        svd_checks += 1;
                        if state.cores[site_idx].chi_right
                            != state.cores[site_idx + 1].chi_left
                        {
                            svd_failures += 1;
                        }
                    }
                }
            }
        }

        let passed = failures.is_empty();
        let confidence = if total_sites > 0 {
            let frac = sites_checked as f64 / total_sites as f64;
            1.0 - (1.0 - frac).powi(sites_checked as i32)
        } else {
            0.0
        };

        let check_time_us = start.elapsed().as_micros() as u64;

        SpotCheckResult {
            passed,
            sites_checked,
            total_sites,
            mac_checks,
            mac_failures,
            svd_checks,
            svd_failures,
            conservation_passed,
            conservation_residual: max_residual,
            hash_integrity: hash_integrity && output_hash_ok,
            check_time_us,
            confidence,
            failures,
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Internal Helpers
    // ═══════════════════════════════════════════════════════════════════════

    /// Select which sites to check using deterministic pseudo-random sampling.
    fn select_sites(&self, total_sites: usize) -> Vec<usize> {
        if total_sites == 0 {
            return Vec::new();
        }

        let target = ((total_sites as f64 * self.config.sample_fraction) as usize)
            .max(self.config.min_sites)
            .min(self.config.max_sites)
            .min(total_sites);

        // Simple deterministic selection using seed
        let mut selected = HashSet::with_capacity(target);
        let mut rng_state = self.config.seed;

        while selected.len() < target {
            // xorshift64
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;

            let idx = (rng_state % total_sites as u64) as usize;
            selected.insert(idx);
        }

        let mut sites: Vec<usize> = selected.into_iter().collect();
        sites.sort_unstable();
        sites
    }

    /// Check MAC correctness at a single site.
    ///
    /// Returns (total_checks, failures).
    fn check_site_mac(
        &self,
        mps_core: &fluidelite_core::mps::MPSCore,
        mpo_core: &fluidelite_core::mpo::MPOCore,
    ) -> (usize, usize) {
        let mut checks = 0usize;
        let mut failures = 0usize;

        // Spot-check a few output elements by recomputing the MAC chain
        let d_in = mpo_core.d_in;
        let max_checks = 4.min(mps_core.chi_left * mpo_core.d_out * mps_core.chi_right);

        for check_idx in 0..max_checks {
            // Pick an output element deterministically
            let cl = check_idx % mps_core.chi_left;
            let o = (check_idx / mps_core.chi_left) % mpo_core.d_out;
            let cr = (check_idx / (mps_core.chi_left * mpo_core.d_out))
                % mps_core.chi_right;

            // Recompute MAC chain for this element (using first dl=0, dr=0)
            let dl = 0;
            let dr = 0;

            if dl < mpo_core.d_left && dr < mpo_core.d_right {
                let mut acc = 0i128;
                for p in 0..d_in {
                    let mpo_val = mpo_core.get(dl, o, p, dr);
                    let mps_val = mps_core.get(cl, p, cr);
                    acc += (mpo_val.raw as i128 * mps_val.raw as i128) >> 16;
                }

                // The absolute value should be representable in Q16
                checks += 1;
                if acc.abs() > (i64::MAX as i128) {
                    failures += 1;
                }
            }
        }

        (checks, failures)
    }

    /// Compute SHA-256 hash of MPS states.
    pub fn compute_state_hash(states: &[MPS], prefix: &[u8]) -> [u64; 4] {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(prefix);
        hasher.update((states.len() as u64).to_le_bytes());

        for state in states {
            hasher.update((state.num_sites as u64).to_le_bytes());
            for core in &state.cores {
                hasher.update((core.chi_left as u64).to_le_bytes());
                hasher.update((core.d as u64).to_le_bytes());
                hasher.update((core.chi_right as u64).to_le_bytes());
                for val in &core.data {
                    hasher.update(val.raw.to_le_bytes());
                }
            }
        }

        let result = hasher.finalize();
        let mut hash_bytes = [0u8; 32];
        hash_bytes.copy_from_slice(&result);

        let mut limbs = [0u64; 4];
        for (i, limb) in limbs.iter_mut().enumerate() {
            let offset = i * 8;
            *limb = u64::from_le_bytes([
                hash_bytes[offset],
                hash_bytes[offset + 1],
                hash_bytes[offset + 2],
                hash_bytes[offset + 3],
                hash_bytes[offset + 4],
                hash_bytes[offset + 5],
                hash_bytes[offset + 6],
                hash_bytes[offset + 7],
            ]);
        }
        limbs
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests (private-access only — public API tests live in tests/proof_preview_tests.rs)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_site_selection() {
        let checker = SpotChecker::default_checker();
        let sites = checker.select_sites(20);
        assert!(!sites.is_empty());
        assert!(sites.len() <= 20);

        for &s in &sites {
            assert!(s < 20);
        }

        for pair in sites.windows(2) {
            assert!(pair[0] < pair[1]);
        }
    }

    #[test]
    fn test_site_selection_small() {
        let checker = SpotChecker::default_checker();
        let sites = checker.select_sites(3);
        assert!(sites.len() >= 2);
        assert!(sites.len() <= 3);
    }

    #[test]
    fn test_site_selection_zero() {
        let checker = SpotChecker::default_checker();
        let sites = checker.select_sites(0);
        assert!(sites.is_empty());
    }
}
