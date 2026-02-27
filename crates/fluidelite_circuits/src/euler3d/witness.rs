//! Witness types and generation for the Euler 3D proof circuit.
//!
//! Uses lightweight `Mps`/`Mpo` types from `crate::tensor` —
//! flat `Vec<Q16>` storage, no `Serialize`/`Deserialize` derives.
//! Full MPS/MPO types are only used at the `generate()` API boundary.
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use std::fmt;

use fluidelite_core::field::Q16;

use crate::tensor::{Mps, Mpo};

use super::config::{
    ConservedVariable, Euler3DCircuitSizing, Euler3DParams, StrangStage,
    NUM_CONSERVED_VARIABLES, NUM_STRANG_STAGES, Q16_FRAC_BITS,
};

// ═══════════════════════════════════════════════════════════════════════════
// Core Witness Types
// ═══════════════════════════════════════════════════════════════════════════

/// Complete witness for proving one Euler 3D timestep.
#[derive(Debug, Clone)]
pub struct Euler3DWitness {
    /// Physics parameters (public).
    pub params: Euler3DParams,

    /// Input states for each conserved variable [ρ, ρu, ρv, ρw, E].
    pub input_states: Vec<Mps>,

    /// Output states for each conserved variable.
    pub output_states: Vec<Mps>,

    /// Shift MPOs for each spatial axis [x, y, z].
    pub shift_mpos: Vec<Mpo>,

    /// Witness data for each Strang splitting stage.
    pub strang_stages: Vec<StrangStageWitness>,

    /// Conservation witness (integrals before/after).
    pub conservation: ConservationWitness,

    /// Hash commitments (public inputs).
    pub hashes: HashWitness,
}

/// Witness for one Strang splitting stage.
#[derive(Debug, Clone)]
pub struct StrangStageWitness {
    /// Which Strang stage this is.
    pub stage: StrangStage,

    /// Per-variable sweep witnesses.
    pub variable_sweeps: Vec<VariableSweepWitness>,
}

/// Witness for one variable's directional sweep within a Strang stage.
#[derive(Debug, Clone)]
pub struct VariableSweepWitness {
    /// Which conserved variable.
    pub variable: ConservedVariable,

    /// MPO×MPS contraction witness (shift MPO applied to state).
    pub contraction: ContractionWitness,

    /// QTT subtraction: shifted - original.
    pub subtraction: QttArithmeticWitness,

    /// QTT scaling: multiply by -dt/dx coefficient.
    pub scaling: QttArithmeticWitness,

    /// QTT addition: original + scaled_delta.
    pub addition: QttArithmeticWitness,

    /// SVD truncation after update.
    pub truncation: SvdTruncationWitness,

    /// State after this sweep (pre-truncation).
    pub state_pre_truncation: Mps,

    /// State after truncation.
    pub state_post_truncation: Mps,
}

// ═══════════════════════════════════════════════════════════════════════════
// Operation-Level Witnesses
// ═══════════════════════════════════════════════════════════════════════════

/// Witness for a single MPO × MPS contraction.
#[derive(Debug, Clone)]
pub struct ContractionWitness {
    /// Number of sites.
    pub num_sites: usize,

    /// Per-site contraction data.
    pub site_data: Vec<SiteContractionData>,

    /// Output MPS after contraction (before truncation).
    pub output_mps: Mps,
}

/// Per-site data for MPO×MPS contraction.
#[derive(Debug, Clone)]
pub struct SiteContractionData {
    /// Accumulator values for each MAC chain.
    pub mac_accumulators: Vec<Vec<Q16>>,

    /// Q16 multiplication remainders for range checks.
    pub fp_remainders: Vec<i64>,

    /// Q16 multiplication quotients.
    pub fp_quotients: Vec<Q16>,
}

/// Witness for QTT arithmetic operations (add, subtract, scale).
#[derive(Debug, Clone)]
pub struct QttArithmeticWitness {
    /// Result MPS.
    pub result: Mps,

    /// For scaling operations: the Q16 remainders from fixed-point multiply.
    pub fp_remainders: Vec<i64>,

    /// For scaling operations: the Q16 quotients.
    pub fp_quotients: Vec<Q16>,
}

/// Witness for SVD truncation.
#[derive(Debug, Clone)]
pub struct SvdTruncationWitness {
    /// Number of SVD operations (one per bond).
    pub num_bonds: usize,

    /// Per-bond SVD data.
    pub bond_data: Vec<BondSvdData>,

    /// Total truncation error.
    pub total_truncation_error: Q16,

    /// Output rank after truncation.
    pub output_rank: usize,
}

/// SVD data for a single bond in the MPS.
#[derive(Debug, Clone)]
pub struct BondSvdData {
    /// Bond index (between site i and site i+1).
    pub bond_index: usize,

    /// Singular values (sorted descending, all non-negative).
    pub singular_values: Vec<Q16>,

    /// Rank after truncation at this bond.
    pub truncated_rank: usize,

    /// Truncation error at this bond.
    pub bond_truncation_error: Q16,

    /// Bit decomposition of (s_i - s_{i+1}) for ordering proof.
    pub sv_ordering_bits: Vec<Vec<bool>>,

    /// Bit decomposition of each singular value for non-negativity proof.
    pub sv_nonneg_bits: Vec<Vec<bool>>,
}

/// Conservation quantities before and after the timestep.
#[derive(Debug, Clone)]
pub struct ConservationWitness {
    /// Integrals of each conserved variable before the timestep.
    pub integrals_before: Vec<Q16>,

    /// Integrals of each conserved variable after the timestep.
    pub integrals_after: Vec<Q16>,

    /// Residual for each variable: |after - before|.
    pub residuals: Vec<Q16>,

    /// Bit decomposition of (tolerance - residual) for each variable.
    pub residual_bound_bits: Vec<Vec<bool>>,
}

/// Hash commitments that become public inputs.
#[derive(Debug, Clone)]
pub struct HashWitness {
    /// SHA-256 of input states (split into 4 × 64-bit limbs).
    pub input_state_hash_limbs: [u64; 4],

    /// SHA-256 of output states (split into 4 × 64-bit limbs).
    pub output_state_hash_limbs: [u64; 4],

    /// SHA-256 of physics parameters (split into 4 × 64-bit limbs).
    pub params_hash_limbs: [u64; 4],
}

// ═══════════════════════════════════════════════════════════════════════════
// Witness Generation
// ═══════════════════════════════════════════════════════════════════════════

/// Generates witnesses by replaying the Euler 3D QTT solver computation.
pub struct WitnessGenerator {
    /// Physics parameters.
    params: Euler3DParams,

    /// Circuit sizing.
    sizing: Euler3DCircuitSizing,
}

impl WitnessGenerator {
    /// Create a new witness generator.
    pub fn new(params: Euler3DParams) -> Self {
        let sizing = Euler3DCircuitSizing::from_params(&params);
        Self { params, sizing }
    }

    /// Get the circuit sizing.
    pub fn sizing(&self) -> &Euler3DCircuitSizing {
        &self.sizing
    }

    /// Generate a complete witness from input states and shift MPOs.
    ///
    /// Full MPS/MPO types are accepted at this boundary and converted
    /// to thin types internally.
    pub fn generate(
        &self,
        input_states: &[fluidelite_core::mps::MPS],
        shift_mpos: &[fluidelite_core::mpo::MPO],
    ) -> Result<Euler3DWitness, WitnessError> {
        if input_states.len() != NUM_CONSERVED_VARIABLES {
            return Err(WitnessError::WrongVariableCount {
                expected: NUM_CONSERVED_VARIABLES,
                got: input_states.len(),
            });
        }
        if shift_mpos.len() != 3 {
            return Err(WitnessError::WrongMpoCount {
                expected: 3,
                got: shift_mpos.len(),
            });
        }

        // ── Convert to thin types at the API boundary ──
        let mut current_states: Vec<Mps> = input_states
            .iter()
            .map(|m| Mps::from_full(m))
            .collect();
        let thin_shift_mpos: Vec<Mpo> = shift_mpos
            .iter()
            .map(|m| Mpo::from_full(m))
            .collect();
        let thin_input_states: Vec<Mps> = current_states.clone();

        // Record input conservation integrals
        let integrals_before: Vec<Q16> = current_states
            .iter()
            .map(|mps| Self::compute_mps_integral(mps))
            .collect();

        // Compute hashes
        let input_refs: Vec<&Mps> = current_states.iter().collect();
        let input_hash_limbs = Self::hash_mps_to_limbs(&input_refs);
        let params_hash_limbs = Self::hash_to_limbs(&self.params.hash());

        // Execute Strang splitting stages
        let mut strang_stages = Vec::with_capacity(NUM_STRANG_STAGES);

        for stage in StrangStage::ALL {
            let stage_witness = self.execute_strang_stage(
                stage,
                &mut current_states,
                &thin_shift_mpos,
            )?;
            strang_stages.push(stage_witness);
        }

        // Record output conservation integrals
        let integrals_after: Vec<Q16> = current_states
            .iter()
            .map(|mps| Self::compute_mps_integral(mps))
            .collect();

        // Compute conservation residuals
        let residuals: Vec<Q16> = integrals_before
            .iter()
            .zip(integrals_after.iter())
            .map(|(before, after)| (*after - *before).abs())
            .collect();

        let residual_bound_bits: Vec<Vec<bool>> = residuals
            .iter()
            .map(|r| {
                let bound = self.params.conservation_tolerance.raw - r.raw;
                Self::decompose_nonneg_to_bits(bound, 32)
            })
            .collect();

        let output_refs: Vec<&Mps> = current_states.iter().collect();
        let output_hash_limbs = Self::hash_mps_to_limbs(&output_refs);

        let conservation = ConservationWitness {
            integrals_before,
            integrals_after,
            residuals,
            residual_bound_bits,
        };

        let hashes = HashWitness {
            input_state_hash_limbs: input_hash_limbs,
            output_state_hash_limbs: output_hash_limbs,
            params_hash_limbs,
        };

        Ok(Euler3DWitness {
            params: self.params.clone(),
            input_states: thin_input_states,
            output_states: current_states,
            shift_mpos: thin_shift_mpos,
            strang_stages,
            conservation,
            hashes,
        })
    }

    /// Execute one Strang splitting stage, recording witness data.
    fn execute_strang_stage(
        &self,
        stage: StrangStage,
        states: &mut [Mps],
        shift_mpos: &[Mpo],
    ) -> Result<StrangStageWitness, WitnessError> {
        let axis = stage.axis();
        let mpo = &shift_mpos[axis];

        // Compute effective dt for this stage
        let effective_dt = if stage.is_half_step() {
            Q16::from_raw(self.params.dt.raw / 2)
        } else {
            self.params.dt
        };

        // Coefficient: -dt/dx
        let neg_dt_over_dx = if self.params.dx.raw != 0 {
            let dt_dx_raw = ((effective_dt.raw as i128) << Q16_FRAC_BITS)
                / (self.params.dx.raw as i128);
            Q16::from_raw(-(dt_dx_raw as i64))
        } else {
            Q16::zero()
        };

        let mut variable_sweeps = Vec::with_capacity(NUM_CONSERVED_VARIABLES);

        for var in ConservedVariable::ALL {
            let idx = var.index();
            let sweep = self.execute_variable_sweep(
                var,
                &states[idx],
                mpo,
                neg_dt_over_dx,
            )?;
            states[idx] = sweep.state_post_truncation.clone();
            variable_sweeps.push(sweep);
        }

        Ok(StrangStageWitness {
            stage,
            variable_sweeps,
        })
    }

    /// Execute one variable's directional sweep, recording witness data.
    fn execute_variable_sweep(
        &self,
        variable: ConservedVariable,
        state: &Mps,
        shift_mpo: &Mpo,
        neg_dt_over_dx: Q16,
    ) -> Result<VariableSweepWitness, WitnessError> {
        let chi_max = self.params.chi_max;

        // Step 1: Apply shift MPO
        let (shifted, contraction) = self.apply_mpo_with_witness(state, shift_mpo)?;

        // Step 2: Subtract original: delta = shifted - state
        let (delta, subtraction) = self.qtt_subtract_with_witness(&shifted, state);

        // Step 3: Scale by -dt/dx: scaled = delta * neg_dt_over_dx
        let (scaled, scaling) = self.qtt_scale_with_witness(&delta, neg_dt_over_dx);

        // Step 4: Add to original: updated = state + scaled
        let (mut updated, addition) = self.qtt_add_with_witness(state, &scaled);

        let state_pre_truncation = updated.clone();

        // Step 5: Truncate
        let truncation = self.truncate_with_witness(&mut updated, chi_max);

        let state_post_truncation = updated;

        Ok(VariableSweepWitness {
            variable,
            contraction,
            subtraction,
            scaling,
            addition,
            truncation,
            state_pre_truncation,
            state_post_truncation,
        })
    }

    /// Apply MPO to MPS, recording contraction witness data.
    fn apply_mpo_with_witness(
        &self,
        mps: &Mps,
        mpo: &Mpo,
    ) -> Result<(Mps, ContractionWitness), WitnessError> {
        if mps.num_sites != mpo.num_sites {
            return Err(WitnessError::SiteMismatch {
                mps_sites: mps.num_sites,
                mpo_sites: mpo.num_sites,
            });
        }

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
                                    let mpo_val = mpo.get(i, dl, o, p, dr);
                                    let mps_val = mps.get(i, cl, p, cr);

                                    let full_product =
                                        mpo_val.raw as i128 * mps_val.raw as i128;
                                    let quotient =
                                        (full_product >> Q16_FRAC_BITS) as i64;
                                    let remainder =
                                        (full_product - ((quotient as i128) << Q16_FRAC_BITS))
                                            as i64;

                                    fp_quotients.push(Q16::from_raw(quotient));
                                    fp_remainders.push(remainder);

                                    acc = Q16::from_raw(acc.raw + quotient);
                                    chain.push(acc);
                                }

                                mac_accumulators.push(chain);
                                new_data[(new_left * d_out + o) * new_chi_right
                                    + new_right] = acc;
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

        Ok((output_mps, witness))
    }

    /// QTT subtraction (a - b) with witness recording.
    fn qtt_subtract_with_witness(&self, a: &Mps, b: &Mps) -> (Mps, QttArithmeticWitness) {
        let neg_b = Self::negate_mps(b);
        let result = crate::tensor::add(a, &neg_b);

        let witness = QttArithmeticWitness {
            result: result.clone(),
            fp_remainders: Vec::new(),
            fp_quotients: Vec::new(),
        };

        (result, witness)
    }

    /// QTT scaling (mps * scalar) with witness recording.
    fn qtt_scale_with_witness(&self, mps: &Mps, scalar: Q16) -> (Mps, QttArithmeticWitness) {
        let mut result = mps.clone();
        let mut fp_remainders = Vec::new();
        let mut fp_quotients = Vec::new();

        if result.num_sites > 0 {
            let core = result.core_data_mut(0);
            for val in core.iter_mut() {
                let full_product = val.raw as i128 * scalar.raw as i128;
                let quotient = (full_product >> Q16_FRAC_BITS) as i64;
                let remainder =
                    (full_product - ((quotient as i128) << Q16_FRAC_BITS)) as i64;

                fp_quotients.push(Q16::from_raw(quotient));
                fp_remainders.push(remainder);

                *val = Q16::from_raw(quotient);
            }
        }

        let witness = QttArithmeticWitness {
            result: result.clone(),
            fp_remainders,
            fp_quotients,
        };

        (result, witness)
    }

    /// QTT addition (a + b) with witness recording.
    fn qtt_add_with_witness(&self, a: &Mps, b: &Mps) -> (Mps, QttArithmeticWitness) {
        let result = crate::tensor::add(a, b);

        let witness = QttArithmeticWitness {
            result: result.clone(),
            fp_remainders: Vec::new(),
            fp_quotients: Vec::new(),
        };

        (result, witness)
    }

    /// Truncate MPS with SVD witness recording.
    fn truncate_with_witness(&self, mps: &mut Mps, chi_max: usize) -> SvdTruncationWitness {
        let num_bonds = if mps.num_sites > 0 {
            mps.num_sites - 1
        } else {
            0
        };
        let mut bond_data = Vec::with_capacity(num_bonds);
        let mut total_error_sq = 0i128;

        for bond_idx in 0..num_bonds {
            let left_chi = mps.chi_right(bond_idx);
            let right_chi = mps.chi_left(bond_idx + 1);
            let current_rank = left_chi.min(right_chi);
            let truncated_rank = current_rank.min(chi_max);

            let singular_values = Self::estimate_singular_values(
                mps,
                bond_idx,
                current_rank,
            );

            let mut bond_error_sq = 0i128;
            for sv in singular_values.iter().skip(truncated_rank) {
                bond_error_sq += (sv.raw as i128) * (sv.raw as i128);
            }
            total_error_sq += bond_error_sq;

            let bond_error = Self::q16_sqrt_approx(bond_error_sq);

            let sv_ordering_bits: Vec<Vec<bool>> = singular_values
                .windows(2)
                .map(|pair| {
                    let diff = pair[0].raw - pair[1].raw;
                    Self::decompose_nonneg_to_bits(diff, 32)
                })
                .collect();

            let sv_nonneg_bits: Vec<Vec<bool>> = singular_values
                .iter()
                .map(|sv| Self::decompose_nonneg_to_bits(sv.raw, 32))
                .collect();

            bond_data.push(BondSvdData {
                bond_index: bond_idx,
                singular_values,
                truncated_rank,
                bond_truncation_error: bond_error,
                sv_ordering_bits,
                sv_nonneg_bits,
            });
        }

        mps.truncate(chi_max);

        let total_truncation_error = Self::q16_sqrt_approx(total_error_sq);
        let output_rank = mps.max_chi();

        SvdTruncationWitness {
            num_bonds,
            bond_data,
            total_truncation_error,
            output_rank,
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Helper Methods
    // ═══════════════════════════════════════════════════════════════════════

    /// Negate an MPS by negating the first core.
    fn negate_mps(mps: &Mps) -> Mps {
        let mut result = mps.clone();
        if result.num_sites > 0 {
            let core = result.core_data_mut(0);
            for val in core.iter_mut() {
                *val = -*val;
            }
        }
        result
    }

    /// Compute a proxy for the integral of an MPS state.
    pub fn compute_mps_integral(mps: &Mps) -> Q16 {
        let mut sum = Q16::zero();
        for val in mps.flat_data() {
            sum = sum + *val;
        }
        sum
    }

    /// Estimate singular values from an MPS core's Frobenius norms.
    fn estimate_singular_values(mps: &Mps, site: usize, rank: usize) -> Vec<Q16> {
        let mut svs = Vec::with_capacity(rank);

        for r in 0..rank.min(mps.chi_right(site)) {
            let mut norm_sq = 0i128;
            for l in 0..mps.chi_left(site) {
                for p in 0..mps.d() {
                    let val = mps.get(site, l, p, r);
                    norm_sq += (val.raw as i128) * (val.raw as i128);
                }
            }
            let norm = Self::q16_sqrt_approx(norm_sq);
            svs.push(norm);
        }

        svs.sort_by(|a, b| b.raw.cmp(&a.raw));
        svs
    }

    /// Approximate integer square root in Q16 domain.
    fn q16_sqrt_approx(sum_sq: i128) -> Q16 {
        if sum_sq <= 0 {
            return Q16::zero();
        }
        let mut x = sum_sq;
        let mut y = (x + 1) / 2;
        while y < x {
            x = y;
            y = (x + sum_sq / x) / 2;
        }
        Q16::from_raw(x.min(i64::MAX as i128) as i64)
    }

    /// Decompose a non-negative value into bits (LSB first).
    fn decompose_nonneg_to_bits(value: i64, num_bits: usize) -> Vec<bool> {
        if value < 0 {
            return vec![false; num_bits];
        }
        let v = value as u64;
        (0..num_bits).map(|i| (v >> i) & 1 == 1).collect()
    }

    /// Convert a 32-byte SHA-256 hash into 4 × 64-bit limbs.
    fn hash_to_limbs(hash: &[u8; 32]) -> [u64; 4] {
        let mut limbs = [0u64; 4];
        for (i, limb) in limbs.iter_mut().enumerate() {
            let offset = i * 8;
            *limb = u64::from_le_bytes([
                hash[offset],
                hash[offset + 1],
                hash[offset + 2],
                hash[offset + 3],
                hash[offset + 4],
                hash[offset + 5],
                hash[offset + 6],
                hash[offset + 7],
            ]);
        }
        limbs
    }

    /// Compute SHA-256 hash of MPS states and return as limbs.
    fn hash_mps_to_limbs(states: &[&Mps]) -> [u64; 4] {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(b"EULER3D_STATE_V1");
        hasher.update((states.len() as u64).to_le_bytes());

        for state in states {
            hasher.update((state.num_sites as u64).to_le_bytes());
            for i in 0..state.num_sites {
                hasher.update((state.chi_left(i) as u64).to_le_bytes());
                hasher.update((state.d() as u64).to_le_bytes());
                hasher.update((state.chi_right(i) as u64).to_le_bytes());
                for val in state.core_data(i) {
                    hasher.update(val.raw.to_le_bytes());
                }
            }
        }

        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        Self::hash_to_limbs(&hash)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Error Types
// ═══════════════════════════════════════════════════════════════════════════

/// Errors during witness generation.
#[derive(Debug, Clone)]
pub enum WitnessError {
    /// Wrong number of conserved variables.
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
    /// MPS and MPO site count mismatch.
    SiteMismatch {
        /// MPS site count.
        mps_sites: usize,
        /// MPO site count.
        mpo_sites: usize,
    },
}

impl fmt::Display for WitnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WitnessError::WrongVariableCount { expected, got } => {
                write!(f, "Expected {} variables, got {}", expected, got)
            }
            WitnessError::WrongMpoCount { expected, got } => {
                write!(f, "Expected {} shift MPOs, got {}", expected, got)
            }
            WitnessError::SiteMismatch {
                mps_sites,
                mpo_sites,
            } => {
                write!(
                    f,
                    "MPS sites ({}) != MPO sites ({})",
                    mps_sites, mpo_sites
                )
            }
        }
    }
}

impl std::error::Error for WitnessError {}

// ═══════════════════════════════════════════════════════════════════════════
// Tests (private method access — integration tests cover public API)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use fluidelite_core::field::Q16;

    #[test]
    fn test_bit_decomposition() {
        let bits = WitnessGenerator::decompose_nonneg_to_bits(42, 8);
        assert_eq!(bits.len(), 8);
        assert!(!bits[0]);
        assert!(bits[1]);
        assert!(!bits[2]);
        assert!(bits[3]);
        assert!(!bits[4]);
        assert!(bits[5]);
        assert!(!bits[6]);
        assert!(!bits[7]);
    }

    #[test]
    fn test_negative_bit_decomposition() {
        let bits = WitnessGenerator::decompose_nonneg_to_bits(-5, 8);
        assert!(bits.iter().all(|&b| !b));
    }

    #[test]
    fn test_q16_sqrt_approx() {
        let two_q16 = Q16::from_f64(2.0).raw as i128;
        let sum_sq = two_q16 * two_q16;
        let result = WitnessGenerator::q16_sqrt_approx(sum_sq);
        assert!(
            (result.to_f64() - 2.0).abs() < 0.01,
            "Expected ~2.0, got {}",
            result.to_f64()
        );
    }
}
