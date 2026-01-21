//! Tensor network operations for MPS and MPO
//!
//! Core operations:
//! 1. MPO × MPS contraction: Apply MPO to MPS
//! 2. MPS + MPS direct sum: Block-diagonal addition
//! 3. MPS truncation: Reduce bond dimension
//!
//! These operations form the computational core of FluidElite inference.

use crate::field::Q16;
use crate::mpo::MPO;
use crate::mps::{MPSCore, MPS};

/// Error type for tensor operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorOpError {
    /// MPS and MPO have different number of sites
    SiteMismatch { mps_sites: usize, mpo_sites: usize },
    /// MPS states have different number of sites
    MpsSiteMismatch { a_sites: usize, b_sites: usize },
}

impl std::fmt::Display for TensorOpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SiteMismatch { mps_sites, mpo_sites } => {
                write!(f, "MPS sites ({}) != MPO sites ({})", mps_sites, mpo_sites)
            }
            Self::MpsSiteMismatch { a_sites, b_sites } => {
                write!(f, "MPS A sites ({}) != MPS B sites ({})", a_sites, b_sites)
            }
        }
    }
}

impl std::error::Error for TensorOpError {}

/// Apply MPO to MPS, producing a new MPS with increased bond dimension
///
/// Input dimensions:
///   MPS core: (χ_l, d, χ_r)
///   MPO core: (D_l, d_out, d_in, D_r)
///
/// Output dimensions:
///   New MPS core: (χ_l × D_l, d_out, χ_r × D_r)
///
/// Contraction: sum over d_in = physical input dimension
///
/// # Errors
/// Returns `TensorOpError::SiteMismatch` if MPS and MPO have different number of sites.
pub fn apply_mpo(mps: &MPS, mpo: &MPO) -> Result<MPS, TensorOpError> {
    if mps.num_sites != mpo.num_sites {
        return Err(TensorOpError::SiteMismatch {
            mps_sites: mps.num_sites,
            mpo_sites: mpo.num_sites,
        });
    }

    let num_sites = mps.num_sites;
    let mut new_cores = Vec::with_capacity(num_sites);

    for i in 0..num_sites {
        let mps_core = &mps.cores[i];
        let mpo_core = &mpo.cores[i];

        // New dimensions
        let new_chi_left = mps_core.chi_left * mpo_core.d_left;
        let new_d = mpo_core.d_out;
        let new_chi_right = mps_core.chi_right * mpo_core.d_right;

        let mut new_data = vec![Q16::zero(); new_chi_left * new_d * new_chi_right];

        // Contraction: sum over input physical dimension
        // new[cl*dl + dl', o, cr*dr + dr'] = sum_i mpo[dl, o, i, dr] * mps[cl, i, cr]
        for cl in 0..mps_core.chi_left {
            for dl in 0..mpo_core.d_left {
                let new_left = cl * mpo_core.d_left + dl;

                for o in 0..mpo_core.d_out {
                    for cr in 0..mps_core.chi_right {
                        for dr in 0..mpo_core.d_right {
                            let new_right = cr * mpo_core.d_right + dr;

                            // Sum over input physical index
                            let mut sum = Q16::zero();
                            for i in 0..mpo_core.d_in {
                                let mpo_val = mpo_core.get(dl, o, i, dr);
                                let mps_val = mps_core.get(cl, i, cr);
                                sum = sum.mac(mpo_val, mps_val);
                            }

                            new_data[(new_left * new_d + o) * new_chi_right + new_right] = sum;
                        }
                    }
                }
            }
        }

        new_cores.push(MPSCore::from_data(new_data, new_chi_left, new_d, new_chi_right));
    }

    Ok(MPS {
        cores: new_cores,
        num_sites,
    })
}

/// Add two MPS using block-diagonal direct sum
///
/// Input:
///   MPS A cores: (χ_a_l, d, χ_a_r)
///   MPS B cores: (χ_b_l, d, χ_b_r)
///
/// Output:
///   New MPS cores: (χ_a_l + χ_b_l, d, χ_a_r + χ_b_r)
///   with block-diagonal structure
///
/// Special handling for boundary cores to maintain valid MPS structure.
pub fn add_mps(a: &MPS, b: &MPS) -> MPS {
    assert_eq!(
        a.num_sites, b.num_sites,
        "MPS must have same number of sites"
    );

    let num_sites = a.num_sites;
    let mut new_cores = Vec::with_capacity(num_sites);

    for i in 0..num_sites {
        let a_core = &a.cores[i];
        let b_core = &b.cores[i];

        assert_eq!(a_core.d, b_core.d, "Physical dimensions must match");
        let d = a_core.d;

        if i == 0 {
            // First site: concatenate along right bond
            // (1, d, χ_a) + (1, d, χ_b) -> (1, d, χ_a + χ_b)
            let new_chi_right = a_core.chi_right + b_core.chi_right;
            let mut new_data = vec![Q16::zero(); 1 * d * new_chi_right];

            for p in 0..d {
                for r in 0..a_core.chi_right {
                    new_data[(0 * d + p) * new_chi_right + r] = a_core.get(0, p, r);
                }
                for r in 0..b_core.chi_right {
                    new_data[(0 * d + p) * new_chi_right + a_core.chi_right + r] =
                        b_core.get(0, p, r);
                }
            }

            new_cores.push(MPSCore::from_data(new_data, 1, d, new_chi_right));
        } else if i == num_sites - 1 {
            // Last site: concatenate along left bond
            // (χ_a, d, 1) + (χ_b, d, 1) -> (χ_a + χ_b, d, 1)
            let new_chi_left = a_core.chi_left + b_core.chi_left;
            let mut new_data = vec![Q16::zero(); new_chi_left * d * 1];

            for l in 0..a_core.chi_left {
                for p in 0..d {
                    new_data[(l * d + p) * 1 + 0] = a_core.get(l, p, 0);
                }
            }
            for l in 0..b_core.chi_left {
                for p in 0..d {
                    new_data[((a_core.chi_left + l) * d + p) * 1 + 0] = b_core.get(l, p, 0);
                }
            }

            new_cores.push(MPSCore::from_data(new_data, new_chi_left, d, 1));
        } else {
            // Middle sites: block diagonal
            // (χ_a_l, d, χ_a_r) + (χ_b_l, d, χ_b_r) -> block diagonal
            let new_chi_left = a_core.chi_left + b_core.chi_left;
            let new_chi_right = a_core.chi_right + b_core.chi_right;
            let mut new_data = vec![Q16::zero(); new_chi_left * d * new_chi_right];

            // A block in top-left
            for l in 0..a_core.chi_left {
                for p in 0..d {
                    for r in 0..a_core.chi_right {
                        new_data[(l * d + p) * new_chi_right + r] = a_core.get(l, p, r);
                    }
                }
            }

            // B block in bottom-right
            for l in 0..b_core.chi_left {
                for p in 0..d {
                    for r in 0..b_core.chi_right {
                        new_data[((a_core.chi_left + l) * d + p) * new_chi_right
                            + a_core.chi_right
                            + r] = b_core.get(l, p, r);
                    }
                }
            }

            new_cores.push(MPSCore::from_data(new_data, new_chi_left, d, new_chi_right));
        }
    }

    MPS {
        cores: new_cores,
        num_sites,
    }
}

/// Complete FluidElite forward step
///
/// context' = W_hidden @ context + W_input @ token_embed
/// then truncate to chi_max
pub fn fluidelite_step(
    context: &MPS,
    token_id: usize,
    w_hidden: &MPO,
    w_input: &MPO,
    chi_max: usize,
) -> MPS {
    let num_sites = context.num_sites;

    // Embed token
    let token_mps = MPS::embed_token(token_id, num_sites);

    // Apply MPOs (unwrap safe: dimensions validated at construction)
    let h_term = apply_mpo(context, w_hidden).expect("MPS/MPO site mismatch in h_term");
    let x_term = apply_mpo(&token_mps, w_input).expect("MPS/MPO site mismatch in x_term");

    // Add
    let mut combined = add_mps(&h_term, &x_term);

    // Truncate
    combined.truncate(chi_max);

    combined
}

/// Readout: extract logits from MPS
/// Takes middle bond and projects to vocabulary size
pub fn readout(mps: &MPS, readout_weights: &[Q16], vocab_size: usize) -> Vec<Q16> {
    let mid = mps.num_sites / 2;
    let mid_core = &mps.cores[mid];

    // Average over physical dimension, flatten bonds
    let bond_size = mid_core.chi_left * mid_core.chi_right;
    let mut features = vec![Q16::zero(); bond_size];

    for l in 0..mid_core.chi_left {
        for r in 0..mid_core.chi_right {
            let idx = l * mid_core.chi_right + r;
            let mut sum = Q16::zero();
            for p in 0..mid_core.d {
                sum = sum + mid_core.get(l, p, r);
            }
            features[idx] = sum;
        }
    }

    // Linear projection to vocab
    // readout_weights: [vocab_size, feature_size]
    let mut logits = vec![Q16::zero(); vocab_size];
    let feature_size = readout_weights.len() / vocab_size;

    for v in 0..vocab_size {
        let mut sum = Q16::zero();
        let effective_features = bond_size.min(feature_size);
        for f in 0..effective_features {
            if v * feature_size + f < readout_weights.len() {
                let w = readout_weights[v * feature_size + f];
                sum = sum + w.mul(features[f.min(bond_size - 1)]);
            }
        }
        logits[v] = sum;
    }

    logits
}

/// Count the number of arithmetic operations for constraint estimation.
/// 
/// Returns `None` if the calculation would overflow `usize`.
/// Use `count_ops_unchecked` for hot paths where overflow is guaranteed not to occur.
pub fn count_ops(mps_chi: usize, mpo_d: usize, d_phys: usize, num_sites: usize) -> Option<usize> {
    // MPO × MPS: For each output element, we sum over d_phys
    // Output size per site: (χ × D) × d_phys × (χ × D)
    let chi_d = mps_chi.checked_mul(mpo_d)?;
    let chi_d_squared = chi_d.checked_mul(chi_d)?;
    let d_phys_squared = d_phys.checked_mul(d_phys)?;
    let per_site = chi_d_squared.checked_mul(d_phys_squared)?;
    let mpo_mps_ops = num_sites.checked_mul(per_site)?;

    // Addition: just copying, no arithmetic
    // Total multiplications (each becomes ~1 constraint in Halo2)
    Some(mpo_mps_ops)
}

/// Count ops without overflow checking (for hot paths with known-safe inputs)
#[inline]
pub fn count_ops_unchecked(mps_chi: usize, mpo_d: usize, d_phys: usize, num_sites: usize) -> usize {
    let chi_d = mps_chi * mpo_d;
    num_sites * chi_d * chi_d * d_phys * d_phys
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_mpo_apply() {
        // Applying identity MPO should preserve the MPS
        let mps = MPS::embed_token(5, 4);
        let identity = MPO::identity(4, 2);

        let result = apply_mpo(&mps, &identity).expect("apply_mpo failed");

        // Check dimensions preserved
        assert_eq!(result.num_sites, 4);
        for i in 0..4 {
            assert_eq!(result.cores[i].d, 2);
        }

        // For identity with D=1, chi should be preserved
        assert_eq!(result.cores[0].chi_left, 1);
        assert_eq!(result.cores[0].chi_right, 1);
    }

    #[test]
    fn test_mps_addition() {
        let a = MPS::embed_token(3, 4);
        let b = MPS::embed_token(5, 4);

        let sum = add_mps(&a, &b);

        // Bond dimensions should double (except boundaries stay correct)
        assert_eq!(sum.cores[0].chi_left, 1);
        assert_eq!(sum.cores[0].chi_right, 2); // 1 + 1
        assert_eq!(sum.cores[1].chi_left, 2);
        assert_eq!(sum.cores[1].chi_right, 2);
        assert_eq!(sum.cores[3].chi_left, 2);
        assert_eq!(sum.cores[3].chi_right, 1);
    }

    #[test]
    fn test_fluidelite_step() {
        let context = MPS::new(4, 4, 2);
        let w_hidden = MPO::identity(4, 2);
        let w_input = MPO::identity(4, 2);

        let new_ctx = fluidelite_step(&context, 7, &w_hidden, &w_input, 8);

        // Should have valid structure
        assert_eq!(new_ctx.num_sites, 4);
        assert_eq!(new_ctx.cores[0].chi_left, 1);
        assert_eq!(new_ctx.cores[3].chi_right, 1);
    }

    #[test]
    fn test_op_count() {
        // L=16, χ=64, D=1, d=2
        let ops = count_ops(64, 1, 2, 16).expect("count_ops should not overflow");
        println!("Operations: {}", ops);

        // Should be around 131k
        assert!(ops > 50_000);
        assert!(ops < 500_000);
    }
}
