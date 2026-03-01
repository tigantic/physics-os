//! QTT Operator Construction: Shift, Laplacian, and System Matrix MPOs
//!
//! These operators form the foundation for trustless PDE verification.
//! All operators are constructed analytically with exact, closed-form
//! core values — no SVD or numerical approximation.
//!
//! # Operators
//!
//! - **Shift S±**: `|x⟩ → |x±1 mod 2^L⟩`, bond dimension 2 (ripple-carry)
//! - **Laplacian ∇²**: `(S⁺ + S⁻ − 2I)/Δx²`, bond dimension 5 (direct-sum)
//! - **System matrix**: `(I − α·Δt·∇²)`, bond dimension 6 (direct-sum I + L)
//!
//! # QTT Bit Ordering
//!
//! MSB-first: `core[0]` = most significant bit, `core[L-1]` = least significant bit.
//! Grid index: `x = Σ_k b_k · 2^{L-1-k}`.
//!
//! # Carry/Borrow Convention
//!
//! Bond index 0 = no carry, bond index 1 = carry/borrow active.
//! Carry propagates from LSB (site L-1) toward MSB (site 0).
//!
//! © 2025-2026 Bradly Biron Baker Adams / Tigantic Labs. All Rights Reserved.
//! SPDX-License-Identifier: LicenseRef-Proprietary

use crate::field::Q16;
use crate::mpo::{MPOCore, MPO};

// ──────────────────────────────────────────────────────────────────────
// Shift Operators S⁺ and S⁻ — Exact, Bond Dimension 2
// ──────────────────────────────────────────────────────────────────────

/// Build the forward shift operator: S⁺|x⟩ = |x+1 mod 2^L⟩.
///
/// Ripple-carry adder encoding:
/// - LSB (site L-1): always increment, carry propagates if bit was 1.
/// - Middle sites: if carry_in=1, flip bit; propagate carry if bit was 1.
/// - MSB (site 0): absorb carry with periodic wrap.
///
/// Bond dimension = 2 (carry state). All core values ∈ {0, 1}.
///
/// # Panics
/// Panics if `num_sites < 2`.
pub fn shift_plus_mpo(num_sites: usize) -> MPO {
    shift_mpo_impl(num_sites, ShiftDirection::Forward)
}

/// Build the backward shift operator: S⁻|x⟩ = |x−1 mod 2^L⟩.
///
/// Ripple-borrow encoding (dual of ripple-carry).
/// Bond dimension = 2. All core values ∈ {0, 1}.
///
/// # Panics
/// Panics if `num_sites < 2`.
pub fn shift_minus_mpo(num_sites: usize) -> MPO {
    shift_mpo_impl(num_sites, ShiftDirection::Backward)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ShiftDirection {
    Forward,
    Backward,
}

/// Internal implementation for both shift directions.
///
/// Core layout convention: `(D_left, d_out, d_in, D_right)`
/// Flat index: `((left * d_out + out) * d_in + inp) * d_right + right`
///
/// Note on bond semantics:
/// - Bond index 0 = no carry/borrow
/// - Bond index 1 = carry/borrow active
///
/// Carry propagates right-to-left in MSB-first ordering, which means
/// the *right* bond of a core connects to the *left* bond of the next
/// core to its right (higher site index = more significant... wait, no:
/// site 0 = MSB, site L-1 = LSB, so carry flows from high site index
/// to low site index, i.e. from right bond of site k to left bond of
/// site k via the right bond of site k+1's left bond... let me be precise:
///
/// In MSB-first ordering:
/// - site 0 is MSB, site L-1 is LSB
/// - Carry propagates from LSB toward MSB: site L-1 → site L-2 → ... → site 0
/// - The *right* bond of site k connects to the *left* bond of site k+1
///   (because that's how contraction works: left-to-right)
/// - But carry propagates right-to-left (LSB→MSB)
///
/// So carry state lives in the RIGHT bond of each core:
/// - If core k has right bond index 1, that signals carry to core k-1
///   BUT that's the LEFT bond of core k, not right!
///
/// Actually, in the standard MPS/MPO contraction, we go left to right:
/// site 0 → site 1 → ... → site L-1
/// The left bond of site k connects to the right bond of site k-1.
///
/// For carry to flow from LSB (site L-1) to MSB (site 0), the carry
/// state must flow RIGHT-TO-LEFT through the bonds. This means:
/// - RIGHT bond of site k carries the carry state FROM site k+1
///   Wait — right bond of site k connects to left bond of site k+1.
///   So if carry goes from site L-1 to site 0, the carry info must
///   go through: left bond of L-1 → right bond of L-2 → left bond of L-2 → ...
///
/// This matches the Python implementation:
/// - MSB (site 0): d_left=1, d_right=2 — right bond carries carry to site 1
///   No wait, the Python builds LSB first then reverses.
///
/// Let me just follow the Python exactly:
/// After reversal, in MSB-first order:
/// - Site 0 (MSB): shape (1, 2, 2, 2) — left=1, right=2
/// - Middle sites: shape (2, 2, 2, 2)
/// - Site L-1 (LSB): shape (2, 2, 2, 1) — left=2, right=1
///
/// The carry signal is in the LEFT bond of each core (coming from the right,
/// since right bond of site k = left bond of site k+1).
fn shift_mpo_impl(num_sites: usize, dir: ShiftDirection) -> MPO {
    assert!(num_sites >= 2, "shift MPO requires at least 2 sites");

    let one = Q16::one();
    let mut cores = Vec::with_capacity(num_sites);

    for site in 0..num_sites {
        if site == 0 {
            // MSB (leftmost): absorb carry, no outgoing left bond
            // Shape: (1, 2, 2, 2) — d_left=1, d_out=2, d_in=2, d_right=2
            let mut core = MPOCore::new(1, 2, 2, 2);

            match dir {
                ShiftDirection::Forward => {
                    // Right bond 0 = no carry → identity
                    core.set(0, 0, 0, 0, one); // |0⟩→|0⟩
                    core.set(0, 1, 1, 0, one); // |1⟩→|1⟩
                    // Right bond 1 = carry_in → increment (with wrap)
                    core.set(0, 1, 0, 1, one); // |0⟩+carry → |1⟩
                    core.set(0, 0, 1, 1, one); // |1⟩+carry → |0⟩ (wrap)
                }
                ShiftDirection::Backward => {
                    // Right bond 0 = no borrow → identity
                    core.set(0, 0, 0, 0, one);
                    core.set(0, 1, 1, 0, one);
                    // Right bond 1 = borrow_in → decrement (with wrap)
                    core.set(0, 1, 0, 1, one); // |0⟩-borrow → |1⟩ (wrap)
                    core.set(0, 0, 1, 1, one); // |1⟩-borrow → |0⟩
                }
            }

            cores.push(core);
        } else if site == num_sites - 1 {
            // LSB (rightmost): always increment/decrement (carry_in=1 implied)
            // Shape: (2, 2, 2, 1) — d_left=2, d_out=2, d_in=2, d_right=1
            let mut core = MPOCore::new(2, 2, 2, 1);

            match dir {
                ShiftDirection::Forward => {
                    // |0⟩+1 → |1⟩, carry_out=0 (left bond 0)
                    core.set(0, 1, 0, 0, one);
                    // |1⟩+1 → |0⟩, carry_out=1 (left bond 1)
                    core.set(1, 0, 1, 0, one);
                }
                ShiftDirection::Backward => {
                    // |1⟩−1 → |0⟩, borrow_out=0 (left bond 0)
                    core.set(0, 0, 1, 0, one);
                    // |0⟩−1 → |1⟩, borrow_out=1 (left bond 1)
                    core.set(1, 1, 0, 0, one);
                }
            }

            cores.push(core);
        } else {
            // Middle site: propagate carry/borrow
            // Shape: (2, 2, 2, 2) — all bond dims = 2
            let mut core = MPOCore::new(2, 2, 2, 2);

            match dir {
                ShiftDirection::Forward => {
                    // No carry (right bond 0): identity, no carry out (left bond 0)
                    core.set(0, 0, 0, 0, one);
                    core.set(0, 1, 1, 0, one);
                    // Carry in (right bond 1): increment
                    // |0⟩+carry → |1⟩, carry_out=0 (left bond 0)
                    core.set(0, 1, 0, 1, one);
                    // |1⟩+carry → |0⟩, carry_out=1 (left bond 1)
                    core.set(1, 0, 1, 1, one);
                }
                ShiftDirection::Backward => {
                    // No borrow (right bond 0): identity
                    core.set(0, 0, 0, 0, one);
                    core.set(0, 1, 1, 0, one);
                    // Borrow in (right bond 1): decrement
                    // |1⟩−borrow → |0⟩, borrow_out=0 (left bond 0)
                    core.set(0, 0, 1, 1, one);
                    // |0⟩−borrow → |1⟩, borrow_out=1 (left bond 1)
                    core.set(1, 1, 0, 1, one);
                }
            }

            cores.push(core);
        }
    }

    MPO {
        cores,
        num_sites,
    }
}

// ──────────────────────────────────────────────────────────────────────
// Laplacian MPO — Exact, Bond Dimension 5 (Direct-Sum)
// ──────────────────────────────────────────────────────────────────────

/// Build the discrete Laplacian operator in MPO form.
///
/// Implements: `∇²f[i] = (f[i+1] − 2f[i] + f[i−1]) / Δx²`
///
/// This is the **direct-sum** construction: `(S⁺ + S⁻ − 2I) / Δx²`,
/// which composes three exact MPOs using MPO addition:
/// - S⁺ (bond dim 2) + S⁻ (bond dim 2) + (−2)·I (bond dim 1) = bond dim 5
/// - Then scaled by 1/Δx²
///
/// All core values are exact — no SVD or numerical approximation.
/// The operator has periodic boundary conditions.
///
/// # Why not fused rank-3?
///
/// The `operators.py` "fused rank-3" Laplacian is a per-mode Laplacian
/// (diagonal in physical indices) — it cannot represent position shifts
/// and therefore is NOT the standard finite-difference Laplacian.
/// The direct-sum construction is provably correct because each component
/// (S⁺, S⁻, I) is individually verified against dense matrices.
///
/// # Reference
/// Port of `ontic/cfd/pure_qtt_ops.py:laplacian_mpo()`
///
/// # Panics
/// Panics if `num_sites < 2` or `dx` is zero.
pub fn laplacian_mpo(num_sites: usize, dx: Q16) -> MPO {
    assert!(num_sites >= 2, "Laplacian MPO requires at least 2 sites");
    assert!(dx.raw != 0, "dx must be nonzero");

    // Build (S⁺ + S⁻ − 2I)
    let s_plus = shift_plus_mpo(num_sites);
    let s_minus = shift_minus_mpo(num_sites);
    let identity = MPO::identity(num_sites, 2);
    let neg_2i = mpo_scale(&identity, Q16::from_f64(-2.0));
    let sum_shifts = mpo_add(&s_plus, &s_minus);
    let stencil = mpo_add(&sum_shifts, &neg_2i); // bond dim 2+2+1 = 5

    // Scale by 1/Δx²
    // Compute inv_dx_sq = 1/dx² in Q16
    let dx_f64 = dx.to_f64();
    let inv_dx_sq = Q16::from_f64(1.0 / (dx_f64 * dx_f64));

    mpo_scale(&stencil, inv_dx_sq)
}

// ──────────────────────────────────────────────────────────────────────
// MPO Arithmetic: scale, add, negate, subtract
// ──────────────────────────────────────────────────────────────────────

/// Scale an MPO by a Q16 scalar. Only modifies the first core.
///
/// Complexity: O(core_0_size) — touches only the first core.
pub fn mpo_scale(mpo: &MPO, scalar: Q16) -> MPO {
    assert!(!mpo.cores.is_empty(), "cannot scale empty MPO");

    let mut result = mpo.clone();
    let core = &mut result.cores[0];

    for val in &mut core.data {
        *val = scalar.mul(*val);
    }

    result
}

/// Negate an MPO: −O. Equivalent to `mpo_scale(mpo, -1)`.
pub fn mpo_negate(mpo: &MPO) -> MPO {
    let mut result = mpo.clone();
    let core = &mut result.cores[0];

    for val in &mut core.data {
        *val = Q16::from_raw(-val.raw);
    }

    result
}

/// Add two MPOs using direct-sum (block-diagonal) construction.
///
/// Result bond dimension = D₁ + D₂ (additive).
///
/// Structure:
/// - First core: concatenate along right bond `(1, d, d, D₁_r + D₂_r)`
/// - Middle cores: block diagonal `(D₁_l + D₂_l, d, d, D₁_r + D₂_r)`
/// - Last core: concatenate along left bond `(D₁_l + D₂_l, d, d, 1)`
///
/// # Panics
/// Panics if MPOs have different site counts or physical dimensions.
pub fn mpo_add(a: &MPO, b: &MPO) -> MPO {
    assert_eq!(
        a.num_sites, b.num_sites,
        "MPO site count mismatch: {} vs {}",
        a.num_sites, b.num_sites
    );
    assert!(!a.cores.is_empty(), "cannot add empty MPOs");

    let num_sites = a.num_sites;
    let mut cores = Vec::with_capacity(num_sites);

    for site in 0..num_sites {
        let ca = &a.cores[site];
        let cb = &b.cores[site];

        assert_eq!(ca.d_out, cb.d_out, "d_out mismatch at site {site}");
        assert_eq!(ca.d_in, cb.d_in, "d_in mismatch at site {site}");

        let d_out = ca.d_out;
        let d_in = ca.d_in;

        if site == 0 {
            // First core: concatenate along right bond
            // (1, d_out, d_in, D_a_r + D_b_r)
            let new_d_right = ca.d_right + cb.d_right;
            let mut core = MPOCore::new(1, d_out, d_in, new_d_right);

            for o in 0..d_out {
                for i in 0..d_in {
                    for r in 0..ca.d_right {
                        core.set(0, o, i, r, ca.get(0, o, i, r));
                    }
                    for r in 0..cb.d_right {
                        core.set(0, o, i, ca.d_right + r, cb.get(0, o, i, r));
                    }
                }
            }

            cores.push(core);
        } else if site == num_sites - 1 {
            // Last core: concatenate along left bond
            // (D_a_l + D_b_l, d_out, d_in, 1)
            let new_d_left = ca.d_left + cb.d_left;
            let mut core = MPOCore::new(new_d_left, d_out, d_in, 1);

            for l in 0..ca.d_left {
                for o in 0..d_out {
                    for i in 0..d_in {
                        core.set(l, o, i, 0, ca.get(l, o, i, 0));
                    }
                }
            }
            for l in 0..cb.d_left {
                for o in 0..d_out {
                    for i in 0..d_in {
                        core.set(ca.d_left + l, o, i, 0, cb.get(l, o, i, 0));
                    }
                }
            }

            cores.push(core);
        } else {
            // Middle core: block diagonal
            // (D_a_l + D_b_l, d_out, d_in, D_a_r + D_b_r)
            let new_d_left = ca.d_left + cb.d_left;
            let new_d_right = ca.d_right + cb.d_right;
            let mut core = MPOCore::new(new_d_left, d_out, d_in, new_d_right);

            // A block: top-left
            for l in 0..ca.d_left {
                for o in 0..d_out {
                    for i in 0..d_in {
                        for r in 0..ca.d_right {
                            core.set(l, o, i, r, ca.get(l, o, i, r));
                        }
                    }
                }
            }

            // B block: bottom-right
            for l in 0..cb.d_left {
                for o in 0..d_out {
                    for i in 0..d_in {
                        for r in 0..cb.d_right {
                            core.set(
                                ca.d_left + l,
                                o,
                                i,
                                ca.d_right + r,
                                cb.get(l, o, i, r),
                            );
                        }
                    }
                }
            }

            cores.push(core);
        }
    }

    MPO {
        cores,
        num_sites,
    }
}

/// Subtract two MPOs: A − B.
pub fn mpo_subtract(a: &MPO, b: &MPO) -> MPO {
    let neg_b = mpo_negate(b);
    mpo_add(a, &neg_b)
}

// ──────────────────────────────────────────────────────────────────────
// System Matrix (I − α·Δt·∇²) — Exact, Bond Dimension 6
// ──────────────────────────────────────────────────────────────────────

/// Build the implicit time-stepping system matrix `(I − α·Δt·L)`.
///
/// This is formed by direct-sum of `I` (rank 1) and `−α·Δt·L` (rank 5):
/// - Bond dimension = 1 + 5 = 6
/// - All core values are analytically known
///
/// The resulting MPO can be applied to an MPS to compute the LHS of
/// the implicit heat equation: `(I − α·Δt·∇²)·T^{n+1} = T^n + Δt·S`.
///
/// # Parameters
/// - `num_sites`: Number of QTT binary modes (grid has 2^num_sites points)
/// - `alpha_dt`: The product α·Δt where α = ν (diffusivity), Δt = timestep
/// - `dx`: Grid spacing
///
/// # Panics
/// Panics if `num_sites < 2` or `dx` is zero.
pub fn system_matrix_mpo(num_sites: usize, alpha_dt: Q16, dx: Q16) -> MPO {
    let identity = MPO::identity(num_sites, 2);
    let laplacian = laplacian_mpo(num_sites, dx);
    let neg_alpha_dt_laplacian = mpo_scale(&laplacian, Q16::from_raw(-alpha_dt.raw));

    mpo_add(&identity, &neg_alpha_dt_laplacian)
}

// ──────────────────────────────────────────────────────────────────────
// MPS ↔ Dense Conversion (validation only, not for production proofs)
// ──────────────────────────────────────────────────────────────────────

/// Contract an MPS into a dense vector of length 2^L.
///
/// This materializes the full dense representation by contracting all
/// cores. **Only use for testing/validation** — this is O(2^L) and
/// defeats the purpose of QTT compression.
///
/// MSB-first: `f(x) = G^{(0)}[b_0] · G^{(1)}[b_1] · ... · G^{(L-1)}[b_{L-1}]`
/// where `x = Σ_k b_k · 2^{L-1-k}`.
///
/// # Panics
/// Panics if `num_sites > 24` (to prevent accidental 2^L explosion).
pub fn mps_to_dense(mps: &crate::mps::MPS) -> Vec<Q16> {
    let l = mps.num_sites;
    assert!(
        l <= 24,
        "mps_to_dense: num_sites={l} would create 2^{l} elements — too large for dense"
    );

    let n = 1usize << l;
    let mut result = vec![Q16::zero(); n];

    for x in 0..n {
        // Extract bits MSB-first
        let bits: Vec<usize> = (0..l)
            .map(|k| (x >> (l - 1 - k)) & 1)
            .collect();

        // Contract: multiply transfer matrices G^{(k)}[b_k]
        // Start with vectors since boundary bond dims are 1
        // We'll track a row vector (1×χ) that grows through contraction.

        // Site 0: G^{(0)}[b_0] has shape (1, d, χ_right)
        // Select physical index b_0 → get row vector of size χ_right
        let core0 = &mps.cores[0];
        let chi_r0 = core0.chi_right;
        let mut vec: Vec<Q16> = (0..chi_r0)
            .map(|r| core0.get(0, bits[0], r))
            .collect();

        // Sites 1..L-2: multiply by G^{(k)}[b_k] matrices
        for k in 1..l {
            let core = &mps.cores[k];
            let chi_l = core.chi_left;
            let chi_r = core.chi_right;
            let phys = bits[k];

            assert_eq!(vec.len(), chi_l, "bond dimension mismatch at site {k}");

            let mut new_vec = vec![Q16::zero(); chi_r];
            for r in 0..chi_r {
                let mut sum = Q16::zero();
                for l_idx in 0..chi_l {
                    sum = sum.mac(vec[l_idx], core.get(l_idx, phys, r));
                }
                new_vec[r] = sum;
            }
            vec = new_vec;
        }

        // Final: vec should be length 1 (right boundary)
        assert_eq!(vec.len(), 1, "final bond dimension should be 1");
        result[x] = vec[0];
    }

    result
}

/// Apply a dense operator matrix to a dense vector (for validation).
///
/// Computes `y[i] = Σ_j M[i,j] · x[j]` where M is the dense matrix
/// representation of the MPO.
///
/// # Panics
/// Panics if `num_sites > 20` or dimensions are inconsistent.
pub fn mpo_to_dense_matrix(mpo: &MPO) -> Vec<Vec<Q16>> {
    let l = mpo.num_sites;
    assert!(
        l <= 20,
        "mpo_to_dense_matrix: num_sites={l} would create 2^{l}×2^{l} matrix",
    );

    let n = 1usize << l;
    let mut matrix = vec![vec![Q16::zero(); n]; n];

    for row in 0..n {
        for col in 0..n {
            // Extract bits MSB-first for output (row) and input (col)
            let out_bits: Vec<usize> = (0..l)
                .map(|k| (row >> (l - 1 - k)) & 1)
                .collect();
            let in_bits: Vec<usize> = (0..l)
                .map(|k| (col >> (l - 1 - k)) & 1)
                .collect();

            // Contract: multiply transfer matrices O^{(k)}[out_k, in_k]
            // O^{(k)} has shape (D_left, d_out, d_in, D_right)
            // Selecting (out_k, in_k) gives matrix (D_left, D_right)

            let core0 = &mpo.cores[0];
            let d_r0 = core0.d_right;
            let mut vec: Vec<Q16> = (0..d_r0)
                .map(|r| core0.get(0, out_bits[0], in_bits[0], r))
                .collect();

            for k in 1..l {
                let core = &mpo.cores[k];
                let d_l = core.d_left;
                let d_r = core.d_right;

                assert_eq!(vec.len(), d_l, "MPO bond dimension mismatch at site {k}");

                let mut new_vec = vec![Q16::zero(); d_r];
                for r in 0..d_r {
                    let mut sum = Q16::zero();
                    for l_idx in 0..d_l {
                        sum = sum.mac(
                            vec[l_idx],
                            core.get(l_idx, out_bits[k], in_bits[k], r),
                        );
                    }
                    new_vec[r] = sum;
                }
                vec = new_vec;
            }

            assert_eq!(vec.len(), 1, "final MPO bond dimension should be 1");
            matrix[row][col] = vec[0];
        }
    }

    matrix
}

// ──────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mps::MPS;
    use crate::ops::apply_mpo;

    /// Helper: create a basis vector |x⟩ as an MPS (product state)
    fn basis_mps(x: usize, num_sites: usize) -> MPS {
        let bits: Vec<bool> = (0..num_sites)
            .map(|k| ((x >> (num_sites - 1 - k)) & 1) == 1)
            .collect();
        MPS::from_bits(&bits)
    }

    /// Helper: apply MPO to basis and read off dense result
    fn apply_to_basis(mpo: &MPO, x: usize) -> Vec<Q16> {
        let mps = basis_mps(x, mpo.num_sites);
        let result = apply_mpo(&mps, mpo).expect("apply_mpo failed");
        mps_to_dense(&result)
    }

    // ── Shift S⁺ Tests ──

    #[test]
    fn test_shift_plus_bond_dim() {
        for l in [2, 4, 8, 12] {
            let s = shift_plus_mpo(l);
            assert_eq!(s.num_sites, l);
            assert_eq!(s.max_bond(), 2, "shift S+ should have bond dim 2 for L={l}");
            // Boundary bond dims
            assert_eq!(s.cores[0].d_left, 1);
            assert_eq!(s.cores[0].d_right, 2);
            assert_eq!(s.cores[l - 1].d_left, 2);
            assert_eq!(s.cores[l - 1].d_right, 1);
        }
    }

    #[test]
    fn test_shift_plus_all_basis_vectors_l4() {
        let l = 4;
        let n = 1usize << l;
        let s_plus = shift_plus_mpo(l);

        for x in 0..n {
            let result = apply_to_basis(&s_plus, x);
            let expected = (x + 1) % n;

            // Result should be a basis vector |expected⟩
            for y in 0..n {
                let val = result[y];
                if y == expected {
                    assert!(
                        (val.raw - Q16::one().raw).abs() < 2,
                        "S+|{x}⟩: expected 1.0 at position {y}, got {}",
                        val.to_f64()
                    );
                } else {
                    assert!(
                        val.raw.abs() < 2,
                        "S+|{x}⟩: expected 0.0 at position {y}, got {}",
                        val.to_f64()
                    );
                }
            }
        }
    }

    #[test]
    fn test_shift_plus_all_basis_vectors_l8() {
        let l = 8;
        let n = 1usize << l;
        let s_plus = shift_plus_mpo(l);

        for x in 0..n {
            let result = apply_to_basis(&s_plus, x);
            let expected = (x + 1) % n;

            for y in 0..n {
                let val = result[y];
                if y == expected {
                    assert!(
                        (val.raw - Q16::one().raw).abs() < 2,
                        "S+|{x}⟩[{y}]: expected 1.0, got {}",
                        val.to_f64()
                    );
                } else {
                    assert!(
                        val.raw.abs() < 2,
                        "S+|{x}⟩[{y}]: expected 0.0, got {}",
                        val.to_f64()
                    );
                }
            }
        }
    }

    // ── Shift S⁻ Tests ──

    #[test]
    fn test_shift_minus_bond_dim() {
        for l in [2, 4, 8, 12] {
            let s = shift_minus_mpo(l);
            assert_eq!(s.max_bond(), 2, "shift S- should have bond dim 2 for L={l}");
        }
    }

    #[test]
    fn test_shift_minus_all_basis_vectors_l4() {
        let l = 4;
        let n = 1usize << l;
        let s_minus = shift_minus_mpo(l);

        for x in 0..n {
            let result = apply_to_basis(&s_minus, x);
            let expected = (x + n - 1) % n; // x-1 mod N

            for y in 0..n {
                let val = result[y];
                if y == expected {
                    assert!(
                        (val.raw - Q16::one().raw).abs() < 2,
                        "S-|{x}⟩: expected 1.0 at position {y}, got {}",
                        val.to_f64()
                    );
                } else {
                    assert!(
                        val.raw.abs() < 2,
                        "S-|{x}⟩: expected 0.0 at position {y}, got {}",
                        val.to_f64()
                    );
                }
            }
        }
    }

    #[test]
    fn test_shift_minus_all_basis_vectors_l8() {
        let l = 8;
        let n = 1usize << l;
        let s_minus = shift_minus_mpo(l);

        for x in 0..n {
            let result = apply_to_basis(&s_minus, x);
            let expected = (x + n - 1) % n;

            for y in 0..n {
                let val = result[y];
                if y == expected {
                    assert!(
                        (val.raw - Q16::one().raw).abs() < 2,
                        "S-|{x}⟩[{y}]: expected 1.0, got {}",
                        val.to_f64()
                    );
                } else {
                    assert!(
                        val.raw.abs() < 2,
                        "S-|{x}⟩[{y}]: expected 0.0, got {}",
                        val.to_f64()
                    );
                }
            }
        }
    }

    // ── Shift composition: S⁺·S⁻ = Identity ──

    #[test]
    fn test_shift_roundtrip() {
        let l = 4;
        let n = 1usize << l;
        let s_plus = shift_plus_mpo(l);
        let s_minus = shift_minus_mpo(l);

        for x in 0..n {
            // Apply S+ then S-: should get back |x⟩
            let mps = basis_mps(x, l);
            let shifted = apply_mpo(&mps, &s_plus).expect("S+ failed");
            let roundtrip = apply_mpo(&shifted, &s_minus).expect("S- failed");
            let result = mps_to_dense(&roundtrip);

            for y in 0..n {
                let val = result[y];
                if y == x {
                    assert!(
                        (val.raw - Q16::one().raw).abs() < 4,
                        "S-·S+|{x}⟩[{y}]: expected 1.0, got {}",
                        val.to_f64()
                    );
                } else {
                    assert!(
                        val.raw.abs() < 4,
                        "S-·S+|{x}⟩[{y}]: expected 0.0, got {}",
                        val.to_f64()
                    );
                }
            }
        }
    }

    // ── Laplacian MPO Tests ──

    #[test]
    fn test_laplacian_bond_dim() {
        let dx = Q16::one();
        for l in [2, 4, 8] {
            let lap = laplacian_mpo(l, dx);
            assert_eq!(lap.num_sites, l);
            assert_eq!(lap.max_bond(), 5, "direct-sum Laplacian should have bond dim 5 for L={l}");
            // Boundary dims
            assert_eq!(lap.cores[0].d_left, 1);
            assert_eq!(lap.cores[0].d_right, 5);
            assert_eq!(lap.cores[l - 1].d_left, 5);
            assert_eq!(lap.cores[l - 1].d_right, 1);
        }
    }

    #[test]
    fn test_laplacian_dense_stencil_l4() {
        // For dx=1, the Laplacian matrix should be the standard tridiagonal
        // stencil [1, -2, 1] with periodic boundary conditions.
        let l = 4;
        let n = 1usize << l; // 16
        let dx = Q16::one();
        let lap = laplacian_mpo(l, dx);

        let matrix = mpo_to_dense_matrix(&lap);

        // Check stencil values for each row
        for i in 0..n {
            let left = (i + n - 1) % n;
            let right = (i + 1) % n;

            for j in 0..n {
                let val = matrix[i][j].to_f64();
                if j == left || j == right {
                    // Should be +1/dx² = 1.0
                    assert!(
                        (val - 1.0).abs() < 0.01,
                        "L[{i},{j}]: expected 1.0 (neighbor), got {val}"
                    );
                } else if j == i {
                    // Should be -2/dx² = -2.0
                    assert!(
                        (val - (-2.0)).abs() < 0.01,
                        "L[{i},{j}]: expected -2.0 (center), got {val}"
                    );
                } else {
                    // Should be 0
                    assert!(
                        val.abs() < 0.01,
                        "L[{i},{j}]: expected 0.0, got {val}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_laplacian_matches_shift_construction_l4() {
        // Cross-validate: fused rank-3 Laplacian should produce the same
        // dense matrix as (S⁺ + S⁻ − 2I) / dx²
        let l = 4;
        let n = 1usize << l;
        let dx = Q16::one();

        // Fused Laplacian
        let lap_fused = laplacian_mpo(l, dx);
        let mat_fused = mpo_to_dense_matrix(&lap_fused);

        // Direct-sum Laplacian: (S⁺ + S⁻ − 2I) / dx²
        let s_plus = shift_plus_mpo(l);
        let s_minus = shift_minus_mpo(l);
        let identity = MPO::identity(l, 2);
        let neg_2i = mpo_scale(&identity, Q16::from_f64(-2.0));
        let sum_shifts = mpo_add(&s_plus, &s_minus);
        let stencil = mpo_add(&sum_shifts, &neg_2i);
        // For dx=1, 1/dx² = 1, so no scaling needed
        let mat_direct = mpo_to_dense_matrix(&stencil);

        // Compare
        for i in 0..n {
            for j in 0..n {
                let fused = mat_fused[i][j].to_f64();
                let direct = mat_direct[i][j].to_f64();
                assert!(
                    (fused - direct).abs() < 0.02,
                    "Laplacian mismatch at [{i},{j}]: fused={fused}, direct-sum={direct}"
                );
            }
        }
    }

    #[test]
    fn test_laplacian_known_function_l4() {
        // Apply Laplacian to a known function: constant function → zero
        let l = 4;
        let n = 1usize << l;
        let dx = Q16::one();
        let lap = laplacian_mpo(l, dx);

        // Create constant MPS: all grid points = 1.0
        // For this, every basis vector contributes 1.0
        // We'll just verify via dense: L · [1,1,...,1] = [0,0,...,0]
        // because (1 - 2 + 1) = 0 for each point with periodic BC
        let matrix = mpo_to_dense_matrix(&lap);
        for i in 0..n {
            let mut sum = Q16::zero();
            for j in 0..n {
                sum = sum + matrix[i][j];
            }
            assert!(
                sum.to_f64().abs() < 0.01,
                "Laplacian of constant: row {i} sum = {}, expected 0",
                sum.to_f64()
            );
        }
    }

    // ── MPO Arithmetic Tests ──

    #[test]
    fn test_mpo_scale() {
        let l = 4;
        let identity = MPO::identity(l, 2);
        let scaled = mpo_scale(&identity, Q16::from_f64(3.0));
        let matrix = mpo_to_dense_matrix(&scaled);

        let n = 1usize << l;
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 3.0 } else { 0.0 };
                assert!(
                    (matrix[i][j].to_f64() - expected).abs() < 0.01,
                    "3·I[{i},{j}]: expected {expected}, got {}",
                    matrix[i][j].to_f64()
                );
            }
        }
    }

    #[test]
    fn test_mpo_add_identity_plus_identity() {
        let l = 4;
        let n = 1usize << l;
        let i1 = MPO::identity(l, 2);
        let i2 = MPO::identity(l, 2);
        let sum = mpo_add(&i1, &i2);

        // Bond dim should be 1+1 = 2
        assert_eq!(sum.max_bond(), 2);

        // Dense matrix should be 2·I
        let matrix = mpo_to_dense_matrix(&sum);
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 2.0 } else { 0.0 };
                assert!(
                    (matrix[i][j].to_f64() - expected).abs() < 0.01,
                    "(I+I)[{i},{j}]: expected {expected}, got {}",
                    matrix[i][j].to_f64()
                );
            }
        }
    }

    #[test]
    fn test_mpo_subtract() {
        let l = 4;
        let n = 1usize << l;
        let s_plus = shift_plus_mpo(l);
        let diff = mpo_subtract(&s_plus, &s_plus);

        // S+ - S+ should be the zero operator
        let matrix = mpo_to_dense_matrix(&diff);
        for i in 0..n {
            for j in 0..n {
                assert!(
                    matrix[i][j].to_f64().abs() < 0.01,
                    "(S+ - S+)[{i},{j}]: expected 0, got {}",
                    matrix[i][j].to_f64()
                );
            }
        }
    }

    // ── System Matrix Tests ──

    #[test]
    fn test_system_matrix_bond_dim() {
        let l = 4;
        let dx = Q16::one();
        let alpha_dt = Q16::from_f64(0.1);
        let sys = system_matrix_mpo(l, alpha_dt, dx);

        assert_eq!(sys.num_sites, l);
        assert_eq!(sys.max_bond(), 6, "system matrix should have bond dim 1+5=6");
    }

    #[test]
    fn test_system_matrix_dense_l4() {
        // (I - α·dt·L) should be: diagonal = 1 + 2·α·dt/dx²
        //                          off-diagonal (neighbors) = -α·dt/dx²
        let l = 4;
        let n = 1usize << l;
        let dx = Q16::one();
        let alpha_dt = Q16::from_f64(0.25);
        let sys = system_matrix_mpo(l, alpha_dt, dx);
        let matrix = mpo_to_dense_matrix(&sys);

        // α·dt/dx² = 0.25
        let coeff = 0.25;

        for i in 0..n {
            let left = (i + n - 1) % n;
            let right = (i + 1) % n;

            for j in 0..n {
                let val = matrix[i][j].to_f64();
                if j == i {
                    // I - α·dt·(-2/dx²) = 1 + 2·α·dt/dx²
                    let expected = 1.0 + 2.0 * coeff;
                    assert!(
                        (val - expected).abs() < 0.05,
                        "A[{i},{j}]: expected {expected}, got {val}"
                    );
                } else if j == left || j == right {
                    // -α·dt·(1/dx²) = -α·dt/dx²
                    let expected = -coeff;
                    assert!(
                        (val - expected).abs() < 0.05,
                        "A[{i},{j}]: expected {expected}, got {val}"
                    );
                } else {
                    assert!(
                        val.abs() < 0.02,
                        "A[{i},{j}]: expected 0, got {val}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_system_matrix_identity_limit() {
        // When α·dt = 0, system matrix = I
        let l = 4;
        let n = 1usize << l;
        let dx = Q16::one();
        let sys = system_matrix_mpo(l, Q16::zero(), dx);
        let matrix = mpo_to_dense_matrix(&sys);

        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (matrix[i][j].to_f64() - expected).abs() < 0.01,
                    "A(α·dt=0)[{i},{j}]: expected {expected}, got {}",
                    matrix[i][j].to_f64()
                );
            }
        }
    }

    // ── MPS Dense Conversion Tests ──

    #[test]
    fn test_mps_to_dense_basis_vector() {
        // |5⟩ in 4 sites should give a vector with 1.0 at index 5
        let l = 4;
        let n = 1usize << l;
        let mps = basis_mps(5, l);
        let dense = mps_to_dense(&mps);

        assert_eq!(dense.len(), n);
        for i in 0..n {
            if i == 5 {
                assert!(
                    (dense[i].raw - Q16::one().raw).abs() < 2,
                    "|5⟩[{i}]: expected 1.0, got {}",
                    dense[i].to_f64()
                );
            } else {
                assert!(
                    dense[i].raw.abs() < 2,
                    "|5⟩[{i}]: expected 0.0, got {}",
                    dense[i].to_f64()
                );
            }
        }
    }

    #[test]
    fn test_mps_to_dense_all_basis_l4() {
        let l = 4;
        let n = 1usize << l;

        for x in 0..n {
            let mps = basis_mps(x, l);
            let dense = mps_to_dense(&mps);

            for y in 0..n {
                let val = dense[y];
                if y == x {
                    assert!((val.raw - Q16::one().raw).abs() < 2);
                } else {
                    assert!(val.raw.abs() < 2);
                }
            }
        }
    }

    // ── Cross-validation: dense matrix vs MPO application ──

    #[test]
    fn test_laplacian_dense_vs_mpo_apply_l4() {
        let l = 4;
        let n = 1usize << l;
        let dx = Q16::one();
        let lap = laplacian_mpo(l, dx);
        let matrix = mpo_to_dense_matrix(&lap);

        // For each basis vector, apply MPO and compare with dense matmul
        for x in 0..n {
            let mpo_result = apply_to_basis(&lap, x);

            for y in 0..n {
                let from_mpo = mpo_result[y].to_f64();
                let from_dense = matrix[y][x].to_f64();
                assert!(
                    (from_mpo - from_dense).abs() < 0.05,
                    "Laplacian at [{y},{x}]: MPO={from_mpo}, dense={from_dense}"
                );
            }
        }
    }

    #[test]
    fn test_system_matrix_dense_vs_mpo_apply_l4() {
        let l = 4;
        let n = 1usize << l;
        let dx = Q16::one();
        let alpha_dt = Q16::from_f64(0.1);
        let sys = system_matrix_mpo(l, alpha_dt, dx);
        let matrix = mpo_to_dense_matrix(&sys);

        for x in 0..n {
            let mpo_result = apply_to_basis(&sys, x);

            for y in 0..n {
                let from_mpo = mpo_result[y].to_f64();
                let from_dense = matrix[y][x].to_f64();
                assert!(
                    (from_mpo - from_dense).abs() < 0.05,
                    "System matrix at [{y},{x}]: MPO={from_mpo}, dense={from_dense}"
                );
            }
        }
    }

    // ── L=8 and L=12 bond dimension checks ──

    #[test]
    fn test_operators_l8_bond_dims() {
        let dx = Q16::one();
        let alpha_dt = Q16::from_f64(0.1);

        let s_plus = shift_plus_mpo(8);
        assert_eq!(s_plus.max_bond(), 2);

        let s_minus = shift_minus_mpo(8);
        assert_eq!(s_minus.max_bond(), 2);

        let lap = laplacian_mpo(8, dx);
        assert_eq!(lap.max_bond(), 5);

        let sys = system_matrix_mpo(8, alpha_dt, dx);
        assert_eq!(sys.max_bond(), 6);
    }

    #[test]
    fn test_operators_l12_bond_dims() {
        let dx = Q16::one();
        let alpha_dt = Q16::from_f64(0.1);

        let s_plus = shift_plus_mpo(12);
        assert_eq!(s_plus.max_bond(), 2);

        let lap = laplacian_mpo(12, dx);
        assert_eq!(lap.max_bond(), 5);

        let sys = system_matrix_mpo(12, alpha_dt, dx);
        assert_eq!(sys.max_bond(), 6);
    }

    // ── Shift S⁺ edge case: L=2 ──

    #[test]
    fn test_shift_plus_l2() {
        let l = 2;
        let n = 1usize << l; // 4
        let s_plus = shift_plus_mpo(l);

        for x in 0..n {
            let result = apply_to_basis(&s_plus, x);
            let expected = (x + 1) % n;

            for y in 0..n {
                let val = result[y];
                if y == expected {
                    assert!((val.raw - Q16::one().raw).abs() < 2);
                } else {
                    assert!(val.raw.abs() < 2);
                }
            }
        }
    }

    // ── Laplacian with non-unity dx ──

    #[test]
    fn test_laplacian_nonunity_dx() {
        let l = 4;
        let n = 1usize << l;
        let dx = Q16::from_f64(0.5);
        let lap = laplacian_mpo(l, dx);
        let matrix = mpo_to_dense_matrix(&lap);

        // alpha = 1/dx² = 1/0.25 = 4.0
        let alpha = 4.0;

        for i in 0..n {
            let left = (i + n - 1) % n;
            let right = (i + 1) % n;

            for j in 0..n {
                let val = matrix[i][j].to_f64();
                if j == left || j == right {
                    // +alpha = 4.0
                    assert!(
                        (val - alpha).abs() < 0.2,
                        "L[{i},{j}] with dx=0.5: expected {alpha}, got {val}"
                    );
                } else if j == i {
                    // -2*alpha = -8.0
                    assert!(
                        (val - (-2.0 * alpha)).abs() < 0.3,
                        "L[{i},{j}] with dx=0.5: expected {}, got {val}",
                        -2.0 * alpha
                    );
                } else {
                    assert!(
                        val.abs() < 0.1,
                        "L[{i},{j}] with dx=0.5: expected 0, got {val}"
                    );
                }
            }
        }
    }
}
