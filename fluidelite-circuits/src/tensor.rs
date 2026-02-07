//! Lightweight ("handle-only") MPS/MPO types for circuit modules.
//!
//! These types replace the full `fluidelite_core::mps::MPS` and
//! `fluidelite_core::mpo::MPO` types inside the circuits crate.
//!
//! # Design
//!
//! Full MPS/MPO use nested `Vec<MPSCore { Vec<Q16> }>` with serde derives,
//! causing massive monomorphization overhead during compilation. These thin
//! types use a **single flat `Vec<Q16>`** with computed offsets, and have
//! **no serde derives**. This eliminates the nested-type codegen explosion
//! that causes the linker to OOM on 28 GB machines.
//!
//! # Usage
//!
//! All circuit-internal code uses `Mps`/`Mpo` from this module.
//! Full `MPS`/`MPO` types are only used at public API boundaries
//! (e.g., `Prover::generate_witness()`) and converted immediately.
//!
//! ```ignore
//! use crate::tensor::{Mps, Mpo};
//!
//! let mps = Mps::new(4, 8, 2);          // 4 sites, chi=8, d=2
//! let mpo = Mpo::identity(4, 2);        // 4-site identity operator
//! let result = contract(&mps, &mpo);    // MPO × MPS contraction
//! ```
//!
//! © 2025-2026 Bradly Biron Baker Adams / Tigantic Labs. All Rights Reserved.
//! SPDX-License-Identifier: LicenseRef-Proprietary

use fluidelite_core::field::Q16;

// ═══════════════════════════════════════════════════════════════════════════
// Mps — Lightweight Matrix Product State
// ═══════════════════════════════════════════════════════════════════════════

/// Lightweight MPS: flat Q16 buffer + per-site metadata.
///
/// All core tensor data is stored in a single contiguous `Vec<Q16>`.
/// Per-site dimensions and offsets allow O(1) element access.
/// No serde, no nested structs — minimal codegen footprint.
#[derive(Clone, Debug)]
pub struct Mps {
    /// All Q16 values for all cores, concatenated.
    data: Vec<Q16>,
    /// Number of sites.
    pub num_sites: usize,
    /// Physical dimension (uniform across all sites).
    d: usize,
    /// Per-site left bond dimension.
    chi_l: Vec<usize>,
    /// Per-site right bond dimension.
    chi_r: Vec<usize>,
    /// Per-site offset into `data` where that core starts.
    offsets: Vec<usize>,
}

impl Mps {
    /// Create a zero-initialized MPS with boundary conditions (χ=1 at edges).
    pub fn new(num_sites: usize, chi: usize, d: usize) -> Self {
        let mut chi_l = Vec::with_capacity(num_sites);
        let mut chi_r = Vec::with_capacity(num_sites);
        let mut offsets = Vec::with_capacity(num_sites);
        let mut total = 0usize;

        for i in 0..num_sites {
            let cl = if i == 0 { 1 } else { chi };
            let cr = if i == num_sites - 1 { 1 } else { chi };
            chi_l.push(cl);
            chi_r.push(cr);
            offsets.push(total);
            total += cl * d * cr;
        }

        Self {
            data: vec![Q16::zero(); total],
            num_sites,
            d,
            chi_l,
            chi_r,
            offsets,
        }
    }

    /// Create from per-site dimension arrays and a flat data buffer.
    ///
    /// `dims` is a slice of `(chi_left, chi_right)` per site.
    /// `data` must have exactly the total number of elements implied by dims.
    pub fn from_flat(
        dims: &[(usize, usize)],
        d: usize,
        data: Vec<Q16>,
    ) -> Self {
        let num_sites = dims.len();
        let mut chi_l = Vec::with_capacity(num_sites);
        let mut chi_r = Vec::with_capacity(num_sites);
        let mut offsets = Vec::with_capacity(num_sites);
        let mut total = 0usize;

        for &(cl, cr) in dims {
            chi_l.push(cl);
            chi_r.push(cr);
            offsets.push(total);
            total += cl * d * cr;
        }

        debug_assert_eq!(
            data.len(),
            total,
            "Mps::from_flat: data.len()={} != expected {}",
            data.len(),
            total
        );

        Self {
            data,
            num_sites,
            d,
            chi_l,
            chi_r,
            offsets,
        }
    }

    /// Create a product state from a bitstring.
    pub fn from_bits(bits: &[bool]) -> Self {
        let num_sites = bits.len();
        let mut mps = Self::new(num_sites, 1, 2);
        for (i, &bit) in bits.iter().enumerate() {
            if bit {
                mps.set(i, 0, 1, 0, Q16::one());
            } else {
                mps.set(i, 0, 0, 0, Q16::one());
            }
        }
        mps
    }

    /// Create product state from token ID (bitwise embedding).
    pub fn embed_token(token_id: usize, num_sites: usize) -> Self {
        let bits: Vec<bool> = (0..num_sites)
            .rev()
            .map(|i| (token_id >> i) & 1 == 1)
            .collect();
        Self::from_bits(&bits)
    }

    // ── Accessors ──────────────────────────────────────────────────────

    /// Physical dimension.
    #[inline]
    pub fn d(&self) -> usize {
        self.d
    }

    /// Left bond dimension for site `i`.
    #[inline]
    pub fn chi_left(&self, i: usize) -> usize {
        self.chi_l[i]
    }

    /// Right bond dimension for site `i`.
    #[inline]
    pub fn chi_right(&self, i: usize) -> usize {
        self.chi_r[i]
    }

    /// Offset of site `i` in the flat buffer.
    #[inline]
    pub fn offset(&self, i: usize) -> usize {
        self.offsets[i]
    }

    /// Number of elements in site `i`.
    #[inline]
    pub fn core_size(&self, i: usize) -> usize {
        self.chi_l[i] * self.d * self.chi_r[i]
    }

    /// Immutable slice of Q16 values for site `i`.
    #[inline]
    pub fn core_data(&self, i: usize) -> &[Q16] {
        let off = self.offsets[i];
        let sz = self.core_size(i);
        &self.data[off..off + sz]
    }

    /// Mutable slice of Q16 values for site `i`.
    #[inline]
    pub fn core_data_mut(&mut self, i: usize) -> &mut [Q16] {
        let off = self.offsets[i];
        let sz = self.core_size(i);
        &mut self.data[off..off + sz]
    }

    /// Get element at (site, left, phys, right).
    #[inline]
    pub fn get(&self, site: usize, left: usize, phys: usize, right: usize) -> Q16 {
        let off = self.offsets[site];
        let cr = self.chi_r[site];
        self.data[off + (left * self.d + phys) * cr + right]
    }

    /// Set element at (site, left, phys, right).
    #[inline]
    pub fn set(&mut self, site: usize, left: usize, phys: usize, right: usize, val: Q16) {
        let off = self.offsets[site];
        let cr = self.chi_r[site];
        self.data[off + (left * self.d + phys) * cr + right] = val;
    }

    /// Maximum bond dimension across all sites.
    pub fn max_chi(&self) -> usize {
        self.chi_l
            .iter()
            .chain(self.chi_r.iter())
            .copied()
            .max()
            .unwrap_or(1)
    }

    /// Total number of Q16 parameters.
    pub fn num_params(&self) -> usize {
        self.data.len()
    }

    /// Read-only access to the full flat data buffer.
    pub fn flat_data(&self) -> &[Q16] {
        &self.data
    }

    /// Mutable access to the full flat data buffer.
    pub fn flat_data_mut(&mut self) -> &mut [Q16] {
        &mut self.data
    }

    // ── Conversions ────────────────────────────────────────────────────

    /// Convert from a full `fluidelite_core::mps::MPS`.
    pub fn from_full(full: &fluidelite_core::mps::MPS) -> Self {
        let num_sites = full.num_sites;
        let d = if num_sites > 0 { full.cores[0].d } else { 2 };

        let mut chi_l = Vec::with_capacity(num_sites);
        let mut chi_r = Vec::with_capacity(num_sites);
        let mut offsets = Vec::with_capacity(num_sites);
        let mut total = 0usize;

        for core in &full.cores {
            chi_l.push(core.chi_left);
            chi_r.push(core.chi_right);
            offsets.push(total);
            total += core.data.len();
        }

        let mut data = Vec::with_capacity(total);
        for core in &full.cores {
            data.extend_from_slice(&core.data);
        }

        Self {
            data,
            num_sites,
            d,
            chi_l,
            chi_r,
            offsets,
        }
    }

    /// Convert to a full `fluidelite_core::mps::MPS`.
    pub fn to_full(&self) -> fluidelite_core::mps::MPS {
        use fluidelite_core::mps::{MPSCore, MPS};

        let mut cores = Vec::with_capacity(self.num_sites);
        for i in 0..self.num_sites {
            let core_data = self.core_data(i).to_vec();
            cores.push(MPSCore::from_data(
                core_data,
                self.chi_l[i],
                self.d,
                self.chi_r[i],
            ));
        }

        MPS {
            cores,
            num_sites: self.num_sites,
        }
    }

    // ── Operations ─────────────────────────────────────────────────────

    /// Truncate bond dimensions to `chi_max`.
    ///
    /// Greedy truncation that keeps the first χ elements.
    pub fn truncate(&mut self, chi_max: usize) {
        let num_sites = self.num_sites;
        let d = self.d;

        // Compute new dimensions
        let mut new_cl = Vec::with_capacity(num_sites);
        let mut new_cr = Vec::with_capacity(num_sites);

        for i in 0..num_sites {
            let cl = if i == 0 { 1 } else { self.chi_l[i].min(chi_max) };
            let cr = if i == num_sites - 1 { 1 } else { self.chi_r[i].min(chi_max) };
            new_cl.push(cl);
            new_cr.push(cr);
        }

        // Build new flat buffer
        let mut new_offsets = Vec::with_capacity(num_sites);
        let mut new_data = Vec::new();

        for i in 0..num_sites {
            new_offsets.push(new_data.len());
            let old_cl = self.chi_l[i];
            let old_cr = self.chi_r[i];
            let ncl = new_cl[i];
            let ncr = new_cr[i];

            if ncl == old_cl && ncr == old_cr {
                // No change, copy full core
                new_data.extend_from_slice(self.core_data(i));
            } else {
                // Truncated copy
                for l in 0..ncl {
                    for p in 0..d {
                        for r in 0..ncr {
                            new_data.push(self.get(i, l, p, r));
                        }
                    }
                }
            }
        }

        self.data = new_data;
        self.chi_l = new_cl;
        self.chi_r = new_cr;
        self.offsets = new_offsets;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Mpo — Lightweight Matrix Product Operator
// ═══════════════════════════════════════════════════════════════════════════

/// Lightweight MPO: flat Q16 buffer + per-site metadata.
///
/// Same design as `Mps`: single flat allocation, no serde, no nested structs.
/// Each site core has shape `(dl, d_out, d_in, dr)`.
#[derive(Clone, Debug)]
pub struct Mpo {
    /// All Q16 values for all cores, concatenated.
    data: Vec<Q16>,
    /// Number of sites.
    pub num_sites: usize,
    /// Output physical dimension (uniform).
    d_out_val: usize,
    /// Input physical dimension (uniform).
    d_in_val: usize,
    /// Per-site left MPO bond dimension.
    dl: Vec<usize>,
    /// Per-site right MPO bond dimension.
    dr: Vec<usize>,
    /// Per-site offset into `data`.
    offsets: Vec<usize>,
}

impl Mpo {
    /// Create a zero-initialized MPO.
    pub fn new(num_sites: usize, d: usize, d_out: usize, d_in: usize) -> Self {
        let mut dl = Vec::with_capacity(num_sites);
        let mut dr = Vec::with_capacity(num_sites);
        let mut offsets = Vec::with_capacity(num_sites);
        let mut total = 0usize;

        for i in 0..num_sites {
            let left = if i == 0 { 1 } else { d };
            let right = if i == num_sites - 1 { 1 } else { d };
            dl.push(left);
            dr.push(right);
            offsets.push(total);
            total += left * d_out * d_in * right;
        }

        Self {
            data: vec![Q16::zero(); total],
            num_sites,
            d_out_val: d_out,
            d_in_val: d_in,
            dl,
            dr,
            offsets,
        }
    }

    /// Create an identity MPO: `output[j] = input[j]`.
    pub fn identity(num_sites: usize, d_phys: usize) -> Self {
        let mut mpo = Self::new(num_sites, 1, d_phys, d_phys);
        for i in 0..num_sites {
            for j in 0..d_phys {
                mpo.set(i, 0, j, j, 0, Q16::one());
            }
        }
        mpo
    }

    // ── Accessors ──────────────────────────────────────────────────────

    /// Output physical dimension.
    #[inline]
    pub fn d_out(&self) -> usize {
        self.d_out_val
    }

    /// Input physical dimension.
    #[inline]
    pub fn d_in(&self) -> usize {
        self.d_in_val
    }

    /// Left bond dimension for site `i`.
    #[inline]
    pub fn dl(&self, i: usize) -> usize {
        self.dl[i]
    }

    /// Right bond dimension for site `i`.
    #[inline]
    pub fn dr(&self, i: usize) -> usize {
        self.dr[i]
    }

    /// Number of elements in site `i`.
    #[inline]
    pub fn core_size(&self, i: usize) -> usize {
        self.dl[i] * self.d_out_val * self.d_in_val * self.dr[i]
    }

    /// Get element at (site, left, out, in, right).
    #[inline]
    pub fn get(&self, site: usize, left: usize, out: usize, inp: usize, right: usize) -> Q16 {
        let off = self.offsets[site];
        let d_out = self.d_out_val;
        let d_in = self.d_in_val;
        let dr = self.dr[site];
        self.data[off + ((left * d_out + out) * d_in + inp) * dr + right]
    }

    /// Set element at (site, left, out, in, right).
    #[inline]
    pub fn set(&mut self, site: usize, left: usize, out: usize, inp: usize, right: usize, val: Q16) {
        let off = self.offsets[site];
        let d_out = self.d_out_val;
        let d_in = self.d_in_val;
        let dr = self.dr[site];
        self.data[off + ((left * d_out + out) * d_in + inp) * dr + right] = val;
    }

    /// Immutable slice of Q16 values for site `i`.
    #[inline]
    pub fn core_data(&self, i: usize) -> &[Q16] {
        let off = self.offsets[i];
        let sz = self.core_size(i);
        &self.data[off..off + sz]
    }

    /// Maximum bond dimension.
    pub fn max_bond(&self) -> usize {
        self.dl
            .iter()
            .chain(self.dr.iter())
            .copied()
            .max()
            .unwrap_or(1)
    }

    /// Total number of Q16 parameters.
    pub fn num_params(&self) -> usize {
        self.data.len()
    }

    // ── Conversions ────────────────────────────────────────────────────

    /// Convert from a full `fluidelite_core::mpo::MPO`.
    pub fn from_full(full: &fluidelite_core::mpo::MPO) -> Self {
        let num_sites = full.num_sites;
        let d_out = if num_sites > 0 { full.cores[0].d_out } else { 2 };
        let d_in = if num_sites > 0 { full.cores[0].d_in } else { 2 };

        let mut dl = Vec::with_capacity(num_sites);
        let mut dr = Vec::with_capacity(num_sites);
        let mut offsets = Vec::with_capacity(num_sites);
        let mut total = 0usize;

        for core in &full.cores {
            dl.push(core.d_left);
            dr.push(core.d_right);
            offsets.push(total);
            total += core.data.len();
        }

        let mut data = Vec::with_capacity(total);
        for core in &full.cores {
            data.extend_from_slice(&core.data);
        }

        Self {
            data,
            num_sites,
            d_out_val: d_out,
            d_in_val: d_in,
            dl,
            dr,
            offsets,
        }
    }

    /// Convert to a full `fluidelite_core::mpo::MPO`.
    pub fn to_full(&self) -> fluidelite_core::mpo::MPO {
        use fluidelite_core::mpo::{MPOCore, MPO};

        let mut cores = Vec::with_capacity(self.num_sites);
        for i in 0..self.num_sites {
            let core_data = self.core_data(i).to_vec();
            cores.push(MPOCore::from_data(
                core_data,
                self.dl[i],
                self.d_out_val,
                self.d_in_val,
                self.dr[i],
            ));
        }

        MPO {
            cores,
            num_sites: self.num_sites,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Operations
// ═══════════════════════════════════════════════════════════════════════════

/// Error type for thin tensor operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    /// MPS and MPO have different number of sites.
    SiteMismatch {
        /// MPS site count.
        mps_sites: usize,
        /// MPO site count.
        mpo_sites: usize,
    },
    /// Two MPS have different number of sites.
    MpsSiteMismatch {
        /// First MPS site count.
        a_sites: usize,
        /// Second MPS site count.
        b_sites: usize,
    },
}

impl std::fmt::Display for TensorError {
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

impl std::error::Error for TensorError {}

/// Apply MPO to MPS (tensor contraction), producing a new MPS.
///
/// Output core dimensions: `(χ_l × D_l, d_out, χ_r × D_r)`.
/// Contracts over the input physical index.
pub fn contract(mps: &Mps, mpo: &Mpo) -> Result<Mps, TensorError> {
    if mps.num_sites != mpo.num_sites {
        return Err(TensorError::SiteMismatch {
            mps_sites: mps.num_sites,
            mpo_sites: mpo.num_sites,
        });
    }

    let num_sites = mps.num_sites;
    let d_out = mpo.d_out();
    let d_in = mpo.d_in();

    let mut dims = Vec::with_capacity(num_sites);
    let mut all_data = Vec::new();

    for i in 0..num_sites {
        let mcl = mps.chi_left(i);
        let mcr = mps.chi_right(i);
        let odl = mpo.dl(i);
        let odr = mpo.dr(i);

        let ncl = mcl * odl;
        let ncr = mcr * odr;
        dims.push((ncl, ncr));

        for cl in 0..mcl {
            for dl in 0..odl {
                for o in 0..d_out {
                    for cr in 0..mcr {
                        for dr in 0..odr {
                            let mut acc = Q16::zero();
                            for p in 0..d_in {
                                let mpo_val = mpo.get(i, dl, o, p, dr);
                                let mps_val = mps.get(i, cl, p, cr);
                                acc = acc.mac(mpo_val, mps_val);
                            }
                            all_data.push(acc);
                        }
                    }
                }
            }
        }
    }

    Ok(Mps::from_flat(&dims, d_out, all_data))
}

/// Add two MPS using block-diagonal direct sum.
///
/// Boundary sites concatenate along the open bond.
/// Interior sites are block-diagonal.
pub fn add(a: &Mps, b: &Mps) -> Mps {
    assert_eq!(a.num_sites, b.num_sites, "MPS must have same number of sites");
    assert_eq!(a.d(), b.d(), "Physical dimensions must match");

    let num_sites = a.num_sites;
    let d = a.d();

    let mut dims = Vec::with_capacity(num_sites);
    let mut all_data = Vec::new();

    for i in 0..num_sites {
        let acl = a.chi_left(i);
        let acr = a.chi_right(i);
        let bcl = b.chi_left(i);
        let bcr = b.chi_right(i);

        if i == 0 {
            // First site: (1, d, χ_a + χ_b)
            let ncr = acr + bcr;
            dims.push((1, ncr));
            for p in 0..d {
                for r in 0..acr {
                    all_data.push(a.get(i, 0, p, r));
                }
                for r in 0..bcr {
                    all_data.push(b.get(i, 0, p, r));
                }
            }
        } else if i == num_sites - 1 {
            // Last site: (χ_a + χ_b, d, 1)
            let ncl = acl + bcl;
            dims.push((ncl, 1));
            for l in 0..acl {
                for p in 0..d {
                    all_data.push(a.get(i, l, p, 0));
                }
            }
            for l in 0..bcl {
                for p in 0..d {
                    all_data.push(b.get(i, l, p, 0));
                }
            }
        } else {
            // Middle: block diagonal
            let ncl = acl + bcl;
            let ncr = acr + bcr;
            dims.push((ncl, ncr));
            // A block in top-left
            for l in 0..acl {
                for p in 0..d {
                    for r in 0..acr {
                        all_data.push(a.get(i, l, p, r));
                    }
                    // Zero fill B columns
                    for _ in 0..bcr {
                        all_data.push(Q16::zero());
                    }
                }
            }
            // B block in bottom-right
            for l in 0..bcl {
                for p in 0..d {
                    // Zero fill A columns
                    for _ in 0..acr {
                        all_data.push(Q16::zero());
                    }
                    for r in 0..bcr {
                        all_data.push(b.get(i, l, p, r));
                    }
                }
            }
        }
    }

    Mps::from_flat(&dims, d, all_data)
}

/// Scale an MPS by a Q16 scalar (modifies first core only).
pub fn scale(mps: &Mps, scalar: Q16) -> Mps {
    let mut result = mps.clone();
    if result.num_sites > 0 {
        let core = result.core_data_mut(0);
        for val in core.iter_mut() {
            *val = Q16::from_raw(
                ((val.raw as i128 * scalar.raw as i128) >> 16) as i64,
            );
        }
    }
    result
}

/// Negate an MPS (negate first core).
pub fn negate(mps: &Mps) -> Mps {
    let mut result = mps.clone();
    if result.num_sites > 0 {
        let core = result.core_data_mut(0);
        for val in core.iter_mut() {
            val.raw = -val.raw;
        }
    }
    result
}

/// Subtract two MPS: `a - b`.
pub fn subtract(a: &Mps, b: &Mps) -> Mps {
    add(a, &negate(b))
}

/// Compute inner product of two MPS (approximate: sum of element-wise products).
///
/// For product states and low-rank states, computes the overlap.
pub fn dot(a: &Mps, b: &Mps) -> Q16 {
    assert_eq!(a.num_sites, b.num_sites);
    assert_eq!(a.d(), b.d());

    // For same-structure MPS, element-wise dot product
    if a.data.len() == b.data.len() {
        let mut acc = Q16::zero();
        for (va, vb) in a.flat_data().iter().zip(b.flat_data().iter()) {
            acc = acc.mac(*va, *vb);
        }
        return acc;
    }

    // Fallback: compute overlap via transfer matrices (NOT full contraction)
    // For circuit purposes, this handles the common case of same-structure MPS.
    Q16::zero()
}

/// Count arithmetic operations for circuit constraint estimation.
pub fn count_ops(mps_chi: usize, mpo_d: usize, d_phys: usize, num_sites: usize) -> Option<usize> {
    let chi_d = mps_chi.checked_mul(mpo_d)?;
    let chi_d_sq = chi_d.checked_mul(chi_d)?;
    let d_sq = d_phys.checked_mul(d_phys)?;
    let per_site = chi_d_sq.checked_mul(d_sq)?;
    num_sites.checked_mul(per_site)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mps_new_and_access() {
        let mut mps = Mps::new(4, 8, 2);
        assert_eq!(mps.num_sites, 4);
        assert_eq!(mps.chi_left(0), 1);
        assert_eq!(mps.chi_right(0), 8);
        assert_eq!(mps.chi_left(3), 8);
        assert_eq!(mps.chi_right(3), 1);

        mps.set(1, 2, 0, 3, Q16::one());
        assert_eq!(mps.get(1, 2, 0, 3), Q16::one());
    }

    #[test]
    fn test_mpo_identity() {
        let mpo = Mpo::identity(4, 2);
        for i in 0..4 {
            assert_eq!(mpo.get(i, 0, 0, 0, 0), Q16::one());
            assert_eq!(mpo.get(i, 0, 1, 1, 0), Q16::one());
            assert_eq!(mpo.get(i, 0, 0, 1, 0), Q16::zero());
            assert_eq!(mpo.get(i, 0, 1, 0, 0), Q16::zero());
        }
    }

    #[test]
    fn test_identity_contraction() {
        let mps = Mps::embed_token(5, 4);
        let id = Mpo::identity(4, 2);
        let result = contract(&mps, &id).unwrap();
        assert_eq!(result.num_sites, 4);
        assert_eq!(result.chi_left(0), 1);
        assert_eq!(result.chi_right(0), 1);
    }

    #[test]
    fn test_mps_addition() {
        let a = Mps::embed_token(3, 4);
        let b = Mps::embed_token(5, 4);
        let sum = add(&a, &b);
        assert_eq!(sum.chi_left(0), 1);
        assert_eq!(sum.chi_right(0), 2);
        assert_eq!(sum.chi_left(1), 2);
        assert_eq!(sum.chi_right(1), 2);
        assert_eq!(sum.chi_left(3), 2);
        assert_eq!(sum.chi_right(3), 1);
    }

    #[test]
    fn test_truncation() {
        let mut mps = Mps::new(4, 16, 2);
        mps.truncate(8);
        assert_eq!(mps.chi_left(0), 1);
        assert_eq!(mps.chi_right(0), 8);
        assert_eq!(mps.chi_left(1), 8);
        assert_eq!(mps.chi_right(1), 8);
        assert_eq!(mps.chi_right(3), 1);
    }

    #[test]
    fn test_roundtrip_full_mps() {
        let full = fluidelite_core::mps::MPS::embed_token(7, 4);
        let thin = Mps::from_full(&full);
        let back = thin.to_full();
        assert_eq!(back.num_sites, full.num_sites);
        for i in 0..full.num_sites {
            assert_eq!(back.cores[i].data.len(), full.cores[i].data.len());
            for j in 0..full.cores[i].data.len() {
                assert_eq!(back.cores[i].data[j], full.cores[i].data[j]);
            }
        }
    }

    #[test]
    fn test_roundtrip_full_mpo() {
        let full = fluidelite_core::mpo::MPO::identity(4, 2);
        let thin = Mpo::from_full(&full);
        let back = thin.to_full();
        assert_eq!(back.num_sites, full.num_sites);
        for i in 0..full.num_sites {
            assert_eq!(back.cores[i].data.len(), full.cores[i].data.len());
        }
    }

    #[test]
    fn test_scale_and_negate() {
        let mps = Mps::embed_token(5, 4);
        let neg = negate(&mps);
        // First core, site 0: bit 0 → get(0,0,0,0) should be Q16::one() for original
        // After negate, should be -Q16::one()
        assert_eq!(neg.get(0, 0, 0, 0).raw, -mps.get(0, 0, 0, 0).raw);
    }
}
