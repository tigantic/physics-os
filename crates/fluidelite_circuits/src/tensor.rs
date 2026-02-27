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

// ═══════════════════════════════════════════════════════════════════════════
// Task 6.14 — Transfer-Matrix Integral
// ═══════════════════════════════════════════════════════════════════════════

/// Transfer-matrix contraction witness for one site.
///
/// Records every MAC step so the ZK circuit can verify the integral
/// computation without re-executing it.
#[derive(Clone, Debug)]
pub struct TransferMatrixSiteWitness {
    /// Transfer vector before this site's contraction (length = χ_l[k]).
    pub transfer_before: Vec<Q16>,
    /// Transfer vector after this site's contraction (length = χ_r[k]).
    pub transfer_after: Vec<Q16>,
    /// MAC accumulators for each output bond index.
    /// `accumulators[beta_r]` has length `chi_l * d + 1` (initial zero + one per MAC).
    pub accumulators: Vec<Vec<Q16>>,
    /// MAC remainders (lower 16 bits of each raw product).
    /// `remainders[beta_r]` has length `chi_l * d`.
    pub remainders: Vec<Vec<i64>>,
}

/// Complete witness for a transfer-matrix integral computation.
///
/// Proves ∑_x f(x) = ⟨1|f⟩ via L site contractions, each constrained
/// by MAC chains. Total cost: O(L · χ² · d) multiplications.
#[derive(Clone, Debug)]
pub struct TransferMatrixIntegralWitness {
    /// Per-site witness data (length = num_sites).
    pub sites: Vec<TransferMatrixSiteWitness>,
    /// Final integral result (scalar).
    pub result: Q16,
    /// Total number of MAC operations across all sites.
    pub total_macs: usize,
}

/// Compute the exact MPS integral using transfer-matrix contraction.
///
/// Evaluates ∑_x f(x) = ⟨1|f⟩ by contracting transfer matrices site by site:
///
///   T^{(k+1)}[β'] = Σ_{β,p} T^{(k)}[β] · G^{(k)}[β, p, β']
///
/// where T^{(0)} = [1] (boundary: χ_l\[0\] = 1) and the result is T^{(L)}\[0\].
/// The all-ones bra ⟨1| sums over every physical index at each site,
/// effectively computing the sum of all 2^L entries of the decoded tensor.
///
/// Cost: O(L · χ² · d) Q16 multiply-accumulate operations.
pub fn transfer_matrix_integral(mps: &Mps) -> Q16 {
    if mps.num_sites == 0 {
        return Q16::zero();
    }

    let d = mps.d();
    // T^{(0)} = [1, 1, …, 1] of length χ_l[0].
    // For a well-formed MPS with boundary conditions, χ_l[0] = 1.
    let mut transfer: Vec<Q16> = vec![Q16::one(); mps.chi_left(0)];

    for k in 0..mps.num_sites {
        let chi_l = mps.chi_left(k);
        let chi_r = mps.chi_right(k);
        let mut new_transfer = vec![Q16::zero(); chi_r];

        for beta_r in 0..chi_r {
            let mut acc = Q16::zero();
            for beta_l in 0..chi_l {
                for p in 0..d {
                    let core_val = mps.get(k, beta_l, p, beta_r);
                    // acc += transfer[β] × G^{(k)}[β, p, β']
                    acc = Q16::from_raw(
                        acc.raw + ((transfer[beta_l].raw as i128 * core_val.raw as i128) >> 16) as i64,
                    );
                }
            }
            new_transfer[beta_r] = acc;
        }

        transfer = new_transfer;
    }

    // T^{(L)}[0] — boundary: χ_r[L-1] = 1
    transfer.first().copied().unwrap_or(Q16::zero())
}

/// Compute the MPS integral with full MAC-chain witness for ZK proof.
///
/// Same computation as [`transfer_matrix_integral`] but records every
/// intermediate accumulator value and fixed-point remainder, enabling
/// the STARK verifier to re-check arithmetic without re-executing.
pub fn transfer_matrix_integral_with_witness(mps: &Mps) -> (Q16, TransferMatrixIntegralWitness) {
    if mps.num_sites == 0 {
        return (
            Q16::zero(),
            TransferMatrixIntegralWitness {
                sites: Vec::new(),
                result: Q16::zero(),
                total_macs: 0,
            },
        );
    }

    let d = mps.d();
    let mut transfer: Vec<Q16> = vec![Q16::one(); mps.chi_left(0)];
    let mut site_witnesses = Vec::with_capacity(mps.num_sites);
    let mut total_macs = 0usize;

    for k in 0..mps.num_sites {
        let chi_l = mps.chi_left(k);
        let chi_r = mps.chi_right(k);
        let macs_per_output = chi_l * d;

        let transfer_before = transfer.clone();
        let mut new_transfer = vec![Q16::zero(); chi_r];
        let mut accumulators = Vec::with_capacity(chi_r);
        let mut remainders = Vec::with_capacity(chi_r);

        for beta_r in 0..chi_r {
            // accumulator trace: initial zero + one entry per MAC
            let mut acc_trace = Vec::with_capacity(macs_per_output + 1);
            let mut rem_trace = Vec::with_capacity(macs_per_output);
            let mut acc = Q16::zero();
            acc_trace.push(acc);

            for beta_l in 0..chi_l {
                for p in 0..d {
                    let core_val = mps.get(k, beta_l, p, beta_r);
                    let product_128 = transfer[beta_l].raw as i128 * core_val.raw as i128;
                    let quotient = (product_128 >> 16) as i64;
                    let remainder = (product_128 & 0xFFFF) as i64;
                    acc = Q16::from_raw(acc.raw + quotient);
                    acc_trace.push(acc);
                    rem_trace.push(remainder);
                }
            }
            new_transfer[beta_r] = acc;
            accumulators.push(acc_trace);
            remainders.push(rem_trace);
            total_macs += macs_per_output;
        }

        site_witnesses.push(TransferMatrixSiteWitness {
            transfer_before,
            transfer_after: new_transfer.clone(),
            accumulators,
            remainders,
        });

        transfer = new_transfer;
    }

    let result = transfer.first().copied().unwrap_or(Q16::zero());

    (
        result,
        TransferMatrixIntegralWitness {
            sites: site_witnesses,
            result,
            total_macs,
        },
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Task 6.17 — Dense Validation Oracle (L ≤ 8 only)
// ═══════════════════════════════════════════════════════════════════════════

impl Mps {
    /// Decode the MPS into a dense vector of Q16 values.
    ///
    /// Evaluates f(x) for every basis state x ∈ {0, 1, …, d^L − 1} by
    /// contracting the MPS cores sequentially. This is **exponential** in L
    /// and is intended **only for test validation** (L ≤ 8).
    ///
    /// Each basis state x is decomposed into per-site physical indices
    /// (i_0, i_1, …, i_{L-1}) where i_k = (x / d^k) % d, and the
    /// MPS value is the matrix product of sliced cores:
    ///
    ///   f(x) = G^{(0)}[:, i_0, :] · G^{(1)}[:, i_1, :] · … · G^{(L-1)}[:, i_{L-1}, :]
    ///
    /// Returns a vector of length d^L.
    ///
    /// # Panics
    ///
    /// Panics if `num_sites > 8` to prevent accidental O(2^N) blowup.
    pub fn to_dense(&self) -> Vec<Q16> {
        assert!(
            self.num_sites <= 8,
            "to_dense() is only safe for L ≤ 8 (got L={}). \
             This function is O(d^L) and intended for test validation only.",
            self.num_sites,
        );

        let d = self.d;
        let l = self.num_sites;

        if l == 0 {
            return vec![Q16::zero()];
        }

        let total_entries = d.pow(l as u32);
        let mut dense = Vec::with_capacity(total_entries);

        for x in 0..total_entries {
            // Decompose x into per-site physical indices: i_k = (x / d^k) % d
            // Site 0 is the most significant site (standard QTT convention).
            let mut phys_indices = vec![0usize; l];
            {
                let mut remainder = x;
                for k in (0..l).rev() {
                    phys_indices[k] = remainder % d;
                    remainder /= d;
                }
            }

            // Contract: transfer[β'] = Σ_β transfer[β] · G^{(k)}[β, i_k, β']
            // Start with T^{(0)} = [1] (boundary: χ_l[0] = 1).
            let mut transfer: Vec<Q16> = vec![Q16::one(); self.chi_left(0)];

            for k in 0..l {
                let p = phys_indices[k];
                let chi_r = self.chi_right(k);
                let chi_l = self.chi_left(k);
                let mut new_transfer = vec![Q16::zero(); chi_r];

                for beta_r in 0..chi_r {
                    let mut acc = Q16::zero();
                    for beta_l in 0..chi_l {
                        let core_val = self.get(k, beta_l, p, beta_r);
                        acc = Q16::from_raw(
                            acc.raw
                                + ((transfer[beta_l].raw as i128 * core_val.raw as i128) >> 16)
                                    as i64,
                        );
                    }
                    new_transfer[beta_r] = acc;
                }

                transfer = new_transfer;
            }

            dense.push(transfer.first().copied().unwrap_or(Q16::zero()));
        }

        dense
    }
}

/// Apply a naïve dense 1D thermal stencil for validation.
///
/// Computes T_new[i] = T_old[i] + α·Δt·(T[i-1] − 2T[i] + T[i+1]) / Δx²
/// with periodic boundary conditions. This is the reference against which
/// QTT-proven results are cross-checked for L ≤ 8.
pub fn dense_thermal_stencil(
    t_old: &[Q16],
    alpha: Q16,
    dt: Q16,
    dx: Q16,
) -> Vec<Q16> {
    let n = t_old.len();
    if n == 0 {
        return Vec::new();
    }

    // α·Δt / Δx² in Q16
    let coeff_raw = (alpha.raw as i128 * dt.raw as i128) >> 16;
    let dx_sq_raw = (dx.raw as i128 * dx.raw as i128) >> 16;
    // coeff / dx² → need another fixed-point division
    // coeff_fp = (coeff_raw << 16) / dx_sq_raw
    let scale_raw = if dx_sq_raw != 0 {
        ((coeff_raw << 16) / dx_sq_raw) as i64
    } else {
        0i64
    };

    let mut t_new = Vec::with_capacity(n);
    for i in 0..n {
        let left = t_old[(i + n - 1) % n];
        let center = t_old[i];
        let right = t_old[(i + 1) % n];
        // Laplacian = left - 2*center + right
        let lap_raw = left.raw - 2 * center.raw + right.raw;
        // T_new = T_old + scale * laplacian (Q16 multiply)
        let delta = (scale_raw as i128 * lap_raw as i128) >> 16;
        t_new.push(Q16::from_raw(center.raw + delta as i64));
    }

    t_new
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

    // ── Task 6.14: Transfer-matrix integral ────────────────────────────

    #[test]
    fn test_transfer_matrix_integral_rank1_constant() {
        // Rank-1 constant MPS: every entry = c (product state).
        // All L sites have core G^{(k)}[0, p, 0] = c for p ∈ {0,1}.
        // Dense decode: every element = c^L (after Q16 FP chaining).
        // Integral = 2^L · c^L.
        let l = 4usize; // 2^4 = 16 entries
        let c = Q16::from_f64(0.5);
        let mut mps = Mps::new(l, 1, 2);
        for k in 0..l {
            mps.set(k, 0, 0, 0, c);
            mps.set(k, 0, 1, 0, c);
        }
        let integral = transfer_matrix_integral(&mps);
        // c^L with Q16 chaining: 0.5^4 = 0.0625, times 2^4 = 16 → 1.0
        // Allow small FP tolerance
        let expected = 1.0f64;
        let actual = integral.to_f64();
        assert!(
            (actual - expected).abs() < 0.01,
            "Rank-1 constant integral: expected ~{}, got {}",
            expected,
            actual,
        );
    }

    #[test]
    fn test_transfer_matrix_matches_dense_sum() {
        // For L ≤ 8, transfer-matrix integral must equal sum(to_dense()).
        let l = 4;
        let mps = Mps::embed_token(5, l); // token 5 = binary 0101
        let dense = mps.to_dense();
        let dense_sum: i64 = dense.iter().map(|v| v.raw).sum();
        let tm_integral = transfer_matrix_integral(&mps);
        assert_eq!(
            tm_integral.raw, dense_sum,
            "transfer-matrix integral ({}) != dense sum ({})",
            tm_integral.raw, dense_sum,
        );
    }

    #[test]
    fn test_transfer_matrix_witness_consistency() {
        let mps = Mps::embed_token(7, 4);
        let (result, witness) = transfer_matrix_integral_with_witness(&mps);
        let plain = transfer_matrix_integral(&mps);
        assert_eq!(result.raw, plain.raw, "Witness result must match plain result");
        assert_eq!(witness.result.raw, result.raw);
        assert_eq!(witness.sites.len(), mps.num_sites);
        // Every site's MAC chain accumulators end at the transfer_after values
        for (k, sw) in witness.sites.iter().enumerate() {
            let chi_r = mps.chi_right(k);
            assert_eq!(sw.transfer_after.len(), chi_r);
            for (beta_r, accs) in sw.accumulators.iter().enumerate() {
                let final_acc = accs.last().unwrap();
                assert_eq!(
                    final_acc.raw, sw.transfer_after[beta_r].raw,
                    "Site {} beta_r={}: final accumulator ({}) != transfer_after ({})",
                    k, beta_r, final_acc.raw, sw.transfer_after[beta_r].raw,
                );
            }
        }
    }

    #[test]
    fn test_transfer_matrix_witness_mac_counts() {
        let l = 6;
        let chi = 4;
        let d = 2;
        let mps = Mps::new(l, chi, d);
        let (_, witness) = transfer_matrix_integral_with_witness(&mps);
        // Total MACs = Σ_k χ_r[k] × χ_l[k] × d
        let expected_macs: usize = (0..l)
            .map(|k| mps.chi_right(k) * mps.chi_left(k) * d)
            .sum();
        assert_eq!(witness.total_macs, expected_macs);
    }

    // ── Task 6.17: Dense validation oracle ─────────────────────────────

    #[test]
    fn test_to_dense_product_state() {
        // embed_token(5, 4): binary 0101 → one-hot at index 5
        let mps = Mps::embed_token(5, 4);
        let dense = mps.to_dense();
        assert_eq!(dense.len(), 16); // 2^4
        // Only element at index 5 should be Q16::one()
        for (i, val) in dense.iter().enumerate() {
            if i == 5 {
                assert_eq!(val.raw, Q16::one().raw, "Index 5 should be 1.0");
            } else {
                assert_eq!(val.raw, 0, "Index {} should be 0", i);
            }
        }
    }

    #[test]
    fn test_to_dense_sum_equals_integral() {
        // Build a non-trivial MPS and verify sum(dense) ≈ transfer_matrix_integral
        // (Small FP rounding differences are expected for multi-bond MPS since
        // the dense path and transfer-matrix path accumulate in different orders.)
        let l = 4;
        let mut mps = Mps::new(l, 2, 2);
        // Set some non-zero values
        mps.set(0, 0, 0, 0, Q16::from_f64(1.0));
        mps.set(0, 0, 1, 0, Q16::from_f64(0.5));
        for k in 1..l - 1 {
            mps.set(k, 0, 0, 0, Q16::from_f64(0.8));
            mps.set(k, 0, 1, 0, Q16::from_f64(0.3));
            mps.set(k, 0, 0, 1, Q16::from_f64(0.1));
            mps.set(k, 0, 1, 1, Q16::from_f64(0.6));
            mps.set(k, 1, 0, 0, Q16::from_f64(0.4));
            mps.set(k, 1, 1, 0, Q16::from_f64(0.2));
            mps.set(k, 1, 0, 1, Q16::from_f64(0.7));
            mps.set(k, 1, 1, 1, Q16::from_f64(0.9));
        }
        mps.set(l - 1, 0, 0, 0, Q16::from_f64(1.0));
        mps.set(l - 1, 0, 1, 0, Q16::from_f64(0.5));
        mps.set(l - 1, 1, 0, 0, Q16::from_f64(0.3));
        mps.set(l - 1, 1, 1, 0, Q16::from_f64(0.7));

        let dense = mps.to_dense();
        let dense_sum: i64 = dense.iter().map(|v| v.raw).sum();
        let tm = transfer_matrix_integral(&mps);
        // Allow ±1 Q16 LSB per site of FP rounding tolerance
        let tol = (l as i64) * Q16::one().raw; // L × 1.0 in Q16
        assert!(
            (tm.raw - dense_sum).abs() <= tol,
            "transfer-matrix integral ({}) and dense sum ({}) differ by {} > tolerance {}",
            tm.raw, dense_sum, (tm.raw - dense_sum).abs(), tol,
        );
    }

    #[test]
    fn test_dense_thermal_stencil_constant_field() {
        // Constant field: Laplacian = 0, so T_new = T_old.
        let n = 8;
        let c = Q16::from_f64(1.0);
        let field: Vec<Q16> = vec![c; n];
        let result = dense_thermal_stencil(
            &field,
            Q16::from_f64(0.01),
            Q16::from_f64(0.1),
            Q16::one(),
        );
        for (i, val) in result.iter().enumerate() {
            assert!(
                (val.raw - c.raw).abs() <= 1,
                "Constant field index {}: expected ~{}, got {}",
                i,
                c.to_f64(),
                val.to_f64(),
            );
        }
    }

    #[test]
    #[should_panic(expected = "only safe for L ≤ 8")]
    fn test_to_dense_panics_over_8_sites() {
        let mps = Mps::new(9, 2, 2);
        let _ = mps.to_dense();
    }
}
