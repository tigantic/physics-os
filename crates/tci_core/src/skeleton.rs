//! TCI Skeleton: Index set management and fiber sampling
//!
//! Manages the left/right pivot indices and skeleton matrices for TCI.

use ndarray::{Array2, Array3};
use std::collections::HashSet;

/// Skeleton state for one TT site
pub struct SiteSkeleton {
    /// Left pivot indices (multi-indices up to this site)
    pub left_pivots: Vec<u64>,
    /// Right pivot indices (multi-indices after this site)
    pub right_pivots: Vec<u64>,
    /// Fiber matrix: shape (|left| * 2, |right|)
    pub fiber_matrix: Option<Array2<f64>>,
}

impl SiteSkeleton {
    pub fn new() -> Self {
        Self {
            left_pivots: Vec::new(),
            right_pivots: Vec::new(),
            fiber_matrix: None,
        }
    }

    /// Initialize with geometric spread of pivots
    pub fn init_geometric(&mut self, site: usize, n_sites: usize, max_pivots: usize) {
        // Left pivots: cover [0, 2^site)
        let n_left = (1u64 << site).min(max_pivots as u64);
        self.left_pivots = (0..n_left).collect();
        
        // Right pivots: cover [0, 2^(n_sites - site - 1))
        let n_right = (1u64 << (n_sites - site - 1)).min(max_pivots as u64);
        self.right_pivots = (0..n_right).collect();
    }

    /// Get number of left pivots
    pub fn n_left(&self) -> usize {
        self.left_pivots.len().max(1)
    }

    /// Get number of right pivots
    pub fn n_right(&self) -> usize {
        self.right_pivots.len().max(1)
    }
}

/// Full TCI skeleton for all sites
pub struct TCISkeleton {
    /// Number of sites (qubits)
    pub n_sites: usize,
    /// Per-site skeletons
    pub sites: Vec<SiteSkeleton>,
    /// Sample cache: full_index -> value
    pub samples: std::collections::HashMap<u64, f64>,
}

impl TCISkeleton {
    /// Create new skeleton
    pub fn new(n_sites: usize, max_rank: usize) -> Self {
        let mut sites = Vec::with_capacity(n_sites);
        for site in 0..n_sites {
            let mut skel = SiteSkeleton::new();
            skel.init_geometric(site, n_sites, max_rank);
            sites.push(skel);
        }
        
        Self {
            n_sites,
            sites,
            samples: std::collections::HashMap::new(),
        }
    }

    /// Get indices to sample (not yet in cache)
    pub fn get_sample_indices(&self, batch_size: usize) -> Vec<u64> {
        let mut indices = HashSet::new();
        let n = 1u64 << self.n_sites;
        
        // Generate fiber indices for each site
        for site in 0..self.n_sites {
            let skel = &self.sites[site];
            
            for &left in &skel.left_pivots {
                for bit in 0..2u64 {
                    for &right in &skel.right_pivots {
                        let idx = compose_index(left, bit, right, site, self.n_sites);
                        if idx < n && !self.samples.contains_key(&idx) {
                            indices.insert(idx);
                        }
                    }
                }
            }
        }
        
        // Limit to batch size
        indices.into_iter().take(batch_size).collect()
    }

    /// Submit sampled values
    pub fn submit_samples(&mut self, indices: &[u64], values: &[f64]) {
        for (&idx, &val) in indices.iter().zip(values.iter()) {
            self.samples.insert(idx, val);
        }
    }

    /// Update pivots based on samples (greedy value-based selection)
    pub fn update_pivots(&mut self, max_rank: usize) {
        for site in 0..self.n_sites {
            self.update_site_pivots(site, max_rank);
        }
    }

    /// Update pivots for a single site
    fn update_site_pivots(&mut self, site: usize, max_rank: usize) {
        let skel = &self.sites[site];
        let n = 1u64 << self.n_sites;
        
        // Collect fiber values
        let mut fiber_vals: Vec<(u64, u64, f64)> = Vec::new();
        
        for &left in &skel.left_pivots {
            for bit in 0..2u64 {
                for &right in &skel.right_pivots {
                    let idx = compose_index(left, bit, right, site, self.n_sites);
                    if let Some(&val) = self.samples.get(&idx) {
                        fiber_vals.push((left, right, val.abs()));
                    }
                }
            }
        }
        
        if fiber_vals.is_empty() {
            return;
        }
        
        // Rank by absolute value
        fiber_vals.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        
        // Select top pivots
        let mut new_left = HashSet::new();
        let mut new_right = HashSet::new();
        
        for (left, right, _) in fiber_vals.iter().take(max_rank * 4) {
            if new_left.len() < max_rank {
                new_left.insert(*left);
            }
            if new_right.len() < max_rank {
                new_right.insert(*right);
            }
        }
        
        // Update (but keep at least 1 pivot)
        if !new_left.is_empty() {
            let mut v: Vec<_> = new_left.into_iter().collect();
            v.sort_unstable();
            self.sites[site].left_pivots = v;
        }
        if !new_right.is_empty() {
            let mut v: Vec<_> = new_right.into_iter().collect();
            v.sort_unstable();
            self.sites[site].right_pivots = v;
        }
    }

    /// Build TT cores from skeleton
    pub fn build_cores(&self) -> Vec<Array3<f64>> {
        let mut cores = Vec::with_capacity(self.n_sites);
        
        for site in 0..self.n_sites {
            let skel = &self.sites[site];
            let n_left = skel.n_left();
            let n_right = skel.n_right();
            
            // Core shape: (n_left, 2, n_right)
            let mut core = Array3::zeros((n_left, 2, n_right));
            
            for (li, &left) in skel.left_pivots.iter().enumerate() {
                for bit in 0..2usize {
                    for (ri, &right) in skel.right_pivots.iter().enumerate() {
                        let idx = compose_index(left, bit as u64, right, site, self.n_sites);
                        if let Some(&val) = self.samples.get(&idx) {
                            core[[li, bit, ri]] = val;
                        }
                    }
                }
            }
            
            // Normalize for stability
            let max_val = core.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
            if max_val > 1e-10 {
                core /= max_val.sqrt();
            }
            
            cores.push(core);
        }
        
        cores
    }
}

/// Compose full index from left, bit, right
fn compose_index(left: u64, bit: u64, right: u64, site: usize, n_sites: usize) -> u64 {
    let left_shift = n_sites - site;
    let bit_shift = n_sites - site - 1;
    (left << left_shift) | (bit << bit_shift) | right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compose_index() {
        // n_sites = 4, site = 1
        // left occupies bits [4, 3), bit at position 2, right at [1, 0]
        let idx = compose_index(0b10, 1, 0b01, 1, 4);
        // Expected: 0b1011 = 11
        assert_eq!(idx, 0b1011);
    }

    #[test]
    fn test_skeleton_init() {
        let skel = TCISkeleton::new(8, 16);
        assert_eq!(skel.n_sites, 8);
        assert_eq!(skel.sites.len(), 8);
        
        // First site: 1 left pivot, up to 16 right pivots
        assert_eq!(skel.sites[0].left_pivots.len(), 1);
        assert!(skel.sites[0].right_pivots.len() <= 16);
    }
}
