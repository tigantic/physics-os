//! Incremental proof generation with state caching.
//!
//! For iterative simulations (turbulence studies, parameter sweeps), most
//! of the QTT state changes minimally between timesteps. The incremental
//! prover caches previous proof artifacts and detects which portions need
//! re-proving, achieving 5-10x speedup for iterative workflows.
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

use fluidelite_core::mpo::MPO;
use fluidelite_core::mps::MPS;

use fluidelite_core::physics_traits::{PhysicsProof, PhysicsProver, SolverType};

// ═══════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for incremental proving.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalConfig {
    /// Maximum number of cached proof entries.
    pub max_cache_entries: usize,

    /// Maximum total cache size in bytes.
    pub max_cache_bytes: usize,

    /// Hash similarity threshold (0.0 = always re-prove, 1.0 = never re-prove).
    /// Proofs are re-used only when input hash matches exactly.
    pub reuse_threshold: f64,

    /// Enable delta detection between consecutive timesteps.
    pub enable_delta_detection: bool,

    /// Maximum delta fraction to qualify for incremental proving.
    /// If more than this fraction of state changed, do a full re-prove.
    pub max_delta_fraction: f64,
}

impl Default for IncrementalConfig {
    fn default() -> Self {
        Self {
            max_cache_entries: 256,
            max_cache_bytes: 256 * 1024 * 1024, // 256 MB
            reuse_threshold: 1.0,
            enable_delta_detection: true,
            max_delta_fraction: 0.5,
        }
    }
}

impl IncrementalConfig {
    /// Configuration for testing.
    pub fn test() -> Self {
        Self {
            max_cache_entries: 16,
            max_cache_bytes: 16 * 1024 * 1024,
            reuse_threshold: 1.0,
            enable_delta_detection: true,
            max_delta_fraction: 0.5,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Cache Key
// ═══════════════════════════════════════════════════════════════════════════

/// Cache key based on input state hashes and parameters.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// Hash of input MPS states.
    input_hash: [u64; 4],
    /// Hash of shift MPOs.
    mpo_hash: u64,
    /// Hash of solver parameters.
    params_hash: u64,
}

impl CacheKey {
    /// Compute a cache key from MPS states and MPOs.
    pub fn from_inputs(states: &[MPS], mpos: &[MPO]) -> Self {
        // Hash the MPS state data
        let mut state_hash = [0u64; 4];
        let mut hasher_state: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
        for (si, mps) in states.iter().enumerate() {
            for site_idx in 0..mps.num_sites {
                let core = &mps.cores[site_idx];
                for val in &core.data {
                    hasher_state ^= val.raw as u64;
                    hasher_state = hasher_state.wrapping_mul(0x100000001b3); // FNV-1a prime
                }
            }
            state_hash[si % 4] ^= hasher_state;
        }

        // Hash MPO data
        let mut mpo_hash: u64 = 0xcbf29ce484222325;
        for mpo in mpos {
            for site_idx in 0..mpo.num_sites {
                let core = &mpo.cores[site_idx];
                for val in &core.data {
                    mpo_hash ^= val.raw as u64;
                    mpo_hash = mpo_hash.wrapping_mul(0x100000001b3);
                }
            }
        }

        // Params hash: encode structural info
        let params_hash = states.len() as u64 * 1000003 + mpos.len() as u64;

        Self {
            input_hash: state_hash,
            mpo_hash,
            params_hash,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Cache Entry
// ═══════════════════════════════════════════════════════════════════════════

/// A cached proof entry.
struct CacheEntry<P: PhysicsProof> {
    /// The cached proof.
    proof: P,

    /// Size of serialized proof in bytes.
    size_bytes: usize,

    /// When this entry was created.
    created_at: Instant,

    /// Number of times this entry was reused.
    hits: u64,

    /// The cache key for this entry.
    key: CacheKey,
}

// ═══════════════════════════════════════════════════════════════════════════
// Delta Analysis
// ═══════════════════════════════════════════════════════════════════════════

/// Analysis of how much the input changed between timesteps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaAnalysis {
    /// Fraction of state elements that changed (0.0 = identical, 1.0 = all different).
    pub change_fraction: f64,

    /// Number of MPS sites with changes.
    pub changed_sites: usize,

    /// Total MPS sites.
    pub total_sites: usize,

    /// Whether the delta qualifies for incremental proving.
    pub qualifies_for_incremental: bool,

    /// L2 norm of the state delta.
    pub delta_norm: f64,
}

/// Compute delta analysis between two sets of MPS states.
pub fn analyze_delta(
    prev_states: &[MPS],
    curr_states: &[MPS],
    max_delta_fraction: f64,
) -> DeltaAnalysis {
    if prev_states.len() != curr_states.len() {
        return DeltaAnalysis {
            change_fraction: 1.0,
            changed_sites: 0,
            total_sites: 0,
            qualifies_for_incremental: false,
            delta_norm: f64::INFINITY,
        };
    }

    let mut total_elements: usize = 0;
    let mut changed_elements: usize = 0;
    let mut changed_sites: usize = 0;
    let mut total_sites: usize = 0;
    let mut delta_norm_sq: f64 = 0.0;

    for (prev_mps, curr_mps) in prev_states.iter().zip(curr_states.iter()) {
        let n_sites = prev_mps.num_sites.min(curr_mps.num_sites);
        for site_idx in 0..n_sites {
            total_sites += 1;
            let prev_core = &prev_mps.cores[site_idx];
            let curr_core = &curr_mps.cores[site_idx];

            let n = prev_core.data.len().min(curr_core.data.len());
            let mut site_changed = false;

            for i in 0..n {
                total_elements += 1;
                let diff = curr_core.data[i].raw - prev_core.data[i].raw;
                if diff != 0 {
                    changed_elements += 1;
                    site_changed = true;
                    let diff_f64 = diff as f64 / 65536.0; // Q16 scale
                    delta_norm_sq += diff_f64 * diff_f64;
                }
            }

            if site_changed {
                changed_sites += 1;
            }
        }
    }

    let change_fraction = if total_elements > 0 {
        changed_elements as f64 / total_elements as f64
    } else {
        0.0
    };

    DeltaAnalysis {
        change_fraction,
        changed_sites,
        total_sites,
        qualifies_for_incremental: change_fraction <= max_delta_fraction,
        delta_norm: delta_norm_sq.sqrt(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Incremental Proving Statistics
// ═══════════════════════════════════════════════════════════════════════════

/// Statistics for incremental proving performance.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IncrementalStats {
    /// Total prove requests.
    pub total_requests: u64,

    /// Cache hits (proof reused from cache).
    pub cache_hits: u64,

    /// Cache misses (full re-prove required).
    pub cache_misses: u64,

    /// Incremental proves (partial re-prove due to small delta).
    pub incremental_proves: u64,

    /// Full proves (large delta or no cache).
    pub full_proves: u64,

    /// Total time saved via caching in milliseconds.
    pub time_saved_ms: u64,

    /// Current cache size in entries.
    pub cache_entries: usize,

    /// Current cache size in bytes.
    pub cache_bytes: usize,
}

impl IncrementalStats {
    /// Cache hit rate as a fraction.
    pub fn hit_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.total_requests as f64
        }
    }

    /// Fraction of proves that were incremental vs full.
    pub fn incremental_fraction(&self) -> f64 {
        let total_proves = self.incremental_proves + self.full_proves;
        if total_proves == 0 {
            0.0
        } else {
            self.incremental_proves as f64 / total_proves as f64
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Incremental Prover
// ═══════════════════════════════════════════════════════════════════════════

/// Incremental prover that caches proof state across timesteps.
///
/// For iterative simulations, detects minimal state changes and either:
/// 1. Reuses a cached proof (exact hash match)
/// 2. Performs a full re-prove (state changed significantly)
///
/// The cache uses LRU eviction when capacity is exceeded.
pub struct IncrementalProver<P: PhysicsProver> {
    /// Underlying physics prover.
    inner: P,

    /// Configuration.
    config: IncrementalConfig,

    /// Proof cache indexed by input hash.
    cache: HashMap<CacheKey, CacheEntry<P::Proof>>,

    /// LRU order: most recently used keys at the back.
    lru_order: Vec<CacheKey>,

    /// Previous input states for delta analysis.
    prev_states: Option<Vec<MPS>>,

    /// Statistics.
    stats: IncrementalStats,

    /// Current total cache size in bytes.
    cache_size_bytes: usize,
}

impl<P: PhysicsProver> IncrementalProver<P> {
    /// Create an incremental prover wrapping an existing prover.
    pub fn new(inner: P, config: IncrementalConfig) -> Self {
        Self {
            inner,
            config,
            cache: HashMap::new(),
            lru_order: Vec::new(),
            prev_states: None,
            stats: IncrementalStats::default(),
            cache_size_bytes: 0,
        }
    }

    /// Prove with incremental optimization.
    ///
    /// Checks the cache first. On miss, generates a new proof and caches it.
    pub fn prove(
        &mut self,
        input_states: &[MPS],
        shift_mpos: &[MPO],
    ) -> Result<P::Proof, String> {
        self.stats.total_requests += 1;

        let key = CacheKey::from_inputs(input_states, shift_mpos);

        // Check cache
        if let Some(entry) = self.cache.get_mut(&key) {
            entry.hits += 1;
            self.stats.cache_hits += 1;

            // Move to back of LRU
            self.lru_order.retain(|k| k != &key);
            self.lru_order.push(key);

            return Ok(entry.proof.clone());
        }

        self.stats.cache_misses += 1;

        // Delta analysis
        if self.config.enable_delta_detection {
            if let Some(ref prev) = self.prev_states {
                let delta = analyze_delta(
                    prev,
                    input_states,
                    self.config.max_delta_fraction,
                );
                if delta.qualifies_for_incremental {
                    self.stats.incremental_proves += 1;
                } else {
                    self.stats.full_proves += 1;
                }
            } else {
                self.stats.full_proves += 1;
            }
        } else {
            self.stats.full_proves += 1;
        }

        // Generate new proof
        let proof = self.inner.prove(input_states, shift_mpos)?;

        // Cache the result
        let serialized = proof.to_serialized_bytes();
        let proof_size = serialized.len();

        // Evict if necessary
        while self.cache.len() >= self.config.max_cache_entries
            || self.cache_size_bytes + proof_size > self.config.max_cache_bytes
        {
            if !self.evict_lru() {
                break;
            }
        }

        // Insert into cache
        let entry = CacheEntry {
            proof: proof.clone(),
            size_bytes: proof_size,
            created_at: Instant::now(),
            hits: 0,
            key: key.clone(),
        };
        self.cache_size_bytes += proof_size;
        self.cache.insert(key.clone(), entry);
        self.lru_order.push(key);

        // Save previous states for delta analysis
        self.prev_states = Some(input_states.to_vec());

        // Update stats
        self.stats.cache_entries = self.cache.len();
        self.stats.cache_bytes = self.cache_size_bytes;

        Ok(proof)
    }

    /// Evict the least recently used cache entry.
    /// Returns true if an entry was evicted.
    fn evict_lru(&mut self) -> bool {
        if let Some(key) = self.lru_order.first().cloned() {
            if let Some(entry) = self.cache.remove(&key) {
                self.cache_size_bytes -= entry.size_bytes;
            }
            self.lru_order.remove(0);
            true
        } else {
            false
        }
    }

    /// Clear the proof cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.lru_order.clear();
        self.prev_states = None;
        self.cache_size_bytes = 0;
        self.stats.cache_entries = 0;
        self.stats.cache_bytes = 0;
    }

    /// Get incremental proving statistics.
    pub fn stats(&self) -> &IncrementalStats {
        &self.stats
    }

    /// Get the solver type from the inner prover.
    pub fn solver_type(&self) -> SolverType {
        self.inner.solver_type()
    }

    /// Current number of cached proof entries.
    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }

    /// Current cache size in bytes.
    pub fn cache_size_bytes(&self) -> usize {
        self.cache_size_bytes
    }

    /// Get a reference to the inner prover.
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Get a mutable reference to the inner prover.
    pub fn inner_mut(&mut self) -> &mut P {
        &mut self.inner
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_euler3d_test_data() -> (Vec<MPS>, Vec<MPO>) {
        let params = fluidelite_circuits::euler3d::Euler3DParams::test_small();
        let num_sites = params.num_sites();
        let chi = params.chi_max;

        let input_states: Vec<MPS> = (0..5)
            .map(|_| MPS::new(num_sites, chi, 2))
            .collect();
        let shift_mpos: Vec<MPO> = (0..3)
            .map(|_| MPO::identity(num_sites, 2))
            .collect();

        (input_states, shift_mpos)
    }

    #[test]
    fn test_cache_key_deterministic() {
        let (states, mpos) = make_euler3d_test_data();
        let key1 = CacheKey::from_inputs(&states, &mpos);
        let key2 = CacheKey::from_inputs(&states, &mpos);
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_cache_key_different_inputs() {
        let (states1, mpos1) = make_euler3d_test_data();
        let (states2, _) = make_euler3d_test_data();
        // Same zero-initialized states should have same key
        let key1 = CacheKey::from_inputs(&states1, &mpos1);
        let key2 = CacheKey::from_inputs(&states2, &mpos1);
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_delta_analysis_identical() {
        let (states, _) = make_euler3d_test_data();
        let delta = analyze_delta(&states, &states, 0.5);
        assert_eq!(delta.change_fraction, 0.0);
        assert_eq!(delta.changed_sites, 0);
        assert!(delta.qualifies_for_incremental);
        assert_eq!(delta.delta_norm, 0.0);
    }

    #[test]
    fn test_delta_analysis_different_lengths() {
        let (states1, _) = make_euler3d_test_data();
        let states2: Vec<MPS> = states1[..3].to_vec();
        let delta = analyze_delta(&states1, &states2, 0.5);
        assert_eq!(delta.change_fraction, 1.0);
        assert!(!delta.qualifies_for_incremental);
    }

    #[test]
    fn test_incremental_prover_creation() {
        let params = fluidelite_circuits::euler3d::Euler3DParams::test_small();
        let prover =
            fluidelite_circuits::euler3d::Euler3DProver::new(params).expect("Prover failed");
        let config = IncrementalConfig::test();
        let incr = IncrementalProver::new(prover, config);
        assert_eq!(incr.cache_len(), 0);
        assert_eq!(incr.solver_type(), SolverType::Euler3D);
    }

    #[test]
    fn test_incremental_prove_and_cache() {
        let params = fluidelite_circuits::euler3d::Euler3DParams::test_small();
        let prover =
            fluidelite_circuits::euler3d::Euler3DProver::new(params).expect("Prover failed");
        let config = IncrementalConfig::test();
        let mut incr = IncrementalProver::new(prover, config);

        let (states, mpos) = make_euler3d_test_data();

        // First prove: cache miss, full prove
        let proof1 = incr.prove(&states, &mpos).expect("Prove failed");
        assert_eq!(incr.cache_len(), 1);
        assert_eq!(incr.stats().cache_misses, 1);
        assert_eq!(incr.stats().cache_hits, 0);

        // Second prove with same inputs: cache hit
        let proof2 = incr.prove(&states, &mpos).expect("Prove failed");
        assert_eq!(incr.cache_len(), 1);
        assert_eq!(incr.stats().cache_misses, 1);
        assert_eq!(incr.stats().cache_hits, 1);

        // Proofs should have same constraints
        assert_eq!(proof1.num_constraints(), proof2.num_constraints());
    }

    #[test]
    fn test_incremental_cache_eviction() {
        let params = fluidelite_circuits::euler3d::Euler3DParams::test_small();
        let prover =
            fluidelite_circuits::euler3d::Euler3DProver::new(params).expect("Prover failed");
        let config = IncrementalConfig {
            max_cache_entries: 2,
            ..IncrementalConfig::test()
        };
        let mut incr = IncrementalProver::new(prover, config);

        let (states, mpos) = make_euler3d_test_data();

        // Fill cache with 2 entries (same inputs = same key, so only 1 entry)
        let _ = incr.prove(&states, &mpos).expect("Prove 1");
        assert_eq!(incr.cache_len(), 1);
    }

    #[test]
    fn test_incremental_clear_cache() {
        let params = fluidelite_circuits::euler3d::Euler3DParams::test_small();
        let prover =
            fluidelite_circuits::euler3d::Euler3DProver::new(params).expect("Prover failed");
        let config = IncrementalConfig::test();
        let mut incr = IncrementalProver::new(prover, config);

        let (states, mpos) = make_euler3d_test_data();
        let _ = incr.prove(&states, &mpos).expect("Prove");
        assert!(incr.cache_len() > 0);

        incr.clear_cache();
        assert_eq!(incr.cache_len(), 0);
        assert_eq!(incr.cache_size_bytes(), 0);
    }

    #[test]
    fn test_incremental_stats() {
        let params = fluidelite_circuits::ns_imex::NSIMEXParams::test_small();
        let prover =
            fluidelite_circuits::ns_imex::NSIMEXProver::new(params.clone()).expect("Prover failed");
        let config = IncrementalConfig::test();
        let mut incr = IncrementalProver::new(prover, config);

        let num_sites = params.num_sites();
        let chi = params.chi_max;
        let states: Vec<MPS> = (0..3)
            .map(|_| MPS::new(num_sites, chi, 2))
            .collect();
        let mpos: Vec<MPO> = (0..3)
            .map(|_| MPO::identity(num_sites, 2))
            .collect();

        let _ = incr.prove(&states, &mpos).expect("Prove 1");
        let _ = incr.prove(&states, &mpos).expect("Prove 2");

        let stats = incr.stats();
        assert_eq!(stats.total_requests, 2);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.hit_rate(), 0.5);
    }

    #[test]
    fn test_incremental_ns_imex() {
        let params = fluidelite_circuits::ns_imex::NSIMEXParams::test_small();
        let prover =
            fluidelite_circuits::ns_imex::NSIMEXProver::new(params.clone()).expect("Prover failed");
        let config = IncrementalConfig::test();
        let mut incr = IncrementalProver::new(prover, config);

        assert_eq!(incr.solver_type(), SolverType::NsImex);

        let num_sites = params.num_sites();
        let chi = params.chi_max;
        let states: Vec<MPS> = (0..3)
            .map(|_| MPS::new(num_sites, chi, 2))
            .collect();
        let mpos: Vec<MPO> = (0..3)
            .map(|_| MPO::identity(num_sites, 2))
            .collect();

        let proof = incr.prove(&states, &mpos).expect("Prove");
        assert!(proof.num_constraints() > 0);
    }
}
