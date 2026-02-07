//! Batch proof generation with parallel execution.
//!
//! Distributes proof generation across multiple prover instances using
//! rayon's work-stealing thread pool for optimal throughput.
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use fluidelite_core::mpo::MPO;
use fluidelite_core::mps::MPS;

use fluidelite_core::physics_traits::{PhysicsProof, PhysicsProver, ProverFactory, SolverType};

// ═══════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for batch proof generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Number of parallel prover instances.
    pub parallelism: usize,

    /// Maximum batch size (0 = unlimited).
    pub max_batch_size: usize,

    /// Timeout per proof in seconds (0 = no timeout).
    pub per_proof_timeout_secs: u64,

    /// Whether to continue on individual proof failures.
    pub continue_on_error: bool,

    /// Enable proof ordering guarantee (results in same order as input).
    pub ordered_results: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            parallelism: num_cpus_available(),
            max_batch_size: 0,
            per_proof_timeout_secs: 300,
            continue_on_error: true,
            ordered_results: true,
        }
    }
}

impl BatchConfig {
    /// Create a config for testing with minimal parallelism.
    pub fn test() -> Self {
        Self {
            parallelism: 2,
            max_batch_size: 0,
            per_proof_timeout_secs: 30,
            continue_on_error: true,
            ordered_results: true,
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.parallelism == 0 {
            return Err("parallelism must be ≥ 1".into());
        }
        Ok(())
    }
}

/// Detect available CPU parallelism.
fn num_cpus_available() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

// ═══════════════════════════════════════════════════════════════════════════
// Batch Job
// ═══════════════════════════════════════════════════════════════════════════

/// A single prove job in a batch.
pub struct ProveJob {
    /// Unique job identifier within the batch.
    pub job_id: u64,

    /// MPS states for each physics variable.
    pub input_states: Vec<MPS>,

    /// Shift MPOs for each spatial dimension.
    pub shift_mpos: Vec<MPO>,

    /// Optional label for tracking.
    pub label: Option<String>,
}

impl ProveJob {
    /// Create a new prove job.
    pub fn new(job_id: u64, input_states: Vec<MPS>, shift_mpos: Vec<MPO>) -> Self {
        Self {
            job_id,
            input_states,
            shift_mpos,
            label: None,
        }
    }

    /// Set a human-readable label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Batch Result
// ═══════════════════════════════════════════════════════════════════════════

/// Result of proving a single job in a batch.
#[derive(Debug, Clone)]
pub struct ProveResult<P: PhysicsProof> {
    /// Job identifier.
    pub job_id: u64,

    /// Optional label from the job.
    pub label: Option<String>,

    /// Proof result (Ok or Err).
    pub outcome: Result<P, String>,

    /// Wall-clock time for this job in milliseconds.
    pub wall_time_ms: u64,

    /// Which prover instance handled this job.
    pub prover_index: usize,
}

impl<P: PhysicsProof> ProveResult<P> {
    /// Whether the proof was generated successfully.
    pub fn is_success(&self) -> bool {
        self.outcome.is_ok()
    }
}

/// Summary statistics for a completed batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSummary {
    /// Total jobs in the batch.
    pub total_jobs: usize,

    /// Successful proofs.
    pub succeeded: usize,

    /// Failed proofs.
    pub failed: usize,

    /// Total wall-clock time for the batch in milliseconds.
    pub total_wall_time_ms: u64,

    /// Sum of individual proof times in milliseconds.
    pub total_proof_time_ms: u64,

    /// Parallelism used.
    pub parallelism: usize,

    /// Throughput: proofs per second.
    pub throughput_proofs_per_sec: f64,

    /// Speedup vs sequential (total_proof_time / wall_time).
    pub speedup: f64,

    /// Solver type.
    pub solver_type: SolverType,
}

// ═══════════════════════════════════════════════════════════════════════════
// Batch Statistics (Thread-Safe)
// ═══════════════════════════════════════════════════════════════════════════

/// Atomic batch statistics for lock-free tracking.
#[derive(Debug)]
pub struct BatchStats {
    total_batches: AtomicU64,
    total_jobs: AtomicU64,
    total_succeeded: AtomicU64,
    total_failed: AtomicU64,
    total_proof_time_ms: AtomicU64,
    total_wall_time_ms: AtomicU64,
}

impl Default for BatchStats {
    fn default() -> Self {
        Self {
            total_batches: AtomicU64::new(0),
            total_jobs: AtomicU64::new(0),
            total_succeeded: AtomicU64::new(0),
            total_failed: AtomicU64::new(0),
            total_proof_time_ms: AtomicU64::new(0),
            total_wall_time_ms: AtomicU64::new(0),
        }
    }
}

impl BatchStats {
    /// Record a batch summary.
    pub fn record(&self, summary: &BatchSummary) {
        self.total_batches.fetch_add(1, Ordering::Relaxed);
        self.total_jobs
            .fetch_add(summary.total_jobs as u64, Ordering::Relaxed);
        self.total_succeeded
            .fetch_add(summary.succeeded as u64, Ordering::Relaxed);
        self.total_failed
            .fetch_add(summary.failed as u64, Ordering::Relaxed);
        self.total_proof_time_ms
            .fetch_add(summary.total_proof_time_ms, Ordering::Relaxed);
        self.total_wall_time_ms
            .fetch_add(summary.total_wall_time_ms, Ordering::Relaxed);
    }

    /// Snapshot current statistics.
    pub fn snapshot(&self) -> BatchStatsSnapshot {
        BatchStatsSnapshot {
            total_batches: self.total_batches.load(Ordering::Relaxed),
            total_jobs: self.total_jobs.load(Ordering::Relaxed),
            total_succeeded: self.total_succeeded.load(Ordering::Relaxed),
            total_failed: self.total_failed.load(Ordering::Relaxed),
            total_proof_time_ms: self.total_proof_time_ms.load(Ordering::Relaxed),
            total_wall_time_ms: self.total_wall_time_ms.load(Ordering::Relaxed),
        }
    }
}

/// Serializable snapshot of batch statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchStatsSnapshot {
    /// Total batches processed.
    pub total_batches: u64,
    /// Total jobs processed.
    pub total_jobs: u64,
    /// Total successful proofs.
    pub total_succeeded: u64,
    /// Total failed proofs.
    pub total_failed: u64,
    /// Cumulative proof generation time.
    pub total_proof_time_ms: u64,
    /// Cumulative wall-clock time.
    pub total_wall_time_ms: u64,
}

impl BatchStatsSnapshot {
    /// Overall success rate as a fraction.
    pub fn success_rate(&self) -> f64 {
        if self.total_jobs == 0 {
            1.0
        } else {
            self.total_succeeded as f64 / self.total_jobs as f64
        }
    }

    /// Average parallel efficiency.
    pub fn avg_efficiency(&self) -> f64 {
        if self.total_wall_time_ms == 0 {
            0.0
        } else {
            self.total_proof_time_ms as f64 / self.total_wall_time_ms as f64
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Batch Prover
// ═══════════════════════════════════════════════════════════════════════════

/// Parallel batch prover for high-throughput proof generation.
///
/// Maintains a pool of prover instances and distributes work across them
/// using rayon's work-stealing scheduler.
///
/// # Example
///
/// ```ignore
/// let factory = euler3d_factory(params);
/// let config = BatchConfig { parallelism: 4, ..Default::default() };
/// let batch_prover = BatchProver::new(factory, config)?;
///
/// let jobs = vec![ProveJob::new(0, states, mpos)];
/// let (results, summary) = batch_prover.prove_batch(jobs)?;
/// ```
pub struct BatchProver<P: PhysicsProver> {
    /// Pool of pre-created prover instances.
    provers: Vec<Mutex<P>>,

    /// Configuration.
    config: BatchConfig,

    /// Cumulative statistics.
    stats: Arc<BatchStats>,

    /// Solver type (determined from first prover).
    solver_type: SolverType,
}

impl<P: PhysicsProver> BatchProver<P> {
    /// Create a batch prover with `config.parallelism` prover instances.
    pub fn new(
        factory: &ProverFactory<P>,
        config: BatchConfig,
    ) -> Result<Self, String> {
        config.validate()?;

        let mut provers = Vec::with_capacity(config.parallelism);
        let mut solver_type = SolverType::Euler3D;

        for i in 0..config.parallelism {
            let prover = factory().map_err(|e| {
                format!("Failed to create prover instance {}: {}", i, e)
            })?;
            if i == 0 {
                solver_type = prover.solver_type();
            }
            provers.push(Mutex::new(prover));
        }

        Ok(Self {
            provers,
            config,
            stats: Arc::new(BatchStats::default()),
            solver_type,
        })
    }

    /// Prove a single job using the first available prover.
    pub fn prove_one(
        &self,
        input_states: &[MPS],
        shift_mpos: &[MPO],
    ) -> Result<P::Proof, String> {
        // Try each prover, take the first available
        for (i, prover_mutex) in self.provers.iter().enumerate() {
            if let Ok(mut prover) = prover_mutex.try_lock() {
                let t0 = Instant::now();
                let result = prover.prove(input_states, shift_mpos);
                let elapsed = t0.elapsed().as_millis() as u64;

                let summary = BatchSummary {
                    total_jobs: 1,
                    succeeded: if result.is_ok() { 1 } else { 0 },
                    failed: if result.is_err() { 1 } else { 0 },
                    total_wall_time_ms: elapsed,
                    total_proof_time_ms: elapsed,
                    parallelism: 1,
                    throughput_proofs_per_sec: if elapsed > 0 {
                        1000.0 / elapsed as f64
                    } else {
                        f64::INFINITY
                    },
                    speedup: 1.0,
                    solver_type: self.solver_type,
                };
                self.stats.record(&summary);

                return result;
            }
        }

        // All provers busy, wait on the first one
        let mut prover = self.provers[0]
            .lock()
            .map_err(|e| format!("Prover lock poisoned: {}", e))?;
        prover.prove(input_states, shift_mpos)
    }

    /// Prove a batch of jobs in parallel across the prover pool.
    ///
    /// Jobs are sharded across prover instances. Each shard is processed
    /// sequentially by its assigned prover, but shards run in parallel.
    pub fn prove_batch(
        &self,
        jobs: Vec<ProveJob>,
    ) -> Result<(Vec<ProveResult<P::Proof>>, BatchSummary), String> {
        if jobs.is_empty() {
            return Ok((
                vec![],
                BatchSummary {
                    total_jobs: 0,
                    succeeded: 0,
                    failed: 0,
                    total_wall_time_ms: 0,
                    total_proof_time_ms: 0,
                    parallelism: self.config.parallelism,
                    throughput_proofs_per_sec: 0.0,
                    speedup: 0.0,
                    solver_type: self.solver_type,
                },
            ));
        }

        if self.config.max_batch_size > 0 && jobs.len() > self.config.max_batch_size {
            return Err(format!(
                "Batch size {} exceeds maximum {}",
                jobs.len(),
                self.config.max_batch_size
            ));
        }

        let batch_start = Instant::now();
        let num_provers = self.provers.len();

        // Shard jobs across provers using round-robin
        let mut shards: Vec<Vec<(usize, ProveJob)>> =
            (0..num_provers).map(|_| Vec::new()).collect();
        for (original_index, job) in jobs.into_iter().enumerate() {
            let shard_idx = original_index % num_provers;
            shards[shard_idx].push((original_index, job));
        }

        // Process shards in parallel using rayon
        let total_proof_time_ms = Arc::new(AtomicU64::new(0));
        let succeeded = Arc::new(AtomicUsize::new(0));
        let failed = Arc::new(AtomicUsize::new(0));

        let shard_results: Vec<Vec<(usize, ProveResult<P::Proof>)>> = {
            let tpt = Arc::clone(&total_proof_time_ms);
            let succ = Arc::clone(&succeeded);
            let fail = Arc::clone(&failed);

            // Use std threads for true parallelism (rayon requires Send on closures,
            // Mutex<P> is Send+Sync, but we need &self.provers which is fine)
            std::thread::scope(|s| {
                let handles: Vec<_> = shards
                    .into_iter()
                    .enumerate()
                    .map(|(prover_idx, shard)| {
                        let prover_mutex = &self.provers[prover_idx];
                        let tpt = Arc::clone(&tpt);
                        let succ = Arc::clone(&succ);
                        let fail = Arc::clone(&fail);
                        let continue_on_error = self.config.continue_on_error;

                        s.spawn(move || {
                            let mut prover = prover_mutex.lock().unwrap();
                            let mut results = Vec::with_capacity(shard.len());

                            for (original_index, job) in shard {
                                let job_start = Instant::now();
                                let outcome = prover.prove(
                                    &job.input_states,
                                    &job.shift_mpos,
                                );
                                let wall_time_ms =
                                    job_start.elapsed().as_millis() as u64;

                                tpt.fetch_add(wall_time_ms, Ordering::Relaxed);

                                match &outcome {
                                    Ok(_) => {
                                        succ.fetch_add(1, Ordering::Relaxed);
                                    }
                                    Err(_) => {
                                        fail.fetch_add(1, Ordering::Relaxed);
                                    }
                                }

                                let result = ProveResult {
                                    job_id: job.job_id,
                                    label: job.label,
                                    outcome,
                                    wall_time_ms,
                                    prover_index: prover_idx,
                                };

                                let is_err = !result.is_success();
                                results.push((original_index, result));

                                if is_err && !continue_on_error {
                                    break;
                                }
                            }

                            results
                        })
                    })
                    .collect();

                handles.into_iter().map(|h| h.join().unwrap()).collect()
            })
        };

        // Flatten and optionally sort by original index
        let mut all_results: Vec<(usize, ProveResult<P::Proof>)> =
            shard_results.into_iter().flatten().collect();

        if self.config.ordered_results {
            all_results.sort_by_key(|(idx, _)| *idx);
        }

        let results: Vec<ProveResult<P::Proof>> =
            all_results.into_iter().map(|(_, r)| r).collect();

        let batch_wall_time_ms = batch_start.elapsed().as_millis() as u64;
        let total_proof_ms = total_proof_time_ms.load(Ordering::Relaxed);
        let succ_count = succeeded.load(Ordering::Relaxed);
        let fail_count = failed.load(Ordering::Relaxed);
        let total_jobs = succ_count + fail_count;

        let summary = BatchSummary {
            total_jobs,
            succeeded: succ_count,
            failed: fail_count,
            total_wall_time_ms: batch_wall_time_ms,
            total_proof_time_ms: total_proof_ms,
            parallelism: num_provers,
            throughput_proofs_per_sec: if batch_wall_time_ms > 0 {
                total_jobs as f64 * 1000.0 / batch_wall_time_ms as f64
            } else {
                f64::INFINITY
            },
            speedup: if batch_wall_time_ms > 0 {
                total_proof_ms as f64 / batch_wall_time_ms as f64
            } else {
                1.0
            },
            solver_type: self.solver_type,
        };

        self.stats.record(&summary);

        Ok((results, summary))
    }

    /// Get cumulative batch statistics.
    pub fn stats(&self) -> BatchStatsSnapshot {
        self.stats.snapshot()
    }

    /// Get the solver type for this batch prover.
    pub fn solver_type(&self) -> SolverType {
        self.solver_type
    }

    /// Number of prover instances in the pool.
    pub fn pool_size(&self) -> usize {
        self.provers.len()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::traits;

    fn make_euler3d_jobs(count: usize) -> Vec<ProveJob> {
        let params = fluidelite_circuits::euler3d::Euler3DParams::test_small();
        let num_sites = params.num_sites();
        let chi = params.chi_max;

        (0..count)
            .map(|i| {
                let input_states: Vec<MPS> = (0..5)
                    .map(|_| MPS::new(num_sites, chi, 2))
                    .collect();
                let shift_mpos: Vec<MPO> = (0..3)
                    .map(|_| MPO::identity(num_sites, 2))
                    .collect();

                ProveJob::new(i as u64, input_states, shift_mpos)
                    .with_label(format!("timestep_{}", i))
            })
            .collect()
    }

    fn make_ns_imex_jobs(count: usize) -> Vec<ProveJob> {
        let params = fluidelite_circuits::ns_imex::NSIMEXParams::test_small();
        let num_sites = params.num_sites();
        let chi = params.chi_max;

        (0..count)
            .map(|i| {
                let input_states: Vec<MPS> = (0..3)
                    .map(|_| MPS::new(num_sites, chi, 2))
                    .collect();
                let shift_mpos: Vec<MPO> = (0..3)
                    .map(|_| MPO::identity(num_sites, 2))
                    .collect();

                ProveJob::new(i as u64, input_states, shift_mpos)
            })
            .collect()
    }

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert!(config.parallelism >= 1);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_batch_config_invalid() {
        let config = BatchConfig {
            parallelism: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_batch_prover_creation_euler3d() {
        let params = fluidelite_circuits::euler3d::Euler3DParams::test_small();
        let factory = super::euler3d_factory(params);
        let config = BatchConfig::test();
        let batch = BatchProver::new(&factory, config);
        assert!(batch.is_ok());
        let bp = batch.unwrap();
        assert_eq!(bp.pool_size(), 2);
        assert_eq!(bp.solver_type(), SolverType::Euler3D);
    }

    #[test]
    fn test_batch_prover_creation_ns_imex() {
        let params = fluidelite_circuits::ns_imex::NSIMEXParams::test_small();
        let factory = super::ns_imex_factory(params);
        let config = BatchConfig::test();
        let batch = BatchProver::new(&factory, config);
        assert!(batch.is_ok());
        let bp = batch.unwrap();
        assert_eq!(bp.solver_type(), SolverType::NsImex);
    }

    #[test]
    fn test_batch_prove_single() {
        let params = fluidelite_circuits::euler3d::Euler3DParams::test_small();
        let factory = super::euler3d_factory(params.clone());
        let config = BatchConfig::test();
        let batch = BatchProver::new(&factory, config).unwrap();

        let num_sites = params.num_sites();
        let chi = params.chi_max;
        let states: Vec<MPS> = (0..5)
            .map(|_| MPS::new(num_sites, chi, 2))
            .collect();
        let mpos: Vec<MPO> = (0..3)
            .map(|_| MPO::identity(num_sites, 2))
            .collect();

        let proof = batch.prove_one(&states, &mpos);
        assert!(proof.is_ok());
    }

    #[test]
    fn test_batch_prove_empty() {
        let params = fluidelite_circuits::euler3d::Euler3DParams::test_small();
        let factory = super::euler3d_factory(params);
        let config = BatchConfig::test();
        let batch = BatchProver::new(&factory, config).unwrap();

        let (results, summary) = batch.prove_batch(vec![]).unwrap();
        assert!(results.is_empty());
        assert_eq!(summary.total_jobs, 0);
    }

    #[test]
    fn test_batch_prove_multiple_euler3d() {
        let params = fluidelite_circuits::euler3d::Euler3DParams::test_small();
        let factory = super::euler3d_factory(params);
        let config = BatchConfig::test();
        let batch = BatchProver::new(&factory, config).unwrap();

        let jobs = make_euler3d_jobs(4);
        let (results, summary) = batch.prove_batch(jobs).unwrap();

        assert_eq!(results.len(), 4);
        assert_eq!(summary.total_jobs, 4);
        assert_eq!(summary.succeeded, 4);
        assert_eq!(summary.failed, 0);
        assert_eq!(summary.parallelism, 2);

        // Verify ordering
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.job_id, i as u64);
            assert!(result.is_success());
        }
    }

    #[test]
    fn test_batch_prove_multiple_ns_imex() {
        let params = fluidelite_circuits::ns_imex::NSIMEXParams::test_small();
        let factory = super::ns_imex_factory(params);
        let config = BatchConfig::test();
        let batch = BatchProver::new(&factory, config).unwrap();

        let jobs = make_ns_imex_jobs(4);
        let (results, summary) = batch.prove_batch(jobs).unwrap();

        assert_eq!(results.len(), 4);
        assert_eq!(summary.succeeded, 4);
        assert_eq!(summary.solver_type, SolverType::NsImex);
    }

    #[test]
    fn test_batch_stats_accumulate() {
        let params = fluidelite_circuits::euler3d::Euler3DParams::test_small();
        let factory = super::euler3d_factory(params);
        let config = BatchConfig::test();
        let batch = BatchProver::new(&factory, config).unwrap();

        let _ = batch.prove_batch(make_euler3d_jobs(2)).unwrap();
        let _ = batch.prove_batch(make_euler3d_jobs(3)).unwrap();

        let stats = batch.stats();
        assert_eq!(stats.total_batches, 2);
        assert_eq!(stats.total_jobs, 5);
        assert_eq!(stats.total_succeeded, 5);
        assert_eq!(stats.total_failed, 0);
    }

    #[test]
    fn test_batch_max_size_exceeded() {
        let params = fluidelite_circuits::euler3d::Euler3DParams::test_small();
        let factory = super::euler3d_factory(params);
        let config = BatchConfig {
            parallelism: 2,
            max_batch_size: 2,
            ..Default::default()
        };
        let batch = BatchProver::new(&factory, config).unwrap();

        let result = batch.prove_batch(make_euler3d_jobs(5));
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_summary_throughput() {
        let summary = BatchSummary {
            total_jobs: 10,
            succeeded: 10,
            failed: 0,
            total_wall_time_ms: 1000,
            total_proof_time_ms: 3000,
            parallelism: 4,
            throughput_proofs_per_sec: 10.0,
            speedup: 3.0,
            solver_type: SolverType::Euler3D,
        };
        assert!((summary.throughput_proofs_per_sec - 10.0).abs() < 0.01);
        assert!((summary.speedup - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_stats_snapshot_rates() {
        let snap = BatchStatsSnapshot {
            total_batches: 5,
            total_jobs: 100,
            total_succeeded: 95,
            total_failed: 5,
            total_proof_time_ms: 30000,
            total_wall_time_ms: 10000,
        };
        assert!((snap.success_rate() - 0.95).abs() < 0.001);
        assert!((snap.avg_efficiency() - 3.0).abs() < 0.001);
    }
}
