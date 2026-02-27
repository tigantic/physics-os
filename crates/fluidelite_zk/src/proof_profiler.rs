//! Proof Generation Latency Profiling
//!
//! Provides flamegraph-annotated breakdown of proof generation phases:
//! keygen, witness generation, commitment, and proof creation.
//!
//! # Usage
//!
//! ```rust
//! use fluidelite_zk::proof_profiler::{ProofProfiler, ProofPhase};
//!
//! let mut profiler = ProofProfiler::new("euler3d_k16");
//!
//! profiler.begin(ProofPhase::KeyGen);
//! // ... keygen ...
//! profiler.end(ProofPhase::KeyGen);
//!
//! profiler.begin(ProofPhase::Witness);
//! // ... witness generation ...
//! profiler.end(ProofPhase::Witness);
//!
//! profiler.begin(ProofPhase::Commit);
//! // ... polynomial commitment ...
//! profiler.end(ProofPhase::Commit);
//!
//! profiler.begin(ProofPhase::Prove);
//! // ... proof generation ...
//! profiler.end(ProofPhase::Prove);
//!
//! let report = profiler.report();
//! println!("{}", report.summary());
//! report.save_json("profile_results/euler3d_k16.json").unwrap();
//! ```

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

/// Phases of proof generation, in execution order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ProofPhase {
    /// Parameter generation (SRS/CRS setup).
    ParamGen,
    /// Circuit keygen (ProvingKey, VerifyingKey).
    KeyGen,
    /// Witness generation (constraint assignment).
    Witness,
    /// Polynomial commitment (MSM for QTT/KZG).
    Commit,
    /// Core proof generation (create_proof call).
    Prove,
    /// Proof verification.
    Verify,
    /// GPU data transfer (host → device).
    GpuUpload,
    /// GPU MSM computation.
    GpuMsm,
    /// GPU NTT computation.
    GpuNtt,
    /// GPU data transfer (device → host).
    GpuDownload,
    /// Merkle tree construction (multi-timestep).
    MerkleBuild,
    /// Certificate serialization and signing.
    CertSign,
    /// Custom user-defined phase.
    Custom(&'static str),
}

impl ProofPhase {
    /// Human-readable label.
    pub fn label(&self) -> &str {
        match self {
            Self::ParamGen => "param_gen",
            Self::KeyGen => "keygen",
            Self::Witness => "witness",
            Self::Commit => "commit",
            Self::Prove => "prove",
            Self::Verify => "verify",
            Self::GpuUpload => "gpu_upload",
            Self::GpuMsm => "gpu_msm",
            Self::GpuNtt => "gpu_ntt",
            Self::GpuDownload => "gpu_download",
            Self::MerkleBuild => "merkle_build",
            Self::CertSign => "cert_sign",
            Self::Custom(name) => name,
        }
    }
}

/// Accumulated timing for a single phase.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PhaseStats {
    /// Total time spent in this phase.
    pub total: Duration,
    /// Number of times this phase was entered.
    pub count: u64,
    /// Minimum single-entry duration.
    pub min: Option<Duration>,
    /// Maximum single-entry duration.
    pub max: Option<Duration>,
}

impl PhaseStats {
    /// Average duration per entry.
    pub fn avg(&self) -> Duration {
        if self.count == 0 {
            Duration::ZERO
        } else {
            self.total / self.count as u32
        }
    }

    /// Total time in milliseconds.
    pub fn total_ms(&self) -> f64 {
        self.total.as_secs_f64() * 1000.0
    }

    /// Percentage of total proof time.
    pub fn pct_of(&self, total: Duration) -> f64 {
        if total.is_zero() {
            0.0
        } else {
            (self.total.as_secs_f64() / total.as_secs_f64()) * 100.0
        }
    }

    fn record(&mut self, duration: Duration) {
        self.total += duration;
        self.count += 1;
        match self.min {
            Some(m) if duration < m => self.min = Some(duration),
            None => self.min = Some(duration),
            _ => {}
        }
        match self.max {
            Some(m) if duration > m => self.max = Some(duration),
            None => self.max = Some(duration),
            _ => {}
        }
    }
}

/// Profiling report containing all phase timings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingReport {
    /// Name/label for this profiling session.
    pub name: String,
    /// Phase → timing statistics.
    pub phases: BTreeMap<String, PhaseStats>,
    /// Wall-clock total from first begin() to last end().
    pub wall_clock: Duration,
    /// Top-3 latency contributors (phase name, percentage).
    pub top_contributors: Vec<(String, f64)>,
    /// Timestamp when profiling started (Unix epoch millis).
    pub started_at_ms: u64,
}

impl ProfilingReport {
    /// Human-readable summary string.
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("=== Proof Profile: {} ===", self.name));
        lines.push(format!(
            "Wall clock: {:.2}ms",
            self.wall_clock.as_secs_f64() * 1000.0
        ));
        lines.push(String::new());

        // Phase breakdown table.
        lines.push(format!(
            "{:<15} {:>10} {:>8} {:>10} {:>10} {:>7}",
            "Phase", "Total(ms)", "Count", "Avg(ms)", "Max(ms)", "% Total"
        ));
        lines.push("-".repeat(65));

        let mut sorted_phases: Vec<_> = self.phases.iter().collect();
        sorted_phases.sort_by(|a, b| b.1.total.cmp(&a.1.total));

        for (name, stats) in &sorted_phases {
            lines.push(format!(
                "{:<15} {:>10.2} {:>8} {:>10.2} {:>10.2} {:>6.1}%",
                name,
                stats.total_ms(),
                stats.count,
                stats.avg().as_secs_f64() * 1000.0,
                stats.max.unwrap_or_default().as_secs_f64() * 1000.0,
                stats.pct_of(self.wall_clock),
            ));
        }

        lines.push(String::new());
        lines.push("Top-3 latency contributors:".to_string());
        for (i, (name, pct)) in self.top_contributors.iter().enumerate() {
            lines.push(format!("  {}. {} ({:.1}%)", i + 1, name, pct));
        }

        lines.join("\n")
    }

    /// Save report as JSON.
    pub fn save_json(&self, path: &str) -> Result<(), String> {
        // Ensure parent directory exists.
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create directory: {}", e))?;
        }

        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("JSON serialization failed: {}", e))?;

        std::fs::write(path, json).map_err(|e| format!("write failed: {}", e))
    }
}

/// Proof generation profiler.
///
/// Tracks timing for each phase of proof generation. Thread-safe for single
/// profiling session (one phase active at a time per profiler instance).
pub struct ProofProfiler {
    /// Session name.
    name: String,
    /// Phase timings (accumulated).
    phases: BTreeMap<String, PhaseStats>,
    /// Currently active phase start time.
    active_phase: Option<(String, Instant)>,
    /// Wall-clock start time.
    started_at: Option<Instant>,
    /// Wall-clock end time.
    ended_at: Option<Instant>,
    /// Epoch millis at creation.
    created_at_ms: u64,
}

impl ProofProfiler {
    /// Create a new profiler for the given session.
    pub fn new(name: &str) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            name: name.to_string(),
            phases: BTreeMap::new(),
            active_phase: None,
            started_at: None,
            ended_at: None,
            created_at_ms: now,
        }
    }

    /// Begin timing a phase.
    ///
    /// If another phase is active, it is automatically ended first.
    pub fn begin(&mut self, phase: ProofPhase) {
        // Auto-end previous phase if still active.
        if self.active_phase.is_some() {
            self.end_active();
        }

        if self.started_at.is_none() {
            self.started_at = Some(Instant::now());
        }

        self.active_phase = Some((phase.label().to_string(), Instant::now()));
    }

    /// End timing the current phase.
    ///
    /// Panics if no phase is active (in debug mode).
    pub fn end(&mut self, phase: ProofPhase) {
        let now = Instant::now();
        self.ended_at = Some(now);

        if let Some((active_name, start)) = self.active_phase.take() {
            debug_assert_eq!(
                active_name,
                phase.label(),
                "end({}) called but {} was active",
                phase.label(),
                active_name
            );
            let duration = now - start;
            self.phases
                .entry(active_name)
                .or_default()
                .record(duration);
        }
    }

    /// Time a closure as a specific phase.
    pub fn time<F, R>(&mut self, phase: ProofPhase, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        self.begin(phase);
        let result = f();
        self.end(phase);
        result
    }

    /// Generate the profiling report.
    pub fn report(&mut self) -> ProfilingReport {
        // Auto-end any active phase.
        self.end_active();

        let wall_clock = match (self.started_at, self.ended_at) {
            (Some(s), Some(e)) => e - s,
            _ => Duration::ZERO,
        };

        // Compute top-3 contributors.
        let mut sorted: Vec<_> = self
            .phases
            .iter()
            .map(|(name, stats)| (name.clone(), stats.pct_of(wall_clock)))
            .collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_contributors: Vec<_> = sorted.into_iter().take(3).collect();

        ProfilingReport {
            name: self.name.clone(),
            phases: self.phases.clone(),
            wall_clock,
            top_contributors,
            started_at_ms: self.created_at_ms,
        }
    }

    fn end_active(&mut self) {
        if let Some((name, start)) = self.active_phase.take() {
            let duration = Instant::now() - start;
            self.ended_at = Some(Instant::now());
            self.phases.entry(name).or_default().record(duration);
        }
    }
}

impl Drop for ProofProfiler {
    fn drop(&mut self) {
        self.end_active();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_profiler_basic_workflow() {
        let mut profiler = ProofProfiler::new("test_basic");

        profiler.begin(ProofPhase::KeyGen);
        thread::sleep(Duration::from_millis(10));
        profiler.end(ProofPhase::KeyGen);

        profiler.begin(ProofPhase::Prove);
        thread::sleep(Duration::from_millis(20));
        profiler.end(ProofPhase::Prove);

        let report = profiler.report();
        assert_eq!(report.name, "test_basic");
        assert!(report.wall_clock >= Duration::from_millis(25));
        assert_eq!(report.phases.len(), 2);

        let keygen = &report.phases["keygen"];
        assert_eq!(keygen.count, 1);
        assert!(keygen.total >= Duration::from_millis(8));

        let prove = &report.phases["prove"];
        assert_eq!(prove.count, 1);
        assert!(prove.total >= Duration::from_millis(15));
    }

    #[test]
    fn test_profiler_time_closure() {
        let mut profiler = ProofProfiler::new("test_closure");

        let result = profiler.time(ProofPhase::Witness, || {
            thread::sleep(Duration::from_millis(5));
            42
        });

        assert_eq!(result, 42);

        let report = profiler.report();
        assert_eq!(report.phases["witness"].count, 1);
    }

    #[test]
    fn test_profiler_repeated_phase() {
        let mut profiler = ProofProfiler::new("test_repeated");

        for _ in 0..5 {
            profiler.begin(ProofPhase::GpuMsm);
            thread::sleep(Duration::from_millis(2));
            profiler.end(ProofPhase::GpuMsm);
        }

        let report = profiler.report();
        let msm = &report.phases["gpu_msm"];
        assert_eq!(msm.count, 5);
        assert!(msm.min.is_some());
        assert!(msm.max.is_some());
        assert!(msm.avg() >= Duration::from_millis(1));
    }

    #[test]
    fn test_profiler_auto_end_on_begin() {
        let mut profiler = ProofProfiler::new("test_auto_end");

        profiler.begin(ProofPhase::KeyGen);
        thread::sleep(Duration::from_millis(5));
        // Begin prove without ending keygen — keygen should auto-end.
        profiler.begin(ProofPhase::Prove);
        thread::sleep(Duration::from_millis(5));
        profiler.end(ProofPhase::Prove);

        let report = profiler.report();
        assert_eq!(report.phases.len(), 2);
        assert!(report.phases.contains_key("keygen"));
        assert!(report.phases.contains_key("prove"));
    }

    #[test]
    fn test_profiler_top_contributors() {
        let mut profiler = ProofProfiler::new("test_top3");

        profiler.begin(ProofPhase::KeyGen);
        thread::sleep(Duration::from_millis(5));
        profiler.end(ProofPhase::KeyGen);

        profiler.begin(ProofPhase::Prove);
        thread::sleep(Duration::from_millis(20));
        profiler.end(ProofPhase::Prove);

        profiler.begin(ProofPhase::Verify);
        thread::sleep(Duration::from_millis(2));
        profiler.end(ProofPhase::Verify);

        let report = profiler.report();
        assert!(!report.top_contributors.is_empty());
        assert_eq!(report.top_contributors[0].0, "prove");
    }

    #[test]
    fn test_report_summary_format() {
        let mut profiler = ProofProfiler::new("test_summary");

        profiler.begin(ProofPhase::KeyGen);
        thread::sleep(Duration::from_millis(5));
        profiler.end(ProofPhase::KeyGen);

        let report = profiler.report();
        let summary = report.summary();

        assert!(summary.contains("Proof Profile: test_summary"));
        assert!(summary.contains("keygen"));
        assert!(summary.contains("Phase"));
        assert!(summary.contains("Total(ms)"));
    }

    #[test]
    fn test_report_save_json() {
        let mut profiler = ProofProfiler::new("test_json");

        profiler.begin(ProofPhase::Prove);
        thread::sleep(Duration::from_millis(1));
        profiler.end(ProofPhase::Prove);

        let report = profiler.report();
        let tmpdir = tempfile::tempdir().unwrap();
        let path = tmpdir.path().join("profile.json");

        report.save_json(path.to_str().unwrap()).unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();
        assert_eq!(parsed["name"], "test_json");
        assert!(parsed["phases"]["prove"]["count"].as_u64().unwrap() > 0);
    }

    #[test]
    fn test_phase_stats_avg() {
        let mut stats = PhaseStats::default();
        stats.record(Duration::from_millis(10));
        stats.record(Duration::from_millis(30));

        assert_eq!(stats.count, 2);
        assert_eq!(stats.avg(), Duration::from_millis(20));
        assert_eq!(stats.min.unwrap(), Duration::from_millis(10));
        assert_eq!(stats.max.unwrap(), Duration::from_millis(30));
    }

    #[test]
    fn test_empty_profiler_report() {
        let mut profiler = ProofProfiler::new("empty");
        let report = profiler.report();
        assert!(report.phases.is_empty());
        assert!(report.top_contributors.is_empty());
        assert_eq!(report.wall_clock, Duration::ZERO);
    }

    #[test]
    fn test_custom_phase() {
        let mut profiler = ProofProfiler::new("custom");

        profiler.begin(ProofPhase::Custom("my_phase"));
        thread::sleep(Duration::from_millis(1));
        profiler.end(ProofPhase::Custom("my_phase"));

        let report = profiler.report();
        assert!(report.phases.contains_key("my_phase"));
    }
}
