//! Benchmark Regression Baseline
//!
//! Captures TPS, latency (p50/p95/p99), and VRAM peak into a
//! `.benchmark_baseline.json` file. CI compares against this baseline
//! to detect ≥10% regressions.
//!
//! # CI Integration
//!
//! ```bash
//! # Generate baseline:
//! cargo run --release --features halo2 --bin benchmark-baseline -- --save
//!
//! # Compare against baseline:
//! cargo run --release --features halo2 --bin benchmark-baseline -- --compare
//! ```

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Default baseline file path relative to workspace root.
pub const DEFAULT_BASELINE_PATH: &str = ".benchmark_baseline.json";

/// Regression threshold: percentage degradation that triggers failure.
pub const REGRESSION_THRESHOLD_PCT: f64 = 10.0;

/// A snapshot of benchmark metrics at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkBaseline {
    /// Schema version for forward compatibility.
    pub schema_version: u32,
    /// Git commit hash at baseline capture.
    pub git_commit: String,
    /// Timestamp (ISO 8601).
    pub timestamp: String,
    /// Rust compiler version.
    pub rustc_version: String,
    /// CUDA toolkit version (if GPU).
    pub cuda_version: Option<String>,
    /// GPU device name (if GPU).
    pub gpu_name: Option<String>,
    /// Per-domain benchmark results.
    pub domains: Vec<DomainBaseline>,
    /// Aggregate metrics.
    pub aggregate: AggregateMetrics,
}

/// Benchmark results for a single physics domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainBaseline {
    /// Domain name (euler3d, ns_imex, thermal).
    pub domain: String,
    /// Circuit k parameter.
    pub k: u32,
    /// Latency statistics (milliseconds).
    pub latency: LatencyStats,
    /// Throughput (proofs per second).
    pub throughput_tps: f64,
    /// Peak VRAM usage (MiB), if measured.
    pub peak_vram_mib: Option<f64>,
    /// Proof size (bytes).
    pub proof_size_bytes: usize,
    /// Verification time (milliseconds).
    pub verify_time_ms: f64,
}

/// Latency percentile statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    /// Median latency (p50) in milliseconds.
    pub p50_ms: f64,
    /// 95th percentile latency in milliseconds.
    pub p95_ms: f64,
    /// 99th percentile latency in milliseconds.
    pub p99_ms: f64,
    /// Mean latency in milliseconds.
    pub mean_ms: f64,
    /// Standard deviation in milliseconds.
    pub stddev_ms: f64,
    /// Number of samples.
    pub sample_count: usize,
}

/// Aggregate metrics across all domains.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateMetrics {
    /// Overall throughput (proofs per second, all domains).
    pub total_tps: f64,
    /// Average latency across domains.
    pub avg_latency_ms: f64,
    /// Peak VRAM across all benchmarks (MiB).
    pub peak_vram_mib: Option<f64>,
}

/// Comparison result between current and baseline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    /// Overall pass/fail.
    pub passed: bool,
    /// Regression threshold used (percentage).
    pub threshold_pct: f64,
    /// Per-domain comparisons.
    pub domains: Vec<DomainComparison>,
    /// Summary text.
    pub summary: String,
}

/// Per-domain comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainComparison {
    /// Domain name.
    pub domain: String,
    /// Circuit k parameter.
    pub k: u32,
    /// Baseline throughput.
    pub baseline_tps: f64,
    /// Current throughput.
    pub current_tps: f64,
    /// Percentage change (positive = improvement, negative = regression).
    pub tps_change_pct: f64,
    /// Baseline p50 latency.
    pub baseline_p50_ms: f64,
    /// Current p50 latency.
    pub current_p50_ms: f64,
    /// Percentage change (positive = slower = regression).
    pub latency_change_pct: f64,
    /// Whether this domain regressed.
    pub regressed: bool,
    /// Specific regression details (if any).
    pub regression_detail: Option<String>,
}

impl BenchmarkBaseline {
    /// Load baseline from a JSON file.
    pub fn load(path: &Path) -> Result<Self, String> {
        let contents =
            std::fs::read_to_string(path).map_err(|e| format!("failed to read baseline: {}", e))?;

        serde_json::from_str(&contents)
            .map_err(|e| format!("failed to parse baseline JSON: {}", e))
    }

    /// Save baseline to a JSON file.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create directory: {}", e))?;
        }

        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("failed to serialize baseline: {}", e))?;

        std::fs::write(path, json).map_err(|e| format!("failed to write baseline: {}", e))
    }

    /// Compare current metrics against this baseline.
    ///
    /// Returns a comparison result indicating whether any domain regressed
    /// beyond the threshold.
    pub fn compare(&self, current: &BenchmarkBaseline) -> ComparisonResult {
        let threshold = REGRESSION_THRESHOLD_PCT;
        let mut domains = Vec::new();
        let mut any_regressed = false;

        for baseline_domain in &self.domains {
            // Find matching domain in current.
            let current_domain = current.domains.iter().find(|d| {
                d.domain == baseline_domain.domain && d.k == baseline_domain.k
            });

            match current_domain {
                Some(curr) => {
                    // TPS regression: current < baseline by threshold%.
                    let tps_change_pct = if baseline_domain.throughput_tps > 0.0 {
                        ((curr.throughput_tps - baseline_domain.throughput_tps)
                            / baseline_domain.throughput_tps)
                            * 100.0
                    } else {
                        0.0
                    };

                    // Latency regression: current > baseline by threshold%.
                    let latency_change_pct = if baseline_domain.latency.p50_ms > 0.0 {
                        ((curr.latency.p50_ms - baseline_domain.latency.p50_ms)
                            / baseline_domain.latency.p50_ms)
                            * 100.0
                    } else {
                        0.0
                    };

                    let regressed = tps_change_pct < -threshold || latency_change_pct > threshold;
                    if regressed {
                        any_regressed = true;
                    }

                    let regression_detail = if regressed {
                        let mut details = Vec::new();
                        if tps_change_pct < -threshold {
                            details.push(format!(
                                "TPS regressed {:.1}% ({:.1} → {:.1})",
                                -tps_change_pct, baseline_domain.throughput_tps, curr.throughput_tps
                            ));
                        }
                        if latency_change_pct > threshold {
                            details.push(format!(
                                "p50 latency regressed {:.1}% ({:.2}ms → {:.2}ms)",
                                latency_change_pct,
                                baseline_domain.latency.p50_ms,
                                curr.latency.p50_ms
                            ));
                        }
                        Some(details.join("; "))
                    } else {
                        None
                    };

                    domains.push(DomainComparison {
                        domain: baseline_domain.domain.clone(),
                        k: baseline_domain.k,
                        baseline_tps: baseline_domain.throughput_tps,
                        current_tps: curr.throughput_tps,
                        tps_change_pct,
                        baseline_p50_ms: baseline_domain.latency.p50_ms,
                        current_p50_ms: curr.latency.p50_ms,
                        latency_change_pct,
                        regressed,
                        regression_detail,
                    });
                }
                None => {
                    // Domain missing from current — not a regression, just skip.
                    tracing::warn!(
                        domain = %baseline_domain.domain,
                        k = baseline_domain.k,
                        "domain present in baseline but missing from current"
                    );
                }
            }
        }

        let summary = if any_regressed {
            let regressed_domains: Vec<_> = domains
                .iter()
                .filter(|d| d.regressed)
                .map(|d| format!("{}@k={}", d.domain, d.k))
                .collect();
            format!(
                "REGRESSION DETECTED (>{:.0}% threshold): {}",
                threshold,
                regressed_domains.join(", ")
            )
        } else {
            format!(
                "All {} domains within {:.0}% threshold — PASS",
                domains.len(),
                threshold
            )
        };

        ComparisonResult {
            passed: !any_regressed,
            threshold_pct: threshold,
            domains,
            summary,
        }
    }

    /// Get the default baseline file path.
    pub fn default_path() -> PathBuf {
        PathBuf::from(DEFAULT_BASELINE_PATH)
    }
}

/// Compute latency percentiles from a sorted slice of durations (ms).
pub fn compute_latency_stats(samples_ms: &mut Vec<f64>) -> LatencyStats {
    if samples_ms.is_empty() {
        return LatencyStats {
            p50_ms: 0.0,
            p95_ms: 0.0,
            p99_ms: 0.0,
            mean_ms: 0.0,
            stddev_ms: 0.0,
            sample_count: 0,
        };
    }

    samples_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = samples_ms.len();
    let mean = samples_ms.iter().sum::<f64>() / n as f64;
    let variance = samples_ms.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    LatencyStats {
        p50_ms: percentile(samples_ms, 50.0),
        p95_ms: percentile(samples_ms, 95.0),
        p99_ms: percentile(samples_ms, 99.0),
        mean_ms: mean,
        stddev_ms: variance.sqrt(),
        sample_count: n,
    }
}

/// Compute the p-th percentile of a sorted slice.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Capture the current git commit hash.
pub fn git_commit_hash() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// Capture the rustc version string.
pub fn rustc_version() -> String {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// Detect CUDA version from nvcc.
pub fn cuda_version() -> Option<String> {
    std::process::Command::new("nvcc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                let output = String::from_utf8_lossy(&o.stdout);
                // Parse "release X.Y" from nvcc output.
                output
                    .lines()
                    .find(|line| line.contains("release"))
                    .and_then(|line| {
                        line.split("release ")
                            .nth(1)
                            .map(|v| v.split(',').next().unwrap_or(v).trim().to_string())
                    })
            } else {
                None
            }
        })
}

/// Detect GPU name from nvidia-smi.
pub fn gpu_name() -> Option<String> {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                let output = String::from_utf8_lossy(&o.stdout);
                output.lines().next().map(|s| s.trim().to_string())
            } else {
                None
            }
        })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_baseline(tps_multiplier: f64) -> BenchmarkBaseline {
        BenchmarkBaseline {
            schema_version: 1,
            git_commit: "abc123".to_string(),
            timestamp: "2025-01-01T00:00:00Z".to_string(),
            rustc_version: "rustc 1.80.0".to_string(),
            cuda_version: Some("12.8".to_string()),
            gpu_name: Some("RTX 5070".to_string()),
            domains: vec![
                DomainBaseline {
                    domain: "euler3d".to_string(),
                    k: 16,
                    latency: LatencyStats {
                        p50_ms: 100.0 / tps_multiplier,
                        p95_ms: 120.0 / tps_multiplier,
                        p99_ms: 150.0 / tps_multiplier,
                        mean_ms: 105.0 / tps_multiplier,
                        stddev_ms: 10.0,
                        sample_count: 100,
                    },
                    throughput_tps: 10.0 * tps_multiplier,
                    peak_vram_mib: Some(2048.0),
                    proof_size_bytes: 1024,
                    verify_time_ms: 5.0,
                },
                DomainBaseline {
                    domain: "thermal".to_string(),
                    k: 14,
                    latency: LatencyStats {
                        p50_ms: 50.0 / tps_multiplier,
                        p95_ms: 60.0 / tps_multiplier,
                        p99_ms: 80.0 / tps_multiplier,
                        mean_ms: 55.0 / tps_multiplier,
                        stddev_ms: 5.0,
                        sample_count: 100,
                    },
                    throughput_tps: 20.0 * tps_multiplier,
                    peak_vram_mib: Some(1024.0),
                    proof_size_bytes: 512,
                    verify_time_ms: 3.0,
                },
            ],
            aggregate: AggregateMetrics {
                total_tps: 30.0 * tps_multiplier,
                avg_latency_ms: 75.0 / tps_multiplier,
                peak_vram_mib: Some(2048.0),
            },
        }
    }

    #[test]
    fn test_baseline_save_load_roundtrip() {
        let baseline = make_test_baseline(1.0);
        let tmpdir = tempfile::tempdir().unwrap();
        let path = tmpdir.path().join("baseline.json");

        baseline.save(&path).unwrap();
        let loaded = BenchmarkBaseline::load(&path).unwrap();

        assert_eq!(loaded.schema_version, 1);
        assert_eq!(loaded.git_commit, "abc123");
        assert_eq!(loaded.domains.len(), 2);
        assert_eq!(loaded.domains[0].domain, "euler3d");
    }

    #[test]
    fn test_comparison_no_regression() {
        let baseline = make_test_baseline(1.0);
        let current = make_test_baseline(1.0); // Same performance.

        let result = baseline.compare(&current);
        assert!(result.passed);
        assert!(result.summary.contains("PASS"));
        assert!(result.domains.iter().all(|d| !d.regressed));
    }

    #[test]
    fn test_comparison_improvement() {
        let baseline = make_test_baseline(1.0);
        let current = make_test_baseline(1.2); // 20% faster.

        let result = baseline.compare(&current);
        assert!(result.passed);

        for d in &result.domains {
            assert!(d.tps_change_pct > 0.0, "expected TPS improvement");
        }
    }

    #[test]
    fn test_comparison_regression_detected() {
        let baseline = make_test_baseline(1.0);
        let current = make_test_baseline(0.8); // 20% slower → regression.

        let result = baseline.compare(&current);
        assert!(!result.passed);
        assert!(result.summary.contains("REGRESSION"));
        assert!(result.domains.iter().any(|d| d.regressed));
    }

    #[test]
    fn test_comparison_marginal_no_regression() {
        let baseline = make_test_baseline(1.0);
        let current = make_test_baseline(0.95); // 5% slower → within threshold.

        let result = baseline.compare(&current);
        assert!(result.passed);
    }

    #[test]
    fn test_latency_stats_computation() {
        let mut samples = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
        let stats = compute_latency_stats(&mut samples);

        assert_eq!(stats.sample_count, 10);
        assert!((stats.mean_ms - 55.0).abs() < 0.01);
        assert!(stats.p50_ms >= 50.0 && stats.p50_ms <= 60.0);
        assert!(stats.p95_ms >= 90.0);
        assert_eq!(stats.p99_ms, 100.0);
        assert!(stats.stddev_ms > 0.0);
    }

    #[test]
    fn test_empty_latency_stats() {
        let mut samples: Vec<f64> = vec![];
        let stats = compute_latency_stats(&mut samples);
        assert_eq!(stats.sample_count, 0);
        assert_eq!(stats.p50_ms, 0.0);
    }

    #[test]
    fn test_single_sample_latency() {
        let mut samples = vec![42.0];
        let stats = compute_latency_stats(&mut samples);
        assert_eq!(stats.sample_count, 1);
        assert_eq!(stats.p50_ms, 42.0);
        assert_eq!(stats.p99_ms, 42.0);
    }

    #[test]
    fn test_git_commit_hash() {
        let hash = git_commit_hash();
        // Should be either a 40-char hex string or "unknown".
        assert!(!hash.is_empty());
    }

    #[test]
    fn test_rustc_version_detection() {
        let version = rustc_version();
        assert!(version.contains("rustc") || version == "unknown");
    }

    #[test]
    fn test_domain_comparison_details() {
        let baseline = make_test_baseline(1.0);
        let current = make_test_baseline(0.7); // 30% slower.

        let result = baseline.compare(&current);
        let euler3d = result.domains.iter().find(|d| d.domain == "euler3d").unwrap();

        assert!(euler3d.regressed);
        assert!(euler3d.regression_detail.is_some());
        let detail = euler3d.regression_detail.as_ref().unwrap();
        assert!(detail.contains("TPS regressed"));
    }

    #[test]
    fn test_default_path() {
        let path = BenchmarkBaseline::default_path();
        assert_eq!(path.to_str().unwrap(), ".benchmark_baseline.json");
    }
}
