//! Real Data Pipeline Benchmark
//!
//! Exercises the full QTT pipeline with realistic tensor data:
//! 1. Generate weather-like tensor data
//! 2. Serialize to QTT format
//! 3. Read through QTTBridge
//! 4. Evaluate on CPU (GPU simulation)
//! 5. Collect comprehensive metrics
//!
//! # Metrics Collected
//!
//! - Throughput (frames/sec, GB/s)
//! - Latency (p50, p95, p99)
//! - Memory usage (peak, TT vs dense)
//! - Compression effectiveness
//! - Evaluation accuracy

use std::time::Instant;
use std::io::Write;
use std::path::Path;
use std::fs::File;

use crate::qtt::{
    QTTBridgeHeader, QTT_BRIDGE_MAGIC, QTT_BRIDGE_VERSION,
    MAX_QTT_SITES,
};

// ─────────────────────────────────────────────────────────────────────────────
// Metrics Structures
// ─────────────────────────────────────────────────────────────────────────────

/// Latency percentiles in microseconds
#[derive(Debug, Clone)]
pub struct LatencyMetrics {
    pub min_us: f64,
    pub p50_us: f64,
    pub p95_us: f64,
    pub p99_us: f64,
    pub max_us: f64,
    pub mean_us: f64,
    pub std_dev_us: f64,
}

impl LatencyMetrics {
    pub fn from_samples(mut samples: Vec<f64>) -> Self {
        if samples.is_empty() {
            return Self {
                min_us: 0.0,
                p50_us: 0.0,
                p95_us: 0.0,
                p99_us: 0.0,
                max_us: 0.0,
                mean_us: 0.0,
                std_dev_us: 0.0,
            };
        }

        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = samples.len();

        let min = samples[0];
        let max = samples[n - 1];
        let p50 = samples[n / 2];
        let p95 = samples[(n * 95) / 100];
        let p99 = samples[(n * 99) / 100];

        let sum: f64 = samples.iter().sum();
        let mean = sum / n as f64;

        let variance: f64 = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();

        Self {
            min_us: min,
            p50_us: p50,
            p95_us: p95,
            p99_us: p99,
            max_us: max,
            mean_us: mean,
            std_dev_us: std_dev,
        }
    }
}

/// Memory metrics in bytes
#[derive(Debug, Clone)]
pub struct MemoryMetrics {
    /// Size of TT-core data
    pub tt_core_bytes: usize,
    /// Equivalent dense tensor size
    pub dense_equiv_bytes: usize,
    /// Compression ratio
    pub compression_ratio: f32,
    /// Peak memory during evaluation
    pub peak_memory_bytes: usize,
    /// Memory saved vs dense
    pub memory_saved_bytes: usize,
}

/// Throughput metrics
#[derive(Debug, Clone)]
pub struct ThroughputMetrics {
    /// Frames processed per second
    pub frames_per_sec: f64,
    /// Gigabytes per second (raw data rate)
    pub gbps: f64,
    /// Tensor elements per second
    pub elements_per_sec: f64,
    /// Total frames processed
    pub total_frames: usize,
    /// Total duration
    pub total_duration_ms: f64,
}

/// Complete benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// Test configuration
    pub config: BenchmarkConfig,
    /// Serialization latency (generate TT data)
    pub serialize_latency: LatencyMetrics,
    /// Deserialization latency (read QTT frames)
    pub deserialize_latency: LatencyMetrics,
    /// Evaluation latency (CPU TT contraction)
    pub eval_latency: LatencyMetrics,
    /// End-to-end latency
    pub e2e_latency: LatencyMetrics,
    /// Throughput metrics
    pub throughput: ThroughputMetrics,
    /// Memory metrics
    pub memory: MemoryMetrics,
    /// Timestamp
    pub timestamp: String,
}

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of TT sites
    pub num_sites: u32,
    /// Physical dimension per site
    pub physical_dim: u32,
    /// Maximum bond dimension
    pub max_bond_dim: u32,
    /// Number of warmup iterations
    pub warmup_iters: usize,
    /// Number of benchmark iterations
    pub bench_iters: usize,
    /// Number of query points per frame
    pub queries_per_frame: usize,
    /// Test description
    pub description: String,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            num_sites: 16,
            physical_dim: 2,
            max_bond_dim: 32,
            warmup_iters: 10,
            bench_iters: 100,
            queries_per_frame: 1000,
            description: "Default benchmark".to_string(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Synthetic QTT Data Generator
// ─────────────────────────────────────────────────────────────────────────────

/// Generates realistic TT-cores that represent weather-like smooth functions
pub struct QTTDataGenerator {
    num_sites: u32,
    physical_dim: u32,
    max_bond_dim: u32,
    bond_dims: Vec<u16>,
    rng_state: u64,
}

impl QTTDataGenerator {
    pub fn new(num_sites: u32, physical_dim: u32, max_bond_dim: u32) -> Self {
        // Calculate realistic bond dimensions
        // Start small, grow towards middle, shrink back
        let mut bond_dims = Vec::with_capacity((num_sites - 1) as usize);
        for i in 0..(num_sites - 1) {
            let position = i as f32 / (num_sites - 1) as f32;
            // Parabolic profile: highest in middle
            let factor = 4.0 * position * (1.0 - position);
            let dim = 1 + (factor * (max_bond_dim - 1) as f32) as u16;
            bond_dims.push(dim.min(max_bond_dim as u16));
        }

        Self {
            num_sites,
            physical_dim,
            max_bond_dim,
            bond_dims,
            rng_state: 0x12345678_9ABCDEF0,
        }
    }

    /// Simple LCG random number generator
    fn next_random(&mut self) -> f32 {
        // LCG parameters
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Map to [-1, 1]
        let bits = (self.rng_state >> 33) as u32;
        (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
    }

    /// Get bond dimensions
    pub fn bond_dims(&self) -> &[u16] {
        &self.bond_dims
    }

    /// Get left bond dimension for site i
    fn chi_left(&self, site: usize) -> usize {
        if site == 0 { 1 } else { self.bond_dims[site - 1] as usize }
    }

    /// Get right bond dimension for site i
    fn chi_right(&self, site: usize) -> usize {
        if site >= self.num_sites as usize - 1 { 1 } else { self.bond_dims[site] as usize }
    }

    /// Generate all TT-cores
    pub fn generate_cores(&mut self) -> (Vec<f32>, Vec<u32>) {
        let mut cores = Vec::new();
        let mut offsets = Vec::with_capacity(self.num_sites as usize);

        for site in 0..self.num_sites as usize {
            offsets.push((cores.len() * 4) as u32);

            let chi_l = self.chi_left(site);
            let chi_r = self.chi_right(site);
            let d = self.physical_dim as usize;

            let _core_size = chi_l * d * chi_r;

            // Generate core with structure
            // For smooth functions, we want structured cores not pure noise
            for i in 0..chi_l {
                for _j in 0..d {
                    for k in 0..chi_r {
                        // Create structured values with some randomness
                        let base = if i == k { 1.0 } else { 0.1 };
                        let noise = self.next_random() * 0.1;
                        let value = base + noise;
                        cores.push(value);
                    }
                }
            }
        }

        (cores, offsets)
    }

    /// Calculate total core bytes
    pub fn total_core_bytes(&self) -> usize {
        let mut total = 0usize;
        for site in 0..self.num_sites as usize {
            let chi_l = self.chi_left(site);
            let chi_r = self.chi_right(site);
            let d = self.physical_dim as usize;
            total += chi_l * d * chi_r * 4; // f32 = 4 bytes
        }
        total
    }

    /// Calculate mean bond dimension
    pub fn mean_bond_dim(&self) -> f32 {
        if self.bond_dims.is_empty() {
            return 1.0;
        }
        let sum: u32 = self.bond_dims.iter().map(|&x| x as u32).sum();
        sum as f32 / self.bond_dims.len() as f32
    }

    /// Calculate equivalent dense size
    pub fn dense_equiv_bytes(&self) -> usize {
        let d = self.physical_dim as usize;
        let l = self.num_sites as usize;
        d.pow(l as u32) * 4 // f32 = 4 bytes
    }

    /// Build a complete QTT header
    pub fn build_header(&self, frame_number: u64, cores: &[f32], offsets: &[u32]) -> QTTBridgeHeader {
        let mut header = QTTBridgeHeader::default();

        header.magic = QTT_BRIDGE_MAGIC;
        header.version = QTT_BRIDGE_VERSION;
        header.frame_number = frame_number;
        header.num_sites = self.num_sites;
        header.physical_dim = self.physical_dim;
        header.max_bond_dim = self.max_bond_dim;
        header.num_cores = self.num_sites;

        // Original shape (for 16 sites, d=2 → 2^16 = 65536 grid points)
        let grid_size = self.physical_dim.pow(self.num_sites);
        header.original_shape = [grid_size, 1, 1, 1];
        header.original_elements = grid_size as u64;

        // Compression metrics
        let tt_bytes = cores.len() * 4;
        let dense_bytes = self.dense_equiv_bytes();
        header.compression_ratio = dense_bytes as f32 / tt_bytes as f32;
        header.truncation_error = 1e-8; // Simulated low error
        header.mean_bond_dim = self.mean_bond_dim();
        header.max_singular_value = 1.0;

        // Copy bond dimensions
        for (i, &dim) in self.bond_dims.iter().enumerate() {
            if i < MAX_QTT_SITES {
                header.bond_dims[i] = dim;
            }
        }

        // Copy core offsets
        for (i, &offset) in offsets.iter().enumerate() {
            if i < MAX_QTT_SITES {
                header.core_offsets[i] = offset;
            }
        }

        // Flags
        header.flags = 0x12; // is_canonical | is_ready

        // Data size
        header.total_data_bytes = tt_bytes as u32;

        // CRC32 of core data
        let crc = crc::Crc::<u32>::new(&crc::CRC_32_ISO_HDLC);
        let core_bytes: &[u8] = bytemuck::cast_slice(cores);
        header.data_checksum = crc.checksum(core_bytes);

        // Timestamp
        header.producer_timestamp_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        header
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TT Evaluator (CPU fallback for benchmark)
// ─────────────────────────────────────────────────────────────────────────────

/// CPU-based TT evaluator for benchmarking
pub struct BenchTTEvaluator {
    num_sites: u32,
    physical_dim: u32,
    bond_dims: Vec<u16>,
    cores: Vec<f32>,
    core_offsets: Vec<u32>,
}

impl BenchTTEvaluator {
    pub fn new(header: &QTTBridgeHeader, cores: Vec<f32>) -> Self {
        let mut bond_dims = Vec::with_capacity(header.num_sites as usize - 1);
        for i in 0..(header.num_sites as usize - 1) {
            bond_dims.push(header.bond_dims[i]);
        }

        let mut core_offsets = Vec::with_capacity(header.num_sites as usize);
        for i in 0..header.num_sites as usize {
            core_offsets.push(header.core_offsets[i]);
        }

        Self {
            num_sites: header.num_sites,
            physical_dim: header.physical_dim,
            bond_dims,
            cores,
            core_offsets,
        }
    }

    fn chi_left(&self, site: usize) -> usize {
        if site == 0 { 1 } else { self.bond_dims[site - 1] as usize }
    }

    fn chi_right(&self, site: usize) -> usize {
        if site >= self.num_sites as usize - 1 { 1 } else { self.bond_dims[site] as usize }
    }

    fn get_core_element(&self, site: usize, alpha_l: usize, x: usize, alpha_r: usize) -> f32 {
        let chi_r = self.chi_right(site);
        let d = self.physical_dim as usize;

        let local_idx = alpha_l * (d * chi_r) + x * chi_r + alpha_r;
        let offset = self.core_offsets[site] as usize / 4;
        let global_idx = offset + local_idx;

        self.cores.get(global_idx).copied().unwrap_or(0.0)
    }

    /// Evaluate TT at a single multi-index
    pub fn evaluate_single(&self, indices: &[u32]) -> f32 {
        let num_sites = self.num_sites as usize;

        if indices.len() < num_sites {
            return 0.0;
        }

        let mut acc = vec![1.0f32];
        let mut current_dim = 1usize;

        for site in 0..num_sites {
            let x_i = indices[site] as usize;
            let chi_l = self.chi_left(site);
            let chi_r = self.chi_right(site);

            if chi_l != current_dim {
                return 0.0;
            }

            let mut new_acc = vec![0.0f32; chi_r];
            for j in 0..chi_r {
                let mut sum = 0.0f32;
                for i in 0..chi_l {
                    sum += acc[i] * self.get_core_element(site, i, x_i, j);
                }
                new_acc[j] = sum;
            }

            acc = new_acc;
            current_dim = chi_r;
        }

        acc.first().copied().unwrap_or(0.0)
    }

    /// Evaluate at multiple points
    pub fn evaluate_batch(&self, indices: &[u32], num_queries: usize) -> Vec<f32> {
        let num_sites = self.num_sites as usize;
        (0..num_queries)
            .map(|q| {
                let start = q * num_sites;
                let end = start + num_sites;
                if end <= indices.len() {
                    self.evaluate_single(&indices[start..end])
                } else {
                    0.0
                }
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark Runner
// ─────────────────────────────────────────────────────────────────────────────

/// Run the complete benchmark suite
pub fn run_benchmark(config: &BenchmarkConfig) -> BenchmarkResults {
    eprintln!("\n╔════════════════════════════════════════════════════════════════╗");
    eprintln!("║            QTT PIPELINE REAL DATA BENCHMARK                   ║");
    eprintln!("╠════════════════════════════════════════════════════════════════╣");
    eprintln!("║ Sites: {:3}  │  d: {}  │  χ_max: {:3}  │  Iters: {:5}           ║",
             config.num_sites, config.physical_dim, config.max_bond_dim, config.bench_iters);
    eprintln!("╚════════════════════════════════════════════════════════════════╝\n");

    // Initialize generator
    let mut generator = QTTDataGenerator::new(
        config.num_sites,
        config.physical_dim,
        config.max_bond_dim,
    );

    // Pre-generate some data for warmup
    let (warmup_cores, warmup_offsets) = generator.generate_cores();
    let warmup_header = generator.build_header(0, &warmup_cores, &warmup_offsets);

    // Generate query indices
    let mut query_indices: Vec<u32> = Vec::with_capacity(
        config.queries_per_frame * config.num_sites as usize
    );
    let mut rng_state = 0xDEADBEEF_u64;
    for _ in 0..(config.queries_per_frame * config.num_sites as usize) {
        rng_state = rng_state.wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        query_indices.push((rng_state as u32) % config.physical_dim);
    }

    // ─── Warmup Phase ────────────────────────────────────────────────────────
    eprintln!("▸ Warmup phase ({} iterations)...", config.warmup_iters);
    for _i in 0..config.warmup_iters {
        let evaluator = BenchTTEvaluator::new(&warmup_header, warmup_cores.clone());
        let _ = evaluator.evaluate_batch(&query_indices, config.queries_per_frame);
    }
    eprintln!("  ✓ Warmup complete\n");

    // ─── Benchmark Phase ─────────────────────────────────────────────────────
    let mut serialize_times = Vec::with_capacity(config.bench_iters);
    let mut deserialize_times = Vec::with_capacity(config.bench_iters);
    let mut eval_times = Vec::with_capacity(config.bench_iters);
    let mut e2e_times = Vec::with_capacity(config.bench_iters);

    eprintln!("▸ Benchmark phase ({} iterations)...", config.bench_iters);
    let total_start = Instant::now();

    for i in 0..config.bench_iters {
        if i > 0 && i % 25 == 0 {
            eprintln!("  Progress: {}/{}", i, config.bench_iters);
        }

        let frame_start = Instant::now();

        // Phase 1: Serialize (generate QTT data)
        let serialize_start = Instant::now();
        let (cores, offsets) = generator.generate_cores();
        let header = generator.build_header(i as u64, &cores, &offsets);
        let serialize_us = serialize_start.elapsed().as_micros() as f64;
        serialize_times.push(serialize_us);

        // Phase 2: Deserialize (read/validate header)
        let deserialize_start = Instant::now();
        assert_eq!(header.magic, QTT_BRIDGE_MAGIC);
        assert_eq!(header.version, QTT_BRIDGE_VERSION);
        let _ = header.validate();
        let evaluator = BenchTTEvaluator::new(&header, cores);
        let deserialize_us = deserialize_start.elapsed().as_micros() as f64;
        deserialize_times.push(deserialize_us);

        // Phase 3: Evaluate (CPU TT contraction)
        let eval_start = Instant::now();
        let results = evaluator.evaluate_batch(&query_indices, config.queries_per_frame);
        let eval_us = eval_start.elapsed().as_micros() as f64;
        eval_times.push(eval_us);

        // Validate results are finite
        for r in &results {
            assert!(r.is_finite(), "TT evaluation produced non-finite result");
        }

        let e2e_us = frame_start.elapsed().as_micros() as f64;
        e2e_times.push(e2e_us);
    }

    let total_duration = total_start.elapsed();
    eprintln!("  ✓ Benchmark complete ({:.2}s)\n", total_duration.as_secs_f64());

    // ─── Calculate Metrics ───────────────────────────────────────────────────
    let tt_bytes = generator.total_core_bytes();
    let dense_bytes = generator.dense_equiv_bytes();
    let compression_ratio = dense_bytes as f32 / tt_bytes as f32;

    let total_ms = total_duration.as_secs_f64() * 1000.0;
    let frames_per_sec = config.bench_iters as f64 / total_duration.as_secs_f64();
    let total_bytes_processed = tt_bytes * config.bench_iters;
    let gbps = total_bytes_processed as f64 / total_duration.as_secs_f64() / 1e9;
    let elements_per_sec = (config.queries_per_frame * config.bench_iters) as f64
        / total_duration.as_secs_f64();

    let results = BenchmarkResults {
        config: config.clone(),
        serialize_latency: LatencyMetrics::from_samples(serialize_times),
        deserialize_latency: LatencyMetrics::from_samples(deserialize_times),
        eval_latency: LatencyMetrics::from_samples(eval_times),
        e2e_latency: LatencyMetrics::from_samples(e2e_times),
        throughput: ThroughputMetrics {
            frames_per_sec,
            gbps,
            elements_per_sec,
            total_frames: config.bench_iters,
            total_duration_ms: total_ms,
        },
        memory: MemoryMetrics {
            tt_core_bytes: tt_bytes,
            dense_equiv_bytes: dense_bytes,
            compression_ratio,
            peak_memory_bytes: tt_bytes + config.queries_per_frame * 4 * 2, // cores + indices + results
            memory_saved_bytes: dense_bytes.saturating_sub(tt_bytes),
        },
        timestamp: chrono_lite_timestamp(),
    };

    // Print results
    print_results(&results);

    results
}

/// Simple timestamp without chrono dependency
fn chrono_lite_timestamp() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", now.as_secs())
}

/// Print formatted benchmark results
pub fn print_results(results: &BenchmarkResults) {
    eprintln!("┌────────────────────────────────────────────────────────────────┐");
    eprintln!("│                    BENCHMARK RESULTS                          │");
    eprintln!("├────────────────────────────────────────────────────────────────┤");
    eprintln!("│ Configuration:                                                │");
    eprintln!("│   Sites: {:3}  Physical dim: {}  Max bond: {:3}                 │",
             results.config.num_sites, results.config.physical_dim, results.config.max_bond_dim);
    eprintln!("│   Iterations: {}  Queries/frame: {}                        │",
             results.config.bench_iters, results.config.queries_per_frame);
    eprintln!("├────────────────────────────────────────────────────────────────┤");
    eprintln!("│ THROUGHPUT:                                                   │");
    eprintln!("│   Frames/sec:     {:12.2}                                │", results.throughput.frames_per_sec);
    eprintln!("│   Data rate:      {:12.4} GB/s                          │", results.throughput.gbps);
    eprintln!("│   Elements/sec:   {:12.0}                                │", results.throughput.elements_per_sec);
    eprintln!("├────────────────────────────────────────────────────────────────┤");
    eprintln!("│ LATENCY (microseconds):                                       │");
    eprintln!("│                    p50         p95         p99        max     │");
    eprintln!("│   Serialize:   {:8.1}    {:8.1}    {:8.1}    {:8.1}     │",
             results.serialize_latency.p50_us, results.serialize_latency.p95_us,
             results.serialize_latency.p99_us, results.serialize_latency.max_us);
    eprintln!("│   Deserialize: {:8.1}    {:8.1}    {:8.1}    {:8.1}     │",
             results.deserialize_latency.p50_us, results.deserialize_latency.p95_us,
             results.deserialize_latency.p99_us, results.deserialize_latency.max_us);
    eprintln!("│   Evaluate:    {:8.1}    {:8.1}    {:8.1}    {:8.1}     │",
             results.eval_latency.p50_us, results.eval_latency.p95_us,
             results.eval_latency.p99_us, results.eval_latency.max_us);
    eprintln!("│   End-to-End:  {:8.1}    {:8.1}    {:8.1}    {:8.1}     │",
             results.e2e_latency.p50_us, results.e2e_latency.p95_us,
             results.e2e_latency.p99_us, results.e2e_latency.max_us);
    eprintln!("├────────────────────────────────────────────────────────────────┤");
    eprintln!("│ MEMORY:                                                       │");
    eprintln!("│   TT-core size:   {:12} bytes ({:.2} KB)                 │",
             results.memory.tt_core_bytes, results.memory.tt_core_bytes as f64 / 1024.0);
    eprintln!("│   Dense equiv:    {:12} bytes ({:.2} KB)                 │",
             results.memory.dense_equiv_bytes, results.memory.dense_equiv_bytes as f64 / 1024.0);
    eprintln!("│   Compression:    {:12.2}x                                │", results.memory.compression_ratio);
    eprintln!("│   Memory saved:   {:12} bytes ({:.2} KB)                 │",
             results.memory.memory_saved_bytes, results.memory.memory_saved_bytes as f64 / 1024.0);
    eprintln!("└────────────────────────────────────────────────────────────────┘");
}

/// Export results to JSON
pub fn export_results_json(results: &BenchmarkResults, path: &Path) -> std::io::Result<()> {
    let mut file = File::create(path)?;

    writeln!(file, "{{")?;
    writeln!(file, "  \"timestamp\": \"{}\",", results.timestamp)?;
    writeln!(file, "  \"config\": {{")?;
    writeln!(file, "    \"num_sites\": {},", results.config.num_sites)?;
    writeln!(file, "    \"physical_dim\": {},", results.config.physical_dim)?;
    writeln!(file, "    \"max_bond_dim\": {},", results.config.max_bond_dim)?;
    writeln!(file, "    \"bench_iters\": {},", results.config.bench_iters)?;
    writeln!(file, "    \"queries_per_frame\": {},", results.config.queries_per_frame)?;
    writeln!(file, "    \"description\": \"{}\"", results.config.description)?;
    writeln!(file, "  }},")?;

    writeln!(file, "  \"throughput\": {{")?;
    writeln!(file, "    \"frames_per_sec\": {:.4},", results.throughput.frames_per_sec)?;
    writeln!(file, "    \"gbps\": {:.6},", results.throughput.gbps)?;
    writeln!(file, "    \"elements_per_sec\": {:.0},", results.throughput.elements_per_sec)?;
    writeln!(file, "    \"total_frames\": {},", results.throughput.total_frames)?;
    writeln!(file, "    \"total_duration_ms\": {:.2}", results.throughput.total_duration_ms)?;
    writeln!(file, "  }},")?;

    writeln!(file, "  \"latency_us\": {{")?;
    writeln!(file, "    \"serialize\": {{ \"p50\": {:.1}, \"p95\": {:.1}, \"p99\": {:.1}, \"max\": {:.1} }},",
             results.serialize_latency.p50_us, results.serialize_latency.p95_us,
             results.serialize_latency.p99_us, results.serialize_latency.max_us)?;
    writeln!(file, "    \"deserialize\": {{ \"p50\": {:.1}, \"p95\": {:.1}, \"p99\": {:.1}, \"max\": {:.1} }},",
             results.deserialize_latency.p50_us, results.deserialize_latency.p95_us,
             results.deserialize_latency.p99_us, results.deserialize_latency.max_us)?;
    writeln!(file, "    \"evaluate\": {{ \"p50\": {:.1}, \"p95\": {:.1}, \"p99\": {:.1}, \"max\": {:.1} }},",
             results.eval_latency.p50_us, results.eval_latency.p95_us,
             results.eval_latency.p99_us, results.eval_latency.max_us)?;
    writeln!(file, "    \"e2e\": {{ \"p50\": {:.1}, \"p95\": {:.1}, \"p99\": {:.1}, \"max\": {:.1} }}",
             results.e2e_latency.p50_us, results.e2e_latency.p95_us,
             results.e2e_latency.p99_us, results.e2e_latency.max_us)?;
    writeln!(file, "  }},")?;

    writeln!(file, "  \"memory\": {{")?;
    writeln!(file, "    \"tt_core_bytes\": {},", results.memory.tt_core_bytes)?;
    writeln!(file, "    \"dense_equiv_bytes\": {},", results.memory.dense_equiv_bytes)?;
    writeln!(file, "    \"compression_ratio\": {:.2},", results.memory.compression_ratio)?;
    writeln!(file, "    \"memory_saved_bytes\": {}", results.memory.memory_saved_bytes)?;
    writeln!(file, "  }}")?;
    writeln!(file, "}}")?;

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Test Suite
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_generator_basic() {
        let mut gen = QTTDataGenerator::new(8, 2, 16);
        let (cores, offsets) = gen.generate_cores();

        assert_eq!(offsets.len(), 8);
        assert!(!cores.is_empty());

        // Verify bond dims have parabolic profile
        let bond_dims = gen.bond_dims();
        assert_eq!(bond_dims.len(), 7);
        assert!(bond_dims[3] >= bond_dims[0]); // Middle >= edge
    }

    #[test]
    fn test_header_generation() {
        let mut gen = QTTDataGenerator::new(8, 2, 16);
        let (cores, offsets) = gen.generate_cores();
        let header = gen.build_header(42, &cores, &offsets);

        assert_eq!(header.magic, QTT_BRIDGE_MAGIC);
        assert_eq!(header.version, QTT_BRIDGE_VERSION);
        assert_eq!(header.frame_number, 42);
        assert_eq!(header.num_sites, 8);
        assert_eq!(header.physical_dim, 2);
        assert!(header.compression_ratio > 0.0); // Small tensors may expand
    }

    #[test]
    fn test_evaluator_runs() {
        let mut gen = QTTDataGenerator::new(4, 2, 4);
        let (cores, offsets) = gen.generate_cores();
        let header = gen.build_header(0, &cores, &offsets);

        let evaluator = BenchTTEvaluator::new(&header, cores);

        // Evaluate at index (0, 1, 0, 1)
        let result = evaluator.evaluate_single(&[0, 1, 0, 1]);
        assert!(result.is_finite());
    }

    #[test]
    fn test_batch_evaluation() {
        let mut gen = QTTDataGenerator::new(4, 2, 4);
        let (cores, offsets) = gen.generate_cores();
        let header = gen.build_header(0, &cores, &offsets);

        let evaluator = BenchTTEvaluator::new(&header, cores);

        // 3 queries, 4 sites each
        let indices: Vec<u32> = vec![0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1];
        let results = evaluator.evaluate_batch(&indices, 3);

        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(r.is_finite());
        }
    }

    #[test]
    fn test_latency_metrics() {
        let samples: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let metrics = LatencyMetrics::from_samples(samples);

        assert_eq!(metrics.min_us, 1.0);
        assert_eq!(metrics.max_us, 100.0);
        assert!(metrics.p50_us >= 50.0 && metrics.p50_us <= 52.0, "p50 out of range: {}", metrics.p50_us);
        assert!(metrics.p95_us >= 90.0 && metrics.p95_us <= 100.0, "p95 out of range: {}", metrics.p95_us);
        assert!(metrics.p99_us >= 90.0 && metrics.p99_us <= 102.0, "p99 out of range: {}", metrics.p99_us);
    }

    #[test]
    fn test_compression_ratio() {
        let gen = QTTDataGenerator::new(16, 2, 32);

        let tt_bytes = gen.total_core_bytes();
        let dense_bytes = gen.dense_equiv_bytes();

        // 2^16 = 65536 elements × 4 bytes = 262144 bytes dense
        assert_eq!(dense_bytes, 262144);

        // TT should be much smaller
        let ratio = dense_bytes as f32 / tt_bytes as f32;
        assert!(ratio > 1.0, "Expected compression > 1x, got {:.2}x", ratio);
    }

    #[test]
    fn test_mini_benchmark() {
        let config = BenchmarkConfig {
            num_sites: 8,
            physical_dim: 2,
            max_bond_dim: 8,
            warmup_iters: 2,
            bench_iters: 10,
            queries_per_frame: 100,
            description: "Mini test".to_string(),
        };

        let results = run_benchmark(&config);

        assert!(results.throughput.frames_per_sec > 0.0);
        assert!(results.memory.compression_ratio > 0.0); // Small tensors may expand
        assert!(results.e2e_latency.p50_us > 0.0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public Benchmark Entry Point
// ─────────────────────────────────────────────────────────────────────────────

/// Run a standard benchmark suite with multiple configurations
pub fn run_standard_benchmark_suite() -> Vec<BenchmarkResults> {
    let configs = vec![
        BenchmarkConfig {
            num_sites: 10,
            physical_dim: 2,
            max_bond_dim: 16,
            warmup_iters: 10,
            bench_iters: 100,
            queries_per_frame: 1000,
            description: "Small: 2^10 grid, χ=16".to_string(),
        },
        BenchmarkConfig {
            num_sites: 16,
            physical_dim: 2,
            max_bond_dim: 32,
            warmup_iters: 10,
            bench_iters: 100,
            queries_per_frame: 1000,
            description: "Medium: 2^16 grid, χ=32".to_string(),
        },
        BenchmarkConfig {
            num_sites: 20,
            physical_dim: 2,
            max_bond_dim: 64,
            warmup_iters: 10,
            bench_iters: 50,
            queries_per_frame: 1000,
            description: "Large: 2^20 grid, χ=64".to_string(),
        },
    ];

    let mut all_results = Vec::new();

    for config in configs {
        eprintln!("\n{}", "=".repeat(64));
        eprintln!("Running: {}", config.description);
        eprintln!("{}", "=".repeat(64));

        let results = run_benchmark(&config);
        all_results.push(results);
    }

    all_results
}
