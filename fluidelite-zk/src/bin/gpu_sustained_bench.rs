//! GPU Sustained Benchmark: Real-World TPS Measurement
//!
//! This benchmark simulates production ZK proof workloads:
//! - Pre-generates all data to eliminate CPU bottleneck
//! - Runs sustained back-to-back MSMs (GPU saturation)
//! - Measures true TPS (transactions per second)
//! - Tracks VRAM, latency percentiles, and throughput
//!
//! # Usage
//!
//! ```bash
//! export ICICLE_BACKEND_INSTALL_DIR=/opt/icicle/lib
//! cargo run --release --bin gpu-sustained-bench --features gpu
//! ```

use std::process::Command;
use std::time::{Duration, Instant};
use fluidelite_zk::gpu::GpuAccelerator;

// ============================================================================
// Configuration
// ============================================================================

/// Proof circuit sizes (MSM points per proof)
const PROOF_SIZES: &[(u32, &str)] = &[
    (14, "Small (16K) - Simple transfers"),
    (16, "Medium (65K) - Token swaps"),
    (18, "Large (262K) - Complex DeFi"),
    (20, "XLarge (1M) - ML inference proofs"),
    (22, "XXLarge (4M) - Full model attestation"),
];

/// Duration to run each benchmark (seconds)
const BENCH_DURATION_SECS: u64 = 10;

/// Warmup iterations before measurement
const WARMUP_ITERS: usize = 3;

// ============================================================================
// Utilities
// ============================================================================

fn get_vram_mb() -> f64 {
    Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(0.0)
}

fn get_gpu_info() -> (String, f64) {
    Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .and_then(|s| {
            let parts: Vec<&str> = s.trim().split(',').collect();
            if parts.len() >= 2 {
                let name = parts[0].trim().to_string();
                let total = parts[1].trim().parse::<f64>().ok()?;
                Some((name, total))
            } else {
                None
            }
        })
        .unwrap_or(("Unknown GPU".to_string(), 0.0))
}

fn format_duration(d: Duration) -> String {
    let us = d.as_micros();
    if us < 1000 {
        format!("{} µs", us)
    } else if us < 1_000_000 {
        format!("{:.2} ms", us as f64 / 1000.0)
    } else {
        format!("{:.3} s", us as f64 / 1_000_000.0)
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p / 100.0).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ============================================================================
// Benchmark Results
// ============================================================================

#[derive(Debug, Clone)]
struct BenchResult {
    proof_size: usize,
    description: String,
    total_proofs: usize,
    total_time_secs: f64,
    tps: f64,
    latency_min_ms: f64,
    latency_p50_ms: f64,
    latency_p95_ms: f64,
    latency_p99_ms: f64,
    latency_max_ms: f64,
    points_per_sec: f64,
    vram_peak_mb: f64,
    vram_baseline_mb: f64,
}

// ============================================================================
// Main Benchmark
// ============================================================================

fn main() {
    let (gpu_name, gpu_vram) = get_gpu_info();
    
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    FLUIDELITE GPU SUSTAINED BENCHMARK                                ║");
    println!("║                      \"Real-World TPS Measurement\"                                    ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║  GPU: {:<66} {:>6.0} MB  ║", gpu_name, gpu_vram);
    println!("║  Mode: Sustained GPU Saturation | Pre-generated Data                                 ║");
    println!("║  Duration: {} seconds per proof size                                                 ║", BENCH_DURATION_SECS);
    println!("╚══════════════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Initialize GPU
    println!("🔧 Initializing GPU accelerator...");
    let gpu = match GpuAccelerator::new() {
        Ok(g) => {
            println!("✅ GPU initialized: {}", g.device_name());
            g
        }
        Err(e) => {
            eprintln!("❌ Failed to initialize GPU: {}", e);
            std::process::exit(1);
        }
    };

    let baseline_vram = get_vram_mb();
    println!("📊 Baseline VRAM: {:.0} MB", baseline_vram);
    println!();

    let mut all_results: Vec<BenchResult> = Vec::new();

    for (log_size, description) in PROOF_SIZES {
        let size = 1usize << log_size;
        
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("📦 PROOF SIZE: 2^{} = {} points", log_size, size);
        println!("   {}", description);
        println!();

        // Pre-generate data
        println!("   ⏳ Pre-generating {} MB of test data...", 
                 size * 96 / 1024 / 1024); // 32B scalar + 64B point
        
        let data_gen_start = Instant::now();
        
        // Use Icicle's random generation (this happens on GPU too!)
        use icicle_bn254::curve::{G1Affine, ScalarField};
        use icicle_core::traits::GenerateRandom;
        
        let scalars = ScalarField::generate_random(size);
        let points = G1Affine::generate_random(size);
        
        let data_gen_time = data_gen_start.elapsed();
        println!("   ✅ Data generated in {}", format_duration(data_gen_time));
        
        // Warmup
        println!("   🔥 Warming up ({} iterations)...", WARMUP_ITERS);
        for _ in 0..WARMUP_ITERS {
            let _ = gpu.msm_bn254(&points, &scalars);
        }
        
        // Sustained benchmark
        println!("   🚀 Running sustained benchmark for {} seconds...", BENCH_DURATION_SECS);
        println!();
        
        let mut latencies: Vec<f64> = Vec::new();
        let mut vram_peak = baseline_vram;
        let bench_start = Instant::now();
        let bench_deadline = Duration::from_secs(BENCH_DURATION_SECS);
        
        let mut last_print = Instant::now();
        let mut proofs_since_print = 0;
        
        while bench_start.elapsed() < bench_deadline {
            let msm_start = Instant::now();
            
            match gpu.msm_bn254(&points, &scalars) {
                Ok(_) => {
                    let latency_ms = msm_start.elapsed().as_secs_f64() * 1000.0;
                    latencies.push(latency_ms);
                    proofs_since_print += 1;
                    
                    // Sample VRAM periodically
                    if latencies.len() % 10 == 0 {
                        let current_vram = get_vram_mb();
                        vram_peak = vram_peak.max(current_vram);
                    }
                    
                    // Live progress every 2 seconds
                    if last_print.elapsed() >= Duration::from_secs(2) {
                        let elapsed = bench_start.elapsed().as_secs_f64();
                        let instant_tps = proofs_since_print as f64 / last_print.elapsed().as_secs_f64();
                        let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
                        println!("      [{:>5.1}s] {} proofs | {:.1} TPS | avg latency: {:.2} ms",
                                elapsed, latencies.len(), instant_tps, avg_latency);
                        last_print = Instant::now();
                        proofs_since_print = 0;
                    }
                }
                Err(e) => {
                    eprintln!("   ❌ MSM failed: {}", e);
                    break;
                }
            }
        }
        
        let total_time = bench_start.elapsed().as_secs_f64();
        let total_proofs = latencies.len();
        
        if total_proofs == 0 {
            eprintln!("   ⚠️  No proofs completed!");
            continue;
        }
        
        // Calculate statistics
        let tps = total_proofs as f64 / total_time;
        let points_per_sec = (total_proofs * size) as f64 / total_time;
        
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let latency_min = latencies[0];
        let latency_max = latencies[latencies.len() - 1];
        let latency_p50 = percentile(&latencies, 50.0);
        let latency_p95 = percentile(&latencies, 95.0);
        let latency_p99 = percentile(&latencies, 99.0);
        
        let result = BenchResult {
            proof_size: size,
            description: description.to_string(),
            total_proofs,
            total_time_secs: total_time,
            tps,
            latency_min_ms: latency_min,
            latency_p50_ms: latency_p50,
            latency_p95_ms: latency_p95,
            latency_p99_ms: latency_p99,
            latency_max_ms: latency_max,
            points_per_sec,
            vram_peak_mb: vram_peak,
            vram_baseline_mb: baseline_vram,
        };
        
        println!();
        println!("   ┌─────────────────────────────────────────────────────────────────┐");
        println!("   │ RESULTS: 2^{} MSM ({})                             │", log_size, description.split(" - ").next().unwrap_or(""));
        println!("   ├─────────────────────────────────────────────────────────────────┤");
        println!("   │ Total Proofs:     {:>10}                                   │", total_proofs);
        println!("   │ Duration:         {:>10.2} s                                  │", total_time);
        println!("   │ Throughput:       {:>10.2} TPS                                │", tps);
        println!("   │ Points/sec:       {:>10.2e}                                │", points_per_sec);
        println!("   ├─────────────────────────────────────────────────────────────────┤");
        println!("   │ Latency (min):    {:>10.2} ms                                 │", latency_min);
        println!("   │ Latency (p50):    {:>10.2} ms                                 │", latency_p50);
        println!("   │ Latency (p95):    {:>10.2} ms                                 │", latency_p95);
        println!("   │ Latency (p99):    {:>10.2} ms                                 │", latency_p99);
        println!("   │ Latency (max):    {:>10.2} ms                                 │", latency_max);
        println!("   ├─────────────────────────────────────────────────────────────────┤");
        println!("   │ VRAM Peak:        {:>10.0} MB (+{:.0} MB)                      │", vram_peak, vram_peak - baseline_vram);
        println!("   └─────────────────────────────────────────────────────────────────┘");
        println!();
        
        all_results.push(result);
    }

    // Final Summary
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                              BENCHMARK SUMMARY                                       ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                                      ║");
    
    println!("║  ┌──────────┬──────────────┬────────────┬────────────┬────────────┬─────────────┐   ║");
    println!("║  │   Size   │  Total Proofs│    TPS     │  P50 (ms)  │  P99 (ms)  │  VRAM (MB)  │   ║");
    println!("║  ├──────────┼──────────────┼────────────┼────────────┼────────────┼─────────────┤   ║");
    
    for r in &all_results {
        let log_size = (r.proof_size as f64).log2() as u32;
        println!("║  │  2^{:<5} │ {:>12} │ {:>10.2} │ {:>10.2} │ {:>10.2} │ {:>11.0} │   ║",
                log_size, r.total_proofs, r.tps, r.latency_p50_ms, r.latency_p99_ms, r.vram_peak_mb);
    }
    
    println!("║  └──────────┴──────────────┴────────────┴────────────┴────────────┴─────────────┘   ║");
    println!("║                                                                                      ║");
    
    // Enterprise target assessment
    let target_tps = 88.0;
    let large_proof_result = all_results.iter().find(|r| r.proof_size == (1 << 18)); // 262K = Large
    
    if let Some(r) = large_proof_result {
        let margin = (r.tps / target_tps - 1.0) * 100.0;
        let status = if r.tps >= target_tps { "✅ PASS" } else { "❌ FAIL" };
        
        println!("║  🎯 ENTERPRISE TARGET (88 TPS @ 2^18):                                             ║");
        println!("║     Achieved: {:.2} TPS | {} | Margin: {:+.1}%                               ║", 
                r.tps, status, margin);
    }
    
    println!("║                                                                                      ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════╝");
    println!();
    
    // JSON output for automation
    println!("📊 JSON Results (for CI/automation):");
    println!("{{");
    println!("  \"gpu\": \"{}\",", gpu_name);
    println!("  \"vram_total_mb\": {:.0},", gpu_vram);
    println!("  \"target_tps\": {},", target_tps);
    println!("  \"results\": [");
    for (i, r) in all_results.iter().enumerate() {
        let comma = if i < all_results.len() - 1 { "," } else { "" };
        println!("    {{");
        println!("      \"proof_size\": {},", r.proof_size);
        println!("      \"log_size\": {},", (r.proof_size as f64).log2() as u32);
        println!("      \"description\": \"{}\",", r.description);
        println!("      \"total_proofs\": {},", r.total_proofs);
        println!("      \"tps\": {:.4},", r.tps);
        println!("      \"latency_p50_ms\": {:.4},", r.latency_p50_ms);
        println!("      \"latency_p99_ms\": {:.4},", r.latency_p99_ms);
        println!("      \"points_per_sec\": {:.0},", r.points_per_sec);
        println!("      \"vram_peak_mb\": {:.0}", r.vram_peak_mb);
        println!("    }}{}", comma);
    }
    println!("  ]");
    println!("}}");
}
