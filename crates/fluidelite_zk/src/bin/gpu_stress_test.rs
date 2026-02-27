//! GPU Stress Test: "The VRAM Ladder" - GPU Edition
//!
//! Tests GPU-accelerated MSM/NTT operations with real VRAM usage.
//!
//! # Usage
//!
//! ```bash
//! export ICICLE_BACKEND_INSTALL_DIR=/opt/icicle/lib
//! cargo run --release --bin gpu-stress-test --features gpu
//! ```

use std::process::Command;
use std::time::Instant;
use fluidelite_zk::gpu::GpuAccelerator;

/// Get GPU VRAM usage via nvidia-smi
fn get_vram_mb() -> f64 {
    Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(0.0)
}

/// Get GPU info
fn get_gpu_info() -> Option<(String, f64)> {
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
}

/// Stress test result
#[derive(Debug, Clone)]
struct StressResult {
    size: usize,
    msm_time_ms: f64,
    points_per_sec: f64,
    vram_used_mb: f64,
    vram_delta_mb: f64,
}

fn print_header(gpu_info: &Option<(String, f64)>) {
    println!();
    println!("╔════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                       FLUIDELITE GPU STRESS TEST                               ║");
    println!("║                          \"The VRAM Ladder\"                                     ║");
    println!("╠════════════════════════════════════════════════════════════════════════════════╣");
    if let Some((name, total)) = gpu_info {
        println!("║  GPU: {:60} {:>6.0} MB  ║", name, total);
    } else {
        println!("║  GPU: Not detected!                                                              ║");
    }
    println!("║  Mode: GPU Prover (Icicle CUDA) | MSM Stress Test                               ║");
    println!("╚════════════════════════════════════════════════════════════════════════════════╝");
    println!();
}

fn print_results_header() {
    println!("┌────────────┬────────────┬──────────────────┬──────────────┬──────────────┐");
    println!("│    Size    │  MSM (ms)  │   Points/sec     │  VRAM (MB)   │  VRAM Δ      │");
    println!("├────────────┼────────────┼──────────────────┼──────────────┼──────────────┤");
}

fn print_result_row(r: &StressResult) {
    println!(
        "│ 2^{:<2} {:>6} │ {:>10.2} │ {:>16.0} │ {:>12.0} │ {:>+12.0} │",
        (r.size as f64).log2() as u32,
        r.size,
        r.msm_time_ms,
        r.points_per_sec,
        r.vram_used_mb,
        r.vram_delta_mb,
    );
}

fn print_results_footer() {
    println!("└────────────┴────────────┴──────────────────┴──────────────┴──────────────┘");
}

fn main() {
    let gpu_info = get_gpu_info();
    print_header(&gpu_info);

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

    print_results_header();

    let mut results: Vec<StressResult> = Vec::new();
    let mut best_throughput = 0.0f64;
    let mut best_size = 0;
    let mut oom_size: Option<usize> = None;

    // Test sizes from 2^10 to 2^28 (push to the limit!)
    let sizes: Vec<usize> = (10..=28).map(|k| 1usize << k).collect();

    for size in sizes {
        let vram_before = get_vram_mb();

        // Run MSM benchmark 3 times and average
        let mut times = Vec::new();
        for _ in 0..3 {
            match gpu.benchmark_msm(size) {
                Ok(duration) => {
                    times.push(duration.as_secs_f64() * 1000.0);
                }
                Err(e) => {
                    println!(
                        "│ 2^{:<2} {:>6} │    OOM!    │       ---        │     ---      │     ---      │",
                        (size as f64).log2() as u32,
                        size
                    );
                    println!("│ Error: {} ", e);
                    oom_size = Some(size);
                    break;
                }
            }
        }

        if oom_size.is_some() {
            break;
        }

        let avg_time = times.iter().sum::<f64>() / times.len() as f64;
        let vram_after = get_vram_mb();
        let points_per_sec = size as f64 / (avg_time / 1000.0);

        let result = StressResult {
            size,
            msm_time_ms: avg_time,
            points_per_sec,
            vram_used_mb: vram_after,
            vram_delta_mb: vram_after - baseline_vram,
        };

        print_result_row(&result);

        if points_per_sec > best_throughput {
            best_throughput = points_per_sec;
            best_size = size;
        }

        // Check for VRAM danger zone (>90% of total)
        if let Some((_, total)) = &gpu_info {
            if vram_after > total * 0.9 {
                println!("│ ⚠️  DANGER ZONE: VRAM usage > 90%                                                │");
            }
        }

        results.push(result);
    }

    print_results_footer();
    println!();

    // Summary
    println!("╔═══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                                  SUMMARY                                      ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════╣");
    if let Some(oom) = oom_size {
        println!(
            "║  OOM Threshold: 2^{} ({} points)                                              ",
            (oom as f64).log2() as u32,
            oom
        );
    }
    println!(
        "║  Peak Throughput: {:>12.0} points/sec @ 2^{}                                   ",
        best_throughput,
        (best_size as f64).log2() as u32
    );
    
    // Calculate equivalent TPS assuming 1 proof = 1 MSM of size 2^16
    let equiv_tps = best_throughput / 65536.0;
    println!(
        "║  Equivalent TPS:  {:>12.2} (assuming 2^16 MSM per proof)                       ",
        equiv_tps
    );
    println!("╚═══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Whale metrics
    if !results.is_empty() {
        let best = results
            .iter()
            .max_by(|a, b| a.points_per_sec.partial_cmp(&b.points_per_sec).unwrap())
            .unwrap();
        
        println!("🐋 GPU WHALE METRICS (at optimal 2^{}):", (best.size as f64).log2() as u32);
        println!("   ├── MSM Time: {:.2} ms", best.msm_time_ms);
        println!("   ├── Throughput: {:.0} points/sec", best.points_per_sec);
        println!("   ├── VRAM Used: {:.0} MB", best.vram_used_mb);
        println!("   └── VRAM Delta: {:+.0} MB from baseline", best.vram_delta_mb);
        
        // Peak VRAM usage
        if let Some(peak) = results.iter().max_by(|a, b| a.vram_used_mb.partial_cmp(&b.vram_used_mb).unwrap()) {
            println!();
            println!("📈 Peak VRAM: {:.0} MB at 2^{} points", 
                peak.vram_used_mb,
                (peak.size as f64).log2() as u32
            );
        }
    }

    println!();
    println!("✅ GPU stress test completed!");
}
