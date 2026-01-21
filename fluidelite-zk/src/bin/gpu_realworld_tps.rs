//! Real-World TPS Benchmark - Simulates actual production workflow
//!
//! Flow per proof:
//! 1. Generate witness data (CPU) - simulates transaction processing
//! 2. Upload to GPU
//! 3. Compute MSM
//! 4. Sync result
//! 5. Repeat
//!
//! This measures TRUE end-to-end TPS including all overheads.

use fluidelite_zk::gpu::GpuAccelerator;
use icicle_bn254::curve::{G1Affine, G1Projective, ScalarField};
use icicle_core::msm::{msm, MSMConfig};
use icicle_core::traits::GenerateRandom;
use icicle_runtime::memory::{DeviceVec, HostSlice};
use std::io::Write;
use std::process::Command;
use std::time::{Duration, Instant};

fn get_vram_mb() -> u64 {
    Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

fn get_gpu_info() -> (String, u64) {
    Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .and_then(|s| {
            let parts: Vec<&str> = s.trim().split(',').collect();
            if parts.len() >= 2 {
                let name = parts[0].trim().to_string();
                let total = parts[1].trim().parse::<u64>().ok()?;
                Some((name, total))
            } else {
                None
            }
        })
        .unwrap_or(("Unknown GPU".to_string(), 0))
}

/// Simulates generating witness data for a transaction
/// In real life this would be hashing, signature verification, state reads, etc.
fn generate_witness(size: usize) -> (Vec<ScalarField>, Vec<G1Affine>) {
    let scalars = ScalarField::generate_random(size);
    let points = G1Affine::generate_random(size);
    (scalars, points)
}

/// Process a single proof end-to-end (real workflow)
fn process_single_proof(
    size: usize,
    cfg: &MSMConfig,
    gpu_scalars: &mut DeviceVec<ScalarField>,
    gpu_points: &mut DeviceVec<G1Affine>,
    gpu_result: &mut DeviceVec<G1Projective>,
) -> Duration {
    let start = Instant::now();
    
    // Step 1: Generate witness (CPU work - simulates tx processing)
    let (scalars, points) = generate_witness(size);
    
    // Step 2: Upload to GPU
    gpu_scalars.copy_from_host(HostSlice::from_slice(&scalars)).expect("upload scalars");
    gpu_points.copy_from_host(HostSlice::from_slice(&points)).expect("upload points");
    
    // Step 3: Compute MSM on GPU
    msm(
        &gpu_scalars[..],
        &gpu_points[..],
        cfg,
        &mut gpu_result[..],
    ).expect("MSM failed");
    
    // Step 4: Sync (ensure GPU is done)
    icicle_runtime::stream::IcicleStream::default().synchronize().ok();
    
    start.elapsed()
}

fn main() {
    let (gpu_name, vram_total) = get_gpu_info();
    
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║        🎯 REAL-WORLD TPS BENCHMARK - END-TO-END PRODUCTION FLOW 🎯           ║");
    println!("║                                                                              ║");
    println!("║   Per proof: Generate Witness → Upload → GPU MSM → Sync → Next              ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    
    let _gpu = GpuAccelerator::new().expect("GPU init failed");
    
    println!("║  GPU: {:60} ║", gpu_name);
    println!("║  VRAM: {} MB                                                             ║", vram_total);
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    // Production proof sizes
    let test_configs: Vec<(u32, &str, u64)> = vec![
        (16, "Token Transfer", 10),      // Run for 10 seconds
        (18, "DeFi Swap", 10),            // Run for 10 seconds  
        (20, "Complex DeFi / ML", 10),    // Run for 10 seconds
    ];

    println!("📋 Test: Continuous proof generation for {} seconds each\n", 10);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut results: Vec<(u32, String, usize, f64, f64, f64, f64, u64)> = Vec::new();
    let cfg = MSMConfig::default();

    for (log_size, desc, duration_secs) in &test_configs {
        let size = 1usize << log_size;
        
        println!("📦 2^{} = {} constraints | {}", log_size, size, desc);
        println!("   Running continuous proofs for {} seconds...\n", duration_secs);

        // Pre-allocate GPU buffers (reused each proof - this IS realistic)
        let mut gpu_scalars = DeviceVec::<ScalarField>::device_malloc(size).expect("alloc scalars");
        let mut gpu_points = DeviceVec::<G1Affine>::device_malloc(size).expect("alloc points");
        let mut gpu_result = DeviceVec::<G1Projective>::device_malloc(1).expect("alloc result");

        // Warmup (3 proofs)
        print!("   🔥 Warmup... ");
        std::io::stdout().flush().unwrap();
        for _ in 0..3 {
            process_single_proof(size, &cfg, &mut gpu_scalars, &mut gpu_points, &mut gpu_result);
        }
        println!("done");

        // Timed run
        let mut latencies: Vec<f64> = Vec::new();
        let mut peak_vram: u64 = 0;
        let bench_start = Instant::now();
        let deadline = Duration::from_secs(*duration_secs);
        
        let mut last_print = Instant::now();
        
        while bench_start.elapsed() < deadline {
            let proof_time = process_single_proof(size, &cfg, &mut gpu_scalars, &mut gpu_points, &mut gpu_result);
            latencies.push(proof_time.as_secs_f64() * 1000.0); // ms
            
            // Sample VRAM
            let vram = get_vram_mb();
            peak_vram = peak_vram.max(vram);
            
            // Progress every 2 seconds
            if last_print.elapsed() > Duration::from_secs(2) {
                let elapsed = bench_start.elapsed().as_secs_f64();
                let tps = latencies.len() as f64 / elapsed;
                let avg_lat = latencies.iter().sum::<f64>() / latencies.len() as f64;
                print!("\r   [{:5.1}s] {:4} proofs | {:6.1} TPS | {:6.2}ms avg | VRAM: {} MB   ", 
                       elapsed, latencies.len(), tps, avg_lat, vram);
                std::io::stdout().flush().unwrap();
                last_print = Instant::now();
            }
        }
        
        let total_time = bench_start.elapsed().as_secs_f64();
        let total_proofs = latencies.len();
        let tps = total_proofs as f64 / total_time;
        
        // Calculate latency stats
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let lat_min = latencies.first().copied().unwrap_or(0.0);
        let lat_p50 = latencies.get(latencies.len() / 2).copied().unwrap_or(0.0);
        let lat_p99 = latencies.get(latencies.len() * 99 / 100).copied().unwrap_or(0.0);
        let lat_max = latencies.last().copied().unwrap_or(0.0);
        let lat_avg = latencies.iter().sum::<f64>() / latencies.len() as f64;

        println!("\n");
        println!("   ┌────────────────────────────────────────────────────────────────┐");
        println!("   │ 2^{:2} {} - END-TO-END RESULTS                      │", log_size, desc);
        println!("   ├────────────────────────────────────────────────────────────────┤");
        println!("   │ 🎯 TPS (end-to-end):    {:8.2}                              │", tps);
        println!("   │ 📊 Total Proofs:        {:8}                              │", total_proofs);
        println!("   │ ⏳ Total Time:          {:8.2} s                            │", total_time);
        println!("   ├────────────────────────────────────────────────────────────────┤");
        println!("   │ ⏱️  Latency min:         {:8.2} ms                           │", lat_min);
        println!("   │ ⏱️  Latency p50:         {:8.2} ms                           │", lat_p50);
        println!("   │ ⏱️  Latency p99:         {:8.2} ms                           │", lat_p99);
        println!("   │ ⏱️  Latency max:         {:8.2} ms                           │", lat_max);
        println!("   ├────────────────────────────────────────────────────────────────┤");
        println!("   │ 💾 Peak VRAM:           {:8} MB ({:.1}%)                   │", 
                 peak_vram, (peak_vram as f64 / vram_total as f64) * 100.0);
        println!("   └────────────────────────────────────────────────────────────────┘\n");

        results.push((*log_size, desc.to_string(), total_proofs, tps, lat_avg, lat_p50, lat_p99, peak_vram));

        // Cleanup
        drop(gpu_scalars);
        drop(gpu_points);
        drop(gpu_result);
        
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    }

    // Final summary
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    REAL-WORLD TPS SUMMARY                                    ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                              ║");
    println!("║  Flow: CPU witness gen → Upload → GPU MSM → Sync (per proof)                ║");
    println!("║                                                                              ║");
    println!("║  ┌────────┬──────────────────┬──────────┬──────────┬──────────┬───────────┐  ║");
    println!("║  │  Size  │    Use Case      │   TPS    │  P50 ms  │  P99 ms  │ Peak VRAM │  ║");
    println!("║  ├────────┼──────────────────┼──────────┼──────────┼──────────┼───────────┤  ║");
    
    for (log_size, desc, _proofs, tps, _avg, p50, p99, vram) in &results {
        let short_desc = if desc.len() > 16 { &desc[..16] } else { desc };
        println!("║  │  2^{:2}  │ {:16} │ {:8.2} │ {:8.2} │ {:8.2} │ {:5} MB  │  ║",
                 log_size, short_desc, tps, p50, p99, vram);
    }
    
    println!("║  └────────┴──────────────────┴──────────┴──────────┴──────────┴───────────┘  ║");
    println!("║                                                                              ║");
    
    // Enterprise target
    if let Some((_, _, _, tps, _, _, _, _)) = results.iter().find(|(log, _, _, _, _, _, _, _)| *log == 18) {
        let target = 88.0;
        if *tps >= target {
            println!("║  ✅ ENTERPRISE TARGET: 88 TPS @ 2^18 → Achieved {:.2} TPS              ║", tps);
        } else {
            println!("║  ❌ ENTERPRISE TARGET: 88 TPS @ 2^18 → Achieved {:.2} TPS              ║", tps);
        }
    }
    
    println!("║                                                                              ║");
    println!("║  Note: These are TRUE end-to-end numbers including CPU witness generation,  ║");
    println!("║        PCIe upload, GPU compute, and synchronization overhead.              ║");
    println!("║                                                                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    // JSON
    println!("📊 JSON:");
    println!("{{");
    println!("  \"benchmark\": \"real-world-e2e\",");
    println!("  \"gpu\": \"{}\",", gpu_name);
    println!("  \"results\": [");
    for (i, (log_size, desc, proofs, tps, avg, p50, p99, vram)) in results.iter().enumerate() {
        let comma = if i < results.len() - 1 { "," } else { "" };
        println!("    {{ \"size\": \"2^{}\", \"desc\": \"{}\", \"proofs\": {}, \"tps\": {:.2}, \"latency_avg_ms\": {:.2}, \"latency_p50_ms\": {:.2}, \"latency_p99_ms\": {:.2}, \"vram_mb\": {} }}{}",
                 log_size, desc, proofs, tps, avg, p50, p99, vram, comma);
    }
    println!("  ]");
    println!("}}");
}
