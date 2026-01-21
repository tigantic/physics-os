//! GPU VRAM Stress Test - Load data INTO VRAM, measure real TPS
//!
//! The previous tests used HostSlice which streams from system RAM.
//! This test uses DeviceSlice to pre-load data into GPU VRAM.

use fluidelite_zk::gpu::GpuAccelerator;
use icicle_bn254::curve::{G1Affine, G1Projective, ScalarField};
use icicle_core::msm::{msm, MSMConfig};
use icicle_core::traits::GenerateRandom;
use icicle_runtime::memory::{DeviceVec, HostSlice};
use std::io::Write;
use std::process::Command;
use std::time::Instant;

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

fn main() {
    let (gpu_name, vram_total) = get_gpu_info();
    let baseline_vram = get_vram_mb();
    
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║           🎯 GPU VRAM STRESS TEST - DATA IN VRAM, REAL TPS 🎯                ║");
    println!("║                  \"Loading data INTO GPU memory, not streaming\"               ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    
    // Initialize GPU
    let _gpu = GpuAccelerator::new().expect("GPU init failed");
    
    println!("║  GPU: {:60} ║", gpu_name);
    println!("║  VRAM Total: {} MB | Baseline: {} MB                                     ║", vram_total, baseline_vram);
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    // Test realistic proof sizes with TPS focus
    // A real ZK proof typically uses 2^16 to 2^20 MSM
    let test_configs: Vec<(u32, &str, usize)> = vec![
        (16, "Token Transfer (65K constraints)", 100),   // Run 100 proofs
        (18, "DeFi Swap (262K constraints)", 50),        // Run 50 proofs  
        (20, "ML Inference (1M constraints)", 20),       // Run 20 proofs
        (22, "Full Model (4M constraints)", 10),         // Run 10 proofs
    ];

    println!("🎯 GOAL: Measure REAL TPS with data pre-loaded in VRAM\n");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut results: Vec<(u32, String, usize, f64, f64, u64)> = Vec::new();

    for (log_size, desc, num_proofs) in &test_configs {
        let size = 1usize << log_size;
        let data_mb = (size * 96) / (1024 * 1024); // 32B scalar + 64B point
        
        println!("📦 2^{} = {} points | {} proofs to run", log_size, size, num_proofs);
        println!("   {}", desc);
        println!("   Data size: {} MB per proof\n", data_mb);

        // Step 1: Generate data on host
        print!("   ⏳ Generating {} points on CPU... ", size);
        std::io::stdout().flush().unwrap();
        
        let gen_start = Instant::now();
        let host_scalars = ScalarField::generate_random(size);
        let host_points = G1Affine::generate_random(size);
        println!("done in {:.2}s", gen_start.elapsed().as_secs_f64());

        // Step 2: Copy to GPU VRAM
        print!("   📤 Uploading to GPU VRAM... ");
        std::io::stdout().flush().unwrap();
        
        let upload_start = Instant::now();
        let mut gpu_scalars = DeviceVec::<ScalarField>::device_malloc(size).expect("Failed to alloc scalars on GPU");
        let mut gpu_points = DeviceVec::<G1Affine>::device_malloc(size).expect("Failed to alloc points on GPU");
        
        gpu_scalars.copy_from_host(HostSlice::from_slice(&host_scalars)).expect("Failed to copy scalars");
        gpu_points.copy_from_host(HostSlice::from_slice(&host_points)).expect("Failed to copy points");
        
        // Force sync
        icicle_runtime::stream::IcicleStream::default().synchronize().ok();
        
        let upload_time = upload_start.elapsed();
        let vram_after_upload = get_vram_mb();
        println!("done in {:.2}s", upload_time.as_secs_f64());
        println!("   📊 VRAM after upload: {} MB (+{} MB)", vram_after_upload, vram_after_upload - baseline_vram);

        // Step 3: Warmup
        println!("   🔥 Warming up (3 iterations)...");
        let cfg = MSMConfig::default();
        let mut result = DeviceVec::<G1Projective>::device_malloc(1).expect("Failed to alloc result");
        
        for _ in 0..3 {
            msm(
                &gpu_scalars[..],
                &gpu_points[..],
                &cfg,
                &mut result[..],
            ).expect("Warmup MSM failed");
        }
        icicle_runtime::stream::IcicleStream::default().synchronize().ok();

        // Step 4: Timed run - measure TPS
        println!("   🚀 Running {} MSM proofs from VRAM...\n", num_proofs);
        
        let vram_during = get_vram_mb();
        let bench_start = Instant::now();
        
        for i in 0..*num_proofs {
            msm(
                &gpu_scalars[..],
                &gpu_points[..],
                &cfg,
                &mut result[..],
            ).expect("MSM failed");
            
            // Progress every 10 or at end
            if (i + 1) % 10 == 0 || i + 1 == *num_proofs {
                icicle_runtime::stream::IcicleStream::default().synchronize().ok();
                let elapsed = bench_start.elapsed().as_secs_f64();
                let current_tps = (i + 1) as f64 / elapsed;
                print!("\r      [{:3}/{}] {:.1} TPS | {:.2}s elapsed", i + 1, num_proofs, current_tps, elapsed);
                std::io::stdout().flush().unwrap();
            }
        }
        
        icicle_runtime::stream::IcicleStream::default().synchronize().ok();
        let total_time = bench_start.elapsed().as_secs_f64();
        let tps = *num_proofs as f64 / total_time;
        let avg_latency_ms = (total_time * 1000.0) / *num_proofs as f64;
        
        println!("\n");
        println!("   ┌────────────────────────────────────────────────────────────────┐");
        println!("   │ 2^{:2} RESULTS: {}                           │", log_size, desc);
        println!("   ├────────────────────────────────────────────────────────────────┤");
        println!("   │ 🎯 TPS (Proofs/sec):     {:8.2}                              │", tps);
        println!("   │ ⏱️  Avg Latency:          {:8.2} ms                           │", avg_latency_ms);
        println!("   │ 📊 Total Proofs:         {:8}                              │", num_proofs);
        println!("   │ ⏳ Total Time:           {:8.2} s                            │", total_time);
        println!("   ├────────────────────────────────────────────────────────────────┤");
        println!("   │ 💾 VRAM During Test:     {:8} MB ({:.1}% of {})           │", 
                 vram_during, (vram_during as f64 / vram_total as f64) * 100.0, vram_total);
        println!("   └────────────────────────────────────────────────────────────────┘\n");

        results.push((*log_size, desc.to_string(), *num_proofs, tps, avg_latency_ms, vram_during));

        // Cleanup GPU memory
        drop(gpu_scalars);
        drop(gpu_points);
        drop(result);
        
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    }

    // Final summary with TPS focus
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                        TPS BENCHMARK SUMMARY                                 ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                              ║");
    println!("║  ┌────────┬─────────────────────────┬──────────┬────────────┬──────────────┐ ║");
    println!("║  │  Size  │      Use Case           │   TPS    │  Latency   │  VRAM Used   │ ║");
    println!("║  ├────────┼─────────────────────────┼──────────┼────────────┼──────────────┤ ║");
    
    for (log_size, desc, _proofs, tps, latency, vram) in &results {
        let short_desc = if desc.len() > 23 { &desc[..23] } else { desc };
        println!("║  │  2^{:2}  │ {:23} │ {:8.2} │ {:8.2}ms │ {:>6} MB    │ ║",
                 log_size, short_desc, tps, latency, vram);
    }
    
    println!("║  └────────┴─────────────────────────┴──────────┴────────────┴──────────────┘ ║");
    println!("║                                                                              ║");
    
    // Enterprise target check (88 TPS @ 2^18)
    if let Some((_, _, _, tps, _, _)) = results.iter().find(|(log, _, _, _, _, _)| *log == 18) {
        let target = 88.0;
        let margin = (tps / target - 1.0) * 100.0;
        if *tps >= target {
            println!("║  ✅ ENTERPRISE TARGET: 88 TPS @ 2^18 → Achieved {:.2} TPS (+{:.1}%)       ║", tps, margin);
        } else {
            println!("║  ❌ ENTERPRISE TARGET: 88 TPS @ 2^18 → Achieved {:.2} TPS ({:.1}%)        ║", tps, margin);
        }
    }
    
    println!("║                                                                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    // JSON output
    println!("📊 JSON Results:");
    println!("{{");
    println!("  \"gpu\": \"{}\",", gpu_name);
    println!("  \"vram_total_mb\": {},", vram_total);
    println!("  \"results\": [");
    for (i, (log_size, desc, proofs, tps, latency, vram)) in results.iter().enumerate() {
        let comma = if i < results.len() - 1 { "," } else { "" };
        println!("    {{ \"size\": \"2^{}\", \"desc\": \"{}\", \"proofs\": {}, \"tps\": {:.2}, \"latency_ms\": {:.2}, \"vram_mb\": {} }}{}",
                 log_size, desc, proofs, tps, latency, vram, comma);
    }
    println!("  ]");
    println!("}}");
}
