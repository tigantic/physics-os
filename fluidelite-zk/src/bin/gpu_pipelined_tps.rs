//! GPU Pipelined TPS Benchmark - Addressing the QTT Memory Bottleneck
//!
//! This benchmark implements ASYNC PIPELINING to solve the CPU/GPU ping-pong:
//!   - While GPU computes MSM(N), CPU prepares scalars for MSM(N+1)
//!   - Double-buffered scalar pools eliminate transfer latency
//!   - Async CUDA streams keep GPU fed continuously
//!
//! Target: Sustained 88 TPS @ 2^18 (enterprise DeFi workload)

#![allow(unused_imports)]

use fluidelite_zk::gpu::GpuAccelerator;
use icicle_bn254::curve::{G1Affine, G1Projective, ScalarField};
use icicle_core::msm::{msm, MSMConfig};
use icicle_core::traits::GenerateRandom;
use icicle_runtime::memory::{DeviceVec, HostSlice};
use icicle_runtime::stream::IcicleStream;
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

fn main() {
    let (gpu_name, vram_total) = get_gpu_info();
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║        🚀 PIPELINED TPS BENCHMARK - SOLVING CPU/GPU PING-PONG 🚀             ║");
    println!("║                                                                              ║");
    println!("║   Problem: Serial execution → GPU waits for CPU → 23 TPS @ 2^18            ║");
    println!("║   Solution: Async pipelining with double buffering → Target 88 TPS         ║");
    println!("║                                                                              ║");
    println!("║   Pipeline Stages:                                                           ║");
    println!("║     [CPU: Prep N+1] ──▶ [Transfer N] ──▶ [GPU: Compute N-1] ──▶ [Result]   ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    
    // Initialize GPU
    let _gpu = GpuAccelerator::new().expect("GPU init failed");
    
    println!("║  GPU: {:60} ║", gpu_name);
    println!("║  VRAM: {} MB                                                             ║", vram_total);
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");
    
    // Create multiple CUDA streams for pipelining
    let num_streams = 3;
    let mut streams: Vec<IcicleStream> = Vec::new();
    for _ in 0..num_streams {
        streams.push(IcicleStream::create().expect("Failed to create stream"));
    }
    
    println!("🔥 Created {} CUDA streams for async pipelining\n", streams.len());
    
    // Test configurations
    let test_sizes: Vec<(u32, &str)> = vec![
        (16, "Token Transfer"),
        (18, "DeFi Swap"),
        (20, "Complex DeFi"),
    ];
    
    // Optimal c values from previous benchmark
    let optimal_c: Vec<i32> = vec![12, 14, 0]; // 0 = auto for 2^20
    
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    
    // PHASE 1: Lock generator points in VRAM (weight caching)
    println!("🔒 PHASE 1: WEIGHT CACHING - Locking generator points in VRAM...\n");
    
    let mut cached_points: Vec<DeviceVec<G1Affine>> = Vec::new();
    let mut total_cached_mb = 0usize;
    
    for (log_size, desc) in &test_sizes {
        let size = 1usize << log_size;
        let mb = size * 64 / (1024 * 1024);
        print!("   📌 Caching 2^{} {} ({} MB)...", log_size, desc, mb);
        std::io::stdout().flush().unwrap();
        
        let points: Vec<G1Affine> = G1Affine::generate_random(size);
        
        let mut points_d = DeviceVec::<G1Affine>::device_malloc(size).expect("VRAM alloc failed");
        points_d.copy_from_host(HostSlice::from_slice(&points)).expect("Copy failed");
        
        cached_points.push(points_d);
        total_cached_mb += mb;
        println!(" ✓ locked");
    }
    
    println!("\n   💾 Total weight cache: {} MB\n", total_cached_mb);
    
    // PHASE 2: Create double-buffered scalar pools
    println!("📌 PHASE 2: DOUBLE BUFFERING - Creating parallel scalar pools...\n");
    
    const NUM_BUFFERS: usize = 4; // Quad-buffering for max throughput
    let mut scalar_buffers: Vec<Vec<Vec<ScalarField>>> = Vec::new();
    
    for (log_size, desc) in &test_sizes {
        let size = 1usize << log_size;
        print!("   📌 Pre-generating {} x {} scalar buffers...", NUM_BUFFERS, desc);
        
        let buffers: Vec<Vec<ScalarField>> = (0..NUM_BUFFERS)
            .map(|_| ScalarField::generate_random(size))
            .collect();
        
        scalar_buffers.push(buffers);
        println!(" ✓");
    }
    println!();
    
    // PHASE 3: Results storage
    let mut results_summary: Vec<(u32, &str, f64, f64, i32)> = Vec::new();
    
    // PHASE 4: Run pipelined benchmarks
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    println!("🚀 PHASE 3: ASYNC PIPELINED BENCHMARK\n");
    
    for (idx, ((log_size, desc), c_val)) in test_sizes.iter().zip(optimal_c.iter()).enumerate() {
        let size = 1usize << log_size;
        
        println!("📦 2^{} = {} constraints | {}", log_size, size, desc);
        println!("   Using optimal c={} from previous benchmark", if *c_val == 0 { "auto".to_string() } else { c_val.to_string() });
        println!();
        
        // Run serial baseline first
        println!("   📊 Serial baseline (for comparison):");
        let serial_tps = run_serial_benchmark(&cached_points[idx], &scalar_buffers[idx], *c_val, 5.0);
        println!("      Serial TPS: {:.2}", serial_tps);
        
        // Run pipelined async benchmark
        println!("\n   📊 Pipelined async benchmark:");
        let (pipelined_tps, latency_p50) = run_pipelined_benchmark(
            &cached_points[idx], 
            &scalar_buffers[idx], 
            &mut streams,
            *c_val, 
            10.0
        );
        
        let speedup = pipelined_tps / serial_tps;
        println!("      Pipelined TPS: {:.2} ({:.1}x speedup)", pipelined_tps, speedup);
        println!("      P50 Latency: {:.2}ms", latency_p50);
        
        results_summary.push((*log_size, *desc, pipelined_tps, latency_p50, *c_val));
        
        println!();
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    }
    
    // Extended 2^18 stress test
    println!("🎯 PHASE 4: EXTENDED 2^18 PIPELINED TEST (30 seconds)\n");
    
    let (sustained_tps, sustained_p50) = run_pipelined_benchmark(
        &cached_points[1], // 2^18
        &scalar_buffers[1],
        &mut streams,
        optimal_c[1],
        30.0
    );
    
    let gap_percent = ((88.0 - sustained_tps) / 88.0 * 100.0).max(0.0);
    let target_met = sustained_tps >= 88.0;
    
    println!("   ┌─────────────────────────────────────────────────────────────────┐");
    println!("   │ 2^18 DeFi Swap - PIPELINED SUSTAINED TPS                        │");
    println!("   ├─────────────────────────────────────────────────────────────────┤");
    println!("   │ 🎯 TPS (sustained):        {:6.2}                              │", sustained_tps);
    println!("   │ 📊 P50 Latency:           {:6.2} ms                            │", sustained_p50);
    println!("   │ 🎯 Enterprise Target:       88.00 TPS                           │");
    println!("   │ 📈 Gap to target:          {:5.1}%                              │", gap_percent);
    println!("   │ {} │", if target_met { "✅ TARGET MET!                                                 " } else { "❌ Still working toward target                                  " });
    println!("   └─────────────────────────────────────────────────────────────────┘\n");
    
    // Final summary
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                      PIPELINED TPS FINAL SUMMARY                             ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                              ║");
    println!("║  Pipeline Configuration:                                                     ║");
    println!("║    ✅ CUDA Streams: {}                                                        ║", streams.len());
    println!("║    ✅ Scalar Buffers: {} (quad-buffered)                                      ║", NUM_BUFFERS);
    println!("║    ✅ Weight Cache: {} MB locked in VRAM                                    ║", total_cached_mb);
    println!("║    ✅ Async Execution: Enabled                                               ║");
    println!("║                                                                              ║");
    println!("║  ┌────────┬──────────────────┬──────────┬──────────┬─────────┐              ║");
    println!("║  │  Size  │    Use Case      │   TPS    │  P50 ms  │  Best c │              ║");
    println!("║  ├────────┼──────────────────┼──────────┼──────────┼─────────┤              ║");
    
    for (log_size, desc, tps, p50, c) in &results_summary {
        let c_str = if *c == 0 { "auto".to_string() } else { c.to_string() };
        let short_desc = if desc.len() > 16 { &desc[..16] } else { desc };
        println!("║  │  2^{:2}  │ {:16} │ {:8.2} │ {:8.2} │ {:>7} │              ║", 
                 log_size, short_desc, tps, p50, c_str);
    }
    
    println!("║  └────────┴──────────────────┴──────────┴──────────┴─────────┘              ║");
    println!("║                                                                              ║");
    
    if target_met {
        println!("║  ✅ ENTERPRISE TARGET: 88 TPS @ 2^18 → ACHIEVED {:.2} TPS             ║", sustained_tps);
    } else {
        println!("║  ⚠️  ENTERPRISE TARGET: 88 TPS @ 2^18 → Achieved {:.2} TPS ({:.1}% gap)      ║", sustained_tps, gap_percent);
    }
    
    println!("║                                                                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");
    
    // Output JSON
    println!("📊 JSON:");
    println!("{{");
    println!("  \"benchmark\": \"pipelined-tps\",");
    println!("  \"gpu\": \"{}\",", gpu_name);
    println!("  \"pipeline\": {{");
    println!("    \"cuda_streams\": {},", num_streams);
    println!("    \"scalar_buffers\": {},", NUM_BUFFERS);
    println!("    \"async_execution\": true");
    println!("  }},");
    println!("  \"cached_vram_mb\": {},", total_cached_mb);
    println!("  \"results\": [");
    for (i, (log_size, desc, tps, p50, c)) in results_summary.iter().enumerate() {
        let comma = if i < results_summary.len() - 1 { "," } else { "" };
        println!("    {{ \"size\": \"2^{}\", \"desc\": \"{}\", \"tps\": {:.2}, \"latency_p50_ms\": {:.2}, \"best_c\": {} }}{}", 
                 log_size, desc, tps, p50, c, comma);
    }
    println!("  ],");
    println!("  \"sustained_2_18\": {{");
    println!("    \"duration_sec\": 30,");
    println!("    \"tps\": {:.2},", sustained_tps);
    println!("    \"p50_ms\": {:.2},", sustained_p50);
    println!("    \"target_met\": {}", target_met);
    println!("  }}");
    println!("}}");
    
    // Cleanup streams
    for mut stream in streams {
        stream.destroy().ok();
    }
}

#[cfg(feature = "gpu")]
fn run_serial_benchmark(
    points_d: &DeviceVec<G1Affine>,
    scalar_buffers: &[Vec<ScalarField>],
    c_val: i32,
    duration_secs: f64,
) -> f64 {
    let mut cfg = MSMConfig::default();
    if c_val > 0 {
        cfg.c = c_val;
    }
    cfg.is_async = false; // Serial!
    
    let mut result_d = DeviceVec::<G1Projective>::device_malloc(1).expect("Result alloc failed");
    
    let start = Instant::now();
    let mut count = 0usize;
    let mut buffer_idx = 0;
    
    while start.elapsed().as_secs_f64() < duration_secs {
        let scalars = &scalar_buffers[buffer_idx % scalar_buffers.len()];
        msm(
            HostSlice::from_slice(scalars),
            &points_d[..],
            &cfg,
            &mut result_d[..],
        ).expect("MSM failed");
        
        count += 1;
        buffer_idx += 1;
    }
    
    let elapsed = start.elapsed().as_secs_f64();
    count as f64 / elapsed
}

#[cfg(feature = "gpu")]
fn run_pipelined_benchmark(
    points_d: &DeviceVec<G1Affine>,
    scalar_buffers: &[Vec<ScalarField>],
    streams: &mut [IcicleStream],
    c_val: i32,
    duration_secs: f64,
) -> (f64, f64) {
    let num_streams = streams.len();
    
    // Create result buffers - one per stream
    let mut results_d: Vec<DeviceVec<G1Projective>> = (0..num_streams)
        .map(|_| DeviceVec::<G1Projective>::device_malloc(1).expect("Result alloc failed"))
        .collect();
    
    // MSM configs - one per stream
    let mut configs: Vec<MSMConfig> = (0..num_streams)
        .map(|i| {
            let mut cfg = MSMConfig::default();
            if c_val > 0 {
                cfg.c = c_val;
            }
            cfg.stream_handle = *streams[i];
            cfg.is_async = true; // ASYNC execution!
            cfg
        })
        .collect();
    
    let mut latencies: Vec<f64> = Vec::new();
    let mut buffer_idx = 0usize;
    let mut completed = 0usize;
    let mut in_flight: Vec<Option<Instant>> = vec![None; num_streams];
    
    let start = Instant::now();
    let mut last_report = Instant::now();
    
    // Prime the pipeline - launch initial MSMs on all streams
    for (i, stream) in streams.iter().enumerate() {
        let scalars = &scalar_buffers[buffer_idx % scalar_buffers.len()];
        
        msm(
            HostSlice::from_slice(scalars),
            &points_d[..],
            &configs[i],
            &mut results_d[i][..],
        ).expect("MSM launch failed");
        
        in_flight[i] = Some(Instant::now());
        buffer_idx += 1;
    }
    
    // Main pipeline loop
    while start.elapsed().as_secs_f64() < duration_secs {
        // Round-robin through streams
        for i in 0..num_streams {
            // Synchronize this stream (blocks until MSM complete)
            streams[i].synchronize().expect("Sync failed");
            
            // Record latency
            if let Some(launch_time) = in_flight[i].take() {
                let lat = launch_time.elapsed().as_secs_f64() * 1000.0;
                latencies.push(lat);
                completed += 1;
            }
            
            // Immediately launch next MSM on this stream (keeps GPU fed)
            if start.elapsed().as_secs_f64() < duration_secs {
                let scalars = &scalar_buffers[buffer_idx % scalar_buffers.len()];
                
                msm(
                    HostSlice::from_slice(scalars),
                    &points_d[..],
                    &configs[i],
                    &mut results_d[i][..],
                ).expect("MSM launch failed");
                
                in_flight[i] = Some(Instant::now());
                buffer_idx += 1;
            }
        }
        
        // Progress report every 5 seconds
        if last_report.elapsed().as_secs_f64() >= 5.0 {
            let elapsed = start.elapsed().as_secs_f64();
            let tps = completed as f64 / elapsed;
            let avg_lat = if !latencies.is_empty() {
                latencies.iter().sum::<f64>() / latencies.len() as f64
            } else { 0.0 };
            
            println!("   [{:5.1}s] {:4} proofs | {:6.2} TPS | {:5.2}ms avg", 
                     elapsed, completed, tps, avg_lat);
            last_report = Instant::now();
        }
    }
    
    // Drain pipeline - wait for all in-flight MSMs
    for i in 0..num_streams {
        if in_flight[i].is_some() {
            streams[i].synchronize().expect("Final sync failed");
            if let Some(launch_time) = in_flight[i].take() {
                let lat = launch_time.elapsed().as_secs_f64() * 1000.0;
                latencies.push(lat);
                completed += 1;
            }
        }
    }
    
    let total_elapsed = start.elapsed().as_secs_f64();
    let tps = completed as f64 / total_elapsed;
    
    // Calculate P50
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = if !latencies.is_empty() {
        latencies[latencies.len() / 2]
    } else { 0.0 };
    
    (tps, p50)
}
