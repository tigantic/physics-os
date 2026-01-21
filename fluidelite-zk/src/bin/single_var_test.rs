//! Single Variable Test - Controlled experiments with one variable change at a time
//!
//! Usage: single-var-test --k <size> --c <c_param> --duration <seconds> [--precompute <factor>] [--big-triangle]

use fluidelite_zk::gpu::GpuAccelerator;
use icicle_bn254::curve::{G1Affine, G1Projective, ScalarField};
use icicle_core::msm::{msm, precompute_bases, MSMConfig, CUDA_MSM_IS_BIG_TRIANGLE, CUDA_MSM_LARGE_BUCKET_FACTOR};
use icicle_core::traits::GenerateRandom;
use icicle_runtime::memory::{DeviceVec, HostSlice};
use icicle_runtime::stream::IcicleStream;
use std::time::{Duration, Instant};

fn main() {
    // Parse args
    let args: Vec<String> = std::env::args().collect();
    
    let mut k: u32 = 18;
    let mut c: i32 = 12;
    let mut duration_secs: u64 = 60;
    let mut precompute_factor: i32 = 0; // 0 = no precompute
    let mut big_triangle: bool = false;
    let mut large_bucket_factor: i32 = 0;
    
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--k" => { k = args[i + 1].parse().unwrap(); i += 2; }
            "--c" => { c = args[i + 1].parse().unwrap(); i += 2; }
            "--duration" => { duration_secs = args[i + 1].parse().unwrap(); i += 2; }
            "--precompute" => { precompute_factor = args[i + 1].parse().unwrap(); i += 2; }
            "--big-triangle" => { big_triangle = true; i += 1; }
            "--large-bucket" => { large_bucket_factor = args[i + 1].parse().unwrap(); i += 2; }
            _ => { i += 1; }
        }
    }
    
    let size = 1usize << k;
    
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║            SINGLE VARIABLE TEST - CONTROLLED EXPERIMENT        ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║  Size:      2^{} = {} points", k, size);
    println!("║  c-param:   {} ({} buckets)", c, 1i32 << c);
    println!("║  Duration:  {} seconds", duration_secs);
    println!("║  Precompute: {}", if precompute_factor > 0 { format!("factor={}", precompute_factor) } else { "OFF".to_string() });
    if big_triangle {
        println!("║  Big Triangle: ON (alternative algorithm)");
    }
    if large_bucket_factor > 0 {
        println!("║  Large Bucket Factor: {}", large_bucket_factor);
    }
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();
    
    // Initialize GPU
    let _gpu = GpuAccelerator::new().expect("GPU init failed");
    
    // Generate points and scalars
    println!("🔧 Setup...");
    let points = G1Affine::generate_random(size);
    let scalars = ScalarField::generate_random(size);
    
    // Optionally precompute bases
    let precomputed: Option<DeviceVec<G1Affine>> = if precompute_factor > 0 {
        println!("   Precomputing bases (factor={})...", precompute_factor);
        let expanded_size = size * precompute_factor as usize;
        let mut precomp_buf = DeviceVec::<G1Affine>::device_malloc(expanded_size).unwrap();
        let start = Instant::now();
        
        let mut precompute_cfg = MSMConfig::default();
        precompute_cfg.precompute_factor = precompute_factor;
        
        precompute_bases::<G1Projective>(
            HostSlice::from_slice(&points),
            &precompute_cfg,
            &mut precomp_buf[..],
        ).unwrap();
        println!("   ✅ Precomputed in {:.2}ms", start.elapsed().as_secs_f64() * 1000.0);
        Some(precomp_buf)
    } else {
        None
    };
    
    // Pre-allocate GPU buffers for triple-buffering
    println!("   Allocating triple-buffer pipeline...");
    let mut scalar_bufs: Vec<DeviceVec<ScalarField>> = (0..3)
        .map(|_| DeviceVec::device_malloc(size).unwrap())
        .collect();
    let mut result_bufs: Vec<DeviceVec<G1Projective>> = (0..3)
        .map(|_| DeviceVec::device_malloc(1).unwrap())
        .collect();
    let streams: Vec<IcicleStream> = (0..3)
        .map(|_| IcicleStream::create().unwrap())
        .collect();
    
    // Pre-upload scalars to first buffer for warmup
    scalar_bufs[0].copy_from_host(HostSlice::from_slice(&scalars)).unwrap();
    
    println!("   ✅ Ready");
    println!();
    
    // Warmup
    println!("🔥 Warmup (10 proofs)...");
    for _ in 0..10 {
        let mut cfg = MSMConfig::default();
        cfg.c = c;
        cfg.are_points_shared_in_batch = true;
        
        // Set backend-specific extensions
        if big_triangle {
            cfg.ext.set_bool(CUDA_MSM_IS_BIG_TRIANGLE, true);
        }
        if large_bucket_factor > 0 {
            cfg.ext.set_int(CUDA_MSM_LARGE_BUCKET_FACTOR, large_bucket_factor);
        }
        
        if precompute_factor > 0 {
            cfg.precompute_factor = precompute_factor as i32;
            msm(
                &scalar_bufs[0][..],
                precomputed.as_ref().unwrap(),
                &cfg,
                &mut result_bufs[0][..],
            ).unwrap();
        } else {
            msm(
                &scalar_bufs[0][..],
                HostSlice::from_slice(&points),
                &cfg,
                &mut result_bufs[0][..],
            ).unwrap();
        }
    }
    
    // Run test
    println!();
    println!("📊 Running {}-second stress test...", duration_secs);
    println!();
    println!("   ┌─────────┬──────────┬──────────┬──────────┐");
    println!("   │  Time   │  Proofs  │   TPS    │  P50 ms  │");
    println!("   ├─────────┼──────────┼──────────┼──────────┤");
    
    let duration = Duration::from_secs(duration_secs);
    let start = Instant::now();
    let mut proof_count = 0u64;
    let mut latencies: Vec<f64> = Vec::with_capacity(10000);
    let mut interval_latencies: Vec<f64> = Vec::new();
    let mut last_report = Instant::now();
    let mut buf_idx = 0usize;
    
    while start.elapsed() < duration {
        let proof_start = Instant::now();
        
        // Upload scalars (async)
        scalar_bufs[buf_idx].copy_from_host_async(
            HostSlice::from_slice(&scalars),
            &streams[buf_idx],
        ).unwrap();
        
        // Configure MSM
        let mut cfg = MSMConfig::default();
        cfg.c = c;
        cfg.are_points_shared_in_batch = true;
        cfg.stream_handle = streams[buf_idx].handle;
        cfg.is_async = true;
        
        // Set backend-specific extensions
        if big_triangle {
            cfg.ext.set_bool(CUDA_MSM_IS_BIG_TRIANGLE, true);
        }
        if large_bucket_factor > 0 {
            cfg.ext.set_int(CUDA_MSM_LARGE_BUCKET_FACTOR, large_bucket_factor);
        }
        
        // Launch MSM
        if precompute_factor > 0 {
            cfg.precompute_factor = precompute_factor as i32;
            msm(
                &scalar_bufs[buf_idx][..],
                precomputed.as_ref().unwrap(),
                &cfg,
                &mut result_bufs[buf_idx][..],
            ).unwrap();
        } else {
            msm(
                &scalar_bufs[buf_idx][..],
                HostSlice::from_slice(&points),
                &cfg,
                &mut result_bufs[buf_idx][..],
            ).unwrap();
        }
        
        // Sync oldest buffer (back-pressure model)
        let sync_idx = (buf_idx + 1) % 3;
        streams[sync_idx].synchronize().ok();
        
        let latency = proof_start.elapsed().as_secs_f64() * 1000.0;
        latencies.push(latency);
        interval_latencies.push(latency);
        proof_count += 1;
        
        // Report every 10 seconds
        if last_report.elapsed() >= Duration::from_secs(10) {
            let elapsed = start.elapsed().as_secs_f64();
            let tps = proof_count as f64 / elapsed;
            
            interval_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p50 = interval_latencies[interval_latencies.len() / 2];
            
            println!("   │ {:5.0}s   │  {:6}  │  {:6.1}  │  {:6.2}  │",
                     elapsed, proof_count, tps, p50);
            
            interval_latencies.clear();
            last_report = Instant::now();
        }
        
        buf_idx = (buf_idx + 1) % 3;
    }
    
    // Final sync
    for stream in &streams {
        stream.synchronize().ok();
    }
    
    // Destroy streams to avoid warning
    for mut stream in streams {
        stream.destroy().ok();
    }
    
    println!("   └─────────┴──────────┴──────────┴──────────┘");
    println!();
    
    // Final stats
    let total_time = start.elapsed().as_secs_f64();
    let avg_tps = proof_count as f64 / total_time;
    
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = latencies[latencies.len() / 2];
    let p99 = latencies[(latencies.len() * 99) / 100];
    let avg_lat = latencies.iter().sum::<f64>() / latencies.len() as f64;
    
    println!("═══════════════════════════════════════════════════════════════════");
    println!("                          FINAL RESULTS");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();
    println!("   Configuration:");
    println!("     Size:       2^{} = {} points", k, size);
    println!("     c-param:    {} ({} buckets)", c, 1u32 << c);
    println!("     Precompute: {}", if precompute_factor > 0 { format!("factor={}", precompute_factor) } else { "OFF".to_string() });
    println!();
    println!("   Performance:");
    println!("     Total proofs: {}", proof_count);
    println!("     Duration:     {:.1}s", total_time);
    println!("     Average TPS:  {:.1}", avg_tps);
    println!();
    println!("   Latency:");
    println!("     Mean:   {:.2} ms", avg_lat);
    println!("     P50:    {:.2} ms", p50);
    println!("     P99:    {:.2} ms", p99);
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
}
