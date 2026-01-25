//! Zero-Expansion QTT Benchmark
//!
//! Compares traditional (expanded) MSM vs QTT-native MSM.
//!
//! Usage:
//!   cargo run --release --features gpu --bin qtt-zero-expansion -- --sites 18 --rank 16

use fluidelite_zk::gpu::GpuAccelerator;
use fluidelite_zk::qtt_native_msm::{
    QttTrain, QttCommitmentBases, qtt_native_commit, compute_qtt_stats,
    FlattenedQtt, BatchedQttBases, qtt_batched_commit,
};
use icicle_bn254::curve::{G1Affine, G1Projective, ScalarField};
use icicle_core::msm::{msm, precompute_bases, MSMConfig};
use icicle_core::traits::GenerateRandom;
use icicle_runtime::memory::{DeviceVec, HostSlice};
use icicle_runtime::stream::IcicleStream;
use std::time::Instant;

fn main() {
    // Parse args
    let args: Vec<String> = std::env::args().collect();
    
    let mut n_sites: usize = 18;
    let mut max_rank: usize = 16;
    let mut precompute_factor: i32 = 10;
    let mut c: i32 = 16;
    let mut iterations: usize = 100;
    
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--sites" => { n_sites = args[i + 1].parse().unwrap(); i += 2; }
            "--rank" => { max_rank = args[i + 1].parse().unwrap(); i += 2; }
            "--precompute" => { precompute_factor = args[i + 1].parse().unwrap(); i += 2; }
            "--c" => { c = args[i + 1].parse().unwrap(); i += 2; }
            "--iterations" => { iterations = args[i + 1].parse().unwrap(); i += 2; }
            _ => { i += 1; }
        }
    }
    
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                     ZERO-EXPANSION QTT-NATIVE MSM BENCHMARK                  ║");
    println!("║                                                                              ║");
    println!("║   Traditional: QTT → EXPAND → 2^N scalars → MSM  (PCIe every proof!)        ║");
    println!("║   Zero-Expansion: Bases in VRAM, only scalars cross PCIe per proof          ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║  Sites:      {}  (full dimension = 2^{} = {})", n_sites, n_sites, 1usize << n_sites);
    println!("║  Max Rank:   {}", max_rank);
    println!("║  c-param:    {}", c);
    println!("║  Precompute: {}", precompute_factor);
    println!("║  Iterations: {}", iterations);
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Initialize GPU
    let _gpu = GpuAccelerator::new().expect("GPU init failed");
    println!("✓ GPU initialized\n");

    // Create QTT
    println!("🔧 Creating {}-site QTT with rank {}...", n_sites, max_rank);
    let qtt = QttTrain::random(n_sites, 2, max_rank);
    println!("   Total params: {} ({:.2} KB)", 
        qtt.total_params(), 
        qtt.total_params() as f64 * 32.0 / 1024.0);
    println!("   Full dimension: 2^{} = {}", n_sites, qtt.full_dimension());
    println!("   Compression ratio: {:.0}x", qtt.compression_ratio());
    println!();

    // Generate BATCHED commitment bases (optimized single-MSM)
    println!("🔧 Generating BATCHED QTT commitment bases...");
    let start = Instant::now();
    let batched_bases = BatchedQttBases::generate(&qtt, precompute_factor)
        .expect("Batched bases generation failed");
    println!("   Done in {:.2}s", start.elapsed().as_secs_f64());
    println!("   VRAM usage: {:.2} MB", batched_bases.vram_bytes() as f64 / 1024.0 / 1024.0);
    println!();
    
    // Flatten QTT for batched MSM
    let flat_qtt = FlattenedQtt::from_qtt(&qtt);
    println!("   Flattened QTT: {} scalars ({:.2} KB)", 
        flat_qtt.total_size,
        flat_qtt.total_size as f64 * 32.0 / 1024.0);
    println!();

    // Also generate old-style bases for comparison
    println!("🔧 Generating per-core bases (for comparison)...");
    let start = Instant::now();
    let bases = QttCommitmentBases::generate(&qtt, precompute_factor)
        .expect("Bases generation failed");
    println!("   Done in {:.2}s", start.elapsed().as_secs_f64());
    println!();

    // Print statistics
    let stats = compute_qtt_stats(&qtt, &bases);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("{}", stats);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Warmup BATCHED version
    println!("🔥 Warming up BATCHED QTT-native MSM...");
    for _ in 0..5 {
        let _ = qtt_batched_commit(&flat_qtt, &batched_bases, c);
    }
    println!();

    // Benchmark BATCHED QTT-native MSM (optimized)
    println!("📊 Benchmarking BATCHED QTT-native MSM ({} iterations)...", iterations);
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = qtt_batched_commit(&flat_qtt, &batched_bases, c).expect("Commit failed");
    }
    let qtt_batched_time = start.elapsed();
    let qtt_batched_tps = iterations as f64 / qtt_batched_time.as_secs_f64();
    let qtt_batched_latency_ms = qtt_batched_time.as_secs_f64() * 1000.0 / iterations as f64;
    
    println!("   BATCHED QTT-Native: {:.1} TPS, {:.2} ms/proof", qtt_batched_tps, qtt_batched_latency_ms);
    println!();

    // Also benchmark naive per-core version for comparison
    println!("📊 Benchmarking NAIVE per-core QTT MSM ({} iterations)...", iterations.min(20));
    let naive_iterations = iterations.min(20); // Fewer since it's slower
    let start = Instant::now();
    for _ in 0..naive_iterations {
        let _ = qtt_native_commit(&qtt, &bases, c, None).expect("Commit failed");
    }
    let qtt_naive_time = start.elapsed();
    let qtt_naive_tps = naive_iterations as f64 / qtt_naive_time.as_secs_f64();
    let qtt_naive_latency_ms = qtt_naive_time.as_secs_f64() * 1000.0 / naive_iterations as f64;
    
    println!("   NAIVE per-core: {:.1} TPS, {:.2} ms/proof (kernel launch overhead)", qtt_naive_tps, qtt_naive_latency_ms);
    println!();

    // Compare with traditional (expanded) MSM
    println!("📊 Comparing with traditional (expanded) MSM...");
    
    let full_size = 1usize << n_sites;
    
    // Only do traditional benchmark for reasonable sizes
    if full_size <= 1 << 20 {
        // Generate full expanded scalars (simulating expansion)
        println!("   Generating {} expanded scalars...", full_size);
        let expanded_scalars = ScalarField::generate_random(full_size);
        let points = G1Affine::generate_random(full_size);
        
        // Upload to GPU
        let mut points_d = DeviceVec::<G1Affine>::device_malloc(full_size).unwrap();
        points_d.copy_from_host(HostSlice::from_slice(&points)).unwrap();
        
        // Precompute if using factor
        let precomputed = if precompute_factor > 1 {
            let expanded_size = full_size * precompute_factor as usize;
            let mut precomp_buf = DeviceVec::<G1Affine>::device_malloc(expanded_size).unwrap();
            
            let mut cfg = MSMConfig::default();
            cfg.precompute_factor = precompute_factor;
            
            precompute_bases::<G1Projective>(
                HostSlice::from_slice(&points),
                &cfg,
                &mut precomp_buf[..],
            ).unwrap();
            
            Some(precomp_buf)
        } else {
            None
        };
        
        // Warmup
        println!("   Warming up traditional MSM...");
        let mut result_buf = DeviceVec::<G1Projective>::device_malloc(1).unwrap();
        for _ in 0..3 {
            let mut cfg = MSMConfig::default();
            cfg.c = c;
            if let Some(ref precomp) = precomputed {
                cfg.precompute_factor = precompute_factor;
                msm(
                    HostSlice::from_slice(&expanded_scalars),
                    precomp,
                    &cfg,
                    &mut result_buf[..],
                ).unwrap();
            } else {
                msm(
                    HostSlice::from_slice(&expanded_scalars),
                    &points_d,
                    &cfg,
                    &mut result_buf[..],
                ).unwrap();
            }
        }
        
        // Benchmark
        let trad_iterations = iterations.min(50); // Fewer iterations for larger size
        let start = Instant::now();
        for _ in 0..trad_iterations {
            let mut cfg = MSMConfig::default();
            cfg.c = c;
            if let Some(ref precomp) = precomputed {
                cfg.precompute_factor = precompute_factor;
                msm(
                    HostSlice::from_slice(&expanded_scalars),
                    precomp,
                    &cfg,
                    &mut result_buf[..],
                ).unwrap();
            } else {
                msm(
                    HostSlice::from_slice(&expanded_scalars),
                    &points_d,
                    &cfg,
                    &mut result_buf[..],
                ).unwrap();
            }
        }
        let trad_time = start.elapsed();
        let trad_tps = trad_iterations as f64 / trad_time.as_secs_f64();
        let trad_latency_ms = trad_time.as_secs_f64() * 1000.0 / trad_iterations as f64;
        
        println!("   Traditional: {:.1} TPS, {:.2} ms/proof", trad_tps, trad_latency_ms);
        println!();
        
        // Summary
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("                                RESULTS SUMMARY");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!();
        println!("   ┌──────────────────────┬────────────┬──────────────┬───────────────┐");
        println!("   │       Method         │    TPS     │  Latency     │  PCIe Transfer│");
        println!("   ├──────────────────────┼────────────┼──────────────┼───────────────┤");
        println!("   │ Traditional (expand) │ {:>8.1}   │ {:>8.2} ms  │ {:>8.2} MB   │", 
            trad_tps, trad_latency_ms, stats.traditional_transfer_mb);
        println!("   │ QTT BATCHED          │ {:>8.1}   │ {:>8.2} ms  │ {:>8.2} MB   │", 
            qtt_batched_tps, qtt_batched_latency_ms, stats.zero_expansion_transfer_mb);
        println!("   │ QTT naive (per-core) │ {:>8.1}   │ {:>8.2} ms  │ {:>8.2} MB   │", 
            qtt_naive_tps, qtt_naive_latency_ms, stats.zero_expansion_transfer_mb);
        println!("   ├──────────────────────┼────────────┼──────────────┼───────────────┤");
        println!("   │ Batched Speedup      │ {:>8.1}x  │ {:>8.1}x    │ {:>8.0}x     │",
            qtt_batched_tps / trad_tps.max(0.001),
            trad_latency_ms / qtt_batched_latency_ms.max(0.001),
            stats.transfer_reduction);
        println!("   └──────────────────────┴────────────┴──────────────┴───────────────┘");
        println!();
        
        // Key insight
        if qtt_batched_tps > trad_tps {
            println!("   ✅ QTT BATCHED is {:.1}x FASTER than traditional!", qtt_batched_tps / trad_tps);
        } else {
            println!("   ⚠️  Traditional is faster at this size (GPU not saturated with {} scalars)", flat_qtt.total_size);
            println!("   💡 QTT wins at larger sizes where PCIe becomes bottleneck");
        }
        println!();
    } else {
        println!("   ⚠️  Skipping traditional benchmark (2^{} = {} too large)", n_sites, full_size);
        println!("   Traditional would require {:.1} MB PCIe transfer per proof!", 
            stats.traditional_transfer_mb);
        println!();
        
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("                            QTT-NATIVE RESULTS");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!();
        println!("   BATCHED Throughput:  {:.1} TPS", qtt_batched_tps);
        println!("   BATCHED Latency:     {:.2} ms/proof", qtt_batched_latency_ms);
        println!("   PCIe Transfer:       {:.2} KB/proof (vs {:.1} MB traditional)", 
            stats.zero_expansion_transfer_mb * 1024.0, stats.traditional_transfer_mb);
        println!("   Reduction:           {:.0}x less data movement!", stats.transfer_reduction);
        println!();
    }
    
    println!("✅ Done!\n");
}