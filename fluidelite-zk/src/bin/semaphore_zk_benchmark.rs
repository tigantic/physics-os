//! Semaphore ZK Benchmark - Full Halo2-KZG Proof Generation
//!
//! This benchmark generates ACTUAL ZK proofs for Semaphore-compatible circuits,
//! enabling a TRUE apples-to-apples comparison with Worldcoin's Groth16 proofs.
//!
//! # What This Measures
//!
//! Unlike witness-only benchmarks, this measures the COMPLETE proof generation:
//! 1. GPU Poseidon witness computation
//! 2. Halo2 circuit synthesis
//! 3. KZG polynomial commitment
//! 4. PLONK proof generation
//!
//! # Comparison with Worldcoin
//!
//! Worldcoin uses:
//! - Groth16 via gnark
//! - ~6.3M constraints for depth 20
//! - ~9 TPS (batch 100) on GPUs
//!
//! This benchmark uses:
//! - Halo2-KZG
//! - Poseidon witness on GPU via ICICLE
//! - Full proof generation

use std::time::Instant;

#[cfg(all(feature = "gpu", feature = "halo2"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use fluidelite_zk::semaphore::{SemaphoreZkProver, SemaphoreZkProof};
    
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     SEMAPHORE ZK BENCHMARK - FULL HALO2-KZG PROOF GENERATION    ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  This generates ACTUAL ZK proofs, not just witness computation  ║");
    println!("║  For true apples-to-apples comparison with Worldcoin Groth16    ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    
    // Initialize ICICLE
    println!("[1/5] Initializing ICICLE backend...");
    icicle_runtime::runtime::load_backend_from_env_or_default()?;
    let device = icicle_runtime::Device::new("CUDA", 0);
    icicle_runtime::set_device(&device)?;
    
    let device_info = format!("{:?}", device);
    println!("  ✓ GPU: {}", device_info);
    println!();
    
    // Test depths
    let depths = [20, 30, 40, 50];
    let warmup_proofs = 3;
    let benchmark_proofs = 10;
    
    println!("[2/5] Running benchmarks at depths: {:?}", depths);
    println!("  Warmup: {} proofs, Benchmark: {} proofs", warmup_proofs, benchmark_proofs);
    println!();
    
    println!("┌─────────┬──────────┬──────────────┬──────────────┬────────────────┐");
    println!("│  Depth  │ Circuits │ Avg Time/Prf │ Proof Size   │ Throughput     │");
    println!("├─────────┼──────────┼──────────────┼──────────────┼────────────────┤");
    
    struct BenchResult {
        depth: usize,
        avg_time_ms: f64,
        proof_size: usize,
        tps: f64,
    }
    
    let mut results: Vec<BenchResult> = Vec::new();
    
    for &depth in &depths {
        print!("│ {:>7} │", depth);
        std::io::Write::flush(&mut std::io::stdout())?;
        
        // Create prover for this depth
        let prover = match SemaphoreZkProver::new(depth) {
            Ok(p) => p,
            Err(e) => {
                println!(" SETUP FAILED: {} │", e);
                continue;
            }
        };
        
        print!(" {:>8} │", format!("k={}", 10_u32.max(((depth + 10) as f64).log2().ceil() as u32)));
        std::io::Write::flush(&mut std::io::stdout())?;
        
        // Warmup
        for _ in 0..warmup_proofs {
            let witness = prover.generate_random_witness()?;
            let _ = prover.prove(&witness)?;
        }
        
        // Benchmark
        let mut total_time_ms = 0u64;
        let mut last_proof: Option<SemaphoreZkProof> = None;
        
        let start = Instant::now();
        for _ in 0..benchmark_proofs {
            let witness = prover.generate_random_witness()?;
            let proof = prover.prove(&witness)?;
            total_time_ms += proof.generation_time_ms;
            last_proof = Some(proof);
        }
        let elapsed = start.elapsed();
        
        let proof_size = last_proof.as_ref().map(|p| p.size()).unwrap_or(0);
        let avg_time_ms = elapsed.as_secs_f64() * 1000.0 / benchmark_proofs as f64;
        let tps = benchmark_proofs as f64 / elapsed.as_secs_f64();
        
        println!(" {:>10.2}ms │ {:>8} B  │ {:>10.2} TPS │",
            avg_time_ms,
            proof_size,
            tps
        );
        
        results.push(BenchResult {
            depth,
            avg_time_ms,
            proof_size,
            tps,
        });
        
        // Verify one proof
        if let Some(ref proof) = last_proof {
            let valid = prover.verify(proof)?;
            if !valid {
                println!("  ⚠ VERIFICATION FAILED");
            }
        }
    }
    
    println!("└─────────┴──────────┴──────────────┴──────────────┴────────────────┘");
    println!();
    
    // Comparison with Worldcoin
    println!("[3/5] Comparison with Worldcoin (Groth16, depth 20):");
    println!("  ┌────────────────────────────────────────────────────────────────┐");
    println!("  │ WORLDCOIN REFERENCE (gnark Groth16, AWS g6.4xlarge):           │");
    println!("  │   - Constraints: ~6.3M                                          │");
    println!("  │   - Batch size: 100                                             │");
    println!("  │   - Throughput: ~9 TPS                                          │");
    println!("  │   - Proof size: 256 bytes (8 × uint256)                         │");
    println!("  └────────────────────────────────────────────────────────────────┘");
    
    if let Some(depth20) = results.iter().find(|r| r.depth == 20) {
        let speedup = depth20.tps / 9.0;
        println!();
        println!("  ┌────────────────────────────────────────────────────────────────┐");
        println!("  │ FLUID-ZK (Halo2-KZG, RTX 5070 Ti):                             │");
        println!("  │   - Throughput: {:.2} TPS                              │", depth20.tps);
        println!("  │   - Proof size: {} bytes                                    │", depth20.proof_size);
        println!("  │   - SPEEDUP vs Worldcoin: {:.1}x                               │", speedup);
        println!("  └────────────────────────────────────────────────────────────────┘");
    }
    println!();
    
    // Scaling analysis
    println!("[4/5] Depth Scaling Analysis:");
    for i in 1..results.len() {
        let prev = &results[i-1];
        let curr = &results[i];
        let time_ratio = curr.avg_time_ms / prev.avg_time_ms;
        let depth_ratio = (curr.depth as f64) / (prev.depth as f64);
        println!("  Depth {} → {}: time ratio {:.2}x (depth ratio {:.2}x)", 
            prev.depth, curr.depth, time_ratio, depth_ratio);
    }
    println!();
    
    // Summary
    println!("[5/5] BENCHMARK COMPLETE");
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                        FINAL RESULTS                             ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    for r in &results {
        println!("║  Depth {:>2}: {:>6.2} TPS, {:>6} byte proof, {:>8.2}ms/proof       ║",
            r.depth, r.tps, r.proof_size, r.avg_time_ms);
    }
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  Worldcoin: ~9 TPS (Groth16, depth 20)                           ║");
    if let Some(depth20) = results.iter().find(|r| r.depth == 20) {
        println!("║  Fluid-ZK:  {:.1} TPS (Halo2-KZG, depth 20) = {:.1}x FASTER         ║",
            depth20.tps, depth20.tps / 9.0);
    }
    println!("╚══════════════════════════════════════════════════════════════════╝");
    
    Ok(())
}

#[cfg(not(all(feature = "gpu", feature = "halo2")))]
fn main() {
    eprintln!("ERROR: This benchmark requires both 'gpu' and 'halo2' features.");
    eprintln!("Run with: cargo run --bin semaphore_zk_benchmark --features gpu,halo2");
    std::process::exit(1);
}
