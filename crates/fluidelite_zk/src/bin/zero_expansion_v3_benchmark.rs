//! Zero-Expansion v3 Benchmark — Batched Structure Proofs
//!
//! Tests the batched prover that achieves 200+ TPS by amortizing
//! structure proof overhead across multiple QTT commitments.
//!
//! Usage:
//!   cargo run --release --features "gpu,halo2" --bin zero_expansion_v3_benchmark -- [OPTIONS]
//!
//! Options:
//!   --sites N     Number of QTT sites (default: 18)
//!   --rank R      Maximum rank (default: 16)
//!   --batch B     Batch size (default: 32)
//!   --total T     Total proofs to generate (default: 256)

#[cfg(all(feature = "gpu", feature = "halo2"))]
use fluidelite_zk::gpu::GpuAccelerator;

#[cfg(all(feature = "gpu", feature = "halo2"))]
use fluidelite_zk::qtt_native_msm::QttTrain;

#[cfg(all(feature = "gpu", feature = "halo2"))]
use fluidelite_zk::zero_expansion_prover_v3::{
    ZeroExpansionProverV3, StreamingZeroExpansionProver, DeferredStructureProver,
};

#[cfg(all(feature = "gpu", feature = "halo2"))]
use std::io::Write;

#[cfg(all(feature = "gpu", feature = "halo2"))]
use std::time::Instant;

#[cfg(all(feature = "gpu", feature = "halo2"))]
fn main() {
    println!("");
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                                                                              ║");
    println!("║      ███████╗███████╗██████╗  ██████╗       ███████╗██╗  ██╗██████╗         ║");
    println!("║      ╚══███╔╝██╔════╝██╔══██╗██╔═══██╗      ██╔════╝╚██╗██╔╝██╔══██╗        ║");
    println!("║        ███╔╝ █████╗  ██████╔╝██║   ██║█████╗█████╗   ╚███╔╝ ██████╔╝        ║");
    println!("║       ███╔╝  ██╔══╝  ██╔══██╗██║   ██║╚════╝██╔══╝   ██╔██╗ ██╔═══╝         ║");
    println!("║      ███████╗███████╗██║  ██║╚██████╔╝      ███████╗██╔╝ ██╗██║             ║");
    println!("║      ╚══════╝╚══════╝╚═╝  ╚═╝ ╚═════╝       ╚══════╝╚═╝  ╚═╝╚═╝             ║");
    println!("║                                                                              ║");
    println!("║              PROVER v3.0 — BATCHED STRUCTURE PROOFS                         ║");
    println!("║                    Target: 200+ TPS with full proofs                        ║");
    println!("║                                                                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!("");

    // Parse arguments
    let args: Vec<String> = std::env::args().collect();
    let mut n_sites = 18usize;
    let mut max_rank = 16usize;
    let mut batch_size = 32usize;
    let mut total_proofs = 256usize;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--sites" => {
                n_sites = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(18);
                i += 2;
            }
            "--rank" => {
                max_rank = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(16);
                i += 2;
            }
            "--batch" => {
                batch_size = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(32);
                i += 2;
            }
            "--total" => {
                total_proofs = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(256);
                i += 2;
            }
            _ => i += 1,
        }
    }

    println!("Configuration:");
    println!("  Scale:      2^{} = {} points", n_sites, 1usize << n_sites);
    println!("  Max Rank:   {}", max_rank);
    println!("  Batch Size: {}", batch_size);
    println!("  Total:      {} proofs", total_proofs);
    println!("");

    // Initialize GPU
    println!("Initializing GPU...");
    let gpu = GpuAccelerator::new().expect("GPU initialization failed");
    println!("  ✓ GPU: {}", gpu.device_name());
    println!("  ✓ VRAM: {} MB", gpu.vram_mb());
    println!("");

    // Create lookup table
    let table: Vec<(u64, u64, u8)> = (0..256)
        .map(|i| (i as u64, 0u64, i as u8))
        .collect();

    // ========================================================================
    // BENCHMARK 1: Batched Prover v3
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("BENCHMARK 1: BATCHED PROVER v3 (batch_size={})", batch_size);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("");

    let prover = ZeroExpansionProverV3::new(
        n_sites,
        max_rank,
        table.clone(),
        10, // precompute_factor
        16, // msm_c
        batch_size,
    ).expect("Prover initialization failed");

    // Warmup
    println!("Warming up...");
    {
        let qtts: Vec<QttTrain> = (0..batch_size)
            .map(|_| QttTrain::random(n_sites, 2, max_rank))
            .collect();
        let contexts: Vec<Vec<u8>> = (0..batch_size)
            .map(|i| vec![i as u8; 12])
            .collect();
        let predictions: Vec<u8> = (0..batch_size).map(|i| i as u8).collect();
        let _ = prover.prove_batch(&qtts, &contexts, &predictions).expect("Warmup failed");
    }
    println!("  ✓ Warmup complete");
    println!("");

    // Run benchmark
    println!("Running {} proofs in batches of {}...", total_proofs, batch_size);
    let start = Instant::now();
    let n_batches = (total_proofs + batch_size - 1) / batch_size;
    let mut total_qtt_ms = 0u64;
    let mut total_structure_ms = 0u64;
    let mut proofs_done = 0;

    for batch_idx in 0..n_batches {
        let this_batch = (total_proofs - proofs_done).min(batch_size);
        
        let qtts: Vec<QttTrain> = (0..this_batch)
            .map(|_| QttTrain::random(n_sites, 2, max_rank))
            .collect();
        let contexts: Vec<Vec<u8>> = (0..this_batch)
            .map(|i| vec![(batch_idx * batch_size + i) as u8; 12])
            .collect();
        let predictions: Vec<u8> = (0..this_batch).map(|i| i as u8).collect();

        let proof = prover.prove_batch(&qtts, &contexts, &predictions)
            .expect("Batch proof failed");

        total_qtt_ms += proof.stats.qtt_commit_total_ms;
        total_structure_ms += proof.stats.structure_proof_ms;
        proofs_done += this_batch;

        let elapsed = start.elapsed().as_secs_f64();
        let tps = proofs_done as f64 / elapsed;
        print!("\r  [Batch {}/{}] {} proofs | {:.1} TPS | QTT: {}ms | Struct: {}ms   ",
            batch_idx + 1, n_batches, proofs_done, tps,
            total_qtt_ms / (batch_idx + 1) as u64,
            total_structure_ms / (batch_idx + 1) as u64);
        std::io::stdout().flush().ok();
    }

    let total_time = start.elapsed();
    let batched_tps = total_proofs as f64 / total_time.as_secs_f64();
    println!("\n");

    // ========================================================================
    // BENCHMARK 2: Streaming Prover
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("BENCHMARK 2: STREAMING PROVER (auto-batch at {})", batch_size);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("");

    let mut streaming = StreamingZeroExpansionProver::new(
        n_sites,
        max_rank,
        table.clone(),
        batch_size,
    ).expect("Streaming prover init failed");

    println!("Submitting {} proofs...", total_proofs);
    let start = Instant::now();

    for i in 0..total_proofs {
        let qtt = QttTrain::random(n_sites, 2, max_rank);
        let context = vec![i as u8; 12];
        let prediction = (i % 256) as u8;
        
        if let Some(_proof) = streaming.submit(qtt, context, prediction).expect("Submit failed") {
            // Batch completed
        }

        if (i + 1) % 32 == 0 || i == total_proofs - 1 {
            let elapsed = start.elapsed().as_secs_f64();
            let tps = streaming.total_proofs() as f64 / elapsed;
            print!("\r  [Progress] {}/{} submitted | {} completed | {:.1} TPS   ",
                i + 1, total_proofs, streaming.total_proofs(), tps);
            std::io::stdout().flush().ok();
        }
    }

    // Flush remaining
    if streaming.total_proofs() < total_proofs {
        let _ = streaming.flush();
    }

    let streaming_tps = streaming.effective_tps();
    println!("\n");

    // ========================================================================
    // BENCHMARK 3: Deferred Mode (Commitment TPS)
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("BENCHMARK 3: DEFERRED MODE (Commit now, prove later)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("");

    let mut deferred = DeferredStructureProver::new(
        n_sites,
        max_rank,
        table.clone(),
    ).expect("Deferred prover init failed");

    println!("Committing {} QTTs (deferred structure proofs)...", total_proofs);
    let start = Instant::now();

    for i in 0..total_proofs {
        let qtt = QttTrain::random(n_sites, 2, max_rank);
        let context = vec![i as u8; 12];
        let prediction = (i % 256) as u8;
        
        let _ = deferred.commit(&qtt, context, prediction).expect("Commit failed");

        if (i + 1) % 64 == 0 || i == total_proofs - 1 {
            let tps = deferred.commitment_tps();
            print!("\r  [Committing] {}/{} | {:.1} TPS (commitments only)   ", i + 1, total_proofs, tps);
            std::io::stdout().flush().ok();
        }
    }

    let commit_time = start.elapsed();
    let commit_tps = total_proofs as f64 / commit_time.as_secs_f64();
    println!("\n");

    println!("Generating deferred structure proofs...");
    let struct_start = Instant::now();
    let proofs = deferred.finalize_all().expect("Finalize failed");
    let struct_time = struct_start.elapsed();
    let total_batches = proofs.len();
    println!("  ✓ {} batches in {:.2}s", total_batches, struct_time.as_secs_f64());
    println!("");

    // ========================================================================
    // RESULTS
    // ========================================================================
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                              RESULTS SUMMARY                                 ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                              ║");
    println!("║  Configuration:                                                              ║");
    println!("║    Scale:            2^{:<4} = {:>10} points                           ║", n_sites, 1usize << n_sites);
    println!("║    Max Rank:         {:<10}                                            ║", max_rank);
    println!("║    Batch Size:       {:<10}                                            ║", batch_size);
    println!("║    Total Proofs:     {:<10}                                            ║", total_proofs);
    println!("║                                                                              ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║  THROUGHPUT COMPARISON                                                       ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                              ║");
    println!("║  v2.0 Sequential:    ~6.3 TPS  (155ms struct proof each)                   ║");
    println!("║                                                                              ║");
    println!("║  v3.0 BATCHED:       {:<6.1} TPS  ← BENCHMARK 1                              ║", batched_tps);
    println!("║  v3.0 STREAMING:     {:<6.1} TPS  ← BENCHMARK 2                              ║", streaming_tps);
    println!("║  v3.0 COMMIT ONLY:   {:<6.1} TPS  ← BENCHMARK 3 (deferred struct)           ║", commit_tps);
    println!("║                                                                              ║");

    let best_tps = batched_tps.max(streaming_tps);
    let speedup = best_tps / 6.3;
    if best_tps >= 200.0 {
        println!("║  ✅ TARGET MET: {:.1} TPS > 200 TPS                                       ║", best_tps);
    } else {
        println!("║  ⚠️  TARGET: 200 TPS | Best: {:.1} TPS | Need {:.0}x larger batches       ║", 
            best_tps, 200.0 / best_tps);
    }
    println!("║  📈 Speedup vs v2.0: {:.1}x                                                  ║", speedup);
    println!("║                                                                              ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║  WHAT THIS MEANS                                                             ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                              ║");
    println!("║  • Batching amortizes Halo2 structure proof overhead                        ║");
    println!("║  • Deferred mode achieves pure MSM throughput for commitments               ║");
    println!("║  • Structure proofs can be generated async/in background                    ║");
    println!("║                                                                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!("");
}

#[cfg(not(all(feature = "gpu", feature = "halo2")))]
fn main() {
    eprintln!("╔══════════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  ERROR: This benchmark requires both gpu and halo2 features                  ║");
    eprintln!("╠══════════════════════════════════════════════════════════════════════════════╣");
    eprintln!("║  Run with:                                                                   ║");
    eprintln!("║    cargo run --release --features=\"gpu,halo2\" --bin zero_expansion_v3_benchmark ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════════════════╝");
}
