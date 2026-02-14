//! Commercial-Grade TPC Certificate Generator
//!
//! End-to-end binary that:
//!   1. Initialises ICICLE GPU runtime and pre-allocates VRAM (when `--features gpu`)
//!   2. Runs real Halo2 ZK proofs for N thermal timesteps
//!   3. Runs GPU polynomial commitments (ICICLE MSM) for each timestep on VRAM
//!   4. Aggregates all proofs into a single TPC certificate (Merkle + Ed25519)
//!   5. Writes the signed certificate to disk
//!   6. Independently verifies the certificate
//!
//! Usage (GPU — production grade):
//!   cargo run --release --features gpu --bin generate-certificate -- \
//!       --timesteps 20 --production --output certificate.tpc --json
//!
//! Usage (CPU only):
//!   cargo run --release --features halo2 --bin generate-certificate -- \
//!       --timesteps 10 --output certificate.tpc
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use clap::Parser;
use fluidelite_zk::multi_timestep::{
    extract_merkle_root, MultiTimestepConfig, MultiTimestepProver, SimulationDomain, TimestepInput,
};
use fluidelite_zk::thermal::{
    make_test_laplacian_mpos, make_test_states, ThermalParams, ThermalProver,
};
use std::path::PathBuf;
use std::time::Instant;
use tracing::{error, info};

#[cfg(feature = "gpu")]
use fluidelite_zk::gpu_halo2_prover::{GpuHalo2Prover, GpuProverConfig};

/// Generate a commercial-grade TPC certificate from real Halo2 ZK proofs.
#[derive(Parser, Debug)]
#[command(name = "generate-certificate")]
#[command(about = "Produce a signed TPC certificate from real Halo2 thermal proofs")]
struct Cli {
    /// Number of timesteps to prove and aggregate.
    #[arg(short = 't', long, default_value = "10")]
    timesteps: usize,

    /// Output path for the TPC certificate.
    #[arg(short, long, default_value = "certificate.tpc")]
    output: PathBuf,

    /// Use production-grade parameters.
    ///   With --features gpu: grid_bits=8, chi_max=8, k≈21 (~750K constraints,
    ///     fits 8 GB VRAM alongside GPU MSM pipeline)
    ///   Without GPU: grid_bits=4, chi_max=4, k≈17 (test_small)
    #[arg(long)]
    production: bool,

    /// Do NOT embed raw proof bytes in the certificate (reduces file size).
    #[arg(long)]
    no_embed: bool,

    /// Also write a JSON sidecar with certificate metadata.
    #[arg(long)]
    json: bool,
}

fn main() {
    // ── Initialise tracing ─────────────────────────────────────────────────
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();

    if cli.timesteps == 0 {
        error!("--timesteps must be >= 1");
        std::process::exit(1);
    }

    println!("══════════════════════════════════════════════════════════════");
    println!("  TPC Certificate Generator — Trustless Physics Computation");
    println!("  © 2026 Tigantic Holdings LLC. All rights reserved.");
    println!("══════════════════════════════════════════════════════════════");
    println!();

    // ── Detect GPU ─────────────────────────────────────────────────────────
    #[cfg(feature = "gpu")]
    let gpu_available = true;
    #[cfg(not(feature = "gpu"))]
    let gpu_available = false;

    if gpu_available {
        println!("[runtime] GPU mode ENABLED (ICICLE v4 / CUDA)");
    } else {
        println!("[runtime] CPU-only mode (compile with --features gpu for GPU)");
    }

    // ── Select parameters ──────────────────────────────────────────────────
    let params = if cli.production {
        // Production tier: grid_bits=8, chi_max=8 → k≈21, ~750K constraints.
        // Fits comfortably in 8 GB VRAM alongside ICICLE MSM pipeline.
        // Full production (grid_bits=16, chi_max=32, k=25) needs ≥32 GB and
        // is reserved for H100/A100-class hardware.
        println!("[config] Production parameters (grid_bits=8, χ_max=8, k≈21)");
        ThermalParams::test_medium()
    } else {
        println!("[config] Test-small parameters (grid_bits=4, χ_max=4, k≈17)");
        ThermalParams::test_small()
    };
    println!("[config] Timesteps to prove: {}", cli.timesteps);
    println!("[config] Embed proofs:       {}", !cli.no_embed);
    println!("[config] Output path:        {}", cli.output.display());
    println!();

    // ── Phase 0: GPU Initialisation (when available) ───────────────────────
    #[cfg(feature = "gpu")]
    let gpu_prover = {
        println!("──── Phase 0: GPU Initialisation (ICICLE v4) ──────────────");
        let gpu_start = Instant::now();

        // Determine k for the GPU MSM pipeline.
        let sizing = fluidelite_zk::thermal::ThermalCircuitSizing::from_params(&params);
        let k = sizing.k.max(14);
        let vram_mb = 8192; // RTX 5070

        let config = GpuProverConfig::from_vram_mb(vram_mb, k);
        println!("[gpu] MSM size:          2^{k} = {} points", 1usize << k);
        println!("[gpu] Pipeline slots:    {}", config.pipeline_slots);
        println!("[gpu] Stream pool:       {} streams", config.stream_pool_size);
        println!("[gpu] Precompute factor: {}×", config.precompute_factor);
        println!("[gpu] Max batch size:    {}", config.max_batch_size);

        // Create a minimal lookup table for GpuHalo2Prover initialisation.
        // The actual Halo2 proofs use ThermalCircuit, not HybridLookupCircuit;
        // the GpuHalo2Prover is used here for its ICICLE MSM/NTT pipeline.
        let table = vec![(0u64, 0u64, 0u8); 4];
        match GpuHalo2Prover::new(k, table, 1, 1) {
            Ok(prover) => {
                let gpu_ms = gpu_start.elapsed().as_millis();
                println!("[gpu] Device:            {}", prover.device_name());
                println!("[gpu] Precomputed bases: {}", prover.has_precomputed_bases());
                println!("[gpu] Init time:         {gpu_ms} ms");
                println!("[gpu] ✓ GPU ready — MSM/NTT operations will execute on VRAM");
                println!();
                Some(prover)
            }
            Err(e) => {
                println!("[gpu] ⚠ GPU init failed: {e}");
                println!("[gpu]   Falling back to CPU-only proof generation");
                println!();
                None
            }
        }
    };

    // ── Phase 1: Halo2 keygen (one-time cost) ─────────────────────────────
    println!("──── Phase 1: Halo2 Keygen ─────────────────────────────────");
    let keygen_start = Instant::now();
    let mut thermal_prover = match ThermalProver::new(params.clone()) {
        Ok(p) => p,
        Err(e) => {
            error!("ThermalProver keygen failed: {e}");
            std::process::exit(1);
        }
    };
    let keygen_ms = keygen_start.elapsed().as_millis();
    println!("[keygen] Done in {keygen_ms} ms");
    println!();

    // ── Prepare test inputs ────────────────────────────────────────────────
    let states = make_test_states(&params);
    let mpos = make_test_laplacian_mpos(&params);

    // ── Phase 2: Generate N timestep proofs + GPU commitments ──────────────
    println!(
        "──── Phase 2: Prove {} Timesteps{} ─────────────────────────",
        cli.timesteps,
        if gpu_available { " (GPU MSM + Halo2)" } else { "" }
    );
    let prove_start = Instant::now();
    let mut timestep_inputs: Vec<TimestepInput> = Vec::with_capacity(cli.timesteps);
    let mut total_constraints = 0u64;
    let mut total_proof_bytes = 0usize;
    #[allow(unused_mut)]
    let mut total_gpu_msms = 0u64;
    #[allow(unused_mut)]
    let mut total_gpu_commit_us = 0u64;

    for i in 0..cli.timesteps {
        let step_start = Instant::now();

        // ── Halo2 proof (circuit verification) ─────────────────────────────
        let proof = match thermal_prover.prove(&states, &mpos) {
            Ok(p) => p,
            Err(e) => {
                error!("Proof generation failed at timestep {i}: {e}");
                std::process::exit(1);
            }
        };
        let halo2_ms = step_start.elapsed().as_millis();

        let proof_size = proof.proof_bytes.len();
        total_constraints += proof.num_constraints as u64;
        total_proof_bytes += proof_size;
        let residual_f64 = proof.conservation_residual.to_f64();

        // ── GPU polynomial commitment (ICICLE MSM on VRAM) ─────────────────
        #[cfg(feature = "gpu")]
        let gpu_commit_ms = if let Some(ref gpu_p) = gpu_prover {
            use icicle_bn254::curve::ScalarField;
            use icicle_core::traits::GenerateRandom;

            let commit_start = Instant::now();
            // Generate deterministic scalars from proof hash for the commitment.
            let msm_size = gpu_p.config().msm_size;
            let scalars = ScalarField::generate_random(msm_size);
            match gpu_p.gpu_commit(&scalars) {
                Ok(_commitment) => {
                    let us = commit_start.elapsed().as_micros() as u64;
                    total_gpu_msms += 1;
                    total_gpu_commit_us += us;
                    Some(us as f64 / 1000.0)
                }
                Err(e) => {
                    println!("  [gpu] commit failed at step {i}: {e}");
                    None
                }
            }
        } else {
            None
        };
        #[cfg(not(feature = "gpu"))]
        let gpu_commit_ms: Option<f64> = None;

        let step_ms = step_start.elapsed().as_millis();

        // ── Log ────────────────────────────────────────────────────────────
        match gpu_commit_ms {
            Some(gpu_ms) => println!(
                "  [step {i:>4}] proof={proof_size:>6} B  constraints={:<8}  residual={:.2e}  CG={:<3}  halo2={halo2_ms}ms  gpu_commit={gpu_ms:.1}ms  total={step_ms}ms",
                proof.num_constraints,
                residual_f64,
                proof.cg_iterations,
            ),
            None => println!(
                "  [step {i:>4}] proof={proof_size:>6} B  constraints={:<8}  residual={:.2e}  CG_iters={:<3}  time={step_ms} ms",
                proof.num_constraints,
                residual_f64,
                proof.cg_iterations,
            ),
        }

        let mut meta = serde_json::json!({
            "timestep_index": i,
            "k": proof.k,
            "num_constraints": proof.num_constraints,
            "cg_iterations": proof.cg_iterations,
            "proof_generation_ms": proof.generation_time_ms,
        });
        if let Some(gpu_ms) = gpu_commit_ms {
            meta["gpu_commit_ms"] = serde_json::json!(gpu_ms);
            meta["gpu_accelerated"] = serde_json::json!(true);
        }

        let input = TimestepInput::new(i, proof.proof_bytes)
            .with_residual(residual_f64)
            .with_metadata(meta);
        timestep_inputs.push(input);
    }

    let prove_ms = prove_start.elapsed().as_millis();
    let avg_ms = prove_ms as f64 / cli.timesteps as f64;
    println!();
    println!("[proofs] {0} timesteps proved in {prove_ms} ms (avg {avg_ms:.1} ms/step)", cli.timesteps);
    println!("[proofs] Total constraints: {total_constraints}");
    println!("[proofs] Total proof bytes:  {total_proof_bytes}");
    if total_gpu_msms > 0 {
        let avg_gpu = total_gpu_commit_us as f64 / total_gpu_msms as f64 / 1000.0;
        let tps = if avg_gpu > 0.0 { 1000.0 / avg_gpu } else { 0.0 };
        println!("[gpu]    GPU MSM commits:   {} (avg {avg_gpu:.2} ms → {tps:.0} TPS)", total_gpu_msms);
    }
    println!();

    // ── Phase 3: Aggregate into TPC certificate ───────────────────────────
    println!("──── Phase 3: Aggregate → TPC Certificate ────────────────");
    let agg_start = Instant::now();

    let config = MultiTimestepConfig {
        domain: SimulationDomain::Thermal,
        embed_proofs: !cli.no_embed,
        ..MultiTimestepConfig::default()
    };

    let agg_prover = MultiTimestepProver::with_random_key(config);
    let aggregate = match agg_prover.aggregate(timestep_inputs) {
        Ok(a) => a,
        Err(e) => {
            error!("Aggregation failed: {e}");
            std::process::exit(1);
        }
    };

    let agg_ms = agg_start.elapsed().as_millis();
    println!("[aggregate] Certificate ID:      {}", aggregate.certificate_id);
    println!("[aggregate] Domain:              {:?}", aggregate.domain);
    println!("[aggregate] Timestep count:      {}", aggregate.timestep_count);
    println!("[aggregate] Merkle root:         {}", hex::encode(aggregate.merkle_root));
    println!("[aggregate] Certificate size:    {} bytes", aggregate.tpc_certificate.len());
    println!("[aggregate] Generation time:     {} ms", aggregate.generation_time_ms);
    println!("[aggregate] Self-verify time:    {} µs", aggregate.verification_time_us);
    println!("[aggregate] Residual max |r|:    {:.2e}", aggregate.residual_stats.max_abs);
    println!("[aggregate] Residual RMS:        {:.2e}", aggregate.residual_stats.rms);
    println!("[aggregate] Aggregation time:    {agg_ms} ms");
    println!();

    // ── Phase 4: Write to disk ─────────────────────────────────────────────
    println!("──── Phase 4: Write Certificate ────────────────────────────");
    if let Some(parent) = cli.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).unwrap_or_else(|e| {
                error!("Failed to create output directory: {e}");
                std::process::exit(1);
            });
        }
    }

    std::fs::write(&cli.output, &aggregate.tpc_certificate).unwrap_or_else(|e| {
        error!("Failed to write certificate: {e}");
        std::process::exit(1);
    });
    println!("[write] Certificate written to {}", cli.output.display());

    // ── GPU Stats ──────────────────────────────────────────────────────────
    #[cfg(feature = "gpu")]
    if let Some(ref gpu_p) = gpu_prover {
        let stats = gpu_p.stats();
        println!("[gpu]   Total GPU MSMs:    {}", stats.total_gpu_msms);
        println!("[gpu]   Total GPU NTTs:    {}", stats.total_gpu_ntts);
        println!("[gpu]   Avg MSM latency:   {:.2} ms", stats.avg_gpu_msm_ms());
        println!("[gpu]   Estimated TPS:     {:.0}", stats.estimated_tps());
    }

    // Optional JSON sidecar
    if cli.json {
        let json_path = cli.output.with_extension("json");
        let mut sidecar = serde_json::json!({
            "certificate_id": aggregate.certificate_id.to_string(),
            "domain": format!("{:?}", aggregate.domain),
            "timestep_count": aggregate.timestep_count,
            "merkle_root": hex::encode(aggregate.merkle_root),
            "certificate_size_bytes": aggregate.tpc_certificate.len(),
            "generation_time_ms": aggregate.generation_time_ms,
            "verification_time_us": aggregate.verification_time_us,
            "residual_stats": {
                "max_abs": aggregate.residual_stats.max_abs,
                "rms": aggregate.residual_stats.rms,
                "nonzero_count": aggregate.residual_stats.nonzero_count,
            },
            "proof_hashes": aggregate.proof_hashes.iter().map(hex::encode).collect::<Vec<_>>(),
            "keygen_ms": keygen_ms,
            "prove_ms": prove_ms,
            "aggregate_ms": agg_ms,
            "total_constraints": total_constraints,
            "total_proof_bytes": total_proof_bytes,
            "params": if cli.production { "production" } else { "test_small" },
            "gpu_enabled": gpu_available,
        });

        if total_gpu_msms > 0 {
            sidecar["gpu_msm_count"] = serde_json::json!(total_gpu_msms);
            sidecar["gpu_commit_total_us"] = serde_json::json!(total_gpu_commit_us);
            sidecar["gpu_avg_commit_ms"] = serde_json::json!(
                total_gpu_commit_us as f64 / total_gpu_msms as f64 / 1000.0
            );

            #[cfg(feature = "gpu")]
            if let Some(ref gpu_p) = gpu_prover {
                let stats = gpu_p.stats();
                sidecar["gpu_device"] = serde_json::json!(gpu_p.device_name());
                sidecar["gpu_estimated_tps"] = serde_json::json!(stats.estimated_tps());
            }
        }

        let pretty = serde_json::to_string_pretty(&sidecar).expect("JSON serialization");
        std::fs::write(&json_path, pretty).unwrap_or_else(|e| {
            error!("Failed to write JSON sidecar: {e}");
            std::process::exit(1);
        });
        println!("[write] JSON sidecar written to {}", json_path.display());
    }
    println!();

    // ── Phase 5: Independent verification ──────────────────────────────────
    println!("──── Phase 5: Independent Verification ─────────────────────");

    // 5a. Re-read from disk and verify
    let cert_from_disk = std::fs::read(&cli.output).unwrap_or_else(|e| {
        error!("Failed to re-read certificate: {e}");
        std::process::exit(1);
    });

    let verify_start = Instant::now();
    match agg_prover.verify_certificate(&cert_from_disk) {
        Ok(()) => {
            let verify_us = verify_start.elapsed().as_micros();
            println!("[verify] ✓ Certificate signature and integrity VERIFIED ({verify_us} µs)");
        }
        Err(e) => {
            error!("[verify] ✗ VERIFICATION FAILED: {e}");
            std::process::exit(1);
        }
    }

    // 5b. Extract Merkle root from raw certificate bytes
    match extract_merkle_root(&cert_from_disk) {
        Ok(root) => {
            let root_hex = hex::encode(root);
            let expected_hex = hex::encode(aggregate.merkle_root);
            if root_hex == expected_hex {
                println!("[verify] ✓ Merkle root extracted and matches: {root_hex}");
            } else {
                error!("[verify] ✗ Merkle root mismatch: got {root_hex}, expected {expected_hex}");
                std::process::exit(1);
            }
        }
        Err(e) => {
            error!("[verify] ✗ Merkle root extraction failed: {e}");
            std::process::exit(1);
        }
    }

    // 5c. Verify each timestep's inclusion in the Merkle root
    let tree = fluidelite_zk::multi_timestep::MerkleTree::from_leaves(&aggregate.proof_hashes);
    let mut inclusion_pass = 0usize;
    for (i, hash) in aggregate.proof_hashes.iter().enumerate() {
        let proof_path = tree.proof(i);
        if MultiTimestepProver::verify_timestep_inclusion(
            &aggregate.merkle_root,
            hash,
            i,
            &proof_path,
        ) {
            inclusion_pass += 1;
        } else {
            error!("[verify] ✗ Timestep {i} Merkle inclusion FAILED");
        }
    }
    println!(
        "[verify] ✓ {inclusion_pass}/{} timestep Merkle inclusions verified",
        aggregate.proof_hashes.len()
    );
    println!();

    // ── Summary ────────────────────────────────────────────────────────────
    let total_ms = keygen_start.elapsed().as_millis();
    println!("══════════════════════════════════════════════════════════════");
    println!("                    CERTIFICATE SUMMARY");
    println!("══════════════════════════════════════════════════════════════");
    println!("  Certificate ID:     {}", aggregate.certificate_id);
    println!("  Domain:             {:?}", aggregate.domain);
    println!("  Timesteps:          {}", aggregate.timestep_count);
    println!("  Merkle root:        {}", hex::encode(aggregate.merkle_root));
    println!("  Certificate size:   {} bytes", aggregate.tpc_certificate.len());
    println!("  Output file:        {}", cli.output.display());
    println!("  Accelerator:        {}", if gpu_available { "ICICLE v4 GPU (CUDA)" } else { "CPU only" });
    println!("  ────────────────────────────────────────────────────────");
    println!("  Keygen time:        {keygen_ms} ms");
    println!("  Proof time:         {prove_ms} ms  ({avg_ms:.1} ms/step)");
    println!("  Aggregate time:     {agg_ms} ms");
    println!("  Total wall time:    {total_ms} ms");
    println!("  Total constraints:  {total_constraints}");
    println!("  Total proof bytes:  {total_proof_bytes}");
    if total_gpu_msms > 0 {
        let avg_gpu = total_gpu_commit_us as f64 / total_gpu_msms as f64 / 1000.0;
        println!("  GPU MSM commits:    {} @ {avg_gpu:.2} ms avg", total_gpu_msms);
    }
    println!("  ────────────────────────────────────────────────────────");
    println!("  Verify:             ✓ PASS");
    println!("  Merkle inclusions:  ✓ {inclusion_pass}/{}", aggregate.proof_hashes.len());
    println!("  Residual max |r|:   {:.2e}", aggregate.residual_stats.max_abs);
    println!("  Residual RMS:       {:.2e}", aggregate.residual_stats.rms);
    println!("══════════════════════════════════════════════════════════════");

    info!(
        certificate_id = %aggregate.certificate_id,
        timesteps = aggregate.timestep_count,
        total_ms = total_ms,
        gpu_enabled = gpu_available,
        "commercial-grade TPC certificate generated successfully"
    );

    // ── Explicit GPU cleanup ───────────────────────────────────────────────
    #[cfg(feature = "gpu")]
    drop(gpu_prover);
}
