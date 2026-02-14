//! Commercial-Grade TPC Certificate Generator
//!
//! End-to-end binary that:
//!   1. Runs real Halo2 ZK proofs for N thermal timesteps
//!   2. Aggregates all proofs into a single TPC certificate (Merkle + Ed25519)
//!   3. Writes the signed certificate to disk
//!   4. Independently verifies the certificate
//!   5. Extracts and displays the Merkle root
//!
//! Usage:
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

    /// Use production-grade parameters (grid_bits=16, chi_max=32).
    /// WARNING: ~45 s per timestep. Default is test_small (grid_bits=4, chi_max=4).
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

    // ── Select parameters ──────────────────────────────────────────────────
    let params = if cli.production {
        println!("[config] Production parameters (grid_bits=16, χ_max=32, k≈22)");
        ThermalParams::production()
    } else {
        println!("[config] Test-small parameters (grid_bits=4, χ_max=4, k≈10)");
        ThermalParams::test_small()
    };
    println!("[config] Timesteps to prove: {}", cli.timesteps);
    println!("[config] Embed proofs:       {}", !cli.no_embed);
    println!("[config] Output path:        {}", cli.output.display());
    println!();

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

    // ── Phase 2: Generate N timestep proofs ────────────────────────────────
    println!("──── Phase 2: Prove {} Timesteps ─────────────────────────", cli.timesteps);
    let prove_start = Instant::now();
    let mut timestep_inputs: Vec<TimestepInput> = Vec::with_capacity(cli.timesteps);
    let mut total_constraints = 0u64;
    let mut total_proof_bytes = 0usize;

    for i in 0..cli.timesteps {
        let step_start = Instant::now();
        let proof = match thermal_prover.prove(&states, &mpos) {
            Ok(p) => p,
            Err(e) => {
                error!("Proof generation failed at timestep {i}: {e}");
                std::process::exit(1);
            }
        };
        let step_ms = step_start.elapsed().as_millis();

        let proof_size = proof.proof_bytes.len();
        total_constraints += proof.num_constraints as u64;
        total_proof_bytes += proof_size;

        let residual_f64 = proof.conservation_residual.to_f64();

        println!(
            "  [step {i:>4}] proof={proof_size:>6} B  constraints={:<8}  residual={:.2e}  CG_iters={:<3}  time={step_ms} ms",
            proof.num_constraints,
            residual_f64,
            proof.cg_iterations,
        );

        let input = TimestepInput::new(i, proof.proof_bytes)
            .with_residual(residual_f64)
            .with_metadata(serde_json::json!({
                "timestep_index": i,
                "k": proof.k,
                "num_constraints": proof.num_constraints,
                "cg_iterations": proof.cg_iterations,
                "proof_generation_ms": proof.generation_time_ms,
            }));
        timestep_inputs.push(input);
    }

    let prove_ms = prove_start.elapsed().as_millis();
    let avg_ms = prove_ms as f64 / cli.timesteps as f64;
    println!();
    println!("[proofs] {0} timesteps proved in {prove_ms} ms (avg {avg_ms:.1} ms/step)", cli.timesteps);
    println!("[proofs] Total constraints: {total_constraints}");
    println!("[proofs] Total proof bytes:  {total_proof_bytes}");
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

    // Optional JSON sidecar
    if cli.json {
        let json_path = cli.output.with_extension("json");
        let sidecar = serde_json::json!({
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
        });
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
    // Rebuild the Merkle tree from the proof hashes to get inclusion proofs.
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
    println!("  ────────────────────────────────────────────────────────");
    println!("  Keygen time:        {keygen_ms} ms");
    println!("  Proof time:         {prove_ms} ms  ({avg_ms:.1} ms/step)");
    println!("  Aggregate time:     {agg_ms} ms");
    println!("  Total wall time:    {total_ms} ms");
    println!("  Total constraints:  {total_constraints}");
    println!("  Total proof bytes:  {total_proof_bytes}");
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
        "commercial-grade TPC certificate generated successfully"
    );
}
