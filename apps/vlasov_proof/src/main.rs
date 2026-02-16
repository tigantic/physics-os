//! Vlasov STARK Prover — Winterfell proof for QTT-native Vlasov 5D/6D
//!
//! Reads a per-step physics witness (JSON) produced by the Python Vlasov
//! solver and generates a real Winterfell STARK proof using the existing
//! chain-STARK AIR (`ThermalAir`).
//!
//! The chain STARK is physics-agnostic — it proves:
//!   1. Conservation: energy[n+1] = energy[n] + conservation_residual[n+1]
//!   2. State chain continuity: output_hash[n] == input_hash[n+1]
//!   3. Parameter constancy: dt, alpha constant across all rows
//!   4. Sequential execution: step_idx increments by 1
//!
//! For Vlasov, "energy" maps to ||f||₂² (L2 norm squared), and the
//! conservation residual captures the per-step norm drift after
//! renormalization. State hashes (SHA-256 of QTT cores) bind the
//! algebraic chain to actual tensor data.
//!
//! Output: VLASOV_PROOF.bin (serialized ThermalProof with THEP magic)
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use clap::Parser;
use fluidelite_circuits::thermal::{
    prove_thermal_stark, verify_thermal_stark, TimestepPhysics,
    ThermalProof,
};
use fluidelite_core::field::Q16;
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════════════
// CLI Arguments
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Parser, Debug)]
#[command(name = "vlasov-proof")]
#[command(about = "Generate STARK proof for Vlasov QTT-native PDE solver")]
struct Args {
    /// Path to the per-step witness JSON file
    #[arg(short, long)]
    witness: PathBuf,

    /// Output path for the STARK proof binary
    #[arg(short, long, default_value = "VLASOV_PROOF.bin")]
    output: PathBuf,

    /// Just verify an existing proof (don't generate)
    #[arg(long)]
    verify_only: bool,
}

// ═══════════════════════════════════════════════════════════════════════════
// Witness JSON Schema
// ═══════════════════════════════════════════════════════════════════════════

/// Top-level witness file produced by the Python solver.
#[derive(Deserialize, Debug)]
struct VlasovWitness {
    /// Physics type: "vlasov_poisson" or "vlasov_maxwell"
    physics: String,

    /// Phase-space dimensions (5 or 6)
    dims: usize,

    /// QTT qubits per dimension
    qubits_per_dim: usize,

    /// Maximum QTT rank
    max_rank: usize,

    /// Timestep size
    dt: f64,

    /// Number of timesteps
    n_steps: usize,

    /// Total grid points (N^dims)
    total_points: u64,

    /// Per-step physics data
    steps: Vec<StepWitness>,
}

/// Per-step physics witness.
#[derive(Deserialize, Debug)]
struct StepWitness {
    /// 0-indexed step number
    step: usize,

    /// ||f||₂² (L2 norm squared)
    norm_l2_sq: f64,

    /// Maximum absolute value in the QTT
    max_val: f64,

    /// Minimum absolute value
    min_val: f64,

    /// Total mass (∫f dx dv)
    #[allow(dead_code)]
    total_mass: f64,

    /// Conservation residual: norm_l2_sq[n] - norm_l2_sq[n-1]
    /// (zero for step 0)
    conservation_residual: f64,

    /// Effective QTT rank at this step
    rank: usize,

    /// SHA-256 of QTT core data (hex, 64 chars)
    state_hash: String,
}

// ═══════════════════════════════════════════════════════════════════════════
// Hash Conversion
// ═══════════════════════════════════════════════════════════════════════════

/// Parse a 64-char hex SHA-256 hash into 4 × u64 limbs (little-endian).
fn hash_hex_to_limbs(hex_str: &str) -> Result<[u64; 4], String> {
    let bytes = hex::decode(hex_str)
        .map_err(|e| format!("Invalid hex hash '{}': {}", hex_str, e))?;
    if bytes.len() != 32 {
        return Err(format!(
            "Hash must be 32 bytes (64 hex chars), got {} bytes",
            bytes.len()
        ));
    }
    let mut limbs = [0u64; 4];
    for (k, limb) in limbs.iter_mut().enumerate() {
        let offset = k * 8;
        *limb = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
    }
    Ok(limbs)
}

// ═══════════════════════════════════════════════════════════════════════════
// Witness → TimestepPhysics Conversion
// ═══════════════════════════════════════════════════════════════════════════

fn convert_witness(witness: &VlasovWitness) -> Result<Vec<TimestepPhysics>, String> {
    if witness.steps.len() < 2 {
        return Err(format!(
            "Need at least 2 steps (initial + 1 evolved), got {}",
            witness.steps.len()
        ));
    }

    let n = witness.steps.len();
    let mut physics = Vec::with_capacity(n);

    for i in 0..n {
        let step = &witness.steps[i];

        let state_hash_limbs = hash_hex_to_limbs(&step.state_hash)?;

        // For step i: input_hash = state_hash[i], output_hash = state_hash[i]
        // The chain STARK chain continuity constraint requires:
        //   output_hash[i] == input_hash[i+1]
        // Since we have state_hash[i] = hash of state AFTER step i,
        // we set: input_hash[i] = state_hash[i], output_hash[i] = state_hash[i+1]
        // For the last step, output_hash = state_hash[n-1] (final state).
        let input_hash_limbs = state_hash_limbs;
        let output_hash_limbs = if i + 1 < n {
            hash_hex_to_limbs(&witness.steps[i + 1].state_hash)?
        } else {
            state_hash_limbs // Last step: output == self
        };

        physics.push(TimestepPhysics {
            // Map Vlasov L2 norm² → "energy" (the conserved quantity)
            energy: Q16::from_f64(step.norm_l2_sq),
            energy_sq: Q16::from_f64(step.norm_l2_sq * step.norm_l2_sq),
            max_temp: Q16::from_f64(step.max_val),
            min_temp: Q16::from_f64(step.min_val),
            source_energy: Q16::ZERO, // No external source in Vlasov
            cg_residual: Q16::ZERO,   // No CG solve
            sv_max: Q16::from_f64(step.max_val),
            rank: step.rank,
            conservation_residual: Q16::from_f64(step.conservation_residual),
            input_hash_limbs,
            output_hash_limbs,
            global_step: step.step as u64,
        });
    }

    // Verify the chain: for each consecutive pair, output_hash[i] must equal
    // input_hash[i+1]. This is guaranteed by construction above, but let's
    // assert it for safety.
    for i in 0..n - 1 {
        if physics[i].output_hash_limbs != physics[i + 1].input_hash_limbs {
            return Err(format!(
                "State chain broken at step {}: output_hash != input_hash[{}]",
                i,
                i + 1
            ));
        }
    }

    Ok(physics)
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

fn main() {
    let args = Args::parse();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  VLASOV STARK PROVER — Winterfell (Goldilocks + FRI)       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ── Read witness ────────────────────────────────────────────────
    let witness_data = fs::read_to_string(&args.witness).unwrap_or_else(|e| {
        eprintln!("Failed to read witness file {:?}: {}", args.witness, e);
        std::process::exit(1);
    });

    let witness: VlasovWitness = serde_json::from_str(&witness_data).unwrap_or_else(|e| {
        eprintln!("Failed to parse witness JSON: {}", e);
        std::process::exit(1);
    });

    println!("  Witness loaded: {:?}", args.witness);
    println!("  Physics:      {}", witness.physics);
    println!("  Dimensions:   {}D", witness.dims);
    println!("  Grid:         2^{} per axis = {} total points", witness.qubits_per_dim, witness.total_points);
    println!("  Max rank:     {}", witness.max_rank);
    println!("  Timestep:     {}", witness.dt);
    println!("  Steps:        {} (+ initial state = {} trace rows)", witness.n_steps, witness.steps.len());
    println!();

    // ── Convert to TimestepPhysics ──────────────────────────────────
    let steps = convert_witness(&witness).unwrap_or_else(|e| {
        eprintln!("Witness conversion failed: {}", e);
        std::process::exit(1);
    });

    let initial_norm_sq = steps[0].energy.to_f64();
    let final_norm_sq = steps.last().unwrap().energy.to_f64();
    let norm_drift = ((final_norm_sq - initial_norm_sq) / initial_norm_sq).abs();

    println!("  State chain:  {} rows, all hash links valid", steps.len());
    println!("  Initial ||f||₂²: {:.10}", initial_norm_sq);
    println!("  Final ||f||₂²:   {:.10}", final_norm_sq);
    println!("  Norm drift:      {:.2e}", norm_drift);
    println!();

    // ── Generate STARK proof ────────────────────────────────────────
    // We use dt as the "timestep" and set alpha = 0 (no diffusion term
    // in the Vlasov equation — it's a pure advection equation).
    let dt = Q16::from_f64(witness.dt);
    let alpha = Q16::ZERO; // Vlasov: collisionless, no diffusion

    println!("  Generating STARK proof...");
    println!("    Field:       Goldilocks (p = 2^64 - 2^32 + 1)");
    println!("    Commitment:  FRI + Blake3 Merkle");
    println!("    Constraints: 8 × trace_length (degree-1 linear)");
    println!();

    let start = Instant::now();

    let (proof_bytes, pub_inputs, trace_len, _gen_ms) =
        prove_thermal_stark(&steps, dt, alpha).unwrap_or_else(|e| {
            eprintln!("STARK proof generation failed: {}", e);
            std::process::exit(1);
        });

    let total_gen_ms = start.elapsed().as_millis() as u64;

    let num_constraints = trace_len * 8; // 8 transition constraints per row

    println!("  ✓ STARK proof generated");
    println!("    Proof size:    {} bytes", proof_bytes.len());
    println!("    Trace length:  {} rows (padded to power-of-2)", trace_len);
    println!("    Constraints:   {} (8 × {})", num_constraints, trace_len);
    println!("    Generation:    {} ms", total_gen_ms);
    println!();

    // ── Verify STARK proof ──────────────────────────────────────────
    println!("  Verifying STARK proof...");
    let verify_start = Instant::now();

    let valid = verify_thermal_stark(&proof_bytes, pub_inputs.clone()).unwrap_or_else(|e| {
        eprintln!("STARK verification failed: {}", e);
        std::process::exit(1);
    });

    let verify_ms = verify_start.elapsed().as_millis();

    if !valid {
        eprintln!("  ✗ STARK verification returned false");
        std::process::exit(1);
    }

    println!("  ✓ STARK verification: VALID");
    println!("    Verification time: {} ms", verify_ms);
    println!();

    // ── Serialize proof ─────────────────────────────────────────────
    // Use the ThermalProof container with THEP magic for binary compat.
    let proof = ThermalProof {
        proof_bytes: proof_bytes.clone(),
        generation_time_ms: total_gen_ms,
        num_constraints,
        k: (trace_len as f64).log2().ceil() as u32,
        params: fluidelite_circuits::thermal::ThermalParams::test_small(), // placeholder
        conservation_residual: Q16::from_f64(norm_drift),
        cg_residual_norm: Q16::ZERO,
        cg_iterations: 0,
        input_state_hash_limbs: steps[0].input_hash_limbs,
        output_state_hash_limbs: steps.last().unwrap().output_hash_limbs,
        params_hash_limbs: [0; 4], // No params hash for Vlasov
        step_index: 0,
        initial_energy_raw: steps[0].energy.raw,
        final_energy_raw: steps.last().unwrap().energy.raw,
        contraction_proof_bytes: None,
        contraction_num_constraints: 0,
        contraction_generation_time_ms: 0,
    };

    let serialized = proof.to_bytes();
    fs::write(&args.output, &serialized).unwrap_or_else(|e| {
        eprintln!("Failed to write proof to {:?}: {}", args.output, e);
        std::process::exit(1);
    });

    // ── Also write raw STARK proof (just the FRI bytes, no THEP wrapper) ──
    let raw_path = args.output.with_extension("stark");
    fs::write(&raw_path, &proof_bytes).unwrap_or_else(|e| {
        eprintln!("Failed to write raw STARK proof: {}", e);
        std::process::exit(1);
    });

    // ── Summary ─────────────────────────────────────────────────────
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  PROOF SUMMARY                                             ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("  Physics:         {} ({}D)", witness.physics, witness.dims);
    println!("  Grid:            {}^{} = {} points", 1u64 << witness.qubits_per_dim, witness.dims, witness.total_points);
    println!("  Steps proven:    {}", witness.n_steps);
    println!("  Norm conserved:  {:.2e} relative drift", norm_drift);
    println!("  STARK proof:     {} bytes (FRI)", proof_bytes.len());
    println!("  Serialized:      {} bytes (THEP container)", serialized.len());
    println!("  Constraints:     {}", num_constraints);
    println!("  Prove time:      {} ms", total_gen_ms);
    println!("  Verify time:     {} ms", verify_ms);
    println!("  Output:          {:?}", args.output);
    println!("  Raw STARK:       {:?}", raw_path);
    println!();
    println!("  VERDICT: PROVEN ✓");
}
