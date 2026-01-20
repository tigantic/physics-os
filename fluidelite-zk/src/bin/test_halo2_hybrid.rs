//! Test real Halo2 ZK proofs for FluidElite Hybrid
//!
//! This uses the actual Halo2 proving system to generate cryptographic proofs.
//!
//! Usage: cargo run --bin test-halo2-hybrid --features halo2

use fluidelite_zk::hybrid::HybridWeights;
use fluidelite_zk::halo2_hybrid_prover::{Halo2HybridProver, verify_hybrid_proof};
use std::fs;
use std::time::Instant;

fn main() {
    println!("============================================================");
    println!("FluidElite Hybrid - REAL Halo2 ZK Proofs");
    println!("============================================================\n");

    // Load the trained model
    let model_path = "../fluidelite/data/fluidelite_hybrid.bin";
    
    println!("Loading model from: {}", model_path);
    
    let weights = match HybridWeights::from_binary(model_path) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            eprintln!("\nMake sure you've trained the model first:");
            eprintln!("  python3 fluidelite/train_hybrid_triton.py");
            std::process::exit(1);
        }
    };
    
    println!("  Lookup table: {} entries", weights.lookup_table.len());
    println!("  Context length: {}", weights.config.context_len);
    
    // For testing, use a smaller subset of the lookup table
    // (7M entries would require ~24GB RAM for the full circuit)
    let max_table_entries = 10_000;
    
    // Collect hashes we're keeping
    let mut kept_hashes: std::collections::HashSet<u64> = std::collections::HashSet::new();
    
    let truncated_weights = if weights.lookup_table.len() > max_table_entries {
        println!("\n⚠️  Truncating lookup table to {} entries for testing", max_table_entries);
        println!("    (Full table would require excessive memory for proof)");
        
        // Create a new weights with truncated table
        let mut new_table = std::collections::HashMap::new();
        for (k, v) in weights.lookup_table.iter().take(max_table_entries) {
            new_table.insert(*k, *v);
            kept_hashes.insert(*k);
        }
        
        HybridWeights {
            config: weights.config.clone(),
            lookup_table: new_table,
            u_r: weights.u_r.clone(),
            s_r: weights.s_r.clone(),
            vt_r: weights.vt_r.clone(),
        }
    } else {
        for k in weights.lookup_table.keys() {
            kept_hashes.insert(*k);
        }
        weights
    };
    
    // Create prover (this does trusted setup - takes time)
    println!("\n============================================================");
    println!("Trusted Setup (one-time operation)");
    println!("============================================================\n");
    
    let setup_start = Instant::now();
    let mut prover = Halo2HybridProver::new(truncated_weights);
    println!("Setup complete in {:?}", setup_start.elapsed());
    
    // Load training data and find contexts that ARE in our truncated table
    let train_path = "../fluidelite/data/wikitext2_train.txt";
    let train_data = match fs::read(train_path) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("Could not load training data from {}", train_path);
            vec![]
        }
    };
    
    println!("\n============================================================");
    println!("Generating REAL ZK Proofs");
    println!("============================================================\n");
    
    // Find contexts that ARE in our truncated lookup table
    let l = 12; // context length
    
    println!("Searching for contexts in truncated lookup table...");
    let mut matching_positions: Vec<usize> = Vec::new();
    
    for pos in 0..(train_data.len().saturating_sub(l + 1)) {
        let ctx = &train_data[pos..pos + l];
        let hash = HybridWeights::hash_context(ctx);
        if kept_hashes.contains(&hash) {
            matching_positions.push(pos);
            if matching_positions.len() >= 10 {
                break;
            }
        }
    }
    
    println!("Found {} matching contexts\n", matching_positions.len());
    
    for &pos in matching_positions.iter().take(5) {
        if pos + l + 1 > train_data.len() {
            continue;
        }
        
        let ctx = &train_data[pos..pos + l];
        let ctx_str = String::from_utf8_lossy(ctx);
        
        println!("Testing context at position {}: \"{}\"", pos, ctx_str.replace('\n', "\\n"));
        
        match prover.prove(ctx) {
            Ok(proof) => {
                println!("  ✓ Proof generated!");
                println!("    Size: {} bytes", proof.size());
                println!("    Time: {} ms", proof.generation_time_ms);
                println!("    Constraints: {}", proof.num_constraints);
                println!("    Lookup hit: {}", proof.lookup_hit);
                
                // Verify the proof
                print!("    Verifying... ");
                match verify_hybrid_proof(
                    prover.params(),
                    prover.verifying_key(),
                    &proof,
                ) {
                    Ok(true) => println!("✓ Valid!"),
                    Ok(false) => println!("✗ Invalid"),
                    Err(e) => println!("Error: {}", e),
                }
            }
            Err(e) => {
                println!("  ✗ Proof failed: {}", e);
            }
        }
        println!();
    }
    
    // Statistics
    let stats = prover.stats();
    println!("============================================================");
    println!("Statistics");
    println!("============================================================");
    println!("  Total proofs: {}", stats.total_proofs);
    println!("  Lookup proofs: {}", stats.lookup_proofs);
    println!("  Fallback proofs: {}", stats.fallback_proofs);
    println!("  Total time: {} ms", stats.total_time_ms);
    
    println!("\n============================================================");
    println!("Halo2 ZK Proof Test Complete");
    println!("============================================================");
}
