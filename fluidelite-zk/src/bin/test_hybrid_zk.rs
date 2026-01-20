//! Test ZK proof generation for FluidElite Hybrid
//!
//! Usage: cargo run --bin test-hybrid-zk

use fluidelite_zk::hybrid::HybridWeights;
use fluidelite_zk::hybrid_prover::HybridProver;
use std::fs;
use std::time::Instant;

fn main() {
    println!("============================================================");
    println!("FluidElite Hybrid ZK Proof Test");
    println!("============================================================\n");

    // Load the trained model
    let model_path = "../fluidelite/data/fluidelite_hybrid.bin";
    
    println!("Loading model from: {}", model_path);
    let start = Instant::now();
    
    let weights = match HybridWeights::from_binary(model_path) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            eprintln!("\nMake sure you've trained the model first:");
            eprintln!("  python3 fluidelite/train_hybrid_triton.py");
            std::process::exit(1);
        }
    };
    
    println!("Loaded in {:?}", start.elapsed());
    println!("  Lookup table: {} entries", weights.lookup_table.len());
    println!("  Context length: {}", weights.config.context_len);
    println!("  Rank: {}", weights.config.rank);
    println!("  Feature dim: {}", weights.config.feature_dim);
    
    // Create prover
    let mut prover = HybridProver::new(weights);
    
    // Load training data for real context tests
    let train_path = "../fluidelite/data/wikitext2_train.txt";
    let train_data = match fs::read(train_path) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("Could not load training data from {}", train_path);
            vec![]
        }
    };
    
    println!("\n============================================================");
    println!("Testing with REAL Training Data Contexts");
    println!("============================================================\n");
    
    // Test actual contexts from training data (should hit lookup table)
    let test_positions = [0usize, 100, 1000, 10000, 100000, 500000, 1000000];
    let l = prover.config().context_len;
    
    let mut lookup_hits = 0;
    let mut total_tests = 0;
    
    for &pos in &test_positions {
        if pos + l + 1 > train_data.len() {
            continue;
        }
        
        let ctx = &train_data[pos..pos + l];
        let expected = train_data[pos + l];
        
        // Run inference
        let result = prover.infer(ctx);
        
        // Generate proof
        let proof = prover.prove(ctx);
        
        let ctx_str = String::from_utf8_lossy(ctx);
        println!("Position {}: \"{}\"", pos, ctx_str.replace('\n', "\\n"));
        println!("  Expected: {} ('{}')", expected, char_display(expected));
        println!("  Predicted: {} ('{}')", result.prediction, char_display(result.prediction));
        println!("  Correct: {}", result.prediction == expected);
        println!("  Lookup hit: {}", result.lookup_hit);
        println!("  Proof constraints: {}", proof.num_constraints);
        println!("  Proof size: {} bytes", proof.size());
        println!();
        
        if result.lookup_hit {
            lookup_hits += 1;
        }
        total_tests += 1;
    }
    
    println!("Lookup hit rate: {}/{} ({:.1}%)", 
        lookup_hits, total_tests, 
        100.0 * lookup_hits as f64 / total_tests as f64);
    
    // Statistics (clone to avoid borrow issues)
    let stats_snapshot = prover.stats().clone();
    println!("\n============================================================");
    println!("Statistics");
    println!("============================================================");
    println!("  Total proofs: {}", stats_snapshot.total_proofs);
    println!("  Lookup hits: {} ({:.1}%)", 
        stats_snapshot.lookup_hits, 
        stats_snapshot.lookup_rate() * 100.0
    );
    println!("  Fallback uses: {}", stats_snapshot.fallback_uses);
    
    // Benchmark with training data contexts
    println!("\n============================================================");
    println!("Benchmark: 1000 proofs from training data");
    println!("============================================================\n");
    
    let start = Instant::now();
    let mut bench_lookup_hits = 0;
    
    for i in 0..1000 {
        let pos = (i * 1000) % (train_data.len() - l - 1);
        let ctx = &train_data[pos..pos + l];
        let proof = prover.prove(ctx);
        if proof.lookup_hit {
            bench_lookup_hits += 1;
        }
    }
    let elapsed = start.elapsed();
    
    println!("1000 proofs in {:?}", elapsed);
    println!("Average: {:.2} µs/proof", elapsed.as_micros() as f64 / 1000.0);
    println!("Throughput: {:.0} proofs/sec", 1000.0 / elapsed.as_secs_f64());
    println!("Lookup hit rate: {}/1000 ({:.1}%)", bench_lookup_hits, bench_lookup_hits as f64 / 10.0);
    
    let final_stats = prover.stats().clone();
    let avg_constraints = if final_stats.total_proofs > 0 {
        final_stats.total_constraints as f64 / final_stats.total_proofs as f64
    } else {
        0.0
    };
    println!("Average constraints: {:.0}", avg_constraints);
    
    // Compare lookup vs fallback cost
    println!("\n============================================================");
    println!("Constraint Analysis");
    println!("============================================================");
    println!("  Lookup path: ~80 constraints (hash + membership)");
    println!("  Fallback path: ~50,000 constraints (features + matmul)");
    let lookup_rate = final_stats.lookup_rate();
    println!("  With {:.1}% lookup hits:", lookup_rate * 100.0);
    let effective_constraints = lookup_rate * 80.0 + (1.0 - lookup_rate) * 50000.0;
    println!("  Effective avg constraints: {:.0}", effective_constraints);
    println!("  Savings vs pure fallback: {:.1}x", 50000.0 / effective_constraints);
    
    println!("\n============================================================");
    println!("ZK Proof Test Complete");
    println!("============================================================");
}

fn char_display(b: u8) -> char {
    if b.is_ascii_graphic() || b == b' ' {
        b as char
    } else if b == b'\n' {
        '↵'
    } else {
        '?'
    }
}
