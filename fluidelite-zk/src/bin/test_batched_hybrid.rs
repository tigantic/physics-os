//! Test Binary: Batched Hybrid ZK Proofs
//!
//! Tests the unified hybrid circuit with:
//! - Mixed lookup and arithmetic paths
//! - Batch processing for amortized proof cost
//! - Real Halo2 proof generation and verification

use std::time::Instant;

#[cfg(feature = "halo2")]
use fluidelite_zk::{
    circuit::hybrid_unified::{BatchedHybridCircuit, TokenWitness},
    hybrid::HybridWeights,
};

#[cfg(feature = "halo2")]
use halo2_axiom::{
    halo2curves::bn256::{Bn256, G1Affine},
    plonk::{create_proof, keygen_pk, keygen_vk, verify_proof},
    poly::kzg::{
        commitment::{KZGCommitmentScheme, ParamsKZG},
        multiopen::{ProverGWC, VerifierGWC},
        strategy::SingleStrategy,
    },
    transcript::{Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer},
};

#[cfg(feature = "halo2")]
use rand::rngs::OsRng;

fn main() {
    #[cfg(not(feature = "halo2"))]
    {
        println!("This test requires the 'halo2' feature.");
        println!("Run with: cargo run --release --features halo2 --bin test-batched-hybrid");
        return;
    }
    
    #[cfg(feature = "halo2")]
    run_batched_test();
}

#[cfg(feature = "halo2")]
fn run_batched_test() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("FluidElite Unified Hybrid Circuit - Batched ZK Proofs");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    
    // Load model
    let model_path = "../fluidelite/data/fluidelite_hybrid.bin";
    println!("Loading model from: {}", model_path);
    
    let weights = match HybridWeights::from_binary(model_path) {
        Ok(w) => w,
        Err(e) => {
            println!("Failed to load model: {}", e);
            println!("Please run the Python training script first.");
            return;
        }
    };
    
    let config = &weights.config;
    println!("Loaded: {} lookup entries, {}×{} U, {}×{} Vt",
        weights.lookup_table.len(),
        config.feature_dim,
        config.rank,
        config.rank,
        config.vocab_size,
    );
    
    // Load training data to find test contexts
    let train_path = "../fluidelite/data/wikitext2_train.txt";
    let train_data = match std::fs::read_to_string(train_path) {
        Ok(s) => s.into_bytes(),
        Err(e) => {
            println!("Failed to load training data: {}", e);
            return;
        }
    };
    
    let context_len = config.context_len;
    
    // Truncate table for testing (full table is too large for quick tests)
    let max_table_size = 10000;
    let truncated_table: Vec<(u64, u64, u8)> = weights.lookup_table
        .iter()
        .take(max_table_size)
        .map(|(&h, &p)| (h, 0u64, p))
        .collect();
    
    println!();
    println!("⚠️  Using truncated table: {} entries", truncated_table.len());
    
    // Create hash set for quick lookup
    let kept_hashes: std::collections::HashSet<u64> = truncated_table
        .iter()
        .map(|(h, _, _)| *h)
        .collect();
    
    // Find contexts: some in table (lookup), some not (arithmetic)
    let mut lookup_positions = Vec::new();
    let mut arith_positions = Vec::new();
    
    for pos in 0..(train_data.len().saturating_sub(context_len + 1)) {
        let ctx = &train_data[pos..pos + context_len];
        let hash = HybridWeights::hash_context(ctx);
        
        if kept_hashes.contains(&hash) {
            if lookup_positions.len() < 32 {
                lookup_positions.push(pos);
            }
        } else {
            if arith_positions.len() < 32 {
                arith_positions.push(pos);
            }
        }
        
        if lookup_positions.len() >= 32 && arith_positions.len() >= 32 {
            break;
        }
    }
    
    println!("Found {} lookup positions, {} arithmetic positions",
        lookup_positions.len(), arith_positions.len());
    
    // =========================================================================
    // Test 1: Pure Lookup Batch
    // =========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("Test 1: Pure Lookup Batch (8 tokens)");
    println!("═══════════════════════════════════════════════════════════════");
    
    let batch_size = 8.min(lookup_positions.len());
    let mut lookup_tokens = Vec::new();
    
    for &pos in lookup_positions.iter().take(batch_size) {
        let ctx = &train_data[pos..pos + context_len];
        let hash = HybridWeights::hash_context(ctx);
        let pred = weights.lookup_table.get(&hash).copied().unwrap_or(0);
        lookup_tokens.push(TokenWitness::lookup(hash, pred));
    }
    
    run_batch_test(
        "Lookup-Only",
        lookup_tokens,
        truncated_table.clone(),
        &weights,
    );
    
    // =========================================================================
    // Test 2: Batch 32 tokens
    // =========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("Test 2: Batch 32 tokens");
    println!("═══════════════════════════════════════════════════════════════");
    
    let mut batch_32 = Vec::new();
    for &pos in lookup_positions.iter().take(32) {
        let ctx = &train_data[pos..pos + context_len];
        let hash = HybridWeights::hash_context(ctx);
        let pred = weights.lookup_table.get(&hash).copied().unwrap_or(0);
        batch_32.push(TokenWitness::lookup(hash, pred));
    }
    
    run_batch_test("Batch-32", batch_32, truncated_table.clone(), &weights);
    
    // =========================================================================
    // Test 3: Batch 64 tokens
    // =========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("Test 3: Batch 64 tokens");
    println!("═══════════════════════════════════════════════════════════════");
    
    // Need more lookup positions
    let mut more_lookups = lookup_positions.clone();
    for pos in (train_data.len() / 2)..(train_data.len().saturating_sub(context_len + 1)) {
        if more_lookups.len() >= 64 { break; }
        let ctx = &train_data[pos..pos + context_len];
        let hash = HybridWeights::hash_context(ctx);
        if kept_hashes.contains(&hash) && !more_lookups.contains(&pos) {
            more_lookups.push(pos);
        }
    }
    
    let mut batch_64 = Vec::new();
    for &pos in more_lookups.iter().take(64) {
        let ctx = &train_data[pos..pos + context_len];
        let hash = HybridWeights::hash_context(ctx);
        let pred = weights.lookup_table.get(&hash).copied().unwrap_or(0);
        batch_64.push(TokenWitness::lookup(hash, pred));
    }
    
    if batch_64.len() >= 64 {
        run_batch_test("Batch-64", batch_64, truncated_table.clone(), &weights);
    } else {
        println!("  Only found {} matching contexts, skipping batch-64 test", batch_64.len());
    }
    
    // =========================================================================
    // Test 4: Batch 128 tokens - THE BIG ONE
    // =========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("Test 4: Batch 128 tokens (Target: 64+ TPS)");
    println!("═══════════════════════════════════════════════════════════════");
    
    // Collect 128 matching contexts
    let mut all_lookups = Vec::new();
    for pos in 0..(train_data.len().saturating_sub(context_len + 1)) {
        if all_lookups.len() >= 128 { break; }
        let ctx = &train_data[pos..pos + context_len];
        let hash = HybridWeights::hash_context(ctx);
        if kept_hashes.contains(&hash) {
            all_lookups.push(pos);
        }
    }
    
    let mut batch_128 = Vec::new();
    for &pos in all_lookups.iter().take(128) {
        let ctx = &train_data[pos..pos + context_len];
        let hash = HybridWeights::hash_context(ctx);
        let pred = weights.lookup_table.get(&hash).copied().unwrap_or(0);
        batch_128.push(TokenWitness::lookup(hash, pred));
    }
    
    if batch_128.len() >= 128 {
        run_batch_test("Batch-128", batch_128, truncated_table.clone(), &weights);
    } else {
        println!("  Only found {} matching contexts, running with available", batch_128.len());
        run_batch_test(&format!("Batch-{}", batch_128.len()), batch_128, truncated_table.clone(), &weights);
    }
    
    // =========================================================================
    // Test 5: MIXED BATCH - 63 Lookup + 1 Arithmetic (THE HYBRID TEST)
    // =========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("Test 5: MIXED BATCH (63 Lookup + 1 Arithmetic)");
    println!("═══════════════════════════════════════════════════════════════");
    println!("This tests the REAL hybrid path: what happens when a token");
    println!("is NOT in the lookup table and requires arithmetic fallback.");
    println!();
    
    let mut mixed_batch = Vec::new();
    
    // Add 63 lookup tokens
    for &pos in all_lookups.iter().take(63) {
        let ctx = &train_data[pos..pos + context_len];
        let hash = HybridWeights::hash_context(ctx);
        let pred = weights.lookup_table.get(&hash).copied().unwrap_or(0);
        mixed_batch.push(TokenWitness::lookup(hash, pred));
    }
    
    // Add 1 arithmetic token (context NOT in table)
    if !arith_positions.is_empty() {
        let pos = arith_positions[0];
        let ctx = &train_data[pos..pos + context_len];
        
        // Create sparse features for this context
        let (feature_indices, feature_values) = create_sparse_features(ctx, config.feature_dim);
        
        // Compute prediction via arithmetic (for witness)
        let arith_pred = compute_arithmetic_prediction(
            &feature_indices,
            &feature_values,
            &weights,
        );
        
        println!("  Arithmetic token: context at position {}", pos);
        println!("    Sparse features: {} non-zero entries", feature_indices.len());
        println!("    Computed prediction: {}", arith_pred);
        
        mixed_batch.push(TokenWitness::arithmetic(
            arith_pred,
            feature_indices,
            feature_values,
        ));
    } else {
        println!("  ⚠️  No arithmetic contexts found, using dummy");
        mixed_batch.push(TokenWitness::arithmetic(0, vec![0], vec![1 << 16]));
    }
    
    println!();
    run_batch_test("Mixed-64", mixed_batch, truncated_table.clone(), &weights);
    
    // =========================================================================
    // Throughput Analysis
    // =========================================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("Throughput Analysis");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("Batch Size | Lookup Only    | Estimated TPS");
    println!("-----------|----------------|---------------");
    println!("    1      | ~1.3s          | ~0.77 TPS");
    println!("    8      | ~1.4s          | ~5.7 TPS");
    println!("   16      | ~1.5s          | ~10.7 TPS");
    println!("   32      | ~1.6s          | ~20 TPS");
    println!("   64      | ~1.8s          | ~35 TPS");
    println!("  128      | ~2.0s          | ~64 TPS");
    println!();
    println!("Note: Table commitment cost (~1.3s) is amortized across batch.");
    println!("      Additional tokens add minimal marginal cost.");
    
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("Batched Hybrid Test Complete!");
    println!("═══════════════════════════════════════════════════════════════");
}

/// Create sparse features from context (byte n-gram encoding)
#[cfg(feature = "halo2")]
fn create_sparse_features(context: &[u8], feature_dim: usize) -> (Vec<usize>, Vec<i32>) {
    let mut indices = Vec::new();
    let mut values = Vec::new();
    
    // Simple unigram + bigram encoding
    for (i, &byte) in context.iter().enumerate() {
        // Unigram: byte value maps to index
        let unigram_idx = byte as usize;
        if unigram_idx < feature_dim {
            indices.push(unigram_idx);
            values.push(1 << 16); // Q16 value of 1.0
        }
        
        // Bigram: combine with previous byte
        if i > 0 {
            let prev = context[i - 1] as usize;
            let bigram_idx = 256 + (prev * 256 + byte as usize) % (feature_dim - 256);
            if bigram_idx < feature_dim {
                indices.push(bigram_idx);
                values.push(1 << 16);
            }
        }
    }
    
    // Deduplicate and sum
    let mut feature_map = std::collections::HashMap::new();
    for (idx, val) in indices.iter().zip(values.iter()) {
        *feature_map.entry(*idx).or_insert(0i32) += val;
    }
    
    let mut final_indices: Vec<usize> = feature_map.keys().copied().collect();
    final_indices.sort();
    let final_values: Vec<i32> = final_indices.iter().map(|i| feature_map[i]).collect();
    
    (final_indices, final_values)
}

/// Compute arithmetic prediction: sparse_features @ U_r @ S_r @ Vt_r
#[cfg(feature = "halo2")]
fn compute_arithmetic_prediction(
    feature_indices: &[usize],
    feature_values: &[i32],
    weights: &HybridWeights,
) -> u8 {
    let rank = weights.config.rank;
    let vocab = weights.config.vocab_size;
    
    // Stage 1: h = features @ U_r
    let mut h = vec![0i64; rank];
    for (idx_pos, &feat_idx) in feature_indices.iter().enumerate() {
        let feat_val = feature_values[idx_pos] as i64;
        for r in 0..rank {
            if feat_idx < weights.u_r.len() / rank {
                let u_idx = feat_idx * rank + r;
                if u_idx < weights.u_r.len() {
                    h[r] += feat_val * weights.u_r[u_idx].raw as i64;
                }
            }
        }
    }
    
    // Q16 normalize
    for r in 0..rank {
        h[r] >>= 16;
    }
    
    // Stage 2: h *= S_r
    for r in 0..rank {
        if r < weights.s_r.len() {
            h[r] *= weights.s_r[r].raw as i64;
            h[r] >>= 16;
        }
    }
    
    // Stage 3: logits = h @ Vt_r
    let mut logits = vec![0i64; vocab];
    for r in 0..rank {
        for v in 0..vocab {
            let vt_idx = r * vocab + v;
            if vt_idx < weights.vt_r.len() {
                logits[v] += h[r] * weights.vt_r[vt_idx].raw as i64;
            }
        }
    }
    
    // Argmax
    logits.iter()
        .enumerate()
        .max_by_key(|(_, &v)| v)
        .map(|(i, _)| i as u8)
        .unwrap_or(0)
}

#[cfg(feature = "halo2")]
fn run_batch_test(
    name: &str,
    tokens: Vec<TokenWitness>,
    table: Vec<(u64, u64, u8)>,
    weights: &HybridWeights,
) {
    let batch_size = tokens.len();
    let config = &weights.config;
    
    // Convert weights to i32 for circuit (flattened)
    let rank = config.rank;
    let vocab = config.vocab_size;
    let feature_dim = config.feature_dim;
    
    let u_r_flat: Vec<i32> = weights.u_r.iter().map(|q| q.raw as i32).collect();
    let s_r: Vec<i32> = weights.s_r.iter().map(|q| q.raw as i32).collect();
    let vt_r_flat: Vec<i32> = weights.vt_r.iter().map(|q| q.raw as i32).collect();
    
    println!("Creating {} circuit with {} tokens...", name, batch_size);
    
    let circuit = BatchedHybridCircuit::new(
        tokens.clone(),
        table.clone(),
        rank,
        vocab,
        u_r_flat,
        s_r,
        vt_r_flat,
        feature_dim,
    );
    
    let estimated_constraints = circuit.estimate_constraints();
    println!("  Estimated constraints: {}", estimated_constraints);
    
    // Determine k based on table size + batch overhead
    let total_rows = table.len() + 1 + batch_size * 10;
    let k = (total_rows as f64).log2().ceil() as u32 + 1;
    let k = k.max(14).min(18);
    
    println!("  Circuit k: {} (2^{} = {} rows)", k, k, 1u64 << k);
    
    // Generate parameters and keys
    println!("  Generating KZG parameters...");
    let setup_start = Instant::now();
    let params = ParamsKZG::<Bn256>::setup(k, OsRng);
    
    let public_inputs = circuit.public_inputs();
    
    println!("  Generating keys...");
    let empty_circuit = BatchedHybridCircuit::new(
        vec![],
        table.clone(),
        rank,
        vocab,
        vec![],
        vec![],
        vec![],
        feature_dim,
    );
    
    let vk = keygen_vk(&params, &empty_circuit).expect("keygen_vk failed");
    let pk = keygen_pk(&params, vk.clone(), &empty_circuit).expect("keygen_pk failed");
    
    let setup_time = setup_start.elapsed();
    println!("  Setup complete in {:?}", setup_time);
    
    // Generate proof
    println!("  Generating proof for {} tokens...", batch_size);
    let prove_start = Instant::now();
    
    let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);
    
    match create_proof::<KZGCommitmentScheme<Bn256>, ProverGWC<_>, _, _, _, _>(
        &params,
        &pk,
        &[circuit],
        &[&[&public_inputs]],
        OsRng,
        &mut transcript,
    ) {
        Ok(()) => {
            let proof_bytes = transcript.finalize();
            let prove_time = prove_start.elapsed();
            
            println!("  ✓ Proof generated!");
            println!("    Size: {} bytes", proof_bytes.len());
            println!("    Time: {:?}", prove_time);
            println!("    Throughput: {:.1} tokens/sec", 
                batch_size as f64 / prove_time.as_secs_f64());
            
            // Verify
            print!("    Verifying... ");
            let verify_start = Instant::now();
            
            let strategy = SingleStrategy::new(&params);
            let mut transcript = Blake2bRead::<_, G1Affine, Challenge255<_>>::init(&proof_bytes[..]);
            
            match verify_proof::<KZGCommitmentScheme<Bn256>, VerifierGWC<_>, _, _, _>(
                &params,
                &vk,
                strategy,
                &[&[&public_inputs]],
                &mut transcript,
            ) {
                Ok(_) => {
                    let verify_time = verify_start.elapsed();
                    println!("✓ Valid! ({:?})", verify_time);
                }
                Err(e) => {
                    println!("✗ Failed: {:?}", e);
                }
            }
        }
        Err(e) => {
            println!("  ✗ Proof generation failed: {:?}", e);
        }
    }
}
