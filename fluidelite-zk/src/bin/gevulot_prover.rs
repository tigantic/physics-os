//! Gevulot Prover Binary
//!
//! This binary runs as a Unikernel on the Gevulot network.
//! It reads input from /mnt/gevulot/task_input.file and writes
//! the ZK proof to /mnt/gevulot/task_output.file.
//!
//! # Deployment
//!
//! 1. Compile with MUSL: `cargo build --release --target x86_64-unknown-linux-musl --bin gevulot-prover`
//! 2. Package with OPS: `ops build target/.../gevulot-prover -c ops_config.json`
//! 3. Register with Gevulot: `gvltctl program deploy ...`

use std::fs;
use std::io::Write;
use std::time::Instant;

use serde::{Deserialize, Serialize};

#[cfg(feature = "halo2")]
use fluidelite_zk::{
    circuit::hybrid_unified::{BatchedHybridCircuit, TokenWitness},
    hybrid::HybridWeights,
};

#[cfg(feature = "halo2")]
use halo2_axiom::{
    halo2curves::bn256::{Bn256, Fr, G1Affine},
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

// Gevulot Firestarter protocol paths (v2 - correct mounts)
const INPUT_PATH: &str = "/mnt/gevulot/input/task_input.json";
const OUTPUT_PATH: &str = "/mnt/gevulot/output/proof_output.json";

// Model path (baked into the container image at root)
const MODEL_PATH: &str = "/fluidelite_v1.bin";

/// Input format from Gevulot network
#[derive(Deserialize)]
struct TaskInput {
    /// The context string to run inference on
    context: String,
    /// Optional: batch multiple contexts
    #[serde(default)]
    contexts: Vec<String>,
    /// Whether to include the full proof in output
    #[serde(default = "default_include_proof")]
    include_proof: bool,
}

fn default_include_proof() -> bool {
    true
}

/// Output format to Gevulot network
#[derive(Serialize)]
struct TaskOutput {
    /// Status of the operation
    status: String,
    /// Predicted tokens (one per context)
    predictions: Vec<PredictionResult>,
    /// The ZK proof (hex encoded)
    #[serde(skip_serializing_if = "Option::is_none")]
    proof: Option<String>,
    /// Proof size in bytes
    proof_size: usize,
    /// Time taken to generate proof (ms)
    prove_time_ms: u64,
    /// Time taken to verify proof (ms)
    verify_time_ms: u64,
    /// Whether the proof verified successfully
    verified: bool,
}

#[derive(Serialize)]
struct PredictionResult {
    /// Input context (truncated for display)
    context_preview: String,
    /// Predicted token ID
    token_id: u8,
    /// Predicted character (if ASCII printable)
    char: Option<char>,
    /// Path used: "Lookup" or "Arithmetic"
    path: String,
    /// Context hash (for verification)
    hash: String,
}

fn main() {
    #[cfg(not(feature = "halo2"))]
    {
        eprintln!("ERROR: This binary requires the 'halo2' feature.");
        eprintln!("Compile with: cargo build --release --features halo2 --bin gevulot-prover");
        std::process::exit(1);
    }
    
    #[cfg(feature = "halo2")]
    run_gevulot_prover();
}

#[cfg(feature = "halo2")]
fn run_gevulot_prover() {
    let start = Instant::now();
    
    // 1. Read input from Gevulot
    let input_str = match fs::read_to_string(INPUT_PATH) {
        Ok(s) => s,
        Err(e) => {
            write_error(&format!("Failed to read input file: {}", e));
            return;
        }
    };
    
    let input: TaskInput = match serde_json::from_str(&input_str) {
        Ok(i) => i,
        Err(e) => {
            write_error(&format!("Failed to parse input JSON: {}", e));
            return;
        }
    };
    
    // 2. Load model
    let weights = match HybridWeights::from_binary(MODEL_PATH) {
        Ok(w) => w,
        Err(e) => {
            write_error(&format!("Failed to load model: {}", e));
            return;
        }
    };
    
    let config = &weights.config;
    let context_len = config.context_len;
    
    // 3. Process contexts
    let mut contexts = input.contexts;
    if contexts.is_empty() && !input.context.is_empty() {
        contexts.push(input.context);
    }
    
    if contexts.is_empty() {
        write_error("No contexts provided");
        return;
    }
    
    // 4. Build token witnesses
    let mut tokens = Vec::new();
    let mut predictions = Vec::new();
    
    for ctx_str in &contexts {
        let ctx_bytes = ctx_str.as_bytes();
        
        // Pad or truncate to context_len
        let ctx: Vec<u8> = if ctx_bytes.len() >= context_len {
            ctx_bytes[ctx_bytes.len() - context_len..].to_vec()
        } else {
            let mut padded = vec![0u8; context_len - ctx_bytes.len()];
            padded.extend_from_slice(ctx_bytes);
            padded
        };
        
        let hash = HybridWeights::hash_context(&ctx);
        
        // Check if in lookup table
        if let Some(&pred) = weights.lookup_table.get(&hash) {
            tokens.push(TokenWitness::lookup(hash, pred));
            predictions.push(PredictionResult {
                context_preview: truncate_context(ctx_str, 32),
                token_id: pred,
                char: if pred.is_ascii_graphic() || pred == b' ' { 
                    Some(pred as char) 
                } else { 
                    None 
                },
                path: "Lookup".to_string(),
                hash: format!("{:016x}", hash),
            });
        } else {
            // Arithmetic fallback
            let (indices, values) = create_sparse_features(&ctx, config.feature_dim);
            let pred = compute_arithmetic_prediction(&indices, &values, &weights);
            
            tokens.push(TokenWitness::arithmetic(pred, indices, values));
            predictions.push(PredictionResult {
                context_preview: truncate_context(ctx_str, 32),
                token_id: pred,
                char: if pred.is_ascii_graphic() || pred == b' ' { 
                    Some(pred as char) 
                } else { 
                    None 
                },
                path: "Arithmetic".to_string(),
                hash: format!("{:016x}", hash),
            });
        }
    }
    
    // 5. Build lookup table for circuit (truncated for efficiency)
    let max_table = 10000;
    let table: Vec<(u64, u64, u8)> = weights.lookup_table
        .iter()
        .take(max_table)
        .map(|(&h, &p)| (h, 0u64, p))
        .collect();
    
    // 6. Create circuit
    let u_r: Vec<i32> = weights.u_r.iter().map(|q| q.raw as i32).collect();
    let s_r: Vec<i32> = weights.s_r.iter().map(|q| q.raw as i32).collect();
    let vt_r: Vec<i32> = weights.vt_r.iter().map(|q| q.raw as i32).collect();
    
    let circuit = BatchedHybridCircuit::new(
        tokens,
        table.clone(),
        config.rank,
        config.vocab_size,
        u_r,
        s_r,
        vt_r,
        config.feature_dim,
    );
    
    let public_inputs = circuit.public_inputs();
    
    // 7. Generate proof
    let k = 15u32;
    let params = ParamsKZG::<Bn256>::setup(k, OsRng);
    
    let empty_circuit = BatchedHybridCircuit::new(
        vec![],
        table,
        config.rank,
        config.vocab_size,
        vec![],
        vec![],
        vec![],
        config.feature_dim,
    );
    
    let vk = keygen_vk(&params, &empty_circuit).expect("keygen_vk failed");
    let pk = keygen_pk(&params, vk.clone(), &empty_circuit).expect("keygen_pk failed");
    
    let prove_start = Instant::now();
    let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);
    
    let proof_result = create_proof::<KZGCommitmentScheme<Bn256>, ProverGWC<_>, _, _, _, _>(
        &params,
        &pk,
        &[circuit],
        &[&[&public_inputs]],
        OsRng,
        &mut transcript,
    );
    
    match proof_result {
        Ok(()) => {
            let proof_bytes = transcript.finalize();
            let prove_time = prove_start.elapsed();
            
            // 8. Verify proof
            let verify_start = Instant::now();
            let strategy = SingleStrategy::new(&params);
            let mut verify_transcript = Blake2bRead::<_, G1Affine, Challenge255<_>>::init(&proof_bytes[..]);
            
            let verified = verify_proof::<KZGCommitmentScheme<Bn256>, VerifierGWC<_>, _, _, _>(
                &params,
                &vk,
                strategy,
                &[&[&public_inputs]],
                &mut verify_transcript,
            ).is_ok();
            
            let verify_time = verify_start.elapsed();
            
            // 9. Write output
            let output = TaskOutput {
                status: "success".to_string(),
                predictions,
                proof: if input.include_proof { 
                    Some(hex::encode(&proof_bytes)) 
                } else { 
                    None 
                },
                proof_size: proof_bytes.len(),
                prove_time_ms: prove_time.as_millis() as u64,
                verify_time_ms: verify_time.as_millis() as u64,
                verified,
            };
            
            write_output(&output);
        }
        Err(e) => {
            write_error(&format!("Proof generation failed: {:?}", e));
        }
    }
}

fn write_error(message: &str) {
    let output = TaskOutput {
        status: format!("error: {}", message),
        predictions: vec![],
        proof: None,
        proof_size: 0,
        prove_time_ms: 0,
        verify_time_ms: 0,
        verified: false,
    };
    write_output(&output);
}

fn write_output(output: &TaskOutput) {
    let json = serde_json::to_string_pretty(output).unwrap_or_else(|_| "{}".to_string());
    
    if let Ok(mut file) = fs::File::create(OUTPUT_PATH) {
        let _ = file.write_all(json.as_bytes());
    } else {
        // Fallback: print to stdout (for local testing)
        println!("{}", json);
    }
}

fn truncate_context(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("...{}", &s[s.len() - max_len..])
    }
}

#[cfg(feature = "halo2")]
fn create_sparse_features(context: &[u8], feature_dim: usize) -> (Vec<usize>, Vec<i32>) {
    let mut indices = Vec::new();
    let mut values = Vec::new();
    
    for (i, &byte) in context.iter().enumerate() {
        let unigram_idx = byte as usize;
        if unigram_idx < feature_dim {
            indices.push(unigram_idx);
            values.push(1 << 16);
        }
        
        if i > 0 {
            let prev = context[i - 1] as usize;
            let bigram_idx = 256 + (prev * 256 + byte as usize) % (feature_dim - 256);
            if bigram_idx < feature_dim {
                indices.push(bigram_idx);
                values.push(1 << 16);
            }
        }
    }
    
    let mut feature_map = std::collections::HashMap::new();
    for (idx, val) in indices.iter().zip(values.iter()) {
        *feature_map.entry(*idx).or_insert(0i32) += val;
    }
    
    let mut final_indices: Vec<usize> = feature_map.keys().copied().collect();
    final_indices.sort();
    let final_values: Vec<i32> = final_indices.iter().map(|i| feature_map[i]).collect();
    
    (final_indices, final_values)
}

#[cfg(feature = "halo2")]
fn compute_arithmetic_prediction(
    feature_indices: &[usize],
    feature_values: &[i32],
    weights: &HybridWeights,
) -> u8 {
    let rank = weights.config.rank;
    let vocab = weights.config.vocab_size;
    
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
    
    for r in 0..rank {
        h[r] >>= 16;
    }
    
    for r in 0..rank {
        if r < weights.s_r.len() {
            h[r] *= weights.s_r[r].raw as i64;
            h[r] >>= 16;
        }
    }
    
    let mut logits = vec![0i64; vocab];
    for r in 0..rank {
        for v in 0..vocab {
            let vt_idx = r * vocab + v;
            if vt_idx < weights.vt_r.len() {
                logits[v] += h[r] * weights.vt_r[vt_idx].raw as i64;
            }
        }
    }
    
    logits.iter()
        .enumerate()
        .max_by_key(|(_, &v)| v)
        .map(|(i, _)| i as u8)
        .unwrap_or(0)
}
