//! FluidElite CLI
//!
//! Command-line interface for ZK proof generation and verification.
//!
//! # Usage
//!
//! ```bash
//! # Generate a proof
//! fluidelite-cli prove --token 42 --context context.bin --output proof.json
//!
//! # Verify a proof
//! fluidelite-cli verify --proof proof.json
//!
//! # Benchmark proof generation
//! fluidelite-cli bench --iterations 10
//!
//! # Generate verification key
//! fluidelite-cli keygen --output vk.bin
//! ```

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, Subcommand};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use fluidelite_zk::circuit::config::CircuitConfig;
use fluidelite_zk::field::Q16;
use fluidelite_zk::mpo::MPO;
use fluidelite_zk::mps::MPS;
use fluidelite_zk::prover::FluidEliteProver;

#[derive(Parser)]
#[command(name = "fluidelite-cli")]
#[command(author = "TiganticLabz")]
#[command(version = "0.1.0")]
#[command(about = "FluidElite ZK Proof Generation CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Verbose output
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate a ZK proof for an inference step
    Prove {
        /// Token ID to prove
        #[arg(short, long)]
        token: u64,

        /// Path to context MPS file (binary)
        #[arg(short, long)]
        context: Option<PathBuf>,

        /// Output proof file (JSON)
        #[arg(short, long, default_value = "proof.json")]
        output: PathBuf,

        /// Model weights file
        #[arg(short, long)]
        weights: Option<PathBuf>,
    },

    /// Verify a ZK proof
    Verify {
        /// Proof file to verify
        #[arg(short, long)]
        proof: PathBuf,

        /// Verification key file
        #[arg(short, long)]
        vk: Option<PathBuf>,
    },

    /// Benchmark proof generation
    Bench {
        /// Number of iterations
        #[arg(short, long, default_value = "5")]
        iterations: usize,

        /// Circuit size (k parameter, 2^k rows)
        #[arg(short, long, default_value = "10")]
        k: u32,
    },

    /// Generate proving/verification keys
    Keygen {
        /// Output file for verification key
        #[arg(short, long, default_value = "vk.bin")]
        output: PathBuf,

        /// Also output proving key (large file)
        #[arg(long)]
        pk: Option<PathBuf>,
    },

    /// Show circuit statistics
    Stats {
        /// Circuit configuration k (2^k rows)
        #[arg(short, long, default_value = "10")]
        k: u32,
    },
}

fn main() {
    let cli = Cli::parse();

    // Initialize logging
    let level = if cli.verbose { Level::DEBUG } else { Level::INFO };
    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    match cli.command {
        Commands::Prove { token, context, output, weights } => {
            cmd_prove(token, context, output, weights);
        }
        Commands::Verify { proof, vk } => {
            cmd_verify(proof, vk);
        }
        Commands::Bench { iterations, k } => {
            cmd_bench(iterations, k);
        }
        Commands::Keygen { output, pk } => {
            cmd_keygen(output, pk);
        }
        Commands::Stats { k } => {
            cmd_stats(k);
        }
    }
}

fn cmd_prove(token: u64, context_path: Option<PathBuf>, output: PathBuf, _weights: Option<PathBuf>) {
    info!("╔═══════════════════════════════════════════════════════╗");
    info!("║           FluidElite ZK Proof Generation              ║");
    info!("╚═══════════════════════════════════════════════════════╝");

    let config = CircuitConfig::default();
    info!("Circuit: k={}, sites={}, chi={}", config.k, config.num_sites, config.chi_max);

    // Load or create context
    let context = if let Some(path) = context_path {
        info!("Loading context from {}", path.display());
        let bytes = fs::read(&path).expect("Failed to read context file");
        bincode::deserialize(&bytes).expect("Failed to deserialize context")
    } else {
        info!("Using default context (no file provided)");
        MPS::new(config.num_sites, config.chi_max, config.phys_dim)
    };

    // Create prover with identity weights
    let w_hidden = MPO::identity(config.num_sites, config.phys_dim);
    let w_input = MPO::identity(config.num_sites, config.phys_dim);
    let readout_weights = vec![Q16::from_f64(0.1); config.chi_max * config.vocab_size];

    info!("Initializing prover...");
    let mut prover = FluidEliteProver::new(w_hidden, w_input, readout_weights, config);

    info!("Generating proof for token {}...", token);
    let start = Instant::now();
    
    let proof = prover.prove(&context, token).expect("Proof generation failed");
    
    let elapsed = start.elapsed();
    info!("✓ Proof generated in {:.2?}", elapsed);
    info!("  Proof size: {} bytes", proof.inner.proof_bytes.len());
    info!("  Generation time: {} ms", proof.inner.generation_time_ms);

    // Serialize proof to JSON
    let proof_json = serde_json::to_string_pretty(&ProofOutput {
        token_id: token,
        proof_bytes: base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &proof.inner.proof_bytes),
        public_inputs: proof.public_inputs.iter().map(|x| format!("{:?}", x)).collect(),
        generation_time_ms: proof.inner.generation_time_ms,
    }).expect("Failed to serialize proof");

    fs::write(&output, proof_json).expect("Failed to write proof file");
    info!("✓ Proof saved to {}", output.display());
}

fn cmd_verify(proof_path: PathBuf, _vk_path: Option<PathBuf>) {
    info!("╔═══════════════════════════════════════════════════════╗");
    info!("║           FluidElite ZK Proof Verification            ║");
    info!("╚═══════════════════════════════════════════════════════╝");

    let proof_json = fs::read_to_string(&proof_path).expect("Failed to read proof file");
    let proof: ProofOutput = serde_json::from_str(&proof_json).expect("Failed to parse proof");

    info!("Proof loaded:");
    info!("  Token ID: {}", proof.token_id);
    info!("  Proof size: {} bytes", proof.proof_bytes.len());
    info!("  Public inputs: {}", proof.public_inputs.len());

    // TODO: Implement actual verification with VK
    info!("⚠ Verification requires keygen to be run first");
    info!("  Use: fluidelite-cli keygen --output vk.bin");
}

fn cmd_bench(iterations: usize, k: u32) {
    info!("╔═══════════════════════════════════════════════════════╗");
    info!("║           FluidElite ZK Benchmark                     ║");
    info!("╚═══════════════════════════════════════════════════════╝");

    let mut config = CircuitConfig::default();
    config.k = k;

    info!("Configuration:");
    info!("  k = {} (2^{} = {} rows)", k, k, 1 << k);
    info!("  iterations = {}", iterations);
    info!("  sites = {}", config.num_sites);
    info!("  chi_max = {}", config.chi_max);

    // Create prover
    let w_hidden = MPO::identity(config.num_sites, config.phys_dim);
    let w_input = MPO::identity(config.num_sites, config.phys_dim);
    let readout_weights = vec![Q16::from_f64(0.1); config.chi_max * config.vocab_size];
    let mut prover = FluidEliteProver::new(w_hidden, w_input, readout_weights, config.clone());

    let context = MPS::new(config.num_sites, config.chi_max, config.phys_dim);

    info!("\nRunning {} iterations...\n", iterations);

    let mut times = Vec::with_capacity(iterations);
    let mut proof_sizes = Vec::with_capacity(iterations);

    for i in 0..iterations {
        let token = (i as u64) % 256;
        let start = Instant::now();
        
        let proof = prover.prove(&context, token).expect("Proof generation failed");
        
        let elapsed = start.elapsed();
        times.push(elapsed.as_millis() as u64);
        proof_sizes.push(proof.inner.proof_bytes.len());

        info!("  [{}] token={:3} time={:>6}ms size={:>5} bytes", 
              i + 1, token, elapsed.as_millis(), proof.inner.proof_bytes.len());
    }

    // Calculate statistics
    let avg_time = times.iter().sum::<u64>() as f64 / times.len() as f64;
    let min_time = *times.iter().min().unwrap();
    let max_time = *times.iter().max().unwrap();
    let avg_size = proof_sizes.iter().sum::<usize>() / proof_sizes.len();

    info!("\n═══════════════════════════════════════════════════════");
    info!("                    BENCHMARK RESULTS                   ");
    info!("═══════════════════════════════════════════════════════");
    info!("  Iterations:     {:>10}", iterations);
    info!("  Avg time:       {:>10.2} ms", avg_time);
    info!("  Min time:       {:>10} ms", min_time);
    info!("  Max time:       {:>10} ms", max_time);
    info!("  Avg proof size: {:>10} bytes", avg_size);
    info!("  Throughput:     {:>10.2} proofs/sec", 1000.0 / avg_time);
    info!("═══════════════════════════════════════════════════════");
}

fn cmd_keygen(output: PathBuf, pk_path: Option<PathBuf>) {
    info!("╔═══════════════════════════════════════════════════════╗");
    info!("║           FluidElite Key Generation                   ║");
    info!("╚═══════════════════════════════════════════════════════╝");

    let config = CircuitConfig::default();
    info!("Configuration: k={}", config.k);

    info!("Generating keys (this may take a while)...");
    let start = Instant::now();

    // TODO: Implement actual key generation and serialization
    // For now, create placeholder
    let vk_placeholder = vec![0u8; 1024];
    
    fs::write(&output, &vk_placeholder).expect("Failed to write verification key");
    info!("✓ Verification key saved to {} ({} bytes)", output.display(), vk_placeholder.len());

    if let Some(pk) = pk_path {
        let pk_placeholder = vec![0u8; 10240];
        fs::write(&pk, &pk_placeholder).expect("Failed to write proving key");
        info!("✓ Proving key saved to {} ({} bytes)", pk.display(), pk_placeholder.len());
    }

    info!("Key generation completed in {:.2?}", start.elapsed());
}

fn cmd_stats(k: u32) {
    info!("╔═══════════════════════════════════════════════════════╗");
    info!("║           FluidElite Circuit Statistics               ║");
    info!("╚═══════════════════════════════════════════════════════╝");

    let mut config = CircuitConfig::default();
    config.k = k;

    let rows = 1u64 << k;
    let constraints = config.estimate_constraints();

    info!("Circuit Parameters:");
    info!("  k (log2 rows):      {:>10}", k);
    info!("  Total rows:         {:>10}", rows);
    info!("  Est. constraints:   {:>10}", constraints);
    info!("  Num sites:          {:>10}", config.num_sites);
    info!("  Chi max:            {:>10}", config.chi_max);
    info!("  Physical dim:       {:>10}", config.phys_dim);
    info!("  Vocab size:         {:>10}", config.vocab_size);
    info!("");
    info!("Estimated Resources:");
    info!("  Proof size:         ~{} KB", (constraints as f64 * 0.1) as u64 / 1024);
    info!("  Proof time:         ~{} ms", (constraints as f64 * 0.001) as u64);
    info!("  Memory (proving):   ~{} MB", (rows as f64 * 32.0) as u64 / 1024 / 1024);
}

/// Proof output format for JSON serialization
#[derive(serde::Serialize, serde::Deserialize)]
struct ProofOutput {
    token_id: u64,
    proof_bytes: String,  // base64 encoded
    public_inputs: Vec<String>,
    generation_time_ms: u64,
}
