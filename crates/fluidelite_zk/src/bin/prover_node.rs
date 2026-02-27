//! FluidElite Prover Node
//!
//! Standalone service that connects to prover networks (Gevulot, Succinct)
//! and earns fees by generating ZK proofs for FluidElite inference.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                      Prover Node                             │
//! ├─────────────────────────────────────────────────────────────┤
//! │  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
//! │  │ Network  │───▶│  Queue   │───▶│  Prover  │              │
//! │  │ Listener │    │ Manager  │    │  Worker  │              │
//! │  └──────────┘    └──────────┘    └──────────┘              │
//! │       │                               │                     │
//! │       │         ┌──────────┐          │                     │
//! │       └────────▶│  Stats   │◀─────────┘                     │
//! │                 │ Reporter │                                │
//! │                 └──────────┘                                │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```bash
//! prover-node --weights model.bin --network gevulot --gpu 0
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use tokio::sync::mpsc;
use tracing::{info, warn, error, Level};
use tracing_subscriber::FmtSubscriber;

use fluidelite_zk::circuit::config::CircuitConfig;
use fluidelite_zk::field::Q16;
use fluidelite_zk::mpo::MPO;
use fluidelite_zk::mps::MPS;
use fluidelite_zk::prover::{FluidEliteProof, FluidEliteProver, Halo2Proof, ProverStats};

/// Command-line arguments
#[derive(Parser, Debug)]
#[command(name = "fluidelite-prover")]
#[command(about = "FluidElite ZK Prover Node for prover networks")]
struct Args {
    /// Path to model weights file
    #[arg(short, long)]
    weights: PathBuf,

    /// Prover network to connect to
    #[arg(short, long, default_value = "gevulot")]
    network: String,

    /// GPU device ID
    #[arg(short, long, default_value = "0")]
    gpu: u32,

    /// Maximum concurrent proof jobs
    #[arg(short, long, default_value = "4")]
    jobs: usize,

    /// Stats reporting interval in seconds
    #[arg(long, default_value = "60")]
    stats_interval: u64,
}

/// Proof request from network
#[derive(Debug, Clone)]
struct ProofRequest {
    /// Unique request ID
    id: String,

    /// Token to prove
    token_id: u64,

    /// Serialized context MPS
    context_bytes: Vec<u8>,

    /// Fee offered (in network token units)
    fee: u64,

    /// Deadline for proof submission
    deadline: Instant,
}

/// Proof response to network
#[derive(Debug)]
struct ProofResponse {
    /// Request ID
    request_id: String,

    /// Generated proof
    proof: Halo2Proof,

    /// Whether proof was submitted successfully
    success: bool,
}

/// Network interface trait
trait ProverNetwork: Send + Sync {
    /// Connect to the network
    fn connect(&mut self) -> Result<(), String>;

    /// Fetch pending proof requests
    fn fetch_requests(&self) -> Vec<ProofRequest>;

    /// Submit a completed proof
    fn submit_proof(&self, response: ProofResponse) -> Result<(), String>;

    /// Get current fee rates
    fn get_fee_rate(&self) -> f64;
}

/// Mock network for testing
struct MockNetwork {
    fee_rate: f64,
}

impl ProverNetwork for MockNetwork {
    fn connect(&mut self) -> Result<(), String> {
        info!("Connected to mock network");
        Ok(())
    }

    fn fetch_requests(&self) -> Vec<ProofRequest> {
        // Generate mock requests for testing
        vec![ProofRequest {
            id: format!("mock-{}", rand::random::<u64>()),
            token_id: rand::random::<u64>() % 256,
            context_bytes: vec![],
            fee: 1000,
            deadline: Instant::now() + Duration::from_secs(60),
        }]
    }

    fn submit_proof(&self, response: ProofResponse) -> Result<(), String> {
        info!("Submitted proof for request {}", response.request_id);
        Ok(())
    }

    fn get_fee_rate(&self) -> f64 {
        self.fee_rate
    }
}

/// Gevulot network implementation
struct GevulotNetwork {
    endpoint: String,
    api_key: String,
}

impl GevulotNetwork {
    fn new(endpoint: &str, api_key: &str) -> Self {
        Self {
            endpoint: endpoint.to_string(),
            api_key: api_key.to_string(),
        }
    }
}

impl ProverNetwork for GevulotNetwork {
    fn connect(&mut self) -> Result<(), String> {
        // TODO: Implement actual Gevulot SDK connection
        info!("Connecting to Gevulot at {}", self.endpoint);
        Ok(())
    }

    fn fetch_requests(&self) -> Vec<ProofRequest> {
        // TODO: Fetch from Gevulot API
        vec![]
    }

    fn submit_proof(&self, _response: ProofResponse) -> Result<(), String> {
        // TODO: Submit to Gevulot
        Ok(())
    }

    fn get_fee_rate(&self) -> f64 {
        // TODO: Fetch current rates
        0.001
    }
}

/// Economics tracker
#[derive(Debug, Default)]
struct Economics {
    /// Total fees earned
    total_earned: f64,

    /// Total electricity cost (estimated)
    total_cost: f64,

    /// Total proofs generated
    proofs_generated: usize,

    /// Uptime in seconds
    uptime_seconds: u64,
}

impl Economics {
    fn profit(&self) -> f64 {
        self.total_earned - self.total_cost
    }

    fn profit_margin(&self) -> f64 {
        if self.total_earned > 0.0 {
            self.profit() / self.total_earned * 100.0
        } else {
            0.0
        }
    }

    fn hourly_rate(&self) -> f64 {
        if self.uptime_seconds > 0 {
            self.profit() / (self.uptime_seconds as f64 / 3600.0)
        } else {
            0.0
        }
    }

    fn record_proof(&mut self, fee: f64, electricity_cost: f64) {
        self.total_earned += fee;
        self.total_cost += electricity_cost;
        self.proofs_generated += 1;
    }

    fn report(&self) {
        info!("═══════════════════════════════════════════════════════");
        info!("                    ECONOMICS REPORT                    ");
        info!("═══════════════════════════════════════════════════════");
        info!("  Proofs generated:  {:>10}", self.proofs_generated);
        info!("  Total earned:      ${:>10.4}", self.total_earned);
        info!("  Total cost:        ${:>10.4}", self.total_cost);
        info!("  Net profit:        ${:>10.4}", self.profit());
        info!("  Profit margin:     {:>10.1}%", self.profit_margin());
        info!("  Hourly rate:       ${:>10.2}/hr", self.hourly_rate());
        info!("═══════════════════════════════════════════════════════");
    }
}

/// Main prover node
struct ProverNode {
    prover: Arc<std::sync::Mutex<FluidEliteProver>>,
    network: Box<dyn ProverNetwork>,
    stats: ProverStats,
    economics: Economics,
    config: CircuitConfig,
    start_time: Instant,
}

impl ProverNode {
    fn new(
        prover: FluidEliteProver,
        network: Box<dyn ProverNetwork>,
        config: CircuitConfig,
    ) -> Self {
        Self {
            prover: Arc::new(std::sync::Mutex::new(prover)),
            network,
            stats: ProverStats::default(),
            economics: Economics::default(),
            config,
            start_time: Instant::now(),
        }
    }

    async fn run(&mut self) {
        info!("Starting FluidElite Prover Node...");

        if let Err(e) = self.network.connect() {
            error!("Failed to connect to network: {}", e);
            return;
        }

        let fee_rate = self.network.get_fee_rate();
        info!("Current fee rate: ${}/proof", fee_rate);

        loop {
            // Fetch requests
            let requests = self.network.fetch_requests();

            if requests.is_empty() {
                tokio::time::sleep(Duration::from_secs(1)).await;
                continue;
            }

            for request in requests {
                if Instant::now() > request.deadline {
                    warn!("Request {} expired, skipping", request.id);
                    continue;
                }

                match self.process_request(&request).await {
                    Ok(response) => {
                        if let Err(e) = self.network.submit_proof(response) {
                            error!("Failed to submit proof: {}", e);
                        }
                    }
                    Err(e) => {
                        error!("Failed to generate proof: {}", e);
                    }
                }
            }

            // Update economics
            self.economics.uptime_seconds = self.start_time.elapsed().as_secs();

            // Periodic stats report
            if self.economics.proofs_generated % 10 == 0 && self.economics.proofs_generated > 0 {
                self.economics.report();
            }
        }
    }

    async fn process_request(&mut self, request: &ProofRequest) -> Result<ProofResponse, String> {
        info!("Processing request {} (token {})", request.id, request.token_id);

        // Deserialize context (or use default for testing)
        let context = if request.context_bytes.is_empty() {
            MPS::new(self.config.num_sites, self.config.chi_max, self.config.phys_dim)
        } else {
            // TODO: Deserialize from bytes
            MPS::new(self.config.num_sites, self.config.chi_max, self.config.phys_dim)
        };

        // Generate proof
        let proof = self.prover.lock().unwrap().prove(&context, request.token_id)?;

        // Update stats
        self.stats.record(&proof.inner);

        // Estimate electricity cost
        // RTX 4090: 450W, $0.12/kWh
        let proof_time_hours = proof.inner.generation_time_ms as f64 / 3_600_000.0;
        let electricity_cost = 0.450 * proof_time_hours * 0.12;

        let fee = request.fee as f64 / 1_000_000.0; // Convert from micro-units
        self.economics.record_proof(fee, electricity_cost);

        Ok(ProofResponse {
            request_id: request.id.clone(),
            proof,
            success: true,
        })
    }
}

#[tokio::main]
async fn main() {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    let args = Args::parse();

    info!("FluidElite Prover Node v0.1.0");
    info!("═══════════════════════════════════════════════════════");
    info!("  Network:     {}", args.network);
    info!("  GPU:         {}", args.gpu);
    info!("  Max jobs:    {}", args.jobs);
    info!("  Weights:     {}", args.weights.display());
    info!("═══════════════════════════════════════════════════════");

    // Load configuration
    let config = CircuitConfig::production();
    info!("Circuit config: k={}, constraints~{}", 
          config.k, config.estimate_constraints());

    // Load or create weights
    let (w_hidden, w_input, readout_weights) = if args.weights.exists() {
        info!("Loading weights from {}", args.weights.display());
        // TODO: Implement weight loading
        let w_hidden = MPO::identity(config.num_sites, config.phys_dim);
        let w_input = MPO::identity(config.num_sites, config.phys_dim);
        let readout_weights = vec![Q16::from_f64(0.1); config.chi_max * config.vocab_size];
        (w_hidden, w_input, readout_weights)
    } else {
        warn!("Weights file not found, using identity weights");
        let w_hidden = MPO::identity(config.num_sites, config.phys_dim);
        let w_input = MPO::identity(config.num_sites, config.phys_dim);
        let readout_weights = vec![Q16::from_f64(0.1); config.chi_max * config.vocab_size];
        (w_hidden, w_input, readout_weights)
    };

    // Create prover
    info!("Initializing prover (this may take a while for key generation)...");
    let prover = FluidEliteProver::new(w_hidden, w_input, readout_weights, config.clone());

    // Create network interface
    let network: Box<dyn ProverNetwork> = match args.network.as_str() {
        "gevulot" => Box::new(GevulotNetwork::new(
            "https://api.gevulot.com",
            &std::env::var("GEVULOT_API_KEY").unwrap_or_default(),
        )),
        "mock" => Box::new(MockNetwork { fee_rate: 0.001 }),
        _ => {
            error!("Unknown network: {}", args.network);
            return;
        }
    };

    // Run prover node
    let mut node = ProverNode::new(prover, network, config);
    node.run().await;
}
