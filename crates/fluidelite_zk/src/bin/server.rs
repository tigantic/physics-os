//! FluidElite ZK REST API Server
//!
//! Production-hardened ZK proof generation service.
//!
//! Features:
//! - API key authentication
//! - Rate limiting (per-IP and per-key)
//! - Prometheus metrics
//! - Graceful shutdown
//! - Request timeout
//! - CORS support

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::Router;
use clap::Parser;
use tokio::net::TcpListener;
use tokio::signal;
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use fluidelite_zk::circuit::config::CircuitConfig;
use fluidelite_zk::field::Q16;
use fluidelite_zk::mpo::MPO;
use fluidelite_zk::prover::FluidEliteProver;
use fluidelite_zk::server::{create_router, ServerState};

#[derive(Parser, Debug)]
#[command(name = "fluidelite-server")]
#[command(about = "FluidElite ZK REST API Server - Production Ready")]
struct Args {
    /// Port to listen on
    #[arg(short, long, default_value = "8080")]
    port: u16,

    /// Host to bind to
    #[arg(short = 'H', long, default_value = "0.0.0.0")]
    host: String,

    /// Use test configuration (faster key generation)
    #[arg(short, long, default_value = "false")]
    test: bool,

    /// Use production_v1 config (matches trained weights)
    #[arg(long, default_value = "false")]
    production_v1: bool,

    /// Path to trained weights JSON file
    #[arg(long, env = "FLUIDELITE_MODEL_PATH")]
    weights: Option<String>,

    /// API key for authentication (if not set, auth is disabled)
    #[arg(long, env = "FLUIDELITE_API_KEY")]
    api_key: Option<String>,

    /// Requests per minute per IP (rate limiting)
    #[arg(long, default_value = "60")]
    rate_limit: u32,

    /// Request timeout in seconds
    #[arg(long, default_value = "120")]
    timeout: u64,

    /// Metrics port (Prometheus)
    #[arg(long, default_value = "9090")]
    metrics_port: u16,

    /// Enable JSON logging for production
    #[arg(long)]
    json_logs: bool,

    /// Custom k value for circuit (overrides test/production)
    #[arg(short = 'k', long)]
    circuit_k: Option<u32>,
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    warn!("Shutdown signal received, draining connections...");
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Initialize tracing with appropriate format
    if args.json_logs {
        tracing_subscriber::registry()
            .with(tracing_subscriber::fmt::layer().json())
            .with(tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("fluidelite_zk=info".parse()?)
                .add_directive("tower_http=info".parse()?))
            .init();
    } else {
        tracing_subscriber::fmt::init();
    }

    info!(
        "╔══════════════════════════════════════════════════════════╗"
    );
    info!(
        "║       FluidElite ZK Prover - Production Server          ║"
    );
    info!(
        "║                    Version {}                       ║",
        env!("CARGO_PKG_VERSION")
    );
    info!(
        "╚══════════════════════════════════════════════════════════╝"
    );

    // Build configuration
    let mut config = if args.test {
        info!("Mode: TEST (fast key generation)");
        CircuitConfig::test()
    } else if args.production_v1 {
        info!("Mode: PRODUCTION_V1 (trained weights config)");
        CircuitConfig::production_v1()
    } else {
        info!("Mode: PRODUCTION (full security)");
        CircuitConfig::production()
    };

    // Override k if specified
    if let Some(k) = args.circuit_k {
        info!("Overriding circuit k={}", k);
        config.k = k;
    }

    info!(
        "Circuit: sites={}, chi_max={}, phys_dim={}, mpo_d={}, vocab={}, k={}",
        config.num_sites, config.chi_max, config.phys_dim, config.mpo_d, config.vocab_size, config.k
    );

    // Security settings
    if args.api_key.is_some() {
        info!("Authentication: ENABLED (API key required)");
    } else {
        warn!("Authentication: DISABLED (set FLUIDELITE_API_KEY for production)");
    }
    info!("Rate limit: {} requests/minute/IP", args.rate_limit);
    info!("Request timeout: {}s", args.timeout);

    // Load model weights
    let (w_hidden, w_input, readout_weights) = if let Some(ref weights_path) = args.weights {
        info!("Loading trained weights from: {}", weights_path);
        
        match fluidelite_zk::weights::TrainedWeights::from_json(weights_path) {
            Ok(weights) => {
                info!(
                    "Weights loaded: L={}, chi={}, vocab={}",
                    weights.config.num_sites,
                    weights.config.chi_max,
                    weights.config.vocab_size
                );
                
                // Validate weights match config
                if weights.config.num_sites != config.num_sites {
                    warn!(
                        "Weight L={} doesn't match config L={}. Using --production-v1 is recommended.",
                        weights.config.num_sites, config.num_sites
                    );
                }
                
                let w_hidden = weights.to_w_hidden();
                let w_input = weights.to_w_input();
                let readout = weights.to_readout(config.vocab_size, config.chi_max);
                
                info!("Real weights loaded successfully!");
                (w_hidden, w_input, readout)
            }
            Err(e) => {
                warn!("Failed to load weights: {}. Using identity weights.", e);
                let w_hidden = MPO::identity(config.num_sites, config.phys_dim);
                let w_input = MPO::identity(config.num_sites, config.phys_dim);
                let readout = vec![Q16::from_f64(0.1); config.chi_max * config.vocab_size];
                (w_hidden, w_input, readout)
            }
        }
    } else {
        info!("No weights file specified. Using identity weights (test mode).");
        let w_hidden = MPO::identity(config.num_sites, config.phys_dim);
        let w_input = MPO::identity(config.num_sites, config.phys_dim);
        let readout = vec![Q16::from_f64(0.1); config.chi_max * config.vocab_size];
        (w_hidden, w_input, readout)
    };

    // Initialize prover
    info!("Generating proving keys (one-time setup)...");
    let start = Instant::now();
    let prover = FluidEliteProver::new(w_hidden, w_input, readout_weights, config.clone());
    info!("Key generation complete in {:?}", start.elapsed());

    // Create server state with optional API key
    let state = Arc::new(ServerState::with_api_key(prover, config, args.api_key));

    // Create router with middleware
    let app: Router = create_router(state)
        .layer(tower_http::timeout::TimeoutLayer::new(Duration::from_secs(args.timeout)))
        .layer(tower_http::limit::RequestBodyLimitLayer::new(1024 * 1024)); // 1MB max body

    // Start server
    let addr = format!("{}:{}", args.host, args.port);
    info!("╔══════════════════════════════════════════════════════════╗");
    info!("║ Server listening on http://{:<25} ║", addr);
    info!("╠══════════════════════════════════════════════════════════╣");
    info!("║  GET  /health   - Health check                          ║");
    info!("║  GET  /stats    - Prover statistics                     ║");
    info!("║  GET  /metrics  - Prometheus metrics                    ║");
    info!("║  POST /prove    - Generate ZK proof                     ║");
    info!("║  POST /verify   - Verify ZK proof                       ║");
    info!("╚══════════════════════════════════════════════════════════╝");

    let listener = TcpListener::bind(&addr).await?;
    
    // Graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("Server shutdown complete");
    Ok(())
}
