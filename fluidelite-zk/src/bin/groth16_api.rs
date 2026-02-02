//! Groth16 ZK Proof API
//! 
//! REST API for Groth16 proof generation.

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use tracing::info;
use ark_bn254::Fr;

use fluidelite_zk::groth16_prover::{Groth16GpuProver, Groth16Proof, SemaphoreProof};

#[derive(Parser, Debug)]
#[command(name = "groth16-api")]
#[command(about = "FluidElite Groth16 ZK Proof API")]
struct Args {
    #[arg(short, long, default_value = "8080")]
    port: u16,
    #[arg(long, default_value = "0.0.0.0")]
    host: String,
}

struct AppState {
    prover: Mutex<Groth16GpuProver>,
    start_time: Instant,
    proofs_generated: std::sync::atomic::AtomicU64,
}

#[derive(Deserialize)]
struct ProveRequest {
    secret: String,
}

#[derive(Serialize)]
struct ProveResponse {
    success: bool,
    proof: Option<Groth16Proof>,
    semaphore_format: Option<SemaphoreProof>,
    error: Option<String>,
}

#[derive(Deserialize)]
struct VerifyRequest {
    proof: Groth16Proof,
}

#[derive(Serialize)]
struct VerifyResponse {
    valid: bool,
    error: Option<String>,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    uptime_seconds: u64,
    proofs_generated: u64,
}

async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let resp = HealthResponse {
        status: "healthy".to_string(),
        uptime_seconds: state.start_time.elapsed().as_secs(),
        proofs_generated: state.proofs_generated.load(std::sync::atomic::Ordering::Relaxed),
    };
    (StatusCode::OK, Json(resp))
}

async fn prove(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ProveRequest>,
) -> impl IntoResponse {
    let secret_bytes = match hex::decode(req.secret.trim_start_matches("0x")) {
        Ok(b) if b.len() == 32 => {
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&b);
            arr
        }
        Ok(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ProveResponse {
                    success: false,
                    proof: None,
                    semaphore_format: None,
                    error: Some("Secret must be exactly 32 bytes".to_string()),
                }),
            );
        }
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ProveResponse {
                    success: false,
                    proof: None,
                    semaphore_format: None,
                    error: Some(format!("Invalid hex: {}", e)),
                }),
            );
        }
    };

    let prover = state.prover.lock().await;
    match prover.prove(
        &secret_bytes,
        &[],
        &[],
        Fr::from(0u64),
        Fr::from(0u64),
        Fr::from(0u64),
    ) {
        Ok(proof) => {
            state.proofs_generated.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let semaphore_format = proof.to_semaphore_format();
            (
                StatusCode::OK,
                Json(ProveResponse {
                    success: true,
                    proof: Some(proof),
                    semaphore_format: Some(semaphore_format),
                    error: None,
                }),
            )
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ProveResponse {
                success: false,
                proof: None,
                semaphore_format: None,
                error: Some(e),
            }),
        ),
    }
}

async fn verify(
    State(state): State<Arc<AppState>>,
    Json(req): Json<VerifyRequest>,
) -> impl IntoResponse {
    let prover = state.prover.lock().await;
    match prover.verify(&req.proof) {
        Ok(valid) => (StatusCode::OK, Json(VerifyResponse { valid, error: None })),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(VerifyResponse {
                valid: false,
                error: Some(e),
            }),
        ),
    }
}

async fn get_verifier() -> impl IntoResponse {
    let contract = r#"// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title FluidElite Groth16 Verifier (BN254)
/// @notice On-chain verification of Groth16 proofs
contract FluidEliteGroth16Verifier {
    uint256 constant SNARK_SCALAR_FIELD = 21888242871839275222246405745257275088548364400416034343698204186575808495617;
    
    function verify(
        uint256[2] calldata a,
        uint256[2][2] calldata b,
        uint256[2] calldata c,
        uint256[] calldata input
    ) external view returns (bool) {
        // Production: embed actual VK coefficients here
        return true;
    }
}"#;
    (StatusCode::OK, contract.to_string())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    let args = Args::parse();
    
    info!("Initializing Groth16 prover...");
    let prover = Groth16GpuProver::new(20)?;
    info!("Groth16 prover ready");

    let state = Arc::new(AppState {
        prover: Mutex::new(prover),
        start_time: Instant::now(),
        proofs_generated: std::sync::atomic::AtomicU64::new(0),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/prove", post(prove))
        .route("/v1/verify", post(verify))
        .route("/v1/verifier/solidity", get(get_verifier))
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    info!("Starting Groth16 API on {}", addr);
    
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    
    Ok(())
}
