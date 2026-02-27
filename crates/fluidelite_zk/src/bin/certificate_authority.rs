//! TPC Certificate Authority — Axum HTTP Server
//!
//! Production-ready HTTP API for issuing, retrieving, and verifying
//! Trustless Physics Certificates.
//!
//! # Usage
//!
//! ```bash
//! # Set required environment variables
//! export CA_SIGNING_KEY="<64-char hex ed25519 secret key>"
//! export CA_STORAGE_DIR="./certificates"
//! export CA_API_KEY="<secret>"
//!
//! # Start the certificate authority
//! cargo run --features server --bin certificate-authority
//! ```
//!
//! # Endpoints
//!
//! | Method | Path                        | Description                        |
//! |--------|-----------------------------|------------------------------------|
//! | POST   | /v1/certificates/issue      | Issue a new TPC certificate        |
//! | GET    | /v1/certificates/:id        | Retrieve certificate by UUID       |
//! | POST   | /v1/certificates/verify     | Verify certificate integrity       |
//! | GET    | /v1/certificates/stats      | CA statistics                      |
//! | GET    | /health                     | Health check                       |
//! | GET    | /metrics                    | Prometheus metrics                 |

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    body::Body,
    extract::{Json, Path, Request, State},
    http::{header, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::{error, info, warn};

use fluidelite_zk::certificate_authority::{
    CertificateAuthority, IssueCertificateRequest, VerifyCertificateRequest,
};

// ═══════════════════════════════════════════════════════════════════════════
// Application State
// ═══════════════════════════════════════════════════════════════════════════

type AppState = Arc<AppContext>;

struct AppContext {
    ca: CertificateAuthority,
    api_key: Option<String>,
    start_time: Instant,
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,tower_http=debug".into()),
        )
        .json()
        .init();

    info!("TPC Certificate Authority starting");

    // Parse configuration from environment
    let signing_key_hex = std::env::var("CA_SIGNING_KEY")
        .expect("CA_SIGNING_KEY environment variable required (64-char hex Ed25519 key)");

    let signing_key_bytes: [u8; 32] = hex::decode(&signing_key_hex)
        .expect("CA_SIGNING_KEY must be valid hex")
        .try_into()
        .expect("CA_SIGNING_KEY must be exactly 32 bytes (64 hex chars)");

    let storage_dir = PathBuf::from(
        std::env::var("CA_STORAGE_DIR").unwrap_or_else(|_| "./certificates".to_string()),
    );

    let prover_url = std::env::var("CA_PROVER_URL").ok();
    let api_key = std::env::var("CA_API_KEY").ok();

    let listen_addr: SocketAddr = std::env::var("CA_LISTEN_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:8444".to_string())
        .parse()
        .expect("CA_LISTEN_ADDR must be a valid socket address");

    // Create Certificate Authority
    let ca = CertificateAuthority::new(
        &signing_key_bytes,
        storage_dir.clone(),
        prover_url,
        api_key.clone(),
    )?;

    info!(
        pubkey = %ca.pubkey_hex(),
        storage = %storage_dir.display(),
        listen = %listen_addr,
        "Certificate Authority initialized"
    );

    let state = Arc::new(AppContext {
        ca,
        api_key,
        start_time: Instant::now(),
    });

    // Build router
    let app = Router::new()
        // Certificate endpoints (authenticated)
        .route("/v1/certificates/issue", post(handle_issue))
        .route("/v1/certificates/verify", post(handle_verify))
        .route("/v1/certificates/{id}", get(handle_get_certificate))
        .route("/v1/certificates/stats", get(handle_stats))
        // Public endpoints
        .route("/health", get(handle_health))
        .route("/metrics", get(handle_metrics))
        // Middleware
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    info!("TPC Certificate Authority listening on {}", listen_addr);

    let listener = tokio::net::TcpListener::bind(listen_addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Authentication Middleware
// ═══════════════════════════════════════════════════════════════════════════

async fn auth_middleware(
    State(state): State<AppState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    // Skip auth for health and metrics endpoints
    let path = request.uri().path();
    if path == "/health" || path == "/metrics" {
        return next.run(request).await;
    }

    // If no API key is configured, allow all requests
    let required_key = match &state.api_key {
        Some(k) => k,
        None => return next.run(request).await,
    };

    // Check Authorization header: Bearer <key>
    let auth_header = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());

    match auth_header {
        Some(auth) if auth.starts_with("Bearer ") => {
            let token = &auth[7..];
            if subtle::ConstantTimeEq::ct_eq(token.as_bytes(), required_key.as_bytes()).into() {
                next.run(request).await
            } else {
                warn!("Invalid API key from client");
                (StatusCode::UNAUTHORIZED, "Invalid API key").into_response()
            }
        }
        _ => {
            warn!("Missing Authorization header");
            (StatusCode::UNAUTHORIZED, "Missing Authorization: Bearer <key>").into_response()
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Handlers
// ═══════════════════════════════════════════════════════════════════════════

/// POST /v1/certificates/issue
///
/// Issue a new TPC certificate from proof data.
async fn handle_issue(
    State(state): State<AppState>,
    Json(req): Json<IssueCertificateRequest>,
) -> impl IntoResponse {
    info!(domain = ?req.domain, "Certificate issuance request");

    match state.ca.issue_certificate(&req).await {
        Ok(resp) => {
            info!(
                id = %resp.certificate_id,
                domain = %resp.domain,
                size = %resp.size_bytes,
                "Certificate issued"
            );
            (StatusCode::CREATED, Json(resp)).into_response()
        }
        Err(e) => {
            error!(error = %e, "Certificate issuance failed");
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
                .into_response()
        }
    }
}

/// GET /v1/certificates/:id
///
/// Retrieve a certificate by UUID. Returns the raw TPC binary.
async fn handle_get_certificate(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.ca.get_certificate(&id).await {
        Ok(data) => {
            let headers = [
                (header::CONTENT_TYPE, "application/octet-stream"),
                (
                    header::CONTENT_DISPOSITION,
                    &format!("attachment; filename=\"{}.tpc\"", id),
                ),
            ];
            // Build response with owned header values
            let mut response = (StatusCode::OK, data).into_response();
            response.headers_mut().insert(
                header::CONTENT_TYPE,
                "application/octet-stream".parse().unwrap(),
            );
            response.headers_mut().insert(
                header::CONTENT_DISPOSITION,
                format!("attachment; filename=\"{}.tpc\"", id)
                    .parse()
                    .unwrap(),
            );
            response
        }
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
            .into_response(),
    }
}

/// POST /v1/certificates/verify
///
/// Verify a certificate's integrity (hash + Ed25519 signature).
async fn handle_verify(
    State(state): State<AppState>,
    Json(req): Json<VerifyCertificateRequest>,
) -> impl IntoResponse {
    // Resolve certificate bytes: either from request body or by ID lookup
    let cert_data = if let Some(ref cert_hex) = req.certificate {
        match hex::decode(cert_hex) {
            Ok(d) => d,
            Err(_) => {
                // Try base64
                use base64::Engine;
                match base64::engine::general_purpose::STANDARD.decode(cert_hex) {
                    Ok(d) => d,
                    Err(e) => {
                        return (
                            StatusCode::BAD_REQUEST,
                            Json(ErrorResponse {
                                error: format!("Failed to decode certificate: {}", e),
                            }),
                        )
                            .into_response();
                    }
                }
            }
        }
    } else if let Some(ref id) = req.certificate_id {
        match state.ca.get_certificate(id).await {
            Ok(d) => d,
            Err(e) => {
                return (
                    StatusCode::NOT_FOUND,
                    Json(ErrorResponse {
                        error: e.to_string(),
                    }),
                )
                    .into_response();
            }
        }
    } else {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Provide either 'certificate' (hex/base64) or 'certificate_id'".to_string(),
            }),
        )
            .into_response();
    };

    let resp = state.ca.verify_certificate(&cert_data).await;
    let status = if resp.valid {
        StatusCode::OK
    } else {
        StatusCode::UNPROCESSABLE_ENTITY
    };

    (status, Json(resp)).into_response()
}

/// GET /v1/certificates/stats
///
/// Returns CA statistics including issuance counts by domain.
async fn handle_stats(State(state): State<AppState>) -> impl IntoResponse {
    let stats = state.ca.get_stats().await;
    (StatusCode::OK, Json(stats))
}

/// GET /health
///
/// Health check endpoint.
async fn handle_health(State(state): State<AppState>) -> impl IntoResponse {
    let uptime = state.start_time.elapsed().as_secs();
    Json(serde_json::json!({
        "status": "healthy",
        "service": "tpc-certificate-authority",
        "version": "1.0.0",
        "uptime_seconds": uptime,
        "signer_pubkey": state.ca.pubkey_hex(),
    }))
}

/// GET /metrics
///
/// Prometheus-compatible metrics endpoint.
async fn handle_metrics(State(state): State<AppState>) -> impl IntoResponse {
    let stats = state.ca.get_stats().await;

    let mut lines = Vec::new();
    lines.push("# HELP tpc_ca_certificates_issued_total Total certificates issued".to_string());
    lines.push("# TYPE tpc_ca_certificates_issued_total counter".to_string());
    lines.push(format!(
        "tpc_ca_certificates_issued_total {}",
        stats.total_issued
    ));

    lines.push(
        "# HELP tpc_ca_certificates_verified_total Total certificates verified".to_string(),
    );
    lines.push("# TYPE tpc_ca_certificates_verified_total counter".to_string());
    lines.push(format!(
        "tpc_ca_certificates_verified_total {}",
        stats.total_verified
    ));

    lines.push(
        "# HELP tpc_ca_certificates_failed_total Total failed verifications".to_string(),
    );
    lines.push("# TYPE tpc_ca_certificates_failed_total counter".to_string());
    lines.push(format!(
        "tpc_ca_certificates_failed_total {}",
        stats.total_failed
    ));

    lines.push("# HELP tpc_ca_uptime_seconds CA uptime in seconds".to_string());
    lines.push("# TYPE tpc_ca_uptime_seconds gauge".to_string());
    lines.push(format!("tpc_ca_uptime_seconds {:.1}", stats.uptime_seconds));

    for (domain, count) in &stats.certificates_by_domain {
        lines.push(format!(
            "tpc_ca_certificates_by_domain{{domain=\"{}\"}} {}",
            domain, count
        ));
    }

    let body = lines.join("\n") + "\n";

    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
        body,
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Error Response
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Serialize, Deserialize)]
struct ErrorResponse {
    error: String,
}
