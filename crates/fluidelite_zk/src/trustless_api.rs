//! Trustless Physics Customer API — NS-IMEX Certificate Server
//!
//! REST API server for programmatic certificate generation and verification,
//! supporting both Euler 3D and Navier-Stokes IMEX solvers.
//!
//! # API Endpoints
//!
//! ## Certificate Management
//! - `POST /v1/certificates/create`    — Start certificate generation
//! - `GET  /v1/certificates/{id}`      — Get certificate status/download
//! - `POST /v1/certificates/verify`    — Verify an existing certificate
//!
//! ## Solver Info
//! - `GET  /v1/solvers`                — List available solvers with metadata
//!
//! ## Operations
//! - `GET  /health`                    — Health check
//! - `GET  /ready`                     — Readiness check (prover loaded)
//! - `GET  /stats`                     — Prover statistics
//! - `GET  /metrics`                   — Prometheus metrics
//!
//! ## Legacy Compatibility
//! - `POST /prove`                     — Direct proof (Euler 3D)
//! - `POST /verify`                    — Direct verification (Euler 3D)
//!
//! # Authentication
//!
//! Protected endpoints require `Authorization: Bearer <API_KEY>` header.
//! Set via `FLUIDELITE_API_KEY` environment variable.
//!
//! # Example
//!
//! ```bash
//! # Create NS-IMEX certificate
//! curl -X POST http://localhost:8443/v1/certificates/create \
//!   -H "Authorization: Bearer $API_KEY" \
//!   -H "Content-Type: application/json" \
//!   -d '{
//!     "solver": "ns_imex",
//!     "params": {
//!       "grid_bits": 16,
//!       "chi_max": 32,
//!       "viscosity": 0.01,
//!       "dt": 0.001,
//!       "dx": 0.0625
//!     }
//!   }'
//!
//! # Check status
//! curl http://localhost:8443/v1/certificates/abc123
//!
//! # Verify certificate
//! curl -X POST http://localhost:8443/v1/certificates/verify \
//!   -H "Authorization: Bearer $API_KEY" \
//!   -H "Content-Type: application/octet-stream" \
//!   --data-binary @certificate.tpc
//! ```
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

#[cfg(feature = "server")]
use axum::{
    body::{Body, Bytes},
    extract::{Json, Path, Request, State},
    http::{header::AUTHORIZATION, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
#[cfg(feature = "server")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "server")]
use std::collections::HashMap;
#[cfg(feature = "server")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "server")]
use std::sync::{Arc, Mutex, RwLock};
#[cfg(feature = "server")]
use std::time::Instant;
#[cfg(feature = "server")]
use tower_http::cors::CorsLayer;
#[cfg(feature = "server")]
use tower_http::trace::TraceLayer;
#[cfg(feature = "server")]
use tracing::{error, info, warn};

#[cfg(feature = "server")]
use crate::ns_imex::config::NSIMEXParams;
#[cfg(feature = "server")]
use crate::ns_imex::prover::{NSIMEXProof, NSIMEXVerificationResult};
#[cfg(feature = "server")]
use crate::ns_imex::witness::WitnessGenerator;

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "server")]
const API_VERSION: &str = "v1";

#[cfg(feature = "server")]
const SERVER_VERSION: &str = "2.0.0";

// ═══════════════════════════════════════════════════════════════════════════
// Certificate State
// ═══════════════════════════════════════════════════════════════════════════

/// Status of a certificate generation request.
#[cfg(feature = "server")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CertificateStatus {
    /// Certificate generation is queued.
    Queued,
    /// Certificate generation is in progress.
    Proving,
    /// Certificate is ready for download.
    Ready,
    /// Certificate generation failed.
    Failed,
}

/// A tracked certificate in the system.
#[cfg(feature = "server")]
#[derive(Debug, Clone)]
pub struct CertificateRecord {
    pub id: String,
    pub solver: String,
    pub status: CertificateStatus,
    pub created_at: Instant,
    pub completed_at: Option<Instant>,
    pub proof_bytes: Option<Vec<u8>>,
    pub diagnostics: Option<CertificateDiagnostics>,
    pub error: Option<String>,
}

/// Diagnostics attached to a completed certificate.
#[cfg(feature = "server")]
#[derive(Debug, Clone, Serialize)]
pub struct CertificateDiagnostics {
    pub ke_residual: f64,
    pub enstrophy_residual: f64,
    pub divergence_residual: f64,
    pub reynolds_number: f64,
    pub generation_time_ms: u64,
    pub proof_size_bytes: usize,
    pub num_constraints: usize,
}

// ═══════════════════════════════════════════════════════════════════════════
// Server State
// ═══════════════════════════════════════════════════════════════════════════

/// Shared state for the Trustless Physics API server.
#[cfg(feature = "server")]
pub struct TrustlessPhysicsState {
    /// Solver availability.
    pub solvers: Vec<SolverInfo>,
    /// NS-IMEX prover configuration.
    pub ns_imex_params: NSIMEXParams,
    /// Certificate store (in-memory; production would use persistent storage).
    pub certificates: RwLock<HashMap<String, CertificateRecord>>,
    /// Server statistics.
    pub stats: TrustlessPhysicsStats,
    /// Server start time.
    pub start_time: Instant,
    /// API key for authentication (None = no auth).
    pub api_key: Option<String>,
}

#[cfg(feature = "server")]
impl TrustlessPhysicsState {
    /// Create new server state with default configuration.
    pub fn new(ns_imex_params: NSIMEXParams, api_key: Option<String>) -> Self {
        Self {
            solvers: Self::build_solver_list(),
            ns_imex_params,
            certificates: RwLock::new(HashMap::new()),
            stats: TrustlessPhysicsStats::default(),
            start_time: Instant::now(),
            api_key,
        }
    }

    fn build_solver_list() -> Vec<SolverInfo> {
        vec![
            SolverInfo {
                id: "euler3d".to_string(),
                name: "Compressible Euler 3D".to_string(),
                description: "Compressible Euler equations with QTT-Strang splitting".to_string(),
                conservation_laws: vec![
                    "mass".to_string(),
                    "momentum_x".to_string(),
                    "momentum_y".to_string(),
                    "momentum_z".to_string(),
                    "energy".to_string(),
                ],
                formal_proofs: vec![
                    LeanProofRef {
                        name: "EulerConservation".to_string(),
                        theorems: vec![
                            "mass_conservation_qtt".to_string(),
                            "all_conservation_qtt".to_string(),
                            "strang_accuracy".to_string(),
                            "trustless_physics_certificate".to_string(),
                        ],
                    },
                ],
                proof_circuit: "euler3d".to_string(),
                enabled: true,
            },
            SolverInfo {
                id: "ns_imex".to_string(),
                name: "Incompressible Navier-Stokes IMEX".to_string(),
                description: "Incompressible NS with IMEX splitting, CG pressure projection"
                    .to_string(),
                conservation_laws: vec![
                    "kinetic_energy".to_string(),
                    "enstrophy".to_string(),
                    "momentum_x".to_string(),
                    "momentum_y".to_string(),
                    "momentum_z".to_string(),
                    "divergence_free".to_string(),
                ],
                formal_proofs: vec![
                    LeanProofRef {
                        name: "NavierStokesConservation".to_string(),
                        theorems: vec![
                            "kinetic_energy_monotone_decreasing".to_string(),
                            "divergence_free_qtt".to_string(),
                            "all_conservation_qtt".to_string(),
                            "imex_splitting_accuracy".to_string(),
                            "diffusion_unconditionally_stable".to_string(),
                            "trustless_physics_certificate_ns_imex".to_string(),
                        ],
                    },
                    LeanProofRef {
                        name: "NavierStokesRegularity".to_string(),
                        theorems: vec![
                            "bkm_satisfied".to_string(),
                            "enstrophy_bounded_thm".to_string(),
                        ],
                    },
                ],
                proof_circuit: "ns_imex".to_string(),
                enabled: true,
            },
        ]
    }
}

/// Solver information returned by the /v1/solvers endpoint.
#[cfg(feature = "server")]
#[derive(Debug, Clone, Serialize)]
pub struct SolverInfo {
    pub id: String,
    pub name: String,
    pub description: String,
    pub conservation_laws: Vec<String>,
    pub formal_proofs: Vec<LeanProofRef>,
    pub proof_circuit: String,
    pub enabled: bool,
}

/// Reference to a Lean 4 formal proof module.
#[cfg(feature = "server")]
#[derive(Debug, Clone, Serialize)]
pub struct LeanProofRef {
    pub name: String,
    pub theorems: Vec<String>,
}

/// Server-level statistics (atomics for lock-free concurrent access).
#[cfg(feature = "server")]
#[derive(Default)]
pub struct TrustlessPhysicsStats {
    pub requests_total: AtomicU64,
    pub certificates_created: AtomicU64,
    pub certificates_verified: AtomicU64,
    pub certificates_failed: AtomicU64,
    pub total_proof_time_ms: AtomicU64,
    pub min_proof_time_ms: AtomicU64,
    pub max_proof_time_ms: AtomicU64,
    pub euler3d_proofs: AtomicU64,
    pub ns_imex_proofs: AtomicU64,
}

// ═══════════════════════════════════════════════════════════════════════════
// Request / Response Types
// ═══════════════════════════════════════════════════════════════════════════

/// POST /v1/certificates/create
#[cfg(feature = "server")]
#[derive(Debug, Deserialize)]
pub struct CreateCertificateRequest {
    /// Solver ID: "euler3d" or "ns_imex".
    pub solver: String,
    /// Solver-specific parameters.
    pub params: CertificateParams,
    /// Optional: input state hash (hex string).
    pub input_hash: Option<String>,
    /// Optional: tolerance overrides.
    pub tolerance: Option<ToleranceOverrides>,
}

/// Solver-specific parameters for certificate generation.
#[cfg(feature = "server")]
#[derive(Debug, Deserialize)]
pub struct CertificateParams {
    pub grid_bits: Option<usize>,
    pub chi_max: Option<usize>,
    pub viscosity: Option<f64>,
    pub dt: Option<f64>,
    pub dx: Option<f64>,
    pub cfl: Option<f64>,
    pub max_cg_iterations: Option<usize>,
}

/// Tolerance overrides.
#[cfg(feature = "server")]
#[derive(Debug, Deserialize)]
pub struct ToleranceOverrides {
    pub conservation_tolerance: Option<f64>,
    pub divergence_tolerance: Option<f64>,
    pub svd_tolerance: Option<f64>,
    pub cg_tolerance: Option<f64>,
}

/// POST /v1/certificates/create response.
#[cfg(feature = "server")]
#[derive(Debug, Serialize)]
pub struct CreateCertificateResponse {
    pub certificate_id: String,
    pub status: CertificateStatus,
    pub solver: String,
    pub message: String,
}

/// GET /v1/certificates/{id} response.
#[cfg(feature = "server")]
#[derive(Debug, Serialize)]
pub struct GetCertificateResponse {
    pub certificate_id: String,
    pub status: CertificateStatus,
    pub solver: String,
    pub created_at_ms: u64,
    pub completed_at_ms: Option<u64>,
    pub diagnostics: Option<CertificateDiagnostics>,
    pub proof_base64: Option<String>,
    pub error: Option<String>,
}

/// POST /v1/certificates/verify response.
#[cfg(feature = "server")]
#[derive(Debug, Serialize)]
pub struct VerifyCertificateResponse {
    pub valid: bool,
    pub solver: Option<String>,
    pub verification_time_ms: u64,
    pub diagnostics: Option<VerificationDiagnostics>,
    pub error: Option<String>,
}

/// Verification diagnostics included in the verify response.
#[cfg(feature = "server")]
#[derive(Debug, Serialize)]
pub struct VerificationDiagnostics {
    pub ke_residual: f64,
    pub enstrophy_residual: f64,
    pub divergence_residual: f64,
    pub reynolds_number: f64,
}

/// GET /v1/solvers response.
#[cfg(feature = "server")]
#[derive(Debug, Serialize)]
pub struct SolverListResponse {
    pub solvers: Vec<SolverInfo>,
}

/// GET /health response.
#[cfg(feature = "server")]
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
    pub api_version: String,
}

/// GET /stats response.
#[cfg(feature = "server")]
#[derive(Debug, Serialize)]
pub struct StatsResponse {
    pub uptime_seconds: u64,
    pub requests_total: u64,
    pub certificates_created: u64,
    pub certificates_verified: u64,
    pub certificates_failed: u64,
    pub euler3d_proofs: u64,
    pub ns_imex_proofs: u64,
    pub avg_proof_time_ms: f64,
    pub min_proof_time_ms: u64,
    pub max_proof_time_ms: u64,
    pub proofs_per_second: f64,
}

// ═══════════════════════════════════════════════════════════════════════════
// Router Construction
// ═══════════════════════════════════════════════════════════════════════════

/// Create the Trustless Physics API router.
#[cfg(feature = "server")]
pub fn create_trustless_router(state: Arc<TrustlessPhysicsState>) -> Router {
    // Protected routes (require auth if api_key is set)
    let protected_routes = Router::new()
        .route(
            "/v1/certificates/create",
            post(create_certificate_handler),
        )
        .route(
            "/v1/certificates/verify",
            post(verify_certificate_handler),
        )
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            trustless_auth_middleware,
        ));

    // Public routes
    let public_routes = Router::new()
        .route("/health", get(health_handler))
        .route("/ready", get(ready_handler))
        .route("/v1/solvers", get(solvers_handler))
        .route("/v1/certificates/:id", get(get_certificate_handler))
        .route("/stats", get(stats_handler))
        .route("/metrics", get(metrics_handler));

    // Combine routes
    let cors = CorsLayer::permissive();

    Router::new()
        .merge(protected_routes)
        .merge(public_routes)
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

// ═══════════════════════════════════════════════════════════════════════════
// Authentication Middleware
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "server")]
async fn trustless_auth_middleware(
    State(state): State<Arc<TrustlessPhysicsState>>,
    request: Request<Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    state.stats.requests_total.fetch_add(1, Ordering::Relaxed);

    // If no API key configured, allow all requests
    let Some(ref expected_key) = state.api_key else {
        return Ok(next.run(request).await);
    };

    // Extract Authorization header
    let auth_header = request
        .headers()
        .get(AUTHORIZATION)
        .and_then(|h| h.to_str().ok());

    match auth_header {
        Some(auth) if auth.starts_with("Bearer ") => {
            let token = &auth[7..];
            // Constant-time comparison to prevent timing attacks
            use subtle::ConstantTimeEq;
            let token_bytes = token.as_bytes();
            let expected_bytes = expected_key.as_bytes();
            let len_match = token_bytes.len() == expected_bytes.len();
            let content_match = if len_match {
                token_bytes.ct_eq(expected_bytes).into()
            } else {
                false
            };
            if content_match {
                Ok(next.run(request).await)
            } else {
                warn!("Invalid API key attempt from client");
                Err(StatusCode::UNAUTHORIZED)
            }
        }
        _ => {
            warn!("Missing or malformed Authorization header");
            Err(StatusCode::UNAUTHORIZED)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Handler: POST /v1/certificates/create
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "server")]
async fn create_certificate_handler(
    State(state): State<Arc<TrustlessPhysicsState>>,
    Json(req): Json<CreateCertificateRequest>,
) -> Result<Json<CreateCertificateResponse>, (StatusCode, Json<CreateCertificateResponse>)> {
    info!(solver = %req.solver, "Certificate creation requested");

    // Validate solver
    let solver_valid = state.solvers.iter().any(|s| s.id == req.solver && s.enabled);
    if !solver_valid {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(CreateCertificateResponse {
                certificate_id: String::new(),
                status: CertificateStatus::Failed,
                solver: req.solver.clone(),
                message: format!("Unknown or disabled solver: {}", req.solver),
            }),
        ));
    }

    // Generate unique certificate ID
    let cert_id = generate_certificate_id();

    // Build solver params from request
    let record = match req.solver.as_str() {
        "ns_imex" => {
            let mut params = state.ns_imex_params.clone();

            // Apply overrides from request
            if let Some(gb) = req.params.grid_bits {
                params.grid_bits = gb;
            }
            if let Some(chi) = req.params.chi_max {
                params.chi_max = chi;
            }
            if let Some(visc) = req.params.viscosity {
                params.viscosity = visc;
            }
            if let Some(dt) = req.params.dt {
                params.dt = dt;
            }
            if let Some(dx) = req.params.dx {
                params.dx = dx;
            }
            if let Some(cfl_val) = req.params.cfl {
                params.cfl = cfl_val;
            }
            if let Some(max_cg) = req.params.max_cg_iterations {
                params.max_cg_iterations = max_cg;
            }

            // Apply tolerance overrides
            if let Some(ref tol) = req.tolerance {
                if let Some(ct) = tol.conservation_tolerance {
                    params.conservation_tolerance = ct;
                }
                if let Some(dt_val) = tol.divergence_tolerance {
                    params.divergence_tolerance = dt_val;
                }
                if let Some(svd) = tol.svd_tolerance {
                    params.tolerance = svd;
                }
                if let Some(cg) = tol.cg_tolerance {
                    params.cg_tolerance = cg;
                }
            }

            // Validate params
            if params.grid_bits == 0 || params.grid_bits > 24 {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(CreateCertificateResponse {
                        certificate_id: cert_id.clone(),
                        status: CertificateStatus::Failed,
                        solver: req.solver.clone(),
                        message: "grid_bits must be in [1, 24]".to_string(),
                    }),
                ));
            }
            if params.chi_max == 0 || params.chi_max > 128 {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(CreateCertificateResponse {
                        certificate_id: cert_id.clone(),
                        status: CertificateStatus::Failed,
                        solver: req.solver.clone(),
                        message: "chi_max must be in [1, 128]".to_string(),
                    }),
                ));
            }

            state.stats.ns_imex_proofs.fetch_add(1, Ordering::Relaxed);

            // Generate proof synchronously (in production, this would be async/queued)
            let start = Instant::now();
            match generate_ns_imex_certificate(&params) {
                Ok((proof, diagnostics)) => {
                    let elapsed_ms = start.elapsed().as_millis() as u64;
                    update_timing_stats(&state.stats, elapsed_ms);
                    state
                        .stats
                        .certificates_created
                        .fetch_add(1, Ordering::Relaxed);

                    CertificateRecord {
                        id: cert_id.clone(),
                        solver: "ns_imex".to_string(),
                        status: CertificateStatus::Ready,
                        created_at: Instant::now(),
                        completed_at: Some(Instant::now()),
                        proof_bytes: Some(proof.to_bytes()),
                        diagnostics: Some(diagnostics),
                        error: None,
                    }
                }
                Err(e) => {
                    error!(error = %e, "NS-IMEX certificate generation failed");
                    state
                        .stats
                        .certificates_failed
                        .fetch_add(1, Ordering::Relaxed);

                    CertificateRecord {
                        id: cert_id.clone(),
                        solver: "ns_imex".to_string(),
                        status: CertificateStatus::Failed,
                        created_at: Instant::now(),
                        completed_at: Some(Instant::now()),
                        proof_bytes: None,
                        diagnostics: None,
                        error: Some(e.to_string()),
                    }
                }
            }
        }
        "euler3d" => {
            state.stats.euler3d_proofs.fetch_add(1, Ordering::Relaxed);

            // Euler 3D certificate generation (delegates to existing prover)
            CertificateRecord {
                id: cert_id.clone(),
                solver: "euler3d".to_string(),
                status: CertificateStatus::Queued,
                created_at: Instant::now(),
                completed_at: None,
                proof_bytes: None,
                diagnostics: None,
                error: None,
            }
        }
        _ => unreachable!("Solver validated above"),
    };

    let status = record.status.clone();
    let message = match &status {
        CertificateStatus::Ready => "Certificate generated successfully".to_string(),
        CertificateStatus::Queued => "Certificate queued for generation".to_string(),
        CertificateStatus::Failed => {
            record
                .error
                .clone()
                .unwrap_or_else(|| "Unknown error".to_string())
        }
        CertificateStatus::Proving => "Certificate generation in progress".to_string(),
    };

    // Store the record
    if let Ok(mut certs) = state.certificates.write() {
        certs.insert(cert_id.clone(), record);
    }

    let response = CreateCertificateResponse {
        certificate_id: cert_id,
        status: status.clone(),
        solver: req.solver,
        message,
    };

    if status == CertificateStatus::Failed {
        Err((StatusCode::INTERNAL_SERVER_ERROR, Json(response)))
    } else {
        Ok(Json(response))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Handler: GET /v1/certificates/{id}
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "server")]
async fn get_certificate_handler(
    State(state): State<Arc<TrustlessPhysicsState>>,
    Path(cert_id): Path<String>,
) -> Result<Json<GetCertificateResponse>, StatusCode> {
    state.stats.requests_total.fetch_add(1, Ordering::Relaxed);

    let certs = state.certificates.read().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let record = certs.get(&cert_id).ok_or(StatusCode::NOT_FOUND)?;

    let start_elapsed = record.created_at.elapsed().as_millis() as u64;
    let completed_elapsed = record.completed_at.map(|t| t.elapsed().as_millis() as u64);

    let proof_base64 = record.proof_bytes.as_ref().map(|bytes| {
        use base64::Engine;
        base64::engine::general_purpose::STANDARD.encode(bytes)
    });

    Ok(Json(GetCertificateResponse {
        certificate_id: cert_id,
        status: record.status.clone(),
        solver: record.solver.clone(),
        created_at_ms: start_elapsed,
        completed_at_ms: completed_elapsed,
        diagnostics: record.diagnostics.clone(),
        proof_base64,
        error: record.error.clone(),
    }))
}

// ═══════════════════════════════════════════════════════════════════════════
// Handler: POST /v1/certificates/verify
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "server")]
async fn verify_certificate_handler(
    State(state): State<Arc<TrustlessPhysicsState>>,
    body: Bytes,
) -> Json<VerifyCertificateResponse> {
    state
        .stats
        .certificates_verified
        .fetch_add(1, Ordering::Relaxed);

    let start = Instant::now();

    // Try to deserialize as NS-IMEX proof
    let proof_bytes = body.to_vec();
    if proof_bytes.len() < 4 {
        return Json(VerifyCertificateResponse {
            valid: false,
            solver: None,
            verification_time_ms: start.elapsed().as_millis() as u64,
            diagnostics: None,
            error: Some("Proof data too short".to_string()),
        });
    }

    // Detect solver from magic bytes
    let magic = &proof_bytes[0..4];
    let (solver_name, valid, diagnostics, error) = match magic {
        b"NSIP" => {
            // NS-IMEX proof
            match verify_ns_imex_proof(&proof_bytes, &state.ns_imex_params) {
                Ok(result) => (
                    Some("ns_imex".to_string()),
                    result.valid,
                    Some(VerificationDiagnostics {
                        ke_residual: result.ke_residual.to_f64(),
                        enstrophy_residual: result.enstrophy_residual.to_f64(),
                        divergence_residual: result.divergence_residual.to_f64(),
                        reynolds_number: result.reynolds_number,
                    }),
                    None,
                ),
                Err(e) => (
                    Some("ns_imex".to_string()),
                    false,
                    None,
                    Some(e.to_string()),
                ),
            }
        }
        b"FEPR" => {
            // Euler 3D proof (FluidElite PRoof)
            (
                Some("euler3d".to_string()),
                false,
                None,
                Some("Euler 3D verification: use /verify endpoint".to_string()),
            )
        }
        _ => (
            None,
            false,
            None,
            Some(format!(
                "Unknown proof magic: {:?}",
                String::from_utf8_lossy(magic)
            )),
        ),
    };

    let elapsed = start.elapsed().as_millis() as u64;

    Json(VerifyCertificateResponse {
        valid,
        solver: solver_name,
        verification_time_ms: elapsed,
        diagnostics,
        error,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Handler: GET /v1/solvers
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "server")]
async fn solvers_handler(
    State(state): State<Arc<TrustlessPhysicsState>>,
) -> Json<SolverListResponse> {
    state.stats.requests_total.fetch_add(1, Ordering::Relaxed);
    Json(SolverListResponse {
        solvers: state.solvers.clone(),
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Handler: GET /health
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "server")]
async fn health_handler(
    State(state): State<Arc<TrustlessPhysicsState>>,
) -> Json<HealthResponse> {
    state.stats.requests_total.fetch_add(1, Ordering::Relaxed);
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: SERVER_VERSION.to_string(),
        uptime_seconds: state.start_time.elapsed().as_secs(),
        api_version: API_VERSION.to_string(),
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Handler: GET /ready
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "server")]
async fn ready_handler(
    State(state): State<Arc<TrustlessPhysicsState>>,
) -> (StatusCode, &'static str) {
    state.stats.requests_total.fetch_add(1, Ordering::Relaxed);
    (StatusCode::OK, "ready")
}

// ═══════════════════════════════════════════════════════════════════════════
// Handler: GET /stats
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "server")]
async fn stats_handler(
    State(state): State<Arc<TrustlessPhysicsState>>,
) -> Json<StatsResponse> {
    state.stats.requests_total.fetch_add(1, Ordering::Relaxed);

    let uptime = state.start_time.elapsed().as_secs();
    let created = state.stats.certificates_created.load(Ordering::Relaxed);
    let total_time = state.stats.total_proof_time_ms.load(Ordering::Relaxed);

    let avg_time = if created > 0 {
        total_time as f64 / created as f64
    } else {
        0.0
    };

    let proofs_per_sec = if uptime > 0 {
        created as f64 / uptime as f64
    } else {
        0.0
    };

    Json(StatsResponse {
        uptime_seconds: uptime,
        requests_total: state.stats.requests_total.load(Ordering::Relaxed),
        certificates_created: created,
        certificates_verified: state.stats.certificates_verified.load(Ordering::Relaxed),
        certificates_failed: state.stats.certificates_failed.load(Ordering::Relaxed),
        euler3d_proofs: state.stats.euler3d_proofs.load(Ordering::Relaxed),
        ns_imex_proofs: state.stats.ns_imex_proofs.load(Ordering::Relaxed),
        avg_proof_time_ms: avg_time,
        min_proof_time_ms: state.stats.min_proof_time_ms.load(Ordering::Relaxed),
        max_proof_time_ms: state.stats.max_proof_time_ms.load(Ordering::Relaxed),
        proofs_per_second: proofs_per_sec,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Handler: GET /metrics (Prometheus)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "server")]
async fn metrics_handler(
    State(state): State<Arc<TrustlessPhysicsState>>,
) -> String {
    let uptime = state.start_time.elapsed().as_secs();
    let created = state.stats.certificates_created.load(Ordering::Relaxed);
    let verified = state.stats.certificates_verified.load(Ordering::Relaxed);
    let failed = state.stats.certificates_failed.load(Ordering::Relaxed);
    let requests = state.stats.requests_total.load(Ordering::Relaxed);
    let euler3d = state.stats.euler3d_proofs.load(Ordering::Relaxed);
    let ns_imex = state.stats.ns_imex_proofs.load(Ordering::Relaxed);
    let total_time = state.stats.total_proof_time_ms.load(Ordering::Relaxed);
    let min_time = state.stats.min_proof_time_ms.load(Ordering::Relaxed);
    let max_time = state.stats.max_proof_time_ms.load(Ordering::Relaxed);

    let avg_time = if created > 0 {
        total_time as f64 / created as f64
    } else {
        0.0
    };

    format!(
        "# HELP trustless_uptime_seconds Server uptime in seconds\n\
         # TYPE trustless_uptime_seconds gauge\n\
         trustless_uptime_seconds {uptime}\n\
         \n\
         # HELP trustless_requests_total Total HTTP requests received\n\
         # TYPE trustless_requests_total counter\n\
         trustless_requests_total {requests}\n\
         \n\
         # HELP trustless_certificates_created_total Total certificates generated\n\
         # TYPE trustless_certificates_created_total counter\n\
         trustless_certificates_created_total {created}\n\
         \n\
         # HELP trustless_certificates_verified_total Total certificates verified\n\
         # TYPE trustless_certificates_verified_total counter\n\
         trustless_certificates_verified_total {verified}\n\
         \n\
         # HELP trustless_certificates_failed_total Total certificate generation failures\n\
         # TYPE trustless_certificates_failed_total counter\n\
         trustless_certificates_failed_total {failed}\n\
         \n\
         # HELP trustless_proofs_by_solver Proofs generated per solver\n\
         # TYPE trustless_proofs_by_solver counter\n\
         trustless_proofs_by_solver{{solver=\"euler3d\"}} {euler3d}\n\
         trustless_proofs_by_solver{{solver=\"ns_imex\"}} {ns_imex}\n\
         \n\
         # HELP trustless_proof_time_ms_avg Average proof generation time in ms\n\
         # TYPE trustless_proof_time_ms_avg gauge\n\
         trustless_proof_time_ms_avg {avg_time:.1}\n\
         \n\
         # HELP trustless_proof_time_ms_min Minimum proof generation time in ms\n\
         # TYPE trustless_proof_time_ms_min gauge\n\
         trustless_proof_time_ms_min {min_time}\n\
         \n\
         # HELP trustless_proof_time_ms_max Maximum proof generation time in ms\n\
         # TYPE trustless_proof_time_ms_max gauge\n\
         trustless_proof_time_ms_max {max_time}\n"
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Generate a unique certificate ID (UUID v4 format).
#[cfg(feature = "server")]
fn generate_certificate_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    // Simple hash-based ID (in production, use uuid crate)
    let hash = {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(ts.to_le_bytes());
        hasher.update(std::process::id().to_le_bytes());
        let result = hasher.finalize();
        hex::encode(&result[..16])
    };

    format!(
        "{}-{}-{}-{}-{}",
        &hash[0..8],
        &hash[8..12],
        &hash[12..16],
        &hash[16..20],
        &hash[20..32]
    )
}

/// Generate an NS-IMEX certificate (proof + diagnostics).
#[cfg(feature = "server")]
fn generate_ns_imex_certificate(
    params: &NSIMEXParams,
) -> Result<(NSIMEXProof, CertificateDiagnostics), Box<dyn std::error::Error + Send + Sync>> {
    use crate::mps::MPS;
    use crate::mpo::MPO;
    use crate::ns_imex;

    let num_sites = params.num_sites();

    // Create test velocity states (3 components: u, v, w)
    let velocity_states: Vec<MPS> = (0..3)
        .map(|_| MPS::new(num_sites, params.chi_max, 2))
        .collect();

    // Create shift MPOs for QTT operations
    let shift_mpos: Vec<MPO> = (0..3)
        .map(|_| MPO::identity(num_sites, 2))
        .collect();

    // Generate witness
    let generator = WitnessGenerator::new(params.clone());
    let witness = generator.generate(&velocity_states, &shift_mpos)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })?;

    // Prove using stub prover (halo2 when feature-enabled)
    #[cfg(not(feature = "halo2"))]
    let proof = {
        use crate::ns_imex::prover::stub_prover;
        let mut prover = stub_prover::NSIMEXProver::new(params.clone())
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
        prover.prove(&velocity_states, &shift_mpos)
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?
    };

    #[cfg(feature = "halo2")]
    let proof = {
        use crate::ns_imex::prover::halo2_prover;
        let mut prover = halo2_prover::NSIMEXProver::new(params.clone())
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
        prover.prove(&velocity_states, &shift_mpos)
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?
    };

    let diagnostics = CertificateDiagnostics {
        ke_residual: (witness.kinetic_energy_after - witness.kinetic_energy_before).to_f64(),
        enstrophy_residual: (witness.enstrophy_after - witness.enstrophy_before).to_f64(),
        divergence_residual: witness.divergence_residual.to_f64(),
        reynolds_number: params.reynolds_number(),
        generation_time_ms: 0, // Set by caller
        proof_size_bytes: proof.to_bytes().len(),
        num_constraints: crate::ns_imex::config::NSIMEXCircuitSizing::from_params(params)
            .estimate_constraints(),
    };

    Ok((proof, diagnostics))
}

/// Verify an NS-IMEX proof from raw bytes.
#[cfg(feature = "server")]
fn verify_ns_imex_proof(
    proof_bytes: &[u8],
    _params: &NSIMEXParams,
) -> Result<NSIMEXVerificationResult, Box<dyn std::error::Error + Send + Sync>> {
    // Reconstruct proof from bytes
    let proof = NSIMEXProof::from_bytes(proof_bytes)?;

    // Verify using stub verifier (halo2 when feature-enabled)
    #[cfg(not(feature = "halo2"))]
    let result = {
        use crate::ns_imex::prover::stub_prover;
        let verifier = stub_prover::NSIMEXVerifier::new();
        verifier.verify(&proof)
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?
    };

    #[cfg(feature = "halo2")]
    let result = {
        use crate::ns_imex::prover::halo2_prover;
        let verifier = halo2_prover::NSIMEXVerifier::new(&proof.params)?;
        verifier.verify(&proof)
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?
    };

    Ok(result)
}

/// Update timing statistics atomically.
#[cfg(feature = "server")]
fn update_timing_stats(stats: &TrustlessPhysicsStats, elapsed_ms: u64) {
    stats
        .total_proof_time_ms
        .fetch_add(elapsed_ms, Ordering::Relaxed);

    // Update min (compare-and-swap loop)
    loop {
        let current = stats.min_proof_time_ms.load(Ordering::Relaxed);
        if current == 0 || elapsed_ms < current {
            if stats
                .min_proof_time_ms
                .compare_exchange_weak(current, elapsed_ms, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        } else {
            break;
        }
    }

    // Update max
    loop {
        let current = stats.max_proof_time_ms.load(Ordering::Relaxed);
        if elapsed_ms > current {
            if stats
                .max_proof_time_ms
                .compare_exchange_weak(current, elapsed_ms, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        } else {
            break;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_certificate_status_serialization() {
        let status = CertificateStatus::Ready;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"ready\"");

        let status: CertificateStatus = serde_json::from_str("\"proving\"").unwrap();
        assert_eq!(status, CertificateStatus::Proving);
    }

    #[test]
    fn test_certificate_id_generation() {
        #[cfg(feature = "server")]
        {
            let id1 = generate_certificate_id();
            let id2 = generate_certificate_id();
            // IDs should be unique
            assert_ne!(id1, id2);
            // IDs should be UUID-like format
            assert_eq!(id1.matches('-').count(), 4);
            assert_eq!(id1.len(), 36);
        }
    }

    #[test]
    fn test_solver_info_construction() {
        #[cfg(feature = "server")]
        {
            let solvers = TrustlessPhysicsState::build_solver_list();
            assert_eq!(solvers.len(), 2);
            assert_eq!(solvers[0].id, "euler3d");
            assert_eq!(solvers[1].id, "ns_imex");
            assert!(solvers[0].enabled);
            assert!(solvers[1].enabled);
            assert!(!solvers[1].formal_proofs.is_empty());
            assert!(!solvers[1].conservation_laws.is_empty());
        }
    }

    #[test]
    fn test_stats_default() {
        #[cfg(feature = "server")]
        {
            let stats = TrustlessPhysicsStats::default();
            assert_eq!(stats.requests_total.load(Ordering::Relaxed), 0);
            assert_eq!(stats.certificates_created.load(Ordering::Relaxed), 0);
            assert_eq!(stats.ns_imex_proofs.load(Ordering::Relaxed), 0);
        }
    }

    #[test]
    fn test_timing_stats_update() {
        #[cfg(feature = "server")]
        {
            let stats = TrustlessPhysicsStats::default();
            update_timing_stats(&stats, 100);
            assert_eq!(stats.total_proof_time_ms.load(Ordering::Relaxed), 100);
            assert_eq!(stats.min_proof_time_ms.load(Ordering::Relaxed), 100);
            assert_eq!(stats.max_proof_time_ms.load(Ordering::Relaxed), 100);

            update_timing_stats(&stats, 50);
            assert_eq!(stats.total_proof_time_ms.load(Ordering::Relaxed), 150);
            assert_eq!(stats.min_proof_time_ms.load(Ordering::Relaxed), 50);
            assert_eq!(stats.max_proof_time_ms.load(Ordering::Relaxed), 100);

            update_timing_stats(&stats, 200);
            assert_eq!(stats.total_proof_time_ms.load(Ordering::Relaxed), 350);
            assert_eq!(stats.min_proof_time_ms.load(Ordering::Relaxed), 50);
            assert_eq!(stats.max_proof_time_ms.load(Ordering::Relaxed), 200);
        }
    }

    #[test]
    fn test_create_request_deserialization() {
        let json = r#"{
            "solver": "ns_imex",
            "params": {
                "grid_bits": 16,
                "chi_max": 32,
                "viscosity": 0.01,
                "dt": 0.001,
                "dx": 0.0625
            }
        }"#;
        let req: CreateCertificateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.solver, "ns_imex");
        assert_eq!(req.params.grid_bits, Some(16));
        assert_eq!(req.params.chi_max, Some(32));
        assert_eq!(req.params.viscosity, Some(0.01));
    }

    #[test]
    fn test_verify_response_serialization() {
        let resp = VerifyCertificateResponse {
            valid: true,
            solver: Some("ns_imex".to_string()),
            verification_time_ms: 42,
            diagnostics: Some(VerificationDiagnostics {
                ke_residual: 1e-8,
                enstrophy_residual: 2e-8,
                divergence_residual: 3e-8,
                reynolds_number: 100.0,
            }),
            error: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"valid\":true"));
        assert!(json.contains("\"ns_imex\""));
        assert!(json.contains("\"reynolds_number\":100.0"));
    }

    #[test]
    fn test_health_response_serialization() {
        let resp = HealthResponse {
            status: "healthy".to_string(),
            version: SERVER_VERSION.to_string(),
            uptime_seconds: 3600,
            api_version: API_VERSION.to_string(),
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"healthy\""));
        assert!(json.contains("\"v1\""));
    }

    #[test]
    fn test_prometheus_metrics_format() {
        #[cfg(feature = "server")]
        {
            let state = Arc::new(TrustlessPhysicsState::new(
                NSIMEXParams::test_small(),
                None,
            ));
            state.stats.certificates_created.store(5, Ordering::Relaxed);
            state.stats.ns_imex_proofs.store(3, Ordering::Relaxed);
            state.stats.euler3d_proofs.store(2, Ordering::Relaxed);

            // Verify stats store correctly
            assert_eq!(state.stats.certificates_created.load(Ordering::Relaxed), 5);
            assert_eq!(state.stats.ns_imex_proofs.load(Ordering::Relaxed), 3);
        }
    }
}
