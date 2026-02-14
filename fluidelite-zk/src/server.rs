//! FluidElite REST API Server
//!
//! Production-ready HTTP API for ZK proof generation.
//!
//! # Endpoints
//!
//! - `POST /prove` - Generate a ZK proof
//! - `POST /verify` - Verify a ZK proof
//! - `GET /health` - Health check
//! - `GET /stats` - Prover statistics
//! - `GET /metrics` - Prometheus metrics
//!
//! # Example
//!
//! ```bash
//! # Start server
//! cargo run --features production --bin fluidelite-server
//!
//! # Generate proof
//! curl -X POST http://localhost:8080/prove \
//!   -H "Content-Type: application/json" \
//!   -H "Authorization: Bearer YOUR_API_KEY" \
//!   -d '{"token_id": 42}'
//! ```

#[cfg(feature = "server")]
use axum::{
    body::Body,
    extract::{Json, Request, State},
    http::{header::AUTHORIZATION, StatusCode},
    middleware::{self, Next},
    response::Response,
    routing::{get, post},
    Router,
};
#[cfg(feature = "server")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "server")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "server")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "server")]
use std::time::Instant;
#[cfg(feature = "server")]
use tower_http::cors::CorsLayer;
#[cfg(feature = "server")]
use tower_http::trace::TraceLayer;
#[cfg(feature = "server")]
use tracing::{info, warn};

#[cfg(feature = "server")]
use crate::circuit::config::CircuitConfig;
#[cfg(feature = "server")]
use crate::field::Q16;
#[cfg(feature = "server")]
use crate::mpo::MPO;
#[cfg(feature = "server")]
use crate::mps::MPS;
#[cfg(feature = "server")]
use crate::prover::FluidEliteProver;
#[cfg(feature = "server")]
use crate::verifier::FluidEliteVerifier;
#[cfg(feature = "server")]
use crate::prover::Halo2Proof;

// Physics domain types (lazily initialized per domain)
#[cfg(feature = "server")]
use crate::euler3d;
#[cfg(feature = "server")]
use crate::ns_imex;
#[cfg(feature = "server")]
use crate::thermal;

/// Magic byte prefixes for physics domain proofs.
#[cfg(feature = "server")]
const EULER3D_MAGIC: &[u8; 4] = b"E3DP";
#[cfg(feature = "server")]
const NS_IMEX_MAGIC: &[u8; 4] = b"NSIP";
#[cfg(feature = "server")]
const THERMAL_MAGIC: &[u8; 4] = b"THEP";

/// Lazily-initialized physics domain provers and verifiers.
///
/// Each domain is initialized on its first `/prove` request.
/// Keygen is expensive (~seconds) and happens once per domain.
/// After initialization, the prover/verifier pair is reused for all subsequent requests.
#[cfg(feature = "server")]
pub struct PhysicsDomainRouter {
    euler3d: Mutex<Option<(euler3d::Euler3DProver, euler3d::Euler3DVerifier)>>,
    ns_imex: Mutex<Option<(ns_imex::NSIMEXProver, ns_imex::NSIMEXVerifier)>>,
    thermal: Mutex<Option<(thermal::ThermalProver, thermal::ThermalVerifier)>>,
}

#[cfg(feature = "server")]
impl PhysicsDomainRouter {
    fn new() -> Self {
        Self {
            euler3d: Mutex::new(None),
            ns_imex: Mutex::new(None),
            thermal: Mutex::new(None),
        }
    }
}

/// Server state shared across requests
#[cfg(feature = "server")]
pub struct ServerState {
    prover: Mutex<FluidEliteProver>,
    verifier: FluidEliteVerifier,
    stats: ServerStats,
    /// Circuit configuration
    pub config: CircuitConfig,
    start_time: Instant,
    /// Optional API key for authentication
    pub api_key: Option<String>,
    /// Physics domain provers (Euler3D, NS-IMEX, Thermal) — lazily initialized
    physics: PhysicsDomainRouter,
}

#[cfg(feature = "server")]
impl ServerState {
    /// Create new server state with prover and config.
    /// The verifier is initialized from the prover's KZG params and verifying key.
    pub fn new(prover: FluidEliteProver, config: CircuitConfig) -> Self {
        let verifier = FluidEliteVerifier::new(
            prover.params().clone(),
            prover.verifying_key().clone(),
        );
        Self {
            prover: Mutex::new(prover),
            verifier,
            stats: ServerStats::default(),
            config,
            start_time: Instant::now(),
            api_key: None,
            physics: PhysicsDomainRouter::new(),
        }
    }

    /// Create new server state with prover, config, and API key
    pub fn with_api_key(prover: FluidEliteProver, config: CircuitConfig, api_key: Option<String>) -> Self {
        let verifier = FluidEliteVerifier::new(
            prover.params().clone(),
            prover.verifying_key().clone(),
        );
        Self {
            prover: Mutex::new(prover),
            verifier,
            stats: ServerStats::default(),
            config,
            start_time: Instant::now(),
            api_key,
            physics: PhysicsDomainRouter::new(),
        }
    }
}

#[cfg(feature = "server")]
#[derive(Default)]
pub struct ServerStats {
    pub requests_total: AtomicU64,
    pub proofs_generated: AtomicU64,
    pub proofs_failed: AtomicU64,
    pub verifications_total: AtomicU64,
    pub total_proof_time_ms: AtomicU64,
    pub min_proof_time_ms: AtomicU64,
    pub max_proof_time_ms: AtomicU64,
}

/// Request to generate a proof
#[cfg(feature = "server")]
#[derive(Debug, Deserialize)]
pub struct ProveRequest {
    /// Token ID to prove (FluidElite inference domain)
    #[serde(default)]
    pub token_id: u64,

    /// Context MPS as base64-encoded bytes (optional)
    pub context_bytes: Option<String>,

    /// Number of sites (if creating default context)
    pub num_sites: Option<usize>,

    /// Chi max (if creating default context)
    pub chi_max: Option<usize>,

    /// Physics domain: "euler3d", "ns_imex", or "thermal".
    /// When absent, defaults to FluidElite inference circuit.
    pub domain: Option<String>,
}

/// Response from proof generation
#[cfg(feature = "server")]
#[derive(Debug, Serialize)]
pub struct ProveResponse {
    pub success: bool,
    pub token_id: u64,
    pub proof_bytes: String, // base64 encoded
    pub public_inputs: Vec<String>,
    pub generation_time_ms: u64,
    pub error: Option<String>,
    /// Physics domain (present for domain proofs, absent for FluidElite inference)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,
}

/// Request to verify a proof
#[cfg(feature = "server")]
#[derive(Debug, Deserialize)]
pub struct VerifyRequest {
    pub proof_bytes: String, // base64 encoded
    pub public_inputs: Vec<String>,
}

/// Response from verification
#[cfg(feature = "server")]
#[derive(Debug, Serialize)]
pub struct VerifyResponse {
    pub valid: bool,
    pub error: Option<String>,
}

/// Health check response
#[cfg(feature = "server")]
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
}

/// Statistics response
#[cfg(feature = "server")]
#[derive(Debug, Serialize)]
pub struct StatsResponse {
    pub uptime_seconds: u64,
    pub requests_total: u64,
    pub proofs_generated: u64,
    pub proofs_failed: u64,
    pub verifications_total: u64,
    pub avg_proof_time_ms: f64,
    pub min_proof_time_ms: u64,
    pub max_proof_time_ms: u64,
    pub proofs_per_second: f64,
    pub circuit_config: CircuitConfigResponse,
}

#[cfg(feature = "server")]
#[derive(Debug, Serialize)]
pub struct CircuitConfigResponse {
    pub num_sites: usize,
    pub chi_max: usize,
    pub vocab_size: usize,
    pub k: u32,
}

/// API key authentication middleware
#[cfg(feature = "server")]
async fn auth_middleware(
    State(state): State<Arc<ServerState>>,
    request: Request<Body>,
    next: Next,
) -> Result<Response, StatusCode> {
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
            // Use constant-time comparison to prevent timing attacks
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
                warn!("Invalid API key attempt");
                Err(StatusCode::UNAUTHORIZED)
            }
        }
        _ => {
            warn!("Missing or malformed Authorization header");
            Err(StatusCode::UNAUTHORIZED)
        }
    }
}

/// Create the API router
#[cfg(feature = "server")]
pub fn create_router(state: Arc<ServerState>) -> Router {
    // Protected routes (require auth if api_key is set)
    let protected_routes = Router::new()
        .route("/prove", post(prove_handler))
        .route("/verify", post(verify_handler))
        .route_layer(middleware::from_fn_with_state(state.clone(), auth_middleware));

    // Public routes (no auth required)
    let public_routes = Router::new()
        .route("/health", get(health_handler))
        .route("/ready", get(ready_handler))
        .route("/stats", get(stats_handler))
        .route("/metrics", get(metrics_handler));

    // Configure CORS based on environment
    // In production, set CORS_ORIGIN env var to restrict origins
    let cors_layer = if let Ok(origin) = std::env::var("CORS_ORIGIN") {
        use tower_http::cors::AllowOrigin;
        CorsLayer::new()
            .allow_origin(AllowOrigin::exact(origin.parse().unwrap_or_else(|_| {
                "*".parse().expect("wildcard always valid")
            })))
            .allow_methods([axum::http::Method::GET, axum::http::Method::POST])
            .allow_headers([axum::http::header::AUTHORIZATION, axum::http::header::CONTENT_TYPE])
    } else {
        // Development mode: permissive CORS (logged for awareness)
        tracing::warn!("CORS_ORIGIN not set, using permissive CORS. Set CORS_ORIGIN in production.");
        CorsLayer::permissive()
    };

    Router::new()
        .merge(public_routes)
        .merge(protected_routes)
        .layer(cors_layer)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Health check endpoint (liveness)
#[cfg(feature = "server")]
async fn health_handler(State(state): State<Arc<ServerState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: state.start_time.elapsed().as_secs(),
    })
}

/// Readiness check (for k8s)
#[cfg(feature = "server")]
async fn ready_handler(State(_state): State<Arc<ServerState>>) -> (StatusCode, &'static str) {
    (StatusCode::OK, "ready")
}

/// Statistics endpoint
#[cfg(feature = "server")]
async fn stats_handler(State(state): State<Arc<ServerState>>) -> Json<StatsResponse> {
    let uptime = state.start_time.elapsed().as_secs();
    let proofs_generated = state.stats.proofs_generated.load(Ordering::Relaxed);
    let total_proof_time = state.stats.total_proof_time_ms.load(Ordering::Relaxed);

    let avg_proof_time = if proofs_generated > 0 {
        total_proof_time as f64 / proofs_generated as f64
    } else {
        0.0
    };

    let proofs_per_second = if uptime > 0 {
        proofs_generated as f64 / uptime as f64
    } else {
        0.0
    };

    Json(StatsResponse {
        uptime_seconds: uptime,
        requests_total: state.stats.requests_total.load(Ordering::Relaxed),
        proofs_generated,
        proofs_failed: state.stats.proofs_failed.load(Ordering::Relaxed),
        verifications_total: state.stats.verifications_total.load(Ordering::Relaxed),
        avg_proof_time_ms: avg_proof_time,
        min_proof_time_ms: state.stats.min_proof_time_ms.load(Ordering::Relaxed),
        max_proof_time_ms: state.stats.max_proof_time_ms.load(Ordering::Relaxed),
        proofs_per_second,
        circuit_config: CircuitConfigResponse {
            num_sites: state.config.num_sites,
            chi_max: state.config.chi_max,
            vocab_size: state.config.vocab_size,
            k: state.config.k,
        },
    })
}

/// Prometheus metrics endpoint
#[cfg(feature = "server")]
async fn metrics_handler(State(state): State<Arc<ServerState>>) -> String {
    let uptime = state.start_time.elapsed().as_secs();
    let proofs = state.stats.proofs_generated.load(Ordering::Relaxed);
    let failed = state.stats.proofs_failed.load(Ordering::Relaxed);
    let total_time = state.stats.total_proof_time_ms.load(Ordering::Relaxed);
    let requests = state.stats.requests_total.load(Ordering::Relaxed);
    let verifications = state.stats.verifications_total.load(Ordering::Relaxed);

    format!(
        r#"# HELP fluidelite_uptime_seconds Server uptime in seconds
# TYPE fluidelite_uptime_seconds gauge
fluidelite_uptime_seconds {}

# HELP fluidelite_requests_total Total HTTP requests received
# TYPE fluidelite_requests_total counter
fluidelite_requests_total {}

# HELP fluidelite_proofs_total Total proofs generated
# TYPE fluidelite_proofs_total counter
fluidelite_proofs_total {}

# HELP fluidelite_proofs_failed_total Total failed proof attempts
# TYPE fluidelite_proofs_failed_total counter
fluidelite_proofs_failed_total {}

# HELP fluidelite_verifications_total Total verification requests
# TYPE fluidelite_verifications_total counter
fluidelite_verifications_total {}

# HELP fluidelite_proof_time_ms_total Cumulative proof generation time in milliseconds
# TYPE fluidelite_proof_time_ms_total counter
fluidelite_proof_time_ms_total {}

# HELP fluidelite_circuit_k Circuit k parameter (log2 rows)
# TYPE fluidelite_circuit_k gauge
fluidelite_circuit_k {}

# HELP fluidelite_circuit_chi_max Maximum bond dimension
# TYPE fluidelite_circuit_chi_max gauge
fluidelite_circuit_chi_max {}

# HELP fluidelite_circuit_sites Number of tensor sites
# TYPE fluidelite_circuit_sites gauge
fluidelite_circuit_sites {}
"#,
        uptime,
        requests,
        proofs,
        failed,
        verifications,
        total_time,
        state.config.k,
        state.config.chi_max,
        state.config.num_sites
    )
}

/// Proof generation endpoint
#[cfg(feature = "server")]
async fn prove_handler(
    State(state): State<Arc<ServerState>>,
    Json(request): Json<ProveRequest>,
) -> Result<Json<ProveResponse>, (StatusCode, String)> {
    // Update request count (atomic)
    state.stats.requests_total.fetch_add(1, Ordering::Relaxed);

    // Dispatch based on physics domain
    match request.domain.as_deref() {
        None => {
            // Default path: FluidElite inference circuit
            info!("Proof request: token_id={}", request.token_id);
            prove_fluidelite(&state, &request).await
        }
        Some("euler3d") => {
            info!("Proof request: domain=euler3d");
            prove_physics_domain(&state, "euler3d").await
        }
        Some("ns_imex") => {
            info!("Proof request: domain=ns_imex");
            prove_physics_domain(&state, "ns_imex").await
        }
        Some("thermal") => {
            info!("Proof request: domain=thermal");
            prove_physics_domain(&state, "thermal").await
        }
        Some(unknown) => {
            Err((
                StatusCode::BAD_REQUEST,
                format!(
                    "Unknown domain: '{}'. Valid domains: euler3d, ns_imex, thermal",
                    unknown
                ),
            ))
        }
    }
}

/// FluidElite inference proof generation (existing path).
#[cfg(feature = "server")]
async fn prove_fluidelite(
    state: &Arc<ServerState>,
    request: &ProveRequest,
) -> Result<Json<ProveResponse>, (StatusCode, String)> {
    use halo2_axiom::halo2curves::ff::PrimeField;

    // Parse or create context
    let context = if let Some(bytes_b64) = &request.context_bytes {
        let bytes = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, bytes_b64)
            .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid base64: {}", e)))?;

        bincode::deserialize(&bytes)
            .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid context: {}", e)))?
    } else {
        let num_sites = request.num_sites.unwrap_or(state.config.num_sites);
        let chi_max = request.chi_max.unwrap_or(state.config.chi_max);
        MPS::new(num_sites, chi_max, state.config.phys_dim)
    };

    // Generate proof
    let start = Instant::now();
    match state.prover.lock().unwrap().prove(&context, request.token_id) {
        Ok(proof) => {
            let elapsed = start.elapsed().as_millis() as u64;

            // Update stats (atomic)
            state.stats.proofs_generated.fetch_add(1, Ordering::Relaxed);
            state.stats.total_proof_time_ms.fetch_add(elapsed, Ordering::Relaxed);

            // Update min/max (best-effort, not perfectly atomic but good enough for metrics)
            let current_min = state.stats.min_proof_time_ms.load(Ordering::Relaxed);
            if current_min == 0 || elapsed < current_min {
                state.stats.min_proof_time_ms.store(elapsed, Ordering::Relaxed);
            }
            let current_max = state.stats.max_proof_time_ms.load(Ordering::Relaxed);
            if elapsed > current_max {
                state.stats.max_proof_time_ms.store(elapsed, Ordering::Relaxed);
            }

            info!(
                "Proof generated: token={}, time={}ms",
                request.token_id, elapsed
            );

            Ok(Json(ProveResponse {
                success: true,
                token_id: request.token_id,
                proof_bytes: base64::Engine::encode(
                    &base64::engine::general_purpose::STANDARD,
                    &proof.inner.proof_bytes,
                ),
                public_inputs: proof
                    .public_inputs
                    .iter()
                    .map(|x| hex::encode(x.to_repr()))
                    .collect(),
                generation_time_ms: elapsed,
                error: None,
                domain: None,
            }))
        }
        Err(e) => {
            // Update failed count (atomic)
            state.stats.proofs_failed.fetch_add(1, Ordering::Relaxed);

            Ok(Json(ProveResponse {
                success: false,
                token_id: request.token_id,
                proof_bytes: String::new(),
                public_inputs: vec![],
                generation_time_ms: 0,
                error: Some(e),
                domain: None,
            }))
        }
    }
}

/// Physics domain proof generation (Euler3D, NS-IMEX, Thermal).
///
/// Lazily initializes the domain prover on first call (one-time keygen).
/// Creates canonical test inputs from domain parameters and generates a real proof.
#[cfg(feature = "server")]
async fn prove_physics_domain(
    state: &Arc<ServerState>,
    domain: &str,
) -> Result<Json<ProveResponse>, (StatusCode, String)> {
    use halo2_axiom::halo2curves::bn256::Fr;
    use halo2_axiom::halo2curves::ff::PrimeField;

    let start = Instant::now();

    // Each match arm is wrapped in a closure so that `?` returns String errors
    // to the `result` binding rather than propagating to the outer function.
    let result: Result<(Vec<u8>, Vec<Fr>), String> = match domain {
        "euler3d" => (|| -> Result<(Vec<u8>, Vec<Fr>), String> {
            let mut guard = state.physics.euler3d.lock().unwrap();

            // Lazy initialization: keygen on first request
            if guard.is_none() {
                info!("Initializing Euler3D prover (one-time keygen)...");
                let params = euler3d::Euler3DParams::default();
                let prover = euler3d::Euler3DProver::new(params)
                    .map_err(|e| format!("Euler3D keygen failed: {}", e))?;
                let verifier = euler3d::Euler3DVerifier::from_prover(&prover);
                *guard = Some((prover, verifier));
                info!("Euler3D prover initialized");
            }

            let (prover, _) = guard.as_mut().unwrap();

            // Create canonical test inputs
            let params = euler3d::Euler3DParams::default();
            let states = euler3d::make_test_states(&params);
            let mpos = euler3d::make_test_shift_mpos(&params);

            // Generate real proof
            let proof = prover.prove(&states, &mpos)?;
            let proof_bytes = proof.to_bytes();
            let public_inputs = proof.reconstruct_public_inputs();
            Ok((proof_bytes, public_inputs))
        })(),
        "ns_imex" => (|| -> Result<(Vec<u8>, Vec<Fr>), String> {
            let mut guard = state.physics.ns_imex.lock().unwrap();

            if guard.is_none() {
                info!("Initializing NS-IMEX prover (one-time keygen)...");
                let params = ns_imex::NSIMEXParams::test_small();
                let prover = ns_imex::NSIMEXProver::new(params)
                    .map_err(|e| format!("NS-IMEX keygen failed: {}", e))?;
                let verifier = ns_imex::NSIMEXVerifier::from_prover(&prover);
                *guard = Some((prover, verifier));
                info!("NS-IMEX prover initialized");
            }

            let (prover, _) = guard.as_mut().unwrap();

            let params = ns_imex::NSIMEXParams::test_small();
            let states = ns_imex::make_test_states(&params);
            let mpos = ns_imex::make_test_shift_mpos(&params);

            let proof = prover.prove(&states, &mpos)?;
            let proof_bytes = proof.to_bytes();
            let public_inputs = proof.reconstruct_public_inputs();
            Ok((proof_bytes, public_inputs))
        })(),
        "thermal" => (|| -> Result<(Vec<u8>, Vec<Fr>), String> {
            let mut guard = state.physics.thermal.lock().unwrap();

            if guard.is_none() {
                info!("Initializing Thermal prover (one-time keygen)...");
                let params = thermal::ThermalParams::test_small();
                let prover = thermal::ThermalProver::new(params)
                    .map_err(|e| format!("Thermal keygen failed: {}", e))?;
                let verifier = thermal::ThermalVerifier::from_prover(&prover);
                *guard = Some((prover, verifier));
                info!("Thermal prover initialized");
            }

            let (prover, _) = guard.as_mut().unwrap();

            let params = thermal::ThermalParams::test_small();
            let states = thermal::make_test_states(&params);
            let mpos = thermal::make_test_laplacian_mpos(&params);

            let proof = prover.prove(&states, &mpos)?;
            let proof_bytes = proof.to_bytes();
            let public_inputs = proof.reconstruct_public_inputs();
            Ok((proof_bytes, public_inputs))
        })(),
        _ => unreachable!("domain already validated"),
    };

    match result {
        Ok((proof_bytes, public_inputs)) => {
            let elapsed = start.elapsed().as_millis() as u64;

            state.stats.proofs_generated.fetch_add(1, Ordering::Relaxed);
            state.stats.total_proof_time_ms.fetch_add(elapsed, Ordering::Relaxed);

            let current_min = state.stats.min_proof_time_ms.load(Ordering::Relaxed);
            if current_min == 0 || elapsed < current_min {
                state.stats.min_proof_time_ms.store(elapsed, Ordering::Relaxed);
            }
            let current_max = state.stats.max_proof_time_ms.load(Ordering::Relaxed);
            if elapsed > current_max {
                state.stats.max_proof_time_ms.store(elapsed, Ordering::Relaxed);
            }

            info!(
                "Physics proof generated: domain={}, time={}ms, proof_size={}",
                domain, elapsed, proof_bytes.len()
            );

            Ok(Json(ProveResponse {
                success: true,
                token_id: 0,
                proof_bytes: base64::Engine::encode(
                    &base64::engine::general_purpose::STANDARD,
                    &proof_bytes,
                ),
                public_inputs: public_inputs
                    .iter()
                    .map(|x| hex::encode(x.to_repr()))
                    .collect(),
                generation_time_ms: elapsed,
                error: None,
                domain: Some(domain.to_string()),
            }))
        }
        Err(e) => {
            state.stats.proofs_failed.fetch_add(1, Ordering::Relaxed);
            warn!("Physics proof failed: domain={}, error={}", domain, e);
            Ok(Json(ProveResponse {
                success: false,
                token_id: 0,
                proof_bytes: String::new(),
                public_inputs: vec![],
                generation_time_ms: 0,
                error: Some(e),
                domain: Some(domain.to_string()),
            }))
        }
    }
}

/// Verification endpoint
///
/// Auto-detects physics domain proofs from magic bytes (E3DP, NSIP, THEP).
/// Falls through to FluidElite inference verification if no magic is recognized.
#[cfg(feature = "server")]
async fn verify_handler(
    State(state): State<Arc<ServerState>>,
    Json(request): Json<VerifyRequest>,
) -> Json<VerifyResponse> {
    info!("Verify request: proof_size={}", request.proof_bytes.len());

    // Update verification count
    state.stats.verifications_total.fetch_add(1, Ordering::Relaxed);

    // Decode proof bytes from base64
    let proof_bytes = match base64::Engine::decode(
        &base64::engine::general_purpose::STANDARD,
        &request.proof_bytes,
    ) {
        Ok(bytes) => bytes,
        Err(e) => {
            return Json(VerifyResponse {
                valid: false,
                error: Some(format!("Invalid base64 proof: {}", e)),
            });
        }
    };

    // Basic size validation (Halo2/KZG proofs are typically 400-2000 bytes)
    if proof_bytes.len() < 100 || proof_bytes.len() > 100_000 {
        return Json(VerifyResponse {
            valid: false,
            error: Some(format!("Invalid proof size: {} bytes", proof_bytes.len())),
        });
    }

    // Parse public inputs from hex-encoded little-endian Fr repr bytes
    let public_inputs = match parse_public_inputs(&request.public_inputs) {
        Ok(inputs) => inputs,
        Err(e) => {
            return Json(VerifyResponse {
                valid: false,
                error: Some(e),
            });
        }
    };

    // Auto-detect physics domain from magic bytes
    if proof_bytes.len() >= 4 {
        let magic = &proof_bytes[0..4];
        if magic == EULER3D_MAGIC || magic == NS_IMEX_MAGIC || magic == THERMAL_MAGIC {
            return verify_physics_domain(&state, &proof_bytes, &public_inputs);
        }
    }

    // Default: FluidElite inference verification
    let proof = Halo2Proof {
        inner: crate::prover::FluidEliteProof {
            proof_bytes,
            generation_time_ms: 0,
            num_constraints: 0,
        },
        public_inputs,
    };

    match state.verifier.verify(&proof) {
        Ok(result) => {
            info!(
                "Verification complete: valid={}, time={}μs",
                result.valid, result.verification_time_us
            );
            Json(VerifyResponse {
                valid: result.valid,
                error: None,
            })
        }
        Err(e) => {
            warn!("Verification error: {}", e);
            Json(VerifyResponse {
                valid: false,
                error: Some(format!("Verification error: {}", e)),
            })
        }
    }
}

/// Parse hex-encoded public inputs into Fr elements.
#[cfg(feature = "server")]
fn parse_public_inputs(hex_inputs: &[String]) -> Result<Vec<halo2_axiom::halo2curves::bn256::Fr>, String> {
    use halo2_axiom::halo2curves::bn256::Fr;
    use halo2_axiom::halo2curves::ff::PrimeField;

    let mut public_inputs: Vec<Fr> = Vec::with_capacity(hex_inputs.len());
    for (i, hex_str) in hex_inputs.iter().enumerate() {
        let hex_clean = hex_str.strip_prefix("0x").unwrap_or(hex_str);
        let bytes = hex::decode(hex_clean)
            .map_err(|e| format!("Invalid hex in public_inputs[{}]: {}", i, e))?;
        if bytes.len() != 32 {
            return Err(format!(
                "public_inputs[{}]: expected 32 bytes, got {}",
                i,
                bytes.len()
            ));
        }
        let repr: [u8; 32] = bytes.try_into().unwrap();
        let fr_opt: subtle::CtOption<Fr> = Fr::from_repr(repr);
        if bool::from(fr_opt.is_none()) {
            return Err(format!(
                "public_inputs[{}]: not a valid BN254 Fr element",
                i
            ));
        }
        public_inputs.push(fr_opt.unwrap());
    }
    Ok(public_inputs)
}

/// Extract raw Halo2 proof bytes from a domain-serialized proof.
///
/// Domain proof wire format (shared by all domains):
///   bytes 0-3:   magic (E3DP | NSIP | THEP)
///   bytes 4-7:   version (u32 LE)
///   bytes 8-11:  proof_bytes_len (u32 LE)
///   bytes 12..:  raw Halo2/KZG proof bytes
#[cfg(feature = "server")]
fn extract_halo2_proof_from_domain(domain_bytes: &[u8]) -> Result<Vec<u8>, String> {
    if domain_bytes.len() < 12 {
        return Err(format!(
            "Domain proof too short: {} bytes, need at least 12",
            domain_bytes.len()
        ));
    }
    let proof_len = u32::from_le_bytes(
        domain_bytes[8..12]
            .try_into()
            .map_err(|_| "Failed to parse proof length")?,
    ) as usize;

    if domain_bytes.len() < 12 + proof_len {
        return Err(format!(
            "Domain proof truncated: header says {} proof bytes, but only {} bytes remain",
            proof_len,
            domain_bytes.len() - 12
        ));
    }

    Ok(domain_bytes[12..12 + proof_len].to_vec())
}

/// Verify a physics domain proof using the appropriate domain verifier.
///
/// Detects domain from magic bytes, extracts the raw Halo2 proof,
/// constructs a minimal domain proof struct, and verifies using
/// the lazily-initialized domain verifier.
#[cfg(feature = "server")]
fn verify_physics_domain(
    state: &Arc<ServerState>,
    domain_bytes: &[u8],
    public_inputs: &[halo2_axiom::halo2curves::bn256::Fr],
) -> Json<VerifyResponse> {
    let magic = &domain_bytes[0..4];

    // Extract the raw Halo2 proof bytes from the domain wire format
    let raw_proof = match extract_halo2_proof_from_domain(domain_bytes) {
        Ok(p) => p,
        Err(e) => {
            return Json(VerifyResponse {
                valid: false,
                error: Some(format!("Failed to parse domain proof: {}", e)),
            });
        }
    };

    let domain_name = if magic == EULER3D_MAGIC {
        "euler3d"
    } else if magic == NS_IMEX_MAGIC {
        "ns_imex"
    } else if magic == THERMAL_MAGIC {
        "thermal"
    } else {
        return Json(VerifyResponse {
            valid: false,
            error: Some("Unknown domain magic bytes".to_string()),
        });
    };

    info!("Verifying physics domain proof: domain={}", domain_name);

    let verify_result: Result<bool, String> = match domain_name {
        "euler3d" => {
            let guard = state.physics.euler3d.lock().unwrap();
            match guard.as_ref() {
                Some((_, verifier)) => {
                    // Construct minimal Euler3DProof with raw Halo2 bytes
                    let proof = euler3d::Euler3DProof {
                        proof_bytes: raw_proof,
                        generation_time_ms: 0,
                        num_constraints: 0,
                        k: 0,
                        params: euler3d::Euler3DParams::default(),
                        conservation_residuals: vec![],
                        input_state_hash_limbs: [0; 4],
                        output_state_hash_limbs: [0; 4],
                        params_hash_limbs: [0; 4],
                    };
                    verifier
                        .verify_with_public_inputs(&proof, public_inputs)
                        .map(|r| r.valid)
                }
                None => Err(
                    "Euler3D verifier not initialized. Generate a proof first.".to_string(),
                ),
            }
        }
        "ns_imex" => {
            let guard = state.physics.ns_imex.lock().unwrap();
            match guard.as_ref() {
                Some((_, verifier)) => {
                    let proof = ns_imex::NSIMEXProof {
                        proof_bytes: raw_proof,
                        generation_time_ms: 0,
                        num_constraints: 0,
                        k: 0,
                        params: ns_imex::NSIMEXParams::test_small(),
                        ke_residual: Q16::from_raw(0),
                        enstrophy_residual: Q16::from_raw(0),
                        divergence_residual: Q16::from_raw(0),
                        input_state_hash_limbs: [0; 4],
                        output_state_hash_limbs: [0; 4],
                        params_hash_limbs: [0; 4],
                    };
                    verifier
                        .verify_with_public_inputs(&proof, public_inputs)
                        .map(|r| r.valid)
                }
                None => Err(
                    "NS-IMEX verifier not initialized. Generate a proof first.".to_string(),
                ),
            }
        }
        "thermal" => {
            let guard = state.physics.thermal.lock().unwrap();
            match guard.as_ref() {
                Some((_, verifier)) => {
                    let proof = thermal::ThermalProof {
                        proof_bytes: raw_proof,
                        generation_time_ms: 0,
                        num_constraints: 0,
                        k: 0,
                        params: thermal::ThermalParams::test_small(),
                        conservation_residual: Q16::from_raw(0),
                        cg_residual_norm: Q16::from_raw(0),
                        cg_iterations: 0,
                        input_state_hash_limbs: [0; 4],
                        output_state_hash_limbs: [0; 4],
                        params_hash_limbs: [0; 4],
                    };
                    verifier
                        .verify_with_public_inputs(&proof, public_inputs)
                        .map(|r| r.valid)
                }
                None => Err(
                    "Thermal verifier not initialized. Generate a proof first.".to_string(),
                ),
            }
        }
        _ => unreachable!(),
    };

    match verify_result {
        Ok(valid) => {
            info!(
                "Physics verification complete: domain={}, valid={}",
                domain_name, valid
            );
            Json(VerifyResponse {
                valid,
                error: None,
            })
        }
        Err(e) => {
            warn!(
                "Physics verification error: domain={}, error={}",
                domain_name, e
            );
            Json(VerifyResponse {
                valid: false,
                error: Some(format!("Verification error ({}): {}", domain_name, e)),
            })
        }
    }
}

/// Start the HTTP server (legacy function, use server binary instead)
#[cfg(feature = "server")]
pub async fn start_server(port: u16, config: CircuitConfig) -> Result<(), Box<dyn std::error::Error>>
{
    info!("Initializing FluidElite API server...");

    // Create prover
    let w_hidden = MPO::identity(config.num_sites, config.phys_dim);
    let w_input = MPO::identity(config.num_sites, config.phys_dim);
    let readout_weights = vec![Q16::from_f64(0.1); config.chi_max * config.vocab_size];

    let prover = FluidEliteProver::new(w_hidden, w_input, readout_weights, config.clone());

    let state = Arc::new(ServerState::new(prover, config));

    let app = create_router(state);

    let addr = format!("0.0.0.0:{}", port);
    info!("Starting server on http://{}", addr);
    info!("Endpoints:");
    info!("  GET  /health  - Health check");
    info!("  GET  /ready   - Readiness check");
    info!("  GET  /stats   - Prover statistics");
    info!("  GET  /metrics - Prometheus metrics");
    info!("  POST /prove   - Generate ZK proof");
    info!("  POST /verify  - Verify ZK proof");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
