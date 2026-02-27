//! Token-bucket rate limiter for the FluidElite ZK prover API.
//!
//! Implements a per-API-key token bucket with configurable capacity and refill
//! rate. Returns HTTP 429 with a `Retry-After` header when the bucket is empty.
//!
//! # Design
//!
//! - **Thread-safe**: Uses `DashMap` for lock-free concurrent access across
//!   Tokio tasks (no write-lock contention on the global map).
//! - **Per-key buckets**: Each API key gets an independent bucket sized by the
//!   global config. Unauthenticated requests share a single "anonymous" bucket.
//! - **Lazy initialization**: Buckets are created on first request (no
//!   pre-registration needed).
//! - **Background cleanup**: A Tokio task periodically evicts expired buckets
//!   to prevent unbounded memory growth from transient API keys.
//!
//! # Integration
//!
//! ```rust,ignore
//! use fluidelite_zk::rate_limit::RateLimiter;
//!
//! let limiter = RateLimiter::new(60, 10); // 60 tokens, refill 10/sec
//! let app = Router::new()
//!     .route("/prove", post(prove_handler))
//!     .layer(axum::middleware::from_fn_with_state(
//!         Arc::new(limiter),
//!         rate_limit_middleware,
//!     ));
//! ```

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, Request, Response, StatusCode};
use axum::middleware::Next;
use dashmap::DashMap;
use tracing::warn;

/// A single token bucket for one API key.
#[derive(Debug)]
struct Bucket {
    /// Current number of available tokens (can go negative during bursts if
    /// we atomically check-and-decrement in a racy way — but DashMap entry
    /// locks prevent this).
    tokens: f64,
    /// When the bucket was last refilled.
    last_refill: Instant,
    /// Maximum capacity (tokens stop accumulating beyond this).
    capacity: f64,
    /// Tokens added per second.
    refill_rate: f64,
}

impl Bucket {
    fn new(capacity: u32, refill_rate: u32) -> Self {
        Self {
            tokens: capacity as f64,
            last_refill: Instant::now(),
            capacity: capacity as f64,
            refill_rate: refill_rate as f64,
        }
    }

    /// Refills the bucket based on elapsed time and attempts to consume one
    /// token. Returns `Ok(())` if allowed, or `Err(retry_after_secs)` if
    /// the bucket is empty.
    fn try_acquire(&mut self) -> Result<(), u64> {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();

        // Refill tokens proportional to elapsed time.
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.capacity);
        self.last_refill = now;

        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            Ok(())
        } else {
            // Compute how long until one token is available.
            let deficit = 1.0 - self.tokens;
            let wait_secs = (deficit / self.refill_rate).ceil() as u64;
            Err(wait_secs.max(1))
        }
    }

    /// Returns true if this bucket has been idle long enough to evict.
    fn is_expired(&self, ttl: Duration) -> bool {
        self.last_refill.elapsed() > ttl
    }
}

/// Global rate limiter managing per-key token buckets.
#[derive(Debug)]
pub struct RateLimiter {
    /// Per-key buckets (keyed by API key or "anonymous").
    buckets: DashMap<String, Bucket>,
    /// Maximum tokens per bucket.
    capacity: u32,
    /// Tokens added per second per bucket.
    refill_rate: u32,
    /// Time after which idle buckets are evicted.
    bucket_ttl: Duration,
}

impl RateLimiter {
    /// Creates a new rate limiter.
    ///
    /// # Arguments
    ///
    /// * `capacity` — Maximum burst size (tokens per bucket). A value of 60
    ///   allows up to 60 requests in a burst before throttling.
    /// * `refill_rate` — Tokens added per second. A value of 1 allows 1
    ///   request/second sustained, with bursts up to `capacity`.
    pub fn new(capacity: u32, refill_rate: u32) -> Self {
        assert!(capacity > 0, "capacity must be > 0");
        assert!(refill_rate > 0, "refill_rate must be > 0");
        Self {
            buckets: DashMap::new(),
            capacity,
            refill_rate,
            bucket_ttl: Duration::from_secs(3600), // 1 hour default TTL
        }
    }

    /// Creates a rate limiter from requests-per-minute configuration.
    ///
    /// Converts RPM to a token bucket: capacity = rpm (allows full burst),
    /// refill_rate = rpm / 60 (sustained rate).
    pub fn from_rpm(requests_per_minute: u32) -> Self {
        let refill_rate = (requests_per_minute / 60).max(1);
        Self::new(requests_per_minute, refill_rate)
    }

    /// Attempts to acquire a token for the given key.
    ///
    /// Returns `Ok(())` if the request is allowed, or `Err(retry_after)`
    /// with the number of seconds to wait.
    pub fn try_acquire(&self, key: &str) -> Result<(), u64> {
        self.buckets
            .entry(key.to_string())
            .or_insert_with(|| Bucket::new(self.capacity, self.refill_rate))
            .try_acquire()
    }

    /// Evicts buckets that have been idle longer than the TTL.
    /// Called periodically by the cleanup task.
    pub fn evict_expired(&self) {
        self.buckets
            .retain(|_key, bucket| !bucket.is_expired(self.bucket_ttl));
    }

    /// Spawns a background Tokio task that evicts expired buckets every
    /// `interval`. Returns a `JoinHandle` that can be used to abort the
    /// task on shutdown.
    pub fn spawn_cleanup_task(
        self: &Arc<Self>,
        interval: Duration,
    ) -> tokio::task::JoinHandle<()> {
        let limiter = Arc::clone(self);
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            loop {
                ticker.tick().await;
                limiter.evict_expired();
            }
        })
    }
}

/// Axum middleware that enforces per-API-key token-bucket rate limiting.
///
/// Skips rate limiting for public endpoints (`/health`, `/ready`, `/metrics`,
/// `/stats`). For authenticated requests, the API key from the `Authorization:
/// Bearer <key>` header is used as the bucket key. Unauthenticated requests
/// share the "anonymous" bucket.
///
/// On rate limit exceeded, returns HTTP 429 with:
/// - `Retry-After` header indicating seconds to wait
/// - JSON error body with rate limit details
pub async fn rate_limit_middleware(
    State(limiter): State<Arc<RateLimiter>>,
    request: Request<Body>,
    next: Next,
) -> Result<Response<Body>, Response<Body>> {
    let path = request.uri().path();

    // Skip rate limiting for public/health endpoints.
    if matches!(path, "/health" | "/ready" | "/metrics" | "/stats") {
        return Ok(next.run(request).await);
    }

    // Extract the API key from the Authorization header, or use "anonymous".
    let key = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|auth| auth.strip_prefix("Bearer "))
        .unwrap_or("anonymous");

    match limiter.try_acquire(key) {
        Ok(()) => Ok(next.run(request).await),
        Err(retry_after) => {
            warn!(
                key_prefix = &key[..key.len().min(8)],
                retry_after_secs = retry_after,
                path = path,
                "Rate limit exceeded"
            );

            let body = serde_json::json!({
                "error": "rate_limit_exceeded",
                "message": "Too many requests. Please retry after the indicated delay.",
                "retry_after_seconds": retry_after,
            });

            let response = Response::builder()
                .status(StatusCode::TOO_MANY_REQUESTS)
                .header(header::RETRY_AFTER, retry_after.to_string())
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap_or_default()))
                .unwrap_or_else(|_| {
                    Response::builder()
                        .status(StatusCode::TOO_MANY_REQUESTS)
                        .body(Body::empty())
                        .expect("infallible empty response")
                });

            Err(response)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_allows_up_to_capacity() {
        let limiter = RateLimiter::new(5, 1);
        for _ in 0..5 {
            assert!(limiter.try_acquire("test-key").is_ok());
        }
        // 6th request should fail.
        assert!(limiter.try_acquire("test-key").is_err());
    }

    #[test]
    fn test_different_keys_independent() {
        let limiter = RateLimiter::new(2, 1);
        assert!(limiter.try_acquire("key-a").is_ok());
        assert!(limiter.try_acquire("key-a").is_ok());
        assert!(limiter.try_acquire("key-a").is_err());
        // key-b should still be full.
        assert!(limiter.try_acquire("key-b").is_ok());
        assert!(limiter.try_acquire("key-b").is_ok());
        assert!(limiter.try_acquire("key-b").is_err());
    }

    #[test]
    fn test_retry_after_is_positive() {
        let limiter = RateLimiter::new(1, 1);
        assert!(limiter.try_acquire("test").is_ok());
        match limiter.try_acquire("test") {
            Err(retry_after) => assert!(retry_after >= 1),
            Ok(()) => panic!("should have been rate limited"),
        }
    }

    #[test]
    fn test_from_rpm() {
        let limiter = RateLimiter::from_rpm(120);
        // capacity=120, refill_rate=2/sec
        // Should allow 120 burst requests.
        for _ in 0..120 {
            assert!(limiter.try_acquire("burst").is_ok());
        }
        assert!(limiter.try_acquire("burst").is_err());
    }

    #[test]
    fn test_evict_expired() {
        let limiter = RateLimiter {
            buckets: DashMap::new(),
            capacity: 10,
            refill_rate: 1,
            bucket_ttl: Duration::from_millis(1), // Very short TTL for testing
        };
        limiter.try_acquire("ephemeral").ok();
        // Wait for TTL to expire.
        std::thread::sleep(Duration::from_millis(10));
        limiter.evict_expired();
        assert!(limiter.buckets.is_empty());
    }

    #[test]
    fn test_refill_restores_tokens() {
        let limiter = RateLimiter::new(2, 1000); // Very fast refill for testing
        assert!(limiter.try_acquire("fast").is_ok());
        assert!(limiter.try_acquire("fast").is_ok());
        assert!(limiter.try_acquire("fast").is_err());

        // Wait for refill — 1000 tokens/sec means ~1ms per token.
        std::thread::sleep(Duration::from_millis(5));
        assert!(limiter.try_acquire("fast").is_ok());
    }
}
