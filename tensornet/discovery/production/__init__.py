"""
Production Hardening Module for Autonomous Discovery Engine

Phase 7: Production-grade resilience, observability, security, and performance.

Components:
    - Resilience: Circuit breakers, rate limiting, retry logic
    - Observability: Structured logging, metrics, health checks
    - Security: Input validation, authentication, audit logging
    - Performance: Caching, connection pooling, batch optimization
"""

from .resilience import (
    CircuitBreaker,
    CircuitState,
    RateLimiter,
    RetryPolicy,
    RetryStrategy,
    Bulkhead,
    Timeout,
    resilient,
)

from .observability import (
    StructuredLogger,
    MetricsCollector,
    HealthChecker,
    HealthStatus,
    Tracer,
    Span,
    get_logger,
    get_metrics,
)

from .security import (
    InputValidator,
    ValidationError,
    APIKeyAuth,
    RequestSigner,
    AuditLogger,
    AuditEvent,
    sanitize_input,
)

from .performance import (
    CacheManager,
    CachePolicy,
    ConnectionPool,
    BatchOptimizer,
    MemoryManager,
    PerformanceProfiler,
)

__all__ = [
    # Resilience
    "CircuitBreaker",
    "CircuitState",
    "RateLimiter",
    "RetryPolicy",
    "RetryStrategy",
    "Bulkhead",
    "Timeout",
    "resilient",
    # Observability
    "StructuredLogger",
    "MetricsCollector",
    "HealthChecker",
    "HealthStatus",
    "Tracer",
    "Span",
    "get_logger",
    "get_metrics",
    # Security
    "InputValidator",
    "ValidationError",
    "APIKeyAuth",
    "RequestSigner",
    "AuditLogger",
    "AuditEvent",
    "sanitize_input",
    # Performance
    "CacheManager",
    "CachePolicy",
    "ConnectionPool",
    "BatchOptimizer",
    "MemoryManager",
    "PerformanceProfiler",
]
