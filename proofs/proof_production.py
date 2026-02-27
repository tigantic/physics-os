"""
Phase 7: Production Hardening - Proof Tests

Comprehensive verification of resilience, observability, security, and performance.
"""

import time
import threading
from datetime import datetime, timezone

# Resilience tests
from tensornet.ml.discovery.production.resilience import (
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError, CircuitState,
    RateLimiter, RateLimiterConfig, RateLimitExceeded,
    RetryPolicy, RetryStrategy,
    Bulkhead, BulkheadConfig, BulkheadFull,
    Timeout, TimeoutError,
    resilient, ResilienceConfig,
)

# Observability tests
from tensornet.ml.discovery.production.observability import (
    StructuredLogger, LogLevel, LogContext,
    MetricsCollector, MetricType,
    HealthChecker, HealthStatus, ComponentHealth,
    Tracer, Span,
    get_logger, get_metrics,
)

# Security tests
from tensornet.ml.discovery.production.security import (
    InputValidator, ValidationError, FieldSpec,
    RequiredRule, TypeRule, RangeRule, LengthRule, PatternRule, EnumRule,
    APIKeyAuth, AuthenticationError, AuthorizationError,
    RequestSigner,
    AuditLogger, AuditEventType, AuditEvent,
    CSPPolicy, get_security_headers, sanitize_input,
)

# Performance tests
from tensornet.ml.discovery.production.performance import (
    CacheManager, CachePolicy, CacheStats,
    ConnectionPool, PooledConnection,
    BatchOptimizer, BatchConfig,
    MemoryManager, MemoryConfig,
    PerformanceProfiler,
    cached,
)


def test_circuit_breaker_transitions():
    """Test circuit breaker state transitions."""
    config = CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout_seconds=0.1,
    )
    cb = CircuitBreaker("test_cb", config)
    
    # Initial state is CLOSED
    assert cb.state == CircuitState.CLOSED
    
    # Record failures to open circuit
    for i in range(3):
        cb.record_failure()
    
    assert cb.state == CircuitState.OPEN
    assert not cb.allow_request()
    
    # Wait for timeout
    time.sleep(0.15)
    
    # Should transition to HALF_OPEN
    assert cb.state == CircuitState.HALF_OPEN
    assert cb.allow_request()
    
    # Record successes to close
    cb.record_success()
    cb.record_success()
    
    assert cb.state == CircuitState.CLOSED


def test_circuit_breaker_decorator():
    """Test circuit breaker as decorator."""
    config = CircuitBreakerConfig(failure_threshold=2)
    cb = CircuitBreaker("decorator_cb", config)
    
    call_count = 0
    
    @cb
    def failing_function():
        nonlocal call_count
        call_count += 1
        raise ValueError("Test error")
    
    # First two calls should fail
    for _ in range(2):
        try:
            failing_function()
        except ValueError:
            pass
    
    assert call_count == 2
    assert cb.state == CircuitState.OPEN
    
    # Third call should be rejected
    try:
        failing_function()
        assert False, "Should raise CircuitBreakerError"
    except CircuitBreakerError as e:
        assert e.circuit_name == "decorator_cb"


def test_rate_limiter_token_bucket():
    """Test rate limiter token bucket algorithm."""
    config = RateLimiterConfig(
        requests_per_second=10.0,
        burst_size=5,
    )
    limiter = RateLimiter(config)
    
    # Should allow burst
    for _ in range(5):
        assert limiter.try_acquire()
    
    # Sixth should fail
    assert not limiter.try_acquire()
    
    # Wait for refill
    time.sleep(0.11)  # Should refill 1 token
    assert limiter.try_acquire()


def test_rate_limiter_decorator():
    """Test rate limiter as decorator."""
    config = RateLimiterConfig(
        requests_per_second=100.0,
        burst_size=3,
    )
    limiter = RateLimiter(config)
    
    @limiter
    def rate_limited_fn():
        return "success"
    
    # First 3 should work
    for _ in range(3):
        assert rate_limited_fn() == "success"


def test_retry_policy_exponential_backoff():
    """Test retry policy with exponential backoff."""
    policy = RetryPolicy(
        max_attempts=3,
        strategy=RetryStrategy.EXPONENTIAL,
        base_delay=0.01,  # 10ms for testing
        max_delay=1.0,
    )
    
    # Test delay calculation
    assert policy.get_delay(0) == 0.01  # 10ms
    assert policy.get_delay(1) == 0.02  # 20ms
    assert policy.get_delay(2) == 0.04  # 40ms


def test_retry_policy_decorator():
    """Test retry policy as decorator."""
    attempt_count = 0
    
    policy = RetryPolicy(
        max_attempts=3,
        strategy=RetryStrategy.FIXED,
        base_delay=0.01,
    )
    
    @policy
    def flaky_function():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ValueError("Not yet")
        return "success"
    
    result = flaky_function()
    assert result == "success"
    assert attempt_count == 3


def test_bulkhead_concurrency_limit():
    """Test bulkhead limits concurrent execution."""
    config = BulkheadConfig(max_concurrent=2, max_wait_seconds=0.1)
    bulkhead = Bulkhead("test_bulkhead", config)
    
    # Acquire 2 slots
    assert bulkhead.acquire()
    assert bulkhead.acquire()
    assert bulkhead.active_count == 2
    assert bulkhead.available == 0
    
    # Third should fail
    assert not bulkhead.acquire(timeout=0.01)
    
    # Release one
    bulkhead.release()
    assert bulkhead.available == 1
    
    # Now should work
    assert bulkhead.acquire()


def test_timeout_wrapper():
    """Test timeout wrapper."""
    @Timeout(0.05)
    def fast_function():
        return "fast"
    
    @Timeout(0.01)
    def slow_function():
        time.sleep(0.1)
        return "slow"
    
    assert fast_function() == "fast"
    
    try:
        slow_function()
        assert False, "Should timeout"
    except TimeoutError:
        pass


def test_resilient_combined_decorator():
    """Test combined resilience patterns."""
    call_count = 0
    
    @resilient(
        name="test_operation",
        config=ResilienceConfig(
            rate_limiter=RateLimiterConfig(requests_per_second=100, burst_size=10),
            bulkhead=BulkheadConfig(max_concurrent=5),
            timeout_seconds=1.0,
        ),
    )
    def protected_function():
        nonlocal call_count
        call_count += 1
        return "protected"
    
    result = protected_function()
    assert result == "protected"
    assert call_count == 1


def test_structured_logger():
    """Test structured logger output."""
    context = LogContext(
        service="test-service",
        version="1.0.0",
        environment="test",
    )
    logger = StructuredLogger("test", LogLevel.DEBUG, context, json_output=False)
    
    # Should not raise
    logger.debug("Debug message", extra_field="value")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")


def test_metrics_collector_counter():
    """Test metrics counter."""
    metrics = MetricsCollector("test")
    
    metrics.counter("requests", 1, labels={"endpoint": "/api"})
    metrics.counter("requests", 1, labels={"endpoint": "/api"})
    metrics.counter("requests", 1, labels={"endpoint": "/health"})
    
    all_metrics = metrics.get_metrics()
    request_metrics = [m for m in all_metrics if "requests" in m.name]
    
    assert len(request_metrics) == 2  # Two label combinations


def test_metrics_collector_gauge():
    """Test metrics gauge."""
    metrics = MetricsCollector("test")
    
    metrics.gauge("temperature", 72.5, labels={"room": "server"})
    metrics.gauge("temperature", 68.0, labels={"room": "server"})  # Update
    
    all_metrics = metrics.get_metrics()
    temp_metrics = [m for m in all_metrics if "temperature" in m.name]
    
    assert len(temp_metrics) == 1
    assert temp_metrics[0].value == 68.0


def test_metrics_collector_histogram():
    """Test metrics histogram."""
    metrics = MetricsCollector("test")
    
    for latency in [0.005, 0.01, 0.05, 0.1, 0.5]:
        metrics.histogram("latency", latency)
    
    all_metrics = metrics.get_metrics()
    latency_metrics = [m for m in all_metrics if "latency" in m.name]
    
    # Should have sum, count, and buckets
    assert any("sum" in m.name for m in latency_metrics)
    assert any("count" in m.name for m in latency_metrics)
    assert any("bucket" in m.name for m in latency_metrics)


def test_metrics_timer_context():
    """Test metrics timer context manager."""
    metrics = MetricsCollector("test")
    
    with metrics.timer("operation_duration"):
        time.sleep(0.01)
    
    all_metrics = metrics.get_metrics()
    duration_metrics = [m for m in all_metrics if "operation_duration" in m.name]
    
    assert len(duration_metrics) > 0


def test_health_checker():
    """Test health checker."""
    checker = HealthChecker("test-service")
    
    # Register healthy check
    def healthy_check() -> ComponentHealth:
        return ComponentHealth(
            name="database",
            status=HealthStatus.HEALTHY,
            message="Connected",
        )
    
    checker.register_check("database", healthy_check)
    
    health = checker.check()
    assert health.status == HealthStatus.HEALTHY
    assert len(health.components) == 1
    assert health.components[0].name == "database"


def test_health_checker_degraded():
    """Test health checker with degraded component."""
    checker = HealthChecker("test-service")
    
    checker.register_check("cache", lambda: ComponentHealth(
        name="cache",
        status=HealthStatus.DEGRADED,
        message="High latency",
    ))
    
    checker.register_check("database", lambda: ComponentHealth(
        name="database",
        status=HealthStatus.HEALTHY,
    ))
    
    health = checker.check()
    assert health.status == HealthStatus.DEGRADED


def test_tracer_span_creation():
    """Test distributed tracing span creation."""
    tracer = Tracer("test-service")
    
    with tracer.trace("test_operation", tags={"key": "value"}) as span:
        span.log("event", message="Processing")
        time.sleep(0.01)
    
    spans = tracer.get_spans()
    assert len(spans) == 1
    assert spans[0].operation_name == "test_operation"
    assert spans[0].duration_ms >= 10
    assert spans[0].status == "OK"


def test_tracer_nested_spans():
    """Test nested span creation."""
    tracer = Tracer("test-service")
    
    with tracer.trace("parent") as parent:
        with tracer.trace("child", parent=parent) as child:
            pass
    
    spans = tracer.get_spans()
    assert len(spans) == 2
    
    child_span = [s for s in spans if s.operation_name == "child"][0]
    parent_span = [s for s in spans if s.operation_name == "parent"][0]
    
    assert child_span.trace_id == parent_span.trace_id
    assert child_span.parent_id == parent_span.span_id


def test_input_validator_rules():
    """Test input validation rules."""
    validator = InputValidator()
    
    # Add field specs
    validator.add_field("user", FieldSpec(
        name="email",
        rules=[RequiredRule(), LengthRule(min_length=5, max_length=100)],
    ))
    validator.add_field("user", FieldSpec(
        name="age",
        rules=[TypeRule(int), RangeRule(min_value=0, max_value=150)],
    ))
    
    # Valid data
    valid_data = {"email": "test@example.com", "age": 25}
    result = validator.validate("user", valid_data)
    assert result == valid_data
    
    # Invalid data
    try:
        validator.validate("user", {"email": "", "age": 25})
        assert False, "Should fail"
    except ValidationError:
        pass


def test_input_sanitization():
    """Test input sanitization."""
    # XSS attempt
    malicious = '<script>alert("xss")</script>'
    sanitized = sanitize_input(malicious)
    assert "<script>" not in sanitized
    assert "&lt;script&gt;" in sanitized
    
    # SQL injection check
    sql_injection = "'; DROP TABLE users; --"
    assert not InputValidator.is_safe(sql_injection)


def test_api_key_auth():
    """Test API key authentication."""
    auth = APIKeyAuth()
    
    # Generate key
    raw_key, api_key = auth.generate_key(
        name="test-key",
        permissions={"read", "write"},
        rate_limit=100,
    )
    
    assert raw_key.startswith("tn_")
    assert api_key.has_permission("read")
    assert api_key.has_permission("write")
    assert not api_key.has_permission("admin")
    
    # Validate key
    validated = auth.validate(raw_key)
    assert validated.name == "test-key"
    
    # Invalid key
    try:
        auth.validate("invalid_key")
        assert False, "Should fail"
    except AuthenticationError:
        pass


def test_api_key_expiration():
    """Test API key expiration."""
    auth = APIKeyAuth()
    
    # Create expired key (hack: set expires_at in past)
    raw_key, api_key = auth.generate_key(
        name="expired-key",
        expires_in_days=0,  # Will set expires_at to now
    )
    
    # Manually expire it
    api_key.expires_at = time.time() - 1
    
    try:
        auth.validate(raw_key)
        assert False, "Should fail"
    except AuthenticationError as e:
        assert "expired" in str(e).lower()


def test_request_signer():
    """Test request signing."""
    signer = RequestSigner(secret_key="test-secret")
    
    # Sign request
    headers = signer.sign("POST", "/api/data", '{"key": "value"}')
    
    assert "X-Signature" in headers
    assert "X-Timestamp" in headers
    
    # Verify signature
    result = signer.verify(
        "POST",
        "/api/data",
        headers["X-Signature"],
        int(headers["X-Timestamp"]),
        '{"key": "value"}',
    )
    assert result
    
    # Invalid signature
    try:
        signer.verify("POST", "/api/data", "invalid", int(time.time()))
        assert False, "Should fail"
    except AuthenticationError:
        pass


def test_audit_logger():
    """Test audit logging."""
    audit = AuditLogger("test-service")
    
    # Log authentication
    event = audit.log_authentication("user123", success=True)
    assert event.event_type == AuditEventType.AUTHENTICATION
    assert event.actor == "user123"
    
    # Log API call
    event = audit.log_api_call(
        actor="user123",
        endpoint="/api/data",
        method="GET",
        status_code=200,
        latency_ms=15.5,
    )
    assert event.event_type == AuditEventType.API_CALL
    
    # Query events
    events = audit.get_events(actor="user123")
    assert len(events) == 2


def test_security_headers():
    """Test security headers generation."""
    csp = CSPPolicy()
    headers = get_security_headers(csp)
    
    assert "X-Content-Type-Options" in headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert "Strict-Transport-Security" in headers
    assert "Content-Security-Policy" in headers


def test_cache_manager_lru():
    """Test LRU cache."""
    cache: CacheManager[str, str] = CacheManager(
        max_size=3,
        policy=CachePolicy.LRU,
    )
    
    cache.set("a", "1")
    cache.set("b", "2")
    cache.set("c", "3")
    
    # Access 'a' to make it recently used
    cache.get("a")
    
    # Add 'd', should evict 'b' (least recently used)
    cache.set("d", "4")
    
    assert cache.get("a") == "1"
    assert cache.get("b") is None  # Evicted
    assert cache.get("c") == "3"
    assert cache.get("d") == "4"


def test_cache_manager_ttl():
    """Test cache TTL expiration."""
    cache: CacheManager[str, str] = CacheManager(
        max_size=100,
        default_ttl=0.05,  # 50ms
    )
    
    cache.set("key", "value")
    assert cache.get("key") == "value"
    
    time.sleep(0.06)
    assert cache.get("key") is None  # Expired


def test_cache_decorator():
    """Test cache decorator."""
    call_count = 0
    cache = CacheManager(max_size=100)
    
    @cached(cache=cache, ttl=1.0)
    def expensive_function(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 2
    
    # First call
    assert expensive_function(5) == 10
    assert call_count == 1
    
    # Cached call
    assert expensive_function(5) == 10
    assert call_count == 1  # Not called again
    
    # Different argument
    assert expensive_function(10) == 20
    assert call_count == 2


def test_connection_pool():
    """Test connection pool."""
    created = []
    closed = []
    
    def factory():
        conn = f"conn-{len(created)}"
        created.append(conn)
        return conn
    
    def cleanup(conn):
        closed.append(conn)
    
    pool: ConnectionPool[str] = ConnectionPool(
        factory=factory,
        max_size=2,
        cleanup=cleanup,
    )
    
    # Acquire connections
    conn1 = pool.acquire()
    conn2 = pool.acquire()
    
    assert len(created) == 2
    
    stats = pool.get_stats()
    assert stats["in_use"] == 2
    assert stats["available"] == 0
    
    # Release and reuse
    pool.release(conn1)
    conn3 = pool.acquire()
    
    assert conn3 == conn1  # Reused
    assert len(created) == 2  # No new creation
    
    # Cleanup
    pool.close()
    assert len(closed) == 2


def test_connection_pool_context_manager():
    """Test connection pool context manager."""
    pool: ConnectionPool[dict] = ConnectionPool(
        factory=lambda: {"id": time.time()},
        max_size=5,
    )
    
    with pool.connection() as conn:
        assert "id" in conn
    
    # Connection should be released
    stats = pool.get_stats()
    assert stats["in_use"] == 0


def test_batch_optimizer():
    """Test batch optimizer."""
    batches_processed = []
    
    def process_batch(items: list) -> list:
        batches_processed.append(len(items))
        return [x * 2 for x in items]
    
    optimizer = BatchOptimizer(
        batch_fn=process_batch,
        config=BatchConfig(max_batch_size=5, max_wait_ms=50),
    )
    
    optimizer.start()
    
    # Submit items
    result = optimizer.submit(5, timeout=1.0)
    assert result == 10
    
    optimizer.stop()


def test_memory_manager():
    """Test memory manager."""
    config = MemoryConfig(
        max_memory_mb=1024.0,
        gc_threshold_pct=80.0,
    )
    mem = MemoryManager(config)
    
    # Get usage
    usage = mem.get_memory_usage()
    assert "rss_mb" in usage or "rss_bytes" in usage
    
    # Check pressure
    pressure = mem.check_pressure()
    assert pressure in ("ok", "warning", "critical")
    
    # Collect garbage
    gc_result = mem.collect_garbage()
    assert "collected_objects" in gc_result


def test_performance_profiler():
    """Test performance profiler."""
    profiler = PerformanceProfiler()
    
    with profiler.profile("test_operation", extra="metadata"):
        time.sleep(0.01)
    
    samples = profiler.get_samples()
    assert len(samples) == 1
    assert samples[0].operation == "test_operation"
    assert samples[0].duration_ms >= 10
    
    stats = profiler.get_statistics("test_operation")
    assert stats["count"] == 1
    assert stats["success_rate"] == 1.0


def test_performance_profiler_decorator():
    """Test profiler decorator."""
    profiler = PerformanceProfiler()
    
    @profiler.profile_fn("decorated_op")
    def decorated_function():
        return "result"
    
    result = decorated_function()
    assert result == "result"
    
    samples = profiler.get_samples("decorated_op")
    assert len(samples) == 1


# Run all tests
PROOF_TESTS = [
    # Resilience (9 tests)
    ("circuit_breaker_transitions", test_circuit_breaker_transitions),
    ("circuit_breaker_decorator", test_circuit_breaker_decorator),
    ("rate_limiter_token_bucket", test_rate_limiter_token_bucket),
    ("rate_limiter_decorator", test_rate_limiter_decorator),
    ("retry_policy_exponential_backoff", test_retry_policy_exponential_backoff),
    ("retry_policy_decorator", test_retry_policy_decorator),
    ("bulkhead_concurrency_limit", test_bulkhead_concurrency_limit),
    ("timeout_wrapper", test_timeout_wrapper),
    ("resilient_combined_decorator", test_resilient_combined_decorator),
    
    # Observability (8 tests)
    ("structured_logger", test_structured_logger),
    ("metrics_collector_counter", test_metrics_collector_counter),
    ("metrics_collector_gauge", test_metrics_collector_gauge),
    ("metrics_collector_histogram", test_metrics_collector_histogram),
    ("metrics_timer_context", test_metrics_timer_context),
    ("health_checker", test_health_checker),
    ("health_checker_degraded", test_health_checker_degraded),
    ("tracer_span_creation", test_tracer_span_creation),
    ("tracer_nested_spans", test_tracer_nested_spans),
    
    # Security (8 tests)
    ("input_validator_rules", test_input_validator_rules),
    ("input_sanitization", test_input_sanitization),
    ("api_key_auth", test_api_key_auth),
    ("api_key_expiration", test_api_key_expiration),
    ("request_signer", test_request_signer),
    ("audit_logger", test_audit_logger),
    ("security_headers", test_security_headers),
    
    # Performance (9 tests)
    ("cache_manager_lru", test_cache_manager_lru),
    ("cache_manager_ttl", test_cache_manager_ttl),
    ("cache_decorator", test_cache_decorator),
    ("connection_pool", test_connection_pool),
    ("connection_pool_context_manager", test_connection_pool_context_manager),
    ("batch_optimizer", test_batch_optimizer),
    ("memory_manager", test_memory_manager),
    ("performance_profiler", test_performance_profiler),
    ("performance_profiler_decorator", test_performance_profiler_decorator),
]


def run_all_tests():
    """Run all proof tests."""
    print("\n" + "=" * 60)
    print("PHASE 7: PRODUCTION HARDENING - PROOF TESTS")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for name, test_fn in PROOF_TESTS:
        try:
            test_fn()
            print(f"  ✓ {name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            failed += 1
    
    print("-" * 60)
    print(f"Results: {passed}/{len(PROOF_TESTS)} tests passed")
    
    if failed == 0:
        print("\n✅ ALL PROOF TESTS PASSED")
    else:
        print(f"\n❌ {failed} TESTS FAILED")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
