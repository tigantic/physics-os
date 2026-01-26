"""
Resilience Patterns for Production Systems

Circuit breakers, rate limiting, retry logic, bulkheads, and timeouts.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union
from enum import Enum
from functools import wraps
import time
import threading
import logging
import random
import asyncio
from collections import deque
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 3          # Successes to close from half-open
    timeout_seconds: float = 30.0       # Time before trying half-open
    excluded_exceptions: tuple = ()     # Exceptions that don't count as failures
    
    def __post_init__(self):
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.success_threshold < 1:
            raise ValueError("success_threshold must be >= 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    def __init__(self, circuit_name: str, state: CircuitState, retry_after: float):
        self.circuit_name = circuit_name
        self.state = state
        self.retry_after = retry_after
        super().__init__(
            f"Circuit '{circuit_name}' is {state.value}. Retry after {retry_after:.1f}s"
        )


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    
    Prevents cascading failures by stopping requests to failing services.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service failing, requests rejected immediately
    - HALF_OPEN: Testing recovery, limited requests allowed
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Circuit breaker name for logging/metrics
            config: Configuration options
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.RLock()
        self._listeners: List[Callable[[CircuitState, CircuitState], None]] = []
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            self._check_state_transition()
            return self._state
    
    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count
    
    def add_listener(self, listener: Callable[[CircuitState, CircuitState], None]) -> None:
        """Add state change listener."""
        self._listeners.append(listener)
    
    def _check_state_transition(self) -> None:
        """Check if state should transition based on timeout."""
        if self._state == CircuitState.OPEN and self._last_failure_time:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.config.timeout_seconds:
                self._transition_to(CircuitState.HALF_OPEN)
    
    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to new state."""
        if self._state != new_state:
            old_state = self._state
            self._state = new_state
            logger.info(f"Circuit '{self.name}': {old_state.value} -> {new_state.value}")
            
            # Reset counters on transition
            if new_state == CircuitState.CLOSED:
                self._failure_count = 0
                self._success_count = 0
            elif new_state == CircuitState.HALF_OPEN:
                self._success_count = 0
            
            # Notify listeners
            for listener in self._listeners:
                try:
                    listener(old_state, new_state)
                except Exception as e:
                    logger.error(f"Listener error: {e}")
    
    def record_success(self) -> None:
        """Record successful operation."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0
    
    def record_failure(self, exception: Optional[Exception] = None) -> None:
        """Record failed operation."""
        with self._lock:
            # Check if exception should be excluded
            if exception and isinstance(exception, self.config.excluded_exceptions):
                return
            
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
    
    def allow_request(self) -> bool:
        """Check if request should be allowed."""
        with self._lock:
            self._check_state_transition()
            
            if self._state == CircuitState.CLOSED:
                return True
            elif self._state == CircuitState.HALF_OPEN:
                return True  # Allow test request
            else:  # OPEN
                return False
    
    def get_retry_after(self) -> float:
        """Get seconds until retry is allowed."""
        with self._lock:
            if self._state != CircuitState.OPEN:
                return 0.0
            
            if self._last_failure_time:
                elapsed = time.time() - self._last_failure_time
                return max(0.0, self.config.timeout_seconds - elapsed)
            
            return self.config.timeout_seconds
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to wrap function with circuit breaker."""
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if not self.allow_request():
                raise CircuitBreakerError(
                    self.name, self._state, self.get_retry_after()
                )
            
            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure(e)
                raise
        
        return wrapper
    
    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure_time": self._last_failure_time,
                "retry_after": self.get_retry_after(),
            }


class RetryStrategy(str, Enum):
    """Retry backoff strategies."""
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    EXPONENTIAL_JITTER = "exponential_jitter"


@dataclass
class RetryPolicy:
    """
    Retry policy configuration.
    
    Attributes:
        max_attempts: Maximum retry attempts (including initial)
        strategy: Backoff strategy
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap in seconds
        jitter_factor: Jitter factor for randomization (0-1)
        retryable_exceptions: Tuple of exceptions to retry
    """
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_JITTER
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter_factor: float = 0.1
    retryable_exceptions: tuple = (Exception,)
    
    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay for given attempt number.
        
        Args:
            attempt: Attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        if self.strategy == RetryStrategy.FIXED:
            delay = self.base_delay
        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.base_delay * (attempt + 1)
        elif self.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.base_delay * (2 ** attempt)
        else:  # EXPONENTIAL_JITTER
            delay = self.base_delay * (2 ** attempt)
            jitter = delay * self.jitter_factor * random.random()
            delay += jitter
        
        return min(delay, self.max_delay)
    
    def should_retry(self, exception: Exception) -> bool:
        """Check if exception should trigger retry."""
        return isinstance(exception, self.retryable_exceptions)
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to wrap function with retry logic."""
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(self.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if not self.should_retry(e):
                        raise
                    
                    if attempt < self.max_attempts - 1:
                        delay = self.get_delay(attempt)
                        logger.warning(
                            f"Retry {attempt + 1}/{self.max_attempts} after {delay:.2f}s: {e}"
                        )
                        time.sleep(delay)
            
            raise last_exception
        
        return wrapper


@dataclass
class RateLimiterConfig:
    """Rate limiter configuration."""
    requests_per_second: float = 10.0
    burst_size: int = 20
    
    def __post_init__(self):
        if self.requests_per_second <= 0:
            raise ValueError("requests_per_second must be > 0")
        if self.burst_size < 1:
            raise ValueError("burst_size must be >= 1")


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    def __init__(self, retry_after: float):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after:.2f}s")


class RateLimiter:
    """
    Token bucket rate limiter.
    
    Allows bursts up to burst_size, refills at requests_per_second.
    """
    
    def __init__(self, config: Optional[RateLimiterConfig] = None):
        """
        Initialize rate limiter.
        
        Args:
            config: Rate limiter configuration
        """
        self.config = config or RateLimiterConfig()
        self._tokens = float(self.config.burst_size)
        self._last_refill = time.time()
        self._lock = threading.Lock()
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_refill
        refill = elapsed * self.config.requests_per_second
        self._tokens = min(self.config.burst_size, self._tokens + refill)
        self._last_refill = now
    
    def acquire(self, tokens: int = 1, block: bool = True) -> bool:
        """
        Acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire
            block: If True, block until tokens available
            
        Returns:
            True if tokens acquired, False if non-blocking and unavailable
        """
        with self._lock:
            self._refill()
            
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            
            if not block:
                return False
            
            # Calculate wait time
            needed = tokens - self._tokens
            wait_time = needed / self.config.requests_per_second
            
            time.sleep(wait_time)
            self._refill()
            self._tokens -= tokens
            return True
    
    def try_acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens without blocking."""
        return self.acquire(tokens, block=False)
    
    def get_wait_time(self, tokens: int = 1) -> float:
        """Get wait time until tokens available."""
        with self._lock:
            self._refill()
            
            if self._tokens >= tokens:
                return 0.0
            
            needed = tokens - self._tokens
            return needed / self.config.requests_per_second
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to rate limit function calls."""
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            wait_time = self.get_wait_time()
            if wait_time > 0 and not self.try_acquire():
                raise RateLimitExceeded(wait_time)
            
            self.acquire()
            return func(*args, **kwargs)
        
        return wrapper
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        with self._lock:
            self._refill()
            return {
                "available_tokens": self._tokens,
                "burst_size": self.config.burst_size,
                "requests_per_second": self.config.requests_per_second,
            }


@dataclass
class BulkheadConfig:
    """Bulkhead configuration."""
    max_concurrent: int = 10
    max_wait_seconds: float = 5.0


class BulkheadFull(Exception):
    """Raised when bulkhead is at capacity."""
    pass


class Bulkhead:
    """
    Bulkhead pattern for limiting concurrent execution.
    
    Prevents resource exhaustion by limiting concurrent operations.
    """
    
    def __init__(self, name: str, config: Optional[BulkheadConfig] = None):
        """
        Initialize bulkhead.
        
        Args:
            name: Bulkhead name for logging
            config: Bulkhead configuration
        """
        self.name = name
        self.config = config or BulkheadConfig()
        self._semaphore = threading.Semaphore(self.config.max_concurrent)
        self._active = 0
        self._lock = threading.Lock()
    
    @property
    def active_count(self) -> int:
        """Get number of active operations."""
        return self._active
    
    @property
    def available(self) -> int:
        """Get available slots."""
        return self.config.max_concurrent - self._active
    
    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire a slot in the bulkhead.
        
        Args:
            timeout: Maximum wait time in seconds
            
        Returns:
            True if acquired, False if timeout
        """
        timeout = timeout or self.config.max_wait_seconds
        acquired = self._semaphore.acquire(timeout=timeout)
        
        if acquired:
            with self._lock:
                self._active += 1
        
        return acquired
    
    def release(self) -> None:
        """Release a slot in the bulkhead."""
        with self._lock:
            self._active = max(0, self._active - 1)
        self._semaphore.release()
    
    def __enter__(self) -> "Bulkhead":
        """Context manager entry."""
        if not self.acquire():
            raise BulkheadFull(f"Bulkhead '{self.name}' is full")
        return self
    
    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.release()
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to wrap function with bulkhead."""
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            with self:
                return func(*args, **kwargs)
        
        return wrapper
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bulkhead statistics."""
        return {
            "name": self.name,
            "active": self._active,
            "available": self.available,
            "max_concurrent": self.config.max_concurrent,
        }


class TimeoutError(Exception):
    """Raised when operation times out."""
    pass


class Timeout:
    """
    Timeout wrapper for operations.
    
    Note: Thread-based timeout, cannot interrupt blocking I/O.
    """
    
    def __init__(self, seconds: float):
        """
        Initialize timeout.
        
        Args:
            seconds: Timeout in seconds
        """
        if seconds <= 0:
            raise ValueError("Timeout must be > 0")
        self.seconds = seconds
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to add timeout to function."""
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            result: List[Any] = []
            exception: List[Exception] = []
            
            def target():
                try:
                    result.append(func(*args, **kwargs))
                except Exception as e:
                    exception.append(e)
            
            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout=self.seconds)
            
            if thread.is_alive():
                raise TimeoutError(f"Operation timed out after {self.seconds}s")
            
            if exception:
                raise exception[0]
            
            return result[0] if result else None
        
        return wrapper


@dataclass
class ResilienceConfig:
    """Combined resilience configuration."""
    circuit_breaker: Optional[CircuitBreakerConfig] = None
    retry_policy: Optional[RetryPolicy] = None
    rate_limiter: Optional[RateLimiterConfig] = None
    bulkhead: Optional[BulkheadConfig] = None
    timeout_seconds: Optional[float] = None


def resilient(
    name: str,
    config: Optional[ResilienceConfig] = None,
    circuit_breaker: Optional[CircuitBreaker] = None,
    retry_policy: Optional[RetryPolicy] = None,
    rate_limiter: Optional[RateLimiter] = None,
    bulkhead: Optional[Bulkhead] = None,
    timeout: Optional[float] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Combined resilience decorator.
    
    Applies resilience patterns in order:
    1. Rate limiting
    2. Bulkhead (concurrency limit)
    3. Circuit breaker
    4. Timeout
    5. Retry
    
    Args:
        name: Operation name for logging/metrics
        config: Combined resilience configuration
        circuit_breaker: Existing circuit breaker instance
        retry_policy: Existing retry policy instance
        rate_limiter: Existing rate limiter instance
        bulkhead: Existing bulkhead instance
        timeout: Timeout in seconds
        
    Returns:
        Decorated function
    """
    config = config or ResilienceConfig()
    
    # Create instances from config if not provided
    _circuit_breaker = circuit_breaker
    if not _circuit_breaker and config.circuit_breaker:
        _circuit_breaker = CircuitBreaker(f"{name}_circuit", config.circuit_breaker)
    
    _retry_policy = retry_policy or config.retry_policy
    
    _rate_limiter = rate_limiter
    if not _rate_limiter and config.rate_limiter:
        _rate_limiter = RateLimiter(config.rate_limiter)
    
    _bulkhead = bulkhead
    if not _bulkhead and config.bulkhead:
        _bulkhead = Bulkhead(f"{name}_bulkhead", config.bulkhead)
    
    _timeout = timeout or config.timeout_seconds
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Apply patterns in order
            result_func = func
            
            # 5. Retry (innermost, applied last)
            if _retry_policy:
                result_func = _retry_policy(result_func)
            
            # 4. Timeout
            if _timeout:
                result_func = Timeout(_timeout)(result_func)
            
            # 3. Circuit breaker
            if _circuit_breaker:
                result_func = _circuit_breaker(result_func)
            
            # 2. Bulkhead
            if _bulkhead:
                result_func = _bulkhead(result_func)
            
            # 1. Rate limiter (outermost, applied first)
            if _rate_limiter:
                result_func = _rate_limiter(result_func)
            
            return result_func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# Async versions for async/await support

class AsyncCircuitBreaker(CircuitBreaker):
    """Async-compatible circuit breaker."""
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for async functions."""
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                if not self.allow_request():
                    raise CircuitBreakerError(
                        self.name, self._state, self.get_retry_after()
                    )
                
                try:
                    result = await func(*args, **kwargs)
                    self.record_success()
                    return result
                except Exception as e:
                    self.record_failure(e)
                    raise
            
            return async_wrapper
        else:
            return super().__call__(func)


class AsyncRetryPolicy(RetryPolicy):
    """Async-compatible retry policy."""
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for async functions."""
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                last_exception = None
                
                for attempt in range(self.max_attempts):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        
                        if not self.should_retry(e):
                            raise
                        
                        if attempt < self.max_attempts - 1:
                            delay = self.get_delay(attempt)
                            logger.warning(
                                f"Retry {attempt + 1}/{self.max_attempts} after {delay:.2f}s: {e}"
                            )
                            await asyncio.sleep(delay)
                
                raise last_exception
            
            return async_wrapper
        else:
            return super().__call__(func)
