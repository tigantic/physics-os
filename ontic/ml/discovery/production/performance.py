"""
Performance Optimization for Production Systems

Caching, connection pooling, batch optimization, and memory management.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, Set, TypeVar, Union
from enum import Enum
from functools import wraps
from collections import OrderedDict
from contextlib import contextmanager
import time
import threading
import hashlib
import json
import sys
import gc
import weakref
import logging
import heapq
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


class CachePolicy(str, Enum):
    """Cache eviction policies."""
    LRU = "lru"       # Least Recently Used
    LFU = "lfu"       # Least Frequently Used
    TTL = "ttl"       # Time-To-Live only
    FIFO = "fifo"     # First In First Out


@dataclass
class CacheEntry(Generic[V]):
    """Cache entry with metadata."""
    value: V
    created_at: float
    accessed_at: float
    access_count: int = 1
    expires_at: Optional[float] = None
    size_bytes: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def touch(self) -> None:
        """Update access time and count."""
        self.accessed_at = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    size: int = 0
    max_size: int = 0
    memory_bytes: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "hit_rate": self.hit_rate,
            "size": self.size,
            "max_size": self.max_size,
            "memory_bytes": self.memory_bytes,
        }


class CacheManager(Generic[K, V]):
    """
    High-performance cache with multiple eviction policies.
    
    Features:
    - LRU, LFU, TTL, FIFO eviction
    - Size-based limits
    - Automatic expiration
    - Cache warming
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        policy: CachePolicy = CachePolicy.LRU,
        default_ttl: Optional[float] = None,
        max_memory_mb: Optional[float] = None,
    ):
        """
        Initialize cache manager.
        
        Args:
            max_size: Maximum number of entries
            policy: Eviction policy
            default_ttl: Default TTL in seconds
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.policy = policy
        self.default_ttl = default_ttl
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024) if max_memory_mb else None
        
        self._cache: Dict[K, CacheEntry[V]] = {}
        self._order: List[K] = []  # For FIFO
        self._lock = threading.RLock()
        self._stats = CacheStats(max_size=max_size)
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        try:
            return sys.getsizeof(value)
        except TypeError:
            return 0
    
    def _should_evict(self) -> bool:
        """Check if eviction is needed."""
        if len(self._cache) >= self.max_size:
            return True
        
        if self.max_memory_bytes:
            if self._stats.memory_bytes >= self.max_memory_bytes:
                return True
        
        return False
    
    def _evict_one(self) -> Optional[K]:
        """Evict one entry based on policy."""
        if not self._cache:
            return None
        
        if self.policy == CachePolicy.LRU:
            # Find least recently accessed
            key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].accessed_at
            )
        elif self.policy == CachePolicy.LFU:
            # Find least frequently accessed
            key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].access_count
            )
        elif self.policy == CachePolicy.FIFO:
            # Remove first inserted
            if self._order:
                key = self._order.pop(0)
            else:
                key = next(iter(self._cache))
        else:  # TTL - evict oldest
            key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].created_at
            )
        
        entry = self._cache.pop(key, None)
        if entry:
            self._stats.evictions += 1
            self._stats.size -= 1
            self._stats.memory_bytes -= entry.size_bytes
        
        return key
    
    def _expire_entries(self) -> int:
        """Remove expired entries."""
        expired = []
        current_time = time.time()
        
        for key, entry in self._cache.items():
            if entry.expires_at and current_time > entry.expires_at:
                expired.append(key)
        
        for key in expired:
            entry = self._cache.pop(key, None)
            if entry:
                self._stats.expirations += 1
                self._stats.size -= 1
                self._stats.memory_bytes -= entry.size_bytes
            if key in self._order:
                self._order.remove(key)
        
        return len(expired)
    
    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default if not found
            
        Returns:
            Cached value or default
        """
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._stats.misses += 1
                return default
            
            if entry.is_expired:
                self._cache.pop(key, None)
                self._stats.expirations += 1
                self._stats.misses += 1
                self._stats.size -= 1
                self._stats.memory_bytes -= entry.size_bytes
                return default
            
            entry.touch()
            self._stats.hits += 1
            return entry.value
    
    def set(
        self,
        key: K,
        value: V,
        ttl: Optional[float] = None,
    ) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
        """
        with self._lock:
            # Check for existing entry
            existing = self._cache.get(key)
            if existing:
                self._stats.memory_bytes -= existing.size_bytes
                self._stats.size -= 1
            
            # Evict if needed
            while self._should_evict():
                if not self._evict_one():
                    break
            
            # Calculate TTL
            ttl = ttl or self.default_ttl
            expires_at = time.time() + ttl if ttl else None
            
            # Create entry
            size_bytes = self._estimate_size(value)
            now = time.time()
            
            entry = CacheEntry(
                value=value,
                created_at=now,
                accessed_at=now,
                expires_at=expires_at,
                size_bytes=size_bytes,
            )
            
            self._cache[key] = entry
            self._stats.size += 1
            self._stats.memory_bytes += size_bytes
            
            if self.policy == CachePolicy.FIFO and key not in self._order:
                self._order.append(key)
    
    def delete(self, key: K) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            entry = self._cache.pop(key, None)
            if entry:
                self._stats.size -= 1
                self._stats.memory_bytes -= entry.size_bytes
                if key in self._order:
                    self._order.remove(key)
                return True
            return False
    
    def clear(self) -> int:
        """Clear all entries. Returns count cleared."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._order.clear()
            self._stats.size = 0
            self._stats.memory_bytes = 0
            return count
    
    def contains(self, key: K) -> bool:
        """Check if key exists (and is not expired)."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired:
                self._cache.pop(key, None)
                self._stats.expirations += 1
                self._stats.size -= 1
                return False
            return True
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                expirations=self._stats.expirations,
                size=self._stats.size,
                max_size=self.max_size,
                memory_bytes=self._stats.memory_bytes,
            )
    
    def keys(self) -> List[K]:
        """Get all cache keys."""
        with self._lock:
            return list(self._cache.keys())
    
    def warm(self, items: Dict[K, V], ttl: Optional[float] = None) -> int:
        """
        Warm cache with multiple items.
        
        Args:
            items: Items to cache
            ttl: TTL for all items
            
        Returns:
            Number of items cached
        """
        count = 0
        for key, value in items.items():
            self.set(key, value, ttl)
            count += 1
        return count


def cached(
    cache: Optional[CacheManager] = None,
    ttl: Optional[float] = None,
    key_fn: Optional[Callable[..., str]] = None,
) -> Callable:
    """
    Decorator for caching function results.
    
    Args:
        cache: Cache manager instance
        ttl: Time-to-live in seconds
        key_fn: Custom key generation function
    """
    _cache = cache or CacheManager(max_size=1000)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            if key_fn:
                key = key_fn(*args, **kwargs)
            else:
                key_parts = [func.__name__]
                key_parts.extend(str(a) for a in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                key = ":".join(key_parts)
            
            # Check cache
            result = _cache.get(key)
            if result is not None:
                return result
            
            # Call function
            result = func(*args, **kwargs)
            
            # Cache result
            _cache.set(key, result, ttl)
            
            return result
        
        wrapper.cache = _cache
        wrapper.cache_clear = _cache.clear
        return wrapper
    
    return decorator


@dataclass
class PooledConnection(Generic[T]):
    """Pooled connection wrapper."""
    connection: T
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    use_count: int = 0
    
    def touch(self) -> None:
        """Update usage stats."""
        self.last_used_at = time.time()
        self.use_count += 1


class ConnectionPool(Generic[T]):
    """
    Generic connection pool.
    
    Features:
    - Connection reuse
    - Automatic cleanup
    - Health checking
    - Size limits
    """
    
    def __init__(
        self,
        factory: Callable[[], T],
        max_size: int = 10,
        min_size: int = 1,
        max_idle_seconds: float = 300.0,
        health_check: Optional[Callable[[T], bool]] = None,
        cleanup: Optional[Callable[[T], None]] = None,
    ):
        """
        Initialize connection pool.
        
        Args:
            factory: Connection factory function
            max_size: Maximum pool size
            min_size: Minimum connections to maintain
            max_idle_seconds: Max idle time before cleanup
            health_check: Function to check connection health
            cleanup: Function to cleanup/close connection
        """
        self.factory = factory
        self.max_size = max_size
        self.min_size = min_size
        self.max_idle_seconds = max_idle_seconds
        self.health_check = health_check
        self.cleanup_fn = cleanup
        
        self._pool: List[PooledConnection[T]] = []
        self._in_use: Set[int] = set()  # IDs of connections in use
        self._lock = threading.Lock()
        self._created = 0
        self._closed = False
    
    def _create_connection(self) -> PooledConnection[T]:
        """Create a new pooled connection."""
        conn = self.factory()
        self._created += 1
        return PooledConnection(connection=conn)
    
    def _is_healthy(self, pooled: PooledConnection[T]) -> bool:
        """Check if connection is healthy."""
        # Check idle time
        idle_time = time.time() - pooled.last_used_at
        if idle_time > self.max_idle_seconds:
            return False
        
        # Custom health check
        if self.health_check:
            try:
                return self.health_check(pooled.connection)
            except Exception:
                return False
        
        return True
    
    def _cleanup_connection(self, pooled: PooledConnection[T]) -> None:
        """Cleanup a connection."""
        if self.cleanup_fn:
            try:
                self.cleanup_fn(pooled.connection)
            except Exception as e:
                logger.warning(f"Connection cleanup error: {e}")
    
    def acquire(self, timeout: Optional[float] = None) -> T:
        """
        Acquire a connection from the pool.
        
        Args:
            timeout: Maximum wait time
            
        Returns:
            Connection instance
            
        Raises:
            RuntimeError: If pool is closed or timeout
        """
        if self._closed:
            raise RuntimeError("Connection pool is closed")
        
        deadline = time.time() + timeout if timeout else None
        
        while True:
            with self._lock:
                # Try to get existing connection
                for i, pooled in enumerate(self._pool):
                    if id(pooled) not in self._in_use:
                        if self._is_healthy(pooled):
                            pooled.touch()
                            self._in_use.add(id(pooled))
                            return pooled.connection
                        else:
                            # Remove unhealthy connection
                            self._cleanup_connection(pooled)
                            self._pool.pop(i)
                            break
                
                # Create new if under limit
                if len(self._pool) < self.max_size:
                    pooled = self._create_connection()
                    self._pool.append(pooled)
                    self._in_use.add(id(pooled))
                    return pooled.connection
            
            # Check timeout
            if deadline and time.time() >= deadline:
                raise RuntimeError("Connection pool timeout")
            
            # Wait briefly
            time.sleep(0.01)
    
    def release(self, connection: T) -> None:
        """
        Release a connection back to the pool.
        
        Args:
            connection: Connection to release
        """
        with self._lock:
            for pooled in self._pool:
                if pooled.connection is connection:
                    self._in_use.discard(id(pooled))
                    pooled.touch()
                    return
    
    @contextmanager
    def connection(self, timeout: Optional[float] = None):
        """
        Context manager for acquiring/releasing connections.
        
        Args:
            timeout: Acquire timeout
        """
        conn = self.acquire(timeout)
        try:
            yield conn
        finally:
            self.release(conn)
    
    def close(self) -> None:
        """Close all connections and the pool."""
        with self._lock:
            self._closed = True
            
            for pooled in self._pool:
                self._cleanup_connection(pooled)
            
            self._pool.clear()
            self._in_use.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                "size": len(self._pool),
                "in_use": len(self._in_use),
                "available": len(self._pool) - len(self._in_use),
                "max_size": self.max_size,
                "min_size": self.min_size,
                "total_created": self._created,
                "closed": self._closed,
            }


@dataclass
class BatchConfig:
    """Batch optimization configuration."""
    max_batch_size: int = 100
    max_wait_ms: float = 10.0
    adaptive: bool = True


class BatchOptimizer(Generic[T, V]):
    """
    Batch request optimizer.
    
    Collects individual requests and processes them in batches.
    """
    
    def __init__(
        self,
        batch_fn: Callable[[List[T]], List[V]],
        config: Optional[BatchConfig] = None,
    ):
        """
        Initialize batch optimizer.
        
        Args:
            batch_fn: Function to process batch
            config: Batch configuration
        """
        self.batch_fn = batch_fn
        self.config = config or BatchConfig()
        
        self._pending: List[tuple[T, threading.Event, List[V]]] = []
        self._lock = threading.Lock()
        self._batch_thread: Optional[threading.Thread] = None
        self._running = False
        self._stats = {
            "batches": 0,
            "items": 0,
            "avg_batch_size": 0.0,
        }
    
    def _process_batch(self) -> None:
        """Background thread for batch processing."""
        while self._running:
            batch_items = []
            events_results = []
            
            with self._lock:
                if len(self._pending) >= self.config.max_batch_size:
                    # Process immediately if batch is full
                    batch_items = [(item, event, result) for item, event, result in self._pending[:self.config.max_batch_size]]
                    self._pending = self._pending[self.config.max_batch_size:]
            
            if not batch_items:
                # Wait for items
                time.sleep(self.config.max_wait_ms / 1000.0)
                
                with self._lock:
                    if self._pending:
                        batch_items = list(self._pending)
                        self._pending.clear()
            
            if batch_items:
                # Process batch
                items = [item for item, _, _ in batch_items]
                try:
                    results = self.batch_fn(items)
                    
                    # Distribute results
                    for (_, event, result_list), result in zip(batch_items, results):
                        result_list.append(result)
                        event.set()
                    
                    # Update stats
                    self._stats["batches"] += 1
                    self._stats["items"] += len(items)
                    self._stats["avg_batch_size"] = (
                        self._stats["items"] / self._stats["batches"]
                    )
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
                    for _, event, _ in batch_items:
                        event.set()  # Unblock waiters
    
    def start(self) -> None:
        """Start batch processing."""
        if not self._running:
            self._running = True
            self._batch_thread = threading.Thread(
                target=self._process_batch,
                daemon=True,
            )
            self._batch_thread.start()
    
    def stop(self) -> None:
        """Stop batch processing."""
        self._running = False
        if self._batch_thread:
            self._batch_thread.join(timeout=1.0)
            self._batch_thread = None
    
    def submit(self, item: T, timeout: Optional[float] = None) -> V:
        """
        Submit item for batch processing.
        
        Args:
            item: Item to process
            timeout: Maximum wait time
            
        Returns:
            Processing result
        """
        if not self._running:
            self.start()
        
        event = threading.Event()
        result: List[V] = []
        
        with self._lock:
            self._pending.append((item, event, result))
        
        # Wait for result
        if not event.wait(timeout):
            raise RuntimeError("Batch processing timeout")
        
        return result[0] if result else None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch optimizer statistics."""
        with self._lock:
            return {
                **self._stats,
                "pending": len(self._pending),
                "running": self._running,
            }


@dataclass
class MemoryConfig:
    """Memory manager configuration."""
    max_memory_mb: float = 1024.0
    gc_threshold_pct: float = 80.0
    warning_threshold_pct: float = 70.0


class MemoryManager:
    """
    Memory usage monitoring and management.
    
    Features:
    - Memory usage tracking
    - Automatic GC triggering
    - Memory pressure callbacks
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize memory manager.
        
        Args:
            config: Memory configuration
        """
        self.config = config or MemoryConfig()
        self._callbacks: List[Callable[[float, float], None]] = []
        self._lock = threading.Lock()
        self._last_check = 0.0
        self._gc_count = 0
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        import os
        
        try:
            # Try to read from /proc on Linux
            with open('/proc/self/status', 'r') as f:
                status = f.read()
            
            vm_rss = 0
            vm_size = 0
            
            for line in status.split('\n'):
                if line.startswith('VmRSS:'):
                    vm_rss = int(line.split()[1]) * 1024  # Convert KB to bytes
                elif line.startswith('VmSize:'):
                    vm_size = int(line.split()[1]) * 1024
            
            return {
                "rss_bytes": vm_rss,
                "rss_mb": vm_rss / (1024 * 1024),
                "virtual_bytes": vm_size,
                "virtual_mb": vm_size / (1024 * 1024),
            }
        except Exception:
            # Fallback to basic tracking
            import tracemalloc
            
            current, peak = 0, 0
            if tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
            
            return {
                "rss_bytes": current,
                "rss_mb": current / (1024 * 1024),
                "peak_bytes": peak,
                "peak_mb": peak / (1024 * 1024),
            }
    
    def check_pressure(self) -> str:
        """
        Check memory pressure level.
        
        Returns:
            "ok", "warning", or "critical"
        """
        usage = self.get_memory_usage()
        usage_mb = usage.get("rss_mb", 0)
        usage_pct = (usage_mb / self.config.max_memory_mb) * 100
        
        if usage_pct >= self.config.gc_threshold_pct:
            return "critical"
        elif usage_pct >= self.config.warning_threshold_pct:
            return "warning"
        return "ok"
    
    def add_pressure_callback(
        self,
        callback: Callable[[float, float], None],
    ) -> None:
        """
        Add memory pressure callback.
        
        Args:
            callback: Function(usage_mb, max_mb)
        """
        with self._lock:
            self._callbacks.append(callback)
    
    def collect_garbage(self, generation: int = 2) -> Dict[str, Any]:
        """
        Run garbage collection.
        
        Args:
            generation: GC generation (0, 1, or 2)
            
        Returns:
            Collection statistics
        """
        before = self.get_memory_usage()
        
        collected = gc.collect(generation)
        self._gc_count += 1
        
        after = self.get_memory_usage()
        
        freed = before.get("rss_bytes", 0) - after.get("rss_bytes", 0)
        
        return {
            "collected_objects": collected,
            "freed_bytes": max(0, freed),
            "freed_mb": max(0, freed) / (1024 * 1024),
            "before_mb": before.get("rss_mb", 0),
            "after_mb": after.get("rss_mb", 0),
            "generation": generation,
        }
    
    def monitor(self) -> Dict[str, Any]:
        """
        Monitor memory and trigger GC if needed.
        
        Returns:
            Monitoring results
        """
        usage = self.get_memory_usage()
        usage_mb = usage.get("rss_mb", 0)
        pressure = self.check_pressure()
        
        result = {
            "usage_mb": usage_mb,
            "max_mb": self.config.max_memory_mb,
            "usage_pct": (usage_mb / self.config.max_memory_mb) * 100,
            "pressure": pressure,
            "gc_triggered": False,
        }
        
        # Notify callbacks on warning/critical
        if pressure in ("warning", "critical"):
            for callback in self._callbacks:
                try:
                    callback(usage_mb, self.config.max_memory_mb)
                except Exception as e:
                    logger.error(f"Memory callback error: {e}")
        
        # Trigger GC on critical
        if pressure == "critical":
            gc_result = self.collect_garbage()
            result["gc_triggered"] = True
            result["gc_result"] = gc_result
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory manager statistics."""
        usage = self.get_memory_usage()
        
        return {
            **usage,
            "max_mb": self.config.max_memory_mb,
            "gc_count": self._gc_count,
            "gc_stats": gc.get_stats(),
            "pressure": self.check_pressure(),
        }


@dataclass
class ProfileSample:
    """Performance profile sample."""
    operation: str
    start_time: float
    end_time: float
    duration_ms: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceProfiler:
    """
    Performance profiling and analysis.
    
    Collects timing data for operations.
    """
    
    def __init__(self, max_samples: int = 10000):
        """
        Initialize profiler.
        
        Args:
            max_samples: Maximum samples to retain
        """
        self.max_samples = max_samples
        self._samples: List[ProfileSample] = []
        self._lock = threading.Lock()
    
    @contextmanager
    def profile(self, operation: str, **metadata):
        """
        Context manager for profiling operations.
        
        Args:
            operation: Operation name
            **metadata: Additional metadata
        """
        start = time.time()
        success = True
        
        try:
            yield
        except Exception:
            success = False
            raise
        finally:
            end = time.time()
            duration_ms = (end - start) * 1000
            
            sample = ProfileSample(
                operation=operation,
                start_time=start,
                end_time=end,
                duration_ms=duration_ms,
                success=success,
                metadata=metadata,
            )
            
            with self._lock:
                self._samples.append(sample)
                if len(self._samples) > self.max_samples:
                    self._samples = self._samples[-self.max_samples:]
    
    def profile_fn(self, operation: Optional[str] = None):
        """
        Decorator for profiling functions.
        
        Args:
            operation: Operation name (defaults to function name)
        """
        def decorator(func: Callable) -> Callable:
            op_name = operation or func.__name__
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.profile(op_name):
                    return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def get_samples(
        self,
        operation: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[ProfileSample]:
        """
        Get profile samples.
        
        Args:
            operation: Filter by operation
            since: Filter by start time
            limit: Maximum samples
            
        Returns:
            List of samples
        """
        with self._lock:
            samples = list(self._samples)
        
        if operation:
            samples = [s for s in samples if s.operation == operation]
        
        if since:
            samples = [s for s in samples if s.start_time >= since]
        
        return samples[-limit:]
    
    def get_statistics(
        self,
        operation: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get operation statistics.
        
        Args:
            operation: Specific operation or all
            
        Returns:
            Statistics dictionary
        """
        samples = self.get_samples(operation, limit=self.max_samples)
        
        if not samples:
            return {"count": 0}
        
        durations = [s.duration_ms for s in samples]
        success_count = sum(1 for s in samples if s.success)
        
        return {
            "count": len(samples),
            "success_count": success_count,
            "failure_count": len(samples) - success_count,
            "success_rate": success_count / len(samples),
            "min_ms": min(durations),
            "max_ms": max(durations),
            "avg_ms": sum(durations) / len(durations),
            "p50_ms": sorted(durations)[len(durations) // 2],
            "p95_ms": sorted(durations)[int(len(durations) * 0.95)] if len(durations) > 20 else max(durations),
            "p99_ms": sorted(durations)[int(len(durations) * 0.99)] if len(durations) > 100 else max(durations),
        }
    
    def clear(self) -> int:
        """Clear samples. Returns count cleared."""
        with self._lock:
            count = len(self._samples)
            self._samples.clear()
            return count
