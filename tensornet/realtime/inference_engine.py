# Copyright (c) 2025 Tigantic
# Phase 18: Real-Time Inference Engine
"""
High-performance inference engine for tensor network models.

Provides optimized inference with dynamic batching, request queuing,
and priority scheduling for real-time applications.
"""

from __future__ import annotations

import time
import threading
import queue
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)
from concurrent.futures import ThreadPoolExecutor, Future

import torch
import numpy as np


class InferencePriority(Enum):
    """Priority levels for inference requests."""
    
    CRITICAL = 0   # Highest priority (safety-critical)
    HIGH = 1       # High priority (real-time guidance)
    NORMAL = 2     # Normal priority (standard inference)
    LOW = 3        # Low priority (background tasks)
    BATCH = 4      # Lowest priority (batch processing)


@dataclass
class InferenceConfig:
    """Configuration for the inference engine.
    
    Attributes:
        max_batch_size: Maximum batch size for batching
        batch_timeout_ms: Timeout before processing incomplete batch
        num_workers: Number of worker threads
        enable_batching: Whether to enable dynamic batching
        enable_caching: Whether to cache results
        cache_size: Maximum cache entries
        warmup_iterations: Warmup iterations for stable timing
        use_mixed_precision: Enable FP16/BF16 inference
        device: Target device (cpu, cuda, mps)
        enable_profiling: Enable detailed profiling
    """
    
    max_batch_size: int = 32
    batch_timeout_ms: float = 5.0
    num_workers: int = 4
    enable_batching: bool = True
    enable_caching: bool = True
    cache_size: int = 1000
    warmup_iterations: int = 3
    use_mixed_precision: bool = False
    device: str = "cpu"
    enable_profiling: bool = False
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if self.batch_timeout_ms <= 0:
            raise ValueError("batch_timeout_ms must be > 0")
        if self.num_workers < 1:
            raise ValueError("num_workers must be >= 1")


@dataclass
class InferenceResult:
    """Result of an inference request.
    
    Attributes:
        output: Model output tensor(s)
        latency_ms: Inference latency in milliseconds
        batch_size: Actual batch size used
        from_cache: Whether result was from cache
        request_id: Unique request identifier
        priority: Request priority
        metadata: Additional metadata
    """
    
    output: Union[torch.Tensor, Dict[str, torch.Tensor]]
    latency_ms: float
    batch_size: int = 1
    from_cache: bool = False
    request_id: str = ""
    priority: InferencePriority = InferencePriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_throughput(self) -> float:
        """Get throughput in samples/second."""
        if self.latency_ms <= 0:
            return float('inf')
        return 1000.0 * self.batch_size / self.latency_ms


@dataclass
class BatchRequest:
    """A batched inference request.
    
    Attributes:
        inputs: List of input tensors
        priorities: Priority for each input
        request_ids: Unique ID for each request
        created_at: Timestamp when batch was created
    """
    
    inputs: List[torch.Tensor]
    priorities: List[InferencePriority]
    request_ids: List[str]
    created_at: float = field(default_factory=time.perf_counter)
    
    def __len__(self) -> int:
        return len(self.inputs)
    
    @property
    def max_priority(self) -> InferencePriority:
        """Get highest priority in the batch."""
        return min(self.priorities, key=lambda p: p.value)


class RequestQueue:
    """Priority queue for inference requests."""
    
    def __init__(self, maxsize: int = 0) -> None:
        """Initialize request queue.
        
        Args:
            maxsize: Maximum queue size (0 = unlimited)
        """
        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize)
        self._counter = 0
        self._lock = threading.Lock()
    
    def put(
        self,
        input_tensor: torch.Tensor,
        priority: InferencePriority = InferencePriority.NORMAL,
        request_id: Optional[str] = None,
    ) -> str:
        """Add a request to the queue.
        
        Args:
            input_tensor: Input tensor
            priority: Request priority
            request_id: Optional request ID
            
        Returns:
            Request ID
        """
        with self._lock:
            self._counter += 1
            if request_id is None:
                request_id = f"req_{self._counter}"
        
        # Priority queue uses (priority, counter, data) for stable sorting
        self._queue.put((
            priority.value,
            self._counter,
            (input_tensor, priority, request_id),
        ))
        
        return request_id
    
    def get(self, timeout: Optional[float] = None) -> Tuple[torch.Tensor, InferencePriority, str]:
        """Get next request from queue.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (input_tensor, priority, request_id)
        """
        try:
            _, _, data = self._queue.get(timeout=timeout)
            return data
        except queue.Empty:
            raise TimeoutError("Queue get timed out")
    
    def get_batch(
        self,
        max_size: int,
        timeout: float = 0.005,
    ) -> Optional[BatchRequest]:
        """Get a batch of requests.
        
        Args:
            max_size: Maximum batch size
            timeout: Timeout in seconds
            
        Returns:
            BatchRequest or None if queue is empty
        """
        inputs = []
        priorities = []
        request_ids = []
        
        # Get first item with timeout
        try:
            tensor, priority, req_id = self.get(timeout=timeout)
            inputs.append(tensor)
            priorities.append(priority)
            request_ids.append(req_id)
        except TimeoutError:
            return None
        
        # Try to get more items without blocking
        while len(inputs) < max_size:
            try:
                tensor, priority, req_id = self.get(timeout=0.001)
                inputs.append(tensor)
                priorities.append(priority)
                request_ids.append(req_id)
            except TimeoutError:
                break
        
        return BatchRequest(
            inputs=inputs,
            priorities=priorities,
            request_ids=request_ids,
        )
    
    def qsize(self) -> int:
        """Get approximate queue size."""
        return self._queue.qsize()
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()


class ResultCache:
    """LRU cache for inference results."""
    
    def __init__(self, max_size: int = 1000) -> None:
        """Initialize result cache.
        
        Args:
            max_size: Maximum number of cached entries
        """
        self.max_size = max_size
        self._cache: Dict[str, InferenceResult] = {}
        self._access_order: List[str] = []
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
    
    def _compute_key(self, tensor: torch.Tensor) -> str:
        """Compute cache key for tensor.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Cache key string
        """
        # Use shape and data hash
        shape_str = str(tuple(tensor.shape))
        data_hash = hash(tensor.cpu().numpy().tobytes())
        return f"{shape_str}_{data_hash}"
    
    def get(self, tensor: torch.Tensor) -> Optional[InferenceResult]:
        """Get cached result for tensor.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Cached result or None
        """
        key = self._compute_key(tensor)
        
        with self._lock:
            if key in self._cache:
                self._hits += 1
                # Move to end (most recently used)
                self._access_order.remove(key)
                self._access_order.append(key)
                result = self._cache[key]
                # Mark as from cache
                return InferenceResult(
                    output=result.output,
                    latency_ms=0.0,
                    batch_size=result.batch_size,
                    from_cache=True,
                    request_id=result.request_id,
                    priority=result.priority,
                    metadata=result.metadata,
                )
            else:
                self._misses += 1
                return None
    
    def put(self, tensor: torch.Tensor, result: InferenceResult) -> None:
        """Cache a result.
        
        Args:
            tensor: Input tensor
            result: Inference result
        """
        key = self._compute_key(tensor)
        
        with self._lock:
            if key in self._cache:
                # Update existing
                self._access_order.remove(key)
            elif len(self._cache) >= self.max_size:
                # Evict oldest
                oldest_key = self._access_order.pop(0)
                del self._cache[oldest_key]
            
            self._cache[key] = result
            self._access_order.append(key)
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
    
    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total


class InferenceEngine:
    """High-performance inference engine with dynamic batching.
    
    Provides optimized inference for tensor network models with
    features including dynamic batching, priority scheduling,
    result caching, and mixed precision support.
    
    Attributes:
        config: Engine configuration
        model: Model for inference (callable)
        is_running: Whether engine is actively processing
    """
    
    def __init__(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        config: Optional[InferenceConfig] = None,
    ) -> None:
        """Initialize inference engine.
        
        Args:
            model: Model callable (input -> output)
            config: Engine configuration
        """
        self.model = model
        self.config = config or InferenceConfig()
        
        # Components
        self._request_queue = RequestQueue()
        self._result_cache = ResultCache(self.config.cache_size) if self.config.enable_caching else None
        self._results: Dict[str, Future] = {}
        
        # Threading
        self._executor = ThreadPoolExecutor(max_workers=self.config.num_workers)
        self._is_running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Statistics
        self._total_requests = 0
        self._total_batches = 0
        self._total_latency_ms = 0.0
        
        # Warmup
        self._warmed_up = False
    
    def start(self) -> None:
        """Start the inference engine."""
        if self._is_running:
            return
        
        self._is_running = True
        
        if self.config.enable_batching:
            self._worker_thread = threading.Thread(target=self._batch_worker, daemon=True)
            self._worker_thread.start()
    
    def stop(self) -> None:
        """Stop the inference engine."""
        self._is_running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=1.0)
        self._executor.shutdown(wait=False)
    
    def warmup(self, sample_input: torch.Tensor) -> None:
        """Warm up the model with sample input.
        
        Args:
            sample_input: Sample input tensor
        """
        for _ in range(self.config.warmup_iterations):
            with torch.no_grad():
                _ = self.model(sample_input)
        
        self._warmed_up = True
    
    def infer(
        self,
        input_tensor: torch.Tensor,
        priority: InferencePriority = InferencePriority.NORMAL,
        timeout: Optional[float] = None,
    ) -> InferenceResult:
        """Run synchronous inference.
        
        Args:
            input_tensor: Input tensor
            priority: Request priority
            timeout: Timeout in seconds
            
        Returns:
            InferenceResult
        """
        # Check cache first
        if self._result_cache:
            cached = self._result_cache.get(input_tensor)
            if cached:
                return cached
        
        # Run inference
        start_time = time.perf_counter()
        
        with torch.no_grad():
            if self.config.use_mixed_precision and input_tensor.dtype == torch.float32:
                with torch.autocast(device_type=self.config.device):
                    output = self.model(input_tensor)
            else:
                output = self.model(input_tensor)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        result = InferenceResult(
            output=output,
            latency_ms=latency_ms,
            batch_size=1,
            from_cache=False,
            request_id=f"sync_{self._total_requests}",
            priority=priority,
        )
        
        # Update statistics
        self._total_requests += 1
        self._total_latency_ms += latency_ms
        
        # Cache result
        if self._result_cache:
            self._result_cache.put(input_tensor, result)
        
        return result
    
    def infer_async(
        self,
        input_tensor: torch.Tensor,
        priority: InferencePriority = InferencePriority.NORMAL,
    ) -> Future:
        """Submit asynchronous inference request.
        
        Args:
            input_tensor: Input tensor
            priority: Request priority
            
        Returns:
            Future that will contain InferenceResult
        """
        future: Future = Future()
        
        if not self._is_running:
            # Run synchronously
            result = self.infer(input_tensor, priority)
            future.set_result(result)
            return future
        
        request_id = self._request_queue.put(input_tensor, priority)
        self._results[request_id] = future
        
        return future
    
    def infer_batch(
        self,
        inputs: List[torch.Tensor],
        priority: InferencePriority = InferencePriority.NORMAL,
    ) -> List[InferenceResult]:
        """Run batch inference.
        
        Args:
            inputs: List of input tensors
            priority: Priority for all inputs
            
        Returns:
            List of InferenceResults
        """
        if not inputs:
            return []
        
        # Stack inputs
        batched = torch.stack(inputs)
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            if self.config.use_mixed_precision and batched.dtype == torch.float32:
                with torch.autocast(device_type=self.config.device):
                    outputs = self.model(batched)
            else:
                outputs = self.model(batched)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        per_sample_latency = latency_ms / len(inputs)
        
        results = []
        for i, input_tensor in enumerate(inputs):
            output = outputs[i] if outputs.dim() > 0 else outputs
            result = InferenceResult(
                output=output,
                latency_ms=per_sample_latency,
                batch_size=len(inputs),
                from_cache=False,
                request_id=f"batch_{self._total_requests}_{i}",
                priority=priority,
            )
            results.append(result)
            
            # Cache result
            if self._result_cache:
                self._result_cache.put(input_tensor, result)
        
        self._total_requests += len(inputs)
        self._total_batches += 1
        self._total_latency_ms += latency_ms
        
        return results
    
    def _batch_worker(self) -> None:
        """Worker thread for dynamic batching."""
        while self._is_running:
            batch = self._request_queue.get_batch(
                max_size=self.config.max_batch_size,
                timeout=self.config.batch_timeout_ms / 1000.0,
            )
            
            if batch is None:
                continue
            
            try:
                # Run batch inference
                results = self.infer_batch(batch.inputs, batch.max_priority)
                
                # Fulfill futures
                for i, request_id in enumerate(batch.request_ids):
                    if request_id in self._results:
                        future = self._results.pop(request_id)
                        future.set_result(results[i])
                        
            except Exception as e:
                # Report errors to all futures
                for request_id in batch.request_ids:
                    if request_id in self._results:
                        future = self._results.pop(request_id)
                        future.set_exception(e)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            "total_requests": self._total_requests,
            "total_batches": self._total_batches,
            "average_latency_ms": (
                self._total_latency_ms / self._total_requests
                if self._total_requests > 0 else 0.0
            ),
            "queue_size": self._request_queue.qsize(),
            "is_running": self._is_running,
            "warmed_up": self._warmed_up,
        }
        
        if self._result_cache:
            stats["cache_hit_rate"] = self._result_cache.hit_rate
            stats["cache_size"] = len(self._result_cache._cache)
        
        return stats


def run_inference(
    model: Callable[[torch.Tensor], torch.Tensor],
    input_tensor: torch.Tensor,
    config: Optional[InferenceConfig] = None,
) -> InferenceResult:
    """Run single inference with optional configuration.
    
    Args:
        model: Model callable
        input_tensor: Input tensor
        config: Inference configuration
        
    Returns:
        InferenceResult
    """
    engine = InferenceEngine(model, config)
    return engine.infer(input_tensor)


def run_batched_inference(
    model: Callable[[torch.Tensor], torch.Tensor],
    inputs: List[torch.Tensor],
    config: Optional[InferenceConfig] = None,
) -> List[InferenceResult]:
    """Run batched inference.
    
    Args:
        model: Model callable
        inputs: List of input tensors
        config: Inference configuration
        
    Returns:
        List of InferenceResults
    """
    engine = InferenceEngine(model, config)
    return engine.infer_batch(inputs)
