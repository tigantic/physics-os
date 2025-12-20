# Copyright (c) 2025 Tigantic
# Phase 18: Memory Management for Real-Time Inference
"""
Memory management and optimization for real-time tensor computations.

Provides memory pooling, caching, streaming buffers, and memory planning
for efficient GPU/CPU memory utilization in real-time applications.
"""

from __future__ import annotations

import weakref
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import torch
import numpy as np


class AllocationStrategy(Enum):
    """Memory allocation strategies."""
    
    ON_DEMAND = auto()      # Allocate as needed
    PREALLOCATED = auto()   # Pre-allocate pool
    STREAMING = auto()      # Double-buffered streaming
    UNIFIED = auto()        # Unified memory (CPU-GPU shared)
    CACHED = auto()         # Cached allocations


@dataclass
class MemoryConfig:
    """Memory management configuration.
    
    Attributes:
        pool_size_mb: Memory pool size in megabytes
        strategy: Allocation strategy
        enable_caching: Enable tensor caching
        cache_size: Maximum cache entries
        alignment: Memory alignment in bytes
        growth_factor: Pool growth factor when exhausted
        max_pool_size_mb: Maximum pool size
        enable_defragmentation: Enable memory defragmentation
        prefetch_enabled: Enable prefetching
    """
    
    pool_size_mb: float = 512.0
    strategy: AllocationStrategy = AllocationStrategy.PREALLOCATED
    enable_caching: bool = True
    cache_size: int = 100
    alignment: int = 256
    growth_factor: float = 1.5
    max_pool_size_mb: float = 4096.0
    enable_defragmentation: bool = True
    prefetch_enabled: bool = True
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.pool_size_mb <= 0:
            raise ValueError("pool_size_mb must be > 0")
        if self.alignment <= 0 or (self.alignment & (self.alignment - 1)) != 0:
            raise ValueError("alignment must be a positive power of 2")
        if self.growth_factor < 1:
            raise ValueError("growth_factor must be >= 1")


@dataclass
class TensorHandle:
    """Handle to a managed tensor allocation.
    
    Attributes:
        id: Unique handle identifier
        tensor: Underlying tensor
        size_bytes: Size in bytes
        in_use: Whether tensor is currently in use
        last_access: Last access timestamp
        shape: Original tensor shape
        dtype: Tensor data type
        device: Tensor device
    """
    
    id: int
    tensor: torch.Tensor
    size_bytes: int
    in_use: bool = True
    last_access: float = 0.0
    shape: Tuple[int, ...] = field(default_factory=tuple)
    dtype: torch.dtype = torch.float32
    device: str = "cpu"
    
    def release(self) -> None:
        """Mark tensor as no longer in use."""
        self.in_use = False
    
    def acquire(self) -> torch.Tensor:
        """Acquire the tensor for use.
        
        Returns:
            The underlying tensor
        """
        self.in_use = True
        import time
        self.last_access = time.perf_counter()
        return self.tensor


class TensorCache:
    """LRU cache for tensor allocations.
    
    Caches tensor allocations by size and dtype to enable reuse
    and reduce allocation overhead.
    
    Attributes:
        max_size: Maximum number of cached tensors
        max_bytes: Maximum cache size in bytes
    """
    
    def __init__(
        self,
        max_size: int = 100,
        max_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        """Initialize tensor cache.
        
        Args:
            max_size: Maximum cached tensors
            max_bytes: Maximum cache size in bytes
        """
        self.max_size = max_size
        self.max_bytes = max_bytes
        
        # Cache keyed by (shape, dtype, device)
        self._cache: OrderedDict[Tuple, List[torch.Tensor]] = OrderedDict()
        self._current_bytes = 0
        self._lock = threading.Lock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
    
    def _make_key(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: str,
    ) -> Tuple:
        """Create cache key.
        
        Args:
            shape: Tensor shape
            dtype: Data type
            device: Device string
            
        Returns:
            Cache key tuple
        """
        return (shape, dtype, device)
    
    def get(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> Optional[torch.Tensor]:
        """Get a cached tensor if available.
        
        Args:
            shape: Desired shape
            dtype: Data type
            device: Device
            
        Returns:
            Cached tensor or None
        """
        key = self._make_key(shape, dtype, device)
        
        with self._lock:
            if key in self._cache and self._cache[key]:
                self._hits += 1
                tensor = self._cache[key].pop()
                
                # Update byte count
                self._current_bytes -= tensor.numel() * tensor.element_size()
                
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                
                return tensor
            
            self._misses += 1
            return None
    
    def put(self, tensor: torch.Tensor) -> bool:
        """Cache a tensor for reuse.
        
        Args:
            tensor: Tensor to cache
            
        Returns:
            True if cached, False if rejected
        """
        size_bytes = tensor.numel() * tensor.element_size()
        
        with self._lock:
            # Check if we have room
            while (
                self._current_bytes + size_bytes > self.max_bytes
                or len(self._cache) >= self.max_size
            ):
                if not self._evict_one():
                    return False
            
            key = self._make_key(
                tuple(tensor.shape),
                tensor.dtype,
                str(tensor.device),
            )
            
            if key not in self._cache:
                self._cache[key] = []
            
            self._cache[key].append(tensor)
            self._current_bytes += size_bytes
            self._cache.move_to_end(key)
            
            return True
    
    def _evict_one(self) -> bool:
        """Evict one entry from cache.
        
        Returns:
            True if an entry was evicted
        """
        if not self._cache:
            return False
        
        # Evict from least recently used
        for key in self._cache:
            if self._cache[key]:
                tensor = self._cache[key].pop(0)
                self._current_bytes -= tensor.numel() * tensor.element_size()
                
                if not self._cache[key]:
                    del self._cache[key]
                
                return True
        
        return False
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._current_bytes = 0
    
    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total
    
    @property
    def size_bytes(self) -> int:
        """Get current cache size in bytes."""
        return self._current_bytes


class StreamingBuffer:
    """Double-buffered streaming for continuous data.
    
    Enables overlap of computation and data transfer by using
    two buffers that alternate roles.
    
    Attributes:
        buffer_size: Size of each buffer in elements
        dtype: Buffer data type
        device: Target device
    """
    
    def __init__(
        self,
        buffer_size: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        """Initialize streaming buffer.
        
        Args:
            buffer_size: Size of each buffer
            dtype: Data type
            device: Device for buffers
        """
        self.buffer_size = buffer_size
        self.dtype = dtype
        self.device = device
        
        # Create double buffers
        self._buffers = [
            torch.empty(buffer_size, dtype=dtype, device=device),
            torch.empty(buffer_size, dtype=dtype, device=device),
        ]
        self._current_idx = 0
        self._lock = threading.Lock()
        
        # For async transfers
        self._streams: List[Any] = []
        if "cuda" in device:
            self._streams = [
                torch.cuda.Stream(),
                torch.cuda.Stream(),
            ]
    
    @property
    def current_buffer(self) -> torch.Tensor:
        """Get the current (computation) buffer."""
        return self._buffers[self._current_idx]
    
    @property
    def next_buffer(self) -> torch.Tensor:
        """Get the next (transfer) buffer."""
        return self._buffers[1 - self._current_idx]
    
    def swap(self) -> None:
        """Swap the current and next buffers."""
        with self._lock:
            self._current_idx = 1 - self._current_idx
    
    def load_async(
        self,
        data: torch.Tensor,
        target_buffer: int = -1,
    ) -> None:
        """Load data into a buffer asynchronously.
        
        Args:
            data: Data to load
            target_buffer: Buffer index (-1 for next buffer)
        """
        if target_buffer < 0:
            target_buffer = 1 - self._current_idx
        
        buffer = self._buffers[target_buffer]
        size = min(len(data), self.buffer_size)
        
        if self._streams:
            with torch.cuda.stream(self._streams[target_buffer]):
                buffer[:size].copy_(data[:size])
        else:
            buffer[:size].copy_(data[:size])
    
    def synchronize(self, buffer_idx: Optional[int] = None) -> None:
        """Synchronize buffer transfer.
        
        Args:
            buffer_idx: Specific buffer to sync, or None for all
        """
        if self._streams:
            if buffer_idx is not None:
                self._streams[buffer_idx].synchronize()
            else:
                for stream in self._streams:
                    stream.synchronize()


class MemoryPool:
    """Pre-allocated memory pool for efficient allocation.
    
    Manages a pool of pre-allocated memory blocks to reduce
    allocation overhead and fragmentation.
    
    Attributes:
        config: Memory configuration
        total_allocated: Total allocated bytes
        peak_usage: Peak memory usage
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None) -> None:
        """Initialize memory pool.
        
        Args:
            config: Memory configuration
        """
        self.config = config or MemoryConfig()
        
        # Allocation tracking
        self._allocations: Dict[int, TensorHandle] = {}
        self._next_id = 0
        self._lock = threading.Lock()
        
        # Free list by size bucket
        self._free_list: Dict[int, List[TensorHandle]] = {}
        
        # Statistics
        self.total_allocated = 0
        self.peak_usage = 0
        self._allocation_count = 0
        self._reuse_count = 0
    
    def _size_to_bucket(self, size_bytes: int) -> int:
        """Map size to bucket for free list.
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            Bucket size (rounded up to alignment)
        """
        alignment = self.config.alignment
        return ((size_bytes + alignment - 1) // alignment) * alignment
    
    def allocate(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> TensorHandle:
        """Allocate a tensor from the pool.
        
        Args:
            shape: Tensor shape
            dtype: Data type
            device: Device
            
        Returns:
            TensorHandle for the allocation
        """
        # Calculate size
        numel = 1
        for dim in shape:
            numel *= dim
        
        element_size = torch.tensor([], dtype=dtype).element_size()
        size_bytes = numel * element_size
        bucket = self._size_to_bucket(size_bytes)
        
        with self._lock:
            # Try to reuse from free list
            if bucket in self._free_list and self._free_list[bucket]:
                handle = self._free_list[bucket].pop()
                
                # Reshape if necessary
                if handle.tensor.numel() >= numel:
                    handle.tensor = handle.tensor.view(-1)[:numel].view(shape)
                    handle.shape = shape
                    handle.in_use = True
                    self._reuse_count += 1
                    return handle
            
            # Allocate new tensor
            tensor = torch.empty(shape, dtype=dtype, device=device)
            
            handle = TensorHandle(
                id=self._next_id,
                tensor=tensor,
                size_bytes=size_bytes,
                in_use=True,
                shape=shape,
                dtype=dtype,
                device=device,
            )
            
            self._next_id += 1
            self._allocations[handle.id] = handle
            self.total_allocated += size_bytes
            self.peak_usage = max(self.peak_usage, self.total_allocated)
            self._allocation_count += 1
            
            return handle
    
    def free(self, handle: TensorHandle) -> None:
        """Return a tensor to the pool.
        
        Args:
            handle: Handle to free
        """
        with self._lock:
            handle.in_use = False
            
            bucket = self._size_to_bucket(handle.size_bytes)
            
            if bucket not in self._free_list:
                self._free_list[bucket] = []
            
            self._free_list[bucket].append(handle)
    
    def defragment(self) -> int:
        """Defragment the memory pool.
        
        Returns:
            Number of bytes freed
        """
        if not self.config.enable_defragmentation:
            return 0
        
        freed = 0
        
        with self._lock:
            # Clear empty buckets and compact
            empty_buckets = []
            for bucket, handles in self._free_list.items():
                if not handles:
                    empty_buckets.append(bucket)
                else:
                    # Sort by last access, keep only recent
                    handles.sort(key=lambda h: h.last_access, reverse=True)
                    
                    # Free old allocations
                    max_keep = 5
                    while len(handles) > max_keep:
                        old_handle = handles.pop()
                        freed += old_handle.size_bytes
                        if old_handle.id in self._allocations:
                            del self._allocations[old_handle.id]
            
            for bucket in empty_buckets:
                del self._free_list[bucket]
        
        self.total_allocated -= freed
        return freed
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pool statistics.
        
        Returns:
            Dictionary of statistics
        """
        total_free = sum(
            sum(h.size_bytes for h in handles)
            for handles in self._free_list.values()
        )
        
        reuse_rate = (
            self._reuse_count / self._allocation_count
            if self._allocation_count > 0 else 0.0
        )
        
        return {
            "total_allocated_bytes": self.total_allocated,
            "peak_usage_bytes": self.peak_usage,
            "free_pool_bytes": total_free,
            "allocation_count": self._allocation_count,
            "reuse_count": self._reuse_count,
            "reuse_rate": reuse_rate,
            "num_buckets": len(self._free_list),
        }


class MemoryPlanner:
    """Static memory planner for computation graphs.
    
    Analyzes computation graphs to plan memory allocation and
    enable memory reuse between non-overlapping tensors.
    
    Attributes:
        config: Memory configuration
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None) -> None:
        """Initialize memory planner.
        
        Args:
            config: Memory configuration
        """
        self.config = config or MemoryConfig()
        self._plans: Dict[str, Dict[str, Any]] = {}
    
    def plan(
        self,
        tensor_specs: List[Dict[str, Any]],
        lifetimes: List[Tuple[int, int]],
    ) -> Dict[str, Any]:
        """Create a memory plan for tensors with known lifetimes.
        
        Args:
            tensor_specs: List of {shape, dtype, name} dicts
            lifetimes: List of (start, end) step ranges
            
        Returns:
            Memory plan with allocation assignments
        """
        if len(tensor_specs) != len(lifetimes):
            raise ValueError("Specs and lifetimes must have same length")
        
        # Sort by size (descending) for better packing
        indexed = list(enumerate(zip(tensor_specs, lifetimes)))
        indexed.sort(key=lambda x: -np.prod(x[1][0].get("shape", (1,))))
        
        # Allocate memory regions
        regions: List[Dict[str, Any]] = []
        assignments: Dict[int, int] = {}  # tensor idx -> region idx
        
        for orig_idx, (spec, (start, end)) in indexed:
            size = np.prod(spec.get("shape", (1,)))
            
            # Try to reuse an existing region
            reused = False
            for region_idx, region in enumerate(regions):
                # Check if lifetime overlaps
                overlaps = any(
                    not (end < r_start or start > r_end)
                    for r_start, r_end in region["lifetimes"]
                )
                
                if not overlaps and region["size"] >= size:
                    # Can reuse this region
                    region["lifetimes"].append((start, end))
                    region["tensors"].append(orig_idx)
                    assignments[orig_idx] = region_idx
                    reused = True
                    break
            
            if not reused:
                # Create new region
                region = {
                    "size": size,
                    "lifetimes": [(start, end)],
                    "tensors": [orig_idx],
                }
                assignments[orig_idx] = len(regions)
                regions.append(region)
        
        # Calculate total memory
        total_size = sum(r["size"] for r in regions)
        naive_size = sum(np.prod(s.get("shape", (1,))) for s in tensor_specs)
        
        plan = {
            "assignments": assignments,
            "regions": regions,
            "total_size": total_size,
            "naive_size": naive_size,
            "memory_saved": naive_size - total_size,
            "savings_ratio": 1 - (total_size / naive_size) if naive_size > 0 else 0,
        }
        
        return plan
    
    def optimize(
        self,
        plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Optimize a memory plan.
        
        Args:
            plan: Initial memory plan
            
        Returns:
            Optimized plan
        """
        # Further optimizations could include:
        # - Memory defragmentation
        # - Alignment optimization
        # - Cache-friendly ordering
        
        return plan


def optimize_memory(
    tensors: List[torch.Tensor],
    lifetimes: Optional[List[Tuple[int, int]]] = None,
    config: Optional[MemoryConfig] = None,
) -> Dict[str, Any]:
    """Optimize memory usage for a list of tensors.
    
    Args:
        tensors: List of tensors
        lifetimes: Optional lifetime ranges
        config: Memory configuration
        
    Returns:
        Optimization result
    """
    specs = [
        {"shape": tuple(t.shape), "dtype": t.dtype}
        for t in tensors
    ]
    
    if lifetimes is None:
        # Assume all tensors live for the entire computation
        lifetimes = [(0, len(tensors))] * len(tensors)
    
    planner = MemoryPlanner(config)
    plan = planner.plan(specs, lifetimes)
    
    return planner.optimize(plan)
