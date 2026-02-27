"""
Memory Management Utilities for FluidElite
===========================================

Provides memory tracking, leak detection, and safe resource management
for CUDA and CPU memory.

Constitutional Compliance:
    - Article III.2: All failures are graceful
    - Article V.4: Error messages include actionable guidance
    - Phase 4: Memory leak detection and prevention

Example:
    >>> from fluidelite.utils.memory import MemoryTracker, memory_scope
    >>> with MemoryTracker() as tracker:
    ...     # Do operations
    ...     model = FluidElite()
    ...     result = model(tokens)
    >>> print(tracker.summary())
"""

from __future__ import annotations

import gc
import weakref
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import time

import torch


@dataclass
class MemorySnapshot:
    """Snapshot of memory state at a point in time."""
    
    timestamp: float
    label: str
    
    # CPU memory (if psutil available)
    cpu_allocated: int = 0
    cpu_percent: float = 0.0
    
    # CUDA memory
    cuda_allocated: int = 0
    cuda_reserved: int = 0
    cuda_max_allocated: int = 0
    cuda_active_tensors: int = 0
    
    def __str__(self) -> str:
        lines = [f"MemorySnapshot({self.label}):"]
        if torch.cuda.is_available():
            lines.append(f"  CUDA allocated: {self.cuda_allocated / 1e6:.2f} MB")
            lines.append(f"  CUDA reserved: {self.cuda_reserved / 1e6:.2f} MB")
            lines.append(f"  CUDA max allocated: {self.cuda_max_allocated / 1e6:.2f} MB")
        if self.cpu_allocated > 0:
            lines.append(f"  CPU allocated: {self.cpu_allocated / 1e6:.2f} MB")
        return "\n".join(lines)


@dataclass
class MemoryDelta:
    """Difference between two memory snapshots."""
    
    label_start: str
    label_end: str
    elapsed_seconds: float
    
    cuda_allocated_delta: int = 0
    cuda_reserved_delta: int = 0
    cpu_allocated_delta: int = 0
    
    @property
    def has_leak(self) -> bool:
        """Check if memory increased (potential leak)."""
        # Allow small allocations (1MB threshold)
        threshold = 1 * 1024 * 1024
        return (self.cuda_allocated_delta > threshold or 
                self.cpu_allocated_delta > threshold)
    
    def __str__(self) -> str:
        lines = [f"MemoryDelta({self.label_start} → {self.label_end}):"]
        lines.append(f"  Elapsed: {self.elapsed_seconds:.3f}s")
        if torch.cuda.is_available():
            sign = "+" if self.cuda_allocated_delta >= 0 else ""
            lines.append(f"  CUDA allocated: {sign}{self.cuda_allocated_delta / 1e6:.2f} MB")
        if self.cpu_allocated_delta != 0:
            sign = "+" if self.cpu_allocated_delta >= 0 else ""
            lines.append(f"  CPU allocated: {sign}{self.cpu_allocated_delta / 1e6:.2f} MB")
        if self.has_leak:
            lines.append("  ⚠️ POTENTIAL MEMORY LEAK DETECTED")
        return "\n".join(lines)


def take_snapshot(label: str = "") -> MemorySnapshot:
    """
    Take a snapshot of current memory state.
    
    Args:
        label: Label for the snapshot
        
    Returns:
        MemorySnapshot with current memory stats
    """
    snapshot = MemorySnapshot(
        timestamp=time.time(),
        label=label
    )
    
    if torch.cuda.is_available():
        snapshot.cuda_allocated = torch.cuda.memory_allocated()
        snapshot.cuda_reserved = torch.cuda.memory_reserved()
        snapshot.cuda_max_allocated = torch.cuda.max_memory_allocated()
        
        # Count active tensors (approximate)
        gc.collect()
        count = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    count += 1
            except Exception:
                pass
        snapshot.cuda_active_tensors = count
    
    # CPU memory (optional dependency)
    try:
        import psutil
        process = psutil.Process()
        snapshot.cpu_allocated = process.memory_info().rss
        snapshot.cpu_percent = process.memory_percent()
    except ImportError:
        pass
    
    return snapshot


def compute_delta(start: MemorySnapshot, end: MemorySnapshot) -> MemoryDelta:
    """
    Compute memory difference between two snapshots.
    
    Args:
        start: Starting snapshot
        end: Ending snapshot
        
    Returns:
        MemoryDelta with differences
    """
    return MemoryDelta(
        label_start=start.label,
        label_end=end.label,
        elapsed_seconds=end.timestamp - start.timestamp,
        cuda_allocated_delta=end.cuda_allocated - start.cuda_allocated,
        cuda_reserved_delta=end.cuda_reserved - start.cuda_reserved,
        cpu_allocated_delta=end.cpu_allocated - start.cpu_allocated
    )


class MemoryTracker:
    """
    Track memory usage and detect leaks.
    
    Use as a context manager to automatically track memory before/after
    and check for potential leaks.
    
    Example:
        >>> with MemoryTracker("training loop") as tracker:
        ...     for epoch in range(100):
        ...         train_one_epoch()
        >>> if tracker.delta.has_leak:
        ...     print("Warning: potential memory leak!")
    """
    
    def __init__(self, label: str = "operation", warn_on_leak: bool = True):
        self.label = label
        self.warn_on_leak = warn_on_leak
        
        self.start_snapshot: Optional[MemorySnapshot] = None
        self.end_snapshot: Optional[MemorySnapshot] = None
        self.intermediate_snapshots: List[MemorySnapshot] = []
        
    def __enter__(self) -> MemoryTracker:
        # Force garbage collection before starting
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        self.start_snapshot = take_snapshot(f"{self.label}_start")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Force garbage collection before measuring
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.end_snapshot = take_snapshot(f"{self.label}_end")
        
        if self.warn_on_leak and self.delta.has_leak:
            warnings.warn(
                f"Potential memory leak detected during '{self.label}': "
                f"CUDA delta = {self.delta.cuda_allocated_delta / 1e6:.2f} MB",
                UserWarning
            )
        
        return False  # Don't suppress exceptions
    
    def checkpoint(self, label: str = ""):
        """Take an intermediate snapshot."""
        if not label:
            label = f"{self.label}_checkpoint_{len(self.intermediate_snapshots)}"
        self.intermediate_snapshots.append(take_snapshot(label))
    
    @property
    def delta(self) -> MemoryDelta:
        """Get memory delta between start and end."""
        if self.start_snapshot is None or self.end_snapshot is None:
            return MemoryDelta("none", "none", 0.0)
        return compute_delta(self.start_snapshot, self.end_snapshot)
    
    def summary(self) -> str:
        """Get a summary of memory usage."""
        lines = [f"Memory Tracking Summary: {self.label}"]
        lines.append("=" * 50)
        
        if self.start_snapshot:
            lines.append(str(self.start_snapshot))
        
        for snap in self.intermediate_snapshots:
            lines.append(str(snap))
        
        if self.end_snapshot:
            lines.append(str(self.end_snapshot))
        
        lines.append("-" * 50)
        lines.append(str(self.delta))
        
        return "\n".join(lines)


@contextmanager
def memory_scope(label: str = "scope", clear_cache: bool = True):
    """
    Context manager that ensures memory is freed after scope exits.
    
    Args:
        label: Label for debugging
        clear_cache: Whether to call torch.cuda.empty_cache() on exit
        
    Example:
        >>> with memory_scope("compute"):
        ...     large_tensor = torch.randn(10000, 10000, device='cuda')
        ...     result = process(large_tensor)
        >>> # large_tensor is freed here
    """
    try:
        yield
    finally:
        gc.collect()
        if clear_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()


class TensorRegistry:
    """
    Registry for tracking tensor lifetimes and detecting leaks.
    
    Uses weak references to track tensors without preventing garbage collection.
    
    Example:
        >>> registry = TensorRegistry()
        >>> tensor = torch.randn(100, 100)
        >>> registry.register(tensor, "my_tensor")
        >>> # ... later ...
        >>> leaks = registry.find_leaks()
    """
    
    def __init__(self):
        self._registry: Dict[int, Dict[str, Any]] = {}
        self._weak_refs: Dict[int, weakref.ref] = {}
    
    def register(self, tensor: torch.Tensor, name: str = "", metadata: Optional[Dict] = None):
        """
        Register a tensor for tracking.
        
        Args:
            tensor: Tensor to track
            name: Human-readable name
            metadata: Additional metadata
        """
        tensor_id = id(tensor)
        
        def cleanup(ref):
            self._registry.pop(tensor_id, None)
            self._weak_refs.pop(tensor_id, None)
        
        self._weak_refs[tensor_id] = weakref.ref(tensor, cleanup)
        self._registry[tensor_id] = {
            "name": name,
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "size_bytes": tensor.element_size() * tensor.numel(),
            "created_at": time.time(),
            "metadata": metadata or {}
        }
    
    def find_leaks(self, max_age_seconds: float = 60.0) -> List[Dict]:
        """
        Find tensors that may be leaking (still alive after max_age).
        
        Args:
            max_age_seconds: Tensors older than this are considered potential leaks
            
        Returns:
            List of leak info dicts
        """
        gc.collect()
        now = time.time()
        leaks = []
        
        for tensor_id, info in list(self._registry.items()):
            ref = self._weak_refs.get(tensor_id)
            if ref is not None and ref() is not None:
                age = now - info["created_at"]
                if age > max_age_seconds:
                    leaks.append({
                        **info,
                        "age_seconds": age
                    })
        
        return leaks
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about tracked tensors."""
        gc.collect()
        
        alive = 0
        total_bytes = 0
        by_device: Dict[str, int] = {}
        
        for tensor_id, info in self._registry.items():
            ref = self._weak_refs.get(tensor_id)
            if ref is not None and ref() is not None:
                alive += 1
                total_bytes += info["size_bytes"]
                device = info["device"]
                by_device[device] = by_device.get(device, 0) + info["size_bytes"]
        
        return {
            "alive_tensors": alive,
            "total_bytes": total_bytes,
            "by_device": by_device
        }
    
    def clear(self):
        """Clear the registry."""
        self._registry.clear()
        self._weak_refs.clear()


# Global registry for module-level tracking
_global_registry: Optional[TensorRegistry] = None


def get_global_registry() -> TensorRegistry:
    """Get or create the global tensor registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = TensorRegistry()
    return _global_registry


def reset_global_registry():
    """Reset the global tensor registry."""
    global _global_registry
    if _global_registry is not None:
        _global_registry.clear()
    _global_registry = None


def get_cuda_memory_summary() -> str:
    """
    Get a human-readable summary of CUDA memory usage.
    
    Returns:
        Formatted string with memory stats
    """
    if not torch.cuda.is_available():
        return "CUDA not available"
    
    lines = ["CUDA Memory Summary:"]
    lines.append("=" * 50)
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i)
        reserved = torch.cuda.memory_reserved(i)
        max_allocated = torch.cuda.max_memory_allocated(i)
        
        lines.append(f"\nDevice {i}: {props.name}")
        lines.append(f"  Total memory: {props.total_memory / 1e9:.2f} GB")
        lines.append(f"  Allocated: {allocated / 1e6:.2f} MB")
        lines.append(f"  Reserved: {reserved / 1e6:.2f} MB")
        lines.append(f"  Max allocated: {max_allocated / 1e6:.2f} MB")
        lines.append(f"  Utilization: {allocated / props.total_memory * 100:.1f}%")
    
    return "\n".join(lines)
