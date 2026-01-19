"""
Kernel Autotuning Cache System
==============================

Profile GPU kernels and cache optimal configurations.

The problem: Different GPUs, problem sizes, and tensor shapes
have different optimal kernel configurations. Finding the best
one every run wastes time.

The solution:
    1. Profile kernels with different configurations
    2. Cache the best configuration per (kernel, shape, device)
    3. Reuse cached config on subsequent runs

Cached parameters:
    - Thread block dimensions
    - Grid dimensions
    - Shared memory size
    - Loop unrolling factors
    - Tile sizes
    - Number of warps

Phase 24: Physics Toolbox Extension
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import threading

import torch
from torch import Tensor


@dataclass
class KernelConfig:
    """Configuration for a kernel execution."""
    block_x: int = 256
    block_y: int = 1
    block_z: int = 1
    grid_x: int = 0  # 0 = auto
    grid_y: int = 1
    grid_z: int = 1
    shared_mem_bytes: int = 0
    stream: Optional[int] = None
    
    # Algorithmic parameters
    tile_size: int = 32
    unroll_factor: int = 4
    num_warps: int = 8
    
    # Extra config (kernel-specific)
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> 'KernelConfig':
        extra = d.pop('extra', {})
        return cls(**d, extra=extra)
    
    def key(self) -> str:
        """Generate unique key for this config."""
        return hashlib.md5(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]


@dataclass
class ProfilingResult:
    """Result from kernel profiling."""
    config: KernelConfig
    time_ms: float
    memory_bytes: int
    achieved_occupancy: float = 0.0
    throughput_gbps: float = 0.0
    
    def __lt__(self, other: 'ProfilingResult') -> bool:
        return self.time_ms < other.time_ms


@dataclass
class CacheEntry:
    """Entry in the autotune cache."""
    kernel_name: str
    shape_hash: str
    device_name: str
    best_config: KernelConfig
    profiling_time_ms: float
    speedup_vs_default: float
    timestamp: str
    pytorch_version: str
    cuda_version: str


class AutotuneCache:
    """
    Persistent cache for autotuned kernel configurations.
    
    Stores optimal configurations per (kernel, shape, device) tuple.
    Cache is persisted to disk and loaded on startup.
    """
    
    def __init__(
        self,
        cache_dir: str = "~/.cache/hypertensor/autotune",
        max_entries: int = 10000,
    ):
        self.cache_dir = Path(os.path.expanduser(cache_dir))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        
        # Load existing cache
        self._load_cache()
    
    def _cache_file(self) -> Path:
        return self.cache_dir / "autotune_cache.json"
    
    def _load_cache(self):
        """Load cache from disk."""
        cache_file = self._cache_file()
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                
                for key, entry_dict in data.items():
                    config_dict = entry_dict.pop('best_config')
                    entry = CacheEntry(
                        best_config=KernelConfig.from_dict(config_dict),
                        **entry_dict
                    )
                    self._cache[key] = entry
                    
            except Exception as e:
                print(f"Warning: Failed to load autotune cache: {e}")
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            data = {}
            for key, entry in self._cache.items():
                entry_dict = {
                    'kernel_name': entry.kernel_name,
                    'shape_hash': entry.shape_hash,
                    'device_name': entry.device_name,
                    'best_config': entry.best_config.to_dict(),
                    'profiling_time_ms': entry.profiling_time_ms,
                    'speedup_vs_default': entry.speedup_vs_default,
                    'timestamp': entry.timestamp,
                    'pytorch_version': entry.pytorch_version,
                    'cuda_version': entry.cuda_version,
                }
                data[key] = entry_dict
            
            with open(self._cache_file(), 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Failed to save autotune cache: {e}")
    
    @staticmethod
    def _shape_hash(shapes: List[Tuple[int, ...]]) -> str:
        """Generate hash for tensor shapes."""
        shape_str = str(shapes)
        return hashlib.md5(shape_str.encode()).hexdigest()[:12]
    
    @staticmethod
    def _device_name() -> str:
        """Get device identifier."""
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return f"{props.name}_{props.major}.{props.minor}"
        return "cpu"
    
    def _make_key(self, kernel_name: str, shapes: List[Tuple[int, ...]]) -> str:
        """Generate cache key."""
        shape_hash = self._shape_hash(shapes)
        device = self._device_name()
        return f"{kernel_name}_{shape_hash}_{device}"
    
    def get(
        self,
        kernel_name: str,
        shapes: List[Tuple[int, ...]],
    ) -> Optional[KernelConfig]:
        """
        Get cached config for kernel and shapes.
        
        Args:
            kernel_name: Name of the kernel
            shapes: Input tensor shapes
            
        Returns:
            Cached KernelConfig or None if not found
        """
        key = self._make_key(kernel_name, shapes)
        with self._lock:
            entry = self._cache.get(key)
            return entry.best_config if entry else None
    
    def put(
        self,
        kernel_name: str,
        shapes: List[Tuple[int, ...]],
        config: KernelConfig,
        profiling_time_ms: float,
        speedup_vs_default: float,
    ):
        """
        Store config in cache.
        
        Args:
            kernel_name: Name of the kernel
            shapes: Input tensor shapes
            config: Optimal configuration
            profiling_time_ms: Time spent profiling
            speedup_vs_default: Speedup over default config
        """
        key = self._make_key(kernel_name, shapes)
        
        from datetime import datetime
        
        entry = CacheEntry(
            kernel_name=kernel_name,
            shape_hash=self._shape_hash(shapes),
            device_name=self._device_name(),
            best_config=config,
            profiling_time_ms=profiling_time_ms,
            speedup_vs_default=speedup_vs_default,
            timestamp=datetime.now().isoformat(),
            pytorch_version=torch.__version__,
            cuda_version=torch.version.cuda or "N/A",
        )
        
        with self._lock:
            # Evict old entries if needed
            if len(self._cache) >= self.max_entries:
                # Remove oldest
                oldest_key = min(self._cache.keys(), 
                                key=lambda k: self._cache[k].timestamp)
                del self._cache[oldest_key]
            
            self._cache[key] = entry
            self._save_cache()
    
    def clear(self):
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            if self._cache_file().exists():
                self._cache_file().unlink()


class KernelAutotuner:
    """
    Automatic kernel tuning with caching.
    
    Profiles different configurations and caches the best one.
    
    Example:
        >>> autotuner = KernelAutotuner()
        >>> 
        >>> @autotuner.autotune("my_kernel")
        ... def my_kernel(a, b, config: KernelConfig):
        ...     # Use config.tile_size, config.block_x, etc.
        ...     return a @ b
        >>> 
        >>> # First call: profiles and caches
        >>> result = my_kernel(tensor_a, tensor_b)
        >>> 
        >>> # Subsequent calls: uses cached config
        >>> result = my_kernel(tensor_a, tensor_b)
    """
    
    def __init__(
        self,
        cache: Optional[AutotuneCache] = None,
        warmup_iters: int = 3,
        profile_iters: int = 10,
    ):
        self.cache = cache or AutotuneCache()
        self.warmup_iters = warmup_iters
        self.profile_iters = profile_iters
    
    def generate_configs(
        self,
        shapes: List[Tuple[int, ...]],
        kernel_type: str = "matmul",
    ) -> List[KernelConfig]:
        """
        Generate candidate configurations to profile.
        
        Args:
            shapes: Input tensor shapes
            kernel_type: Type of kernel (matmul, conv, elementwise, etc.)
            
        Returns:
            List of configurations to try
        """
        configs = []
        
        # Get device info
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            max_threads = props.max_threads_per_block
            max_shared = props.max_shared_memory_per_block
            sm_count = props.multi_processor_count
        else:
            max_threads = 1024
            max_shared = 48 * 1024
            sm_count = 1
        
        if kernel_type == "matmul":
            # Matrix multiply: vary tile size and block dims
            for tile in [16, 32, 64, 128]:
                for block in [64, 128, 256, 512]:
                    if block <= max_threads:
                        configs.append(KernelConfig(
                            block_x=block,
                            tile_size=tile,
                            num_warps=block // 32,
                        ))
        
        elif kernel_type == "elementwise":
            # Elementwise: mainly vary block size
            for block in [64, 128, 256, 512, 1024]:
                if block <= max_threads:
                    configs.append(KernelConfig(
                        block_x=block,
                        unroll_factor=4,
                    ))
        
        elif kernel_type == "reduction":
            # Reduction: block size and warp count matter
            for block in [32, 64, 128, 256, 512]:
                if block <= max_threads:
                    for warps in [1, 2, 4, 8]:
                        if warps * 32 <= block:
                            configs.append(KernelConfig(
                                block_x=block,
                                num_warps=warps,
                            ))
        
        elif kernel_type == "contraction":
            # Tensor contraction: tile size critical
            for tile in [8, 16, 32]:
                for block in [64, 128, 256]:
                    if block <= max_threads:
                        configs.append(KernelConfig(
                            block_x=block,
                            tile_size=tile,
                            num_warps=block // 32,
                        ))
        
        else:
            # Default: sweep block sizes
            for block in [64, 128, 256, 512]:
                if block <= max_threads:
                    configs.append(KernelConfig(block_x=block))
        
        return configs
    
    def profile_config(
        self,
        kernel_fn: Callable,
        args: Tuple,
        config: KernelConfig,
    ) -> ProfilingResult:
        """
        Profile a single configuration.
        
        Args:
            kernel_fn: Kernel function to profile
            args: Arguments to pass to kernel
            config: Configuration to use
            
        Returns:
            ProfilingResult with timing and metrics
        """
        if not torch.cuda.is_available():
            # CPU fallback
            start = time.perf_counter()
            for _ in range(self.profile_iters):
                kernel_fn(*args, config=config)
            elapsed = (time.perf_counter() - start) / self.profile_iters * 1000
            return ProfilingResult(config=config, time_ms=elapsed, memory_bytes=0)
        
        # GPU profiling with CUDA events
        torch.cuda.synchronize()
        
        # Warmup
        for _ in range(self.warmup_iters):
            kernel_fn(*args, config=config)
        
        torch.cuda.synchronize()
        
        # Profile
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(self.profile_iters):
            kernel_fn(*args, config=config)
        end_event.record()
        
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event) / self.profile_iters
        
        # Get memory usage
        memory_bytes = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        
        return ProfilingResult(
            config=config,
            time_ms=elapsed_ms,
            memory_bytes=memory_bytes,
        )
    
    def find_best_config(
        self,
        kernel_name: str,
        kernel_fn: Callable,
        args: Tuple,
        kernel_type: str = "matmul",
    ) -> Tuple[KernelConfig, float]:
        """
        Find optimal config by profiling all candidates.
        
        Args:
            kernel_name: Name for caching
            kernel_fn: Kernel function
            args: Arguments to kernel
            kernel_type: Type of kernel for config generation
            
        Returns:
            Tuple of (best_config, speedup_vs_default)
        """
        shapes = [arg.shape if hasattr(arg, 'shape') else () for arg in args]
        
        # Check cache first
        cached = self.cache.get(kernel_name, shapes)
        if cached is not None:
            return cached, 1.0  # Already optimal
        
        print(f"Autotuning {kernel_name}...")
        start_time = time.time()
        
        # Generate configs
        configs = self.generate_configs(shapes, kernel_type)
        
        # Profile default first
        default_config = KernelConfig()
        default_result = self.profile_config(kernel_fn, args, default_config)
        
        # Profile all configs
        results = [default_result]
        for config in configs:
            try:
                result = self.profile_config(kernel_fn, args, config)
                results.append(result)
            except Exception as e:
                continue  # Skip configs that error
        
        # Find best
        best_result = min(results)
        speedup = default_result.time_ms / best_result.time_ms
        
        profiling_time_ms = (time.time() - start_time) * 1000
        
        # Cache result
        self.cache.put(
            kernel_name, shapes, best_result.config,
            profiling_time_ms, speedup
        )
        
        print(f"  Best config: {best_result.config.key()}")
        print(f"  Time: {best_result.time_ms:.3f}ms (speedup: {speedup:.2f}x)")
        
        return best_result.config, speedup
    
    def autotune(
        self,
        kernel_name: str,
        kernel_type: str = "matmul",
    ) -> Callable:
        """
        Decorator for auto-tuned kernels.
        
        The decorated function must accept a `config` keyword argument.
        
        Args:
            kernel_name: Name for caching
            kernel_type: Type of kernel
            
        Returns:
            Decorator function
        """
        def decorator(fn: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                # Extract shapes
                shapes = [arg.shape if hasattr(arg, 'shape') else () 
                         for arg in args]
                
                # Check cache
                config = self.cache.get(kernel_name, shapes)
                
                if config is None:
                    # Need to autotune
                    config, _ = self.find_best_config(
                        kernel_name, fn, args, kernel_type
                    )
                
                return fn(*args, config=config, **kwargs)
            
            wrapper.__name__ = fn.__name__
            wrapper.__doc__ = fn.__doc__
            return wrapper
        
        return decorator


# =============================================================================
# QTT-Specific Autotuning
# =============================================================================

class QTTAutotuner(KernelAutotuner):
    """
    Autotuner specialized for QTT operations.
    
    Optimizes:
    - Core contractions
    - Hadamard products
    - SVD truncations
    - Rounding operations
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._contraction_cache: Dict[str, KernelConfig] = {}
    
    def tune_contraction(
        self,
        left_shape: Tuple[int, ...],
        right_shape: Tuple[int, ...],
        contract_dims: Tuple[int, ...],
    ) -> KernelConfig:
        """
        Find optimal config for tensor contraction.
        
        Args:
            left_shape: Shape of left tensor
            right_shape: Shape of right tensor
            contract_dims: Dimensions to contract
            
        Returns:
            Optimal KernelConfig
        """
        key = f"contract_{left_shape}_{right_shape}_{contract_dims}"
        shapes = [left_shape, right_shape]
        
        cached = self.cache.get(key, shapes)
        if cached:
            return cached
        
        # Create dummy tensors for profiling
        device = "cuda" if torch.cuda.is_available() else "cpu"
        left = torch.randn(left_shape, device=device)
        right = torch.randn(right_shape, device=device)
        
        def contract_kernel(l, r, config: KernelConfig):
            # Use config.tile_size for chunking if needed
            return torch.tensordot(l, r, dims=[contract_dims, contract_dims])
        
        config, _ = self.find_best_config(
            key, contract_kernel, (left, right), "contraction"
        )
        
        return config
    
    def tune_hadamard(self, shape: Tuple[int, ...]) -> KernelConfig:
        """Find optimal config for Hadamard product."""
        key = f"hadamard_{shape}"
        shapes = [shape, shape]
        
        cached = self.cache.get(key, shapes)
        if cached:
            return cached
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        a = torch.randn(shape, device=device)
        b = torch.randn(shape, device=device)
        
        def hadamard_kernel(x, y, config: KernelConfig):
            return x * y
        
        config, _ = self.find_best_config(
            key, hadamard_kernel, (a, b), "elementwise"
        )
        
        return config


# =============================================================================
# Convenience Functions
# =============================================================================

# Global autotuner instance
_global_autotuner: Optional[KernelAutotuner] = None


def get_autotuner() -> KernelAutotuner:
    """Get global autotuner instance."""
    global _global_autotuner
    if _global_autotuner is None:
        _global_autotuner = KernelAutotuner()
    return _global_autotuner


def autotune(kernel_name: str, kernel_type: str = "matmul") -> Callable:
    """
    Decorator for auto-tuned kernels using global autotuner.
    
    Example:
        >>> @autotune("my_matmul", "matmul")
        ... def optimized_matmul(a, b, config: KernelConfig):
        ...     return a @ b
    """
    return get_autotuner().autotune(kernel_name, kernel_type)


def clear_autotune_cache():
    """Clear the global autotune cache."""
    get_autotuner().cache.clear()


if __name__ == "__main__":
    print("Testing Kernel Autotune Cache System...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Test autotuner
    autotuner = KernelAutotuner(warmup_iters=2, profile_iters=5)
    
    # Define a tunable kernel
    def tunable_matmul(a: Tensor, b: Tensor, config: KernelConfig) -> Tensor:
        # In a real kernel, we'd use config.tile_size, etc.
        return a @ b
    
    # Test tensors
    sizes = [(256, 256), (512, 512), (1024, 1024)]
    
    for size in sizes:
        print(f"\nMatrix size: {size}")
        a = torch.randn(size, device=device)
        b = torch.randn(size, device=device)
        
        # First call: profiles
        config, speedup = autotuner.find_best_config(
            f"matmul_{size[0]}", tunable_matmul, (a, b), "matmul"
        )
        
        # Second call: uses cache
        cached_config = autotuner.cache.get(f"matmul_{size[0]}", [size, size])
        assert cached_config is not None, "Config should be cached"
        print(f"  Cached config found: {cached_config.key()}")
    
    # Test QTT autotuner
    print("\nTesting QTT-specific autotuner...")
    qtt_tuner = QTTAutotuner()
    
    # Tune contraction
    config = qtt_tuner.tune_contraction(
        left_shape=(16, 2, 2, 32),
        right_shape=(32, 2, 2, 16),
        contract_dims=(3,),
    )
    print(f"  Contraction config: tile={config.tile_size}, block={config.block_x}")
    
    print("\n✓ Autotune cache test passed!")
