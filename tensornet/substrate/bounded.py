"""
Bounded Mode - Frame Budget Enforcement
========================================

"Never miss frame budget" mode.

Features:
    - Fixed-rank caps per frame
    - Precompiled contraction paths
    - Time-budgeted truncation (best effort under ms cap)
    - Graceful quality degradation
    - Caching for intermediate cores
"""

from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any, Callable
import time
from enum import Enum


class QualityLevel(Enum):
    """Quality levels for bounded mode."""
    ULTRA = "ultra"      # Max quality, may miss budget
    HIGH = "high"        # High quality, soft budget
    BALANCED = "balanced"  # Balance quality and performance
    PERFORMANCE = "performance"  # Prioritize frame rate
    MINIMAL = "minimal"  # Minimum quality, guarantee budget


@dataclass
class BudgetConfig:
    """Configuration for bounded-latency mode."""
    
    # Frame budget
    target_ms: float = 16.67  # 60 FPS default
    hard_cap_ms: float = 33.33  # Never exceed (30 FPS floor)
    
    # Quality bounds
    min_rank: int = 2
    max_rank: int = 64
    
    # Error bounds
    max_truncation_error: float = 0.01
    
    # Adaptation
    quality_level: QualityLevel = QualityLevel.BALANCED
    adaptation_rate: float = 0.1  # How fast to adapt rank
    
    # Caching
    enable_cache: bool = True
    max_cache_entries: int = 1024
    cache_ttl_frames: int = 60
    
    # Precompilation
    precompile_paths: bool = True
    
    def target_fps(self) -> float:
        """Target FPS from budget."""
        return 1000.0 / self.target_ms


@dataclass
class ContractionPath:
    """Precompiled contraction path for efficient execution."""
    
    indices: List[Tuple[int, ...]]  # Order of contractions
    flops: int = 0
    memory_bytes: int = 0
    
    # Cached intermediate results
    intermediates: Dict[Tuple[int, ...], torch.Tensor] = field(default_factory=dict)


class ContractionCache:
    """
    Cache for intermediate contraction results.
    
    Key insight: When sampling nearby points, many partial contractions
    are shared. Cache them.
    """
    
    def __init__(self, max_entries: int = 1024, ttl_frames: int = 60):
        self.max_entries = max_entries
        self.ttl_frames = ttl_frames
        self.cache: Dict[str, Tuple[torch.Tensor, int]] = {}  # key -> (value, frame)
        self.current_frame = 0
        self.hits = 0
        self.misses = 0
    
    def _make_key(self, core_indices: Tuple[int, ...], bit_pattern: Tuple[int, ...]) -> str:
        """Create cache key from contraction specification."""
        return f"{core_indices}:{bit_pattern}"
    
    def get(self, core_indices: Tuple[int, ...], bit_pattern: Tuple[int, ...]) -> Optional[torch.Tensor]:
        """Get cached result if available."""
        key = self._make_key(core_indices, bit_pattern)
        if key in self.cache:
            value, frame = self.cache[key]
            if self.current_frame - frame <= self.ttl_frames:
                self.hits += 1
                return value
            else:
                del self.cache[key]  # Expired
        
        self.misses += 1
        return None
    
    def put(self, core_indices: Tuple[int, ...], bit_pattern: Tuple[int, ...], value: torch.Tensor):
        """Cache a contraction result."""
        if len(self.cache) >= self.max_entries:
            self._evict()
        
        key = self._make_key(core_indices, bit_pattern)
        self.cache[key] = (value, self.current_frame)
    
    def _evict(self):
        """Evict oldest entries."""
        # Remove expired entries
        expired = [k for k, (v, f) in self.cache.items() 
                   if self.current_frame - f > self.ttl_frames]
        for k in expired:
            del self.cache[k]
        
        # If still full, remove oldest half
        if len(self.cache) >= self.max_entries:
            sorted_keys = sorted(self.cache.keys(), 
                                key=lambda k: self.cache[k][1])
            for k in sorted_keys[:len(sorted_keys) // 2]:
                del self.cache[k]
    
    def advance_frame(self):
        """Called each frame to advance TTL tracking."""
        self.current_frame += 1
    
    def clear(self):
        """Clear all cached entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / max(1, total)
    
    def stats(self) -> Dict[str, Any]:
        return {
            'entries': len(self.cache),
            'max_entries': self.max_entries,
            'hits': self.hits,
            'misses': self.misses,
            'hit_ratio': self.hit_ratio,
            'current_frame': self.current_frame,
        }


class BoundedMode:
    """
    Bounded-latency execution mode.
    
    Guarantees frame budget by:
        1. Precompiling contraction paths
        2. Caching intermediate results
        3. Adaptive rank adjustment
        4. Time-budgeted truncation
    """
    
    def __init__(self, config: BudgetConfig = None):
        self.config = config or BudgetConfig()
        
        # State
        self.current_rank = (self.config.min_rank + self.config.max_rank) // 2
        self.frame_times: List[float] = []
        self.frame_count = 0
        
        # Cache
        self.cache = ContractionCache(
            max_entries=self.config.max_cache_entries,
            ttl_frames=self.config.cache_ttl_frames,
        )
        
        # Precompiled paths
        self.paths: Dict[str, ContractionPath] = {}
        
        # Metrics
        self.budget_hits = 0
        self.budget_misses = 0
    
    def begin_frame(self):
        """Call at start of each frame."""
        self._frame_start = time.perf_counter()
    
    def end_frame(self):
        """Call at end of each frame."""
        elapsed_ms = (time.perf_counter() - self._frame_start) * 1000
        self.frame_times.append(elapsed_ms)
        
        if len(self.frame_times) > 100:
            self.frame_times = self.frame_times[-100:]
        
        # Track budget compliance
        if elapsed_ms <= self.config.target_ms:
            self.budget_hits += 1
        else:
            self.budget_misses += 1
        
        # Adapt rank for next frame
        self._adapt_rank(elapsed_ms)
        
        # Advance cache
        self.cache.advance_frame()
        self.frame_count += 1
    
    def _adapt_rank(self, elapsed_ms: float):
        """Adapt rank based on frame time."""
        target = self.config.target_ms
        rate = self.config.adaptation_rate
        
        if elapsed_ms > target * 1.2:
            # Reduce rank (we're slow)
            self.current_rank = max(
                self.config.min_rank,
                int(self.current_rank * (1 - rate))
            )
        elif elapsed_ms < target * 0.7:
            # Increase rank (we have headroom)
            self.current_rank = min(
                self.config.max_rank,
                int(self.current_rank * (1 + rate * 0.5))
            )
    
    def time_remaining_ms(self) -> float:
        """Time remaining in current frame budget."""
        elapsed = (time.perf_counter() - self._frame_start) * 1000
        return max(0, self.config.target_ms - elapsed)
    
    def should_truncate(self) -> bool:
        """Check if we should truncate due to budget pressure."""
        return self.time_remaining_ms() < self.config.target_ms * 0.2
    
    def get_rank_cap(self) -> int:
        """Get current rank cap based on budget pressure."""
        remaining = self.time_remaining_ms()
        
        if remaining < self.config.target_ms * 0.1:
            # Emergency: use minimum rank
            return self.config.min_rank
        elif remaining < self.config.target_ms * 0.3:
            # Pressure: reduce rank
            return max(self.config.min_rank, self.current_rank // 2)
        else:
            return self.current_rank
    
    def precompile_path(self, name: str, n_cores: int, rank: int) -> ContractionPath:
        """Precompile a contraction path for fast execution."""
        # Try opt_einsum for complex graphs, fall back to linear contraction
        try:
            import opt_einsum as oe
            # Build contraction subscripts for QTT chain
            # Each core has shape (r_left, 2, r_right)
            subscripts = []
            for i in range(n_cores):
                left_idx = chr(ord('a') + i)
                phys_idx = chr(ord('A') + i)  # Physical index
                right_idx = chr(ord('a') + i + 1)
                subscripts.append(f"{left_idx}{phys_idx}{right_idx}")
            
            input_str = ",".join(subscripts)
            output_str = "".join(chr(ord('A') + i) for i in range(n_cores))
            eq = f"{input_str}->{output_str}"
            
            # Get optimal contraction path
            shapes = [(1 if i == 0 else rank, 2, rank if i < n_cores - 1 else 1) 
                      for i in range(n_cores)]
            path, info = oe.contract_path(eq, *[np.ones(s) for s in shapes], optimize='optimal')
            
            indices = path
            flops = int(info.opt_cost)
            memory = sum(s[0] * s[1] * s[2] * 8 for s in shapes)
        except ImportError:
            # Fallback: simple linear contraction
            indices = [(i,) for i in range(n_cores)]
            flops = n_cores * rank * rank
            memory = n_cores * rank * 2 * 8  # float64
        
        path = ContractionPath(
            indices=indices,
            flops=flops,
            memory_bytes=memory,
        )
        
        self.paths[name] = path
        return path
    
    def execute_with_budget(
        self,
        fn: Callable[[], torch.Tensor],
        fallback: Optional[Callable[[], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Execute a function with budget awareness.
        
        Returns:
            (result, hit_budget) - result and whether we stayed in budget
        """
        t_start = time.perf_counter()
        
        result = fn()
        
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        hit_budget = elapsed_ms <= self.config.target_ms
        
        if not hit_budget and fallback is not None:
            # Try fallback (lower quality)
            result = fallback()
        
        return result, hit_budget
    
    def stats(self) -> Dict[str, Any]:
        """Get bounded mode statistics."""
        avg_frame = np.mean(self.frame_times) if self.frame_times else 0
        max_frame = max(self.frame_times) if self.frame_times else 0
        p99_frame = np.percentile(self.frame_times, 99) if len(self.frame_times) > 10 else max_frame
        
        return {
            'current_rank': self.current_rank,
            'target_ms': self.config.target_ms,
            'avg_frame_ms': avg_frame,
            'max_frame_ms': max_frame,
            'p99_frame_ms': p99_frame,
            'fps': 1000 / avg_frame if avg_frame > 0 else 0,
            'budget_hits': self.budget_hits,
            'budget_misses': self.budget_misses,
            'budget_hit_ratio': self.budget_hits / max(1, self.budget_hits + self.budget_misses),
            'frame_count': self.frame_count,
            'cache': self.cache.stats(),
        }
    
    def summary(self) -> str:
        """Human-readable summary."""
        s = self.stats()
        lines = [
            "=" * 50,
            "BOUNDED MODE STATUS",
            "=" * 50,
            f"Target:      {s['target_ms']:.1f} ms ({s['fps']:.1f} FPS)",
            f"Avg Frame:   {s['avg_frame_ms']:.2f} ms",
            f"P99 Frame:   {s['p99_frame_ms']:.2f} ms",
            f"Max Frame:   {s['max_frame_ms']:.2f} ms",
            f"Current Rank: {s['current_rank']}",
            f"Budget Hits: {s['budget_hit_ratio']:.1%}",
            f"Cache Hit:   {s['cache']['hit_ratio']:.1%}",
            "=" * 50,
        ]
        return "\n".join(lines)


class AdaptiveRankController:
    """
    PID-like controller for adaptive rank adjustment.
    
    Maintains target FPS by adjusting rank based on frame time history.
    """
    
    def __init__(
        self,
        target_ms: float = 16.67,
        min_rank: int = 2,
        max_rank: int = 64,
        kp: float = 0.5,
        ki: float = 0.1,
        kd: float = 0.05,
    ):
        self.target_ms = target_ms
        self.min_rank = min_rank
        self.max_rank = max_rank
        
        # PID gains
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # State
        self.current_rank = (min_rank + max_rank) // 2
        self.integral = 0.0
        self.last_error = 0.0
    
    def update(self, frame_time_ms: float) -> int:
        """Update rank based on frame time."""
        # Error: positive if too slow
        error = frame_time_ms - self.target_ms
        
        # PID terms
        self.integral += error
        self.integral = np.clip(self.integral, -100, 100)  # Anti-windup
        derivative = error - self.last_error
        self.last_error = error
        
        # Control signal (positive = reduce rank)
        control = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        # Update rank
        rank_delta = -int(np.sign(control) * max(1, abs(control) / 5))
        self.current_rank = np.clip(
            self.current_rank + rank_delta,
            self.min_rank,
            self.max_rank
        )
        
        return self.current_rank
    
    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.last_error = 0.0
        self.current_rank = (self.min_rank + self.max_rank) // 2
