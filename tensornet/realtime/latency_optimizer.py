# Copyright (c) 2025 Tigantic
# Phase 18: Latency Optimization
"""
Latency optimization for real-time inference.

Provides latency profiling, mixed precision scheduling, pipeline
optimization, and automatic tuning for meeting latency targets.
"""

from __future__ import annotations

import time
import statistics
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import torch
import numpy as np


class PrecisionPolicy(Enum):
    """Precision policies for inference."""
    
    FULL = auto()          # Full precision (FP32)
    MIXED_FP16 = auto()    # Mixed FP32/FP16
    MIXED_BF16 = auto()    # Mixed FP32/BF16
    FP16 = auto()          # Full FP16
    INT8 = auto()          # INT8 quantization
    DYNAMIC = auto()       # Dynamic precision selection


@dataclass
class LatencyTarget:
    """Latency target specification.
    
    Attributes:
        target_ms: Target latency in milliseconds
        max_ms: Maximum acceptable latency
        percentile: Target percentile (e.g., 0.99 for p99)
        priority: Target priority (lower = more important)
    """
    
    target_ms: float
    max_ms: float = 0.0
    percentile: float = 0.99
    priority: int = 1
    
    def __post_init__(self) -> None:
        """Validate and set defaults."""
        if self.max_ms <= 0:
            self.max_ms = self.target_ms * 2
        if not 0 < self.percentile <= 1:
            raise ValueError("percentile must be in (0, 1]")


@dataclass
class PipelineConfig:
    """Pipeline parallelism configuration.
    
    Attributes:
        num_stages: Number of pipeline stages
        micro_batch_size: Size of each micro-batch
        enable_async: Enable asynchronous execution
        overlap_compute_transfer: Overlap computation and data transfer
    """
    
    num_stages: int = 4
    micro_batch_size: int = 1
    enable_async: bool = True
    overlap_compute_transfer: bool = True
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.num_stages < 1:
            raise ValueError("num_stages must be >= 1")
        if self.micro_batch_size < 1:
            raise ValueError("micro_batch_size must be >= 1")


@dataclass
class LatencyProfile:
    """Latency profile from benchmarking.
    
    Attributes:
        measurements: List of latency measurements (ms)
        mean_ms: Mean latency
        std_ms: Standard deviation
        p50_ms: 50th percentile (median)
        p90_ms: 90th percentile
        p99_ms: 99th percentile
        min_ms: Minimum latency
        max_ms: Maximum latency
        samples: Number of samples
        metadata: Additional metadata
    """
    
    measurements: List[float]
    mean_ms: float = 0.0
    std_ms: float = 0.0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    samples: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_measurements(
        cls,
        measurements: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "LatencyProfile":
        """Create profile from measurements.
        
        Args:
            measurements: List of latency measurements (ms)
            metadata: Additional metadata
            
        Returns:
            LatencyProfile instance
        """
        if not measurements:
            return cls(measurements=[], metadata=metadata or {})
        
        sorted_measurements = sorted(measurements)
        n = len(sorted_measurements)
        
        return cls(
            measurements=measurements,
            mean_ms=statistics.mean(measurements),
            std_ms=statistics.stdev(measurements) if n > 1 else 0.0,
            p50_ms=sorted_measurements[int(n * 0.5)],
            p90_ms=sorted_measurements[int(n * 0.9)],
            p99_ms=sorted_measurements[int(n * 0.99)] if n >= 100 else sorted_measurements[-1],
            min_ms=sorted_measurements[0],
            max_ms=sorted_measurements[-1],
            samples=n,
            metadata=metadata or {},
        )
    
    def meets_target(self, target: LatencyTarget) -> bool:
        """Check if profile meets latency target.
        
        Args:
            target: Latency target
            
        Returns:
            True if target is met
        """
        if target.percentile >= 0.99:
            measured = self.p99_ms
        elif target.percentile >= 0.90:
            measured = self.p90_ms
        elif target.percentile >= 0.50:
            measured = self.p50_ms
        else:
            measured = self.mean_ms
        
        return measured <= target.target_ms and self.max_ms <= target.max_ms
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "p50_ms": self.p50_ms,
            "p90_ms": self.p90_ms,
            "p99_ms": self.p99_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "samples": self.samples,
            "metadata": self.metadata,
        }


class PrecisionScheduler:
    """Dynamic precision scheduling for latency optimization.
    
    Selects precision levels based on latency requirements and
    accuracy constraints.
    
    Attributes:
        policy: Current precision policy
        accuracy_threshold: Minimum accuracy to maintain
    """
    
    def __init__(
        self,
        policy: PrecisionPolicy = PrecisionPolicy.FULL,
        accuracy_threshold: float = 0.99,
    ) -> None:
        """Initialize precision scheduler.
        
        Args:
            policy: Initial precision policy
            accuracy_threshold: Minimum relative accuracy
        """
        self.policy = policy
        self.accuracy_threshold = accuracy_threshold
        
        # Layer-wise precision assignments
        self._layer_precisions: Dict[str, PrecisionPolicy] = {}
        
        # Profiling data
        self._latency_by_precision: Dict[PrecisionPolicy, float] = {}
        self._accuracy_by_precision: Dict[PrecisionPolicy, float] = {}
    
    def get_dtype(self) -> torch.dtype:
        """Get torch dtype for current policy.
        
        Returns:
            Appropriate torch dtype
        """
        dtype_map = {
            PrecisionPolicy.FULL: torch.float32,
            PrecisionPolicy.MIXED_FP16: torch.float16,
            PrecisionPolicy.MIXED_BF16: torch.bfloat16,
            PrecisionPolicy.FP16: torch.float16,
            PrecisionPolicy.INT8: torch.int8,
            PrecisionPolicy.DYNAMIC: torch.float32,
        }
        return dtype_map.get(self.policy, torch.float32)
    
    def should_use_autocast(self) -> bool:
        """Check if autocast should be used.
        
        Returns:
            True if autocast is appropriate
        """
        return self.policy in (
            PrecisionPolicy.MIXED_FP16,
            PrecisionPolicy.MIXED_BF16,
        )
    
    def profile_precision(
        self,
        model: Callable,
        sample_input: torch.Tensor,
        reference_output: Optional[torch.Tensor] = None,
        num_iterations: int = 100,
    ) -> Dict[PrecisionPolicy, Dict[str, float]]:
        """Profile model at different precisions.
        
        Args:
            model: Model to profile
            sample_input: Sample input tensor
            reference_output: Reference output for accuracy comparison
            num_iterations: Number of profiling iterations
            
        Returns:
            Dict mapping precision to latency/accuracy
        """
        results: Dict[PrecisionPolicy, Dict[str, float]] = {}
        
        for policy in [PrecisionPolicy.FULL, PrecisionPolicy.MIXED_FP16]:
            # Warm up
            with torch.no_grad():
                for _ in range(10):
                    if policy == PrecisionPolicy.MIXED_FP16:
                        with torch.autocast(device_type="cpu"):
                            _ = model(sample_input)
                    else:
                        _ = model(sample_input)
            
            # Profile
            latencies = []
            with torch.no_grad():
                for _ in range(num_iterations):
                    start = time.perf_counter()
                    
                    if policy == PrecisionPolicy.MIXED_FP16:
                        with torch.autocast(device_type="cpu"):
                            output = model(sample_input)
                    else:
                        output = model(sample_input)
                    
                    latencies.append((time.perf_counter() - start) * 1000)
            
            # Compute accuracy if reference provided
            accuracy = 1.0
            if reference_output is not None:
                diff = torch.abs(output.float() - reference_output.float())
                accuracy = 1.0 - float(torch.mean(diff / (torch.abs(reference_output) + 1e-8)))
            
            results[policy] = {
                "latency_ms": statistics.mean(latencies),
                "latency_std_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
                "accuracy": accuracy,
            }
            
            self._latency_by_precision[policy] = results[policy]["latency_ms"]
            self._accuracy_by_precision[policy] = accuracy
        
        return results
    
    def select_policy(
        self,
        target: LatencyTarget,
        min_accuracy: float = 0.99,
    ) -> PrecisionPolicy:
        """Select best precision policy for target.
        
        Args:
            target: Latency target
            min_accuracy: Minimum required accuracy
            
        Returns:
            Selected precision policy
        """
        # Sort by latency, filter by accuracy
        valid_policies = [
            (policy, latency)
            for policy, latency in self._latency_by_precision.items()
            if self._accuracy_by_precision.get(policy, 1.0) >= min_accuracy
        ]
        
        if not valid_policies:
            return PrecisionPolicy.FULL
        
        valid_policies.sort(key=lambda x: x[1])
        
        # Find lowest latency that meets target
        for policy, latency in valid_policies:
            if latency <= target.target_ms:
                return policy
        
        # Return fastest if none meet target
        return valid_policies[0][0]


class PipelineOptimizer:
    """Pipeline parallelism optimizer.
    
    Optimizes inference pipeline for latency and throughput
    using micro-batching and stage balancing.
    
    Attributes:
        config: Pipeline configuration
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        """Initialize pipeline optimizer.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self._stage_profiles: List[LatencyProfile] = []
    
    def profile_stages(
        self,
        stages: List[Callable],
        sample_input: torch.Tensor,
        num_iterations: int = 100,
    ) -> List[LatencyProfile]:
        """Profile each pipeline stage.
        
        Args:
            stages: List of stage callables
            sample_input: Sample input
            num_iterations: Profiling iterations
            
        Returns:
            LatencyProfile for each stage
        """
        profiles = []
        current_input = sample_input
        
        for i, stage in enumerate(stages):
            latencies = []
            
            # Warm up
            with torch.no_grad():
                for _ in range(10):
                    _ = stage(current_input)
            
            # Profile
            with torch.no_grad():
                for _ in range(num_iterations):
                    start = time.perf_counter()
                    output = stage(current_input)
                    latencies.append((time.perf_counter() - start) * 1000)
            
            profile = LatencyProfile.from_measurements(
                latencies,
                metadata={"stage": i, "name": f"stage_{i}"},
            )
            profiles.append(profile)
            current_input = output
        
        self._stage_profiles = profiles
        return profiles
    
    def compute_pipeline_latency(self) -> float:
        """Compute end-to-end pipeline latency.
        
        Returns:
            Total pipeline latency (ms)
        """
        if not self._stage_profiles:
            return 0.0
        
        # Pipeline latency is dominated by the slowest stage
        # plus one-time fill/drain overhead
        stage_latencies = [p.mean_ms for p in self._stage_profiles]
        
        # For a filled pipeline: latency = max_stage + (N-1) * max_stage
        # For single item: sum of all stages
        max_stage = max(stage_latencies)
        
        if self.config.enable_async:
            # Pipelined: approximately max_stage (when filled)
            return max_stage
        else:
            # Sequential: sum of all stages
            return sum(stage_latencies)
    
    def compute_throughput(self) -> float:
        """Compute pipeline throughput.
        
        Returns:
            Throughput in items/second
        """
        if not self._stage_profiles:
            return 0.0
        
        # Throughput limited by slowest stage
        max_stage_ms = max(p.mean_ms for p in self._stage_profiles)
        
        if max_stage_ms <= 0:
            return float('inf')
        
        # Items per second = 1000 / max_stage_ms
        return 1000.0 / max_stage_ms
    
    def optimize_batch_size(
        self,
        stages: List[Callable],
        sample_input: torch.Tensor,
        target: LatencyTarget,
        min_batch: int = 1,
        max_batch: int = 64,
    ) -> int:
        """Find optimal batch size for latency target.
        
        Args:
            stages: Pipeline stages
            sample_input: Sample input
            target: Latency target
            min_batch: Minimum batch size
            max_batch: Maximum batch size
            
        Returns:
            Optimal batch size
        """
        best_batch = min_batch
        best_throughput = 0.0
        
        for batch_size in range(min_batch, max_batch + 1):
            # Create batched input
            if sample_input.dim() == 0:
                batched = sample_input.unsqueeze(0).expand(batch_size)
            else:
                batched = sample_input.unsqueeze(0).expand(batch_size, *sample_input.shape)
            
            # Profile with this batch size
            latencies = []
            with torch.no_grad():
                for _ in range(20):
                    start = time.perf_counter()
                    x = batched
                    for stage in stages:
                        x = stage(x)
                    latencies.append((time.perf_counter() - start) * 1000)
            
            mean_latency = statistics.mean(latencies)
            
            # Check if meets target
            if mean_latency <= target.target_ms:
                throughput = batch_size * 1000.0 / mean_latency
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_batch = batch_size
            else:
                # Latency exceeded, stop increasing
                break
        
        return best_batch
    
    def balance_stages(
        self,
        stage_profiles: List[LatencyProfile],
    ) -> List[float]:
        """Compute suggested stage rebalancing weights.
        
        Args:
            stage_profiles: Current stage profiles
            
        Returns:
            Suggested computation weights per stage
        """
        if not stage_profiles:
            return []
        
        latencies = [p.mean_ms for p in stage_profiles]
        total = sum(latencies)
        
        if total <= 0:
            return [1.0 / len(latencies)] * len(latencies)
        
        # Ideal: all stages have equal latency
        ideal = total / len(latencies)
        
        # Weight = how much to scale each stage
        weights = [ideal / lat if lat > 0 else 1.0 for lat in latencies]
        
        return weights


class LatencyOptimizer:
    """Main latency optimization engine.
    
    Coordinates precision scheduling, pipeline optimization,
    and automatic tuning to meet latency targets.
    
    Attributes:
        target: Latency target
        precision_scheduler: Precision scheduler
        pipeline_optimizer: Pipeline optimizer
    """
    
    def __init__(
        self,
        target: Optional[LatencyTarget] = None,
        precision_policy: PrecisionPolicy = PrecisionPolicy.FULL,
        pipeline_config: Optional[PipelineConfig] = None,
    ) -> None:
        """Initialize latency optimizer.
        
        Args:
            target: Latency target
            precision_policy: Initial precision policy
            pipeline_config: Pipeline configuration
        """
        self.target = target or LatencyTarget(target_ms=10.0)
        self.precision_scheduler = PrecisionScheduler(precision_policy)
        self.pipeline_optimizer = PipelineOptimizer(pipeline_config)
        
        self._optimization_history: List[Dict[str, Any]] = []
    
    def profile(
        self,
        model: Callable,
        sample_input: torch.Tensor,
        num_iterations: int = 100,
        warmup_iterations: int = 10,
    ) -> LatencyProfile:
        """Profile model latency.
        
        Args:
            model: Model to profile
            sample_input: Sample input
            num_iterations: Profiling iterations
            warmup_iterations: Warmup iterations
            
        Returns:
            LatencyProfile
        """
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(sample_input)
        
        # Profile
        latencies = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.perf_counter()
                _ = model(sample_input)
                latencies.append((time.perf_counter() - start) * 1000)
        
        return LatencyProfile.from_measurements(latencies)
    
    def optimize(
        self,
        model: Callable,
        sample_input: torch.Tensor,
        max_iterations: int = 10,
    ) -> Dict[str, Any]:
        """Optimize model for latency target.
        
        Args:
            model: Model to optimize
            sample_input: Sample input
            max_iterations: Maximum optimization iterations
            
        Returns:
            Optimization result
        """
        initial_profile = self.profile(model, sample_input)
        
        result = {
            "initial_latency_ms": initial_profile.mean_ms,
            "target_ms": self.target.target_ms,
            "optimizations_applied": [],
            "final_latency_ms": initial_profile.mean_ms,
            "target_met": initial_profile.meets_target(self.target),
        }
        
        if result["target_met"]:
            return result
        
        # Try precision optimization
        self.precision_scheduler.profile_precision(model, sample_input)
        best_policy = self.precision_scheduler.select_policy(self.target)
        
        if best_policy != PrecisionPolicy.FULL:
            self.precision_scheduler.policy = best_policy
            result["optimizations_applied"].append(f"precision: {best_policy.name}")
        
        # Re-profile with optimizations
        final_profile = self.profile(model, sample_input)
        
        result["final_latency_ms"] = final_profile.mean_ms
        result["target_met"] = final_profile.meets_target(self.target)
        result["speedup"] = initial_profile.mean_ms / final_profile.mean_ms
        
        self._optimization_history.append(result)
        
        return result
    
    def get_recommendations(
        self,
        profile: LatencyProfile,
    ) -> List[str]:
        """Get optimization recommendations.
        
        Args:
            profile: Current latency profile
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if profile.mean_ms > self.target.target_ms:
            gap_ratio = profile.mean_ms / self.target.target_ms
            
            if gap_ratio > 2:
                recommendations.append(
                    "Consider reducing model complexity or using a smaller model"
                )
            
            if gap_ratio > 1.5:
                recommendations.append(
                    "Enable mixed precision (FP16) for ~2x speedup on supported hardware"
                )
            
            recommendations.append(
                "Profile individual layers to identify bottlenecks"
            )
        
        if profile.std_ms > profile.mean_ms * 0.2:
            recommendations.append(
                "High latency variance detected - consider warmup or CPU pinning"
            )
        
        if profile.max_ms > profile.p99_ms * 1.5:
            recommendations.append(
                "Outlier latencies detected - check for GC pauses or thermal throttling"
            )
        
        return recommendations


def optimize_for_latency(
    model: Callable,
    sample_input: torch.Tensor,
    target_ms: float,
    precision_policy: PrecisionPolicy = PrecisionPolicy.DYNAMIC,
) -> Dict[str, Any]:
    """Optimize a model for a latency target.
    
    Args:
        model: Model to optimize
        sample_input: Sample input tensor
        target_ms: Target latency in milliseconds
        precision_policy: Precision policy to use
        
    Returns:
        Optimization result dictionary
    """
    target = LatencyTarget(target_ms=target_ms)
    optimizer = LatencyOptimizer(target, precision_policy)
    
    return optimizer.optimize(model, sample_input)
