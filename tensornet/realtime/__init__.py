# Copyright (c) 2025 Tigantic
# Phase 18: Real-Time Inference Optimization
"""
Real-time inference optimization for tensor network models.

This module provides high-performance inference capabilities including:
- Optimized inference engine with dynamic batching
- Kernel fusion for reduced overhead
- Memory management and caching
- Latency optimization with mixed precision
"""

from .inference_engine import (
    InferenceConfig,
    InferencePriority,
    InferenceResult,
    BatchRequest,
    InferenceEngine,
    run_inference,
    run_batched_inference,
)

from .kernel_fusion import (
    FusionPattern,
    FusionType,
    OperatorNode,
    OperatorGraph,
    KernelFuser,
    fuse_operators,
    optimize_graph,
)

from .memory_manager import (
    MemoryConfig,
    AllocationStrategy,
    TensorHandle,
    TensorCache,
    StreamingBuffer,
    MemoryPool,
    MemoryPlanner,
    optimize_memory,
)

from .latency_optimizer import (
    LatencyTarget,
    PrecisionPolicy,
    PipelineConfig,
    LatencyProfile,
    LatencyOptimizer,
    PipelineOptimizer,
    PrecisionScheduler,
    optimize_for_latency,
)

__all__ = [
    # Inference engine
    "InferenceConfig",
    "InferencePriority",
    "InferenceResult",
    "BatchRequest",
    "InferenceEngine",
    "run_inference",
    "run_batched_inference",
    # Kernel fusion
    "FusionPattern",
    "FusionType",
    "OperatorNode",
    "OperatorGraph",
    "KernelFuser",
    "fuse_operators",
    "optimize_graph",
    # Memory management
    "MemoryConfig",
    "AllocationStrategy",
    "TensorHandle",
    "TensorCache",
    "StreamingBuffer",
    "MemoryPool",
    "MemoryPlanner",
    "optimize_memory",
    # Latency optimization
    "LatencyTarget",
    "PrecisionPolicy",
    "PipelineConfig",
    "LatencyProfile",
    "LatencyOptimizer",
    "PipelineOptimizer",
    "PrecisionScheduler",
    "optimize_for_latency",
]
