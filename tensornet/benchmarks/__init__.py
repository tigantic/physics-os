"""
TensorRT integration benchmarking module.

This module provides comprehensive benchmarking utilities for evaluating
TensorRT export and inference performance across different configurations.
"""

from .benchmark_suite import (
    BenchmarkConfig,
    BenchmarkResult,
    PrecisionBenchmark,
    LatencyBenchmark,
    ThroughputBenchmark,
    MemoryBenchmark,
    AccuracyBenchmark,
    TensorRTBenchmarkSuite,
    run_tensorrt_benchmarks,
    compare_precision_modes,
)
from .profiler import (
    ProfileConfig,
    ProfileResult,
    LayerProfile,
    OperationProfile,
    TensorRTProfiler,
    profile_model,
    profile_inference,
)
from .reports import (
    BenchmarkReport,
    ReportFormat,
    generate_report,
    export_to_csv,
    export_to_json,
    export_to_markdown,
)
from .analysis import (
    PerformanceAnalysis,
    BottleneckAnalysis,
    OptimizationRecommendation,
    analyze_performance,
    identify_bottlenecks,
    recommend_optimizations,
)

__all__ = [
    # Benchmark Suite
    'BenchmarkConfig',
    'BenchmarkResult',
    'PrecisionBenchmark',
    'LatencyBenchmark',
    'ThroughputBenchmark',
    'MemoryBenchmark',
    'AccuracyBenchmark',
    'TensorRTBenchmarkSuite',
    'run_tensorrt_benchmarks',
    'compare_precision_modes',
    # Profiler
    'ProfileConfig',
    'ProfileResult',
    'LayerProfile',
    'OperationProfile',
    'TensorRTProfiler',
    'profile_model',
    'profile_inference',
    # Reports
    'BenchmarkReport',
    'ReportFormat',
    'generate_report',
    'export_to_csv',
    'export_to_json',
    'export_to_markdown',
    # Analysis
    'PerformanceAnalysis',
    'BottleneckAnalysis',
    'OptimizationRecommendation',
    'analyze_performance',
    'identify_bottlenecks',
    'recommend_optimizations',
]
