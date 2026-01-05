"""
TensorRT integration benchmarking module.

This module provides comprehensive benchmarking utilities for evaluating
TensorRT export and inference performance across different configurations.
"""

from .analysis import (
                       BottleneckAnalysis,
                       OptimizationRecommendation,
                       PerformanceAnalysis,
                       analyze_performance,
                       identify_bottlenecks,
                       recommend_optimizations,
)
from .benchmark_suite import (
                       AccuracyBenchmark,
                       BenchmarkConfig,
                       BenchmarkResult,
                       LatencyBenchmark,
                       MemoryBenchmark,
                       PrecisionBenchmark,
                       TensorRTBenchmarkSuite,
                       ThroughputBenchmark,
                       compare_precision_modes,
                       run_tensorrt_benchmarks,
)
from .profiler import (
                       LayerProfile,
                       OperationProfile,
                       ProfileConfig,
                       ProfileResult,
                       TensorRTProfiler,
                       profile_inference,
                       profile_model,
)
from .reports import (
                       BenchmarkReport,
                       ReportFormat,
                       export_to_csv,
                       export_to_json,
                       export_to_markdown,
                       generate_report,
)

__all__ = [
    # Benchmark Suite
    "BenchmarkConfig",
    "BenchmarkResult",
    "PrecisionBenchmark",
    "LatencyBenchmark",
    "ThroughputBenchmark",
    "MemoryBenchmark",
    "AccuracyBenchmark",
    "TensorRTBenchmarkSuite",
    "run_tensorrt_benchmarks",
    "compare_precision_modes",
    # Profiler
    "ProfileConfig",
    "ProfileResult",
    "LayerProfile",
    "OperationProfile",
    "TensorRTProfiler",
    "profile_model",
    "profile_inference",
    # Reports
    "BenchmarkReport",
    "ReportFormat",
    "generate_report",
    "export_to_csv",
    "export_to_json",
    "export_to_markdown",
    # Analysis
    "PerformanceAnalysis",
    "BottleneckAnalysis",
    "OptimizationRecommendation",
    "analyze_performance",
    "identify_bottlenecks",
    "recommend_optimizations",
]
