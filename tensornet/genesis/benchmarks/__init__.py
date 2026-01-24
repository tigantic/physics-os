"""
GENESIS Benchmark Suite

The Massacre.
"""

from .massacre import (
    BenchmarkResult,
    BenchmarkSuite,
    benchmark_wasserstein,
    benchmark_laplacian,
    benchmark_floyd_warshall,
    benchmark_mmd,
    benchmark_geometric_algebra,
    mega_scale_demo,
    print_massacre_report,
)

__all__ = [
    "BenchmarkResult",
    "BenchmarkSuite", 
    "benchmark_wasserstein",
    "benchmark_laplacian",
    "benchmark_floyd_warshall",
    "benchmark_mmd",
    "benchmark_geometric_algebra",
    "mega_scale_demo",
    "print_massacre_report",
]
