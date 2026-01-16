"""
FluidElite Benchmarks
=====================

Perplexity, memory, and throughput benchmarks for FluidElite.
"""

from fluidelite.benchmarks.wikitext import evaluate_perplexity, download_wikitext2
from fluidelite.benchmarks.memory_profile import profile_memory, memory_vs_length
from fluidelite.benchmarks.throughput import benchmark_throughput, throughput_vs_length

__all__ = [
    "evaluate_perplexity",
    "download_wikitext2",
    "profile_memory",
    "memory_vs_length",
    "benchmark_throughput",
    "throughput_vs_length",
]
