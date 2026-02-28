"""
TensorRT benchmark suite for comprehensive performance evaluation.

This module provides benchmarking tools for evaluating TensorRT
inference performance across different configurations.
"""

import gc
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch


class PrecisionMode(Enum):
    """TensorRT precision modes."""

    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    TF32 = "tf32"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    # Run parameters
    warmup_runs: int = 10
    benchmark_runs: int = 100
    batch_sizes: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    precision_modes: list[PrecisionMode] = field(
        default_factory=lambda: [PrecisionMode.FP32, PrecisionMode.FP16]
    )

    # Input configuration
    input_shape: tuple[int, ...] = (1, 3, 224, 224)
    input_dtype: torch.dtype = torch.float32

    # Measurement options
    measure_latency: bool = True
    measure_throughput: bool = True
    measure_memory: bool = True
    measure_accuracy: bool = False

    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_id: int = 0

    # Options
    sync_cuda: bool = True
    collect_gc: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "warmup_runs": self.warmup_runs,
            "benchmark_runs": self.benchmark_runs,
            "batch_sizes": self.batch_sizes,
            "precision_modes": [p.value for p in self.precision_modes],
            "input_shape": self.input_shape,
            "device": self.device,
            "gpu_id": self.gpu_id,
        }


@dataclass
class LatencyStats:
    """Latency statistics."""

    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float

    @classmethod
    def from_measurements(cls, measurements_ms: list[float]) -> "LatencyStats":
        """Create from measurements."""
        if not measurements_ms:
            return cls(0, 0, 0, 0, 0, 0, 0, 0)

        sorted_ms = sorted(measurements_ms)
        n = len(sorted_ms)

        return cls(
            mean_ms=statistics.mean(measurements_ms),
            std_ms=statistics.stdev(measurements_ms) if n > 1 else 0,
            min_ms=min(measurements_ms),
            max_ms=max(measurements_ms),
            p50_ms=sorted_ms[int(n * 0.50)],
            p90_ms=sorted_ms[int(n * 0.90)],
            p95_ms=sorted_ms[int(n * 0.95)],
            p99_ms=sorted_ms[min(int(n * 0.99), n - 1)],
        )

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "mean_ms": round(self.mean_ms, 4),
            "std_ms": round(self.std_ms, 4),
            "min_ms": round(self.min_ms, 4),
            "max_ms": round(self.max_ms, 4),
            "p50_ms": round(self.p50_ms, 4),
            "p90_ms": round(self.p90_ms, 4),
            "p95_ms": round(self.p95_ms, 4),
            "p99_ms": round(self.p99_ms, 4),
        }


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    peak_memory_mb: float
    allocated_memory_mb: float
    reserved_memory_mb: float
    model_size_mb: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "allocated_memory_mb": round(self.allocated_memory_mb, 2),
            "reserved_memory_mb": round(self.reserved_memory_mb, 2),
            "model_size_mb": round(self.model_size_mb, 2),
        }


@dataclass
class ThroughputStats:
    """Throughput statistics."""

    samples_per_second: float
    batches_per_second: float
    effective_batch_size: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "samples_per_second": round(self.samples_per_second, 2),
            "batches_per_second": round(self.batches_per_second, 2),
            "effective_batch_size": self.effective_batch_size,
        }


@dataclass
class AccuracyStats:
    """Accuracy comparison statistics."""

    max_absolute_error: float
    mean_absolute_error: float
    max_relative_error: float
    mean_relative_error: float
    cosine_similarity: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "max_absolute_error": self.max_absolute_error,
            "mean_absolute_error": self.mean_absolute_error,
            "max_relative_error": self.max_relative_error,
            "mean_relative_error": self.mean_relative_error,
            "cosine_similarity": self.cosine_similarity,
        }


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""

    name: str
    precision: PrecisionMode
    batch_size: int

    # Statistics
    latency: LatencyStats | None = None
    throughput: ThroughputStats | None = None
    memory: MemoryStats | None = None
    accuracy: AccuracyStats | None = None

    # Metadata
    config: BenchmarkConfig | None = None
    timestamp: float = field(default_factory=time.time)
    device_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "precision": self.precision.value,
            "batch_size": self.batch_size,
            "timestamp": self.timestamp,
            "device_info": self.device_info,
        }

        if self.latency:
            result["latency"] = self.latency.to_dict()
        if self.throughput:
            result["throughput"] = self.throughput.to_dict()
        if self.memory:
            result["memory"] = self.memory.to_dict()
        if self.accuracy:
            result["accuracy"] = self.accuracy.to_dict()

        return result


class LatencyBenchmark:
    """Benchmark for measuring inference latency."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark."""
        self.config = config

    def run(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
    ) -> LatencyStats:
        """
        Run latency benchmark.

        Args:
            model: Model to benchmark
            input_tensor: Input tensor

        Returns:
            LatencyStats with measurements
        """
        # Warmup
        for _ in range(self.config.warmup_runs):
            with torch.no_grad():
                _ = model(input_tensor)
            if self.config.sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()

        # Benchmark
        measurements = []
        for _ in range(self.config.benchmark_runs):
            if self.config.sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()
            with torch.no_grad():
                _ = model(input_tensor)

            if self.config.sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            measurements.append(elapsed)

        return LatencyStats.from_measurements(measurements)


class ThroughputBenchmark:
    """Benchmark for measuring inference throughput."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark."""
        self.config = config

    def run(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        duration_seconds: float = 10.0,
    ) -> ThroughputStats:
        """
        Run throughput benchmark.

        Args:
            model: Model to benchmark
            input_tensor: Input tensor
            duration_seconds: Duration to run

        Returns:
            ThroughputStats with measurements
        """
        batch_size = input_tensor.shape[0]

        # Warmup
        for _ in range(self.config.warmup_runs):
            with torch.no_grad():
                _ = model(input_tensor)
            if self.config.sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()

        # Measure throughput
        if self.config.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        batches = 0

        while time.perf_counter() - start < duration_seconds:
            with torch.no_grad():
                _ = model(input_tensor)
            batches += 1

        if self.config.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        return ThroughputStats(
            samples_per_second=batches * batch_size / elapsed,
            batches_per_second=batches / elapsed,
            effective_batch_size=batch_size,
        )


class MemoryBenchmark:
    """Benchmark for measuring memory usage."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark."""
        self.config = config

    def run(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
    ) -> MemoryStats:
        """
        Run memory benchmark.

        Args:
            model: Model to benchmark
            input_tensor: Input tensor

        Returns:
            MemoryStats with measurements
        """
        if not torch.cuda.is_available():
            # CPU memory estimation
            model_size = sum(
                p.numel() * p.element_size() for p in model.parameters()
            ) / (1024 * 1024)

            return MemoryStats(
                peak_memory_mb=0,
                allocated_memory_mb=0,
                reserved_memory_mb=0,
                model_size_mb=model_size,
            )

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        if self.config.collect_gc:
            gc.collect()

        # Measure model size
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (
            1024 * 1024
        )

        # Run inference to measure peak memory
        for _ in range(self.config.warmup_runs):
            with torch.no_grad():
                _ = model(input_tensor)
            torch.cuda.synchronize()

        # Capture memory stats
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)

        return MemoryStats(
            peak_memory_mb=peak_memory,
            allocated_memory_mb=allocated,
            reserved_memory_mb=reserved,
            model_size_mb=model_size,
        )


class AccuracyBenchmark:
    """Benchmark for measuring accuracy differences."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark."""
        self.config = config

    def run(
        self,
        reference_model: torch.nn.Module,
        optimized_model: torch.nn.Module,
        input_tensor: torch.Tensor,
    ) -> AccuracyStats:
        """
        Compare accuracy between reference and optimized models.

        Args:
            reference_model: Reference (FP32) model
            optimized_model: Optimized model
            input_tensor: Input tensor

        Returns:
            AccuracyStats with comparison
        """
        with torch.no_grad():
            ref_output = reference_model(input_tensor)
            opt_output = optimized_model(input_tensor)

        # D-005 FIX: Use torch ops for analysis (offline benchmark, acceptable)
        ref = ref_output.cpu().flatten()
        opt = opt_output.cpu().flatten()

        # Calculate errors using torch
        abs_error = torch.abs(ref - opt)
        rel_error = abs_error / (torch.abs(ref) + 1e-10)

        # Cosine similarity using torch
        cos_sim = torch.dot(ref, opt) / (torch.norm(ref) * torch.norm(opt) + 1e-10)

        return AccuracyStats(
            max_absolute_error=float(abs_error.max().item()),
            mean_absolute_error=float(abs_error.mean().item()),
            max_relative_error=float(rel_error.max().item()),
            mean_relative_error=float(rel_error.mean().item()),
            cosine_similarity=float(cos_sim.item()),
        )


class PrecisionBenchmark:
    """Benchmark comparing different precision modes."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark."""
        self.config = config
        self._latency_bench = LatencyBenchmark(config)
        self._memory_bench = MemoryBenchmark(config)
        self._accuracy_bench = AccuracyBenchmark(config)

    def run(
        self,
        model_factory: Callable[[PrecisionMode], torch.nn.Module],
        input_shape: tuple[int, ...],
    ) -> dict[PrecisionMode, BenchmarkResult]:
        """
        Run precision comparison benchmark.

        Args:
            model_factory: Function to create model for precision mode
            input_shape: Input tensor shape

        Returns:
            Dictionary of precision mode to results
        """
        results = {}
        reference_model = None

        for precision in self.config.precision_modes:
            # Create model
            model = model_factory(precision)

            if precision == PrecisionMode.FP32:
                reference_model = model

            # Create input tensor
            dtype = torch.float16 if precision == PrecisionMode.FP16 else torch.float32
            input_tensor = torch.randn(
                input_shape, dtype=dtype, device=self.config.device
            )

            # Run benchmarks
            latency = self._latency_bench.run(model, input_tensor)
            memory = self._memory_bench.run(model, input_tensor)

            accuracy = None
            if reference_model is not None and precision != PrecisionMode.FP32:
                # Compare to FP32 reference
                ref_input = input_tensor.float()
                accuracy = self._accuracy_bench.run(reference_model, model, ref_input)

            results[precision] = BenchmarkResult(
                name=f"precision_{precision.value}",
                precision=precision,
                batch_size=input_shape[0],
                latency=latency,
                memory=memory,
                accuracy=accuracy,
                config=self.config,
            )

        return results


class TensorRTBenchmarkSuite:
    """
    Complete TensorRT benchmark suite.

    Runs comprehensive benchmarks across precision modes,
    batch sizes, and optimization configurations.
    """

    def __init__(self, config: BenchmarkConfig | None = None):
        """
        Initialize benchmark suite.

        Args:
            config: Benchmark configuration
        """
        self.config = config or BenchmarkConfig()
        self.results: list[BenchmarkResult] = []

    def run_latency_sweep(
        self,
        model: torch.nn.Module,
        input_shape_base: tuple[int, ...],
    ) -> list[BenchmarkResult]:
        """
        Run latency benchmarks across batch sizes.

        Args:
            model: Model to benchmark
            input_shape_base: Base input shape (batch dim will be modified)

        Returns:
            List of benchmark results
        """
        results = []
        latency_bench = LatencyBenchmark(self.config)
        memory_bench = MemoryBenchmark(self.config)

        for batch_size in self.config.batch_sizes:
            # Create input with batch size
            input_shape = (batch_size,) + input_shape_base[1:]
            input_tensor = torch.randn(
                input_shape,
                dtype=self.config.input_dtype,
                device=self.config.device,
            )

            latency = latency_bench.run(model, input_tensor)
            memory = memory_bench.run(model, input_tensor)

            throughput = ThroughputStats(
                samples_per_second=batch_size / (latency.mean_ms / 1000),
                batches_per_second=1 / (latency.mean_ms / 1000),
                effective_batch_size=batch_size,
            )

            result = BenchmarkResult(
                name=f"latency_sweep_bs{batch_size}",
                precision=PrecisionMode.FP32,
                batch_size=batch_size,
                latency=latency,
                throughput=throughput,
                memory=memory,
                config=self.config,
            )
            results.append(result)

        self.results.extend(results)
        return results

    def run_precision_comparison(
        self,
        model_factory: Callable[[PrecisionMode], torch.nn.Module],
        input_shape: tuple[int, ...],
    ) -> dict[PrecisionMode, BenchmarkResult]:
        """
        Compare precision modes.

        Args:
            model_factory: Factory function for models
            input_shape: Input tensor shape

        Returns:
            Dictionary of results by precision
        """
        precision_bench = PrecisionBenchmark(self.config)
        results = precision_bench.run(model_factory, input_shape)

        self.results.extend(results.values())
        return results

    def run_full_suite(
        self,
        model_factory: Callable[[PrecisionMode], torch.nn.Module],
        input_shape_base: tuple[int, ...],
    ) -> list[BenchmarkResult]:
        """
        Run full benchmark suite.

        Args:
            model_factory: Factory function for models
            input_shape_base: Base input shape

        Returns:
            List of all benchmark results
        """
        all_results = []

        for precision in self.config.precision_modes:
            model = model_factory(precision)

            for batch_size in self.config.batch_sizes:
                input_shape = (batch_size,) + input_shape_base[1:]
                dtype = (
                    torch.float16 if precision == PrecisionMode.FP16 else torch.float32
                )
                input_tensor = torch.randn(
                    input_shape,
                    dtype=dtype,
                    device=self.config.device,
                )

                # Run all benchmarks
                latency_bench = LatencyBenchmark(self.config)
                memory_bench = MemoryBenchmark(self.config)
                throughput_bench = ThroughputBenchmark(self.config)

                latency = latency_bench.run(model, input_tensor)
                memory = memory_bench.run(model, input_tensor)
                throughput = throughput_bench.run(
                    model, input_tensor, duration_seconds=5.0
                )

                result = BenchmarkResult(
                    name=f"full_{precision.value}_bs{batch_size}",
                    precision=precision,
                    batch_size=batch_size,
                    latency=latency,
                    throughput=throughput,
                    memory=memory,
                    config=self.config,
                    device_info=self._get_device_info(),
                )
                all_results.append(result)

        self.results = all_results
        return all_results

    def _get_device_info(self) -> dict[str, Any]:
        """Get device information."""
        info = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            info.update(
                {
                    "cuda_version": torch.version.cuda,
                    "device_name": torch.cuda.get_device_name(self.config.gpu_id),
                    "device_capability": torch.cuda.get_device_capability(
                        self.config.gpu_id
                    ),
                    "device_count": torch.cuda.device_count(),
                }
            )

        return info

    def get_summary(self) -> dict[str, Any]:
        """Get benchmark summary."""
        if not self.results:
            return {"error": "No results available"}

        # Group by precision
        by_precision: dict[str, list[BenchmarkResult]] = {}
        for result in self.results:
            key = result.precision.value
            if key not in by_precision:
                by_precision[key] = []
            by_precision[key].append(result)

        summary = {
            "total_benchmarks": len(self.results),
            "config": self.config.to_dict(),
            "by_precision": {},
        }

        for precision, results in by_precision.items():
            latencies = [r.latency.mean_ms for r in results if r.latency]
            summary["by_precision"][precision] = {
                "count": len(results),
                "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
                "min_latency_ms": min(latencies) if latencies else 0,
                "max_latency_ms": max(latencies) if latencies else 0,
            }

        return summary


def run_tensorrt_benchmarks(
    model: torch.nn.Module,
    input_shape: tuple[int, ...],
    config: BenchmarkConfig | None = None,
) -> list[BenchmarkResult]:
    """
    Run TensorRT benchmarks on a model.

    Args:
        model: Model to benchmark
        input_shape: Input tensor shape
        config: Benchmark configuration

    Returns:
        List of benchmark results
    """
    config = config or BenchmarkConfig()
    suite = TensorRTBenchmarkSuite(config)

    # Simple factory that returns the same model
    def model_factory(precision: PrecisionMode) -> torch.nn.Module:
        if precision == PrecisionMode.FP16:
            return model.half()
        return model

    return suite.run_full_suite(model_factory, input_shape)


def compare_precision_modes(
    model: torch.nn.Module,
    input_shape: tuple[int, ...],
) -> dict[str, Any]:
    """
    Compare model performance across precision modes.

    Args:
        model: Model to compare
        input_shape: Input tensor shape

    Returns:
        Comparison results
    """
    config = BenchmarkConfig(
        precision_modes=[PrecisionMode.FP32, PrecisionMode.FP16],
        batch_sizes=[1, 8, 32],
    )

    results = run_tensorrt_benchmarks(model, input_shape, config)

    # Calculate speedups
    fp32_results = [r for r in results if r.precision == PrecisionMode.FP32]
    fp16_results = [r for r in results if r.precision == PrecisionMode.FP16]

    comparison = {
        "fp32_avg_latency_ms": (
            sum(r.latency.mean_ms for r in fp32_results) / len(fp32_results)
            if fp32_results
            else 0
        ),
        "fp16_avg_latency_ms": (
            sum(r.latency.mean_ms for r in fp16_results) / len(fp16_results)
            if fp16_results
            else 0
        ),
        "speedup": 0,
        "results": [r.to_dict() for r in results],
    }

    if comparison["fp16_avg_latency_ms"] > 0:
        comparison["speedup"] = (
            comparison["fp32_avg_latency_ms"] / comparison["fp16_avg_latency_ms"]
        )

    return comparison
