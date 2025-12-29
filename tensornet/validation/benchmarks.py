"""
Benchmark Module for Project HyperTensor.

Provides performance benchmarking utilities including:
- Timing measurement with statistical analysis
- Memory tracking and profiling
- Scalability testing (weak/strong scaling)
- Benchmark suite management and comparison

These tools enable systematic performance evaluation across configurations.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Union, Tuple, Callable, Any
from abc import ABC, abstractmethod
from contextlib import contextmanager
import torch
import numpy as np
from pathlib import Path
import json
import time
import gc
import traceback


@dataclass
class BenchmarkConfig:
    """
    Configuration for benchmark execution.
    
    Attributes:
        warmup_runs: Number of warmup iterations (not timed)
        benchmark_runs: Number of timed iterations
        gc_collect: Force garbage collection between runs
        sync_cuda: Synchronize CUDA between runs
        timeout_seconds: Maximum time per benchmark
        memory_tracking: Enable memory tracking
    """
    warmup_runs: int = 3
    benchmark_runs: int = 10
    gc_collect: bool = True
    sync_cuda: bool = True
    timeout_seconds: float = 300.0
    memory_tracking: bool = True
    save_raw_timings: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'warmup_runs': self.warmup_runs,
            'benchmark_runs': self.benchmark_runs,
            'gc_collect': self.gc_collect,
            'sync_cuda': self.sync_cuda,
            'timeout_seconds': self.timeout_seconds,
            'memory_tracking': self.memory_tracking,
            'save_raw_timings': self.save_raw_timings,
        }


@dataclass
class BenchmarkResult:
    """
    Result from a benchmark run.
    
    Attributes:
        name: Benchmark name
        mean_time: Mean execution time in seconds
        std_time: Standard deviation of execution time
        min_time: Minimum execution time
        max_time: Maximum execution time
        n_runs: Number of runs completed
        raw_timings: Individual timings (if saved)
        memory_peak: Peak memory usage in bytes
        memory_allocated: Total memory allocated
        metadata: Additional benchmark metadata
    """
    name: str
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    n_runs: int
    raw_timings: Optional[List[float]] = None
    memory_peak: Optional[int] = None
    memory_allocated: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def median_time(self) -> float:
        """Compute median time if raw timings available."""
        if self.raw_timings:
            return float(np.median(self.raw_timings))
        return self.mean_time
    
    @property
    def throughput(self) -> Optional[float]:
        """Compute throughput if work size specified in metadata."""
        if 'work_size' in self.metadata:
            return self.metadata['work_size'] / self.mean_time
        return None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = {
            'name': self.name,
            'mean_time': self.mean_time,
            'std_time': self.std_time,
            'min_time': self.min_time,
            'max_time': self.max_time,
            'n_runs': self.n_runs,
            'memory_peak': self.memory_peak,
            'memory_allocated': self.memory_allocated,
            'metadata': self.metadata,
        }
        if self.raw_timings:
            result['raw_timings'] = self.raw_timings
        return result
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Benchmark: {self.name}",
            f"  Mean time: {self.mean_time*1000:.3f} ms ± {self.std_time*1000:.3f} ms",
            f"  Range: [{self.min_time*1000:.3f}, {self.max_time*1000:.3f}] ms",
            f"  Runs: {self.n_runs}",
        ]
        if self.memory_peak:
            lines.append(f"  Peak memory: {self.memory_peak / 1e6:.2f} MB")
        if self.throughput:
            lines.append(f"  Throughput: {self.throughput:.2e} ops/sec")
        return "\n".join(lines)


class TimerContext:
    """
    Context manager for precise timing.
    
    Uses high-resolution timer and optionally CUDA synchronization.
    
    Example:
        with TimerContext(sync_cuda=True) as timer:
            # code to time
        print(f"Elapsed: {timer.elapsed:.3f} s")
    """
    
    def __init__(self, sync_cuda: bool = True):
        """
        Initialize timer context.
        
        Args:
            sync_cuda: Synchronize CUDA before/after timing
        """
        self.sync_cuda = sync_cuda and torch.cuda.is_available()
        self.start_time = 0.0
        self.end_time = 0.0
        self.elapsed = 0.0
    
    def __enter__(self):
        if self.sync_cuda:
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        if self.sync_cuda:
            torch.cuda.synchronize()
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time


class PerformanceTimer:
    """
    Performance timer with statistical tracking.
    
    Tracks multiple runs and computes statistics.
    
    Example:
        timer = PerformanceTimer("my_function")
        for _ in range(10):
            with timer.time():
                my_function()
        print(timer.summary())
    """
    
    def __init__(self, name: str = "benchmark", sync_cuda: bool = True):
        """
        Initialize performance timer.
        
        Args:
            name: Timer name for reporting
            sync_cuda: Synchronize CUDA for timing
        """
        self.name = name
        self.sync_cuda = sync_cuda
        self.timings: List[float] = []
    
    @contextmanager
    def time(self):
        """Context manager for a single timing."""
        timer = TimerContext(sync_cuda=self.sync_cuda)
        with timer:
            yield timer
        self.timings.append(timer.elapsed)
    
    def reset(self):
        """Clear all recorded timings."""
        self.timings = []
    
    @property
    def mean(self) -> float:
        """Mean timing."""
        return float(np.mean(self.timings)) if self.timings else 0.0
    
    @property
    def std(self) -> float:
        """Standard deviation of timings."""
        return float(np.std(self.timings)) if len(self.timings) > 1 else 0.0
    
    @property
    def min(self) -> float:
        """Minimum timing."""
        return float(np.min(self.timings)) if self.timings else 0.0
    
    @property
    def max(self) -> float:
        """Maximum timing."""
        return float(np.max(self.timings)) if self.timings else 0.0
    
    def to_result(self, **metadata) -> BenchmarkResult:
        """Convert to BenchmarkResult."""
        return BenchmarkResult(
            name=self.name,
            mean_time=self.mean,
            std_time=self.std,
            min_time=self.min,
            max_time=self.max,
            n_runs=len(self.timings),
            raw_timings=list(self.timings),
            metadata=metadata,
        )
    
    def summary(self) -> str:
        """Generate summary string."""
        return self.to_result().summary()


@dataclass
class MemorySnapshot:
    """
    Snapshot of memory usage.
    
    Attributes:
        timestamp: When snapshot was taken
        cpu_allocated: CPU memory allocated in bytes
        cpu_reserved: CPU memory reserved
        gpu_allocated: GPU memory allocated
        gpu_reserved: GPU memory reserved
        gpu_cached: GPU cached memory
    """
    timestamp: float
    cpu_allocated: int = 0
    cpu_reserved: int = 0
    gpu_allocated: int = 0
    gpu_reserved: int = 0
    gpu_cached: int = 0
    
    @classmethod
    def capture(cls) -> 'MemorySnapshot':
        """Capture current memory state."""
        import sys
        
        # CPU memory (approximate via gc)
        gc.collect()
        
        snapshot = cls(timestamp=time.time())
        
        if torch.cuda.is_available():
            snapshot.gpu_allocated = torch.cuda.memory_allocated()
            snapshot.gpu_reserved = torch.cuda.memory_reserved()
            snapshot.gpu_cached = torch.cuda.memory_reserved()
        
        return snapshot
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'cpu_allocated': self.cpu_allocated,
            'gpu_allocated': self.gpu_allocated,
            'gpu_reserved': self.gpu_reserved,
        }


class MemoryTracker:
    """
    Track memory usage over time.
    
    Records memory snapshots during execution to identify
    peak usage and memory leaks.
    
    Example:
        tracker = MemoryTracker()
        tracker.start()
        # ... operations ...
        tracker.stop()
        print(f"Peak GPU: {tracker.peak_gpu / 1e9:.2f} GB")
    """
    
    def __init__(self, interval: float = 0.1):
        """
        Initialize memory tracker.
        
        Args:
            interval: Sampling interval in seconds
        """
        self.interval = interval
        self.snapshots: List[MemorySnapshot] = []
        self._running = False
        self._thread = None
    
    def start(self):
        """Start memory tracking."""
        self._running = True
        self.snapshots = [MemorySnapshot.capture()]
    
    def stop(self):
        """Stop memory tracking and take final snapshot."""
        self._running = False
        self.snapshots.append(MemorySnapshot.capture())
    
    def snapshot(self):
        """Take a manual snapshot."""
        self.snapshots.append(MemorySnapshot.capture())
    
    @property
    def peak_gpu(self) -> int:
        """Peak GPU memory allocated."""
        if not self.snapshots:
            return 0
        return max(s.gpu_allocated for s in self.snapshots)
    
    @property
    def peak_cpu(self) -> int:
        """Peak CPU memory allocated."""
        if not self.snapshots:
            return 0
        return max(s.cpu_allocated for s in self.snapshots)
    
    def report(self) -> Dict:
        """Generate memory usage report."""
        if not self.snapshots:
            return {}
        
        return {
            'n_snapshots': len(self.snapshots),
            'peak_gpu_mb': self.peak_gpu / 1e6,
            'peak_cpu_mb': self.peak_cpu / 1e6,
            'final_gpu_mb': self.snapshots[-1].gpu_allocated / 1e6,
            'duration_seconds': self.snapshots[-1].timestamp - self.snapshots[0].timestamp,
        }


class ScalabilityTest(ABC):
    """
    Abstract base class for scalability testing.
    
    Tests how performance scales with problem size or resources.
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[BenchmarkConfig] = None,
    ):
        """
        Initialize scalability test.
        
        Args:
            name: Test name
            config: Benchmark configuration
        """
        self.name = name
        self.config = config or BenchmarkConfig()
        self.results: List[BenchmarkResult] = []
    
    @abstractmethod
    def setup(self, scale: int) -> Any:
        """
        Set up test for given scale.
        
        Args:
            scale: Scale parameter (problem size, # processors, etc.)
            
        Returns:
            Setup data needed for run
        """
        pass
    
    @abstractmethod
    def run(self, setup_data: Any) -> None:
        """
        Run the benchmark at given scale.
        
        Args:
            setup_data: Data from setup()
        """
        pass
    
    def teardown(self, setup_data: Any) -> None:
        """
        Clean up after test (optional).
        
        Args:
            setup_data: Data from setup()
        """
        pass
    
    def execute(self, scales: List[int]) -> List[BenchmarkResult]:
        """
        Execute scalability test across scales.
        
        Args:
            scales: List of scale values to test
            
        Returns:
            List of BenchmarkResult for each scale
        """
        self.results = []
        
        for scale in scales:
            setup_data = self.setup(scale)
            
            timer = PerformanceTimer(f"{self.name}_scale{scale}")
            tracker = MemoryTracker()
            
            # Warmup runs
            for _ in range(self.config.warmup_runs):
                self.run(setup_data)
            
            # Timed runs
            tracker.start()
            for _ in range(self.config.benchmark_runs):
                if self.config.gc_collect:
                    gc.collect()
                with timer.time():
                    self.run(setup_data)
            tracker.stop()
            
            result = timer.to_result(
                scale=scale,
                memory_peak=tracker.peak_gpu,
            )
            result.memory_peak = tracker.peak_gpu
            self.results.append(result)
            
            self.teardown(setup_data)
        
        return self.results
    
    def analyze_scaling(self) -> Dict:
        """
        Analyze scaling behavior from results.
        
        Returns:
            Dictionary with scaling analysis
        """
        if len(self.results) < 2:
            return {}
        
        scales = [r.metadata.get('scale', i) for i, r in enumerate(self.results)]
        times = [r.mean_time for r in self.results]
        
        # Log-log fit for power law: T = a * N^b
        log_scales = np.log(scales)
        log_times = np.log(times)
        
        # Linear regression
        A = np.vstack([log_scales, np.ones(len(log_scales))]).T
        b, log_a = np.linalg.lstsq(A, log_times, rcond=None)[0]
        
        # Theoretical ideal scaling
        # Strong scaling: T = T0 / P (b = -1)
        # Weak scaling: T = T0 (b = 0)
        
        return {
            'exponent': float(b),
            'coefficient': float(np.exp(log_a)),
            'scales': scales,
            'times': times,
            'efficiency': [times[0] / (t * scales[i] / scales[0]) 
                          for i, t in enumerate(times)],
        }


class WeakScalingTest(ScalabilityTest):
    """
    Weak scaling test: problem size grows with resources.
    
    Ideal weak scaling maintains constant execution time as both
    problem size and resources increase proportionally.
    """
    
    def __init__(
        self,
        name: str,
        workload_generator: Callable[[int], Any],
        benchmark_fn: Callable[[Any], None],
        config: Optional[BenchmarkConfig] = None,
    ):
        """
        Initialize weak scaling test.
        
        Args:
            name: Test name
            workload_generator: Function(scale) -> workload
            benchmark_fn: Function(workload) -> None
            config: Benchmark configuration
        """
        super().__init__(name, config)
        self.workload_generator = workload_generator
        self.benchmark_fn = benchmark_fn
    
    def setup(self, scale: int) -> Any:
        """Generate workload for scale."""
        return self.workload_generator(scale)
    
    def run(self, setup_data: Any) -> None:
        """Run benchmark on workload."""
        self.benchmark_fn(setup_data)


class StrongScalingTest(ScalabilityTest):
    """
    Strong scaling test: fixed problem size, varying resources.
    
    Ideal strong scaling reduces execution time proportionally
    to the increase in resources.
    """
    
    def __init__(
        self,
        name: str,
        fixed_workload: Any,
        benchmark_fn: Callable[[Any, int], None],
        config: Optional[BenchmarkConfig] = None,
    ):
        """
        Initialize strong scaling test.
        
        Args:
            name: Test name
            fixed_workload: The workload (fixed across scales)
            benchmark_fn: Function(workload, n_workers) -> None
            config: Benchmark configuration
        """
        super().__init__(name, config)
        self.fixed_workload = fixed_workload
        self.benchmark_fn = benchmark_fn
    
    def setup(self, scale: int) -> Tuple[Any, int]:
        """Return workload and worker count."""
        return (self.fixed_workload, scale)
    
    def run(self, setup_data: Tuple[Any, int]) -> None:
        """Run benchmark with given worker count."""
        workload, n_workers = setup_data
        self.benchmark_fn(workload, n_workers)


@dataclass
class BenchmarkSuite:
    """
    Collection of benchmarks to run together.
    
    Manages multiple benchmarks with consistent configuration
    and generates comparative reports.
    """
    name: str
    config: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    benchmarks: Dict[str, Callable[[], None]] = field(default_factory=dict)
    results: Dict[str, BenchmarkResult] = field(default_factory=dict)
    
    def add(self, name: str, fn: Callable[[], None], **metadata):
        """
        Add a benchmark to the suite.
        
        Args:
            name: Benchmark name
            fn: Function to benchmark (no arguments)
            **metadata: Additional metadata for this benchmark
        """
        self.benchmarks[name] = (fn, metadata)
    
    def run_all(self, verbose: bool = True) -> Dict[str, BenchmarkResult]:
        """
        Run all benchmarks in the suite.
        
        Args:
            verbose: Print progress
            
        Returns:
            Dictionary mapping names to results
        """
        self.results = {}
        
        for name, (fn, metadata) in self.benchmarks.items():
            if verbose:
                print(f"Running: {name}...")
            
            try:
                result = run_benchmark(fn, name, self.config, **metadata)
                self.results[name] = result
                
                if verbose:
                    print(f"  {result.mean_time*1000:.2f} ms ± {result.std_time*1000:.2f} ms")
            except Exception as e:
                if verbose:
                    print(f"  FAILED: {e}")
        
        return self.results
    
    def report(self, format: str = "text") -> str:
        """
        Generate benchmark report.
        
        Args:
            format: Output format ("text", "markdown", "csv")
            
        Returns:
            Formatted report string
        """
        if format == "markdown":
            return self._report_markdown()
        elif format == "csv":
            return self._report_csv()
        else:
            return self._report_text()
    
    def _report_text(self) -> str:
        """Generate text report."""
        lines = [
            f"Benchmark Suite: {self.name}",
            "=" * 60,
            "",
        ]
        
        for name, result in sorted(self.results.items()):
            lines.append(result.summary())
            lines.append("")
        
        return "\n".join(lines)
    
    def _report_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# Benchmark Suite: {self.name}",
            "",
            "| Benchmark | Mean (ms) | Std (ms) | Min (ms) | Max (ms) | Peak Mem (MB) |",
            "|-----------|-----------|----------|----------|----------|---------------|",
        ]
        
        for name, result in sorted(self.results.items()):
            mem = result.memory_peak / 1e6 if result.memory_peak else 0
            lines.append(
                f"| {name} | {result.mean_time*1000:.2f} | "
                f"{result.std_time*1000:.2f} | {result.min_time*1000:.2f} | "
                f"{result.max_time*1000:.2f} | {mem:.1f} |"
            )
        
        return "\n".join(lines)
    
    def _report_csv(self) -> str:
        """Generate CSV report."""
        lines = ["name,mean_ms,std_ms,min_ms,max_ms,peak_mem_mb"]
        
        for name, result in sorted(self.results.items()):
            mem = result.memory_peak / 1e6 if result.memory_peak else 0
            lines.append(
                f"{name},{result.mean_time*1000:.4f},"
                f"{result.std_time*1000:.4f},{result.min_time*1000:.4f},"
                f"{result.max_time*1000:.4f},{mem:.2f}"
            )
        
        return "\n".join(lines)
    
    def save(self, filepath: Union[str, Path]):
        """Save results to JSON file."""
        data = {
            'name': self.name,
            'config': self.config.to_dict(),
            'results': {name: r.to_dict() for name, r in self.results.items()},
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


def run_benchmark(
    fn: Callable[[], Any],
    name: str = "benchmark",
    config: Optional[BenchmarkConfig] = None,
    **metadata,
) -> BenchmarkResult:
    """
    Run a single benchmark with given configuration.
    
    Args:
        fn: Function to benchmark (no arguments)
        name: Benchmark name
        config: Benchmark configuration
        **metadata: Additional metadata
        
    Returns:
        BenchmarkResult with timing statistics
    """
    config = config or BenchmarkConfig()
    timer = PerformanceTimer(name, sync_cuda=config.sync_cuda)
    tracker = MemoryTracker() if config.memory_tracking else None
    
    # Warmup
    for _ in range(config.warmup_runs):
        fn()
    
    # Benchmark runs
    if tracker:
        tracker.start()
    
    for _ in range(config.benchmark_runs):
        if config.gc_collect:
            gc.collect()
        with timer.time():
            fn()
    
    if tracker:
        tracker.stop()
    
    result = timer.to_result(**metadata)
    
    if tracker:
        result.memory_peak = tracker.peak_gpu
    
    if not config.save_raw_timings:
        result.raw_timings = None
    
    return result


def run_benchmark_suite(
    suite: BenchmarkSuite,
    output_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> Dict[str, BenchmarkResult]:
    """
    Run a benchmark suite and optionally save results.
    
    Args:
        suite: The benchmark suite to run
        output_path: Optional path to save results
        verbose: Print progress
        
    Returns:
        Dictionary of benchmark results
    """
    results = suite.run_all(verbose=verbose)
    
    if output_path:
        suite.save(output_path)
    
    return results


def compare_benchmarks(
    baseline: Dict[str, BenchmarkResult],
    current: Dict[str, BenchmarkResult],
    threshold: float = 0.1,
) -> Dict[str, Dict]:
    """
    Compare two sets of benchmark results.
    
    Args:
        baseline: Baseline benchmark results
        current: Current benchmark results
        threshold: Threshold for significant change (fraction)
        
    Returns:
        Comparison report with speedups and regressions
    """
    comparison = {}
    
    for name in set(baseline.keys()) & set(current.keys()):
        base = baseline[name]
        curr = current[name]
        
        speedup = base.mean_time / curr.mean_time if curr.mean_time > 0 else float('inf')
        change = (curr.mean_time - base.mean_time) / base.mean_time
        
        if abs(change) < threshold:
            status = "unchanged"
        elif change < 0:
            status = "improved"
        else:
            status = "regressed"
        
        comparison[name] = {
            'baseline_ms': base.mean_time * 1000,
            'current_ms': curr.mean_time * 1000,
            'speedup': speedup,
            'change_percent': change * 100,
            'status': status,
        }
    
    # Find benchmarks only in one set
    for name in set(baseline.keys()) - set(current.keys()):
        comparison[name] = {'status': 'removed'}
    
    for name in set(current.keys()) - set(baseline.keys()):
        comparison[name] = {'status': 'added', 'current_ms': current[name].mean_time * 1000}
    
    return comparison
