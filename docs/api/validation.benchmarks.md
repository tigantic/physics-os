# Module `validation.benchmarks`

Benchmark Module for Project HyperTensor.

Provides performance benchmarking utilities including:
- Timing measurement with statistical analysis
- Memory tracking and profiling
- Scalability testing (weak/strong scaling)
- Benchmark suite management and comparison

These tools enable systematic performance evaluation across configurations.

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `BenchmarkConfig`

Configuration for benchmark execution.

#### Attributes

- **warmup_runs** (`<class 'int'>`): 
- **benchmark_runs** (`<class 'int'>`): 
- **gc_collect** (`<class 'bool'>`): 
- **sync_cuda** (`<class 'bool'>`): 
- **timeout_seconds** (`<class 'float'>`): 
- **memory_tracking** (`<class 'bool'>`): 
- **save_raw_timings** (`<class 'bool'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, warmup_runs: int = 3, benchmark_runs: int = 10, gc_collect: bool = True, sync_cuda: bool = True, timeout_seconds: float = 300.0, memory_tracking: bool = True, save_raw_timings: bool = False) -> None
```

##### `to_dict`

```python
def to_dict(self) -> Dict
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:48](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L48)*

### class `BenchmarkResult`

Result from a benchmark run.

#### Attributes

- **name** (`<class 'str'>`): 
- **mean_time** (`<class 'float'>`): 
- **std_time** (`<class 'float'>`): 
- **min_time** (`<class 'float'>`): 
- **max_time** (`<class 'float'>`): 
- **n_runs** (`<class 'int'>`): 
- **raw_timings** (`typing.Optional[typing.List[float]]`): 
- **memory_peak** (`typing.Optional[int]`): 
- **memory_allocated** (`typing.Optional[int]`): 
- **metadata** (`typing.Dict[str, typing.Any]`): 

#### Properties

##### `median_time`

```python
def median_time(self) -> float
```

Compute median time if raw timings available.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:89](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L89)*

##### `throughput`

```python
def throughput(self) -> Optional[float]
```

Compute throughput if work size specified in metadata.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:96](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L96)*

#### Methods

##### `__init__`

```python
def __init__(self, name: str, mean_time: float, std_time: float, min_time: float, max_time: float, n_runs: int, raw_timings: Optional[List[float]] = None, memory_peak: Optional[int] = None, memory_allocated: Optional[int] = None, metadata: Dict[str, Any] = <factory>) -> None
```

##### `summary`

```python
def summary(self) -> str
```

Generate human-readable summary.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:120](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L120)*

##### `to_dict`

```python
def to_dict(self) -> Dict
```

Convert to dictionary for serialization.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:103](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L103)*

### class `BenchmarkSuite`

Collection of benchmarks to run together.

Manages multiple benchmarks with consistent configuration
and generates comparative reports.

#### Attributes

- **name** (`<class 'str'>`): 
- **config** (`<class 'benchmarks.BenchmarkConfig'>`): 
- **benchmarks** (`typing.Dict[str, typing.Callable[[], NoneType]]`): 
- **results** (`typing.Dict[str, benchmarks.BenchmarkResult]`): 

#### Methods

##### `__init__`

```python
def __init__(self, name: str, config: benchmarks.BenchmarkConfig = <factory>, benchmarks: Dict[str, Callable[[], NoneType]] = <factory>, results: Dict[str, benchmarks.BenchmarkResult] = <factory>) -> None
```

##### `add`

```python
def add(self, name: str, fn: Callable[[], NoneType], **metadata)
```

Add a benchmark to the suite.

**Parameters:**

- **name** (`<class 'str'>`): Benchmark name
- **fn** (`typing.Callable[[], NoneType]`): Function to benchmark (no arguments) **metadata: Additional metadata for this benchmark

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:584](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L584)*

##### `report`

```python
def report(self, format: str = 'text') -> str
```

Generate benchmark report.

**Parameters:**

- **format** (`<class 'str'>`): Output format ("text", "markdown", "csv")

**Returns**: `<class 'str'>` - Formatted report string

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:623](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L623)*

##### `run_all`

```python
def run_all(self, verbose: bool = True) -> Dict[str, benchmarks.BenchmarkResult]
```

Run all benchmarks in the suite.

**Parameters:**

- **verbose** (`<class 'bool'>`): Print progress

**Returns**: `typing.Dict[str, benchmarks.BenchmarkResult]` - Dictionary mapping names to results

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:595](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L595)*

##### `save`

```python
def save(self, filepath: Union[str, pathlib.Path])
```

Save results to JSON file.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:687](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L687)*

### class `MemorySnapshot`

Snapshot of memory usage.

#### Attributes

- **timestamp** (`<class 'float'>`): 
- **cpu_allocated** (`<class 'int'>`): 
- **cpu_reserved** (`<class 'int'>`): 
- **gpu_allocated** (`<class 'int'>`): 
- **gpu_reserved** (`<class 'int'>`): 
- **gpu_cached** (`<class 'int'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, timestamp: float, cpu_allocated: int = 0, cpu_reserved: int = 0, gpu_allocated: int = 0, gpu_reserved: int = 0, gpu_cached: int = 0) -> None
```

##### `capture`

```python
def capture() -> 'MemorySnapshot'
```

Capture current memory state.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:268](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L268)*

##### `to_dict`

```python
def to_dict(self) -> Dict
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:285](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L285)*

### class `MemoryTracker`

Track memory usage over time.

Records memory snapshots during execution to identify
peak usage and memory leaks.

#### Properties

##### `peak_cpu`

```python
def peak_cpu(self) -> int
```

Peak CPU memory allocated.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:343](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L343)*

##### `peak_gpu`

```python
def peak_gpu(self) -> int
```

Peak GPU memory allocated.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:336](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L336)*

#### Methods

##### `__init__`

```python
def __init__(self, interval: float = 0.1)
```

Initialize memory tracker.

**Parameters:**

- **interval** (`<class 'float'>`): Sampling interval in seconds

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:310](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L310)*

##### `report`

```python
def report(self) -> Dict
```

Generate memory usage report.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:350](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L350)*

##### `snapshot`

```python
def snapshot(self)
```

Take a manual snapshot.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:332](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L332)*

##### `start`

```python
def start(self)
```

Start memory tracking.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:322](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L322)*

##### `stop`

```python
def stop(self)
```

Stop memory tracking and take final snapshot.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:327](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L327)*

### class `PerformanceTimer`

Performance timer with statistical tracking.

Tracks multiple runs and computes statistics.

#### Properties

##### `max`

```python
def max(self) -> float
```

Maximum timing.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:225](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L225)*

##### `mean`

```python
def mean(self) -> float
```

Mean timing.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:210](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L210)*

##### `min`

```python
def min(self) -> float
```

Minimum timing.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:220](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L220)*

##### `std`

```python
def std(self) -> float
```

Standard deviation of timings.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:215](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L215)*

#### Methods

##### `__init__`

```python
def __init__(self, name: str = 'benchmark', sync_cuda: bool = True)
```

Initialize performance timer.

**Parameters:**

- **name** (`<class 'str'>`): Timer name for reporting
- **sync_cuda** (`<class 'bool'>`): Synchronize CUDA for timing

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:186](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L186)*

##### `reset`

```python
def reset(self)
```

Clear all recorded timings.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:206](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L206)*

##### `summary`

```python
def summary(self) -> str
```

Generate summary string.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:243](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L243)*

##### `time`

*@wraps*

```python
def time(self)
```

Context manager for a single timing.

*Source: [C:\Users\bradl\AppData\Local\Programs\Python\Python311\Lib\contextlib.py:198](C:\Users\bradl\AppData\Local\Programs\Python\Python311\Lib\contextlib.py#L198)*

##### `to_result`

```python
def to_result(self, **metadata) -> benchmarks.BenchmarkResult
```

Convert to BenchmarkResult.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:230](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L230)*

### class `ScalabilityTest`(ABC)

Abstract base class for scalability testing.

Tests how performance scales with problem size or resources.

#### Methods

##### `__init__`

```python
def __init__(self, name: str, config: Optional[benchmarks.BenchmarkConfig] = None)
```

Initialize scalability test.

**Parameters:**

- **name** (`<class 'str'>`): Test name
- **config** (`typing.Optional[benchmarks.BenchmarkConfig]`): Benchmark configuration

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:371](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L371)*

##### `analyze_scaling`

```python
def analyze_scaling(self) -> Dict
```

Analyze scaling behavior from results.

**Returns**: `typing.Dict` - Dictionary with scaling analysis

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:461](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L461)*

##### `execute`

```python
def execute(self, scales: List[int]) -> List[benchmarks.BenchmarkResult]
```

Execute scalability test across scales.

**Parameters:**

- **scales** (`typing.List[int]`): List of scale values to test

**Returns**: `typing.List[benchmarks.BenchmarkResult]` - List of BenchmarkResult for each scale

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:419](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L419)*

##### `run`

```python
def run(self, setup_data: Any) -> None
```

Run the benchmark at given scale.

**Parameters:**

- **setup_data** (`typing.Any`): Data from setup()

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:400](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L400)*

##### `setup`

```python
def setup(self, scale: int) -> Any
```

Set up test for given scale.

**Parameters:**

- **scale** (`<class 'int'>`): Scale parameter (problem size, # processors, etc.)

**Returns**: `typing.Any` - Setup data needed for run

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:387](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L387)*

##### `teardown`

```python
def teardown(self, setup_data: Any) -> None
```

Clean up after test (optional).

**Parameters:**

- **setup_data** (`typing.Any`): Data from setup()

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:410](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L410)*

### class `StrongScalingTest`(ScalabilityTest)

Strong scaling test: fixed problem size, varying resources.

Ideal strong scaling reduces execution time proportionally
to the increase in resources.

#### Methods

##### `__init__`

```python
def __init__(self, name: str, fixed_workload: Any, benchmark_fn: Callable[[Any, int], NoneType], config: Optional[benchmarks.BenchmarkConfig] = None)
```

Initialize strong scaling test.

**Parameters:**

- **name** (`<class 'str'>`): Test name
- **fixed_workload** (`typing.Any`): The workload (fixed across scales)
- **benchmark_fn** (`typing.Callable[[typing.Any, int], NoneType]`): Function(workload, n_workers) -> None
- **config** (`typing.Optional[benchmarks.BenchmarkConfig]`): Benchmark configuration

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:541](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L541)*

##### `run`

```python
def run(self, setup_data: Tuple[Any, int]) -> None
```

Run benchmark with given worker count.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:565](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L565)*

##### `setup`

```python
def setup(self, scale: int) -> Tuple[Any, int]
```

Return workload and worker count.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:561](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L561)*

### class `TimerContext`

Context manager for precise timing.

Uses high-resolution timer and optionally CUDA synchronization.

#### Methods

##### `__init__`

```python
def __init__(self, sync_cuda: bool = True)
```

Initialize timer context.

**Parameters:**

- **sync_cuda** (`<class 'bool'>`): Synchronize CUDA before/after timing

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:147](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L147)*

### class `WeakScalingTest`(ScalabilityTest)

Weak scaling test: problem size grows with resources.

Ideal weak scaling maintains constant execution time as both
problem size and resources increase proportionally.

#### Methods

##### `__init__`

```python
def __init__(self, name: str, workload_generator: Callable[[int], Any], benchmark_fn: Callable[[Any], NoneType], config: Optional[benchmarks.BenchmarkConfig] = None)
```

Initialize weak scaling test.

**Parameters:**

- **name** (`<class 'str'>`): Test name
- **workload_generator** (`typing.Callable[[int], typing.Any]`): Function(scale) -> workload
- **benchmark_fn** (`typing.Callable[[typing.Any], NoneType]`): Function(workload) -> None
- **config** (`typing.Optional[benchmarks.BenchmarkConfig]`): Benchmark configuration

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:504](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L504)*

##### `run`

```python
def run(self, setup_data: Any) -> None
```

Run benchmark on workload.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:528](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L528)*

##### `setup`

```python
def setup(self, scale: int) -> Any
```

Generate workload for scale.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:524](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L524)*

## Functions

### `compare_benchmarks`

```python
def compare_benchmarks(baseline: Dict[str, benchmarks.BenchmarkResult], current: Dict[str, benchmarks.BenchmarkResult], threshold: float = 0.1) -> Dict[str, Dict]
```

Compare two sets of benchmark results.

**Parameters:**

- **baseline** (`typing.Dict[str, benchmarks.BenchmarkResult]`): Baseline benchmark results
- **current** (`typing.Dict[str, benchmarks.BenchmarkResult]`): Current benchmark results
- **threshold** (`<class 'float'>`): Threshold for significant change (fraction)

**Returns**: `typing.Dict[str, typing.Dict]` - Comparison report with speedups and regressions

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:772](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L772)*

### `run_benchmark`

```python
def run_benchmark(fn: Callable[[], Any], name: str = 'benchmark', config: Optional[benchmarks.BenchmarkConfig] = None, **metadata) -> benchmarks.BenchmarkResult
```

Run a single benchmark with given configuration.

**Parameters:**

- **fn** (`typing.Callable[[], typing.Any]`): Function to benchmark (no arguments)
- **name** (`<class 'str'>`): Benchmark name
- **config** (`typing.Optional[benchmarks.BenchmarkConfig]`): Benchmark configuration **metadata: Additional metadata

**Returns**: `<class 'benchmarks.BenchmarkResult'>` - BenchmarkResult with timing statistics

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:698](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L698)*

### `run_benchmark_suite`

```python
def run_benchmark_suite(suite: benchmarks.BenchmarkSuite, output_path: Union[str, pathlib.Path, NoneType] = None, verbose: bool = True) -> Dict[str, benchmarks.BenchmarkResult]
```

Run a benchmark suite and optionally save results.

**Parameters:**

- **suite** (`<class 'benchmarks.BenchmarkSuite'>`): The benchmark suite to run
- **output_path** (`typing.Union[str, pathlib.Path, NoneType]`): Optional path to save results
- **verbose** (`<class 'bool'>`): Print progress

**Returns**: `typing.Dict[str, benchmarks.BenchmarkResult]` - Dictionary of benchmark results

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py:748](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\benchmarks.py#L748)*
