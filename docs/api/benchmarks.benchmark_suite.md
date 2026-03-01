# Module `benchmarks.benchmark_suite`

TensorRT benchmark suite for comprehensive performance evaluation.

This module provides benchmarking tools for evaluating TensorRT
inference performance across different configurations.

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `AccuracyBenchmark`

Benchmark for measuring accuracy differences.

#### Methods

##### `__init__`

```python
def __init__(self, config: benchmark_suite.BenchmarkConfig)
```

Initialize benchmark.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py:384](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py#L384)*

##### `run`

```python
def run(self, reference_model: torch.nn.modules.module.Module, optimized_model: torch.nn.modules.module.Module, input_tensor: torch.Tensor) -> benchmark_suite.AccuracyStats
```

Compare accuracy between reference and optimized models.

**Parameters:**

- **reference_model** (`<class 'torch.nn.modules.module.Module'>`): Reference (FP32) model
- **optimized_model** (`<class 'torch.nn.modules.module.Module'>`): Optimized model
- **input_tensor** (`<class 'torch.Tensor'>`): Input tensor

**Returns**: `<class 'benchmark_suite.AccuracyStats'>` - AccuracyStats with comparison

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py:388](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py#L388)*

### class `AccuracyStats`

Accuracy comparison statistics.

#### Attributes

- **max_absolute_error** (`<class 'float'>`): 
- **mean_absolute_error** (`<class 'float'>`): 
- **max_relative_error** (`<class 'float'>`): 
- **mean_relative_error** (`<class 'float'>`): 
- **cosine_similarity** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, max_absolute_error: float, mean_absolute_error: float, max_relative_error: float, mean_relative_error: float, cosine_similarity: float) -> None
```

##### `to_dict`

```python
def to_dict(self) -> Dict[str, float]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py:157](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py#L157)*

### class `BenchmarkConfig`

Configuration for benchmark runs.

#### Attributes

- **warmup_runs** (`<class 'int'>`): 
- **benchmark_runs** (`<class 'int'>`): 
- **batch_sizes** (`typing.List[int]`): 
- **precision_modes** (`typing.List[benchmark_suite.PrecisionMode]`): 
- **input_shape** (`typing.Tuple[int, ...]`): 
- **input_dtype** (`<class 'torch.dtype'>`): 
- **measure_latency** (`<class 'bool'>`): 
- **measure_throughput** (`<class 'bool'>`): 
- **measure_memory** (`<class 'bool'>`): 
- **measure_accuracy** (`<class 'bool'>`): 
- **device** (`<class 'str'>`): 
- **gpu_id** (`<class 'int'>`): 
- **sync_cuda** (`<class 'bool'>`): 
- **collect_gc** (`<class 'bool'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, warmup_runs: int = 10, benchmark_runs: int = 100, batch_sizes: List[int] = <factory>, precision_modes: List[benchmark_suite.PrecisionMode] = <factory>, input_shape: Tuple[int, ...] = (1, 3, 224, 224), input_dtype: torch.dtype = torch.float32, measure_latency: bool = True, measure_throughput: bool = True, measure_memory: bool = True, measure_accuracy: bool = False, device: str = 'cpu', gpu_id: int = 0, sync_cuda: bool = True, collect_gc: bool = True) -> None
```

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py:55](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py#L55)*

### class `BenchmarkResult`

Complete benchmark result.

#### Attributes

- **name** (`<class 'str'>`): 
- **precision** (`<enum 'PrecisionMode'>`): 
- **batch_size** (`<class 'int'>`): 
- **latency** (`typing.Optional[benchmark_suite.LatencyStats]`): 
- **throughput** (`typing.Optional[benchmark_suite.ThroughputStats]`): 
- **memory** (`typing.Optional[benchmark_suite.MemoryStats]`): 
- **accuracy** (`typing.Optional[benchmark_suite.AccuracyStats]`): 
- **config** (`typing.Optional[benchmark_suite.BenchmarkConfig]`): 
- **timestamp** (`<class 'float'>`): 
- **device_info** (`typing.Dict[str, typing.Any]`): 

#### Methods

##### `__init__`

```python
def __init__(self, name: str, precision: benchmark_suite.PrecisionMode, batch_size: int, latency: Optional[benchmark_suite.LatencyStats] = None, throughput: Optional[benchmark_suite.ThroughputStats] = None, memory: Optional[benchmark_suite.MemoryStats] = None, accuracy: Optional[benchmark_suite.AccuracyStats] = None, config: Optional[benchmark_suite.BenchmarkConfig] = None, timestamp: float = <factory>, device_info: Dict[str, Any] = <factory>) -> None
```

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py:186](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py#L186)*

### class `LatencyBenchmark`

Benchmark for measuring inference latency.

#### Methods

##### `__init__`

```python
def __init__(self, config: benchmark_suite.BenchmarkConfig)
```

Initialize benchmark.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py:211](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py#L211)*

##### `run`

```python
def run(self, model: torch.nn.modules.module.Module, input_tensor: torch.Tensor) -> benchmark_suite.LatencyStats
```

Run latency benchmark.

**Parameters:**

- **model** (`<class 'torch.nn.modules.module.Module'>`): Model to benchmark
- **input_tensor** (`<class 'torch.Tensor'>`): Input tensor

**Returns**: `<class 'benchmark_suite.LatencyStats'>` - LatencyStats with measurements

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py:215](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py#L215)*

### class `LatencyStats`

Latency statistics.

#### Attributes

- **mean_ms** (`<class 'float'>`): 
- **std_ms** (`<class 'float'>`): 
- **min_ms** (`<class 'float'>`): 
- **max_ms** (`<class 'float'>`): 
- **p50_ms** (`<class 'float'>`): 
- **p90_ms** (`<class 'float'>`): 
- **p95_ms** (`<class 'float'>`): 
- **p99_ms** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, mean_ms: float, std_ms: float, min_ms: float, max_ms: float, p50_ms: float, p90_ms: float, p95_ms: float, p99_ms: float) -> None
```

##### `from_measurements`

```python
def from_measurements(measurements_ms: List[float]) -> 'LatencyStats'
```

Create from measurements.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py:80](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py#L80)*

##### `to_dict`

```python
def to_dict(self) -> Dict[str, float]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py:100](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py#L100)*

### class `MemoryBenchmark`

Benchmark for measuring memory usage.

#### Methods

##### `__init__`

```python
def __init__(self, config: benchmark_suite.BenchmarkConfig)
```

Initialize benchmark.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py:316](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py#L316)*

##### `run`

```python
def run(self, model: torch.nn.modules.module.Module, input_tensor: torch.Tensor) -> benchmark_suite.MemoryStats
```

Run memory benchmark.

**Parameters:**

- **model** (`<class 'torch.nn.modules.module.Module'>`): Model to benchmark
- **input_tensor** (`<class 'torch.Tensor'>`): Input tensor

**Returns**: `<class 'benchmark_suite.MemoryStats'>` - MemoryStats with measurements

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py:320](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py#L320)*

### class `MemoryStats`

Memory usage statistics.

#### Attributes

- **peak_memory_mb** (`<class 'float'>`): 
- **allocated_memory_mb** (`<class 'float'>`): 
- **reserved_memory_mb** (`<class 'float'>`): 
- **model_size_mb** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, peak_memory_mb: float, allocated_memory_mb: float, reserved_memory_mb: float, model_size_mb: float) -> None
```

##### `to_dict`

```python
def to_dict(self) -> Dict[str, float]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py:122](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py#L122)*

### class `PrecisionBenchmark`

Benchmark comparing different precision modes.

#### Methods

##### `__init__`

```python
def __init__(self, config: benchmark_suite.BenchmarkConfig)
```

Initialize benchmark.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py:432](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py#L432)*

##### `run`

```python
def run(self, model_factory: Callable[[benchmark_suite.PrecisionMode], torch.nn.modules.module.Module], input_shape: Tuple[int, ...]) -> Dict[benchmark_suite.PrecisionMode, benchmark_suite.BenchmarkResult]
```

Run precision comparison benchmark.

**Parameters:**

- **model_factory** (`typing.Callable[[benchmark_suite.PrecisionMode], torch.nn.modules.module.Module]`): Function to create model for precision mode
- **input_shape** (`typing.Tuple[int, ...]`): Input tensor shape

**Returns**: `typing.Dict[benchmark_suite.PrecisionMode, benchmark_suite.BenchmarkResult]` - Dictionary of precision mode to results

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py:439](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py#L439)*

### class `PrecisionMode`(Enum)

TensorRT precision modes.

### class `TensorRTBenchmarkSuite`

Complete TensorRT benchmark suite.

Runs comprehensive benchmarks across precision modes,
batch sizes, and optimization configurations.

#### Methods

##### `__init__`

```python
def __init__(self, config: Optional[benchmark_suite.BenchmarkConfig] = None)
```

Initialize benchmark suite.

**Parameters:**

- **config** (`typing.Optional[benchmark_suite.BenchmarkConfig]`): Benchmark configuration

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py:499](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py#L499)*

##### `get_summary`

```python
def get_summary(self) -> Dict[str, Any]
```

Get benchmark summary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py:651](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py#L651)*

##### `run_full_suite`

```python
def run_full_suite(self, model_factory: Callable[[benchmark_suite.PrecisionMode], torch.nn.modules.module.Module], input_shape_base: Tuple[int, ...]) -> List[benchmark_suite.BenchmarkResult]
```

Run full benchmark suite.

**Parameters:**

- **model_factory** (`typing.Callable[[benchmark_suite.PrecisionMode], torch.nn.modules.module.Module]`): Factory function for models
- **input_shape_base** (`typing.Tuple[int, ...]`): Base input shape

**Returns**: `typing.List[benchmark_suite.BenchmarkResult]` - List of all benchmark results

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py:581](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py#L581)*

##### `run_latency_sweep`

```python
def run_latency_sweep(self, model: torch.nn.modules.module.Module, input_shape_base: Tuple[int, ...]) -> List[benchmark_suite.BenchmarkResult]
```

Run latency benchmarks across batch sizes.

**Parameters:**

- **model** (`<class 'torch.nn.modules.module.Module'>`): Model to benchmark
- **input_shape_base** (`typing.Tuple[int, ...]`): Base input shape (batch dim will be modified)

**Returns**: `typing.List[benchmark_suite.BenchmarkResult]` - List of benchmark results

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py:509](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py#L509)*

##### `run_precision_comparison`

```python
def run_precision_comparison(self, model_factory: Callable[[benchmark_suite.PrecisionMode], torch.nn.modules.module.Module], input_shape: Tuple[int, ...]) -> Dict[benchmark_suite.PrecisionMode, benchmark_suite.BenchmarkResult]
```

Compare precision modes.

**Parameters:**

- **model_factory** (`typing.Callable[[benchmark_suite.PrecisionMode], torch.nn.modules.module.Module]`): Factory function for models
- **input_shape** (`typing.Tuple[int, ...]`): Input tensor shape

**Returns**: `typing.Dict[benchmark_suite.PrecisionMode, benchmark_suite.BenchmarkResult]` - Dictionary of results by precision

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py:560](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py#L560)*

### class `ThroughputBenchmark`

Benchmark for measuring inference throughput.

#### Methods

##### `__init__`

```python
def __init__(self, config: benchmark_suite.BenchmarkConfig)
```

Initialize benchmark.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py:259](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py#L259)*

##### `run`

```python
def run(self, model: torch.nn.modules.module.Module, input_tensor: torch.Tensor, duration_seconds: float = 10.0) -> benchmark_suite.ThroughputStats
```

Run throughput benchmark.

**Parameters:**

- **model** (`<class 'torch.nn.modules.module.Module'>`): Model to benchmark
- **input_tensor** (`<class 'torch.Tensor'>`): Input tensor
- **duration_seconds** (`<class 'float'>`): Duration to run

**Returns**: `<class 'benchmark_suite.ThroughputStats'>` - ThroughputStats with measurements

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py:263](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py#L263)*

### class `ThroughputStats`

Throughput statistics.

#### Attributes

- **samples_per_second** (`<class 'float'>`): 
- **batches_per_second** (`<class 'float'>`): 
- **effective_batch_size** (`<class 'int'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, samples_per_second: float, batches_per_second: float, effective_batch_size: int) -> None
```

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py:139](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py#L139)*

## Functions

### `compare_precision_modes`

```python
def compare_precision_modes(model: torch.nn.modules.module.Module, input_shape: Tuple[int, ...]) -> Dict[str, Any]
```

Compare model performance across precision modes.

**Parameters:**

- **model** (`<class 'torch.nn.modules.module.Module'>`): Model to compare
- **input_shape** (`typing.Tuple[int, ...]`): Input tensor shape

**Returns**: `typing.Dict[str, typing.Any]` - Comparison results

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py:710](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py#L710)*

### `run_tensorrt_benchmarks`

```python
def run_tensorrt_benchmarks(model: torch.nn.modules.module.Module, input_shape: Tuple[int, ...], config: Optional[benchmark_suite.BenchmarkConfig] = None) -> List[benchmark_suite.BenchmarkResult]
```

Run TensorRT benchmarks on a model.

**Parameters:**

- **model** (`<class 'torch.nn.modules.module.Module'>`): Model to benchmark
- **input_shape** (`typing.Tuple[int, ...]`): Input tensor shape
- **config** (`typing.Optional[benchmark_suite.BenchmarkConfig]`): Benchmark configuration

**Returns**: `typing.List[benchmark_suite.BenchmarkResult]` - List of benchmark results

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py:682](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\benchmarks\benchmark_suite.py#L682)*
