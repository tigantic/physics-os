# Module `benchmarks.profiler`

TensorRT inference profiler for detailed performance analysis.

This module provides profiling utilities to analyze TensorRT
inference at the layer and operation level.

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `LayerProfile`

Profile for a single layer.

#### Attributes

- **name** (`<class 'str'>`): 
- **layer_type** (`<class 'str'>`): 
- **operation_type** (`<enum 'OperationType'>`): 
- **total_time_ms** (`<class 'float'>`): 
- **avg_time_ms** (`<class 'float'>`): 
- **min_time_ms** (`<class 'float'>`): 
- **max_time_ms** (`<class 'float'>`): 
- **percentage** (`<class 'float'>`): 
- **input_size_bytes** (`<class 'int'>`): 
- **output_size_bytes** (`<class 'int'>`): 
- **weight_size_bytes** (`<class 'int'>`): 
- **workspace_bytes** (`<class 'int'>`): 
- **input_shapes** (`typing.List[typing.Tuple[int, ...]]`): 
- **output_shapes** (`typing.List[typing.Tuple[int, ...]]`): 
- **flops** (`<class 'int'>`): 
- **memory_bandwidth_gbps** (`<class 'float'>`): 
- **compute_utilization** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, name: str, layer_type: str, operation_type: profiler.OperationType, total_time_ms: float = 0.0, avg_time_ms: float = 0.0, min_time_ms: float = 0.0, max_time_ms: float = 0.0, percentage: float = 0.0, input_size_bytes: int = 0, output_size_bytes: int = 0, weight_size_bytes: int = 0, workspace_bytes: int = 0, input_shapes: List[Tuple[int, ...]] = <factory>, output_shapes: List[Tuple[int, ...]] = <factory>, flops: int = 0, memory_bandwidth_gbps: float = 0.0, compute_utilization: float = 0.0) -> None
```

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\profiler.py:86](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\profiler.py#L86)*

### class `LayerTimer`

Timer for individual layer profiling.

#### Methods

##### `__init__`

```python
def __init__(self, use_cuda: bool = True)
```

Initialize timer.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\profiler.py:179](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\profiler.py#L179)*

##### `get_stats`

```python
def get_stats(self) -> Dict[str, float]
```

Get timing statistics.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\profiler.py:207](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\profiler.py#L207)*

##### `start`

```python
def start(self)
```

Start timing.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\profiler.py:188](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\profiler.py#L188)*

##### `stop`

```python
def stop(self)
```

Stop timing and record measurement.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\profiler.py:195](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\profiler.py#L195)*

### class `OperationProfile`

Profile for a specific operation type.

#### Attributes

- **operation_type** (`<enum 'OperationType'>`): 
- **count** (`<class 'int'>`): 
- **total_time_ms** (`<class 'float'>`): 
- **avg_time_ms** (`<class 'float'>`): 
- **percentage** (`<class 'float'>`): 
- **total_flops** (`<class 'int'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, operation_type: profiler.OperationType, count: int = 0, total_time_ms: float = 0.0, avg_time_ms: float = 0.0, percentage: float = 0.0, total_flops: int = 0) -> None
```

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\profiler.py:111](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\profiler.py#L111)*

### class `OperationType`(Enum)

Types of neural network operations.

### class `ProfileConfig`

Configuration for profiling.

#### Attributes

- **warmup_runs** (`<class 'int'>`): 
- **profile_runs** (`<class 'int'>`): 
- **layer_level** (`<class 'bool'>`): 
- **operation_level** (`<class 'bool'>`): 
- **memory_tracking** (`<class 'bool'>`): 
- **timeline_export** (`<class 'bool'>`): 
- **use_cuda_events** (`<class 'bool'>`): 
- **record_shapes** (`<class 'bool'>`): 
- **with_stack** (`<class 'bool'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, warmup_runs: int = 5, profile_runs: int = 20, layer_level: bool = True, operation_level: bool = True, memory_tracking: bool = True, timeline_export: bool = False, use_cuda_events: bool = True, record_shapes: bool = True, with_stack: bool = False) -> None
```

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\profiler.py:46](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\profiler.py#L46)*

### class `ProfileResult`

Complete profiling result.

#### Attributes

- **model_name** (`<class 'str'>`): 
- **total_time_ms** (`<class 'float'>`): 
- **layer_profiles** (`typing.List[profiler.LayerProfile]`): 
- **operation_profiles** (`typing.Dict[profiler.OperationType, profiler.OperationProfile]`): 
- **total_memory_bytes** (`<class 'int'>`): 
- **peak_memory_bytes** (`<class 'int'>`): 
- **device_info** (`typing.Dict[str, typing.Any]`): 
- **config** (`typing.Optional[profiler.ProfileConfig]`): 

#### Methods

##### `__init__`

```python
def __init__(self, model_name: str, total_time_ms: float, layer_profiles: List[profiler.LayerProfile] = <factory>, operation_profiles: Dict[profiler.OperationType, profiler.OperationProfile] = <factory>, total_memory_bytes: int = 0, peak_memory_bytes: int = 0, device_info: Dict[str, Any] = <factory>, config: Optional[profiler.ProfileConfig] = None) -> None
```

##### `get_operation_breakdown`

```python
def get_operation_breakdown(self) -> Dict[str, float]
```

Get time breakdown by operation type.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\profiler.py:153](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\profiler.py#L153)*

##### `get_top_layers`

```python
def get_top_layers(self, n: int = 10) -> List[profiler.LayerProfile]
```

Get top N layers by time.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\profiler.py:145](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\profiler.py#L145)*

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\profiler.py:160](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\profiler.py#L160)*

### class `TensorRTProfiler`

Profiler for TensorRT inference.

Provides detailed layer-by-layer and operation-level profiling.

#### Methods

##### `__init__`

```python
def __init__(self, config: Optional[profiler.ProfileConfig] = None)
```

Initialize profiler.

**Parameters:**

- **config** (`typing.Optional[profiler.ProfileConfig]`): Profiling configuration

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\profiler.py:227](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\profiler.py#L227)*

##### `profile`

```python
def profile(self, model: torch.nn.modules.module.Module, input_tensor: torch.Tensor) -> profiler.ProfileResult
```

Profile model inference.

**Parameters:**

- **model** (`<class 'torch.nn.modules.module.Module'>`): Model to profile
- **input_tensor** (`<class 'torch.Tensor'>`): Input tensor

**Returns**: `<class 'profiler.ProfileResult'>` - ProfileResult with detailed analysis

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\profiler.py:239](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\profiler.py#L239)*

## Functions

### `profile_inference`

```python
def profile_inference(model: torch.nn.modules.module.Module, input_tensor: torch.Tensor, num_runs: int = 20) -> Dict[str, Any]
```

Quick inference profiling.

**Parameters:**

- **model** (`<class 'torch.nn.modules.module.Module'>`): Model to profile
- **input_tensor** (`<class 'torch.Tensor'>`): Input tensor
- **num_runs** (`<class 'int'>`): Number of profiling runs

**Returns**: `typing.Dict[str, typing.Any]` - Dictionary with profiling results

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\profiler.py:516](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\profiler.py#L516)*

### `profile_model`

```python
def profile_model(model: torch.nn.modules.module.Module, input_shape: Tuple[int, ...], config: Optional[profiler.ProfileConfig] = None) -> profiler.ProfileResult
```

Profile a model with given input shape.

**Parameters:**

- **model** (`<class 'torch.nn.modules.module.Module'>`): Model to profile
- **input_shape** (`typing.Tuple[int, ...]`): Input tensor shape
- **config** (`typing.Optional[profiler.ProfileConfig]`): Profiling configuration

**Returns**: `<class 'profiler.ProfileResult'>` - ProfileResult with detailed analysis

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\profiler.py:491](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\benchmarks\profiler.py#L491)*
