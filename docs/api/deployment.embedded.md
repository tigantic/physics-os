# Module `deployment.embedded`

Embedded Deployment Utilities =============================

Tools for deploying tensor network CFD models to embedded hardware,
with focus on NVIDIA Jetson platforms.

Target Platforms:
    - Jetson AGX Orin Industrial: 275 TOPS, 2048 CUDA, 64GB LPDDR5
    - Jetson Orin NX 16GB: 100 TOPS, 1024 CUDA
    - Jetson Orin Nano 8GB: 40 TOPS, 512 CUDA

Key Optimizations:
    - Power mode management (MAXN, 50W, 30W, 15W)
    - Memory-aware inference scheduling
    - Thermal throttling prevention
    - Real-time deadline guarantees

SWaP Constraints (Size, Weight, Power):
    - Volume: < 100 cm³ for module
    - Weight: < 200g
    - Power: 15W - 60W depending on mode

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `EmbeddedRuntime`

Runtime manager for embedded inference deployment.

Provides:
- Model loading and management
- Real-time inference scheduling
- Power and thermal management
- Performance monitoring

#### Methods

##### `__init__`

```python
def __init__(self, config: embedded.JetsonConfig = None)
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py:266](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py#L266)*

##### `get_metrics`

```python
def get_metrics(self) -> embedded.InferenceMetrics
```

Get current performance metrics.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py:430](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py#L430)*

##### `infer`

```python
def infer(self, model_name: str, inputs: Dict[str, torch.Tensor], check_deadline: bool = True) -> Dict[str, torch.Tensor]
```

Run inference with deadline checking.

**Parameters:**

- **model_name** (`<class 'str'>`): Name of loaded model
- **inputs** (`typing.Dict[str, torch.Tensor]`): Input tensors
- **check_deadline** (`<class 'bool'>`): Whether to check deadline

**Returns**: `typing.Dict[str, torch.Tensor]` - Output tensors

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py:372](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py#L372)*

##### `initialize`

```python
def initialize(self, pool_size_mb: int = 1024)
```

Initialize runtime for inference.

**Parameters:**

- **pool_size_mb** (`<class 'int'>`): Size of memory pool

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py:278](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py#L278)*

##### `load_model`

```python
def load_model(self, name: str, model_path: Union[str, pathlib.Path], warmup_iterations: int = 10)
```

Load and warm up a model.

**Parameters:**

- **name** (`<class 'str'>`): Model name for reference
- **model_path** (`typing.Union[str, pathlib.Path]`): Path to ONNX or TRT model
- **warmup_iterations** (`<class 'int'>`): Number of warmup inferences

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py:295](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py#L295)*

##### `shutdown`

```python
def shutdown(self)
```

Clean shutdown of runtime.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py:443](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py#L443)*

### class `InferenceMetrics`

Real-time inference performance metrics.

#### Attributes

- **latency_ms** (`<class 'float'>`): 
- **throughput_hz** (`<class 'float'>`): 
- **gpu_temp_c** (`<class 'float'>`): 
- **cpu_temp_c** (`<class 'float'>`): 
- **power_w** (`<class 'float'>`): 
- **memory_used_mb** (`<class 'float'>`): 
- **deadline_misses** (`<class 'int'>`): 
- **total_inferences** (`<class 'int'>`): 

#### Properties

##### `deadline_hit_rate`

```python
def deadline_hit_rate(self) -> float
```

Percentage of deadlines met.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py:106](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py#L106)*

#### Methods

##### `__init__`

```python
def __init__(self, latency_ms: float = 0.0, throughput_hz: float = 0.0, gpu_temp_c: float = 0.0, cpu_temp_c: float = 0.0, power_w: float = 0.0, memory_used_mb: float = 0.0, deadline_misses: int = 0, total_inferences: int = 0) -> None
```

### class `JetsonConfig`

Configuration for Jetson deployment.

#### Attributes

- **power_mode** (`<enum 'PowerMode'>`): 
- **max_gpu_freq_mhz** (`<class 'int'>`): 
- **max_cpu_freq_mhz** (`<class 'int'>`): 
- **memory_growth** (`<class 'bool'>`): 
- **enable_dla** (`<class 'bool'>`): 
- **dla_core** (`<class 'int'>`): 
- **enable_gpu** (`<class 'bool'>`): 
- **target_fps** (`<class 'float'>`): 
- **thermal_limit_c** (`<class 'float'>`): 
- **fan_speed_pct** (`<class 'int'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, power_mode: embedded.PowerMode = <PowerMode.MODE_30W: '30W'>, max_gpu_freq_mhz: int = 1300, max_cpu_freq_mhz: int = 2200, memory_growth: bool = True, enable_dla: bool = True, dla_core: int = 0, enable_gpu: bool = True, target_fps: float = 100.0, thermal_limit_c: float = 85.0, fan_speed_pct: int = 75) -> None
```

### class `MemoryPool`

Pre-allocated memory pool for deterministic allocation.

Eliminates allocation jitter during real-time inference.

#### Methods

##### `__init__`

```python
def __init__(self, pool_size_mb: int = 1024, dtype: torch.dtype = torch.float32, device: str = 'cuda')
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py:121](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py#L121)*

##### `allocate`

```python
def allocate(self, name: str, shape: Tuple[int, ...]) -> torch.Tensor
```

Allocate a tensor from the pool.

**Parameters:**

- **name** (`<class 'str'>`): Unique name for allocation
- **shape** (`typing.Tuple[int, ...]`): Shape of tensor to allocate

**Returns**: `<class 'torch.Tensor'>` - Pre-allocated tensor view

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py:143](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py#L143)*

##### `get_usage_mb`

```python
def get_usage_mb(self) -> float
```

Current memory usage in MB.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py:173](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py#L173)*

##### `reset`

```python
def reset(self)
```

Reset pool for new inference cycle.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py:167](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py#L167)*

### class `MemoryProfile`

Memory allocation profile for embedded systems.

#### Attributes

- **total_system_mb** (`<class 'int'>`): 
- **reserved_system_mb** (`<class 'int'>`): 
- **model_weights_mb** (`<class 'int'>`): 
- **inference_buffer_mb** (`<class 'int'>`): 
- **io_buffer_mb** (`<class 'int'>`): 
- **safety_margin_mb** (`<class 'int'>`): 

#### Properties

##### `available_mb`

```python
def available_mb(self) -> int
```

Available memory for application.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py:79](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py#L79)*

##### `utilization_pct`

```python
def utilization_pct(self) -> float
```

Memory utilization percentage.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py:86](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py#L86)*

#### Methods

##### `__init__`

```python
def __init__(self, total_system_mb: int = 64000, reserved_system_mb: int = 4000, model_weights_mb: int = 500, inference_buffer_mb: int = 1000, io_buffer_mb: int = 200, safety_margin_mb: int = 500) -> None
```

### class `PowerMode`(Enum)

Jetson power modes.

### class `ThermalMonitor`

Monitor and manage thermal state for Jetson.

Prevents thermal throttling by proactive management.

#### Methods

##### `__init__`

```python
def __init__(self, config: embedded.JetsonConfig)
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py:185](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py#L185)*

##### `get_recommended_power_mode`

```python
def get_recommended_power_mode(self) -> embedded.PowerMode
```

Recommend power mode based on thermal state.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py:225](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py#L225)*

##### `get_temperatures`

```python
def get_temperatures(self) -> Dict[str, float]
```

Read current temperatures.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py:192](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py#L192)*

##### `start_monitoring`

```python
def start_monitoring(self, interval_s: float = 1.0)
```

Start background thermal monitoring.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py:236](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py#L236)*

##### `stop_monitoring`

```python
def stop_monitoring(self)
```

Stop thermal monitoring.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py:248](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py#L248)*

##### `update_thermal_state`

```python
def update_thermal_state(self)
```

Update thermal state based on current temperatures.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py:209](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py#L209)*

### class `ThermalState`(Enum)

Thermal throttling states.

## Functions

### `configure_jetson_power`

```python
def configure_jetson_power(config: embedded.JetsonConfig)
```

Configure Jetson power mode.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py:453](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py#L453)*

### `create_inference_pipeline`

```python
def create_inference_pipeline(model_paths: Dict[str, pathlib.Path], config: embedded.JetsonConfig = None) -> embedded.EmbeddedRuntime
```

Create a complete inference pipeline for embedded deployment.

**Parameters:**

- **model_paths** (`typing.Dict[str, pathlib.Path]`): Dict mapping model names to paths
- **config** (`<class 'embedded.JetsonConfig'>`): Jetson configuration

**Returns**: `<class 'embedded.EmbeddedRuntime'>` - Configured EmbeddedRuntime

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py:518](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py#L518)*

### `optimize_memory_layout`

```python
def optimize_memory_layout(tensors: List[torch.Tensor], target_alignment: int = 256) -> List[torch.Tensor]
```

Optimize tensor memory layout for cache efficiency.

**Parameters:**

- **tensors** (`typing.List[torch.Tensor]`): List of tensors to optimize
- **target_alignment** (`<class 'int'>`): Byte alignment for cache lines

**Returns**: `typing.List[torch.Tensor]` - List of optimized tensors

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py:489](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py#L489)*

### `validate_embedded_module`

```python
def validate_embedded_module()
```

Validate embedded deployment module.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py:548](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\deployment\embedded.py#L548)*
