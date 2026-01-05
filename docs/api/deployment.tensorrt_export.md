# Module `deployment.tensorrt_export`

TensorRT Export Pipeline ========================

Export PyTorch tensor network models to TensorRT for high-performance
inference on NVIDIA Jetson and other GPU platforms.

Pipeline:
    1. PyTorch Model → ONNX (torch.onnx.export)
    2. ONNX → TensorRT Engine (trtexec or Python API)
    3. Validation against reference outputs
    4. Benchmarking with target hardware

Optimization Strategies:
    - FP16/INT8 quantization for Tensor Core acceleration
    - Layer fusion for reduced memory bandwidth
    - Dynamic batching for variable workloads
    - Sparsity pruning for reduced compute

Target Hardware:
    - Jetson AGX Orin: 275 TOPS INT8, 64 Tensor Cores
    - Jetson Orin NX: 100 TOPS INT8
    - Jetson Orin Nano: 40 TOPS INT8

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `BenchmarkResult`

Inference benchmark results.

#### Attributes

- **latency_ms** (`<class 'float'>`): 
- **throughput_samples_per_sec** (`<class 'float'>`): 
- **memory_mb** (`<class 'float'>`): 
- **gpu_utilization** (`<class 'float'>`): 
- **power_w** (`typing.Optional[float]`): 

#### Methods

##### `__init__`

```python
def __init__(self, latency_ms: float, throughput_samples_per_sec: float, memory_mb: float, gpu_utilization: float, power_w: Optional[float] = None) -> None
```

### class `CFDInferenceModule`(Module)

Wrapper module for CFD inference suitable for TensorRT export.

Encapsulates the tensor network CFD computation in a form
that can be traced and exported.

#### Methods

##### `__init__`

```python
def __init__(self, grid_shape: Tuple[int, ...], n_vars: int = 4, gamma: float = 1.4)
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\deployment\tensorrt_export.py:105](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\deployment\tensorrt_export.py#L105)*

##### `forward`

```python
def forward(self, state: torch.Tensor) -> torch.Tensor
```

Compute one Euler time step.

**Parameters:**

- **state** (`<class 'torch.Tensor'>`): Conservative variables (batch, n_vars, Nx, Ny)

**Returns**: `<class 'torch.Tensor'>` - Updated state after one time step

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\deployment\tensorrt_export.py:120](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\deployment\tensorrt_export.py#L120)*

### class `ExportConfig`

Configuration for model export.

#### Attributes

- **precision** (`<enum 'Precision'>`): 
- **optimization_level** (`<enum 'OptimizationLevel'>`): 
- **max_batch_size** (`<class 'int'>`): 
- **workspace_size_mb** (`<class 'int'>`): 
- **dynamic_axes** (`typing.Optional[typing.Dict[str, typing.Dict[int, str]]]`): 
- **input_names** (`typing.List[str]`): 
- **output_names** (`typing.List[str]`): 
- **opset_version** (`<class 'int'>`): 
- **enable_sparsity** (`<class 'bool'>`): 
- **enable_timing_cache** (`<class 'bool'>`): 
- **calibration_data** (`typing.Optional[torch.Tensor]`): 

#### Methods

##### `__init__`

```python
def __init__(self, precision: tensorrt_export.Precision = <Precision.FP16: 'fp16'>, optimization_level: tensorrt_export.OptimizationLevel = <OptimizationLevel.O2: 2>, max_batch_size: int = 1, workspace_size_mb: int = 1024, dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None, input_names: List[str] = <factory>, output_names: List[str] = <factory>, opset_version: int = 17, enable_sparsity: bool = False, enable_timing_cache: bool = True, calibration_data: Optional[torch.Tensor] = None) -> None
```

### class `ExportResult`

Result from model export.

#### Attributes

- **onnx_path** (`typing.Optional[pathlib.Path]`): 
- **trt_engine_path** (`typing.Optional[pathlib.Path]`): 
- **input_shapes** (`typing.Dict[str, typing.Tuple[int, ...]]`): 
- **output_shapes** (`typing.Dict[str, typing.Tuple[int, ...]]`): 
- **precision** (`<enum 'Precision'>`): 
- **export_time_s** (`<class 'float'>`): 
- **model_size_mb** (`<class 'float'>`): 
- **validation_passed** (`<class 'bool'>`): 
- **max_error** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, onnx_path: Optional[pathlib.Path], trt_engine_path: Optional[pathlib.Path], input_shapes: Dict[str, Tuple[int, ...]], output_shapes: Dict[str, Tuple[int, ...]], precision: tensorrt_export.Precision, export_time_s: float, model_size_mb: float, validation_passed: bool, max_error: float) -> None
```

### class `OptimizationLevel`(Enum)

TensorRT optimization levels.

### class `Precision`(Enum)

Inference precision modes.

### class `TTContraction`(Module)

Tensor Train contraction as a neural network module.

Enables TensorRT optimization of TT operations.

#### Methods

##### `__init__`

```python
def __init__(self, cores: List[torch.Tensor])
```

Args:

cores: List of TT cores [G_1, G_2, ..., G_d]

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\deployment\tensorrt_export.py:175](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\deployment\tensorrt_export.py#L175)*

##### `forward`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Contract TT with input vector.

**Parameters:**

- **x** (`<class 'torch.Tensor'>`): Input vector (batch, input_dim)

**Returns**: `<class 'torch.Tensor'>` - Output vector (batch, output_dim)

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\deployment\tensorrt_export.py:186](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\deployment\tensorrt_export.py#L186)*

### class `TensorRTExporter`

High-level interface for exporting models to TensorRT.

#### Methods

##### `__init__`

```python
def __init__(self, config: tensorrt_export.ExportConfig = None)
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\deployment\tensorrt_export.py:483](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\deployment\tensorrt_export.py#L483)*

##### `export`

```python
def export(self, model: torch.nn.modules.module.Module, sample_input: torch.Tensor, name: str, output_dir: Union[str, pathlib.Path] = './exports') -> tensorrt_export.ExportResult
```

Export a model to ONNX and optionally TensorRT.

**Parameters:**

- **model** (`<class 'torch.nn.modules.module.Module'>`): PyTorch module
- **sample_input** (`<class 'torch.Tensor'>`): Example input tensor
- **name** (`<class 'str'>`): Model name for output files
- **output_dir** (`typing.Union[str, pathlib.Path]`): Directory for exported files

**Returns**: `<class 'tensorrt_export.ExportResult'>` - ExportResult with paths and metadata

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\deployment\tensorrt_export.py:487](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\deployment\tensorrt_export.py#L487)*

##### `export_cfd_solver`

```python
def export_cfd_solver(self, grid_shape: Tuple[int, int], output_dir: Union[str, pathlib.Path] = './exports') -> tensorrt_export.ExportResult
```

Export CFD inference module for embedded deployment.

**Parameters:**

- **grid_shape** (`typing.Tuple[int, int]`): (Nx, Ny) grid dimensions
- **output_dir** (`typing.Union[str, pathlib.Path]`): Directory for exported files

**Returns**: `<class 'tensorrt_export.ExportResult'>` - ExportResult for the CFD solver

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\deployment\tensorrt_export.py:552](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\deployment\tensorrt_export.py#L552)*

## Functions

### `benchmark_inference`

```python
def benchmark_inference(model_path: Union[str, pathlib.Path], input_shape: Tuple[int, ...], n_warmup: int = 10, n_iterations: int = 100, device: str = 'cuda') -> tensorrt_export.BenchmarkResult
```

Benchmark inference latency and throughput.

**Parameters:**

- **model_path** (`typing.Union[str, pathlib.Path]`): Path to model (ONNX or TRT)
- **input_shape** (`typing.Tuple[int, ...]`): Shape of input tensor
- **n_warmup** (`<class 'int'>`): Number of warmup iterations
- **n_iterations** (`<class 'int'>`): Number of timed iterations
- **device** (`<class 'str'>`): Device to run on

**Returns**: `<class 'tensorrt_export.BenchmarkResult'>` - BenchmarkResult with timing information

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\deployment\tensorrt_export.py:394](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\deployment\tensorrt_export.py#L394)*

### `export_to_onnx`

```python
def export_to_onnx(model: torch.nn.modules.module.Module, sample_input: torch.Tensor, output_path: Union[str, pathlib.Path], config: tensorrt_export.ExportConfig = None) -> pathlib.Path
```

Export PyTorch model to ONNX format.

**Parameters:**

- **model** (`<class 'torch.nn.modules.module.Module'>`): PyTorch module to export
- **sample_input** (`<class 'torch.Tensor'>`): Example input for tracing
- **output_path** (`typing.Union[str, pathlib.Path]`): Path for ONNX file
- **config** (`<class 'tensorrt_export.ExportConfig'>`): Export configuration

**Returns**: `<class 'pathlib.Path'>` - Path to exported ONNX file

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\deployment\tensorrt_export.py:213](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\deployment\tensorrt_export.py#L213)*

### `optimize_for_tensorrt`

```python
def optimize_for_tensorrt(onnx_path: Union[str, pathlib.Path], output_path: Union[str, pathlib.Path], config: tensorrt_export.ExportConfig = None) -> pathlib.Path
```

Optimize ONNX model for TensorRT inference.

**Parameters:**

- **onnx_path** (`typing.Union[str, pathlib.Path]`): Path to ONNX model
- **output_path** (`typing.Union[str, pathlib.Path]`): Path for TensorRT engine
- **config** (`<class 'tensorrt_export.ExportConfig'>`): Export configuration

**Returns**: `<class 'pathlib.Path'>` - Path to TensorRT engine file

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\deployment\tensorrt_export.py:265](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\deployment\tensorrt_export.py#L265)*

### `validate_exported_model`

```python
def validate_exported_model(original_model: torch.nn.modules.module.Module, exported_path: Union[str, pathlib.Path], test_inputs: List[torch.Tensor], rtol: float = 0.001, atol: float = 1e-05) -> Tuple[bool, float]
```

Validate exported model against original PyTorch model.

**Parameters:**

- **original_model** (`<class 'torch.nn.modules.module.Module'>`): Original PyTorch module
- **exported_path** (`typing.Union[str, pathlib.Path]`): Path to exported ONNX/TRT model
- **test_inputs** (`typing.List[torch.Tensor]`): List of test input tensors rtol, atol: Relative and absolute tolerance

**Returns**: `typing.Tuple[bool, float]` - (validation_passed, max_error)

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\deployment\tensorrt_export.py:342](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\deployment\tensorrt_export.py#L342)*

### `validate_tensorrt_export`

```python
def validate_tensorrt_export()
```

Run validation tests for TensorRT export.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\deployment\tensorrt_export.py:575](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\deployment\tensorrt_export.py#L575)*
