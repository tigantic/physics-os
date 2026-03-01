# Module `core.gpu`

CUDA/GPU Acceleration for Tensor Network CFD =============================================

Optimized GPU kernels for performance-critical operations:

    1. Tensor Contractions:
       - Batched Einstein summation
       - Optimized TT-vector products

    2. CFD Flux Computations:
       - Roe flux assembly
       - AUSM+ flux evaluation
       - Viscous stress tensor

    3. Memory Management:
       - Efficient data layout (AoS vs SoA)
       - Pinned memory for async transfers
       - Memory pool for temporary tensors

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `DeviceType`(Enum)

Compute device types.

### class `GPUConfig`

Configuration for GPU acceleration.

#### Attributes

- **device** (`<enum 'DeviceType'>`): 
- **device_id** (`<class 'int'>`): 
- **use_mixed_precision** (`<class 'bool'>`): 
- **memory_pool_size** (`<class 'int'>`): 
- **enable_tensor_cores** (`<class 'bool'>`): 
- **prefetch_factor** (`<class 'int'>`): 
- **pin_memory** (`<class 'bool'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, device: gpu.DeviceType = <DeviceType.CUDA: 'cuda'>, device_id: int = 0, use_mixed_precision: bool = False, memory_pool_size: int = 536870912, enable_tensor_cores: bool = True, prefetch_factor: int = 2, pin_memory: bool = True) -> None
```

### class `KernelStats`

Performance statistics for kernel execution.

#### Attributes

- **name** (`<class 'str'>`): 
- **elapsed_ms** (`<class 'float'>`): 
- **memory_read_bytes** (`<class 'int'>`): 
- **memory_write_bytes** (`<class 'int'>`): 
- **flops** (`<class 'int'>`): 

#### Properties

##### `bandwidth_gb_s`

```python
def bandwidth_gb_s(self) -> float
```

Compute achieved memory bandwidth.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\gpu.py:74](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\gpu.py#L74)*

##### `gflops`

```python
def gflops(self) -> float
```

Compute achieved GFLOP/s.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\gpu.py:80](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\gpu.py#L80)*

#### Methods

##### `__init__`

```python
def __init__(self, name: str, elapsed_ms: float, memory_read_bytes: int, memory_write_bytes: int, flops: int) -> None
```

### class `MemoryLayout`(Enum)

Memory layout for CFD arrays.

### class `MemoryPool`

Simple memory pool for temporary GPU allocations.

Reduces allocation overhead by reusing memory.

#### Methods

##### `__init__`

```python
def __init__(self, device: torch.device, pool_size: int = 536870912)
```

Args:

device: Target device
    pool_size: Size in bytes

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\gpu.py:144](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\gpu.py#L144)*

##### `allocate`

```python
def allocate(self, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor
```

Allocate tensor from pool.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\gpu.py:155](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\gpu.py#L155)*

##### `reset`

```python
def reset(self)
```

Reset pool for next iteration.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\gpu.py:173](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\gpu.py#L173)*

## Functions

### `batched_tt_matvec`

```python
def batched_tt_matvec(cores: List[torch.Tensor], vectors: torch.Tensor, device: torch.device = None) -> torch.Tensor
```

Batched Tensor-Train matrix-vector product.

Computes y = A·x where A is in TT format and x is a batch of vectors.
Uses optimized contraction order for GPU.

**Parameters:**

- **cores** (`typing.List[torch.Tensor]`): TT cores [G_1, G_2, ..., G_d]
- **vectors** (`<class 'torch.Tensor'>`): Input vectors (batch_size, n1*n2*...*nd)
- **device** (`<class 'torch.device'>`): Compute device

**Returns**: `<class 'torch.Tensor'>` - Result vectors (batch_size, m1*m2*...*md)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\gpu.py:180](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\gpu.py#L180)*

### `benchmark_kernel`

```python
def benchmark_kernel(kernel_fn, *args, n_warmup: int = 3, n_runs: int = 10, device: torch.device = None) -> gpu.KernelStats
```

Benchmark a GPU kernel.

**Parameters:**

- **kernel_fn**: Function to benchmark
- **args**: Arguments to pass
- **n_warmup** (`<class 'int'>`): Number of warmup runs
- **n_runs** (`<class 'int'>`): Number of timed runs
- **device** (`<class 'torch.device'>`): Compute device

**Returns**: `<class 'gpu.KernelStats'>` - KernelStats with timing information

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\gpu.py:538](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\gpu.py#L538)*

### `compute_strain_rate_gpu`

```python
def compute_strain_rate_gpu(u: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor, dz: Optional[torch.Tensor] = None) -> torch.Tensor
```

GPU-optimized strain rate tensor computation.

S_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)

**Parameters:**

- **u** (`<class 'torch.Tensor'>`): Velocity field (n_dims, Nx, Ny, [Nz]) dx, dy, dz: Grid spacing

**Returns**: `<class 'torch.Tensor'>` - Strain rate magnitude |S| = √(2 S_ij S_ij)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\gpu.py:379](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\gpu.py#L379)*

### `get_device`

```python
def get_device(config: gpu.GPUConfig = None) -> torch.device
```

Get the appropriate compute device.

**Parameters:**

- **config** (`<class 'gpu.GPUConfig'>`): GPU configuration

**Returns**: `<class 'torch.device'>` - torch.device object

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\gpu.py:86](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\gpu.py#L86)*

### `optimized_einsum`

```python
def optimized_einsum(equation: str, *operands: torch.Tensor, optimize: str = 'optimal') -> torch.Tensor
```

Optimized Einstein summation with contraction path optimization.

**Parameters:**

- **equation** (`<class 'str'>`): Einsum equation string
- **operands** (`<class 'torch.Tensor'>`): Input tensors
- **optimize** (`<class 'str'>`): Optimization strategy ("optimal", "greedy", "none")

**Returns**: `<class 'torch.Tensor'>` - Result tensor

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\gpu.py:234](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\gpu.py#L234)*

### `roe_flux_gpu`

```python
def roe_flux_gpu(rho_L: torch.Tensor, rho_R: torch.Tensor, u_L: torch.Tensor, u_R: torch.Tensor, p_L: torch.Tensor, p_R: torch.Tensor, E_L: torch.Tensor, E_R: torch.Tensor, gamma: float = 1.4, normal: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

GPU-optimized Roe flux computation.

Computes Roe approximate Riemann solver fluxes for all faces
in parallel on GPU.

**Parameters:**

- **gamma** (`<class 'float'>`): Specific heat ratio
- **normal** (`<class 'int'>`): Direction index (0=x, 1=y, 2=z)

**Returns**: `typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]` - (F_rho, F_rhou, F_rhov, F_E) flux tensors

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\gpu.py:258](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\gpu.py#L258)*

### `to_device`

```python
def to_device(tensor: torch.Tensor, device: torch.device, non_blocking: bool = True) -> torch.Tensor
```

Transfer tensor to device with optimal settings.

**Parameters:**

- **tensor** (`<class 'torch.Tensor'>`): Input tensor
- **device** (`<class 'torch.device'>`): Target device
- **non_blocking** (`<class 'bool'>`): Use async transfer if possible

**Returns**: `<class 'torch.Tensor'>` - Tensor on target device

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\gpu.py:115](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\gpu.py#L115)*

### `validate_gpu`

```python
def validate_gpu()
```

Run validation tests for GPU acceleration.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\gpu.py:589](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\gpu.py#L589)*

### `viscous_flux_gpu`

```python
def viscous_flux_gpu(rho: torch.Tensor, u: torch.Tensor, T: torch.Tensor, mu: torch.Tensor, k: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor, dz: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]
```

GPU-optimized viscous flux computation.

Computes viscous stresses and heat conduction fluxes
on GPU with fused operations.

**Parameters:**

- **rho** (`<class 'torch.Tensor'>`): Density
- **u** (`<class 'torch.Tensor'>`): Velocity (n_dims, Nx, Ny, [Nz])
- **T** (`<class 'torch.Tensor'>`): Temperature
- **mu** (`<class 'torch.Tensor'>`): Dynamic viscosity
- **k** (`<class 'torch.Tensor'>`): Thermal conductivity dx, dy, dz: Grid spacing

**Returns**: `typing.Tuple[torch.Tensor, ...]` - Viscous flux tensors

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\gpu.py:448](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\core\gpu.py#L448)*
