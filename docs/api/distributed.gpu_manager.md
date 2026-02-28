# Module `distributed.gpu_manager`

Multi-GPU resource management for distributed CFD.

This module provides GPU device management, memory pooling,
and workload distribution for multi-GPU simulations.

Author: Tigantic Holdings LLC

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `GPUConfig`

Configuration for GPU management.

#### Attributes

- **device_ids** (`typing.Optional[typing.List[int]]`): 
- **memory_fraction** (`<class 'float'>`): 
- **enable_memory_pool** (`<class 'bool'>`): 
- **pool_size_mb** (`<class 'int'>`): 
- **use_amp** (`<class 'bool'>`): 
- **cudnn_benchmark** (`<class 'bool'>`): 
- **deterministic** (`<class 'bool'>`): 
- **distribution_strategy** (`<class 'str'>`): 
- **enable_checkpointing** (`<class 'bool'>`): 
- **checkpoint_interval** (`<class 'int'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, device_ids: Optional[List[int]] = None, memory_fraction: float = 0.9, enable_memory_pool: bool = True, pool_size_mb: int = 512, use_amp: bool = True, cudnn_benchmark: bool = True, deterministic: bool = False, distribution_strategy: str = 'data_parallel', enable_checkpointing: bool = True, checkpoint_interval: int = 100) -> None
```

### class `GPUDevice`

Information about a GPU device.

#### Attributes

- **device_id** (`<class 'int'>`): 
- **name** (`<class 'str'>`): 
- **total_memory** (`<class 'int'>`): 
- **available_memory** (`<class 'int'>`): 
- **compute_capability** (`typing.Tuple[int, int]`): 
- **is_active** (`<class 'bool'>`): 
- **current_memory_used** (`<class 'int'>`): 
- **current_utilization** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, device_id: int, name: str, total_memory: int, available_memory: int, compute_capability: Tuple[int, int], is_active: bool = False, current_memory_used: int = 0, current_utilization: float = 0.0) -> None
```

### class `GPUManager`

Multi-GPU resource manager.

Handles device selection, memory management, and workload
distribution across multiple GPUs.

#### Methods

##### `__init__`

```python
def __init__(self, config: gpu_manager.GPUConfig)
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\distributed\gpu_manager.py:150](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\distributed\gpu_manager.py#L150)*

##### `cleanup`

```python
def cleanup(self)
```

Cleanup resources.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\distributed\gpu_manager.py:279](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\distributed\gpu_manager.py#L279)*

##### `device_context`

*@wraps*

```python
def device_context(self, device_id: int)
```

Context manager for device operations.

*Source: [C:\Users\bradl\AppData\Local\Programs\Python\Python311\Lib\contextlib.py:221](C:\Users\bradl\AppData\Local\Programs\Python\Python311\Lib\contextlib.py#L221)*

##### `get_device`

```python
def get_device(self, device_id: Optional[int] = None) -> torch.device
```

Get torch device.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\distributed\gpu_manager.py:208](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\distributed\gpu_manager.py#L208)*

##### `get_memory_summary`

```python
def get_memory_summary(self) -> Dict[int, Dict[str, float]]
```

Get memory summary for all devices.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\distributed\gpu_manager.py:254](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\distributed\gpu_manager.py#L254)*

##### `get_pool`

```python
def get_pool(self, device_id: Optional[int] = None) -> Optional[gpu_manager.MemoryPool]
```

Get memory pool for device.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\distributed\gpu_manager.py:236](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\distributed\gpu_manager.py#L236)*

##### `initialize`

```python
def initialize(self)
```

Initialize GPU management.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\distributed\gpu_manager.py:159](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\distributed\gpu_manager.py#L159)*

##### `synchronize`

```python
def synchronize(self, device_id: Optional[int] = None)
```

Synchronize GPU operations.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\distributed\gpu_manager.py:269](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\distributed\gpu_manager.py#L269)*

##### `update_memory_stats`

```python
def update_memory_stats(self)
```

Update memory usage statistics.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\distributed\gpu_manager.py:242](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\distributed\gpu_manager.py#L242)*

### class `MemoryPool`

GPU memory pool for efficient allocation.

Pre-allocates memory blocks to reduce allocation overhead
during CFD time-stepping.

#### Methods

##### `__init__`

```python
def __init__(self, device: torch.device, pool_size_mb: int = 512)
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\distributed\gpu_manager.py:67](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\distributed\gpu_manager.py#L67)*

##### `allocate`

```python
def allocate(self, name: str, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor
```

Allocate a named tensor from the pool.

**Parameters:**

- **name** (`<class 'str'>`): Unique identifier for this allocation
- **shape** (`typing.Tuple[int, ...]`): Tensor shape
- **dtype** (`<class 'torch.dtype'>`): Data type

**Returns**: `<class 'torch.Tensor'>` - Allocated tensor

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\distributed\gpu_manager.py:78](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\distributed\gpu_manager.py#L78)*

##### `clear`

```python
def clear(self)
```

Clear all allocations.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\distributed\gpu_manager.py:114](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\distributed\gpu_manager.py#L114)*

##### `get_stats`

```python
def get_stats(self) -> Dict[str, Any]
```

Get pool statistics.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\distributed\gpu_manager.py:121](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\distributed\gpu_manager.py#L121)*

##### `release`

```python
def release(self, name: str)
```

Mark a block as available for reuse.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\distributed\gpu_manager.py:108](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\distributed\gpu_manager.py#L108)*

## Functions

### `distribute_workload`

```python
def distribute_workload(n_elements: int, n_devices: int, balance_strategy: str = 'equal') -> Dict[int, Tuple[int, int]]
```

Distribute workload across devices.

**Parameters:**

- **n_elements** (`<class 'int'>`): Total number of elements
- **n_devices** (`<class 'int'>`): Number of devices
- **balance_strategy** (`<class 'str'>`): 'equal' or 'compute_weighted'

**Returns**: `typing.Dict[int, typing.Tuple[int, int]]` - Dictionary mapping device ID to (start, end) indices

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\distributed\gpu_manager.py:350](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\distributed\gpu_manager.py#L350)*

### `get_available_gpus`

```python
def get_available_gpus() -> List[gpu_manager.GPUDevice]
```

Get list of available GPUs.

**Returns**: `typing.List[gpu_manager.GPUDevice]` - List of GPU device information

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\distributed\gpu_manager.py:288](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\distributed\gpu_manager.py#L288)*

### `select_optimal_device`

```python
def select_optimal_device(workload_size: int, memory_required: int, prefer_compute: bool = True) -> int
```

Select optimal GPU device for a workload.

**Parameters:**

- **workload_size** (`<class 'int'>`): Size of computation
- **memory_required** (`<class 'int'>`): Memory required in bytes
- **prefer_compute** (`<class 'bool'>`): Prefer compute capability over memory

**Returns**: `<class 'int'>` - Selected device ID

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\distributed\gpu_manager.py:316](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\distributed\gpu_manager.py#L316)*

### `test_gpu_manager`

```python
def test_gpu_manager()
```

Test GPU management.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\distributed\gpu_manager.py:410](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\distributed\gpu_manager.py#L410)*
