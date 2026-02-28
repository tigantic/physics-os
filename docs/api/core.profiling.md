# Module `core.profiling`

Profiling Utilities ====================

Memory and performance profiling decorators per Article VIII.8.2.

Usage:
    from ontic.core.profiling import profile, memory_profile

    @profile
    def my_function():
        ...

    @memory_profile
    def memory_intensive_function():
        ...

Enable profiling by setting environment variable:
    TENSORNET_PROFILE=1 python script.py

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `PerformanceReport`

Collect and report performance statistics.

#### Methods

##### `__init__`

```python
def __init__(self)
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\profiling.py:152](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\profiling.py#L152)*

##### `measure`

```python
def measure(self, name: str)
```

Context manager to measure a named operation.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\profiling.py:155](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\profiling.py#L155)*

##### `summary`

```python
def summary(self) -> str
```

Generate summary report.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\profiling.py:178](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\profiling.py#L178)*

## Functions

### `memory_profile`

```python
def memory_profile(func: ~F) -> ~F
```

Decorator that profiles function memory usage.

Only active when TENSORNET_PROFILE=1 environment variable is set.
Uses tracemalloc to measure peak memory allocation.

**Parameters:**

- **func** (`~F`): Function to profile

**Returns**: `~F` - Wrapped function with memory tracking

**Examples:**

```python
@memory_profile
def build_environments(psi, H):
    ...
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\profiling.py:68](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\profiling.py#L68)*

### `profile`

```python
def profile(func: ~F) -> ~F
```

Decorator that profiles function execution time.

Only active when TENSORNET_PROFILE=1 environment variable is set.

**Parameters:**

- **func** (`~F`): Function to profile

**Returns**: `~F` - Wrapped function with timing

**Examples:**

```python
@profile
def dmrg_sweep(psi, H):
    ...
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\profiling.py:37](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\profiling.py#L37)*

### `profile_block`

```python
def profile_block(name: str)
```

Context manager for profiling code blocks.

Only active when TENSORNET_PROFILE=1 environment variable is set.

**Parameters:**

- **name** (`<class 'str'>`): Name for the profiled block

**Examples:**

```python
with profile_block("SVD computation"):
    U, S, V = torch.linalg.svd(A)
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\profiling.py:109](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\core\profiling.py#L109)*
