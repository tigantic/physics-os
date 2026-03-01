# Module `integration.diagnostics`

System Diagnostics for Project The Physics OS.

Provides:
- System information gathering
- Health checks and monitoring
- Debugging utilities
- Performance profiling

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `DebugContext`

Debug context for capturing state.

#### Attributes

- **name** (`<class 'str'>`): 
- **variables** (`typing.Dict[str, typing.Any]`): 
- **stack_trace** (`<class 'str'>`): 
- **timestamp** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, name: str, variables: Dict[str, Any] = <factory>, stack_trace: str = '', timestamp: float = <factory>) -> None
```

##### `capture`

```python
def capture(name: str, **variables) -> 'DebugContext'
```

Capture current context.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py:649](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py#L649)*

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py:658](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py#L658)*

### class `DiagnosticsReport`

Complete diagnostics report.

#### Attributes

- **system_info** (`<class 'diagnostics.SystemInfo'>`): 
- **health_results** (`typing.Dict[str, diagnostics.HealthCheckResult]`): 
- **timestamp** (`<class 'float'>`): 
- **issues** (`typing.List[str]`): 
- **recommendations** (`typing.List[str]`): 

#### Methods

##### `__init__`

```python
def __init__(self, system_info: diagnostics.SystemInfo, health_results: Dict[str, diagnostics.HealthCheckResult], timestamp: float = <factory>, issues: List[str] = <factory>, recommendations: List[str] = <factory>) -> None
```

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py:527](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py#L527)*

##### `to_markdown`

```python
def to_markdown(self) -> str
```

Generate markdown report.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py:537](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py#L537)*

### class `GPUInfo`

GPU device information.

#### Attributes

- **device_id** (`<class 'int'>`): 
- **name** (`<class 'str'>`): 
- **memory_total** (`<class 'int'>`): 
- **memory_used** (`<class 'int'>`): 
- **memory_free** (`<class 'int'>`): 
- **compute_capability** (`typing.Optional[str]`): 
- **driver_version** (`typing.Optional[str]`): 

#### Properties

##### `memory_utilization`

```python
def memory_utilization(self) -> float
```

Memory utilization percentage.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py:50](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py#L50)*

#### Methods

##### `__init__`

```python
def __init__(self, device_id: int, name: str, memory_total: int, memory_used: int = 0, memory_free: int = 0, compute_capability: Optional[str] = None, driver_version: Optional[str] = None) -> None
```

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py:57](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py#L57)*

### class `HealthCheck`

Single health check definition.

#### Methods

##### `__init__`

```python
def __init__(self, name: str, check_fn: Callable[[], Dict[str, Any]], description: str = '', timeout: float = 30.0)
```

Initialize health check.

**Parameters:**

- **name** (`<class 'str'>`): Check name
- **check_fn** (`typing.Callable[[], typing.Dict[str, typing.Any]]`): Function that returns check result
- **description** (`<class 'str'>`): Check description
- **timeout** (`<class 'float'>`): Check timeout in seconds

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py:272](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py#L272)*

##### `run`

```python
def run(self) -> diagnostics.HealthCheckResult
```

Run the health check.

**Returns**: `<class 'diagnostics.HealthCheckResult'>` - HealthCheckResult with status

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py:293](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py#L293)*

### class `HealthCheckResult`

Result of a health check.

#### Attributes

- **name** (`<class 'str'>`): 
- **status** (`<enum 'HealthStatus'>`): 
- **message** (`<class 'str'>`): 
- **duration** (`<class 'float'>`): 
- **details** (`typing.Dict[str, typing.Any]`): 

#### Methods

##### `__init__`

```python
def __init__(self, name: str, status: diagnostics.HealthStatus, message: str = '', duration: float = 0.0, details: Dict[str, Any] = <factory>) -> None
```

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py:256](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py#L256)*

### class `HealthStatus`(Enum)

Health check status.

### class `MemoryInfo`

System memory information.

#### Attributes

- **total** (`<class 'int'>`): 
- **available** (`<class 'int'>`): 
- **used** (`<class 'int'>`): 
- **percent** (`<class 'float'>`): 
- **python_used** (`<class 'int'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, total: int, available: int, used: int, percent: float, python_used: int = 0) -> None
```

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py:89](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py#L89)*

### class `Profiler`

Simple profiler for timing code sections.

#### Methods

##### `__init__`

```python
def __init__(self)
```

Initialize profiler.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py:673](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py#L673)*

##### `profile`

*@wraps*

```python
def profile(self, name: str)
```

Context manager for profiling.

*Source: [C:\Users\bradl\AppData\Local\Programs\Python\Python311\Lib\contextlib.py:699](C:\Users\bradl\AppData\Local\Programs\Python\Python311\Lib\contextlib.py#L699)*

##### `report`

```python
def report(self) -> str
```

Generate profiling report.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py:723](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py#L723)*

##### `start`

```python
def start(self, name: str)
```

Start timing a section.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py:679](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py#L679)*

##### `stop`

```python
def stop(self, name: str) -> float
```

Stop timing a section.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py:684](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py#L684)*

##### `summary`

```python
def summary(self) -> Dict[str, Dict[str, float]]
```

Get profiling summary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py:708](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py#L708)*

### class `SystemHealthMonitor`

Monitor system health with multiple checks.

#### Properties

##### `overall_status`

```python
def overall_status(self) -> diagnostics.HealthStatus
```

Get overall system health status.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py:475](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py#L475)*

#### Methods

##### `__init__`

```python
def __init__(self)
```

Initialize health monitor.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py:334](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py#L334)*

##### `register`

```python
def register(self, check: diagnostics.HealthCheck)
```

Register a health check.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py:450](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py#L450)*

##### `run_all`

```python
def run_all(self) -> Dict[str, diagnostics.HealthCheckResult]
```

Run all health checks.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py:466](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py#L466)*

##### `run_check`

```python
def run_check(self, name: str) -> Optional[diagnostics.HealthCheckResult]
```

Run a specific health check.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py:454](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py#L454)*

##### `summary`

```python
def summary(self) -> str
```

Generate health summary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py:493](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py#L493)*

### class `SystemInfo`

Complete system information.

#### Attributes

- **platform** (`<class 'str'>`): 
- **python_version** (`<class 'str'>`): 
- **pytorch_version** (`<class 'str'>`): 
- **numpy_version** (`<class 'str'>`): 
- **cpu_count** (`<class 'int'>`): 
- **memory** (`<class 'diagnostics.MemoryInfo'>`): 
- **gpus** (`typing.List[diagnostics.GPUInfo]`): 
- **hostname** (`<class 'str'>`): 
- **timestamp** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, platform: str, python_version: str, pytorch_version: str, numpy_version: str, cpu_count: int, memory: diagnostics.MemoryInfo, gpus: List[diagnostics.GPUInfo] = <factory>, hostname: str = '', timestamp: float = <factory>) -> None
```

##### `summary`

```python
def summary(self) -> str
```

Generate a summary string.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py:140](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py#L140)*

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py:126](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py#L126)*

### class `TracingSpan`

Tracing span for distributed tracing.

#### Attributes

- **name** (`<class 'str'>`): 
- **trace_id** (`<class 'str'>`): 
- **span_id** (`<class 'str'>`): 
- **parent_id** (`typing.Optional[str]`): 
- **start_time** (`<class 'float'>`): 
- **end_time** (`typing.Optional[float]`): 
- **tags** (`typing.Dict[str, str]`): 
- **logs** (`typing.List[typing.Dict[str, typing.Any]]`): 

#### Properties

##### `duration`

```python
def duration(self) -> Optional[float]
```

Get span duration.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py:784](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py#L784)*

#### Methods

##### `__init__`

```python
def __init__(self, name: str, trace_id: str, span_id: str = '', parent_id: Optional[str] = None, start_time: float = <factory>, end_time: Optional[float] = None, tags: Dict[str, str] = <factory>, logs: List[Dict[str, Any]] = <factory>) -> None
```

##### `finish`

```python
def finish(self)
```

Finish the span.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py:768](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py#L768)*

##### `log`

```python
def log(self, event: str, **data)
```

Add a log entry.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py:772](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py#L772)*

##### `tag`

```python
def tag(self, key: str, value: str)
```

Add a tag.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py:780](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py#L780)*

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py:791](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py#L791)*

## Functions

### `check_system_health`

```python
def check_system_health() -> bool
```

Quick health check.

**Returns**: `<class 'bool'>` - True if system is healthy

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py:616](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py#L616)*

### `get_system_info`

```python
def get_system_info() -> diagnostics.SystemInfo
```

Gather complete system information.

**Returns**: `<class 'diagnostics.SystemInfo'>` - SystemInfo with current system state

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py:157](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py#L157)*

### `run_diagnostics`

```python
def run_diagnostics() -> diagnostics.DiagnosticsReport
```

Run full system diagnostics.

**Returns**: `<class 'diagnostics.DiagnosticsReport'>` - DiagnosticsReport with results

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py:580](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\integration\diagnostics.py#L580)*
