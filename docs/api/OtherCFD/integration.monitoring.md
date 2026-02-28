# Module `integration.monitoring`

Monitoring and Logging for Project The Physics OS.

Provides:
- Structured logging with multiple outputs
- Metrics collection and aggregation
- Telemetry for performance tracking
- Alerting for anomaly detection

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `Alert`

System alert.

#### Attributes

- **name** (`<class 'str'>`): 
- **severity** (`<enum 'AlertSeverity'>`): 
- **message** (`<class 'str'>`): 
- **timestamp** (`<class 'float'>`): 
- **context** (`typing.Dict[str, typing.Any]`): 
- **resolved** (`<class 'bool'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, name: str, severity: monitoring.AlertSeverity, message: str, timestamp: float = <factory>, context: Dict[str, Any] = <factory>, resolved: bool = False) -> None
```

##### `resolve`

```python
def resolve(self)
```

Mark alert as resolved.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:591](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L591)*

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:595](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L595)*

### class `AlertManager`

Manages system alerts.

#### Methods

##### `__init__`

```python
def __init__(self)
```

Initialize alert manager.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:612](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L612)*

##### `add_handler`

```python
def add_handler(self, handler: Callable[[monitoring.Alert], NoneType])
```

Add an alert handler.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:618](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L618)*

##### `get_active`

```python
def get_active(self, severity: Optional[monitoring.AlertSeverity] = None) -> List[monitoring.Alert]
```

Get active (unresolved) alerts.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:659](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L659)*

##### `raise_alert`

```python
def raise_alert(self, name: str, severity: monitoring.AlertSeverity, message: str, **context) -> monitoring.Alert
```

Raise a new alert.

**Parameters:**

- **name** (`<class 'str'>`): Alert name
- **severity** (`<enum 'AlertSeverity'>`): Severity level
- **message** (`<class 'str'>`): Alert message **context: Additional context

**Returns**: `<class 'monitoring.Alert'>` - The raised alert

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:622](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L622)*

##### `resolve`

```python
def resolve(self, name: str)
```

Resolve alerts by name.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:669](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L669)*

### class `AlertSeverity`(Enum)

Alert severity levels.

### class `LogEntry`

Structured log entry.

#### Attributes

- **timestamp** (`<class 'float'>`): 
- **level** (`<enum 'LogLevel'>`): 
- **message** (`<class 'str'>`): 
- **logger_name** (`<class 'str'>`): 
- **context** (`typing.Dict[str, typing.Any]`): 
- **source** (`typing.Optional[str]`): 
- **line** (`typing.Optional[int]`): 

#### Methods

##### `__init__`

```python
def __init__(self, timestamp: float, level: monitoring.LogLevel, message: str, logger_name: str = 'ontic', context: Dict[str, Any] = <factory>, source: Optional[str] = None, line: Optional[int] = None) -> None
```

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:53](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L53)*

##### `to_json`

```python
def to_json(self) -> str
```

Convert to JSON string.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:66](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L66)*

### class `LogFormatter`

Format log entries for output.

#### Methods

##### `__init__`

```python
def __init__(self, format: str = '{datetime} [{level}] {message}')
```

Initialize formatter.

**Parameters:**

- **format** (`<class 'str'>`): Format string or "json"

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:80](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L80)*

##### `format`

```python
def format(self, entry: monitoring.LogEntry) -> str
```

Format a log entry.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:89](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L89)*

### class `LogLevel`(Enum)

Log severity levels.

### class `Metric`

Single metric value.

#### Attributes

- **name** (`<class 'str'>`): 
- **value** (`<class 'float'>`): 
- **metric_type** (`<enum 'MetricType'>`): 
- **labels** (`typing.Dict[str, str]`): 
- **timestamp** (`<class 'float'>`): 
- **unit** (`<class 'str'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, name: str, value: float, metric_type: monitoring.MetricType = <MetricType.GAUGE: 2>, labels: Dict[str, str] = <factory>, timestamp: float = <factory>, unit: str = '') -> None
```

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:245](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L245)*

### class `MetricCollector`

Collects and aggregates metrics.

#### Methods

##### `__init__`

```python
def __init__(self, name: str = 'default')
```

Initialize collector.

**Parameters:**

- **name** (`<class 'str'>`): Collector name

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:262](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L262)*

##### `clear`

```python
def clear(self, name: Optional[str] = None)
```

Clear metrics.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:355](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L355)*

##### `get_all`

```python
def get_all(self, name: str) -> List[monitoring.Metric]
```

Get all values for a metric.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:332](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L332)*

##### `get_latest`

```python
def get_latest(self, name: str) -> Optional[monitoring.Metric]
```

Get the latest value for a metric.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:325](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L325)*

##### `increment`

```python
def increment(self, name: str, value: float = 1.0, **kwargs)
```

Increment a counter.

**Parameters:**

- **name** (`<class 'str'>`): Counter name
- **value** (`<class 'float'>`): Increment amount

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:298](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L298)*

##### `record`

```python
def record(self, name: str, value: float, **kwargs)
```

Record a metric value.

**Parameters:**

- **name** (`<class 'str'>`): Metric name
- **value** (`<class 'float'>`): Metric value **kwargs: Additional metric attributes

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:274](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L274)*

##### `summary`

```python
def summary(self) -> Dict[str, Dict]
```

Get summary statistics for all metrics.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:337](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L337)*

##### `timing`

```python
def timing(self, name: str, duration: float, **kwargs)
```

Record a timing metric.

**Parameters:**

- **name** (`<class 'str'>`): Timer name
- **duration** (`<class 'float'>`): Duration in seconds

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:314](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L314)*

### class `MetricType`(Enum)

Types of metrics.

### class `MetricsRegistry`

Global registry for metric collectors.

#### Methods

##### `get_collector`

```python
def get_collector(name: str = 'default') -> monitoring.MetricCollector
```

Get or create a collector.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:372](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L372)*

##### `record`

```python
def record(name: str, value: float, collector: str = 'default', **kwargs)
```

Record a metric in a collector.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:380](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L380)*

##### `summary`

```python
def summary() -> Dict[str, Dict]
```

Get summary from all collectors.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:385](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L385)*

### class `StructuredLogger`

Structured logger with multiple handlers.

#### Methods

##### `__init__`

```python
def __init__(self, name: str = 'ontic', level: monitoring.LogLevel = <LogLevel.INFO: 20>)
```

Initialize logger.

**Parameters:**

- **name** (`<class 'str'>`): Logger name
- **level** (`<enum 'LogLevel'>`): Minimum log level

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:111](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L111)*

##### `add_handler`

```python
def add_handler(self, handler: Callable[[monitoring.LogEntry], NoneType])
```

Add a log handler.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:138](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L138)*

##### `critical`

```python
def critical(self, message: str, **kwargs)
```

Log critical message.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:203](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L203)*

##### `debug`

```python
def debug(self, message: str, **kwargs)
```

Log debug message.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:187](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L187)*

##### `error`

```python
def error(self, message: str, **kwargs)
```

Log error message.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:199](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L199)*

##### `get_buffer`

```python
def get_buffer(self) -> List[monitoring.LogEntry]
```

Get buffered log entries.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:207](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L207)*

##### `get_logger`

```python
def get_logger(name: str = 'ontic') -> 'StructuredLogger'
```

Get or create a logger by name.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:131](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L131)*

##### `info`

```python
def info(self, message: str, **kwargs)
```

Log info message.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:191](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L191)*

##### `warning`

```python
def warning(self, message: str, **kwargs)
```

Log warning message.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:195](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L195)*

### class `TelemetryCollector`

Collects telemetry events for tracing.

#### Methods

##### `__init__`

```python
def __init__(self, max_events: int = 10000)
```

Initialize collector.

**Parameters:**

- **max_events** (`<class 'int'>`): Maximum events to buffer

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:453](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L453)*

##### `end_event`

```python
def end_event(self, event_id: str) -> Optional[monitoring.TelemetryEvent]
```

End a telemetry event.

**Parameters:**

- **event_id** (`<class 'str'>`): Event ID to end

**Returns**: `typing.Optional[monitoring.TelemetryEvent]` - The completed event

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:493](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L493)*

##### `get_events`

```python
def get_events(self, name: Optional[str] = None, since: Optional[float] = None) -> List[monitoring.TelemetryEvent]
```

Get telemetry events.

**Parameters:**

- **name** (`typing.Optional[str]`): Filter by name
- **since** (`typing.Optional[float]`): Filter by timestamp

**Returns**: `typing.List[monitoring.TelemetryEvent]` - Matching events

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:511](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L511)*

##### `get_trace`

```python
def get_trace(self, event_id: str) -> List[monitoring.TelemetryEvent]
```

Get full trace for an event.

**Parameters:**

- **event_id** (`<class 'str'>`): Root event ID

**Returns**: `typing.List[monitoring.TelemetryEvent]` - Events in the trace

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:536](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L536)*

##### `start_event`

```python
def start_event(self, name: str, parent_id: Optional[str] = None, **metadata) -> monitoring.TelemetryEvent
```

Start a new telemetry event.

**Parameters:**

- **name** (`<class 'str'>`): Event name
- **parent_id** (`typing.Optional[str]`): Parent event ID **metadata: Event metadata

**Returns**: `<class 'monitoring.TelemetryEvent'>` - The started event

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:464](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L464)*

### class `TelemetryEvent`

Telemetry event for performance tracking.

#### Attributes

- **name** (`<class 'str'>`): 
- **start_time** (`<class 'float'>`): 
- **end_time** (`typing.Optional[float]`): 
- **metadata** (`typing.Dict[str, typing.Any]`): 
- **parent_id** (`typing.Optional[str]`): 
- **event_id** (`<class 'str'>`): 

#### Properties

##### `duration`

```python
def duration(self) -> Optional[float]
```

Get event duration.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:424](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L424)*

#### Methods

##### `__init__`

```python
def __init__(self, name: str, start_time: float, end_time: Optional[float] = None, metadata: Dict[str, Any] = <factory>, parent_id: Optional[str] = None, event_id: str = '') -> None
```

##### `finish`

```python
def finish(self)
```

Mark event as finished.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:431](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L431)*

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:435](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L435)*

## Functions

### `get_logger`

```python
def get_logger(name: str = 'ontic') -> monitoring.StructuredLogger
```

Get a structured logger.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:682](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L682)*

### `log_error`

```python
def log_error(message: str, **context)
```

Log an error message.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:697](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L697)*

### `log_info`

```python
def log_info(message: str, **context)
```

Log an info message.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:687](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L687)*

### `log_warning`

```python
def log_warning(message: str, **context)
```

Log a warning message.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:692](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L692)*

### `record_metric`

```python
def record_metric(name: str, value: float, **kwargs)
```

Record a metric value.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py:702](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\integration\monitoring.py#L702)*
