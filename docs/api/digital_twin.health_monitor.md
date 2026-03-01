# Module `digital_twin.health_monitor`

Health monitoring for hypersonic vehicle digital twins.

This module provides real-time structural and thermal health monitoring
with anomaly detection capabilities. It enables predictive maintenance
by tracking cumulative damage and detecting off-nominal conditions.

Key capabilities:
    - Structural health monitoring with damage accumulation
    - Thermal protection system integrity assessment
    - Anomaly detection using statistical and ML methods
    - Fatigue life estimation and monitoring

Author: Tigantic Holdings LLC

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `AnomalyDetector`

Statistical anomaly detection system.

Uses baseline statistics and deviation thresholds to
detect off-nominal behavior across multiple channels.

#### Methods

##### `__init__`

```python
def __init__(self, config: health_monitor.HealthConfig)
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\digital_twin\health_monitor.py:415](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\digital_twin\health_monitor.py#L415)*

##### `check`

```python
def check(self, measurements: Dict[str, float]) -> Dict[str, Any]
```

Check measurements for anomalies.

**Parameters:**

- **measurements** (`typing.Dict[str, float]`): Dictionary of current measurements

**Returns**: `typing.Dict[str, typing.Any]` - Anomaly detection result

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\digital_twin\health_monitor.py:433](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\digital_twin\health_monitor.py#L433)*

##### `reset_baseline`

```python
def reset_baseline(self)
```

Reset baseline (e.g., after system change).

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\digital_twin\health_monitor.py:510](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\digital_twin\health_monitor.py#L510)*

### class `AnomalyType`(Enum)

Type of detected anomaly.

### class `HealthConfig`

Configuration for health monitoring.

#### Attributes

- **structural_warning** (`<class 'float'>`): 
- **structural_critical** (`<class 'float'>`): 
- **thermal_margin_warning** (`<class 'float'>`): 
- **thermal_margin_critical** (`<class 'float'>`): 
- **fatigue_exponent** (`<class 'float'>`): 
- **fatigue_limit_cycles** (`<class 'float'>`): 
- **anomaly_window** (`<class 'int'>`): 
- **anomaly_threshold** (`<class 'float'>`): 
- **update_interval** (`<class 'float'>`): 
- **tps_design_temp** (`<class 'float'>`): 
- **tps_max_temp** (`<class 'float'>`): 
- **vibration_warning** (`<class 'float'>`): 
- **vibration_critical** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, structural_warning: float = 0.8, structural_critical: float = 0.95, thermal_margin_warning: float = 0.1, thermal_margin_critical: float = 0.02, fatigue_exponent: float = 3.0, fatigue_limit_cycles: float = 10000000.0, anomaly_window: int = 100, anomaly_threshold: float = 3.0, update_interval: float = 0.1, tps_design_temp: float = 2000.0, tps_max_temp: float = 2200.0, vibration_warning: float = 5.0, vibration_critical: float = 10.0) -> None
```

### class `HealthMetric`(Enum)

Specific health metrics tracked.

### class `HealthMonitor`

Comprehensive health monitoring system for hypersonic vehicles.

Tracks structural integrity, thermal state, fatigue accumulation,
and performs anomaly detection across multiple subsystems.

#### Methods

##### `__init__`

```python
def __init__(self, config: health_monitor.HealthConfig)
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\digital_twin\health_monitor.py:136](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\digital_twin\health_monitor.py#L136)*

##### `get_summary`

```python
def get_summary(self) -> Dict[str, Any]
```

Get summary of current health state.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\digital_twin\health_monitor.py:305](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\digital_twin\health_monitor.py#L305)*

##### `update`

```python
def update(self, timestamp: float, loads: Optional[numpy.ndarray] = None, temperatures: Optional[numpy.ndarray] = None, vibrations: Optional[numpy.ndarray] = None, control_state: Optional[numpy.ndarray] = None) -> health_monitor.HealthState
```

Update health state with new measurements.

**Parameters:**

- **timestamp** (`<class 'float'>`): Current time
- **loads** (`typing.Optional[numpy.ndarray]`): Structural loads array
- **temperatures** (`typing.Optional[numpy.ndarray]`): Temperature distribution
- **vibrations** (`typing.Optional[numpy.ndarray]`): Vibration measurements
- **control_state** (`typing.Optional[numpy.ndarray]`): Control surface states

**Returns**: `<class 'health_monitor.HealthState'>` - Updated health state

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\digital_twin\health_monitor.py:166](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\digital_twin\health_monitor.py#L166)*

### class `HealthState`

Current health state of vehicle.

#### Attributes

- **timestamp** (`<class 'float'>`): 
- **status** (`<enum 'HealthStatus'>`): 
- **structural_damage** (`<class 'float'>`): 
- **tps_damage** (`<class 'float'>`): 
- **fatigue_fraction** (`<class 'float'>`): 
- **max_temperature** (`<class 'float'>`): 
- **max_vibration** (`<class 'float'>`): 
- **thermal_margin** (`<class 'float'>`): 
- **anomaly_detected** (`<class 'bool'>`): 
- **anomaly_type** (`<enum 'AnomalyType'>`): 
- **anomaly_confidence** (`<class 'float'>`): 
- **remaining_life_hours** (`typing.Optional[float]`): 

#### Methods

##### `__init__`

```python
def __init__(self, timestamp: float, status: health_monitor.HealthStatus, structural_damage: float = 0.0, tps_damage: float = 0.0, fatigue_fraction: float = 0.0, max_temperature: float = 0.0, max_vibration: float = 0.0, thermal_margin: float = 1.0, anomaly_detected: bool = False, anomaly_type: health_monitor.AnomalyType = <AnomalyType.NONE: 1>, anomaly_confidence: float = 0.0, remaining_life_hours: Optional[float] = None) -> None
```

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\digital_twin\health_monitor.py:111](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\digital_twin\health_monitor.py#L111)*

### class `HealthStatus`(Enum)

Overall health status of vehicle system.

### class `StructuralHealth`

Structural health monitoring subsystem.

Tracks load history, estimates damage accumulation,
and monitors for structural anomalies.

#### Methods

##### `__init__`

```python
def __init__(self, config: health_monitor.HealthConfig)
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\digital_twin\health_monitor.py:328](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\digital_twin\health_monitor.py#L328)*

##### `reset`

```python
def reset(self)
```

Reset damage tracking (e.g., after repair).

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\digital_twin\health_monitor.py:354](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\digital_twin\health_monitor.py#L354)*

##### `update`

```python
def update(self, loads: numpy.ndarray)
```

Update with new load measurements.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\digital_twin\health_monitor.py:340](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\digital_twin\health_monitor.py#L340)*

### class `ThermalHealth`

Thermal protection system health monitoring.

Tracks temperature history, thermal cycling, and
estimates TPS degradation.

#### Methods

##### `__init__`

```python
def __init__(self, config: health_monitor.HealthConfig)
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\digital_twin\health_monitor.py:369](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\digital_twin\health_monitor.py#L369)*

##### `update`

```python
def update(self, temperatures: numpy.ndarray)
```

Update with new temperature measurements.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\digital_twin\health_monitor.py:382](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\digital_twin\health_monitor.py#L382)*

## Functions

### `compute_damage_index`

```python
def compute_damage_index(load_history: numpy.ndarray, design_load: float, fatigue_exponent: float = 3.0) -> float
```

Compute damage index from load history using Miner's rule.

**Parameters:**

- **load_history** (`<class 'numpy.ndarray'>`): Array of load values
- **design_load** (`<class 'float'>`): Reference design load
- **fatigue_exponent** (`<class 'float'>`): S-N curve exponent

**Returns**: `<class 'float'>` - Cumulative damage index (0-1)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\digital_twin\health_monitor.py:519](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\digital_twin\health_monitor.py#L519)*

### `estimate_thermal_margin`

```python
def estimate_thermal_margin(temperatures: numpy.ndarray, limit_temp: float, safety_factor: float = 1.1) -> float
```

Estimate thermal margin from temperature distribution.

**Parameters:**

- **temperatures** (`<class 'numpy.ndarray'>`): Current temperature array
- **limit_temp** (`<class 'float'>`): Temperature limit
- **safety_factor** (`<class 'float'>`): Safety factor for margin calculation

**Returns**: `<class 'float'>` - Thermal margin (0-1, 0=at limit)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\digital_twin\health_monitor.py:544](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\digital_twin\health_monitor.py#L544)*

### `test_health_monitor`

```python
def test_health_monitor()
```

Test health monitoring module.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\digital_twin\health_monitor.py:565](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\digital_twin\health_monitor.py#L565)*
