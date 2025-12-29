# Module `digital_twin.state_sync`

Real-time state synchronization for digital twin systems.

This module provides the infrastructure for synchronizing state between
physical vehicles and their digital counterparts, enabling real-time
monitoring, prediction, and control optimization.

Key features:
    - Low-latency state transfer with configurable update rates
    - Automatic state interpolation and extrapolation for timing mismatches
    - Divergence detection and correction mechanisms
    - Network resilience with buffering and recovery
    - Multi-fidelity state representation

Author: HyperTensor Team

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `StateBuffer`

Thread-safe circular buffer for state history.

Maintains temporal history of states for interpolation,
extrapolation, and analysis purposes.

#### Methods

##### `__init__`

```python
def __init__(self, max_size: int = 1000)
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py:232](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py#L232)*

##### `clear`

```python
def clear(self)
```

Clear buffer.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py:277](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py#L277)*

##### `get_at_time`

```python
def get_at_time(self, t: float) -> Optional[state_sync.StateVector]
```

Get state closest to time t.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py:250](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py#L250)*

##### `get_latest`

```python
def get_latest(self, n: int = 1) -> List[state_sync.StateVector]
```

Get n most recent states.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py:242](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py#L242)*

##### `get_range`

```python
def get_range(self, t_start: float, t_end: float) -> List[state_sync.StateVector]
```

Get states in time range.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py:271](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py#L271)*

##### `push`

```python
def push(self, state: state_sync.StateVector)
```

Add state to buffer.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py:237](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py#L237)*

### class `StateSync`

State synchronization manager for digital twin.

Handles real-time state transfer, divergence detection,
and automatic correction between physical and digital systems.

#### Methods

##### `__init__`

```python
def __init__(self, config: state_sync.SyncConfig)
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py:528](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py#L528)*

##### `get_statistics`

```python
def get_statistics(self) -> Dict[str, Any]
```

Get synchronization statistics.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py:754](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py#L754)*

##### `get_synchronized_state`

```python
def get_synchronized_state(self, t: Optional[float] = None) -> Optional[state_sync.StateVector]
```

Get best estimate of vehicle state at time t.

Combines physical and digital states with appropriate weighting.

**Parameters:**

- **t** (`typing.Optional[float]`): Target time (None = latest)

**Returns**: `typing.Optional[state_sync.StateVector]` - Best state estimate

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py:600](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py#L600)*

##### `push_digital_state`

```python
def push_digital_state(self, state: state_sync.StateVector)
```

Push new digital state (from simulation).

**Parameters:**

- **state** (`<class 'state_sync.StateVector'>`): Current digital twin state

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py:590](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py#L590)*

##### `push_physical_state`

```python
def push_physical_state(self, state: state_sync.StateVector)
```

Push new physical state (from vehicle/HIL).

**Parameters:**

- **state** (`<class 'state_sync.StateVector'>`): Current physical vehicle state

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py:576](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py#L576)*

##### `start`

```python
def start(self, mode: state_sync.SyncMode = <SyncMode.REAL_TIME: 1>)
```

Start synchronization process.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py:556](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py#L556)*

##### `stop`

```python
def stop(self)
```

Stop synchronization.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py:569](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py#L569)*

### class `StateSynchronizer`

High-level state synchronizer with automatic mode selection.

Provides simplified interface for common synchronization patterns.

#### Properties

##### `is_synced`

```python
def is_synced(self) -> bool
```

Check if currently synchronized.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py:803](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py#L803)*

#### Methods

##### `__init__`

```python
def __init__(self, sync_rate: float = 100.0)
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py:779](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py#L779)*

##### `connect`

```python
def connect(self, mode: state_sync.SyncMode = <SyncMode.REAL_TIME: 1>)
```

Connect and start synchronization.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py:783](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py#L783)*

##### `disconnect`

```python
def disconnect(self)
```

Disconnect and stop synchronization.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py:787](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py#L787)*

##### `get_state`

```python
def get_state(self, t: Optional[float] = None) -> Optional[state_sync.StateVector]
```

Get synchronized state.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py:799](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py#L799)*

##### `update_digital`

```python
def update_digital(self, state: state_sync.StateVector)
```

Update with digital state.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py:795](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py#L795)*

##### `update_physical`

```python
def update_physical(self, state: state_sync.StateVector)
```

Update with physical state.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py:791](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py#L791)*

### class `StateVector`

Complete state vector for hypersonic vehicle.

Represents the full state including position, velocity, attitude,
rates, and internal states for both physical and digital systems.

#### Attributes

- **timestamp** (`<class 'float'>`): 
- **position** (`<class 'numpy.ndarray'>`): 
- **velocity** (`<class 'numpy.ndarray'>`): 
- **quaternion** (`<class 'numpy.ndarray'>`): 
- **angular_rates** (`<class 'numpy.ndarray'>`): 
- **accelerations** (`<class 'numpy.ndarray'>`): 
- **control_surfaces** (`<class 'numpy.ndarray'>`): 
- **thrust** (`<class 'float'>`): 
- **fuel_mass** (`<class 'float'>`): 
- **temperatures** (`typing.Optional[numpy.ndarray]`): 
- **loads** (`typing.Optional[numpy.ndarray]`): 
- **mach** (`<class 'float'>`): 
- **dynamic_pressure** (`<class 'float'>`): 
- **altitude** (`<class 'float'>`): 
- **covariance** (`typing.Optional[numpy.ndarray]`): 

#### Methods

##### `__init__`

```python
def __init__(self, timestamp: float, position: numpy.ndarray, velocity: numpy.ndarray, quaternion: numpy.ndarray, angular_rates: numpy.ndarray, accelerations: numpy.ndarray, control_surfaces: numpy.ndarray, thrust: float = 0.0, fuel_mass: float = 0.0, temperatures: Optional[numpy.ndarray] = None, loads: Optional[numpy.ndarray] = None, mach: float = 0.0, dynamic_pressure: float = 0.0, altitude: float = 0.0, covariance: Optional[numpy.ndarray] = None) -> None
```

##### `copy`

```python
def copy(self) -> 'StateVector'
```

Create deep copy.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py:173](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py#L173)*

##### `from_vector`

```python
def from_vector(vector: numpy.ndarray, timestamp: float, n_control: int = 3, n_temps: int = 0, n_loads: int = 0) -> 'StateVector'
```

Reconstruct from flat vector.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py:128](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py#L128)*

##### `to_vector`

```python
def to_vector(self) -> numpy.ndarray
```

Convert to flat state vector.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py:110](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py#L110)*

### class `SyncConfig`

Configuration for state synchronization.

#### Attributes

- **sync_rate** (`<class 'float'>`): 
- **max_latency** (`<class 'float'>`): 
- **timeout** (`<class 'float'>`): 
- **buffer_size** (`<class 'int'>`): 
- **interpolation_window** (`<class 'int'>`): 
- **position_threshold** (`<class 'float'>`): 
- **velocity_threshold** (`<class 'float'>`): 
- **attitude_threshold** (`<class 'float'>`): 
- **max_divergence_time** (`<class 'float'>`): 
- **recovery_gain** (`<class 'float'>`): 
- **retry_attempts** (`<class 'int'>`): 
- **heartbeat_interval** (`<class 'float'>`): 
- **enable_prediction** (`<class 'bool'>`): 
- **prediction_horizon** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, sync_rate: float = 100.0, max_latency: float = 0.05, timeout: float = 1.0, buffer_size: int = 1000, interpolation_window: int = 5, position_threshold: float = 10.0, velocity_threshold: float = 5.0, attitude_threshold: float = 0.1, max_divergence_time: float = 0.5, recovery_gain: float = 0.1, retry_attempts: int = 3, heartbeat_interval: float = 0.1, enable_prediction: bool = True, prediction_horizon: float = 0.1) -> None
```

### class `SyncMode`(Enum)

Synchronization mode for digital twin.

### class `SyncStatus`(Enum)

Status of synchronization process.

## Functions

### `compute_state_divergence`

```python
def compute_state_divergence(physical: state_sync.StateVector, digital: state_sync.StateVector) -> Dict[str, float]
```

Compute divergence metrics between physical and digital states.

**Returns**: `typing.Dict[str, float]` - Dictionary of divergence metrics for each state component

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py:473](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py#L473)*

### `extrapolate_state`

```python
def extrapolate_state(states: List[state_sync.StateVector], t: float, order: int = 2) -> state_sync.StateVector
```

Extrapolate state forward in time.

Uses polynomial extrapolation based on recent state history.

**Parameters:**

- **states** (`typing.List[state_sync.StateVector]`): Recent state history (newest last)
- **t** (`<class 'float'>`): Target time
- **order** (`<class 'int'>`): Polynomial order (1=linear, 2=quadratic)

**Returns**: `<class 'state_sync.StateVector'>` - Extrapolated state at time t

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py:358](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py#L358)*

### `interpolate_state`

```python
def interpolate_state(state1: state_sync.StateVector, state2: state_sync.StateVector, t: float) -> state_sync.StateVector
```

Interpolate state between two time points.

Uses linear interpolation for positions and velocities,
SLERP for quaternions, and linear for other quantities.

**Parameters:**

- **state1** (`<class 'state_sync.StateVector'>`): Earlier state
- **state2** (`<class 'state_sync.StateVector'>`): Later state
- **t** (`<class 'float'>`): Target time

**Returns**: `<class 'state_sync.StateVector'>` - Interpolated state at time t

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py:287](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py#L287)*

### `test_state_sync`

```python
def test_state_sync()
```

Test state synchronization module.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py:809](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\digital_twin\state_sync.py#L809)*
