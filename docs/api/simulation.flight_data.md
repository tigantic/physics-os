# Module `simulation.flight_data`

Flight Data Integration Module ==============================

Parses, validates, and reconstructs trajectories from flight telemetry data
for model validation and post-flight analysis.

Capabilities:
    - Multi-format telemetry parsing (IRIG-106, CSV, binary)
    - Trajectory reconstruction with EKF/UKF
    - Model vs flight comparison metrics
    - Aerodynamic parameter estimation
    - Data quality assessment

Data Flow:
    ┌────────────────┐     ┌──────────────────┐     ┌──────────────────┐
    │  Raw Telemetry │────►│  Parser/Decoder  │────►│ TelemetryFrame[] │
    │  (Files/Stream)│     │                  │     │                  │
    └────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                             │
                           ┌──────────────────┐              │
                           │    FlightRecord  │◄─────────────┘
                           │                  │
                           └────────┬─────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         ▼                          ▼                          ▼
    ┌─────────────┐          ┌─────────────┐           ┌─────────────┐
    │ Trajectory  │          │  Parameter  │           │   Model     │
    │ Reconstruct │          │  Estimation │           │  Validation │
    └─────────────┘          └─────────────┘           └─────────────┘

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `DataQuality`(Enum)

Data quality classification.

### class `FlightRecord`

Complete flight record with metadata and telemetry.

#### Attributes

- **flight_id** (`<class 'str'>`): 
- **vehicle_id** (`<class 'str'>`): 
- **date** (`<class 'str'>`): 
- **frames** (`typing.List[flight_data.TelemetryFrame]`): 
- **metadata** (`typing.Dict[str, typing.Any]`): 
- **events** (`typing.List[typing.Tuple[float, str]]`): 

#### Properties

##### `duration`

```python
def duration(self) -> float
```

Flight duration in seconds.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\flight_data.py:141](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\flight_data.py#L141)*

##### `sample_rate`

```python
def sample_rate(self) -> float
```

Average sample rate in Hz.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\flight_data.py:148](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\flight_data.py#L148)*

#### Methods

##### `__init__`

```python
def __init__(self, flight_id: str, vehicle_id: str, date: str, frames: List[flight_data.TelemetryFrame] = <factory>, metadata: Dict[str, Any] = <factory>, events: List[Tuple[float, str]] = <factory>) -> None
```

##### `get_time_series`

```python
def get_time_series(self, field: str) -> Tuple[numpy.ndarray, numpy.ndarray]
```

Extract time series data for a field.

**Parameters:**

- **field** (`<class 'str'>`): Field name ('position', 'velocity', 'mach', etc.)

**Returns**: `typing.Tuple[numpy.ndarray, numpy.ndarray]` - (times, values) arrays

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\flight_data.py:155](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\flight_data.py#L155)*

##### `resample`

```python
def resample(self, target_rate_hz: float) -> 'FlightRecord'
```

Resample to target rate using linear interpolation.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\flight_data.py:201](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\flight_data.py#L201)*

##### `trim_to_range`

```python
def trim_to_range(self, t_start: float, t_end: float) -> 'FlightRecord'
```

Return a new FlightRecord trimmed to time range.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\flight_data.py:185](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\flight_data.py#L185)*

### class `TelemetryFormat`(Enum)

Supported telemetry formats.

### class `TelemetryFrame`

Single telemetry frame with synchronized sensor data.

#### Attributes

- **timestamp** (`<class 'float'>`): 
- **position** (`typing.Optional[numpy.ndarray]`): 
- **velocity** (`typing.Optional[numpy.ndarray]`): 
- **attitude** (`typing.Optional[numpy.ndarray]`): 
- **rates** (`typing.Optional[numpy.ndarray]`): 
- **accelerations** (`typing.Optional[numpy.ndarray]`): 
- **air_data** (`typing.Optional[typing.Dict[str, float]]`): 
- **sensor_status** (`typing.Dict[str, bool]`): 
- **quality** (`<enum 'DataQuality'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, timestamp: float, position: Optional[numpy.ndarray] = None, velocity: Optional[numpy.ndarray] = None, attitude: Optional[numpy.ndarray] = None, rates: Optional[numpy.ndarray] = None, accelerations: Optional[numpy.ndarray] = None, air_data: Optional[Dict[str, float]] = None, sensor_status: Dict[str, bool] = <factory>, quality: flight_data.DataQuality = <DataQuality.GOOD: 'good'>) -> None
```

##### `from_dict`

```python
def from_dict(d: Dict[str, Any]) -> 'TelemetryFrame'
```

Create from dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\flight_data.py:105](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\flight_data.py#L105)*

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\flight_data.py:91](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\flight_data.py#L91)*

### class `TrajectoryReconstruction`

Reconstructed trajectory with uncertainty.

Uses Extended Kalman Filter or Unscented Kalman Filter
to optimally combine sensor data.

#### Attributes

- **state_dim** (`<class 'int'>`): 
- **P** (`typing.Optional[numpy.ndarray]`): 
- **Q** (`typing.Optional[numpy.ndarray]`): 
- **R_gps** (`typing.Optional[numpy.ndarray]`): 
- **R_imu** (`typing.Optional[numpy.ndarray]`): 
- **states** (`typing.List[numpy.ndarray]`): 
- **covariances** (`typing.List[numpy.ndarray]`): 
- **times** (`typing.List[float]`): 

#### Methods

##### `__init__`

```python
def __init__(self, state_dim: int = 12, P: Optional[numpy.ndarray] = None, Q: Optional[numpy.ndarray] = None, R_gps: Optional[numpy.ndarray] = None, R_imu: Optional[numpy.ndarray] = None, states: List[numpy.ndarray] = <factory>, covariances: List[numpy.ndarray] = <factory>, times: List[float] = <factory>) -> None
```

##### `get_uncertainty`

```python
def get_uncertainty(self, time_idx: int) -> numpy.ndarray
```

Get 1-sigma uncertainty at time index.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\flight_data.py:419](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\flight_data.py#L419)*

##### `process_frame`

```python
def process_frame(self, frame: flight_data.TelemetryFrame, x_prev: numpy.ndarray, dt: float) -> numpy.ndarray
```

Process a telemetry frame with EKF.

**Parameters:**

- **frame** (`<class 'flight_data.TelemetryFrame'>`): Telemetry frame
- **x_prev** (`<class 'numpy.ndarray'>`): Previous state estimate
- **dt** (`<class 'float'>`): Time step

**Returns**: `<class 'numpy.ndarray'>` - Updated state estimate

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\flight_data.py:324](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\flight_data.py#L324)*

##### `reconstruct`

```python
def reconstruct(self, record: flight_data.FlightRecord) -> Tuple[numpy.ndarray, numpy.ndarray]
```

Reconstruct trajectory from flight record.

**Parameters:**

- **record** (`<class 'flight_data.FlightRecord'>`): Flight record with telemetry

**Returns**: `typing.Tuple[numpy.ndarray, numpy.ndarray]` - (times, states) arrays

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\flight_data.py:378](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\flight_data.py#L378)*

##### `state_transition`

```python
def state_transition(self, x: numpy.ndarray, dt: float) -> numpy.ndarray
```

State transition function.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\flight_data.py:309](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\flight_data.py#L309)*

## Functions

### `compute_reconstruction_error`

```python
def compute_reconstruction_error(reconstruction: flight_data.TrajectoryReconstruction, flight_record: flight_data.FlightRecord) -> Dict[str, float]
```

Compute reconstruction error metrics.

**Parameters:**

- **reconstruction** (`<class 'flight_data.TrajectoryReconstruction'>`): Reconstructed trajectory
- **flight_record** (`<class 'flight_data.FlightRecord'>`): Original flight record

**Returns**: `typing.Dict[str, float]` - Error metrics

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\flight_data.py:640](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\flight_data.py#L640)*

### `create_synthetic_flight_record`

```python
def create_synthetic_flight_record(trajectory: numpy.ndarray, times: numpy.ndarray, add_noise: bool = True) -> flight_data.FlightRecord
```

Create a synthetic flight record for testing.

**Parameters:**

- **trajectory** (`<class 'numpy.ndarray'>`): (N, 6+) state trajectory [pos, vel, ...]
- **times** (`<class 'numpy.ndarray'>`): Time array
- **add_noise** (`<class 'bool'>`): Whether to add sensor noise

**Returns**: `<class 'flight_data.FlightRecord'>` - Synthetic FlightRecord

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\flight_data.py:701](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\flight_data.py#L701)*

### `parse_telemetry`

```python
def parse_telemetry(source: Union[str, pathlib.Path, bytes], format: flight_data.TelemetryFormat = <TelemetryFormat.CSV: 'csv'>) -> flight_data.FlightRecord
```

Parse telemetry data from various formats.

**Parameters:**

- **source** (`typing.Union[str, pathlib.Path, bytes]`): File path, string data, or bytes
- **format** (`<enum 'TelemetryFormat'>`): Telemetry format

**Returns**: `<class 'flight_data.FlightRecord'>` - FlightRecord with parsed data

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\flight_data.py:426](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\flight_data.py#L426)*

### `validate_against_flight`

```python
def validate_against_flight(model_trajectory: numpy.ndarray, model_times: numpy.ndarray, flight_record: flight_data.FlightRecord) -> Dict[str, float]
```

Validate model predictions against flight data.

**Parameters:**

- **model_trajectory** (`<class 'numpy.ndarray'>`): Model predicted states
- **model_times** (`<class 'numpy.ndarray'>`): Model time points
- **flight_record** (`<class 'flight_data.FlightRecord'>`): Flight record to compare against

**Returns**: `typing.Dict[str, float]` - Dict of validation metrics

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\flight_data.py:577](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\flight_data.py#L577)*

### `validate_flight_data_module`

```python
def validate_flight_data_module()
```

Validate flight data module.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\flight_data.py:744](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\simulation\flight_data.py#L744)*
