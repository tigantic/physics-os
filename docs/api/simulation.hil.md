# Module `simulation.hil`

Hardware-in-the-Loop (HIL) Simulation Interface ================================================

Provides interfaces for HIL testing of guidance algorithms with
realistic sensor noise, actuator dynamics, and communication latency.

Architecture:
    ┌─────────────────┐     ┌──────────────────┐
    │  Flight Computer │◄───►│   HIL Interface   │
    │  (Target HW)     │     │   (This Module)   │
    └─────────────────┘     └────────┬─────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
    ┌─────────┐               ┌─────────────┐             ┌───────────┐
    │ Sensors │               │ Environment │             │ Actuators │
    │ Models  │               │  Dynamics   │             │  Models   │
    └─────────┘               └─────────────┘             └───────────┘

Sensor Models:
    - IMU: Accelerometer + Gyroscope with bias, noise, drift
    - GPS: Position with ionospheric delay and multipath
    - Air Data: Pitot-static with temperature effects
    - Star Tracker: Attitude with sky visibility

Actuator Models:
    - Control Surfaces: Rate limits, position limits, hysteresis
    - Reaction Control: Thrust uncertainty, ignition delay
    - Propulsion: Throttle lag, thrust vectoring

Timing:
    - Synchronous mode: Lock-step with simulation
    - Asynchronous mode: Real-time with hardware clock
    - Accelerated mode: Faster than real-time for Monte Carlo

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `ActuatorModel`(ABC)

Abstract base class for actuator models.

#### Methods

##### `__init__`

```python
def __init__(self, actuator_type: hil.ActuatorType, rate_limit: float = 100.0, position_limit_min: float = -30.0, position_limit_max: float = 30.0, latency_ms: float = 2.0)
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py:389](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py#L389)*

##### `update`

```python
def update(self, command: float, dt: float) -> float
```

Update actuator position based on command.

**Parameters:**

- **command** (`<class 'float'>`): Commanded position
- **dt** (`<class 'float'>`): Time step

**Returns**: `<class 'float'>` - Actual position after dynamics

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py:406](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py#L406)*

### class `ActuatorType`(Enum)

Actuator type classification.

### class `AirDataSensor`(SensorModel)

Air data system sensor model.

Models pitot-static system with:
- Static pressure
- Dynamic pressure
- Total temperature
- Angle of attack vanes
- Sideslip angle vanes

#### Methods

##### `__init__`

```python
def __init__(self, update_rate_hz: float = 100.0, latency_ms: float = 5.0)
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py:318](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py#L318)*

##### `measure`

```python
def measure(self, true_state: Dict[str, float], t: float) -> Dict[str, float]
```

Generate air data measurement.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py:346](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py#L346)*

### class `ControlSurface`(ActuatorModel)

Aerodynamic control surface actuator model.

Models:
- Servo dynamics (first/second order)
- Rate limits
- Position limits
- Hysteresis
- Load-dependent performance

#### Methods

##### `__init__`

```python
def __init__(self, name: str = 'elevator', rate_limit_deg_s: float = 60.0, position_limit_deg: float = 30.0, bandwidth_hz: float = 10.0, latency_ms: float = 5.0)
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py:433](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py#L433)*

##### `update`

```python
def update(self, command: float, dt: float) -> float
```

Update control surface position with second-order dynamics.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py:454](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py#L454)*

### class `GPSSensor`(SensorModel)

GPS receiver sensor model.

Models GPS with:
- Position accuracy (CEP)
- Velocity accuracy
- Dilution of precision (DOP)
- Ionospheric/tropospheric delays
- Signal acquisition/loss

#### Methods

##### `__init__`

```python
def __init__(self, update_rate_hz: float = 10.0, latency_ms: float = 50.0)
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py:238](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py#L238)*

##### `measure`

```python
def measure(self, true_state: Dict[str, float], t: float) -> Dict[str, float]
```

Generate GPS measurement.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py:267](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py#L267)*

### class `HILConfig`

Configuration for HIL simulation.

#### Attributes

- **mode** (`<enum 'HILMode'>`): 
- **dt_s** (`<class 'float'>`): 
- **sensor_latency_ms** (`<class 'float'>`): 
- **actuator_latency_ms** (`<class 'float'>`): 
- **communication_latency_ms** (`<class 'float'>`): 
- **enable_sensor_noise** (`<class 'bool'>`): 
- **enable_actuator_dynamics** (`<class 'bool'>`): 
- **time_acceleration** (`<class 'float'>`): 
- **log_telemetry** (`<class 'bool'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, mode: hil.HILMode = <HILMode.SYNCHRONOUS: 'sync'>, dt_s: float = 0.001, sensor_latency_ms: float = 1.0, actuator_latency_ms: float = 2.0, communication_latency_ms: float = 0.5, enable_sensor_noise: bool = True, enable_actuator_dynamics: bool = True, time_acceleration: float = 1.0, log_telemetry: bool = True) -> None
```

### class `HILInterface`

Hardware-in-the-Loop simulation interface.

Manages the complete HIL loop including:
- Sensor data generation
- Actuator command processing
- Timing and synchronization
- Telemetry logging

#### Methods

##### `__init__`

```python
def __init__(self, config: hil.HILConfig = None)
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py:550](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py#L550)*

##### `add_actuator`

```python
def add_actuator(self, name: str, actuator: hil.ActuatorModel)
```

Add an actuator to the HIL interface.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py:567](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py#L567)*

##### `add_sensor`

```python
def add_sensor(self, name: str, sensor: hil.SensorModel)
```

Add a sensor to the HIL interface.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py:563](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py#L563)*

##### `get_sensor_data`

```python
def get_sensor_data(self, true_state: Dict[str, float]) -> Dict[str, Dict]
```

Get all sensor measurements.

**Parameters:**

- **true_state** (`typing.Dict[str, float]`): True vehicle state

**Returns**: `typing.Dict[str, typing.Dict]` - Dict of sensor name -> measurement

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py:571](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py#L571)*

##### `get_telemetry`

```python
def get_telemetry(self) -> List[Dict]
```

Get logged telemetry.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py:708](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py#L708)*

##### `reset`

```python
def reset(self)
```

Reset HIL state.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py:699](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py#L699)*

##### `run_realtime`

```python
def run_realtime(self, get_true_state: Callable[[], Dict[str, float]], get_commands: Callable[[Dict], Dict[str, float]], duration_s: float)
```

Run HIL in real-time mode.

**Parameters:**

- **get_true_state** (`typing.Callable[[], typing.Dict[str, float]]`): Callback to get current true state
- **get_commands** (`typing.Callable[[typing.Dict], typing.Dict[str, float]]`): Callback to get actuator commands from sensor data
- **duration_s** (`<class 'float'>`): Duration to run

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py:646](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py#L646)*

##### `set_actuator_commands`

```python
def set_actuator_commands(self, commands: Dict[str, float], dt: float) -> Dict[str, float]
```

Set actuator commands and get actual positions.

**Parameters:**

- **commands** (`typing.Dict[str, float]`): Dict of actuator name -> command
- **dt** (`<class 'float'>`): Time step

**Returns**: `typing.Dict[str, float]` - Dict of actuator name -> actual position

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py:588](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py#L588)*

##### `step`

```python
def step(self, true_state: Dict[str, float], commands: Dict[str, float]) -> Tuple[Dict[str, Dict], Dict[str, float]]
```

Execute one HIL step.

**Parameters:**

- **true_state** (`typing.Dict[str, float]`): True vehicle state
- **commands** (`typing.Dict[str, float]`): Actuator commands

**Returns**: `typing.Tuple[typing.Dict[str, typing.Dict], typing.Dict[str, float]]` - (sensor_measurements, actuator_positions)

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py:609](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py#L609)*

##### `stop`

```python
def stop(self)
```

Stop real-time HIL.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py:695](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py#L695)*

### class `HILMode`(Enum)

HIL timing mode.

### class `IMUSensor`(SensorModel)

Inertial Measurement Unit sensor model.

Models a 6-DOF IMU with:
- 3-axis accelerometer
- 3-axis gyroscope
- Bias instability and drift
- Temperature sensitivity

#### Methods

##### `__init__`

```python
def __init__(self, update_rate_hz: float = 400.0, latency_ms: float = 0.5)
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py:167](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py#L167)*

##### `measure`

```python
def measure(self, true_state: Dict[str, float], t: float) -> Dict[str, float]
```

Generate IMU measurement.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py:198](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py#L198)*

### class `SensorModel`(ABC)

Abstract base class for sensor models.

#### Methods

##### `__init__`

```python
def __init__(self, sensor_type: hil.SensorType, update_rate_hz: float = 100.0, latency_ms: float = 1.0)
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py:106](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py#L106)*

##### `apply_noise`

```python
def apply_noise(self, value: float, noise: hil.SensorNoise, dt: float) -> float
```

Apply noise model to measurement.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py:132](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py#L132)*

##### `measure`

```python
def measure(self, true_state: Dict[str, float], t: float) -> Dict[str, float]
```

Generate sensor measurement from true state.

**Parameters:**

- **true_state** (`typing.Dict[str, float]`): True vehicle state
- **t** (`<class 'float'>`): Current simulation time

**Returns**: `typing.Dict[str, float]` - Sensor measurement with noise

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py:118](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py#L118)*

### class `SensorNoise`

Sensor noise parameters.

#### Attributes

- **white_noise_std** (`<class 'float'>`): 
- **bias** (`<class 'float'>`): 
- **bias_drift_rate** (`<class 'float'>`): 
- **quantization** (`<class 'float'>`): 
- **saturation_min** (`<class 'float'>`): 
- **saturation_max** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, white_noise_std: float = 0.0, bias: float = 0.0, bias_drift_rate: float = 0.0, quantization: float = 0.0, saturation_min: float = -10000000000.0, saturation_max: float = 10000000000.0) -> None
```

### class `SensorType`(Enum)

Sensor type classification.

### class `ThrustActuator`(ActuatorModel)

Propulsion thrust actuator model.

Models:
- Throttle lag (first order)
- Ignition delay
- Thrust uncertainty
- Thrust vectoring

#### Methods

##### `__init__`

```python
def __init__(self, max_thrust_N: float = 100000.0, throttle_lag_s: float = 0.1, ignition_delay_s: float = 0.05, thrust_uncertainty_pct: float = 3.0)
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py:491](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py#L491)*

##### `ignite`

```python
def ignite(self, t: float)
```

Command ignition.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py:512](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py#L512)*

##### `update`

```python
def update(self, command: float, dt: float) -> float
```

Update thrust level.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py:516](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py#L516)*

## Functions

### `create_vehicle_actuators`

```python
def create_vehicle_actuators(vehicle_class: str = 'hypersonic') -> Dict[str, hil.ActuatorModel]
```

Create a standard actuator suite for a vehicle class.

**Parameters:**

- **vehicle_class** (`<class 'str'>`): 'hypersonic', 'missile', 'aircraft'

**Returns**: `typing.Dict[str, hil.ActuatorModel]` - Dict of actuator name -> ActuatorModel

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py:740](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py#L740)*

### `create_vehicle_sensors`

```python
def create_vehicle_sensors(vehicle_class: str = 'hypersonic') -> Dict[str, hil.SensorModel]
```

Create a standard sensor suite for a vehicle class.

**Parameters:**

- **vehicle_class** (`<class 'str'>`): 'hypersonic', 'missile', 'aircraft'

**Returns**: `typing.Dict[str, hil.SensorModel]` - Dict of sensor name -> SensorModel

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py:713](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py#L713)*

### `validate_hil_module`

```python
def validate_hil_module()
```

Validate HIL module.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py:786](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\hil.py#L786)*
