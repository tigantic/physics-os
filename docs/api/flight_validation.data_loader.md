# Module `flight_validation.data_loader`

Flight data loading and parsing utilities.

This module provides tools for loading flight data from various
sources and formats for validation against CFD simulations.

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `AerodynamicData`

Aerodynamic forces and moments.

#### Attributes

- **timestamp** (`<class 'float'>`): 
- **cl** (`<class 'float'>`): 
- **cd** (`<class 'float'>`): 
- **cy** (`<class 'float'>`): 
- **cm** (`<class 'float'>`): 
- **cn** (`<class 'float'>`): 
- **croll** (`<class 'float'>`): 
- **cp_distribution** (`typing.Optional[numpy.ndarray]`): 
- **heat_flux_distribution** (`typing.Optional[numpy.ndarray]`): 
- **cl_uncertainty** (`<class 'float'>`): 
- **cd_uncertainty** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, timestamp: float, cl: float = 0.0, cd: float = 0.0, cy: float = 0.0, cm: float = 0.0, cn: float = 0.0, croll: float = 0.0, cp_distribution: Optional[numpy.ndarray] = None, heat_flux_distribution: Optional[numpy.ndarray] = None, cl_uncertainty: float = 0.0, cd_uncertainty: float = 0.0) -> None
```

##### `to_dict`

```python
def to_dict(self) -> Dict[str, float]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\flight_validation\data_loader.py:114](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\flight_validation\data_loader.py#L114)*

### class `FlightCondition`

Flight condition at a specific time.

#### Attributes

- **timestamp** (`<class 'float'>`): 
- **altitude_m** (`<class 'float'>`): 
- **mach_number** (`<class 'float'>`): 
- **velocity_m_s** (`<class 'float'>`): 
- **pressure_pa** (`<class 'float'>`): 
- **temperature_k** (`<class 'float'>`): 
- **density_kg_m3** (`<class 'float'>`): 
- **angle_of_attack_deg** (`<class 'float'>`): 
- **sideslip_angle_deg** (`<class 'float'>`): 
- **roll_angle_deg** (`<class 'float'>`): 
- **latitude_deg** (`<class 'float'>`): 
- **longitude_deg** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, timestamp: float, altitude_m: float = 0.0, mach_number: float = 0.0, velocity_m_s: float = 0.0, pressure_pa: float = 101325.0, temperature_k: float = 288.15, density_kg_m3: float = 1.225, angle_of_attack_deg: float = 0.0, sideslip_angle_deg: float = 0.0, roll_angle_deg: float = 0.0, latitude_deg: float = 0.0, longitude_deg: float = 0.0) -> None
```

##### `dynamic_pressure_pa`

```python
def dynamic_pressure_pa(self) -> float
```

Calculate dynamic pressure.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\flight_validation\data_loader.py:86](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\flight_validation\data_loader.py#L86)*

##### `to_dict`

```python
def to_dict(self) -> Dict[str, float]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\flight_validation\data_loader.py:72](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\flight_validation\data_loader.py#L72)*

### class `FlightDataFormat`(Enum)

Flight data file formats.

### class `FlightDataLoader`

Loader for flight data from various sources.

#### Methods

##### `__init__`

```python
def __init__(self)
```

Initialize loader.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\flight_validation\data_loader.py:216](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\flight_validation\data_loader.py#L216)*

##### `load`

```python
def load(self, path: Union[str, pathlib.Path], format: Optional[data_loader.FlightDataFormat] = None, source: data_loader.FlightDataSource = <FlightDataSource.FLIGHT_TEST: 2>) -> data_loader.FlightRecord
```

Load flight data from file.

**Parameters:**

- **path** (`typing.Union[str, pathlib.Path]`): Path to data file
- **format** (`typing.Optional[data_loader.FlightDataFormat]`): Data format (auto-detected if not specified)
- **source** (`<enum 'FlightDataSource'>`): Data source type

**Returns**: `<class 'data_loader.FlightRecord'>` - FlightRecord with loaded data

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\flight_validation\data_loader.py:223](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\flight_validation\data_loader.py#L223)*

### class `FlightDataSource`(Enum)

Sources of flight data.

### class `FlightRecord`

Complete flight data record.

#### Attributes

- **record_id** (`<class 'str'>`): 
- **source** (`<enum 'FlightDataSource'>`): 
- **start_time** (`<class 'float'>`): 
- **end_time** (`<class 'float'>`): 
- **conditions** (`typing.List[data_loader.FlightCondition]`): 
- **aero_data** (`typing.List[data_loader.AerodynamicData]`): 
- **sensor_readings** (`typing.Dict[str, typing.List[data_loader.SensorReading]]`): 
- **vehicle_name** (`<class 'str'>`): 
- **test_name** (`<class 'str'>`): 
- **date** (`<class 'str'>`): 
- **notes** (`<class 'str'>`): 
- **quality_score** (`<class 'float'>`): 
- **calibration_status** (`<class 'str'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, record_id: str, source: data_loader.FlightDataSource, start_time: float = 0.0, end_time: float = 0.0, conditions: List[data_loader.FlightCondition] = <factory>, aero_data: List[data_loader.AerodynamicData] = <factory>, sensor_readings: Dict[str, List[data_loader.SensorReading]] = <factory>, vehicle_name: str = '', test_name: str = '', date: str = '', notes: str = '', quality_score: float = 1.0, calibration_status: str = 'unknown') -> None
```

##### `get_aero_at_time`

```python
def get_aero_at_time(self, t: float) -> Optional[data_loader.AerodynamicData]
```

Get aerodynamic data at specific time.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\flight_validation\data_loader.py:187](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\flight_validation\data_loader.py#L187)*

##### `get_condition_at_time`

```python
def get_condition_at_time(self, t: float) -> Optional[data_loader.FlightCondition]
```

Get flight condition at specific time (interpolated).

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\flight_validation\data_loader.py:159](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\flight_validation\data_loader.py#L159)*

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\flight_validation\data_loader.py:196](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\flight_validation\data_loader.py#L196)*

### class `SensorReading`

Single sensor reading.

#### Attributes

- **timestamp** (`<class 'float'>`): 
- **value** (`<class 'float'>`): 
- **sensor_id** (`<class 'str'>`): 
- **unit** (`<class 'str'>`): 
- **uncertainty** (`<class 'float'>`): 
- **quality_flag** (`<class 'int'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, timestamp: float, value: float, sensor_id: str, unit: str = '', uncertainty: float = 0.0, quality_flag: int = 0) -> None
```

## Functions

### `load_flight_data`

```python
def load_flight_data(path: Union[str, pathlib.Path], source: data_loader.FlightDataSource = <FlightDataSource.FLIGHT_TEST: 2>) -> data_loader.FlightRecord
```

Load flight data from file.

**Parameters:**

- **path** (`typing.Union[str, pathlib.Path]`): Path to data file
- **source** (`<enum 'FlightDataSource'>`): Data source type

**Returns**: `<class 'data_loader.FlightRecord'>` - FlightRecord with loaded data

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\flight_validation\data_loader.py:401](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\flight_validation\data_loader.py#L401)*

### `parse_telemetry`

```python
def parse_telemetry(data: Union[str, bytes, Dict], format: str = 'json') -> data_loader.FlightRecord
```

Parse telemetry data from string/bytes.

**Parameters:**

- **data** (`typing.Union[str, bytes, typing.Dict]`): Telemetry data
- **format** (`<class 'str'>`): Data format ("json", "binary", etc.)

**Returns**: `<class 'data_loader.FlightRecord'>` - FlightRecord with parsed data

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\flight_validation\data_loader.py:419](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\flight_validation\data_loader.py#L419)*
