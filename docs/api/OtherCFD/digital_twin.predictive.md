# Module `digital_twin.predictive`

Predictive maintenance for hypersonic vehicle digital twins.

This module provides remaining useful life (RUL) estimation,
maintenance scheduling optimization, and reliability analysis
for hypersonic vehicle subsystems.

Key capabilities:
    - RUL estimation using physics-based and data-driven methods
    - Optimal maintenance scheduling considering mission constraints
    - Reliability and risk quantification
    - Component-level and system-level analysis

Author: Tigantic Holdings LLC

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `ComponentState`

State of a vehicle component.

#### Attributes

- **component_id** (`<class 'str'>`): 
- **component_type** (`<enum 'ComponentType'>`): 
- **age_hours** (`<class 'float'>`): 
- **cycles** (`<class 'int'>`): 
- **damage_index** (`<class 'float'>`): 
- **failure_modes** (`typing.List[predictive.FailureMode]`): 
- **rul_mean** (`<class 'float'>`): 
- **rul_lower** (`<class 'float'>`): 
- **rul_upper** (`<class 'float'>`): 
- **last_maintenance** (`typing.Optional[float]`): 
- **next_scheduled** (`typing.Optional[float]`): 
- **maintenance_count** (`<class 'int'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, component_id: str, component_type: predictive.ComponentType, age_hours: float = 0.0, cycles: int = 0, damage_index: float = 0.0, failure_modes: List[predictive.FailureMode] = <factory>, rul_mean: float = inf, rul_lower: float = inf, rul_upper: float = inf, last_maintenance: Optional[float] = None, next_scheduled: Optional[float] = None, maintenance_count: int = 0) -> None
```

### class `ComponentType`(Enum)

Types of vehicle components.

### class `FailureMode`(Enum)

Failure modes for vehicle components.

### class `MaintenanceAction`(Enum)

Types of maintenance actions.

### class `MaintenanceConfig`

Configuration for predictive maintenance.

#### Attributes

- **rul_confidence_level** (`<class 'float'>`): 
- **min_rul_threshold** (`<class 'float'>`): 
- **planning_horizon** (`<class 'float'>`): 
- **min_maintenance_interval** (`<class 'float'>`): 
- **cost_inspection** (`<class 'float'>`): 
- **cost_minor_repair** (`<class 'float'>`): 
- **cost_major_repair** (`<class 'float'>`): 
- **cost_replacement** (`<class 'float'>`): 
- **cost_failure** (`<class 'float'>`): 
- **acceptable_failure_probability** (`<class 'float'>`): 
- **warning_failure_probability** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, rul_confidence_level: float = 0.95, min_rul_threshold: float = 10.0, planning_horizon: float = 1000.0, min_maintenance_interval: float = 24.0, cost_inspection: float = 1000.0, cost_minor_repair: float = 10000.0, cost_major_repair: float = 100000.0, cost_replacement: float = 500000.0, cost_failure: float = 10000000.0, acceptable_failure_probability: float = 1e-06, warning_failure_probability: float = 1e-07) -> None
```

### class `MaintenanceSchedule`

Optimized maintenance schedule.

#### Attributes

- **component_id** (`<class 'str'>`): 
- **scheduled_time** (`<class 'float'>`): 
- **action** (`<enum 'MaintenanceAction'>`): 
- **priority** (`<class 'int'>`): 
- **estimated_cost** (`<class 'float'>`): 
- **estimated_downtime** (`<class 'float'>`): 
- **reason** (`<class 'str'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, component_id: str, scheduled_time: float, action: predictive.MaintenanceAction, priority: int, estimated_cost: float, estimated_downtime: float, reason: str) -> None
```

### class `MaintenanceScheduler`

Optimal maintenance scheduling.

Determines optimal maintenance timing and actions based on
component health, RUL estimates, cost, and constraints.

#### Methods

##### `__init__`

```python
def __init__(self, config: predictive.MaintenanceConfig)
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\digital_twin\predictive.py:275](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\digital_twin\predictive.py#L275)*

##### `schedule_maintenance`

```python
def schedule_maintenance(self, components: List[predictive.ComponentState], constraints: Optional[Dict[str, Any]] = None) -> List[predictive.MaintenanceSchedule]
```

Generate optimal maintenance schedule.

**Parameters:**

- **components** (`typing.List[predictive.ComponentState]`): List of component states
- **constraints** (`typing.Optional[typing.Dict[str, typing.Any]]`): Scheduling constraints

**Returns**: `typing.List[predictive.MaintenanceSchedule]` - List of scheduled maintenance actions

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\digital_twin\predictive.py:279](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\digital_twin\predictive.py#L279)*

### class `PredictiveMaintenance`

Complete predictive maintenance system.

Integrates RUL estimation, reliability analysis, and
maintenance scheduling for vehicle fleet management.

#### Methods

##### `__init__`

```python
def __init__(self, config: predictive.MaintenanceConfig)
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\digital_twin\predictive.py:408](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\digital_twin\predictive.py#L408)*

##### `generate_maintenance_plan`

```python
def generate_maintenance_plan(self) -> List[predictive.MaintenanceSchedule]
```

Generate maintenance plan for all components.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\digital_twin\predictive.py:499](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\digital_twin\predictive.py#L499)*

##### `get_fleet_health`

```python
def get_fleet_health(self) -> Dict[str, Any]
```

Get overall fleet health summary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\digital_twin\predictive.py:477](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\digital_twin\predictive.py#L477)*

##### `record_maintenance`

```python
def record_maintenance(self, component_id: str, action: predictive.MaintenanceAction, timestamp: float, notes: str = '')
```

Record completed maintenance action.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\digital_twin\predictive.py:445](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\digital_twin\predictive.py#L445)*

##### `register_component`

```python
def register_component(self, component_id: str, component_type: predictive.ComponentType) -> predictive.ComponentState
```

Register a new component for monitoring.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\digital_twin\predictive.py:417](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\digital_twin\predictive.py#L417)*

##### `update_component`

```python
def update_component(self, component_id: str, delta_hours: float = 0.0, delta_cycles: int = 0, damage_increment: float = 0.0, operating_conditions: Optional[Dict[str, float]] = None)
```

Update component state with new usage data.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\digital_twin\predictive.py:427](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\digital_twin\predictive.py#L427)*

### class `RULEstimator`

Remaining Useful Life estimator.

Combines physics-based degradation models with data-driven
approaches for accurate RUL prediction with uncertainty.

#### Methods

##### `__init__`

```python
def __init__(self, config: predictive.MaintenanceConfig)
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\digital_twin\predictive.py:127](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\digital_twin\predictive.py#L127)*

##### `estimate_failure_probability`

```python
def estimate_failure_probability(self, component: predictive.ComponentState, time_horizon: float) -> float
```

Estimate probability of failure within time horizon.

Uses Weibull distribution based on failure modes and age.

**Parameters:**

- **component** (`<class 'predictive.ComponentState'>`): Component state
- **time_horizon** (`<class 'float'>`): Time horizon in hours

**Returns**: `<class 'float'>` - Probability of failure

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\digital_twin\predictive.py:230](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\digital_twin\predictive.py#L230)*

##### `estimate_rul`

```python
def estimate_rul(self, component: predictive.ComponentState, operating_conditions: Optional[Dict[str, float]] = None) -> Tuple[float, float, float]
```

Estimate remaining useful life for a component.

**Parameters:**

- **component** (`<class 'predictive.ComponentState'>`): Current component state
- **operating_conditions** (`typing.Optional[typing.Dict[str, float]]`): Current/expected operating conditions

**Returns**: `typing.Tuple[float, float, float]` - Tuple of (mean RUL, lower bound, upper bound) in hours

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\digital_twin\predictive.py:174](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\digital_twin\predictive.py#L174)*

## Functions

### `compute_reliability`

```python
def compute_reliability(failure_rate: float, time: float) -> float
```

Compute reliability (survival probability) assuming constant failure rate.

R(t) = exp(-lambda * t)

**Parameters:**

- **failure_rate** (`<class 'float'>`): Failure rate (per hour)
- **time** (`<class 'float'>`): Time horizon (hours)

**Returns**: `<class 'float'>` - Reliability (0-1)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\digital_twin\predictive.py:525](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\digital_twin\predictive.py#L525)*

### `estimate_remaining_life`

```python
def estimate_remaining_life(damage: float, damage_rate: float, threshold: float = 1.0) -> float
```

Simple RUL estimation from current damage and rate.

**Parameters:**

- **damage** (`<class 'float'>`): Current damage index (0-1)
- **damage_rate** (`<class 'float'>`): Damage accumulation rate per hour
- **threshold** (`<class 'float'>`): Failure threshold

**Returns**: `<class 'float'>` - Remaining life in hours

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\digital_twin\predictive.py:504](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\digital_twin\predictive.py#L504)*

### `optimize_maintenance_schedule`

```python
def optimize_maintenance_schedule(components: List[predictive.ComponentState], config: predictive.MaintenanceConfig, objective: str = 'cost') -> List[predictive.MaintenanceSchedule]
```

Optimize maintenance schedule for minimum cost or maximum availability.

**Parameters:**

- **components** (`typing.List[predictive.ComponentState]`): List of component states
- **config** (`<class 'predictive.MaintenanceConfig'>`): Maintenance configuration
- **objective** (`<class 'str'>`): 'cost' or 'availability'

**Returns**: `typing.List[predictive.MaintenanceSchedule]` - Optimized maintenance schedule

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\digital_twin\predictive.py:541](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\digital_twin\predictive.py#L541)*

### `test_predictive_maintenance`

```python
def test_predictive_maintenance()
```

Test predictive maintenance module.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\digital_twin\predictive.py:570](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\digital_twin\predictive.py#L570)*
