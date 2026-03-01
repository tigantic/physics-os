# Module `guidance.controller`

Guidance Controller ===================

Physics-aware guidance for hypersonic vehicles with trajectory tracking,
constraint handling, and real-time CFD integration.

Guidance Laws:
    - Bank-to-turn for hypersonic glide
    - Proportional navigation for terminal guidance
    - Energy management for range control
    - Predictive guidance with forward simulation

Constraints:
    - Thermal: q̇ < q̇_max, Q < Q_max (heat load)
    - Structural: g_n < g_max, q_dyn < q_max
    - Corridor: altitude bounds, skip-out prevention
    - Range: TAEM energy management

Real-Time Considerations:
    - 100 Hz update rate for guidance loop
    - Pre-computed lookup tables
    - GPU-accelerated trajectory prediction
    - Graceful degradation under overload

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `ConstraintType`(Enum)

Constraint type classification.

### class `CorridorBounds`

Entry corridor definition.

#### Attributes

- **min_altitude_m** (`<class 'float'>`): 
- **max_altitude_m** (`<class 'float'>`): 
- **min_velocity_m_s** (`<class 'float'>`): 
- **max_heat_rate_W_cm2** (`<class 'float'>`): 
- **max_heat_load_J_cm2** (`<class 'float'>`): 
- **max_g_load** (`<class 'float'>`): 
- **max_bank_angle_rad** (`<class 'float'>`): 
- **max_aoa_rad** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, min_altitude_m: float = 15000.0, max_altitude_m: float = 120000.0, min_velocity_m_s: float = 100.0, max_heat_rate_W_cm2: float = 100.0, max_heat_load_J_cm2: float = 10000.0, max_g_load: float = 3.0, max_bank_angle_rad: float = 1.3962634015954636, max_aoa_rad: float = 0.6981317007977318) -> None
```

### class `GuidanceCommand`

Output command from guidance law.

#### Attributes

- **bank_angle_rad** (`<class 'float'>`): 
- **angle_of_attack_rad** (`<class 'float'>`): 
- **bank_rate_rad_s** (`<class 'float'>`): 
- **mode** (`<enum 'GuidanceMode'>`): 
- **constraint_margin** (`<class 'float'>`): 
- **predicted_range_m** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, bank_angle_rad: float = 0.0, angle_of_attack_rad: float = 0.0, bank_rate_rad_s: float = 0.0, mode: controller.GuidanceMode = <GuidanceMode.ENTRY: 'entry'>, constraint_margin: float = 1.0, predicted_range_m: float = 0.0) -> None
```

##### `to_controls`

```python
def to_controls(self) -> Dict[str, float]
```

Convert to control surface deflections.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py:71](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py#L71)*

### class `GuidanceController`

Main guidance controller with constraint handling and CFD integration.

Implements a predictor-corrector guidance scheme with:
- Reference trajectory generation
- Online trajectory prediction
- Constraint monitoring and handling
- CFD-based aerodynamic lookup

#### Methods

##### `__init__`

```python
def __init__(self, corridor: controller.CorridorBounds = None, target: controller.WaypointTarget = None, dt: float = 0.01)
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py:290](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py#L290)*

##### `apply_constraint_limiting`

```python
def apply_constraint_limiting(self, command: controller.GuidanceCommand, state: ontic.guidance.trajectory.VehicleState) -> controller.GuidanceCommand
```

Modify command to respect constraints.

Uses a priority-based approach:
1. Thermal rate (reduce bank to reduce heating)
2. G-load (reduce bank and AoA)
3. Altitude (adjust vertical control)

**Parameters:**

- **command** (`<class 'controller.GuidanceCommand'>`): Nominal guidance command
- **state** (`<class 'ontic.guidance.trajectory.VehicleState'>`): Current vehicle state

**Returns**: `<class 'controller.GuidanceCommand'>` - Constraint-limited command

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py:492](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py#L492)*

##### `compute_guidance`

```python
def compute_guidance(self, state: ontic.guidance.trajectory.VehicleState, target: Optional[controller.WaypointTarget] = None) -> controller.GuidanceCommand
```

Main guidance computation loop.

**Parameters:**

- **state** (`<class 'ontic.guidance.trajectory.VehicleState'>`): Current vehicle state
- **target** (`typing.Optional[controller.WaypointTarget]`): Optional override for target

**Returns**: `<class 'controller.GuidanceCommand'>` - Guidance command

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py:565](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py#L565)*

##### `estimate_g_load`

```python
def estimate_g_load(self, state: ontic.guidance.trajectory.VehicleState, atm: ontic.guidance.trajectory.AtmosphericModel, bank_angle: float, aoa: float) -> float
```

Estimate normal g-load.

**Parameters:**

- **state** (`<class 'ontic.guidance.trajectory.VehicleState'>`): Vehicle state
- **atm** (`<class 'ontic.guidance.trajectory.AtmosphericModel'>`): Atmospheric conditions
- **bank_angle** (`<class 'float'>`): Bank angle in radians
- **aoa** (`<class 'float'>`): Angle of attack in radians

**Returns**: `<class 'float'>` - Normal load factor (g's)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py:416](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py#L416)*

##### `estimate_heating`

```python
def estimate_heating(self, state: ontic.guidance.trajectory.VehicleState, atm: ontic.guidance.trajectory.AtmosphericModel) -> float
```

Estimate stagnation point heat rate.

Uses Sutton-Graves correlation for convective heating.

**Parameters:**

- **state** (`<class 'ontic.guidance.trajectory.VehicleState'>`): Vehicle state
- **atm** (`<class 'ontic.guidance.trajectory.AtmosphericModel'>`): Atmospheric conditions

**Returns**: `<class 'float'>` - Heat rate in W/cm²

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py:382](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py#L382)*

##### `lookup_aerodynamics`

```python
def lookup_aerodynamics(self, mach: float, alpha_deg: float) -> Tuple[float, float, float]
```

Look up aerodynamic coefficients.

**Parameters:**

- **mach** (`<class 'float'>`): Mach number
- **alpha_deg** (`<class 'float'>`): Angle of attack in degrees

**Returns**: `typing.Tuple[float, float, float]` - (CL, CD, Cm)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py:348](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py#L348)*

##### `predict_trajectory`

```python
def predict_trajectory(self, state: ontic.guidance.trajectory.VehicleState, command: controller.GuidanceCommand, horizon_s: float = 30.0) -> List[ontic.guidance.trajectory.VehicleState]
```

Predict future trajectory with current command.

**Parameters:**

- **state** (`<class 'ontic.guidance.trajectory.VehicleState'>`): Current state
- **command** (`<class 'controller.GuidanceCommand'>`): Command to hold
- **horizon_s** (`<class 'float'>`): Prediction horizon

**Returns**: `typing.List[ontic.guidance.trajectory.VehicleState]` - Predicted trajectory

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py:541](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py#L541)*

##### `reset`

```python
def reset(self)
```

Reset controller state for new trajectory.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py:610](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py#L610)*

##### `set_aero_table`

```python
def set_aero_table(self, table: Dict)
```

Set aerodynamic lookup table from CFD.

**Parameters:**

- **table** (`typing.Dict`): Dict with keys (Mach, alpha_deg) -> (CL, CD, Cm)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py:339](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py#L339)*

##### `update_constraints`

```python
def update_constraints(self, state: ontic.guidance.trajectory.VehicleState, command: controller.GuidanceCommand)
```

Update constraint values based on current state.

**Parameters:**

- **state** (`<class 'ontic.guidance.trajectory.VehicleState'>`): Current vehicle state
- **command** (`<class 'controller.GuidanceCommand'>`): Current guidance command

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py:456](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py#L456)*

### class `GuidanceMode`(Enum)

Guidance mode phases.

### class `TrajectoryConstraint`

Definition of a trajectory constraint.

#### Attributes

- **constraint_type** (`<enum 'ConstraintType'>`): 
- **max_value** (`<class 'float'>`): 
- **current_value** (`<class 'float'>`): 
- **margin** (`<class 'float'>`): 

#### Properties

##### `is_active`

```python
def is_active(self) -> bool
```

Whether constraint is active (within margin of limit).

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py:102](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py#L102)*

##### `relative_margin`

```python
def relative_margin(self) -> float
```

Relative margin to constraint (1 = at limit, 0 = far from limit).

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py:95](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py#L95)*

##### `violation`

```python
def violation(self) -> float
```

Constraint violation (0 if satisfied, positive if violated).

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py:90](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py#L90)*

#### Methods

##### `__init__`

```python
def __init__(self, constraint_type: controller.ConstraintType, max_value: float, current_value: float = 0.0, margin: float = 0.1) -> None
```

### class `WaypointTarget`

Target waypoint for guidance.

#### Attributes

- **latitude_rad** (`<class 'float'>`): 
- **longitude_rad** (`<class 'float'>`): 
- **altitude_m** (`<class 'float'>`): 
- **velocity_m_s** (`<class 'float'>`): 
- **heading_rad** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, latitude_rad: float, longitude_rad: float, altitude_m: float = 0.0, velocity_m_s: float = 0.0, heading_rad: float = 0.0) -> None
```

## Functions

### `bank_angle_guidance`

```python
def bank_angle_guidance(current_state: ontic.guidance.trajectory.VehicleState, target: controller.WaypointTarget, corridor: controller.CorridorBounds, L_over_D: float = 1.5) -> controller.GuidanceCommand
```

Bank-to-turn guidance for hypersonic glide vehicles.

Modulates bank angle to control range while maintaining
equilibrium glide within thermal and structural constraints.

**Parameters:**

- **current_state** (`<class 'ontic.guidance.trajectory.VehicleState'>`): Current vehicle state
- **target** (`<class 'controller.WaypointTarget'>`): Target waypoint
- **corridor** (`<class 'controller.CorridorBounds'>`): Constraint corridor
- **L_over_D** (`<class 'float'>`): Current lift-to-drag ratio

**Returns**: `<class 'controller.GuidanceCommand'>` - GuidanceCommand with bank angle and mode

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py:178](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py#L178)*

### `closed_loop_simulation`

```python
def closed_loop_simulation(initial_state: ontic.guidance.trajectory.VehicleState, target: controller.WaypointTarget, duration_s: float = 60.0, dt: float = 0.01) -> Tuple[List[ontic.guidance.trajectory.VehicleState], List[controller.GuidanceCommand]]
```

Run closed-loop guidance simulation.

**Parameters:**

- **initial_state** (`<class 'ontic.guidance.trajectory.VehicleState'>`): Initial vehicle state
- **target** (`<class 'controller.WaypointTarget'>`): Target waypoint
- **duration_s** (`<class 'float'>`): Simulation duration
- **dt** (`<class 'float'>`): Time step

**Returns**: `typing.Tuple[typing.List[ontic.guidance.trajectory.VehicleState], typing.List[controller.GuidanceCommand]]` - (trajectory, commands)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py:617](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py#L617)*

### `proportional_navigation`

```python
def proportional_navigation(vehicle_state: ontic.guidance.trajectory.VehicleState, target_state: ontic.guidance.trajectory.VehicleState, nav_ratio: float = 3.0) -> float
```

Proportional navigation guidance law.

Used for terminal homing where line-of-sight rate
is nulled by commanding acceleration perpendicular to LOS.

**Parameters:**

- **vehicle_state** (`<class 'ontic.guidance.trajectory.VehicleState'>`): Current vehicle state
- **target_state** (`<class 'ontic.guidance.trajectory.VehicleState'>`): Target state
- **nav_ratio** (`<class 'float'>`): Navigation ratio (typically 3-5)

**Returns**: `<class 'float'>` - Commanded lateral acceleration (m/s²)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py:131](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py#L131)*

### `validate_guidance_module`

```python
def validate_guidance_module()
```

Validate guidance controller module.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py:658](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\guidance\controller.py#L658)*
