# Module `guidance.trajectory`

6-DOF Trajectory Solver =======================

Real-time trajectory propagation for hypersonic vehicles with
physics-based atmospheric and aerodynamic models.

Dynamics:
    - 6-DOF rigid body equations in body and NED frames
    - Quaternion attitude representation (singularity-free)
    - Rotating spherical Earth model
    - ISA/exponential/tabular atmospheric models

Integration:
    - RK4 (4th order Runge-Kutta) for fixed step
    - RK45 (Dormand-Prince) for adaptive step
    - Symplectic integrators for energy preservation

Performance:
    - Vectorized tensor operations
    - Pre-allocated memory for real-time execution
    - Target: 1000 Hz update rate for HIL simulation

Coordinate Frames:
    - NED: North-East-Down (navigation frame)
    - ECEF: Earth-Centered Earth-Fixed
    - Body: Vehicle body-fixed frame
    - Wind: Velocity-aligned frame

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `AeroCoefficients`

Aerodynamic coefficients for the vehicle.

These can be constant, tabulated, or computed from CFD.

#### Attributes

- **CD** (`<class 'float'>`): 
- **CL** (`<class 'float'>`): 
- **CY** (`<class 'float'>`): 
- **Cl** (`<class 'float'>`): 
- **Cm** (`<class 'float'>`): 
- **Cn** (`<class 'float'>`): 
- **CD_alpha** (`<class 'float'>`): 
- **CL_alpha** (`<class 'float'>`): 
- **Cm_alpha** (`<class 'float'>`): 
- **Cn_beta** (`<class 'float'>`): 
- **CL_de** (`<class 'float'>`): 
- **Cm_de** (`<class 'float'>`): 
- **Cn_dr** (`<class 'float'>`): 
- **Cm_q** (`<class 'float'>`): 
- **Cn_r** (`<class 'float'>`): 
- **Cl_p** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, CD: float = 0.02, CL: float = 0.5, CY: float = 0.0, Cl: float = 0.0, Cm: float = 0.0, Cn: float = 0.0, CD_alpha: float = 0.0, CL_alpha: float = 5.7, Cm_alpha: float = -0.5, Cn_beta: float = 0.1, CL_de: float = 0.5, Cm_de: float = -1.0, Cn_dr: float = -0.1, Cm_q: float = -10.0, Cn_r: float = -0.5, Cl_p: float = -0.4) -> None
```

### class `AtmosphereType`(Enum)

Atmospheric model type.

### class `AtmosphericModel`

Atmospheric properties at a given altitude.

#### Attributes

- **altitude_m** (`<class 'float'>`): 
- **temperature_K** (`<class 'float'>`): 
- **pressure_Pa** (`<class 'float'>`): 
- **density_kg_m3** (`<class 'float'>`): 
- **speed_of_sound_m_s** (`<class 'float'>`): 
- **viscosity_Pa_s** (`<class 'float'>`): 

#### Properties

##### `mach_from_velocity`

```python
def mach_from_velocity(self) -> Callable[[float], float]
```

Return function to compute Mach from velocity.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py:80](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py#L80)*

#### Methods

##### `__init__`

```python
def __init__(self, altitude_m: float, temperature_K: float, pressure_Pa: float, density_kg_m3: float, speed_of_sound_m_s: float, viscosity_Pa_s: float = 1.789e-05) -> None
```

### class `IntegrationMethod`(Enum)

Time integration methods.

### class `TrajectoryConfig`

Configuration for trajectory propagation.

#### Attributes

- **dt_s** (`<class 'float'>`): 
- **integration_method** (`<enum 'IntegrationMethod'>`): 
- **atmosphere_type** (`<enum 'AtmosphereType'>`): 
- **include_earth_rotation** (`<class 'bool'>`): 
- **include_gravity_variation** (`<class 'bool'>`): 
- **max_simulation_time_s** (`<class 'float'>`): 
- **save_interval** (`<class 'int'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, dt_s: float = 0.001, integration_method: trajectory.IntegrationMethod = <IntegrationMethod.RK4: 'rk4'>, atmosphere_type: trajectory.AtmosphereType = <AtmosphereType.ISA: 'isa'>, include_earth_rotation: bool = False, include_gravity_variation: bool = True, max_simulation_time_s: float = 300.0, save_interval: int = 10) -> None
```

### class `TrajectorySolver`

6-DOF trajectory propagator with real-time performance.

Features:
    - Vectorized state propagation
    - Configurable integration methods
    - Pre-allocated buffers for HIL
    - CFD aerodynamic coupling interface

#### Methods

##### `__init__`

```python
def __init__(self, config: trajectory.TrajectoryConfig = None, geometry: trajectory.VehicleGeometry = None, aero: trajectory.AeroCoefficients = None)
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py:377](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py#L377)*

##### `compute_aero_forces`

```python
def compute_aero_forces(self, state: trajectory.VehicleState, atm: trajectory.AtmosphericModel, controls: Optional[Dict[str, float]] = None) -> Tuple[torch.Tensor, torch.Tensor]
```

Compute aerodynamic forces and moments.

**Parameters:**

- **state** (`<class 'trajectory.VehicleState'>`): Current vehicle state
- **atm** (`<class 'trajectory.AtmosphericModel'>`): Atmospheric conditions
- **controls** (`typing.Optional[typing.Dict[str, float]]`): Control surface deflections

**Returns**: `typing.Tuple[torch.Tensor, torch.Tensor]` - (forces_body, moments_body) in N and N*m

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py:417](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py#L417)*

##### `compute_gravity_force`

```python
def compute_gravity_force(self, state: trajectory.VehicleState) -> torch.Tensor
```

Compute gravity force in body frame.

**Parameters:**

- **state** (`<class 'trajectory.VehicleState'>`): Current vehicle state

**Returns**: `<class 'torch.Tensor'>` - Gravity force vector in body frame

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py:504](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py#L504)*

##### `get_atmosphere`

```python
def get_atmosphere(self, altitude_m: float) -> trajectory.AtmosphericModel
```

Get atmospheric properties at altitude.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py:410](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py#L410)*

##### `propagate`

```python
def propagate(self, initial_state: trajectory.VehicleState, duration_s: float, controls_fn: Optional[Callable[[float, trajectory.VehicleState], Dict[str, float]]] = None) -> List[trajectory.VehicleState]
```

Propagate trajectory for given duration.

**Parameters:**

- **initial_state** (`<class 'trajectory.VehicleState'>`): Initial vehicle state
- **duration_s** (`<class 'float'>`): Simulation duration in seconds
- **controls_fn** (`typing.Optional[typing.Callable[[float, trajectory.VehicleState], typing.Dict[str, float]]]`): Function(t, state) -> controls dict

**Returns**: `typing.List[trajectory.VehicleState]` - List of states at save intervals

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py:668](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py#L668)*

##### `set_cfd_lookup`

```python
def set_cfd_lookup(self, lookup_fn: Callable[[float, float, float], trajectory.AeroCoefficients])
```

Set CFD-based aerodynamic coefficient lookup.

**Parameters:**

- **lookup_fn** (`typing.Callable[[float, float, float], trajectory.AeroCoefficients]`): Function(Mach, alpha, beta) -> AeroCoefficients

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py:401](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py#L401)*

##### `single_step`

```python
def single_step(self, state: trajectory.VehicleState, controls: Optional[Dict[str, float]] = None) -> trajectory.VehicleState
```

Advance one time step (for real-time loop).

**Parameters:**

- **state** (`<class 'trajectory.VehicleState'>`): Current state
- **controls** (`typing.Optional[typing.Dict[str, float]]`): Control inputs

**Returns**: `<class 'trajectory.VehicleState'>` - New state

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py:719](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py#L719)*

##### `state_derivative`

```python
def state_derivative(self, state: trajectory.VehicleState, controls: Optional[Dict[str, float]] = None) -> torch.Tensor
```

Compute state derivative for integration.

**Parameters:**

- **state** (`<class 'trajectory.VehicleState'>`): Current state
- **controls** (`typing.Optional[typing.Dict[str, float]]`): Control inputs

**Returns**: `<class 'torch.Tensor'>` - State derivative vector (14 elements)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py:530](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py#L530)*

##### `step_rk4`

```python
def step_rk4(self, state: trajectory.VehicleState, dt: float, controls: Optional[Dict[str, float]] = None) -> trajectory.VehicleState
```

Advance state using RK4 integration.

**Parameters:**

- **state** (`<class 'trajectory.VehicleState'>`): Current state
- **dt** (`<class 'float'>`): Time step
- **controls** (`typing.Optional[typing.Dict[str, float]]`): Control inputs

**Returns**: `<class 'trajectory.VehicleState'>` - New state after dt

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py:624](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py#L624)*

### class `VehicleGeometry`

Vehicle geometry and inertia properties.

#### Attributes

- **reference_area_m2** (`<class 'float'>`): 
- **reference_length_m** (`<class 'float'>`): 
- **reference_span_m** (`<class 'float'>`): 
- **Ixx** (`<class 'float'>`): 
- **Iyy** (`<class 'float'>`): 
- **Izz** (`<class 'float'>`): 
- **Ixz** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, reference_area_m2: float = 10.0, reference_length_m: float = 5.0, reference_span_m: float = 3.0, Ixx: float = 1000.0, Iyy: float = 5000.0, Izz: float = 5500.0, Ixz: float = 100.0) -> None
```

### class `VehicleState`

Complete vehicle state vector.

Position in NED or geodetic, velocity in body or NED,
attitude as quaternion or Euler angles.

#### Attributes

- **latitude_rad** (`<class 'float'>`): 
- **longitude_rad** (`<class 'float'>`): 
- **altitude_m** (`<class 'float'>`): 
- **u_m_s** (`<class 'float'>`): 
- **v_m_s** (`<class 'float'>`): 
- **w_m_s** (`<class 'float'>`): 
- **q0** (`<class 'float'>`): 
- **q1** (`<class 'float'>`): 
- **q2** (`<class 'float'>`): 
- **q3** (`<class 'float'>`): 
- **p_rad_s** (`<class 'float'>`): 
- **q_rad_s** (`<class 'float'>`): 
- **r_rad_s** (`<class 'float'>`): 
- **mass_kg** (`<class 'float'>`): 

#### Properties

##### `angle_of_attack`

```python
def angle_of_attack(self) -> float
```

Angle of attack (alpha) in radians.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py:142](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py#L142)*

##### `euler_angles`

```python
def euler_angles(self) -> Tuple[float, float, float]
```

Convert quaternion to Euler angles (phi, theta, psi).

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py:124](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py#L124)*

##### `sideslip_angle`

```python
def sideslip_angle(self) -> float
```

Sideslip angle (beta) in radians.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py:149](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py#L149)*

##### `velocity_magnitude`

```python
def velocity_magnitude(self) -> float
```

Total velocity magnitude.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py:119](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py#L119)*

#### Methods

##### `__init__`

```python
def __init__(self, latitude_rad: float = 0.0, longitude_rad: float = 0.0, altitude_m: float = 0.0, u_m_s: float = 0.0, v_m_s: float = 0.0, w_m_s: float = 0.0, q0: float = 1.0, q1: float = 0.0, q2: float = 0.0, q3: float = 0.0, p_rad_s: float = 0.0, q_rad_s: float = 0.0, r_rad_s: float = 0.0, mass_kg: float = 1000.0) -> None
```

##### `from_tensor`

```python
def from_tensor(t: torch.Tensor) -> 'VehicleState'
```

Create state from tensor.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py:167](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py#L167)*

##### `to_tensor`

```python
def to_tensor(self) -> torch.Tensor
```

Convert state to tensor for integration.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py:157](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py#L157)*

## Functions

### `create_reentry_trajectory`

```python
def create_reentry_trajectory(entry_altitude_m: float = 80000, entry_velocity_m_s: float = 7000, entry_angle_deg: float = -3.0, duration_s: float = 100.0) -> List[trajectory.VehicleState]
```

Create a simple reentry trajectory for testing.

**Parameters:**

- **entry_altitude_m** (`<class 'float'>`): Initial altitude
- **entry_velocity_m_s** (`<class 'float'>`): Initial velocity
- **entry_angle_deg** (`<class 'float'>`): Initial flight path angle
- **duration_s** (`<class 'float'>`): Simulation duration

**Returns**: `typing.List[trajectory.VehicleState]` - List of vehicle states

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py:737](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py#L737)*

### `exponential_atmosphere`

```python
def exponential_atmosphere(altitude_m: float, scale_height_m: float = 8500.0) -> trajectory.AtmosphericModel
```

Simple exponential atmospheric model.

Fast computation for trajectory optimization.

**Parameters:**

- **altitude_m** (`<class 'float'>`): Altitude in meters
- **scale_height_m** (`<class 'float'>`): Scale height H (default 8.5 km for Earth) Default: `8`.

**Returns**: `<class 'trajectory.AtmosphericModel'>` - AtmosphericModel

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py:306](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py#L306)*

### `gravity_model`

```python
def gravity_model(altitude_m: float, latitude_rad: float = 0.0) -> float
```

Compute gravitational acceleration.

Uses WGS84 ellipsoidal Earth model with J2 perturbation.

**Parameters:**

- **altitude_m** (`<class 'float'>`): Altitude above sea level
- **latitude_rad** (`<class 'float'>`): Geodetic latitude

**Returns**: `<class 'float'>` - Gravitational acceleration in m/s²

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py:339](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py#L339)*

### `isa_atmosphere`

```python
def isa_atmosphere(altitude_m: float) -> trajectory.AtmosphericModel
```

International Standard Atmosphere (ISA) model.

Valid for altitudes up to 85 km.

**Parameters:**

- **altitude_m** (`<class 'float'>`): Geometric altitude in meters

**Returns**: `<class 'trajectory.AtmosphericModel'>` - AtmosphericModel with atmospheric properties

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py:248](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py#L248)*

### `validate_trajectory_module`

```python
def validate_trajectory_module()
```

Validate trajectory solver module.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py:777](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\guidance\trajectory.py#L777)*
