# Module `simulation.realtime_cfd`

Real-Time CFD Coupling Module =============================

Provides interfaces for coupling high-fidelity CFD solutions with
real-time guidance algorithms through aerodynamic tables and
surrogate models.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    OFFLINE (Pre-computation)                    │
    │  ┌────────────┐    ┌──────────────┐    ┌────────────────────┐  │
    │  │ Mesh/Grid  │───►│ CFD Solver   │───►│ Post-Processing    │  │
    │  │ Generation │    │ (Euler/NS)   │    │ (Forces/Moments)   │  │
    │  └────────────┘    └──────────────┘    └─────────┬──────────┘  │
    │                                                   │             │
    │                                          ┌───────▼────────┐    │
    │                                          │  AeroTable     │    │
    │                                          │  Generation    │    │
    │                                          └───────┬────────┘    │
    └──────────────────────────────────────────────────┼─────────────┘
                                                       │
    ┌──────────────────────────────────────────────────┼─────────────┐
    │                    ONLINE (Real-time)            │             │
    │                                          ┌───────▼────────┐    │
    │                                          │   AeroTable    │    │
    │                                          │   Lookup       │    │
    │                                          └───────┬────────┘    │
    │                                                   │             │
    │  ┌────────────┐    ┌──────────────┐    ┌────────▼─────────┐   │
    │  │ Vehicle    │───►│ RealTimeCFD  │◄───│ Interpolate      │   │
    │  │ State      │    │ Interface    │    │ CL, CD, Cm, ...  │   │
    │  └────────────┘    └──────────────┘    └──────────────────┘   │
    └─────────────────────────────────────────────────────────────────┘

Interpolation Methods:
    - Linear (fastest, for monotonic data)
    - Cubic spline (smooth, general purpose)
    - Kriging (uncertainty quantification)
    - Neural network (complex nonlinear)

Table Dimensions:
    - Mach number (primary)
    - Angle of attack (primary)
    - Sideslip angle (secondary)
    - Control surface deflections
    - Reynolds number (optional)

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `AeroCoefficient`

Single aerodynamic coefficient with derivatives.

#### Attributes

- **base** (`<class 'float'>`): 
- **d_alpha** (`<class 'float'>`): 
- **d_beta** (`<class 'float'>`): 
- **d_p** (`<class 'float'>`): 
- **d_q** (`<class 'float'>`): 
- **d_r** (`<class 'float'>`): 
- **d_de** (`<class 'float'>`): 
- **d_da** (`<class 'float'>`): 
- **d_dr** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, base: float = 0.0, d_alpha: float = 0.0, d_beta: float = 0.0, d_p: float = 0.0, d_q: float = 0.0, d_r: float = 0.0, d_de: float = 0.0, d_da: float = 0.0, d_dr: float = 0.0) -> None
```

### class `AeroPoint`

Aerodynamic coefficients at a single flight condition.

#### Attributes

- **CL** (`<class 'float'>`): 
- **CD** (`<class 'float'>`): 
- **CY** (`<class 'float'>`): 
- **Cl** (`<class 'float'>`): 
- **Cm** (`<class 'float'>`): 
- **Cn** (`<class 'float'>`): 
- **CL_alpha** (`<class 'float'>`): 
- **CD_alpha** (`<class 'float'>`): 
- **Cm_alpha** (`<class 'float'>`): 
- **Cm_q** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, CL: float = 0.0, CD: float = 0.0, CY: float = 0.0, Cl: float = 0.0, Cm: float = 0.0, Cn: float = 0.0, CL_alpha: float = 0.0, CD_alpha: float = 0.0, Cm_alpha: float = 0.0, Cm_q: float = 0.0) -> None
```

##### `from_vector`

```python
def from_vector(v: numpy.ndarray) -> 'AeroPoint'
```

Create from coefficient vector.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py:148](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py#L148)*

##### `to_vector`

```python
def to_vector(self) -> numpy.ndarray
```

Convert to coefficient vector.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py:144](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py#L144)*

### class `AeroTable`

Multi-dimensional aerodynamic lookup table.

Provides fast interpolation of aerodynamic coefficients
across the flight envelope.

#### Methods

##### `__init__`

```python
def __init__(self, config: realtime_cfd.AeroTableConfig = None)
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py:162](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py#L162)*

##### `load`

```python
def load(filepath: Union[str, pathlib.Path]) -> 'AeroTable'
```

Load table from file.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py:347](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py#L347)*

##### `lookup`

```python
def lookup(self, mach: float, alpha_deg: float, beta_deg: float = 0.0) -> realtime_cfd.AeroPoint
```

Look up aerodynamic coefficients.

**Parameters:**

- **mach** (`<class 'float'>`): Mach number
- **alpha_deg** (`<class 'float'>`): Angle of attack (degrees)
- **beta_deg** (`<class 'float'>`): Sideslip angle (degrees)

**Returns**: `<class 'realtime_cfd.AeroPoint'>` - AeroPoint with interpolated coefficients

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py:268](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py#L268)*

##### `lookup_batch`

```python
def lookup_batch(self, mach: numpy.ndarray, alpha_deg: numpy.ndarray, beta_deg: numpy.ndarray) -> Dict[str, numpy.ndarray]
```

Batch lookup for multiple points.

**Parameters:**

- **mach** (`<class 'numpy.ndarray'>`): Array of Mach numbers
- **alpha_deg** (`<class 'numpy.ndarray'>`): Array of angles of attack
- **beta_deg** (`<class 'numpy.ndarray'>`): Array of sideslip angles

**Returns**: `typing.Dict[str, numpy.ndarray]` - Dict of coefficient arrays

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py:294](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py#L294)*

##### `populate_from_arrays`

```python
def populate_from_arrays(self, CL: numpy.ndarray, CD: numpy.ndarray, Cm: numpy.ndarray, CY: Optional[numpy.ndarray] = None, Cl: Optional[numpy.ndarray] = None, Cn: Optional[numpy.ndarray] = None)
```

Populate from pre-computed arrays.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py:214](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py#L214)*

##### `populate_from_cfd`

```python
def populate_from_cfd(self, cfd_solver: Callable[[float, float, float], Dict[str, float]])
```

Populate table from CFD solver.

**Parameters:**

- **cfd_solver** (`typing.Callable[[float, float, float], typing.Dict[str, float]]`): Function(mach, alpha_deg, beta_deg) -> aero_dict

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py:188](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py#L188)*

##### `save`

```python
def save(self, filepath: Union[str, pathlib.Path])
```

Save table to file.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py:325](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py#L325)*

### class `AeroTableConfig`

Configuration for aerodynamic table generation.

#### Attributes

- **mach_range** (`typing.Tuple[float, float]`): 
- **alpha_range** (`typing.Tuple[float, float]`): 
- **beta_range** (`typing.Tuple[float, float]`): 
- **n_mach** (`<class 'int'>`): 
- **n_alpha** (`<class 'int'>`): 
- **n_beta** (`<class 'int'>`): 
- **elevator_range** (`typing.Tuple[float, float]`): 
- **aileron_range** (`typing.Tuple[float, float]`): 
- **rudder_range** (`typing.Tuple[float, float]`): 
- **n_deflection** (`<class 'int'>`): 
- **method** (`<enum 'InterpolationMethod'>`): 
- **S_ref** (`<class 'float'>`): 
- **c_ref** (`<class 'float'>`): 
- **b_ref** (`<class 'float'>`): 
- **bounds_error** (`<class 'bool'>`): 
- **fill_value** (`typing.Optional[float]`): 

#### Methods

##### `__init__`

```python
def __init__(self, mach_range: Tuple[float, float] = (0.5, 10.0), alpha_range: Tuple[float, float] = (-5.0, 25.0), beta_range: Tuple[float, float] = (-10.0, 10.0), n_mach: int = 20, n_alpha: int = 31, n_beta: int = 11, elevator_range: Tuple[float, float] = (-30.0, 30.0), aileron_range: Tuple[float, float] = (-25.0, 25.0), rudder_range: Tuple[float, float] = (-30.0, 30.0), n_deflection: int = 13, method: realtime_cfd.InterpolationMethod = <InterpolationMethod.LINEAR: 'linear'>, S_ref: float = 10.0, c_ref: float = 2.0, b_ref: float = 5.0, bounds_error: bool = False, fill_value: Optional[float] = None) -> None
```

### class `InterpolationMethod`(Enum)

Interpolation method for aero tables.

### class `RealTimeCFD`

Real-time CFD interface for guidance systems.

Combines pre-computed aero tables with optional online
corrections for real-time simulation.

#### Methods

##### `__init__`

```python
def __init__(self, aero_table: realtime_cfd.AeroTable, config: realtime_cfd.AeroTableConfig = None)
```

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py:388](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py#L388)*

##### `get_aero`

```python
def get_aero(self, state: Dict[str, float], controls: Dict[str, float] = None) -> Dict[str, float]
```

Get aerodynamic forces and moments.

**Parameters:**

- **state** (`typing.Dict[str, float]`): Vehicle state (mach, alpha_deg, beta_deg, alt, V, q_bar)
- **controls** (`typing.Dict[str, float]`): Control deflections (de, da, dr)

**Returns**: `typing.Dict[str, float]` - Aero dict with forces, moments, and coefficients

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py:409](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py#L409)*

##### `get_derivatives`

```python
def get_derivatives(self, state: Dict[str, float], epsilon: float = 0.1) -> Dict[str, float]
```

Estimate aerodynamic derivatives numerically.

**Parameters:**

- **state** (`typing.Dict[str, float]`): Current flight state
- **epsilon** (`<class 'float'>`): Perturbation size (degrees for alpha)

**Returns**: `typing.Dict[str, float]` - Dict of derivatives

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py:507](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py#L507)*

##### `get_performance_stats`

```python
def get_performance_stats(self) -> Dict[str, float]
```

Get lookup performance statistics.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py:538](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py#L538)*

##### `update_correction`

```python
def update_correction(self, measured_CL: float, measured_CD: float, measured_Cm: float)
```

Update correction factors from in-flight measurements.

**Parameters:**

- **measured_CL** (`<class 'float'>`): Measured lift coefficient
- **measured_CD** (`<class 'float'>`): Measured drag coefficient
- **measured_Cm** (`<class 'float'>`): Measured pitching moment coefficient

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py:486](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py#L486)*

### class `TableDimension`(Enum)

Standard aerodynamic table dimensions.

## Functions

### `build_aero_table`

```python
def build_aero_table(cfd_solver: Callable, config: realtime_cfd.AeroTableConfig = None, parallel: bool = False) -> realtime_cfd.AeroTable
```

Build aerodynamic table from CFD solver.

**Parameters:**

- **cfd_solver** (`typing.Callable`): Function(mach, alpha, beta) -> aero_dict
- **config** (`<class 'realtime_cfd.AeroTableConfig'>`): Table configuration
- **parallel** (`<class 'bool'>`): Use parallel computation

**Returns**: `<class 'realtime_cfd.AeroTable'>` - Populated AeroTable

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py:550](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py#L550)*

### `create_hypersonic_waverider_model`

```python
def create_hypersonic_waverider_model() -> Callable
```

Create a simplified hypersonic waverider aerodynamic model.

**Returns**: `typing.Callable` - CFD solver function

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py:630](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py#L630)*

### `interpolate_coefficients`

```python
def interpolate_coefficients(table: realtime_cfd.AeroTable, mach: float, alpha_deg: float, beta_deg: float = 0.0) -> Tuple[float, float, float]
```

Convenience function for interpolating CL, CD, Cm.

**Parameters:**

- **table** (`<class 'realtime_cfd.AeroTable'>`): AeroTable
- **mach** (`<class 'float'>`): Mach number
- **alpha_deg** (`<class 'float'>`): Angle of attack
- **beta_deg** (`<class 'float'>`): Sideslip angle

**Returns**: `typing.Tuple[float, float, float]` - (CL, CD, Cm) tuple

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py:572](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py#L572)*

### `validate_aero_table`

```python
def validate_aero_table(table: realtime_cfd.AeroTable, validation_points: List[Tuple[float, float, float, Dict[str, float]]]) -> Dict[str, float]
```

Validate aero table against known points.

**Parameters:**

- **table** (`<class 'realtime_cfd.AeroTable'>`): AeroTable to validate
- **validation_points** (`typing.List[typing.Tuple[float, float, float, typing.Dict[str, float]]]`): List of (mach, alpha, beta, expected_aero)

**Returns**: `typing.Dict[str, float]` - Validation metrics

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py:594](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py#L594)*

### `validate_realtime_cfd_module`

```python
def validate_realtime_cfd_module()
```

Validate real-time CFD module.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py:682](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\simulation\realtime_cfd.py#L682)*
