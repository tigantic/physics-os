# Module `validation.physical`

Physical Validation Module for Project HyperTensor.

Provides physics-based validation tests including:
- Conservation law verification (mass, momentum, energy)
- Analytical solution comparisons
- Benchmark problem validation

These tests ensure numerical methods satisfy fundamental physical principles.

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `AnalyticalValidator`(ABC)

Abstract base class for validation against analytical solutions.

These validators compare numerical results against known exact solutions
to verify accuracy and convergence rates.

#### Methods

##### `__init__`

```python
def __init__(self, tolerance: float = 0.001, error_norm: str = 'L2')
```

Initialize analytical validator.

**Parameters:**

- **tolerance** (`<class 'float'>`): Maximum acceptable error
- **error_norm** (`<class 'str'>`): Error norm to use (L1, L2, Linf)

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py:464](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py#L464)*

##### `compute_analytical_solution`

```python
def compute_analytical_solution(self, x: torch.Tensor, t: float, **kwargs) -> torch.Tensor
```

Compute the analytical solution.

**Parameters:**

- **x** (`<class 'torch.Tensor'>`): Spatial coordinates
- **t** (`<class 'float'>`): Time **kwargs: Problem parameters

**Returns**: `<class 'torch.Tensor'>` - Analytical solution at given points and time

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py:479](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py#L479)*

##### `compute_error`

```python
def compute_error(self, numerical: torch.Tensor, analytical: torch.Tensor) -> float
```

Compute error between numerical and analytical solutions.

**Parameters:**

- **numerical** (`<class 'torch.Tensor'>`): Numerical solution
- **analytical** (`<class 'torch.Tensor'>`): Analytical solution

**Returns**: `<class 'float'>` - Error in specified norm

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py:499](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py#L499)*

##### `validate`

```python
def validate(self, numerical: torch.Tensor, x: torch.Tensor, t: float, test_name: str = 'Analytical Comparison', **kwargs) -> physical.ValidationResult
```

Validate numerical solution against analytical.

**Parameters:**

- **numerical** (`<class 'torch.Tensor'>`): Numerical solution
- **x** (`<class 'torch.Tensor'>`): Spatial coordinates
- **t** (`<class 'float'>`): Time
- **test_name** (`<class 'str'>`): Name for this validation **kwargs: Problem parameters

**Returns**: `<class 'physical.ValidationResult'>` - ValidationResult with comparison outcome

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py:525](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py#L525)*

### class `BlasiusValidator`(AnalyticalValidator)

Validator for Blasius flat plate boundary layer.

The Blasius solution is the similarity solution for laminar flow
over a flat plate, providing exact velocity profiles.

#### Methods

##### `__init__`

```python
def __init__(self, U_inf: float = 1.0, nu: float = 0.0001, tolerance: float = 0.02)
```

Initialize Blasius validator.

**Parameters:**

- **U_inf** (`<class 'float'>`): Freestream velocity
- **nu** (`<class 'float'>`): Kinematic viscosity
- **tolerance** (`<class 'float'>`): Maximum acceptable error

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py:792](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py#L792)*

##### `compute_analytical_solution`

```python
def compute_analytical_solution(self, x: torch.Tensor, t: float, y: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor
```

Compute Blasius velocity profile.

**Parameters:**

- **x** (`<class 'torch.Tensor'>`): Streamwise coordinates
- **t** (`<class 'float'>`): Not used (steady flow)
- **y** (`typing.Optional[torch.Tensor]`): Wall-normal coordinates

**Returns**: `<class 'torch.Tensor'>` - Velocity u/U_inf at each (x, y) point

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py:843](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py#L843)*

### class `ConservationValidator`(ABC)

Abstract base class for conservation law validators.

Conservation laws are fundamental physical principles that must be
satisfied by any valid numerical simulation.

#### Methods

##### `__init__`

```python
def __init__(self, tolerance: float = 1e-10, relative: bool = True)
```

Initialize conservation validator.

**Parameters:**

- **tolerance** (`<class 'float'>`): Tolerance for conservation check
- **relative** (`<class 'bool'>`): Use relative (True) or absolute (False) tolerance

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py:185](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py#L185)*

##### `compute_conserved_quantity`

```python
def compute_conserved_quantity(self, state: torch.Tensor, **kwargs) -> float
```

Compute the conserved quantity from the state.

**Parameters:**

- **state** (`<class 'torch.Tensor'>`): Current simulation state **kwargs: Additional parameters (geometry, etc.)

**Returns**: `<class 'float'>` - Value of the conserved quantity

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py:200](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py#L200)*

##### `validate`

```python
def validate(self, initial_state: torch.Tensor, final_state: torch.Tensor, test_name: str = 'Conservation', **kwargs) -> physical.ValidationResult
```

Validate conservation between initial and final states.

**Parameters:**

- **initial_state** (`<class 'torch.Tensor'>`): State at beginning
- **final_state** (`<class 'torch.Tensor'>`): State at end
- **test_name** (`<class 'str'>`): Name for this validation test **kwargs: Additional parameters

**Returns**: `<class 'physical.ValidationResult'>` - ValidationResult with conservation check outcome

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py:218](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py#L218)*

### class `EnergyConservationTest`(ConservationValidator)

Validator for total energy conservation in CFD simulations.

Total energy = ∫E dV should be conserved in inviscid flow
with adiabatic boundaries.

#### Methods

##### `__init__`

```python
def __init__(self, dx: float = 1.0, dy: float = 1.0, dz: float = 1.0, energy_index: int = -1, tolerance: float = 1e-10)
```

Initialize energy conservation validator.

**Parameters:**

- **energy_index** (`<class 'int'>`): Index of energy in state vector (-1 for last)
- **tolerance** (`<class 'float'>`): Conservation tolerance

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py:407](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py#L407)*

##### `compute_conserved_quantity`

```python
def compute_conserved_quantity(self, state: torch.Tensor, **kwargs) -> float
```

Compute total energy from state.

**Parameters:**

- **state** (`<class 'torch.Tensor'>`): Conservative state tensor with energy as last variable

**Returns**: `<class 'float'>` - Total energy in domain

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py:429](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py#L429)*

### class `IsentropicVortexValidator`(AnalyticalValidator)

Validator for isentropic vortex advection.

The isentropic vortex is an exact solution to the Euler equations
that advects without changing shape, making it ideal for accuracy testing.

#### Methods

##### `__init__`

```python
def __init__(self, gamma: float = 1.4, vortex_strength: float = 5.0, tolerance: float = 0.01)
```

Initialize isentropic vortex validator.

**Parameters:**

- **gamma** (`<class 'float'>`): Ratio of specific heats
- **vortex_strength** (`<class 'float'>`): Vortex circulation parameter
- **tolerance** (`<class 'float'>`): Maximum acceptable error

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py:998](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py#L998)*

##### `compute_analytical_solution`

```python
def compute_analytical_solution(self, x: torch.Tensor, t: float, y: Optional[torch.Tensor] = None, x0: float = 5.0, y0: float = 5.0, **kwargs) -> torch.Tensor
```

Compute isentropic vortex solution.

**Parameters:**

- **x** (`<class 'torch.Tensor'>`): x-coordinates
- **t** (`<class 'float'>`): Time
- **y** (`typing.Optional[torch.Tensor]`): y-coordinates x0, y0: Initial vortex center

**Returns**: `<class 'torch.Tensor'>` - Primitive state (rho, u, v, p) at each point

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py:1022](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py#L1022)*

### class `MassConservationTest`(ConservationValidator)

Validator for mass conservation in CFD simulations.

For compressible flow, total mass = ∫ρ dV should be conserved
in the absence of mass sources/sinks.

#### Methods

##### `__init__`

```python
def __init__(self, dx: float = 1.0, dy: float = 1.0, dz: float = 1.0, tolerance: float = 1e-10)
```

Initialize mass conservation validator.

**Parameters:**

- **tolerance** (`<class 'float'>`): Conservation tolerance

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py:288](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py#L288)*

##### `compute_conserved_quantity`

```python
def compute_conserved_quantity(self, state: torch.Tensor, **kwargs) -> float
```

Compute total mass from density field.

**Parameters:**

- **state** (`<class 'torch.Tensor'>`): State tensor with density in first channel
- **Shape**: (nvar, nx) for 1D, (nvar, nx, ny) for 2D, etc.

**Returns**: `<class 'float'>` - Total mass in domain

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py:307](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py#L307)*

### class `MomentumConservationTest`(ConservationValidator)

Validator for momentum conservation in CFD simulations.

Total momentum = ∫ρu dV should be conserved in the absence
of external forces and with appropriate boundary conditions.

#### Methods

##### `__init__`

```python
def __init__(self, dx: float = 1.0, dy: float = 1.0, dz: float = 1.0, component: int = 0, tolerance: float = 1e-10)
```

Initialize momentum conservation validator.

**Parameters:**

- **component** (`<class 'int'>`): Momentum component to check (0=x, 1=y, 2=z)
- **tolerance** (`<class 'float'>`): Conservation tolerance

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py:347](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py#L347)*

##### `compute_conserved_quantity`

```python
def compute_conserved_quantity(self, state: torch.Tensor, **kwargs) -> float
```

Compute total momentum from state.

**Parameters:**

- **state** (`<class 'torch.Tensor'>`): Conservative state tensor For Euler: (rho, rho*u, rho*v, rho*w, E)

**Returns**: `<class 'float'>` - Total momentum component

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py:369](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py#L369)*

### class `ObliqueShockValidator`(AnalyticalValidator)

Validator for oblique shock waves.

Compares numerical shock angles and post-shock states against
the θ-β-M analytical relations.

#### Methods

##### `__init__`

```python
def __init__(self, gamma: float = 1.4, tolerance: float = 0.01)
```

Initialize oblique shock validator.

**Parameters:**

- **gamma** (`<class 'float'>`): Ratio of specific heats
- **tolerance** (`<class 'float'>`): Maximum acceptable error

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py:885](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py#L885)*

##### `compute_analytical_solution`

```python
def compute_analytical_solution(self, x: torch.Tensor, t: float, M1: float = 2.0, theta: float = 10.0, **kwargs) -> torch.Tensor
```

Compute post-shock state from oblique shock relations.

**Parameters:**

- **x** (`<class 'torch.Tensor'>`): Not used directly
- **t** (`<class 'float'>`): Not used (steady flow)
- **M1** (`<class 'float'>`): Upstream Mach number
- **theta** (`<class 'float'>`): Wedge/deflection angle in degrees

**Returns**: `<class 'torch.Tensor'>` - Post-shock state (rho2/rho1, p2/p1, M2)

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py:900](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py#L900)*

##### `compute_shock_angle`

```python
def compute_shock_angle(self, M1: float, theta: float) -> float
```

Compute shock angle for given Mach and deflection.

**Parameters:**

- **M1** (`<class 'float'>`): Upstream Mach number
- **theta** (`<class 'float'>`): Deflection angle in degrees

**Returns**: `<class 'float'>` - Shock angle in degrees

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py:975](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py#L975)*

### class `SodShockValidator`(AnalyticalValidator)

Validator for Sod shock tube problem.

The Sod shock tube is a 1D Riemann problem with known analytical solution,
widely used for CFD code verification.

#### Methods

##### `__init__`

```python
def __init__(self, gamma: float = 1.4, tolerance: float = 0.01, x_discontinuity: float = 0.5)
```

Initialize Sod shock tube validator.

**Parameters:**

- **gamma** (`<class 'float'>`): Ratio of specific heats
- **tolerance** (`<class 'float'>`): Maximum L2 error
- **x_discontinuity** (`<class 'float'>`): Initial discontinuity location

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py:586](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py#L586)*

##### `compute_analytical_solution`

```python
def compute_analytical_solution(self, x: torch.Tensor, t: float, **kwargs) -> torch.Tensor
```

Compute Sod shock tube analytical solution.

Uses the exact Riemann solver for the shock tube problem.

**Parameters:**

- **x** (`<class 'torch.Tensor'>`): Spatial coordinates
- **t** (`<class 'float'>`): Time

**Returns**: `<class 'torch.Tensor'>` - Primitive state (rho, u, p) at each point

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py:608](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py#L608)*

### class `ValidationReport`

Comprehensive validation report containing multiple test results.

#### Attributes

- **title** (`<class 'str'>`): 
- **timestamp** (`<class 'float'>`): 
- **results** (`typing.List[physical.ValidationResult]`): 
- **summary** (`typing.Dict[str, int]`): 
- **configuration** (`typing.Dict[str, typing.Any]`): 

#### Properties

##### `all_passed`

```python
def all_passed(self) -> bool
```

Check if all tests passed.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py:111](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py#L111)*

##### `pass_rate`

```python
def pass_rate(self) -> float
```

Compute pass rate as percentage.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py:116](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py#L116)*

#### Methods

##### `__init__`

```python
def __init__(self, title: str, timestamp: float, results: List[physical.ValidationResult], summary: Dict[str, int] = <factory>, configuration: Dict[str, Any] = <factory>) -> None
```

##### `save`

```python
def save(self, filepath: Union[str, pathlib.Path])
```

Save report to JSON file.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py:171](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py#L171)*

##### `to_dict`

```python
def to_dict(self) -> Dict
```

Convert to dictionary for serialization.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py:161](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py#L161)*

##### `to_markdown`

```python
def to_markdown(self) -> str
```

Generate markdown report.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py:123](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py#L123)*

### class `ValidationResult`

Result of a single validation test.

#### Attributes

- **test_name** (`<class 'str'>`): 
- **passed** (`<class 'bool'>`): 
- **severity** (`<enum 'ValidationSeverity'>`): 
- **metric_name** (`<class 'str'>`): 
- **computed_value** (`<class 'float'>`): 
- **expected_value** (`<class 'float'>`): 
- **tolerance** (`<class 'float'>`): 
- **relative_error** (`typing.Optional[float]`): 
- **absolute_error** (`typing.Optional[float]`): 
- **message** (`<class 'str'>`): 
- **details** (`typing.Dict[str, typing.Any]`): 

#### Methods

##### `__init__`

```python
def __init__(self, test_name: str, passed: bool, severity: physical.ValidationSeverity, metric_name: str, computed_value: float, expected_value: float, tolerance: float, relative_error: Optional[float] = None, absolute_error: Optional[float] = None, message: str = '', details: Dict[str, Any] = <factory>) -> None
```

##### `to_dict`

```python
def to_dict(self) -> Dict
```

Convert to dictionary for serialization.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py:61](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py#L61)*

### class `ValidationSeverity`(Enum)

Severity level for validation results.

## Functions

### `run_physical_validation`

```python
def run_physical_validation(validation_tests: List[Union[physical.ConservationValidator, physical.AnalyticalValidator]], **kwargs) -> physical.ValidationReport
```

Run a suite of physical validation tests.

**Parameters:**

- **validation_tests** (`typing.List[typing.Union[physical.ConservationValidator, physical.AnalyticalValidator]]`): List of validators to run **kwargs: Arguments passed to each validator

**Returns**: `<class 'physical.ValidationReport'>` - ValidationReport with all results

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py:1074](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\validation\physical.py#L1074)*
