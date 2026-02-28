# Module `flight_validation.uncertainty`

Uncertainty quantification for flight data validation.

This module provides tools for propagating and analyzing
uncertainties in both flight data and CFD simulations.

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `GridConvergenceIndex`

Grid Convergence Index (GCI) for CFD uncertainty estimation.

Based on Roache's method for estimating discretization uncertainty.

#### Methods

##### `__init__`

```python
def __init__(self, refinement_ratio: float = 2.0, safety_factor: float = 1.25, target_order: float = 2.0)
```

Initialize GCI calculator.

**Parameters:**

- **refinement_ratio** (`<class 'float'>`): Grid refinement ratio
- **safety_factor** (`<class 'float'>`): Safety factor for GCI
- **target_order** (`<class 'float'>`): Target order of accuracy

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py:432](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py#L432)*

##### `calculate_gci`

```python
def calculate_gci(self, coarse_value: float, medium_value: float, fine_value: float) -> Dict[str, float]
```

Calculate GCI from three grid solutions.

**Parameters:**

- **coarse_value** (`<class 'float'>`): Value on coarsest grid
- **medium_value** (`<class 'float'>`): Value on medium grid
- **fine_value** (`<class 'float'>`): Value on finest grid

**Returns**: `typing.Dict[str, float]` - Dictionary with GCI metrics

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py:450](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py#L450)*

### class `MeasurementUncertainty`

Uncertainty analysis for measurements.

#### Attributes

- **parameter_name** (`<class 'str'>`): 
- **measured_value** (`<class 'float'>`): 
- **unit** (`<class 'str'>`): 
- **components** (`typing.List[uncertainty.UncertaintyComponent]`): 
- **combined_standard_uncertainty** (`<class 'float'>`): 
- **expanded_uncertainty** (`<class 'float'>`): 
- **coverage_factor** (`<class 'float'>`): 
- **percent_uncertainty** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, parameter_name: str, measured_value: float, unit: str = '', components: List[uncertainty.UncertaintyComponent] = <factory>, combined_standard_uncertainty: float = 0.0, expanded_uncertainty: float = 0.0, coverage_factor: float = 2.0, percent_uncertainty: float = 0.0) -> None
```

##### `add_component`

```python
def add_component(self, component: uncertainty.UncertaintyComponent)
```

Add uncertainty component.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py:97](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py#L97)*

##### `get_interval`

```python
def get_interval(self) -> Tuple[float, float]
```

Get uncertainty interval.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py:138](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py#L138)*

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py:125](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py#L125)*

### class `ModelUncertainty`

Uncertainty from CFD model.

#### Attributes

- **parameter_name** (`<class 'str'>`): 
- **nominal_value** (`<class 'float'>`): 
- **grid_uncertainty** (`<class 'float'>`): 
- **turbulence_model_uncertainty** (`<class 'float'>`): 
- **numerical_uncertainty** (`<class 'float'>`): 
- **boundary_condition_uncertainty** (`<class 'float'>`): 
- **total_model_uncertainty** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, parameter_name: str, nominal_value: float, grid_uncertainty: float = 0.0, turbulence_model_uncertainty: float = 0.0, numerical_uncertainty: float = 0.0, boundary_condition_uncertainty: float = 0.0, total_model_uncertainty: float = 0.0) -> None
```

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py:174](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py#L174)*

### class `UncertaintyBudget`

Complete uncertainty budget.

#### Attributes

- **name** (`<class 'str'>`): 
- **description** (`<class 'str'>`): 
- **measurement_uncertainties** (`typing.Dict[str, uncertainty.MeasurementUncertainty]`): 
- **model_uncertainties** (`typing.Dict[str, uncertainty.ModelUncertainty]`): 
- **validation_uncertainties** (`typing.Dict[str, uncertainty.ValidationUncertainty]`): 

#### Methods

##### `__init__`

```python
def __init__(self, name: str, description: str = '', measurement_uncertainties: Dict[str, uncertainty.MeasurementUncertainty] = <factory>, model_uncertainties: Dict[str, uncertainty.ModelUncertainty] = <factory>, validation_uncertainties: Dict[str, uncertainty.ValidationUncertainty] = <factory>) -> None
```

##### `add_measurement_uncertainty`

```python
def add_measurement_uncertainty(self, unc: uncertainty.MeasurementUncertainty)
```

Add measurement uncertainty.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py:249](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py#L249)*

##### `add_model_uncertainty`

```python
def add_model_uncertainty(self, unc: uncertainty.ModelUncertainty)
```

Add model uncertainty.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py:254](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py#L254)*

##### `get_summary`

```python
def get_summary(self) -> Dict[str, Any]
```

Get budget summary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py:272](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py#L272)*

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py:282](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py#L282)*

### class `UncertaintyComponent`

Single uncertainty component.

#### Attributes

- **name** (`<class 'str'>`): 
- **source** (`<enum 'UncertaintySource'>`): 
- **uncertainty_type** (`<enum 'UncertaintyType'>`): 
- **value** (`<class 'float'>`): 
- **value_percent** (`<class 'float'>`): 
- **distribution** (`<class 'str'>`): 
- **confidence_level** (`<class 'float'>`): 
- **degrees_of_freedom** (`<class 'int'>`): 
- **correlations** (`typing.Dict[str, float]`): 

#### Methods

##### `__init__`

```python
def __init__(self, name: str, source: uncertainty.UncertaintySource, uncertainty_type: uncertainty.UncertaintyType, value: float = 0.0, value_percent: float = 0.0, distribution: str = 'normal', confidence_level: float = 0.95, degrees_of_freedom: int = 0, correlations: Dict[str, float] = <factory>) -> None
```

##### `to_dict`

```python
def to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py:66](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py#L66)*

### class `UncertaintyPropagation`

Propagation of uncertainties through calculations.

#### Methods

##### `__init__`

```python
def __init__(self, method: str = 'linear')
```

Initialize propagation.

**Parameters:**

- **method** (`<class 'str'>`): Propagation method ("linear", "monte_carlo")

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py:301](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py#L301)*

##### `propagate`

```python
def propagate(self, func: Callable[..., float], inputs: Dict[str, float], uncertainties: Dict[str, float]) -> Tuple[float, float]
```

Propagate uncertainties using configured method.

**Parameters:**

- **func** (`typing.Callable[..., float]`): Function to evaluate
- **inputs** (`typing.Dict[str, float]`): Nominal input values
- **uncertainties** (`typing.Dict[str, float]`): Input uncertainties

**Returns**: `typing.Tuple[float, float]` - (output_value, output_uncertainty)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py:401](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py#L401)*

##### `propagate_linear`

```python
def propagate_linear(self, func: Callable[..., float], inputs: Dict[str, float], uncertainties: Dict[str, float], delta: float = 1e-06) -> Tuple[float, float]
```

Propagate uncertainties using linear approximation.

Uses finite difference to estimate sensitivities.

**Parameters:**

- **func** (`typing.Callable[..., float]`): Function to evaluate
- **inputs** (`typing.Dict[str, float]`): Nominal input values
- **uncertainties** (`typing.Dict[str, float]`): Input uncertainties
- **delta** (`<class 'float'>`): Finite difference step

**Returns**: `typing.Tuple[float, float]` - (output_value, output_uncertainty)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py:310](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py#L310)*

##### `propagate_monte_carlo`

```python
def propagate_monte_carlo(self, func: Callable[..., float], inputs: Dict[str, float], uncertainties: Dict[str, float], n_samples: int = 10000, seed: Optional[int] = None) -> Tuple[float, float, numpy.ndarray]
```

Propagate uncertainties using Monte Carlo sampling.

**Parameters:**

- **func** (`typing.Callable[..., float]`): Function to evaluate
- **inputs** (`typing.Dict[str, float]`): Nominal input values
- **uncertainties** (`typing.Dict[str, float]`): Input uncertainties (standard deviations)
- **n_samples** (`<class 'int'>`): Number of Monte Carlo samples
- **seed** (`typing.Optional[int]`): Random seed

**Returns**: `typing.Tuple[float, float, numpy.ndarray]` - (mean_output, std_output, samples)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py:357](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py#L357)*

### class `UncertaintySource`(Enum)

Sources of uncertainty.

### class `UncertaintyType`(Enum)

Types of uncertainty.

### class `ValidationUncertainty`

Combined uncertainty for validation.

#### Attributes

- **parameter_name** (`<class 'str'>`): 
- **measurement_uncertainty** (`typing.Optional[uncertainty.MeasurementUncertainty]`): 
- **model_uncertainty** (`typing.Optional[uncertainty.ModelUncertainty]`): 
- **combined_uncertainty** (`<class 'float'>`): 
- **comparison_error** (`<class 'float'>`): 
- **validation_metric** (`<class 'float'>`): 

#### Methods

##### `__init__`

```python
def __init__(self, parameter_name: str, measurement_uncertainty: Optional[uncertainty.MeasurementUncertainty] = None, model_uncertainty: Optional[uncertainty.ModelUncertainty] = None, combined_uncertainty: float = 0.0, comparison_error: float = 0.0, validation_metric: float = 0.0) -> None
```

##### `is_validated`

```python
def is_validated(self, threshold: float = 2.0) -> bool
```

Check if simulation is validated.

Validation passes if:
|E| < U_val * threshold

where E is comparison error and U_val is validation uncertainty.

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py:222](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py#L222)*

## Functions

### `calculate_gci`

```python
def calculate_gci(coarse_value: float, medium_value: float, fine_value: float, refinement_ratio: float = 2.0) -> Dict[str, float]
```

Calculate Grid Convergence Index.

**Parameters:**

- **coarse_value** (`<class 'float'>`): Value on coarsest grid
- **medium_value** (`<class 'float'>`): Value on medium grid
- **fine_value** (`<class 'float'>`): Value on finest grid
- **refinement_ratio** (`<class 'float'>`): Grid refinement ratio

**Returns**: `typing.Dict[str, float]` - Dictionary with GCI metrics

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py:570](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py#L570)*

### `calculate_measurement_uncertainty`

```python
def calculate_measurement_uncertainty(measured_value: float, sensor_accuracy: float, calibration_uncertainty: float = 0.0, repeatability: float = 0.0, coverage_factor: float = 2.0) -> uncertainty.MeasurementUncertainty
```

Calculate measurement uncertainty.

**Parameters:**

- **measured_value** (`<class 'float'>`): Measured value
- **sensor_accuracy** (`<class 'float'>`): Sensor accuracy (same units as value)
- **calibration_uncertainty** (`<class 'float'>`): Calibration uncertainty
- **repeatability** (`<class 'float'>`): Repeatability uncertainty
- **coverage_factor** (`<class 'float'>`): Coverage factor for expanded uncertainty

**Returns**: `<class 'uncertainty.MeasurementUncertainty'>` - MeasurementUncertainty object

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py:515](C:\TiganticLabz\Main_Projects\The Physics OS\tensornet\flight_validation\uncertainty.py#L515)*
